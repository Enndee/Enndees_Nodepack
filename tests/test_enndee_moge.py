"""Tests for the MoGe-3 depth source (`enndee_moge` + the fast-depth MoGe branch).

The `moge` package is optional and is NOT installed in this environment - which is exactly the
state the suite has to keep working in. Everything here fakes the model, so the MoGe path is
covered end to end without downloading a checkpoint, and the "not installed" behaviour is
asserted rather than assumed.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PACK_DIR = Path(__file__).resolve().parents[1]
NODES_DIR = PACK_DIR / "nodes"
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(NODES_DIR))

import enndee_moge  # noqa: E402
import enndee_meridian_fast_depth as fast_depth  # noqa: E402


def _gradient_image(height=64, width=112):
    """[1, H, W, 3] in [0, 1] with a left-to-right ramp in the red channel."""
    ramp = torch.linspace(0.0, 1.0, width).view(1, 1, width).expand(1, height, width)
    return torch.stack([ramp, ramp.flip(-1), torch.full_like(ramp, 0.5)], dim=-1)


class _FakeMoGeV3:
    """MoGe-3 as documented: `infer` takes `refine_steps`, depth comes back as a dict."""

    def __init__(self):
        self.calls = []
        self.repo_id = None

    @classmethod
    def from_pretrained(cls, repo_id):
        instance = cls()
        instance.repo_id = repo_id
        return instance

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer(self, image, resolution_level=9, use_fp16=False, refine_steps=3):
        self.calls.append({"image": tuple(image.shape), "resolution_level": resolution_level,
                           "use_fp16": use_fp16, "refine_steps": refine_steps})
        return {"depth": image[0]}          # (H, W); larger = farther, like the real model


class _FakeMoGeNoRefine:
    """An older `moge` build whose `infer` has no `refine_steps` keyword."""

    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, repo_id):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer(self, image, resolution_level=9, use_fp16=False):
        self.calls.append({"image": tuple(image.shape)})
        return {"depth": image[0]}


class RefineStepsTests(unittest.TestCase):
    """`moge_refine_steps` is validated, not clamped - a typo must not be hidden."""

    def test_defaults_when_the_widget_is_absent(self):
        for value in (None, ""):
            self.assertEqual(enndee_moge.resolve_refine_steps(value),
                             enndee_moge.DEFAULT_REFINE_STEPS)

    def test_accepts_the_whole_documented_range(self):
        self.assertEqual(enndee_moge.resolve_refine_steps(0), 0)     # 0 = stage disabled
        self.assertEqual(enndee_moge.resolve_refine_steps(3), 3)     # MoGe's own default
        self.assertEqual(enndee_moge.resolve_refine_steps(12),
                         enndee_moge.MAX_REFINE_STEPS)
        self.assertEqual(enndee_moge.resolve_refine_steps("7"), 7)   # a stringified widget

    def test_rejects_out_of_range_and_non_numeric_values(self):
        for value in (-1, enndee_moge.MAX_REFINE_STEPS + 1, "many", 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "moge_refine_steps"):
                enndee_moge.resolve_refine_steps(value)


class LoaderTests(unittest.TestCase):
    def test_status_reports_a_missing_package_without_raising(self):
        with mock.patch.object(enndee_moge, "load_moge_v3",
                               side_effect=RuntimeError("not installed")):
            usable, reason = enndee_moge.moge_status()
        self.assertFalse(usable)
        self.assertIn("not usable", reason)

    def test_status_reports_ready_when_the_class_loads(self):
        with mock.patch.object(enndee_moge, "load_moge_v3", return_value=_FakeMoGeV3):
            usable, reason = enndee_moge.moge_status()
        self.assertTrue(usable)
        self.assertIn(enndee_moge.MOGE_V3_REPO, reason)

    def test_a_missing_package_raises_with_the_install_hint(self):
        # `None` in sys.modules makes the import raise ImportError deterministically, whether or
        # not the real package happens to be installed on the machine running the suite.
        with mock.patch.dict(sys.modules, {"moge": None, "moge.model": None}):
            with self.assertRaises(RuntimeError) as caught:
                enndee_moge.load_moge_v3()
        self.assertIn("MoGe", str(caught.exception))
        self.assertIn("pip install", str(caught.exception))


class DepthTensorTests(unittest.TestCase):
    """`_moge_depth_tensor` accepts the shapes MoGe actually returns."""

    def test_plain_and_batched_dict_depths_become_hw(self):
        plain = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertTrue(torch.equal(fast_depth._moge_depth_tensor({"depth": plain}), plain))
        batched = plain.unsqueeze(0)
        self.assertTrue(torch.equal(fast_depth._moge_depth_tensor({"depth": batched}), plain))

    def test_an_object_with_a_depth_attribute_is_accepted(self):
        class _Out:
            depth = torch.ones(2, 3)

        self.assertEqual(tuple(fast_depth._moge_depth_tensor(_Out()).shape), (2, 3))

    def test_a_result_without_depth_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "no 'depth' map"):
            fast_depth._moge_depth_tensor({"points": torch.ones(2, 3)})

    def test_a_non_map_shape_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected shape"):
            fast_depth._moge_depth_tensor({"depth": torch.ones(2, 3, 4)})


class InferKwargsTests(unittest.TestCase):
    """`refine_steps` is v3-only, so the installed signature decides - not an assumption."""

    def _kwargs(self, model, steps=3):
        with mock.patch.object(fast_depth, "_GLOBAL_MOGE_MODEL", model), \
                mock.patch.object(fast_depth, "_MOGE_INFER_PARAMS", None):
            return fast_depth._moge_infer_kwargs(steps, fast_depth.MOGE_RESOLUTION_LEVEL)

    def test_v3_receives_the_refinement_step_count(self):
        kwargs = self._kwargs(_FakeMoGeV3(), steps=5)
        self.assertEqual(kwargs["refine_steps"], 5)
        self.assertEqual(kwargs["resolution_level"], fast_depth.MOGE_RESOLUTION_LEVEL)
        self.assertTrue(kwargs["use_fp16"])

    def test_an_older_build_without_refine_steps_is_not_passed_it(self):
        self.assertNotIn("refine_steps", self._kwargs(_FakeMoGeNoRefine()))


class PredictMogeDepthTests(unittest.TestCase):
    def setUp(self):
        for patch in (
            mock.patch.object(fast_depth, "load_moge_v3", return_value=_FakeMoGeV3),
            mock.patch.object(fast_depth, "_GLOBAL_MOGE_MODEL", None),
            mock.patch.object(fast_depth, "_GLOBAL_MOGE_ID", None),
            mock.patch.object(fast_depth, "_MOGE_INFER_PARAMS", None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_returns_hw_depth_and_forwards_the_refinement_steps(self):
        image = _gradient_image()
        depth = fast_depth._predict_moge_depth("MoGe-3-ViT-L", image, torch.device("cpu"),
                                               refine_steps=5)
        self.assertEqual(tuple(depth.shape), (64, 112))
        # the fake returns channel 0 of the [3, H, W] image it was handed
        self.assertTrue(torch.allclose(depth, image[0][..., 0]))
        call = fast_depth._GLOBAL_MOGE_MODEL.calls[-1]
        self.assertEqual(call["refine_steps"], 5)
        self.assertEqual(call["resolution_level"], fast_depth.MOGE_RESOLUTION_LEVEL)
        self.assertEqual(call["image"], (3, 64, 112))     # [3, H, W], not [1, H, W, 3]

    def test_the_model_is_loaded_once_and_reused(self):
        for _ in range(3):
            fast_depth._predict_moge_depth("MoGe-3-ViT-L", _gradient_image(),
                                           torch.device("cpu"))
        self.assertEqual(fast_depth._GLOBAL_MOGE_ID, enndee_moge.MOGE_V3_REPO)
        self.assertEqual(len(fast_depth._GLOBAL_MOGE_MODEL.calls), 3)   # 3 passes, 1 load

    def test_an_unknown_variant_is_rejected_before_any_load(self):
        with self.assertRaisesRegex(ValueError, "Unknown MoGe variant"):
            fast_depth._predict_moge_depth("MoGe-9-Nope", _gradient_image(),
                                           torch.device("cpu"))
        fast_depth.load_moge_v3.assert_not_called()

    def test_an_out_of_range_step_count_is_rejected_before_any_load(self):
        with self.assertRaisesRegex(ValueError, "moge_refine_steps"):
            fast_depth._predict_moge_depth("MoGe-3-ViT-L", _gradient_image(),
                                           torch.device("cpu"), refine_steps=99)
        fast_depth.load_moge_v3.assert_not_called()


class MogeDispatchTests(unittest.TestCase):
    """`render_depth_aligned` picks MoGe-3 by model id and passes the widget value through."""

    _OPTIONS = dict(frames=3, canvas_mode="custom", custom_width=112, custom_height=64,
                    cloud_scale=1, point_size=0, edge_cull=True, edge_threshold=0.30,
                    back_face_cull=False, camera=None, custom_camera=None)

    def test_moge_is_dispatched_with_the_widget_value(self):
        seen = []

        def fake(model_name, first, device, refine_steps=fast_depth.MOGE_REFINE_STEPS,
                 resolution_level=fast_depth.MOGE_RESOLUTION_LEVEL):
            seen.append((model_name, refine_steps))
            return torch.ones(64, 112), 42.5        # depth, measured vertical FOV

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), \
                mock.patch.object(fast_depth, "_predict_moge_depth_and_fov", fake):
            fast_depth.render_depth_aligned(_gradient_image(), torch.device("cpu"),
                                            model_size="MoGe-3-ViT-L",
                                            moge_refine_steps=7, **self._OPTIONS)
        self.assertEqual(seen, [("MoGe-3-ViT-L", 7)])
        self.assertIn("42.50", buffer.getvalue())   # the measured FOV reached the renderer

    def test_the_other_sources_ignore_the_refinement_widget(self):
        seen = []

        def fake_da3(model_name, first, device, process_res=fast_depth.DA3_RES):
            seen.append(model_name)
            return torch.ones(64, 112)

        with mock.patch.object(fast_depth, "_predict_da3_depth", fake_da3), \
                mock.patch.object(fast_depth, "_predict_moge_depth_and_fov",
                                  side_effect=AssertionError("MoGe must not run")):
            fast_depth.render_depth_aligned(_gradient_image(), torch.device("cpu"),
                                            model_size="Depth-Anything-3-Small",
                                            moge_refine_steps=7, **self._OPTIONS)
        self.assertEqual(seen, ["Depth-Anything-3-Small"])

    def test_an_out_of_range_widget_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "moge_refine_steps"):
            fast_depth.render_depth_aligned(_gradient_image(), torch.device("cpu"),
                                            model_size="MoGe-3-ViT-L",
                                            moge_refine_steps=99, **self._OPTIONS)


class InvalidDepthTests(unittest.TestCase):
    """MoGe marks unvouched pixels (sky) `inf`; the backend must turn that into the far plane.

    Measured on beach.jpg: `apply_mask=True`, the model's own default, leaves 20.4 % of the frame
    non-finite. Left alone that `inf` wins `cloud_gauge`'s 99th percentile and hands `inf`
    coordinates to the unprojection.
    """

    def test_a_finite_map_is_returned_untouched(self):
        depth = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertIs(fast_depth._fill_invalid_depth(depth), depth)   # identity, no copy

    def test_invalid_pixels_become_the_clips_far_end(self):
        depth = torch.tensor([[1.0, float("inf")], [3.0, float("nan")]])
        filled = fast_depth._fill_invalid_depth(depth)
        self.assertTrue(torch.isfinite(filled).all())
        # The 99 % percentile of the two measured pixels, 1 and 3 - never the raw maximum, which
        # would let a speck define the far plane (see the test below).
        self.assertAlmostEqual(filled[0, 1].item(), 2.98, places=5)
        self.assertAlmostEqual(filled[1, 1].item(), 2.98, places=5)
        self.assertEqual(filled[0, 0].item(), 1.0)     # valid pixels are untouched
        self.assertEqual(filled[1, 0].item(), 3.0)

    def test_the_fill_cannot_define_the_far_plane(self):
        """Found on a real still, and it is a *scale* error, not a cosmetic one.

        The fill covers 20.4 % of beach.jpg - twenty times more than the 1 % the 99th percentile
        needs - so the value it carries becomes `d_hi` and therefore the far plane of the window
        `render_depth_aligned` builds its cloud in. Filling at `valid.max()` stretched that window
        from 1.2 .. 36 to 1.2 .. 93 and squashed the measured scene into **1.000 .. 1.020** of the
        1.0 .. 5.0 window: a flat card, near-field depth contrast 1.016x against the 2.43x of
        Depth-Anything-3-Large, the source this backend was tuned on. The clip's own docstring
        says it exists so that "one speck cannot squash the range" - handing the raw maximum to a
        fifth of the frame is the one way to defeat it.

        Filling at the clip's own high end instead lands the fill *on* that value rather than past
        it, so the 99th percentile of the filled map is exactly the 99th percentile of the
        measured pixels, whatever the fill fraction.
        """
        depth = torch.linspace(1.0, 4.0, 1000).reshape(1, 1000).repeat(10, 1)   # 10 000 pixels
        depth[0, 0] = 500.0                     # a far speck the clip has to ignore
        depth[:, 600:] = float("inf")           # 40 % unvouched, like a sky-heavy still
        measured = depth[torch.isfinite(depth)]
        self.assertEqual(measured.numel(), 6000)

        filled = fast_depth._fill_invalid_depth(depth)
        self.assertTrue(torch.isfinite(filled).all())

        def percentile(map_, q):
            return float(fast_depth._flat_quantile(map_.reshape(-1), q))

        measured_far = percentile(measured, fast_depth.DEPTH_PCT_HI)
        self.assertLess(measured_far, 10.0)                      # the 500 speck is clipped away
        self.assertAlmostEqual(percentile(filled, fast_depth.DEPTH_PCT_HI), measured_far, places=5)

        # The consequence the eye reads: the scene must not be squashed into the bottom of the
        # window. With the fill at the raw maximum the measured median rendered at 1.007 - a flat
        # card - instead of ~2.98, so a quarter of the window separates the two cleanly. Ask
        # `cloud_gauge` itself rather than re-deriving the map here: the copy of the formula is how
        # the *shape* half of this bug (a subject 5.7x flatter than the model's own depth, see
        # `test_the_gauge_preserves_the_shape_of_the_scene` in the fast-depth module) survived a
        # passing check.
        near, far = fast_depth.DEPTH_NEAR, fast_depth.DEPTH_FAR
        mid = percentile(measured, 0.5)
        self.assertAlmostEqual(float(filled[5, 300]), mid, delta=0.02)   # a pixel at the median
        rendered_mid = float(fast_depth.cloud_gauge(filled)[5, 300])
        self.assertGreater(rendered_mid, near + 0.25 * (far - near))

    def test_the_fill_is_reported_rather_than_applied_silently(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            fast_depth._fill_invalid_depth(torch.tensor([[1.0, float("inf")]]))
        self.assertIn("unvouched", buffer.getvalue())

    def test_a_fully_invalid_map_is_an_error_not_a_render(self):
        with self.assertRaisesRegex(RuntimeError, "every pixel"):
            fast_depth._fill_invalid_depth(torch.full((2, 2), float("inf")))


if __name__ == "__main__":
    unittest.main()

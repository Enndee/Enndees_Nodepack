"""Tests for the Meridian fast-depth backend (`enndee_meridian_fast_depth`).

The depth model is faked by patching the module's `_get_depth_model` (Depth-Anything-V2) and
`_predict_da3_depth` (Depth-Anything-3) hooks, so the suite covers signal parsing, the arguments
parser, canvas bucketing, the ported camera math, depth inversion, the Meridian-style edge keep /
all-parents upsample, the working-still resolution cap, the strided percentile clip, the `--cull`
normals and the full unproject + point-render contract without downloading or loading any depth
model.

The V2 fake returns *inverse* depth like the real Depth-Anything-V2 (larger = closer): a near
pier post reads 4.4 while the far sky reads 0.3. The V3 fakes return true relative depth (larger
= farther), which is what the real Depth-Anything-3 models emit.
"""

import contextlib
import importlib.util
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PACK_DIR = Path(__file__).resolve().parents[1]
NODES_DIR = PACK_DIR / "nodes"
sys.path.insert(0, str(NODES_DIR))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402


class _FakeOutput:
    def __init__(self, depth):
        self.predicted_depth = depth


class _FakeDepthModel:
    """Deterministic stand-in for Depth-Anything-V2 (inverse depth: larger = closer).

    `disparity(fx, fy)` receives the 518-grid position as 0..1 fractions; the default is the
    gentle horizontal ramp the original tests used.
    """

    def __init__(self, disparity=None):
        self._disparity = disparity or (lambda fx, fy: 0.2 + 0.8 * fx)

    def __call__(self, pixel_values=None, **kwargs):
        height, width = pixel_values.shape[-2], pixel_values.shape[-1]
        fx = torch.linspace(0.0, 1.0, width, device=pixel_values.device, dtype=torch.float32)
        fy = torch.linspace(0.0, 1.0, height, device=pixel_values.device, dtype=torch.float32)
        rows = [torch.tensor([self._disparity(float(x), float(y)) for x in fx], dtype=torch.float32)
                for y in fy]
        return _FakeOutput(torch.stack(rows).unsqueeze(0))


def _gradient_image(height=64, width=112):
    yy = torch.linspace(0, 1, height).view(height, 1, 1)
    xx = torch.linspace(0, 1, width).view(1, width, 1)
    image = torch.cat([xx.expand(height, width, 1), yy.expand(height, width, 1),
                       (1 - xx).expand(height, width, 1)], dim=-1)
    return image.unsqueeze(0).contiguous()


def _two_key_path(frames=73, end=(0.5, 0.0, 0.5)):
    return {"frames": frames, "path": [
        {"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
        {"t": frames - 1, "src": frames - 1, "pos": list(end), "look": [0.0, 0.0, 1.0]},
    ]}


def _camera(*tokens):
    return fast_depth.parse_camera_settings(list(tokens))


class MeridianFastDepthHelperTests(unittest.TestCase):
    def test_bucket_480_matches_meridian_ladder(self):
        self.assertEqual(fast_depth._bucket_480(1920, 1080), (832, 480))
        self.assertEqual(fast_depth._bucket_480(512, 512), (640, 640))
        self.assertEqual(fast_depth._bucket_480(1080, 1920), (480, 832))

    def test_parse_camera_signal_round_trips_and_rejects_bad_payloads(self):
        data = _two_key_path()
        parsed, frames = fast_depth._parse_camera_signal(json.dumps(data))
        self.assertEqual(frames, 73)
        self.assertEqual(parsed["path"][1]["pos"], [0.5, 0.0, 0.5])

        with self.assertRaises(ValueError):
            fast_depth._parse_camera_signal(json.dumps({"frames": 71, "path": data["path"]}))
        with self.assertRaises(ValueError):
            fast_depth._parse_camera_signal(json.dumps({"frames": 73, "path": data["path"][:1]}))
        with self.assertRaises(ValueError):
            fast_depth._parse_camera_signal(json.dumps(
                {"frames": 73, "path": [{"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0]}]}))
        with self.assertRaises(ValueError):
            fast_depth._parse_camera_signal("not json")
        # real-time indexing and frame 0..frames-1 coverage, like the VGGT-side parser
        with self.assertRaises(ValueError):
            fast_depth._parse_camera_signal(json.dumps({"frames": 73, "path": [
                {"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
                {"t": 72, "src": 71, "pos": [0.5, 0.0, 0.5], "look": [0.0, 0.0, 1.0]}]}))
        with self.assertRaises(ValueError):
            fast_depth._parse_camera_signal(json.dumps({"frames": 73, "path": [
                {"t": 20, "src": 20, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
                {"t": 72, "src": 72, "pos": [0.5, 0.0, 0.5], "look": [0.0, 0.0, 1.0]}]}))

    def test_parse_camera_settings_reads_values_flags_and_inline_forms(self):
        settings = _camera("--frames", "90", "--yaw-from", "-15", "--yaw", "15", "--truck=0.25",
                           "--boom", "-0.5", "--dolly", "0.8", "--zoom", "1.6", "--sweep",
                           "--ease", "--aim", "--pivot", "0.4,0.55", "--pivot-to", "0.6,0.5",
                           "--pivot-lock", "--fast-back", "2", "--cull")
        self.assertEqual(settings["frames"], 90)
        self.assertEqual(settings["yaw_from"], -15.0)
        self.assertEqual(settings["yaw"], 15.0)
        self.assertEqual(settings["truck"], 0.25)
        self.assertEqual(settings["boom"], -0.5)
        self.assertEqual(settings["dolly"], 0.8)
        self.assertEqual(settings["zoom"], 1.6)
        self.assertTrue(settings["sweep"] and settings["ease"] and settings["aim"])
        self.assertEqual(settings["pivot"], "0.4,0.55")
        self.assertEqual(settings["pivot_to"], "0.6,0.5")
        self.assertTrue(settings["pivot_lock"] and settings["cull"])
        self.assertEqual(settings["fast_back"], 2.0)
        self.assertFalse(settings["follow"])

    def test_parse_camera_settings_rejects_bad_values_and_reports_follow(self):
        with self.assertRaises(ValueError):
            _camera("--yaw")
        with self.assertRaises(ValueError):
            _camera("--frames", "ninety")
        with self.assertRaises(ValueError):
            _camera("--truck", "left")
        self.assertTrue(_camera("--follow")["follow"])

    def test_invert_disparity_puts_near_content_at_small_z(self):
        pred = torch.tensor([0.3, 1.0, 4.4])   # sky, mid, near post (inverse depth)
        depth = fast_depth._invert_disparity(pred)
        self.assertGreater(float(depth[0]), float(depth[1]))     # the sky ends up far
        self.assertGreater(float(depth[1]), float(depth[2]))     # the post ends up near
        self.assertLess(float(depth[2]), 0.25)

    def test_da3_process_res_zero_means_the_stills_own_resolution(self):
        self.assertEqual(fast_depth._da3_process_res(0, 1024, 1536), 1536)
        self.assertEqual(fast_depth._da3_process_res(-4, 800, 600), 800)     # defended at the backend
        self.assertEqual(fast_depth._da3_process_res(504, 1024, 1536), 504)  # the fast default
        self.assertEqual(fast_depth._da3_process_res(1008, 64, 112), 1008)   # explicit, even upscaling

    def test_fit_working_still_caps_the_longest_side_and_never_upscales(self):
        still = _gradient_image(height=400, width=600)
        capped, note = fast_depth._fit_working_still(still, 300)
        self.assertEqual(tuple(capped.shape), (1, 200, 300, 3))              # 600x400 -> 300x200
        self.assertEqual(note, "still 600x400 -> 300x200 (depth_res 300 cap)")
        wide, note = fast_depth._fit_working_still(still, 2048)              # above the own side
        self.assertIs(wide, still)                                           # never upscales
        self.assertIsNone(note)
        own, note = fast_depth._fit_working_still(still, 0)                  # 0 = the own resolution
        self.assertIs(own, still)
        self.assertIsNone(note)

    def test_fit_working_still_falls_back_to_the_cloud_ceiling_with_depth_res_zero(self):
        # MAX_CLOUD_PIXELS is patched small so the 2**24-pixel ceiling is testable on a tiny still.
        with mock.patch.object(fast_depth, "MAX_CLOUD_PIXELS", 1000):
            small, note = fast_depth._fit_working_still(torch.zeros(1, 100, 200, 3), 0)
        self.assertEqual(tuple(small.shape), (1, 22, 44, 3))                 # both sides * sqrt(1000/20000)
        self.assertLessEqual(22 * 44, 1000)
        self.assertEqual(note, "still 200x100 -> 44x22 (MAX_CLOUD_PIXELS 1000 cap)")

    def test_flat_quantile_strides_pools_over_the_aten_element_limit(self):
        ramp = torch.arange(1, 4097, dtype=torch.float32)
        self.assertEqual(float(fast_depth._flat_quantile(ramp, 0.5)),
                         float(torch.quantile(ramp, 0.5)))                   # small pools stay exact
        huge = torch.arange((1 << 24) + 2, dtype=torch.float32)              # just past the ceiling
        with self.assertRaises(RuntimeError):
            torch.quantile(huge, 0.5)                                        # the crash this helper prevents
        self.assertAlmostEqual(float(fast_depth._flat_quantile(huge, 0.5)),
                               (huge.numel() - 1) / 2, delta=4.0)            # the stride keeps the median

    def test_edge_keep_culls_only_the_depth_step(self):
        depth = torch.ones(9, 9)
        depth[:, 5:] = 3.0
        keep = fast_depth._edge_keep(depth, 0.30)
        self.assertTrue(bool(keep[:, :4].all()))
        self.assertTrue(bool(keep[:, 6:].all()))
        self.assertFalse(bool(keep[:, 4:6].any()))   # the 3x3 window sees the jump

    def test_upsample_depth_keep_requires_every_parent(self):
        depth = torch.ones(2, 2)
        keep = torch.tensor([[False, True], [True, True]])
        dense_depth, dense_keep = fast_depth._upsample_depth_keep(depth, keep, 4, 4)
        self.assertEqual(tuple(dense_depth.shape), (4, 4))
        self.assertFalse(bool(dense_keep[0, 0]))     # a killed parent drops the whole corner
        self.assertTrue(bool(dense_keep[-1, -1]))    # fully-kept parents survive
        _, all_kept = fast_depth._upsample_depth_keep(depth, torch.ones(2, 2, dtype=torch.bool), 4, 4)
        self.assertTrue(bool(all_kept.all()))

    def test_cull_normals_face_the_source_camera(self):
        yy, xx = torch.meshgrid(torch.arange(4, dtype=torch.float32),
                                torch.arange(4, dtype=torch.float32), indexing="ij")
        plane = torch.stack([xx - 1.5, yy - 1.5, torch.full_like(xx, 2.0)], dim=-1)
        normals = fast_depth._orient_toward_source(plane)
        self.assertEqual(tuple(normals.shape), (16, 3))
        toward = (normals * -plane.reshape(-1, 3)).sum(-1)
        self.assertTrue(bool((toward > 0).all()))          # every normal points at the origin
        self.assertTrue(bool((normals[:, 2] < 0).all()))   # i.e. against the plane's +z side

    def test_custom_path_starts_at_identity_like_the_source_camera(self):
        c2w, focal = fast_depth._evaluate_camera_path(_two_key_path(), 73, 2.0)
        self.assertEqual(c2w.shape, (73, 4, 4))
        np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(focal, np.ones(73, dtype=np.float32))

    def test_parametric_orbit_keeps_the_pivot_on_the_optical_axis(self):
        device = torch.device("cpu")
        pivot = torch.tensor([0.0, 0.0, 2.0])
        c2w, _ = fast_depth._build_parametric_c2w(3, pivot, 2.0, yaw=90.0, truck=0.0, boom=0.0,
                                                  dolly=1.0, sweep=True, ease=False, aim=False,
                                                  device=device)
        for frame in (1, 2):
            self.assertAlmostEqual(float((c2w[frame, :3, 3] - pivot).norm()), 2.0, places=4)

    def test_truck_shifts_the_pivot_and_aim_recenters_it(self):
        device = torch.device("cpu")
        pivot = torch.tensor([0.0, 0.0, 2.0])

        def pivot_in_camera(c2w):
            rotation = c2w[:3, :3]
            return rotation.transpose(0, 1) @ pivot - rotation.transpose(0, 1) @ c2w[:3, 3]

        shifted, _ = fast_depth._build_parametric_c2w(3, pivot, 2.0, yaw=0.0, truck=0.4, boom=0.0,
                                                      dolly=1.0, sweep=False, ease=False, aim=False,
                                                      device=device)
        # sample.py moves the camera by +truck*zm in the rotated frame, so the pivot sits at -0.8 in camera x
        self.assertAlmostEqual(float(pivot_in_camera(shifted[-1])[0]), -0.8, places=4)

        aimed, _ = fast_depth._build_parametric_c2w(3, pivot, 2.0, yaw=0.0, truck=0.4, boom=0.0,
                                                    dolly=1.0, sweep=False, ease=False, aim=True,
                                                    device=device)
        aim_cam = pivot_in_camera(aimed[-1])
        self.assertAlmostEqual(float(aim_cam[0]), 0.0, places=4)
        self.assertAlmostEqual(float(aim_cam[1]), 0.0, places=4)
        self.assertAlmostEqual(float(aim_cam[2]), float(np.hypot(0.8, 2.0)), places=4)

    def test_yaw_from_sweep_and_zoom_match_sample_py(self):
        device = torch.device("cpu")
        pivot = torch.tensor([0.0, 0.0, 2.0])
        c2w, focal = fast_depth._build_parametric_c2w(3, pivot, 2.0, yaw=15.0, truck=0.0, boom=0.0,
                                                      dolly=1.0, sweep=True, ease=False, aim=False,
                                                      device=device, yaw_from=-15.0, zoom=1.6)
        # sample.py ramps the yaw from --yaw-from to --yaw: frame 0 sits at -15 degrees ...
        self.assertAlmostEqual(float(c2w[0, 0, 0]), float(np.cos(np.radians(-15.0))), places=4)
        # ... the middle frame at exactly 0 degrees (identity rotation) ...
        np.testing.assert_allclose(c2w[1, :3, :3].numpy(), np.eye(3, dtype=np.float32), atol=1e-5)
        # ... and --zoom ramps the focal multiplier from 1 to its final value
        self.assertAlmostEqual(float(focal[0]), 1.0, places=5)
        self.assertAlmostEqual(float(focal[-1]), 1.6, places=5)

    def test_bounce_and_swing_shape_the_ramp(self):
        device = torch.device("cpu")
        pivot = torch.tensor([0.0, 0.0, 2.0])
        bounce, _ = fast_depth._build_parametric_c2w(5, pivot, 2.0, yaw=90.0, truck=0.0, boom=0.0,
                                                     dolly=1.0, sweep=False, ease=False, aim=False,
                                                     device=device, bounce=True)
        np.testing.assert_allclose(bounce[0].numpy(), np.eye(4, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(bounce[-1].numpy(), np.eye(4, dtype=np.float32), atol=1e-5)
        self.assertGreater(float((bounce[2, :3, 3] - pivot).norm()), 0.5)   # out on the arc mid-way

        swing, _ = fast_depth._build_parametric_c2w(5, pivot, 2.0, yaw=90.0, truck=0.0, boom=0.0,
                                                    dolly=1.0, sweep=False, ease=False, aim=False,
                                                    device=device, swing=True)
        np.testing.assert_allclose(swing[2].numpy(), np.eye(4, dtype=np.float32), atol=1e-5)
        self.assertAlmostEqual(float(swing[1, 0, 3]), -float(swing[3, 0, 3]), places=4)  # mirrored sides
        self.assertAlmostEqual(float(swing[1, 2, 3]), float(swing[3, 2, 3]), places=4)

class MeridianFastDepthEngineTests(unittest.TestCase):
    def _generate(self, image=None, model=None, da3_depth=None, camera=None, custom_camera=None, **overrides):
        options = dict(
            model_size="Depth-Anything-V2-Small-hf", frames=73, canvas_mode="custom",
            custom_width=112, custom_height=64, cloud_scale=1, point_size=0,
            edge_cull=True, edge_threshold=0.30, back_face_cull=False,
        )
        options.update(overrides)
        with mock.patch.object(fast_depth, "_get_depth_model",
                               lambda *args, **kwargs_: model or _FakeDepthModel()), \
             mock.patch.object(fast_depth, "_predict_da3_depth",
                               lambda *args, **kwargs_: da3_depth):
            return fast_depth.render_depth_aligned(
                image if image is not None else _gradient_image(), torch.device("cpu"),
                camera=camera, custom_camera=custom_camera, **options)

    def test_static_camera_reproduces_the_depth_aligned_texture(self):
        source, render, width, height, length = self._generate()
        self.assertEqual((width, height, length), (112, 64, 73))
        self.assertEqual(tuple(source.shape), (73, 64, 112, 3))
        self.assertEqual(tuple(render.shape), (73, 64, 112, 3))
        for frame in (0, -1):
            difference = (render[frame] - source[frame]).abs().max().item()
            self.assertLess(difference, 0.005)

    def test_yaw_flight_moves_the_view_and_keeps_coverage(self):
        source, render, width, height, length = self._generate(
            camera=_camera("--yaw", "15", "--sweep", "--ease"), point_size=1)
        self.assertEqual((width, height, length), (112, 64, 73))
        movement = (render[-1] - render[0]).abs().mean().item()
        self.assertGreater(movement, 0.01)
        holes = render[-1].mean(dim=-1).sub(128 / 255).abs() < 1e-6
        self.assertLess(holes.float().mean().item(), 0.6)

    def test_custom_camera_signal_drives_the_frame_count(self):
        signal = json.dumps(_two_key_path(frames=90, end=(0.6, 0.1, 0.6)))
        source, render, width, height, length = self._generate(
            custom_camera=signal, canvas_mode="auto_meridian480", custom_width=0,
            custom_height=0, point_size=1, cloud_scale=2)
        self.assertEqual(length, 90)
        self.assertEqual(tuple(source.shape), (90, 480, 832, 3))
        self.assertEqual(tuple(render.shape), (90, 480, 832, 3))
        self.assertEqual((width, height), (832, 480))

    def test_invalid_pivot_string_is_rejected(self):
        with self.assertRaises(ValueError):
            self._generate(camera=_camera("--pivot", "not-a-pair"))

    def test_disparity_step_places_the_near_half_at_the_pivot_depth(self):
        step = _FakeDepthModel(disparity=lambda fx, fy: 4.0 if fx < 0.5 else 0.2)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            source, render, width, height, length = self._generate(
                model=step, camera=_camera("--pivot", "0.25,0.5"))
        text = buffer.getvalue()
        # `raw` is the model's own gauge - it says *which* depth the +-5 % window picked - while
        # `zm` is that value re-gauged into the cloud's DEPTH_NEAR..DEPTH_FAR window.
        self.assertIn("raw 0.250", text)   # 1/4.0 = 0.25: the near half is the pivot depth
        self.assertNotIn("raw 4", text)    # a raw (uninverted) disparity would report 4.000
        self.assertEqual(length, 73)

    def test_back_face_cull_empties_a_flat_plane_viewed_from_behind(self):
        flat = _FakeDepthModel(disparity=lambda fx, fy: 1.0)
        _, without, _, _, _ = self._generate(model=flat, camera=_camera("--yaw", "170"),
                                             point_size=1)
        _, with_cull, _, _, _ = self._generate(model=flat, camera=_camera("--yaw", "170", "--cull"),
                                               point_size=1)
        hole = 128 / 255
        without_holes = (without[-1].mean(dim=-1).sub(hole).abs() < 1e-6).float().mean().item()
        culled_holes = (with_cull[-1].mean(dim=-1).sub(hole).abs() < 1e-6).float().mean().item()
        self.assertLess(without_holes, 0.5)
        self.assertGreater(culled_holes, 0.95)  # a 180-degree view is a hole, not the mirrored front

    def test_back_face_cull_widget_matches_the_args_flag(self):
        flat = _FakeDepthModel(disparity=lambda fx, fy: 1.0)
        _, widget_cull, _, _, _ = self._generate(model=flat, camera=_camera("--yaw", "170"),
                                                 point_size=1, back_face_cull=True)
        _, args_cull, _, _, _ = self._generate(model=flat, camera=_camera("--yaw", "170", "--cull"),
                                               point_size=1)
        torch.testing.assert_close(widget_cull, args_cull)

    def test_da3_models_use_true_depth_without_the_disparity_inversion(self):
        # Depth-Anything-3 predicts depth directly: left half near (0.25), right half far (4.0).
        columns = torch.linspace(0.0, 1.0, 21).view(1, 21).expand(7, 21)
        da3_depth = torch.where(columns < 0.5, torch.tensor(0.25), torch.tensor(4.0))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            source, render, width, height, length = self._generate(
                model_size="Depth-Anything-3-Small", da3_depth=da3_depth,
                camera=_camera("--pivot", "0.25,0.5"))
        text = buffer.getvalue()
        self.assertIn("raw 0.250", text)    # the near half is the pivot depth, unconverted
        self.assertNotIn("raw 4", text)     # an inversion would report 4.000
        self.assertEqual((width, height, length), (112, 64, 73))
        self.assertEqual(tuple(source.shape), (73, 64, 112, 3))
        self.assertEqual(tuple(render.shape), (73, 64, 112, 3))

    def test_da3_pivot_window_maps_onto_the_grid_height(self):
        # A 7x21 grid whose value is the row index + 1. The +-5 % window around fy=0.75 covers
        # rows 4..5; torch.median takes the lower middle value, so the raw pivot depth prints as
        # `raw 5.000`. A square-grid mapping (the old `res = shape[-1]`) would read rows past the
        # end, fall back to the global median and print `raw 4.000` instead. The 3x3 edge rule is
        # off because a 2-step row jump exceeds EDGE_RTOL and would cull every non-last row.
        rows = torch.arange(1.0, 8.0).view(7, 1).expand(7, 21).contiguous()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._generate(model_size="Depth-Anything-3-Small", da3_depth=rows, edge_cull=False,
                           camera=_camera("--pivot", "0.5,0.75"))
        self.assertIn("raw 5.000", buffer.getvalue())

    def test_zm_is_the_cloud_gauge_not_the_raw_model_depth(self):
        # The camera keys arrive in "median-depth units" (1.0 = the cloud's median depth), so the
        # scale that turns them into world coordinates has to be the median of the *rendered*
        # cloud - the DEPTH_NEAR..DEPTH_FAR map - and not the raw model depth. Scaling by the raw
        # median left the whole rig ~2x too close to the origin: the orbit centre then sat in
        # front of the subject and the subject swung out of the picture along the path.
        columns = torch.linspace(0.0, 1.0, 21).view(1, 21).expand(7, 21)
        da3_depth = torch.where(columns < 0.5, torch.tensor(0.25), torch.tensor(4.0))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._generate(model_size="Depth-Anything-3-Small", da3_depth=da3_depth)
        match = re.search(r"zm ([0-9.]+) \[raw ([0-9.]+)\]", buffer.getvalue())
        self.assertIsNotNone(match, buffer.getvalue())
        zm, raw = (float(value) for value in match.groups())
        self.assertNotEqual(zm, raw)                                  # the two gauges differ
        self.assertGreaterEqual(zm, fast_depth.DEPTH_NEAR - 1e-6)     # cloud gauge, not raw
        self.assertLessEqual(zm, fast_depth.DEPTH_FAR + 1e-6)

    def test_da3_depth_resolution_reaches_the_model_with_the_stills_own_side(self):
        # render_depth_aligned forwards `depth_res` as the exact DA3 `process_res`, except for 0,
        # which turns into the still's own longest side (the default 64x112 fake -> 112).
        seen = []

        def fake(model_name, first, device, process_res=fast_depth.DA3_RES):
            seen.append(process_res)
            return torch.ones(7, 21)

        options = dict(model_size="Depth-Anything-3-Small", frames=73, canvas_mode="custom",
                       custom_width=112, custom_height=64, cloud_scale=1, point_size=0,
                       edge_cull=True, edge_threshold=0.30, back_face_cull=False)
        with mock.patch.object(fast_depth, "_predict_da3_depth", fake):
            for overrides in (dict(), dict(depth_res=0), dict(depth_res=1008)):
                fast_depth.render_depth_aligned(_gradient_image(), torch.device("cpu"),
                                                camera=None, custom_camera=None,
                                                **options, **overrides)
        self.assertEqual(seen, [fast_depth.DA3_RES, 112, 1008])

    def test_big_stills_are_resized_to_the_depth_res_cap_before_the_model(self):
        # A 24 Mpx photo used to crash the percentile clip (torch.quantile rejects pools over
        # 2**24 elements); with the working-still cap the 3.8 Mpx test still reaches the depth
        # model already resized, and the whole flight runs off the smaller working still.
        seen = []

        def fake(model_name, first, device, process_res=fast_depth.DA3_RES):
            seen.append((tuple(first.shape), process_res))
            return torch.ones(7, 21)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), \
                mock.patch.object(fast_depth, "_predict_da3_depth", fake):
            source, render, width, height, length = fast_depth.render_depth_aligned(
                _gradient_image(height=1600, width=2400), torch.device("cpu"),
                model_size="Depth-Anything-3-Small", frames=3, canvas_mode="custom",
                custom_width=112, custom_height=64, cloud_scale=1, point_size=0,
                edge_cull=True, edge_threshold=0.30, back_face_cull=False,
                camera=None, custom_camera=None, depth_res=400)
        self.assertEqual(seen, [((1, 266, 400, 3), 400)])                    # model sees the capped still
        self.assertEqual((width, height, length), (112, 64, 3))
        self.assertEqual(tuple(source.shape), (3, 64, 112, 3))
        self.assertEqual(tuple(render.shape), (3, 64, 112, 3))
        self.assertIn("still 2400x1600 -> 400x266 (depth_res 400 cap)", buffer.getvalue())

    def test_da3_variants_map_to_hugging_face_repos_and_unknown_ones_fail_loudly(self):
        self.assertEqual(fast_depth.DA3_MODEL_REPOS["Depth-Anything-3-Small"],
                         "depth-anything/DA3-SMALL")
        self.assertIn("Depth-Anything-3-Mono-Large", fast_depth.DA3_MODEL_REPOS)
        self.assertTrue(all(repo.startswith("depth-anything/")
                            for repo in fast_depth.DA3_MODEL_REPOS.values()))
        with self.assertRaisesRegex(ValueError, "Unknown Depth-Anything-3 variant"):
            fast_depth._predict_da3_depth("Depth-Anything-3-Giant", torch.zeros(1, 4, 4, 3),
                                          torch.device("cpu"))

    def test_da3_api_loads_with_the_lightweight_stubs(self):
        if importlib.util.find_spec("depth_anything_3") is None:
            self.skipTest("depth-anything-3 is not installed in this interpreter")
        DepthAnything3 = fast_depth._load_da3_api()
        self.assertTrue(callable(DepthAnything3.from_pretrained))
        # The export dispatcher and pose alignment stay stubbed out (their deps are absent);
        # both are unreachable from the fast-depth inference path.
        with self.assertRaises(ImportError):
            sys.modules["depth_anything_3.utils.export"].export(None, "glb", "out")
        with self.assertRaises(ImportError):
            sys.modules["depth_anything_3.utils.pose_align"].align_poses_umeyama(None, None)


if __name__ == "__main__":
    unittest.main()

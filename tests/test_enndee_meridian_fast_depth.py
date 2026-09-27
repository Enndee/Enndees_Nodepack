"""Tests for the Meridian fast-depth backend (`enndee_meridian_fast_depth`).

The depth model is faked by patching the module's `_get_depth_model`, so the suite covers
signal parsing, the arguments parser, canvas bucketing, the ported camera math, depth
inversion, the Meridian-style edge keep / all-parents upsample, the `--cull` normals and the
full unproject + point-render contract without downloading or loading Depth-Anything-V2.

The fake model returns *inverse* depth like the real Depth-Anything-V2 (larger = closer):
a near pier post reads 4.4 while the far sky reads 0.3.
"""

import contextlib
import io
import json
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
    def _generate(self, image=None, model=None, camera=None, custom_camera=None, **overrides):
        options = dict(
            model_size="Depth-Anything-V2-Small-hf", frames=73, canvas_mode="custom",
            custom_width=112, custom_height=64, cloud_scale=1, point_size=0,
            edge_cull=True, edge_threshold=0.30, back_face_cull=False,
        )
        options.update(overrides)
        with mock.patch.object(fast_depth, "_get_depth_model",
                               lambda *args, **kwargs_: model or _FakeDepthModel()):
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
        self.assertIn("zm 0.2", text)      # 1/4.0 ~= 0.25: the near half is the pivot depth
        self.assertNotIn("zm 4", text)     # a raw (uninverted) disparity would report 4.000
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


if __name__ == "__main__":
    unittest.main()

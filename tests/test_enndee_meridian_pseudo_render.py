"""Tests for the Meridian Fast Depth Splat node (`Enndee_MeridianPseudoRender`).

The depth model is faked by patching the node module's `_get_depth_model`, so the suite
covers signal parsing, canvas bucketing, the ported camera math, the unproject + splat
contract and the output signature without downloading or loading Depth-Anything-V2.
"""

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

import enndee_meridian_pseudo_render as pseudo  # noqa: E402


class _FakeOutput:
    def __init__(self, depth):
        self.predicted_depth = depth


class _FakeDepthModel:
    """Deterministic stand-in for Depth-Anything-V2: a horizontal depth ramp, no weights."""

    def __call__(self, pixel_values=None, **kwargs):
        height, width = pixel_values.shape[-2], pixel_values.shape[-1]
        ramp = torch.linspace(0.2, 1.0, width, device=pixel_values.device, dtype=torch.float32)
        depth = ramp.view(1, 1, width).expand(1, height, width).contiguous()
        return _FakeOutput(depth)


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

class MeridianPseudoRenderHelperTests(unittest.TestCase):
    def test_bucket_480_matches_meridian_ladder(self):
        self.assertEqual(pseudo._bucket_480(1920, 1080), (832, 480))
        self.assertEqual(pseudo._bucket_480(512, 512), (640, 640))
        self.assertEqual(pseudo._bucket_480(1080, 1920), (480, 832))

    def test_parse_camera_signal_round_trips_and_rejects_bad_payloads(self):
        data = _two_key_path()
        parsed, frames = pseudo._parse_camera_signal(json.dumps(data))
        self.assertEqual(frames, 73)
        self.assertEqual(parsed["path"][1]["pos"], [0.5, 0.0, 0.5])

        with self.assertRaises(ValueError):
            pseudo._parse_camera_signal(json.dumps({"frames": 71, "path": data["path"]}))
        with self.assertRaises(ValueError):
            pseudo._parse_camera_signal(json.dumps({"frames": 73, "path": data["path"][:1]}))
        with self.assertRaises(ValueError):
            pseudo._parse_camera_signal(json.dumps(
                {"frames": 73, "path": [{"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0]}]}))
        with self.assertRaises(ValueError):
            pseudo._parse_camera_signal("not json")

    def test_custom_path_starts_at_identity_like_the_source_camera(self):
        c2w, focal = pseudo._evaluate_camera_path(_two_key_path(), 73, 2.0)
        self.assertEqual(c2w.shape, (73, 4, 4))
        np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(focal, np.ones(73, dtype=np.float32))

    def test_parametric_orbit_keeps_the_pivot_on_the_optical_axis(self):
        device = torch.device("cpu")
        pivot = torch.tensor([0.0, 0.0, 2.0])
        c2w, _ = pseudo._build_parametric_c2w(9, pivot, 2.0, yaw=30.0, truck=0.0, boom=0.0, dolly=1.0,
                                              sweep=True, ease=False, aim=False, device=device)
        self.assertTrue(torch.allclose(c2w[0], torch.eye(4)))
        for frame in range(9):
            rotation = c2w[frame, :3, :3]
            cam = rotation.transpose(0, 1) @ pivot - rotation.transpose(0, 1) @ c2w[frame, :3, 3]
            self.assertAlmostEqual(float(cam[0]), 0.0, places=4)
            self.assertAlmostEqual(float(cam[1]), 0.0, places=4)
            self.assertGreater(float(cam[2]), 1.0)
            self.assertAlmostEqual(float((c2w[frame, :3, 3] - pivot).norm()), 2.0, places=4)

    def test_truck_shifts_the_pivot_and_aim_recenters_it(self):
        device = torch.device("cpu")
        pivot = torch.tensor([0.0, 0.0, 2.0])

        def pivot_in_camera(c2w):
            rotation = c2w[:3, :3]
            return rotation.transpose(0, 1) @ pivot - rotation.transpose(0, 1) @ c2w[:3, 3]

        shifted, _ = pseudo._build_parametric_c2w(3, pivot, 2.0, yaw=0.0, truck=0.4, boom=0.0, dolly=1.0,
                                                  sweep=False, ease=False, aim=False, device=device)
        # sample.py moves the camera by +truck*zm in the rotated frame, so the pivot sits at -0.8 in camera x
        self.assertAlmostEqual(float(pivot_in_camera(shifted[-1])[0]), -0.8, places=4)

        aimed, _ = pseudo._build_parametric_c2w(3, pivot, 2.0, yaw=0.0, truck=0.4, boom=0.0, dolly=1.0,
                                                sweep=False, ease=False, aim=True, device=device)
        aim_cam = pivot_in_camera(aimed[-1])
        self.assertAlmostEqual(float(aim_cam[0]), 0.0, places=4)
        self.assertAlmostEqual(float(aim_cam[1]), 0.0, places=4)
        self.assertAlmostEqual(float(aim_cam[2]), float(np.hypot(0.8, 2.0)), places=4)

class MeridianPseudoRenderNodeTests(unittest.TestCase):
    def _generate(self, **overrides):
        node = pseudo.EnndeeMeridianPseudoRender()
        kwargs = dict(
            image=_gradient_image(), model_size="Depth-Anything-V2-Small-hf", frames="73",
            canvas_mode="custom", custom_width=112, custom_height=64, cloud_scale=1, splat_size=0,
            edge_cull=True, edge_threshold=0.30, yaw=0.0, sweep=True, ease=True,
        )
        kwargs.update(overrides)
        with mock.patch.object(pseudo, "_get_depth_model", lambda *args, **kwargs_: _FakeDepthModel()):
            return node.generate(**kwargs)

    def test_static_camera_reproduces_the_depth_aligned_texture(self):
        source, render, width, height, length = self._generate()
        self.assertEqual((width, height, length), (112, 64, 73))
        self.assertEqual(tuple(source.shape), (73, 64, 112, 3))
        self.assertEqual(tuple(render.shape), (73, 64, 112, 3))
        for frame in (0, -1):
            difference = (render[frame] - source[frame]).abs().max().item()
            self.assertLess(difference, 0.005)

    def test_yaw_flight_moves_the_view_and_keeps_coverage(self):
        source, render, width, height, length = self._generate(yaw=15.0, splat_size=1)
        self.assertEqual((width, height, length), (112, 64, 73))
        movement = (render[-1] - render[0]).abs().mean().item()
        self.assertGreater(movement, 0.01)
        holes = render[-1].mean(dim=-1).sub(128 / 255).abs() < 1e-6
        self.assertLess(holes.float().mean().item(), 0.6)

    def test_custom_camera_signal_drives_the_frame_count(self):
        signal = json.dumps(_two_key_path(frames=90, end=(0.6, 0.1, 0.6)))
        source, render, width, height, length = self._generate(
            frames="73", custom_camera=signal, canvas_mode="auto_meridian480", custom_width=0,
            custom_height=0, splat_size=1)
        self.assertEqual(length, 90)
        self.assertEqual(tuple(source.shape), (90, 480, 832, 3))
        self.assertEqual(tuple(render.shape), (90, 480, 832, 3))
        self.assertEqual((width, height), (832, 480))

    def test_invalid_pivot_string_is_rejected(self):
        with self.assertRaises(ValueError):
            self._generate(pivot="not-a-pair")


if __name__ == "__main__":
    unittest.main()



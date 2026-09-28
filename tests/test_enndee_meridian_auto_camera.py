"""Tests for the automatic Meridian camera estimator; no depth model is loaded.

The surface comes from synthetic depth maps (a flat background with a near "subject" square),
so the orbit geometry, the subject heuristic and the speed cap are pinned exactly.
"""

import json
import math
import sys
import unittest
from pathlib import Path

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402
from enndee_meridian_auto_camera import (  # noqa: E402
    DEFAULT_MAX_SPEED,
    MAX_AZIMUTH_PER_FRAME,
    SCENE_TARGET,
    SUBJECT_TARGET,
    estimate_camera_path,
    format_summary,
    sphere_of,
    speed_limited_span,
    split_subject,
    surface_points,
)


def _reference(height=64, width=64):
    return torch.zeros(1, height, width, 3)


def _depth_with_subject(height=64, width=64, background=4.0, subject=1.0, size=24):
    """Flat background with a square 'subject' in the middle, `size` pixels wide."""
    depth = torch.full((height, width), background)
    top, left = (height - size) // 2, (width - size) // 2
    depth[top:top + size, left:left + size] = subject
    return depth


def _estimate(depth, **overrides):
    options = dict(frames=73, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED,
                   depth_fn=lambda reference: depth)
    options.update(overrides)
    return estimate_camera_path(_reference(), **options)


class AutoCameraSurfaceTests(unittest.TestCase):
    def test_surface_points_unproject_like_the_renderer(self):
        depth = torch.full((8, 16), 2.5)
        points = surface_points(depth)
        self.assertEqual(tuple(points.shape), (128, 3))
        self.assertTrue(torch.allclose(points[:, 2], torch.full((128,), 2.5)))
        # the renderer's convention: pixel centres at (width - 1) / 2, so an even grid sits
        # half a pixel left of the optical axis - mirrored exactly here
        focal = 0.5 * 8 / math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0)
        self.assertAlmostEqual(float(points[:, 0].mean()), -0.5 / focal * 2.5, places=5)
        self.assertAlmostEqual(float(points[:, 1].mean()), -0.5 / focal * 2.5, places=5)
        self.assertGreater(float(points[:, 0].max()), float(points[:, 0].min()))

    def test_split_subject_finds_the_near_depth_layer(self):
        points = surface_points(_depth_with_subject())
        near, label = split_subject(points)
        self.assertIn("near depth layer", label)
        self.assertLess(float(near[:, 2].median()), 1.5)          # the near square
        self.assertLessEqual(near.shape[0], points.shape[0] // 2)

    def test_sphere_of_is_centred_on_the_subject(self):
        near, _ = split_subject(surface_points(_depth_with_subject()))
        centre, radius = sphere_of(near)
        self.assertAlmostEqual(float(centre[2]), 1.0, places=3)   # the subject depth
        self.assertGreater(radius, 0.0)
        self.assertLess(radius, float(centre[2]))                 # a square narrower than its depth


class AutoCameraPathTests(unittest.TestCase):
    def test_subject_orbit_wraps_the_sphere_and_respects_the_caps(self):
        document, summary = _estimate(_depth_with_subject(), max_speed=0.5)
        payload = json.loads(document)
        self.assertEqual(payload["frames"], 73)
        keys = payload["path"]
        self.assertGreaterEqual(len(keys), 5)
        self.assertLessEqual(len(keys), 21)
        self.assertEqual(keys[0]["t"], 0)
        self.assertEqual(keys[-1]["t"], 72)
        ticks = [key["t"] for key in keys]
        self.assertEqual(ticks, sorted(ticks))
        self.assertEqual([key["src"] for key in keys], ticks)
        centre = torch.tensor(summary["centre"])
        radius = summary["orbit_radius"]
        for key in keys:
            distance = float((torch.tensor(key["pos"]) - centre).norm())
            self.assertAlmostEqual(distance, radius, delta=1e-3 * max(1.0, radius))
        self.assertGreater(float(torch.tensor(keys[0]["pos"])[2]), float(centre[2]))
        self.assertFalse(torch.allclose(torch.tensor(keys[0]["pos"]),
                                        torch.tensor(keys[-1]["pos"])))
        self.assertLessEqual(summary["azimuth_per_frame"], MAX_AZIMUTH_PER_FRAME + 1e-9)
        self.assertLessEqual(summary["travel_per_frame"], 0.5 * summary["content_radius"] + 1e-9)
        self.assertFalse(summary["speed_limited"])
        self.assertAlmostEqual(summary["swing"], 360.0, places=6)
        self.assertIn("subject", format_summary(summary))

    def test_low_speed_budget_shortens_the_swing_and_says_so(self):
        _, summary = _estimate(_depth_with_subject(), max_speed=0.05)
        self.assertTrue(summary["speed_limited"])
        self.assertLess(summary["swing"], 360.0)
        self.assertGreater(summary["swing"], 0.0)
        self.assertIn("swing shortened", format_summary(summary))
        self.assertLessEqual(summary["travel_per_frame"],
                             0.05 * summary["content_radius"] + 1e-9)

    def test_few_frames_also_shorten_the_swing(self):
        _, many = _estimate(_depth_with_subject(), frames=243, max_speed=0.05)
        _, few = _estimate(_depth_with_subject(), frames=73, max_speed=0.05)
        self.assertGreater(many["swing"], few["swing"])

    def test_scene_target_orbits_farther_out_than_the_subject_target(self):
        depth = _depth_with_subject()
        _, subject = _estimate(depth, target=SUBJECT_TARGET)
        _, scene = _estimate(depth, target=SCENE_TARGET)
        self.assertGreater(scene["content_radius"], subject["content_radius"])
        self.assertGreater(scene["orbit_radius"], subject["orbit_radius"])
        self.assertIn("whole surface (scene)", scene["source"])
        self.assertAlmostEqual(scene["centre"][2], 4.0, delta=0.5)   # background-dominated

    def test_mask_input_overrides_the_depth_heuristic(self):
        depth = _depth_with_subject()
        mask = torch.zeros(64, 64)
        mask[:16, :16] = 1.0                       # a corner patch, NOT the near square
        _, summary = _estimate(depth, subject_mask=mask)
        self.assertEqual(summary["source"], "input mask")
        self.assertEqual(summary["points"], 16 * 16)
        self.assertGreater(summary["centre"][2], 2.0)      # follows the mask, not the subject

    def test_estimate_scales_with_the_depth_gauge(self):
        depth = _depth_with_subject()
        _, small = _estimate(depth)
        _, large = _estimate(depth * 3.0)
        self.assertAlmostEqual(small["swing"], large["swing"], places=6)
        self.assertAlmostEqual(small["azimuth_per_frame"], large["azimuth_per_frame"], places=6)
        self.assertAlmostEqual(large["orbit_radius"] / small["orbit_radius"], 3.0, places=4)
        self.assertAlmostEqual(large["travel_per_frame"] / small["travel_per_frame"], 3.0, places=4)

    def test_invalid_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Meridian frame count"):
            _estimate(_depth_with_subject(), frames=100)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _estimate(_depth_with_subject(), target="banana")
        with self.assertRaisesRegex(ValueError, "empty"):
            _estimate(_depth_with_subject(), subject_mask=torch.zeros(64, 64))

    def test_speed_limited_span_caps_the_azimuth_per_frame(self):
        span, travel, limited, per_frame = speed_limited_span(360.0, 1.0, 1.0, 11, 10.0)
        self.assertAlmostEqual(span, MAX_AZIMUTH_PER_FRAME * 10, places=6)
        self.assertEqual(per_frame, MAX_AZIMUTH_PER_FRAME)
        self.assertTrue(limited)
        self.assertGreater(travel, 0.0)


if __name__ == "__main__":
    unittest.main()

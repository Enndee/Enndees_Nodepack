"""Tests for the automatic Meridian camera estimator; no depth model is loaded.

The surface comes from synthetic depth maps: a flat background with a near "subject" wedge whose
two halves sit at different depths, so the *geometric* pivot, the composite subject path, the
scene oval, the speed fit and the collision guard are all pinned exactly.
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
    COLLISION_MARGIN,
    DEFAULT_MAX_SPEED,
    FRONT_ELEVATION,
    FRONT_ORBIT_SHARE,
    FRONT_YAW_AMPLITUDE,
    REST_ELEVATION_HIGH,
    REST_YAW_SPAN,
    SCENE_ELEVATION_HIGH,
    SCENE_ELEVATION_LOW,
    SCENE_SPAN,
    SCENE_TARGET,
    SUBJECT_TARGET,
    automatic_keys,
    document_from_keys,
    estimate_camera_path,
    format_summary,
    geometric_pivot,
    guard_collisions,
    offset_pivot,
    pivot_radius,
    probe_surface,
    scene_samples,
    split_subject,
    subject_samples,
    surface_points,
)


def _reference(height=64, width=64):
    return torch.zeros(1, height, width, 3)


def _depth_with_subject(height=64, width=64, background=4.0, near=1.0, far=1.0, size=24):
    """Background with a near square 'subject'; its left 3/4 sit at `near`, the rest at `far`.

    The two tones let a test push the subject's depth profile apart (the geometric pivot must
    then sit between them); the default keeps the plate flat so the subject heuristic is not part
    of what is being measured.
    """
    depth = torch.full((height, width), background)
    top, left = (height - size) // 2, (width - size) // 2
    depth[top:top + size, left:left + size] = far
    split = left + int(size * 0.75)
    depth[top:top + size, left:split] = near
    return depth


def _estimate(depth, **overrides):
    options = dict(frames=73, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED,
                   depth_fn=lambda reference: depth)
    options.update(overrides)
    return estimate_camera_path(_reference(), **options)


def _orbit_angles(position, pivot):
    """(azimuth, elevation) of a camera position around a pivot; yaw 0 = in front of it."""
    delta = [position[axis] - pivot[axis] for axis in range(3)]
    horizontal = math.hypot(delta[0], delta[2])
    return (math.degrees(math.atan2(delta[0], -delta[2])),
            math.degrees(math.atan2(-delta[1], horizontal)))


def _unwrapped(azimuths):
    """Azimuths made monotone again by adding whole turns where they wrap at +/-180."""
    result = []
    for index, value in enumerate(azimuths):
        if index:
            while value - result[-1] < -180.0:
                value += 360.0
            while value - result[-1] > 180.0:
                value -= 360.0
        result.append(value)
    return result


def _wrap_delta(first, second):
    """Smallest angle between two azimuths."""
    return abs((first - second + 180.0) % 360.0 - 180.0)


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
        self.assertLess(float(near[:, 2].median()), 2.0)          # the near square, both halves
        self.assertLessEqual(near.shape[0], points.shape[0] // 2)

    def test_geometric_pivot_sits_in_the_middle_of_the_depth_profile(self):
        near = torch.zeros(75, 3)                         # a dense, near front face ...
        near[:, 0] = torch.linspace(-0.5, 0.5, 75)
        near[:, 2] = 1.0
        far = torch.zeros(25, 3)                          # ... and a sparse far tail
        far[:, 0] = torch.linspace(-0.5, 0.5, 25)
        far[:, 2] = 3.0
        cloud = torch.cat([near, far])
        pivot, extents = geometric_pivot(cloud)
        self.assertAlmostEqual(float(pivot[2]), 2.0, delta=0.05)         # (near + far) / 2
        self.assertAlmostEqual(float(cloud[:, 2].median()), 1.0, delta=0.05)  # the dense half
        self.assertAlmostEqual(float(extents[2]), 2.0, delta=0.05)       # the depth extent
        self.assertAlmostEqual(float(pivot[0]), 0.0, delta=0.05)         # centred horizontally
        self.assertAlmostEqual(float(extents[0]), 1.0, delta=0.05)

    def test_geometric_pivot_of_a_flat_plate_is_the_plate(self):
        plate = torch.zeros(50, 3)
        plate[:, 0] = torch.linspace(-0.5, 0.5, 50)
        plate[:, 2] = 1.25
        pivot, extents = geometric_pivot(plate)
        self.assertAlmostEqual(float(pivot[2]), 1.25, delta=1e-5)
        self.assertLess(float(extents[2]), 0.05)

    def test_pivot_radius_is_measured_from_the_pivot(self):
        points = surface_points(_depth_with_subject())
        centre = torch.tensor([0.0, 0.0, 2.0])
        front = torch.tensor([0.0, 0.0, -10.0])
        radius = pivot_radius(points, centre)
        self.assertGreater(radius, 0.0)
        self.assertLess(radius, pivot_radius(points, front))     # follows its own pivot


class AutoCameraProbeTests(unittest.TestCase):
    def test_probe_surface_places_the_scene_pivot_between_the_depth_extremes(self):
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        self.assertEqual(surface["target"], SUBJECT_TARGET)
        self.assertIn("near depth layer", surface["source"])
        self.assertAlmostEqual(surface["pivot"][2], 1.0, delta=0.05)        # the subject's plane
        self.assertAlmostEqual(surface["scene_pivot"][2], 2.5, delta=0.1)   # (near + far) / 2
        self.assertGreater(surface["scene_radius"], surface["content_radius"])
        self.assertEqual(surface["points"], 64 * 64)
        self.assertEqual(surface["content_points"], 24 * 24)
        self.assertGreater(surface["scene_extents"][2], 2.5)               # 4.0 - 1.0, roughly

    def test_offset_pivot_scales_with_the_content_radius(self):
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        offsets = (0.5, -0.5, 0.25)
        pivot = offset_pivot(surface, offsets)
        for axis, offset in enumerate(offsets):
            self.assertAlmostEqual(pivot[axis],
                                   surface["pivot"][axis] + offset * surface["content_radius"],
                                   places=4)


class AutoCameraPathTests(unittest.TestCase):
    def test_subject_path_loops_the_front_then_laps_the_rest_at_a_new_height(self):
        pivot = [0.0, 0.0, 0.0]
        samples = subject_samples(73, pivot, 5.0, 1.0)
        self.assertEqual(len(samples), 73)
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        split = math.floor((73 - 1) * FRONT_ORBIT_SHARE)
        self.assertAlmostEqual(angles[0][0], 0.0, delta=1.0)
        self.assertAlmostEqual(angles[0][1], -FRONT_ELEVATION, delta=1.0)     # starts below front
        self.assertAlmostEqual(angles[split][0], 0.0, delta=1.0)              # the O closes
        self.assertAlmostEqual(angles[split][1], -FRONT_ELEVATION, delta=1.0)
        front = angles[:split + 1]
        self.assertAlmostEqual(max(value for value, _ in front), FRONT_YAW_AMPLITUDE, delta=3.0)
        self.assertAlmostEqual(min(value for value, _ in front), -FRONT_YAW_AMPLITUDE, delta=3.0)
        self.assertAlmostEqual(max(value for _, value in front), FRONT_ELEVATION, delta=3.0)
        self.assertAlmostEqual(min(value for _, value in front), -FRONT_ELEVATION, delta=3.0)
        rest = angles[split + 1:]
        azimuths = _unwrapped([value for value, _ in rest])
        self.assertGreater(azimuths[0], 0.0)                             # continues the loop
        self.assertLess(azimuths[0], 10.0)
        self.assertAlmostEqual(azimuths[-1], REST_YAW_SPAN, delta=3.0)
        self.assertEqual(azimuths, sorted(azimuths))                      # laps only forward
        elevations = [value for _, value in rest]
        self.assertEqual(elevations, sorted(elevations))                  # the height only rises
        self.assertAlmostEqual(elevations[-1], REST_ELEVATION_HIGH, delta=3.0)
        # the last frame lands past the loop's reach: new surface *and* a new height
        self.assertGreater(_wrap_delta(rest[-1][0], 0.0), FRONT_YAW_AMPLITUDE + 20.0)
        self.assertGreater(elevations[-1], FRONT_ELEVATION + 5.0)          # a new height too

    def test_scene_path_is_a_big_oval_over_the_scene(self):
        pivot = [0.0, 0.0, 0.0]
        samples = scene_samples(73, pivot, 5.0, 1.0)
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        azimuths = _unwrapped([value for value, _ in angles])
        self.assertAlmostEqual(azimuths[0], 0.0, delta=1.0)
        self.assertAlmostEqual(azimuths[-1], SCENE_SPAN, delta=3.0)       # one big lap
        self.assertEqual(azimuths, sorted(azimuths))
        elevations = [value for _, value in angles]
        self.assertAlmostEqual(elevations[0], SCENE_ELEVATION_LOW, delta=1.0)
        self.assertAlmostEqual(elevations[-1], SCENE_ELEVATION_HIGH, delta=1.0)
        self.assertLess(elevations[0], 0.0)                               # starts below the horizon
        self.assertGreater(elevations[-1], 0.0)                           # ends above it
    def test_amplitude_fit_scales_the_path_to_the_speed_budget(self):
        pivot, radius, content = [0.0, 0.0, 0.0], 5.0, 2.0
        _, fast = automatic_keys(73, pivot, radius, content, SUBJECT_TARGET, max_speed=100.0)
        _, tight = automatic_keys(73, pivot, radius, content, SUBJECT_TARGET, max_speed=0.05)
        _, floored = automatic_keys(73, pivot, radius, content, SUBJECT_TARGET, max_speed=1e-6)
        _, scene = automatic_keys(73, pivot, radius, content, SCENE_TARGET, max_speed=100.0)
        self.assertEqual(fast["amplitude_scale"], 1.0)
        self.assertLess(tight["amplitude_scale"], 1.0)
        self.assertLessEqual(tight["travel_per_frame"], tight["budget_per_frame"] + 1e-9)
        self.assertLess(tight["travel_per_frame"], fast["travel_per_frame"])
        self.assertEqual(tight["style"], "front O-orbit + height lap")
        self.assertEqual(scene["style"], "big scene oval")
        self.assertEqual(floored["amplitude_scale"], 0.025)               # the ladder's floor
        self.assertGreater(floored["travel_per_frame"], floored["budget_per_frame"])

    def test_keys_look_at_the_pivot_and_span_every_frame(self):
        keys, _ = automatic_keys(73, [1.0, 2.0, 3.0], 4.0, 2.0, SUBJECT_TARGET, max_speed=100.0)
        self.assertGreaterEqual(len(keys), 5)
        self.assertEqual(keys[0]["src"], 0)
        self.assertEqual(keys[-1]["src"], 72)
        sources = [key["src"] for key in keys]
        self.assertEqual(sources, sorted(set(sources)))
        for key in keys:
            self.assertEqual(key["look"], [1.0, 2.0, 3.0])
            self.assertEqual(key["t"], key["src"])


class AutoCameraGuardTests(unittest.TestCase):
    def test_guard_pushes_offending_keys_out_of_the_scene(self):
        wall = torch.zeros(400, 3)                     # a wall in the z = 0 plane
        wall[:, 0] = torch.linspace(-1.0, 1.0, 400)
        wall[:, 1] = torch.linspace(-1.0, 1.0, 400)
        surface = {"scene_points": wall, "scene_radius": 2.0, "content_radius": 2.0,
                   "pivot": [0.0, 0.0, 5.0]}
        keys = [{"pos": [0.0, 0.0, 5.0], "look": [0.0, 0.0, 5.0], "src": 0, "t": 0},
                {"pos": [0.0, 0.0, 0.05], "look": [0.0, 0.0, 5.0], "src": 1, "t": 1}]
        keys, fixed, before, after = guard_collisions(keys, surface)
        self.assertEqual(fixed, 1)
        self.assertLess(before, COLLISION_MARGIN)
        self.assertGreaterEqual(after, COLLISION_MARGIN * 0.99)
        self.assertEqual(keys[0]["pos"], [0.0, 0.0, 5.0])          # the clear key is untouched
        self.assertLess(keys[1]["pos"][2], 0.0)                    # the offender moved away

    def test_guard_is_a_no_op_without_scene_points(self):
        keys = [{"pos": [0.0, 0.0, 1.0], "look": [0.0, 0.0, 0.0], "src": 0, "t": 0}]
        empty = {"scene_points": torch.zeros(0, 3), "scene_radius": 1.0, "content_radius": 1.0,
                 "pivot": [0.0, 0.0, 0.0]}
        untouched, fixed, _, _ = guard_collisions(keys, empty)
        self.assertEqual(fixed, 0)
        self.assertEqual(untouched[0]["pos"], [0.0, 0.0, 1.0])


class AutoCameraEstimateTests(unittest.TestCase):
    def test_estimate_uses_the_geometric_pivot_plus_the_user_offset(self):
        depth = _depth_with_subject()
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        document, summary = _estimate(depth, pivot_offset=(0.0, 0.0, 0.5))
        self.assertAlmostEqual(summary["pivot"][2],
                               surface["pivot"][2] + 0.5 * surface["content_radius"], delta=1e-5)
        keys = json.loads(document)["path"]
        self.assertGreaterEqual(len(keys), 5)
        for key in keys:
            self.assertAlmostEqual(key["look"][2], summary["pivot"][2], delta=1e-5)
        plain = json.loads(_estimate(depth)[0])["path"]
        self.assertAlmostEqual(plain[0]["look"][2], surface["pivot"][2], delta=1e-5)

    def test_summary_reports_the_pivot_style_and_collision_fields(self):
        _, summary = _estimate(_depth_with_subject())
        text = format_summary(summary)
        self.assertIn("geometric midpoint of the subject depth profile", text)
        self.assertIn("front O-orbit + height lap", text)
        self.assertEqual(summary["style"], "front O-orbit + height lap")
        self.assertEqual(summary["collision_fixes"], 0)
        self.assertGreaterEqual(summary["clearance_after"], 0.0)
        self.assertEqual(summary["points"], 64 * 64)
        self.assertGreater(summary["content_radius"], 0.0)

    def test_scene_target_builds_the_big_oval(self):
        document, summary = _estimate(_depth_with_subject(), target=SCENE_TARGET)
        self.assertEqual(summary["style"], "big scene oval")
        self.assertIn("big scene oval", format_summary(summary))
        self.assertIn("whole surface (scene)", summary["source"])
        self.assertEqual(json.loads(document)["frames"], 73)
        self.assertGreater(summary["content_points"], 0)

    def test_mask_input_overrides_the_depth_heuristic(self):
        depth = _depth_with_subject()
        mask = torch.zeros(64, 64)
        mask[:16, :16] = 1.0                       # a corner patch, NOT the near square
        _, summary = _estimate(depth, subject_mask=mask)
        self.assertEqual(summary["source"], "input mask")
        self.assertEqual(summary["content_points"], 16 * 16)
        self.assertGreater(summary["pivot"][2], 3.5)      # follows the mask, not the subject

    def test_estimate_scales_with_the_depth_gauge(self):
        depth = _depth_with_subject()
        _, small = _estimate(depth)
        _, large = _estimate(depth * 3.0)
        self.assertEqual(small["style"], large["style"])
        self.assertEqual(small["keys"], large["keys"])
        self.assertAlmostEqual(small["amplitude_scale"], large["amplitude_scale"], places=6)
        self.assertAlmostEqual(large["pivot"][2] / small["pivot"][2], 3.0, places=4)
        self.assertAlmostEqual(large["orbit_radius"] / small["orbit_radius"], 3.0, places=4)
        self.assertAlmostEqual(large["travel_per_frame"] / small["travel_per_frame"], 3.0, places=4)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_estimate_handles_a_cuda_depth_map(self):
        """The depth map arrives on the GPU in production: every helper must stay on its device."""
        _, summary = _estimate(_depth_with_subject().to("cuda"))
        self.assertGreater(summary["content_radius"], 0.0)
        self.assertEqual(summary["style"], "front O-orbit + height lap")
        mask = torch.zeros(64, 64)                 # masks come from ComfyUI on the CPU
        mask[:16, :16] = 1.0
        _, masked = _estimate(_depth_with_subject().to("cuda"), subject_mask=mask)
        self.assertEqual(masked["source"], "input mask")
        self.assertEqual(masked["content_points"], 16 * 16)

    def test_invalid_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Meridian frame count"):
            _estimate(_depth_with_subject(), frames=100)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _estimate(_depth_with_subject(), target="banana")
        with self.assertRaisesRegex(ValueError, "empty"):
            _estimate(_depth_with_subject(), subject_mask=torch.zeros(64, 64))
        with self.assertRaisesRegex(ValueError, "Unsupported Meridian frame count"):
            automatic_keys(100, [0.0, 0.0, 0.0], 1.0, 1.0, SUBJECT_TARGET)
        with self.assertRaisesRegex(ValueError, "Unsupported Meridian frame count"):
            document_from_keys(100, [], "Auto", "description")


if __name__ == "__main__":
    unittest.main()



"""Tests for the automatic Meridian camera estimator; no depth model is loaded.

The surface comes from synthetic depth maps: a flat background with a near "subject" wedge whose
two halves sit at different depths, so the *geometric* pivot, the composite subject path, the
scene survey rows, the speed fit and the collision guard are all pinned exactly.
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
    CANVAS_HEIGHT,
    COLLISION_MARGIN,
    DEFAULT_MAX_SPEED,
    FRONT_ELEVATION,
    FRONT_ELEVATION_LIMIT,
    FRONT_ORBIT_RISE,
    FRONT_YAW_AMPLITUDE,
    FRONT_YAW_LIMIT,
    FULL_CIRCLE,
    ORBIT_DIRECTION_DEFAULT,
    REST_ELEVATION_HIGH,
    SCENE_ELEVATION_HIGH,
    SCENE_ELEVATION_LOW,
    SCENE_FILL,
    SCENE_FILL_MAX,
    SCENE_FILL_MIN,
    SCENE_MAX_LANES,
    SCENE_MIN_LANES,
    SCENE_TARGET,
    SCENE_YAW_LIMIT,
    SUBJECT_FILL,
    SUBJECT_TARGET,
    VISIBILITY_MARGIN,
    VISIBILITY_QUARTILES,
    automatic_keys,
    balanced_front_share,
    direction_mirror,
    document_from_keys,
    enforce_subject_visibility,
    equalize_pivot,
    estimate_camera_path,
    fit_subject_cylinder,
    format_summary,
    front_amplitudes,
    front_camera,
    geometric_pivot,
    guard_collisions,
    lap_span_for_end,
    offset_pivot,
    pivot_radius,
    probe_surface,
    project_points,
    scene_coverage,
    scene_samples,
    scene_survey_fill,
    split_subject,
    subject_box,
    subject_framing,
    subject_samples,
    subject_share_curve,
    surface_points,
    visibility_clearances,
    _place,
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


def _safe_share(height=CANVAS_HEIGHT, aspect=1.0):
    """Share of the picture the fill target refers to (the frame minus the visibility border)."""
    margin = VISIBILITY_MARGIN * height
    return ((height - 2.0 * margin) * (height * aspect - 2.0 * margin)) / (height * height * aspect)


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


def _front_split(frames, size=1.0, scale=1.0, end=FULL_CIRCLE):
    """Last frame index of the front O: the frame the balanced split hands over to the orbit."""
    yaw, _elevation = front_amplitudes(scale, size)
    return math.floor((frames - 1) * balanced_front_share(yaw, max(0.0, end - yaw)))


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
    def test_subject_path_opens_in_the_middle_rises_and_closes_the_round(self):
        """The O opens on the framed front pose, rises to 12 o'clock, swings counter-clockwise past
        9 and 6 to 3 o'clock - and the concluding orbit carries on from there around the back until
        it is back at the start point (the default 360 deg), where the clip loops."""
        pivot = [0.0, 0.0, 0.0]
        samples = subject_samples(73, pivot, 5.0, 1.0)
        self.assertEqual(len(samples), 73)
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        split = _front_split(73)
        rise = math.floor(split * FRONT_ORBIT_RISE)
        # the middle of the O: frame 0 is the framed front pose (yaw 0, elevation 0) ...
        self.assertAlmostEqual(angles[0][0], 0.0, delta=1e-4)
        self.assertAlmostEqual(angles[0][1], 0.0, delta=1e-4)
        # ... from where the camera rises straight up to 12 o'clock, the top of the O
        for index in range(rise + 1):
            self.assertAlmostEqual(angles[index][0], 0.0, delta=1e-4)
        self.assertAlmostEqual(angles[rise][1], FRONT_ELEVATION, delta=0.05)
        # counter-clockwise 12 -> 9 -> 6 -> 3: the subject's left side first, the low point second,
        # the right side last - where the loop hands over to the lap, back at the framed height
        left = min(range(rise, split + 1), key=lambda index: angles[index][0])
        low = min(range(rise, split + 1), key=lambda index: angles[index][1])
        self.assertLess(rise, left)
        self.assertLess(left, low)
        self.assertLess(low, split)
        # the arc's exact extremes fall between frames, so the *sampled* ones sit up to half a degree
        # short of the amplitudes
        self.assertAlmostEqual(angles[left][0], -FRONT_YAW_AMPLITUDE, delta=0.5)
        self.assertAlmostEqual(angles[low][1], -FRONT_ELEVATION, delta=0.5)
        self.assertAlmostEqual(angles[split][0], FRONT_YAW_AMPLITUDE, delta=0.5)
        # the handover sits *between* two frames, so the sampled loop frame is a few degrees short of
        # the level height and the lap's first frame picks it up - they bracket 0
        self.assertLess(abs(angles[split][1]), 5.0)
        self.assertAlmostEqual(angles[split + 1][1], 0.0, delta=0.5)
        front = angles[:split + 1]
        self.assertAlmostEqual(max(value for value, _ in front), FRONT_YAW_AMPLITUDE, delta=0.5)
        self.assertAlmostEqual(min(value for value, _ in front), -FRONT_YAW_AMPLITUDE, delta=0.5)
        self.assertAlmostEqual(max(value for _, value in front), FRONT_ELEVATION, delta=0.5)
        self.assertAlmostEqual(min(value for _, value in front), -FRONT_ELEVATION, delta=0.5)
        # the concluding orbit starts exactly where the O ended (its right side) and only runs
        # forward - around the back and back to the start azimuth, which is frame 0's
        rest = angles[split + 1:]
        azimuths = _unwrapped([value for value, _ in rest])
        self.assertEqual(azimuths, sorted(azimuths))                      # laps only forward
        self.assertGreater(azimuths[0], FRONT_YAW_AMPLITUDE)              # continues the loop
        self.assertLess(azimuths[0] - FRONT_YAW_AMPLITUDE, 10.0)
        self.assertAlmostEqual(azimuths[-1], FULL_CIRCLE, delta=1e-3)     # back at the start point
        self.assertAlmostEqual(_wrap_delta(azimuths[-1] % FULL_CIRCLE, 0.0), 0.0, delta=1e-6)
        # the lane climbs to the new height at the lap's middle (the subject's back) and eases back
        # down, so the last frame sits on the first one and the clip loops
        elevations = [value for _, value in rest]
        top = max(range(len(elevations)), key=lambda index: elevations[index])
        self.assertAlmostEqual(elevations[top], REST_ELEVATION_HIGH, delta=0.1)
        self.assertAlmostEqual(top / (len(elevations) - 1), 0.5, delta=0.06)
        self.assertAlmostEqual(elevations[-1], 0.0, delta=1e-4)
        self.assertLess(math.dist(samples[-1], samples[0]), 1e-6)         # the round is closed
        # ... and the whole path really surrounds the subject: every azimuth is covered, the front
        # twice (the O shows it, the lap visits it again on its way back to the start)
        covered = _unwrapped([value for value, _ in angles])
        self.assertGreaterEqual(max(covered) - min(covered), FULL_CIRCLE - 0.5)
        self.assertAlmostEqual(min(covered), -FRONT_YAW_AMPLITUDE, delta=0.5)
        self.assertAlmostEqual(max(covered), FULL_CIRCLE, delta=0.5)

    def test_lap_span_for_end_closes_the_round_whatever_the_O_swings(self):
        """The lap starts where the O ended, so the default 360 is reached whatever the O's swing."""
        for swing in (5.0, 31.0, FRONT_YAW_AMPLITUDE, FRONT_YAW_LIMIT):
            span = lap_span_for_end(swing)
            self.assertAlmostEqual(swing + span, FULL_CIRCLE, places=9)
        # ... and a shorter end leaves exactly that much azimuth for the lap
        self.assertAlmostEqual(lap_span_for_end(FRONT_YAW_AMPLITUDE, 180.0),
                               180.0 - FRONT_YAW_AMPLITUDE, places=9)
        self.assertAlmostEqual(lap_span_for_end(FRONT_YAW_AMPLITUDE, 0.0), 0.0, places=9)
        yaw, elevation = front_amplitudes(1.5, 1.0)
        self.assertLessEqual(yaw, FRONT_YAW_LIMIT)
        self.assertLessEqual(elevation, FRONT_ELEVATION_LIMIT)
        self.assertAlmostEqual(lap_span_for_end(yaw), FULL_CIRCLE - yaw, places=9)

    def test_scene_path_is_a_lateral_survey_not_a_surround(self):
        """The scene gets drone-style rows across its width - side coverage, no 360 deg lap."""
        pivot = [0.0, 0.0, 0.0]
        plan = scene_coverage(None, 5.0, 2.0, 73)
        samples = scene_samples(73, pivot, 5.0, plan)
        self.assertEqual(len(samples), 73)
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        yaws = [value for value, _ in angles]
        elevations = [value for _, value in angles]
        for sample in samples:
            self.assertAlmostEqual(math.dist(sample, pivot), 5.0, places=4)   # holds its distance
        # rows cover both sides and turn around instead of travelling round the scene
        self.assertAlmostEqual(max(yaws), plan["half_yaw"], delta=2.0)
        self.assertAlmostEqual(min(yaws), -plan["half_yaw"], delta=2.0)
        self.assertLessEqual(max(yaws), SCENE_YAW_LIMIT + 1e-6)
        self.assertGreaterEqual(min(yaws), -SCENE_YAW_LIMIT - 1e-6)
        # every row sweeps the height range: below the horizon for under-surfaces, above for
        # looking down into the scene
        self.assertAlmostEqual(min(elevations), SCENE_ELEVATION_LOW, delta=2.0)
        self.assertAlmostEqual(max(elevations), SCENE_ELEVATION_HIGH, delta=2.0)
        # the survey ends on the other side at the other height, never repeating the start pose
        self.assertGreater(abs(yaws[-1] - yaws[0]), plan["half_yaw"])
        self.assertGreater(abs(elevations[-1] - elevations[0]), 10.0)

    def test_scene_coverage_adapts_the_rows_to_the_area(self):
        """Rows follow the scene's width and the 70 % side-lap rule; never more than the budget."""
        surface = {"aspect": 16.0 / 9.0, "hfov": 85.0}
        narrow = scene_coverage(dict(surface, lateral_half_width=0.2), 4.0, 2.0, 73)
        wide = scene_coverage(dict(surface, lateral_half_width=3.0), 4.0, 2.0, 73)
        self.assertGreaterEqual(narrow["rows"], SCENE_MIN_LANES)
        self.assertGreater(wide["rows"], narrow["rows"])                  # more area, more rows
        self.assertLessEqual(wide["rows"], SCENE_MAX_LANES)
        self.assertLess(narrow["half_yaw"], wide["half_yaw"])             # a narrow scene sweeps less
        self.assertLessEqual(wide["lane_step"], wide["footprint"])        # rows always overlap
        self.assertGreater(wide["lane_overlap"], 0.0)
        # the frame budget caps the rows: 16 frames cannot pay for nine rows
        short = scene_coverage(dict(surface, lateral_half_width=3.0), 4.0, 2.0, 16)
        self.assertLessEqual(short["rows"], 4)                            # 16 frames / 4 per row
        self.assertLess(short["rows"], wide["rows"])
        # Auto Orbit Size grows the whole envelope
        small = scene_coverage(dict(surface, lateral_half_width=1.0), 4.0, 2.0, 73, orbit_size=0.5)
        large = scene_coverage(dict(surface, lateral_half_width=1.0), 4.0, 2.0, 73, orbit_size=2.0)
        self.assertLess(small["half_yaw"], large["half_yaw"])
        self.assertLess(small["high_elevation"], large["high_elevation"])
    def test_subject_fill_is_the_only_distance_control(self):
        """Auto Subject Fill drives the distance; Auto Orbit Distance is deprecated and ignored."""
        near = _estimate(_depth_with_subject(), subject_fill=60.0)[1]
        far = _estimate(_depth_with_subject(), subject_fill=25.0)[1]
        self.assertLess(near["orbit_radius"], far["orbit_radius"])      # more fill = closer camera
        # the deprecated widget changes nothing at all
        ignored = _estimate(_depth_with_subject(), subject_fill=40.0, orbit_distance=1.5)[1]
        plain = _estimate(_depth_with_subject(), subject_fill=40.0)[1]
        self.assertAlmostEqual(ignored["orbit_radius"], plain["orbit_radius"], places=6)
        # without a fill request the built-in stand-off applies
        _, default = _estimate(_depth_with_subject())
        self.assertAlmostEqual(default["orbit_fill"], SUBJECT_FILL)
        _, scene = _estimate(_depth_with_subject(), target=SCENE_TARGET)
        # the scene survey *solves* its stand-off (reach + frame fill), so it is bounded, not fixed
        self.assertGreaterEqual(scene["orbit_fill"], SCENE_FILL_MIN)
        self.assertLessEqual(scene["orbit_fill"], SCENE_FILL_MAX)
        self.assertAlmostEqual(scene["scene_fill_solved"], scene["orbit_fill"])

    def test_orbit_size_stretches_the_front_loop_only(self):
        """Auto Orbit Size scales the O-orbit's swings; the distance to the pivot never moves."""
        pivot, radius = [0.0, 0.0, 0.0], 5.0
        tight = subject_samples(73, pivot, radius, 1.0, 0.5)
        wide = subject_samples(73, pivot, radius, 1.0, 1.2)   # 1.2 * 62 deg stays inside the limit
        for sample in tight + wide:
            self.assertAlmostEqual(math.dist(sample, pivot), radius, places=4)
        tight_front = tight[:_front_split(73, 0.5) + 1]
        wide_front = wide[:_front_split(73, 1.2) + 1]
        tight_yaws = [abs(_orbit_angles(sample, pivot)[0]) for sample in tight_front]
        wide_yaws = [abs(_orbit_angles(sample, pivot)[0]) for sample in wide_front]
        self.assertAlmostEqual(max(wide_yaws) / max(tight_yaws), 1.2 / 0.5, delta=0.05)

    def test_front_loop_never_swings_past_the_view_limits(self):
        """The O-orbit growth is capped: +/-85 deg azimuth and +/-60 deg elevation, never the lap."""
        pivot = [0.0, 0.0, 0.0]
        samples = subject_samples(73, pivot, 5.0, 4.0, 4.0)        # absurd amplitudes on purpose
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        front = angles[:_front_split(73, 4.0) + 1]
        # the loop really reaches its limits (the last sampled frame of the arc sits a hair short of
        # the peak, hence the 1 deg tolerance) and never swings past them
        self.assertGreaterEqual(max(value for value, _ in front), FRONT_YAW_LIMIT - 1.0)
        self.assertLessEqual(max(value for value, _ in front), FRONT_YAW_LIMIT + 1e-6)
        self.assertLessEqual(min(value for value, _ in front), -FRONT_YAW_LIMIT + 1.0)
        self.assertGreaterEqual(min(value for value, _ in front), -FRONT_YAW_LIMIT - 1e-6)
        self.assertAlmostEqual(max(value for _, value in front), FRONT_ELEVATION_LIMIT, delta=0.1)

    def test_orbit_size_widget_shapes_without_moving_the_camera(self):
        _, tight = _estimate(_depth_with_subject(), orbit_size=0.4)
        _, wide = _estimate(_depth_with_subject(), orbit_size=1.6)
        self.assertAlmostEqual(tight["orbit_size"], 0.4)
        self.assertAlmostEqual(wide["orbit_size"], 1.6)
        self.assertAlmostEqual(tight["orbit_radius"], wide["orbit_radius"], places=6)
        self.assertAlmostEqual(tight["orbit_fill"], wide["orbit_fill"], places=6)
        # without the speed fit the wider orbit needs the longer per-frame travel
        _, small = automatic_keys(73, [0.0, 0.0, 0.0], 5.0, 2.0, SUBJECT_TARGET,
                                  max_speed=100.0, orbit_size=0.5)
        _, large = automatic_keys(73, [0.0, 0.0, 0.0], 5.0, 2.0, SUBJECT_TARGET,
                                  max_speed=100.0, orbit_size=2.0)
        self.assertEqual(large["amplitude_scale"], 1.0)
        self.assertGreater(large["travel_per_frame"], small["travel_per_frame"])

    def test_subject_framing_solves_the_distance_for_the_fill(self):
        """Auto Subject Fill: the along-path fill centres on the requested share of the frame."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        near = subject_framing(surface, 60.0)[0]
        far = subject_framing(surface, 25.0)[0]
        self.assertLess(near, far)                                 # more fill, closer camera
        for target in (25.0, 40.0, 60.0):
            _, _, metrics = subject_framing(surface, target)
            # the fill is measured on the full canvas, but *solved* against the safe area (frame
            # minus the visibility border) and the room the swing/lap need, so the geometric mean
            # lands at or (when the room binds) below the request
            mean = math.sqrt(metrics["fill_min"] * metrics["fill_max"])
            self.assertLessEqual(mean, target / 100.0 * _safe_share() + 0.02)
            self.assertGreater(mean, 0.55 * target / 100.0 * _safe_share())
            self.assertLess(abs(metrics["offset_px"][0]), 2.0)     # the box sits in the middle
            self.assertLess(abs(metrics["offset_px"][1]), 2.0)
            self.assertGreater(metrics["radius_px"], 0.0)

    def test_subject_framing_centres_the_cylinder_axis(self):
        """The pivot is the cylinder centre *and* the aim: the axis projects to the image centre.

        This is the new contract that replaced the old per-axis `_solve_shift` re-centring. The
        camera always looks at the cylinder axis, so the axis lands exactly on the centre of the
        picture in every pose - and the camera-to-pivot distance is one constant for the whole path
        (`_place` walks a sphere of that radius). A subject whose own shape is not symmetric about
        the cylinder (a deep wedge) still has its *box* centre a little off; that is the subject's
        asymmetry, not the aim's, and it is what the cylinder approximation deliberately accepts.
        """
        depth = _depth_with_subject(near=1.0, far=2.5)
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        _, pivot, metrics = subject_framing(surface, 40.0)
        centre, radius, coverage, keep = fit_subject_cylinder(surface)
        for axis in range(3):
            self.assertAlmostEqual(pivot[axis], float(centre[axis]), places=6)
        self.assertAlmostEqual(metrics["pivot_shift"][0], 0.0, places=9)     # no aim shift any more
        self.assertAlmostEqual(metrics["pivot_shift"][1], 0.0, places=9)
        self.assertAlmostEqual(metrics["cylinder"]["radius"], radius, places=6)
        self.assertGreater(coverage, 0.85)                                   # ~95 % of the surface
        self.assertLessEqual(coverage, 1.0)
        # the axis is what the camera looks at -> it sits exactly in the image centre, every pose
        canvas_width = float(CANVAS_HEIGHT) * float(surface["aspect"])
        axis = torch.tensor([pivot], dtype=torch.float32)
        for yaw, elevation in ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0), (0.0, 30.0),
                               (0.0, -20.0)):
            position = _place(pivot, metrics["distance"], yaw, elevation)
            self.assertAlmostEqual(math.dist(position, pivot), metrics["distance"], places=5)
            u, v, _z = project_points(axis, position, pivot, surface)
            self.assertAlmostEqual(float(u[0]) / canvas_width, 0.5, delta=1e-6)
            self.assertAlmostEqual(float(v[0]) / float(CANVAS_HEIGHT), 0.5, delta=1e-6)

    def test_subject_speed_budget_is_measured_in_subject_pixels(self):
        """The fit grows the loop to the view limit when the pixels allow it, and never past it."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, metrics = subject_framing(surface, 40.0)
        cap = DEFAULT_MAX_SPEED * metrics["radius_px"]
        keys, info = automatic_keys(243, pivot, distance, surface["content_radius"], SUBJECT_TARGET,
                                    max_speed=DEFAULT_MAX_SPEED, orbit_size=1.0,
                                    surface=surface, cap_px=cap)
        ceiling = min(FRONT_YAW_LIMIT / FRONT_YAW_AMPLITUDE,
                      FRONT_ELEVATION_LIMIT / FRONT_ELEVATION)
        self.assertEqual(info["fit"], "subject pixels")
        self.assertLessEqual(info["drift_px"], info["drift_cap_px"] + 1e-9)
        self.assertGreater(info["amplitude_scale"], 1.0)            # bigger than the built-in swing
        self.assertLessEqual(info["amplitude_scale"], ceiling + 1e-9)
        yaws = [abs(_orbit_angles(key["pos"], pivot)[0]) for key in keys
                if key["t"] <= _front_split(243, 1.0, info["amplitude_scale"])]   # the O, not the lap
        self.assertGreater(max(yaws), FRONT_YAW_AMPLITUDE)          # ... and it is really flown
        self.assertLessEqual(max(yaws), FRONT_YAW_LIMIT + 1e-6)

    def test_fewer_frames_shrink_the_loop_and_the_background_stays_out(self):
        """Same subject, tighter budget -> smaller loop; a 10x farther background changes nothing."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject(background=4.0))
        distance, pivot, metrics = subject_framing(surface, 40.0)
        cap = DEFAULT_MAX_SPEED * metrics["radius_px"]
        options = dict(target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED, orbit_size=1.0,
                       surface=surface, cap_px=cap)
        _, long_run = automatic_keys(243, pivot, distance, surface["content_radius"], **options)
        _, short_run = automatic_keys(73, pivot, distance, surface["content_radius"], **options)
        # 73 frames cannot always pay for the cap *and* the whole round: the speed cap wins, the lap
        # is cut where the budget ends (the console line names the levers) and the drift stays inside
        self.assertLessEqual(short_run["drift_px"], short_run["drift_cap_px"] + 1e-9)
        self.assertLessEqual(short_run["orbit_end"], FULL_CIRCLE + 1e-9)
        self.assertGreater(short_run["orbit_coverage"], 0.5 * FULL_CIRCLE)
        self.assertLess(short_run["amplitude_scale"], long_run["amplitude_scale"])
        # the background is never part of the budget: the same subject 10x farther out is identical
        far_surface = probe_surface(_reference(),
                                    depth_fn=lambda reference: _depth_with_subject(background=40.0))
        far_distance, far_pivot, far_metrics = subject_framing(far_surface, 40.0)
        self.assertAlmostEqual(far_metrics["radius_px"], metrics["radius_px"], places=6)
        self.assertAlmostEqual(far_distance, distance, places=6)
        _, far_run = automatic_keys(73, far_pivot, far_distance, far_surface["content_radius"],
                                    **dict(options, surface=far_surface,
                                           cap_px=DEFAULT_MAX_SPEED * far_metrics["radius_px"]))
        self.assertAlmostEqual(far_run["drift_px"], short_run["drift_px"], places=6)
        self.assertAlmostEqual(far_run["amplitude_scale"], short_run["amplitude_scale"], places=6)

    def test_keys_are_in_median_depth_units(self):
        """The emitted keys are relative to the median depth, exactly like the manual path.

        The fast-depth backend multiplies a path by the cloud's median depth (`zm`), so absolute
        keys would be scaled a second time - the automatic path must emit median-depth units.
        """
        document, summary = _estimate(_depth_with_subject(), subject_fill=40.0)
        unit = summary["median_depth"]
        keys = json.loads(document)["path"]
        self.assertGreater(unit, 0.0)
        for key in keys:
            self.assertAlmostEqual(math.dist(key["pos"], key["look"]),
                                   summary["orbit_radius"] / unit, delta=0.05)
            for axis in range(3):
                # the keys are rounded to six decimals in the JSON, hence the loose tolerance
                self.assertAlmostEqual(key["look"][axis], summary["pivot"][axis] / unit, delta=1e-5)
        self.assertLess(max(abs(value) for key in keys for value in key["look"]), 5.0)

    def test_depth_gauge_leaves_the_emitted_keys_unchanged(self):
        """Three times the depth units must not move the path: the keys are relative."""
        base, _summary = _estimate(_depth_with_subject())
        scaled, _scaled_summary = _estimate(_depth_with_subject() * 3.0)
        base_keys = json.loads(base)["path"]
        scaled_keys = json.loads(scaled)["path"]
        self.assertEqual(len(base_keys), len(scaled_keys))
        for left, right in zip(base_keys, scaled_keys):
            for axis in range(3):
                self.assertAlmostEqual(left["pos"][axis], right["pos"][axis], places=2)
                self.assertAlmostEqual(left["look"][axis], right["look"][axis], places=2)

    def test_subject_framing_keeps_the_fill_band_along_the_path(self):
        """The requested fill has to hold along the orbit, not only at the front pose.

        The framing corrects the distance against the swing envelope, so the *minimum* and *maximum*
        of the along-path fill straddle the target instead of both sitting below it.
        """
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        _, _, metrics = subject_framing(surface, 40.0)
        low = metrics["fill_min"] * 100.0
        high = metrics["fill_max"] * 100.0
        self.assertLess(low, high)
        # the mean hits the target on the safe area - and stays there when the room the swing needs
        # (the room check that keeps the lap inside the frame) does not bind: it may only pull back
        mean = math.sqrt(low * high)
        self.assertLessEqual(mean, 40.0 * _safe_share() + 2.0)
        self.assertGreater(mean, 0.55 * 40.0 * _safe_share())
        # the synthetic wedge is a worst case (2.5x depth split inside the subject), so its band is
        # wider than the requested one - the console line always reports what was achieved
        self.assertGreater(low, 15.0)
        self.assertLess(high, 55.0)
        self.assertLessEqual(metrics["area"] * 100.0, high + 1e-6)        # the front pose is inside
        self.assertGreaterEqual(metrics["area"] * 100.0, low - 1e-6)

    def test_amplitude_fit_scales_the_loop_and_the_speed_cap_cuts_the_round(self):
        """A tight budget shrinks the O - and the *speed cap is hard*: the round gives way, not the
        speed. A budget that cannot pay is reported instead of silently flown over."""
        pivot, radius, content = [0.0, 0.0, 0.0], 5.0, 2.0
        _, fast = automatic_keys(73, pivot, radius, content, SUBJECT_TARGET, max_speed=100.0)
        _, tight = automatic_keys(73, pivot, radius, content, SUBJECT_TARGET, max_speed=0.05)
        _, floored = automatic_keys(73, pivot, radius, content, SUBJECT_TARGET, max_speed=1e-6)
        _, scene = automatic_keys(73, pivot, radius, content, SCENE_TARGET, max_speed=100.0)
        self.assertEqual(fast["amplitude_scale"], 1.0)
        self.assertLess(tight["amplitude_scale"], 1.0)
        self.assertLess(tight["travel_per_frame"], fast["travel_per_frame"])
        self.assertEqual(tight["style"], "front O-orbit + closing orbit")
        self.assertEqual(scene["style"], "lateral survey rows")
        # a budget that pays for everything closes the requested round around the subject ...
        self.assertAlmostEqual(fast["orbit_end"], FULL_CIRCLE, places=6)
        self.assertAlmostEqual(fast["orbit_coverage"], FULL_CIRCLE, places=6)
        # ... and the tighter the cap, the earlier the orbit ends: the coverage is what gives way
        self.assertLess(tight["orbit_end"], fast["orbit_end"])
        self.assertLess(floored["orbit_end"], tight["orbit_end"])
        for info in (tight, floored):
            self.assertGreaterEqual(info["orbit_end"], info["front_yaw"] - 1e-9)
            self.assertLessEqual(info["lap_span"], FULL_CIRCLE)
        # ... and the estimate says so instead of leaving the user with an over-budget path
        _, summary = _estimate(_depth_with_subject(), subject_fill=0.0, max_speed=1e-6)
        self.assertIn("Auto Max Speed", summary["hint"])

    def test_auto_orbit_size_is_the_floor_for_the_front(self):
        """The fit grows the O above Auto Orbit Size when the cap pays for it, and never trades the
        front away for coverage: with too few frames the *orbit* ends earlier, not the O (the case
        the real-image QC caught: a +/-3 deg O attached to a 351 deg orbit)."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, metrics = subject_framing(surface, 40.0)
        cap = DEFAULT_MAX_SPEED * metrics["radius_px"]
        options = dict(target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED, orbit_size=1.0,
                       surface=surface, cap_px=cap)
        _, roomy = automatic_keys(243, pivot, distance, surface["content_radius"], **options)
        _, tight = automatic_keys(73, pivot, distance, surface["content_radius"], **options)
        self.assertGreaterEqual(roomy["amplitude_scale"], 1.0 - 1e-9)        # grown, never shrunk
        self.assertAlmostEqual(roomy["orbit_end"], FULL_CIRCLE, places=6)     # the round is paid
        self.assertLessEqual(roomy["drift_px"], roomy["drift_cap_px"] + 1e-9)
        # 73 frames cannot pay for a 1.0x O *and* the round at once: the orbit gives way, the front
        # keeps its width as far as the cap allows, and the drift still stays inside the cap
        self.assertLessEqual(tight["drift_px"], tight["drift_cap_px"] + 1e-9)
        self.assertGreaterEqual(tight["amplitude_scale"], 0.5)
        self.assertLess(tight["orbit_end"], roomy["orbit_end"])
        self.assertGreater(tight["orbit_coverage"], tight["front_yaw"])
        self.assertAlmostEqual(tight["orbit_end"],
                               tight["front_yaw"] + tight["lap_span"], places=6)

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
    def test_pivot_equaliser_steadies_the_subject_size_around_the_orbit(self):
        """An off-centre orbit centre is moved onto the subject so the size holds; the aim stays."""
        depth = _depth_with_subject()
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        _, summary = _estimate(depth, subject_fill=40.0)
        centre = surface["pivot"]
        offset = [centre[0], centre[1], centre[2] + 0.12 * surface["content_radius"]]
        moved, metrics = equalize_pivot(surface, offset, summary["orbit_radius"],
                                        summary["amplitude_scale"], summary["orbit_size"],
                                        int(summary["frames"]),
                                        orbit_end=summary["orbit_end_deg"],
                                        direction=summary["orbit_direction"])
        self.assertTrue(metrics["applied"])
        self.assertLess(metrics["spread_after"], metrics["spread_before"] * 0.5)
        before = abs(offset[2] - centre[2])
        after = abs(moved[2] - centre[2])
        self.assertLess(after, before * 0.5)

    def test_auto_pivot_offsets_move_the_final_aim_with_framing(self):
        """The Auto Pivot X/Y/Z widgets survive the framed solve: applied last, exact delta."""
        depth = _depth_with_subject()
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        _, base = _estimate(depth, subject_fill=40.0)
        _, moved = _estimate(depth, subject_fill=40.0, pivot_offset=(0.1, -0.05, 0.0))
        unit = surface["content_radius"]
        self.assertAlmostEqual(moved["pivot"][0] - base["pivot"][0], 0.1 * unit, delta=1e-4)
        self.assertAlmostEqual(moved["pivot"][1] - base["pivot"][1], -0.05 * unit, delta=1e-4)
        self.assertAlmostEqual(moved["pivot"][2] - base["pivot"][2], 0.0, delta=1e-4)
        self.assertAlmostEqual(moved["pivot_shift_final"][0], 0.1 * unit, delta=1e-4)

    def test_orbit_centre_is_reported_and_the_keys_ride_on_it(self):
        """Emitted keys sit on the orbit sphere around `orbit_centre`; the look stays on the pivot."""
        depth = _depth_with_subject()
        document, summary = _estimate(depth, subject_fill=40.0)
        self.assertIn("orbit_centre", summary)
        for axis in range(3):
            self.assertAlmostEqual(summary["orbit_centre"][axis],
                                   summary["pivot"][axis] + summary["orbit_shift"][axis],
                                   delta=1e-9)
        unit = summary["median_depth"]
        for key in json.loads(document)["path"]:
            position = [value * unit for value in key["pos"]]
            self.assertAlmostEqual(math.dist(position, summary["orbit_centre"]),
                                   summary["orbit_radius"], delta=0.05)

    def test_estimate_uses_the_geometric_pivot_plus_the_user_offset(self):
        depth = _depth_with_subject()
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        document, summary = _estimate(depth, pivot_offset=(0.0, 0.0, 0.5))
        self.assertAlmostEqual(summary["pivot"][2],
                               surface["pivot"][2] + 0.5 * surface["content_radius"], delta=1e-5)
        keys = json.loads(document)["path"]
        self.assertGreaterEqual(len(keys), 5)
        unit = summary["median_depth"]
        for key in keys:                                  # the keys are median-depth relative
            self.assertAlmostEqual(key["look"][2] * unit, summary["pivot"][2], delta=1e-4)
        plain = json.loads(_estimate(depth)[0])["path"]
        self.assertAlmostEqual(plain[0]["look"][2] * summary["median_depth"],
                               surface["pivot"][2], delta=1e-4)

    def test_summary_reports_the_pivot_style_and_collision_fields(self):
        _, summary = _estimate(_depth_with_subject())
        text = format_summary(summary)
        self.assertIn("cylindrical centre of the subject depth profile", text)
        self.assertIn("front O-orbit + closing orbit", text)
        self.assertEqual(summary["style"], "front O-orbit + closing orbit")
        self.assertEqual(summary["collision_fixes"], 0)
        self.assertGreaterEqual(summary["clearance_after"], 0.0)
        self.assertEqual(summary["points"], 64 * 64)
        self.assertGreater(summary["content_radius"], 0.0)

    def test_scene_target_builds_the_lateral_survey(self):
        document, summary = _estimate(_depth_with_subject(), target=SCENE_TARGET)
        self.assertEqual(summary["style"], "lateral survey rows")
        self.assertIn("lateral survey rows", format_summary(summary))
        self.assertIn("whole surface (scene)", summary["source"])
        self.assertEqual(json.loads(document)["frames"], 73)
        self.assertGreater(summary["content_points"], 0)
        self.assertGreaterEqual(summary["rows"], SCENE_MIN_LANES)
        self.assertGreater(summary["lane_overlap"], 0.0)
        self.assertIn("survey:", format_summary(summary))
        # no surround: every key stays inside the survey's yaw limit (keys are median-depth relative)
        unit = summary["median_depth"]
        yaws = [abs(_orbit_angles([value * unit for value in key["pos"]], summary["pivot"])[0])
                for key in json.loads(document)["path"]]
        self.assertLessEqual(max(yaws), SCENE_YAW_LIMIT + 5.0)

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
        self.assertEqual(summary["style"], "front O-orbit + closing orbit")
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
        with self.assertRaisesRegex(ValueError, "Unknown orbit direction"):
            automatic_keys(73, [0.0, 0.0, 0.0], 1.0, 1.0, SUBJECT_TARGET, direction="sideways")
        self.assertEqual(direction_mirror(), 1.0)                  # counter-clockwise is the default
        self.assertEqual(direction_mirror("clockwise"), -1.0)


class AutoCameraVisibilityTests(unittest.TestCase):
    """The *whole* subject box has to stay inside the picture along the entire path."""

    def test_subject_box_brackets_the_subject_and_ignores_stray_points(self):
        points = torch.zeros(1000, 3)
        points[:, 0] = torch.linspace(-0.5, 0.5, 1000)
        points[:, 1] = torch.linspace(-0.25, 0.25, 1000)
        points[:, 2] = torch.linspace(1.0, 3.0, 1000)
        points[0] = torch.tensor([9.0, 9.0, 99.0])            # a spike outwards ...
        points[1] = torch.tensor([-9.0, -9.0, -99.0])         # ... and inwards
        corners, extents = subject_box({"content_cloud": points})
        self.assertEqual(tuple(corners.shape), (8, 3))
        low = corners.min(dim=0).values
        high = corners.max(dim=0).values
        for axis in range(3):                                 # brackets the body of the cloud ...
            self.assertLess(float(low[axis]), float(points[:, axis].median()))
            self.assertGreater(float(high[axis]), float(points[:, axis].median()))
            self.assertGreater(extents[axis], 0.0)
        self.assertLess(float(corners.abs().max()), 5.0)      # ... the +/-9/99 spikes stay outside

    def test_visibility_shrinks_the_swing_when_the_room_check_cannot_help(self):
        """Standing too close for the fitted swing: the pass shortens the O instead of cropping it.

        The estimate makes room for the fitted path itself (`_room_distance` in the framing), so this
        test hands the pass a deliberately close distance - the case a user hits by typing a small
        Auto Orbit Distance - and checks that the pass shrinks the swing *and* still clears the frame.
        """
        depth = _depth_with_subject(near=1.0, far=1.6)        # the wedge has real depth
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0                              # the square, both depth halves
        surface = probe_surface(_reference(), subject_mask=mask,
                                depth_fn=lambda reference: depth)
        distance, pivot, metrics = subject_framing(surface, 40.0)
        close, pivot, visibility = enforce_subject_visibility(surface, pivot, distance * 0.5, 1.0,
                                                              1.0, 73)
        self.assertLess(visibility["amplitude_cap"], 1.0)      # the swing had to shrink ...
        self.assertTrue(visibility["ok"])
        self.assertGreaterEqual(visibility["clearance_px"], 0.0)
        self.assertGreaterEqual(metrics["area"], 0.0)
        positions = subject_samples(73, pivot, close, visibility["amplitude_cap"], 1.0)
        clearances = visibility_clearances(surface["content_cloud"], positions, pivot, surface)
        self.assertGreaterEqual(min(clearances), -1e-6)        # ... and the path really fits

    def test_visibility_estimate_stays_visible_without_losing_the_fill(self):
        """The estimate's own result: visible everywhere, the requested fill kept, the trade named."""
        depth = _depth_with_subject(near=1.0, far=1.6)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        surface = probe_surface(_reference(), subject_mask=mask,
                                depth_fn=lambda reference: depth)
        _, summary = _estimate(depth, subject_mask=mask, subject_fill=40.0)
        self.assertEqual(summary["source"], "input mask")
        self.assertTrue(summary["visibility_ok"])
        self.assertGreaterEqual(summary["visibility_clearance_px"], 0.0)
        # the framing already made room for the swing, so the pass itself did not have to crop-fix
        self.assertEqual(summary["visibility_cropped"], 0)
        self.assertGreaterEqual(summary["visibility_scale"], 1.0)
        self.assertGreater(summary["subject_fill_area"], 0.05)     # a real subject in the frame
        _, _, _, keep = fit_subject_cylinder(surface)              # the framing/guarantee pool (95 %)
        points = surface["content_cloud"][keep]
        positions = subject_samples(summary["frames"], summary["pivot"], summary["orbit_radius"],
                                    summary["amplitude_scale"], summary["orbit_size"],
                                    summary["orbit_end_deg"], summary["orbit_direction"])
        clearances = visibility_clearances(points, positions, summary["pivot"], surface)
        self.assertGreaterEqual(min(clearances), -1e-6)

    def test_visibility_keeps_the_front_framing_while_it_fixes_the_path(self):
        """The guarantee may re-aim and shorten the swing, but the front-pose fill stays in band."""
        _, summary = _estimate(_depth_with_subject(), subject_fill=40.0)
        self.assertTrue(summary["visibility_ok"])
        self.assertGreaterEqual(summary["visibility_clearance_px"], 0.0)
        # re-aiming the pivot at the worst pose moves the front-pose box a little (perspective), so
        # the fill lands inside the requested band rather than exactly on the target
        self.assertGreater(summary["subject_fill_area"], 0.29)
        self.assertLess(summary["subject_fill_area"], 0.52)
        self.assertGreater(summary["subject_fill_min"], 0.10)      # nothing collapses to a sliver
        self.assertLessEqual(summary["subject_fill_max"], 1.0)     # and nothing leaves the frame
        self.assertGreaterEqual(summary["visibility_scale"], 1.0)  # a pull-back never tightens
        self.assertLessEqual(summary["amplitude_scale"],
                             summary["visibility_amplitude_cap"] + 1e-6)
        self.assertLessEqual(summary["visibility_amplitude_cap"],
                             summary["visibility_amplitude_before"] + 1e-6)

    def test_visibility_reports_the_quartile_frames_and_the_worst_one(self):
        depth = _depth_with_subject(near=1.0, far=2.6)
        _, summary = _estimate(depth, subject_fill=40.0)
        quarters = summary["visibility_quarters_px"]
        self.assertEqual(len(quarters), len(VISIBILITY_QUARTILES))     # 0/25/50/75/100 %
        self.assertGreaterEqual(min(quarters), 0.0)
        self.assertLessEqual(summary["visibility_clearance_px"], min(quarters) + 1e-9)
        self.assertGreaterEqual(summary["visibility_worst_frame"], 0)
        self.assertLess(summary["visibility_worst_frame"], summary["frames"])
        self.assertLessEqual(summary["visibility_worst_pct"], 1.0)
        self.assertGreaterEqual(summary["visibility_margin_px"], 0.0)

    def test_orbit_end_widget_chooses_where_the_round_stops(self):
        """`orbit_end` is measured from the start azimuth: 180 stops behind the subject, 0 is the O."""
        pivot = [0.0, 0.0, 0.0]
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, 180.0)
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        split = _front_split(73, 1.0, 1.0, 180.0)
        front = [value for value, _ in angles[:split + 1]]
        self.assertAlmostEqual(max(front), FRONT_YAW_AMPLITUDE, delta=0.1)       # the O is untouched
        self.assertAlmostEqual(min(front), -FRONT_YAW_AMPLITUDE, delta=0.1)
        rest = _unwrapped([value for value, _ in angles[split + 1:]])
        self.assertEqual(rest, sorted(rest))
        self.assertGreater(rest[0], FRONT_YAW_AMPLITUDE)                        # starts on the side
        self.assertLess(rest[0] - FRONT_YAW_AMPLITUDE, 10.0)
        self.assertAlmostEqual(rest[-1], 180.0, delta=1e-3)                     # the back of the subject
        covered = _unwrapped([value for value, _ in angles])
        self.assertAlmostEqual(max(covered) - min(covered), 180.0 + FRONT_YAW_AMPLITUDE,
                               delta=0.5)                                      # the O's -A..0 is extra
        # 0 skips the lap: the path is the front O alone, nothing else
        o_only = subject_samples(73, pivot, 5.0, 1.0, 1.0, 0.0)
        self.assertEqual(len(o_only), 73)
        yaws = [abs(_orbit_angles(sample, pivot)[0]) for sample in o_only]
        self.assertAlmostEqual(max(yaws), FRONT_YAW_AMPLITUDE, delta=1e-3)

    def test_orbit_direction_widget_mirrors_the_path(self):
        """'clockwise' mirrors the whole path: the O ends on the subject's left and the lap runs the
        other way round, so the two paths are exact reflections of each other (the default unchanged)."""
        pivot = [0.0, 0.0, 0.0]
        ccw = subject_samples(73, pivot, 5.0, 1.0, 1.0, 360.0, ORBIT_DIRECTION_DEFAULT)
        cw = subject_samples(73, pivot, 5.0, 1.0, 1.0, 360.0, "clockwise")
        for left, right in zip(ccw, cw):
            self.assertAlmostEqual(left[0], -right[0], places=6)     # the yaw mirrors (world x)
            self.assertAlmostEqual(left[1], right[1], places=6)
            self.assertAlmostEqual(left[2], right[2], places=6)
        # the clockwise O turns 12 -> 3 -> 6 -> 9 o'clock: the subject's *right* side first
        split = _front_split(73)
        angles = [_orbit_angles(sample, pivot) for sample in cw]
        self.assertAlmostEqual(angles[split][0], -FRONT_YAW_AMPLITUDE, delta=0.6)
        rest = _unwrapped([value for value, _ in angles[split + 1:]])
        self.assertEqual(rest, sorted(rest, reverse=True))              # now it laps backwards
        self.assertLess(rest[0], -FRONT_YAW_AMPLITUDE)
        self.assertAlmostEqual(rest[-1], -FULL_CIRCLE, delta=1e-3)      # back at the start point

    def test_estimate_keeps_the_round_inside_the_speed_cap(self):
        """The round is the point of the subject path: the *room* it needs pulls the camera back (the
        fill pays), the *speed* cap is hard - the orbit is cut there - and the summary reports both."""
        depth = _depth_with_subject(near=1.0, far=1.6)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        _, summary = _estimate(depth, subject_mask=mask, subject_fill=40.0)
        self.assertTrue(summary["visibility_ok"])
        self.assertLessEqual(summary["orbit_end_deg"], summary["orbit_end_requested_deg"] + 1e-9)
        self.assertAlmostEqual(summary["orbit_end_deg"],
                               summary["front_yaw_deg"] + summary["lap_span_deg"], delta=1e-6)
        self.assertLessEqual(summary["drift_px"], summary["drift_cap_px"] + 1e-9)
        self.assertGreater(summary["subject_fill_area"], 0.05)      # the fill paid for the room
        if summary["orbit_end_deg"] < summary["orbit_end_requested_deg"] - 0.5:
            self.assertIn("Auto Max Speed", summary["hint"])        # ... and the cut is named
        surface = probe_surface(_reference(), subject_mask=mask,
                                depth_fn=lambda reference: depth)
        _, _, _, keep = fit_subject_cylinder(surface)      # the trimmed 95 % subject the guard keeps
        points = surface["content_cloud"][keep]
        positions = subject_samples(summary["frames"], summary["pivot"], summary["orbit_radius"],
                                    summary["amplitude_scale"], summary["orbit_size"],
                                    summary["orbit_end_deg"], summary["orbit_direction"])
        clearances = visibility_clearances(points, positions, summary["pivot"], surface)
        self.assertGreaterEqual(min(clearances), -1e-6)             # and the orbit really fits
        azimuths = _unwrapped([_orbit_angles(position, summary["pivot"])[0]
                               for position in positions])
        self.assertAlmostEqual(max(azimuths) - min(azimuths),
                               summary["front_yaw_deg"]
                               + max(summary["front_yaw_deg"], summary["orbit_end_deg"]),
                               delta=0.5)               # key interpolation, not the sampled frames
        text = format_summary(summary)
        self.assertIn("deg around", text)                           # the console names the coverage

    def test_visibility_checks_the_whole_round_not_only_the_front_loop(self):
        """The orbit is part of the path, so the clearance guarantee has to cover every degree of it."""
        depth = _depth_with_subject(near=1.0, far=2.6)
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        points = surface["content_cloud"]
        pivot = surface["pivot"]
        positions = subject_samples(73, pivot, 5.0, 1.0)      # the built-in amplitudes
        azimuths = _unwrapped([_orbit_angles(position, pivot)[0] for position in positions])
        self.assertGreaterEqual(max(azimuths) - min(azimuths), FULL_CIRCLE - 1e-3)
        clearances = visibility_clearances(points, positions, pivot, surface)
        self.assertEqual(len(clearances), 73)                 # every frame has an answer




class AutoCameraSceneTests(unittest.TestCase):
    """The scene survey: rows that reach the scene's edge, frames that are not mostly empty,
    and the subject inside the scene framed well enough to read."""

    def _surface(self, half_width=1.0, aspect=16.0 / 9.0):
        return {"aspect": aspect, "hfov": 85.0, "lateral_half_width": half_width,
                "content_radius": 1.0}

    def test_scene_survey_fill_reaches_wide_scenes_and_zooms_narrow_ones(self):
        narrow = scene_survey_fill(self._surface(half_width=0.2), 1.0)[0]
        base = scene_survey_fill(self._surface(half_width=1.0), 1.0)
        wide = scene_survey_fill(self._surface(half_width=4.0), 1.0)
        # a narrow scene is flown closer than the built-in stand-off (the frame would be empty ...
        self.assertLess(narrow, SCENE_FILL)
        self.assertGreaterEqual(narrow, SCENE_FILL_MIN)
        # ... and a wide one further out, so the outermost row really reaches its edge
        self.assertGreater(wide[0], base[0])
        for ratio, metrics in (base, wide):
            # the reach is a hard lower bound - unless the safety floor (never inside the content)
            # is the binding one, in which case the console reports the shortfall
            self.assertTrue(metrics["reach"] >= 1.0 - 1e-9
                            or ratio <= SCENE_FILL_MIN + 1e-9)
            self.assertTrue(metrics["reach_fill"] <= ratio + 1e-9
                            or ratio <= SCENE_FILL_MIN + 1e-9)
            self.assertLessEqual(ratio, SCENE_FILL_MAX)
            self.assertGreater(metrics["footprint"], 0.0)
        self.assertGreaterEqual(wide[1]["reach"], 1.0 - 1e-9)     # a wide scene is really reached

    def test_scene_survey_fill_stays_inside_the_orbit_bounds(self):
        for half_width in (0.01, 0.2, 1.0, 3.0, 40.0):
            ratio, metrics = scene_survey_fill(self._surface(half_width=half_width), 1.0)
            self.assertGreaterEqual(ratio, SCENE_FILL_MIN)
            self.assertLessEqual(ratio, SCENE_FILL_MAX)
            self.assertGreater(metrics["radius"], 0.0)

    def test_probe_surface_reports_the_subject_inside_a_scene(self):
        surface = probe_surface(_reference(), target=SCENE_TARGET,
                                depth_fn=lambda reference: _depth_with_subject())
        reference_subject = probe_surface(_reference(), target=SUBJECT_TARGET,
                                          depth_fn=lambda reference: _depth_with_subject())
        self.assertIn("near depth layer", surface["subject_source"])
        self.assertIsNotNone(surface["subject_cloud"])
        # the subject layer is the same one the subject target would use, not the whole scene
        self.assertEqual(int(surface["subject_cloud"].shape[0]),
                         reference_subject["content_points"])
        self.assertLess(int(surface["subject_cloud"].shape[0]), surface["content_points"])
        self.assertAlmostEqual(surface["subject_pivot"][2], 1.0, delta=0.05)   # the subject's plane
        self.assertGreater(surface["subject_radius"], 0.0)
        # a scene without a near layer reports no subject instead of pretending the scene is one
        flat = probe_surface(_reference(), target=SCENE_TARGET,
                             depth_fn=lambda reference: torch.full((64, 64), 4.0))
        self.assertIsNone(flat["subject_cloud"])
        self.assertEqual(flat["subject_source"], "none")

    def test_subject_share_curve_counts_frames_that_show_the_subject_whole(self):
        depth = _depth_with_subject()
        surface = probe_surface(_reference(), depth_fn=lambda reference: depth)
        points = surface["content_cloud"]
        pivot = surface["pivot"]
        far = subject_share_curve(points, [front_camera(pivot, 2.5)], pivot, surface)
        near = subject_share_curve(points, [front_camera(pivot, 1.5)], pivot, surface)
        self.assertEqual(len(far), 1)
        share, whole, shown = far[0]
        self.assertTrue(shown)
        self.assertTrue(whole)                       # a centred camera holds the subject
        self.assertGreater(share, 0.01)
        self.assertGreater(near[0][0], share)        # closer is bigger (that is the survey's dial)
        # a survey row that looks *past* the subject does not show it at all (the camera aims at the
        # survey pivot; the subject sits 60 units to the side of that aim)
        side_pivot = [pivot[0] + 60.0, pivot[1], pivot[2]]
        away = subject_share_curve(points, [front_camera(side_pivot, 5.0)], side_pivot, surface)
        self.assertEqual(away, [(0.0, False, False)])

    def test_estimate_scene_summary_holds_the_reach_and_the_subject_numbers(self):
        document, summary = _estimate(_depth_with_subject(), target=SCENE_TARGET)
        self.assertEqual(summary["style"], "lateral survey rows")
        self.assertGreaterEqual(summary["rows"], SCENE_MIN_LANES)
        self.assertGreaterEqual(summary["scene_reach"], 1.0 - 1e-9)   # the rows reach the edge
        self.assertLessEqual(summary["scene_reach_fill"], summary["scene_fill_solved"] + 1e-9)
        self.assertIn("near depth layer", summary["subject_source"])
        self.assertGreater(summary["scene_subject_frames"], 0)        # the survey passes the subject
        self.assertGreater(summary["scene_subject_share"], 0.0)
        self.assertLessEqual(summary["scene_subject_whole_frames"],
                             summary["scene_subject_frames"])
        # the emitted keys stay inside the survey's own yaw limit and are median-depth relative
        unit = summary["median_depth"]
        yaws = [abs(_orbit_angles([value * unit for value in key["pos"]], summary["pivot"])[0])
                for key in json.loads(document)["path"]]
        self.assertLessEqual(max(yaws), SCENE_YAW_LIMIT + 5.0)
        self.assertIn("survey", json.loads(document)["description"])


if __name__ == "__main__":
    unittest.main()



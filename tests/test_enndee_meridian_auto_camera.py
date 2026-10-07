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
from unittest import mock

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402
from enndee_meridian_auto_camera import (  # noqa: E402
    CANVAS_HEIGHT,
    COLLISION_MARGIN,
    DEFAULT_MAX_SPEED,
    DRIFT_POOL,
    FIT_GROW_STEP,
    FRONT_ELEVATION,
    FRONT_ELEVATION_LIMIT,
    FRONT_ORBIT_AMPLITUDE,
    FRONT_ORBIT_ANGLE_MIN,
    FRONT_ORBIT_LIMIT,
    FRONT_ORBIT_RISE,
    FRONT_YAW_AMPLITUDE,
    FRONT_YAW_LIMIT,
    FULL_CIRCLE,
    NO_SUBJECT_SOURCE,
    ORBIT_COVERAGE_DEGREES,
    ORBIT_COVERAGES,
    ORBIT_DIRECTION_DEFAULT,
    ORBIT_VIEW_ANGLE_DEFAULT,
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
    depth_from_reference,
    enforce_subject_visibility,
    equalize_pivot,
    estimate_camera_path,
    fit_subject_cylinder,
    format_summary,
    front_amplitudes,
    front_camera,
    front_orbit_amplitude,
    geometric_pivot,
    guard_collisions,
    lap_span_for_end,
    offset_pivot,
    pivot_radius,
    probe_surface,
    project_points,
    resolve_front_orbit_amplitude,
    scene_coverage,
    scene_samples,
    scene_survey_fill,
    split_subject,
    subject_box,
    subject_drift,
    subject_framing,
    subject_samples,
    subject_share_curve,
    surface_points,
    visibility_clearances,
    _decimate,
    _envelope_angles,
    _fit_subject_amplitude,
    _fit_orbit_world,
    _front_travel_share,
    _orbit_point,
    _path_fits,
    _place,
    _shrink_amplitude,
    _spiral_geometry,
    _spiral_key_frames,
    _spiral_point,
    _spiral_sweep,
    SPIRAL_COVERAGE,
    SPIRAL_ELEVATION_CEILING,
    SPIRAL_END_ARC_DEFAULT,
    SPIRAL_END_ARC_MAX,
    SPIRAL_END_ARC_MIN,
    SPIRAL_END_DEFAULT,
    SPIRAL_END_MAX,
    SPIRAL_END_MIN,
    SPIRAL_SLOPE_DEFAULT,
    SPIRAL_SLOPE_MAX,
    SPIRAL_SLOPE_MIN,
    spiral_arc,
    spiral_clock,
    spiral_pose,
    resolve_spiral_end,
    resolve_spiral_end_arc,
    resolve_spiral_slope,
)
import enndee_meridian_auto_camera as auto_camera  # noqa: E402  (module state: the active winding)


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


class _FakeV2Output:
    def __init__(self, depth):
        self.predicted_depth = depth


class _FakeV2Model:
    """Depth-Anything-V2 stand-in: a near 'subject' block on a far background (inverse depth).

    `render_depth_aligned` and the estimator both run through `fast_depth._get_depth_model`, so
    patching it with this gives an estimate *and* a render of the exact same scene - which is what
    the end-to-end gauge test needs.
    """

    def __init__(self, size=128, block=32, centre_x=0.30, near=4.0, far=0.4):
        self.size, self.block, self.centre_x, self.near, self.far = size, block, centre_x, near, far

    def __call__(self, pixel_values=None, **kwargs):
        height, width = pixel_values.shape[-2], pixel_values.shape[-1]
        fx = torch.linspace(0.0, 1.0, width).view(1, width)
        fy = torch.linspace(0.0, 1.0, height).view(height, 1)
        inside = (((fx - self.centre_x).abs() < 0.5 * self.block / self.size)
                  & ((fy - 0.5).abs() < 0.5 * self.block / self.size))
        disparity = torch.where(inside, torch.full_like(fx * fy, self.near),
                                torch.full_like(fx * fy, self.far))
        return _FakeV2Output(disparity.unsqueeze(0))


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


def _travelled(samples):
    """World length of the polyline through `samples` - how far the camera actually flies."""
    return sum(math.dist(samples[index - 1], samples[index]) for index in range(1, len(samples)))


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
        # The scene pivot is the midpoint of the *cloud's* gauge (raw 1.0..4.0 -> mapped 1.0..5.0),
        # because the estimator runs in the same window the renderer unprojects.
        self.assertAlmostEqual(surface["scene_pivot"][2], 3.0, delta=0.1)   # (mapped near + far) / 2
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
        # the crane peaks exactly on 12 o'clock; the sampled frame before it sits a little lower now
        # that the circle is 45 deg (the old 30 deg ellipse peaked closer to a whole frame index)
        self.assertAlmostEqual(angles[rise][1], FRONT_ELEVATION, delta=0.5)
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

    def test_an_unusable_mask_raises_instead_of_orbiting_the_whole_surface(self):
        """A connected mask decides - so a mask that selects nothing must FAIL, not fall back.

        The old code silently swapped in the depth heuristic below `_minimum_subject_pixels`, and
        `split_subject` answers "the whole surface" as soon as its Otsu cut collapses: the orbit
        then circled the centre of the whole cloud (background included) while the path itself
        looked perfectly fine. The RMBG node even answers its own exceptions with an all-zero
        mask, so this is reachable from the workflow alone.
        """
        depth = _depth_with_subject()
        mask = torch.zeros(64, 64)
        mask[:2, :2] = 1.0                         # 4 pixels: no subject at all
        with self.assertRaises(ValueError) as caught:
            _estimate(depth, subject_mask=mask)
        message = str(caught.exception)
        self.assertIn("subject mask selects only 4", message)
        self.assertIn("white = subject", message)

    def test_the_subject_floor_accepts_a_small_detail(self):
        """The orbit floor is ~200 px of the real depth canvas, not the old 1 % of the cloud.

        A SAM3 mask that isolates a small detail used to fail ("selects only 1689 of 169344
        pixels, at least 1693 are needed") because 1 % of the canvas was required.
        """
        import enndee_meridian_auto_camera as meridian
        self.assertEqual(meridian.MIN_SUBJECT_PIXELS, 200)
        self.assertLessEqual(meridian._minimum_subject_pixels(169344), 210)

    def test_the_spiral_frames_keep_a_constant_camera_speed(self):
        """Equal arc length per frame - the coil used to crawl at the pole and race at its end."""
        import enndee_meridian_auto_camera as meridian
        frames = 121
        samples = meridian._spiral_samples(frames, [0.0, 0.0, 0.0], 4.0, 0.0, 1.0)
        steps = [math.dist(samples[index - 1], samples[index]) for index in range(1, frames)]
        self.assertGreater(min(steps), 0.0)
        self.assertLess(max(steps) / min(steps), 1.15)      # ~9x before the arc-length remap
        # the endpoints stay exactly what the widgets asked for
        self.assertAlmostEqual(meridian._spiral_point(0.0, 0.0, 1.0)[0], 0.0, places=6)
        end = meridian._spiral_point(1.0, 0.0, 1.0)
        expected = meridian.spiral_pose(meridian.spiral_end_arc(),
                                        -meridian._spiral_sweep(), meridian.spiral_slope())
        self.assertAlmostEqual(end[0], expected[0], places=6)
        self.assertAlmostEqual(end[1], expected[1], places=6)

    def test_the_pivot_reports_the_exact_points_it_was_built_from(self):
        """The console line must say which points fed the pivot - the orbit can hide a bad one."""
        depth = _depth_with_subject()
        mask = torch.zeros(64, 64)
        mask[:16, :16] = 1.0
        document, summary = _estimate(depth, subject_mask=mask)
        line = format_summary(summary)
        self.assertIn("input mask", line)
        self.assertIn(f"{summary['content_points']} of {summary['points']} points used", line)
        self.assertIn("the pivot is computed from those and only those", line)
        description = json.loads(document)["description"]
        self.assertIn(f"built from {summary['content_points']} of {summary['points']} points",
                      description)

    def test_a_mask_that_covers_the_frame_is_named_in_the_hint(self):
        """A mask selecting the whole frame segments nothing - say so, the pivot is the scene's."""
        depth = _depth_with_subject()
        _, summary = _estimate(depth, subject_mask=torch.ones(64, 64))
        self.assertEqual(summary["content_points"], summary["points"])   # it IS the whole cloud
        # the pivot is now the midpoint of the WHOLE profile: subject at z=1.0, background at
        # z=4.0 -> 2.5, i.e. well behind the subject - exactly the reported symptom
        self.assertGreater(summary["pivot"][2], 1.5)
        self.assertLess(summary["pivot"][2], 3.5)
        self.assertIn("covers 100 % of the frame", summary["hint"])
        self.assertIn("white on the SUBJECT", summary["hint"])

    def test_no_mask_and_no_near_layer_is_named_in_the_hint(self):
        """Without a mask the depth split may collapse onto the whole surface: that has to be said."""
        flat = _depth_with_subject(near=4.0, far=4.0, background=4.0)   # no layer to find
        _, summary = _estimate(flat)
        self.assertEqual(summary["source"], NO_SUBJECT_SOURCE)
        self.assertIn("centre of the WHOLE surface (background included)", summary["hint"])
        self.assertIn("connect a subject mask", summary["hint"])

    def test_the_document_names_the_depth_model_that_produced_the_pivot(self):
        """The keys are in THAT map's median units, so the renderer has to be able to check it.

        Two depth models do not share a scale: a path estimated on one and rendered on another
        puts the aim (the pivot) at a different depth than intended, and the orbit then circles a
        point in the background while the path itself looks fine. The record travels with the
        document; `warn_depth_model_mismatch` on the render side consumes it.
        """
        depth = _depth_with_subject()
        document, _ = _estimate(depth)
        self.assertEqual(json.loads(document)["depth_model"], "(injected depth map)")
        with_model = json.loads(document_from_keys(
            73, json.loads(document)["path"], "n", "d",
            extra={"depth_model": "Depth-Anything-3-Mono-Large"}))
        self.assertEqual(with_model["depth_model"], "Depth-Anything-3-Mono-Large")
        plain = json.loads(document_from_keys(73, json.loads(document)["path"], "n", "d"))
        self.assertNotIn("depth_model", plain)                    # other callers stay unchanged

    def test_depth_from_reference_returns_depth_and_model_for_both_model_families(self):
        """`probe_surface` unpacks (depth, model_name) - BOTH model families must honour that.

        The DA3 branch goes through `fast_depth._predict_da3_depth`, which returns a bare (H, W)
        tensor, while the V2 branch inlines its own inference. The tuple contract was only added
        to the V2 branch, so the first real DA3 pick on the new Depth Model widget
        (production: `Depth-Anything-3-Mono-Large`) died with
        "too many values to unpack (expected 2)" - every test injects `depth_fn`, which is why
        nothing caught it. Both branches run here against fakes, so no weights are needed.
        """
        reference = torch.zeros(1, 16, 16, 3)
        sentinel = torch.full((6, 4), 2.5)                 # non-square, like a portrait DA3 grid
        seen = {}

        class _Prediction:
            predicted_depth = torch.full((1, 4, 4), 0.4)

        class _Model:
            def __call__(self, pixel_values=None):
                return _Prediction()

        def fake_da3(model_name, first, device, process_res=0):
            seen["da3"] = model_name
            return sentinel

        def fake_v2(model_name, device):
            seen["v2"] = model_name
            return _Model()

        original_da3, original_v2 = fast_depth._predict_da3_depth, fast_depth._get_depth_model
        fast_depth._predict_da3_depth, fast_depth._get_depth_model = fake_da3, fake_v2
        try:
            depth, name = depth_from_reference(reference,
                                               model_size="Depth-Anything-3-Mono-Large")
            self.assertIs(depth, sentinel)                  # the DA3 branch returns the bare tensor
            self.assertEqual(name, "Depth-Anything-3-Mono-Large")   # ... plus its name now
            depth, name = depth_from_reference(reference,
                                               model_size="Depth-Anything-V2-Small-hf")
            self.assertIsNotNone(depth)
            self.assertEqual(name, "Depth-Anything-V2-Small-hf")
            self.assertEqual(seen.get("da3"), "Depth-Anything-3-Mono-Large")
            self.assertEqual(seen.get("v2"), "Depth-Anything-V2-Small-hf")
        finally:
            fast_depth._predict_da3_depth, fast_depth._get_depth_model = original_da3, original_v2

    def test_estimate_is_invariant_to_the_depth_gauge(self):
        """The estimate runs in the cloud's gauge, so a raw depth's absolute scale must not matter.

        `render_depth_aligned` clips the model's depth to the 1 %/99 % percentiles and maps it onto
        `DEPTH_NEAR .. DEPTH_FAR` before unprojecting, so the rendered world is scale-free: a depth
        map and the same map times three render identically. The estimator now shares that gauge
        (`fast_depth.cloud_gauge`), so its pivot, orbit radius and travel are identical too - before
        this they scaled with the raw depth, which is exactly what put the rig at the wrong depth in
        the render.
        """
        depth = _depth_with_subject()
        _, small = _estimate(depth)
        _, large = _estimate(depth * 3.0)
        self.assertEqual(small["style"], large["style"])
        self.assertEqual(small["keys"], large["keys"])
        self.assertAlmostEqual(small["amplitude_scale"], large["amplitude_scale"], places=6)
        self.assertAlmostEqual(large["pivot"][2] / small["pivot"][2], 1.0, places=4)
        self.assertAlmostEqual(large["orbit_radius"] / small["orbit_radius"], 1.0, places=4)
        self.assertAlmostEqual(large["travel_per_frame"] / small["travel_per_frame"], 1.0, places=4)

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

    def test_visibility_keeps_the_swing_when_no_amplitude_fits(self):
        """Standing too close for EVERY swing: keep the O and let the dolly fix the frame.

        The estimate makes room for the fitted path itself (`_room_distance` in the framing), so this
        test hands the pass a deliberately close distance - half the framed stand-off, where the
        subject is too big for the frame at *any* viewing angle. Shrinking then buys nothing (the
        offender is the distance, not the swing), so the pass keeps the incoming amplitude and pulls
        the camera back instead. Returning the ladder's floor here - the old behaviour, justified by
        "even the smallest loop does not fit: the dolly must help" - is what collapsed a wide front
        O to +/-3 deg in production while the frame stayed exactly as cropped as before.
        """
        depth = _depth_with_subject(near=1.0, far=1.6)        # the wedge has real depth
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0                              # the square, both depth halves
        surface = probe_surface(_reference(), subject_mask=mask,
                                depth_fn=lambda reference: depth)
        distance, pivot, metrics = subject_framing(surface, 40.0)
        close, pivot, visibility = enforce_subject_visibility(surface, pivot, distance * 0.5, 1.0,
                                                              1.0, 73)
        self.assertAlmostEqual(visibility["amplitude_cap"], 1.0, places=9)   # no useless shrink
        self.assertGreater(visibility["scale"], 1.0)          # ... the dolly did the work instead
        self.assertTrue(visibility["ok"])
        self.assertGreaterEqual(visibility["clearance_px"], 0.0)
        self.assertGreaterEqual(metrics["area"], 0.0)
        positions = subject_samples(73, pivot, close, visibility["amplitude_cap"], 1.0)
        clearances = visibility_clearances(surface["content_cloud"], positions, pivot, surface)
        self.assertGreaterEqual(min(clearances), -1e-6)       # ... and the path really fits

    def test_visibility_shrinks_the_swing_when_that_buys_room(self):
        """A wide swing that clips while a narrow one fits: shorten the O - that is the cheap fix.

        Shrink must stay available when it genuinely helps (the front pose - and with it the fill -
        does not move, only the viewing angles narrow), so this drives `_shrink_amplitude` straight
        at the framed distance with a 100 px border: measured on this fixture the 1.0x swing clips
        and 0.05x clears, i.e. there is a real answer between the two. The returned amplitude has to
        keep the frame inside - a shrink that crops is worth nothing.
        """
        depth = _depth_with_subject(near=1.0, far=1.6)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        surface = probe_surface(_reference(), subject_mask=mask,
                                depth_fn=lambda reference: depth)
        distance, pivot, _ = subject_framing(surface, 40.0)
        _, _, _coverage, keep = fit_subject_cylinder(surface)
        pool = surface["content_cloud"][keep]
        margin = 0.10 * CANVAS_HEIGHT
        end, direction = 0.0, ORBIT_DIRECTION_DEFAULT         # no lap: narrowing only buys room
        self.assertFalse(_path_fits(pool, surface, pivot, distance, 1.0, 1.0, 73, CANVAS_HEIGHT,
                                    margin, end, direction))
        self.assertTrue(_path_fits(pool, surface, pivot, distance, FIT_GROW_STEP, 1.0, 73,
                                   CANVAS_HEIGHT, margin, end, direction))
        cap = _shrink_amplitude(pool, surface, pivot, distance, 1.0, 1.0, 73, CANVAS_HEIGHT,
                                margin, end, direction)
        self.assertGreater(cap, FIT_GROW_STEP)                # found a real width ...
        self.assertLess(cap, 1.0)                             # ... smaller than the fitting swing
        self.assertTrue(_path_fits(pool, surface, pivot, distance, cap, 1.0, 73, CANVAS_HEIGHT,
                                   margin, end, direction))

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
        # the circle is 45 deg now, so the discrete frames sit a little inside 3/9 o'clock
        self.assertAlmostEqual(max(front), FRONT_YAW_AMPLITUDE, delta=0.5)       # the O is untouched
        self.assertAlmostEqual(min(front), -FRONT_YAW_AMPLITUDE, delta=0.5)
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

    def test_no_lap_path_is_the_front_alone_ending_on_three_oclock(self):
        """`orbit_end` 0 = "the front O alone": every frame the O's, no tail, no phantom crane.

        The tail the 0.92 share used to leave for a lap with *no azimuth left* still ran that lap's
        elevation arc: a 38 deg crane crammed into the last ~6 frames. Measured on the real subject
        that cost ~56 px/frame against a 14 px cap at EVERY rung of the amplitude ladder, so the
        cap could never be met, the fit chose its winner off a flat 55-60 px curve (1.15x by noise)
        and the clip ended on a crank instead of on 3 o'clock.
        """
        pivot = [0.0, 0.0, 0.0]
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, 0.0, ORBIT_DIRECTION_DEFAULT)
        self.assertEqual(len(samples), 73)                 # all frames belong to the O
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        elevations = [value for _yaw, value in angles]
        # the elevation stays inside the O's own envelope: no 38 deg lap arc hiding in the tail
        self.assertLessEqual(max(elevations), FRONT_ELEVATION + 1e-6)
        self.assertGreaterEqual(min(elevations), -FRONT_ELEVATION - 1e-6)
        # frame 0 is the centre of the O, the path pans up to 12 o'clock first ...
        self.assertAlmostEqual(angles[0][0], 0.0, places=6)
        self.assertAlmostEqual(angles[0][1], 0.0, places=6)
        self.assertAlmostEqual(max(elevations[:40]), FRONT_ELEVATION, delta=0.1)
        # ... and the LAST frame sits on 3 o'clock at level elevation - the clip ends where the
        # concluding orbit would take over from
        self.assertAlmostEqual(angles[-1][0], FRONT_YAW_AMPLITUDE, places=4)
        self.assertAlmostEqual(angles[-1][1], 0.0, places=4)
        # the reported split says the same thing, and 'clockwise' mirrors the landing point
        self.assertEqual(balanced_front_share(FRONT_YAW_AMPLITUDE, 0.0), 1.0)
        mirrored = subject_samples(73, pivot, 5.0, 1.0, 1.0, 0.0, "clockwise")
        self.assertAlmostEqual(_orbit_angles(mirrored[-1], pivot)[0], -FRONT_YAW_AMPLITUDE,
                               places=4)

    def test_zero_end_estimate_stays_under_the_cap_and_keeps_a_real_front_orbit(self):
        """The production settings (v23: 73 frames, fill 25, speed 5 %, end 0): cap met, real O.

        With the phantom tail the cap was unreachable (measured 55.6 px/frame vs a 14 px cap), so
        no rung of the ladder could win and the front's width was effectively noise. The same
        request must now land inside the cap with an orbit wide enough to read as one - and every
        emitted key's aim must be the pivot itself, which is what puts the pivot in the middle of
        the screen for every frame (all `look` values are identical, so the renderer's Catmull-Rom
        interpolation of them is that same point).
        """
        depth = _depth_with_subject(near=1.0, far=1.6)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        document, summary = _estimate(depth, subject_mask=mask, subject_fill=25.0,
                                      max_speed=0.05, orbit_end=0.0)
        self.assertTrue(summary["visibility_ok"])
        self.assertLessEqual(summary["drift_px"], summary["drift_cap_px"] * 1.01)
        self.assertEqual(summary["orbit_end_deg"], 0.0)
        self.assertEqual(summary["front_share"], 1.0)          # every frame is the O's
        # An orbit, not a wobble. The estimate now runs in the cloud's gauge (the renderer's
        # 1 %/99 % -> DEPTH_NEAR..DEPTH_FAR window), which stretches this fixture's narrow raw depth
        # range (1.0..1.6) into a deeper scene, so the fit lands a slightly tighter O than before.
        self.assertGreaterEqual(summary["front_yaw_deg"], 10.0)
        keys = json.loads(document)["path"]
        aim = {tuple(key["look"]) for key in keys}
        self.assertEqual(len(aim), 1)                          # one aim for the whole clip ...
        unit = summary["median_depth"]
        want = [value / unit for value in summary["pivot"]]
        got = list(aim)[0]
        for axis in range(3):                                  # ... and it is exactly the pivot
            self.assertAlmostEqual(got[axis], want[axis], delta=2e-6)

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


class AutoCameraOrbitShapeTests(unittest.TestCase):
    """Auto Orbit View Angle + Auto Orbit Coverage: where the fixed circle sits and what it adds.

    The front O is a 45 deg circle on the pivot's sphere now (`FRONT_ORBIT_AMPLITUDE` both ways), so
    "where do I stand" and "how much do I see" are two separate questions: the view angle turns the
    whole choreography around the subject's vertical axis, the coverage picks between the O alone
    ("Front only") and the O plus the shortest level connection to a mirrored O on the far side
    ("Front and Back"). This is the balance the rework is about - the far side used to be reached by
    a full 360 deg lap with its 38 deg crane, i.e. the long way round the subject.
    """

    def test_view_angle_offsets_the_whole_path(self):
        """The O, the connection and the far O all move with the view angle - nothing else does."""
        pivot = [0.0, 0.0, 0.0]
        plain = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                view_angle=0.0, coverage=ORBIT_COVERAGES[1])
        turned = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                 view_angle=90.0, coverage=ORBIT_COVERAGES[1])
        for before, after in zip(plain, turned):
            before_angles, after_angles = _orbit_angles(before, pivot), _orbit_angles(after, pivot)
            self.assertAlmostEqual(_wrap_delta(after_angles[0], before_angles[0]), 90.0, places=4)
            self.assertAlmostEqual(after_angles[1], before_angles[1], places=6)
        # frame 0 is the middle of the O, so it sits exactly on the view angle
        self.assertAlmostEqual(_orbit_angles(turned[0], pivot)[0], 90.0, places=6)
        self.assertAlmostEqual(_orbit_angles(turned[0], pivot)[1], 0.0, places=6)
        # the default view angle IS the frontal view the old path opened on
        self.assertEqual(ORBIT_VIEW_ANGLE_DEFAULT, 0.0)

    def test_the_front_circle_is_a_circle_on_the_pivots_sphere(self):
        """45 deg swing and 45 deg rise, one radius: the camera keeps its distance to the pivot."""
        pivot = [0.0, 0.0, 0.0]
        amplitude, _ = front_amplitudes()
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  view_angle=0.0, coverage=ORBIT_COVERAGES[0])
        self.assertEqual(len(samples), 73)
        for sample in samples:
            self.assertAlmostEqual(math.dist(sample, pivot), 5.0, places=6)   # a circle, not an O
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        azimuths = [value for value, _ in angles]
        elevations = [value for _yaw, value in angles]
        self.assertAlmostEqual(max(azimuths), amplitude, delta=0.5)
        self.assertAlmostEqual(min(azimuths), -amplitude, delta=0.5)
        self.assertAlmostEqual(max(elevations), amplitude, delta=0.5)
        self.assertAlmostEqual(min(elevations), -amplitude, delta=0.5)

    def test_front_only_coverage_is_the_circle_alone(self):
        """'Front only' ships the O and nothing else: no connection, no far O, no tail."""
        pivot = [0.0, 0.0, 0.0]
        amplitude, _ = front_amplitudes()
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  view_angle=0.0, coverage=ORBIT_COVERAGES[0])
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        azimuths = [value for value, _ in angles]
        self.assertLessEqual(max(azimuths), amplitude + 1e-6)   # never leaves the front sector
        self.assertGreaterEqual(min(azimuths), -amplitude - 1e-6)
        # ... and the front share says every frame belongs to the O
        self.assertEqual(_front_travel_share(amplitude, 0.0, False), 1.0)

    def test_the_front_o_begins_its_sweep_at_two_oclock(self):
        """The crane from the middle now lands on 2 o'clock (it used to land on the top, 12) and
        the sweep runs the REST of the circle from there - 330 deg of it - still landing on the
        level side at 3, so the connection, the back part and the clip's end keep their poses.
        'clockwise' mirrors the opening to 10 o'clock and the landing to 9."""
        amplitude, _ = front_amplitudes()
        middle = _orbit_point(0.0, 0.0, amplitude, 1.0)             # frame 0: the middle of the O
        self.assertAlmostEqual(middle[0], 0.0, delta=1e-9)
        self.assertAlmostEqual(middle[1], 0.0, delta=1e-9)
        two = _orbit_point(FRONT_ORBIT_RISE, 0.0, amplitude, 1.0)    # the crane's landing pose
        self.assertAlmostEqual(two[0], amplitude * math.cos(math.radians(30.0)), delta=1e-9)
        self.assertAlmostEqual(two[1], amplitude * math.sin(math.radians(30.0)), delta=1e-9)
        end = _orbit_point(1.0, 0.0, amplitude, 1.0)
        self.assertAlmostEqual(end[0], amplitude, delta=1e-6)        # ... and still 3 o'clock,
        self.assertAlmostEqual(end[1], 0.0, delta=1e-6)              #     level elevation
        mirrored = _orbit_point(FRONT_ORBIT_RISE, 0.0, amplitude, -1.0)
        self.assertAlmostEqual(mirrored[0], -amplitude * math.cos(math.radians(30.0)), delta=1e-9)
        self.assertAlmostEqual(mirrored[1], amplitude * math.sin(math.radians(30.0)), delta=1e-9)
        mirrored_end = _orbit_point(1.0, 0.0, amplitude, -1.0)
        self.assertAlmostEqual(mirrored_end[0], -amplitude, delta=1e-6)
        self.assertAlmostEqual(mirrored_end[1], 0.0, delta=1e-6)

    def test_front_and_back_coverage_follows_the_back_clock(self):
        """The front O, the shortest LEVEL connection, and the back orbit as a full clock loop.

        The connection stops at the first point of the back orbit it can reach - the orbit's level
        point on the arrival side, the back clock's "9 o'clock" - and the loop starts exactly there
        and sweeps clockwise 9 -> 12 (over the back's head) -> 3 (the far level point) -> 6 (under
        the back) -> 8, one hour short of its start so the closing frames never repeat the opening
        pose - and a final glide from 8 in to the dial's CENTRE: the level pose straight behind the
        subject, the mirror of the front O's own opening (its circle's middle).
        """
        pivot = [0.0, 0.0, 0.0]
        amplitude, _ = front_amplitudes()
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  view_angle=0.0, coverage=ORBIT_COVERAGES[1])
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        azimuths = _unwrapped([value for value, _ in angles])
        # it opens on the middle of the front O and visits one pass around the subject: -A, over the
        # front, +A, level out to the back, then the full loop that reaches +A on the far side
        self.assertAlmostEqual(azimuths[0], 0.0, delta=1e-6)
        self.assertAlmostEqual(angles[0][1], 0.0, delta=1e-6)
        self.assertAlmostEqual(min(azimuths), -amplitude, delta=0.5)
        self.assertAlmostEqual(max(azimuths), ORBIT_COVERAGE_DEGREES + amplitude, delta=0.5)
        # the connection hands over at the back clock's 9 o'clock: the continuous pose falls between
        # two samples, so its last level frame sits just short of it and the loop's first frame just
        # past it - and NO level frame ever runs through the connection's own azimuth range again
        # (the only level pose beyond the handover is the glide's HOME: the dial's centre at 180)
        nine = ORBIT_COVERAGE_DEGREES - amplitude
        home = ORBIT_COVERAGE_DEGREES
        level_yaws = [azimuths[index] for index in range(len(angles))
                      if abs(angles[index][1]) < 1e-9]
        self.assertTrue(all(yaw <= nine + 1e-6 or _wrap_delta(yaw, home) < 1e-6
                            for yaw in level_yaws))
        self.assertGreater(max(level_yaws), nine - 12.0)        # the handover really happens
        last_level = max(index for index, (_yaw, elevation) in enumerate(angles)
                         if abs(elevation) < 1e-9 and azimuths[index] <= nine + 1e-6)
        loop = angles[last_level + 1:]
        self.assertGreater(len(loop), 20)                      # the loop owns the tail of the clip
        # ... and the loop is a FULL circle: over the back's head (+A) and under it (-A), both
        # straight behind the subject (its middle azimuth)
        apex_yaw, apex_elevation = max(loop, key=lambda pose: pose[1])
        bottom_yaw, bottom_elevation = min(loop, key=lambda pose: pose[1])
        # the apex/bottom sit at the back's middle azimuth, where atan2 wraps (+-180): compare the
        # circular difference, the sample may land a few degrees either side of it
        self.assertAlmostEqual(apex_elevation, amplitude, delta=1.5)
        self.assertAlmostEqual(_wrap_delta(apex_yaw, ORBIT_COVERAGE_DEGREES), 0.0, delta=3.0)
        self.assertAlmostEqual(bottom_elevation, -amplitude, delta=1.5)
        self.assertAlmostEqual(_wrap_delta(bottom_yaw, ORBIT_COVERAGE_DEGREES), 0.0, delta=3.0)
        # ... the loop still passes 8 o'clock (az 180 - 0.866A, el -A/2) and then glides HOME: the
        # closing pose is the dial's centre - level, straight behind the subject - instead of the
        # clock's lower tick
        eight_yaw = ORBIT_COVERAGE_DEGREES - 0.8660254 * amplitude
        self.assertTrue(any(abs(yaw - eight_yaw) < 8.0 and abs(elevation + 0.5 * amplitude) < 6.0
                            for yaw, elevation in angles[last_level + 1:]))
        self.assertAlmostEqual(angles[-1][0], ORBIT_COVERAGE_DEGREES, delta=1e-6)
        self.assertAlmostEqual(angles[-1][1], 0.0, delta=1e-6)
        gap = math.hypot(angles[-1][0] - angles[last_level][0],
                         angles[-1][1] - angles[last_level][1])
        self.assertGreater(gap, 10.0)                           # never re-lands on 9 o'clock
        for sample in samples:                                  # every phase keeps the radius
            self.assertAlmostEqual(math.dist(sample, pivot), 5.0, places=6)

    def test_the_back_loop_never_re_flies_the_connection(self):
        """No redundant frames: the loop never walks back along the level connection.

        The old junction ran the connection all the way to the back circle's *middle* and the back
        orbit then swung back over the same level azimuth (measured: 44.5 deg of backwards azimuth
        on top of the front O's own 45 - the same poses twice). Now the connection stops where the
        loop begins, the loop climbs away from the level line immediately, and every near-level
        frame of the loop lies outside the connection's azimuth range: measured 0 retracing frames
        (see `q_loop.py`).
        """
        pivot = [0.0, 0.0, 0.0]
        amplitude, _ = front_amplitudes()
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  view_angle=0.0, coverage=ORBIT_COVERAGES[1])
        poses = [_orbit_angles(sample, pivot) for sample in samples]
        azimuths = _unwrapped([value for value, _ in poses])
        # the connection's true end is the 9 o'clock pose; the shared pose falls between two samples,
        # so measure the run by the level frames themselves: every level frame is either ON the
        # connection (at or below its handover) or - the only exception - the glide's HOME pose at
        # the dial's centre, which is where the whole path ends
        nine = ORBIT_COVERAGE_DEGREES - amplitude
        home = ORBIT_COVERAGE_DEGREES
        level_yaws = [azimuths[index] for index in range(len(poses))
                      if abs(poses[index][1]) < 1e-9]
        self.assertTrue(all(yaw <= nine + 1e-6 or _wrap_delta(yaw, home) < 1e-6
                            for yaw in level_yaws))
        junction = max(index for index, (yaw, elevation) in enumerate(poses)
                       if abs(elevation) < 1e-9 and azimuths[index] <= nine + 1e-6)
        retracing = 0
        for index in range(junction + 1, len(poses)):
            yaw, elevation = azimuths[index], poses[index][1]
            if abs(elevation) < 5.0 and amplitude + 1.0 < yaw < nine - 5.0:
                retracing += 1                                  # a level frame inside the
        self.assertEqual(retracing, 0)                          # connection's own run = re-flying it
        # the loop leaves the level line immediately (it climbs toward 12 o'clock), so the level
        # frames after the junction are the clock's own ticks and the glide's last steps settling
        # at home - all of them BEYOND the loop's start, none inside the connection's run
        self.assertGreater(poses[junction + 1][1], 1.0)
        near_level = [azimuths[index] for index in range(junction + 1, len(poses))
                      if abs(poses[index][1]) < 0.5]
        self.assertTrue(all(yaw > nine - 1e-6 for yaw in near_level))
        self.assertLessEqual(len(near_level), 4)


    def test_front_only_is_the_cheapest_visit_and_the_far_side_costs_less_azimuth(self):
        """What the rework buys: 'Front only' is one circle, and the far side is 270 deg, not 405.

        The far side is reached by the shortest level connection plus the back clock's loop, so the
        azimuth the camera sweeps drops from 405 deg (the old 360 deg end, which still paid for the
        front O's own half) to 270 (-A .. 180 + A), and the old lap's 45 deg crane at the back is
        gone. The full back loop (9 -> 12 -> 3 -> 6 -> 8, then the glide to the back's centre) is
        what the frames buy: it costs a little more than the old lap did in world travel - the
        front sweep alone grew from 270 to 330 deg - and 'Front only' stays the cheap option at a
        quarter of the price, so the speed cap decides from there (see `_cut_back_span`).
        """
        pivot = [0.0, 0.0, 0.0]
        amplitude, _ = front_amplitudes()
        alone = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                view_angle=0.0, coverage=ORBIT_COVERAGES[0])
        both = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                               view_angle=0.0, coverage=ORBIT_COVERAGES[1])
        legacy = subject_samples(73, pivot, 5.0, 1.0, 1.0, FULL_CIRCLE, ORBIT_DIRECTION_DEFAULT)

        def span(samples):
            values = _unwrapped([_orbit_angles(sample, pivot)[0] for sample in samples])
            return max(values) - min(values)

        self.assertLess(_travelled(alone), _travelled(both))          # a back loop costs travel
        self.assertLess(_travelled(alone), _travelled(legacy))        # ... and a lap costs travel
        # the full back loop costs a little more than the old lap (270 deg of azimuth, but a 330
        # deg arc at the O's radius PLUS the front sweep's 330 deg and the glide home - measured
        # 54.9 vs 46.6 world units = +8.3); 'Front only' must still be clearly cheaper than both
        self.assertAlmostEqual(_travelled(both), _travelled(legacy), delta=12.0)
        self.assertAlmostEqual(span(both), ORBIT_COVERAGE_DEGREES + 2.0 * amplitude, delta=2.0)
        self.assertAlmostEqual(span(legacy), FULL_CIRCLE + amplitude, delta=1.0)   # the old round

    def test_clockwise_mirrors_the_whole_shape(self):
        """Direction still mirrors everything - the view angle stays where the user put it."""
        pivot = [0.0, 0.0, 0.0]
        amplitude, _ = front_amplitudes()
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, "clockwise",
                                  view_angle=90.0, coverage=ORBIT_COVERAGES[1])
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        azimuths = _unwrapped([value for value, _ in angles])
        self.assertAlmostEqual(max(azimuths), 90.0 + amplitude, delta=1.0)
        self.assertAlmostEqual(min(azimuths), 90.0 - ORBIT_COVERAGE_DEGREES - amplitude, delta=1.0)

    def test_the_envelope_makes_room_for_the_far_circle_only_when_it_runs(self):
        """The room check sees the front extremes, the connection and (only) a full far O."""
        amplitude, _ = front_amplitudes()
        front = _envelope_angles(1.0, include_back=True, view_angle=None, coverage=ORBIT_COVERAGES[0])
        back = _envelope_angles(1.0, include_back=True, view_angle=None, coverage=ORBIT_COVERAGES[1])
        turned = _envelope_angles(1.0, include_back=True, view_angle=90.0,
                                  coverage=ORBIT_COVERAGES[1])
        self.assertLessEqual(max(yaw for yaw, _ in front), amplitude + 1e-6)
        self.assertAlmostEqual(max(yaw for yaw, _ in back),
                               ORBIT_COVERAGE_DEGREES + amplitude, delta=1e-6)
        self.assertAlmostEqual(min(yaw for yaw, _ in back), -amplitude, delta=1e-6)
        self.assertAlmostEqual(max(yaw for yaw, _ in turned),
                               90.0 + ORBIT_COVERAGE_DEGREES + amplitude, delta=1e-6)
        # a cut connection must NOT reserve room for a far O that never runs
        cut = _envelope_angles(1.0, include_back=True, view_angle=None, coverage=ORBIT_COVERAGES[1],
                               back_span=10.0)
        self.assertAlmostEqual(max(yaw for yaw, _ in cut), amplitude + 10.0, delta=1e-6)
        # ... and the back orbit's own loop ticks: level at its 9 o'clock edge (the connection's end),
        # +A over the back's head, -A under it, and the far 3 o'clock edge on the other side
        arc = [(yaw, elevation) for yaw, elevation in back if yaw >= ORBIT_COVERAGE_DEGREES - 90.0]
        self.assertAlmostEqual(max(elevation for _yaw, elevation in arc), amplitude, delta=1e-9)
        self.assertAlmostEqual(min(elevation for _yaw, elevation in arc), -amplitude, delta=1e-9)
        self.assertAlmostEqual(max(yaw for yaw, _e in arc),
                               ORBIT_COVERAGE_DEGREES + amplitude, delta=1e-9)   # 3 o'clock
        self.assertTrue(any(abs(yaw - (ORBIT_COVERAGE_DEGREES - amplitude)) < 1e-9
                            for yaw, _e in arc))                  # 9 o'clock (the arrival point)

    def test_estimate_reports_the_view_angle_and_the_coverage(self):
        """Both modes stay inside the speed cap and say what they did in the summary.

        The speed cap is the only thing allowed to shorten the path, and it takes it out of the FAR
        part first (`_cut_back_span`): a tight cap flies the front O with a cut connection and says
        so, a roomy one flies the whole front + back visit.
        """
        depth = _depth_with_subject(near=1.0, far=1.6)
        mask = torch.zeros(64, 64)
        mask[20:44, 20:44] = 1.0
        circle = _estimate(depth, subject_mask=mask, subject_fill=25.0, max_speed=0.05,
                           view_angle=90.0, coverage=ORBIT_COVERAGES[0])[1]
        self.assertEqual(circle["coverage"], ORBIT_COVERAGES[0])
        self.assertEqual(circle["view_angle_deg"], 90.0)
        self.assertTrue(circle["visibility_ok"])
        self.assertLessEqual(circle["drift_px"], circle["drift_cap_px"] * 1.01)
        self.assertEqual(circle["lap_span_deg"], 0.0)             # nothing beyond the circle
        self.assertFalse(circle["back_orbit"])
        self.assertGreaterEqual(circle["front_yaw_deg"], 5.0)     # a real orbit, not a wobble
        self.assertLessEqual(circle["orbit_coverage_deg"], 2.0 * circle["front_yaw_deg"] + 1e-6)
        self.assertIn("front circle only", format_summary(circle))   # no back visit is claimed
        for speed in (0.05, 0.6):                                 # the front + back visit
            _document, summary = _estimate(depth, subject_mask=mask, subject_fill=25.0,
                                           max_speed=speed, view_angle=90.0,
                                           coverage=ORBIT_COVERAGES[1])
            self.assertEqual(summary["coverage"], ORBIT_COVERAGES[1])
            self.assertEqual(summary["view_angle_deg"], 90.0)
            self.assertTrue(summary["visibility_ok"])
            self.assertLessEqual(summary["drift_px"], summary["drift_cap_px"] * 1.01)
            self.assertGreater(summary["lap_span_deg"], 0.0)      # the far side is reached
            self.assertLessEqual(summary["lap_span_deg"],
                                 ORBIT_COVERAGE_DEGREES - 2.0 * summary["front_yaw_deg"] + 1e-6)
            self.assertIn("back connection", format_summary(summary))

    def test_the_far_part_gives_way_first_when_the_budget_is_tight(self):
        """The cut is taken out of the connection, never out of the front O (the user's move)."""
        roomy = _fit_orbit_world(73, [0.0, 0.0, 0.0], 5.0, None, 1.0, 100.0,
                                 direction=ORBIT_DIRECTION_DEFAULT, view_angle=0.0,
                                 coverage=ORBIT_COVERAGES[1])[3]
        tight = _fit_orbit_world(73, [0.0, 0.0, 0.0], 5.0, None, 1.0, 0.02,
                                 direction=ORBIT_DIRECTION_DEFAULT, view_angle=0.0,
                                 coverage=ORBIT_COVERAGES[1])[3]
        self.assertTrue(roomy["back_orbit"])                      # a roomy cap flies the far O
        self.assertAlmostEqual(roomy["back_span"],
                               ORBIT_COVERAGE_DEGREES - 2.0 * roomy["front_yaw"], delta=1e-6)
        self.assertFalse(tight["back_orbit"])                     # a tight one cuts the far side
        self.assertLess(tight["back_span"], roomy["back_span"])
        self.assertGreater(tight["back_span"], 0.0)

    def test_the_legacy_orbit_end_path_is_untouched(self):
        """Without a coverage mode the old choreography still runs (saved workflows, callers)."""
        pivot = [0.0, 0.0, 0.0]
        legacy = subject_samples(73, pivot, 5.0, 1.0, 1.0, 270.0, ORBIT_DIRECTION_DEFAULT)
        angles = [_orbit_angles(sample, pivot) for sample in legacy]
        azimuths = _unwrapped([value for value, _ in angles])
        elevations = [value for _yaw, value in angles]
        split = _front_split(73, 1.0, 1.0, 270.0)
        self.assertAlmostEqual(max(azimuths), 270.0, delta=1e-3)     # the requested end is honoured
        self.assertGreater(max(azimuths), ORBIT_COVERAGE_DEGREES + 45.0)   # a lap, not 270+2A
        self.assertAlmostEqual(max(elevations[split:]), REST_ELEVATION_HIGH, delta=0.5)  # back crane
        self.assertAlmostEqual(azimuths[0], 0.0, delta=1e-6)         # still opens on the front pose


class AutoCameraOrbitAngleTests(unittest.TestCase):
    """The node's O Orbit Angle (`orbit_amplitude`) scales the front O for one estimate."""

    def test_resolve_clamps_the_widget_value(self):
        self.assertAlmostEqual(resolve_front_orbit_amplitude(None), FRONT_ORBIT_AMPLITUDE)
        self.assertAlmostEqual(resolve_front_orbit_amplitude(0.0), FRONT_ORBIT_AMPLITUDE)
        self.assertAlmostEqual(resolve_front_orbit_amplitude(-10.0), FRONT_ORBIT_AMPLITUDE)
        self.assertAlmostEqual(resolve_front_orbit_amplitude(1.0), FRONT_ORBIT_ANGLE_MIN)
        self.assertAlmostEqual(resolve_front_orbit_amplitude(90.0), FRONT_ORBIT_LIMIT)
        self.assertAlmostEqual(resolve_front_orbit_amplitude(30.0), 30.0)

    def test_the_override_is_active_inside_the_estimate_and_then_restored(self):
        seen = []

        def depth_fn(reference):
            seen.append(front_orbit_amplitude())
            return _depth_with_subject()

        self.assertAlmostEqual(front_orbit_amplitude(), FRONT_ORBIT_AMPLITUDE, places=6)
        _estimate(None, depth_fn=depth_fn, orbit_amplitude=25.0)
        self.assertEqual(seen, [25.0])            # the radius was in effect while the path was built
        self.assertAlmostEqual(front_orbit_amplitude(), FRONT_ORBIT_AMPLITUDE, places=6)
        _estimate(None, depth_fn=depth_fn)        # no widget -> the built-in radius
        self.assertAlmostEqual(seen[-1], FRONT_ORBIT_AMPLITUDE, places=6)

    def test_a_smaller_angle_scales_the_front_orbit_down(self):
        """The fit keeps its headroom, so the flown O scales with the widget value."""
        def front_span(signal):
            keys = json.loads(signal)["path"]
            pivot = keys[0]["look"]
            return max(abs(_orbit_angles(key["pos"], pivot)[0]) for key in keys)

        narrow = front_span(_estimate(_depth_with_subject(), subject_fill=40.0, max_speed=0.5,
                                      orbit_amplitude=20.0)[0])
        wide = front_span(_estimate(_depth_with_subject(), subject_fill=40.0, max_speed=0.5,
                                    orbit_amplitude=45.0)[0])
        self.assertLess(narrow, wide)


class AutoCameraSpiralCoverageTests(unittest.TestCase):
    """The Spiral coverage: a spherical spiral around the pivot, front view -> side view."""

    def test_spiral_is_a_coverage_option(self):
        self.assertIn(SPIRAL_COVERAGE, ORBIT_COVERAGES)
        self.assertEqual(ORBIT_COVERAGES[-1], SPIRAL_COVERAGE)

    def _with_winding(self, winding):
        """Fly the spiral with one winding (the module state `estimate_camera_path` sets)."""
        previous = auto_camera._ACTIVE_SPIRAL_END
        auto_camera._ACTIVE_SPIRAL_END = winding
        self.addCleanup(setattr, auto_camera, "_ACTIVE_SPIRAL_END", previous)

    def test_pose_round_trip_matches_the_sketch(self):
        """phi = the arc from the view axis, psi = the clock around it (the user's sketch)."""
        for phi, psi in ((0.0, 0.0), (10.0, 60.0), (45.0, 120.0), (90.0, 270.0),
                         (70.0, -30.0)):
            yaw, elevation = spiral_pose(phi, psi)
            self.assertAlmostEqual(spiral_arc(yaw, elevation), phi, places=6)
            delta = (spiral_clock(yaw, elevation) - psi + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual(delta, 0.0, places=6)

    def test_phi_runs_from_the_view_axis_to_the_picture_plane(self):
        """phi = 0 at the FIRST frame, 90 (a side view of the picture) at the END - the spec."""
        self.assertEqual(_spiral_point(0.0, 0.0), (0.0, 0.0))      # the middle of the picture
        # phi does NOT climb linearly: the frames are spread over the coil's ARC LENGTH (constant
        # camera speed), so the angle per frame is small where the coil opens and large where it
        # winds. A quarter of the way along the path is therefore well past a quarter of the angle.
        quarter = spiral_arc(*_spiral_point(0.25, 0.0))
        self.assertGreater(quarter, SPIRAL_END_ARC_DEFAULT * 0.25)
        self.assertLess(quarter, SPIRAL_END_ARC_DEFAULT * 0.5)
        end_yaw, end_elevation = _spiral_point(1.0, 0.0)
        self.assertAlmostEqual(spiral_arc(end_yaw, end_elevation), SPIRAL_END_ARC_DEFAULT, places=6)
        # phi = 90 is the picture's own plane: the camera looks at the scene from the side, whatever
        # the winding - the end elevation follows from the winding alone.
        self.assertAlmostEqual(abs(end_yaw), 90.0, places=6)
        self.assertAlmostEqual(
            end_elevation,
            math.degrees(math.asin(math.cos(math.radians(SPIRAL_END_DEFAULT)))), places=6)
        self.assertAlmostEqual(end_elevation, -30.0, places=6)      # 840 deg = 2 1/3 rounds

    def test_the_winding_is_the_parameter_and_decides_the_end_pose(self):
        """Spiral End is flown as asked - no whole-round snapping, no cap cutting it."""
        for winding in (90.0, 270.0, 810.0, 840.0, 1500.0):
            with self.subTest(winding=winding):
                self._with_winding(winding)
                self.assertAlmostEqual(_spiral_sweep(), winding, places=6)
                yaw, elevation = _spiral_point(1.0, 0.0)
                self.assertAlmostEqual(spiral_arc(yaw, elevation), SPIRAL_END_ARC_DEFAULT, places=6)
                self.assertAlmostEqual(abs(yaw), 90.0, places=6)     # always a side view
                self.assertAlmostEqual(
                    elevation, math.degrees(math.asin(math.cos(math.radians(winding)))), places=6)
        # 810 deg ends level on the side, 840 deg (the default) 30 deg below the pivot
        self._with_winding(810.0)
        self.assertAlmostEqual(_spiral_point(1.0, 0.0)[1], 0.0, places=6)
        self._with_winding(840.0)
        self.assertAlmostEqual(_spiral_point(1.0, 0.0)[1], -30.0, places=6)
        # 0 deg is the plain quarter circle. psi starts at 12 o'clock (the clock angle 0), so the
        # end of a winding that is a multiple of 360 lands on the TOP of the picture's plane - the
        # pole, where the up vector is undefined and the ceiling holds it at +/-88 deg.
        self._with_winding(0.0)
        yaw, elevation = _spiral_point(1.0, 0.0)
        self.assertAlmostEqual(yaw, 0.0, places=6)
        self.assertEqual(elevation, SPIRAL_ELEVATION_CEILING)

    def test_resolve_spiral_end_clamps(self):
        self.assertEqual(resolve_spiral_end(None), SPIRAL_END_DEFAULT)
        self.assertEqual(resolve_spiral_end(0), SPIRAL_END_MIN)
        self.assertEqual(resolve_spiral_end(-90), SPIRAL_END_MIN)
        self.assertEqual(resolve_spiral_end(99999), SPIRAL_END_MAX)
        self.assertEqual(resolve_spiral_end(840), 840.0)
        self.assertEqual(auto_camera.spiral_end(), SPIRAL_END_DEFAULT)

    def test_resolve_spiral_slope_clamps(self):
        self.assertEqual(resolve_spiral_slope(None), SPIRAL_SLOPE_DEFAULT)
        self.assertEqual(resolve_spiral_slope(0), 0.0)
        self.assertEqual(resolve_spiral_slope(45), 45.0)
        self.assertEqual(resolve_spiral_slope(180), SPIRAL_SLOPE_MAX)
        self.assertEqual(resolve_spiral_slope(-180), SPIRAL_SLOPE_MIN)
        self.assertEqual(auto_camera.spiral_slope(), SPIRAL_SLOPE_DEFAULT)

    def test_the_slope_leans_the_spiral_s_axis(self):
        """0 = the view axis, +90 = straight above, -90 = straight below - the axis, not the aim."""
        self.assertEqual(_spiral_point(0.0, 0.0, 1.0, SPIRAL_SLOPE_DEFAULT), (0.0, 0.0))
        # the first frame of a +90 lean opens straight above the subject - the elevation ceiling
        # holds it the last 2 deg short of the pole, so the arc out of the tilted axis reads 2 deg
        above = _spiral_point(0.0, 0.0, 1.0, SPIRAL_SLOPE_MAX)
        self.assertEqual(above, (0.0, SPIRAL_ELEVATION_CEILING))
        self.assertAlmostEqual(spiral_arc(*above, slope_degrees=SPIRAL_SLOPE_MAX),
                               SPIRAL_END_ARC_DEFAULT - SPIRAL_ELEVATION_CEILING, places=6)
        self.assertAlmostEqual(spiral_arc(*spiral_pose(0.0, 0.0, SPIRAL_SLOPE_MAX),
                                          slope_degrees=SPIRAL_SLOPE_MAX), 0.0, places=6)
        end = _spiral_point(1.0, 0.0, 1.0, SPIRAL_SLOPE_MAX)
        self.assertAlmostEqual(spiral_arc(*end, slope_degrees=SPIRAL_SLOPE_MAX), SPIRAL_END_ARC_DEFAULT,
                               places=4)
        below = _spiral_point(0.0, 0.0, 1.0, SPIRAL_SLOPE_MIN)
        self.assertEqual(below, (0.0, -SPIRAL_ELEVATION_CEILING))
        # ... and the round trip through the tilted axis holds for any slope
        for slope in (-90.0, -45.0, 0.0, 30.0, 90.0):
            for phi, psi in ((0.0, 0.0), (20.0, 120.0), (90.0, 200.0)):
                with self.subTest(slope=slope, phi=phi, psi=psi):
                    yaw, elevation = spiral_pose(phi, psi, slope)
                    self.assertAlmostEqual(spiral_arc(yaw, elevation, slope), phi, places=6)
                    delta = (spiral_clock(yaw, elevation, slope) - psi + 180.0) % 360.0 - 180.0
                    self.assertAlmostEqual(delta, 0.0, places=6)

    def test_a_leaned_spiral_keeps_the_distance_and_the_ends(self):
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, _metrics = subject_framing(surface, 40.0)
        previous = auto_camera._ACTIVE_SPIRAL_SLOPE
        auto_camera._ACTIVE_SPIRAL_SLOPE = 60.0
        self.addCleanup(setattr, auto_camera, "_ACTIVE_SPIRAL_SLOPE", previous)
        samples = subject_samples(73, pivot, distance, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        for sample in samples:
            self.assertAlmostEqual(math.dist(sample, pivot), distance, places=6)
        self.assertAlmostEqual(samples[0][1], pivot[1] - distance * math.sin(math.radians(60.0)),
                               places=4)                       # the opening frame is high above
        self.assertAlmostEqual(
            spiral_arc(*_orbit_angles(samples[-1], pivot), slope_degrees=60.0), SPIRAL_END_ARC_DEFAULT,
            places=4)

    def test_arc_rises_monotonically_and_the_clock_winds(self):
        steps = 60
        points = [_spiral_point(index / steps, 0.0) for index in range(steps + 1)]
        arcs = [spiral_arc(yaw, elevation) for yaw, elevation in points]
        self.assertEqual(arcs, sorted(arcs))                      # the arc never turns back
        self.assertAlmostEqual(arcs[-1], SPIRAL_END_ARC_DEFAULT, places=6)
        # unwrap the clock angle: the spiral winds the whole sweep, counter-clockwise (falling)
        clocks = [spiral_clock(yaw, elevation) for yaw, elevation in points]
        unwrapped = [clocks[0]]
        for value in clocks[1:]:
            unwrapped.append(unwrapped[-1] + ((value - unwrapped[-1] + 180.0) % 360.0 - 180.0))
        self.assertLess(unwrapped[-1] - unwrapped[0], 0.0)
        self.assertAlmostEqual(abs(unwrapped[-1] - unwrapped[0]), _spiral_sweep(), delta=1.0)

    def test_direction_mirrors_the_winding(self):
        ccw = [_spiral_point(index / 12, 0.0, 1.0) for index in range(13)]
        cw = [_spiral_point(index / 12, 0.0, -1.0) for index in range(13)]
        self.assertEqual(ccw[0], (0.0, 0.0))                       # both start on the view axis
        self.assertEqual(cw[0], (0.0, 0.0))
        self.assertAlmostEqual(ccw[-1][0], -cw[-1][0], places=6)   # mirrored yaw ...
        self.assertAlmostEqual(ccw[-1][1], cw[-1][1], places=6)    # ... same elevation
        for left, right in zip(ccw, cw):
            self.assertAlmostEqual(left[0], -right[0], places=6)
            self.assertAlmostEqual(left[1], right[1], places=6)

    def test_samples_keep_the_distance_and_the_frame_count(self):
        pivot = [0.0, 0.0, 3.0]
        samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        self.assertEqual(len(samples), 73)
        for sample in samples:
            self.assertAlmostEqual(math.dist(sample, pivot), 5.0, places=6)
        angles = [_orbit_angles(sample, pivot) for sample in samples]
        self.assertAlmostEqual(spiral_arc(*angles[0]), 0.0, places=4)      # the front pose
        self.assertAlmostEqual(spiral_arc(*angles[-1]), SPIRAL_END_ARC_DEFAULT, places=4)

    def test_the_fit_never_shortens_the_winding(self):
        """The Spiral End parameter is flown as asked - the fit reports the drift, it does not cut."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, metrics = subject_framing(surface, 40.0)
        # a deliberately tiny cap: the old ladder would have cut the winding to a fraction of a round
        tiny = 0.02 * metrics["radius_px"]
        chosen = _fit_subject_amplitude(73, pivot, distance, surface, 1.0, tiny,
                                        coverage=SPIRAL_COVERAGE)
        self.assertIsNotNone(chosen)
        scale, samples, drift, _typical, info = chosen
        self.assertEqual(scale, 1.0)                   # no ladder: the winding is flown as asked
        self.assertAlmostEqual(info["spiral_end"], SPIRAL_END_DEFAULT, places=6)
        self.assertAlmostEqual(info["orbit_end"], SPIRAL_END_DEFAULT, places=6)
        self.assertEqual(len(samples), 73)
        arcs = [spiral_arc(*_orbit_angles(sample, pivot)) for sample in samples]
        self.assertAlmostEqual(arcs[0], 0.0, places=4)
        self.assertAlmostEqual(arcs[-1], SPIRAL_END_ARC_DEFAULT, places=4)
        self.assertGreater(drift, 0.0)                 # measured, so the console can name the price
        self.assertGreater(drift, tiny)                # ... and it *is* over the cap: no silent cut

    def test_the_o_orbit_angle_grows_evenly_along_the_path(self):
        """O-orbits like in the Front-only setting, their angle growing 0 -> the end angle.

        The frames are spread over the coil's **arc length** (constant camera speed), so the
        O-orbit angle per frame is deliberately NOT constant: it is small where the coil opens at
        the pole and large out at the winding ring - that is what equalises the camera speed.
        """
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, _metrics = subject_framing(surface, 40.0)
        samples = subject_samples(73, pivot, distance, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        self.assertEqual(len(samples), 73)
        for sample in samples:                            # every pose keeps the one distance
            self.assertAlmostEqual(math.dist(sample, pivot), distance, places=6)
        arcs = [spiral_arc(*_orbit_angles(sample, pivot)) for sample in samples]
        self.assertAlmostEqual(arcs[0], 0.0, places=6)    # the family's first O-orbit: the axis
        self.assertAlmostEqual(arcs[-1], SPIRAL_END_ARC_DEFAULT, places=4)
        # the angle still grows monotonically, and the O-orbit angle per frame *falls* along the
        # path: right at the pole the arc is pure `phi`, while at the winding ring it is almost
        # all `psi` - that is exactly what keeps the camera's speed constant
        self.assertTrue(all(arcs[index] > arcs[index - 1] for index in range(1, 73)))
        self.assertGreater(arcs[1] - arcs[0], arcs[-1] - arcs[-2])
        # ... because what grows evenly now is the ARC on the orbit sphere: constant speed. The
        # residual spread comes from the elevation ceiling that flattens the coil's top.
        steps = [math.dist(samples[index - 1], samples[index]) for index in range(1, 73)]
        self.assertLess(max(steps) / min(steps), 1.25)     # ~9x before the arc-length remap

    def test_the_end_angle_is_the_parameter(self):
        """Spiral End Angle is the O-orbit family's last radius - flown as asked, whatever it is."""
        pivot = [0.0, 0.0, 3.0]
        previous_slope = auto_camera._ACTIVE_SPIRAL_SLOPE
        auto_camera._ACTIVE_SPIRAL_SLOPE = 0.0
        self.addCleanup(setattr, auto_camera, "_ACTIVE_SPIRAL_SLOPE", previous_slope)
        for arc in (20.0, 45.0, 75.0, 90.0):
            with self.subTest(arc=arc):
                previous = auto_camera._ACTIVE_SPIRAL_END_ARC
                auto_camera._ACTIVE_SPIRAL_END_ARC = arc
                self.addCleanup(setattr, auto_camera, "_ACTIVE_SPIRAL_END_ARC", previous)
                samples = subject_samples(73, pivot, 5.0, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                          coverage=SPIRAL_COVERAGE, spiral_slope=0.0)
                arcs = [spiral_arc(*_orbit_angles(sample, pivot)) for sample in samples]
                # the O-orbit angle grows along the path from the axis out to the end angle: the
                # first frame is the frontal pose (on the axis) and the last one is the end angle
                self.assertLess(arcs[0], 1.0)                 # starts on (or within a hair of) the axis
                self.assertAlmostEqual(arcs[-1], arc, places=4)   # ... and ends on the end angle
                self.assertEqual(arcs, sorted(arcs))          # the angle never turns back

    def test_resolve_spiral_end_arc_clamps(self):
        self.assertEqual(resolve_spiral_end_arc(None), SPIRAL_END_ARC_DEFAULT)
        self.assertEqual(resolve_spiral_end_arc(0), SPIRAL_END_ARC_MIN)      # no spiral at all
        self.assertEqual(resolve_spiral_end_arc(180), SPIRAL_END_ARC_MAX)    # the picture's plane
        self.assertEqual(resolve_spiral_end_arc(45), 45.0)
        self.assertEqual(auto_camera.spiral_end_arc(), SPIRAL_END_ARC_DEFAULT)

    def test_the_drift_is_reported_not_fitted_away(self):
        """The winding and the end angle stay what the widgets asked for; the drift is *named*."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, metrics = subject_framing(surface, 40.0)
        pool = _decimate(surface["content_cloud"], DRIFT_POOL)
        # a deliberately tiny cap: the old ladder would have cut the spiral to a fraction of a round
        tiny = 0.02 * metrics["radius_px"]
        samples = subject_samples(73, pivot, distance, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        drift, _typical = subject_drift(pool, samples, pivot, surface)
        self.assertGreater(drift, tiny)                   # over the cap - and reported, not cut
        arcs = [spiral_arc(*_orbit_angles(sample, pivot)) for sample in samples]
        self.assertAlmostEqual(arcs[-1], SPIRAL_END_ARC_DEFAULT, places=4)   # the path is flown

    def test_the_spiral_frame_stays_upright_and_never_flips_over_the_pole(self):
        """The frame is the world-up look-at: a coil comes out upright, with no roll and no flip."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, _metrics = subject_framing(surface, 40.0)
        samples = subject_samples(73, pivot, distance, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        pos = torch.as_tensor(samples, dtype=torch.float32).reshape(-1, 3)
        piv = torch.as_tensor(pivot, dtype=torch.float32).reshape(1, 3)
        forwards = piv - pos
        forwards = forwards / forwards.norm(dim=1, keepdim=True).clamp(min=1e-8)
        right, _down = fast_depth.camera_frames(forwards.numpy())
        right = torch.as_tensor(right)
        # no flip anywhere along the spiral (the clock winds over the top of the pivot)
        dots = [float((right[i] * right[i - 1]).sum()) for i in range(1, right.shape[0])]
        self.assertGreater(min(dots), 0.0)
        # upright wherever a horizon exists: `right` is the look's level-horizon right, so the clip
        # never rolls about the optical axis (the spiral no longer accumulates transport roll)
        for i in range(forwards.shape[0]):
            level = fast_depth._horizon_right(forwards[i].numpy())
            if level is not None:
                self.assertTrue(torch.allclose(right[i], torch.as_tensor(level), atol=1e-4))
        # frame 0 is the plain world-up zero-roll look-at: the opening pose is unchanged
        up = torch.tensor([0.0, -1.0, 0.0])
        r0 = torch.cross(forwards[0], up, dim=0)
        r0 = r0 / r0.norm()
        self.assertTrue(torch.allclose(right[0], r0, atol=1e-4))

    def test_the_rendered_estimate_keeps_the_subject_in_the_picture_centre(self):
        """End to end: the emitted keys must land the rig where the renderer's cloud lives.

        The estimator and the renderer have to share one depth gauge (`fast_depth.cloud_gauge` -
        the 1 %/99 % percentile window mapped onto `DEPTH_NEAR .. DEPTH_FAR`). Estimating in the
        model's raw units instead put the whole rig at the wrong depth *and* scale (a key's x/y
        follow its z), so the subject - which the estimate promises to hold in the horizontal
        picture centre - swung out of the frame along the path. This flies a real spiral over an
        off-centre subject and measures where it actually lands in the rendered frames.
        """
        size, block, centre_x = 128, 32, 0.30
        image = torch.zeros(1, size, size, 3)
        image[0, :, :, 2] = 0.15                                    # dim blue background
        top, left = size // 2 - block // 2, int(centre_x * size) - block // 2
        image[0, top:top + block, left:left + block, 0] = 1.0       # bright red subject
        image[0, top:top + block, left:left + block, 2] = 0.0
        model = _FakeV2Model(size=size, block=block, centre_x=centre_x)
        device = torch.device("cpu")
        with mock.patch.object(fast_depth, "_get_depth_model", lambda *a, **k: model):
            document, _summary = estimate_camera_path(
                image, 73, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED, subject_fill=40.0,
                coverage=SPIRAL_COVERAGE, device=device, model_size="Depth-Anything-V2-Small-hf")
            _source, render, width, height, length = fast_depth.render_depth_aligned(
                image, device, model_size="Depth-Anything-V2-Small-hf", frames=73,
                canvas_mode="custom", custom_width=size, custom_height=size, cloud_scale=1,
                point_size=1, edge_cull=True, edge_threshold=0.30, custom_camera=document)
        self.assertEqual(length, 73)
        for index in range(length):
            subject = ((render[index, :, :, 0] > 0.35)
                       & (render[index, :, :, 0] > render[index, :, :, 2] + 0.2))
            self.assertGreater(int(subject.sum()), 12, f"the subject left frame {index}")
            ys, xs = torch.nonzero(subject, as_tuple=True)
            offset = math.hypot(float(xs.float().mean()) - width / 2.0,
                                float(ys.float().mean()) - height / 2.0)
            self.assertLess(offset, 0.12 * height, f"the subject sits off-centre at frame {index}")

    def test_the_sloped_spiral_carries_its_own_up(self):
        """A sloped spiral is levelled to its own (axis-tilted) up; a level one keeps the world up.

        The renderer levels the clip to the document's `up`. A coil levelled to the *world* up swings
        against its own frame as the clock winds (measured 78.6 deg at 60 deg end angle / 30 deg
        slope) and crosses the world's pole - the flip - at that slope; the spiral's own up is
        perpendicular to its tilted axis, so the coil stays upright in it and that pole moves to
        phi = 90.
        """
        document, _summary = _estimate(_depth_with_subject(), coverage=SPIRAL_COVERAGE,
                                       spiral_slope=30.0)
        up = json.loads(document).get("up")
        self.assertIsNotNone(up)
        slope = math.radians(30.0)
        self.assertAlmostEqual(up[0], 0.0, places=5)
        self.assertAlmostEqual(up[1], -math.cos(slope), places=5)     # the world up, tilted
        self.assertAlmostEqual(up[2], math.sin(slope), places=5)
        self.assertAlmostEqual(math.dist(up, [0.0, 0.0, 0.0]), 1.0, places=5)
        # a level spiral - and every other coverage - leaves the field out: the world up it always had
        for overrides in ({"coverage": SPIRAL_COVERAGE}, {"coverage": ORBIT_COVERAGES[1]}):
            plain, _summary = _estimate(_depth_with_subject(), **overrides)
            self.assertNotIn("up", json.loads(plain))

    def test_the_keys_follow_the_spiral_s_turning(self):
        """The renderer splines the keys, so they sit where the path bends - not every Nth frame."""
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, _metrics = subject_framing(surface, 40.0)
        samples = subject_samples(175, pivot, distance, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        ticks = _spiral_key_frames(samples, 175)
        self.assertEqual(ticks[0], 0)
        self.assertEqual(ticks[-1], 174)
        self.assertEqual(ticks, sorted(set(ticks)))            # strictly increasing, no repeats
        self.assertGreater(len(ticks), 17)                     # denser than the even frame list ...
        # ... and densest where the path turns fastest: right after the pole
        first = [right - left for left, right in zip(ticks, ticks[1:])]
        self.assertLess(first[0], max(first))
        self.assertLess(sum(first[:4]), sum(first[-4:]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_the_spiral_spacing_survives_a_cuda_surface(self):
        """A depth model on the GPU hands us CUDA points - the estimate must not mix devices.

        `torch.linspace` always builds on the CPU while the pool lives on `cuda:0` when the depth map
        came from a CUDA model. That combination is a hard RuntimeError in `torch.searchsorted`:
        "got self is on cpu, different from other tensors on cuda:0" (`self` is ATen's name for the
        *values* argument).
        """
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        surface = {key: value.cuda() if torch.is_tensor(value) else value
                   for key, value in surface.items()}
        distance, pivot, _metrics = subject_framing(surface, 40.0)
        samples = subject_samples(73, pivot, distance, 1.0, 1.0, None, ORBIT_DIRECTION_DEFAULT,
                                  coverage=SPIRAL_COVERAGE)
        self.assertEqual(len(samples), 73)
        self.assertAlmostEqual(math.dist(samples[-1], pivot), distance, places=6)
        # ... and the full entry point the node calls, with a CUDA image *and* a CUDA depth map
        depth = _depth_with_subject().cuda()
        signal, summary = estimate_camera_path(_reference().cuda(), 73, target=SUBJECT_TARGET,
                                               max_speed=DEFAULT_MAX_SPEED,
                                               depth_fn=lambda image: depth, subject_fill=40.0,
                                               coverage=SPIRAL_COVERAGE)
        keys = json.loads(signal)["path"]
        self.assertGreater(len(keys), 5)               # a real path came back (keys are sparse)
        self.assertEqual(summary["frames"], 73)
        self.assertTrue(summary["visibility_ok"])

    def test_envelope_and_fit_report_the_spiral(self):
        surface = probe_surface(_reference(), depth_fn=lambda reference: _depth_with_subject())
        distance, pivot, metrics = subject_framing(surface, 40.0)
        cap = DEFAULT_MAX_SPEED * metrics["radius_px"]
        keys, info = automatic_keys(73, pivot, distance, surface["content_radius"], SUBJECT_TARGET,
                                    max_speed=DEFAULT_MAX_SPEED, orbit_size=1.0, surface=surface,
                                    cap_px=cap, coverage=SPIRAL_COVERAGE)
        self.assertEqual(info["coverage"], SPIRAL_COVERAGE)
        # The winding is the Spiral End parameter, flown as asked: the fit reports the drift the
        # path costs instead of shortening the coil to fit the cap.
        self.assertAlmostEqual(info["spiral_end"], SPIRAL_END_DEFAULT, places=6)
        self.assertAlmostEqual(info["orbit_end"], SPIRAL_END_DEFAULT, places=6)
        self.assertEqual(info["amplitude_scale"], 1.0)
        self.assertFalse(info["back_orbit"])
        self.assertEqual(info["lap_span"], 0.0)
        self.assertEqual(info["spiral_end_arc"], SPIRAL_END_ARC_DEFAULT)
        self.assertAlmostEqual(info["spiral_end_elevation"], -30.0, places=6)
        self.assertEqual(info["drift_cap_px"], float(cap))
        # the emitted keys really leave the view axis and reach the picture's own plane
        arcs = [spiral_arc(*_orbit_angles(key["pos"], pivot)) for key in keys]
        self.assertGreater(max(arcs), 45.0)
        self.assertAlmostEqual(max(arcs), SPIRAL_END_ARC_DEFAULT, delta=1e-4)
        # the room/fill envelope sees the spiral's own poses, not the front O's
        envelope = _envelope_angles(1.0, coverage=SPIRAL_COVERAGE)
        self.assertEqual(len(envelope), 5)
        envelope_arcs = [spiral_arc(yaw, elevation) for yaw, elevation in envelope]
        self.assertEqual(envelope_arcs, sorted(envelope_arcs))
        self.assertAlmostEqual(envelope_arcs[-1], SPIRAL_END_ARC_DEFAULT, places=6)

    def test_estimate_describes_the_spherical_spiral(self):
        document, summary = _estimate(_depth_with_subject(), coverage=SPIRAL_COVERAGE)
        description = json.loads(document)["description"]
        self.assertIn("spherical spiral", description)
        self.assertIn("side view of the picture", description)
        self.assertIn(f"{SPIRAL_END_DEFAULT:g} deg", description)
        self.assertNotIn("the front O", description)
        self.assertEqual(summary.get("coverage"), SPIRAL_COVERAGE)
        self.assertAlmostEqual(summary.get("spiral_end_deg"), SPIRAL_END_DEFAULT, places=6)
        self.assertAlmostEqual(summary.get("spiral_end_arc_deg"), SPIRAL_END_ARC_DEFAULT, places=6)
        self.assertAlmostEqual(summary.get("spiral_end_elevation_deg"), -30.0, places=6)

    def test_the_spiral_slope_widget_rides_through_the_entry_point(self):
        """`estimate_camera_path(spiral_slope=...)` leans the axis for the whole estimate."""
        _document, summary = _estimate(_depth_with_subject(), coverage=SPIRAL_COVERAGE,
                                       spiral_slope=60.0)
        self.assertAlmostEqual(summary.get("spiral_slope_deg"), 60.0, places=6)
        self.assertEqual(auto_camera.spiral_slope(), SPIRAL_SLOPE_DEFAULT)   # restored afterwards
        # 90 deg is "from straight above": the opening frame sits above the pivot
        _document, summary = _estimate(_depth_with_subject(), coverage=SPIRAL_COVERAGE,
                                       spiral_slope=90.0)
        self.assertAlmostEqual(summary.get("spiral_slope_deg"), SPIRAL_SLOPE_MAX, places=6)
        keys = json.loads(_document)["path"]
        self.assertLess(keys[0]["pos"][1], keys[0]["look"][1])   # above the aim (world y is down)
        # every other coverage ignores it
        _document, summary = _estimate(_depth_with_subject(), coverage=ORBIT_COVERAGES[0])
        self.assertIsNone(summary.get("spiral_slope_deg"))

    def test_the_spiral_end_widget_rides_through_the_entry_point(self):
        """`estimate_camera_path(spiral_end=...)` sets the winding for the whole estimate."""
        _document, summary = _estimate(_depth_with_subject(), coverage=SPIRAL_COVERAGE,
                                       spiral_end=810.0)
        self.assertAlmostEqual(summary.get("spiral_end_deg"), 810.0, places=6)
        self.assertAlmostEqual(summary.get("spiral_end_elevation_deg"), 0.0, places=6)
        self.assertEqual(auto_camera.spiral_end(), SPIRAL_END_DEFAULT)   # restored afterwards


if __name__ == "__main__":
    unittest.main()



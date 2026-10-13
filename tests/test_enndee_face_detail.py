"""Tests for the face-detail pipeline (enndee_face_detect + the tracker mask pass).

No SAM3 model is ever loaded: the detector is patched, so these run in milliseconds
and cover the parts that actually broke while wiring the feature in - the gate's
"never raise" contract, the outlier rule that picks the close-up frames, the mask
geometry, and the widget order that saved workflows depend on.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_face_detect as fd  # noqa: E402
import glomap_lichtfeld_node as glomap_mod  # noqa: E402


def _box(cx=0.475, cy=0.275, width=0.15, height=0.15, score=0.9):
    return {"x0": cx - width / 2, "y0": cy - height / 2,
            "x1": cx + width / 2, "y1": cy + height / 2,
            "cx": cx, "cy": cy, "width": width, "height": height,
            "area": width * height, "score": score,
            "width_px": width * 100, "height_px": height * 100}


class FaceFrameIndicesTests(unittest.TestCase):
    """The close-ups are an outlier group: >= 4x the median face size."""

    def test_the_close_ups_are_picked_out_of_a_body_capture(self):
        # ~0.5 % body frames, then three close-ups at 20 %
        areas = [0.004, 0.005, 0.006, 0.004, 0.005, 0.200, 0.190, 0.210, 0.004]
        self.assertEqual(fd.face_frame_indices(areas), [5, 6, 7])

    def test_a_uniform_capture_selects_nothing(self):
        """No orbit was flown, so no frame is a close-up - the honest answer.

        This is the case a ratio-of-the-maximum rule got wrong: measured on the real
        Lara 180 deg capture the face runs 0.30-0.60 % across the front arc, so
        "within 50 % of the biggest" selected 130 of 175 frames.
        """
        areas = [0.0030, 0.0040, 0.0050, 0.0060, 0.0055, 0.0045]
        self.assertEqual(fd.face_frame_indices(areas), [])

    def test_the_real_lara_distribution_selects_nothing(self):
        areas = [0.00601, 0.00597, 0.00596, 0.00592, 0.00588, 0.00587, 0.00573,
                 0.00560, 0.00555, 0.00541, 0.00300, 0.00310, 0.00320]
        self.assertEqual(fd.face_frame_indices(areas), [])

    def test_frames_without_a_detection_are_never_selected(self):
        self.assertEqual(fd.face_frame_indices([0.0, 0.0, 0.0]), [])
        # the zeros must not drag the median down and make frame 0 look like an outlier
        self.assertEqual(fd.face_frame_indices([0.02, 0.0, 0.0, 0.021, 0.019]), [])

    def test_an_empty_list_is_not_an_error(self):
        self.assertEqual(fd.face_frame_indices([]), [])

    def test_the_multiple_is_tunable(self):
        areas = [0.005, 0.005, 0.02]
        self.assertEqual(fd.face_frame_indices(areas, multiple=4.0), [2])
        self.assertEqual(fd.face_frame_indices(areas, multiple=1.5), [2])
        self.assertEqual(fd.face_frame_indices(areas, multiple=10.0), [])


class FaceBoxRoundTripTests(unittest.TestCase):
    def test_the_string_form_survives_the_round_trip(self):
        face = _box()
        parsed = fd.parse_face_bbox(fd.face_bbox_string(face))
        for key in ("cx", "cy", "width", "height"):
            self.assertAlmostEqual(parsed[key], face[key], places=5)

    def test_no_face_is_an_empty_string_and_parses_to_none(self):
        self.assertEqual(fd.face_bbox_string(None), "")
        self.assertIsNone(fd.parse_face_bbox(""))
        self.assertIsNone(fd.parse_face_bbox(None))

    def test_junk_is_rejected_rather_than_crashing(self):
        for junk in ("abc", "1,2,3", "1,2,3,4,5", "1,2,0,4"):
            self.assertIsNone(fd.parse_face_bbox(junk), junk)

    def test_largest_face_is_the_first_and_none_when_empty(self):
        big, small = _box(width=0.2), _box(width=0.05)
        self.assertIs(fd.largest_face([big, small]), big)
        self.assertIsNone(fd.largest_face([]))


class GateContractTests(unittest.TestCase):
    """A broken or missing detector must never break the run."""

    def test_a_missing_detector_returns_no_faces_and_says_why(self):
        with mock.patch.object(fd, "sam3_available", return_value=(False, "no sam3")):
            faces, report = fd.detect_faces(torch.zeros(1, 8, 8, 3), verbose=False)
        self.assertEqual(faces, [])
        self.assertIn("no sam3", report)

    def test_a_failing_model_load_returns_no_faces_and_says_why(self):
        with mock.patch.object(fd, "sam3_available", return_value=(True, "ok")), \
                mock.patch.object(fd, "load_processor", side_effect=RuntimeError("boom")):
            faces, report = fd.detect_faces(torch.zeros(1, 8, 8, 3), verbose=False)
        self.assertEqual(faces, [])
        self.assertIn("boom", report)

    def test_a_bad_image_is_reported_not_raised(self):
        faces, report = fd.detect_faces(object(), verbose=False)
        self.assertEqual(faces, [])

    def test_the_device_choice_never_raises(self):
        self.assertIn(fd.resolve_device("Auto"), ("cuda", "cpu"))
        self.assertEqual(fd.resolve_device("CPU"), "cpu")


class FaceMaskTests(unittest.TestCase):
    """The COLMAP node's mask pass: geometry, both modes, and the off switch."""

    COUNT, HEIGHT, WIDTH = 6, 96, 128

    def setUp(self):
        self.node = glomap_mod.GLOMAPLichtfeldTracker()
        self.images = torch.rand(self.COUNT, self.HEIGHT, self.WIDTH, 3)
        alpha = torch.zeros(self.COUNT, self.HEIGHT, self.WIDTH, 4)
        alpha[..., :3] = self.images
        alpha[..., 3] = 0.7                     # the RMBG foreground
        self.alpha = alpha
        self.boxes = [_box() for _ in range(self.COUNT)]

    def _masks(self, chosen, mode, dilate=8):
        return self.node._face_mask_from_boxes(
            self.images, self.alpha, chosen, self.boxes, self.WIDTH, self.HEIGHT,
            self.COUNT, dilate, mode, "test")

    def _head(self, mask):
        return mask[0, int(0.25 * self.HEIGHT):int(0.32 * self.HEIGHT),
                    int(0.42 * self.WIDTH):int(0.53 * self.WIDTH)]

    def test_cut_removes_the_head_from_the_body_frames_only(self):
        masks, report = self._masks([1, 2], glomap_mod.FACE_MASK_CUT)
        self.assertIn("head removed from 4 body frame(s)", report)
        for index in range(self.COUNT):
            kept = float(self._head(masks[index]).mean())
            if index in (1, 2):
                self.assertAlmostEqual(kept, 1.0, places=5)     # close-up kept whole
            else:
                self.assertAlmostEqual(kept, 0.0, places=5)     # head cut out

    def test_cut_keeps_the_rest_of_the_rmbg_foreground(self):
        """Only the (dilated) head box goes; the rest stays at the RMBG alpha."""
        masks, _ = self._masks([1, 2], glomap_mod.FACE_MASK_CUT, dilate=0)
        self.assertAlmostEqual(float(masks[0].mean()), 0.7 * (1 - 0.0225), places=3)
        # inside the undilated box: 0, outside: the alpha
        self.assertAlmostEqual(float(masks[0][0, int(0.25 * self.HEIGHT),
                                               int(0.45 * self.WIDTH)]), 0.0, places=5)
        self.assertAlmostEqual(float(masks[0][0, 2, 2]), 0.7, places=5)

    def test_the_dilated_cut_covers_the_dilated_box(self):
        masks, _ = self._masks([1], glomap_mod.FACE_MASK_CUT, dilate=8)
        # 8 px of margin around the 19x14 px box, so the cut is bigger than the box
        self.assertLess(float(masks[0].mean()), 0.7 * (1 - 0.0225))
        self.assertGreater(float(masks[0].mean()), 0.5)

    def test_union_only_adds_and_never_removes(self):
        masks, report = self._masks([1, 2], glomap_mod.FACE_MASK_UNION)
        self.assertIn("face region added to 4 body frame(s)", report)
        for index in range(self.COUNT):
            self.assertAlmostEqual(float(self._head(masks[index]).mean()), 1.0, places=5)

    def test_dilation_grows_the_cut(self):
        tight, _ = self._masks([1], glomap_mod.FACE_MASK_CUT, dilate=0)
        loose, _ = self._masks([1], glomap_mod.FACE_MASK_CUT, dilate=8)
        self.assertLess(float(loose[0].mean()), float(tight[0].mean()))

    def test_the_off_switch_does_nothing(self):
        masks, report = self.node._prepare_face_masks(
            self.images, self.alpha, "", False)
        self.assertIsNone(masks)
        self.assertIn("off", report)

    def test_a_scan_without_a_face_changes_nothing(self):
        with mock.patch.object(glomap_mod.GLOMAPLichtfeldTracker, "_scan_faces",
                               return_value=([0.0] * self.COUNT, [None] * self.COUNT, "none")):
            masks, report = self.node._prepare_face_masks(
                self.images, self.alpha, "", True)
        self.assertIsNone(masks)
        self.assertIn("no frame stands out", report)

    def test_the_scan_picks_the_close_ups_when_no_signal_is_connected(self):
        areas = [0.005] * self.COUNT
        areas[3] = 0.08
        with mock.patch.object(glomap_mod.GLOMAPLichtfeldTracker, "_scan_faces",
                               return_value=(areas, self.boxes, "fake")):
            masks, report = self.node._prepare_face_masks(
                self.images, self.alpha, "", True)
        self.assertIn("close-up", report)
        self.assertAlmostEqual(float(self._head(masks[3]).mean()), 1.0, places=5)

    def test_frame_numbers_are_accepted_0_based_and_clamped(self):
        """0-based, matching the camera node's output; out-of-range is dropped."""
        with mock.patch.object(glomap_mod.GLOMAPLichtfeldTracker, "_scan_faces",
                               return_value=([0.0] * self.COUNT, self.boxes, "fake")):
            masks, report = self.node._prepare_face_masks(
                self.images, self.alpha, "1, 2, 99", True)
        self.assertIn("face_frames input (2 frame(s))", report)
        self.assertAlmostEqual(float(self._head(masks[1]).mean()), 1.0, places=5)
        self.assertAlmostEqual(float(self._head(masks[2]).mean()), 1.0, places=5)

    def test_a_1_based_list_is_dropped_rather_than_shifted(self):
        """count itself is out of range in 0-based, so 6 frames + '6' = nothing."""
        with mock.patch.object(glomap_mod.GLOMAPLichtfeldTracker, "_scan_faces",
                               return_value=([0.0] * self.COUNT, self.boxes, "fake")):
            masks, report = self.node._prepare_face_masks(
                self.images, self.alpha, "6", True)
        self.assertIn("no usable frame", report)
        self.assertIsNone(masks)

    def test_junk_in_the_signal_is_dropped(self):
        with mock.patch.object(glomap_mod.GLOMAPLichtfeldTracker, "_scan_faces",
                               return_value=([0.0] * self.COUNT, self.boxes, "fake")):
            masks, report = self.node._prepare_face_masks(
                self.images, self.alpha, "2, banana, 3.0", True)
        self.assertIn("face_frames input (2 frame(s))", report)
        self.assertIsNotNone(masks)

    def test_a_face_frames_signal_still_measures_the_head_boxes(self):
        """The cut-out needs WHERE the head is, so the scan runs even with a signal."""
        calls = []

        def scan(_images, _prompt, _threshold):
            calls.append(1)
            return [0.0] * self.COUNT, self.boxes, "fake"

        with mock.patch.object(glomap_mod.GLOMAPLichtfeldTracker, "_scan_faces",
                               side_effect=scan):
            self.node._prepare_face_masks(self.images, self.alpha, "2", True)
        self.assertEqual(len(calls), 1)

    def test_without_rmbg_the_mask_is_the_whole_frame(self):
        masks, _ = self._masks([1], glomap_mod.FACE_MASK_CUT)
        self.assertAlmostEqual(float(masks[1].mean()), 1.0, places=5)

    def test_a_gpu_alpha_is_moved_to_the_cpu(self):
        """The mixed-device twin of the camera-path bug.

        `masks_lichtfeld` comes out of `_as_mask_tensor` on the CPU and `track()`
        then does `torch.maximum` against this, so a GPU alpha must not leak through.
        """
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device - the mixed-device path needs one")
        masks, _ = self.node._face_mask_from_boxes(
            self.images, self.alpha.cuda(), [1], self.boxes,
            self.WIDTH, self.HEIGHT, self.COUNT, 8,
            glomap_mod.FACE_MASK_CUT, "test")
        self.assertEqual(masks.device.type, "cpu")
        # exactly the merge `track()` performs
        merged = torch.maximum(torch.zeros_like(masks), masks)
        self.assertEqual(merged.device.type, "cpu")
        self.assertGreater(float(merged.mean()), 0.0)


class WidgetContractTests(unittest.TestCase):
    """The pack's rule: new widgets go LAST, or saved workflows shift."""

    def test_both_tracker_nodes_offer_the_same_trailing_widgets(self):
        import colmap_lichtfeld_node as colmap_mod

        face = list(glomap_mod.GLOMAPLichtfeldTracker._face_widgets())
        added = list(glomap_mod.GLOMAPLichtfeldTracker._new_widgets())
        for cls in (glomap_mod.GLOMAPLichtfeldTracker, colmap_mod.ColmapLichtfeldTracker):
            optional = list(cls.INPUT_TYPES()["optional"])
            self.assertEqual(optional[-len(face) - len(added):], face + added,
                             cls.__name__)

    def test_the_dense_mvs_widgets_keep_their_positions(self):
        """The pycolmap subclass must re-append the face widgets after its own."""
        import colmap_lichtfeld_node as colmap_mod

        optional = list(colmap_mod.ColmapLichtfeldTracker.INPUT_TYPES()["optional"])
        self.assertLess(optional.index("export_depth_maps"), optional.index("face_frames"))
        self.assertLess(optional.index("mesh_method"), optional.index("face_frames"))

    def test_the_signature_defaults_match_the_widgets(self):
        import inspect

        import colmap_lichtfeld_node as colmap_mod

        widgets = glomap_mod.GLOMAPLichtfeldTracker._face_widgets()
        for cls in (glomap_mod.GLOMAPLichtfeldTracker, colmap_mod.ColmapLichtfeldTracker):
            params = inspect.signature(cls.track).parameters
            for name, widget in widgets.items():
                self.assertIn(name, params, f"{cls.__name__}.{name}")
                self.assertEqual(params[name].default, widget[1]["default"],
                                 f"{cls.__name__}.{name}")

    def test_the_camera_node_carries_the_face_orbit_widgets(self):
        import enndee_meridian_parameters as meridian

        required = meridian.MeridianParametersAndCamera.INPUT_TYPES()["required"]
        for name in ("face_orbit", "face_fill", "face_yaw_span", "face_prompt",
                     "face_threshold"):
            self.assertIn(name, required, name)
        self.assertEqual(required["face_orbit"][0], ["Off", "Auto (SAM3)"])
        self.assertEqual(required["face_orbit"][1]["default"], "Off")
        self.assertEqual(required["face_fill"][1]["default"], 20.0)
        self.assertEqual(required["face_yaw_span"][1]["default"], 60.0)
        self.assertEqual(required["face_threshold"][1]["default"], 0.75)

    def test_the_camera_node_appends_its_new_widgets_last(self):
        import enndee_meridian_parameters as meridian

        required = list(meridian.MeridianParametersAndCamera.INPUT_TYPES()["required"])
        # The face block, with `depth_relief` appended after it (2026-10-13) - both go LAST
        # so no saved workflow's positional widget_values move.
        self.assertEqual(required[-6:], ["face_orbit", "face_fill", "face_yaw_span",
                                         "face_prompt", "face_threshold", "depth_relief"])

    def test_the_camera_node_exposes_face_frames(self):
        import enndee_meridian_parameters as meridian

        cls = meridian.MeridianParametersAndCamera
        self.assertIn("face_frames", cls.RETURN_NAMES)
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))


class FaceOrbitGeometryTests(unittest.TestCase):
    """The window selection and the yaw/elevation inverse of `_place`."""

    def setUp(self):
        import enndee_meridian_auto_camera as cam
        self.cam = cam

    def test_yaw_elevation_inverts_place(self):
        pivot = [0.0, 0.0, 0.0]
        for yaw in (-90.0, -30.0, 0.0, 45.0, 90.0):
            for elevation in (-20.0, 0.0, 30.0):
                position = self.cam._place(pivot, 2.5, yaw, elevation)
                got_yaw, got_elevation, radius = self.cam._yaw_elevation_of(position, pivot)
                self.assertAlmostEqual(got_yaw, yaw, places=4)
                self.assertAlmostEqual(got_elevation, elevation, places=4)
                self.assertAlmostEqual(radius, 2.5, places=4)

    def test_the_window_is_the_whole_top_band(self):
        pivot = [0.0, 0.0, 0.0]
        keys = [{"pos": self.cam._place(pivot, 2.0, yaw, elevation), "look": list(pivot),
                 "src": "t", "t": 0.0}
                for elevation in (-20.0, 20.0)
                for yaw in (50.0, 25.0, 0.0, -25.0, -50.0)]
        # FACE_ORBIT_WHOLE_TOP_BAND (the current design): the yaw span is ignored and the
        # whole +20 deg band flies around the FACE pivot end to end.
        self.assertEqual(self.cam.face_orbit_window(keys, pivot, span=60.0), [5, 6, 7, 8, 9])
        # With the switch off, `span` narrows the band to centre_yaw +- span/2.
        with mock.patch.object(self.cam, "FACE_ORBIT_WHOLE_TOP_BAND", False):
            self.assertEqual(self.cam.face_orbit_window(keys, pivot, span=60.0), [6, 7, 8])

    def test_a_narrow_span_can_leave_nothing_to_do(self):
        pivot = [0.0, 0.0, 0.0]
        keys = [{"pos": self.cam._place(pivot, 2.0, yaw, 20.0), "look": list(pivot),
                 "src": "t", "t": 0.0} for yaw in (50.0, -50.0)]
        # only when the span narrows the selection: whole-top-band mode keeps the band
        with mock.patch.object(self.cam, "FACE_ORBIT_WHOLE_TOP_BAND", False):
            self.assertEqual(self.cam.face_orbit_window(keys, pivot, span=10.0), [])
        self.assertEqual(self.cam.face_orbit_window(keys, pivot, span=10.0), [0, 1])

    def test_no_face_box_leaves_the_keys_untouched(self):
        pivot = [0.0, 0.0, 0.0]
        keys = [{"pos": self.cam._place(pivot, 2.0, 0.0, 20.0), "look": list(pivot),
                 "src": "t", "t": 0.0}]
        before = [dict(key) for key in keys]
        surface = {"pivot": pivot, "aspect": 0.75}
        out, info = self.cam.apply_face_orbit(keys, surface, None, pivot=pivot)
        self.assertEqual(out, before)
        self.assertEqual(info["status"], "no face box")

    def test_a_face_box_with_no_surface_points_is_reported(self):
        pivot = [0.0, 0.0, 0.0]
        keys = [{"pos": self.cam._place(pivot, 2.0, yaw, 20.0), "look": list(pivot),
                 "src": "t", "t": 0.0} for yaw in (-20.0, 0.0, 20.0)]
        surface = {"pivot": pivot, "aspect": 0.75, "subject_cloud": torch.zeros(0, 3)}
        out, info = self.cam.apply_face_orbit(
            keys, surface, _box(), pivot=pivot, front_position=[0.0, 0.0, -2.0])
        self.assertIn("no surface points", info["status"])
        self.assertEqual(len(out), len(keys))

    def test_no_reference_pose_is_reported(self):
        pivot = [0.0, 0.0, 0.0]
        keys = [{"pos": self.cam._place(pivot, 2.0, yaw, 20.0), "look": list(pivot),
                 "src": "t", "t": 0.0} for yaw in (-20.0, 0.0, 20.0)]
        surface = {"pivot": pivot, "aspect": 0.75}
        _out, info = self.cam.apply_face_orbit(keys, surface, _box(), pivot=pivot)
        self.assertEqual(info["status"], "no reference pose")


class FaceOrbitDeviceTests(unittest.TestCase):
    """The projection must follow the points' device.

    Regression: the camera basis was built with CPU constants while the surface cloud
    arrives on ``cuda:0`` in a real ComfyUI run, so ``pool - origin`` raised
    "Expected all tensors to be on the same device, cuda:0 and cpu". CPU-only tests
    cannot see that, so this class runs the geometry on the GPU when one is present
    and skips otherwise.
    """

    #: where the face sits in the reference frame - matches the measured Lara box
    FACE_U, FACE_V = 0.44, 0.28
    FACE_W, FACE_H = 0.06, 0.07
    HEIGHT, ASPECT = 1000.0, 0.75

    def setUp(self):
        import enndee_meridian_auto_camera as cam
        self.cam = cam
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device - the mixed-device path needs one")
        self.device = torch.device("cuda")
        self.pivot = [0.0, 0.0, 0.0]
        # camera at _place(pivot, 2, 0, 0) looking at the pivot: x right, y down,
        # z forward (see `WORLD_UP = (0, -1, 0)`), so world = cam + (X, Y, Z - 2).
        self.front = self.cam._place(self.pivot, 2.0, 0.0, 0.0)
        self.look = self.cam._aim(self.pivot, self.front)
        # The fixture builds its cloud at `fast_depth.VFOV_DEGREES`, so the vfov the orbit
        # resolves and projects with must be the same one. `_resolve_fill_vfov` PREFERs the
        # published `LAST_VFOV_DEGREES`, which is module state and leaks between tests run
        # in one process - with a stale value `_project_source` uses a focal the fixture was
        # not built with and the box selects nothing ("the face box holds no surface points").
        self.cam.fast_depth.LAST_VFOV_DEGREES = self.cam.fast_depth.VFOV_DEGREES

    def _face_cloud(self, count=400):
        """A tight cloud that projects into the face box, on the GPU.

        Built in the SOURCE camera frame - origin at the camera, +z along the view axis -
        because that is exactly what `surface_points` unprojects and therefore exactly what
        `_project_source` registers the SAM3 box against. The old fixture shifted the cloud
        into the orbit's front-pose frame instead, which is why the three `apply_face_orbit`
        tests broke the moment box registration started using the correct camera.
        """
        import math
        focal = 0.5 * self.HEIGHT / math.tan(
            math.radians(self.cam.fast_depth.VFOV_DEGREES) / 2.0)
        width = self.HEIGHT * self.ASPECT
        z = 1.5
        centre = torch.tensor(
            [(self.FACE_U - 0.5) * width / focal * z,
             (self.FACE_V - 0.5) * self.HEIGHT / focal * z, z],
            dtype=torch.float64, device=self.device)
        half = torch.tensor(
            [self.FACE_W * 0.5 * width / focal * z,
             self.FACE_H * 0.5 * self.HEIGHT / focal * z, 0.05],
            dtype=torch.float64, device=self.device)
        jitter = (torch.rand(count, 3, dtype=torch.float64, device=self.device) - 0.5) * 2.0
        return centre.reshape(1, 3) + jitter * half.reshape(1, 3)

    def test_projection_follows_the_points_device(self):
        points = self._face_cloud(8)
        u, v, z = self.cam._project_with_pose(points, self.front, self.look,
                                              self.HEIGHT, self.ASPECT)
        for tensor in (u, v, z):
            self.assertEqual(tensor.device.type, points.device.type)

    def test_the_face_patch_projects_back_into_the_box(self):
        """The cloud <-> box round trip must agree, or the zoom aims at nothing.

        The round trip is `_project_source` - the exact inverse of `surface_points` and the
        projection `_face_points` registers the SAM3 box with. Asserting it here is what
        catches a registration that has silently drifted onto the wrong camera.
        """
        points = self._face_cloud()
        u, v, _z = self.cam._project_source(points, self.HEIGHT, self.ASPECT)
        self.assertGreater(float(u.mean()), self.FACE_U - 0.02)
        self.assertLess(float(u.mean()), self.FACE_U + 0.02)
        self.assertGreater(float(v.mean()), self.FACE_V - 0.02)
        self.assertLess(float(v.mean()), self.FACE_V + 0.02)

    def test_project_source_inverts_surface_points(self):
        """A pixel round-tripped through unprojection then projection lands where it began."""
        import math
        height, width = self.HEIGHT, int(self.HEIGHT * self.ASPECT)
        focal = 0.5 * height / math.tan(
            math.radians(self.cam.fast_depth.VFOV_DEGREES) / 2.0)
        col, row, z = 12.0, 7.0, 1.5
        p = torch.tensor([[(col - width / 2.0) / focal * z,
                           (row - height / 2.0) / focal * z, z]],
                         dtype=torch.float64, device=self.device)
        u, v, _ = self.cam._project_source(p, height, self.ASPECT)
        self.assertAlmostEqual(float(u), col / width, places=6)
        self.assertAlmostEqual(float(v), row / height, places=6)

    def test_apply_face_orbit_runs_on_a_gpu_cloud(self):
        keys = [{"pos": self.cam._place(self.pivot, 2.0, yaw, 20.0), "look": list(self.pivot),
                 "src": "t", "t": 0.0}
                for yaw in (25.0, 0.0, -25.0)]
        surface = {"pivot": self.pivot, "aspect": self.ASPECT,
                   "subject_cloud": self._face_cloud()}
        out, info = self.cam.apply_face_orbit(
            keys, surface, _box(cx=self.FACE_U, cy=self.FACE_V,
                                width=self.FACE_W, height=self.FACE_H),
            fill=20.0, span=60.0, pivot=self.pivot,
            front_position=self.front, front_look=self.look)
        self.assertEqual(info["status"], info["status"])          # never raises
        self.assertEqual(info["status"], "applied")
        self.assertEqual(len(info["frames"]), 3)
        # The distance now comes from `head_distance_for_fill`, which models the patch as
        # a SPHERE of radius r. `_projection_area` measures the percentile bbox of the
        # actual points, so for a non-spherical patch the two differ by the shape factor.
        # The old assertion (`fill_after == 0.20` to 3 places) tested the fixed-point loop
        # converging on a metric we have since shown is disconnected from the render, so
        # it is no longer the right contract. Assert the invariants that matter now:
        # the camera moved toward the face, and the achieved area is in the same order as
        # the target rather than the 0.3 % the buggy version produced.
        self.assertGreater(info["fill_after"], info["fill_before"])
        # The patch is a jittered BOX (uniform jitter across x/y/z half-extents), and
        # `head_distance_for_fill` models it as a SPHERE of radius r - so the achieved disc
        # area lands below the geometric target by the shape factor. Measured: 0.096 for a
        # 0.20 target (~0.48x). The bound below is an order-of-magnitude guard, not an
        # equality: what must hold is that the camera moved onto the face and the area is in
        # the target's league rather than the 0.3 % the mis-registered version produced.
        self.assertGreater(info["fill_after"], 0.4 * 0.20)
        # `face_fill` is a GEOOMETRIC frame share and the closed form reaches it directly;
        # the `SAM3_FACE_AREA_RATIO` calibration that briefly drove the target to 0.755 was
        # removed (it overshot the measured peak). The disc area is bounded by the raw
        # target again.
        self.assertLess(info["fill_after"], 2.0 * 0.20)
        for key in out:
            self.assertTrue(all(isinstance(value, float) for value in key["pos"]))

    def test_moving_keys_keeps_the_src_equals_t_invariant(self):
        """The face orbit must never break real-time source indexing.

        Regression: ``apply_face_orbit`` stamped ``keys[i]["src"] = "face"``, and both
        path consumers - ``enndee_meridian_geometry._parse_custom_camera`` and
        ``enndee_meridian_fast_depth._parse_custom_camera`` - do
        ``int(key["src"])`` and then require ``src == t``. A workflow with a face orbit
        therefore died with "Every camera-path key must have integer t and src frame
        indices." The orbit moves the CAMERA only: the frame count and the source
        indexing are deliberately unchanged, so ``src`` must stay equal to ``t`` and the
        close-up marker belongs in a separate field.

        Every other camera-path test already asserts ``src == t``; this one covers the
        orbit, which is the one writer that could violate it.
        """
        keys = [{"pos": self.cam._place(self.pivot, 2.0, yaw, 20.0), "look": list(self.pivot),
                 "src": index, "t": index}
                for index, yaw in enumerate((25.0, 0.0, -25.0))]
        surface = {"pivot": self.pivot, "aspect": self.ASPECT,
                   "subject_cloud": self._face_cloud()}
        out, info = self.cam.apply_face_orbit(
            keys, surface, _box(cx=self.FACE_U, cy=self.FACE_V,
                                width=self.FACE_W, height=self.FACE_H),
            fill=20.0, span=60.0, pivot=self.pivot,
            front_position=self.front, front_look=self.look)
        self.assertEqual(info["status"], "applied")
        self.assertEqual(len(info["frames"]), 3)

        for key in out:
            self.assertIsInstance(key["src"], int, f"src became {key['src']!r}")
            self.assertEqual(key["src"], key["t"],
                             "real-time source indexing broken - the parsers would raise")
        # and the close-up marker is preserved separately
        for index in info["frames"]:
            self.assertTrue(out[index].get("face"),
                            "the face marker was lost when src stopped carrying it")

    def test_a_cpu_cloud_still_works(self):
        """The device fix must not have made CPU tensors unusable."""
        keys = [{"pos": self.cam._place(self.pivot, 2.0, yaw, 20.0), "look": list(self.pivot),
                 "src": "t", "t": 0.0}
                for yaw in (25.0, 0.0, -25.0)]
        cloud = self._face_cloud().cpu()
        surface = {"pivot": self.pivot, "aspect": self.ASPECT, "subject_cloud": cloud}
        _out, info = self.cam.apply_face_orbit(
            keys, surface, _box(cx=self.FACE_U, cy=self.FACE_V,
                                width=self.FACE_W, height=self.FACE_H),
            fill=20.0, span=60.0, pivot=self.pivot,
            front_position=self.front, front_look=self.look)
        self.assertEqual(info["status"], "applied")


if __name__ == "__main__":
    unittest.main()

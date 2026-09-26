"""Tests for the unified Enndee Meridian parameter and camera-path builder."""

import importlib.util
import json
import math
import re
import shlex
import sys
import unittest
from pathlib import Path


PACK_DIR = Path(__file__).resolve().parents[1]
PICKER_PATH = PACK_DIR / "nodes" / "enndee_meridian_parameter_picker.py"
sys.path.insert(0, str(PICKER_PATH.parent))
SPEC = importlib.util.spec_from_file_location("enndee_meridian_parameter_picker_test", PICKER_PATH)
PICKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PICKER_MODULE)
PICKER = PICKER_MODULE.MeridianParameterPickerEnndee

from enndee_meridian_camera_path import SWEEP_SUBSTEPS  # noqa: E402


def args_defaults():
    """Argument-builder widgets only; the camera-path and toggle widgets are filtered out."""
    return {
        name: metadata.get("default", options[0] if isinstance(options, list) else None)
        for name, (options, metadata) in PICKER.INPUT_TYPES()["required"].items()
        if name not in PICKER_MODULE.PATH_WIDGET_NAMES and name != "use_custom_camera"
    }


def all_defaults():
    return {
        name: metadata.get("default", options[0] if isinstance(options, list) else None)
        for name, (options, metadata) in PICKER.INPUT_TYPES()["required"].items()
    }


class MeridianParameterPickerTests(unittest.TestCase):
    def test_still_image_default_emits_full_73_frame_camera_sweep(self):
        tokens = shlex.split(PICKER_MODULE.build_meridian_arguments(**args_defaults()))
        self.assertEqual(tokens[tokens.index("--frames") + 1], "73")
        self.assertEqual(tokens[tokens.index("--freeze") + 1], "0:73")
        self.assertEqual(tokens[tokens.index("--yaw-from") + 1], "-15")
        self.assertEqual(tokens[tokens.index("--yaw") + 1], "15")
        self.assertIn("--sweep", tokens)
        self.assertIn("--ease", tokens)

    def test_158_frame_180_degree_orbit(self):
        values = args_defaults()
        values.update(output_frames="158", yaw_from=90.0, yaw_to=-90.0)
        tokens = shlex.split(PICKER_MODULE.build_meridian_arguments(**values))
        self.assertEqual(tokens[tokens.index("--frames") + 1], "158")
        self.assertEqual(tokens[tokens.index("--freeze") + 1], "0:158")
        self.assertEqual(tokens[tokens.index("--yaw-from") + 1], "90")
        self.assertEqual(tokens[tokens.index("--yaw") + 1], "-90")

    def test_bounce_on_frozen_source_gets_linear_base_sweep(self):
        values = args_defaults()
        values["path_mode"] = "Bounce"
        tokens = shlex.split(PICKER_MODULE.build_meridian_arguments(**values))
        self.assertIn("--sweep", tokens)
        self.assertIn("--bounce", tokens)

    def test_windows_paths_with_spaces_survive_shlex_roundtrip(self):
        values = args_defaults()
        values.update(vggt_repo=r"D:\A Path\VGGT", vggt_checkpoint=r"D:\Model Store\vggt.pt")
        tokens = shlex.split(PICKER_MODULE.build_meridian_arguments(**values))
        self.assertEqual(tokens[tokens.index("--vggt-repo") + 1], "D:/A Path/VGGT")
        self.assertEqual(tokens[tokens.index("--vggt") + 1], "D:/Model Store/vggt.pt")

    def test_invalid_output_length_is_rejected(self):
        values = args_defaults()
        values["output_frames"] = "150"
        with self.assertRaisesRegex(ValueError, "73, 90, 107, 124, 141, 158, 175, 243"):
            PICKER_MODULE.build_meridian_arguments(**values)

    def test_freeze_window_must_fit_output_length(self):
        values = args_defaults()
        values.update(freeze_full_output=False, freeze_length=90)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            PICKER_MODULE.build_meridian_arguments(**values)

    def test_camera_path_cannot_be_combined_with_frozen_still(self):
        values = args_defaults()
        values["camera_path"] = "path.json"
        with self.assertRaisesRegex(ValueError, "requires Freeze Source"):
            PICKER_MODULE.build_meridian_arguments(**values)

    def test_diagnostics_toggle_emits_geometry_only_flag(self):
        values = args_defaults()
        values["diagnostics_only"] = True
        tokens = shlex.split(PICKER_MODULE.build_meridian_arguments(**values))
        self.assertIn("--gauge-only", tokens)

    def test_custom_camera_off_returns_empty_signal_and_keeps_motion_args(self):
        args, signal = PICKER().build(**all_defaults())
        self.assertEqual(signal, "")
        tokens = shlex.split(args)
        self.assertIn("--freeze", tokens)
        self.assertIn("--yaw", tokens)

    def test_custom_camera_mode_drops_flags_and_builds_the_path(self):
        values = all_defaults()
        values.update(use_custom_camera=True, output_frames="124")
        args, signal = PICKER().build(**values)
        tokens = shlex.split(args)
        for flag in ("--start", "--freeze", "--yaw-from", "--yaw", "--truck", "--boom",
                     "--dolly", "--pivot", "--pivot-lock", "--aim", "--pivot-to",
                     "--follow", "--smooth", "--sweep", "--ease", "--live-speed"):
            self.assertNotIn(flag, tokens)
        self.assertEqual(tokens[tokens.index("--frames") + 1], "124")
        self.assertIn("--seed", tokens)
        document = json.loads(signal)
        self.assertEqual(document["frames"], 124)
        self.assertEqual(document["stations"], ["Right", "Front", "Left"])
        self.assertEqual(document["path"][-1]["t"], 123)

    def test_custom_camera_mode_ignores_hidden_freeze_values(self):
        values = all_defaults()
        values.update(use_custom_camera=True, freeze_source=True, freeze_full_output=False,
                      freeze_length=90, freeze_frame=5, source_start=10)
        args, signal = PICKER().build(**values)
        self.assertNotIn("--freeze", shlex.split(args))
        self.assertEqual(json.loads(signal)["frames"], 73)


    def test_camera_path_is_unified_with_output_frames_and_start_station(self):
        values = all_defaults()
        values.update(use_custom_camera=True, output_frames="124", path_start_station="RightBack",
                      path_orbit_left_back=True, path_orbit_right_back=True)
        args, signal = PICKER().build(**values)
        document = json.loads(signal)
        self.assertEqual(document["frames"], 124)
        self.assertEqual(document["stations"], ["RightBack", "Right", "Front", "Left", "LeftBack"])
        tokens = shlex.split(args)
        self.assertEqual(tokens[tokens.index("--frames") + 1], "124")

    def test_alternating_height_mode_builds_yaw_sweep_with_alternating_arcs(self):
        values = all_defaults()
        values.update(
            use_custom_camera=True, output_frames="124",
            path_camera_mode=PICKER_MODULE.HEIGHT_SWEEP_MODE,
            path_start_yaw=-90.0, path_target_yaw=90.0,
            path_low_elevation=-30.0, path_high_elevation=30.0,
            path_arc_switches=3, path_first_arc="Low arc",
        )
        args, signal = PICKER().build(**values)
        document = json.loads(signal)
        self.assertEqual(document["frames"], 124)
        self.assertEqual(document["stations"], ["HeightSweep"])
        keys = document["path"]
        self.assertEqual(keys[0]["t"], 0)
        self.assertEqual(keys[-1]["t"], 123)
        self.assertTrue(all(key["src"] == key["t"] for key in keys))
        self.assertEqual(len(keys), 1 + 3 * SWEEP_SUBSTEPS)
        pivot = (0.0, 0.0, 1.0)
        radius = math.hypot(*pivot)

        def yaw_of(key):
            return math.degrees(math.atan2(key["pos"][0] - pivot[0], pivot[2] - key["pos"][2]))

        def elevation_of(key):
            return math.degrees(math.asin((pivot[1] - key["pos"][1]) / radius))

        # Azimuth runs monotonically from the start to the target yaw.
        yaws = [yaw_of(key) for key in keys]
        self.assertAlmostEqual(yaws[0], -90.0, places=3)
        self.assertAlmostEqual(yaws[-1], 90.0, places=3)
        self.assertTrue(all(right > left for left, right in zip(yaws, yaws[1:])))
        # Every key keeps the source-camera-to-pivot radius and looks at the pivot.
        for key in keys:
            self.assertAlmostEqual(math.dist(key["pos"], pivot), radius, places=6)
            self.assertEqual(key["look"], [0.0, 0.0, 1.0])
        # Odd switch count: low -> high -> low -> high, easing at each apex.
        apexes = [elevation_of(keys[index]) for index in range(0, len(keys), SWEEP_SUBSTEPS)]
        self.assertEqual(len(apexes), 4)
        for observed, expected in zip(apexes, (-30.0, 30.0, -30.0, 30.0)):
            self.assertAlmostEqual(observed, expected, places=2)
        # Mid-segment keys stay between the arcs (the pendulum is moving there);
        # the 0.001-degree slack absorbs the 6-decimal position rounding.
        self.assertTrue(all(-30.001 <= elevation_of(key) <= 30.001 for key in keys))
        self.assertIn("--frames", shlex.split(args))

    def test_alternating_height_mode_validates_arcs_and_sweep(self):
        base = all_defaults()
        base.update(use_custom_camera=True, path_camera_mode=PICKER_MODULE.HEIGHT_SWEEP_MODE)

        with self.assertRaisesRegex(ValueError, "below High Arc Elevation"):
            PICKER().build(**{**base, "path_low_elevation": 40.0, "path_high_elevation": -40.0})
        with self.assertRaisesRegex(ValueError, "must differ"):
            PICKER().build(**{**base, "path_start_yaw": 45.0, "path_target_yaw": 45.0})
        with self.assertRaisesRegex(ValueError, "at least 1"):
            PICKER().build(**{**base, "path_arc_switches": 0})

    def test_spiral_sweep_mode_builds_a_monotone_low_speed_sweep(self):
        values = all_defaults()
        values.update(
            use_custom_camera=True, output_frames="175",
            path_camera_mode=PICKER_MODULE.SPIRAL_SWEEP_MODE,
            path_start_yaw=0.0, path_target_yaw=270.0,
            path_spiral_start_elevation=-20.0, path_spiral_end_elevation=45.0,
        )
        args, signal = PICKER().build(**values)
        document = json.loads(signal)
        self.assertEqual(document["frames"], 175)
        self.assertEqual(document["stations"], ["SpiralSweep"])
        keys = document["path"]
        self.assertEqual((keys[0]["t"], keys[-1]["t"]), (0, 174))
        pivot = (0.0, 0.0, 1.0)
        radius = math.hypot(*pivot)

        def elevation_of(key):
            return math.degrees(math.asin((pivot[1] - key["pos"][1]) / radius))

        def yaw_of(key):
            return math.degrees(math.atan2(key["pos"][0] - pivot[0], pivot[2] - key["pos"][2]))

        self.assertAlmostEqual(yaw_of(keys[0]), 0.0, places=3)
        self.assertAlmostEqual(elevation_of(keys[0]), -20.0, places=2)
        self.assertAlmostEqual(elevation_of(keys[-1]), 45.0, places=2)
        # 270 degrees of travel end 90 degrees away from the start, so the start
        # surface never reappears in the last frames.
        separation = abs(((yaw_of(keys[-1]) - yaw_of(keys[0])) + 180.0) % 360.0 - 180.0)
        self.assertAlmostEqual(separation, 90.0, places=3)
        for key in keys:
            self.assertAlmostEqual(math.dist(key["pos"], pivot), radius, places=5)
        self.assertIn("--frames", shlex.split(args))

    def test_path_dolly_zooms_out_every_path_style(self):
        pivot = (0.0, 0.0, 1.0)
        base_distance = math.hypot(*pivot)
        for mode, extra in (
            (PICKER_MODULE.HEIGHT_SWEEP_MODE,
             {"path_low_elevation": -20.0, "path_high_elevation": 30.0}),
            (PICKER_MODULE.SPIRAL_SWEEP_MODE,
             {"path_spiral_start_elevation": -20.0, "path_spiral_end_elevation": 30.0}),
        ):
            with self.subTest(mode=mode):
                values = all_defaults()
                values.update(use_custom_camera=True, output_frames="124", path_camera_mode=mode,
                              path_start_yaw=0.0, path_target_yaw=180.0, path_dolly=0.4, **extra)
                _args, signal = PICKER().build(**values)
                for key in json.loads(signal)["path"]:
                    self.assertAlmostEqual(math.dist(key["pos"], pivot), base_distance + 0.4, places=5)
        # The O-orbit stations move out too and keep their loop diameter on the
        # shifted sphere.
        values = all_defaults()
        values.update(
            use_custom_camera=True, path_camera_mode=PICKER_MODULE.CAMERA_MODE_OPTIONS[0],
            path_orbit_front=True, path_orbit_left=False, path_orbit_right=False,
            path_orbit_back=False, path_orbit_up=False, path_orbit_down=False,
            path_orbit_left_back=False, path_orbit_right_back=False, path_dolly=0.5,
        )
        _args, signal = PICKER().build(**values)
        distances = [math.dist(key["pos"], pivot) for key in json.loads(signal)["path"]]
        self.assertGreater(min(distances), base_distance)

    def test_path_dolly_must_keep_the_camera_outside_the_pivot(self):
        values = all_defaults()
        values.update(use_custom_camera=True, path_camera_mode=PICKER_MODULE.SPIRAL_SWEEP_MODE,
                      path_dolly=-1.0)
        with self.assertRaisesRegex(ValueError, "at least 0.15"):
            PICKER().build(**values)

    def test_picker_exposes_args_and_camera_signal_outputs(self):
        self.assertEqual(PICKER.RETURN_TYPES, ("STRING", PICKER_MODULE.CAMERA_SIGNAL_TYPE))
        self.assertEqual(PICKER.RETURN_NAMES, ("args", "custom_camera"))

    def test_each_picker_control_has_a_tooltip(self):
        controls = PICKER.INPUT_TYPES()["required"]
        self.assertTrue(controls)
        self.assertTrue(all("tooltip" in metadata for _, metadata in controls.values()))

    def test_frontend_visibility_groups_cover_the_widgets(self):
        script = (PACK_DIR / "web" / "js" / "enndee_meridian_parameter_picker.js").read_text(encoding="utf-8")
        widget_names = set(PICKER.INPUT_TYPES()["required"])

        def array(name):
            match = re.search(rf"const {name} = \[([^\]]*)\];", script)
            self.assertIsNotNone(match, f"{name} missing from the visibility extension")
            return {value.strip().strip('"') for value in match.group(1).split(",") if value.strip()}

        o_orbit_panel = array("O_ORBIT_PANEL")
        height_panel = array("HEIGHT_SWEEP_PANEL")
        spiral_panel = array("SPIRAL_SWEEP_PANEL")
        shared_panel = array("CUSTOM_CAMERA_SHARED")
        custom_panel = o_orbit_panel | height_panel | spiral_panel | shared_panel
        hidden_in_custom = array("HIDDEN_IN_CUSTOM_MODE")
        detail_groups = (
            array("FREEZE_DETAILS") | array("PIVOT_DETAILS") | array("PIVOT_TO_DETAILS")
            | array("FOLLOW_DETAILS") | array("CANVAS_DETAILS") | array("FULL_DETAILS")
        )
        always_visible = {
            "output_frames", "use_custom_camera", "cull", "diagnostics_only", "seed",
            "canvas_enabled", "full_enabled", "vggt_repo", "vggt_checkpoint",
        }

        self.assertTrue(custom_panel <= widget_names)
        self.assertTrue(hidden_in_custom <= widget_names)
        self.assertTrue(detail_groups <= widget_names)
        self.assertEqual(custom_panel | hidden_in_custom | detail_groups | always_visible, widget_names)
        self.assertEqual(len(o_orbit_panel), 10)
        self.assertEqual(height_panel, {
            "path_start_yaw", "path_target_yaw", "path_low_elevation",
            "path_high_elevation", "path_arc_switches", "path_first_arc",
        })
        self.assertEqual(spiral_panel, {
            "path_start_yaw", "path_target_yaw",
            "path_spiral_start_elevation", "path_spiral_end_elevation",
        })
        self.assertEqual(shared_panel, {
            "path_camera_mode", "path_dolly",
            "path_pivot_x", "path_pivot_y", "path_pivot_z",
        })
        for master in ("use_custom_camera", "path_camera_mode", "freeze_source", "freeze_full_output",
                       "pivot_enabled", "aim", "pivot_to_enabled", "follow", "canvas_enabled",
                       "full_enabled"):
            self.assertIn(f'"{master}"', script)
        # The camera-path mode labels must stay identical in python and in the
        # frontend dependency map.
        for constant, label in (
            ("O_ORBIT_MODE", PICKER_MODULE.CAMERA_MODE_OPTIONS[0]),
            ("HEIGHT_SWEEP_MODE", PICKER_MODULE.HEIGHT_SWEEP_MODE),
            ("SPIRAL_SWEEP_MODE", PICKER_MODULE.SPIRAL_SWEEP_MODE),
        ):
            mode_match = re.search(rf'const {constant} = "([^"]*)";', script)
            self.assertIsNotNone(mode_match, f"{constant} missing from the visibility extension")
            self.assertEqual(mode_match.group(1), label)
        # Panels may share widgets (the yaw pair lives in both sweep panels), so
        # the runtime must clear every panel and re-add only the active one;
        # deleting the inactive lists from a global set hid the shared yaw
        # widgets (regression: Spiral Sweep showed no Start/Target Yaw).
        self.assertIn("panel.forEach((name) => visible.delete(name));", script)
        self.assertIn("activePanel.forEach((name) => visible.add(name));", script)
        # Regression (custom-camera widgets never reappeared without a browser
        # refresh): the collapse helper must track its own overrides with an
        # explicit flag and restore the originals exactly - standard widgets own
        # no computeSize/draw, so a restore test based on the saved value would
        # never run.
        self.assertIn("widget.__enndeeCollapsed = true;", script)
        self.assertIn("delete widget.computeSize;", script)
        self.assertIn("delete widget.draw;", script)


if __name__ == "__main__":
    unittest.main()


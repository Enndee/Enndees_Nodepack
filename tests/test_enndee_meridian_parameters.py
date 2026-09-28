"""Tests for "Meridian Parameters and Camera (Enndee)" - the unified node.

The manual camera modes run the real path builders (pure maths, no model). The automatic mode
stubs `estimate_camera_path`, so no depth model is loaded anywhere in this suite; the estimator
itself is covered by `test_enndee_meridian_auto_camera.py`.
"""

import contextlib
import importlib.util
import io
import json
import math
import shlex
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_meridian_parameters as parameters  # noqa: E402
from enndee_meridian_camera_path import ORBIT_OPTIONS, SWEEP_SUBSTEPS  # noqa: E402

NODE = parameters.MeridianParametersAndCamera
JS_PATH = PACK_DIR / "web" / "js" / "enndee_meridian_parameters.js"


def all_defaults():
    return {
        name: metadata.get("default", options[0] if isinstance(options, list) else None)
        for name, (options, metadata) in NODE.INPUT_TYPES()["required"].items()
    }


def manual_defaults(**overrides):
    values = all_defaults()
    values["camera_mode"] = parameters.MANUAL_MODE
    values.update(overrides)
    return values


def estimate_stub(document='{"frames": 73, "path": []}', summary=None):
    """A fake estimator recording its call, so no depth model runs."""
    calls = []
    report = {
        "target": "subject", "frames": 73, "source": "input mask", "points": 4096,
        "content_points": 1024, "pivot": [0.0, 0.0, 2.0], "extents": [1.0, 1.0, 1.0],
        "pivot_offset": [0.0, 0.0, 0.0], "content_radius": 0.5, "orbit_radius": 1.1,
        "style": "front O-orbit + height lap", "amplitude_scale": 0.75,
        "travel_per_frame": 0.02, "budget_per_frame": 0.06, "keys": 19, "max_speed": 0.02,
        "scene_radius": 3.0, "scene_pivot": [0.0, 0.0, 2.5], "collision_fixes": 0,
        "collision_margin": 0.15, "clearance_before": 0.4, "clearance_after": 0.4,
    }
    report.update(summary or {})

    def fake(reference, frames, **kwargs):
        calls.append({"reference": reference, "frames": frames, **kwargs})
        return document, report

    return fake, calls, report


def surface_stub(pivot=(0.1, -0.2, 2.0), radius=0.5, cloud=None):
    """A fake `probe_surface` result: pure torch/vector maths, no depth model."""
    points = cloud if cloud is not None else torch.zeros(64, 3)
    return {
        "target": "subject", "source": "input mask",
        "scene_points": points, "scene_radius": 3.0,
        "scene_pivot": [0.0, 0.0, 2.0], "scene_extents": [1.0, 1.0, 1.0],
        "points": int(points.shape[0]), "content_points": 64,
        "pivot": list(pivot), "extents": [1.0, 1.0, 1.0],
        "content_radius": radius,
    }


class MeridianParametersWidgetTests(unittest.TestCase):
    def test_widget_set_matches_the_new_design(self):
        required = NODE.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required)[:5],
            ["output_frames", "camera_mode", "auto_target", "auto_max_speed", "auto_path_mode"],
        )
        self.assertIn("cull", required)
        for name in parameters.PATH_WIDGET_NAMES:
            self.assertIn(name, required)
        self.assertTrue(all("tooltip" in metadata for _options, metadata in required.values()))
        self.assertEqual(set(NODE.INPUT_TYPES()["optional"]), {"reference_image", "subject_mask"})

    def test_removed_widgets_are_gone(self):
        required = NODE.INPUT_TYPES()["required"]
        for removed in (
            "use_custom_camera", "diagnostics_only", "seed", "camera_path",
            "canvas_enabled", "canvas_width", "canvas_height", "full_enabled", "full_size",
            "vggt_repo", "vggt_checkpoint",
            "source_start", "freeze_source", "freeze_frame", "freeze_full_output", "freeze_length",
            "yaw_from", "yaw_to", "truck", "boom", "dolly", "zoom",
            "pivot_enabled", "pivot_x", "pivot_y", "pivot_lock", "aim",
            "pivot_to_enabled", "pivot_to_x", "pivot_to_y",
            "path_mode", "ease", "live_speed", "fast_back", "follow", "smooth",
        ):
            self.assertNotIn(removed, required)

    def test_camera_mode_and_auto_defaults(self):
        required = NODE.INPUT_TYPES()["required"]
        self.assertEqual(required["camera_mode"][0], list(parameters.CAMERA_MODES))
        self.assertEqual(required["camera_mode"][1]["default"], parameters.MANUAL_MODE)
        self.assertEqual(required["auto_target"][1]["default"], "subject")
        self.assertAlmostEqual(required["auto_max_speed"][1]["default"], 12.0)
        self.assertEqual(required["auto_path_mode"][0], list(parameters.AUTO_PATH_MODES))
        self.assertEqual(required["auto_path_mode"][1]["default"], parameters.AUTOMATIC_PATH)
        for axis in ("x", "y", "z"):
            metadata = required[f"auto_pivot_{axis}"][1]
            self.assertEqual(metadata["default"], 0.0)
            self.assertEqual((metadata["min"], metadata["max"]), (-1.0, 1.0))
        self.assertFalse(all_defaults()["cull"])

    def test_javascript_mirrors_the_widget_names_and_mode_labels(self):
        source = JS_PATH.read_text(encoding="utf-8")
        for name in parameters.PATH_WIDGET_NAMES + parameters.AUTO_WIDGET_NAMES:
            self.assertIn(f'"{name}"', source)
        self.assertIn(f'"{parameters.MANUAL_MODE}"', source)
        self.assertIn(f'"{parameters.AUTOMATIC_MODE}"', source)
        self.assertIn(f'"{parameters.AUTOMATIC_PATH}"', source)
        self.assertIn('"Enndee_MeridianParametersAndCamera"', source)


class MeridianArgumentTests(unittest.TestCase):
    def test_args_carry_only_frames_and_cull(self):
        self.assertEqual(parameters.build_meridian_arguments("124"), "--frames 124")
        self.assertEqual(parameters.build_meridian_arguments(73, True), "--frames 73 --cull")
        with self.assertRaisesRegex(ValueError, "Unsupported Meridian output length"):
            parameters.build_meridian_arguments("150")

    def test_node_rejects_an_unsupported_output_length(self):
        values = manual_defaults(output_frames="150")
        with self.assertRaisesRegex(ValueError, "Unsupported Meridian output length"):
            NODE().build(**values)


class MeridianManualCameraTests(unittest.TestCase):
    def test_o_orbits_rotate_to_the_picked_start_station(self):
        values = manual_defaults(output_frames="124", path_orbit_left=False,
                                 path_orbit_right=False, path_orbit_left_back=True,
                                 path_orbit_right_back=True, path_start_station="RightBack")
        args, signal = NODE().build(**values)
        document = json.loads(signal)
        self.assertEqual(document["frames"], 124)
        self.assertEqual(document["stations"], ["RightBack", "Front", "LeftBack"])
        self.assertEqual(shlex.split(args), ["--frames", "124"])

    def test_default_selection_uses_the_builders_sweep_order(self):
        args, signal = NODE().build(**manual_defaults())
        document = json.loads(signal)
        self.assertEqual(document["stations"], ["Right", "Front", "Left"])
        keys = document["path"]
        self.assertEqual(keys[0]["t"], 0)
        self.assertEqual(keys[-1]["t"], 72)
        self.assertTrue(all(key["src"] == key["t"] for key in keys))

    def test_alternating_height_mode_builds_the_pendulum(self):
        values = manual_defaults(output_frames="124",
                                 path_camera_mode=parameters.HEIGHT_SWEEP_MODE,
                                 path_start_yaw=-90.0, path_target_yaw=90.0,
                                 path_low_elevation=-30.0, path_high_elevation=30.0,
                                 path_arc_switches=3, path_first_arc="Low arc")
        _, signal = NODE().build(**values)
        document = json.loads(signal)
        self.assertEqual(document["stations"], ["HeightSweep"])
        keys = document["path"]
        self.assertEqual(len(keys), 1 + 3 * SWEEP_SUBSTEPS)

        def yaw_of(key):
            return math.degrees(math.atan2(key["pos"][0], 1.0 - key["pos"][2]))

        def elevation_of(key):
            return math.degrees(math.asin(-key["pos"][1]))

        yaws = [yaw_of(key) for key in keys]
        self.assertAlmostEqual(yaws[0], -90.0, places=3)
        self.assertAlmostEqual(yaws[-1], 90.0, places=3)
        self.assertTrue(all(right > left for left, right in zip(yaws, yaws[1:])))
        apexes = [elevation_of(keys[index]) for index in range(0, len(keys), SWEEP_SUBSTEPS)]
        for observed, expected in zip(apexes, (-30.0, 30.0, -30.0, 30.0)):
            self.assertAlmostEqual(observed, expected, places=2)

    def test_spiral_sweep_mode_starts_in_front_and_ends_higher(self):
        values = manual_defaults(output_frames="175",
                                 path_camera_mode=parameters.SPIRAL_SWEEP_MODE,
                                 path_start_yaw=0.0, path_target_yaw=270.0,
                                 path_spiral_start_elevation=-20.0,
                                 path_spiral_end_elevation=45.0)
        _, signal = NODE().build(**values)
        document = json.loads(signal)
        self.assertEqual(document["stations"], ["SpiralSweep"])
        keys = document["path"]
        # yaw 0 sits between the source camera and the pivot, so the first key's z is *below*
        # the pivot depth (1.0 by default) - the source-camera side of the orbit.
        self.assertLess(keys[0]["pos"][2], 1.0)
        self.assertLess(keys[-1]["pos"][1], keys[0]["pos"][1])    # ends above the start (y is down)


class MeridianAutomaticCameraTests(unittest.TestCase):
    def test_automatic_mode_estimates_and_prints_the_summary(self):
        fake, calls, _report = estimate_stub(document='{"frames": 73, "path": []}')
        values = all_defaults()
        values.update(camera_mode=parameters.AUTOMATIC_MODE, auto_target="scene",
                      auto_max_speed=20.0, auto_pivot_x=0.25, output_frames="73", cull=True)
        reference = torch.zeros(1, 8, 8, 3)
        mask = torch.ones(8, 8)
        buffer = io.StringIO()
        with mock.patch.object(parameters, "estimate_camera_path", side_effect=fake), \
                contextlib.redirect_stdout(buffer):
            args, signal = NODE().build(reference_image=reference, subject_mask=mask, **values)
        self.assertEqual(signal, '{"frames": 73, "path": []}')
        self.assertEqual(shlex.split(args), ["--frames", "73", "--cull"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["frames"], 73)
        self.assertEqual(calls[0]["target"], "scene")
        self.assertAlmostEqual(calls[0]["max_speed"], 0.2)
        self.assertEqual(calls[0]["pivot_offset"], (0.25, 0.0, 0.0))
        self.assertIs(calls[0]["subject_mask"], mask)
        self.assertIs(calls[0]["reference"], reference)
        printed = buffer.getvalue()
        self.assertIn("[Enndee] Meridian auto camera:", printed)
        self.assertIn("2 %/frame", printed)          # the speed cap is reported in percent

    def test_automatic_mode_requires_the_reference_image(self):
        values = all_defaults()
        values["camera_mode"] = parameters.AUTOMATIC_MODE
        with self.assertRaisesRegex(ValueError, "needs the reference image"):
            NODE().build(**values)

    def test_manual_mode_never_calls_the_estimator(self):
        def explode(*args, **kwargs):
            raise AssertionError("the estimator must not run in manual mode")

        with mock.patch.object(parameters, "estimate_camera_path", side_effect=explode):
            args, signal = NODE().build(**manual_defaults())
        self.assertIn("--frames", args)
        self.assertTrue(signal)

    def test_invalid_camera_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown Meridian camera mode"):
            NODE().build(**manual_defaults(camera_mode="bogus"))

    def test_automatic_manual_path_flies_the_estimated_pivot(self):
        values = all_defaults()
        values.update(camera_mode=parameters.AUTOMATIC_MODE,
                      auto_path_mode=parameters.MANUAL_PATH,
                      auto_pivot_y=0.5, output_frames="73",
                      path_camera_mode=parameters.SPIRAL_SWEEP_MODE,
                      path_start_yaw=0.0, path_target_yaw=90.0,
                      path_spiral_start_elevation=-20.0, path_spiral_end_elevation=20.0)
        surface = surface_stub(pivot=(0.1, -0.2, 2.0), radius=0.5)
        expected = [0.1, -0.2 + 0.5 * 0.5, 2.0]          # pivot + 0.5 content radii on y

        def explode(*args, **kwargs):
            raise AssertionError("the estimated path must not run when the manual one is picked")

        buffer = io.StringIO()
        with mock.patch.object(parameters, "probe_surface", return_value=surface), \
                mock.patch.object(parameters, "estimate_camera_path", side_effect=explode), \
                contextlib.redirect_stdout(buffer):
            args, signal = NODE().build(reference_image=torch.zeros(1, 8, 8, 3), **values)
        document = json.loads(signal)
        self.assertEqual(document["frames"], 73)
        self.assertIn("on the estimated pivot", document["name"])
        self.assertIn("geometric midpoint", document["description"])
        self.assertIn("Collision guard", document["description"])
        self.assertEqual(shlex.split(args), ["--frames", "73"])
        for key in document["path"]:
            for axis in range(3):
                self.assertAlmostEqual(key["look"][axis], expected[axis], places=5)
        printed = buffer.getvalue()
        self.assertIn("manual path on the estimated pivot", printed)
        self.assertIn("collision fix(es)", printed)

    def test_automatic_path_mode_is_validated(self):
        values = all_defaults()
        values.update(camera_mode=parameters.AUTOMATIC_MODE, auto_path_mode="bogus")
        with self.assertRaisesRegex(ValueError, "Unknown automatic Meridian path mode"):
            NODE().build(reference_image=torch.zeros(1, 8, 8, 3), **values)


class MeridianParametersRegistrationTests(unittest.TestCase):
    def test_registration_replaces_the_retired_nodes(self):
        spec = importlib.util.spec_from_file_location(
            "enndee_nodepack_parameters_test",
            PACK_DIR / "__init__.py",
            submodule_search_locations=[str(PACK_DIR)],
        )
        nodepack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodepack)
        self.assertIs(nodepack.NODE_CLASS_MAPPINGS["Enndee_MeridianParametersAndCamera"], NODE)
        self.assertEqual(
            nodepack.NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianParametersAndCamera"],
            "Meridian Parameters and Camera (Enndee)",
        )
        self.assertNotIn("Enndee_MeridianParameterPicker", nodepack.NODE_CLASS_MAPPINGS)
        self.assertNotIn("Enndee_MeridianCameraPath", nodepack.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()

"""Tests for the Enndee Meridian camera-path configurator and Geometry runner."""

import av
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PACK_DIR = Path(__file__).resolve().parents[1]
NODES_DIR = PACK_DIR / "nodes"
sys.path.insert(0, str(NODES_DIR))

from enndee_meridian_camera_path import (  # noqa: E402
    CAMERA_FRAME_OPTIONS,
    CAMERA_SIGNAL_TYPE,
    CONNECTOR_STEPS,
    ORBIT_OPTIONS,
    ORBIT_WIDGET_NAMES,
    SPIRAL_KEY_COUNT,
    START_STATION_OPTIONS,
    SWEEP_SUBSTEPS,
    MeridianCameraPathConfigurator,
    _order_selected_stations,
    _orbit_position,
    _round_position,
    _station_direction,
    _sweep_direction,
    build_meridian_custom_camera,
    build_meridian_height_sweep,
    build_meridian_spiral_sweep,
)
import enndee_meridian_geometry as geometry  # noqa: E402


class MeridianCameraPathConfiguratorTests(unittest.TestCase):
    def test_builds_user_selected_loop_combinations_in_stable_order(self):
        combinations = (
            ["Front"],
            ["Left"],
            ["Back", "Up"],
            ["Front", "Right", "Down"],
            list(ORBIT_OPTIONS),
        )
        for selected in combinations:
            with self.subTest(selected=selected):
                document = json.loads(build_meridian_custom_camera("124", selected, 0.28, 0, 0, 1))
                self.assertEqual(document["frames"], 124)
                self.assertEqual(document["stations"], _order_selected_stations(selected))
                self.assertEqual(document["path"][0]["t"], 0)
                self.assertEqual(document["path"][-1]["t"], 123)
                self.assertTrue(all(key["src"] == key["t"] for key in document["path"]))
                # Each selected station has a nine-key closed O; connector endpoints are shared.
                for station_index in range(len(document["stations"])):
                    loop_start = station_index * (8 + CONNECTOR_STEPS)
                    loop = document["path"][loop_start:loop_start + 9]
                    self.assertEqual(len(loop), 9)
                    self.assertEqual(loop[0]["pos"], loop[-1]["pos"])
                    self.assertTrue(all(key["look"] == [0.0, 0.0, 1.0] for key in loop))

    def test_custom_station_ordering_rules(self):
        # 1. Front + Left + Right exactly -> ['Right', 'Front', 'Left']
        doc = json.loads(build_meridian_custom_camera("124", ["Front", "Left", "Right"], 0.28, 0, 0, 1))
        self.assertEqual(doc["stations"], ["Right", "Front", "Left"])

        # 2. Left + Right without Back, with Up -> ['Front', 'Left', 'Up', 'Right']
        doc = json.loads(build_meridian_custom_camera("124", ["Front", "Left", "Right", "Up"], 0.28, 0, 0, 1))
        self.assertEqual(doc["stations"], ["Front", "Left", "Up", "Right"])

        # 3. Left + Right with Back -> ['Front', 'Left', 'Back', 'Right', 'Up']
        doc = json.loads(build_meridian_custom_camera("175", ["Front", "Left", "Right", "Back", "Up"], 0.28, 0, 0, 1))
        self.assertEqual(doc["stations"], ["Front", "Left", "Back", "Right", "Up"])

        # All eight stations -> full clockwise sweep including the 120-degree rear stations.
        self.assertEqual(
            _order_selected_stations(ORBIT_OPTIONS),
            ["Front", "Left", "LeftBack", "Back", "RightBack", "Right", "Up", "Down"],
        )

        # 4. The 120-degree trio keeps front first and sweeps back-left before back-right.
        self.assertEqual(
            _order_selected_stations(["Front", "LeftBack", "RightBack"]),
            ["Front", "LeftBack", "RightBack"],
        )

        # 5. Rear stations slot between Left/Back and Back/Right in the sweep.
        self.assertEqual(
            _order_selected_stations(["Front", "Left", "Right", "Back", "LeftBack", "RightBack"]),
            ["Front", "Left", "LeftBack", "Back", "RightBack", "Right"],
        )

    def test_rear_stations_sit_120_degrees_from_front_and_each_other(self):
        for pivot in ((0.0, 0.0, 1.0), (0.0, 0.25, 1.0), (0.3, 0.0, 0.85)):
            with self.subTest(pivot=pivot):
                front = _station_direction("Front", pivot)
                left_back = _station_direction("LeftBack", pivot)
                right_back = _station_direction("RightBack", pivot)
                for first, second in ((front, left_back), (front, right_back), (left_back, right_back)):
                    dot = sum(a * b for a, b in zip(first, second))
                    self.assertAlmostEqual(dot, math.cos(math.radians(120.0)), places=6)
                self.assertLess(left_back[0], 0.0)
                self.assertGreater(right_back[0], 0.0)
                self.assertGreater(left_back[2], 0.0)
                self.assertGreater(right_back[2], 0.0)
                # Symmetric triple: the rear stations share -front_y / 2.
                self.assertAlmostEqual(left_back[1], -front[1] / 2.0, places=6)
                self.assertAlmostEqual(right_back[1], -front[1] / 2.0, places=6)
                self.assertAlmostEqual(left_back[1], right_back[1], places=6)

        document = json.loads(build_meridian_custom_camera("124", ["Front", "LeftBack", "RightBack"], 0.28, 0, 0, 1))
        self.assertEqual(document["stations"], ["Front", "LeftBack", "RightBack"])
        self.assertEqual(document["path"][0]["t"], 0)
        self.assertEqual(document["path"][-1]["t"], 123)
        self.assertTrue(all(key["look"] == [0.0, 0.0, 1.0] for key in document["path"]))

    def test_start_station_rotates_the_cycle_to_the_chosen_station(self):
        base = ["Front", "Left", "Right", "Back", "Up"]

        default = json.loads(build_meridian_custom_camera("175", base, 0.28, 0, 0, 1))
        self.assertEqual(default["stations"], ["Front", "Left", "Back", "Right", "Up"])

        document = json.loads(build_meridian_custom_camera("175", base, 0.28, 0, 0, 1, "Back"))
        self.assertEqual(document["stations"], ["Back", "Right", "Up", "Front", "Left"])
        self.assertEqual(document["path"][0]["t"], 0)
        self.assertEqual(document["path"][-1]["t"], 174)
        self.assertTrue(all(key["src"] == key["t"] for key in document["path"]))

        document = json.loads(build_meridian_custom_camera("175", base, 0.28, 0, 0, 1, "Up"))
        self.assertEqual(document["stations"], ["Up", "Front", "Left", "Back", "Right"])

        # A station outside the selection falls back to the optimized visit order.
        document = json.loads(build_meridian_custom_camera("175", base, 0.28, 0, 0, 1, "Down"))
        self.assertEqual(document["stations"], ["Front", "Left", "Back", "Right", "Up"])

        # The 120-degree trio can start at any station in its cycle.
        trio = ["Front", "LeftBack", "RightBack"]
        document = json.loads(build_meridian_custom_camera("124", trio, 0.28, 0, 0, 1, "RightBack"))
        self.assertEqual(document["stations"], ["RightBack", "Front", "LeftBack"])
        document = json.loads(build_meridian_custom_camera("124", trio, 0.28, 0, 0, 1, "LeftBack"))
        self.assertEqual(document["stations"], ["LeftBack", "RightBack", "Front"])

    def test_height_sweep_uses_azimuth_and_elevation_conventions(self):
        document = json.loads(build_meridian_height_sweep("124", -90, 90, -30, 30, 3, 0, 0, 1))
        keys = document["path"]
        self.assertEqual(document["frames"], 124)
        self.assertEqual(document["stations"], ["HeightSweep"])
        self.assertEqual(keys[0]["t"], 0)
        self.assertEqual(keys[-1]["t"], 123)
        self.assertTrue(all(key["src"] == key["t"] for key in keys))
        self.assertEqual(len(keys), 1 + 3 * SWEEP_SUBSTEPS)
        pivot = (0.0, 0.0, 1.0)
        radius = math.hypot(*pivot)
        # 0 degrees = front (here the source camera side), +90 = the subject's
        # right; the start arc elevation is the lower one.
        self.assertAlmostEqual(
            math.degrees(math.atan2(keys[0]["pos"][0] - pivot[0], pivot[2] - keys[0]["pos"][2])),
            -90.0, places=3)
        self.assertAlmostEqual(
            math.degrees(math.atan2(keys[-1]["pos"][0] - pivot[0], pivot[2] - keys[-1]["pos"][2])),
            90.0, places=3)
        for key in keys:
            self.assertAlmostEqual(math.dist(key["pos"], pivot), radius, places=6)
            self.assertEqual(key["look"], [0.0, 0.0, 1.0])

    def test_height_sweep_alternates_arcs_with_odd_and_even_switch_counts(self):
        pivot = (0.0, 0.0, 1.0)
        radius = math.hypot(*pivot)
        for start_arc, switches, low, high in (
            ("Low arc", 3, -30.0, 30.0),
            ("High arc", 4, -25.0, 45.0),
            ("Low arc", 1, -10.0, 60.0),
        ):
            with self.subTest(start_arc=start_arc, switches=switches):
                document = json.loads(build_meridian_height_sweep(
                    "124", -90, 90, low, high, switches, 0, 0, 1, start_arc))
                keys = document["path"]
                apexes = [
                    math.degrees(math.asin((pivot[1] - keys[index]["pos"][1]) / radius))
                    for index in range(0, len(keys), SWEEP_SUBSTEPS)
                ]
                expected = []
                angle = low if start_arc == "Low arc" else high
                for _ in range(switches + 1):
                    expected.append(angle)
                    angle = high if angle == low else low
                self.assertEqual(len(apexes), switches + 1)
                for observed, wanted in zip(apexes, expected):
                    self.assertAlmostEqual(observed, wanted, places=2)
                yaws = [
                    math.degrees(math.atan2(key["pos"][0] - pivot[0], pivot[2] - key["pos"][2]))
                    for key in keys
                ]
                self.assertAlmostEqual(yaws[0], -90.0, places=3)
                self.assertAlmostEqual(yaws[-1], 90.0, places=3)
                self.assertTrue(all(right > left for left, right in zip(yaws, yaws[1:])))

    def test_height_sweep_rejects_invalid_configurations(self):
        with self.assertRaisesRegex(ValueError, "below High Arc Elevation"):
            build_meridian_height_sweep("124", -90, 90, 40, -40, 3, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "must differ"):
            build_meridian_height_sweep("124", 30, 30, -30, 30, 3, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "gimbal lock"):
            build_meridian_height_sweep("124", -90, 90, -30, 80, 3, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_meridian_height_sweep("124", -90, 90, -30, 30, 0, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "between -360 and 360"):
            build_meridian_height_sweep("124", -400, 90, -30, 30, 3, 0, 0, 1)

    def test_spiral_sweep_follows_the_analytic_spiral(self):
        document = json.loads(build_meridian_spiral_sweep("175", 0, 270, -20, 45, 0, 0, 1))
        keys = document["path"]
        self.assertEqual(document["frames"], 175)
        self.assertEqual(document["stations"], ["SpiralSweep"])
        self.assertEqual(len(keys), SPIRAL_KEY_COUNT)
        self.assertEqual((keys[0]["t"], keys[-1]["t"]), (0, 174))
        self.assertTrue(all(key["src"] == key["t"] for key in keys))
        pivot = (0.0, 0.0, 1.0)
        radius = math.hypot(*pivot)
        for key in keys:
            fraction = key["t"] / 174
            direction = _sweep_direction(pivot, 270.0 * fraction, -20.0 + 65.0 * fraction)
            expected = _round_position(
                tuple(pivot[index] + direction[index] * radius for index in range(3)))
            self.assertEqual(key["pos"], expected)
            self.assertEqual(key["look"], [0.0, 0.0, 1.0])
            self.assertAlmostEqual(math.dist(key["pos"], pivot), radius, places=5)

    def test_spiral_sweep_wraps_yaw_and_allows_descending_sweeps(self):
        direct = json.loads(build_meridian_spiral_sweep("124", -90, 90, -20, 30, 0, 0, 1))
        wrapped = json.loads(build_meridian_spiral_sweep("124", 270, 90, -20, 30, 0, 0, 1))
        self.assertEqual(direct["path"][0]["pos"], wrapped["path"][0]["pos"])
        pivot = (0.0, 0.0, 1.0)
        radius = math.hypot(*pivot)
        descending = json.loads(build_meridian_spiral_sweep("124", 0, 180, 50, -30, 0, 0, 1))

        def elevation_of(key):
            return math.degrees(math.asin((pivot[1] - key["pos"][1]) / radius))

        self.assertAlmostEqual(elevation_of(descending["path"][0]), 50.0, places=2)
        self.assertAlmostEqual(elevation_of(descending["path"][-1]), -30.0, places=2)

    def test_spiral_sweep_rejects_invalid_configurations(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            build_meridian_spiral_sweep("124", 45, 45, -20, 30, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "gimbal lock"):
            build_meridian_spiral_sweep("124", 0, 180, -20, 80, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "between -360 and 360"):
            build_meridian_spiral_sweep("124", 0, 400, -20, 30, 0, 0, 1)

    def test_path_dolly_shifts_all_three_path_styles(self):
        pivot = [0.0, 0.0, 1.0]
        radius = math.hypot(*pivot)
        diameter = 0.28
        document = json.loads(build_meridian_custom_camera("124", ["Front"], diameter, 0.0, 0.0, 1.0, dolly=0.5))
        direction = _station_direction("Front", pivot)
        loop_duration = document["path"][-1]["t"] - document["path"][0]["t"]
        for key in document["path"]:
            phase = 360.0 * (key["t"] - document["path"][0]["t"]) / loop_duration
            expected = _round_position(_orbit_position(direction, phase, pivot, diameter, radius + 0.5))
            self.assertEqual(key["pos"], expected)
        spiral = json.loads(build_meridian_spiral_sweep("124", 0, 180, -20, 30, 0, 0, 1, dolly=0.5))
        for key in spiral["path"]:
            self.assertAlmostEqual(math.dist(key["pos"], pivot), radius + 0.5, places=5)
        sweep = json.loads(build_meridian_height_sweep("124", -90, 90, -30, 30, 3, 0, 0, 1, dolly=0.25))
        for key in sweep["path"]:
            self.assertAlmostEqual(math.dist(key["pos"], pivot), radius + 0.25, places=5)
        with self.assertRaisesRegex(ValueError, "at least 0.15"):
            build_meridian_spiral_sweep("124", 0, 180, -20, 30, 0, 0, 1, dolly=-1.0)

    def test_configure_passes_rear_stations_and_start_station(self):
        node = MeridianCameraPathConfigurator()
        document = json.loads(node.configure(
            "124", True, False, False, False, False, False, 0.28, 0.0, 0.0, 1.0,
            orbit_left_back=True, orbit_right_back=True, start_station="RightBack",
        )[0])
        self.assertEqual(document["stations"], ["RightBack", "Front", "LeftBack"])

        legacy = json.loads(node.configure("124", True, True, True, False, False, False, 0.28, 0.0, 0.0, 1.0)[0])
        self.assertEqual(legacy["stations"], ["Right", "Front", "Left"])

    def test_up_station_looks_down_and_orbits_in_camera_plane(self):
        pivot = [0.0, 0.0, 1.0]
        diameter = 0.28
        doc = json.loads(build_meridian_custom_camera("124", ["Up"], diameter, pivot[0], pivot[1], pivot[2]))
        self.assertEqual(doc["stations"], ["Up"])
        path = doc["path"]
        # In OpenCV frame (UP = 0, -1, 0), Up station is elevated above the pivot (y < 0, z < 1).
        direction_up = _station_direction("Up", pivot)
        self.assertLess(direction_up[1], -0.9)  # sin(70 deg) ~ 0.9396
        self.assertLess(direction_up[2], -0.3)  # -cos(70 deg) ~ -0.3420
        # Loop keys circle in the camera viewing plane (cam_right and cam_down), and each
        # key sits at the exact fraction of the O its integer frame offset represents, so
        # the camera keeps a constant angular speed across uneven integer frame gaps.
        loop_duration = path[-1]["t"] - path[0]["t"]
        for key in path:
            phase = 360.0 * (key["t"] - path[0]["t"]) / loop_duration
            expected = _round_position(_orbit_position(direction_up, phase, pivot, diameter, 1.0))
            self.assertEqual(key["pos"], expected)
        # Phase 0 deg and 360 deg share the pure cam_right offset -> x = 0.14.
        self.assertEqual(path[0]["pos"][0], 0.14)
        self.assertEqual(path[-1]["pos"][0], 0.14)
        # The near-90 deg key stays close to the pure cam_down offset -> x ~ 0.
        self.assertLess(abs(path[2]["pos"][0]), 0.02)
        # The near-180 deg key stays close to the opposite cam_right offset -> x ~ -0.14.
        self.assertAlmostEqual(path[4]["pos"][0], -0.14, places=3)
        # All positions look at pivot
        self.assertTrue(all(key["look"] == pivot for key in path))

    def test_integer_key_frames_keep_constant_angular_speed(self):
        """Regression: rounded key times must not turn short frame gaps into speed spikes."""
        pivot = [0.0, 0.0, 1.0]
        per_station = 8 + CONNECTOR_STEPS
        for frames in CAMERA_FRAME_OPTIONS:
            # Eight stations need at least 8 * 8 + 7 * 4 = 92 frames for smooth keys.
            selections = (
                ["Front"],
                ["Front", "Left", "Right"],
                ["Front", "Left", "Right", "Back", "Up"],
            ) + ((list(ORBIT_OPTIONS),) if int(frames) >= 107 else ())
            for selected in selections:
                with self.subTest(frames=frames, selected=selected):
                    doc = json.loads(build_meridian_custom_camera(frames, selected, 0.28, pivot[0], pivot[1], pivot[2]))
                    keys = doc["path"]
                    pivot_point = np.asarray(keys[0]["look"])

                    def angular_rates(group):
                        rates = []
                        for first, second in zip(group, group[1:]):
                            first_u = np.asarray(first["pos"]) - pivot_point
                            second_u = np.asarray(second["pos"]) - pivot_point
                            cosine = float(first_u @ second_u / (np.linalg.norm(first_u) * np.linalg.norm(second_u)))
                            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                            rates.append(angle / (second["t"] - first["t"]))
                        return rates

                    for station in range(len(doc["stations"])):
                        loop_start = station * per_station
                        loop_rates = angular_rates(keys[loop_start:loop_start + 9])
                        self.assertLess(max(loop_rates) / min(loop_rates), 1.10)
                        connector = keys[loop_start + 8:loop_start + 8 + 1 + CONNECTOR_STEPS]
                        if len(connector) > 1:
                            connector_rates = angular_rates(connector)
                            self.assertLess(max(connector_rates) / min(connector_rates), 1.25)

    def test_only_selected_stations_are_included_and_order_is_fixed(self):
        document = json.loads(build_meridian_custom_camera("124", ["Down", "Back", "Left"], 0.28, 0, 0, 1))
        self.assertEqual(document["stations"], ["Left", "Back", "Down"])
        left_start = np.asarray(document["path"][0]["pos"])
        back_start = np.asarray(document["path"][12]["pos"])
        down_start = np.asarray(document["path"][24]["pos"])
        self.assertLess(left_start[0], 0.0)
        self.assertGreater(back_start[2], 1.0)
        self.assertGreater(down_start[1], 0.0)

    def test_vertical_and_antipodal_stations_generate_finite_connected_paths(self):
        selections = (["Front", "Up"], ["Front", "Down"], ["Front", "Back"], ["Up", "Down"])
        for selected in selections:
            with self.subTest(selected=selected):
                document = json.loads(build_meridian_custom_camera("124", selected, 0.28, 0, 0, 1))
                self.assertEqual(document["stations"], list(selected))
                self.assertEqual(document["path"][-1]["t"], 123)
                self.assertTrue(all(math.isfinite(value) for key in document["path"] for vector in ("pos", "look") for value in key[vector]))

    def test_empty_or_unknown_station_selection_is_rejected(self):
        for selected in ([], ["Somewhere else"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                build_meridian_custom_camera("124", selected, 0.28, 0, 0, 1)

    def test_pivot_y_moves_the_orbit_center_and_look_target(self):
        document = json.loads(build_meridian_custom_camera("124", ["Front"], 0.28, 0, 0.25, 1))
        self.assertEqual(document["path"][0]["look"], [0.0, 0.25, 1.0])
        self.assertEqual(document["path"][0]["pos"], [0.14, 0.0, 0.0])

    def test_supported_frame_counts_match_meridian_assets(self):
        self.assertEqual(
            [int(value) for value in CAMERA_FRAME_OPTIONS],
            [73, 90, 107, 124, 141, 158, 175, 243],
        )
        for frames in CAMERA_FRAME_OPTIONS:
            with self.subTest(frames=frames):
                document = json.loads(build_meridian_custom_camera(frames, ["Front", "Left", "Right"], 0.28, 0, 0, 1))
                self.assertEqual(document["path"][-1]["t"], int(frames) - 1)
        # The full eight-station sweep needs the larger Meridian lengths (>= 107 frames).
        for frames in CAMERA_FRAME_OPTIONS:
            if int(frames) < 107:
                continue
            with self.subTest(frames=frames, selection="all"):
                document = json.loads(build_meridian_custom_camera(frames, list(ORBIT_OPTIONS), 0.28, 0, 0, 1))
                self.assertEqual(document["path"][-1]["t"], int(frames) - 1)

    def test_invalid_configuration_is_rejected(self):
        invalid = (
            (("150", ["Front"], 0.28, 0, 0, 1), "Unsupported Meridian frame count"),
            (("124", [], 0.28, 0, 0, 1), "Select at least one"),
            (("124", ["Front"], 0.28, 0, 0, 0), "Pivot X/Z"),
            (("124", ["Front"], float("nan"), 0, 0, 1), "finite number"),
            (("124", ["Front"], 0.0, 0, 0, 1), "greater than zero"),
            (("124", ["NonExistent"], 0.28, 0, 0, 1), "Unknown orbit station"),
            (("73", list(ORBIT_OPTIONS), 0.28, 0, 0, 1), "not enough"),
        )
        for values, message in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                build_meridian_custom_camera(*values)

    def test_every_required_parameter_has_an_explanatory_tooltip(self):
        parameters = MeridianCameraPathConfigurator.INPUT_TYPES()["required"]
        self.assertEqual(
            set(parameters),
            {
                "number_of_frames", "orbit_front", "orbit_left", "orbit_right", "orbit_back",
                "orbit_up", "orbit_down", "orbit_left_back", "orbit_right_back",
                "orbit_diameter", "pivot_x", "pivot_y", "pivot_z", "start_station",
            },
        )
        self.assertTrue(all("tooltip" in metadata for _options, metadata in parameters.values()))
        self.assertTrue(all(
            parameters[ORBIT_WIDGET_NAMES[name]][1]["default"] == (name in ("Front", "Left", "Right"))
            for name in ORBIT_OPTIONS
        ))
        self.assertEqual(parameters["start_station"][1]["default"], "Visit order")
        self.assertEqual(list(parameters["start_station"][0]), list(START_STATION_OPTIONS))
        # Legacy widgets keep their serialized positions; new widgets are appended.
        self.assertEqual(
            list(parameters)[:11],
            ["number_of_frames", "orbit_front", "orbit_left", "orbit_right", "orbit_back",
             "orbit_up", "orbit_down", "orbit_diameter", "pivot_x", "pivot_y", "pivot_z"],
        )
        self.assertEqual(MeridianCameraPathConfigurator.RETURN_TYPES, (CAMERA_SIGNAL_TYPE,))
        self.assertEqual(MeridianCameraPathConfigurator.RETURN_NAMES, ("custom_camera",))


class MeridianGeometryCustomCameraTests(unittest.TestCase):
    class CompletedProcess:
        stdout = "source 512x512 -> canvas (736, 544)"

        @staticmethod
        def check_returncode():
            return None

    def test_custom_camera_repeats_first_image_and_overrides_motion_arguments(self):
        signal = build_meridian_custom_camera("73", ["Front", "Left"], 0.28, 0, 0, 1)
        image = torch.zeros((4, 16, 24, 3), dtype=torch.float32)
        image[0, ..., 0] = 1.0
        image[1, ..., 1] = 1.0
        image[2, ..., 2] = 1.0
        image[3, ...] = 0.5
        observed = {}

        def run_meridian(command, **kwargs):
            observed["command"] = command
            video_path = Path(command[command.index("--video") + 1])
            path_path = Path(command[command.index("--camera-path") + 1])
            observed["video_path"] = video_path
            observed["path_path"] = path_path
            observed["camera_json"] = json.loads(path_path.read_text(encoding="utf-8"))
            with av.open(str(video_path)) as container:
                observed["frames"] = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            observed["video_exists_during_run"] = video_path.is_file()
            observed["json_exists_during_run"] = path_path.is_file()
            return self.CompletedProcess()

        with (
            mock.patch.object(geometry.subprocess, "run", side_effect=run_meridian),
            mock.patch.object(
                geometry,
                "_frames",
                side_effect=[torch.zeros((73, 8, 12, 3)), torch.ones((73, 8, 12, 3))],
            ),
        ):
            result = geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4",
                args="--freeze 0:73 --yaw 40 --frames 124 --camera-path stale.json --vggt-repo repo --vggt checkpoint.pt --cull",
                repo="meridian",
                python="python",
                image=image,
                custom_camera=signal,
            )

        command = observed["command"]
        self.assertEqual(command[command.index("--frames") + 1], "73")
        self.assertEqual(command[command.index("--camera-path") + 1], str(observed["path_path"]))
        self.assertEqual(command[command.index("--start") + 1], "0")
        self.assertIn("--cull", command)
        for conflicting in ("--freeze", "--yaw", "--follow", "stale.json"):
            self.assertNotIn(conflicting, command)
        self.assertEqual(observed["camera_json"]["frames"], 73)
        self.assertEqual(observed["camera_json"]["path"][-1]["t"], 72)
        self.assertEqual(len(observed["frames"]), 73)
        self.assertTrue(observed["video_exists_during_run"])
        self.assertTrue(observed["json_exists_during_run"])
        for frame in observed["frames"]:
            self.assertTrue(np.all(frame[..., 0] > 240))
            self.assertTrue(np.all(frame[..., 1] < 10))
        self.assertFalse(observed["video_path"].exists())
        self.assertFalse(observed["path_path"].exists())
        self.assertEqual(result[2:], (736, 544, 73))

    def test_custom_camera_can_use_first_frame_from_video_path(self):
        signal = build_meridian_custom_camera("73", ["Front"], 0.28, 0, 0, 1)
        frames = np.zeros((2, 12, 16, 3), dtype=np.uint8)
        frames[0, ..., 2] = 255
        frames[1, ..., 0] = 255
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video = Path(temp_dir) / "input.mp4"
            geometry._write_rgb_frames_video(frames, str(input_video))
            observed = {}

            def run_meridian(command, **kwargs):
                path = Path(command[command.index("--video") + 1])
                with av.open(str(path)) as container:
                    observed["frames"] = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
                observed["path"] = path
                return self.CompletedProcess()

            with (
                mock.patch.object(geometry.subprocess, "run", side_effect=run_meridian),
                mock.patch.object(
                    geometry,
                    "_frames",
                    side_effect=[torch.zeros((73, 8, 12, 3)), torch.zeros((73, 8, 12, 3))],
                ),
            ):
                geometry.EnndeeMeridianGeometry().build(
                    video=str(input_video),
                    args="",
                    repo="meridian",
                    python="python",
                    custom_camera=signal,
                )

        self.assertEqual(len(observed["frames"]), 73)
        self.assertTrue(all(frame[..., 2].mean() > frame[..., 0].mean() for frame in observed["frames"]))
        self.assertFalse(observed["path"].exists())

    def test_custom_camera_signal_validation_rejects_invalid_source_mapping(self):
        data = json.loads(build_meridian_custom_camera("73", ["Front"], 0.28, 0, 0, 1))
        data["path"][-1]["src"] = 71
        with self.assertRaisesRegex(ValueError, "src equal to t"):
            geometry._parse_custom_camera(json.dumps(data))

    def test_legacy_single_image_mode_keeps_png_behavior_without_custom_signal(self):
        image = torch.zeros((1, 8, 10, 3), dtype=torch.float32)
        observed = {}

        def run_meridian(command, **kwargs):
            observed["command"] = command
            input_path = Path(command[command.index("--video") + 1])
            observed["path"] = input_path
            observed["exists_during_run"] = input_path.is_file()
            return self.CompletedProcess()

        with (
            mock.patch.object(geometry.subprocess, "run", side_effect=run_meridian),
            mock.patch.object(
                geometry,
                "_frames",
                side_effect=[torch.zeros((73, 8, 10, 3)), torch.zeros((73, 8, 10, 3))],
            ),
        ):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4",
                args="--freeze 0:73",
                repo="meridian",
                python="python",
                image=image,
            )

        self.assertEqual(observed["path"].suffix, ".png")
        self.assertTrue(observed["exists_during_run"])
        self.assertFalse(observed["path"].exists())
        self.assertIn("--freeze", observed["command"])

    def test_custom_path_removes_equals_form_motion_flags_and_keeps_quoted_windows_paths(self):
        args = (
            '--freeze=0:73 --yaw=90 --vggt-repo "D:\\Model Store\\VGGT" '
            '--vggt "D:\\Weights\\vggt.pt" --cull'
        )
        command_args = geometry._args_for_custom_camera(args, "camera path.json", 73)
        self.assertNotIn("--freeze", command_args)
        self.assertNotIn("--yaw", command_args)
        self.assertEqual(command_args[command_args.index("--vggt-repo") + 1], "D:/Model Store/VGGT")
        self.assertEqual(command_args[command_args.index("--vggt") + 1], "D:/Weights/vggt.pt")
        self.assertEqual(command_args[command_args.index("--camera-path") + 1], "camera path.json")
        self.assertIn("--cull", command_args)


if __name__ == "__main__":
    unittest.main()
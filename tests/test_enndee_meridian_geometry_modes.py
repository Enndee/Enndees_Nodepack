"""Tests for the unified Meridian Geometry (Enndee) node: mode dispatch and fast-depth routing.

The Depth-Anything backend is patched at the geometry module's `render_depth_aligned` symbol,
so these tests assert the routing contract - which mode runs which backend, what the fast mode
passes through, and which argument combinations it rejects - without loading a model or a
subprocess. The backend's own behaviour lives in test_enndee_meridian_fast_depth.py.
"""

import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PACK_DIR = Path(__file__).resolve().parents[1]
NODES_DIR = PACK_DIR / "nodes"
sys.path.insert(0, str(NODES_DIR))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402
import enndee_meridian_geometry as geometry  # noqa: E402


def _two_key_path(frames=90):
    return {"frames": frames, "path": [
        {"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
        {"t": frames - 1, "src": frames - 1, "pos": [0.6, 0.1, 0.6], "look": [0.0, 0.0, 1.0]},
    ]}


class _FastCapture:
    """Records the fast-backend call and returns a fixed condition pair."""

    def __init__(self):
        self.kwargs = None
        self.first = None

    def __call__(self, first, device, **kwargs):
        self.first = first
        self.kwargs = kwargs
        length = kwargs.get("frames", 73)
        return (torch.zeros(length, 64, 112, 3), torch.zeros(length, 64, 112, 3), 112, 64, length)


class CompletedProcess:
    stdout = "canvas (64, 64)"

    @staticmethod
    def check_returncode():
        return None


class MeridianGeometryModeTests(unittest.TestCase):
    def test_input_types_expose_the_mode_and_the_fast_widgets(self):
        inputs = geometry.EnndeeMeridianGeometry.INPUT_TYPES()
        self.assertEqual(tuple(inputs["required"]["mode"][0]), geometry.MODE_OPTIONS)
        for name in ("video", "args", "repo", "python", "cache", "cache_dir", "mode", "model_size",
                     "canvas_mode", "custom_width", "custom_height", "cloud_scale", "point_size",
                     "edge_cull", "edge_threshold", "back_face_cull", "depth_res", "canvas_enabled",
                     "canvas_width", "canvas_height", "full_enabled", "full_size", "vggt_repo",
                     "vggt_checkpoint"):
            self.assertIn(name, inputs["required"])
            self.assertIn("tooltip", inputs["required"][name][1], f"{name} needs a tooltip")

    def test_fast_mode_routes_to_the_engine_with_parsed_arguments(self):
        capture = _FastCapture()
        images = torch.zeros(3, 64, 112, 3)
        with mock.patch.object(geometry, "render_depth_aligned", capture), \
             mock.patch.object(geometry.subprocess, "run",
                               side_effect=AssertionError("no subprocess in fast mode")):
            result = geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 90 --yaw-from -15 --yaw 15 --sweep --ease --cull",
                repo="meridian", python="python", image=images, mode=geometry.FAST_DEPTH_MODE)
        self.assertEqual(capture.first.shape, (1, 64, 112, 3))     # first frame of the batch only
        self.assertEqual(capture.kwargs["frames"], 90)
        self.assertEqual(capture.kwargs["camera"]["yaw"], 15.0)
        self.assertEqual(capture.kwargs["camera"]["yaw_from"], -15.0)
        self.assertTrue(capture.kwargs["camera"]["sweep"] and capture.kwargs["camera"]["cull"])
        self.assertIsNone(capture.kwargs["custom_camera"])
        self.assertEqual(result[2:], (112, 64, 90))

    def test_fast_mode_frame_count_defaults_to_meridian_73(self):
        capture = _FastCapture()
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--yaw 10", repo="meridian", python="python",
                image=torch.zeros(1, 64, 112, 3), mode=geometry.FAST_DEPTH_MODE)
        self.assertEqual(capture.kwargs["frames"], 73)

    def test_fast_mode_forwards_the_da3_depth_resolution(self):
        spec = geometry.EnndeeMeridianGeometry.INPUT_TYPES()["required"]["depth_res"]
        self.assertEqual(spec[1]["default"], fast_depth.DA3_RES)   # the fast default
        self.assertEqual(spec[1]["min"], 0)                        # 0 = the still's own resolution
        capture = _FastCapture()
        images = torch.zeros(1, 64, 112, 3)
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", repo="meridian", python="python",
                image=images, mode=geometry.FAST_DEPTH_MODE)
            self.assertEqual(capture.kwargs["depth_res"], fast_depth.DA3_RES)
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", repo="meridian", python="python",
                image=images, mode=geometry.FAST_DEPTH_MODE, depth_res=0)
            self.assertEqual(capture.kwargs["depth_res"], 0)
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", repo="meridian", python="python",
                image=images, mode=geometry.FAST_DEPTH_MODE, depth_res=1008)
            self.assertEqual(capture.kwargs["depth_res"], 1008)

    def test_fast_mode_rejects_unsupported_lengths_and_follow(self):
        node = geometry.EnndeeMeridianGeometry()
        with self.assertRaisesRegex(ValueError, "Meridian output length"):
            node.build(video="unused.mp4", args="--frames 71", repo="meridian", python="python",
                       image=torch.zeros(1, 64, 112, 3), mode=geometry.FAST_DEPTH_MODE)
        with self.assertRaisesRegex(ValueError, "--follow"):
            node.build(video="unused.mp4", args="--follow --frames 73", repo="meridian",
                       python="python", image=torch.zeros(1, 64, 112, 3),
                       mode=geometry.FAST_DEPTH_MODE)

    def test_fast_mode_takes_the_frame_count_and_path_from_custom_camera(self):
        capture = _FastCapture()
        signal = json.dumps(_two_key_path(frames=90))
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--follow --frames 73", repo="meridian", python="python",
                image=torch.zeros(1, 64, 112, 3), custom_camera=signal, mode=geometry.FAST_DEPTH_MODE)
        self.assertEqual(capture.kwargs["frames"], 90)        # the path wins over --frames
        self.assertEqual(capture.kwargs["custom_camera"], signal)

    def test_fast_mode_reads_the_first_video_frame_without_an_image(self):
        capture = _FastCapture()
        frame = torch.full((64, 112, 3), 0.5)
        with mock.patch.object(geometry, "render_depth_aligned", capture), \
             mock.patch.object(geometry, "_read_first_video_frame", return_value=frame), \
             mock.patch.object(geometry.os.path, "isfile", return_value=True):
            geometry.EnndeeMeridianGeometry().build(
                video="clip.mp4", args="--frames 73", repo="meridian", python="python",
                mode=geometry.FAST_DEPTH_MODE)
        self.assertEqual(capture.first.shape, (1, 64, 112, 3))
        with self.assertRaisesRegex(ValueError, "video path"):
            geometry.EnndeeMeridianGeometry().build(
                video="missing.mp4", args="--frames 73", repo="meridian", python="python",
                mode=geometry.FAST_DEPTH_MODE)

    def test_vggt_mode_still_runs_the_subprocess_and_never_the_engine(self):
        capture = _FastCapture()
        with mock.patch.object(geometry, "render_depth_aligned", capture), \
             mock.patch.object(geometry.subprocess, "run", return_value=CompletedProcess()) as run, \
             mock.patch.object(geometry, "_frames", side_effect=[torch.zeros(3, 64, 64, 3)] * 2):
            result = geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--freeze 0:73", repo="meridian", python="python",
                image=torch.zeros(1, 64, 64, 3))
        run.assert_called_once()
        self.assertIsNone(capture.kwargs)
        self.assertEqual(result[2:], (64, 64, 3))


class MeridianGeometryVggtPanelTests(unittest.TestCase):
    """The VGGT canvas/source settings that used to live only on the picker, now on the node."""

    def _run_vggt(self, args, **widgets):
        with mock.patch.object(geometry.subprocess, "run", return_value=CompletedProcess()) as run, \
             mock.patch.object(geometry, "_frames", side_effect=[torch.zeros(3, 64, 64, 3)] * 2):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args=args, repo="meridian", python="python", cache=False,
                image=torch.zeros(1, 64, 64, 3), **widgets)
        return run.call_args[0][0]

    def test_vggt_panel_widgets_append_only_missing_flags(self):
        cmd = self._run_vggt("--freeze 0:73", canvas_enabled=True, canvas_width=864, canvas_height=1184,
                             full_enabled=True, full_size=1280, vggt_repo="D:/vggt",
                             vggt_checkpoint="D:/vggt.pt")
        pairs = list(zip(cmd, cmd[1:]))
        self.assertIn(("--canvas", "864x1184"), pairs)
        self.assertIn(("--full", "1280"), pairs)
        self.assertIn(("--vggt-repo", "D:/vggt"), pairs)
        self.assertIn(("--vggt", "D:/vggt.pt"), pairs)

    def test_vggt_panel_widgets_respect_the_args_string(self):
        cmd = self._run_vggt("--freeze 0:73 --canvas 640x640 --full 1024 "
                             "--vggt-repo D:/picker --vggt D:/picker.pt",
                             canvas_enabled=True, canvas_width=864, canvas_height=1184,
                             full_enabled=True, full_size=1280, vggt_repo="D:/node",
                             vggt_checkpoint="D:/node.pt")
        self.assertEqual(cmd.count("--canvas"), 1)
        self.assertIn("640x640", cmd)
        self.assertNotIn("864x1184", cmd)
        self.assertEqual(cmd.count("--full"), 1)
        self.assertIn("1024", cmd)
        self.assertEqual(cmd.count("--vggt"), 1)
        self.assertIn("D:/picker.pt", cmd)
        self.assertNotIn("D:/node.pt", cmd)

    def test_vggt_panel_widgets_validate_their_sizes(self):
        node = geometry.EnndeeMeridianGeometry()
        with mock.patch.object(geometry.subprocess, "run",
                               side_effect=AssertionError("sizes must validate before any run")):
            with self.assertRaisesRegex(ValueError, "multiples of 32"):
                node.build(video="unused.mp4", args="--freeze 0:73", repo="meridian", python="python",
                           cache=False, image=torch.zeros(1, 64, 64, 3),
                           canvas_enabled=True, canvas_width=100)
            with self.assertRaisesRegex(ValueError, "at least 128"):
                node.build(video="unused.mp4", args="--freeze 0:73", repo="meridian", python="python",
                           cache=False, image=torch.zeros(1, 64, 64, 3),
                           full_enabled=True, full_size=100)

    def test_fast_mode_ignores_the_vggt_panel(self):
        capture = _FastCapture()
        with mock.patch.object(geometry, "render_depth_aligned", capture), \
             mock.patch.object(geometry.subprocess, "run",
                               side_effect=AssertionError("no subprocess in fast mode")):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", repo="meridian", python="python", cache=False,
                image=torch.zeros(1, 64, 112, 3), mode=geometry.FAST_DEPTH_MODE,
                canvas_enabled=True, canvas_width=100, canvas_height=100,   # invalid sizes must not raise
                full_enabled=True, full_size=100, vggt_repo="D:/vggt", vggt_checkpoint="D:/vggt.pt")
        self.assertEqual(capture.kwargs["frames"], 73)

    def test_model_size_options_cover_both_depth_families(self):
        options = geometry.EnndeeMeridianGeometry.INPUT_TYPES()["required"]["model_size"][0]
        self.assertTrue(all("Depth-Anything-V2" in option for option in options[:3]))
        self.assertTrue(set(fast_depth.DA3_MODEL_REPOS) <= set(options))

    def test_frontend_visibility_groups_cover_the_widgets(self):
        script = (PACK_DIR / "web" / "js" / "enndee_meridian_geometry.js").read_text(encoding="utf-8")

        def array(name):
            match = re.search(rf"const {name} = \[([^\]]*)\];", script)
            self.assertIsNotNone(match, f"{name} missing from the visibility extension")
            return {value.strip().strip('"') for value in match.group(1).split(",") if value.strip()}

        fast_panel = array("FAST_DEPTH_PANEL")
        vggt_panel = array("VGGT_PANEL")
        detail_groups = (array("CUSTOM_CANVAS_DETAILS") | array("VGGT_CANVAS_DETAILS")
                         | array("VGGT_FULL_DETAILS"))
        widget_names = set(geometry.EnndeeMeridianGeometry.INPUT_TYPES()["required"])

        self.assertTrue(fast_panel <= widget_names)
        self.assertTrue(vggt_panel <= widget_names)
        # Detail widgets stay inside their master's group (revealed only while it is on).
        self.assertTrue(detail_groups <= fast_panel | vggt_panel)
        # The panels plus the always-visible controls must cover every widget exactly.
        self.assertEqual(fast_panel | vggt_panel | {"video", "args", "mode"}, widget_names)
        # The mode labels must stay identical in python and in the frontend.
        for constant, label in (("VGGT_MODE", geometry.VGGT_MODE),
                                ("FAST_DEPTH_MODE", geometry.FAST_DEPTH_MODE)):
            match = re.search(rf'const {constant} = "([^"]*)";', script)
            self.assertIsNotNone(match, f"{constant} missing from the visibility extension")
            self.assertEqual(match.group(1), label)


if __name__ == "__main__":
    unittest.main()

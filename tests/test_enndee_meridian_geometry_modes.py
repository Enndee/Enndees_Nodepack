"""Tests for the fast-depth-only Meridian Geometry (Enndee) node.

The Depth-Anything backend is patched at the geometry module's `render_depth_aligned` symbol, so
these tests assert the routing contract - what the node passes through, which argument
combinations it rejects - and that the VGGT/subprocess machinery is really gone, without loading
a model or a subprocess.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
NODES_DIR = PACK_DIR / "nodes"
sys.path.insert(0, str(NODES_DIR))
sys.path.insert(0, str(PACK_DIR))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402
import enndee_meridian_geometry as geometry  # noqa: E402


def _two_key_path(frames=90):
    return {"frames": frames, "path": [
        {"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
        {"t": frames - 1, "src": frames - 1, "pos": [0.6, 0.1, 0.6], "look": [0.0, 0.0, 1.0]},
    ]}


class _FastCapture:
    """Records the engine call and returns a fixed condition pair."""

    def __init__(self):
        self.kwargs = None
        self.first = None

    def __call__(self, first, device, **kwargs):
        self.first = first
        self.kwargs = kwargs
        length = kwargs.get("frames", 73)
        return (torch.zeros(length, 64, 112, 3), torch.zeros(length, 64, 112, 3), 112, 64, length)


class MeridianGeometryFastDepthTests(unittest.TestCase):
    def test_widgets_are_the_fast_depth_set(self):
        inputs = geometry.EnndeeMeridianGeometry.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            ["video", "args", "model_size", "canvas_mode", "custom_width", "custom_height",
             "cloud_scale", "point_size", "edge_cull", "edge_threshold", "back_face_cull",
             "depth_res", "external_depth_polarity"],
        )
        self.assertEqual(list(inputs["optional"]),
                         ["image", "args_override", "custom_camera", "external_depth"])
        for name, (_options, metadata) in inputs["required"].items():
            self.assertIn("tooltip", metadata, f"{name} needs a tooltip")

    def test_the_external_depth_inputs_are_appended_last(self):
        # ComfyUI maps a saved workflow's widget_values by INSERTION ORDER, so a widget added
        # in the middle would silently remap every later widget of existing workflows. Any new
        # control therefore goes last.
        inputs = geometry.EnndeeMeridianGeometry.INPUT_TYPES()
        self.assertEqual(list(inputs["required"])[-1], "external_depth_polarity")
        self.assertEqual(list(inputs["optional"])[-1], "external_depth")

    def test_the_vggt_surface_is_gone(self):
        inputs = geometry.EnndeeMeridianGeometry.INPUT_TYPES()
        for removed in ("mode", "repo", "python", "cache", "cache_dir", "canvas_enabled",
                        "canvas_width", "canvas_height", "full_enabled", "full_size",
                        "vggt_repo", "vggt_checkpoint"):
            self.assertNotIn(removed, inputs["required"])
        for symbol in ("VGGT_MODE", "MODE_OPTIONS", "FAST_DEPTH_MODE", "subprocess", "_frames",
                       "_cache_load", "_cache_key_for", "_write_repeated_frame_video",
                       "_write_image_batch_video", "_args_for_custom_camera",
                       "_apply_vggt_source_settings", "_add_default_vggt_paths"):
            self.assertFalse(hasattr(geometry, symbol), f"{symbol} should be gone")
        self.assertFalse((PACK_DIR / "web" / "js" / "enndee_meridian_geometry.js").exists())


    def test_build_routes_to_the_engine_with_parsed_arguments(self):
        capture = _FastCapture()
        images = torch.zeros(3, 64, 112, 3)
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            result = geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 90 --yaw-from -15 --yaw 15 --sweep --ease --cull",
                image=images)
        self.assertEqual(capture.first.shape, (1, 64, 112, 3))     # first frame of the batch only
        self.assertEqual(capture.kwargs["frames"], 90)
        self.assertEqual(capture.kwargs["camera"]["yaw"], 15.0)
        self.assertEqual(capture.kwargs["camera"]["yaw_from"], -15.0)
        self.assertTrue(capture.kwargs["camera"]["sweep"] and capture.kwargs["camera"]["cull"])
        self.assertIsNone(capture.kwargs["custom_camera"])
        self.assertEqual(result[2:], (112, 64, 90))

    def test_args_override_wins_over_the_args_widget(self):
        capture = _FastCapture()
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", args_override="--frames 124",
                image=torch.zeros(1, 64, 112, 3))
        self.assertEqual(capture.kwargs["frames"], 124)

    def test_frame_count_defaults_to_meridian_73(self):
        capture = _FastCapture()
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(video="unused.mp4", args="--yaw 10",
                                                    image=torch.zeros(1, 64, 112, 3))
        self.assertEqual(capture.kwargs["frames"], 73)

    def test_forwards_the_widgets_and_the_da3_depth_resolution(self):
        spec = geometry.EnndeeMeridianGeometry.INPUT_TYPES()["required"]["depth_res"]
        # the default is the example workflow's value: the still's own side (DA3_RES stays the
        # model's native grid and the 0 = "keep the still's resolution" sentinel)
        self.assertEqual(spec[1]["default"], 1920)
        self.assertEqual(spec[1]["min"], 0)                        # 0 = the still's own resolution
        capture = _FastCapture()
        images = torch.zeros(1, 64, 112, 3)
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            for depth_res in (fast_depth.DA3_RES, 0, 1008):
                geometry.EnndeeMeridianGeometry().build(
                    video="unused.mp4", args="--frames 73", image=images, depth_res=depth_res,
                    model_size="Depth-Anything-3-Small", canvas_mode="custom", custom_width=112,
                    custom_height=64, cloud_scale=1, point_size=0, edge_threshold=0.2)
                self.assertEqual(capture.kwargs["depth_res"], depth_res)
        self.assertEqual(capture.kwargs["model_size"], "Depth-Anything-3-Small")
        self.assertEqual(capture.kwargs["canvas_mode"], "custom")
        self.assertEqual((capture.kwargs["custom_width"], capture.kwargs["custom_height"]),
                         (112, 64))
        self.assertEqual(capture.kwargs["cloud_scale"], 1)
        self.assertEqual(capture.kwargs["point_size"], 0)
        self.assertEqual(capture.kwargs["edge_threshold"], 0.2)


    def test_rejects_unsupported_lengths_and_follow(self):
        node = geometry.EnndeeMeridianGeometry()
        with self.assertRaisesRegex(ValueError, "Meridian output length"):
            node.build(video="unused.mp4", args="--frames 71", image=torch.zeros(1, 64, 112, 3))
        with self.assertRaisesRegex(ValueError, "--follow"):
            node.build(video="unused.mp4", args="--follow --frames 73",
                       image=torch.zeros(1, 64, 112, 3))

    def test_custom_camera_supplies_the_frame_count_and_path(self):
        capture = _FastCapture()
        signal = json.dumps(_two_key_path(frames=90))
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--follow --frames 73",
                image=torch.zeros(1, 64, 112, 3), custom_camera=signal)
        self.assertEqual(capture.kwargs["frames"], 90)        # the path wins over --frames
        self.assertEqual(capture.kwargs["custom_camera"], signal)

    def test_invalid_custom_camera_signals_are_rejected(self):
        node = geometry.EnndeeMeridianGeometry()
        image = torch.zeros(1, 64, 112, 3)
        with self.assertRaisesRegex(ValueError, "valid signal"):
            node.build(video="unused.mp4", args="--frames 73", image=image,
                       custom_camera="not json")
        broken = _two_key_path(frames=73)
        broken["path"][-1]["t"] = 5
        with self.assertRaisesRegex(ValueError, "frame count minus one"):
            node.build(video="unused.mp4", args="--frames 73", image=image,
                       custom_camera=json.dumps(broken))

    def test_reads_the_first_video_frame_without_an_image(self):
        capture = _FastCapture()
        frame = torch.full((64, 112, 3), 0.5)
        with mock.patch.object(geometry, "render_depth_aligned", capture), \
                mock.patch.object(geometry, "_read_first_video_frame", return_value=frame), \
                mock.patch.object(geometry.os.path, "isfile", return_value=True):
            geometry.EnndeeMeridianGeometry().build(video="clip.mp4", args="--frames 73")
        self.assertEqual(capture.first.shape, (1, 64, 112, 3))
        with self.assertRaisesRegex(ValueError, "video path"):
            geometry.EnndeeMeridianGeometry().build(video="missing.mp4", args="--frames 73")

    def test_model_size_offers_only_da3_mono_large_or_external(self):
        spec = geometry.EnndeeMeridianGeometry.INPUT_TYPES()["required"]["model_size"]
        self.assertEqual(list(spec[0]), ["Depth Anything 3 Mono Large", "external"])
        self.assertEqual(spec[1]["default"], "Depth Anything 3 Mono Large")

        self.assertEqual(geometry.resolve_depth_model("Depth Anything 3 Mono Large"),
                         ("Depth-Anything-3-Mono-Large", False))
        self.assertEqual(geometry.resolve_depth_model("external"),
                         ("Depth-Anything-3-Mono-Large", True))
        # a workflow saved before the picker shrank must still route instead of exploding
        for legacy in geometry.LEGACY_DEPTH_MODELS:
            self.assertEqual(geometry.resolve_depth_model(legacy), (legacy, False))
        with self.assertRaisesRegex(ValueError, "Unsupported depth model"):
            geometry.resolve_depth_model("nonsense")

    def test_external_depth_replaces_the_depth_model(self):
        capture = _FastCapture()
        depth = torch.linspace(0.0, 1.0, 64 * 112).view(1, 64, 112, 1)
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", image=torch.zeros(1, 64, 112, 3),
                model_size="external", external_depth=depth)
        self.assertIs(capture.kwargs["external_depth"], depth)
        self.assertIs(capture.kwargs["external_depth_invert"], False)
        # the backend still receives a usable model id (it is simply never consulted)
        self.assertEqual(capture.kwargs["model_size"], "Depth-Anything-3-Mono-Large")

    def test_external_depth_polarity_sets_the_invert_flag(self):
        depth = torch.linspace(0.0, 1.0, 64 * 112).view(1, 64, 112, 1)
        for label, expected in geometry.EXTERNAL_DEPTH_POLARITIES.items():
            capture = _FastCapture()
            with mock.patch.object(geometry, "render_depth_aligned", capture):
                geometry.EnndeeMeridianGeometry().build(
                    video="unused.mp4", args="--frames 73", image=torch.zeros(1, 64, 112, 3),
                    model_size="external", external_depth=depth,
                    external_depth_polarity=label)
            self.assertIs(capture.kwargs["external_depth_invert"], expected, label)

    def test_external_model_without_a_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "external_depth input"):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", image=torch.zeros(1, 64, 112, 3),
                model_size="external")

    def test_a_connected_map_is_ignored_unless_the_model_is_external(self):
        capture = _FastCapture()
        depth = torch.linspace(0.0, 1.0, 64 * 112).view(1, 64, 112, 1)
        with mock.patch.object(geometry, "render_depth_aligned", capture):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", image=torch.zeros(1, 64, 112, 3),
                model_size="Depth Anything 3 Mono Large", external_depth=depth)
        self.assertIsNone(capture.kwargs["external_depth"])

    def test_an_unknown_polarity_is_rejected(self):
        depth = torch.linspace(0.0, 1.0, 64 * 112).view(1, 64, 112, 1)
        with self.assertRaisesRegex(ValueError, "polarity"):
            geometry.EnndeeMeridianGeometry().build(
                video="unused.mp4", args="--frames 73", image=torch.zeros(1, 64, 112, 3),
                model_size="external", external_depth=depth,
                external_depth_polarity="sideways")

    def test_registration_maps_the_geometry_node(self):
        spec = importlib.util.spec_from_file_location(
            "enndee_nodepack_geometry_test",
            PACK_DIR / "__init__.py",
            submodule_search_locations=[str(PACK_DIR)],
        )
        nodepack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodepack)
        self.assertIs(nodepack.NODE_CLASS_MAPPINGS["Enndee_MeridianGeometry"],
                      geometry.EnndeeMeridianGeometry)


class MeridianExternalDepthMapTests(unittest.TestCase):
    """The external depth map contract: resize, ordering, polarity, and the guards."""

    def test_resizes_onto_the_working_still_and_keeps_the_ordering(self):
        depth = torch.zeros(1, 32, 64, 1)
        depth[..., :, :32, :] = 1.0                # left half of the WIDTH is far
        out, note = fast_depth.prepare_external_depth(depth, 64, 128)
        self.assertEqual(tuple(out.shape), (64, 128))
        self.assertIsNone(note)                    # 64/32 == 128/64, so the aspect matches
        self.assertGreater(float(out[:, :32].mean()), float(out[:, -32:].mean()))

    def test_invert_flips_the_ordering(self):
        depth = torch.zeros(1, 32, 64, 1)
        depth[..., :, :32, :] = 1.0
        out, _note = fast_depth.prepare_external_depth(depth, 64, 128, invert=True)
        self.assertLess(float(out[:, :32].mean()), float(out[:, -32:].mean()))

    def test_a_colourised_map_is_averaged_to_luminance(self):
        depth = torch.zeros(1, 32, 64, 3)
        depth[..., :, :32, 0] = 1.0                # pure red on the left half of the WIDTH
        out, _note = fast_depth.prepare_external_depth(depth, 64, 128)
        self.assertEqual(tuple(out.shape), (64, 128))
        # the map is resized 64 -> 128 wide, so the boundary lands on column 64; keep a margin
        self.assertAlmostEqual(float(out[:, :56].mean()), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(out[:, 72:].mean()), 0.0, places=5)

    def test_an_aspect_mismatch_is_reported_instead_of_hidden(self):
        depth = torch.linspace(0.0, 1.0, 32 * 64).view(1, 32, 64, 1)   # aspect 2.0
        _out, note = fast_depth.prepare_external_depth(depth, 64, 64)  # still is 1.0
        self.assertIsNotNone(note)
        self.assertIn("stretched", note)

    def test_a_constant_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            fast_depth.prepare_external_depth(torch.ones(1, 32, 64, 1), 64, 64)

    def test_non_finite_values_are_rejected(self):
        bad = torch.zeros(1, 32, 64, 1)
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fast_depth.prepare_external_depth(bad, 64, 64)

    def test_a_non_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "IMAGE tensor"):
            fast_depth.prepare_external_depth([[1.0, 2.0]], 64, 64)

    def test_the_result_moves_to_the_requested_device(self):
        """`device=` is the fix for the Marigold/ComfyUI CPU-map crash."""
        depth = torch.linspace(0.0, 1.0, 32 * 64).view(1, 32, 64, 1)
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        moved, _note = fast_depth.prepare_external_depth(depth, 64, 128, device=target)
        self.assertEqual(moved.device.type, target.type)

        # no device argument keeps the input's device - what every other test here relies on
        kept, _note = fast_depth.prepare_external_depth(depth, 64, 128)
        self.assertEqual(kept.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()

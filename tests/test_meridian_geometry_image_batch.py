"""Tests for how Meridian Geometry converts ComfyUI image input to a source clip."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np
import torch


NODE_PATH = Path(__file__).resolve().parents[2] / "meridian_geometry.py"
SPEC = importlib.util.spec_from_file_location("meridian_geometry_test", NODE_PATH)
MERIDIAN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERIDIAN_MODULE)


class MeridianGeometryImageBatchTests(unittest.TestCase):
    def test_image_batch_is_encoded_as_video_with_all_frames(self):
        rng = np.random.default_rng(1234)
        frames = rng.integers(0, 256, size=(73, 31, 47, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "batch.mp4"

            MERIDIAN_MODULE._write_image_batch_video(frames, str(video_path))

            with av.open(str(video_path)) as container:
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

        self.assertEqual(len(decoded), len(frames))
        self.assertTrue(all(frame.shape == (31, 47, 3) for frame in decoded))
        for expected, actual in zip(frames, decoded):
            np.testing.assert_array_equal(actual, expected)


    def test_build_passes_multiframe_image_as_temporary_video_and_cleans_it_up(self):
        images = torch.zeros((3, 8, 10, 3), dtype=torch.float32)
        images[0, ..., 0] = 1.0
        images[1, ..., 1] = 1.0
        images[2, ..., 2] = 1.0
        observed = {}

        class CompletedProcess:
            stdout = "canvas (64, 64)"

            @staticmethod
            def check_returncode():
                return None

        def fake_run(command, **kwargs):
            observed["command"] = command
            observed["video_path"] = Path(command[command.index("--video") + 1])
            with av.open(str(observed["video_path"])) as container:
                observed["frames"] = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            return CompletedProcess()

        with (
            mock.patch.object(MERIDIAN_MODULE.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                MERIDIAN_MODULE,
                "_frames",
                side_effect=[torch.zeros((3, 8, 10, 3)), torch.zeros((3, 8, 10, 3))],
            ),
        ):
            result = MERIDIAN_MODULE.MeridianGeometry().build(
                video="unused.mp4",
                args="--camera-path path.json",
                repo="meridian",
                python="python",
                image=images,
            )

        video_arg = observed["command"][observed["command"].index("--video") + 1]
        self.assertTrue(video_arg.endswith(".mp4"))
        self.assertTrue(Path(video_arg).name.startswith("meridian_batch_"))
        self.assertEqual(len(observed["frames"]), 3)
        self.assertEqual(result[0].shape[0], 3)
        self.assertEqual(result[1].shape[0], 3)
        self.assertEqual(result[2:], (64, 64, 3))
        self.assertFalse(observed["video_path"].exists())


    def test_one_image_keeps_still_png_behavior(self):
        image = torch.zeros((1, 8, 10, 3), dtype=torch.float32)
        observed = {}

        class CompletedProcess:
            stdout = "canvas (64, 64)"

            @staticmethod
            def check_returncode():
                return None

        def fake_run(command, **kwargs):
            observed["video_path"] = Path(command[command.index("--video") + 1])
            observed["exists_during_run"] = observed["video_path"].is_file()
            return CompletedProcess()

        with (
            mock.patch.object(MERIDIAN_MODULE.subprocess, "run", side_effect=fake_run),
            mock.patch.object(
                MERIDIAN_MODULE,
                "_frames",
                side_effect=[torch.zeros((1, 8, 10, 3)), torch.zeros((1, 8, 10, 3))],
            ),
        ):
            MERIDIAN_MODULE.MeridianGeometry().build(
                video="unused.mp4",
                args="--freeze 0:73",
                repo="meridian",
                python="python",
                image=image,
            )

        self.assertTrue(observed["video_path"].name.startswith("meridian_still_"))
        self.assertEqual(observed["video_path"].suffix, ".png")
        self.assertTrue(observed["exists_during_run"])
        self.assertFalse(observed["video_path"].exists())


if __name__ == "__main__":
    unittest.main()
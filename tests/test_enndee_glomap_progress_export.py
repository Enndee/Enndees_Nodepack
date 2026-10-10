"""Tests for live SfM console output and dataset image-format export."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

from enndee_colmap.colmap_wrapper import run_streaming_command  # noqa: E402
from enndee_colmap.glomap_wrapper import GLOMAPWrapper  # noqa: E402
from glomap_lichtfeld_node import GLOMAPLichtfeldTracker  # noqa: E402


class StreamingCommandTests(unittest.TestCase):
    def test_streams_newline_and_carriage_return_progress(self):
        messages = []
        command = [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('feature stage 1/2'); "
            "sys.stdout.write('matching 50%\\r'); sys.stdout.flush(); "
            "sys.stdout.write('matching 100%\\n'); sys.stdout.flush()",
        ]

        code, output = run_streaming_command(
            command, "test COLMAP", timeout=20, progress_callback=messages.append
        )

        self.assertEqual(code, 0)
        self.assertIn("feature stage 1/2", output)
        self.assertIn("matching 50%", output)
        self.assertIn("matching 100%", output)
        self.assertTrue(any("feature stage 1/2" in message for message in messages))
        self.assertTrue(any("matching 50%" in message for message in messages))

    def test_returns_output_and_nonzero_status(self):
        messages = []
        code, output = run_streaming_command(
            [sys.executable, "-u", "-c", "print('simulated failure'); raise SystemExit(7)"],
            "test mapper",
            timeout=20,
            progress_callback=messages.append,
        )
        self.assertEqual(code, 7)
        self.assertIn("simulated failure", output)
        self.assertTrue(any("simulated failure" in message for message in messages))


class DatasetImageExportTests(unittest.TestCase):
    @staticmethod
    def export(folder, images, **kwargs):
        GLOMAPLichtfeldTracker._export_dataset_images(
            None, folder, images, kwargs.pop("alpha_images", None), **kwargs
        )

    def test_png_default_preserves_alpha_when_requested(self):
        rgb = np.zeros((1, 4, 5, 3), dtype=np.float32)
        alpha = np.ones((1, 4, 5, 4), dtype=np.float32)
        alpha[..., 3] = 0.25
        with tempfile.TemporaryDirectory() as temp_dir:
            self.export(Path(temp_dir), rgb, alpha_images=alpha, embed_alpha=True)
            output = Path(temp_dir) / "images" / "0001.png"
            self.assertTrue(output.is_file())
            with Image.open(output) as saved:
                self.assertEqual(saved.format, "PNG")
                self.assertEqual(saved.mode, "RGBA")
                self.assertEqual(saved.getpixel((0, 0))[3], 64)

    def test_jpeg_uses_quality_and_drops_alpha_channel(self):
        rgba = np.ones((1, 12, 14, 4), dtype=np.float32)
        rgba[..., 3] = 0.2
        with tempfile.TemporaryDirectory() as temp_dir:
            self.export(Path(temp_dir), rgba, image_format="JPEG", jpeg_quality=90)
            output = Path(temp_dir) / "images" / "0001.jpeg"
            self.assertTrue(output.is_file())
            with Image.open(output) as saved:
                self.assertEqual(saved.format, "JPEG")
                self.assertEqual(saved.mode, "RGB")

    def test_switching_format_removes_stale_numbered_files(self):
        images = np.zeros((1, 3, 3, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            (image_dir / "0001.png").write_bytes(b"stale")
            (image_dir / "0002.jpeg").write_bytes(b"stale")
            self.export(Path(temp_dir), images, image_format="JPEG")
            self.assertEqual([p.name for p in image_dir.iterdir()], ["0001.jpeg"])

    def test_invalid_format_and_jpeg_quality_are_rejected(self):
        images = np.zeros((1, 2, 2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            for kwargs, expected in (
                ({"image_format": "WEBP"}, "Unsupported dataset image format"),
                ({"image_format": "JPEG", "jpeg_quality": 0}, "between 1 and 100"),
                ({"image_format": "JPEG", "jpeg_quality": 101}, "between 1 and 100"),
            ):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, expected):
                    self.export(Path(temp_dir), images, **kwargs)

    def test_progress_is_logged_for_each_external_stage(self):
        wrapper = GLOMAPWrapper.__new__(GLOMAPWrapper)
        messages = []
        wrapper.progress_callback = messages.append
        wrapper._log_progress("Feature extraction started")
        self.assertEqual(messages, ["[SfM] Feature extraction started"])


class ImageBatchNormalisationTests(unittest.TestCase):
    """A 0..255 batch must be rescaled, not clamped into a binary mask.

    Regression: driving ``track()`` with an 8-bit array used to make the dataset
    export write ``clamp(x, 0, 1) * 255`` - every non-zero pixel became 255, so
    the frames came out as white/magenta masks and the depth model was fed the
    same mask.  Nothing warned; the run reported success.
    """

    @staticmethod
    def batch(images):
        return GLOMAPLichtfeldTracker._as_image_batch(images)

    def test_uint8_batch_is_rescaled_to_unit_range(self):
        photo = (np.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3) % 256).astype(np.uint8)
        batch = self.batch(photo)
        self.assertEqual(batch.dtype, torch.float32)
        expected = torch.from_numpy(photo.astype(np.float32) / 255.0)
        self.assertTrue(torch.allclose(batch, expected, atol=1e-6))

    def test_torch_uint8_batch_is_rescaled_too(self):
        photo = (np.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3) % 256).astype(np.uint8)
        batch = self.batch(torch.from_numpy(photo.copy()))
        expected = torch.from_numpy(photo.astype(np.float32) / 255.0)
        self.assertTrue(torch.allclose(batch, expected, atol=1e-6))

    def test_unit_float_batch_is_untouched(self):
        unit = np.linspace(0.0, 1.0, 2 * 3 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 3)
        self.assertTrue(np.array_equal(self.batch(unit).numpy(), unit))

    def test_uint8_input_does_not_export_a_binary_image(self):
        photo = np.zeros((1, 8, 6, 3), dtype=np.uint8)
        photo[0, :, :, 0] = 17   # dark but non-zero
        photo[0, :, :, 1] = 0    # exactly zero -> the old magenta channel
        photo[0, :, :, 2] = 34
        batch = self.batch(photo)
        with tempfile.TemporaryDirectory() as temp_dir:
            GLOMAPLichtfeldTracker._export_dataset_images(
                None, Path(temp_dir), batch, None
            )
            output = Path(temp_dir) / "images" / "0001.png"
            written = np.asarray(Image.open(output).convert("RGB"))
        self.assertEqual(int(written[0, 0, 0]), 17)
        self.assertEqual(int(written[0, 0, 1]), 0)
        self.assertEqual(int(written[0, 0, 2]), 34)
        self.assertEqual(len(np.unique(written)), 3)

    def test_empty_saturated_and_boolean_batches_survive(self):
        empty = self.batch(np.zeros((0, 4, 4, 3), dtype=np.float32))
        self.assertEqual(tuple(empty.shape), (0, 4, 4, 3))
        self.assertEqual(float(self.batch(np.zeros((1, 4, 4, 3), np.float32)).max()), 0.0)
        self.assertEqual(float(self.batch(np.ones((1, 4, 4, 3), np.float32)).min()), 1.0)
        self.assertEqual(float(self.batch(np.ones((1, 4, 4, 3), bool)).min()), 1.0)

    def test_single_image_gets_a_batch_axis(self):
        self.assertEqual(tuple(self.batch(np.zeros((4, 4, 3), np.uint8)).shape),
                         (1, 4, 4, 3))

    def test_non_four_dimensional_input_is_rejected(self):
        self.assertIsNone(self.batch(np.zeros((4, 4), np.float32)))


if __name__ == "__main__":
    unittest.main()
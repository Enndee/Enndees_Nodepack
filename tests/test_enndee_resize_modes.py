"""Tests for the resize-mode math copied from ComfyUI's ResizeImageMaskNode."""

import sys
import unittest
from pathlib import Path

import torch


PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR.parents[1]))

import comfy.utils  # noqa: E402

from enndee_resize_modes import (  # noqa: E402
    RESIZE_TYPES,
    fit_within,
    match_multiple,
    pad_to_size,
    parse_color,
    resize_like,
    resolve_size,
    source_size,
)


class ResizeModeTests(unittest.TestCase):
    def test_resize_types_match_core_node_order(self):
        self.assertEqual(RESIZE_TYPES, (
            "scale dimensions",
            "scale by multiplier",
            "scale longer dimension",
            "scale shorter dimension",
            "scale width",
            "scale height",
            "scale total pixels",
            "match size",
            "scale to multiple",
        ))

    def test_scale_dimensions_derives_a_missing_side(self):
        self.assertEqual(resolve_size(200, 100, "scale dimensions", width=400), (400, 200))
        self.assertEqual(resolve_size(200, 100, "scale dimensions", height=400), (800, 400))
        self.assertEqual(resolve_size(200, 100, "scale dimensions"), (200, 100))
        self.assertEqual(resolve_size(200, 100, "scale dimensions", width=64, height=64), (64, 64))

    def test_scale_by_multiplier_rounds_source_size(self):
        self.assertEqual(resolve_size(200, 100, "scale by multiplier", multiplier=1.5), (300, 150))
        self.assertEqual(resolve_size(101, 51, "scale by multiplier", multiplier=0.5), (50, 26))

    def test_longer_and_shorter_dimension_keep_aspect(self):
        self.assertEqual(resolve_size(200, 100, "scale longer dimension", longer_size=512), (512, 256))
        self.assertEqual(resolve_size(100, 200, "scale longer dimension", longer_size=512), (256, 512))
        self.assertEqual(resolve_size(100, 100, "scale longer dimension", longer_size=64), (64, 64))
        self.assertEqual(resolve_size(200, 100, "scale shorter dimension", shorter_size=512), (1024, 512))
        self.assertEqual(resolve_size(100, 200, "scale shorter dimension", shorter_size=512), (512, 1024))
        self.assertEqual(resolve_size(200, 100, "scale longer dimension", longer_size=0), (200, 100))

    def test_width_and_height_modes_follow_the_source_aspect(self):
        self.assertEqual(resolve_size(200, 100, "scale width", width=300), (300, 150))
        self.assertEqual(resolve_size(100, 200, "scale height", height=300), (150, 300))
        self.assertEqual(resolve_size(200, 100, "scale width", width=0), (200, 100))

    def test_total_pixels_uses_the_1024_square_base(self):
        self.assertEqual(resolve_size(16, 8, "scale total pixels", megapixels=0.125), (512, 256))
        self.assertEqual(resolve_size(200, 100, "scale total pixels", megapixels=0.1), (458, 229))

    def test_match_size_uses_the_reference_or_falls_back(self):
        self.assertEqual(resolve_size(200, 100, "match size", match_size=(300, 150)), (300, 150))
        self.assertEqual(resolve_size(200, 100, "match size"), (200, 100))

    def test_scale_to_multiple_floors_and_keeps_tiny_inputs(self):
        self.assertEqual(resolve_size(100, 50, "scale to multiple", multiple=8), (96, 48))
        self.assertEqual(resolve_size(5, 3, "scale to multiple", multiple=8), (5, 3))

    def test_parse_color_accepts_hex_and_names(self):
        self.assertEqual(parse_color("#ff0000"), (1.0, 0.0, 0.0))
        self.assertEqual(parse_color("#f00"), (1.0, 0.0, 0.0))
        self.assertEqual(parse_color("white"), (1.0, 1.0, 1.0))
        self.assertEqual(parse_color(""), (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_color("not-a-color")

    def test_fit_within_keeps_the_aspect_and_pad_centers(self):
        self.assertEqual(fit_within(200, 100, 100, 100), (100, 50))
        image = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
        image[..., 1] = 0.5
        padded = pad_to_size(image, 4, 4, (1.0, 0.0, 0.0))
        self.assertEqual(tuple(padded.shape), (1, 4, 4, 3))
        self.assertEqual(tuple(padded[0, 0, 0].tolist()), (1.0, 0.0, 0.0))
        self.assertEqual(tuple(padded[0, 1, 0].tolist()), (0.0, 0.5, 0.0))
        self.assertEqual(tuple(padded[0, 3, 3].tolist()), (1.0, 0.0, 0.0))

    def test_resize_like_handles_images_masks_and_matching_sizes(self):
        image = torch.rand((1, 4, 8, 3))
        self.assertIs(resize_like(image, 8, 4, "area"), image)
        self.assertEqual(tuple(resize_like(image, 16, 8, "lanczos").shape), (1, 8, 16, 3))
        mask = torch.rand((1, 4, 8))
        self.assertEqual(tuple(resize_like(mask, 16, 8, "lanczos").shape), (1, 8, 16))
        self.assertEqual(source_size(mask), (8, 4))

    def test_match_multiple_covers_then_center_crops(self):
        image = torch.zeros((1, 10, 22, 3), dtype=torch.float32)
        result = match_multiple(image, 8, "area")
        self.assertEqual(tuple(result.shape), (1, 8, 16, 3))
        reference = comfy.utils.common_upscale(image.movedim(-1, 1), 18, 8, "area", "disabled").movedim(1, -1)
        self.assertTrue(torch.equal(result, reference[:, :, 1:17, :]))
        mask = torch.zeros((1, 10, 22))
        self.assertEqual(tuple(match_multiple(mask, 8, "area").shape), (1, 8, 16))

    def test_match_multiple_returns_input_when_already_divisible(self):
        image = torch.rand((1, 16, 32, 3))
        self.assertIs(match_multiple(image, 8), image)


if __name__ == "__main__":
    unittest.main()

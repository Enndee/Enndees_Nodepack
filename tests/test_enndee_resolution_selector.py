"""Tests for the Enndee Resolution Selector with core resize types."""

import sys
import unittest
from pathlib import Path

import torch


PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR.parents[1]))

import comfy.utils  # noqa: E402

from enndee_resize_modes import RESIZE_TYPES  # noqa: E402
from enndee_resolution_selector import (  # noqa: E402
    ASPECT_RATIO_OPTIONS,
    ResolutionSelectorEnndee,
    dimensions_from_ratio,
    dimensions_from_source,
)


class ResolutionSelectorEnndeeTests(unittest.TestCase):
    def gradient(self, width, height):
        image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        image[..., 0] = torch.linspace(0.0, 1.0, width).view(1, 1, width)
        return image

    def select(self, **overrides):
        node = ResolutionSelectorEnndee()
        kwargs = dict(
            aspect_ratio=ASPECT_RATIO_OPTIONS[0],
            megapixels=4.0,
            multiple=32,
            keep_source_aspect_ratio=False,
            resize_type="scale dimensions",
            multiplier=1.0,
            longer_size=512,
            shorter_size=512,
            crop="center",
            scale_method="area",
            image=None,
            match=None,
        )
        kwargs.update(overrides)
        return node.select(**kwargs)

    def test_return_types_and_legacy_widget_order(self):
        self.assertEqual(ResolutionSelectorEnndee.RETURN_TYPES, ("INT", "INT", "IMAGE"))
        self.assertEqual(ResolutionSelectorEnndee.RETURN_NAMES, ("width", "height", "image"))
        required = ResolutionSelectorEnndee.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required)[:4],
            ["aspect_ratio", "megapixels", "multiple", "keep_source_aspect_ratio"],
        )
        self.assertEqual(required["resize_type"][1]["default"], "scale dimensions")
        self.assertEqual(list(required["resize_type"][0]), list(RESIZE_TYPES))
        self.assertTrue(all("tooltip" in metadata for _options, metadata in required.values()))

    def test_selected_dimensions_without_image(self):
        width, height, image = self.select()
        self.assertEqual((width, height), (2048, 2048))
        self.assertIsNone(image)
        expected = dimensions_from_ratio(16, 9, 2.0, 32)
        width, height, image = self.select(aspect_ratio="16:9 (Widescreen)", megapixels=2.0)
        self.assertEqual((width, height), expected)
        self.assertIsNone(image)

    def test_keep_source_aspect_ratio_and_scale_dimensions_resize(self):
        source = self.gradient(1000, 500)
        expected = dimensions_from_source(1000, 500, 0.1, 8)
        width, height, image = self.select(
            image=source, keep_source_aspect_ratio=True, megapixels=0.1, multiple=8)
        self.assertEqual((width, height), expected)
        self.assertEqual(tuple(image.shape), (1, expected[1], expected[0], 3))

    def test_scale_dimensions_crops_to_the_selected_aspect(self):
        source = self.gradient(200, 100)
        width, height, image = self.select(image=source, megapixels=0.1, multiple=8)
        self.assertEqual((width, height), (320, 320))
        self.assertEqual(tuple(image.shape), (1, 320, 320, 3))
        self.assertAlmostEqual(float(image[0, 160, 0, 0]), 0.2513, delta=0.02)
        _width, _height, stretched = self.select(
            image=source, megapixels=0.1, multiple=8, crop="disabled")
        self.assertLess(float(stretched[0, 160, 0, 0]), 0.02)

    def test_scale_width_and_height_follow_the_selected_dimensions(self):
        source = self.gradient(200, 100)
        width, height, _image = self.select(
            image=source, megapixels=0.1, multiple=8, resize_type="scale width")
        self.assertEqual((width, height), (320, 160))
        width, height, _image = self.select(
            image=source, megapixels=0.1, multiple=8, resize_type="scale height")
        self.assertEqual((width, height), (640, 320))

    def test_scale_by_multiplier_mode(self):
        source = self.gradient(200, 100)
        width, height, image = self.select(
            image=source, resize_type="scale by multiplier", multiplier=1.5)
        self.assertEqual((width, height), (300, 150))
        self.assertEqual(tuple(image.shape), (1, 150, 300, 3))

    def test_longer_and_shorter_dimension_modes(self):
        source = self.gradient(200, 100)
        width, height, _image = self.select(
            image=source, resize_type="scale longer dimension", longer_size=512)
        self.assertEqual((width, height), (512, 256))
        width, height, _image = self.select(
            image=source, resize_type="scale shorter dimension", shorter_size=512)
        self.assertEqual((width, height), (1024, 512))

    def test_total_pixels_and_match_size_modes(self):
        source = self.gradient(16, 8)
        width, height, _image = self.select(
            image=source, resize_type="scale total pixels", megapixels=0.125)
        self.assertEqual((width, height), (512, 256))

        match = torch.zeros((1, 30, 40, 3), dtype=torch.float32)
        width, height, image = self.select(image=source, resize_type="match size", match=match)
        self.assertEqual((width, height), (40, 30))
        self.assertEqual(tuple(image.shape), (1, 30, 40, 3))

        width, height, _image = self.select(
            image=source, resize_type="match size", megapixels=0.1, multiple=8)
        self.assertEqual((width, height), (320, 320))

    def test_scale_to_multiple_matches_core_cover_crop(self):
        source = self.gradient(22, 10)
        width, height, image = self.select(
            image=source, resize_type="scale to multiple", multiple=8)
        self.assertEqual((width, height), (16, 8))
        reference = comfy.utils.common_upscale(source.movedim(-1, 1), 18, 8, "area", "disabled")
        reference = reference.movedim(1, -1)[:, :, 1:17, :]
        self.assertTrue(torch.equal(image, reference))


if __name__ == "__main__":
    unittest.main()

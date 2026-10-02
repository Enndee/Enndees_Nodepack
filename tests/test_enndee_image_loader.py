"""Tests for the Load & Resize Image (Enndee) node."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR.parents[1]))

import comfy.utils  # noqa: E402

import enndee_image_loader as loader  # noqa: E402
import enndee_resize_modes as resize_modes  # noqa: E402


class ImageLoaderResizeEnndeeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_dir = Path(self.temp_dir.name)
        patcher = mock.patch.object(
            loader.folder_paths, "get_input_directory", return_value=str(self.input_dir))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_png(self, name, array, mode=None):
        path = self.input_dir / name
        Image.fromarray(array, mode=mode).save(path)
        return name

    def gradient(self, name, width, height, channels=3):
        array = np.zeros((height, width, channels), dtype=np.uint8)
        array[..., 0] = np.linspace(0, 255, width).astype(np.uint8)[None, :]
        return self.write_png(name, array)

    def load(self, **overrides):
        node = loader.ImageLoaderResizeEnndee()
        kwargs = dict(
            resize=False, resize_type="scale dimensions", width=512, height=512,
            repeat=1, keep_proportion=True, divisible_by=2, mask_channel="alpha",
            background_color="#000000", multiplier=1.0, longer_size=512,
            shorter_size=512, megapixels=1.0, multiple=8, scale_method="lanczos",
            match=None,
        )
        kwargs.update(overrides)
        return node.load(**kwargs)

    def test_input_types_list_input_folder_images_with_upload(self):
        self.gradient("b.png", 4, 2)
        self.gradient("a.png", 4, 2)
        (self.input_dir / "note.txt").write_text("not an image", encoding="utf-8")
        options, metadata = loader.ImageLoaderResizeEnndee.INPUT_TYPES()["required"]["image"]
        self.assertEqual(options, ["a.png", "b.png"])
        self.assertTrue(metadata["image_upload"])

    def test_input_types_expose_types_and_tooltips(self):
        required = loader.ImageLoaderResizeEnndee.INPUT_TYPES()["required"]
        self.assertEqual(list(required["resize_type"][0]), [
            "scale dimensions", "scale by multiplier", "scale longer dimension",
            "scale shorter dimension", "scale width", "scale height",
            "scale total pixels", "match size", "scale to multiple"])
        self.assertEqual(required["resize"][1]["default"], False)
        self.assertEqual(required["keep_proportion"][1]["default"], True)
        self.assertEqual(required["no_upscale"][1]["default"], False)
        self.assertEqual(required["mask_channel"][0], ["alpha", "red", "green", "blue"])
        self.assertTrue(all("tooltip" in metadata for _options, metadata in required.values()))
        self.assertEqual(
            loader.ImageLoaderResizeEnndee.RETURN_NAMES,
            ("image", "original_image", "mask", "width", "height", "image_path",
             "megapixels"),
        )

    def test_resize_off_keeps_original_and_reports_source_size(self):
        name = self.gradient("photo.png", 8, 4)
        image, original, mask, width, height, image_path, _mp = self.load(image=name)
        self.assertTrue(torch.equal(image, original))
        self.assertEqual(tuple(image.shape), (1, 4, 8, 3))
        self.assertEqual((width, height), (8, 4))
        self.assertEqual(Path(image_path), self.input_dir / name)
        self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))

    def test_scale_dimensions_stretches_without_keep_proportion(self):
        name = self.gradient("photo.png", 8, 4)
        image, _original, mask, width, height, _path, _mp = self.load(
            image=name, resize=True, width=16, height=16, keep_proportion=False)
        self.assertEqual((width, height), (16, 16))
        self.assertEqual(tuple(image.shape), (1, 16, 16, 3))
        self.assertEqual(tuple(mask.shape), (1, 16, 16))
        self.assertGreater(float(image[0, 0, -1, 0]), 0.9)

    def test_keep_proportion_pads_with_background_color(self):
        name = self.gradient("photo.png", 16, 8)
        image, _original, mask, width, height, _path, _mp = self.load(
            image=name, resize=True, width=16, height=16, keep_proportion=True,
            background_color="#ff0000")
        self.assertEqual((width, height), (16, 16))
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(image[0, 0, 0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(image[0, 4, 0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(mask[0, 0, 0]), 0.0, places=5)

    def test_divisible_by_snaps_the_target_size(self):
        name = self.gradient("photo.png", 16, 8)
        image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, width=20, height=10, keep_proportion=False,
            divisible_by=8)
        self.assertEqual((width, height), (16, 8))
        self.assertEqual(tuple(image.shape), (1, 8, 16, 3))


    def test_no_upscale_keeps_the_source_size(self):
        name = self.gradient("photo.png", 16, 8)
        source, _original, _mask, _w, _h, _path, _mp = self.load(image=name)

        image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, width=64, height=64, keep_proportion=False,
            no_upscale=True)
        self.assertEqual((width, height), (16, 8))
        self.assertTrue(torch.equal(image, source))

        image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale by multiplier", multiplier=2.0,
            no_upscale=True)
        self.assertEqual((width, height), (16, 8))
        self.assertTrue(torch.equal(image, source))

        # keep_proportion still fills the requested canvas with the background.
        image, _original, mask, width, height, _path, _mp = self.load(
            image=name, resize=True, width=64, height=64, keep_proportion=True,
            background_color="#ff0000", no_upscale=True)
        self.assertEqual((width, height), (64, 64))
        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 1.0, places=5)
        self.assertTrue(torch.equal(image[0, 28:36, 24:40, :], source[0]))
        self.assertEqual(tuple(mask.shape), (1, 64, 64))

        # Downscaling still works with no_upscale enabled.
        image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, width=8, height=4, keep_proportion=False,
            no_upscale=True)
        self.assertEqual((width, height), (8, 4))
        self.assertEqual(tuple(image.shape), (1, 4, 8, 3))

    def test_multiplier_longer_and_shorter_modes(self):
        name = self.gradient("photo.png", 16, 8)
        image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale by multiplier", multiplier=1.5,
            divisible_by=1)
        self.assertEqual((width, height), (24, 12))
        _image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale longer dimension", longer_size=512)
        self.assertEqual((width, height), (512, 256))
        _image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale shorter dimension", shorter_size=512)
        self.assertEqual((width, height), (1024, 512))

    def test_width_and_height_modes_follow_the_source_aspect(self):
        name = self.gradient("photo.png", 16, 8)
        _image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale width", width=100, divisible_by=1)
        self.assertEqual((width, height), (100, 50))
        _image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale height", height=100, divisible_by=1)
        self.assertEqual((width, height), (200, 100))

    def test_total_pixels_match_size_and_multiple_modes(self):
        name = self.gradient("photo.png", 16, 8)
        _image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="scale total pixels", megapixels=0.125,
            divisible_by=1)
        self.assertEqual((width, height), (512, 256))

        match = torch.zeros((1, 30, 40, 3), dtype=torch.float32)
        image, _original, mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="match size", match=match)
        self.assertEqual((width, height), (40, 30))
        self.assertEqual(tuple(image.shape), (1, 30, 40, 3))
        self.assertEqual(tuple(mask.shape), (1, 30, 40))

        _image, _original, _mask, width, height, _path, _mp = self.load(
            image=name, resize=True, resize_type="match size")
        self.assertEqual((width, height), (512, 512))

        wide = self.gradient("wide.png", 22, 10)
        image, _original, mask, width, height, _path, _mp = self.load(
            image=wide, resize=True, resize_type="scale to multiple", multiple=8,
            scale_method="area")
        self.assertEqual((width, height), (16, 8))
        self.assertEqual(tuple(mask.shape), (1, 8, 16))
        source = torch.from_numpy(
            np.asarray(Image.open(self.input_dir / wide), dtype=np.float32) / 255.0)[None]
        reference = comfy.utils.common_upscale(source.movedim(-1, 1), 18, 8, "area", "disabled")
        reference = reference.movedim(1, -1)[:, :, 1:17, :]
        self.assertTrue(torch.equal(image, reference))

    def test_mask_channels_alpha_and_color(self):
        alpha_values = np.linspace(0, 255, 8).astype(np.uint8)
        rgba = np.zeros((4, 8, 4), dtype=np.uint8)
        rgba[..., 1] = 128
        rgba[..., 3] = alpha_values[None, :]
        name = self.write_png("ramp.png", rgba, mode="RGBA")
        _image, _original, mask, _w, _h, _path, _mp = self.load(image=name, mask_channel="alpha")
        expected_alpha = torch.from_numpy(alpha_values / 255.0).to(torch.float32)[None, None, :].expand(1, 4, 8)
        self.assertTrue(torch.allclose(mask, 1.0 - expected_alpha, atol=1e-6))
        _image, _original, mask, _w, _h, _path, _mp = self.load(image=name, mask_channel="green")
        self.assertTrue(torch.allclose(mask, torch.full((1, 4, 8), 128 / 255.0), atol=1e-6))

        plain = self.gradient("plain.png", 8, 4)
        _image, _original, mask, _w, _h, _path, _mp = self.load(image=plain, mask_channel="alpha")
        self.assertTrue(torch.equal(mask, torch.zeros((1, 4, 8))))

    def test_repeat_expands_the_batch(self):
        name = self.gradient("photo.png", 8, 4)
        image, original, mask, _w, _h, _path, _mp = self.load(image=name, repeat=3)
        self.assertEqual(tuple(image.shape), (3, 4, 8, 3))
        self.assertEqual(tuple(original.shape), (3, 4, 8, 3))
        self.assertEqual(tuple(mask.shape), (3, 4, 8))
        self.assertTrue(torch.equal(image[0], image[2]))

    def test_animated_webp_loads_every_frame(self):
        frames = []
        for color in ((255, 0, 0), (0, 0, 255), (0, 255, 0)):
            array = np.zeros((4, 8, 3), dtype=np.uint8)
            array[..., 0], array[..., 1], array[..., 2] = color
            frames.append(Image.fromarray(array))
        path = self.input_dir / "anim.webp"
        try:
            frames[0].save(path, format="WEBP", save_all=True, append_images=frames[1:],
                           lossless=True)
        except (OSError, ValueError) as error:
            self.skipTest(f"WebP encoding unavailable: {error}")
        image, original, _mask, _w, _h, _path, _mp = self.load(image="anim.webp")
        self.assertEqual(tuple(image.shape), (3, 4, 8, 3))
        self.assertEqual(tuple(original.shape), (3, 4, 8, 3))
        self.assertGreater(float(image[0][0, 0, 0]), 0.9)
        self.assertGreater(float(image[1][0, 0, 2]), 0.9)

    def test_is_changed_and_validate_inputs_track_the_file(self):
        name = self.gradient("photo.png", 8, 4)
        self.assertTrue(loader.ImageLoaderResizeEnndee.VALIDATE_INPUTS(image=name))
        self.assertIn(
            "Invalid image file",
            loader.ImageLoaderResizeEnndee.VALIDATE_INPUTS(image="missing.png"),
        )
        first = loader.ImageLoaderResizeEnndee.IS_CHANGED(image=name)
        self.assertEqual(first, loader.ImageLoaderResizeEnndee.IS_CHANGED(image=name))
        self.gradient("photo.png", 8, 4)
        self.assertEqual(first, loader.ImageLoaderResizeEnndee.IS_CHANGED(image=name))
        self.gradient("photo.png", 9, 4)
        self.assertNotEqual(first, loader.ImageLoaderResizeEnndee.IS_CHANGED(image=name))

    def test_megapixels_is_a_float_slider_parameter(self):
        node_types = loader.ImageLoaderResizeEnndee.INPUT_TYPES()
        spec_type, metadata = node_types["required"]["megapixels"]
        # A slider like every size parameter - and its slot must accept FLOAT
        # links (a primitive float as source) instead of ghosting as Boolean.
        self.assertEqual(spec_type, "FLOAT")
        self.assertNotIn("forceInput", metadata)
        self.assertEqual(metadata["default"], 1.0)
        self.assertEqual(metadata["min"], 0.01)
        self.assertEqual(metadata["max"], 16.0)
        self.assertIn("tooltip", metadata)
        self.assertNotIn("megapixels_in", node_types["optional"])
        self.assertNotIn("megapixels", node_types["optional"])

    def test_megapixels_parameter_value_sizes_the_image(self):
        name = self.gradient("photo.png", 8, 4)
        linked = self.load(
            image=name, resize=True, resize_type="scale total pixels",
            divisible_by=1, megapixels=0.5)
        self.assertEqual((linked[3], linked[4]), (1024, 512))

    def test_default_megapixels_when_parameter_unspecified(self):
        name = self.gradient("photo.png", 8, 4)
        node = loader.ImageLoaderResizeEnndee()
        kwargs = dict(
            image=name, resize=True, resize_type="scale total pixels",
            width=512, height=512, repeat=1, keep_proportion=True,
            divisible_by=1, mask_channel="alpha", background_color="#000000",
            multiplier=1.0, longer_size=512, shorter_size=512,
            multiple=8, scale_method="lanczos",
        )
        _image, _original, _mask, width, height, _path, _mp = node.load(**kwargs)
        self.assertEqual((width, height), (1448, 724))  # spec default 1.0 MP

    def test_legacy_megapixels_in_kwarg_is_still_honoured(self):
        name = self.gradient("photo.png", 8, 4)
        legacy = self.load(
            image=name, resize=True, resize_type="scale total pixels",
            divisible_by=1, megapixels=1.0, megapixels_in=0.5)
        self.assertEqual((legacy[3], legacy[4]), (1024, 512))

    def test_megapixels_value_is_clamped_to_the_documented_bounds(self):
        name = self.gradient("photo.png", 8, 4)
        below = self.load(
            image=name, resize=True, resize_type="scale total pixels",
            divisible_by=1, megapixels=0.0)
        self.assertEqual((below[3], below[4]), (145, 72))
        above = self.load(
            image=name, resize=True, resize_type="scale total pixels",
            divisible_by=1, megapixels=99.0)
        self.assertEqual((above[3], above[4]), (5793, 2896))

    def test_megapixels_output_echoes_the_effective_value(self):
        # The FLOAT output carries the resolved parameter so other nodes can
        # consume it: socket value, legacy alias and the clamp all echo.
        name = self.gradient("photo.png", 8, 4)
        self.assertEqual(self.load(image=name, megapixels=0.5)[6], 0.5)
        self.assertEqual(
            self.load(image=name, megapixels=1.0, megapixels_in=0.75)[6], 0.75)
        self.assertEqual(self.load(image=name, megapixels=99.0)[6], 16.0)
        node = loader.ImageLoaderResizeEnndee()
        kwargs = dict(
            image=name, resize=False, resize_type="scale dimensions",
            width=512, height=512, repeat=1, keep_proportion=True,
            divisible_by=2, mask_channel="alpha", background_color="#000000",
            multiplier=1.0, longer_size=512, shorter_size=512,
            multiple=8, scale_method="lanczos",
        )
        self.assertEqual(node.load(**kwargs)[6], 1.0)  # unconnected default

    def test_frontend_cleans_up_legacy_megapixels_inputs(self):
        script = (PACK_DIR / "web" / "js" / "enndee_image_loader.js").read_text(
            encoding="utf-8")
        # orphan megapixels_in sockets are removed and live links rewired ...
        self.assertIn('findSlot("megapixels_in")', script)
        self.assertIn("node.removeInput(legacySlot)", script)
        self.assertIn("origin.connect(link.origin_slot, node, target)", script)
        # ... saved widget values are repaired by name (the positional restore
        # shifts them and once fed a boolean into megapixels - the "Boolean"
        # ghost that poisoned the slot type) ...
        self.assertIn("repairWidgetValuesByName", script)
        self.assertIn("LEGACY_WIDGET_ORDER", script)
        # ... and the slot type is forced to the declared FLOAT format.
        self.assertIn('megapixels.type = "FLOAT"', script)
        self.assertNotIn("PrimitiveNode", script)

    def test_frontend_visibility_rules_match_the_backend_parameters(self):
        script = (PACK_DIR / "web" / "js" / "enndee_image_loader.js").read_text(encoding="utf-8")
        for resize_type in resize_modes.RESIZE_TYPES:
            self.assertIn(f'"{resize_type}"', script)
        expected_rules = (
            '"scale dimensions": ["width", "height", "keep_proportion", "divisible_by"],',
            '"scale by multiplier": ["multiplier", "divisible_by"],',
            '"scale longer dimension": ["longer_size", "divisible_by"],',
            '"scale shorter dimension": ["shorter_size", "divisible_by"],',
            '"scale width": ["width", "divisible_by"],',
            '"scale height": ["height", "divisible_by"],',
            '"scale total pixels": ["megapixels", "divisible_by"],',
            '"match size": ["keep_proportion", "divisible_by"],',
            '"scale to multiple": ["multiple"],',
        )
        for rule in expected_rules:
            self.assertIn(rule, script)
        self.assertIn(
            'const ALWAYS_VISIBLE_WHILE_RESIZING = ["resize_type", "scale_method", "no_upscale"];',
            script,
        )
        required = set(loader.ImageLoaderResizeEnndee.INPUT_TYPES()["required"])
        self.assertTrue({
            "resize_type", "width", "height", "multiplier", "longer_size", "shorter_size",
            "megapixels", "multiple", "keep_proportion", "divisible_by",
            "background_color", "scale_method", "no_upscale",
        } <= required)
        # Regression (custom-camera widgets never reappeared): the collapse
        # helper must track its own overrides with an explicit flag and restore
        # the originals exactly - standard widgets own no computeSize/draw, so a
        # restore test based on the saved value would never run.
        self.assertIn("widget.__enndeeCollapsed = true;", script)
        self.assertIn("delete widget.computeSize;", script)
        self.assertIn("delete widget.draw;", script)


if __name__ == "__main__":
    unittest.main()


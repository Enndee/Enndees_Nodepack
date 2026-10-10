"""Tests for the VGGT for Lichtfeld node (feed-forward dataset export).

Two layers:

* **Pure geometry / IO** - intrinsics scaling, quaternion round trip, depth
  normalisation, 16-bit depth PNGs, depth unprojection and the COLMAP TXT model
  writer.  These run everywhere, no GPU and no checkpoint required.
* **Node end to end** - the full ``track()`` pipeline against a *stub backend*
  that returns a known synthetic reconstruction, so the exported dataset can be
  verified on disk and parsed back with the vendored ``COLMAPParser``.
* **Real model smoke test** - skipped unless a checkpoint is present *and* CUDA
  is available.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR.parents[1]))          # the ComfyUI root (comfy.*)

import enndee_feedforward as ff  # noqa: E402
from enndee_colmap.colmap_parser import COLMAPParser  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reconstruction(count=4, model_hw=(24, 32), depth_value=2.0):
    """A synthetic, perfectly known reconstruction (identity world-to-camera)."""
    height, width = model_hw
    intrinsics = np.tile(
        np.array([[16.0, 0.0, width / 2.0],
                  [0.0, 16.0, height / 2.0],
                  [0.0, 0.0, 1.0]]), (count, 1, 1))
    extrinsics = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (count, 1, 1))
    depth = np.full((count, height, width), depth_value, np.float32)
    depth += np.linspace(0.0, 1.0, width, dtype=np.float32)[None, None, :]
    confidence = np.ones((count, height, width), np.float32)
    return ff.Reconstruction(extrinsics, intrinsics, depth, confidence,
                             model_hw, "stub:test")


def make_images(count=4, height=48, width=64, channels=3):
    """A deterministic image batch in [0, 1] with the ComfyUI IMAGE layout.

    ``np.tile`` promotes the input to the number of reps by *prepending* axes
    and then multiplies axis by axis, so the gradient needs an explicit
    trailing (channel) axis to land on the intended shape.
    """
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, None, :, None]
    batch = np.tile(gradient, (count, height, 1, channels))
    assert batch.shape == (count, height, width, channels), batch.shape
    return batch.astype(np.float32)


def make_mask(count=3, height=48, width=64, keep_columns=32):
    """A splat mask ``[N, 1, H, W]`` with WHITE (1) = keep on the left columns."""
    mask = np.zeros((count, 1, height, width), dtype=np.float32)
    mask[:, :, :, :keep_columns] = 1.0
    return torch.from_numpy(mask)


class StubBackend:
    """Backend stand-in: returns :func:`make_reconstruction` and counts calls."""

    def __init__(self, reconstruction=None):
        self.reconstruction = reconstruction or make_reconstruction()
        self.calls = 0
        self.released = 0

    def status(self):
        return True, "stub backend"

    def run(self, images, image_resolution=512, device="cuda"):
        self.calls += 1
        return self.reconstruction

    def release(self):
        self.released += 1


class GeometryTests(unittest.TestCase):
    def test_balanced_target_shape_is_patch_aligned_and_keeps_aspect(self):
        for height, width in ((48, 64), (1080, 1920), (1920, 1080), (512, 512)):
            target_h, target_w = ff.balanced_target_shape(height, width, 512)
            self.assertEqual(target_h % 16, 0, f"{height}x{width}")
            self.assertEqual(target_w % 16, 0, f"{height}x{width}")
            tokens = (target_h // 16) * (target_w // 16)
            self.assertLessEqual(abs(tokens - (512 // 16) ** 2), 64)
            self.assertAlmostEqual(target_h / target_w, height / width, delta=0.08)

    def test_preprocess_images_shape_and_range(self):
        tensor, (height, width) = ff.preprocess_images(make_images(3, 48, 64), 512)
        self.assertEqual(tuple(tensor.shape), (1, 3, 3, height, width))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_preprocess_images_composites_alpha_on_white(self):
        rgba = np.zeros((1, 16, 16, 4), np.float32)   # fully transparent -> white
        tensor, _ = ff.preprocess_images(rgba, 256)
        self.assertTrue(torch.allclose(tensor, torch.ones_like(tensor), atol=1e-5))

    def test_rotmat_to_qvec_round_trips_through_the_colmap_parser(self):
        rng = np.random.default_rng(7)
        parser = COLMAPParser(".")
        for _ in range(50):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1.0
            qvec = ff.rotmat_to_qvec(rotation)
            self.assertAlmostEqual(float(np.linalg.norm(qvec)), 1.0, places=6)
            self.assertTrue(np.allclose(parser.qvec_to_rotmat(qvec), rotation, atol=1e-8))

    def test_scale_intrinsics_matches_the_image_resolution(self):
        intrinsics = np.array([[[16.0, 0.0, 16.0], [0.0, 16.0, 12.0], [0.0, 0.0, 1.0]]])
        scaled = ff.scale_intrinsics(intrinsics, (24, 32), (48, 64))
        self.assertAlmostEqual(scaled[0, 0, 0], 32.0)
        self.assertAlmostEqual(scaled[0, 1, 1], 32.0)
        self.assertAlmostEqual(scaled[0, 0, 2], 32.0)
        self.assertAlmostEqual(scaled[0, 1, 2], 24.0)

    def test_unproject_depth_puts_the_principal_point_on_the_optical_axis(self):
        reconstruction = make_reconstruction(1)
        points = ff.unproject_depth(reconstruction.depth[:1],
                                    reconstruction.intrinsics[:1],
                                    reconstruction.extrinsics[:1])
        row, column = 12, 16
        self.assertAlmostEqual(points[0, row, column, 0], 0.0, places=5)
        self.assertAlmostEqual(points[0, row, column, 1], 0.0, places=5)
        self.assertAlmostEqual(points[0, row, column, 2],
                               float(reconstruction.depth[0, row, column]), places=5)
        # +x to the right, +y down (OpenCV), +z forward
        self.assertGreater(points[0, row, column + 4, 0], 0.0)
        self.assertGreater(points[0, row + 4, column, 1], 0.0)


class DepthMapTests(unittest.TestCase):
    def test_valid_mask_gates_on_confidence_percentile(self):
        depth = np.full((1, 10, 10), 1.0, np.float32)
        confidence = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(1, 10, 10)
        mask = ff.depth_valid_mask(depth, confidence, conf_percentile=50.0)
        self.assertAlmostEqual(float(mask.mean()), 0.5, delta=0.02)

    def test_valid_mask_rejects_zero_and_non_finite_depth(self):
        depth = np.ones((1, 4, 4), np.float32)
        depth[0, 0, 0] = 0.0
        depth[0, 1, 1] = np.nan
        mask = ff.depth_valid_mask(depth, np.ones_like(depth), conf_percentile=0.0)
        self.assertFalse(bool(mask[0, 0, 0]))
        self.assertFalse(bool(mask[0, 1, 1]))

    def test_upsample_depth_matches_the_image_resolution_and_keeps_invalid_zero(self):
        reconstruction = make_reconstruction(2)
        valid = ff.depth_valid_mask(reconstruction.depth, reconstruction.depth_conf, 0.0)
        valid[0, :, :4] = False
        full = ff.upsample_depth_maps(reconstruction.depth, valid, (48, 64))
        self.assertEqual(full.shape, (2, 48, 64))
        self.assertTrue(np.all(full[0, :, :4] == 0.0))
        self.assertTrue(np.all(full[1] > 0.0))

    def test_depth_to_uint16_is_monotonic_and_zero_marks_invalid(self):
        depth = np.linspace(1.0, 10.0, 100, dtype=np.float32).reshape(1, 10, 10)
        valid = np.ones_like(depth, bool)
        valid[0, 0, 0] = False
        encoded = ff.depth_to_uint16(depth, valid)
        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual(int(encoded[0, 0, 0]), 0)
        self.assertGreater(int(encoded[0, 9, 9]), int(encoded[0, 1, 1]))
        self.assertGreater(int(encoded.max()), 60000)      # the full range is used

    def test_depth_to_uint16_survives_a_flat_depth_map(self):
        encoded = ff.depth_to_uint16(np.full((1, 4, 4), 3.0, np.float32))
        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual(int(encoded.min()), int(encoded.max()))

    def test_write_depth_maps_creates_16_bit_pngs_at_the_image_resolution(self):
        from PIL import Image

        encoded = ff.depth_to_uint16(
            np.linspace(1.0, 9.0, 4 * 48 * 64, dtype=np.float32).reshape(4, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            written = ff.write_depth_maps(folder, ["0001", "0002", "0003", "0004"], encoded)
            self.assertEqual(written, 4)
            path = Path(folder) / "0002.depth.png"
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertEqual(image.size, (64, 48))
                self.assertEqual(image.mode, "I;16")

    def test_write_depth_maps_removes_stale_maps(self):
        encoded = ff.depth_to_uint16(np.ones((1, 4, 4), np.float32))
        with tempfile.TemporaryDirectory() as folder:
            stale = Path(folder) / "9999.depth.png"
            stale.write_bytes(b"stale")
            ff.write_depth_maps(folder, ["0001"], encoded)
            self.assertFalse(stale.exists())


# ---------------------------------------------------------------------------
# COLMAP TXT reader (the vendored parser only reads the .bin files)
# ---------------------------------------------------------------------------

def read_cameras_txt(path):
    cameras = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cameras[int(parts[0])] = {
            "model_name": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": np.array([float(value) for value in parts[4:]]),
        }
    return cameras


def read_images_txt(path):
    images = {}
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line:
            parts = line.split()
            images[int(parts[0])] = {
                "qvec": np.array([float(value) for value in parts[1:5]]),
                "tvec": np.array([float(value) for value in parts[5:8]]),
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
            index += 2          # skip the (empty) points2D line
        else:
            index += 1
    return images


def read_points_txt(path):
    points = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        points[int(parts[0])] = {
            "xyz": np.array([float(value) for value in parts[1:4]]),
            "rgb": np.array([int(value) for value in parts[4:7]]),
        }
    return points


class WriterTests(unittest.TestCase):
    def test_write_colmap_model_writes_a_lichtfeld_ready_txt_model(self):
        from PIL import Image

        reconstruction = make_reconstruction(3)
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder)
            images_dir = export / "images"
            images_dir.mkdir()
            names = [f"{index + 1:04d}.png" for index in range(3)]
            for name in names:
                Image.new("RGB", (64, 48)).save(images_dir / name)

            intrinsics = ff.scale_intrinsics(reconstruction.intrinsics, (24, 32), (48, 64))
            points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
            colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
            target = ff.write_colmap_model(export, reconstruction.extrinsics, intrinsics,
                                           names, (48, 64), points=points, colors=colors)

            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                self.assertTrue((target / name).is_file(), name)

            cameras = read_cameras_txt(target / "cameras.txt")
            self.assertEqual(len(cameras), 1)
            camera = cameras[1]
            self.assertEqual(camera["model_name"], "PINHOLE")
            self.assertEqual((camera["width"], camera["height"]), (64, 48))
            np.testing.assert_allclose(camera["params"], [32.0, 32.0, 32.0, 24.0], atol=1e-4)

            images = read_images_txt(target / "images.txt")
            self.assertEqual(len(images), 3)
            self.assertEqual(images[1]["name"], "0001.png")
            self.assertEqual(images[3]["name"], "0003.png")
            self.assertTrue(np.allclose(images[1]["qvec"], [1.0, 0.0, 0.0, 0.0], atol=1e-6))
            self.assertTrue(np.allclose(images[1]["tvec"], 0.0, atol=1e-6))
            self.assertTrue(all(entry["camera_id"] == 1 for entry in images.values()))

            points_out = read_points_txt(target / "points3D.txt")
            self.assertEqual(len(points_out), 2)
            np.testing.assert_allclose(points_out[1]["xyz"], [1.0, 2.0, 3.0], atol=1e-5)
            np.testing.assert_array_equal(points_out[1]["rgb"], [255, 0, 0])

    def test_written_cameras_are_readable_by_the_vendored_parser_accessors(self):
        reconstruction = make_reconstruction(2)
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder)
            intrinsics = ff.scale_intrinsics(reconstruction.intrinsics, (24, 32), (48, 64))
            target = ff.write_colmap_model(export, reconstruction.extrinsics, intrinsics,
                                           ["0001.png", "0002.png"], (48, 64))

            parser = COLMAPParser(str(target))
            parser.cameras = read_cameras_txt(target / "cameras.txt")
            for camera in parser.cameras.values():
                camera["model_id"] = 1
            # the COLMAP node reports [fx, fy, cx, cy] through exactly this helper
            np.testing.assert_allclose(parser.get_intrinsics(), [32.0, 32.0, 32.0, 24.0],
                                       atol=1e-4)

    def test_per_view_intrinsics_writes_one_camera_per_view(self):
        reconstruction = make_reconstruction(3)
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder)
            intrinsics = ff.scale_intrinsics(reconstruction.intrinsics, (24, 32), (48, 64))
            target = ff.write_colmap_model(export, reconstruction.extrinsics, intrinsics,
                                           [f"{index + 1:04d}.png" for index in range(3)],
                                           (48, 64), per_view_intrinsics=True)
            self.assertEqual(len(read_cameras_txt(target / "cameras.txt")), 3)
            images = read_images_txt(target / "images.txt")
            self.assertEqual(sorted(entry["camera_id"] for entry in images.values()), [1, 2, 3])

    def test_sample_point_cloud_respects_the_cap_and_keeps_colours_normalised(self):
        points = np.zeros((2, 8, 8, 3), np.float32)
        colors = np.full((2, 8, 8, 3), 0.5, np.float32)
        valid = np.ones((2, 8, 8), bool)
        sampled_points, sampled_colors = ff.sample_point_cloud(points, colors, valid,
                                                              max_points=50)
        self.assertEqual(sampled_points.shape, (50, 3))
        self.assertEqual(sampled_colors.shape, (50, 3))
        self.assertLessEqual(float(sampled_colors.max()), 1.0)

    def test_sample_point_cloud_without_valid_pixels_is_empty(self):
        empty_points, empty_colors = ff.sample_point_cloud(
            np.zeros((1, 4, 4, 3), np.float32), np.zeros((1, 4, 4, 3), np.float32),
            np.zeros((1, 4, 4), bool), max_points=10)
        self.assertEqual(len(empty_points), 0)
        self.assertEqual(len(empty_colors), 0)


class BackendRegistryTests(unittest.TestCase):
    def test_backend_names_are_the_three_expected_models(self):
        self.assertEqual(list(ff.BACKEND_NAMES),
                         ["VGGT-Omega", "DA3-AnyView", "VGG-T3"])

    def test_make_backend_selects_by_name_case_insensitively(self):
        self.assertIsInstance(ff.make_backend("VGGT-Omega"), ff.VGGTOmegaBackend)
        self.assertIsInstance(ff.make_backend("vggt-omega"), ff.VGGTOmegaBackend)
        self.assertIsInstance(ff.make_backend("DA3-AnyView"), ff.DA3AnyViewBackend)
        self.assertIsInstance(ff.make_backend("da3-anyview"), ff.DA3AnyViewBackend)
        self.assertIsInstance(ff.make_backend("VGG-T3"), ff.VGGT3Backend)
        self.assertIsInstance(ff.make_backend("vgg-t3"), ff.VGGT3Backend)

    def test_make_backend_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            ff.make_backend("COLMAP")

    def test_backend_reports_a_missing_checkpoint_instead_of_raising(self):
        usable, reason = ff.VGGTOmegaBackend(checkpoint=None).status()
        self.assertFalse(usable)
        self.assertIn("no checkpoint", reason)

    # ------------------------------------------------------------ hub backends
    def test_only_vggt_omega_needs_a_local_checkpoint(self):
        self.assertTrue(ff.VGGTOmegaBackend(checkpoint=None).needs_local_checkpoint())
        for name in ("DA3-AnyView", "VGG-T3"):
            backend = ff.make_backend(name, checkpoint=None)
            self.assertFalse(backend.needs_local_checkpoint(), name)
            self.assertTrue(backend.hub_repo, name)

    def test_a_hub_backend_without_a_checkpoint_is_usable(self):
        """The VGG-T3 bug: "no checkpoint found" although it needs no file."""
        _usable, reason = ff.make_backend("VGG-T3", checkpoint=None).status()
        self.assertNotIn("no checkpoint", reason)
        self.assertTrue(reason.startswith("VGG-T3:"), reason)

    def test_hub_repo_resolution_order(self):
        self.assertEqual(ff.resolve_hub_repo("DA3-AnyView"),
                         ff.DA3_ANYVIEW_REPO)
        self.assertEqual(ff.resolve_hub_repo("VGG-T3"), "nvidia/vgg-ttt")
        # no hub repo -> the backend requires a local checkpoint
        self.assertEqual(ff.resolve_hub_repo("VGGT-Omega"), "")
        # a widget value that is not an existing path is read as a repo id
        self.assertEqual(ff.resolve_hub_repo("DA3-AnyView", "depth-anything/DA3-SMALL"),
                         "depth-anything/DA3-SMALL")
        # the env var sits between the widget and the default
        with mock.patch.dict("os.environ", {"ENNDEE_DA3_REPO": "org/other"}):
            self.assertEqual(ff.resolve_hub_repo("DA3-AnyView"), "org/other")
            self.assertEqual(ff.resolve_hub_repo("DA3-AnyView", "a/b"), "a/b")

    def test_a_hub_backend_never_scans_folders_for_a_checkpoint(self):
        """It would otherwise match an unrelated model.safetensors in the cache."""
        self.assertIsNone(ff.resolve_checkpoint("DA3-AnyView"))
        self.assertIsNone(ff.resolve_checkpoint("VGG-T3"))

    def test_a_hub_backend_still_takes_an_explicit_local_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(ff.resolve_checkpoint("VGG-T3", folder), Path(folder))

    def test_da3_any_view_uses_the_dinov2_patch_size_and_the_hub(self):
        backend = ff.make_backend("DA3-AnyView")
        self.assertEqual(backend.PATCH_SIZE, 14)
        self.assertEqual(backend.missing_module(), "depth_anything_3")
        self.assertIn("depth-anything/DA3", backend.hub_repo)

    def test_vgg_t3_install_hint_warns_about_the_torch_pin(self):
        hint = ff.VGGT3Backend(checkpoint=None).install_hint()
        self.assertIn("--no-deps", hint)
        self.assertIn("torch", hint)
        self.assertIn("vgg-ttt", hint)

    def test_resolve_checkpoint_honours_an_explicit_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vggt_omega_1b_512.pt"
            path.write_bytes(b"stub")
            self.assertEqual(ff.resolve_checkpoint("VGGT-Omega", str(path)), path)

    def test_resolve_checkpoint_returns_none_for_a_missing_explicit_path(self):
        self.assertIsNone(ff.resolve_checkpoint("VGGT-Omega",
                                                str(Path(tempfile.gettempdir()) / "nope.pt")))

    def test_backend_report_never_raises(self):
        self.assertIsInstance(ff.backend_report(), str)

    def test_backend_report_has_one_line_per_backend(self):
        lines = ff.backend_report().splitlines()
        self.assertEqual(len(lines), len(ff.BACKEND_NAMES))
        for name, line in zip(ff.BACKEND_NAMES, lines):
            self.assertIn(name, line)

    def test_candidate_folders_are_unique(self):
        folders = ff.candidate_folders("VGGT-Omega")
        self.assertEqual(len(folders), len(set(folders)))


class VggT3InputShapeTests(unittest.TestCase):
    """VGG-T^3's ``infer()`` takes ``[N, 3, H, W]``, not VGGT-Omega's 5-D batch.

    This is the second half of the "VGG-T3 does nothing" report: once the package
    was installed the model loaded fine and then died on
    ``N, _, H, W = images.shape -> too many values to unpack``.
    """

    class FakeModel:
        """Mirrors ``vggttt``: it unpacks four dimensions and adds the batch dim."""

        def __init__(self):
            self.seen = None

        def infer(self, images):
            count, _, height, width = images.shape
            self.seen = tuple(images.shape)
            return {
                "pose": torch.eye(4).repeat(count, 1, 1),
                "intrinsics": torch.eye(3).repeat(count, 1, 1),
                "depth": torch.ones(count, height, width, 1),
                "conf": torch.full((count, height, width), 2.0),
            }

    def test_infer_receives_four_dimensions(self):
        backend = ff.VGGT3Backend(checkpoint=None)
        fake = self.FakeModel()
        backend._model = fake  # skip the real load
        images = np.random.RandomState(0).rand(3, 24, 32, 3).astype(np.float32)

        reconstruction = backend.run(images, image_resolution=224, device="cpu")

        self.assertEqual(len(fake.seen), 4, f"infer saw {fake.seen}")
        self.assertEqual(fake.seen[0], 3)
        # camera-to-world is inverted to the COLMAP w2c convention
        self.assertEqual(reconstruction.extrinsics.shape, (3, 3, 4))
        self.assertEqual(reconstruction.intrinsics.shape, (3, 3, 3))
        self.assertEqual(reconstruction.depth.shape[0], 3)
        self.assertEqual(reconstruction.depth_conf.shape, reconstruction.depth.shape)

    def test_a_five_dimensional_input_would_raise(self):
        """Guards the fix: the raw 5-D form really is rejected by that unpack."""
        fake = self.FakeModel()
        with self.assertRaises(ValueError):
            fake.infer(torch.zeros(1, 3, 3, 24, 32))


class NodeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import vggt_lichtfeld_node

        cls.module = vggt_lichtfeld_node
        cls.node = vggt_lichtfeld_node.VGGTLichtfeldTracker

    def test_node_outputs_match_the_colmap_tracker(self):
        self.assertEqual(self.node.RETURN_TYPES,
                         ("CAMERA_TRAJECTORY", "POINTCLOUD", "FLOAT", "STRING"))
        self.assertEqual(self.node.RETURN_NAMES,
                         ("trajectory", "point_cloud", "confidence", "dataset_path"))
        self.assertEqual(self.node.FUNCTION, "track")
        self.assertEqual(self.node.CATEGORY, "Enndee/3D")

    def test_input_types_offer_all_models_and_the_dataset_widgets(self):
        spec = self.node.INPUT_TYPES()
        models = spec["required"]["model"][0]
        self.assertEqual(list(models), ["VGGT-Omega", "DA3-AnyView", "VGG-T3"])
        self.assertEqual(spec["required"]["model"][1]["default"], "VGGT-Omega")
        for name in ("model_path", "image_resolution", "export_depth",
                     "depth_confidence_percentile", "point_cloud_source",
                     "max_points", "per_view_intrinsics", "images_path",
                     "lichtfeld_export_path", "frame_step", "downscale_factor",
                     "use_rmbg", "image_format", "jpeg_quality", "subject_focus"):
            self.assertIn(name, spec["required"], name)
        self.assertIn("images", spec["optional"])
        self.assertEqual(list(spec["required"]["subject_focus"][0]),
                         ["full frame", "subject only"])
        self.assertEqual(spec["required"]["subject_focus"][1]["default"], "full frame")

    def test_track_signature_covers_every_declared_widget(self):
        import inspect

        spec = self.node.INPUT_TYPES()
        expected = set(spec["required"]) | set(spec["optional"])
        actual = set(inspect.signature(self.node.track).parameters) - {"self"}
        self.assertEqual(expected, actual)

    def test_depth_sub_folder_name_is_what_lichtfeld_scans(self):
        self.assertEqual(self.module.DEPTH_DIR, "depth")


class SubjectFocusTests(unittest.TestCase):
    """The priority control: how far depth/points follow the splat mask."""

    @classmethod
    def setUpClass(cls):
        import vggt_lichtfeld_node

        cls.focus = vggt_lichtfeld_node.VGGTLichtfeldTracker._apply_subject_focus

    def test_full_frame_leaves_the_depth_validity_untouched(self):
        valid = np.ones((2, 8, 8), bool)
        result = self.focus(valid, make_mask(2, 8, 8), "full frame", (8, 8), 2)
        self.assertTrue(np.array_equal(result, valid))

    def test_subject_only_ands_the_mask_into_the_validity(self):
        valid = np.ones((2, 8, 8), bool)
        result = self.focus(valid, make_mask(2, 8, 8, keep_columns=4),
                            "subject only", (8, 8), 2)
        self.assertTrue(np.all(result[:, :, :4]))
        self.assertFalse(np.any(result[:, :, 4:]))

    def test_subject_only_resamples_the_mask_to_the_model_resolution(self):
        valid = np.ones((1, 4, 4), bool)
        result = self.focus(valid, make_mask(1, 8, 8, keep_columns=4),
                            "subject only", (4, 4), 1)
        self.assertEqual(result.shape, (1, 4, 4))
        self.assertTrue(np.all(result[:, :, :2]))
        self.assertFalse(np.any(result[:, :, 2:]))

    def test_subject_only_intersects_with_an_already_invalid_pixel(self):
        valid = np.ones((1, 8, 8), bool)
        valid[0, 0, 0] = False                      # no confidence there
        result = self.focus(valid, make_mask(1, 8, 8), "subject only", (8, 8), 1)
        self.assertFalse(bool(result[0, 0, 0]))

    def test_subject_only_without_a_mask_stays_full_frame(self):
        valid = np.ones((1, 4, 4), bool)
        result = self.focus(valid, None, "subject only", (4, 4), 1)
        self.assertTrue(np.array_equal(result, valid))

    def test_subject_only_with_a_mismatched_mask_count_stays_full_frame(self):
        valid = np.ones((2, 4, 4), bool)
        result = self.focus(valid, make_mask(3, 4, 4), "subject only", (4, 4), 2)
        self.assertTrue(np.array_equal(result, valid))

    def test_subject_only_accepts_a_numpy_mask(self):
        valid = np.ones((1, 8, 8), bool)
        result = self.focus(valid, make_mask(1, 8, 8, keep_columns=2).numpy(),
                            "subject only", (8, 8), 1)
        self.assertFalse(np.any(result[:, :, 2:]))


class EndToEndTests(unittest.TestCase):
    """The full ``track()`` pipeline against a stub backend."""

    @classmethod
    def setUpClass(cls):
        import vggt_lichtfeld_node

        cls.module = vggt_lichtfeld_node
        cls.node = vggt_lichtfeld_node.VGGTLichtfeldTracker

    def _run_track(self, backend, export, images, **overrides):
        """Run ``track()`` and return its four output values.

        ``track()`` answers with ComfyUI's ``{"ui": ..., "result": ...}`` form so a
        startup failure can be shown on the node; the ``ui`` block is checked
        separately through :meth:`_run_track_raw`.
        """
        return self._run_track_raw(backend, export, images, **overrides)["result"]

    def _run_track_raw(self, backend, export, images, **overrides):
        options = dict(
            model="VGGT-Omega", model_path="stub", image_resolution=512,
            images_path="", lichtfeld_export_path=str(export),
            export_depth=True, depth_confidence_percentile=0.0,
            point_cloud_source="depth", max_points=500,
            per_view_intrinsics=False, auto_align=False, frame_step=1,
            downscale_factor=1.0, masks_path="", offset_splat=12,
            use_rmbg=False, rmbg_mode="base", rmbg_threshold=0.5,
            rmbg_resize="static", use_gpu=True, autocast=True,
            keep_workspace=False, embed_alpha_in_images=False,
            image_format="PNG", jpeg_quality=90,
        )
        options.update(overrides)
        node = self.node()
        original = self.module.make_backend
        self.module.make_backend = lambda *args, **kwargs: backend
        try:
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                return node.track(images=images, **options)
        finally:
            self.module.make_backend = original

    def test_track_writes_a_complete_lichtfeld_dataset(self):
        from PIL import Image

        backend = StubBackend(make_reconstruction(4))
        images = torch.from_numpy(make_images(4, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            trajectory, point_cloud, confidence, dataset_path = self._run_track(
                backend, export, images)

            self.assertEqual(backend.calls, 1)
            self.assertEqual(backend.released, 1)
            self.assertEqual(dataset_path, str(export))

            # ---- node outputs -----------------------------------------
            self.assertEqual(trajectory["reconstructed_frames"], 4)
            self.assertEqual(trajectory["num_frames"], 4)
            self.assertEqual(trajectory["format"], "opengl")
            self.assertEqual(trajectory["source"], "stub:test")
            self.assertEqual(trajectory["image_names"], ["0001.png", "0002.png",
                                                         "0003.png", "0004.png"])
            self.assertEqual(len(trajectory["matrices"]), 4)
            np.testing.assert_allclose(trajectory["matrices"][0], np.eye(4), atol=1e-5)
            np.testing.assert_allclose(trajectory["intrinsics"],
                                       [32.0, 32.0, 32.0, 24.0], atol=1e-4)
            self.assertEqual(point_cloud["num_points"], 500)
            self.assertEqual(point_cloud["points"].shape, (500, 3))
            self.assertGreater(confidence, 0.0)

            # ---- dataset layout ---------------------------------------
            for name in ("0001.png", "0004.png"):
                self.assertTrue((export / "images" / name).is_file(), name)
            with Image.open(export / "images" / "0001.png") as dataset_image:
                self.assertEqual(dataset_image.size, (64, 48))
                self.assertEqual(dataset_image.mode, "RGB")
            for name in ("0001.depth.png", "0004.depth.png"):
                self.assertTrue((export / "depth" / name).is_file(), name)
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                self.assertTrue((export / "sparse" / "0" / name).is_file(), name)

            # depth maps must match the image resolution and be 16-bit
            with Image.open(export / "depth" / "0002.depth.png") as depth_image:
                self.assertEqual(depth_image.size, (64, 48))
                self.assertEqual(depth_image.mode, "I;16")

            # the sparse model carries every view and the dense cloud
            self.assertEqual(len(read_cameras_txt(export / "sparse" / "0" / "cameras.txt")), 1)
            self.assertEqual(len(read_images_txt(export / "sparse" / "0" / "images.txt")), 4)
            self.assertEqual(len(read_points_txt(export / "sparse" / "0" / "points3D.txt")), 500)

    def test_track_without_depth_writes_no_depth_folder(self):
        backend = StubBackend(make_reconstruction(3))
        images = torch.from_numpy(make_images(3, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            self._run_track(backend, export, images, export_depth=False)
            self.assertFalse((export / "depth").exists())
            self.assertTrue((export / "sparse" / "0" / "images.txt").is_file())

    def test_track_writes_lichtfeld_masks_from_the_alpha_channel(self):
        """Regression: the alpha -> masks/ branch used to hit an undefined name."""
        from PIL import Image

        backend = StubBackend(make_reconstruction(3))
        rgba = make_images(3, 48, 64, channels=4)
        rgba[..., 3] = np.linspace(0.0, 1.0, 64, dtype=np.float32)[None, None, :]
        images = torch.from_numpy(rgba)
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            self._run_track(backend, export, images)

            masks = sorted((export / "masks").glob("*.png"))
            self.assertEqual([path.name for path in masks],
                             ["0001.png", "0002.png", "0003.png"])
            with Image.open(masks[0]) as mask:
                self.assertEqual(mask.size, (64, 48))
                self.assertEqual(mask.mode, "L")
            # the alpha gradient has to survive as a grey ramp (white = keep)
            with Image.open(masks[0]) as mask:
                values = np.array(mask)
            self.assertLess(int(values[:, 0].max()), 10)
            self.assertGreater(int(values[:, -1].max()), 245)

    def test_track_writes_lichtfeld_masks_from_an_explicit_mask_input(self):
        """Regression: the explicit mask branch also uses MASK_LICHTFELD_DIR."""
        from PIL import Image

        backend = StubBackend(make_reconstruction(2))
        images = torch.from_numpy(make_images(2, 48, 64))
        masks = torch.zeros((2, 48, 64), dtype=torch.float32)
        masks[:, 14:34, 14:34] = 1.0
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            self._run_track(backend, export, images, masks_lichtfeld=masks,
                            offset_splat=0)

            written = sorted((export / "masks").glob("*.png"))
            self.assertEqual([path.name for path in written], ["0001.png", "0002.png"])
            with Image.open(written[0]) as mask:
                self.assertEqual(mask.mode, "L")
                values = np.array(mask)
            self.assertGreater(int(values[24, 24]), 200)     # inside the square
            self.assertLess(int(values[2, 2]), 10)          # outside it

    def test_track_with_auto_align_produces_a_ground_aligned_trajectory(self):
        """The workflow runs with auto_align on - that path needs a parser too."""
        backend = StubBackend(make_reconstruction(3))
        images = torch.from_numpy(make_images(3, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            trajectory, point_cloud, _, dataset_path = self._run_track(
                backend, export, images, auto_align=True)

            self.assertEqual(dataset_path, str(export))
            self.assertEqual(len(trajectory["matrices"]), 3)
            self.assertEqual(trajectory["format"], "opengl")
            self.assertTrue(np.all(np.isfinite(trajectory["matrices"])))
            self.assertTrue(np.all(np.isfinite(point_cloud["points"])))
            # alignment touches only the node outputs, never the written dataset
            self.assertEqual(len(read_images_txt(export / "sparse" / "0" / "images.txt")), 3)

    def test_track_subject_only_follows_the_mask_in_depth_and_points(self):
        from PIL import Image

        backend = StubBackend(make_reconstruction(3, model_hw=(24, 32)))
        images = torch.from_numpy(make_images(3, 48, 64))
        masks = make_mask(3, 48, 64, keep_columns=32)     # left half = subject
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            _, point_cloud, _, _ = self._run_track(
                backend, export, images, masks_lichtfeld=masks,
                offset_splat=0, subject_focus="subject only")

            with Image.open(export / "depth" / "0001.depth.png") as depth_image:
                depth = np.array(depth_image)
            self.assertEqual(depth.shape, (48, 64))
            self.assertTrue(np.all(depth[:, :32] > 0))    # the subject keeps its depth
            self.assertFalse(np.any(depth[:, 32:]))       # the background is "no depth"

            # the initial cloud follows the same mask (left half -> x < 0)
            self.assertGreater(len(point_cloud["points"]), 0)
            self.assertTrue(np.all(point_cloud["points"][:, 0] < 0))

    def test_track_full_frame_keeps_the_background_depth(self):
        from PIL import Image

        backend = StubBackend(make_reconstruction(3, model_hw=(24, 32)))
        images = torch.from_numpy(make_images(3, 48, 64))
        masks = make_mask(3, 48, 64, keep_columns=32)
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            _, point_cloud, _, _ = self._run_track(
                backend, export, images, masks_lichtfeld=masks,
                offset_splat=0, subject_focus="full frame")

            with Image.open(export / "depth" / "0001.depth.png") as depth_image:
                depth = np.array(depth_image)
            self.assertTrue(np.all(depth > 0))            # whole frame keeps its depth
            self.assertTrue(np.any(point_cloud["points"][:, 0] > 0))
            # the mask is still written - the trainer decides how to use it
            self.assertEqual(len(list((export / "masks").glob("*.png"))), 3)

    def test_track_without_a_point_cloud_still_writes_poses(self):
        backend = StubBackend(make_reconstruction(2))
        images = torch.from_numpy(make_images(2, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            trajectory, point_cloud, _, _ = self._run_track(
                backend, export, images, point_cloud_source="none")
            self.assertEqual(point_cloud["num_points"], 0)
            self.assertEqual(len(trajectory["matrices"]), 2)
            self.assertEqual(len(read_images_txt(export / "sparse" / "0" / "images.txt")), 2)
            self.assertEqual(len(read_points_txt(export / "sparse" / "0" / "points3D.txt")), 0)

    def test_track_honours_frame_step(self):
        backend = StubBackend(make_reconstruction(3))
        images = torch.from_numpy(make_images(6, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            export = Path(folder) / "dataset"
            trajectory, _, _, _ = self._run_track(backend, export, images, frame_step=2)
            self.assertEqual(trajectory["num_frames"], 3)
            self.assertEqual(len(read_images_txt(export / "sparse" / "0" / "images.txt")), 3)
            self.assertTrue((export / "images" / "0003.png").is_file())
            self.assertFalse((export / "images" / "0004.png").exists())

    def test_track_returns_an_empty_result_without_a_checkpoint(self):
        images = torch.from_numpy(make_images(2, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            # force "no checkpoint" instead of relying on the machine state
            original = self.module.resolve_checkpoint
            self.module.resolve_checkpoint = lambda *args, **kwargs: None
            try:
                with mock.patch.object(torch.cuda, "is_available", return_value=True):
                    result = self.node().track(
                        model="VGGT-Omega", model_path="", image_resolution=512,
                        images_path="", lichtfeld_export_path=str(Path(folder) / "dataset"),
                        export_depth=True, depth_confidence_percentile=20.0,
                        point_cloud_source="depth", max_points=100,
                        per_view_intrinsics=False, auto_align=False, frame_step=1,
                        downscale_factor=1.0, masks_path="", offset_splat=12,
                        use_rmbg=False, rmbg_mode="base", rmbg_threshold=0.5,
                        rmbg_resize="static", use_gpu=True, autocast=True,
                        keep_workspace=False, embed_alpha_in_images=False,
                        image_format="PNG", jpeg_quality=90, images=images,
                    )
            finally:
                self.module.resolve_checkpoint = original
        # the failure must be visible on the node, not only in the console
        self.assertIn("no checkpoint", result["ui"]["text"][0])
        self.assertIn("Available backends:", result["ui"]["text"][0])
        trajectory, point_cloud, confidence, dataset_path = result["result"]
        self.assertEqual(dataset_path, "")
        self.assertEqual(confidence, 0.0)
        self.assertEqual(point_cloud["num_points"], 0)
        self.assertEqual(trajectory["reconstructed_frames"], 0)

    def test_track_without_cuda_returns_an_empty_result(self):
        images = torch.from_numpy(make_images(2, 48, 64))
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                result = self.node().track(
                    model="VGGT-Omega", model_path="stub", image_resolution=512,
                    images_path="", lichtfeld_export_path=str(Path(folder) / "dataset"),
                    export_depth=True, depth_confidence_percentile=20.0,
                    point_cloud_source="depth", max_points=100,
                    per_view_intrinsics=False, auto_align=False, frame_step=1,
                    downscale_factor=1.0, masks_path="", offset_splat=12,
                    use_rmbg=False, rmbg_mode="base", rmbg_threshold=0.5,
                    rmbg_resize="static", use_gpu=True, autocast=True,
                    keep_workspace=False, embed_alpha_in_images=False,
                    image_format="PNG", jpeg_quality=90, images=images,
                )
        self.assertIn("CUDA is not available", result["ui"]["text"][0])
        self.assertEqual(result["result"][3], "")
        self.assertEqual(result["result"][2], 0.0)


class RealModelTests(unittest.TestCase):
    """Smoke test against the real checkpoint - skipped when unavailable."""

    def test_vggt_omega_reconstructs_a_synthetic_batch(self):
        checkpoint = ff.resolve_checkpoint("VGGT-Omega")
        if checkpoint is None or not torch.cuda.is_available():
            self.skipTest("no VGGT-Omega checkpoint / no CUDA")
        backend = ff.make_backend("VGGT-Omega", checkpoint=checkpoint)
        usable, reason = backend.status()
        if not usable:
            self.skipTest(reason)

        images = make_images(3, 64, 64)
        try:
            reconstruction = backend.run(images, image_resolution=256)
        finally:
            backend.release()
        self.assertEqual(reconstruction.num_views, 3)
        self.assertEqual(reconstruction.extrinsics.shape, (3, 3, 4))
        self.assertEqual(reconstruction.intrinsics.shape, (3, 3, 3))
        self.assertEqual(reconstruction.depth.shape[0], 3)
        self.assertTrue(np.all(np.isfinite(reconstruction.depth)))
        self.assertTrue(np.all(reconstruction.depth > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

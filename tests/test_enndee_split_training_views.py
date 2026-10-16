"""Synthetic tests for shared-pose complementary subject/background exports."""

import json
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

from enndee_colmap.split_training_views import (  # noqa: E402
    export_complementary_training_views,
    prepare_subject_mask,
    remove_stale_background_views,
)
from glomap_lichtfeld_node import GLOMAPLichtfeldTracker  # noqa: E402


class FakeParser:
    # Source names intentionally sort differently from numeric image IDs.
    images = {
        10: {"name": "frame_01.jpg", "camera_id": 4,
             "qvec": np.array([1.0, 0.1, 0.2, 0.3]),
             "tvec": np.array([1.0, 2.0, 3.0])},
        3: {"name": "frame_00.jpg", "camera_id": 2,
            "qvec": np.array([0.9, 0.4, 0.1, 0.2]),
            "tvec": np.array([-1.0, -2.0, -3.0])},
    }


class SplitMaskTests(unittest.TestCase):
    def test_mask_remains_aligned_to_selection_and_frame_step(self):
        images = torch.zeros((2, 4, 5, 3), dtype=torch.float32)
        masks = np.zeros((4, 1, 4, 5), dtype=np.float32)
        for index in range(4):
            masks[index].fill(index / 3)
        selected = GLOMAPLichtfeldTracker._subset_prepared_masks(
            torch.from_numpy(masks), [1, 3], 4, "split")
        aligned = prepare_subject_mask(selected, images)
        stepped = aligned[::2]
        self.assertEqual(tuple(stepped.shape), (1, 4, 5))
        self.assertTrue(torch.allclose(stepped, torch.full_like(stepped, 1 / 3)))

    def test_mask_count_mismatch_fails_closed(self):
        images = torch.zeros((3, 4, 5, 3))
        masks = torch.zeros((2, 1, 4, 5))
        with self.assertRaisesRegex(ValueError, "frames"):
            prepare_subject_mask(masks, images)


class ComplementaryExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        (self.root / "sparse" / "0").mkdir(parents=True)
        (self.root / "depth").mkdir()
        (self.root / "sparse" / "0" / "images.txt").write_text(
            "# Image list with two lines of data per image:\n"
            "# Number of images: 2\n"
            "3 0.9 0.4 0.1 0.2 -1 -2 -3 2 0001.png\n\n"
            "10 1 0.1 0.2 0.3 1 2 3 4 0002.png\n\n",
            encoding="utf-8")
        Image.fromarray(np.full((4, 5, 3), 90, np.uint8)).save(
            self.root / "images" / "0001.png")
        Image.fromarray(np.full((4, 5, 3), 150, np.uint8)).save(
            self.root / "images" / "0002.png")
        Image.fromarray(np.arange(20, dtype=np.uint16).reshape(4, 5)).save(
            self.root / "depth" / "0001.depth.png")

    def tearDown(self):
        self.temp.cleanup()

    def test_inverse_masks_share_poses_and_create_valid_colmap_rows(self):
        subject = np.zeros((2, 4, 5), dtype=np.float32)
        subject[0, :2] = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        subject[1] = 0.4
        pairs = export_complementary_training_views(
            self.root, FakeParser(), subject, log=lambda _message: None)

        for source_index in range(1, 3):
            original = np.asarray(Image.open(self.root / "masks" / f"{source_index:04d}.png"))
            inverse = np.asarray(Image.open(
                self.root / "masks" / f"{source_index:04d}_background.png"))
            self.assertTrue(np.array_equal(inverse, 255 - original))
            self.assertTrue((self.root / "images" /
                             f"{source_index:04d}_background.png").is_file())

        self.assertEqual([pair["source_image_id"] for pair in pairs], [3, 10])
        self.assertEqual([pair["background_image_id"] for pair in pairs], [11, 12])
        self.assertEqual([pair["camera_id"] for pair in pairs], [2, 4])
        rows = (self.root / "sparse" / "0" / "images.txt").read_text(
            encoding="utf-8").splitlines()
        pose_rows = [line for line in rows if "_background.png" in line]
        self.assertEqual(len(pose_rows), 2)
        self.assertIn("# Number of images: 4", rows)
        self.assertIn("0.900000 0.400000 0.100000 0.200000 -1.000000 -2.000000 -3.000000 2 "
                      "0001_background.png", pose_rows[0])
        self.assertEqual(rows[rows.index(pose_rows[0]) + 1], "")
        metadata = json.loads((self.root / "subject_background_split.json").read_text(
            encoding="utf-8"))
        self.assertEqual(metadata["camera_solve"], "single shared COLMAP/GLOMAP reconstruction")

    def test_stale_background_artifacts_are_removed(self):
        for folder, name in (("images", "0001_background.png"),
                             ("masks", "0001_background.png"),
                             ("depth", "0001_background.depth.png")):
            (self.root / folder / name).write_bytes(b"old")
        sidecar = self.root / "subject_background_split.json"
        sidecar.write_text("{}", encoding="utf-8")
        remove_stale_background_views(self.root)
        self.assertFalse(sidecar.exists())
        self.assertFalse(list((self.root / "images").glob("*_background*")))
        self.assertFalse(list((self.root / "masks").glob("*_background*")))
        self.assertFalse(list((self.root / "depth").glob("*_background*")))


if __name__ == "__main__":
    unittest.main()
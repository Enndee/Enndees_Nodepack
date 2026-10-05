"""Tests for the Enndee Sharpness Analyzer and Top-N Sharp Frame Selector."""

import sys
import unittest
from pathlib import Path

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))

from enndee_sharp_selector import (  # noqa: E402
    Enndee_SharpFrameSelector,
    Enndee_SharpnessAnalyzer,
)


def frames_with_ids(count, height=4, width=4):
    """Frame ``i`` is filled with the constant ``i`` so picks are verifiable."""
    batch = torch.empty((count, height, width, 3), dtype=torch.float32)
    for index in range(count):
        batch[index].fill_(float(index))
    return batch


def selected_ids(images):
    return [int(round(float(images[i, 0, 0, 0]))) for i in range(images.shape[0])]


class SharpFrameSelectorTests(unittest.TestCase):
    def select(self, images, scores, **overrides):
        node = Enndee_SharpFrameSelector()
        kwargs = dict(
            selection_method="batched_topn",
            batch_size=4,
            batch_buffer=0,
            num_frames=3,
            min_sharpness=0.0,
        )
        kwargs.update(overrides)
        return node.select_frames(images, scores, **kwargs)

    def test_return_types_and_inputs(self):
        self.assertEqual(Enndee_SharpFrameSelector.RETURN_TYPES, ("IMAGE", "INT"))
        self.assertEqual(Enndee_SharpFrameSelector.RETURN_NAMES,
                         ("selected_images", "count"))
        required = Enndee_SharpFrameSelector.INPUT_TYPES()["required"]
        self.assertEqual(list(required)[:2], ["images", "scores"])
        self.assertEqual(list(required["selection_method"][0]),
                         ["batched_topn", "batched", "best_n"])
        self.assertEqual(required["selection_method"][1]["default"], "batched_topn")
        self.assertTrue(all("tooltip" in metadata for _o, metadata in required.values()))

    def test_batched_topn_keeps_three_of_every_four_73_frames(self):
        images = frames_with_ids(73)
        scores = [float(i) for i in range(73)]
        selected, count = self.select(images, scores)
        self.assertEqual(count, 55)
        expected = []
        for start in range(0, 72, 4):
            expected.extend([start + 1, start + 2, start + 3])
        expected.append(72)
        self.assertEqual(selected_ids(selected), expected)

    def test_batched_topn_keeps_three_of_every_four_243_frames(self):
        images = frames_with_ids(243)
        scores = [float(i) for i in range(243)]
        selected, count = self.select(images, scores)
        self.assertEqual(count, 183)
        self.assertEqual(selected.shape[0], 183)

    def test_batched_topn_respects_min_sharpness(self):
        images = frames_with_ids(8)
        scores = [float(i) for i in range(8)]
        selected, count = self.select(images, scores, min_sharpness=1.5)
        self.assertEqual(count, 5)
        self.assertEqual(selected_ids(selected), [2, 3, 5, 6, 7])

    def test_batched_topn_uses_batch_buffer_as_gap(self):
        images = frames_with_ids(12)
        scores = [float(i) for i in range(12)]
        selected, count = self.select(images, scores, num_frames=1, batch_buffer=2)
        self.assertEqual(count, 2)
        self.assertEqual(selected_ids(selected), [3, 9])

    def test_batched_mode_keeps_single_sharpest_per_chunk(self):
        images = frames_with_ids(12)
        scores = [float(i) for i in range(12)]
        selected, count = self.select(images, scores, selection_method="batched")
        self.assertEqual(count, 3)
        self.assertEqual(selected_ids(selected), [3, 7, 11])

    def test_best_n_keeps_global_top_frames(self):
        images = frames_with_ids(10)
        scores = [float(i) for i in range(10)]
        selected, count = self.select(images, scores, selection_method="best_n",
                                      num_frames=3)
        self.assertEqual(count, 3)
        self.assertEqual(selected_ids(selected), [7, 8, 9])

    def test_mismatched_lengths_are_truncated(self):
        images = frames_with_ids(6)
        scores = [float(i) for i in range(4)]
        selected, count = self.select(images, scores, num_frames=1)
        self.assertEqual(count, 1)
        self.assertEqual(selected_ids(selected), [3])

    def test_empty_selection_returns_placeholder(self):
        images = frames_with_ids(4)
        scores = [0.0, 0.0, 0.0, 0.0]
        selected, count = self.select(images, scores, num_frames=1,
                                      min_sharpness=5.0)
        self.assertEqual(count, 0)
        self.assertEqual(tuple(selected.shape), (1, 4, 4, 3))


class SharpnessAnalyzerTests(unittest.TestCase):
    def test_sharp_frame_scores_higher_than_flat_frame(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("cv2 not available")

        torch.manual_seed(0)
        noise = torch.rand((8, 8, 3), dtype=torch.float32)
        flat = torch.zeros((8, 8, 3), dtype=torch.float32)
        batch = torch.stack([noise, flat], dim=0)
        (scores,) = Enndee_SharpnessAnalyzer().analyze_sharpness(batch)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])
        self.assertAlmostEqual(scores[1], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

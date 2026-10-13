"""In-tracker sharp selection + subject focus (2026-10-14) for the COLMAP/GLOMAP trackers.

Pure logic, no binaries and no model: the shared selection functions, the face_frames
remapping that makes sharp selection safe for face-detail masks, the subject-mask
derivation for subject-focused alignment, and the widget-order contract.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_sharp_selector as sharp  # noqa: E402
from glomap_lichtfeld_node import GLOMAPLichtfeldTracker  # noqa: E402
from colmap_lichtfeld_node import ColmapLichtfeldTracker  # noqa: E402


def _checkerboard(size=32):
    """Pixel-alternating board: maximal Laplacian variance (the 'sharp' frame)."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return (((xx + yy) % 2).float().unsqueeze(-1).expand(size, size, 3))


def _flat(size=32):
    return torch.zeros(size, size, 3)


class SharedSelectionFunctionTests(unittest.TestCase):
    def test_scores_rank_a_sharp_frame_above_a_flat_one(self):
        scores = sharp.laplacian_scores(torch.stack([_flat(), _checkerboard()]))
        self.assertGreater(scores[1], scores[0] * 10 + 1.0)

    def test_batched_topn_keeps_the_top_of_every_chunk(self):
        scores = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]
        self.assertEqual(sharp.select_frames(scores, "batched_topn", 2, 1),
                         [1, 3, 5])
        self.assertEqual(sharp.select_frames(scores, "batched", 2, 1), [1, 3, 5])

    def test_best_n_keeps_the_global_top(self):
        scores = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]
        self.assertEqual(sharp.select_frames(scores, "best_n", 2, 2), [1, 3])

    def test_the_node_and_the_shared_function_pick_the_same_frames(self):
        scores = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]
        self.assertEqual(sharp.select_frames(scores, "batched_topn", 2, 1),
                         sharp.Enndee_SharpFrameSelector._select_batched_topn(
                             scores, 2, 0, 1, 0.0))


class SharpSelectionTrackerTests(unittest.TestCase):
    def test_select_sharp_indices_keeps_the_sharp_frames(self):
        images = torch.stack([_flat(), _checkerboard(), _flat(), _checkerboard()])
        kept = GLOMAPLichtfeldTracker._select_sharp_indices(
            images, "best_n", 4, 2)
        self.assertEqual(kept, [1, 3])

    def test_remapping_moves_face_frames_onto_the_kept_batch(self):
        self.assertEqual(
            GLOMAPLichtfeldTracker._remap_face_frames("0,5,9", [0, 2, 5]),
            "0,2")

    def test_remapping_drops_frames_the_selection_removed(self):
        self.assertEqual(GLOMAPLichtfeldTracker._remap_face_frames("7", [0, 1]),
                         "")

    def test_remapping_handles_semicolons_and_whitespace(self):
        self.assertEqual(
            GLOMAPLichtfeldTracker._remap_face_frames(" 1 ; 3 ", [1, 3]),
            "0,1")

    def test_empty_face_frames_stay_empty(self):
        self.assertEqual(GLOMAPLichtfeldTracker._remap_face_frames("", [0]), "")


class SubjectFocusTests(unittest.TestCase):
    def test_alpha_subject_reads_the_alpha_channel(self):
        rgba = torch.zeros(2, 8, 8, 4)
        rgba[..., 3] = 0.25
        subject = GLOMAPLichtfeldTracker._alpha_subject(rgba)
        self.assertEqual(tuple(subject.shape), (2, 1, 8, 8))
        self.assertTrue(torch.all(subject == 0.25))

    def test_rgb_batches_have_no_subject(self):
        self.assertIsNone(GLOMAPLichtfeldTracker._alpha_subject(torch.zeros(1, 4, 4, 3)))

    def test_feature_masks_exclude_the_background(self):
        # masks_glomap convention: WHITE = excluded from feature extraction. With the
        # subject at alpha=1 the background (alpha=0) must come out WHITE.
        rgba = torch.zeros(1, 4, 4, 4)
        rgba[0, :2, :, 3] = 1.0                    # top half = subject
        subject = GLOMAPLichtfeldTracker._alpha_subject(rgba)
        feature = 1.0 - subject
        self.assertTrue(torch.all(feature[0, :, 2:, :] == 1.0))   # background excluded
        self.assertTrue(torch.all(feature[0, :, :2, :] == 0.0))   # subject kept


class MaskSubsetTests(unittest.TestCase):
    def test_full_batch_masks_follow_the_selection(self):
        masks = torch.arange(4.0).view(4, 1, 1, 1).expand(4, 1, 2, 2).contiguous()
        subset = GLOMAPLichtfeldTracker._subset_prepared_masks(masks, [0, 3], 4, "G")
        self.assertEqual(tuple(subset.shape), (2, 1, 2, 2))
        self.assertEqual(float(subset[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(subset[1, 0, 0, 0]), 3.0)

    def test_already_selected_masks_are_left_alone(self):
        masks = torch.zeros(2, 1, 2, 2)
        self.assertIs(GLOMAPLichtfeldTracker._subset_prepared_masks(
            masks, [0, 3], 4, "G"), masks)

    def test_missing_masks_stay_missing(self):
        self.assertIsNone(GLOMAPLichtfeldTracker._subset_prepared_masks(
            None, [0], 1, "G"))


class SidecarAndWidgetContractTests(unittest.TestCase):
    def test_sidecar_records_the_original_frame_numbers(self):
        record = {"selection_method": "batched_topn", "input_frames": 6,
                  "kept_input_indices": [0, 2, 5], "dropped_input_indices": [1, 3, 4]}
        with tempfile.TemporaryDirectory() as folder:
            GLOMAPLichtfeldTracker._write_sharp_selection_sidecar(Path(folder), record)
            data = json.loads((Path(folder) / "sharp_selection.json").read_text(
                encoding="utf-8"))
        self.assertEqual(data["kept_input_indices"], [0, 2, 5])
        self.assertEqual(data["dropped_input_indices"], [1, 3, 4])

    def test_no_record_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            GLOMAPLichtfeldTracker._write_sharp_selection_sidecar(Path(folder), None)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_both_trackers_end_with_face_then_the_new_widgets(self):
        face = list(GLOMAPLichtfeldTracker._face_widgets())
        added = list(GLOMAPLichtfeldTracker._new_widgets())
        self.assertEqual(added, ["subject_focus", "sharp_selection", "sharp_method",
                                 "sharp_batch", "sharp_num_frames"])
        for cls in (GLOMAPLichtfeldTracker, ColmapLichtfeldTracker):
            optional = list(cls.INPUT_TYPES()["optional"])
            self.assertEqual(optional[-len(face) - len(added):], face + added,
                             cls.__name__)

    def test_the_rmbg_widget_key_is_unchanged_for_saved_workflows(self):
        for cls in (GLOMAPLichtfeldTracker, ColmapLichtfeldTracker):
            self.assertIn("use_rmbg", cls.INPUT_TYPES()["optional"])

    def test_track_accepts_the_new_widgets(self):
        import inspect

        for cls in (GLOMAPLichtfeldTracker, ColmapLichtfeldTracker):
            params = inspect.signature(cls.track).parameters
            for name in ("subject_focus", "sharp_selection", "sharp_method",
                         "sharp_batch", "sharp_num_frames"):
                self.assertIn(name, params, f"{cls.__name__}.{name}")


if __name__ == "__main__":
    unittest.main()
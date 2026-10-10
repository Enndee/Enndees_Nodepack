"""Tests for enndee_rmbg (RMBG-2.0 + the RMBG-1.4 fallback); no model is loaded."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_rmbg as rmbg  # noqa: E402


FRAME = np.zeros((6, 8, 3), dtype=np.uint8)
FRAME[1:5, 2:6] = 220


def _soft(frames):
    """A deterministic matte: half the frame is foreground."""
    mattes = []
    for frame in frames:
        matte = np.zeros(frame.shape[:2], dtype=np.float32)
        matte[: frame.shape[0] // 2] = 1.0
        mattes.append(matte)
    return mattes


class WidgetSurfaceTests(unittest.TestCase):
    def test_2_0_is_first_and_the_default(self):
        self.assertEqual(rmbg.RMBG_MODES[0], "2.0")
        self.assertEqual(rmbg.RMBG_DEFAULT_MODE, "2.0")

    def test_the_legacy_ids_are_kept_so_old_workflows_still_validate(self):
        for legacy in ("base", "fast", "base-nightly"):
            self.assertIn(legacy, rmbg.RMBG_MODES)
        self.assertEqual(list(rmbg.LEGACY_MODES), ["base", "fast", "base-nightly"])

    def test_every_tracker_node_offers_the_same_list(self):
        import colmap_lichtfeld_node as colmap_mod
        import glomap_lichtfeld_node as glomap_mod
        import vggt_lichtfeld_node as vggt_mod

        for module, cls in ((glomap_mod, glomap_mod.GLOMAPLichtfeldTracker),
                            (colmap_mod, colmap_mod.ColmapLichtfeldTracker),
                            (vggt_mod, vggt_mod.VGGTLichtfeldTracker)):
            spec = cls.INPUT_TYPES()
            widget = spec["required"].get("rmbg_mode") or spec["optional"]["rmbg_mode"]
            self.assertEqual(list(widget[0]), list(rmbg.RMBG_MODES), cls.__name__)
            self.assertEqual(widget[1]["default"], rmbg.RMBG_DEFAULT_MODE, cls.__name__)

    def test_the_signature_default_matches_the_widget(self):
        """The drift that was found while wiring 2.0 in."""
        import inspect

        import colmap_lichtfeld_node as colmap_mod
        import glomap_lichtfeld_node as glomap_mod

        for cls in (glomap_mod.GLOMAPLichtfeldTracker, colmap_mod.ColmapLichtfeldTracker):
            default = inspect.signature(cls.track).parameters["rmbg_mode"].default
            self.assertEqual(default, rmbg.RMBG_DEFAULT_MODE, cls.__name__)


class RemoveBackgroundTests(unittest.TestCase):
    """The dispatch, the threshold contract and - most importantly - the fallback."""

    def setUp(self):
        self._saved = (rmbg._rmbg2_failed, rmbg._rmbg2_model, rmbg._rmbg2_key)
        rmbg._rmbg2_failed = False
        rmbg._rmbg2_model = None
        rmbg._rmbg2_key = None
        self.addCleanup(self._restore)

    def _restore(self):
        rmbg._rmbg2_failed, rmbg._rmbg2_model, rmbg._rmbg2_key = self._saved

    def test_threshold_binarises_the_matte(self):
        with mock.patch.object(rmbg, "_rmbg2_alpha", lambda f, d, log: _soft(f)):
            mattes = rmbg.remove_background([FRAME], mode="2.0", device="cpu",
                                            threshold=0.5)
        self.assertIsNotNone(mattes)
        self.assertEqual(set(np.unique(mattes[0]).tolist()), {0.0, 1.0})

    def test_no_threshold_keeps_the_soft_matte(self):
        with mock.patch.object(rmbg, "_rmbg2_alpha", lambda f, d, log: _soft(f)):
            mattes = rmbg.remove_background([FRAME], mode="2.0", device="cpu",
                                            threshold=None)
        self.assertEqual(len(np.unique(mattes[0])), 2)   # 0.0 and 1.0 in the fake
        self.assertEqual(mattes[0].dtype, np.float32)

    def test_a_gated_2_0_falls_back_on_every_later_call(self):
        """The KeyError regression: a second call must not retry mode='2.0'."""
        used = []

        def failing(_frames, _device, _log):
            raise OSError("You are trying to access a gated repo. 401 Client Error.")

        def legacy(_frames, mode, _resize, _device, _threshold, _log, _warn):
            used.append(mode)
            return _soft(_frames)

        with mock.patch.object(rmbg, "_rmbg2_alpha", failing), \
                mock.patch.object(rmbg, "_legacy_alpha", legacy):
            first = rmbg.remove_background([FRAME], mode="2.0", device="cpu")
            second = rmbg.remove_background([FRAME], mode="2.0", device="cpu")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        # "base" both times - never "2.0", which the Remover would reject
        self.assertEqual(used, ["base", "base"])

    def test_the_legacy_modes_go_straight_to_1_4(self):
        used = []
        with mock.patch.object(
                rmbg, "_legacy_alpha",
                lambda _f, mode, *_a: (used.append(mode), _soft([FRAME]))[1]):
            mattes = rmbg.remove_background([FRAME], mode="fast", device="cpu")
        self.assertEqual(used, ["fast"])
        self.assertIsNotNone(mattes)

    def test_nothing_loadable_returns_none(self):
        with mock.patch.object(rmbg, "_legacy_alpha",
                               side_effect=ImportError("no transparent_background")):
            self.assertIsNone(rmbg.remove_background([FRAME], mode="base",
                                                     device="cpu"))

    def test_no_frames_is_not_an_error(self):
        self.assertIsNone(rmbg.remove_background([], mode="2.0", device="cpu"))


class HintTests(unittest.TestCase):
    def test_a_gated_repo_gets_an_actionable_hint(self):
        hint = rmbg.rmbg2_hint(OSError("You are trying to access a gated repo. 401"))
        self.assertIn("HF_TOKEN", hint)
        self.assertIn("briaai/RMBG-2.0", hint)

    def test_an_offline_error_mentions_the_local_repo_override(self):
        hint = rmbg.rmbg2_hint(OSError("Connection error: cannot resolve huggingface.co"))
        self.assertIn("ENNDEE_RMBG2_REPO", hint)

    def test_an_unknown_error_is_passed_through(self):
        hint = rmbg.rmbg2_hint(ValueError("something else"))
        self.assertIn("something else", hint)

    def test_the_repo_can_be_overridden(self):
        with mock.patch.dict("os.environ", {"ENNDEE_RMBG2_REPO": "D:/local/rmbg2"}):
            self.assertEqual(rmbg.rmbg2_repo(), "D:/local/rmbg2")
        self.assertEqual(rmbg.rmbg2_repo(), rmbg.RMBG2_REPO)


if __name__ == "__main__":
    unittest.main()

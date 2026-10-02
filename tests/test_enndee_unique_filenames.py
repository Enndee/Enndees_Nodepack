"""Tests for the unique-filename save hook (counter only on collision)."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR.parents[1]))

import folder_paths  # noqa: E402

import enndee_unique_filenames as unique  # noqa: E402


class StripCounterTests(unittest.TestCase):
    def test_strip_counter_patterns(self):
        cases = {
            # core format: trailing underscore before the extension
            "MiniMax_H3_00001_.mp4": "MiniMax_H3.mp4",
            # VHS formats: no trailing underscore, -audio sibling
            "Preview_00025.mp4": "Preview.mp4",
            "Clip_00001-audio.mp4": "Clip-audio.mp4",
            # 6-digit DateTime suffix must survive; only the LAST counter goes
            "Preview_20261002_143022_00001_.png": "Preview_20261002_143022.png",
            "run_00042_00001_.png": "run_00042.png",
            # no counter at all -> untouched
            "already_unique.png": "already_unique.png",
            # documented limit: 6-digit counters are not stripped
            "X_100000_.png": "X_100000_.png",
        }
        for source, expected in cases.items():
            self.assertEqual(unique.strip_counter(source), expected, source)

    def test_uniquify_numbers_only_on_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(unique.uniquify(folder, "a.png"), "a.png")
            (Path(folder) / "a.png").write_bytes(b"x")
            self.assertEqual(unique.uniquify(folder, "a.png"), "a_1.png")
            (Path(folder) / "a_1.png").write_bytes(b"x")
            self.assertEqual(unique.uniquify(folder, "a.png"), "a_2.png")

    def test_rename_family_renames_siblings_and_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("S_00001.mp4", "S_00001-audio.mp4", "S_00001.png",
                         "Other_00001.mp4"):
                (Path(folder) / name).write_bytes(b"x")
            mapping = unique.rename_family(folder, "S_00001.mp4")
            self.assertEqual(mapping["S_00001.mp4"], "S.mp4")
            self.assertEqual(mapping["S_00001-audio.mp4"], "S-audio.mp4")
            self.assertEqual(mapping["S_00001.png"], "S.png")
            self.assertTrue((Path(folder) / "Other_00001.mp4").exists())
            # collision: target taken -> plain number appended
            (Path(folder) / "T.mp4").write_bytes(b"x")
            (Path(folder) / "T_00001.mp4").write_bytes(b"x")
            mapping = unique.rename_family(folder, "T_00001.mp4")
            self.assertEqual(mapping["T_00001.mp4"], "T_1.mp4")


class WrapTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = str(self.temp_dir.name)
        patcher = mock.patch.object(
            folder_paths, "get_output_directory", return_value=self.output_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_legacy_dict_result_is_renamed_and_ui_updated(self):
        save_dir = self.output_dir

        class FakeSaver:
            def save_all(self, name):
                with open(os.path.join(save_dir, name), "wb") as handle:
                    handle.write(b"x")
                return {"ui": {"images": [
                    {"filename": name, "subfolder": "", "type": "output"},
                    {"filename": "keep_00001_.png", "subfolder": "",
                     "type": "temp"},
                ]}}

        self.assertTrue(unique._wrap_callable(FakeSaver, "save_all"))
        result = FakeSaver().save_all("Job_00001_.png")
        images = result["ui"]["images"]
        self.assertEqual(images[0]["filename"], "Job.png")
        self.assertTrue(os.path.exists(os.path.join(save_dir, "Job.png")))
        # temp entries (PreviewImage etc.) are never touched
        self.assertEqual(images[1]["filename"], "keep_00001_.png")

    def test_wrap_callable_is_idempotent(self):
        class Fake:
            def save(self):
                return {}

        self.assertTrue(unique._wrap_callable(Fake, "save"))
        first = Fake.save
        self.assertTrue(unique._wrap_callable(Fake, "save"))
        self.assertIs(Fake.save, first)
        self.assertTrue(getattr(Fake.save, "_enndee_unique", False))

    def test_node_output_style_ui_is_processed(self):
        class FakePreview:
            def __init__(self, values):
                self.values = values

        class FakeNodeOutput:
            def __init__(self, ui):
                self.ui = ui

        name = "Clip_00007_.png"
        with open(os.path.join(self.output_dir, name), "wb") as handle:
            handle.write(b"x")
        entry = {"filename": name, "subfolder": "", "type": "output"}
        unique._process_result(FakeNodeOutput(FakePreview([entry])))
        self.assertEqual(entry["filename"], "Clip.png")

    def test_vhs_lookup_wraps_via_node_class_mappings(self):
        class FakeVHS:
            FUNCTION = "combine_video"

            @classmethod
            def combine_video(cls):
                return {"ui": {"gifs": []}}

        module = types.ModuleType("fake_vhs_pack_for_test")
        module.NODE_CLASS_MAPPINGS = {"VHS_VideoCombine": FakeVHS}
        sys.modules[module.__name__] = module
        try:
            self.assertTrue(unique._install_vhs())
            self.assertTrue(getattr(FakeVHS.combine_video, "_enndee_unique", False))
            self.assertTrue(unique._install_vhs())  # second run is a no-op
        finally:
            del sys.modules[module.__name__]

    def test_install_can_be_disabled_by_env(self):
        with mock.patch.dict(os.environ, {"ENNDEE_KEEP_FILE_COUNTER": "1"}):
            self.assertFalse(unique.install())


class SaveImageIntegrationTests(unittest.TestCase):
    """End-to-end through the real core SaveImage node (when importable)."""

    def setUp(self):
        core = None
        try:
            import nodes

            if hasattr(nodes, "SaveImage"):
                core = nodes
        except Exception:
            core = None
        if core is None:
            # Another test may have put the pack root on sys.path so that
            # "import nodes" resolves to the pack's nodes package - load the
            # core module straight from its file instead.
            try:
                import importlib.util

                core_path = str(PACK_DIR.parents[1] / "nodes.py")
                spec = importlib.util.spec_from_file_location(
                    "_core_nodes_for_unique_test", core_path)
                core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(core)
            except Exception as error:  # pragma: no cover - depends on boot
                self.skipTest(f"core nodes not importable: {error}")
        self.nodes = core
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = str(self.temp_dir.name)
        patcher = mock.patch.object(
            folder_paths, "get_output_directory", return_value=self.output_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_three_saves_number_only_on_collision(self):
        import torch

        saver = self.nodes.SaveImage()
        saver.output_dir = self.output_dir
        unique._wrap_callable(self.nodes.SaveImage, self.nodes.SaveImage.FUNCTION)

        image = torch.zeros((1, 8, 8, 3))
        for wanted in ("UniqTest.png", "UniqTest_1.png", "UniqTest_2.png"):
            result = saver.save_images(image, filename_prefix="UniqTest")
            reported = result["ui"]["images"][0]["filename"]
            self.assertEqual(reported, wanted)
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, wanted)))


if __name__ == "__main__":
    unittest.main()


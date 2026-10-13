"""Tests for the gsplat headless trainer node.

The node exists because the measured gap between gsplat and the LichtFeld trainer was large,
and because two flags decide whether a run is usable at all. Those are what these tests
pin: the flags, the widget/signature contract ComfyUI enforces, and drop-in compatibility
with the LichtFeld node it can replace.

Training itself is not run here - the trainer is a subprocess against a CUDA environment.
The command it would run is captured instead, which is where the interesting mistakes live.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(PACK / "nodes"))

module = importlib.import_module("gsplat_trainer_node")


class _Result:
    returncode = 1
    stdout = ""
    stderr = "stopped by the test"


def _capture_command(**overrides):
    """Run train() with the subprocess stubbed, and return the command it built."""
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        return _Result()

    original = module.subprocess.run
    module.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "ds"
            (dataset / "images").mkdir(parents=True)
            (dataset / "sparse" / "0").mkdir(parents=True)
            (dataset / "sparse" / "0" / "images.bin").write_bytes(b"")
            options = dict(
                dataset_path=str(dataset),
                output_path=str(Path(folder) / "out"),
                output_name="", iterations=7000, sh_degree=3, data_factor=1,
                normalize_world_space=False, prepare_dataset=False, uniform_crop=False,
                max_image_size=1600, test_every=8,
                python_executable=sys.executable, trainer_script=str(Path(__file__)),
                overwrite_output=True,
            )
            options.update(overrides)
            module.GsplatHeadlessTrainer().train(**options)
    finally:
        module.subprocess.run = original
    return captured.get("command", [])


class CommandTests(unittest.TestCase):
    """The flags that decide whether the result is usable."""

    def test_normalize_world_space_is_disabled_by_default(self):
        self.assertIn("--no-normalize-world-space",
                      _capture_command(normalize_world_space=False))

    def test_enabling_normalize_world_space_drops_the_flag(self):
        self.assertNotIn("--no-normalize-world-space",
                         _capture_command(normalize_world_space=True))

    def test_full_resolution_is_the_default(self):
        command = _capture_command()
        self.assertEqual(command[command.index("--data-factor") + 1], "1")

    def test_ply_and_viewer_flags_are_always_set(self):
        command = _capture_command()
        self.assertIn("--save-ply", command)
        self.assertIn("--disable-viewer", command)

    def test_the_iteration_steps_all_match(self):
        command = _capture_command(iterations=7000)
        for flag in ("--max-steps", "--save-steps", "--eval-steps", "--ply-steps"):
            self.assertEqual(command[command.index(flag) + 1], "7000", flag)

    def test_the_holdout_uses_test_every_not_a_no_eval_flag(self):
        """gsplat's tyro config has no --no-eval; the holdout is --test-every."""
        command = _capture_command(test_every=8)
        self.assertNotIn("--no-eval", command)
        self.assertEqual(command[command.index("--test-every") + 1], "8")

    def test_the_trainer_is_invoked_as_a_subprocess_of_the_named_interpreter(self):
        command = _capture_command(python_executable=sys.executable)
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[2], "default")


class ContractTests(unittest.TestCase):
    """What ComfyUI itself enforces."""

    def test_train_signature_covers_every_declared_widget(self):
        spec = module.GsplatHeadlessTrainer.INPUT_TYPES()
        expected = set(spec["required"]) | set(spec.get("optional", {}))
        actual = set(inspect.signature(
            module.GsplatHeadlessTrainer.train).parameters) - {"self"}
        self.assertEqual(expected, actual)

    def test_outputs_match_the_lichtfeld_trainer(self):
        """Drop-in: a workflow can swap the node without rewiring downstream."""
        node = module.GsplatHeadlessTrainer
        self.assertEqual(node.RETURN_TYPES, ("STRING", "STRING", "STRING", "STRING"))
        self.assertEqual(node.RETURN_NAMES,
                         ("output_path", "command", "log_file", "summary"))
        self.assertEqual(node.FUNCTION, "train")
        self.assertEqual(node.CATEGORY, "Enndee/3D")

    def test_a_missing_dataset_is_reported_clearly(self):
        with self.assertRaises(ValueError):
            module.GsplatHeadlessTrainer().train(
                dataset_path=str(Path(tempfile.gettempdir()) / "definitely_not_here"),
                output_path="", output_name="", iterations=7000, sh_degree=3,
                data_factor=1, normalize_world_space=False, prepare_dataset=False,
                uniform_crop=False, max_image_size=1600, test_every=8,
                python_executable=sys.executable, trainer_script=str(Path(__file__)),
                overwrite_output=True)


class HelperTests(unittest.TestCase):
    def test_model_dir_is_found_in_the_usual_places(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sparse" / "0").mkdir(parents=True)
            (root / "sparse" / "0" / "images.bin").write_bytes(b"")
            self.assertEqual(module._find_model_dir(root), root / "sparse" / "0")

    def test_images_dir_is_found(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            self.assertEqual(module._find_images_dir(root), root / "images")

    def test_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(module._find_model_dir(Path(folder)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

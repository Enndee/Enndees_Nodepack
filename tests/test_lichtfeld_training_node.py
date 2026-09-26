"""Unit tests for the Lichtfeld headless trainer; no Studio training is run."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "nodes"))

from lichtfeld_training_node import (  # noqa: E402
    LichtfeldHeadlessTrainer,
    build_lfs_settings_script,
    build_training_command,
    parse_iteration_steps,
    run_streaming_command,
    resolve_studio_executable,
    resolve_training_output,
    validate_refinement_steps,
    validate_dataset,
)
import glomap_lichtfeld_node  # noqa: E402


def make_dataset(root):
    """Create a tiny structural-only dataset to exercise path validation."""
    root = Path(root)
    images = root / "images"
    masks = root / "masks"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    sparse.mkdir(parents=True)
    (images / "0001.png").write_bytes(b"not decoded by these tests")
    (masks / "0001.png").write_bytes(b"mask test fixture")
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("# test fixture\n", encoding="utf-8")
    return root


def default_command_options():
    return {
        "executable": Path(sys.executable),
        "dataset": Path("dataset with spaces"),
        "output": Path("output folder"),
        "log_file": Path("training log.txt"),
        "iterations": 30000,
        "strategy": "mcmc",
        "sh_degree": 3,
        "max_cap": 6000000,
        "steps_scaler": 1.0,
        "mask_mode": "segment",
        "invert_masks": False,
        "bg_mode": "solidcolor",
        "bg_color": "#FFFFFF",
        "enable_mip": False,
        "bilateral_grid": False,
        "enable_eval": False,
        "enable_sparsity": False,
        "log_level": "info",
        "output_name": "final_splat",
        "config_file": Path("studio config.json"),
        "centralize_dataset": "off",
        "resize_factor": "auto",
        "max_image_width": 3840,
        "disable_downscaling": False,
    }


class LichtfeldDatasetTests(unittest.TestCase):
    def test_resolves_legacy_studio_executable_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "LichtFeld-Studio.exe"
            executable.touch()

            resolved = resolve_studio_executable(
                Path(temp_dir) / "!LichtFeld-Studio.exe"
            )

            self.assertEqual(resolved, executable.resolve())

    def test_parses_and_validates_comma_separated_iteration_steps(self):
        self.assertEqual(
            parse_iteration_steps("15000, 5000,15000", "Save Steps", 30000),
            [5000, 15000],
        )
        self.assertIsNone(parse_iteration_steps("", "Save Steps", 30000))
        with self.assertRaisesRegex(ValueError, "between 1 and the total iterations"):
            parse_iteration_steps("30001", "Save Steps", 30000)
        with self.assertRaisesRegex(ValueError, "integer iteration numbers"):
            parse_iteration_steps("5000,late", "Save Steps", 30000)

    def test_validates_refinement_cutoffs_against_total_iterations(self):
        validate_refinement_steps(0, 30000, 30000)
        with self.assertRaisesRegex(ValueError, "cannot exceed total iterations"):
            validate_refinement_steps(30001, 0, 30000)

    def test_accepts_tracker_export_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            self.assertEqual(validate_dataset(dataset), dataset.resolve())

    def test_rejects_dataset_without_colmap_sparse_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            (dataset / "images").mkdir(parents=True)
            (dataset / "images" / "0001.jpg").touch()
            with self.assertRaisesRegex(ValueError, "sparse/0"):
                validate_dataset(dataset)

    def test_rejects_empty_dataset_path(self):
        with self.assertRaisesRegex(ValueError, "Connect a GLOMAP Tracker"):
            validate_dataset("")

    def test_empty_training_output_uses_timestamped_dataset_subfolder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            dataset.mkdir()

            output = resolve_training_output(dataset)

            self.assertEqual(output.parent, dataset.resolve() / "output")
            self.assertTrue(output.name.startswith("ComfyUI_Training_"))
            self.assertNotEqual(output, dataset.resolve())

    def test_dataset_root_widget_value_uses_timestamped_subfolder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            dataset.mkdir()

            output = resolve_training_output(dataset, str(dataset))

            self.assertEqual(output.parent, dataset.resolve() / "output")
            self.assertTrue(output.name.startswith("ComfyUI_Training_"))
            self.assertNotEqual(output, dataset.resolve())

    def test_relative_training_output_resolves_under_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            dataset.mkdir()

            self.assertEqual(
                resolve_training_output(dataset, "training/experiment"),
                dataset.resolve() / "training" / "experiment",
            )

    def test_mask_training_requires_a_lichtfeld_masks_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            (dataset / "masks" / "0001.png").unlink()
            (dataset / "masks").rmdir()
            inputs = {
                name: metadata.get("default", options[0] if isinstance(options, list) else None)
                for name, (options, metadata)
                in LichtfeldHeadlessTrainer.INPUT_TYPES()["required"].items()
            }
            inputs.update(
                studio_executable=sys.executable,
                dataset_path=str(dataset),
                preview_only=True,
            )
            with self.assertRaisesRegex(ValueError, "requires Lichtfeld mask PNGs"):
                LichtfeldHeadlessTrainer().train(**inputs)


class LichtfeldCommandTests(unittest.TestCase):
    def test_builds_headless_white_background_mask_command(self):
        command = build_training_command(**default_command_options())
        self.assertEqual(command[0], str(Path(sys.executable)))
        for flag in (
            "--headless", "--train", "--data-path", "--output-path", "--iter",
            "--strategy", "--sh-degree", "--max-cap", "--steps-scaler",
            "--mask-mode", "--bg-mode", "--bg-color", "--log-file",
            "--output-name", "--config",
            "--centralize=off", "--resize_factor=auto", "--max-width=3840",
        ):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--bg-color") + 1], "#FFFFFF")
        self.assertEqual(command[command.index("--mask-mode") + 1], "segment")
        self.assertEqual(command[command.index("--data-path") + 1], "dataset with spaces")
        self.assertEqual(command[command.index("--log-file") + 1], "training log.txt")
        self.assertEqual(command[command.index("--output-name") + 1], "final_splat")
        self.assertEqual(command[command.index("--config") + 1], "studio config.json")

    def test_command_centralizes_and_disables_downscaling(self):
        options = default_command_options()
        options.update(
            centralize_dataset="by_pointcloud",
            resize_factor="4",
            max_image_width=2048,
            disable_downscaling=True,
        )

        command = build_training_command(**options)

        self.assertIn("--centralize=by_pointcloud", command)
        self.assertIn("--resize_factor=1", command)
        self.assertIn("--max-width=0", command)
        self.assertNotIn("--resize_factor=4", command)
        self.assertNotIn("--max-width=2048", command)

    def test_rejects_invalid_dataset_scaling_options(self):
        options = default_command_options()
        options["centralize_dataset"] = "by_invalid"
        with self.assertRaisesRegex(ValueError, "centering mode"):
            build_training_command(**options)

        options = default_command_options()
        options["resize_factor"] = "3"
        with self.assertRaisesRegex(ValueError, "resize factor"):
            build_training_command(**options)

    def test_generated_settings_script_applies_refinement_save_and_eval_steps(self):
        source = build_lfs_settings_script(
            grow_until_iter=12000,
            stop_refine=20000,
            save_steps=[5000, 10000, 20000],
            eval_steps=[10000, 20000],
            enable_eval=True,
        )
        registered = []

        class MockParams:
            def __init__(self):
                self.values = {}
                self.save_steps = []
                self.eval_steps = []
                self.enable_eval = False

            def has_params(self):
                return True

            def set(self, name, value):
                self.values[name] = value

            def clear_save_steps(self):
                self.save_steps.clear()

            def add_save_step(self, step):
                self.save_steps.append(step)

            def clear_eval_steps(self):
                self.eval_steps.clear()

            def add_eval_step(self, step):
                self.eval_steps.append(step)

        params = MockParams()
        mock_lf = types.ModuleType("lichtfeld")
        mock_lf.optimization_params = lambda: params
        mock_lf.on_iteration_start = registered.append

        with mock.patch.dict(sys.modules, {"lichtfeld": mock_lf}):
            exec(compile(source, "<generated Lichtfeld settings>", "exec"), {})

        self.assertEqual(len(registered), 1)
        registered[0](object())
        registered[0](object())
        self.assertEqual(params.values["grow_until_iter"], 12000)
        self.assertEqual(params.values["stop_refine"], 20000)
        self.assertEqual(params.save_steps, [5000, 10000, 20000])
        self.assertEqual(params.eval_steps, [10000, 20000])
        self.assertTrue(params.enable_eval)

    def test_eval_without_explicit_eval_steps_mirrors_save_steps(self):
        source = build_lfs_settings_script(
            save_steps=[5000, 15000],
            enable_eval=True,
        )
        self.assertIn('"eval_steps": [5000, 15000]', source)

    def test_builds_lfs_settings_script_with_defaults_only_when_overridden(self):
        self.assertEqual(build_lfs_settings_script(), "")

    def test_appends_optional_training_flags_only_when_enabled(self):
        options = default_command_options()
        options.update(
            invert_masks=True,
            enable_mip=True,
            bilateral_grid=True,
            enable_eval=True,
            enable_sparsity=True,
        )
        command = build_training_command(**options)
        for flag in (
            "--invert-masks", "--enable-mip", "--bilateral-grid", "--eval",
            "--enable-sparsity",
        ):
            self.assertIn(flag, command)

    def test_rejects_invalid_strategy_and_iterations(self):
        options = default_command_options()
        options["strategy"] = "unknown"
        with self.assertRaisesRegex(ValueError, "optimization strategy"):
            build_training_command(**options)

        options = default_command_options()
        options["iterations"] = 0
        with self.assertRaisesRegex(ValueError, "at least 1"):
            build_training_command(**options)

    def test_preview_returns_validated_command_without_launching_studio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "preview output"
            inputs = {
                name: metadata.get("default", choices[0] if isinstance(choices, list) else None)
                for name, (choices, metadata)
                in LichtfeldHeadlessTrainer.INPUT_TYPES()["required"].items()
            }
            inputs.update(
                studio_executable=sys.executable,
                dataset_path=str(dataset),
                output_path=str(output),
                iterations=20000,
                centralize_dataset="by_pointcloud",
                disable_downscaling=True,
                grow_until_iter=8000,
                stop_refine=18000,
                save_steps="5000,10000,20000",
                eval_steps="10000,20000",
                preview_only=True,
            )
            result = LichtfeldHeadlessTrainer().train(**inputs)
            self.assertIn("--headless", result["result"][1])
            self.assertIn("--bg-color", result["result"][1])
            self.assertIn("--centralize=by_pointcloud", result["result"][1])
            self.assertIn("--resize_factor=1", result["result"][1])
            self.assertIn("--max-width=0", result["result"][1])
            self.assertIn("--python-script", result["result"][1])
            self.assertIn("<temporary Lichtfeld settings script>", result["result"][1])
            self.assertIn("Preview only", result["result"][3])
            self.assertFalse(output.exists())

    def test_preview_routes_dataset_root_output_to_a_new_output_subfolder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            inputs = {
                name: metadata.get("default", options[0] if isinstance(options, list) else None)
                for name, (options, metadata)
                in LichtfeldHeadlessTrainer.INPUT_TYPES()["required"].items()
            }
            inputs.update(
                studio_executable=sys.executable,
                dataset_path=str(dataset),
                output_path=str(dataset),
                preview_only=True,
            )

            result = LichtfeldHeadlessTrainer().train(**inputs)
            output = Path(result["result"][0])

            self.assertEqual(output.parent, dataset.resolve() / "output")
            self.assertTrue(output.name.startswith("ComfyUI_Training_"))
            self.assertFalse(output.exists())

    def test_refuses_to_start_beside_an_existing_studio_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "must remain untouched"
            inputs = {
                name: metadata.get("default", options[0] if isinstance(options, list) else None)
                for name, (options, metadata)
                in LichtfeldHeadlessTrainer.INPUT_TYPES()["required"].items()
            }
            inputs.update(
                studio_executable=sys.executable,
                dataset_path=str(dataset),
                output_path=str(output),
            )
            with mock.patch(
                "lichtfeld_training_node._check_running_studio_processes",
                return_value=[55540],
            ):
                with self.assertRaisesRegex(RuntimeError, r"PID\(s\): 55540"):
                    LichtfeldHeadlessTrainer().train(**inputs)
            self.assertFalse(output.exists())

    def test_refuses_to_create_training_output_inside_dataset_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            inputs = {
                name: metadata.get("default", options[0] if isinstance(options, list) else None)
                for name, (options, metadata)
                in LichtfeldHeadlessTrainer.INPUT_TYPES()["required"].items()
            }
            inputs.update(
                studio_executable=sys.executable,
                dataset_path=str(dataset),
                output_path=str(dataset / "images" / "training"),
                preview_only=True,
            )
            with self.assertRaisesRegex(ValueError, "protected data folders"):
                LichtfeldHeadlessTrainer().train(**inputs)


class LichtfeldStreamingTests(unittest.TestCase):
    def test_streams_child_output_and_returns_the_tail(self):
        messages = []
        code, tail = run_streaming_command(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; sys.stdout.write('mock training progress\\r'); "
                "sys.stdout.flush(); print('mock training finished')",
            ],
            PACK_DIR,
            progress_callback=messages.append,
        )
        self.assertEqual(code, 0)
        self.assertIn("mock training progress", tail)
        self.assertIn("mock training finished", tail)
        self.assertEqual(len(messages), 2)
        self.assertIn("mock training progress", messages[0])
        self.assertIn("mock training finished", messages[1])

    def test_reports_child_process_failure(self):
        code, tail = run_streaming_command(
            [sys.executable, "-u", "-c", "print('simulated trainer error'); raise SystemExit(7)"],
            PACK_DIR,
        )
        self.assertEqual(code, 7)
        self.assertIn("simulated trainer error", tail)


class ComfyNodeRegistrationTests(unittest.TestCase):
    def test_headless_trainer_registration_and_legacy_tracker_output_order(self):
        spec = importlib.util.spec_from_file_location(
            "enndee_nodepack_registration_test",
            PACK_DIR / "__init__.py",
            submodule_search_locations=[str(PACK_DIR)],
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        nodepack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nodepack)

        self.assertIs(
            nodepack.NODE_CLASS_MAPPINGS["Enndee_LichtfeldHeadlessTrainer"],
            LichtfeldHeadlessTrainer,
        )
        self.assertEqual(
            glomap_lichtfeld_node.GLOMAPLichtfeldTracker.RETURN_NAMES,
            ("trajectory", "point_cloud", "confidence", "dataset_path"),
        )
        self.assertEqual(
            len(glomap_lichtfeld_node.GLOMAPLichtfeldTracker._empty(1)), 4
        )


if __name__ == "__main__":
    unittest.main()
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
    build_conversion_command,
    build_lfs_settings_script,
    build_training_command,
    check_studio_choice,
    filter_supported_flags,
    parse_iteration_steps,
    parse_studio_capabilities,
    probe_studio_support,
    resolve_export_support,
    run_streaming_command,
    resolve_studio_executable,
    resolve_trained_splat,
    resolve_training_output,
    validate_export_format,
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


def capable_studio(**overrides):
    """Probed capabilities of a current LichtFeld Studio build (for mocked probes).

    `flags`/`value_flags`/choice lists stay None on purpose: a mocked build then filters and
    validates nothing, which keeps these tests focused on the export/fallback logic.
    """
    support = {
        "version": "v0.5.3 (d8c50c6a)",
        "flags": None,
        "value_flags": None,
        "convert": True,
        "formats": frozenset({"ply", "sog", "spz", "html"}),
        "strategies": None,
        "mask_modes": None,
        "log_levels": None,
    }
    support.update(overrides)
    return support


OLD_FLAGS = frozenset({
    "--data-path", "--output-path", "--iter", "--strategy", "--sh-degree", "--max-cap",
    "--steps-scaler", "--mask-mode", "--log-level", "--log-file", "--resize_factor",
    "--max-width", "--headless", "--train", "--invert-masks", "--enable-mip",
    "--bilateral-grid", "--eval", "--enable-sparsity", "--config", "--python-script",
})
OLD_VALUE_FLAGS = frozenset({
    "--data-path", "--output-path", "--iter", "--strategy", "--sh-degree", "--max-cap",
    "--steps-scaler", "--mask-mode", "--log-level", "--log-file", "--config",
    "--python-script",
})


def old_studio(**overrides):
    """Probed capabilities of an older free build: no bg-mode/bg-color/centralize/output-name,
    a shorter strategy list, and `--version` printing "unknown"."""
    support = capable_studio(
        version="unknown (bdd8f92)",
        convert=False,
        formats=frozenset({"ply"}),
        flags=OLD_FLAGS,
        value_flags=OLD_VALUE_FLAGS,
        strategies=frozenset({"mcmc", "adc", "igs+"}),
        mask_modes=frozenset({"none", "segment", "ignore", "alpha_consistent"}),
        log_levels=frozenset({"trace", "debug", "info", "perf", "warn", "error",
                              "critical", "off"}),
    )
    support.update(overrides)
    return support


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

    def test_validates_splat_export_formats(self):
        self.assertEqual(validate_export_format("SOG"), "sog")
        self.assertEqual(validate_export_format(".Spz"), "spz")
        self.assertEqual(validate_export_format(""), "ply")
        with self.assertRaisesRegex(ValueError, "Unsupported splat export format"):
            validate_export_format("glb")

    def test_resolve_trained_splat_prefers_the_named_stem_then_the_last_iteration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "splat_5000.ply").write_bytes(b"checkpoint")
            (folder / "splat_30000.ply").write_bytes(b"final")
            (folder / "final_splat.ply").write_bytes(b"named")

            self.assertEqual(resolve_trained_splat(folder).name, "splat_30000.ply")
            self.assertEqual(
                resolve_trained_splat(folder, "final_splat").name, "final_splat.ply"
            )
            self.assertIsNone(resolve_trained_splat(folder / "missing"))

    def test_builds_the_studio_convert_command(self):
        command = build_conversion_command(Path("Studio.exe"), Path("in.ply"), Path("out.sog"))
        self.assertEqual(
            command, ["Studio.exe", "convert", "in.ply", "out.sog", "--overwrite"]
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


    def test_train_converts_the_finished_splat_to_the_chosen_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "trained output"
            calls = []

            def fake_run(command, cwd, progress_callback=None):
                calls.append(list(command))
                if len(calls) == 1:              # simulate Studio's splat_ITER.ply export
                    (output / "splat_20000.ply").write_bytes(b"synthetic splat")
                else:                            # simulate Studio's convert writing the target
                    Path(command[3]).write_bytes(b"synthetic sog")
                return 0, "mock training output"

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
                export_format="sog",
            )
            with mock.patch(
                "lichtfeld_training_node._check_running_studio_processes",
                return_value=[],
            ), mock.patch(
                "lichtfeld_training_node.run_streaming_command",
                side_effect=fake_run,
            ), mock.patch(
                "lichtfeld_training_node.probe_studio_support",
                return_value=capable_studio(),
            ):
                result = LichtfeldHeadlessTrainer().train(**inputs)

            self.assertEqual(len(calls), 2)
            self.assertIn("--train", calls[0])
            self.assertEqual(calls[1][1], "convert")
            self.assertEqual(calls[1][2], str(output / "splat_20000.ply"))
            self.assertEqual(calls[1][3], str(output / "splat_20000.sog"))
            self.assertIn("--overwrite", calls[1])
            self.assertEqual(result["result"][0], str(output))
            self.assertIn("splat_20000.sog", result["result"][3])

    def test_train_keeps_the_plain_ply_export_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "trained output"
            calls = []

            def fake_run(command, cwd, progress_callback=None):
                calls.append(list(command))
                return 0, "mock training output"

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
            )
            with mock.patch(
                "lichtfeld_training_node._check_running_studio_processes",
                return_value=[],
            ), mock.patch(
                "lichtfeld_training_node.run_streaming_command",
                side_effect=fake_run,
            ):
                result = LichtfeldHeadlessTrainer().train(**inputs)

            self.assertEqual(len(calls), 1)
            self.assertNotIn("convert", calls[0])
            self.assertNotIn("Splat:", result["result"][3])

    def test_train_reports_a_failed_splat_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "trained output"
            calls = []

            def fake_run(command, cwd, progress_callback=None):
                calls.append(list(command))
                if len(calls) == 1:
                    (output / "splat_20000.ply").write_bytes(b"synthetic splat")
                    return 0, "mock training output"
                return 3, "convert exploded"

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
                export_format="spz",
            )
            with mock.patch(
                "lichtfeld_training_node._check_running_studio_processes",
                return_value=[],
            ), mock.patch(
                "lichtfeld_training_node.run_streaming_command",
                side_effect=fake_run,
            ), mock.patch(
                "lichtfeld_training_node.probe_studio_support",
                return_value=capable_studio(),
            ):
                with self.assertRaisesRegex(RuntimeError, r"\.spz conversion exited with code 3"):
                    LichtfeldHeadlessTrainer().train(**inputs)

    def test_preview_mentions_the_requested_export(self):
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
                export_format="spz",
                preview_only=True,
            )
            with mock.patch(
                "lichtfeld_training_node.probe_studio_support",
                return_value=capable_studio(),
            ):
                result = LichtfeldHeadlessTrainer().train(**inputs)
            self.assertIn(".spz export would follow", result["result"][3])
            self.assertFalse(output.exists())


OLD_HELP_SNIPPET = """  LichtFeld-Studio.exe {OPTIONS}
        -i[iterations], --iter=[iterations]
                                          Number of iterations
        --strategy=[strategy]             Optimization strategy: mcmc, adc, igs+
        --max-cap=[max_cap]               Max Gaussians for MCMC or igs+
        -o[output_path], --output-path=[output_path]
                                          Path to output
        -r[resize_factor], --resize_factor=[resize_factor]
                                          Resize resolution by factor: auto, 1, 2, 4, 8 (default: auto)
        --mask-mode=[mask_mode]           Mask mode: none, segment, ignore, alpha_consistent (default: none)
        --log-level=[level]               Log level: trace, debug, info, perf, warn, error, critical, off
        -q, --quiet                       Suppress non-error output (equivalent to --log-level error)
        --headless                        Disable visualization during training
        --train                           Start training immediately on startup
    SUBCOMMANDS:
    convert -- Convert between .ply, .sog, .spz, .html
"""

NEW_HELP_SNIPPET = """  LichtFeld-Studio.exe {OPTIONS}
        --output-name=[output_name]       Output filename (replaces default splat_ITER.ply stem)
        --strategy=[strategy]             Optimization strategy: mcmc, mrnf, igs+ (legacy aliases: mnrf, lfs)
        --bg-mode=[mode]                  Background mode: solidcolor, modulation, image, random (default: solidcolor)
        --bg-color=[color]                solidcolor background color as #RRGGBB (default: #000000)
        --mask-mode=[mask_mode]           Mask mode: none, segment, ignore, segment_and_ignore, alpha_consistent
        --centralize=[centralize]         Dataset origin: off, by_pointcloud, by_cameras
    SUBCOMMANDS:
    convert -- Convert between .ply, .sog, .spz, .usd/.usda/.usdc, .html
    mesh2splat -- Convert a mesh file to Gaussian splats
"""

NEW_CONVERT_SNIPPET = """  LichtFeld-Studio.exe convert {OPTIONS} [input] [output]
      -f[format], --format=[format]     Output format: ply, sog, spz, html, usd, usda, usdc, rad
    SUPPORTED FORMATS:
      Input: .ply, .sog, .spz, .usd, .resume (checkpoint)
      Output: .ply, .sog, .spz, .usd, .usda, .usdc, .html, .rad
"""


class LichtfeldStudioCompatibilityTests(unittest.TestCase):
    """The <0.5.3 fallbacks: capabilities are probed, never assumed from a version number."""

    def test_parse_studio_capabilities_reads_old_and_new_help(self):
        old = parse_studio_capabilities(OLD_HELP_SNIPPET)
        self.assertTrue(old["convert"])                      # 0.5.0 already ships convert
        self.assertLessEqual({"ply", "sog", "spz"}, old["formats"])
        self.assertNotIn("--bg-mode", old["flags"])          # flags that abort old builds
        self.assertNotIn("--centralize", old["flags"])
        self.assertNotIn("--output-name", old["flags"])
        self.assertIn("--resize_factor", old["flags"])
        self.assertIn("--resize_factor", old["value_flags"])
        self.assertIn("--iter", old["value_flags"])
        self.assertNotIn("--headless", old["value_flags"])   # booleans take no value
        self.assertEqual(old["strategies"], frozenset({"mcmc", "adc", "igs+"}))
        self.assertEqual(old["mask_modes"],
                         frozenset({"none", "segment", "ignore", "alpha_consistent"}))
        self.assertIn("off", old["log_levels"])

        new = parse_studio_capabilities(NEW_HELP_SNIPPET, NEW_CONVERT_SNIPPET)
        self.assertTrue(new["convert"])
        self.assertLessEqual({"ply", "sog", "spz", "rad"}, new["formats"])
        self.assertIn("--bg-mode", new["flags"])
        self.assertIn("--centralize", new["flags"])
        self.assertIn("--output-name", new["value_flags"])
        self.assertEqual(new["strategies"], frozenset({"mcmc", "mrnf", "igs+"}))
        self.assertIn("segment_and_ignore", new["mask_modes"])

        # the probe may hand a convert help dump over as the single text argument
        convert_only = parse_studio_capabilities(NEW_CONVERT_SNIPPET)
        self.assertTrue(convert_only["convert"])
        self.assertLessEqual({"ply", "sog", "spz", "rad", "usd", "usda", "usdc", "html"},
                             convert_only["formats"])

    def test_probe_studio_support_reports_unknown_for_a_non_lichtfeld_binary(self):
        # The test interpreter answers --version/--help but never identifies as LichtFeld, so
        # nothing is filtered and nothing is promised (the wrong-executable case).
        support = probe_studio_support(sys.executable)
        self.assertEqual(support["version"], "")
        self.assertFalse(support["convert"])
        self.assertEqual(support["formats"], frozenset({"ply"}))
        self.assertIsNone(support["flags"])
        self.assertIsNone(support["strategies"])

    def test_filter_supported_flags_drops_unknown_flags_with_their_values(self):
        command = ["Studio.exe", "--data-path", "dataset", "--iter", "20000",
                   "--bg-mode", "solidcolor", "--bg-color", "#FFFFFF", "--centralize=off",
                   "--output-name", "final_splat", "--headless", "--train"]
        filtered, dropped = filter_supported_flags(
            command, {"--data-path", "--iter", "--headless", "--train"},
        )
        self.assertEqual(
            filtered,
            ["Studio.exe", "--data-path", "dataset", "--iter", "20000", "--headless", "--train"],
        )
        self.assertEqual(sorted(dropped),
                         ["--bg-color", "--bg-mode", "--centralize", "--output-name"])
        untouched, nothing_dropped = filter_supported_flags(command, None)
        self.assertEqual(untouched, command)
        self.assertEqual(nothing_dropped, [])

    def test_resolve_export_support_falls_back_to_ply_on_old_builds(self):
        self.assertEqual(resolve_export_support(old_studio(), "ply"), ("ply", ""))
        export_format, note = resolve_export_support(old_studio(), "sog")
        self.assertEqual(export_format, "ply")
        self.assertIn("has no `convert` subcommand", note)
        self.assertIn("unknown (bdd8f92)", note)

        partial = capable_studio(formats=frozenset({"ply", "sog", "html"}))
        self.assertEqual(resolve_export_support(partial, "sog"), ("sog", ""))
        export_format, note = resolve_export_support(partial, "spz")
        self.assertEqual(export_format, "ply")
        self.assertIn("cannot write .spz", note)

    def test_check_studio_choice_raises_or_falls_back_against_the_builds_list(self):
        strategies = frozenset({"mcmc", "adc", "igs+"})
        self.assertEqual(check_studio_choice("mcmc", strategies, "strategy"), ("mcmc", ""))
        with self.assertRaisesRegex(ValueError, "does not support strategy 'mrnf'"):
            check_studio_choice("mrnf", strategies, "strategy")
        self.assertEqual(check_studio_choice("mrnf", None, "strategy"), ("mrnf", ""))
        level, note = check_studio_choice("trace", frozenset({"info", "warn"}), "log level",
                                          fallback="info")
        self.assertEqual(level, "info")
        self.assertIn("using 'info'", note)

    def test_train_drops_flags_an_old_build_rejects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "trained output"
            calls = []

            def fake_run(command, cwd, progress_callback=None):
                calls.append(list(command))
                return 0, "mock training output"

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
                output_name="final_splat",
            )
            with mock.patch(
                "lichtfeld_training_node._check_running_studio_processes",
                return_value=[],
            ), mock.patch(
                "lichtfeld_training_node.run_streaming_command",
                side_effect=fake_run,
            ), mock.patch(
                "lichtfeld_training_node.probe_studio_support",
                return_value=old_studio(),
            ):
                result = LichtfeldHeadlessTrainer().train(**inputs)

            command = calls[0]
            for flag in ("--bg-mode", "--bg-color", "--centralize", "--output-name"):
                self.assertNotIn(flag, command)
            self.assertNotIn("final_splat", command)          # the value left with its flag
            self.assertIn("--mask-mode", command)
            self.assertIn("--resize_factor=auto", command)
            self.assertIn("does not support", result["result"][3])

    def test_train_falls_back_to_ply_when_the_build_cannot_convert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            output = Path(temp_dir) / "trained output"
            calls = []

            def fake_run(command, cwd, progress_callback=None):
                calls.append(list(command))
                if len(calls) == 1:
                    (output / "splat_20000.ply").write_bytes(b"synthetic splat")
                return 0, "mock training output"

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
                export_format="sog",
            )
            with mock.patch(
                "lichtfeld_training_node._check_running_studio_processes",
                return_value=[],
            ), mock.patch(
                "lichtfeld_training_node.run_streaming_command",
                side_effect=fake_run,
            ), mock.patch(
                "lichtfeld_training_node.probe_studio_support",
                return_value=old_studio(),
            ):
                result = LichtfeldHeadlessTrainer().train(**inputs)

            self.assertEqual(len(calls), 1)                    # training only, no convert
            self.assertNotIn("convert", calls[0])
            self.assertIn("exporting .ply instead", result["result"][3])
            self.assertNotIn("Splat:", result["result"][3])

    def test_train_raises_for_a_strategy_the_build_lacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            inputs = {
                name: metadata.get("default", choices[0] if isinstance(choices, list) else None)
                for name, (choices, metadata)
                in LichtfeldHeadlessTrainer.INPUT_TYPES()["required"].items()
            }
            inputs.update(
                studio_executable=sys.executable,
                dataset_path=str(dataset),
                strategy="mrnf",
                preview_only=True,
            )
            with mock.patch(
                "lichtfeld_training_node.probe_studio_support",
                return_value=old_studio(),
            ):
                with self.assertRaisesRegex(ValueError, "does not support strategy 'mrnf'"):
                    LichtfeldHeadlessTrainer().train(**inputs)

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
"""Unit tests for the Lichtfeld headless trainer; no Studio training is run."""

import importlib.util
import json
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
    MASK_OPACITY_PENALTIES,
    EVALUATION_SPLITS,
    SUBJECT_MODES,
    _LFS_VALUE_FLAGS,
    ANCHOR_MESH_NAMES,
    ANCHOR_SIGMA,
    ANCHOR_WORK_DIR,
    UNDISTORT_MODES,
    dataset_camera_models,
    dataset_needs_undistort,
    resolve_depth_loss_mode,
    LichtfeldHeadlessTrainer,
    build_conversion_command,
    build_lfs_optimization_section,
    build_mesh2splat_command,
    default_eval_steps,
    build_lfs_settings_script,
    build_training_command,
    check_studio_choice,
    describe_lfs_settings_status,
    filter_supported_flags,
    load_lfs_optimization_template,
    parse_iteration_steps,
    parse_studio_capabilities,
    prepare_surface_anchors,
    probe_studio_support,
    read_lfs_settings_status,
    resolve_anchor_mesh,
    resolve_export_support,
    write_lfs_config_file,
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

    def test_depth_loss_downgrades_to_a_warning_when_the_dataset_has_no_depth(self):
        # The depth loss is ON by default (it measured better), so a dataset without a
        # depth/ folder - a GLOMAP-only one - must NOT fail the run. The node warns, drops
        # the depth term and continues, matching its "adapt, don't fail" pattern (compare
        # resolve_export_support). make_dataset() deliberately creates no depth folder.
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
                preview_only=True,
            )
            self.assertTrue(inputs["use_depth_loss"])
            self.assertFalse((dataset / "depth").exists())

            result = LichtfeldHeadlessTrainer().train(**inputs)

            summary = result["result"][3]
            self.assertIn("depth loss skipped", summary)

    def test_depth_loss_is_kept_when_the_dataset_carries_depth_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = make_dataset(Path(temp_dir) / "dataset")
            (dataset / "depth").mkdir()
            (dataset / "depth" / "0001.depth.png").write_bytes(b"depth test fixture")
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
            result = LichtfeldHeadlessTrainer().train(**inputs)
            self.assertNotIn("depth loss skipped", result["result"][3])


class LichtfeldCommandTests(unittest.TestCase):
    def test_builds_headless_white_background_mask_command(self):
        command = build_training_command(**default_command_options())
        self.assertEqual(command[0], str(Path(sys.executable)))
        for flag in (
            "--headless", "--train", "--data-path", "--output-path", "--iter",
            "--strategy", "--sh-degree", "--max-cap",
            "--mask-mode", "--bg-mode", "--bg-color", "--log-file",
            "--output-name", "--config",
            "--centralize=off", "--resize_factor=auto", "--max-width=3840",
        ):
            self.assertIn(flag, command)
        # 0.5.4 made --iter and --steps-scaler mutually exclusive, so the 1.0 no-op scaler
        # is never sent (see test_iter_and_steps_scaler_are_never_sent_together).
        self.assertNotIn("--steps-scaler", command)
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

    def _run_settings_hook(self, source, params, *, calls=1):
        """Run the generated hook against a mock lichtfeld; fire every callback `calls` times."""
        registered = []
        mock_lf = types.ModuleType("lichtfeld")
        mock_lf.optimization_params = lambda: params
        mock_lf.on_training_start = registered.append
        mock_lf.on_iteration_start = registered.append
        printed = []
        with mock.patch.dict(sys.modules, {"lichtfeld": mock_lf}), mock.patch(
            "builtins.print", lambda *args, **kwargs: printed.append(" ".join(map(str, args)))
        ):
            exec(compile(source, "<generated Lichtfeld settings>", "exec"), {})
            for callback in registered:
                for _ in range(calls):
                    callback(object())
        return registered, printed

    def test_settings_hook_never_raises_when_parameters_are_unavailable(self):
        # Headless Studio hands out a parameter object whose has_params() is False
        # (the GUI ParameterManager does not exist); Studio logs a traceback for every
        # single iteration if the hook raises, which floods the console.
        source = build_lfs_settings_script(grow_until_iter=12000, save_steps=[5000])

        class HeadlessParams:
            def __init__(self):
                self.calls = []

            def has_params(self):
                return False

            def set(self, name, value):
                self.calls.append((name, value))

            def clear_save_steps(self):
                self.calls.append("clear_save_steps")

            def add_save_step(self, step):
                self.calls.append(("add_save_step", step))

        params = HeadlessParams()
        registered, printed = self._run_settings_hook(source, params, calls=2)
        self.assertEqual(len(registered), 2)
        warnings = [line for line in printed if "does not expose optimization parameters" in line]
        self.assertEqual(len(warnings), 1)
        # best effort: the writes are still attempted once, they just do not reach the trainer
        self.assertEqual(params.calls.count("clear_save_steps"), 1)

    def test_settings_hook_writes_one_status_report_for_the_node_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "status.json"
            source = build_lfs_settings_script(
                grow_until_iter=12000, save_steps=[5000], status_path=str(status_path)
            )

            class HeadlessParams:
                def __init__(self):
                    self.values = {}

                def has_params(self):
                    return False

                def set(self, name, value):
                    self.values[name] = value

                def clear_save_steps(self):
                    self.values["save_steps"] = []

                def add_save_step(self, step):
                    self.values["save_steps"].append(step)

            params = HeadlessParams()
            self._run_settings_hook(source, params, calls=1)

            self.assertEqual(params.values["grow_until_iter"], 12000)
            self.assertEqual(params.values["save_steps"], [5000])
            payload = read_lfs_settings_status(status_path)
            self.assertEqual(
                payload, {"applied": True, "has_params": False, "reason": ""}
            )
            self.assertIn(
                "does not expose optimization parameters",
                describe_lfs_settings_status(payload),
            )

    def test_settings_hook_reports_a_failing_hook_to_the_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "status.json"
            source = build_lfs_settings_script(save_steps=[5000], status_path=str(status_path))

            class ExplodingParams:
                def has_params(self):
                    return True

                def clear_save_steps(self):
                    raise RuntimeError("studio says no")

            registered, printed = self._run_settings_hook(source, ExplodingParams(), calls=1)

            payload = read_lfs_settings_status(status_path)
            self.assertFalse(payload["applied"])
            self.assertIn("studio says no", payload["reason"])
            self.assertIn("hook could not be applied", describe_lfs_settings_status(payload))
            self.assertTrue(
                any("applying the Lichtfeld training settings failed" in line for line in printed)
            )

    def test_describe_lfs_settings_status_is_silent_when_the_hook_worked(self):
        self.assertEqual(describe_lfs_settings_status(None), "")
        self.assertEqual(read_lfs_settings_status(""), None)
        self.assertEqual(
            describe_lfs_settings_status({"applied": True, "has_params": True, "reason": ""}),
            "",
        )

    def test_optimization_template_carries_every_studio_key(self):
        # Studio's config parser requires the complete section (verified against 0.5.3).
        section = load_lfs_optimization_template()
        self.assertTrue(section)
        for key in ("iterations", "strategy", "stop_refine", "grow_until_iter", "save_steps",
                    "eval_steps", "enable_eval", "mask_mode", "bg_mode"):
            self.assertIn(key, section)
        self.assertIsInstance(section["iterations"], int)
        self.assertIsInstance(section["means_lr"], float)
        # Studio stores these two enums as strings in the config file, not as numbers
        self.assertIsInstance(section["mask_mode"], str)
        self.assertIsInstance(section["bg_mode"], str)

    def test_build_lfs_optimization_section_merges_the_requested_settings(self):
        self.assertEqual(build_lfs_optimization_section(), {})
        section = build_lfs_optimization_section(
            grow_until_iter=12000, stop_refine=20000, save_steps=[5000, 10000],
            enable_eval=True, mask_mode="segment", bg_mode="solidcolor",
        )
        self.assertEqual(section["grow_until_iter"], 12000)
        self.assertEqual(section["stop_refine"], 20000)
        self.assertEqual(section["save_steps"], [5000, 10000])
        self.assertEqual(section["eval_steps"], [5000, 10000])     # mirrors the save steps
        self.assertTrue(section["enable_eval"])
        self.assertEqual(section["mask_mode"], "segment")
        self.assertEqual(section["bg_mode"], "solidcolor")
        self.assertIn("iterations", section)                       # template keys survive

    def test_build_lfs_optimization_section_applies_the_depth_loss_settings(self):
        # off by default: Studio's own use_depth_loss=false from the template stands
        self.assertEqual(build_lfs_optimization_section(), {})

        section = build_lfs_optimization_section(
            use_depth_loss=True, depth_loss_mode="pearson", depth_loss_weight=3.5)
        self.assertTrue(section["use_depth_loss"])
        self.assertEqual(section["depth_loss_mode"], "pearson")
        self.assertEqual(section["depth_loss_weight"], 3.5)

        # mode / weight are only overridden when they were actually requested
        plain = build_lfs_optimization_section(use_depth_loss=True)
        self.assertTrue(plain["use_depth_loss"])
        self.assertEqual(plain["depth_loss_mode"], "adaptive-warped-l1")   # template default
        self.assertEqual(plain["depth_loss_weight"], 2.0)                  # template default

    def test_depth_loss_widgets_are_declared_and_covered_by_the_signature(self):
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        for name in ("use_depth_loss", "depth_loss_mode", "depth_loss_weight"):
            self.assertIn(name, spec, name)
        self.assertEqual(list(spec["depth_loss_mode"][0]),
                         ["adaptive-warped-l1", "pearson"])
        self.assertEqual(spec["depth_loss_mode"][1]["default"], "adaptive-warped-l1")
        # These are the MEASURED winner of the subject-mode A/B, not Studio's defaults:
        # depth supervision scored 12.91 vs 12.66 dB held out at 6000 iterations, and
        # weight 8 only overtakes weight 2 after ~6000 iterations. Both are documented in
        # the widget tooltips and in the README section "Tuned defaults: the two subject
        # scenarios". Do not reset them to 2.0 / False without re-running that A/B.
        self.assertEqual(spec["depth_loss_weight"][1]["default"], 8.0)
        self.assertEqual(spec["use_depth_loss"][1]["default"], True)

        import inspect

        parameters = set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters)
        for name in ("use_depth_loss", "depth_loss_mode", "depth_loss_weight"):
            self.assertIn(name, parameters, name)

    def test_mask_opacity_penalty_maps_to_the_studio_weight(self):
        # the default must NOT override Studio's own value, so the section stays empty
        self.assertEqual(build_lfs_optimization_section(), {})
        self.assertEqual(build_lfs_optimization_section(mask_opacity_penalty="studio default"), {})

        for label, expected in (("off", 0.0), ("low", 1.0), ("medium", 5.0), ("high", 15.0)):
            section = build_lfs_optimization_section(mask_opacity_penalty=label)
            self.assertEqual(section["mask_opacity_penalty_weight"], expected, label)

        # an unknown label leaves the value alone instead of guessing
        self.assertEqual(build_lfs_optimization_section(mask_opacity_penalty="nonsense"), {})

    def test_mask_priority_widgets_are_declared_and_covered_by_the_signature(self):
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        self.assertIn("mask_opacity_penalty", spec)
        self.assertEqual(list(spec["mask_opacity_penalty"][0]), list(MASK_OPACITY_PENALTIES))
        self.assertEqual(spec["mask_opacity_penalty"][1]["default"], "studio default")
        self.assertEqual(spec["mask_mode"][1]["default"], "segment")
        self.assertEqual(list(spec["mask_mode"][0]),
                         ["none", "segment", "ignore", "segment_and_ignore", "alpha_consistent"])

        import inspect

        self.assertIn("mask_opacity_penalty",
                      set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters))

    def test_mask_penalty_and_mask_mode_combine_in_one_section(self):
        section = build_lfs_optimization_section(
            mask_mode="segment", mask_opacity_penalty="medium", use_depth_loss=True)
        self.assertEqual(section["mask_mode"], "segment")
        self.assertEqual(section["mask_opacity_penalty_weight"], 5.0)
        self.assertTrue(section["use_depth_loss"])

    def test_geometry_consistency_knobs_land_in_the_section(self):
        # None everywhere -> nothing requested -> the section collapses to {}
        self.assertEqual(build_lfs_optimization_section(), {})

        section = build_lfs_optimization_section(
            pause_refine_after_reset=200, scale_reg=0.02, scale_decay=0.004,
            lambda_dssim=0.3, grad_threshold=0.0003, growth_grad_threshold=0.004,
            prune_opacity=0.01, prune_scale2d=0.2, prune_scale3d=0.05,
            ppisp=False, bg_modulation=False,
        )
        # Studio's schema keeps int and float strictly apart
        self.assertIsInstance(section["pause_refine_after_reset"], int)
        self.assertEqual(section["pause_refine_after_reset"], 200)
        for key, expected in (("scale_reg", 0.02), ("scale_decay", 0.004),
                              ("lambda_dssim", 0.3), ("grad_threshold", 0.0003),
                              ("growth_grad_threshold", 0.004), ("prune_opacity", 0.01),
                              ("prune_scale2d", 0.2), ("prune_scale3d", 0.05)):
            self.assertIsInstance(section[key], float, key)
            self.assertAlmostEqual(section[key], expected, places=6, msg=key)
        self.assertIs(section["ppisp"], False)
        self.assertIs(section["bg_modulation"], False)

    def test_geometry_consistency_widgets_are_declared_and_in_the_signature(self):
        names = ("pause_refine_after_reset", "scale_reg", "scale_decay", "lambda_dssim",
                 "grad_threshold", "growth_grad_threshold", "prune_opacity",
                 "prune_scale2d", "prune_scale3d", "ppisp", "bg_modulation",
                 "evaluation", "save_eval_images")
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        for name in names:
            self.assertIn(name, spec, name)

        import inspect

        parameters = set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters)
        for name in names:
            self.assertIn(name, parameters, name)

        # defaults mirror the MEASURED subject-mode winner, not Studio's own values - see
        # the README section "Tuned defaults: the two subject scenarios" and the widget
        # tooltips. The three below were the anti-inflation cluster that raised held-out
        # PSNR *and* SSIM together (+0.152 dB / +0.020 over the same run without them).
        self.assertEqual(spec["evaluation"][1]["default"], "off")
        self.assertEqual(list(spec["evaluation"][0]), ["off", "1/2", "1/3", "1/4"])
        self.assertEqual(spec["save_eval_images"][1]["default"], False)
        self.assertEqual(spec["pause_refine_after_reset"][1]["default"], 200)
        self.assertAlmostEqual(spec["scale_reg"][1]["default"], 0.03)
        self.assertAlmostEqual(spec["lambda_dssim"][1]["default"], 0.3)
        self.assertAlmostEqual(spec["scale_decay"][1]["default"], 0.002)
        self.assertEqual(spec["ppisp"][1]["default"], False)
        self.assertEqual(spec["bg_modulation"][1]["default"], False)
        # The cap is a RASTERIZER limit: above ~1.9M primitives the FastGS int32
        # (primitive x tile) counter overflows and the run aborts.
        self.assertEqual(spec["max_gaussians"][1]["default"], 1000000)

    def test_subject_mode_presets_are_declared_and_land_in_the_section(self):
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        self.assertIn("subject_mode", spec)
        self.assertEqual(list(spec["subject_mode"][0]), list(SUBJECT_MODES))
        self.assertEqual(spec["subject_mode"][1]["default"],
                         "subject priority (background kept)")

        import inspect

        self.assertIn("subject_mode",
                      set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters))

        # Both presets keep the mask AND keep the opacity penalty on: 0.0 was measured to
        # crash the rasterizer (2,424 tiles per splat), and mask_mode='none' crashed at
        # 2,261. Only `segment` + penalty >= 1.0 survives.
        soft = SUBJECT_MODES["subject priority (background kept)"]
        hard = SUBJECT_MODES["subject cut-out (background removed)"]
        for preset in (soft, hard):
            self.assertEqual(preset["mask_mode"], "segment")
            self.assertGreaterEqual(MASK_OPACITY_PENALTIES[preset["mask_opacity_penalty"]], 1.0)
        self.assertLess(MASK_OPACITY_PENALTIES[soft["mask_opacity_penalty"]],
                        MASK_OPACITY_PENALTIES[hard["mask_opacity_penalty"]])
        self.assertIsNone(SUBJECT_MODES["custom (use the widgets below)"])

        section = build_lfs_optimization_section(
            mask_mode=soft["mask_mode"],
            mask_opacity_penalty=soft["mask_opacity_penalty"],
        )
        self.assertEqual(section["mask_mode"], "segment")
        self.assertEqual(section["mask_opacity_penalty_weight"], 1.0)

    def test_subject_mode_is_the_last_widget_so_saved_workflows_keep_their_mapping(self):
        # ComfyUI maps a saved workflow's widget_values by INSERTION ORDER, so appending a
        # new widget in the middle would silently remap every later widget of existing
        # workflows. Any new control must therefore go last.
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        names = list(spec)
        # the 0.5.4 supervision block is the newest addition, so it owns the tail;
        # subject_mode is the newest control before the anchor block and must precede it.
        self.assertEqual(names[-7:], ["undistort_cameras", "use_normal_loss",
                                      "normal_loss_weight", "normal_consistency_weight",
                                      "normal_flatten_weight", "normal_loss_space",
                                      "freeze_lr_scale"])
        self.assertEqual(names[names.index("subject_mode") + 1], "use_surface_anchors")

    # ---------------------------------------------------------------- surface anchors
    def test_surface_anchor_widgets_are_declared_and_in_the_signature(self):
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        names = ("use_surface_anchors", "surface_anchor_mesh", "anchor_resolution",
                 "anchor_freeze")
        for name in names:
            self.assertIn(name, spec, name)

        import inspect

        parameters = set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters)
        for name in names:
            self.assertIn(name, parameters, name)

        # OFF by default: anchors are a HARD constraint (`--freeze` gives them no gradients
        # and no densification), so they must never switch themselves on.
        self.assertEqual(spec["use_surface_anchors"][1]["default"], False)
        self.assertEqual(spec["surface_anchor_mesh"][1]["default"], "")
        self.assertEqual(spec["anchor_freeze"][1]["default"], True)
        # MEASURED anchor counts on a 167k-vertex Poisson mesh: 128 -> 30,398,
        # 256 -> 121,434, 512 -> 486,479. Studio's own 1024 default lands near the whole
        # 1,000,000 cap, and frozen anchors cannot be pruned, so 256 is the sane start.
        self.assertEqual(spec["anchor_resolution"][1]["default"], 256)

    def test_build_training_command_appends_anchors_and_freezes_after_them(self):
        # Studio's own wording: --freeze freezes "the immediately preceding --add-splat
        # rows", so the freeze has to come AFTER the rows it pins.
        command = build_training_command(**default_command_options(),
                                         anchor_splats=[Path("a.ply")],
                                         anchor_freeze=True)
        self.assertEqual(command[-3:], ["--add-splat", "a.ply", "--freeze"])
        self.assertLess(command.index("--add-splat"), command.index("--freeze"))

        # without anchors nothing is emitted at all
        plain = build_training_command(**default_command_options(), anchor_freeze=True)
        self.assertNotIn("--freeze", plain)
        self.assertNotIn("--add-splat", plain)

        # anchors without the freeze: a warm start, not a constraint
        warm = build_training_command(**default_command_options(),
                                      anchor_splats=[Path("a.ply"), Path("b.ply")],
                                      anchor_freeze=False)
        self.assertEqual(warm.count("--add-splat"), 2)
        self.assertNotIn("--freeze", warm)

    def test_add_splat_is_a_value_flag_so_an_old_build_cannot_leave_a_stray_path(self):
        # filter_supported_flags drops flags an older Studio rejects. If --add-splat were
        # not known to take a value, the anchor PATH would survive as a bare argument and
        # Studio would fail to parse the command.
        self.assertIn("--add-splat", _LFS_VALUE_FLAGS)
        command = build_training_command(**default_command_options(),
                                         anchor_splats=[Path("anchors.ply")],
                                         anchor_freeze=True)
        filtered, dropped = filter_supported_flags(command, {"--data-path", "--iter"})
        self.assertIn("--add-splat", dropped)
        self.assertIn("--freeze", dropped)
        self.assertNotIn("anchors.ply", filtered)

    def test_resolve_anchor_mesh_prefers_the_dataset_mesh_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            self.assertIsNone(resolve_anchor_mesh(dataset))

            (dataset / "mesh").mkdir()
            self.assertIsNone(resolve_anchor_mesh(dataset))

            for name in ANCHOR_MESH_NAMES:
                candidate = dataset / "mesh" / name
                candidate.write_bytes(b"mesh")
                self.assertEqual(resolve_anchor_mesh(dataset), candidate)
                candidate.unlink()

            # any .obj / .glb is the fallback, then the dataset root
            obj = dataset / "mesh" / "surface.obj"
            obj.write_bytes(b"mesh")
            self.assertEqual(resolve_anchor_mesh(dataset), obj)

            explicit = dataset / "elsewhere.ply"
            explicit.write_bytes(b"mesh")
            self.assertEqual(resolve_anchor_mesh(dataset, str(explicit)), explicit)

            with self.assertRaises(ValueError):
                resolve_anchor_mesh(dataset, str(dataset / "missing.obj"))

    def test_build_mesh2splat_command_pins_the_documented_defaults(self):
        command = build_mesh2splat_command("studio.exe", "m.ply", "a.ply", 256)
        self.assertEqual(command[:3], ["studio.exe", "mesh2splat", "m.ply"])
        self.assertEqual(command[command.index("-o") + 1], "a.ply")
        self.assertEqual(command[command.index("--resolution") + 1], "256")
        # --sigma is pinned so a future Studio release cannot silently change the look
        self.assertEqual(command[command.index("--sigma") + 1], str(ANCHOR_SIGMA))
        self.assertEqual(command[-1], "-y")

    def test_prepare_surface_anchors_reports_and_degrades_instead_of_raising(self):
        notes = []
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / "anchors"
            with mock.patch("lichtfeld_training_node.subprocess.run") as runner:
                runner.return_value = types.SimpleNamespace(returncode=1, stdout="boom",
                                                            stderr="")
                self.assertEqual(prepare_surface_anchors("studio.exe", "m.ply", work_dir,
                                                         256, notes), [])
            self.assertEqual(len(notes), 1)
            self.assertIn("no anchors", notes[0])
            self.assertIn("boom", notes[0])

            notes.clear()
            anchors = work_dir / "surface_anchors_m_128.ply"

            def fake_run(command, **_kwargs):
                # the anchors must land in work_dir, never in the training output
                self.assertEqual(Path(command[command.index("-o") + 1]).parent, work_dir)
                anchors.write_bytes(b"ply")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch("lichtfeld_training_node.subprocess.run", fake_run):
                result = prepare_surface_anchors("studio.exe", "m.ply", work_dir, 128,
                                                 notes)
            self.assertEqual(result, [anchors])
            self.assertEqual(notes, [])

    def test_anchor_work_dir_is_outside_the_training_output(self):
        # A file written into the output folder would make it non-empty and trip the
        # node's own overwrite guard on the very next run.
        self.assertTrue(ANCHOR_WORK_DIR.is_absolute())
        self.assertEqual(ANCHOR_WORK_DIR.name, "enndee_lichtfeld_anchors")

    def test_undistort_cameras_widget_and_flag(self):
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        self.assertIn("undistort_cameras", spec)
        self.assertEqual(list(spec["undistort_cameras"][0]), list(UNDISTORT_MODES))
        self.assertEqual(spec["undistort_cameras"][1]["default"], "auto")

        import inspect

        self.assertIn("undistort_cameras",
                      set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters))

        plain = build_training_command(**default_command_options())
        self.assertNotIn("--undistort", plain)
        forced = build_training_command(**default_command_options(), undistort=True)
        self.assertIn("--undistort", forced)

    def test_dataset_needs_undistort_reads_the_camera_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            # no cameras.txt at all -> never add a flag we cannot justify
            self.assertEqual(dataset_camera_models(dataset), set())
            self.assertFalse(dataset_needs_undistort(dataset))

            sparse = dataset / "sparse" / "0"
            sparse.mkdir(parents=True)
            cameras = sparse / "cameras.txt"
            header = "# Camera list with one line of data per camera:\n"
            for model, expected in (("PINHOLE", False), ("SIMPLE_PINHOLE", False),
                                    ("SIMPLE_RADIAL", True), ("OPENCV", True),
                                    ("RADIAL", True)):
                cameras.write_text(f"{header}1 {model} 3456 2304 2786.2 1728 1152\n",
                                   encoding="utf-8")
                self.assertEqual(dataset_camera_models(dataset), {model})
                self.assertIs(dataset_needs_undistort(dataset), expected, model)

            # a mixed dataset counts as distorted if ANY camera is
            cameras.write_text(f"{header}1 PINHOLE 3456 2304 2786.2 1728 1152\n"
                               f"2 OPENCV 3456 2304 2786.2 1728 1152 0 0 0 0\n",
                               encoding="utf-8")
            self.assertTrue(dataset_needs_undistort(dataset))

            # a bare sparse/ folder (Lichtfeld also accepts this) is probed too
            (dataset / "sparse" / "0").rename(dataset / "sparse_only")
            (dataset / "sparse" / "cameras.txt").write_text(
                f"{header}1 SIMPLE_RADIAL 3456 2304 2786.2 1728 1152 0.01\n",
                encoding="utf-8")
            self.assertTrue(dataset_needs_undistort(dataset))

    # ------------------------------------------------- Lichtfeld 0.5.4 supervision
    def test_normal_loss_widgets_are_declared_and_in_the_signature(self):
        spec = LichtfeldHeadlessTrainer.INPUT_TYPES()["required"]
        names = ("use_normal_loss", "normal_loss_weight", "normal_consistency_weight",
                 "normal_flatten_weight", "normal_loss_space", "freeze_lr_scale")
        for name in names:
            self.assertIn(name, spec, name)

        import inspect

        parameters = set(inspect.signature(LichtfeldHeadlessTrainer.train).parameters)
        for name in names:
            self.assertIn(name, parameters, name)

        # OFF by default, and the weights keep Studio's own 0.5.4 defaults: normals are a
        # much more direct constraint than depth, so the weight is two orders smaller.
        self.assertEqual(spec["use_normal_loss"][1]["default"], False)
        self.assertAlmostEqual(spec["normal_loss_weight"][1]["default"], 0.005)
        self.assertAlmostEqual(spec["normal_consistency_weight"][1]["default"], 0.001)
        self.assertAlmostEqual(spec["normal_flatten_weight"][1]["default"], 0.0)
        self.assertEqual(spec["normal_loss_space"][1]["default"], "auto")
        self.assertEqual(list(spec["normal_loss_space"][0]),
                         ["auto", "camera-opencv", "camera-opengl", "world"])
        self.assertAlmostEqual(spec["freeze_lr_scale"][1]["default"], 0.0)

    def test_build_training_command_emits_the_normal_flags(self):
        plain = build_training_command(**default_command_options())
        for flag in ("--use-normal-loss", "--normal-loss-weight",
                     "--normal-consistency-weight", "--normal-flatten-weight",
                     "--normal-loss-space"):
            self.assertNotIn(flag, plain, flag)

        command = build_training_command(**default_command_options(), use_normal_loss=True)
        self.assertIn("--use-normal-loss", command)
        self.assertEqual(command[command.index("--normal-loss-weight") + 1], "0.005")
        self.assertEqual(command[command.index("--normal-consistency-weight") + 1], "0.001")
        # flatten 0 / space auto are the build's own defaults, so they are not sent
        self.assertNotIn("--normal-flatten-weight", command)
        self.assertNotIn("--normal-loss-space", command)

        tuned = build_training_command(**default_command_options(), use_normal_loss=True,
                                       normal_flatten_weight=0.05,
                                       normal_loss_space="camera-opencv")
        self.assertEqual(tuned[tuned.index("--normal-flatten-weight") + 1], "0.05")
        self.assertEqual(tuned[tuned.index("--normal-loss-space") + 1], "camera-opencv")

    def test_normal_and_freeze_flags_are_value_flags(self):
        # a dropped flag must take its value with it, or Studio gets a stray argument
        for flag in ("--normal-loss-weight", "--normal-consistency-weight",
                     "--normal-flatten-weight", "--normal-loss-space",
                     "--depth-loss-mode", "--depth-loss-weight", "--freeze-lr-scale"):
            self.assertIn(flag, _LFS_VALUE_FLAGS, flag)

    def test_freeze_lr_scale_needs_frozen_anchors(self):
        # 0.0 is the hard freeze Studio already defaults to, so nothing is sent
        command = build_training_command(**default_command_options(),
                                         anchor_splats=[Path("a.ply")],
                                         anchor_freeze=True, freeze_lr_scale=0.05)
        self.assertIn("--freeze", command)
        self.assertEqual(command[command.index("--freeze-lr-scale") + 1], "0.05")

        hard = build_training_command(**default_command_options(),
                                      anchor_splats=[Path("a.ply")],
                                      anchor_freeze=True, freeze_lr_scale=0.0)
        self.assertNotIn("--freeze-lr-scale", hard)

        # without anchors there is nothing to freeze, so the scale is meaningless
        warm = build_training_command(**default_command_options(),
                                      anchor_splats=[Path("a.ply")],
                                      anchor_freeze=False, freeze_lr_scale=0.05)
        self.assertNotIn("--freeze-lr-scale", warm)

    def test_depth_loss_mode_is_translated_for_a_0_5_4_build(self):
        # 0.5.3 named them adaptive-warped-l1 / pearson; 0.5.4 replaced both with the ssi
        # family, so a saved workflow's value has to be translated instead of rejected.
        new = frozenset({"ssi", "ssi-disparity", "ssi-depth"})
        self.assertEqual(resolve_depth_loss_mode("adaptive-warped-l1", new), "ssi")
        self.assertEqual(resolve_depth_loss_mode("pearson", new), "ssi")
        self.assertEqual(resolve_depth_loss_mode("ssi-depth", new), "ssi-depth")
        # an unknown build says nothing usable -> pass the value through untouched
        self.assertEqual(resolve_depth_loss_mode("pearson", None), "pearson")
        # a value the build does not know and we cannot map -> the build's own default
        self.assertEqual(resolve_depth_loss_mode("something-else", new), "ssi")
        self.assertEqual(resolve_depth_loss_mode("", new), "")

    def test_parse_studio_capabilities_reads_the_0_5_4_depth_modes(self):
        help_text = (
            "        --depth-loss-mode=[depth_loss_mode]\n"
            "                                          Depth prior convention: ssi "
            "(auto-detect), ssi-disparity, or ssi-depth (default: ssi)\n"
        )
        capabilities = parse_studio_capabilities(help_text)
        self.assertEqual(capabilities["depth_loss_modes"],
                         frozenset({"ssi", "ssi-disparity", "ssi-depth"}))
        # a 0.5.3-style help without that sentence leaves it unknown
        self.assertIsNone(
            parse_studio_capabilities("--depth-loss-mode=[m]\n")["depth_loss_modes"])

    def test_write_lfs_config_file_merges_a_user_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "user.json"
            base.write_text(
                json.dumps({"dataset": {"images": "images"},
                            "optimization": {"iterations": 1234}}),
                encoding="utf-8",
            )
            section = build_lfs_optimization_section(stop_refine=99, mask_mode="none")
            path = write_lfs_config_file(section, str(base))
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            finally:
                Path(path).unlink(missing_ok=True)
            self.assertEqual(payload["optimization"]["stop_refine"], 99)
            self.assertEqual(payload["optimization"]["mask_mode"], "none")
            self.assertIn("iterations", payload["optimization"])
            self.assertEqual(payload["dataset"]["images"], "images")



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

    def test_iter_and_steps_scaler_are_never_sent_together(self):
        # Lichtfeld 0.5.4 rejects both at once:
        # "Error: --iter and --steps-scaler are mutually exclusive: --iter sets the
        #  iteration count exactly, --steps-scaler ..."
        # which made EVERY run fail instantly until the node stopped sending both.
        plain = build_training_command(**default_command_options())
        self.assertIn("--iter", plain)
        self.assertNotIn("--steps-scaler", plain)

        options = default_command_options()
        options["steps_scaler"] = 2.0
        scaled = build_training_command(**options)
        self.assertNotIn("--iter", scaled)
        self.assertEqual(scaled[scaled.index("--steps-scaler") + 1], "2.0")

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
            self.assertIn("--config", result["result"][1])
            self.assertNotIn("--python-script", result["result"][1])
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


class LichtfeldEvaluationTests(unittest.TestCase):
    """`--test-every` / `--no-save-eval-images` plumbing (held-out evaluation)."""

    @staticmethod
    def _kwargs(**overrides):
        values = dict(
            executable="studio", dataset="dataset", output="output", log_file="",
            iterations=1000, strategy="mcmc", sh_degree=3, max_cap=1000,
            steps_scaler=1.0, mask_mode="none", invert_masks=False,
            bg_mode="solidcolor", bg_color="#FFFFFF", enable_mip=False,
            bilateral_grid=False, enable_eval=True, enable_sparsity=False,
            log_level="info",
        )
        values.update(overrides)
        return values

    def test_split_labels_map_to_studios_test_every(self):
        self.assertEqual(EVALUATION_SPLITS, {"off": 0, "1/2": 2, "1/3": 3, "1/4": 4})

    def test_off_sends_no_test_every_flag(self):
        command = build_training_command(**self._kwargs(
            test_every=EVALUATION_SPLITS["off"]))
        self.assertNotIn("--test-every", command)

    def test_each_ratio_sends_its_test_every_value(self):
        for label, expected in (("1/2", "2"), ("1/3", "3"), ("1/4", "4")):
            command = build_training_command(**self._kwargs(
                test_every=EVALUATION_SPLITS[label]))
            self.assertIn("--test-every", command, label)
            self.assertEqual(command[command.index("--test-every") + 1], expected, label)

    def test_evaluation_still_sends_the_eval_flag(self):
        command = build_training_command(**self._kwargs(enable_eval=True, test_every=2))
        self.assertIn("--eval", command)

    def test_save_eval_images_is_an_opt_out(self):
        keep = build_training_command(**self._kwargs(test_every=2, save_eval_images=True))
        self.assertNotIn("--no-save-eval-images", keep)
        drop = build_training_command(**self._kwargs(test_every=2, save_eval_images=False))
        self.assertIn("--no-save-eval-images", drop)

    def test_a_degenerate_split_is_rejected(self):
        for bad in (1, -1):
            with self.assertRaises(ValueError):
                build_training_command(**self._kwargs(test_every=bad))

    def test_test_every_counts_as_a_value_flag(self):
        # filter_supported_flags must drop its value together with the flag
        self.assertIn("--test-every", _LFS_VALUE_FLAGS)

    def test_default_eval_steps_always_include_the_final_iteration(self):
        # Studio writes an empty report when eval_steps is empty, so the node
        # fills it in whenever evaluation is on without explicit steps.
        self.assertEqual(default_eval_steps(8000), [2000, 4000, 6000, 8000])
        self.assertEqual(default_eval_steps(30000), [7500, 15000, 22500, 30000])
        self.assertEqual(default_eval_steps(4), [1, 2, 3, 4])
        self.assertEqual(default_eval_steps(3), [1, 2, 3])
        self.assertEqual(default_eval_steps(1), [1])
        self.assertEqual(default_eval_steps(0), [1])


if __name__ == "__main__":
    unittest.main()
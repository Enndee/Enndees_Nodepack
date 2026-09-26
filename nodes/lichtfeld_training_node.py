"""Run LichtFeld Studio's supported headless Gaussian-splat trainer in ComfyUI."""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time


_DEFAULT_STUDIO = (
    Path(r"D:\Gaussian_Splatting\Lichtfeld_Gaussian_Splatting")
    / "LichtFeld-Studio-windows-v0.5.3"
    / "bin"
    / "LichtFeld-Studio.exe"
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_TAIL_LINES = 80


def _default_executable():
    """Prefer an explicit user setting, then PATH, then this machine's install."""
    configured = os.environ.get("LICHTFELD_STUDIO_EXE", "").strip()
    if configured:
        return configured
    on_path = shutil.which("!LichtFeld-Studio.exe") or shutil.which("LichtFeld-Studio.exe")
    if on_path:
        return on_path
    return str(_DEFAULT_STUDIO) if _DEFAULT_STUDIO.is_file() else ""


def resolve_studio_executable(path):
    """Resolve legacy saved !LichtFeld-Studio.exe paths to the installed file."""
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("Set the path to Lichtfeld Studio's executable.")
    executable = Path(raw).expanduser().resolve()
    if executable.is_file():
        return executable
    if executable.name.lower() == "!lichtfeld-studio.exe":
        unprefixed = executable.with_name("LichtFeld-Studio.exe")
        if unprefixed.is_file():
            return unprefixed.resolve()
    raise ValueError(
        f"Lichtfeld Studio executable not found: {executable}. "
        "Set the Studio executable path or LICHTFELD_STUDIO_EXE."
    )


def validate_dataset(dataset_path):
    """Check that a directory contains images and the tracker's COLMAP export."""
    raw_path = str(dataset_path or "").strip()
    if not raw_path:
        raise ValueError("Connect a GLOMAP Tracker dataset_path or enter a dataset folder.")

    dataset = Path(raw_path).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f"Lichtfeld training dataset folder does not exist: {dataset}")

    image_dir = dataset / "images"
    if not image_dir.is_dir() or not any(
        path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        for path in image_dir.iterdir()
    ):
        raise ValueError(f"Dataset must contain at least one supported image in {image_dir}")

    sparse_dir = dataset / "sparse" / "0"
    missing = [name for name in ("cameras.txt", "images.txt", "points3D.txt")
               if not (sparse_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"Dataset is missing sparse/0/{', sparse/0/'.join(missing)}: {dataset}"
        )
    return dataset


def resolve_training_output(dataset, output_path=""):
    """Choose a unique training folder, treating the dataset root as auto output."""
    dataset = Path(dataset).expanduser().resolve()
    raw_output = str(output_path or "").strip()
    if raw_output:
        output = Path(raw_output).expanduser()
        if not output.is_absolute():
            output = dataset / output
        output = output.resolve()
        if output != dataset:
            return output

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return dataset / "output" / f"ComfyUI_Training_{stamp}"


def _check_running_studio_processes():
    """Return PIDs for existing LichtFeld Studio apps without stopping them."""
    try:
        import psutil
    except ImportError as exc:  # psutil is in this pack's requirements.txt
        raise RuntimeError(
            "Cannot safely check for a running LichtFeld Studio session because "
            "psutil is unavailable. Install the nodepack requirements, or use "
            "the explicit Allow Concurrent Studio Process option."
        ) from exc

    current_pid = os.getpid()
    found = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            info = process.info
            if (info.get("pid") != current_pid
                    and str(info.get("name", "")).lower()
                    in {"!lichtfeld-studio.exe", "lichtfeld-studio.exe"}):
                found.append(int(info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found


def parse_iteration_steps(value, name, iterations):
    """Parse an optional comma-separated, sorted list of training step numbers."""
    raw = str(value or "").strip()
    if not raw:
        return None
    parts = raw.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(f"{name} must be comma-separated positive iteration numbers.")
    try:
        steps = sorted({int(part.strip()) for part in parts})
    except ValueError as exc:
        raise ValueError(f"{name} must contain comma-separated integer iteration numbers.") from exc
    invalid = [step for step in steps if step < 1 or step > int(iterations)]
    if invalid:
        raise ValueError(
            f"{name} steps must be between 1 and the total iterations "
            f"({int(iterations):,}); invalid: {', '.join(map(str, invalid))}."
        )
    return steps


def validate_refinement_steps(grow_until_iter, stop_refine, iterations):
    """Validate optional refinement cutoffs against configured iterations."""
    total = int(iterations)
    for name, value in (("Grow Until Iter", grow_until_iter), ("Stop Refine", stop_refine)):
        number = int(value)
        if number < 0:
            raise ValueError(f"{name} must be zero (use Studio defaults) or greater.")
        if number > total:
            raise ValueError(f"{name} cannot exceed total iterations ({total:,}).")


def build_lfs_settings_script(
    grow_until_iter=0,
    stop_refine=0,
    save_steps=None,
    eval_steps=None,
    enable_eval=False,
):
    """Build a Lichtfeld --python-script hook for parameters without CLI flags."""
    settings = {}
    if int(grow_until_iter) > 0:
        settings["grow_until_iter"] = int(grow_until_iter)
    if int(stop_refine) > 0:
        settings["stop_refine"] = int(stop_refine)
    if save_steps is not None:
        settings["save_steps"] = sorted({int(step) for step in save_steps})
    if eval_steps is not None:
        settings["eval_steps"] = sorted({int(step) for step in eval_steps})
    if enable_eval or eval_steps is not None:
        settings["enable_eval"] = True
        if eval_steps is None and save_steps:
            # Studio's GUI mirrors save steps as evaluation steps when eval is on.
            settings["eval_steps"] = sorted({int(step) for step in save_steps})
    if not settings:
        return ""

    encoded_settings = repr(json.dumps(settings, sort_keys=True))
    return (
        "import json\n"
        "import lichtfeld as lf\n\n"
        f"_ENNDEE_SETTINGS = json.loads({encoded_settings})\n\n"
        "_ENNDEE_SETTINGS_APPLIED = False\n\n"
        "def _enndee_apply_training_settings(*_args, **_kwargs):\n"
        "    global _ENNDEE_SETTINGS_APPLIED\n"
        "    if _ENNDEE_SETTINGS_APPLIED:\n"
        "        return\n"
        "    params = lf.optimization_params()\n"
        "    if params is None or not params.has_params():\n"
        "        raise RuntimeError('Lichtfeld optimization parameters are unavailable.')\n"
        "    for name in ('grow_until_iter', 'stop_refine'):\n"
        "        if name in _ENNDEE_SETTINGS:\n"
        "            params.set(name, _ENNDEE_SETTINGS[name])\n"
        "    if 'enable_eval' in _ENNDEE_SETTINGS:\n"
        "        params.enable_eval = True\n"
        "    if 'save_steps' in _ENNDEE_SETTINGS:\n"
        "        params.clear_save_steps()\n"
        "        for step in _ENNDEE_SETTINGS['save_steps']:\n"
        "            params.add_save_step(step)\n"
        "    if 'eval_steps' in _ENNDEE_SETTINGS:\n"
        "        params.clear_eval_steps()\n"
        "        for step in _ENNDEE_SETTINGS['eval_steps']:\n"
        "            params.add_eval_step(step)\n"
        "    _ENNDEE_SETTINGS_APPLIED = True\n\n"
        "lf.on_iteration_start(_enndee_apply_training_settings)\n"
    )


def write_lfs_settings_script(contents):
    """Write the generated Lichtfeld callback into an auto-cleaned temp file."""
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".py",
        prefix="enndee_lfs_training_",
        delete=False,
    )
    path = Path(handle.name)
    try:
        with handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def build_training_command(
    executable,
    dataset,
    output,
    log_file,
    iterations,
    strategy,
    sh_degree,
    max_cap,
    steps_scaler,
    mask_mode,
    invert_masks,
    bg_mode,
    bg_color,
    enable_mip,
    bilateral_grid,
    enable_eval,
    enable_sparsity,
    log_level,
    output_name="",
    config_file="",
    centralize_dataset="off",
    resize_factor="auto",
    max_image_width=3840,
    disable_downscaling=False,
    python_script="",
):
    """Build an argument vector for the documented LichtFeld Studio CLI."""
    if int(iterations) < 1:
        raise ValueError("Training iterations must be at least 1.")
    if int(max_cap) < 1:
        raise ValueError("Maximum Gaussian count must be at least 1.")
    if int(sh_degree) not in range(4):
        raise ValueError("Maximum SH degree must be between 0 and 3.")
    if float(steps_scaler) <= 0:
        raise ValueError("Training steps scaler must be greater than 0.")
    if strategy not in {"mcmc", "mrnf", "igs+"}:
        raise ValueError(f"Unsupported Lichtfeld optimization strategy: {strategy}")
    if mask_mode not in {
        "none", "segment", "ignore", "segment_and_ignore", "alpha_consistent"
    }:
        raise ValueError(f"Unsupported Lichtfeld mask mode: {mask_mode}")
    if bg_mode not in {"solidcolor", "modulation", "random"}:
        raise ValueError(f"Unsupported Lichtfeld background mode: {bg_mode}")
    if log_level not in {"trace", "debug", "info", "perf", "warn", "error"}:
        raise ValueError(f"Unsupported Lichtfeld log level: {log_level}")
    if centralize_dataset not in {"off", "by_pointcloud", "by_cameras"}:
        raise ValueError(f"Unsupported Lichtfeld dataset centering mode: {centralize_dataset}")
    if str(resize_factor) not in {"auto", "1", "2", "4", "8"}:
        raise ValueError("Image resize factor must be auto, 1, 2, 4, or 8.")
    if int(max_image_width) < 0 or (
        int(max_image_width) != 0 and int(max_image_width) < 64
    ):
        raise ValueError("Maximum image width must be 0 (unlimited) or at least 64 pixels.")
    effective_resize_factor = "1" if disable_downscaling else str(resize_factor)
    effective_max_width = 0 if disable_downscaling else int(max_image_width)

    color = str(bg_color).strip()
    if bg_mode == "solidcolor" and not color:
        raise ValueError("Solid-color background color cannot be empty.")
    output_name = str(output_name or "").strip()
    if output_name and Path(output_name).name != output_name:
        raise ValueError("Output name must be a filename stem, not a path.")
    config_file = str(config_file or "").strip()

    command = [
        str(executable),
        "--data-path", str(dataset),
        "--output-path", str(output),
        "--iter", str(int(iterations)),
        "--strategy", strategy,
        "--sh-degree", str(int(sh_degree)),
        "--max-cap", str(int(max_cap)),
        "--steps-scaler", str(float(steps_scaler)),
        "--mask-mode", mask_mode,
        "--bg-mode", bg_mode,
        "--log-level", log_level,
        f"--centralize={centralize_dataset}",
        f"--resize_factor={effective_resize_factor}",
        f"--max-width={effective_max_width}",
        "--headless",
        "--train",
    ]
    if bg_mode == "solidcolor":
        command.extend(["--bg-color", color])
    if output_name:
        command.extend(["--output-name", output_name])
    if config_file:
        command.extend(["--config", config_file])
    if python_script:
        command.extend(["--python-script", str(python_script)])
    if invert_masks:
        command.append("--invert-masks")
    if enable_mip:
        command.append("--enable-mip")
    if bilateral_grid:
        command.append("--bilateral-grid")
    if enable_eval:
        command.append("--eval")
    if enable_sparsity:
        command.append("--enable-sparsity")
    if log_file:
        command.extend(["--log-file", str(log_file)])
    return command


def run_streaming_command(command, cwd, progress_callback=None):
    """Stream newline/CR progress and quiet-stage heartbeats from a child process."""
    popen_options = {
        "cwd": str(cwd),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "bufsize": 0,
    }
    if os.name == "nt":
        popen_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    process = subprocess.Popen(command, **popen_options)
    tail = deque(maxlen=_TAIL_LINES)
    started = time.monotonic()
    next_heartbeat = started + 30.0
    output_queue = queue.Queue()

    def read_output():
        try:
            assert process.stdout is not None
            while True:
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                output_queue.put(chunk)
        except Exception as exc:  # relay output-reader failures to the main thread
            output_queue.put(exc)
        finally:
            output_queue.put(None)

    reader = threading.Thread(
        target=read_output, name="enndee-lichtfeld-output", daemon=True
    )
    reader.start()
    pending = bytearray()
    reader_finished = False

    def emit_record(raw_record):
        nonlocal next_heartbeat
        line = raw_record.decode("utf-8", errors="replace").strip()
        if not line:
            return
        tail.append(line)
        elapsed = int(time.monotonic() - started)
        message = f"[Enndee Lichtfeld +{elapsed // 60:02d}:{elapsed % 60:02d}] {line}"
        print(message, flush=True)
        if progress_callback is not None:
            progress_callback(message)
        next_heartbeat = time.monotonic() + 30.0

    def consume_records(final=False):
        while pending:
            boundary = next(
                (index for index, byte in enumerate(pending) if byte in (10, 13)),
                None,
            )
            if boundary is None:
                if final:
                    record = bytes(pending)
                    pending.clear()
                    emit_record(record)
                elif len(pending) > 65536:
                    # A child may print progress without a newline. Keep both
                    # Comfy's console and the staging buffer responsive/bounded.
                    record = bytes(pending[:65536])
                    del pending[:65536]
                    emit_record(record)
                return
            record = bytes(pending[:boundary])
            delimiter = pending[boundary]
            del pending[:boundary + 1]
            if delimiter == 13 and pending[:1] == b"\n":
                del pending[:1]
            emit_record(record)

    try:
        while not reader_finished or process.poll() is None:
            now = time.monotonic()
            try:
                item = output_queue.get(timeout=0.25)
            except queue.Empty:
                item = "__timeout__"

            if item is None:
                reader_finished = True
                consume_records(final=True)
            elif isinstance(item, Exception):
                print(f"[Enndee Lichtfeld] Output reader warning: {item}", flush=True)
                reader_finished = True
                if process.poll() is None:
                    process.kill()
            elif item != "__timeout__":
                pending.extend(item)
                consume_records()
            elif process.poll() is None and now >= next_heartbeat:
                elapsed = int(now - started)
                message = (
                    f"[Enndee Lichtfeld +{elapsed // 60:02d}:{elapsed % 60:02d}] "
                    "still running"
                )
                print(message, flush=True)
                if progress_callback is not None:
                    progress_callback(message)
                next_heartbeat = now + 30.0

        return_code = process.wait()
        reader.join(timeout=2.0)
        return return_code, "\n".join(tail)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()

    return return_code, "\n".join(tail)


class LichtfeldHeadlessTrainer:
    """Launch the installed LichtFeld Studio CLI as a ComfyUI training job."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "studio_executable": ("STRING", {
                    "default": _default_executable(),
                    "tooltip": (
                        "Path to LichtFeld-Studio.exe. This node uses Studio's own "
                        "--headless/--train CLI; it does not install CUDA or Torch."
                    ),
                }),
                "dataset_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Connect the GLOMAP Tracker dataset_path output, or enter a "
                        "folder containing images/ and sparse/0/{cameras.txt, "
                        "images.txt, points3D.txt}."
                    ),
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Training output folder. Empty creates a new timestamped "
                        "ComfyUI_Training folder inside <dataset>/output."
                    ),
                }),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional Lichtfeld output filename stem. Leave empty for "
                        "Studio's default splat_ITER naming."
                    ),
                }),
                "config_file": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional JSON Lichtfeld Studio training configuration "
                        "(--config), for settings beyond the widgets below."
                    ),
                }),
                "iterations": ("INT", {
                    "default": 30000, "min": 1, "max": 10000000, "step": 1000,
                    "tooltip": "Number of Lichtfeld optimization iterations.",
                }),
                "strategy": (["mcmc", "mrnf", "igs+"], {
                    "default": "mcmc",
                    "tooltip": "Lichtfeld optimization strategy.",
                }),
                "max_sh_degree": ("INT", {
                    "default": 3, "min": 0, "max": 3, "step": 1,
                    "tooltip": "Maximum spherical-harmonics degree (0-3).",
                }),
                "max_gaussians": ("INT", {
                    "default": 6000000, "min": 1000, "max": 50000000, "step": 100000,
                    "tooltip": "Maximum number of trained Gaussians (--max-cap).",
                }),
                "steps_scaler": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1,
                    "tooltip": "Lichtfeld training-steps scale factor.",
                }),
                "mask_mode": ([
                    "none", "segment", "ignore", "segment_and_ignore", "alpha_consistent"
                ], {
                    "default": "segment",
                    "tooltip": (
                        "The tracker exports masks/ with WHITE=subject/keep and "
                        "BLACK=background. Try segment first; check the rendered "
                        "result if your dataset mask polarity differs. RGBA input "
                        "images may additionally provide an automatic alpha mask."
                    ),
                }),
                "invert_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask pixels if the foreground/background is reversed.",
                }),
                "background_mode": (["solidcolor", "modulation", "random"], {
                    "default": "solidcolor",
                    "tooltip": "Studio training background mode. Use solidcolor for a white backdrop.",
                }),
                "background_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Solid background color as #RRGGBB (default white).",
                }),
                "enable_mip": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Lichtfeld's mip-filter antialiasing.",
                }),
                "bilateral_grid": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Lichtfeld bilateral-grid appearance filtering.",
                }),
                "enable_eval": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run evaluation during training (uses extra time and storage).",
                }),
                "enable_sparsity": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run Lichtfeld sparsity optimization after regular training.",
                }),
                "log_level": (["info", "debug", "perf", "warn", "error", "trace"], {
                    "default": "info",
                    "tooltip": "Lichtfeld console/log-file verbosity.",
                }),
                "overwrite_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Allow using a non-empty output folder. Off by default to "
                        "protect existing checkpoints and splat exports."
                    ),
                }),
                "allow_concurrent_studio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "By default, refuse to launch if any LichtFeld Studio GUI "
                        "process is open. Enable only when you explicitly want two "
                        "Studio instances (and accept the GPU contention risk)."
                    ),
                }),
                "preview_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Validate the dataset and show the exact command without launching training.",
                }),
                # New controls intentionally come last so old saved workflows keep
                # mapping their widget_values to the existing inputs.
                "centralize_dataset": (["off", "by_pointcloud", "by_cameras"], {
                    "default": "off",
                    "tooltip": (
                        "Centralize the dataset coordinate origin during Studio's "
                        "dataset load: off, by point cloud, or by camera positions."
                    ),
                }),
                "image_resize_factor": (["auto", "1", "2", "4", "8"], {
                    "default": "auto",
                    "tooltip": (
                        "Studio dataset input resize factor. Select 1 for native image "
                        "dimensions; auto may downscale based on the next limit."
                    ),
                }),
                "max_image_width": ("INT", {
                    "default": 3840, "min": 0, "max": 32768, "step": 256,
                    "tooltip": (
                        "Maximum input image width in pixels; set 0 to disable the "
                        "width cap. Has no effect when Disable Downscaling is enabled."
                    ),
                }),
                "disable_downscaling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Force native-resolution training images (resize factor 1, "
                        "no maximum-width cap). Higher VRAM use and compute cost."
                    ),
                }),
                "grow_until_iter": ("INT", {
                    "default": 0, "min": 0, "max": 10000000, "step": 1000,
                    "tooltip": (
                        "Studio MRNF refinement Grow Until Iter. 0 leaves the Studio "
                        "default/config-file value unchanged. Applied with the "
                        "Lichtfeld Python-script callback."
                    ),
                }),
                "stop_refine": ("INT", {
                    "default": 0, "min": 0, "max": 10000000, "step": 1000,
                    "tooltip": (
                        "Studio refinement Stop Refine iteration. 0 leaves the Studio "
                        "default/config-file value unchanged. Applied with the "
                        "Lichtfeld Python-script callback."
                    ),
                }),
                "save_steps": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional comma-separated checkpoint save iterations, e.g. "
                        "5000,10000,15000. Empty preserves Studio/config defaults."
                    ),
                }),
                "eval_steps": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional comma-separated evaluation iterations. Evaluation "
                        "is enabled automatically when these steps are supplied. "
                        "Empty mirrors save steps when evaluation is enabled."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_path", "command", "log_file", "summary")
    FUNCTION = "train"
    CATEGORY = "Enndee/3D"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Trains a GLOMAP-exported dataset with Lichtfeld Studio's official "
        "headless CLI. Progress is streamed to the ComfyUI console."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Training is a side effect: a new queue run must train even if inputs match."""
        return float("nan")

    def train(
        self,
        studio_executable,
        dataset_path,
        output_path="",
        output_name="",
        config_file="",
        centralize_dataset="off",
        image_resize_factor="auto",
        max_image_width=3840,
        disable_downscaling=False,
        grow_until_iter=0,
        stop_refine=0,
        save_steps="",
        eval_steps="",
        iterations=30000,
        strategy="mcmc",
        max_sh_degree=3,
        max_gaussians=6000000,
        steps_scaler=1.0,
        mask_mode="segment",
        invert_masks=False,
        background_mode="solidcolor",
        background_color="#FFFFFF",
        enable_mip=False,
        bilateral_grid=False,
        enable_eval=False,
        enable_sparsity=False,
        log_level="info",
        overwrite_output=False,
        allow_concurrent_studio=False,
        preview_only=False,
    ):
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Training iterations must be at least 1.")
        if steps_scaler <= 0:
            raise ValueError("Training steps scaler must be greater than 0.")
        validate_refinement_steps(grow_until_iter, stop_refine, iterations)
        parsed_save_steps = parse_iteration_steps(save_steps, "Save Steps", iterations)
        parsed_eval_steps = parse_iteration_steps(eval_steps, "Eval Steps", iterations)
        effective_enable_eval = bool(enable_eval or parsed_eval_steps is not None)

        dataset = validate_dataset(dataset_path)
        if mask_mode != "none":
            mask_dir = dataset / "masks"
            if not mask_dir.is_dir() or not any(
                path.is_file() and path.suffix.lower() == ".png"
                for path in mask_dir.iterdir()
            ):
                raise ValueError(
                    f"Mask mode '{mask_mode}' requires Lichtfeld mask PNGs under "
                    f"{mask_dir}. Use mask_mode='none' for datasets without masks."
                )
        executable = resolve_studio_executable(studio_executable)
        config_path = str(config_file or "").strip()
        if config_path:
            config = Path(config_path).expanduser().resolve()
            if not config.is_file():
                raise ValueError(f"Lichtfeld Studio JSON config not found: {config}")
            try:
                with config.open("r", encoding="utf-8") as config_stream:
                    json.load(config_stream)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Lichtfeld Studio config is not readable valid JSON: {config} ({exc})"
                ) from exc
            config_path = str(config)

        output = resolve_training_output(dataset, output_path)
        if output in dataset.parents:
            raise ValueError("Training output cannot be a parent of the dataset.")
        protected_folders = (
            dataset / "images",
            dataset / "masks",
            dataset / "masks_GLOMAP",
            dataset / "sparse",
        )
        for protected in protected_folders:
            if output == protected or protected in output.parents:
                raise ValueError(
                    "Training output must not overwrite the dataset or write into "
                    f"its protected data folders: {protected}"
                )

        log_file = ""
        if not preview_only:
            log_file = str(
                output / f"lichtfeld_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )

        settings_script = build_lfs_settings_script(
            grow_until_iter=grow_until_iter,
            stop_refine=stop_refine,
            save_steps=parsed_save_steps,
            eval_steps=parsed_eval_steps,
            enable_eval=effective_enable_eval,
        )
        command = build_training_command(
            executable=executable,
            dataset=dataset,
            output=output,
            log_file=log_file,
            iterations=iterations,
            strategy=strategy,
            sh_degree=max_sh_degree,
            max_cap=max_gaussians,
            steps_scaler=steps_scaler,
            mask_mode=mask_mode,
            invert_masks=invert_masks,
            bg_mode=background_mode,
            bg_color=background_color,
            enable_mip=enable_mip,
            bilateral_grid=bilateral_grid,
            enable_eval=effective_enable_eval,
            enable_sparsity=enable_sparsity,
            log_level=log_level,
            output_name=output_name,
            config_file=config_path,
            centralize_dataset=centralize_dataset,
            resize_factor=image_resize_factor,
            max_image_width=max_image_width,
            disable_downscaling=disable_downscaling,
            python_script="<temporary Lichtfeld settings script>" if settings_script else "",
        )
        preview_command = subprocess.list2cmdline(command)
        print(f"[Enndee Lichtfeld] Dataset: {dataset}", flush=True)
        print(f"[Enndee Lichtfeld] Output: {output}", flush=True)
        print(f"[Enndee Lichtfeld] Command: {preview_command}", flush=True)

        if preview_only:
            summary = "Preview only: dataset validated; training was not started."
            return {"ui": {"text": [summary, preview_command]},
                    "result": (str(output), preview_command, "", summary)}

        if output.is_dir() and any(output.iterdir()) and not overwrite_output:
            raise ValueError(
                f"Training output is not empty: {output}. Choose a new output "
                "folder or explicitly enable overwrite_output."
            )
        if output.exists() and not output.is_dir():
            raise ValueError(f"Training output path is not a folder: {output}")
        if not allow_concurrent_studio:
            running = _check_running_studio_processes()
            if running:
                raise RuntimeError(
                    "An existing LichtFeld Studio GUI process is open "
                    f"(PID(s): {', '.join(map(str, running))}). It has NOT been "
                    "closed or modified. Close Studio before headless training, "
                    "or explicitly enable allow_concurrent_studio."
                )

        settings_script_path = None
        if settings_script:
            settings_script_path = write_lfs_settings_script(settings_script)
            script_index = command.index("--python-script") + 1
            command[script_index] = str(settings_script_path)

        try:
            output.mkdir(parents=True, exist_ok=True)
        except Exception:
            if settings_script_path is not None:
                settings_script_path.unlink(missing_ok=True)
            raise

        try:
            return_code, tail = run_streaming_command(
                command, cwd=executable.parent
            )
        finally:
            if settings_script_path is not None:
                settings_script_path.unlink(missing_ok=True)
        if return_code:
            raise RuntimeError(
                f"Lichtfeld Studio training exited with code {return_code}. "
                f"Full log: {log_file}\nLast output:\n{tail or '(no console output)'}"
            )

        summary = (
            f"Lichtfeld training completed successfully ({int(iterations):,} configured "
            f"iterations). Output: {output}\nLog: {log_file}"
        )
        return {"ui": {"text": [summary, tail]},
                "result": (str(output), preview_command, log_file, summary)}
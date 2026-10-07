"""Run LichtFeld Studio's supported headless Gaussian-splat trainer in ComfyUI."""

from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
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
_EXPORT_SUFFIXES = {"ply": ".ply", "sog": ".sog", "spz": ".spz"}
_EXPORT_FORMATS = tuple(_EXPORT_SUFFIXES)
# Flags the node emits that consume the *next* token as their value: when an unsupported flag is
# filtered out, its value has to go as well (inline `--flag=value` forms are single tokens).
_LFS_VALUE_FLAGS = frozenset({
    "--data-path", "--output-path", "--iter", "--strategy", "--sh-degree", "--max-cap",
    "--steps-scaler", "--mask-mode", "--bg-mode", "--bg-color", "--log-level", "--log-file",
    "--output-name", "--config", "--python-script",
})
_STUDIO_PROBE_CACHE = {}
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


def validate_export_format(value):
    """Normalize the splat export choice: Studio trains to .ply and converts to sog/spz."""
    export_format = str(value or "ply").strip().lower().lstrip(".")
    if export_format not in _EXPORT_SUFFIXES:
        raise ValueError(
            f"Unsupported splat export format {value!r}; expected one of "
            f"{', '.join(_EXPORT_FORMATS)}."
        )
    return export_format


def resolve_trained_splat(output, output_name=""):
    """Pick the trained splat .ply inside `output` for the optional format conversion.

    Studio's headless trainer always writes `splat_ITER.ply` (or the `--output-name`
    stem); with checkpoint saves the highest iteration is the finished model, so it wins
    over the earlier ones. Returns None when the folder holds no splat at all.
    """
    folder = Path(output)
    candidates = [path for path in folder.glob("*.ply") if path.is_file()]
    preferred = str(output_name or "").strip()
    if preferred:
        stem = Path(preferred).stem.lower()
        named = [path for path in candidates if path.stem.lower().startswith(stem)]
        if named:
            candidates = named
    if not candidates:
        return None

    def rank(path):
        digits = re.findall(r"\d+", path.stem)
        return (int(digits[-1]) if digits else -1, path.stat().st_mtime)

    return max(candidates, key=rank)


def build_conversion_command(executable, source, target):
    """Build Studio's `convert` argv; --overwrite keeps the headless run from prompting."""
    return [str(executable), "convert", str(source), str(target), "--overwrite"]


def parse_studio_capabilities(help_text, convert_text=""):
    """Read a LichtFeld Studio CLI surface out of its own `--help` output.

    Older free builds predate flags the node sends (`--bg-mode`, `--bg-color`,
    `--centralize`, `--output-name`, ...) and reject unknown ones with
    "Error: Parse error: Flag could not be matched", so the node has to know what the
    installed binary accepts before building the command. `convert_text` is the output of
    `<studio> convert --help` when the build has that subcommand. Anything the text does not
    answer stays None and is then passed through unchecked.

    Returns a dict: flags, value_flags, convert, formats, strategies, mask_modes, log_levels.
    """
    def option_flags(text, with_value):
        found = set()
        for line in text.splitlines():
            if not line.lstrip().startswith("-"):
                continue          # option definition lines, not prose or examples
            for flag, inline in re.findall(r"(--[a-z0-9][a-z0-9_-]*)(=\S+)?", line):
                # `re.findall` reports a non-participating group as "", not None
                if (inline != "") == with_value:
                    found.add(flag)
        return found

    def choices(pattern, text):
        match = re.search(pattern, text)
        if not match:
            return None
        items = {item.strip().lower() for item in match.group(1).split(",") if item.strip()}
        return frozenset(items) or None

    value_flags = option_flags(help_text, with_value=True)
    flags = value_flags | option_flags(help_text, with_value=False)

    # A `convert --help` dump may arrive as `help_text` (callers only have one text); its
    # "Output: .ply, .sog, ..." block is then the authoritative format list.
    if not convert_text and re.search(r"(?m)^\s*Output:\s*\.", help_text):
        convert_text = help_text

    listing = ""
    if convert_text:
        match = re.search(r"(?m)^\s*Output:\s*([^\n]+)", convert_text)
        listing = match.group(1) if match else convert_text
    else:
        match = re.search(r"(?m)^\s*convert\s--\s[^\n]*?(\.ply[^\n]*)", help_text)
        listing = match.group(1) if match else ""
    formats = {"ply"} | {
        token.strip().lstrip(".").lower()
        for token in re.split(r"[,\s/]+", listing)
        if token.strip().startswith(".")
    }
    convert_available = bool(convert_text) or bool(
        re.search(r"(?m)^\s*convert\s--\s", help_text)
    )

    return {
        "flags": frozenset(flags) or None,
        "value_flags": frozenset(value_flags) or None,
        "convert": convert_available,
        "formats": frozenset(formats),
        "strategies": choices(r"Optimization strategy:\s*([^\n(]+)", help_text),
        "mask_modes": choices(r"Mask mode:\s*([^\n(]+)", help_text),
        "log_levels": choices(r"Log level:\s*([^\n(]+)", help_text),
    }


def _run_studio_console(command, timeout=20):
    """Run `<studio> --version` / `--help` and return the text ("" on any failure).

    `--help` and `--version` print and exit instead of opening the GUI, and the timeout kills
    a build that answers neither. Never raises: a probe that fails just means the node works
    with "unknown" capabilities, in which case nothing gets filtered or promised.
    """
    options = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "cwd": str(Path(command[0]).parent),
    }
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(command, timeout=timeout, check=False, **options)
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout or ""


def probe_studio_support(executable):
    """Ask the installed Studio what it supports; cached per executable (path, size, mtime).

    Older free builds answer `--version` with "unknown" and may not ship the `convert`
    subcommand at all, so capabilities - not version numbers - drive every fallback here. A
    binary that does not identify itself as LichtFeld Studio (wrong widget value) yields
    "unknown" capabilities: the command is left untouched and a requested SOG/SPZ export
    falls back to PLY with a warning.
    """
    path = Path(executable)
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        key = (str(path), 0, 0)
    cached = _STUDIO_PROBE_CACHE.get(key)
    if cached is not None:
        return cached

    version_text = _run_studio_console([str(path), "--version"])
    help_text = _run_studio_console([str(path), "--help"])
    if version_text and "lichtfeld" not in version_text.lower():
        version_text = ""
    if help_text and "lichtfeld" not in help_text.lower():
        help_text = ""

    capabilities = parse_studio_capabilities(help_text)
    if capabilities["convert"]:
        convert_text = _run_studio_console([str(path), "convert", "--help"])
        if convert_text and "convert" in convert_text.lower():
            convert_formats = parse_studio_capabilities(convert_text)["formats"]
            capabilities["formats"] = capabilities["formats"] | convert_formats
    version = ""
    for line in version_text.splitlines():
        line = line.strip()
        if line:
            match = re.match(r"LichtFeld Studio\s*(.+)", line)
            version = match.group(1).strip() if match else line
            break
    capabilities["version"] = version
    _STUDIO_PROBE_CACHE[key] = capabilities
    return capabilities


def resolve_export_support(support, requested):
    """(format, note): fall back to the plain .ply export when the build cannot convert.

    SOG/SPZ output needs a build with the `convert` subcommand (LichtFeld Studio 0.5+); the
    `.ply` that every build writes stays the result then and the note explains why.
    """
    if requested == "ply":
        return "ply", ""
    if support.get("convert") and requested in support.get("formats", frozenset()):
        return requested, ""
    version = support.get("version") or "unknown version"
    if support.get("convert"):
        reason = (f"cannot write .{requested} (its `convert` supports "
                  f"{', '.join(sorted(support.get('formats', ())))})")
    else:
        reason = "has no `convert` subcommand"
    note = (f"this LichtFeld Studio build ({version}) {reason}; exporting .ply instead - "
            "SOG/SPZ output needs a build with the convert subcommand (LichtFeld Studio 0.5+).")
    return "ply", note


def check_studio_choice(value, supported, label, fallback=None):
    """Validate a widget value against the build's own choice list (None = unchecked).

    Training-shaping options (strategy, mask mode) raise, so the user picks a value the build
    knows instead of silently training differently. `fallback` turns the check into a warning
    for options a build merely labels otherwise (log level).
    """
    if not supported or str(value).lower() in supported:
        return value, ""
    if fallback is None:
        raise ValueError(
            f"This LichtFeld Studio build does not support {label} {value!r}; "
            f"it lists: {', '.join(sorted(supported))}."
        )
    note = (f"{label} {value!r} is not available in this build "
            f"({', '.join(sorted(supported))}); using {fallback!r}.")
    return fallback, note


def filter_supported_flags(command, supported_flags, value_flags=None):
    """Drop flags the installed build does not know; returns (command, dropped flag names).

    Old free builds reject unknown flags outright ("Flag could not be matched: bg-mode"), which
    would abort the whole run, so the probed flag list prunes the command instead. `None` (no
    parsed help) leaves the command untouched.
    """
    if not supported_flags:
        return list(command), []
    value_flags = _LFS_VALUE_FLAGS if value_flags is None else value_flags
    filtered, dropped, index = [], [], 0
    while index < len(command):
        token = command[index]
        name = token.partition("=")[0] if token.startswith("--") else ""
        if name and name not in supported_flags:
            dropped.append(name)
            if "=" not in token and name in value_flags:
                index += 1                       # the value goes with its flag
            index += 1
            continue
        filtered.append(token)
        index += 1
    return filtered, dropped


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
    status_path="",
):
    """Build a Lichtfeld --python-script hook for parameters without CLI flags.

    Studio only exposes Grow Until / Stop Refine / Save Steps / Eval Steps through
    ``lichtfeld.optimization_params()``.  A *headless* build (v0.5.3 among them) hands
    out a parameter object whose ``has_params()`` is False, because the GUI-side
    ``ParameterManager`` that pushes edits into the trainer does not exist in headless
    mode - the trainer keeps its own defaults no matter what the hook writes.

    The hook therefore never raises: Studio logs an exception for *every* iteration,
    which floods the console with tracebacks.  It applies what it can, reports the
    outcome once (to Studio's log and, when ``status_path`` is set, to that JSON file)
    and stops.
    """
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
    encoded_status = repr(str(status_path or ""))
    return (
        "import json\n"
        "import lichtfeld as lf\n\n"
        f"_ENNDEE_SETTINGS = json.loads({encoded_settings})\n"
        f"_ENNDEE_STATUS_PATH = {encoded_status}\n"
        "_ENNDEE_DONE = False\n"
        "_ENNDEE_NO_PARAMS = (\n"
        "    'Enndee: this Lichtfeld build does not expose optimization parameters to a'\n"
        "    ' headless --python-script run (has_params() is false), so Grow Until / Stop'\n"
        "    ' Refine / Save Steps / Eval Steps cannot be applied - Studio defaults apply.'\n"
        ")\n\n\n"
        "def _enndee_report(payload):\n"
        "    if not _ENNDEE_STATUS_PATH:\n"
        "        return\n"
        "    try:\n"
        "        with open(_ENNDEE_STATUS_PATH, 'w', encoding='utf-8') as handle:\n"
        "            json.dump(payload, handle, sort_keys=True)\n"
        "    except Exception:\n"
        "        pass\n\n\n"
        "def _enndee_warn(message):\n"
        "    logger = getattr(lf, 'log', None)\n"
        "    if logger is not None and hasattr(logger, 'warn'):\n"
        "        try:\n"
        "            logger.warn(message)\n"
        "            return\n"
        "        except Exception:\n"
        "            pass\n"
        "    print(message, flush=True)\n\n\n"
        "def _enndee_apply_training_settings(*_args, **_kwargs):\n"
        "    global _ENNDEE_DONE\n"
        "    if _ENNDEE_DONE:\n"
        "        return\n"
        "    _ENNDEE_DONE = True\n"
        "    try:\n"
        "        params = lf.optimization_params()\n"
        "    except Exception as exc:\n"
        "        _enndee_warn('Enndee: lf.optimization_params() failed: ' + repr(exc))\n"
        "        _enndee_report({'applied': False, 'has_params': False, 'reason': repr(exc)})\n"
        "        return\n"
        "    if params is None:\n"
        "        _enndee_warn('Enndee: Lichtfeld has no optimization parameter object for this run.')\n"
        "        _enndee_report({'applied': False, 'has_params': False,\n"
        "                        'reason': 'no parameter object'})\n"
        "        return\n"
        "    try:\n"
        "        has_params = bool(params.has_params())\n"
        "    except Exception:\n"
        "        has_params = False\n"
        "    try:\n"
        "        for name in ('grow_until_iter', 'stop_refine'):\n"
        "            if name in _ENNDEE_SETTINGS:\n"
        "                params.set(name, _ENNDEE_SETTINGS[name])\n"
        "        if 'enable_eval' in _ENNDEE_SETTINGS:\n"
        "            params.enable_eval = True\n"
        "        if 'save_steps' in _ENNDEE_SETTINGS:\n"
        "            params.clear_save_steps()\n"
        "            for step in _ENNDEE_SETTINGS['save_steps']:\n"
        "                params.add_save_step(step)\n"
        "        if 'eval_steps' in _ENNDEE_SETTINGS:\n"
        "            params.clear_eval_steps()\n"
        "            for step in _ENNDEE_SETTINGS['eval_steps']:\n"
        "                params.add_eval_step(step)\n"
        "    except Exception as exc:\n"
        "        _enndee_warn('Enndee: applying the Lichtfeld training settings failed: '\n"
        "                     + repr(exc))\n"
        "        _enndee_report({'applied': False, 'has_params': has_params,\n"
        "                        'reason': repr(exc)})\n"
        "        return\n"
        "    if not has_params:\n"
        "        _enndee_warn(_ENNDEE_NO_PARAMS)\n"
        "    _enndee_report({'applied': True, 'has_params': has_params, 'reason': ''})\n\n\n"
        "# Studio fires training_start before the first iteration; both are registered so\n"
        "# older builds that lack one of them still get the settings applied exactly once.\n"
        "for _enndee_name in ('on_training_start', 'on_iteration_start'):\n"
        "    _enndee_register = getattr(lf, _enndee_name, None)\n"
        "    if callable(_enndee_register):\n"
        "        _enndee_register(_enndee_apply_training_settings)\n"
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


def read_lfs_settings_status(path):
    """Read the settings hook's JSON status report (missing/empty file -> None)."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


LFS_OPTIMIZATION_DEFAULTS = (Path(__file__).resolve().parent.parent
                             / "lfs_optimization_defaults.json")
_LFS_OPTIMIZATION_TEMPLATE = {}


def load_lfs_optimization_template():
    """Studio's complete ``optimization`` config section (it rejects partial configs).

    ``LichtFeld-Studio.exe --config <file>`` reads the parameters *before* the trainer
    exists, so - unlike the ``--python-script`` hook - it also works headless.  Studio's
    parser requires every key of the section, so the template below (dumped from Studio's
    own ``optimization_params().properties()``) is merged with the requested settings.
    """
    if _LFS_OPTIMIZATION_TEMPLATE:
        return dict(_LFS_OPTIMIZATION_TEMPLATE)
    try:
        with open(LFS_OPTIMIZATION_DEFAULTS, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    section = data.get("optimization") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        return {}
    _LFS_OPTIMIZATION_TEMPLATE.update(section)
    return dict(_LFS_OPTIMIZATION_TEMPLATE)


def build_lfs_optimization_section(grow_until_iter=0, stop_refine=0, save_steps=None,
                                   eval_steps=None, enable_eval=False, mask_mode="",
                                   bg_mode=""):
    """Complete ``optimization`` section with the requested settings applied.

    Returns ``{}`` when nothing was requested (or the template is unavailable), so the
    caller can fall back to the ``--python-script`` hook.
    """
    overrides = {}
    if int(grow_until_iter) > 0:
        overrides["grow_until_iter"] = int(grow_until_iter)
    if int(stop_refine) > 0:
        overrides["stop_refine"] = int(stop_refine)
    if save_steps is not None:
        overrides["save_steps"] = sorted({int(step) for step in save_steps})
    if eval_steps is not None:
        overrides["eval_steps"] = sorted({int(step) for step in eval_steps})
    if enable_eval or eval_steps is not None:
        overrides["enable_eval"] = True
        if eval_steps is None and save_steps:
            # Studio's GUI mirrors save steps as evaluation steps when eval is on.
            overrides["eval_steps"] = sorted({int(step) for step in save_steps})
    # Studio stores these two as strings in the config file, not as numbers.
    if mask_mode:
        overrides["mask_mode"] = str(mask_mode)
    if bg_mode:
        overrides["bg_mode"] = str(bg_mode)
    if not overrides:
        return {}
    section = load_lfs_optimization_template()
    if not section:
        return {}
    section.update(overrides)
    return section


def write_lfs_config_file(section, base_config_path=""):
    """Write a Studio config file: the user's config (optional) plus our section."""
    payload = {}
    if base_config_path:
        try:
            with open(base_config_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload.update(loaded)
        except Exception:
            payload = {}
    payload["optimization"] = section
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix="enndee_lfs_config_", delete=False
    )
    path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def describe_lfs_settings_status(payload):
    """Compatibility note for a hook that could not apply its settings (else "")."""
    if not isinstance(payload, dict):
        return ""
    if payload.get("applied") and payload.get("has_params"):
        return ""
    if payload.get("applied"):
        return ("this Studio build does not expose optimization parameters to a headless "
                "--python-script run (has_params() is false), so the Grow Until / Stop "
                "Refine / Save Steps / Eval Steps settings were ignored and Studio's own "
                "defaults applied.")
    reason = str(payload.get("reason") or "")
    return ("the Lichtfeld settings hook could not be applied"
            + (f" ({reason})" if reason else "") + ".")


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
                "export_format": (list(_EXPORT_FORMATS), {
                    "default": "ply",
                    "tooltip": (
                        "Splat format to leave in the training output folder. 'ply' is "
                        "Studio's own training export. 'sog' (SuperSplat) and 'spz' "
                        "(Niantic) additionally run Studio's `convert` subcommand on "
                        "the finished splat right after training, so the compressed "
                        "file appears next to the .ply (which is kept for re-export). "
                        "Older Studio builds without `convert` keep the .ply and log a "
                        "warning instead of failing."
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
        "headless CLI and can leave the splat as .ply, .sog or .spz. Progress is "
        "streamed to the ComfyUI console."
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
        export_format="ply",
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
        requested_export = validate_export_format(export_format)

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
        # Older/free builds differ: probe what this binary actually accepts before promising
        # anything (its `--version` may literally print "unknown") and fall back where needed.
        studio = probe_studio_support(executable)
        export_format, export_note = resolve_export_support(studio, requested_export)
        strategy, strategy_note = check_studio_choice(strategy, studio["strategies"], "strategy")
        mask_mode, mask_note = check_studio_choice(mask_mode, studio["mask_modes"], "mask mode")
        log_level, log_note = check_studio_choice(log_level, studio["log_levels"], "log level",
                                                  fallback="info")
        compatibility_notes = [note for note in (export_note, strategy_note, mask_note, log_note)
                               if note]
        for note in compatibility_notes:
            print(f"[Enndee Lichtfeld] {note}", flush=True)
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

        settings_section = build_lfs_optimization_section(
            grow_until_iter=grow_until_iter,
            stop_refine=stop_refine,
            save_steps=parsed_save_steps,
            eval_steps=parsed_eval_steps,
            enable_eval=effective_enable_eval,
            mask_mode=mask_mode,
            bg_mode=background_mode,
        )
        config_override_path = None
        settings_status_path = ""
        if settings_section:
            # Studio reads --config before the trainer exists, so - unlike the
            # --python-script hook - this also reaches a headless trainer.
            config_override_path = write_lfs_config_file(settings_section, config_path)
            settings_script = ""
        else:
            handle, settings_status_path = tempfile.mkstemp(
                prefix="enndee_lfs_status_", suffix=".json"
            )
            os.close(handle)
            settings_script = build_lfs_settings_script(
                grow_until_iter=grow_until_iter,
                stop_refine=stop_refine,
                save_steps=parsed_save_steps,
                eval_steps=parsed_eval_steps,
                enable_eval=effective_enable_eval,
                status_path=settings_status_path,
            )
            if not settings_script:
                Path(settings_status_path).unlink(missing_ok=True)
                settings_status_path = ""
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
            config_file=str(config_override_path) if config_override_path else config_path,
            centralize_dataset=centralize_dataset,
            resize_factor=image_resize_factor,
            max_image_width=max_image_width,
            disable_downscaling=disable_downscaling,
            python_script="<temporary Lichtfeld settings script>" if settings_script else "",
        )
        command, dropped_flags = filter_supported_flags(command, studio["flags"])
        if dropped_flags:
            note = ("this Studio build does not support "
                    + ", ".join(sorted(set(dropped_flags)))
                    + " - the option(s) were left out (the build's own defaults apply).")
            compatibility_notes.append(note)
            print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
        if settings_script and "--python-script" not in command:
            note = ("this Studio build has no --python-script; the Grow Until / Stop Refine / "
                    "Save Steps / Eval Steps settings are ignored.")
            compatibility_notes.append(note)
            print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
            settings_script = ""
            if settings_status_path:
                Path(settings_status_path).unlink(missing_ok=True)
                settings_status_path = ""
        preview_command = subprocess.list2cmdline(command)
        print(f"[Enndee Lichtfeld] Dataset: {dataset}", flush=True)
        print(f"[Enndee Lichtfeld] Output: {output}", flush=True)
        print(f"[Enndee Lichtfeld] Command: {preview_command}", flush=True)
        if export_format != "ply":
            print(
                f"[Enndee Lichtfeld] Splat export: .{export_format} "
                "(Studio's convert subcommand runs on the trained .ply afterwards)",
                flush=True,
            )

        if preview_only:
            if settings_status_path:
                Path(settings_status_path).unlink(missing_ok=True)
                settings_status_path = ""
            if config_override_path is not None:
                config_override_path.unlink(missing_ok=True)
            summary = "Preview only: dataset validated; training was not started."
            if compatibility_notes:
                summary += " " + " ".join(compatibility_notes)
            if export_format != "ply":
                summary += f" A .{export_format} export would follow the training run."
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

        settings_status = None
        try:
            return_code, tail = run_streaming_command(
                command, cwd=executable.parent
            )
            settings_status = read_lfs_settings_status(settings_status_path)
        finally:
            if settings_script_path is not None:
                settings_script_path.unlink(missing_ok=True)
            if settings_status_path:
                Path(settings_status_path).unlink(missing_ok=True)
            if config_override_path is not None:
                config_override_path.unlink(missing_ok=True)
        if return_code:
            raise RuntimeError(
                f"Lichtfeld Studio training exited with code {return_code}. "
                f"Full log: {log_file}\nLast output:\n{tail or '(no console output)'}"
            )

        settings_note = describe_lfs_settings_status(settings_status)
        if settings_note:
            compatibility_notes.append(settings_note)
            print(f"[Enndee Lichtfeld] Note: {settings_note}", flush=True)
        if config_override_path is not None:
            note = ("Grow Until / Stop Refine / Save Steps / Eval Steps were applied through "
                    "a generated Lichtfeld config file (--config).")
            compatibility_notes.append(note)
            print(f"[Enndee Lichtfeld] Note: {note}", flush=True)

        export_path = None
        export_tail = ""
        if export_format != "ply":
            trained_splat = resolve_trained_splat(output, output_name)
            if trained_splat is None:
                raise RuntimeError(
                    "Lichtfeld training finished but no splat .ply was found in "
                    f"{output}; cannot export .{export_format}. Check the training "
                    f"log: {log_file}"
                )
            export_path = trained_splat.with_suffix(_EXPORT_SUFFIXES[export_format])
            conversion_command = build_conversion_command(executable, trained_splat, export_path)
            print(
                f"[Enndee Lichtfeld] Export: {subprocess.list2cmdline(conversion_command)}",
                flush=True,
            )
            export_return_code, export_tail = run_streaming_command(
                conversion_command, cwd=executable.parent
            )
            if export_return_code:
                raise RuntimeError(
                    f"Lichtfeld .{export_format} conversion exited with code "
                    f"{export_return_code}. Source: {trained_splat}\nLast output:\n"
                    f"{export_tail or '(no console output)'}"
                )
            if not export_path.is_file():
                raise RuntimeError(
                    "Lichtfeld reported a successful conversion but wrote no "
                    f"{export_path.name} next to {trained_splat}."
                )

        exported = f"\nSplat: {export_path}" if export_path is not None else ""
        noted = ("\nNotes: " + " ".join(compatibility_notes)) if compatibility_notes else ""
        summary = (
            f"Lichtfeld training completed successfully ({int(iterations):,} configured "
            f"iterations). Output: {output}{exported}{noted}\nLog: {log_file}"
        )
        ui_text = [summary, tail] if export_path is None else [summary, export_tail, tail]
        return {"ui": {"text": ui_text},
                "result": (str(output), preview_command, log_file, summary)}
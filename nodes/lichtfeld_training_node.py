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
    "--output-name", "--config", "--python-script", "--test-every",
    # one --add-splat per anchor file, so the path has to go with the flag when it is dropped
    "--add-splat",
    # 0.5.4 supervision + freeze flags: all take a value, so it must go with them
    "--depth-loss-mode", "--depth-loss-weight", "--normal-loss-weight",
    "--normal-consistency-weight", "--normal-flatten-weight", "--normal-loss-space",
    "--normal-start-fraction", "--normal-end-fraction", "--freeze-lr-scale",
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

    def depth_loss_modes(text):
        """Accepted ``--depth-loss-mode`` values, or None when the help does not say.

        0.5.4 prints `Depth prior convention: ssi (auto-detect), ssi-disparity, or
        ssi-depth (default: ssi)`; 0.5.3 named them differently and prints nothing usable,
        in which case this returns None and the node sends the user's value unchanged.
        """
        match = re.search(r"Depth prior convention:\s*([^\n]+)", text)
        if not match:
            return None
        listing = re.split(r"\(default", match.group(1))[0]
        items = set()
        for item in re.split(r",| or ", listing):
            item = re.sub(r"\(.*?\)", "", item).strip().lower()
            if item and " " not in item:
                items.add(item)
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
        # 0.5.3 used adaptive-warped-l1 / pearson; 0.5.4 renamed them to ssi*.
        "depth_loss_modes": depth_loss_modes(help_text),
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


def default_eval_steps(iterations: int) -> list:
    """Evenly spaced evaluation steps that always include the final iteration.

    Studio evaluates **only** at the iterations listed in ``eval_steps``. With an
    empty list it still writes ``metrics_report.txt`` and ``metrics.csv``, but
    both contain nothing but their header - which looks exactly like a broken
    feature. So whenever evaluation is enabled without explicit steps, the node
    fills in these.
    """
    iterations = max(1, int(iterations))
    return sorted({max(1, iterations // 4), max(1, iterations // 2),
                   max(1, 3 * iterations // 4), iterations})


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


#: ``mask_opacity_penalty`` widget labels -> Studio's ``mask_opacity_penalty_weight``.
#: ``None`` leaves Studio's own default (1.0) untouched. The weight penalises
#: gaussian opacity in the masked-out (background) region and is therefore the
#: **soft** dial between "the background fades away" and "the background is fully
#: kept" - use it to prioritise the subject without a hard cut-out.
MASK_OPACITY_PENALTIES = {
    "studio default": None,
    "off": 0.0,
    "low": 1.0,
    "medium": 5.0,
    "high": 15.0,
}

#: ``evaluation`` widget labels -> Studio's ``--test-every`` ("use every Nth image
#: as test"). 0 = no held-out split, all images train. Studio performs this split
#: *internally*, so the evaluation frames stay in the dataset with their poses -
#: there is no separate folder and nothing to exclude on the reconstruction side.
EVALUATION_SPLITS = {
    "off": 0,
    "1/2": 2,
    "1/3": 3,
    "1/4": 4,
}

#: ``subject_mode`` widget -> the two knobs that actually DEFINE the scenario.
#: ``mask_mode`` decides which pixels are supervised at all; ``mask_opacity_penalty``
#: decides how hard the un-supervised background is pushed to low opacity. Everything
#: else (the geometry / anti-inflation cluster) is shared by both scenarios and stays
#: exactly as the individual widgets say. ``None`` = leave the widgets alone.
#:
#: MEASURED, and the reason both scenarios are built on ``segment`` with the penalty ON:
#: the background is a large low-detail region, and if it is not actively pushed down it
#: inflates until the rasterizer's 32-bit (primitive x tile) counter overflows. Tiles per
#: splat at a 1,000,000 cap on the test set:
#:
#:     segment + penalty 1.0   -> survives (12.91 dB / 0.386 SSIM held out at 6000 iters)
#:     segment + penalty 0.0   -> CRASH, 2,424 tiles/splat
#:     none    + penalty 1.0   -> CRASH, 2,261 tiles/splat
#:     alpha_consistent        -> runs, but the render is empty (PSNR 2.52)
#:
#: So the opacity penalty is a STABILITY control as much as a priority dial, and there is
#: no reliable "background fully in the loss" recipe on this Studio build. A higher
#: penalty is always the safer direction (it removes background Gaussians), which is why
#: the hard cut-out can be safely turned up while the soft one cannot be turned down.
SUBJECT_MODES = {
    "subject priority (background kept)": {
        "mask_mode": "segment",
        "mask_opacity_penalty": "low",       # 1.0 - Studio's own value, the measured winner
    },
    "subject cut-out (background removed)": {
        "mask_mode": "segment",
        "mask_opacity_penalty": "high",      # 15.0 - harder push, strictly safer to run
    },
    "custom (use the widgets below)": None,
}


def build_lfs_optimization_section(grow_until_iter=0, stop_refine=0, save_steps=None,
                                   eval_steps=None, enable_eval=False, mask_mode="",
                                   bg_mode="", use_depth_loss=False,
                                   depth_loss_mode="", depth_loss_weight=0.0,
                                   mask_opacity_penalty="studio default",
                                   pause_refine_after_reset=None, scale_reg=None,
                                   scale_decay=None, grad_threshold=None,
                                   growth_grad_threshold=None, lambda_dssim=None,
                                   prune_opacity=None, prune_scale2d=None,
                                   prune_scale3d=None, ppisp=None, bg_modulation=None):
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
    # The soft priority dial: how hard the background is pushed to low opacity.
    penalty = MASK_OPACITY_PENALTIES.get(str(mask_opacity_penalty))
    if penalty is not None:
        overrides["mask_opacity_penalty_weight"] = float(penalty)
    # Geometry-consistency knobs. ``None`` leaves Studio's own value in place, so
    # the section still collapses to {} when nothing at all was requested.
    # NOTE: Studio's schema keeps int and float strictly apart, hence the split.
    if pause_refine_after_reset is not None:
        overrides["pause_refine_after_reset"] = int(pause_refine_after_reset)
    for key, value in (
        ("scale_reg", scale_reg),
        ("scale_decay", scale_decay),
        ("grad_threshold", grad_threshold),
        ("growth_grad_threshold", growth_grad_threshold),
        ("lambda_dssim", lambda_dssim),
        ("prune_opacity", prune_opacity),
        ("prune_scale2d", prune_scale2d),
        ("prune_scale3d", prune_scale3d),
    ):
        if value is not None:
            overrides[key] = float(value)
    if ppisp is not None:
        overrides["ppisp"] = bool(ppisp)
    if bg_modulation is not None:
        overrides["bg_modulation"] = bool(bg_modulation)
    # Depth supervision: only meaningful when the dataset carries a depth/ folder
    # (written by the "VGGT for Lichtfeld (Enndee)" tracker). Both loss modes are
    # affine-invariant, so a relative depth prior is sufficient.
    if use_depth_loss:
        overrides["use_depth_loss"] = True
        if depth_loss_mode:
            overrides["depth_loss_mode"] = str(depth_loss_mode)
        if float(depth_loss_weight) > 0.0:
            overrides["depth_loss_weight"] = float(depth_loss_weight)
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


#: Gaussian scale `mesh2splat` uses for the surface anchors (Studio's own default)
ANCHOR_SIGMA = 0.65

#: mesh file names the anchor lookup accepts inside `<dataset>/mesh/` or the dataset root
ANCHOR_MESH_NAMES = ("dense_mesh.ply", "dense_mesh.obj", "mesh.ply", "mesh.obj")

#: Where `mesh2splat` writes the anchor splats. Deliberately NOT the training output
#: folder: the anchors are an intermediate, and dropping them next to the result would
#: make the output folder non-empty and trip the node's own overwrite guard.
ANCHOR_WORK_DIR = Path(tempfile.gettempdir()) / "enndee_lichtfeld_anchors"


#: COLMAP camera models that carry NO lens distortion, i.e. need no undistort pass
PINHOLE_CAMERA_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})

#: `undistort_cameras` choices: auto reads the dataset, on/off force the flag
UNDISTORT_MODES = ("auto", "on", "off")


def dataset_camera_models(dataset):
    """The camera model names in a dataset's ``cameras.txt`` (empty when unreadable).

    Lichtfeld accepts ``sparse/0/`` and a bare ``sparse/``, so both are probed.
    """
    dataset = Path(dataset)
    for folder in (dataset / "sparse" / "0", dataset / "sparse"):
        cameras = folder / "cameras.txt"
        if not cameras.is_file():
            continue
        models = set()
        try:
            text = cameras.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                models.add(parts[1].upper())
        if models:
            return models
    return set()


def dataset_needs_undistort(dataset):
    """True when the dataset's cameras carry distortion, so Studio needs ``--undistort``.

    Lichtfeld Studio REFUSES a distorted dataset with "Distorted images detected. Use
    --gut or --undistort" - so a COLMAP node run with `SIMPLE_RADIAL` / `RADIAL` / `OPENCV`
    produced a dataset the trainer could not train at all. PINHOLE / SIMPLE_PINHOLE (the
    COLMAP node's default) are fine. Unknown/absent cameras answer False: never add a
    flag we cannot justify.
    """
    models = dataset_camera_models(dataset)
    return bool(models) and not models <= PINHOLE_CAMERA_MODELS


#: Lichtfeld 0.5.3 -> 0.5.4 depth-prior naming. 0.5.4 replaced the two 0.5.3 losses with
#: one auto-detecting `ssi` family, so a saved workflow's old value must be translated
#: instead of sent verbatim (0.5.4 would reject it).
LEGACY_DEPTH_LOSS_MODES = {"adaptive-warped-l1": "ssi", "pearson": "ssi"}


def resolve_depth_loss_mode(mode, supported):
    """Map a `depth_loss_mode` widget value onto one the installed build accepts.

    ``supported`` is the set parsed from `--help` (``studio["depth_loss_modes"]``). When it
    is None the build said nothing usable, so the value is passed through unchanged.
    """
    mode = str(mode or "").strip().lower()
    if not mode or not supported or mode in supported:
        return mode
    translated = LEGACY_DEPTH_LOSS_MODES.get(mode)
    if translated and translated in supported:
        return translated
    if "ssi" in supported:
        return "ssi"
    return mode


def resolve_anchor_mesh(dataset, explicit=""):
    """The mesh to turn into surface anchors, or ``None`` when there is none.

    An explicit path wins. Otherwise the dataset's ``mesh/`` folder (where the COLMAP
    node writes ``dense_mesh.ply``) is searched, then the dataset root - so a dataset
    that carries a surface is picked up without wiring anything.
    """
    explicit = str(explicit or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f"The surface anchor mesh does not exist: {path}")
        return path
    dataset = Path(dataset)
    for folder in (dataset / "mesh", dataset):
        if not folder.is_dir():
            continue
        for name in ANCHOR_MESH_NAMES:
            candidate = folder / name
            if candidate.is_file():
                return candidate
        for pattern in ("*.obj", "*.glb", "*.gltf"):
            hits = sorted(folder.glob(pattern))
            if hits:
                return hits[0]
    return None


def build_mesh2splat_command(executable, mesh, output, resolution: int = 256) -> list:
    """``<studio> mesh2splat <mesh> -o <out> --resolution R --sigma S -y``.

    ``--resolution`` is the *surface rasterisation* target, i.e. the Gaussian budget of
    the anchors: at Studio's 1024 default a full mesh lands near 1M Gaussians, which is
    the whole `max_gaussians` cap. `--sigma` is pinned to the documented default so a
    future Studio release cannot silently change the anchor look.
    """
    return [str(executable), "mesh2splat", str(mesh), "-o", str(output),
            "--resolution", str(int(resolution)), "--sigma", str(ANCHOR_SIGMA), "-y"]


def prepare_surface_anchors(executable, mesh, work_dir, resolution, notes):
    """``mesh2splat`` the mesh into anchor Gaussians; ``[]`` when it does not work out.

    Returns a one-element list of anchor paths, or an empty list - and appends a note to
    ``notes`` explaining why - so a missing mesh or a Studio build without
    ``mesh2splat`` degrades to a plain training run instead of failing the queue.

    ``work_dir`` must be outside the training output folder (see ``ANCHOR_WORK_DIR``):
    the anchors are an intermediate, and a file in the output folder would trip the
    node's "Training output is not empty" guard.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(mesh).stem)[:40] or "mesh"
    anchors = work_dir / f"surface_anchors_{stem}_{int(resolution)}.ply"
    command = build_mesh2splat_command(executable, mesh, anchors, resolution)
    print(f"[Enndee Lichtfeld] Anchors: {subprocess.list2cmdline(command)}", flush=True)
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except Exception as exc:  # noqa: BLE001 - anchors are optional
        note = f"mesh2splat could not run ({type(exc).__name__}: {exc}) - no anchors."
        notes.append(note)
        print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
        return []
    if not anchors.is_file():
        tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-3:]
        note = (f"mesh2splat produced no anchors (rc={proc.returncode}): "
                f"{' | '.join(tail)} - training continues without anchors.")
        notes.append(note)
        print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
        return []
    print(f"[Enndee Lichtfeld] Anchors: {anchors} from {Path(mesh).name} "
          f"({anchors.stat().st_size / 1e6:.1f} MB, resolution {int(resolution)})",
          flush=True)
    return [anchors]


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
    test_every=0,
    save_eval_images=True,
    anchor_splats=(),
    anchor_freeze=False,
    undistort=False,
    use_normal_loss=False,
    normal_loss_weight=0.005,
    normal_consistency_weight=0.001,
    normal_flatten_weight=0.0,
    normal_loss_space="auto",
    normal_start_fraction=0.08,
    normal_end_fraction=1.0,
    freeze_lr_scale=None,
    depth_loss_mode="",
    depth_loss_weight=0.0,
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
    if int(test_every) not in (0,) and int(test_every) < 2:
        raise ValueError(
            "Test Every must be 0 (off) or at least 2 - 'use every Nth image as test' "
            "with N=1 would leave no training images."
        )
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
    # `test_every` splits the dataset inside Studio ("use every Nth image as test"),
    # so the evaluation frames stay in the dataset with their poses.
    if int(test_every) >= 2:
        command.extend(["--test-every", str(int(test_every))])
    if enable_eval and not save_eval_images:
        command.append("--no-save-eval-images")
    if enable_sparsity:
        command.append("--enable-sparsity")
    if undistort:
        # Studio REFUSES a distorted dataset without this ("Distorted images detected.
        # Use --gut or --undistort"), and it adjusts the intrinsics itself.
        command.append("--undistort")
    # Depth / normal supervision as CLI flags. 0.5.4 exposes both on the command line and
    # renamed the depth-prior convention (`adaptive-warped-l1`/`pearson` -> `ssi*`), so the
    # caller passes an already-translated depth_loss_mode (see resolve_depth_loss_mode).
    if depth_loss_mode:
        command.extend(["--depth-loss-mode", str(depth_loss_mode)])
    if float(depth_loss_weight) > 0.0:
        command.extend(["--depth-loss-weight", str(float(depth_loss_weight))])
    if use_normal_loss:
        command.append("--use-normal-loss")
        command.extend(["--normal-loss-weight", str(float(normal_loss_weight))])
        command.extend(["--normal-consistency-weight",
                        str(float(normal_consistency_weight))])
        if float(normal_flatten_weight) > 0.0:
            command.extend(["--normal-flatten-weight", str(float(normal_flatten_weight))])
        if str(normal_loss_space) not in ("", "auto"):
            command.extend(["--normal-loss-space", str(normal_loss_space)])
        if float(normal_start_fraction) != 0.08:
            command.extend(["--normal-start-fraction", str(float(normal_start_fraction))])
        if float(normal_end_fraction) != 1.0:
            command.extend(["--normal-end-fraction", str(float(normal_end_fraction))])
    if log_file:
        command.extend(["--log-file", str(log_file)])
    # Surface anchors. Studio's `--add-splat` appends trained Gaussians *before* the
    # optimiser is built, and `--freeze` freezes "the immediately preceding --add-splat
    # rows" - so one flag per file, with the freeze AFTER the rows it pins.
    for anchor in anchor_splats or ():
        command.extend(["--add-splat", str(anchor)])
    if anchor_splats and anchor_freeze:
        command.append("--freeze")
        # 0.5.4: a frozen splat can still absorb a little appearance mismatch instead of
        # being a hard 0-gradient wall (0 = fully frozen, the default).
        if freeze_lr_scale is not None and float(freeze_lr_scale) > 0.0:
            command.extend(["--freeze-lr-scale", str(float(freeze_lr_scale))])
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
                    "tooltip": (
                        "Number of Lichtfeld optimization iterations. Studio's own "
                        "default is 30000 and the densification schedule is designed for "
                        "it. MEASURED (subject set, 1920 px wide, scaled schedule): "
                        "10000-12000 iterations is the useful range - the winner went "
                        "12.91 dB at 6000, 13.60 at 10000 and only 13.68 at 20000, i.e. "
                        "the last 10000 iterations bought +0.074 dB PSNR while SSIM "
                        "actually fell 0.0075. For A/B comparisons use 6000-10000 (each "
                        "run is ~6 min at 1920 px); the full 30000 only pays off at "
                        "native resolution. If you shorten the run, the densification "
                        "schedule MUST be scaled with it - see Grow Until Iter / Stop "
                        "Refine."
                    ),
                }),
                "strategy": (["mcmc", "mrnf", "igs+"], {
                    "default": "mcmc",
                    "tooltip": (
                        "Lichtfeld optimization strategy. 'mcmc' is the measured one "
                        "(every number in the node's tuning notes comes from it) and is "
                        "the strategy whose densification is driven by the opacity "
                        "resets - so it is also the one that grows screen-filling splats "
                        "when the background is in the loss. 'mrnf' uses the MRNF "
                        "refine/grow schedule (Grow Until Iter, Stop Refine) and 'igs+' "
                        "is the third option; neither was swept in the measurement, so "
                        "their geometry knobs (Scale Reg, Prune Scale2D, Pause Refine "
                        "After Reset) may need different values."
                    ),
                }),
                "max_sh_degree": ("INT", {
                    "default": 3, "min": 0, "max": 3, "step": 1,
                    "tooltip": "Maximum spherical-harmonics degree (0-3).",
                }),
                "max_gaussians": ("INT", {
                    "default": 1000000, "min": 1000, "max": 50000000, "step": 100000,
                    "tooltip": (
                        "Maximum number of trained Gaussians (--max-cap). Keep this at "
                        "1,000,000 - Studio's own default. MEASURED on v0.5.3: the FastGS "
                        "tile rasterizer counts (primitive x tile) pairs in a 32-BIT "
                        "integer and aborts the whole run with 'FastGS instance count "
                        "exceeds 32-bit range' as soon as that passes 2^31 = "
                        "2,147,483,648. That happened at a 2,000,000 cap as soon as the "
                        "splats inflated (1,674 tiles per splat = 15% of the whole tile "
                        "grid each). It is NOT a VRAM limit - only 4.0 of 32.6 GB were in "
                        "use when it died - and it is not really a count limit either: an "
                        "unscaled run reached 5,854,091 Gaussians safely while they stayed "
                        "small. A bigger cap buys nothing except the rope to hang the run "
                        "with. If you need more detail, fix the splat SIZE (Scale Reg, "
                        "Prune Scale2D, Pause Refine After Reset) instead."
                    ),
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
                        "How strictly the exported masks/ (WHITE = subject/keep, "
                        "BLACK = background) are used - this is the subject priority "
                        "control. 'segment' (default) = supervise only inside the mask, "
                        "so the subject gets all the photometric gradient; this is the "
                        "MEASURED winner. 'none' = ignore the masks and splat the full "
                        "picture. WARNING - 'none' did NOT survive on the test build: "
                        "the background is a large low-detail region, MCMC answers it "
                        "with screen-filling splats and the rasterizer's 32-bit "
                        "(primitive x tile) counter overflows ('FastGS instance count "
                        "exceeds 32-bit range: 2,205 tiles per splat'). That happened at "
                        "BOTH caps tested (1M and 2M) and with the full anti-inflation "
                        "cluster on (Scale Reg 0.03, Prune Scale2D 0.1, Pause Refine "
                        "200) - it only improved the tile count by 2.5%, nowhere near "
                        "enough. Lowering Max Gaussians makes it WORSE, not better: "
                        "fewer splats each have to cover more screen. So for a subject "
                        "with a visible background do NOT use 'none' - use 'segment' and "
                        "keep the 'mask opacity penalty' at 1.0 or higher (lowering THAT "
                        "to 0.0 also crashed: 2,424 tiles per splat). 'ignore' = exclude "
                        "the masked region from the loss without forcing it empty. "
                        "'segment_and_ignore' = both (the mask is read as bands, so "
                        "invert_masks is ignored). 'alpha_consistent' = make the "
                        "rendered alpha agree with the mask. Prefer the 'Subject Mode' "
                        "preset over setting this by hand."
                    ),
                }),
                "invert_masks": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask pixels if the foreground/background is reversed.",
                }),
                "mask_opacity_penalty": (list(MASK_OPACITY_PENALTIES), {
                    "default": "studio default",
                    "tooltip": (
                        "How hard Gaussians are pushed to low opacity in the masked-out "
                        "(background) region - the soft subject-priority dial, and "
                        "MEASURED as a stability control too. 'studio default' leaves "
                        "Studio's own 1.0, which is the measured winner; 'low' is also "
                        "1.0; 'medium'/'high' (5/15) approach a hard cut-out. "
                        "'off' (0.0) does NOT keep a nice background - it was measured to "
                        "CRASH: with the background left opaque and unconstrained the "
                        "splats inflated to 2,424 tiles each and the rasterizer's 32-bit "
                        "(primitive x tile) counter overflowed. A HIGHER penalty is "
                        "always the safer direction because it removes background "
                        "Gaussians. So keep this at 1.0 or above; only lower it if you "
                        "are prepared for the run to die. Only has an effect when "
                        "mask_mode is not 'none'."
                    ),
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
                "use_depth_loss": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Enable Lichtfeld's depth supervision (--use-depth-loss). "
                        "Requires a depth/ folder with '<image stem>.depth.png' maps - "
                        "export them with the 'VGGT for Lichtfeld (Enndee)' tracker "
                        "(export_depth on). That tracker predicts poses AND depth in "
                        "one forward pass, so the prior lives in the same gauge as the "
                        "poses and reinforces the geometry instead of fighting it. "
                        "Verify the result with LFS_DEPTH_LOSS_DIAG=1. ON by default "
                        "because it MEASURED better (12.91 vs 12.66 dB at 6000 iters); "
                        "a dataset without depth maps is not an error - the node warns "
                        "and runs without the depth term."
                    ),
                }),
                "depth_loss_mode": (["adaptive-warped-l1", "pearson"], {
                    "default": "adaptive-warped-l1",
                    "tooltip": (
                        "adaptive-warped-l1 (Studio default, and the measured winner) "
                        "fits a per-image scale+shift warp to the prior and then applies "
                        "L1 - robust against a relative, up-to-scale depth prior. "
                        "pearson only maximises the correlation, ignoring scale and shift "
                        "entirely. Both modes are affine-invariant, so either works."
                    ),
                }),
                "depth_loss_weight": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": (
                        "Weight of the depth term. Studio default 2.0; MEASURED 8.0 is "
                        "better - but it is a LONGER-RUN setting, which is the trap: at "
                        "iteration 3000 weight 2 leads by +0.92 dB, and only by 6000 does "
                        "weight 8 overtake it (12.76 vs 12.66). The affine-invariant depth "
                        "loss constrains *shape*, so at weight 8 its gradient dominates "
                        "while the RGB term has not found the layout yet and pays off "
                        "later. Expect a noisier loss (4x the depth gradient) - judge it "
                        "on held-out PSNR/SSIM, never on the logged loss value, and do not "
                        "judge it on a run shorter than ~6000 iterations. Lower it (2-4) "
                        "if the prior is noisy."
                    ),
                }),
                "evaluation": (["off", "1/2", "1/3", "1/4"], {
                    "default": "off",
                    "tooltip": (
                        "Held-out evaluation - a MEASUREMENT, not a refinement: "
                        "Lichtfeld renders the held-out frames at the eval steps and "
                        "reports PSNR/SSIM/LPIPS plus a CSV and a final report. "
                        "Nothing feeds back into the optimisation, so the splat is "
                        "not changed by it. Studio performs the split internally "
                        "(--test-every N: every Nth image becomes a test image), so "
                        "the evaluation frames stay in the dataset with their poses - "
                        "no separate folder, nothing to exclude. Cost: '1/4' means "
                        "25% of the frames no longer train, so use it while comparing "
                        "settings and set it back to 'off' for the final run. "
                        "MEASURED: the evaluation POINT decides the winner - the "
                        "ranking at iteration 3000 was almost the exact reverse of the "
                        "ranking at 6000, and the early leader finished third. Always "
                        "evaluate at the final iteration (Eval Steps does this by "
                        "default) and use at least two points. NOTE: this renders FULL "
                        "frames, so it rewards background reconstruction - a masked "
                        "cut-out subject is scored partly on a background it was told "
                        "not to build."
                    ),
                }),
                "save_eval_images": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Also write the GT-vs-rendered comparison images for every "
                        "eval step (adds --no-save-eval-images when off). OFF by "
                        "default: the PSNR/SSIM/LPIPS numbers are the automatic "
                        "output and need no manual inspection. Switch it on only when "
                        "you want to eyeball structure - metrics can improve while "
                        "floaters get worse, and the image pairs are the only way to "
                        "catch that."
                    ),
                }),
                "pause_refine_after_reset": ("INT", {
                    "default": 200, "min": 0, "max": 5000, "step": 50,
                    "tooltip": (
                        "Geometry consistency: steps after each opacity reset during "
                        "which densification stays paused, so the opacities can "
                        "re-converge before new Gaussians are added. Studio default 0. "
                        "MEASURED (subject set, 6000 iters, held-out PSNR/SSIM): 200 was "
                        "part of the winning anti-inflation cluster (+0.152 dB PSNR and "
                        "+0.020 SSIM over the same run without the cluster). It stops "
                        "fresh Gaussians being added while the old ones are still "
                        "re-learning their opacity, which is exactly when floaters and "
                        "oversized splats get born. 100-500 is the useful range."
                    ),
                }),
                "scale_reg": ("FLOAT", {
                    "default": 0.03, "min": 0.0, "max": 0.2, "step": 0.005,
                    "tooltip": (
                        "Regularisation on the Gaussian scale - the strongest "
                        "anti-inflation knob, and half of the measured win. Big "
                        "stretched Gaussians are the classic way to 'cheat' geometry; a "
                        "higher value punishes them. Studio default 0.01; MEASURED 0.03 "
                        "as part of the winning cluster (+0.152 dB PSNR and +0.020 SSIM "
                        "over the same run without it). It is also the first line of "
                        "defence against the rasterizer's 32-bit (primitive x tile) "
                        "overflow - but MEASURED, it is NOT enough on its own: an "
                        "UNMASKED run with the full cluster still overflowed at 2,205 "
                        "tiles per splat. Only the mask keeps that counter bounded. Too "
                        "high (0.1+) thins out legitimate fine detail."
                    ),
                }),
                "scale_decay": ("FLOAT", {
                    "default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001,
                    "tooltip": (
                        "Shrink applied to Gaussian scales over training. Higher = "
                        "smaller, more surface-hugging splats. Studio default 0.002, "
                        "left unchanged by the measured winner - raise it only if you "
                        "still see blobby surfaces after raising Scale Reg."
                    ),
                }),
                "lambda_dssim": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Weight of the SSIM/DSSIM term relative to L1. DSSIM is a "
                        "*structure* metric. Studio default 0.2; MEASURED 0.3 as part of "
                        "the winning cluster, and it is one of the two changes that "
                        "improved PSNR *and* SSIM together (a trade-off-free win, i.e. "
                        "smaller and better-placed splats rather than a metric trick). "
                        "0.3-0.4 is the useful range; too high blurs fine detail."
                    ),
                }),
                "grad_threshold": ("FLOAT", {
                    "default": 0.0002, "min": 0.0, "max": 0.01, "step": 0.00005,
                    "tooltip": (
                        "Gradient threshold used by the densification/pruning "
                        "schedule. Studio default 0.0002."
                    ),
                }),
                "growth_grad_threshold": ("FLOAT", {
                    "default": 0.003, "min": 0.0, "max": 0.05, "step": 0.0005,
                    "tooltip": (
                        "Gradient threshold above which a Gaussian is split or "
                        "cloned. Lower = more aggressive densification (more detail "
                        "in under-covered areas, more floaters). Studio default "
                        "0.003."
                    ),
                }),
                "prune_opacity": ("FLOAT", {
                    "default": 0.005, "min": 0.0, "max": 0.2, "step": 0.001,
                    "tooltip": (
                        "Opacity below which a Gaussian is pruned. Higher removes "
                        "faint floaters more aggressively. Studio default 0.005."
                    ),
                }),
                "prune_scale2d": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Screen-space scale above which a Gaussian is pruned. Removes "
                        "the big screen-filling blobs that hide bad geometry. Studio "
                        "default 0.15; MEASURED 0.1 as part of the winning "
                        "anti-inflation cluster - a LOWER threshold prunes MORE "
                        "aggressively, so oversized splats are removed earlier. This is "
                        "the second half of the fix for the int32 rasterizer overflow "
                        "(see Max Gaussians). CAUTION: only lower this while the "
                        "background is being supervised - with a masked-out (unmatched) "
                        "background nothing pulls the pruned splats back and the "
                        "background degrades."
                    ),
                }),
                "prune_scale3d": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "World-space scale above which a Gaussian is pruned. Removes "
                        "large stretched splats. Studio default 0.1."
                    ),
                }),
                "ppisp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Per-primitive intrinsic/appearance modelling. It adds "
                        "appearance freedom that can absorb *geometric* error, so "
                        "leave it off when you care about structural consistency. "
                        "Studio default off."
                    ),
                }),
                "bg_modulation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Modulate the background colour during training. Same "
                        "reasoning as ppisp: extra appearance freedom can hide "
                        "geometry problems. Studio default off."
                    ),
                }),
                # Newest control, kept LAST so older saved workflows keep mapping their
                # widget_values onto the existing inputs.
                "subject_mode": (list(SUBJECT_MODES), {
                    "default": "subject priority (background kept)",
                    "tooltip": (
                        "The two scenarios that matter, as one choice. It sets ONLY Mask "
                        "Mode + Mask Opacity Penalty; the geometry knobs stay as you set "
                        "them. "
                        "'subject priority (background kept)': the subject gets all the "
                        "photometric gradient and the background is only softly faded, so "
                        "the surrounding scene stays visible at reduced presence. This is "
                        "the MEASURED winner - 12.91 dB / 0.386 SSIM held out at 6000 "
                        "iterations, and 13.68 / 0.428 at 20000. "
                        "'subject cut-out (background removed)': the background is pushed "
                        "to nothing for a clean cut-out splat to composite. Same mask "
                        "mode, a much harder opacity penalty - it was not run separately, "
                        "but a higher penalty only ever REMOVES background Gaussians, so "
                        "it is strictly safer for the rasterizer than the measured one. "
                        "'custom': leave Mask Mode / Mask Opacity Penalty exactly as the "
                        "widgets below say. "
                        "Both presets keep mask_mode='segment' and keep the opacity "
                        "penalty ON, and that is a measured constraint, not a "
                        "simplification: the background is a big low-detail region and if "
                        "it is not actively pushed down it inflates until the rasterizer's "
                        "32-bit (primitive x tile) counter overflows. Tiles per splat at a "
                        "1,000,000 cap: segment+penalty 1.0 SURVIVES, segment+penalty 0.0 "
                        "CRASHED at 2,424, none+penalty 1.0 CRASHED at 2,261, and "
                        "alpha_consistent ran but rendered an empty picture (PSNR 2.52). "
                        "So there is no reliable 'background fully in the loss' recipe "
                        "here - keep the penalty at 1.0 or higher."
                    ),
                }),
                # Surface anchors. Appended LAST for the same workflow-compatibility
                # reason as subject_mode.
                "use_surface_anchors": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Rasterise a surface mesh into Gaussians and train with them "
                        "pinned as scaffolding. Studio's 'mesh2splat' turns the mesh "
                        "into splats, then '--add-splat' loads them before the "
                        "optimiser is built and '--freeze' stops them ever moving or "
                        "densifying. This is the STRONGEST consistency lever there is: "
                        "the surface is a hard constraint, so the Gaussians cannot "
                        "drift off it - and equally, a wrong surface stays wrong "
                        "forever. Pair it with the COLMAP node's 'mesh_dense_surface' "
                        "(writes mesh/dense_mesh.ply), which is auto-detected. "
                        "Measured anchor budgets at mesh2splat --resolution 256: ~121k "
                        "Gaussians from a Poisson mesh, ~97k from a Delaunay one - "
                        "about a tenth of the 1,000,000 cap, so the rest of the "
                        "budget is still free for the actual scene."
                    ),
                }),
                "surface_anchor_mesh": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to the mesh to use for the anchors (.ply, .obj, .glb...). "
                        "EMPTY (recommended) auto-detects: <dataset>/mesh/dense_mesh.ply "
                        "first - which is exactly what the COLMAP node writes - then "
                        "<dataset>/mesh/*.obj and the dataset root. Point it elsewhere "
                        "to anchor against any mesh you like."
                    ),
                }),
                "anchor_resolution": ("INT", {
                    "default": 256, "min": 64, "max": 2048, "step": 64,
                    "tooltip": (
                        "mesh2splat's surface rasterisation target - the anchor "
                        "Gaussian budget. MEASURED on a real 167k-vertex Poisson mesh: "
                        "128 -> 30k anchors, 256 -> 121k, 512 -> 486k. Studio's own "
                        "default is 1024, which lands at roughly 1M - the entire "
                        "max_gaussians cap, and frozen anchors cannot be pruned, so "
                        "the training has nothing left to grow into. 256 is the "
                        "recommended starting point; raise it only with max_gaussians."
                    ),
                }),
                "anchor_freeze": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON (recommended): '--freeze' - the anchor Gaussians get no "
                        "gradients and are never densified or pruned, so they act as a "
                        "fixed skeleton the rest of the splat must agree with. This is "
                        "the whole point of anchors and the strongest anti-drift "
                        "measure available here. OFF: the anchors are only a warm start "
                        "- Studio may move, grow or delete them like any other "
                        "Gaussian, which is far gentler but gives up the constraint."
                    ),
                }),
                "undistort_cameras": (list(UNDISTORT_MODES), {
                    "default": "auto",
                    "tooltip": (
                        "Lichtfeld Studio REFUSES a dataset whose cameras carry lens "
                        "distortion ('Distorted images detected. Use --gut or "
                        "--undistort'), which is exactly what the COLMAP node produces "
                        "when its camera_model is SIMPLE_RADIAL / RADIAL / OPENCV - so "
                        "without this the two nodes could not be used together at all. "
                        "'auto' (recommended) reads sparse/0/cameras.txt and adds "
                        "'--undistort' only when the model actually has distortion "
                        "(PINHOLE / SIMPLE_PINHOLE, the COLMAP node's default, need "
                        "nothing). 'on' always adds it, 'off' never does. Studio "
                        "undistorts on the fly and adjusts the intrinsics itself."
                    ),
                }),
                # ---- Lichtfeld 0.5.4: normal supervision -------------------
                "use_normal_loss": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Lichtfeld 0.5.4+. Supervise the surface with per-pixel NORMAL "
                        "maps, which is the strongest soft geometry constraint the "
                        "trainer has: normals pin the SURFACE ORIENTATION, which a depth "
                        "map alone cannot (depth says where, normals say which way it "
                        "faces - and that is what floaters and 'double surfaces' get "
                        "wrong). Needs a 'normals/' folder in the dataset; run Studio's "
                        "'preprocess' (MoGe-2, depth + normals in one forward pass, so "
                        "they agree with each other) or leave Studio's own auto-generate "
                        "on. Needs a 0.5.4 build - older builds drop the flags."
                    ),
                }),
                "normal_loss_weight": ("FLOAT", {
                    "default": 0.005, "min": 0.0, "max": 0.5, "step": 0.001,
                    "tooltip": (
                        "Studio's own default (0.005) and a sensible start. Normals are a "
                        "much more direct constraint than depth, so this stays two orders "
                        "of magnitude below 'Depth Loss Weight' - pushing it up is how "
                        "you trade photometric sharpness for surface correctness. Raise "
                        "it if the surface stays blobby; lower it if textures go flat."
                    ),
                }),
                "normal_consistency_weight": ("FLOAT", {
                    "default": 0.001, "min": 0.0, "max": 0.2, "step": 0.001,
                    "tooltip": (
                        "Studio's own default (0.001). Ties the DEPTH and NORMAL priors "
                        "together so they cannot contradict each other - the two are "
                        "generated in one MoGe-2 pass, so disagreement between them is a "
                        "sign that one of the two is being over-weighted. This is the "
                        "cheapest way to keep the pair honest."
                    ),
                }),
                "normal_flatten_weight": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Studio's default is 0 (off). While normal supervision is active "
                        "it flattens the smallest Gaussian axis, i.e. it attacks "
                        "NEEDLE-SHAPED splats - the ones that exist to explain one "
                        "disagreeing view and that read as spikes and floaters. Worth "
                        "trying when a scene looks correct but 'hairy'. It only applies "
                        "during the normal-supervision window, so it cannot damage the "
                        "final iterations."
                    ),
                }),
                "normal_loss_space": (["auto", "camera-opencv", "camera-opengl", "world"], {
                    "default": "auto",
                    "tooltip": (
                        "Coordinate space the normal prior is compared in. 'auto' lets "
                        "Studio decide and is right unless you know your maps are in a "
                        "specific frame. 'world' is the interesting one for multi-view "
                        "consistency (the normals are then anchored to the "
                        "reconstruction, not to each camera), but a monocular estimator "
                        "like MoGe-2 predicts CAMERA-space normals, so 'auto' / "
                        "'camera-opencv' is what those maps actually mean."
                    ),
                }),
                "freeze_lr_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Lichtfeld 0.5.4+. Only used when Surface Anchors are frozen. "
                        "0 (default) is the hard freeze: the anchor Gaussians get no "
                        "gradients at all. 0.01-0.1 lets them absorb a small APPEARANCE "
                        "mismatch (exposure, white balance, a slightly wrong colour) "
                        "without moving the geometry much - the middle ground between "
                        "'anchors fight the images' and 'anchors are gone'. Geometry-wise "
                        "the hard freeze is still the stronger constraint."
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
        max_gaussians=1000000,
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
        use_depth_loss=True,
        depth_loss_mode="adaptive-warped-l1",
        depth_loss_weight=8.0,
        mask_opacity_penalty="studio default",
        evaluation="off",
        save_eval_images=True,
        pause_refine_after_reset=200,
        scale_reg=0.03,
        scale_decay=0.002,
        lambda_dssim=0.3,
        grad_threshold=0.0002,
        growth_grad_threshold=0.003,
        prune_opacity=0.005,
        prune_scale2d=0.1,
        prune_scale3d=0.1,
        ppisp=False,
        bg_modulation=False,
        subject_mode="subject priority (background kept)",
        use_surface_anchors=False,
        surface_anchor_mesh="",
        anchor_resolution=256,
        anchor_freeze=True,
        undistort_cameras="auto",
        use_normal_loss=False,
        normal_loss_weight=0.005,
        normal_consistency_weight=0.001,
        normal_flatten_weight=0.0,
        normal_loss_space="auto",
        freeze_lr_scale=0.0,
    ):
        iterations = int(iterations)
        if iterations < 1:
            raise ValueError("Training iterations must be at least 1.")
        if steps_scaler <= 0:
            raise ValueError("Training steps scaler must be greater than 0.")
        # The scenario preset owns exactly two knobs (mask mode + mask opacity penalty)
        # and nothing else, so resolve it before any mask validation and before the
        # optimization config section is built.
        preset = SUBJECT_MODES.get(str(subject_mode))
        if preset:
            overridden = []
            if preset["mask_mode"] != mask_mode:
                overridden.append(f"mask mode {mask_mode} -> {preset['mask_mode']}")
            mask_mode = preset["mask_mode"]
            if preset["mask_opacity_penalty"] != mask_opacity_penalty:
                overridden.append(
                    f"mask opacity penalty {mask_opacity_penalty} -> "
                    f"{preset['mask_opacity_penalty']}"
                )
            mask_opacity_penalty = preset["mask_opacity_penalty"]
            if overridden:
                print(f"[Enndee Lichtfeld] Subject Mode '{subject_mode}' sets: "
                      + "; ".join(overridden), flush=True)
        validate_refinement_steps(grow_until_iter, stop_refine, iterations)
        parsed_save_steps = parse_iteration_steps(save_steps, "Save Steps", iterations)
        parsed_eval_steps = parse_iteration_steps(eval_steps, "Eval Steps", iterations)
        # Studio splits the dataset itself for evaluation ("use every Nth image as
        # test"), so the evaluation frames stay in images/ + sparse/ and keep their
        # poses - nothing to exclude on the reconstruction side.
        test_every = EVALUATION_SPLITS.get(str(evaluation), 0)
        effective_enable_eval = bool(enable_eval or test_every >= 2
                                     or parsed_eval_steps is not None)
        # Studio evaluates ONLY at the iterations listed in eval_steps: with an
        # empty list it still writes a report, but one with no rows at all. So
        # whenever evaluation is on and no steps were requested, spread a few
        # evenly across the run and always include the final iteration.
        if effective_enable_eval and parsed_eval_steps is None:
            parsed_eval_steps = default_eval_steps(iterations)
            print("[Enndee Lichtfeld] Evaluation is on but no Eval Steps were given - "
                  f"using {parsed_eval_steps} (otherwise the report stays empty).",
                  flush=True)
        requested_export = validate_export_format(export_format)

        dataset = validate_dataset(dataset_path)
        # The evaluation split is announced here, where `dataset` is known.
        if test_every >= 2:
            image_dir = dataset / "images"
            frames = ([path for path in image_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")]
                      if image_dir.is_dir() else [])
            if frames:
                test_count = len(frames) // test_every
                train_count = len(frames) - test_count
                print(f"[Enndee Lichtfeld] Evaluation split: {train_count} training / "
                      f"{test_count} test frames (every {test_every}th image is test)",
                      flush=True)
                if train_count < 20:
                    print(f"[Enndee Lichtfeld] WARNING: only {train_count} training "
                          f"frames left - the splat will degrade noticeably. Lower the "
                          f"evaluation ratio or raise the tracker's frame_step.",
                          flush=True)
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
        depth_note = ""
        if use_depth_loss:
            # Lichtfeld only sees the depth maps if they sit next to the images as
            # '<image stem>.depth.png' - without them the loss would silently do nothing.
            # The depth loss is ON by default because it measured better, and it is an
            # enhancement rather than a requirement - so a dataset without depth maps
            # (e.g. a GLOMAP-only one) downgrades to a warning instead of failing the run.
            depth_folders = (dataset / "depth", dataset / "depths")
            if not any(folder.is_dir() and any(folder.glob("*.depth.png"))
                       for folder in depth_folders):
                print(
                    "[Enndee Lichtfeld] WARNING: the depth loss is enabled but no "
                    f"'<image stem>.depth.png' maps were found under {depth_folders[0]} "
                    "(or 'depths/') - continuing WITHOUT the depth term. Export depth "
                    "with the 'VGGT for Lichtfeld (Enndee)' tracker (export_depth on) to "
                    "get it, or switch Use Depth Loss off to silence this.",
                    flush=True,
                )
                use_depth_loss = False
                depth_note = ("depth loss skipped: the dataset carries no "
                              "'<image stem>.depth.png' maps")
        executable = resolve_studio_executable(studio_executable)
        # Older/free builds differ: probe what this binary actually accepts before promising
        # anything (its `--version` may literally print "unknown") and fall back where needed.
        studio = probe_studio_support(executable)
        export_format, export_note = resolve_export_support(studio, requested_export)
        strategy, strategy_note = check_studio_choice(strategy, studio["strategies"], "strategy")
        mask_mode, mask_note = check_studio_choice(mask_mode, studio["mask_modes"], "mask mode")
        log_level, log_note = check_studio_choice(log_level, studio["log_levels"], "log level",
                                                  fallback="info")
        compatibility_notes = [note for note in (export_note, strategy_note, mask_note, log_note,
                                                 depth_note)
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

        # 0.5.3 named the depth-prior conventions `adaptive-warped-l1` / `pearson`, 0.5.4
        # replaced them with the `ssi` family. Translate once, here, so BOTH the config file
        # and the command line carry a value the installed build accepts.
        supported_depth_modes = studio.get("depth_loss_modes") if studio else None
        depth_loss_mode = resolve_depth_loss_mode(depth_loss_mode, supported_depth_modes)
        if supported_depth_modes and depth_loss_mode not in supported_depth_modes:
            note = (f"this Studio build does not accept depth loss mode '{depth_loss_mode}' "
                    f"(it offers {', '.join(sorted(supported_depth_modes))}) - "
                    "the build's own default is used.")
            compatibility_notes.append(note)
            print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
            depth_loss_mode = ""

        settings_section = build_lfs_optimization_section(
            grow_until_iter=grow_until_iter,
            stop_refine=stop_refine,
            save_steps=parsed_save_steps,
            eval_steps=parsed_eval_steps,
            enable_eval=effective_enable_eval,
            mask_mode=mask_mode,
            bg_mode=background_mode,
            use_depth_loss=use_depth_loss,
            depth_loss_mode=depth_loss_mode,
            depth_loss_weight=depth_loss_weight,
            mask_opacity_penalty=mask_opacity_penalty,
            pause_refine_after_reset=pause_refine_after_reset,
            scale_reg=scale_reg,
            scale_decay=scale_decay,
            grad_threshold=grad_threshold,
            growth_grad_threshold=growth_grad_threshold,
            lambda_dssim=lambda_dssim,
            prune_opacity=prune_opacity,
            prune_scale2d=prune_scale2d,
            prune_scale3d=prune_scale3d,
            ppisp=ppisp,
            bg_modulation=bg_modulation,
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
        # ---- distorted cameras: Studio refuses them without --undistort ----
        if undistort_cameras not in UNDISTORT_MODES:
            raise ValueError(
                f"Undistort Cameras must be one of {', '.join(UNDISTORT_MODES)}."
            )
        undistort = undistort_cameras == "on"
        if undistort_cameras == "auto":
            undistort = dataset_needs_undistort(dataset)
            if undistort:
                models = ", ".join(sorted(dataset_camera_models(dataset)))
                print(f"[Enndee Lichtfeld] {models} carries lens distortion - adding "
                      "--undistort (Studio refuses a distorted dataset otherwise).",
                      flush=True)
        # ---- surface anchors: rasterise the mesh, then hand it to --add-splat ----
        anchor_splats = []
        if use_surface_anchors:
            mesh = None
            try:
                mesh = resolve_anchor_mesh(dataset, surface_anchor_mesh)
            except ValueError as exc:
                note = str(exc)
                compatibility_notes.append(note)
                print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
            if mesh is None:
                note = ("Surface anchors are on but no mesh was found - looked for "
                        f"{Path(dataset) / 'mesh' / 'dense_mesh.ply'} and any .obj/.glb "
                        "next to the dataset. Training continues WITHOUT anchors; switch "
                        "the COLMAP node's 'mesh_dense_surface' on to create one.")
                compatibility_notes.append(note)
                print(f"[Enndee Lichtfeld] Note: {note}", flush=True)
            else:
                anchor_splats = prepare_surface_anchors(
                    executable, mesh, ANCHOR_WORK_DIR, int(anchor_resolution),
                    compatibility_notes,
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
            test_every=test_every,
            save_eval_images=save_eval_images,
            enable_sparsity=enable_sparsity,
            log_level=log_level,
            output_name=output_name,
            config_file=str(config_override_path) if config_override_path else config_path,
            centralize_dataset=centralize_dataset,
            resize_factor=image_resize_factor,
            max_image_width=max_image_width,
            disable_downscaling=disable_downscaling,
            python_script="<temporary Lichtfeld settings script>" if settings_script else "",
            anchor_splats=anchor_splats,
            anchor_freeze=bool(anchor_freeze),
            undistort=bool(undistort),
            use_normal_loss=bool(use_normal_loss),
            normal_loss_weight=float(normal_loss_weight),
            normal_consistency_weight=float(normal_consistency_weight),
            normal_flatten_weight=float(normal_flatten_weight),
            normal_loss_space=str(normal_loss_space),
            freeze_lr_scale=float(freeze_lr_scale),
            depth_loss_mode=str(depth_loss_mode or ""),
            depth_loss_weight=float(depth_loss_weight) if use_depth_loss else 0.0,
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
"""Save files without ComfyUI's running counter - number only on collision.

ComfyUI appends a running number to EVERY saved file: ``folder_paths.
get_save_image_path`` computes ``counter`` (max existing + 1) and each save
node formats it unconditionally, e.g. ``f"{filename}_{counter:05}_.png"``
(core) or ``f"{filename}_{counter:05}.{ext}"`` (VHS). There is no core option
to disable it.

This module wraps the save methods of the nodes that matter and renames the
files AFTER they were written, updating the UI result entries in place so the
frontend previews keep working::

    MiniMax_H3_00001_.mp4   ->   MiniMax_H3.mp4       (free)
                             ->   MiniMax_H3_1.mp4     (already exists)

Covered nodes:

- core ``SaveImage`` (and its ``PreviewImage`` subclass - temp files are
  deliberately left untouched) and ``SaveLatent``
- core io-style ``SaveVideo`` / ``SaveWEBM`` (``execute`` classmethod)
- VHS ``VHS_VideoCombine`` (``combine_video``), including sibling files of
  the same save run such as ``-audio`` variants and poster frames

Sibling files sharing the counter stem are renamed together. Only
``type == "output"`` entries are touched; temp previews keep their names.

Limits: only the standard 5-digit counters (``_00001`` / ``_00001_``) are
stripped - a user prefix ending in an exact ``_12345_`` 5-digit block would
be affected too (6-digit DateTime suffixes like ``_143022`` are safe).

Disable completely by setting the environment variable
``ENNDEE_KEEP_FILE_COUNTER=1``.
"""

import functools
import inspect
import os
import re
import sys

import folder_paths

# _00001 (core: followed by "_." before the extension; VHS: followed by
# ".", "-audio" ...). Exactly five digits, never part of a longer digit run.
# The trailing underscore is handled in strip_counter so that neighbour
# counter blocks (run_00042_00001_.png) stay parseable.
_COUNTER_RE = re.compile(r"_\d{5}(?!\d)")

_LOG_PREFIX = "[Enndee] unique filenames: "

_patched_note = set()


def strip_counter(filename):
    """Remove the LAST counter block of a save name; keep everything else."""
    matches = list(_COUNTER_RE.finditer(filename))
    if not matches:
        return filename
    last = matches[-1]
    end = last.end()
    if end < len(filename) and filename[end] == "_":
        end += 1  # core style "_00001_" - swallow the trailing underscore
    return filename[: last.start()] + filename[end:]


def uniquify(folder, filename):
    """Return ``filename`` or ``base_1.ext`` / ``base_2.ext`` ... if taken."""
    if not os.path.exists(os.path.join(folder, filename)):
        return filename
    base, ext = os.path.splitext(filename)
    for n in range(1, 10000):
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(os.path.join(folder, candidate)):
            return candidate
    return filename


def rename_family(folder, filename):
    """Rename ``filename`` and every sibling sharing its counter stem.

    Returns the ``{old: new}`` mapping for renamed files. Siblings cover
    multi-file saves (VHS ``-audio`` variants, poster frames).
    """
    stem = os.path.splitext(filename)[0]
    if not stem:
        return {}
    try:
        existing = os.listdir(folder)
    except OSError:
        return {}
    # The listed result file first, deterministic order for the rest.
    family = sorted(
        (f for f in existing if f.startswith(stem)),
        key=lambda f: (f != filename, f),
    )
    mapping = {}
    for old in family:
        new = strip_counter(old)
        if new == old:
            continue
        new = uniquify(folder, new)
        if new == old:
            continue
        try:
            os.rename(os.path.join(folder, old), os.path.join(folder, new))
        except OSError as error:
            _log_once(f"could not rename {old!r}: {error}")
            continue
        mapping[old] = new
    return mapping


def _handle_entry(entry):
    """Rename the file behind one UI result entry (output files only)."""
    name = entry.get("filename")
    if not name or entry.get("type") != "output":
        return
    root = folder_paths.get_output_directory()
    folder = os.path.join(root, entry.get("subfolder") or "")
    mapping = rename_family(folder, name)
    if name in mapping:
        entry["filename"] = mapping[name]


def _walk(obj, depth=0):
    """Find file-entry dicts inside UI result structures."""
    if depth > 8:
        return
    if isinstance(obj, dict):
        if "filename" in obj and "type" in obj:
            _handle_entry(obj)
            return
        for value in obj.values():
            _walk(value, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, depth + 1)
    else:
        # ui.PreviewVideo.values / ui.SavedImages.results / ui.SavedAudios.results
        for attr in ("values", "results"):
            values = getattr(obj, attr, None)
            if isinstance(values, (list, tuple)):
                _walk(values, depth + 1)


def _process_result(result):
    """Post-process a node return value (legacy dict or io.NodeOutput)."""
    if isinstance(result, dict):
        ui = result.get("ui")
        if ui is not None:
            _walk(ui)
        return result
    ui = getattr(result, "ui", None)
    if ui is not None:
        _walk(ui)
    return result


def _safe_process(result):
    try:
        _process_result(result)
    except Exception as error:  # never break a save
        _log_once(f"post-rename failed: {error}")
    return result


def _log_once(message):
    if message in _patched_note:
        return
    _patched_note.add(message)
    print(_LOG_PREFIX + message)


def _wrap_callable(cls, name):
    """Wrap ``cls.name`` so its return value is post-processed once."""
    try:
        static = inspect.getattr_static(cls, name)
    except AttributeError:
        return False
    bound = getattr(cls, name)
    if getattr(bound, "_enndee_unique", False):
        return True
    is_classmethod = isinstance(static, classmethod)

    if is_classmethod:
        def wrapper(cls_, *args, **kwargs):
            return _safe_process(bound(*args, **kwargs))
    else:
        @functools.wraps(bound)
        def wrapper(*args, **kwargs):
            return _safe_process(bound(*args, **kwargs))

    wrapper._enndee_unique = True
    setattr(cls, name, classmethod(wrapper) if is_classmethod else wrapper)
    return True


def _install_core_targets():
    """Wrap the core save nodes; always finished (core modules are loaded)."""
    try:
        import nodes as core_nodes

        for class_name in ("SaveImage", "SaveLatent"):
            cls = getattr(core_nodes, class_name, None)
            if cls is not None:
                _wrap_callable(cls, getattr(cls, "FUNCTION", ""))
    except Exception as error:
        _log_once(f"core SaveImage/SaveLatent wrap issue: {error}")
    try:
        import comfy_extras.nodes_video as nodes_video

        for class_name in ("SaveVideo", "SaveWEBM"):
            cls = getattr(nodes_video, class_name, None)
            if cls is not None:
                _wrap_callable(cls, "execute")
    except Exception as error:
        _log_once(f"core SaveVideo/SaveWEBM wrap issue: {error}")
    return True


def _install_vhs():
    """Wrap VHS VideoCombine once its pack finished loading."""
    for module in list(sys.modules.values()):
        mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
        if isinstance(mappings, dict) and "VHS_VideoCombine" in mappings:
            cls = mappings["VHS_VideoCombine"]
            _wrap_callable(cls, getattr(cls, "FUNCTION", "combine_video"))
            return True
    return False


_PENDING = [_install_core_targets, _install_vhs]
_HOOK_TRIES = 0


def _run_pending():
    global _PENDING, _HOOK_TRIES
    if not _PENDING:
        return
    _HOOK_TRIES += 1
    remaining = []
    for install_step in _PENDING:
        try:
            done = install_step()
        except Exception as error:
            _log_once(f"install step failed: {error}")
            done = True  # do not retry broken steps forever
        if not done:
            remaining.append(install_step)
    _PENDING = remaining
    if not _PENDING:
        _log_once("all save nodes patched")


def _on_prompt_handler(json_data):
    if _PENDING and _HOOK_TRIES < 500:
        _run_pending()
    return json_data


def install():
    """Enable counter-free saving; returns True when active."""
    if os.environ.get("ENNDEE_KEEP_FILE_COUNTER"):
        print(_LOG_PREFIX + "DISABLED (ENNDEE_KEEP_FILE_COUNTER is set)")
        return False
    _run_pending()
    if _PENDING:
        # VHS may load after this pack - retry right before the first prompt.
        try:
            from server import PromptServer

            if PromptServer.instance is not None:
                PromptServer.instance.add_on_prompt_handler(_on_prompt_handler)
        except Exception as error:
            _log_once(f"no deferred retry hook ({error})")
    print(_LOG_PREFIX + "active - number appended only when the file exists")
    return True


__all__ = [
    "strip_counter",
    "uniquify",
    "rename_family",
    "install",
]



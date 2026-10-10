"""
RMBG background removal for the Lichtfeld tracker nodes (Enndee).
=================================================================

One place that knows how to turn RGB frames into a foreground alpha matte, so
the GLOMAP, COLMAP and VGGT tracker nodes all share it.

Two generations of the BRIA RMBG family are available:

===============================================  ================================
model                                            how it runs
===============================================  ================================
**RMBG-2.0** (``2.0``)                           ``transformers`` +
                                                 ``briaai/RMBG-2.0`` (BiRefNet)
**RMBG-1.4** (``base`` / ``fast`` /              the ``transparent_background``
``base-nightly``)                                package (InSPyReNet)
===============================================  ================================

Why 2.0 is the default
----------------------
RMBG-2.0 is a BiRefNet at 1024x1024 with a much cleaner matte around hair,
motion blur and semi-transparent edges than RMBG-1.4's InSPyReNet.  That matters
here twice over: the alpha becomes the Lichtfeld **splat mask** *and* the
**feature mask**, and the tune in ``Output/VGGT_Tests/enndee_tuning`` showed the
mask is a stability control, not just a priority knob - cleaner edges mean less
eroded foreground and fewer background features.

The three 1.4 ids stay selectable on purpose:

* they need no ``transformers`` download (only the ``transparent_background``
  checkpoint), so an offline machine keeps working;
* they are ~4x faster;
* saved workflows that already selected ``base`` keep validating.

First run downloads
-------------------
``briaai/RMBG-2.0`` is fetched into the Hugging Face cache on first use (~900 MB,
fp32).  ``ENNDEE_RMBG2_REPO`` points the loader somewhere else without touching
the code.  If the load fails (no network, no ``trust_remote_code``, out of VRAM)
the caller falls back to RMBG-1.4 ``base`` with a warning instead of failing the
run.

Threshold semantics (identical for both generations)
----------------------------------------------------
``threshold`` is the *binarisation* point of the probability map, matching
``transparent_background.Remover.process(..., threshold=...)``:

* ``0.5`` (default) -> hard matte, alpha is 0 or 255;
* ``0.3``-``0.4`` -> keeps more foreground, ``0.6``-``0.7`` keeps less;
* ``None`` -> the raw soft matte (API only, the node always passes a float).
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Widget surface
# ---------------------------------------------------------------------------

#: RMBG-2.0 option id (BiRefNet through transformers)
RMBG2_MODE = "2.0"

#: RMBG-1.4 ids, exactly the strings ``transparent_background`` accepts
LEGACY_MODES = ("base", "fast", "base-nightly")

#: Combo values for the ``rmbg_mode`` widget.  2.0 is first so it is the default
#: for new nodes, the legacy ids stay so existing workflows still validate.
RMBG_MODES = [RMBG2_MODE, *LEGACY_MODES]

#: Default mode for every node
RMBG_DEFAULT_MODE = RMBG2_MODE

RMBG_MODE_TOOLTIP = (
    "RMBG model. 2.0 = briaai/RMBG-2.0 (BiRefNet, 1024 px, cleanest matte, "
    "downloads ~900 MB on first use). base = RMBG-1.4 best quality, "
    "fast = RMBG-1.4 quicker with lower quality, base-nightly = newest 1.4 "
    "base build. The three 1.4 modes need no transformers download, so they "
    "are the offline fallback."
)

# ---------------------------------------------------------------------------
# RMBG-2.0 constants
# ---------------------------------------------------------------------------

#: Hub repo of the BiRefNet checkpoint
RMBG2_REPO = "briaai/RMBG-2.0"

#: Square resolution the model was trained at
RMBG2_RESOLUTION = 1024

#: ImageNet statistics used by the RMBG-2.0 preprocessing recipe
RMBG2_MEAN = (0.485, 0.456, 0.406)
RMBG2_STD = (0.229, 0.224, 0.225)

#: Frames per forward pass.  Keeps VRAM flat on small cards; the model is fast
#: enough that batching is not the bottleneck.
RMBG2_CHUNK = 4

# ---------------------------------------------------------------------------
# Caches (module level: a ComfyUI node instance is recreated per execution)
# ---------------------------------------------------------------------------

_rmbg2_model = None
_rmbg2_key = None
_rmbg2_failed = False

_legacy_remover = None
_legacy_key = None


def rmbg2_repo() -> str:
    """Hub repo id of the RMBG-2.0 checkpoint (env override wins).

    ``ENNDEE_RMBG2_REPO`` also accepts a **local directory** holding a copy of
    the checkpoint (``config.json``, ``birefnet.py``, ``model.safetensors``), which
    is the offline path for a machine that may not talk to huggingface.co.
    """
    return (os.environ.get("ENNDEE_RMBG2_REPO") or "").strip() or RMBG2_REPO


def rmbg2_hint(exc: Exception) -> str:
    """One actionable line explaining why RMBG-2.0 would not load.

    ``briaai/RMBG-2.0`` is a **gated** repo (non-commercial licence), which is by
    far the most common failure and deserves more than a raw 401.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "gated" in lowered or "401" in lowered or "authenticated" in lowered:
        return ("briaai/RMBG-2.0 is a gated model - accept the licence at "
                "https://huggingface.co/briaai/RMBG-2.0, then set HF_TOKEN (or "
                "run 'huggingface-cli login') and restart ComfyUI. "
                "ENNDEE_RMBG2_REPO can also point at a local copy.")
    if "offline" in lowered or "connection" in lowered or "resolve" in lowered \
            or "max retries" in lowered:
        return ("huggingface.co is unreachable - download briaai/RMBG-2.0 once on "
                "a connected machine, or set ENNDEE_RMBG2_REPO to a local copy.")
    return text


def _noop(_message: str) -> None:
    return None


# ---------------------------------------------------------------------------
# RMBG-2.0 (BiRefNet through transformers)
# ---------------------------------------------------------------------------


def _load_rmbg2(device: str, log: Callable[[str], None]):
    """Load (and cache) the RMBG-2.0 model on ``device``.

    Raises on any failure so the caller can fall back to RMBG-1.4.
    """
    global _rmbg2_model, _rmbg2_key, _rmbg2_failed

    import torch
    from transformers import AutoModelForImageSegmentation

    repo = rmbg2_repo()
    if _rmbg2_model is not None and _rmbg2_key == (repo, device):
        return _rmbg2_model

    model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
    model.eval()
    model.to(device)
    # BiRefNet runs a large convolution stack at 1024x1024; bf16 halves the
    # memory and matches the reference CUDA implementation.
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)

    _rmbg2_model = model
    _rmbg2_key = (repo, device)
    _rmbg2_failed = False
    log(f"RMBG-2.0 model loaded ({repo}, device={device}, "
        f"{RMBG2_RESOLUTION}x{RMBG2_RESOLUTION})")
    return model


def _rmbg2_logits(output):
    """Pull the highest-resolution map out of whatever the model returned.

    ``AutoModelForImageSegmentation`` hands back an ``ImageSegmentationOutput``
    when the checkpoint ships its own model class, while plain BiRefNet
    implementations return ``[d1, d2, d3]`` (or a nested list of those).  All
    shapes are accepted.
    """
    if hasattr(output, "logits"):
        output = output.logits
    while isinstance(output, (list, tuple)):
        output = output[-1]
    return output


def _rmbg2_alpha_chunk(model, frames: Sequence[np.ndarray], device: str) -> List[np.ndarray]:
    """Probability mattes for one chunk of ``HxWx3`` uint8 frames.

    Returns one ``HxW`` float32 array in [0, 1] per frame, at the *input*
    resolution (the 1024x1024 prediction is resampled back per frame).
    """
    import torch
    import torch.nn.functional as F

    sizes = [(int(frame.shape[0]), int(frame.shape[1])) for frame in frames]
    batch = np.stack([np.ascontiguousarray(frame[..., :3]) for frame in frames])
    tensor = torch.from_numpy(batch).to(torch.float32).permute(0, 3, 1, 2) / 255.0

    tensor = F.interpolate(tensor, size=(RMBG2_RESOLUTION, RMBG2_RESOLUTION),
                           mode="bilinear", align_corners=False)
    mean = torch.tensor(RMBG2_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(RMBG2_STD, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.to(device, dtype=next(model.parameters()).dtype)

    with torch.inference_mode():
        logits = _rmbg2_logits(model(tensor))

    if logits.ndim == 3:                      # [B, H, W] without a channel axis
        logits = logits.unsqueeze(1)
    probabilities = logits.float().sigmoid()

    mattes: List[np.ndarray] = []
    for index, (height, width) in enumerate(sizes):
        single = probabilities[index:index + 1]
        if (int(single.shape[-2]), int(single.shape[-1])) != (height, width):
            single = F.interpolate(single, size=(height, width),
                                   mode="bilinear", align_corners=False)
        mattes.append(single[0, 0].float().cpu().numpy().astype(np.float32))
    return mattes


def _rmbg2_alpha(frames: Sequence[np.ndarray], device: str,
                 log: Callable[[str], None]) -> List[np.ndarray]:
    """Soft mattes for every frame, chunked so VRAM stays bounded."""
    import torch

    model = _load_rmbg2(device, log)
    mattes: List[np.ndarray] = []
    total = len(frames)
    for start in range(0, total, RMBG2_CHUNK):
        chunk = frames[start:start + RMBG2_CHUNK]
        try:
            mattes.extend(_rmbg2_alpha_chunk(model, chunk, device))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if len(chunk) > 1:
                # Retry one frame at a time before giving up on the GPU.
                log(f"RMBG-2.0 chunk of {len(chunk)} failed ({exc}) - "
                    f"retrying frame by frame")
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
                for frame in chunk:
                    mattes.extend(_rmbg2_alpha_chunk(model, [frame], device))
            else:
                raise
        if (start + len(chunk)) % 20 == 0 or start + len(chunk) == total:
            log(f"RMBG {start + len(chunk)}/{total} frames")
    return mattes


# ---------------------------------------------------------------------------
# RMBG-1.4 (transparent_background / InSPyReNet)
# ---------------------------------------------------------------------------


def _load_legacy(mode: str, resize: str, device: str, log: Callable[[str], None]):
    """Load (and cache) a ``transparent_background`` Remover."""
    global _legacy_remover, _legacy_key

    from transparent_background import Remover

    key = (mode, resize, device)
    if _legacy_remover is not None and _legacy_key == key:
        return _legacy_remover

    remover = Remover(mode=mode, device=device, resize=resize)
    _legacy_remover = remover
    _legacy_key = key
    log(f"RMBG-1.4 model loaded (mode={mode}, device={device}, resize={resize})")
    return remover


def _legacy_alpha(frames: Sequence[np.ndarray], mode: str, resize: str,
                  device: str, threshold: Optional[float],
                  log: Callable[[str], None],
                  log_warn: Callable[[str], None]) -> List[np.ndarray]:
    """Alpha mattes from RMBG-1.4, one frame per ``process`` call."""
    remover = _load_legacy(mode, resize, device, log)
    mattes: List[np.ndarray] = []
    total = len(frames)
    for index, frame in enumerate(frames):
        try:
            rgba = np.asarray(remover.process(frame, threshold=threshold),
                              dtype=np.uint8)
            mattes.append(rgba[..., 3].astype(np.float32) / 255.0)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the run
            log_warn(f"RMBG-1.4 failed for frame {index}: {exc}")
            mattes.append(np.ones(frame.shape[:2], dtype=np.float32))
        if (index + 1) % 10 == 0 or index + 1 == total:
            log(f"RMBG {index + 1}/{total} frames")
    return mattes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def remove_background(frames: Sequence[np.ndarray], mode: str = RMBG_DEFAULT_MODE,
                      device: str = "cuda", threshold: Optional[float] = 0.5,
                      resize: str = "static",
                      log: Optional[Callable[[str], None]] = None,
                      log_warn: Optional[Callable[[str], None]] = None,
                      ) -> Optional[List[np.ndarray]]:
    """Foreground mattes for ``frames`` (``HxWx3`` uint8), or ``None``.

    ``None`` means "no background removal happened at all" (nothing installed /
    nothing loadable) - the caller then keeps the frames untouched.  A single
    frame that fails gets a fully-opaque matte instead, so a run never dies
    halfway through a sequence.

    The mattes are ``HxW`` float32 in [0, 1] at the input resolution, already
    thresholded when ``threshold`` is not ``None``.
    """
    global _rmbg2_model, _rmbg2_key, _rmbg2_failed

    log = log or _noop
    log_warn = log_warn or _noop

    frames = [np.asarray(frame) for frame in frames]
    if not frames:
        return None

    mode = str(mode)
    mattes: Optional[List[np.ndarray]] = None

    if mode == RMBG2_MODE:
        if _rmbg2_failed:
            log_warn("RMBG-2.0 already failed in this session - not retrying")
        else:
            candidates = [device, "cpu"] if device != "cpu" else ["cpu"]
            for candidate in candidates:
                try:
                    mattes = _rmbg2_alpha(frames, candidate, log)
                    if candidate != device:
                        log_warn(f"RMBG-2.0 ran on {candidate}, not {device}")
                    break
                except Exception as exc:  # noqa: BLE001 - fall back, never raise
                    _rmbg2_model, _rmbg2_key = None, None
                    if candidate != "cpu":
                        log_warn(f"RMBG-2.0 could not start on {candidate}: {exc}")
                        continue
                    _rmbg2_failed = True
                    log_warn(f"RMBG-2.0 unavailable - {rmbg2_hint(exc)}")
        # A failed 2.0 load must still produce a mask, so continue on 1.4.
        if mattes is None:
            log_warn("Falling back to RMBG-1.4 "
                     f"'{LEGACY_MODES[0]}' for this run")
            mode = LEGACY_MODES[0]

    if mattes is None:
        try:
            mattes = _legacy_alpha(frames, mode, resize, device, threshold,
                                   log, log_warn)
        except ImportError:
            log("transparent_background is not installed - RMBG disabled "
                "(pip install transparent_background)")
            return None
        except Exception as exc:  # noqa: BLE001
            log_warn(f"RMBG could not start: {type(exc).__name__}: {exc}")
            return None

    if threshold is not None:
        cutoff = float(threshold)
        mattes = [(matte >= cutoff).astype(np.float32) for matte in mattes]
    return mattes


__all__ = [
    "LEGACY_MODES",
    "RMBG2_MODE",
    "RMBG2_REPO",
    "RMBG2_RESOLUTION",
    "RMBG_DEFAULT_MODE",
    "RMBG_MODES",
    "RMBG_MODE_TOOLTIP",
    "remove_background",
    "rmbg2_hint",
    "rmbg2_repo",
]

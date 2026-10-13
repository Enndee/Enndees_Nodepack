"""MoGe-3 loader shared by the Meridian fast-depth backend and its nodes.

`moge` is Microsoft's monocular geometry model family (MoGe-1 -> MoGe-2 -> **MoGe-3**,
2026-08-18). One forward pass returns a metric point map, depth, normals and camera FOV, and
MoGe-3's headline addition over MoGe-2 is a *self-guided sparse volumetric refinement* stage:
the coarse point map is voxelised into a sparse 3D grid and walked by a `Sparse3DUNet`, which
is exactly what `infer(..., refine_steps=N)` exposes. `refine_steps=0` disables the stage and
returns the single-pass result; the library default is 3.

Why a separate module (and not just an import in the node)
---------------------------------------------------------
`moge` is *not* a ComfyUI dependency and is deliberately optional: the Meridian Geometry node
only loads it when the picker actually selects MoGe 3, and every other depth source keeps
working without it. :func:`moge_status` therefore never raises, so a node can report
availability without paying for the import, and :func:`load_moge_v3` raises with an install
hint only when a run really needs the weights.

MoGe-3 is heavier to install than Depth-Anything-3: it needs **FlexGEMM** (Triton-based), so
macOS is unsupported and a Windows install needs a Triton build - ComfyUI's `python_embeded`
already ships `triton-windows`, which is why this is worth offering at all.

    git clone https://github.com/microsoft/MoGe.git
    python -m pip install ./MoGe

The checkpoint is ``Ruicheng/moge-3-vitl`` and is **MIT** licensed (the MoGe code is MIT too,
except the vendored DINOv2 modules which are Apache-2.0). Unlike the DA3 wheel there is no
dependency pin to work around - MoGe wants numpy/torch as ComfyUI already has them - so no
import stubbing is needed here.
"""

from __future__ import annotations

from typing import Tuple

#: The only MoGe-3 checkpoint: the ViT-L release, on the Hugging Face hub.
MOGE_V3_REPO = "Ruicheng/moge-3-vitl"

#: MoGe's own version tag, resolved through ``import_model_class_by_version``.
MOGE_V3_VERSION = "v3"

#: Picker label -> backend id, kept here so the model id lives in one place.
MOGE_V3_MODEL_ID = "MoGe-3-ViT-L"

#: `refine_steps` of MoGe-3's self-guided sparse volumetric refinement. 0 disables the
#: stage (a single-pass MoGe-2-style result); 3 is what the library and its CLI default to.
DEFAULT_REFINE_STEPS = 3

#: Ceiling for the widget. The library has no upper bound, but every step is another sparse
#: 3D-UNet pass over the voxelised point map, and past ~12 the extra time buys nothing visible.
MAX_REFINE_STEPS = 12

#: MoGe's `resolution_level` (0-9): higher = more encoder tokens = finer detail, slower. 9 is
#: the library default and what the checkpoint was tuned around; the output map is always the
#: input's own size, so this is the only token knob the fast-depth backend needs.
DEFAULT_RESOLUTION_LEVEL = 9

INSTALL_HINT = ("git clone https://github.com/microsoft/MoGe.git && "
                "python -m pip install ./MoGe   "
                "(MoGe-3 also needs FlexGEMM, which is Triton-based: "
                "https://github.com/JeffreyXiang/FlexGEMM)")


def load_moge_v3():
    """Return the MoGe-3 ``MoGeModel`` class, or raise with an install hint.

    Idempotent in the sense that matters: the heavy work is the caller's
    ``from_pretrained``, and the module import is cached by Python.
    """
    try:
        from moge.model import import_model_class_by_version
    except ImportError as exc:
        raise RuntimeError(
            f"MoGe-3 is not installed for this interpreter. In the ComfyUI "
            f"python_embeded run: {INSTALL_HINT}"
        ) from exc
    return import_model_class_by_version(MOGE_V3_VERSION)


def moge_status() -> Tuple[bool, str]:
    """``(usable, reason)`` - never raises, for the nodes' availability report."""
    try:
        load_moge_v3()
    except Exception as exc:  # noqa: BLE001 - a missing package is a normal state
        return False, f"moge: not usable ({type(exc).__name__}: {exc})"
    return True, (f"moge: ready (MoGe-3 {MOGE_V3_REPO}, "
                  f"refine_steps {DEFAULT_REFINE_STEPS})")


def resolve_refine_steps(value) -> int:
    """Validate ``moge_refine_steps``: 0 disables the refinement stage, MoGe wants >= 0.

    Raising beats clamping here: silently turning a requested 40 into 12 would hide the
    typo, and the widget already caps the range it can produce.
    """
    if value is None or value == "":
        return DEFAULT_REFINE_STEPS
    if isinstance(value, float) and not value.is_integer():
        # int(1.5) would quietly become 1 - the same class of silent wrongness as a mis-scaled
        # depth map, so a fractional request is refused rather than rounded away.
        raise ValueError(f"moge_refine_steps must be a whole number, got {value!r}.")
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"moge_refine_steps must be a whole number, got {value!r}."
        ) from exc
    if not 0 <= steps <= MAX_REFINE_STEPS:
        raise ValueError(
            f"moge_refine_steps must be between 0 and {MAX_REFINE_STEPS} "
            f"(0 disables MoGe-3's refinement stage), got {steps}."
        )
    return steps


__all__ = [
    "DEFAULT_REFINE_STEPS",
    "DEFAULT_RESOLUTION_LEVEL",
    "INSTALL_HINT",
    "MAX_REFINE_STEPS",
    "MOGE_V3_MODEL_ID",
    "MOGE_V3_REPO",
    "MOGE_V3_VERSION",
    "load_moge_v3",
    "moge_status",
    "resolve_refine_steps",
]

"""
Depth Anything 3 loader shared by the Meridian fast-depth node and the VGGT node.
=================================================================================

``depth_anything_3`` is a *mono* **and** an *any-view* model family in one pip
package.  Both users of this pack need the same awkward import dance, so it lives
here once:

* :mod:`enndee_meridian_fast_depth` (Meridian Geometry, ``depth_res`` mode) uses
  the **mono** models for a single still.
* :mod:`enndee_feedforward`'s ``DA3-AnyView`` backend uses the **any-view**
  models as a drop-in replacement for VGGT-Omega: depth *and* camera poses out of
  one forward pass over the whole frame set.

Why the import needs stubbing
-----------------------------
The wheel declares a dependency set meant for a dedicated environment (``numpy<2``,
xformers, open3d, pycolmap, moviepy, gsplat, evo), so it is installed here with
``python -m pip install --no-deps depth-anything-3`` to keep ComfyUI's own
numpy/torch.  Two **eager** imports then stay unsatisfied, and both are dead ends
for inference-only use:

* ``depth_anything_3.utils.export`` - the export dispatcher pulls the glb
  (trimesh), colmap (pycolmap), gs (gsplat) and vis (moviepy) exporters.  A
  raise-on-use stub replaces the package's ``export``.
* ``depth_anything_3.utils.pose_align`` - needs ``evo`` and is only reached when
  input extrinsics are handed to ``inference()``.  The any-view backend passes
  none (it *wants* the model's own poses), the mono backend has no poses at all.

The model code itself needs torch, einops, addict and omegaconf, all present in
ComfyUI's embedded python.  The api module also flips
``torch.backends.cudnn.benchmark`` to False at import time - the previous value is
restored so other ComfyUI models keep their setting.
"""

from __future__ import annotations

import sys
import types
from typing import Optional

#: default any-view repo for the feed-forward backend: the LARGE release, and
#: specifically the **1.1** revision because that is the Apache-2.0 one - plain
#: ``DA3-LARGE`` (and GIANT / NESTED-GIANT) are CC-BY-NC 4.0.
DA3_ANYVIEW_REPO = "depth-anything/DA3-LARGE-1.1"

#: any-view repos usable as a reconstruction backend, cheapest first.  Licences
#: differ per revision - check before shipping anything: SMALL/BASE and
#: LARGE-1.1 are Apache-2.0, LARGE is CC-BY-NC 4.0.
DA3_ANYVIEW_REPOS = ("depth-anything/DA3-SMALL",
                     "depth-anything/DA3-BASE",
                     "depth-anything/DA3-LARGE-1.1",
                     "depth-anything/DA3-LARGE")

#: the mono repo the Meridian fast-depth node defaults to
DA3_MONO_REPO = "depth-anything/DA3MONO-LARGE"

INSTALL_HINT = ("python -m pip install --no-deps depth-anything-3 "
                "(addict and omegaconf must be importable too)")


def load_da3_api():
    """Return ``DepthAnything3`` with the two optional sub-packages stubbed out.

    Idempotent: a second call returns the already-imported class.
    """
    import torch

    cached = sys.modules.get("depth_anything_3.api")
    if cached is not None:
        return cached.DepthAnything3

    benchmark = torch.backends.cudnn.benchmark
    try:
        import depth_anything_3.utils  # noqa: F401  (namespace parent for the stubs)

        export_stub = types.ModuleType("depth_anything_3.utils.export")

        def _export_unavailable(*args, **kwargs):
            raise ImportError("Depth-Anything-3 export formats are not installed; "
                              "the depth backends only run in-memory inference.")

        export_stub.export = _export_unavailable
        export_stub.SUPPORTED_EXPORT_FORMATS = frozenset()
        sys.modules["depth_anything_3.utils.export"] = export_stub

        pose_stub = types.ModuleType("depth_anything_3.utils.pose_align")

        def _pose_align_unavailable(*args, **kwargs):
            raise ImportError("Depth-Anything-3 pose alignment needs the 'evo' package "
                              "and input extrinsics; the depth backends pass neither.")

        pose_stub.align_poses_umeyama = _pose_align_unavailable
        pose_stub.batch_align_poses_umeyama = _pose_align_unavailable
        sys.modules["depth_anything_3.utils.pose_align"] = pose_stub

        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            f"Depth-Anything-3 is not installed for this interpreter. In the ComfyUI "
            f"python_embeded run: {INSTALL_HINT}"
        ) from exc
    finally:
        torch.backends.cudnn.benchmark = benchmark
    return DepthAnything3


def da3_status() -> tuple:
    """``(usable, reason)`` - never raises, for the nodes' availability report."""
    try:
        load_da3_api()
    except Exception as exc:  # noqa: BLE001 - a missing package is a normal state
        return False, f"depth_anything_3: not usable ({type(exc).__name__}: {exc})"
    return True, f"depth_anything_3: ready (default any-view repo {DA3_ANYVIEW_REPO})"


def resolve_da3_repo(explicit: str = "", env_var: str = "ENNDEE_DA3_REPO") -> str:
    """Any-view repo id: explicit value -> env var -> the default.

    An explicit value that is an existing directory wins as a *local* path, which
    is how a machine that may not talk to huggingface.co uses the model.
    """
    import os

    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    return (os.environ.get(env_var) or "").strip() or DA3_ANYVIEW_REPO


def local_da3_repo(explicit: str = "") -> Optional[str]:
    """The explicit value as a local directory, or ``None``."""
    from pathlib import Path

    explicit = (explicit or "").strip()
    if explicit and Path(explicit).is_dir():
        return str(Path(explicit))
    return None


__all__ = [
    "DA3_ANYVIEW_REPO",
    "DA3_ANYVIEW_REPOS",
    "DA3_MONO_REPO",
    "INSTALL_HINT",
    "da3_status",
    "load_da3_api",
    "local_da3_repo",
    "resolve_da3_repo",
]

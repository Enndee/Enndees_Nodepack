"""Meridian Geometry + Extrinsics (Enndee): the geometry pass with its camera rig as an output.

A drop-in copy of ``EnndeeMeridianGeometry`` that adds a sixth output, ``extrinsics`` -
a ``MERIDIAN_EXTRINSICS`` JSON signal describing the EXACT camera rig the render flew:

* ``c2w``   per-frame 4x4 camera-to-world matrices (columns right, down, forward - the
  renderer's OpenCV frame), positions in the cloud's ``zm`` gauge,
* ``focal`` per-frame focal multipliers (the path keys' zoom),
* ``vfov_degrees`` the vertical FOV the rig was built at,
* ``width`` / ``height`` / ``frames`` the render canvas and frame count.

The rig is read from ``enndee_meridian_fast_depth.LAST_CAMERA_EVAL``, which
``render_depth_aligned`` publishes for both camera branches (custom path and the
parametric offsets), so the signal can never disagree with the rendered flight.

Wire ``extrinsics`` into the **Path Pose Dataset (Enndee)** node to export a
Lichtfeld/3DGS dataset with these ground-truth poses instead of running COLMAP.
"""

import json
import sys
from pathlib import Path

import numpy as np

_PACK_DIR = Path(__file__).resolve().parent.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402  (path setup must run first)
from enndee_meridian_geometry import EnndeeMeridianGeometry  # noqa: E402

#: ComfyUI socket type of the extrinsics signal. A plain custom type string: the
#: Path Pose Dataset node accepts exactly this, nothing else.
EXTRINSICS_SIGNAL_TYPE = "MERIDIAN_EXTRINSICS"

#: Format tag so the consumer can reject foreign JSON on the wire.
EXTRINSICS_FORMAT = "meridian_extrinsics_v1"


def log(message: str) -> None:
    print(f"[Enndee] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[Enndee] WARNING: {message}", flush=True)


def build_extrinsics_signal(eval_data):
    """Camera rig dict (``LAST_CAMERA_EVAL``) -> the ``MERIDIAN_EXTRINSICS`` JSON string.

    Empty frames -> a valid zero-frame document, so a workflow can wire the socket
    before the first render without breaking; the consumer refuses to export from it
    and says why.
    """
    if not eval_data:
        return json.dumps({"format": EXTRINSICS_FORMAT, "frames": 0,
                           "c2w": [], "focal": [], "vfov_degrees": 0.0,
                           "zm": 1.0, "width": 0, "height": 0})
    c2w = np.asarray(eval_data["c2w"], dtype=np.float64)
    focal = np.asarray(eval_data["focal"], dtype=np.float64).reshape(-1)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"camera rig must be [F,4,4], got {c2w.shape}")
    if focal.shape[0] != c2w.shape[0]:
        raise ValueError("focal multipliers must be one per frame")
    document = {
        "format": EXTRINSICS_FORMAT,
        "frames": int(eval_data["frames"]),
        "width": int(eval_data["width"]),
        "height": int(eval_data["height"]),
        "vfov_degrees": round(float(eval_data["vfov_degrees"]), 6),
        "zm": round(float(eval_data.get("zm", 1.0)), 6),
        "focal": [round(float(value), 6) for value in focal],
        "c2w": [[[round(float(value), 8) for value in row] for row in matrix]
                for matrix in c2w],
    }
    return json.dumps(document)


class EnndeeMeridianGeometryExtrinsics(EnndeeMeridianGeometry):
    """Meridian Geometry with the rendered camera rig appended as an output."""

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", EXTRINSICS_SIGNAL_TYPE)
    RETURN_NAMES = ("source", "render", "width", "height", "length", "extrinsics")
    FUNCTION = "build"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Meridian Geometry (Enndee) plus the exact per-frame camera rig it rendered "
        "with: c2w matrices, focal multipliers and the vFOV as a MERIDIAN_EXTRINSICS "
        "signal for the Path Pose Dataset (Enndee) node - ground-truth extrinsics for "
        "Gaussian-splatting datasets, no COLMAP pass needed.")

    def build(self, *args, **kwargs):
        outputs = super().build(*args, **kwargs)
        length = int(outputs[4])
        eval_data = getattr(fast_depth, "LAST_CAMERA_EVAL", None)
        if not eval_data:
            log_warn("no camera evaluation was published by the render - the "
                     "extrinsics output is an empty signal")
        elif int(eval_data.get("frames", -1)) != length:
            log_warn(f"published camera rig has {eval_data.get('frames')} frames but "
                     f"the render returned {length} - consumers map by normalized "
                     "time, but check for upstream trimming/VFI")
        return (*outputs, build_extrinsics_signal(eval_data))


NODE_CLASS_MAPPINGS = {"Enndee_MeridianGeometryExtrinsics": EnndeeMeridianGeometryExtrinsics}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_MeridianGeometryExtrinsics": "Meridian Geometry + Extrinsics (Enndee)"}

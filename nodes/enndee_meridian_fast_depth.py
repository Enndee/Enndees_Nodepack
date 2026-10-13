"""Meridian fast-depth backend (Enndee): the VGGT-free geometry engine behind Meridian Geometry.

Builds the Meridian / MiniMax-H3 reference pair in-process, with no external conda
environment and no VGGT checkpoint:

    starting frame --Depth model--> relative depth (larger = farther)
    relative depth --3x3 edge keep + all-parents upsample--> kept depth
    depth + RGB --unproject--> 3D point cloud (the "depth-aligned texture")
    camera path (custom_camera signal, or the sample.py motion flags) --render--> flight frames

Three depth families are available, picked by Meridian Geometry's `model_size`:

    Depth-Anything-V2-*  predicts *inverse* depth (disparity-like: larger = closer - the near
                         pier post reads 4.4 while the far sky reads 0.3), so the engine
                         inverts it (1/x) before the percentile rescale. Feeding the raw
                         output as z flips the scene front-to-back: the subject lands behind
                         the background and a moving camera sees the mirrored "back" of a
                         shell whose front side was never reconstructed.
    Depth-Anything-3-*   predicts depth directly, and the raw output already is true relative
                         depth (measured on beach.jpg: the near pillar reads ~0.72 while the
                         far sea/sky reads ~5.2, verified against a turbo-mapped depth strip),
                         so the v3 path skips the inversion. Small/Base/Large are the any-view
                         models (camera poses, unused here); Mono-Large is the monocular
                         series tuned for single stills.
    MoGe-3-ViT-L         predicts depth directly as well (the point map's z), so it skips the
                         inversion too. Its extra stage is the *self-guided sparse volumetric
                         refinement*: the coarse point map is voxelised into a sparse 3D grid
                         and walked by a Sparse3DUNet, `moge_refine_steps` times (0 = off,
                         3 = MoGe's default). It is the only family that is OPTIONAL - the
                         `moge` package is imported lazily, so every other depth source keeps
                         working without it (see :mod:`enndee_moge`).

Depth-Anything-3 comes from the `depth-anything-3` pip package, installed with `--no-deps`
(its numpy<2 pin plus xformers/open3d/pycolmap/moviepy/gsplat would fight ComfyUI's embedded
python). The api module eagerly imports its export dispatcher and its `evo`-based pose
alignment, so `_load_da3_api` swaps in raise-on-use stubs for
`depth_anything_3.utils.export` and `depth_anything_3.utils.pose_align` before importing:
the model code itself then needs only torch, einops, addict and omegaconf. Import takes
~1.7 s and a still costs ~0.4 s on a Blackwell GPU.

Both families resize before inference. `depth_res` caps the *still's* longest side: a larger
picture is bilinearly downsized before the depth model, the frame colours and the unprojection
grid are built (`_fit_working_still`), so a 24 Mpx camera photo cannot build a cloud that trips
`torch.quantile`'s hard 2**24-element ceiling - `cloud_scale` alone could never shrink a source
that was already over `MAX_CLOUD_PIXELS` on its own. `depth_res=0` keeps the still's own pixels
(bounded only by that ceiling) and a still at or below the cap is never upscaled.
Depth-Anything-3 additionally receives the cap as its own `process_res` (aspect preserved,
rounded to multiples of 14 by the library; values above the still's side upscale), while
Depth-Anything-V2 keeps its native 518 square (`DEPTH_RES`) - the still and the cloud follow
the same cap for both families, so a finer grid keeps smaller depth steps alive through the
edge mask and the cloud rebuild at a cost that grows with the square of the pixel count.

Two render guards mirror Meridian's own pipeline (recam/geometry.py, inference/sample.py):

    edge keep       the 3x3 local depth-spread rule (EDGE_RTOL = 0.30) is applied on the
                    model's own depth grid; at the cloud grid a hi-res pixel survives only
                    where every parent survived (`upsample`'s >0.999 rule) - this is what
                    kills the bilinear "flying pixels" that smear silhouettes into depth.
    back-face cull  depth-map normals oriented toward the source camera (`--cull`): points
                    the target camera sees from behind are dropped, so a 180-degree view is
                    a hole, not the mirrored front.

This module is a library, not a node: Meridian Geometry (Enndee) drives it with
`mode = "Fast depth (Depth-Anything-V2)"` (the mode label predates v3 and stays for saved
workflows), and the unittests call `render_depth_aligned` directly with a fake depth model.
Depth-Anything-V2-Small runs at ~7 ms per frame in FP16 on a Blackwell GPU; a 73-frame pass
typically completes in well under a second.
"""

import inspect
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS

# MoGe's loader shim sits at the pack root next to `enndee_da3`, so make sure it is importable
# before the constants below are read. The shim itself imports nothing heavy - the `moge`
# package is pulled in lazily by `load_moge_v3`, which is what keeps MoGe-3 optional.
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from enndee_moge import (  # noqa: E402  (path setup must run first)
    DEFAULT_REFINE_STEPS as MOGE_REFINE_STEPS,
    DEFAULT_RESOLUTION_LEVEL as MOGE_RESOLUTION_LEVEL,
    MOGE_V3_MODEL_ID,
    MOGE_V3_REPO,
    load_moge_v3,
    resolve_refine_steps,
)

# Meridian's 480-class aspect ladder (width, height), recam/h3.py LADDERS[480]
LADDER_480 = [
    (416, 960), (448, 896), (480, 832), (544, 736), (640, 640),
    (736, 544), (832, 480), (896, 448), (960, 416),
]
HOLE_COLOR = 128                      # uncovered pixels: mid grey, as in recam/geometry.py (no mask channel)
#: Inpaint uncovered pixels from their rendered neighbours so the frame reads as a picture
#: rather than a cut-out. The mesh surface only closes holes WITHIN the subject; where RMBG
#: removed the background there is no geometry at all and a close-up shows a grey field.
#: Pure post-process - geometry and every measurement are untouched. See `_fill_render_holes`.
FILL_RENDER_HOLES = True
#: The vertical field of view the cloud was actually BUILT with - the MEASURED one when
#: MoGe-3 reports intrinsics, else ``VFOV_DEGREES``. Published because the auto-camera's
#: face orbit must project with the same focal; see ``_predict_moge_depth``.
LAST_VFOV_DEGREES = None
VFOV_DEGREES = 55.0                   # assumed vertical field of view of the source camera
# A *measured* source FOV (MoGe-3 reports intrinsics) is only trusted inside this window; a
# wild value would stretch the whole unprojection, so out-of-range falls back to the assumption.
MIN_VFOV_DEGREES, MAX_VFOV_DEGREES = 5.0, 150.0
DEPTH_NEAR, DEPTH_FAR = 1.0, 5.0      # metric window the relative depth is mapped onto (the gauge is zm-relative)
DEPTH_PCT_LO, DEPTH_PCT_HI = 0.01, 0.99   # percentile clip so one speck cannot squash the range
PIVOT_WINDOW = 0.05                   # Meridian's +-5 % window around a picked pivot point
MAX_CLOUD_PIXELS = 16_777_216         # unprojection-grid + working-still ceiling (2**24 pixels)
QUANTILE_MAX = 1 << 24                # torch.quantile's hard element limit (`numel <= 2**24`)
DEPTH_RES = 518                       # Depth-Anything-V2's native square input side
DA3_RES = 504                         # default `depth_res`: still + Depth-Anything-3 longest-side cap (0 = source)
DISPARITY_EPS = 0.001                 # floor before the 1/x inversion, so the far plane stays finite
KEEP_PARENT_RATIO = 0.999             # recam/geometry.py `upsample`: a hi-res pixel needs every parent kept

# The camera's roll. The frame is the world-up look-at: the horizon stays level, so an orbit - the
# spiral's O-orbits especially - comes out upright and nothing twists about the optical axis. The
# one pose without a horizon is the pole, the look straight up/down the world axis, where the
# world-up frame has to reverse (its right vector flips sign as the look crosses vertical). There
# the previous frame is carried by parallel transport and walked back to level over the next
# HORIZON_RELOCK frames, so a coil that grazes the top re-locks smoothly instead of flipping.
WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float32)   # OpenCV frame: -y is up
HORIZON_MIN_SIN = 1e-6                # |cross(look, up)| below this = the look is vertical (no horizon)
HORIZON_RELOCK = 24                   # frames to settle back to level after the look crosses the pole

# Depth-Anything-3 variants the fast backend can load, as Hugging Face repo ids. Small/Base
# are the any-view series (relative depth + camera poses, the poses unused here); Mono-Large
# is the monocular series tuned for high-quality single-still depth. Metric-Large is left out
# because the percentile rescale below re-gauges every prediction anyway, and the Giant/Nested
# models are much heavier. Licences differ per repo: SMALL / BASE / MONO-LARGE are Apache-2.0,
# while plain LARGE is CC-BY-NC 4.0 (its Apache revision is depth-anything/DA3-LARGE-1.1).
DA3_PREFIX = "Depth-Anything-3"
DA3_MODEL_REPOS = {
    "Depth-Anything-3-Small": "depth-anything/DA3-SMALL",
    "Depth-Anything-3-Base": "depth-anything/DA3-BASE",
    "Depth-Anything-3-Large": "depth-anything/DA3-LARGE",
    "Depth-Anything-3-Mono-Large": "depth-anything/DA3MONO-LARGE",
}

# MoGe-3 (Microsoft, 2026-08-18) is the third depth family. Like Depth-Anything-3 it predicts
# *depth* directly - its point map's z - so the backend does not invert it either. What sets
# MoGe-3 apart from MoGe-2 is the self-guided sparse volumetric refinement stage, exposed as
# `refine_steps` (0 = off, 3 = the library default): the coarse point map is voxelised into a
# sparse 3D grid and walked by a Sparse3DUNet, which is what sharpens the fine geometry. Only
# the ViT-L v3 checkpoint exists and it is MIT licensed. The package is `moge` and it is
# OPTIONAL - see :mod:`enndee_moge`; nothing here imports it until a run selects MoGe 3.
#: Matched with ``str.startswith``, so it must cover BOTH spellings the nodes use: the
#: widget value is ``"MoGe 3"`` (space) while the model keys are hyphenated. With the old
#: ``"MoGe-"`` prefix, ``"MoGe 3".startswith("MoGe-")`` is False - the probe silently fell
#: through to the generic depth branch, never measured the source FOV, and the face orbit
#: projected the SAM3 box at the hard-coded 55 deg (a 1.509x error in f), selecting a head
#: patch ~4x too big and leaving `face_fill` unattained.
MOGE_PREFIX = "MoGe"
MOGE_MODEL_REPOS = {
    MOGE_V3_MODEL_ID: MOGE_V3_REPO,
}

# sample.py camera flags the fast backend honours, so one Meridian Parameter Picker args
# string drives both backends: value flags carry a following token, boolean flags do not.
CAMERA_VALUE_FLAGS = {
    "--frames": "frames",
    "--yaw": "yaw",
    "--yaw-from": "yaw_from",
    "--truck": "truck",
    "--boom": "boom",
    "--dolly": "dolly",
    "--zoom": "zoom",
    "--pivot": "pivot",
    "--pivot-to": "pivot_to",
    "--fast-back": "fast_back",
}
CAMERA_FLAGS = {
    "--sweep": "sweep",
    "--bounce": "bounce",
    "--swing": "swing",
    "--ease": "ease",
    "--aim": "aim",
    "--pivot-lock": "pivot_lock",
    "--cull": "cull",
}


_GLOBAL_DEPTH_MODEL = None
_GLOBAL_MODEL_ID = None


def _get_depth_model(model_name: str, device: torch.device):
    """Cache Depth-Anything-V2 in FP16 on the GPU across runs; reload only if the variant changed."""
    global _GLOBAL_DEPTH_MODEL, _GLOBAL_MODEL_ID
    repo_id = model_name if "/" in model_name else f"depth-anything/{model_name}"
    if _GLOBAL_DEPTH_MODEL is None or _GLOBAL_MODEL_ID != repo_id:
        from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingForDepthEstimation
        print(f"[Enndee] Meridian Fast Depth: loading {repo_id} (fp16)...", flush=True)
        _GLOBAL_DEPTH_MODEL = DepthAnythingForDepthEstimation.from_pretrained(repo_id).to(device).half().eval()
        _GLOBAL_MODEL_ID = repo_id
    return _GLOBAL_DEPTH_MODEL


_GLOBAL_DA3_MODEL = None
_GLOBAL_DA3_ID = None


def _load_da3_api():
    """Import ``depth_anything_3.api`` with its optional heavy sub-packages stubbed out.

    The pip package declares a dependency set meant for a dedicated environment
    (``numpy<2``, xformers, open3d, pycolmap, moviepy, gsplat, evo) and is therefore installed
    here with ``python -m pip install --no-deps depth-anything-3``, keeping ComfyUI's own
    numpy/torch. Two eager imports then stay unsatisfied, and both are dead ends for the
    fast-depth backend:

    * ``depth_anything_3.utils.export`` - the export dispatcher pulls the glb (trimesh),
      colmap (pycolmap), gs (gsplat) and vis (moviepy) exporters; this backend only runs
      in-memory inference, so a raise-on-use stub replaces the package's ``export``.
    * ``depth_anything_3.utils.pose_align`` - needs ``evo`` and is only reached when input
      extrinsics are handed to ``inference()``; this backend passes none.

    The stub dance now lives in :mod:`enndee_da3`, shared with the VGGT node's ``DA3-AnyView``
    backend, so both stay in sync; this name is kept because the unittests and
    :func:`_predict_da3_depth` call it.
    """
    import os

    pack = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pack not in sys.path:
        sys.path.insert(0, pack)
    from enndee_da3 import load_da3_api

    return load_da3_api()


def _da3_process_res(depth_res, width: int, height: int) -> int:
    """Depth-Anything-3 `process_res` for one still: `depth_res` px, 0 = the still's own side.

    The Depth-Anything-3 input processor (`upper_bound_resize`) scales the image so its *longest*
    side equals `process_res` - upscaling as happily as downscaling, which is why the default is
    passed through unchanged - and then rounds each dimension to the nearest multiple of 14 (the
    ViT patch size). `max(width, height)` therefore means "as captured": the depth grid keeps the
    source's own detail budget, and since the attention cost grows with the square of the pixel
    count this is the knob that trades time for depth detail.
    """
    longest = max(int(width), int(height))
    cap = int(depth_res or 0)
    return cap if cap > 0 else longest


def _fit_working_still(first: torch.Tensor, depth_res) -> Tuple[torch.Tensor, Optional[str]]:
    """Downscale a too-big still to `depth_res`' longest side (0 = its own resolution).

    `depth_res` is the *working-resolution* cap of the whole fast path: the depth model, the
    frame colours and the unprojection grid are all built from this still, so a camera photo
    (24 Mpx and up) is resized here instead of further down, where the pool handed to
    `torch.quantile` would exceed ATen's hard 2**24-element ceiling. The `MAX_CLOUD_PIXELS`
    ceiling also applies when `depth_res=0` asks for the own resolution, because there
    `cloud_scale` can no longer shrink the base grid. Never upscales: a still at or below both
    caps is returned unchanged, together with the note to print (or `None` when nothing moved).
    """
    height, width = int(first.shape[1]), int(first.shape[2])
    cap = int(depth_res or 0)
    side_cap = cap / float(max(width, height)) if cap > 0 else 1.0
    area_cap = math.sqrt(MAX_CLOUD_PIXELS / float(width * height))
    factor = min(side_cap, area_cap)
    if factor >= 1.0:
        return first, None
    # the 1e-6 tolerance absorbs the float error of `cap / max(width, height)`, so an exact
    # fraction (2400 px at cap 400) lands on 400 instead of one pixel short
    new_w = max(1, int(math.floor(width * factor + 1e-6)))
    new_h = max(1, int(math.floor(height * factor + 1e-6)))
    small = F.interpolate(first.permute(0, 3, 1, 2), size=(new_h, new_w), mode="bilinear",
                          align_corners=False).permute(0, 2, 3, 1)
    reason = f"depth_res {cap} cap" if side_cap <= area_cap else f"MAX_CLOUD_PIXELS {MAX_CLOUD_PIXELS} cap"
    return small, f"still {width}x{height} -> {new_w}x{new_h} ({reason})"


def _flat_quantile(pool: torch.Tensor, q: float) -> torch.Tensor:
    """`torch.quantile` for pools of any size: ATen rejects inputs above 2**24 elements.

    The unprojection grid sits exactly on `MAX_CLOUD_PIXELS`, which is that same 2**24, so a
    boundary overshoot alone would crash a long flight. The percentile clip only needs the
    shape of the distribution, so bigger pools are strided down first - the stride walks the
    whole field instead of cropping it to one corner, and the measured 1 %/99 % boundaries
    stay identical on a monotone ramp.
    """
    flat = pool.reshape(-1)
    if flat.numel() > QUANTILE_MAX:
        flat = flat[:: -(-flat.numel() // QUANTILE_MAX)]
    return torch.quantile(flat, q)


def depth_gauge_scale(pool: torch.Tensor) -> float:
    """The ONE depth -> cloud map, `z_cloud = scale * z_raw`. Returns its slope.

    `render_depth_aligned`, the automatic camera estimator and the camera path all have to live in
    the same gauge (the keys are authored in "median-depth units"), so the map is defined once,
    here, and everything calls it.

    **It is proportional, and that is the whole point.** An affine map (`a * z + b`) does not just
    rescale a scene - it distorts its *shape*. The unprojection is

        x = (u - cx) / f * z_cloud,   y = (v - cy) / f * z_cloud,   z = z_cloud

    so x and y both carry the mapped depth and every point is scaled radially by `k = z_cloud / z`
    - a factor that *varies with depth* whenever `b != 0`. For an object at raw depth `z` with
    depth relief `dz` and lateral size `L`, the aspect (lateral : depth) comes out multiplied by

        C = z_cloud / (a * z) = 1 + b / (a * z),   a = (FAR - NEAR) / span,  b = NEAR - a * d_lo

    `C > 1` flattens the object, `C < 1` deepens it. Measured on beach.jpg with MoGe-3, the affine
    map gave `C = 5.7` at a 1.6 m subject: a person's thighs - the part of a body that extends
    *toward* the camera - showed 17 % of the depth relief the model actually predicted. `b = 0` is
    a pure scale, and a pure scale is invisible here: the cloud, the pivot and the camera rig all
    scale together, so the *ratio* of the scene's depth extent to the camera distance (the
    parallax, which is what the eye reads as "how deep") is preserved for every source and every
    scene.

    Why the old affine map looked right on Depth-Anything-3 and wrong on MoGe-3: `b` is only small
    when the model's p1..p99 span happens to be close to the `DEPTH_NEAR .. DEPTH_FAR` window.
    DA3's relative depth spans ~3.8 against that 4.0 window (`C ~ 1.03`, indistinguishable from
    correct), while MoGe-3 returns *metric* depth whose p99 is the sea or horizon tens of metres
    away (span ~ 35 -> `C ~ 5.7`).

    The *scale* is anchored on the rig distance this backend has always used - the old affine
    map's value at the median, `NEAR + (FAR - NEAR) * (d_median - d_lo) / span` - so `zm`, the
    orbit radius and every authored camera amount stay exactly where they were: this changes the
    shape, not the scale. The percentiles keep their two jobs: the anchor, and the span floor that
    stops a nearly constant map from turning float noise into a fake bumpy surface (which would
    scramble the `--cull` normals).
    """
    d_lo = _flat_quantile(pool, DEPTH_PCT_LO)
    d_hi = _flat_quantile(pool, DEPTH_PCT_HI)
    span = (d_hi - d_lo).clamp(min=1e-3 * float(d_hi.abs() + d_lo.abs()) + 1e-9)
    d_med = _flat_quantile(pool, 0.5)
    reference = float(DEPTH_NEAR + (DEPTH_FAR - DEPTH_NEAR)
                      * (float(d_med) - float(d_lo)) / float(span))
    if float(d_med) > 0.0 and reference > 0.0:
        return reference / float(d_med)
    # A map centred on or below zero cannot be anchored proportionally without flipping it. The
    # window's own slope is still proportional (`b = 0`), so the shape guarantee holds either way.
    return float(DEPTH_FAR - DEPTH_NEAR) / float(span)


def aspect_x_scale(cloud_w: int, cloud_h: int, out_w: int, out_h: int) -> float:
    """The x factor that makes the identity frame reproject EXACTLY.

    The cloud is unprojected at the working still's aspect but rendered into the
    480-class ladder canvas, and the two differ (measured on the Hanks still: 1.4884 vs
    1.3529). With `x = (u-0.5) * cloud_w / f_cloud * z` and
    `u_out = 0.5 + f_canvas * x / (z * out_w)`, the mapping `u -> u_out` is the identity
    iff `f_canvas * cloud_w / (f_cloud * out_w) == 1`, and since
    `f = 0.5 * H / tan(vfov/2)` on both sides that reduces to
    `aspect_cloud == aspect_canvas`. Scaling x by this factor restores it for any pair.

    Without the correction the subject is drawn `aspect_cloud / aspect_canvas` too wide
    - 1.100x on the Hanks still, a real ~10 % "too flat" contribution.
    """
    aspect_canvas = float(out_w) / float(out_h)
    aspect_cloud = float(cloud_w) / float(cloud_h)
    return aspect_canvas / max(aspect_cloud, 1e-9)


def apply_depth_relief(z: torch.Tensor, relief: float) -> torch.Tensor:
    """Exaggerate (or compress) depth relief about the map median."""
    if relief is None or float(relief) == 1.0:
        return z
    med = z.median()
    return (med + (z - med) * float(relief)).clamp(min=0.05)


def resolve_depth_relief(value=1.0) -> float:
    """Validate the shared depth-relief control used by the camera estimate and renderer."""
    try:
        relief = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_relief must be a finite number in 0.25..4.0.") from exc
    if not math.isfinite(relief) or not 0.25 <= relief <= 4.0:
        raise ValueError(f"depth_relief must be in 0.25..4.0, got {value!r}.")
    return relief



def cloud_gauge(depth: torch.Tensor) -> torch.Tensor:
    """A raw model depth map expressed in the *cloud's* gauge - the map `render_depth_aligned`
    builds its point cloud with.

    `render_depth_aligned` does not unproject the model's raw depth: it maps it into this gauge
    first and builds the cloud from *that*. The rendered cloud therefore lives in this gauge, not
    in the model's own units, and a camera path must live in the same one: the keys' x/y follow
    their z (a key is `(px - cx) / f * z`), so a rig placed in the wrong gauge is not just at the
    wrong distance - it is at the wrong *scale*, and the subject drifts out of the picture along
    the path.

    The map is proportional (`z_cloud = scale * z_raw`, see `depth_gauge_scale`), so the estimator
    cannot get there by rescaling its keys alone: it has to build its surface in this gauge.
    `probe_surface` calls this before unprojecting, so the emitted keys - in units of the mapped
    median - are exactly what the renderer's `zm` (`mapped_depth(median)`) expects.
    """
    return (depth * depth_gauge_scale(depth.reshape(-1))).clamp(min=0.05)


def _predict_da3_depth(model_name: str, first: torch.Tensor, device: torch.device,
                       process_res: int = DA3_RES) -> torch.Tensor:
    """Depth-Anything-3 relative depth for one still: (H, W) float tensor, larger = farther.

    Unlike Depth-Anything-V2 there is no 1/x conversion - DA3 predicts depth directly and the
    raw output already is true relative depth (beach.jpg: the near pillar ~0.72, the far
    sea/sky ~5.2). The grid is aspect-preserving (`process_res` caps the longest side), so it
    can be non-square. `process_res=DA3_RES` (504) is the fast default; the caller passes the
    still's own longest side when its `depth_res` is 0. The model is cached in VRAM across
    runs; only the variant reloads it.
    """
    global _GLOBAL_DA3_MODEL, _GLOBAL_DA3_ID
    repo_id = DA3_MODEL_REPOS.get(model_name)
    if repo_id is None:
        raise ValueError(f"Unknown Depth-Anything-3 variant {model_name!r}; "
                         f"expected one of {', '.join(DA3_MODEL_REPOS)}.")
    if _GLOBAL_DA3_MODEL is None or _GLOBAL_DA3_ID != repo_id:
        DepthAnything3 = _load_da3_api()
        print(f"[Enndee] Meridian fast depth: loading {repo_id}...", flush=True)
        _GLOBAL_DA3_MODEL = DepthAnything3.from_pretrained(repo_id).to(device).eval()
        _GLOBAL_DA3_ID = repo_id
    frame = first[0].detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    print(f"[Enndee] Meridian fast depth: {model_name} still {frame.shape[1]}x{frame.shape[0]} "
          f"-> process_res {int(process_res)}...", flush=True)
    with torch.no_grad():
        prediction = _GLOBAL_DA3_MODEL.inference([frame], process_res=int(process_res))
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    return torch.from_numpy(depth).to(device)


_GLOBAL_MOGE_MODEL = None
_GLOBAL_MOGE_ID = None
_MOGE_INFER_PARAMS = None


def _moge_depth_tensor(prediction) -> torch.Tensor:
    """``[H, W]`` depth out of whatever ``MoGeModel.infer`` handed back.

    MoGe returns a dict with ``depth`` / ``points`` / ``mask`` / ``intrinsics`` / ``normal``.
    The depth entry is ``[H, W]`` for a single un-batched image and ``[1, H, W]`` when the
    model kept a batch axis, so both are accepted.
    """
    if isinstance(prediction, dict):
        depth = prediction.get("depth")
        if depth is None:
            raise RuntimeError("MoGe-3 returned no 'depth' map - the installed `moge` build "
                               "does not match the documented v3 output contract.")
    else:
        depth = getattr(prediction, "depth", prediction)
    tensor = torch.as_tensor(depth)
    while tensor.ndim > 2 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise RuntimeError(f"MoGe-3 depth map has an unexpected shape {tuple(tensor.shape)}.")
    return tensor


def _moge_infer_kwargs(steps: int, resolution_level: int) -> dict:
    """The ``infer`` kwargs this ``moge`` build actually accepts.

    ``refine_steps`` is MoGe-**3**-only. Reading the installed signature once beats both
    assumptions: an older build must not die on an unexpected keyword, and a newer one must
    not lose the refinement stage because it was hard-coded off. (Same lesson as Lichtfeld
    Studio's renamed ``--depth-loss-mode``: ask the thing you are about to call.)
    """
    global _MOGE_INFER_PARAMS
    if _MOGE_INFER_PARAMS is None:
        try:
            _MOGE_INFER_PARAMS = frozenset(
                inspect.signature(_GLOBAL_MOGE_MODEL.infer).parameters)
        except (TypeError, ValueError):          # a wrapped or C-level callable
            _MOGE_INFER_PARAMS = frozenset()
    kwargs = {"resolution_level": int(resolution_level), "use_fp16": True}
    if "refine_steps" in _MOGE_INFER_PARAMS or not _MOGE_INFER_PARAMS:
        kwargs["refine_steps"] = int(steps)
    elif steps:
        print("[Enndee] Meridian fast depth: this `moge` build has no `refine_steps` - "
              "MoGe-3's refinement stage is unavailable, running a single pass.", flush=True)
    return kwargs


def _fill_invalid_depth(depth: torch.Tensor) -> torch.Tensor:
    """MoGe's unvouched pixels (`inf`) become the far plane instead of poisoning the render.

    `apply_mask=True` is the model's own validity signal, and `infer` writes ``inf`` into every
    masked pixel - measured on beach.jpg that is **20.4 %** of the frame (the sky). That value
    cannot survive this backend: `cloud_gauge` takes percentiles over the whole map, so an
    ``inf`` wins the 99th percentile and flattens the real scene into the bottom of the window,
    and the unprojection would hand ``inf``/``nan`` coordinates to the renderer.

    The renderer only needs "larger = farther", so an unvouched pixel is placed at the far end
    of the *measured* depth: sky becomes a backdrop, which is what a moving camera should see
    behind the subject. The count is printed rather than applied silently - a model that has
    started masking everything must be visible in the log, not discovered in the render.

    That value is the valid ``DEPTH_PCT_HI`` percentile, **not** the valid maximum, and the
    difference is not cosmetic. `render_depth_aligned` clips the map to
    ``DEPTH_PCT_LO .. DEPTH_PCT_HI`` and builds the cloud from the result, so ``d_hi`` *is* the
    far plane. The filled pixels are 20.4 % of the frame - twenty times more than the 1 % the
    99th percentile needs - so whatever value they carry becomes ``d_hi`` and defines the whole
    window. Filling at the maximum therefore stretched the window from 1.2 .. 36 to 1.2 .. 93 and
    squashed the measured scene into **1.000 .. 1.020** of the 1.0 .. 5.0 window: a flat card.
    Measured on beach.jpg the near-field depth contrast (rendered p90 / p10) came out at 1.016x
    against Depth-Anything-3-Large's 2.43x. The clip exists precisely so that "one speck cannot
    squash the range" - handing the raw maximum to a fifth of the frame is the one way to defeat
    it. Filling at the clip's own high end leaves ``d_hi`` exactly where the valid pixels put it
    (the fill lands on that value rather than past it) and restores the contrast to 2.10x.
    """
    invalid = ~torch.isfinite(depth)
    if not bool(invalid.any()):
        return depth
    valid = depth[~invalid]
    if valid.numel() == 0:
        raise RuntimeError("MoGe-3 marked every pixel of the still invalid - there is no "
                           "geometry to render. Try a different still.")
    # NOT `valid.max()` - see the docstring: the fill is a fifth of the frame, so its value
    # becomes the clip's 99th percentile and therefore the far plane itself. The clip's own high
    # end is the one value that leaves `d_hi` exactly where the measured pixels put it.
    far = _flat_quantile(valid, DEPTH_PCT_HI)
    print(f"[Enndee] Meridian fast depth: MoGe-3 left {int(invalid.sum())} pixel(s) "
          f"({float(invalid.to(torch.float32).mean()) * 100:.1f}%) unvouched - placing them at "
          f"the far plane ({float(far):.3f}, the {DEPTH_PCT_HI:.0%} percentile of the measured "
          f"depth).", flush=True)
    return torch.where(invalid, far, depth)


def _fov_y_degrees_from_intrinsics(intrinsics) -> Optional[float]:
    """MoGe's normalised ``intrinsics`` -> the source camera's vertical FOV, in degrees.

    MoGe normalises by the image *sides* (``fx`` by width, ``fy`` by height), so ``fy`` alone
    carries the vertical field of view: ``fov_y = 2 * atan(0.5 / fy)``. Measured on beach.jpg
    that is **42.53 degrees**, against the 55 this backend used to assume - and the assumption
    is applied to *both* the unprojection and the target canvas, so a wrong value stretches the
    depth and skews the parallax the moment the camera moves, which is the whole point of this
    renderer.

    Returns ``None`` when the value is missing or implausible, so the caller keeps its fallback
    rather than rendering nonsense.
    """
    if intrinsics is None:
        return None
    try:
        tensor = torch.as_tensor(intrinsics, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):   # a string, a dict, anything non-numeric
        return None
    while tensor.ndim > 2 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or int(tensor.shape[0]) < 2 or int(tensor.shape[1]) < 2:
        return None
    fy = float(tensor[1, 1])
    if not math.isfinite(fy) or fy <= 0:
        return None
    fov = math.degrees(2.0 * math.atan(0.5 / fy))
    return fov if MIN_VFOV_DEGREES <= fov <= MAX_VFOV_DEGREES else None


def _predict_moge_depth_and_fov(model_name: str, first: torch.Tensor, device: torch.device,
                                refine_steps: int = MOGE_REFINE_STEPS,
                                resolution_level: int = MOGE_RESOLUTION_LEVEL):
    """``(depth, measured vertical FOV in degrees | None)`` for one still - see the wrapper.

    MoGe-3 is the one depth source here that also reports the camera it saw the still through:
    ``infer`` returns a 3x3 ``intrinsics``, and its ``fy`` is the source camera's vertical focal
    in image heights. That turns the renderer's hard-coded FOV assumption into a measurement,
    which is what :func:`_fov_y_degrees_from_intrinsics` extracts.
    """
    global _GLOBAL_MOGE_MODEL, _GLOBAL_MOGE_ID
    # Match by NORMALISED FAMILY PREFIX, not by exact key. The widget value is `"MoGe 3"`
    # (space) and the only key is `"MoGe-3-ViT-L"` (hyphen, plus a variant suffix), so every
    # exact lookup - including space<->hyphen normalisation - returned None. The ValueError
    # was then swallowed by the caller's `except Exception`, and the probe silently fell
    # through to a depth source with NO measured FOV. That is the reason the face orbit kept
    # projecting at the hard-coded 55 deg, the head patch stayed ~4x too big, and
    # `face_fill` looked ignored. Measured: `MOGE_MODEL_REPOS = {'MoGe-3-ViT-L': ...}`.
    def _norm(text):
        return str(text).replace("-", "").replace(" ", "").lower()
    wanted = _norm(model_name)
    repo_id = MOGE_MODEL_REPOS.get(model_name)
    if repo_id is None:
        for _key, _value in MOGE_MODEL_REPOS.items():
            if _norm(_key).startswith(wanted) or wanted.startswith(_norm(_key)):
                repo_id = _value
                break
    if repo_id is None:
        raise ValueError(f"Unknown MoGe variant {model_name!r}; "
                         f"expected one of {', '.join(MOGE_MODEL_REPOS)}.")
    # Validate before the (slow) weight load, so a bad widget value fails fast.
    steps = resolve_refine_steps(refine_steps)
    if _GLOBAL_MOGE_MODEL is None or _GLOBAL_MOGE_ID != repo_id:
        MoGeModel = load_moge_v3()
        print(f"[Enndee] Meridian fast depth: loading {repo_id}...", flush=True)
        _GLOBAL_MOGE_MODEL = MoGeModel.from_pretrained(repo_id).to(device).eval()
        _GLOBAL_MOGE_ID = repo_id
    # MoGe takes one [3, H, W] float image in [0, 1] on the model's device; the working still
    # is [1, H, W, 3].
    image = first[0].detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    image = image.permute(2, 0, 1).contiguous()
    print(f"[Enndee] Meridian fast depth: {model_name} still "
          f"{int(image.shape[2])}x{int(image.shape[1])} -> resolution_level "
          f"{int(resolution_level)}, refine_steps {steps}...", flush=True)
    with torch.no_grad():
        prediction = _GLOBAL_MOGE_MODEL.infer(
            image, **_moge_infer_kwargs(steps, resolution_level))
    depth = _fill_invalid_depth(_moge_depth_tensor(prediction).to(device=device,
                                                                 dtype=torch.float32))
    intrinsics = (prediction.get("intrinsics") if isinstance(prediction, dict)
                  else getattr(prediction, "intrinsics", None))
    _fov = _fov_y_degrees_from_intrinsics(intrinsics)
    # Publish at the DEEPEST common point so every caller sees it, whichever wrapper it
    # went through. `auto_camera.probe_surface` needs this to project the SAM3 face box
    # onto the cloud at the focal the cloud was built with; at the hard-coded 55 deg the
    # box selects a head patch ~4x too big and the zoom solve never moves the camera.
    global LAST_VFOV_DEGREES
    LAST_VFOV_DEGREES = float(_fov) if _fov is not None else None
    return depth, _fov


def _predict_moge_depth(model_name: str, first: torch.Tensor, device: torch.device,
                        refine_steps: int = MOGE_REFINE_STEPS,
                        resolution_level: int = MOGE_RESOLUTION_LEVEL) -> torch.Tensor:
    """MoGe-3 relative depth for one still: (H, W) float tensor, larger = farther.

    MoGe predicts depth, not disparity (it is the point map's z), so - exactly like
    Depth-Anything-3, and unlike Depth-Anything-V2 - the raw output is already the right way
    round and there is no 1/x inversion. The map comes back at the working still's own
    resolution: MoGe always returns maps at the input size, and `resolution_level` only
    decides how many tokens the encoder gets (9 is the library default, and the node's
    `depth_res` has already capped the still).

    `refine_steps` is MoGe-3's self-guided sparse volumetric refinement: 0 skips the stage and
    returns the single-pass MoGe-2-style result, higher values trade time for fine-grained
    geometry (3 is the library default). The model is cached in VRAM across runs; only the
    variant reloads it.

    This is the depth-only view of :func:`_predict_moge_depth_and_fov`, which the renderer uses
    when it also wants the measured FOV.
    """
    depth, _fov = _predict_moge_depth_and_fov(model_name, first, device, refine_steps,
                                              resolution_level)
    # PUBLISH the measurement even on this depth-only path. `auto_camera.probe_surface`
    # calls THIS wrapper - not `_predict_moge_depth_and_fov` - so the measured FOV used to
    # die here as `_fov`, and the face orbit then projected the SAM3 box onto the cloud at
    # the hard-coded 55 deg instead of the measured ~38 deg (a 1.509x error in f). That
    # selected a head patch ~4x too big (face_radius 0.748 for a ~0.18 head) and made the
    # zoom solve ask for 2.63 units of stand-off where ~0.65 is right - i.e. `face_fill`
    # looked ignored. `render_depth_aligned` sets this too, but it runs AFTER the probe.
    global LAST_VFOV_DEGREES
    LAST_VFOV_DEGREES = float(_fov) if _fov is not None else None
    return depth


def _bucket_480(width: int, height: int) -> Tuple[int, int]:
    """Meridian 480-class ladder entry nearest in log-aspect (recam/h3.py bucket())."""
    return min(LADDER_480, key=lambda c: abs(math.log((c[0] / c[1]) * (height / width))))


def warn_depth_model_mismatch(path_data, model_size):
    """The warning text when a camera path was estimated on a different depth model, else None.

    The path's keys are in *its* depth map's median units (median-depth units) and the render
    scales them by THIS map's median (`zm`). Two models do not share a depth scale, so a path
    estimated on one and rendered on another puts the aim - the pivot - at a different depth than
    intended: the orbit then circles a point in the background while the path itself looks fine.
    The estimator records the model it used; this is where that record is checked.
    """
    declared_model = str((path_data or {}).get("depth_model") or "").strip()
    used_model = str(model_size or "").strip()
    if not declared_model or not used_model or declared_model == used_model:
        return None
    if declared_model.startswith("("):          # injected depth map (tests): nothing to compare
        return None
    return (f"the camera path was estimated with depth model '{declared_model}' but this render "
            f"uses '{used_model}' - their depth maps do not share a scale, so the orbit's aim (the "
            f"pivot) lands at the wrong depth and the camera circles a point in the background. "
            f"Set the SAME depth model on Meridian Parameters and Camera (Depth Model) and on "
            f"Meridian Geometry.")


def _parse_camera_signal(signal: str) -> Tuple[dict, int]:
    """Parse the custom_camera JSON emitted by Meridian Camera Path Configurator (Enndee).

    Validated exactly like the VGGT-side parser: keys must run from frame 0 to frames - 1
    with strictly increasing, real-time `t`/`src` indices, so a signal that would be rejected
    by the subprocess is rejected here too.
    """
    try:
        data = json.loads(signal)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("custom_camera must be a valid signal from Meridian Camera Path Configurator (Enndee).") from exc
    if not isinstance(data, dict) or not isinstance(data.get("path"), list):
        raise ValueError("custom_camera is missing its camera-path keyframes.")
    frames = int(data.get("frames", 73))
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(f"Custom camera path has unsupported frame count {frames}.")
    path = data["path"]
    if len(path) < 2 or not all(isinstance(key, dict) for key in path):
        raise ValueError("Camera path must contain at least two valid keyframe objects.")
    try:
        times = [int(key["t"]) for key in path]
        source_indices = [int(key["src"]) for key in path]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every camera-path key must have integer t and src frame indices.") from exc
    if times[0] != 0 or times[-1] != frames - 1 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Camera-path key times must start at frame 0, end at the frame count minus one and strictly increase.")
    if source_indices != times:
        raise ValueError("Custom camera paths must use real-time source indexing: every key must have src equal to t.")
    for key in path:
        for field in ("pos", "look"):
            vector = key.get(field)
            if not isinstance(vector, list) or len(vector) != 3:
                raise ValueError(f"Camera-path key {field} must be a 3D vector.")
            try:
                finite = all(math.isfinite(float(value)) for value in vector)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"Camera-path key {field} values must be finite.")
    return data, frames


def default_camera_settings() -> dict:
    """All-zero sample.py camera settings the fast backend can honour (frames: picker/args)."""
    return dict(frames=None, yaw=0.0, yaw_from=0.0, truck=0.0, boom=0.0, dolly=1.0, zoom=0.0,
                sweep=False, bounce=False, swing=False, ease=False, aim=False, pivot="",
                pivot_to="", pivot_lock=False, fast_back=1.0, cull=False, follow=False)


def parse_camera_settings(tokens: List[str]) -> dict:
    """Extract the sample.py camera flags the fast backend honours from a token list.

    Meridian Geometry sends the same `args` string to the VGGT subprocess and to this
    backend, so one Meridian Parameter Picker output drives either mode. Flags the fast
    backend cannot honour are simply left out of the result: `--start`, `--freeze` (the
    still is repeated as-is), `--follow` (reported by the caller, which knows whether a
    custom path overrides it), `--smooth`, `--live-speed`, `--canvas`, `--full`,
    `--seed`, `--vggt*`, `--gauge-only` and `--camera-path`.
    """
    settings = default_camera_settings()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option, _, inline = token.partition("=")
        if option in CAMERA_VALUE_FLAGS:
            if inline:
                value = inline
            else:
                index += 1
                if index >= len(tokens):
                    raise ValueError(f"{option} needs a value.")
                value = tokens[index]
            key = CAMERA_VALUE_FLAGS[option]
            if key == "frames":
                try:
                    settings[key] = int(value)
                except ValueError as exc:
                    raise ValueError(f"{option} must be an integer frame count.") from exc
            elif key in ("pivot", "pivot_to"):
                settings[key] = value
            else:
                try:
                    settings[key] = float(value)
                except ValueError as exc:
                    raise ValueError(f"{option} must be a number.") from exc
        elif option in CAMERA_FLAGS:
            settings[CAMERA_FLAGS[option]] = True
        elif option == "--follow":
            settings["follow"] = True
        index += 1
    return settings


def _invert_disparity(pred: torch.Tensor) -> torch.Tensor:
    """Depth-Anything-V2 disparity -> relative depth (larger = farther).

    The model's "depth" is inverse depth: the near pier post reads 4.4 while the far sky
    reads 0.3. 1/x restores depth ordering; `DISPARITY_EPS` keeps the far plane finite and
    the percentile clip downstream absorbs the spike.
    """
    return 1.0 / (pred.clamp(min=0.0) + DISPARITY_EPS)


def prepare_external_depth(external_depth, height: int, width: int, invert: bool = False,
                           aspect_tolerance: float = 0.02, device=None):
    """An external depth map resized onto the working still, in the backend's own gauge.

    Everything downstream needs exactly one convention - **larger value = farther** - because
    the cloud is built by re-gauging the map through a percentile clip onto
    ``DEPTH_NEAR``..``DEPTH_FAR``. That also means an external map only has to be *relative*
    and correctly oriented; its absolute scale is irrelevant. ``invert`` is the switch for
    "bright = near" (disparity-style) exports.

    ``device`` moves the result onto the device the renderer works on. ComfyUI hands an IMAGE
    (or MASK) tensor over on the **CPU**, while every tensor ``render_depth_aligned`` builds
    lives on CUDA - so the map has to be moved here. Leaving it on the CPU did not fail here,
    it failed much later at the cloud unprojection with ``Expected all tensors to be on the
    same device, but found at least two devices, cuda:0 and cpu!``. ``None`` keeps the input's
    device (what the unit tests use).

    Returns ``(depth, note)`` where ``depth`` is ``[height, width]`` float32 and ``note`` is a
    warning string (or None) when the map's aspect ratio does not match the still - the resize
    then stretches it, and the reprojected geometry is subtly skewed.
    """
    if not isinstance(external_depth, torch.Tensor):
        raise ValueError("The external depth input must be a ComfyUI IMAGE tensor.")
    depth = external_depth
    if depth.ndim == 4:
        depth = depth[0]                                   # single-still node: the first frame
    elif depth.ndim == 3:
        # ComfyUI IMAGE is [frames, H, W, C] and a MASK is [frames, H, W]; anything whose last
        # dimension is not a plausible channel count is a frame stack, not a channel-last map.
        if depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0] if depth.shape[-1] == 1 else depth[..., :3].mean(dim=-1)
        else:
            depth = depth[0]
    if depth.ndim == 3:
        depth = depth[..., 0] if depth.shape[-1] == 1 else depth[..., :3].mean(dim=-1)
    if depth.ndim != 2:
        raise ValueError("The external depth map must be [H,W] or [frames,H,W,channels].")
    depth = depth.to(dtype=torch.float32)
    if not bool(torch.isfinite(depth).all()):
        raise ValueError("The external depth map contains non-finite values.")
    span = float(depth.max() - depth.min())
    if span <= 0.0:
        raise ValueError("The external depth map is constant - there is no geometry in it.")

    note = None
    map_h, map_w = int(depth.shape[0]), int(depth.shape[1])
    if map_h > 0 and map_w > 0 and height > 0 and width > 0:
        map_aspect = map_w / map_h
        still_aspect = width / height
        if abs(map_aspect - still_aspect) > aspect_tolerance * still_aspect:
            note = (f"the external depth map is {map_w}x{map_h} (aspect {map_aspect:.3f}) but "
                    f"the still is {width}x{height} (aspect {still_aspect:.3f}) - the map is "
                    "stretched onto the still, so the geometry will be skewed. Render the "
                    "depth from the same framing as the image input.")

    resized = F.interpolate(depth.view(1, 1, map_h, map_w), size=(int(height), int(width)),
                            mode="bilinear", align_corners=False)[0, 0]
    if invert:
        # The percentile clip below re-gauges anyway, so a sign flip is all that is needed.
        resized = -resized
    if device is not None:
        resized = resized.to(device)
    return resized, note


def _edge_keep(depth: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Meridian's 3x3 depth-edge rule: keep pixels whose local spread stays within `tolerance`."""
    mx = F.max_pool2d(depth[None, None], 3, 1, 1)[0, 0]
    mn = -F.max_pool2d(-depth[None, None], 3, 1, 1)[0, 0]
    return ((mx - mn) / depth.clamp(min=1e-6)) <= float(tolerance)


def _upsample_depth_keep(depth: torch.Tensor, keep: torch.Tensor, height: int, width: int):
    """recam/geometry.py `upsample`: bilinear depth, and a hi-res pixel survives only where
    all of its low-res parents did (`> 0.999`). That is what kills bilinear's flying pixels
    across a depth discontinuity instead of smearing silhouettes into the background."""
    d = F.interpolate(depth[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    k = F.interpolate(keep[None, None].float(), size=(height, width),
                      mode="bilinear", align_corners=False)[0, 0] > KEEP_PARENT_RATIO
    return d, k


def _orient_toward_source(points: torch.Tensor) -> torch.Tensor:
    """`--cull` normals: finite differences on the cloud grid, flipped to face the source camera.

    Meridian computes these on its 512-depth grid and orients every normal toward the source
    camera (the world origin of frame 0). It uses `torch.roll`, which wraps the outermost
    row/column into the difference; close to the reference we use forward differences with a
    backward step at the last row/column instead, so the border ring keeps valid normals.
    """
    next_x = torch.zeros_like(points)
    next_x[:, :-1] = points[:, 1:] - points[:, :-1]
    next_x[:, -1] = points[:, -1] - points[:, -2]
    next_y = torch.zeros_like(points)
    next_y[:-1] = points[1:] - points[:-1]
    next_y[-1] = points[-1] - points[-2]
    normals = torch.cross(next_x, next_y, dim=-1)
    normals = normals * torch.sign((normals * -points).sum(-1, keepdim=True))
    return normals.reshape(-1, 3)


def _zip_align(tk: np.ndarray, pk: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson tangents (no overshoot, holds stay perfectly still), matching recam/path.py."""
    d = np.diff(pk, axis=0) / np.diff(tk)[:, None]
    m = np.zeros_like(pk)
    m[0], m[-1] = d[0], d[-1]
    m[1:-1] = 0.5 * (d[:-1] + d[1:])
    m[1:-1][d[:-1] * d[1:] <= 0] = 0.0
    lim = 3.0 * np.minimum(np.abs(d[:-1]), np.abs(d[1:]))
    m[1:-1] = np.clip(m[1:-1], -lim, lim)
    return m


def _catmull_rom(tk: np.ndarray, pk: np.ndarray, t: np.ndarray, ease: List[bool]) -> np.ndarray:
    """Catmull-Rom through (tk, pk) evaluated at t, with Meridian's per-segment cosine ease."""
    m = _zip_align(tk, pk, t)
    K = len(tk)
    out = np.zeros((len(t), pk.shape[1]), dtype=np.float32)
    for i, x in enumerate(t):
        k = min(int(np.searchsorted(tk, x, side="right")) - 1, K - 2)
        h = tk[k + 1] - tk[k]
        s = (x - tk[k]) / h
        if ease[k]:
            s = (1.0 - math.cos(math.pi * s)) / 2.0
        h00 = 2.0 * s ** 3 - 3.0 * s ** 2 + 1.0
        h10 = s ** 3 - 2.0 * s ** 2 + s
        h01 = -2.0 * s ** 3 + 3.0 * s ** 2
        h11 = s ** 3 - s ** 2
        out[i] = h00 * pk[k] + h10 * h * m[k] + h01 * pk[k + 1] + h11 * h * m[k + 1]
    return out


def _horizon_right(f: np.ndarray, up: np.ndarray = WORLD_UP) -> Optional[np.ndarray]:
    """The level-horizon (zero-roll) right of a unit look direction, or None when `f` is vertical.

    This is the look's own right in the world, so the horizon stays level and the frame never rolls
    about the optical axis. `up` is the direction the clip is levelled to: the world up for a level
    path, and the spiral's own (slope-tilted) up for a sloped one - a coil levelled to the world up
    swings tens of degrees relative to its own axis, which is the roll the spiral used to show. It is
    undefined - and reverses - exactly at the pole (the look straight up/down `up`); `camera_frames`
    / `_look_at` hold the previous frame there instead of following this vector over the flip.
    """
    r = np.cross(f, up)
    norm = float(np.linalg.norm(r))
    if norm < HORIZON_MIN_SIN:
        return None
    return (r / norm).astype(np.float32)


def _signed_roll(r_from: np.ndarray, r_to: np.ndarray, f: np.ndarray) -> float:
    """Signed angle (rad) that turns `r_from` into `r_to` about the unit axis `f` (-pi..pi]."""
    return math.atan2(float(np.dot(np.cross(r_from, r_to), f)), float(np.dot(r_from, r_to)))


def _world_up_frame(f: np.ndarray, up: np.ndarray = WORLD_UP) -> Tuple[np.ndarray, np.ndarray]:
    """(right, down) of the level-horizon look-at from a unit forward `f` - zero roll about x.

    This is the renderer's reference frame: the horizon stays level, so an orbit never rolls. It
    only degenerates when `f` is vertical (parallel to `up`), where no horizon exists; there world x
    is kept as the right so the frame stays orthonormal (`camera_frames` handles that pose with
    continuity instead of calling this on its own).
    """
    r = _horizon_right(f, up)
    if r is None:                              # looking straight up/down: keep world x as right
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        r = r - float(np.dot(r, f)) * f
        norm = float(np.linalg.norm(r))
        r = (r / norm).astype(np.float32) if norm > 1e-8 else r.astype(np.float32)
    return r, np.cross(f, r).astype(np.float32)


def _rodrigues(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector `v` about a unit `axis` by `theta` (Rodrigues' rotation formula)."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return (v * cos_t + np.cross(axis, v) * sin_t
            + axis * float(np.dot(axis, v)) * (1.0 - cos_t))


def _transport_step(r_prev: np.ndarray, f_prev: np.ndarray, f_cur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(right, down) of the next frame by *parallel transport* - zero roll about the optical axis.

    The whole frame is carried by the single minimal rotation that takes `f_prev` to `f_cur` (about
    `cross(f_prev, f_cur)`), so `right`/`up` twist as little as possible - zero roll about the view
    axis *per step*. Because the frame is transported rather than re-derived from world up, it stays
    smooth through the pole (the camera looking straight down over the pivot): no flip. If the look
    reverses exactly (the minimal rotation is undefined) the previous `right` is already
    perpendicular to `f_cur` and is kept - still no flip. This alone is *not* the render rule:
    transport accumulates the holonomy of a coil, so `camera_frames` uses it only at the pole
    (`_horizon_step`).
    """
    v = np.cross(f_prev, f_cur)
    s = float(np.linalg.norm(v))
    c = float(np.dot(f_prev, f_cur))
    if s < 1e-8:                                # parallel (same or exact reversal): keep right
        r = r_prev.astype(np.float32).copy()
    else:
        theta = math.atan2(s, c)
        r = _rodrigues(r_prev.astype(np.float32), v / s, theta)
    r = r - float(np.dot(r, f_cur)) * f_cur     # re-orthonormalize against the new forward
    rn = float(np.linalg.norm(r))
    if rn < 1e-8:                               # fully degenerate: any perpendicular will do
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        r = r - float(np.dot(r, f_cur)) * f_cur
        rn = float(np.linalg.norm(r)) or 1.0
    r = (r / rn).astype(np.float32)
    d = np.cross(f_cur, r).astype(np.float32)
    return r, d


def _horizon_step(r_prev: np.ndarray, f_prev: np.ndarray, f_cur: np.ndarray, weight: float,
                  up: np.ndarray = WORLD_UP) -> np.ndarray:
    """Carry `r_prev` to `f_cur` (parallel transport) and roll it `weight` of the way to level.

    `weight` 1 = the level-horizon frame (`_horizon_right`, levelled to `up`): zero roll, the clip
    stays upright. 0 = pure transport: the previous frame carried along the look direction, which is
    what keeps a pole graze continuous. In between the frame walks back to level after such a graze.
    """
    r, _ = _transport_step(r_prev, f_prev, f_cur)
    horizon = _horizon_right(f_cur, up)
    if horizon is None or weight <= 0.0:
        return r
    turned = _rodrigues(r, f_cur, weight * _signed_roll(r, horizon, f_cur))
    turned = turned - float(np.dot(turned, f_cur)) * f_cur
    norm = float(np.linalg.norm(turned))
    return (turned / norm).astype(np.float32) if norm > 1e-8 else r


def camera_frames(forward, up: np.ndarray = WORLD_UP) -> Tuple[np.ndarray, np.ndarray]:
    """(right, down) [N,3] for a sequence of unit forwards: horizon level, no roll, no flip.

    The frame is the level-horizon look-at levelled to `up` - the world up for a level path, the
    spiral's own (slope-tilted) up for a sloped one, so a tilted coil stays upright in *its* frame
    instead of swinging tens of degrees against it. Nothing twists about the optical axis. The one
    pose without a horizon is the pole (the look straight along `up`), where the frame would reverse;
    there the previous frame is carried by parallel transport (`_transport_step`) and walked back to
    level over the next `HORIZON_RELOCK` frames, so a coil that grazes the top re-locks smoothly
    instead of flipping 180 degrees.

    Frame 0 is the plain zero-roll look-at, so a clip opens exactly as it always has. This is the
    single source of the camera's roll: both the renderer (`_evaluate_camera_path`) and the
    auto-camera's measurements use it, so a pixel measured is a pixel rendered.
    """
    f = np.asarray(forward, dtype=np.float32)
    if f.ndim == 1:
        f = f.reshape(1, 3)
    n = f.shape[0]
    right = np.empty_like(f)
    down = np.empty_like(f)
    right[0], down[0] = _world_up_frame(f[0], up)
    previous_horizon = _horizon_right(f[0], up)
    relock = 0
    for i in range(1, n):
        horizon = _horizon_right(f[i], up)
        if horizon is not None:
            if previous_horizon is None or float(np.dot(horizon, previous_horizon)) < 0.0:
                relock = HORIZON_RELOCK         # the horizon just reversed over the pole
            previous_horizon = horizon
        if horizon is None:
            weight = 0.0                        # no horizon to lock to: hold the transported frame
        elif relock > 0:
            weight = (HORIZON_RELOCK - relock) / float(HORIZON_RELOCK)
            relock -= 1
        else:
            weight = 1.0
        right[i] = _horizon_step(right[i - 1], f[i - 1], f[i], weight, up)
        down[i] = np.cross(f[i], right[i]).astype(np.float32)
    return right, down


def _look_at(pos: np.ndarray, look: np.ndarray, prev: Optional[np.ndarray] = None,
             up: np.ndarray = WORLD_UP) -> np.ndarray:
    """c2w rotation (columns right, down, forward) looking from pos at look with the horizon level.

    The frame is the level-horizon look-at (`_world_up_frame`) - zero roll about x - so a moving
    camera keeps the horizon level instead of accumulating roll. `up` is the direction the clip is
    levelled to (the world up, or a tilted path's own up). `prev` (the previous frame's 3x3) supplies
    continuity at the pole only: where the look is along `up`, or the level frame just reversed, the
    previous frame is transported instead, so crossing the top never flips. `prev` is also the hard
    fallback when the look vector itself collapses (pos ~== look). The whole-sequence renderer uses
    `camera_frames`, which spreads that re-lock over `HORIZON_RELOCK` frames.
    """
    f = look - pos
    norm_f = float(np.linalg.norm(f))
    if norm_f < 1e-6:
        return prev if prev is not None else np.eye(3, dtype=np.float32)
    f = (f / norm_f).astype(np.float32)
    if prev is None:
        r, d = _world_up_frame(f, up)
        return np.stack([r, d, f], axis=1)
    previous_horizon = _horizon_right(prev[:, 2], up)
    horizon = _horizon_right(f, up)
    flipped = horizon is not None and (previous_horizon is None
                                       or float(np.dot(horizon, previous_horizon)) < 0.0)
    r = _horizon_step(prev[:, 0], prev[:, 2], f, 0.0 if horizon is None or flipped else 1.0, up)
    return np.stack([r, np.cross(f, r).astype(np.float32), f], axis=1)


def path_up(path_data: Optional[dict]) -> np.ndarray:
    """The direction a camera-path document is levelled to - `up` in the document, else the world up.

    A sloped spiral carries its own up (the world up rotated by the Spiral Center Slope), because a
    coil levelled to the world up swings against its own axis. Everything else - the manual paths,
    the LLM's plans, older documents - has no `up` and keeps the world up it always had.
    """
    values = (path_data or {}).get("up")
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return WORLD_UP
    try:
        vector = np.array([float(value) for value in values], dtype=np.float32)
    except (TypeError, ValueError):
        return WORLD_UP
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-6:
        return WORLD_UP
    return (vector / norm).astype(np.float32)



def _evaluate_camera_path(path_data: dict, frames: int, zm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Custom path keys -> per-frame c2w (F,4,4, float32) and focal multipliers (F,).

    Keys are in frame-0 camera coordinates / pivot depth, exactly like recam/path.py's plan_path.
    The clip is levelled to the document's `up` (the world up when it carries none, a sloped
    spiral's own up when it does - see `path_up`)."""
    keys = path_data["path"]
    tk = np.array([int(key["t"]) for key in keys], dtype=np.float32)
    ease = [bool(key.get("ease", False)) for key in keys]
    pos_keys = np.array([key["pos"] for key in keys], dtype=np.float32)
    look_keys = np.array([key["look"] for key in keys], dtype=np.float32)

    t = np.arange(frames, dtype=np.float32)
    pos = _catmull_rom(tk, pos_keys, t, ease) * zm
    look = _catmull_rom(tk, look_keys, t, ease) * zm
    focal = np.interp(t, tk, [float(key.get("focal", 1.0)) for key in keys]).astype(np.float32)

    c2w = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    forward = look - pos
    lengths = np.linalg.norm(forward, axis=1)
    collapsed = lengths < 1e-6            # pos ~== look: no direction, so the frame cannot move
    forward = forward / np.where(collapsed, 1.0, lengths)[:, None]
    right, down = camera_frames(forward, path_up(path_data))
    rotation = np.eye(3, dtype=np.float32)
    for i in range(frames):
        if not collapsed[i]:
            rotation = np.stack([right[i], down[i], forward[i]], axis=1)
        c2w[i, :3, :3] = rotation
        c2w[i, :3, 3] = pos[i]
    return c2w, focal


def _build_parametric_c2w(frames: int, piv: torch.Tensor, zm: float, yaw: float, truck: float, boom: float,
                          dolly: float, sweep: bool, ease: bool, aim: bool, device: torch.device,
                          yaw_from: float = 0.0, zoom: float = 0.0, bounce: bool = False,
                          swing: bool = False, fast_back: float = 1.0, pivot_to: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Meridian's parametric camera offsets (inference/sample.py) for a static source frame.

    `piv` is the world-space pivot (frame-0 camera coordinates); the source camera sits at the
    origin with identity rotation, so the moved camera is exactly `delta`, as `c2w[t] = I @ delta`.
    The ramp shapes, the `yaw_from -> yaw` sweep, the `--zoom` focal override, the `--fast-back`
    speed map and the sliding `--pivot-to` aim target all follow sample.py line for line."""
    # --bounce/--swing shape a 0->1 ramp; on the constant `ones` ramp they would collapse to
    # zero motion, so they imply the linear base ramp (sample.py: `if args.sweep or ... or ...`).
    if sweep or bounce or swing:
        ramp = torch.linspace(0.0, 1.0, frames, device=device)
    else:
        ramp = torch.ones(frames, device=device)
    if ease:
        ramp = (1.0 - torch.cos(math.pi * ramp)) / 2.0
    if bounce:
        ramp = (1.0 - torch.cos(2.0 * math.pi * ramp)) / 2.0
    if swing:
        ramp = torch.sin(2.0 * math.pi * ramp)
    if fast_back > 1.0:  # piecewise-linear time->angle map: speed v on the outer quarters, K*v on the middle half
        K, v = float(fast_back), 0.5 * (1.0 + 1.0 / float(fast_back))
        t1 = 0.25 / v
        ramp = torch.where(ramp < t1, v * ramp,
                           torch.where(ramp < 1.0 - t1, 0.25 + K * v * (ramp - t1),
                                       0.75 + v * (ramp - 1.0 + t1)))

    c2w = torch.eye(4, device=device, dtype=torch.float32).repeat(frames, 1, 1)
    focals = torch.ones(frames, device=device, dtype=torch.float32)
    for ti in range(frames):
        alpha = float(ramp[ti])
        th = math.radians(yaw_from + (yaw - yaw_from) * alpha)
        r = 1.0 + (dolly - 1.0) * alpha
        cos_t, sin_t = math.cos(th), math.sin(th)
        R = torch.tensor([[cos_t, 0.0, sin_t],
                          [0.0, 1.0, 0.0],
                          [-sin_t, 0.0, cos_t]], device=device, dtype=torch.float32)
        delta = torch.eye(4, device=device, dtype=torch.float32)
        delta[:3, :3] = R
        # orbit about `piv`: sit at r*|piv| from it along the rotated line of sight, then truck/boom in the rotated frame
        delta[:3, 3] = piv - R @ (r * piv) - R @ torch.tensor(
            [-truck * zm * alpha, boom * zm * alpha, 0.0], device=device)
        if aim:  # re-point at the pivot: boom/truck reframe the shot instead of sliding the subject out of frame
            a = piv / piv.norm()
            b = piv + (pivot_to - piv) * alpha - delta[:3, 3] if pivot_to is not None else piv - delta[:3, 3]
            b = b / b.norm()
            v = torch.cross(a, b, dim=0)
            c = float(a @ b)
            sn = float(v.norm())
            if sn > 1e-8:
                K = torch.zeros(3, 3, device=device)
                K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -v[2], v[1], v[2], -v[0], -v[1], v[0]
                delta[:3, :3] = torch.eye(3, device=device) + K + K @ K * ((1.0 - c) / sn ** 2)
        c2w[ti] = delta
        focals[ti] = 1.0 + (zoom - 1.0) * alpha if zoom else r  # sample.py: `--zoom` overrides `f = r`
    return c2w, focals


#: How the depth grid becomes a picture. `mesh` triangulates it (a closed surface: no holes, and
#: the z-buffer resolves occlusion by itself); `points` is the original square-footprint splat.
#:
#: **`points` is still the default, and the near plane turned out not to be the reason.** The mesh
#: used to drop a triangle with any vertex behind the camera; `_clip_triangles` now splits those
#: against the near plane instead (verified against an independent projection of the true 3D
#: crossing, and bounded by `RASTER_GUARD` so a crossing cannot smear the frame). That fixed a real
#: defect and still did not make the mesh usable as a default. Measuring the auto-camera spiral end
#: to end - 73 frames, a bright subject on a dim backdrop, the scenario
#: `test_the_rendered_estimate_keeps_the_subject_in_the_picture_centre` flies:
#:
#:     points   0/73 frames without the subject   worst offset 8.9 px
#:     mesh     1/73 before the clip, 2/73 after   subject 0 px at frame 72
#:
#: At frame 72 the camera has flown *through* the backdrop (diagnostic: `min z` reaches -2.9, and
#: 10171 of 31874 triangles are entirely behind it). A surface renderer then shows the surface - the
#: backdrop fills the view and the subject is legitimately occluded - while the point splat hides
#: that by dropping the occluder's near vertices. So what separates the two is not the near plane at
#: all: it is that this spiral puts the camera inside the cloud, which is a camera-path question.
#: Keep `points` until that is addressed.
SURFACE_MODES = ("points", "mesh")
SURFACE_POINTS = SURFACE_MODES[0]
SURFACE_MESH = SURFACE_MODES[1]
SURFACE_DEFAULT = SURFACE_POINTS


def _quad_triangles(keep: torch.Tensor, depth: torch.Tensor, width: int,
                    tolerance: float = 0.30) -> torch.Tensor:
    """Depth-grid quads -> triangles, keeping only quads that are safe to draw as a surface.

    Two tests, and the second one is the important one:

    1. **Every corner must be trusted** (`keep`: the 3x3 depth-edge rule at model resolution,
       re-checked against the parents on the way up).
    2. **The quad's own four corners must be close enough in depth.** This is the test the point
       renderer got for free - a point cannot stretch - and without it a quad that straddles a
       depth step is drawn as two long triangles that reach from the near surface to the far one.
       Measured on the auto-camera spiral scene, that bridged geometry left the subject 43 px
       off-centre (34 % of the frame height) against the point renderer's 8.9 px, and it made the
       subject's pixel count *grow* mid-flight instead of shrinking. `tolerance` is the same
       relative-spread ratio `_edge_keep` uses, so one widget governs both.

    Returns `[T, 3]` long indices into the flattened grid (row-major, `width` columns), two
    triangles per surviving quad, both wound the same way.
    """
    quad_keep = keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]
    corners = (depth[:-1, :-1], depth[:-1, 1:], depth[1:, :-1], depth[1:, 1:])
    highest = torch.maximum(torch.maximum(corners[0], corners[1]),
                            torch.maximum(corners[2], corners[3]))
    lowest = torch.minimum(torch.minimum(corners[0], corners[1]),
                           torch.minimum(corners[2], corners[3]))
    spread = (highest - lowest) / lowest.abs().clamp(min=1e-6)
    quad_keep = quad_keep & (spread <= float(tolerance))

    rows, cols = torch.nonzero(quad_keep, as_tuple=True)
    v00 = rows * width + cols
    v01 = v00 + 1
    v10 = v00 + width
    v11 = v10 + 1
    return torch.stack([torch.stack([v00, v01, v11], dim=1),
                        torch.stack([v00, v11, v10], dim=1)], dim=1).reshape(-1, 3)


#: Triangles that cross this plane are *split* against it rather than dropped (`_clip_triangles`).
#: The cloud lives in the `DEPTH_NEAR .. DEPTH_FAR` (1.0 .. 5.0) gauge, so this sits 1 % in front of
#: the nearest depth that gauge allows: below any real geometry, and not so small that the clip's
#: perspective weights (`z / near`) lose their precision - at 1e-6 they reach ~1e6 and their
#: difference cancels the interpolated screen position away in float32.
NEAR_PLANE = DEPTH_NEAR * 1e-2
#: A clipped crossing this many canvases away from the frame is treated as "not visible at all" and
#: its triangle is dropped, exactly as the pre-clip code dropped every crossing triangle.
#:
#: Clipping is geometrically right, but a crossing's *screen* position is magnified by `z / near`,
#: so a surface a hair in front of the lens projects millions of pixels off-canvas. That piece is
#: then drawn across the whole frame at a depth that interpolates to ~`near` - it covers everything
#: (measured on the auto-camera spiral: the subject went from 3529 px to 0 px from frame 36 on, and
#: the flight's 73 frames went from 1 empty to 39) - and at 1e8 px the float32 edge functions in
#: `_rasterize_bucket` cancel the barycentric weights away entirely. Measured, sweeping the bound:
#: 1e8 -> 40 empty frames, 1e6 -> 23, 1e4 -> 23, one canvas -> 2, against 1 for dropping outright.
#: A bound of a canvas or two is also the principled place for it: past that, the `1 / z`
#: interpolation across the frame is dominated by the crossing rather than by the visible vertices.
#: Every crossing that lands near the frame is still clipped and drawn.
RASTER_GUARD = 2


def _clip_triangles(tu, tv, tz, tc, near=NEAR_PLANE, guard=None):
    """Split every triangle that crosses `z = near` instead of dropping it whole.

    A depth grid is a *surface*, so the quad at the edge of the frustum is usually half visible: one
    or two of its corners are behind the camera and the rest is in front of it. Dropping such a
    triangle whole is what emptied the subject at the end of the auto-camera spiral (frame 72: 0
    subject pixels, against the point renderer's 318 - a point loses only the vertex that crossed,
    so a surface must lose only the part behind the camera). This is a Sutherland-Hodgman clip
    against the single plane `z = near`:

        three vertices in front -> the triangle, untouched
        two in front            -> a quad  (a, b, p, q) -> two triangles
        one in front            -> a triangle (a, p, q)  -> one triangle
        none                    -> dropped

    All of it is vectorised: each triangle is rotated so that its pattern of inside/outside vertices
    matches one of those cases, and one code path then covers all three.

    The crossing point needs no camera intrinsics. The projection is `u = c + f * x / z`, so on an
    edge from a vertex in front to one behind, the crossing's screen position is the
    perspective-correct blend `u_in + beta * (u_out - u_in)` with `beta = z_out * t / near` and
    `t = (near - z_in) / (z_out - z_in)`: the endpoint weights `z_in(1-t)/near` and `z_out*t/near`
    sum to 1 by construction of `t`, so `c` and `f` cancel out. Depth and colour do **not** use that
    weight - both are linear along the edge in 3D - so they take `t`, and the crossing's depth is
    exactly `near`.

    Returns the clipped `(tu, tv, tz, tc)`: at most twice as long as the input, every vertex at or
    in front of the near plane. With `guard` set, a triangle whose crossings land further than that
    from the origin is dropped rather than drawn - see `RASTER_GUARD`.
    """
    inside = tz >= float(near)
    count = inside.sum(dim=1)
    keep = count > 0
    if not bool(keep.any()):
        return tu[:0], tv[:0], tz[:0], tc[:0]
    tu, tv, tz, tc = tu[keep], tv[keep], tz[keep], tc[keep]
    count = count[keep]

    # Rotate so that `a` is in front and `c` behind in both clipped patterns: the single *behind*
    # vertex goes last when two are in front, and the single *in front* vertex goes first when only
    # one is. A triangle that is entirely in front is already (in, in, in) - `first` lands on 0.
    flags = inside[keep].to(torch.uint8)
    first = torch.where(count == 2, (torch.argmin(flags, dim=1) + 1) % 3,
                        torch.argmax(flags, dim=1))
    order = (first[:, None] + torch.arange(3, device=tu.device)[None, :]) % 3
    attrs = torch.stack([torch.gather(part, 1, order) for part in (tu, tv, tz)], dim=2)
    attrs = torch.cat([attrs, torch.gather(tc, 1, order[:, :, None].expand(-1, -1, 3))], dim=2)

    def vertex(index):
        """`[T, 3, 6]` -> the `(u, v, z, r, g, b)` at `index`, one index per triangle."""
        picked = torch.gather(attrs, 1, index[:, None, None].expand(-1, 1, attrs.shape[2]))
        return picked.squeeze(1)

    def crossing(start, end):
        """The near-plane crossing of `start -> end` (`start` in front, `end` behind)."""
        z_in, z_out = start[:, 2], end[:, 2]
        denominator = z_out - z_in
        # Two vertices at the same depth never straddle the plane, so this only guards the
        # all-in-front case - whose result is discarded below anyway.
        t = (float(near) - z_in) / torch.where(denominator == 0, torch.ones_like(denominator),
                                               denominator)
        beta = (z_out * t / float(near))[:, None]
        return torch.cat([start[:, :2] + beta * (end[:, :2] - start[:, :2]),
                          torch.full_like(start[:, :1], float(near)),
                          start[:, 3:] + t[:, None] * (end[:, 3:] - start[:, 3:])], dim=1)

    zeros = torch.zeros_like(count)
    twos = torch.full_like(count, 2)
    a, b, c = vertex(zeros), vertex(twos - 1), vertex(twos)
    # `q` always crosses the c -> a edge. `p` starts at the second vertex in front - b when two are
    # in front, a when only one is - which is exactly where the visible piece begins.
    p_from = torch.where(count == 2, twos - 1, zeros)
    p = crossing(vertex(p_from), vertex(p_from + 1))
    q = crossing(c, a)
    # A triangle that is entirely in front has nothing to clip, so it comes back whole: `p` becomes
    # its third vertex and `q` its first, which makes the second slot degenerate rather than wrong.
    in_front = (count == 3)[:, None]
    p, q = torch.where(in_front, c, p), torch.where(in_front, a, q)

    tris = torch.stack([torch.stack([a, b, p], dim=1),
                        torch.stack([a, p, q], dim=1)], dim=1).reshape(-1, 3, attrs.shape[2])
    # Drop the slots that cannot hold area: the second one for a fully-visible triangle, and the
    # first for a one-vertex clip, where a, b and p are collinear. Exact, not an area threshold.
    live = torch.stack([count >= 2, count <= 2], dim=1).reshape(-1)
    if guard is not None:
        # A crossing this far out is the surface a hair in front of the lens: its piece covers the
        # frame at ~`near` depth and its coordinates break the rasteriser's float32 maths.
        beyond = ((p[:, :2].abs().amax(dim=1) > float(guard))
                  | (q[:, :2].abs().amax(dim=1) > float(guard)))
        live = live & ~beyond.repeat_interleave(2)
    tris = tris[live]
    return tris[:, :, 0], tris[:, :, 1], tris[:, :, 2], tris[:, :, 3:]


#: Rasteriser size buckets. A triangle whose screen-space bounding box is wider than
#: `RASTER_BUCKETS[i]` lands in bucket `i + 1`, and each bucket pads its offset grid to its own
#: cap. Heightfield quads mostly land in the first buckets (a few pixels), so the padding waste
#: stays close to the work actually needed instead of being dominated by the one huge
#: near-camera triangle.
RASTER_BUCKETS = (2, 4, 8, 16, 32, 64, 128)
#: Candidate pixels per rasteriser chunk - keeps each chunk's temporaries a few MB.
RASTER_CHUNK_BUDGET = 1_000_000


def _fill_render_holes(img, hole_color=HOLE_COLOR, levels=6):
    """Fill `hole_color` pixels from their rendered neighbours - MULTI-SCALE.

    The mesh surface closes holes *within* the subject, but RMBG removed the background so
    there is no geometry there at all: a tight face close-up reads as a solid head floating
    in a grey field. This inpaints the uncovered pixels so the frame reads as a picture.

    A previous version ran an iterative 3x3 convolution. It was BOTH slow (32 passes per
    frame x 158 frames, and the user noticed the camera-path render grinding) and far too
    weak - N passes only reach N pixels inward, so a full-frame background needs hundreds.
    The pyramid does it in O(log N) passes: repeatedly halve the image (weighting by the
    known mask so holes do not dilute the estimate), fill whatever is left at the coarse
    level with the known mean, then upsample and composite the real pixels back over the
    estimate. Arbitrarily large holes, a handful of passes.

    Pure post-process: depth, the splat and the camera are untouched, so no geometry
    measurement changes. `FILL_RENDER_HOLES = False` returns the raw rasteriser output.
    """
    if img is None or img.numel() == 0 or not FILL_RENDER_HOLES:
        return img
    height, width = int(img.shape[0]), int(img.shape[1])
    flat = img.reshape(height, width, -1).to(torch.float32)
    is_hole = (flat == float(hole_color)).all(dim=-1)
    if not bool(is_hole.any()):
        return img
    known = (~is_hole).to(torch.float32)

    # PASS 1: build a weighted pyramid of colour and coverage.
    colours, masks = [flat * known.unsqueeze(-1)], [known]
    for _ in range(int(levels)):
        c, m = colours[-1], masks[-1]
        if min(int(c.shape[0]), int(c.shape[1])) < 2:
            break
        cc = torch.nn.functional.avg_pool2d(
            c.permute(2, 0, 1).unsqueeze(0), 2, 2)[0].permute(1, 2, 0)
        mm = torch.nn.functional.avg_pool2d(
            m.unsqueeze(0).unsqueeze(0), 2, 2)[0, 0]
        colours.append(cc)
        masks.append(mm)

    # PASS 2: coarsest level - anything still unknown takes the overall known mean.
    total = masks[0].sum().clamp(min=1.0)
    mean_colour = colours[0].sum(dim=(0, 1)) / total
    guess = colours[-1] / masks[-1].unsqueeze(-1).clamp(min=1e-6)
    hole_at_top = masks[-1] < 0.999
    if bool(hole_at_top.any()):
        guess = torch.where(hole_at_top.unsqueeze(-1),
                            mean_colour.reshape(1, 1, -1), guess)

    # PASS 3: upsample the estimate, composite the real pixels back over it.
    for level in range(len(colours) - 2, -1, -1):
        up = torch.nn.functional.interpolate(
            guess.permute(2, 0, 1).unsqueeze(0),
            size=(int(colours[level].shape[0]), int(colours[level].shape[1])),
            mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        real = colours[level] / masks[level].unsqueeze(-1).clamp(min=1e-6)
        w = (masks[level] > 0.999).unsqueeze(-1)
        guess = torch.where(w, real, up)

    return guess.reshape(height, width, -1).clamp(0.0, 255).to(torch.uint8)


def _rasterize_mesh(verts_uv, verts_z, colors, tris, out_w, out_h, hole_color):
    """Z-buffered triangle rasteriser: a depth-grid mesh drawn with real occlusion and no holes.

    `verts_uv` is `[N, 2]` in canvas pixels, `verts_z` `[N]` camera depth (larger = farther),
    `colors` `[N, 3]` uint8, `tris` `[T, 3]` indices into them.

    Triangles are visited through their screen-space bounding box, grouped into size buckets so
    the padding waste tracks the work actually needed. Coverage is `>= 0` on all three edge
    functions for **both** triangles of a quad, so their shared diagonal is covered twice rather
    than not at all - double coverage costs nothing (the z-test keeps one) and it is what makes
    the surface watertight. Depth is interpolated perspective-correctly through `1/z`; the colour
    is interpolated affinely, the usual "smooth shading" approximation, which is exact enough at
    the quad sizes this grid produces.

    Anything crossing the camera plane is clipped against `NEAR_PLANE` first
    (`_clip_triangles`), so a triangle is only lost when the whole of it is behind the camera.
    """
    zbuf = torch.full((out_h * out_w,), float("inf"), device=verts_uv.device)
    img = torch.full((out_h * out_w, 3), hole_color, dtype=torch.uint8, device=verts_uv.device)
    if tris.numel() == 0:
        return img.view(out_h, out_w, 3)

    tu, tv = verts_uv[tris, 0], verts_uv[tris, 1]
    tz = verts_z[tris]
    tc = colors[tris].to(torch.float32)
    # A triangle that crosses the camera plane is *split* against the near plane, not dropped: a
    # depth grid is a surface, so the quad at the edge of the frustum is usually half visible.
    tu, tv, tz, tc = _clip_triangles(tu, tv, tz, tc,
                                     guard=RASTER_GUARD * max(out_w, out_h))
    if tz.numel() == 0:
        return img.view(out_h, out_w, 3)
    # The clip leaves every vertex at or in front of the near plane, which is what the projection
    # and the `1 / z` depth below need. Kept as the rasteriser's own contract, since it is also
    # called directly.
    live = (tz > 0).all(dim=1)

    x0 = torch.floor(tu.min(1).values).clamp(0, out_w - 1)
    y0 = torch.floor(tv.min(1).values).clamp(0, out_h - 1)
    x1 = torch.ceil(tu.max(1).values).clamp(0, out_w - 1)
    y1 = torch.ceil(tv.max(1).values).clamp(0, out_h - 1)
    live &= (x1 >= x0) & (y1 >= y0)

    extents = torch.maximum(x1 - x0 + 1, y1 - y0 + 1)
    bounds = torch.tensor(RASTER_BUCKETS, device=verts_uv.device, dtype=torch.float32)
    bucket = torch.bucketize(extents, bounds)
    caps = tuple(RASTER_BUCKETS) + (max(out_w, out_h),)
    largest = len(caps) - 1

    for index, cap in enumerate(caps):
        picked = torch.nonzero(live & (bucket == index), as_tuple=True)[0]
        if picked.numel() == 0:
            continue
        if index == largest:      # pad to what this bucket holds, not to the whole canvas
            cap = min(cap, int(extents[picked].max().item()) + 1)
            print(f"[Enndee] Meridian fast depth: {picked.numel()} triangle(s) project wider than "
                  f"{RASTER_BUCKETS[-1]} px (up to {cap}); those cost a canvas-sized pass each.",
                  flush=True)
        axis = torch.arange(cap, device=verts_uv.device, dtype=torch.float32)
        oy, ox = torch.meshgrid(axis, axis, indexing="ij")
        _rasterize_bucket(tu, tv, tz, tc, picked, cap, ox.reshape(1, -1), oy.reshape(1, -1),
                          zbuf, img, out_w, out_h)

    return img.view(out_h, out_w, 3)


def _rasterize_bucket(tu, tv, tz, tc, picked, cap, ox, oy, zbuf, img, out_w, out_h):
    """Rasterise one size bucket's triangles into `zbuf` / `img` (see `_rasterize_mesh`).

    `ox`/`oy` are the bucket's offset grids (`[1, cap*cap]`); every triangle in the bucket fits
    inside a `cap x cap` box, so the box is padded to that and masked back down to its real size.
    """
    chunk = max(1, min(200_000, RASTER_CHUNK_BUDGET // (cap * cap)))
    for start in range(0, picked.numel(), chunk):
        sel = picked[start:start + chunk]
        ax, ay, az = tu[sel, 0], tv[sel, 0], tz[sel, 0]
        bx, by, bz = tu[sel, 1], tv[sel, 1], tz[sel, 1]
        cx, cy, cz = tu[sel, 2], tv[sel, 2], tz[sel, 2]

        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        usable = area != 0
        if not bool(usable.any()):
            continue
        # A signed area means half the triangles wind the other way; folding the sign into the
        # edge tests lets both windings share one code path.
        sign = torch.where(area < 0, -1.0, 1.0)
        inv_area = 1.0 / torch.where(usable, area, torch.ones_like(area))

        left = torch.floor(torch.minimum(torch.minimum(ax, bx), cx)).clamp(0, out_w - 1)
        top = torch.floor(torch.minimum(torch.minimum(ay, by), cy)).clamp(0, out_h - 1)
        right = torch.ceil(torch.maximum(torch.maximum(ax, bx), cx)).clamp(0, out_w - 1)
        bottom = torch.ceil(torch.maximum(torch.maximum(ay, by), cy)).clamp(0, out_h - 1)

        px = left[:, None] + ox
        py = top[:, None] + oy
        # Clamping the box above is what keeps the scatter index in range as well as on-canvas.
        valid = ((ox < (right - left + 1)[:, None]) & (oy < (bottom - top + 1)[:, None])
                 & usable[:, None])

        e0 = (bx - ax)[:, None] * (py - ay[:, None]) - (by - ay)[:, None] * (px - ax[:, None])
        e1 = (cx - bx)[:, None] * (py - by[:, None]) - (cy - by)[:, None] * (px - bx[:, None])
        e2 = (ax - cx)[:, None] * (py - cy[:, None]) - (ay - cy)[:, None] * (px - cx[:, None])
        inside = (valid & (e0 * sign[:, None] >= 0) & (e1 * sign[:, None] >= 0)
                  & (e2 * sign[:, None] >= 0))
        if not bool(inside.any()):
            continue

        wa = e1 * inv_area[:, None]      # weight of vertex a (it sits opposite edge b->c)
        wb = e2 * inv_area[:, None]
        wc = e0 * inv_area[:, None]
        depth = 1.0 / (wa / az[:, None] + wb / bz[:, None] + wc / cz[:, None])
        colour = (wa[..., None] * tc[sel, 0][:, None, :]
                  + wb[..., None] * tc[sel, 1][:, None, :]
                  + wc[..., None] * tc[sel, 2][:, None, :])

        pix = (py.long() * out_w + px.long())[inside]
        zz = depth[inside]
        cc = colour[inside].round().clamp(0, 255).to(torch.uint8)
        # Same two-step z-buffer the point renderer uses: keep the nearest depth per pixel, then
        # write the colours that own it. A later bucket overwrites only where it is nearer.
        zbuf.scatter_reduce_(0, pix, zz, "amin", include_self=True)
        win = zz == zbuf[pix]
        img[pix[win]] = cc[win]


def render_depth_aligned(first, device, model_size="Depth-Anything-V2-Small-hf", frames=73,
                         canvas_mode="auto_meridian480", custom_width=832, custom_height=480,
                         cloud_scale=2, point_size=1, edge_cull=True, edge_threshold=0.30,
                         back_face_cull=False, camera=None, custom_camera=None, depth_res=DA3_RES,
                         external_depth=None, external_depth_invert=False,
                         moge_refine_steps=MOGE_REFINE_STEPS, surface_mode=SURFACE_DEFAULT,
                         depth_relief=1.0):
    """Depth-aligned camera-flight condition renderer (Depth-Anything-V2/V3 + GPU point-cloud renderer).

    `first`          [1,H,W,3] float tensor in [0,1]: the still the flight starts from.
    `camera`         settings dict from `parse_camera_settings` (None = every default).
    `custom_camera`  Meridian Parameters and Camera signal; its path and frame count win.
    `back_face_cull` mirrors Meridian's `--cull` and is the switch the Enndee nodes use (the
                     Geometry node's widget); a `--cull` token in `camera` - hand-written args
                     or the original CLI - is OR-ed onto it, since the widget cannot express
                     "leave it alone".
    `depth_res`      working-resolution cap in pixels: the still's longest side is resized down
                     to it (`_fit_working_still`, never up) before the depth model, the frame
                     colours and the cloud grid are built, and Depth-Anything-3 additionally
                     receives it as its own `process_res` (the library rounds to multiples of
                     14). `DA3_RES` (504) is the fast default; 0 = the still's own resolution
                     (maximum depth detail, bounded only by the `MAX_CLOUD_PIXELS` ceiling).
                     The V2 models keep their native 518 depth grid but the still and the cloud
                     follow the same cap.
    `external_depth` an optional ComfyUI IMAGE depth map that REPLACES the depth model. It is
                     resized onto the working still and then goes through the very same edge
                     cull, percentile clip and cloud grid as a predicted map, so it only has to
                     be *relative* depth with the right orientation: larger value = farther,
                     unless `external_depth_invert` flips a "bright = near" export. This is the
                     hook for a prior the model cannot beat - metric depth from a stereo/COLMAP
                     pass, a hand-painted map, or a multi-view-consistent depth - and it is the
                     only way to guarantee that two shots of the same scene share one geometry.
    `moge_refine_steps` MoGe-3 only: the number of self-guided sparse volumetric refinement
                     steps (0 disables the stage, 3 is MoGe's own default, 12 is the widget's
                     ceiling). Every other depth source ignores it.
    `surface_mode`   `points` (default) is the original square-footprint splat renderer;
                     `mesh` triangulates the depth grid instead, so the flight is a closed
                     surface: real occlusion, no holes between points, no square footprints.
                     `point_size` and the back-face cull only apply to `points`;
                     `edge_cull`/`edge_threshold` apply to both - for the mesh they are what cuts
                     the sheet at a silhouette (and at a depth step *inside* the sheet) instead of
                     bridging it with stretched triangles. The mesh is opt-in while it lacks
                     near-plane clipping; see `SURFACE_DEFAULT`.

    Returns exactly what the VGGT geometry pass returns - (source, render, width, height, length) -
    so the pair drops straight into MeridianRefConditioning as `<Video 1>` / `<Video 2>`.
    """
    settings = dict(default_camera_settings())
    if camera:
        settings.update(camera)
    if str(surface_mode) not in SURFACE_MODES:
        raise ValueError(f"surface_mode must be one of {', '.join(SURFACE_MODES)}, "
                         f"got {surface_mode!r}.")
    use_mesh = str(surface_mode) == SURFACE_MESH
    if not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[0] < 1:
        raise ValueError("The image input must be a ComfyUI IMAGE batch [frames, height, width, channels].")
    if first.shape[-1] < 3:
        raise ValueError("The image batch must carry at least 3 channels per frame (RGB).")
    if first.shape[-1] > 4:
        raise ValueError("The image batch must have at most 4 channels per frame.")
    first = first[0:1].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    _, src_h, src_w, _ = first.shape

    # working still: `depth_res` caps the picture's longest side, so an oversized input (a 24 Mpx
    # camera photo) is resized right here - the depth model, the frame colours and the cloud grid
    # all work from the smaller still, which keeps the pool of the percentile clip below
    # torch.quantile's hard 2**24-element ceiling. `cloud_scale` alone could never do that: the
    # scale loop below stops at 1, so a source already over MAX_CLOUD_PIXELS stayed huge.
    first, resize_note = _fit_working_still(first, depth_res)
    if resize_note:
        print(f"[Enndee] Meridian fast depth: {resize_note}", flush=True)
    _, work_h, work_w, _ = first.shape

    # canvas: Meridian's 480-class condition ladder (the trained reference canvas), or explicit
    if canvas_mode == "custom":
        out_w, out_h = int(custom_width), int(custom_height)
    else:
        out_w, out_h = _bucket_480(src_w, src_h)

    # --- depth: an EXTERNAL map, MoGe-3 or Depth-Anything-V3 (both already depth) or V2
    # (disparity -> invert). Every branch returns "larger = farther" at the working still's size.
    # `measured_fov` stays None unless the source can report the camera it saw the still through.
    measured_fov = None
    if external_depth is not None:
        depth_low, aspect_note = prepare_external_depth(
            external_depth, work_h, work_w, invert=bool(external_depth_invert),
            device=device)
        if aspect_note:
            print(f"[Enndee] Meridian fast depth: {aspect_note}", flush=True)
        print("[Enndee] Meridian fast depth: using the CONNECTED depth map "
              f"(external depth, {work_w}x{work_h} working still) - no depth model is loaded.",
              flush=True)
    elif model_size.startswith(MOGE_PREFIX):
        depth_low, measured_fov = _predict_moge_depth_and_fov(model_size, first, device,
                                                              moge_refine_steps)
    elif model_size.startswith(DA3_PREFIX):
        depth_low = _predict_da3_depth(model_size, first, device,
                                       _da3_process_res(depth_res, work_w, work_h))
    else:
        depth_model = _get_depth_model(model_size, device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x518 = F.interpolate(first.permute(0, 3, 1, 2), size=(DEPTH_RES, DEPTH_RES),
                             mode="bilinear", align_corners=False)
        with torch.no_grad():
            pred = depth_model(pixel_values=((x518 - mean) / std).half()).predicted_depth[0].float()
        depth_low = _invert_disparity(pred)

    # Every depth source must live on the render device. The external map arrives on the CPU from
    # ComfyUI, and the unprojection below multiplies it with CUDA meshgrids - without this the run
    # dies with "Expected all tensors to be on the same device, cuda:0 and cpu".
    depth_low = depth_low.to(device)

    # The source camera's vertical FOV, shared by BOTH the unprojection and the target canvas -
    # they have to be the same physical camera. A depth model that also reports intrinsics
    # (MoGe-3 does) beats the constant assumption, and the difference is not cosmetic: measured
    # 42.53 against an assumed 55 stretches the cloud in depth and skews the parallax of a moving
    # camera, while the *identity* frame still looks perfect. That is precisely the failure this
    # renderer cannot afford.
    vfov_degrees = VFOV_DEGREES if measured_fov is None else float(measured_fov)
    # PUBLISHED so the auto-camera's face orbit can project with the SAME focal the cloud
    # was built with. This value used to be a local, so `surface["vfov"]` came back None
    # and every face-orbit projection fell back to the hard-coded VFOV_DEGREES - a 1.509x
    # error in f on the Hanks still - which selected a head patch ~4x too big and made the
    # zoom solve ask for 2.63 units of stand-off instead of ~0.65.
    LAST_VFOV_DEGREES = vfov_degrees
    if measured_fov is not None:
        print(f"[Enndee] Meridian fast depth: source FOV measured at {vfov_degrees:.2f} deg "
              f"(vertical) instead of the assumed {VFOV_DEGREES:.0f} deg.", flush=True)

    # keep (model grid): Meridian's 3x3 depth-edge rule; the 2 % confidence pruning of
    # recam/geometry.py is not ported for either family, so nothing else is culled
    keep_low = _edge_keep(depth_low, edge_threshold) if edge_cull else torch.ones_like(depth_low, dtype=torch.bool)

    # cloud grid: the frame (and the depth under it) upscaled so silhouettes get finer points;
    # `_upsample_depth_keep` re-checks the parents, so bilinear cannot bridge a depth jump
    scale = max(1, int(cloud_scale))
    while scale > 1 and (work_w * scale) * (work_h * scale) > MAX_CLOUD_PIXELS:
        scale -= 1
    cloud_w, cloud_h = work_w * scale, work_h * scale
    depth_cloud, keep_cloud = _upsample_depth_keep(depth_low, keep_low, cloud_h, cloud_w)

    pool = depth_cloud[keep_cloud] if keep_cloud.any() else depth_cloud.flatten()
    # ONE map, shared with the automatic estimator and the camera path - see `depth_gauge_scale`.
    # It is proportional on purpose: an affine map with an offset scales every point radially by a
    # depth-dependent factor, which squashes the depth relief of whatever the flight is about
    # (measured: 5.7x flat on MoGe-3, whose metric depth spans ~35 against Depth-Anything-3's ~3.8,
    # while the same map is ~1.03x - i.e. correct - on DA3).
    gauge_scale = depth_gauge_scale(pool)
    depth_map = (depth_cloud * gauge_scale).clamp(min=0.05)

    def mapped_depth(raw):
        """A raw model depth expressed in the *cloud's* gauge - the same map the cloud is built with.

        The camera keys arrive in "median-depth units" (the Camera Path Configurator's and the
        automatic estimator's contract: 1.0 = the cloud's median depth), so the scale that turns
        them into world units must be the median of the *mapped* depth the rendered cloud uses.
        Using the raw model median instead left the whole camera rig ~2.3x too close to the
        origin: the orbit centre then sat well *in front of* the subject and the subject swung out
        of the picture along the path (measured: pivot z 0.47 world vs the subject at 0.94-1.20).

        `gauge_scale` is `depth_gauge_scale`'s slope, so this is a plain scale - the one property
        that keeps the map shape-preserving.
        """
        return float(gauge_scale) * float(raw)

    # --- pivot: median depth in the +-5 % window of the picked point, on the model's own grid ----
    # Depth-Anything-3 returns an aspect-preserving grid (the longest side is `process_res`), so
    # crop fractions map onto its own width/height instead of a single square side.
    low_h, low_w = depth_low.shape[-2:]
    f_cloud = 0.5 * cloud_h / math.tan(math.radians(vfov_degrees) / 2.0)
    cxc, cyc = cloud_w / 2.0, cloud_h / 2.0

    def picked_point(value, label):
        """'x,y' crop fractions -> (fractions, median depth in their +-5 % window); None if unset."""
        text = "" if value is None else str(value).strip()
        if not text or text == "none":
            return None, None
        try:
            fx, fy = (float(part) for part in text.replace(" ", "").split(","))
        except ValueError as exc:
            raise ValueError(f"{label} must be 'x,y' in 0..1 crop coordinates, 'none', or empty.") from exc
        fx, fy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        px, py = fx * low_w, fy * low_h
        rw, rh = PIVOT_WINDOW * low_w, PIVOT_WINDOW * low_h
        window = (slice(max(0, int(py - rh)), min(low_h, int(py + rh) + 1)),
                  slice(max(0, int(px - rw)), min(low_w, int(px + rw) + 1)))
        sample = depth_low[window][keep_low[window]]
        if not sample.numel():
            sample = depth_low[keep_low] if keep_low.any() else depth_low.flatten()
        return (fx, fy), float(sample.median())

    pivot_frac, pivot_depth = picked_point(settings.get("pivot"), "--pivot")
    if pivot_depth is not None:
        zm_raw = pivot_depth
    else:
        zm_raw = float(depth_low[keep_low].median()) if keep_low.any() else float(depth_low.median())
    zm = mapped_depth(zm_raw)     # the gauge the cloud lives in; `zm_raw` is kept for the log
    piv = torch.tensor([0.0, 0.0, zm], device=device)
    if bool(settings.get("pivot_lock")) and pivot_frac is not None:  # orbit about the picked pixel
        fx, fy = pivot_frac
        piv = torch.tensor([(fx * cloud_w - cxc) / f_cloud * zm,
                            (fy * cloud_h - cyc) / f_cloud * zm, zm], device=device)
    pivot_to = piv
    aim_frac, aim_depth = picked_point(settings.get("pivot_to"), "--pivot-to")
    if aim_frac is not None and aim_depth is not None:   # --pivot-to: the aim target slides to it
        fx, fy = aim_frac
        aim_z = mapped_depth(aim_depth)
        pivot_to = torch.tensor([(fx * cloud_w - cxc) / f_cloud * aim_z,
                                 (fy * cloud_h - cyc) / f_cloud * aim_z, aim_z], device=device)

    # --- cloud: frame + depth unprojected in frame-0 camera coordinates (the source c2w is I) ----
    yy, xx = torch.meshgrid(torch.arange(cloud_h, device=device, dtype=torch.float32),
                            torch.arange(cloud_w, device=device, dtype=torch.float32), indexing="ij")
    # Depth RELIEF exaggeration about the median. `zm` is the median, so this leaves the
    # gauge - and therefore the whole camera rig - untouched, while scaling the depth
    # spread. x/y use the SAME relieved z, so x/z is unchanged and the identity frame
    # still reprojects onto the source still exactly.
    depth_map = apply_depth_relief(depth_map, depth_relief)
    # ASPECT CORRECTION. The cloud is built at the working still's aspect but rendered
    # into the 480-class ladder canvas, and the two differ (measured on the Hanks still:
    # 1.4884 vs 1.3529). Exact reprojection of the identity frame needs
    # `f_canvas * cloud_w / (f_cloud * out_w) == 1`, i.e. `aspect_cloud == aspect_canvas`.
    # When they disagree the subject is drawn `aspect_cloud / aspect_canvas` = 1.100x too
    # wide and overflows the frame horizontally - a real ~10 % "too flat" contribution.
    # Scaling x by `aspect_canvas / aspect_cloud` restores exact reprojection.
    aspect_canvas = float(out_w) / float(out_h)
    aspect_cloud = float(cloud_w) / float(cloud_h)
    x_scale = aspect_x_scale(cloud_w, cloud_h, out_w, out_h)
    points = torch.stack([(xx - cxc) / f_cloud * depth_map * x_scale,
                          (yy - cyc) / f_cloud * depth_map, depth_map], dim=-1)
    colors_hi = F.interpolate(first.permute(0, 3, 1, 2), size=(cloud_h, cloud_w),
                              mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    flat_keep = keep_cloud.reshape(-1)
    pts = points.reshape(-1, 3)[flat_keep]
    cols = (colors_hi.reshape(-1, 3)[flat_keep] * 255.0).round().to(torch.uint8)

    # --- cull: `--cull` drops the splats the target camera would see from behind -----------------
    # Only the point renderer needs it: a rasterised surface is occluded by the z-buffer already.
    cull = (bool(settings.get("cull")) or bool(back_face_cull)) and not use_mesh
    normals = _orient_toward_source(points)[flat_keep] if cull else None

    # --- camera: the authored path wins; otherwise Meridian's parametric offsets -----------------
    if custom_camera is not None:
        path_data, num_frames = _parse_camera_signal(custom_camera)
        mismatch = warn_depth_model_mismatch(path_data, model_size)
        if mismatch:
            print(f"[Enndee] WARNING: {mismatch}", flush=True)
        c2w_np, focal_np = _evaluate_camera_path(path_data, num_frames, zm)
        c2w = torch.from_numpy(c2w_np).to(device)
        focal = torch.from_numpy(focal_np).to(device)
    else:
        num_frames = int(frames)
        c2w, focal = _build_parametric_c2w(num_frames, piv, zm, float(settings["yaw"]),
                                           float(settings["truck"]), float(settings["boom"]),
                                           float(settings["dolly"]), bool(settings["sweep"]),
                                           bool(settings["ease"]), bool(settings["aim"]), device,
                                           yaw_from=float(settings["yaw_from"]), zoom=float(settings["zoom"]),
                                           bounce=bool(settings["bounce"]), swing=bool(settings["swing"]),
                                           fast_back=float(settings["fast_back"]), pivot_to=pivot_to)


    # target intrinsics at the canvas: same vertical FOV, focal multipliers from the path applied
    # to both axes exactly as `intr_t[ti, 0, 0] *= f; intr_t[ti, 1, 1] *= f` in sample.py
    f_canvas = 0.5 * out_h / math.tan(math.radians(vfov_degrees) / 2.0)
    focal_px = f_canvas * focal.to(device)      # [F]

    # --- surface: the depth grid as a mesh or as splatted points (`surface_mode`) ----------------
    # The mesh's topology and per-vertex colour are fixed for the whole flight - only the
    # projection changes per frame. `keep_cloud` doubles as the silhouette cut, so `edge_cull` /
    # `edge_threshold` keep their meaning; `point_size` and the back-face cull were point-renderer
    # workarounds that a z-buffered surface does not need.
    if use_mesh:
        mesh_verts = points.reshape(-1, 3)
        mesh_cols = (colors_hi.reshape(-1, 3) * 255.0).round().to(torch.uint8)
        tris = _quad_triangles(keep_cloud, depth_cloud, cloud_w, edge_threshold)
    else:
        mesh_verts = mesh_cols = tris = None

    # --- render: z-buffered flight along the camera path (recam/geometry.py render_hw) ---------
    cxc_out, cyc_out = out_w / 2.0, out_h / 2.0
    radius = max(0, int(point_size))
    offs = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    hole = torch.full((out_h * out_w, 3), HOLE_COLOR, dtype=torch.uint8, device=device)
    rendered = []
    with torch.no_grad():
        for ti in range(num_frames):
            w2c_R = c2w[ti, :3, :3].transpose(0, 1)
            w2c_t = -w2c_R @ c2w[ti, :3, 3]
            fpx = float(focal_px[ti])
            if use_mesh:
                cam = mesh_verts @ w2c_R.T + w2c_t
                # The clamped z only keeps u/v finite; the rasteriser is handed the true depth so
                # its own test can drop whatever ended up behind the camera.
                safe_z = cam[:, 2].clamp(min=1e-6)
                u = cam[:, 0] / safe_z * fpx + cxc_out
                v = cam[:, 1] / safe_z * fpx + cyc_out
                rendered.append(_fill_render_holes(_rasterize_mesh(
                    torch.stack([u, v], dim=-1), cam[:, 2], mesh_cols,
                    tris, out_w, out_h, HOLE_COLOR)))
                continue
            if normals is None:
                pts_ti, cols_ti = pts, cols
            else:  # --cull: drop every splat whose surface normal faces away from this camera
                facing = ((c2w[ti, :3, 3] - pts) * normals).sum(-1) > 0
                pts_ti, cols_ti = pts[facing], cols[facing]
            cam = pts_ti @ w2c_R.T + w2c_t
            front = cam[:, 2] > 1e-6
            cam_f, col_f = cam[front], cols_ti[front]
            z = cam_f[:, 2]
            fpx = float(focal_px[ti])
            u = cam_f[:, 0] / z * fpx + cxc_out
            v = cam_f[:, 1] / z * fpx + cyc_out
            x = torch.cat([(u + dx).round() for dx, _ in offs]).long()
            y = torch.cat([(v + dy).round() for _, dy in offs]).long()
            zz = z.repeat(len(offs))
            cc = col_f.repeat(len(offs), 1)
            inside = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
            idx, zk, ck = y[inside] * out_w + x[inside], zz[inside], cc[inside]
            zbuf = torch.full((out_h * out_w,), float("inf"), device=device)
            zbuf.scatter_reduce_(0, idx, zk, "amin", include_self=True)
            win = zk == zbuf[idx]
            img = hole.clone()
            img[idx[win]] = ck[win]
            # Hole filling is a mesh presentation step. Keep point splats raw: filling them would
            # hide the coverage differences from `point_size` and the holes intentionally left by
            # back-face culling, changing the point renderer's established semantics.
            rendered.append(img.view(out_h, out_w, 3))

    render = torch.stack(rendered).float().div_(255.0).cpu()
    source = F.interpolate(first.permute(0, 3, 1, 2), size=(out_h, out_w),
                           mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    source = source.repeat(num_frames, 1, 1, 1).clamp(0.0, 1.0).cpu()
    culled = ", back-face culled" if normals is not None else ""
    drawn = (f"{int(tris.shape[0])} triangles (mesh)" if use_mesh
             else f"{pts.shape[0]} points (point splat)")
    print(f"[Enndee] Meridian fast depth: {drawn} -> {out_w}x{out_h}, {num_frames} frames "
          f"(zm {zm:.3f} [raw {zm_raw:.3f}], cloud {cloud_w}x{cloud_h}, depth {low_w}x{low_h}, "
          f"{model_size}, {vfov_degrees:.1f} deg vFOV{culled})", flush=True)
    return source, render, out_w, out_h, num_frames


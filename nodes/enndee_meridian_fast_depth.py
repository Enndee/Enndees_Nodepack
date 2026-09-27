"""Meridian fast-depth backend (Enndee): the VGGT-free geometry engine behind Meridian Geometry.

Builds the Meridian / MiniMax-H3 reference pair in-process, with no external conda
environment and no VGGT checkpoint:

    starting frame --Depth-Anything-V2--> inverse depth --invert--> relative depth
    relative depth --3x3 edge keep + all-parents upsample--> kept depth
    depth + RGB --unproject--> 3D point cloud (the "depth-aligned texture")
    camera path (custom_camera signal, or the sample.py motion flags) --render--> flight frames

Depth-Anything-V2 predicts *inverse* depth (disparity-like: larger = closer - the near pier
post reads 4.4 while the far sky reads 0.3), so the engine inverts it into relative depth
before the percentile rescale. Feeding the raw output as z flips the scene front-to-back:
the subject lands behind the background and a moving camera sees the mirrored "back" of a
shell whose front side was never reconstructed.

Two render guards mirror Meridian's own pipeline (recam/geometry.py, inference/sample.py):

    edge keep       the 3x3 local depth-spread rule (EDGE_RTOL = 0.30) is applied on the
                    model's own depth grid; at the cloud grid a hi-res pixel survives only
                    where every parent survived (`upsample`'s >0.999 rule) - this is what
                    kills the bilinear "flying pixels" that smear silhouettes into depth.
    back-face cull  depth-map normals oriented toward the source camera (`--cull`): points
                    the target camera sees from behind are dropped, so a 180-degree view is
                    a hole, not the mirrored front.

This module is a library, not a node: Meridian Geometry (Enndee) drives it with
`mode = "Fast depth (Depth-Anything-V2)"`, and the unittests call `render_depth_aligned`
directly with a fake depth model. Depth-Anything-V2-Small runs at ~7 ms per frame in FP16
on a Blackwell GPU; a 73-frame pass typically completes in well under a second.
"""

import json
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS

# Meridian's 480-class aspect ladder (width, height), recam/h3.py LADDERS[480]
LADDER_480 = [
    (416, 960), (448, 896), (480, 832), (544, 736), (640, 640),
    (736, 544), (832, 480), (896, 448), (960, 416),
]
HOLE_COLOR = 128                      # uncovered pixels: mid grey, as in recam/geometry.py (no mask channel)
VFOV_DEGREES = 55.0                   # assumed vertical field of view of the source camera
DEPTH_NEAR, DEPTH_FAR = 1.0, 5.0      # metric window the relative depth is mapped onto (the gauge is zm-relative)
DEPTH_PCT_LO, DEPTH_PCT_HI = 0.01, 0.99   # percentile clip so one speck cannot squash the range
PIVOT_WINDOW = 0.05                   # Meridian's +-5 % window around a picked pivot point
MAX_CLOUD_PIXELS = 16_777_216         # unprojection-grid safety cap (16 M points)
DEPTH_RES = 518                       # Depth-Anything-V2's native square input side
DISPARITY_EPS = 0.001                 # floor before the 1/x inversion, so the far plane stays finite
KEEP_PARENT_RATIO = 0.999             # recam/geometry.py `upsample`: a hi-res pixel needs every parent kept

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


def _bucket_480(width: int, height: int) -> Tuple[int, int]:
    """Meridian 480-class ladder entry nearest in log-aspect (recam/h3.py bucket())."""
    return min(LADDER_480, key=lambda c: abs(math.log((c[0] / c[1]) * (height / width))))


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


def _look_at(pos: np.ndarray, look: np.ndarray, prev: Optional[np.ndarray] = None) -> np.ndarray:
    """c2w rotation (columns right, down, forward) looking from pos at look with zero roll (recam/path.py)."""
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    f = look - pos
    norm_f = float(np.linalg.norm(f))
    if norm_f < 1e-6:
        return prev if prev is not None else np.eye(3, dtype=np.float32)
    f = f / norm_f
    r = np.cross(f, up)
    norm_r = float(np.linalg.norm(r))
    if norm_r < 1e-6:              # looking straight up or down: keep x as right
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / norm_r
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=1)


def _evaluate_camera_path(path_data: dict, frames: int, zm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Custom path keys -> per-frame c2w (F,4,4, float32) and focal multipliers (F,).

    Keys are in frame-0 camera coordinates / pivot depth, exactly like recam/path.py's plan_path."""
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
    r_prev = None
    for i in range(frames):
        r_prev = _look_at(pos[i], look[i], r_prev)
        c2w[i, :3, :3] = r_prev
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


def render_depth_aligned(first, device, model_size="Depth-Anything-V2-Small-hf", frames=73,
                         canvas_mode="auto_meridian480", custom_width=832, custom_height=480,
                         cloud_scale=2, point_size=1, edge_cull=True, edge_threshold=0.30,
                         back_face_cull=False, camera=None, custom_camera=None):
    """Depth-aligned camera-flight condition renderer (Depth-Anything-V2 + GPU point-cloud renderer).

    `first`          [1,H,W,3] float tensor in [0,1]: the still the flight starts from.
    `camera`         settings dict from `parse_camera_settings` (None = every default).
    `custom_camera`  Meridian Camera Path Configurator signal; its path and frame count win.
    `back_face_cull` mirrors Meridian's `--cull`; a `--cull` token in `camera` also turns it on.

    Returns exactly what the VGGT geometry pass returns - (source, render, width, height, length) -
    so the pair drops straight into MeridianRefConditioning as `<Video 1>` / `<Video 2>`.
    """
    settings = dict(default_camera_settings())
    if camera:
        settings.update(camera)
    if not isinstance(first, torch.Tensor) or first.ndim != 4 or first.shape[0] < 1:
        raise ValueError("The image input must be a ComfyUI IMAGE batch [frames, height, width, channels].")
    if first.shape[-1] < 3:
        raise ValueError("The image batch must carry at least 3 channels per frame (RGB).")
    if first.shape[-1] > 4:
        raise ValueError("The image batch must have at most 4 channels per frame.")
    first = first[0:1].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    _, src_h, src_w, _ = first.shape

    # canvas: Meridian's 480-class condition ladder (the trained reference canvas), or explicit
    if canvas_mode == "custom":
        out_w, out_h = int(custom_width), int(custom_height)
    else:
        out_w, out_h = _bucket_480(src_w, src_h)

    # --- depth: Depth-Anything-V2 at its native 518 px input, inverted to relative depth ---------
    depth_model = _get_depth_model(model_size, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x518 = F.interpolate(first.permute(0, 3, 1, 2), size=(DEPTH_RES, DEPTH_RES),
                         mode="bilinear", align_corners=False)
    with torch.no_grad():
        pred = depth_model(pixel_values=((x518 - mean) / std).half()).predicted_depth[0].float()
    depth_low = _invert_disparity(pred)

    # keep (model grid): Meridian's 3x3 depth-edge rule; no confidence head exists here, so the
    # 2 % confidence pruning of recam/geometry.py has no equivalent and nothing else is culled
    keep_low = _edge_keep(depth_low, edge_threshold) if edge_cull else torch.ones_like(depth_low, dtype=torch.bool)

    # cloud grid: the frame (and the depth under it) upscaled so silhouettes get finer points;
    # `_upsample_depth_keep` re-checks the parents, so bilinear cannot bridge a depth jump
    scale = max(1, int(cloud_scale))
    while scale > 1 and (src_w * scale) * (src_h * scale) > MAX_CLOUD_PIXELS:
        scale -= 1
    cloud_w, cloud_h = src_w * scale, src_h * scale
    depth_cloud, keep_cloud = _upsample_depth_keep(depth_low, keep_low, cloud_h, cloud_w)

    pool = depth_cloud[keep_cloud] if keep_cloud.any() else depth_cloud.flatten()
    d_lo = torch.quantile(pool, DEPTH_PCT_LO)
    d_hi = torch.quantile(pool, DEPTH_PCT_HI)
    # The span floor is relative to the depth magnitude: an absolute floor turns float noise
    # into a fake bumpy surface when the depth is (nearly) constant, which would scramble the
    # `--cull` normals below.
    span = (d_hi - d_lo).clamp(min=1e-3 * float(d_hi.abs() + d_lo.abs()) + 1e-9)
    depth_map = (DEPTH_NEAR + (DEPTH_FAR - DEPTH_NEAR)
                 * (depth_cloud - d_lo) / span).clamp(min=0.05)

    # --- pivot: median depth in the +-5 % window of the picked point, on the model's own grid ----
    res = depth_low.shape[-1]
    f_cloud = 0.5 * cloud_h / math.tan(math.radians(VFOV_DEGREES) / 2.0)
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
        px, py = fx * res, fy * res
        rw, rh = PIVOT_WINDOW * res, PIVOT_WINDOW * res
        window = (slice(max(0, int(py - rh)), min(res, int(py + rh) + 1)),
                  slice(max(0, int(px - rw)), min(res, int(px + rw) + 1)))
        sample = depth_low[window][keep_low[window]]
        if not sample.numel():
            sample = depth_low[keep_low] if keep_low.any() else depth_low.flatten()
        return (fx, fy), float(sample.median())

    pivot_frac, pivot_depth = picked_point(settings.get("pivot"), "--pivot")
    if pivot_depth is not None:
        zm = pivot_depth
    else:
        zm = float(depth_low[keep_low].median()) if keep_low.any() else float(depth_low.median())
    piv = torch.tensor([0.0, 0.0, zm], device=device)
    if bool(settings.get("pivot_lock")) and pivot_frac is not None:  # orbit about the picked pixel
        fx, fy = pivot_frac
        piv = torch.tensor([(fx * cloud_w - cxc) / f_cloud * zm,
                            (fy * cloud_h - cyc) / f_cloud * zm, zm], device=device)
    pivot_to = piv
    aim_frac, aim_depth = picked_point(settings.get("pivot_to"), "--pivot-to")
    if aim_frac is not None and aim_depth is not None:   # --pivot-to: the aim target slides to it
        fx, fy = aim_frac
        pivot_to = torch.tensor([(fx * cloud_w - cxc) / f_cloud * aim_depth,
                                 (fy * cloud_h - cyc) / f_cloud * aim_depth, aim_depth], device=device)

    # --- cloud: frame + depth unprojected in frame-0 camera coordinates (the source c2w is I) ----
    yy, xx = torch.meshgrid(torch.arange(cloud_h, device=device, dtype=torch.float32),
                            torch.arange(cloud_w, device=device, dtype=torch.float32), indexing="ij")
    points = torch.stack([(xx - cxc) / f_cloud * depth_map,
                          (yy - cyc) / f_cloud * depth_map, depth_map], dim=-1)
    colors_hi = F.interpolate(first.permute(0, 3, 1, 2), size=(cloud_h, cloud_w),
                              mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    flat_keep = keep_cloud.reshape(-1)
    pts = points.reshape(-1, 3)[flat_keep]
    cols = (colors_hi.reshape(-1, 3)[flat_keep] * 255.0).round().to(torch.uint8)

    # --- cull: `--cull` drops the splats the target camera would see from behind -----------------
    cull = bool(settings.get("cull")) or bool(back_face_cull)
    normals = _orient_toward_source(points)[flat_keep] if cull else None

    # --- camera: the authored path wins; otherwise Meridian's parametric offsets -----------------
    if custom_camera is not None:
        path_data, num_frames = _parse_camera_signal(custom_camera)
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
    f_canvas = 0.5 * out_h / math.tan(math.radians(VFOV_DEGREES) / 2.0)
    focal_px = f_canvas * focal.to(device)      # [F]

    # --- render: z-buffered point-cloud flight along the camera path (recam/geometry.py render_hw)
    cxc_out, cyc_out = out_w / 2.0, out_h / 2.0
    radius = max(0, int(point_size))
    offs = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    hole = torch.full((out_h * out_w, 3), HOLE_COLOR, dtype=torch.uint8, device=device)
    rendered = []
    with torch.no_grad():
        for ti in range(num_frames):
            w2c_R = c2w[ti, :3, :3].transpose(0, 1)
            w2c_t = -w2c_R @ c2w[ti, :3, 3]
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
            rendered.append(img.view(out_h, out_w, 3))

    render = torch.stack(rendered).float().div_(255.0).cpu()
    source = F.interpolate(first.permute(0, 3, 1, 2), size=(out_h, out_w),
                           mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    source = source.repeat(num_frames, 1, 1, 1).clamp(0.0, 1.0).cpu()
    culled = ", back-face culled" if normals is not None else ""
    print(f"[Enndee] Meridian fast depth: {pts.shape[0]} points -> {out_w}x{out_h}, {num_frames} frames "
          f"(zm {zm:.3f}, cloud {cloud_w}x{cloud_h}, {model_size}{culled})", flush=True)
    return source, render, out_w, out_h, num_frames


"""Meridian Fast Depth Splat (Enndee): a VGGT-free geometry condition renderer for MiniMax-H3.

Builds the Meridian / MiniMax-H3 reference pair in-process, with no external conda
environment and no VGGT checkpoint:

    starting frame --Depth-Anything-V2--> relative depth --metric rescale--> depth map
    depth map + RGB --unproject--> 3D point cloud (the "depth-aligned texture")
    camera path (custom_camera signal, or yaw/truck/boom/dolly) --splat--> render frames

The outputs mirror ``Enndee_MeridianGeometry`` (source, render, width, height, length)
at the 480-class Meridian canvas, so this node is a drop-in, much faster stand-in for the
VGGT condition render whenever the source is a single still (image-to-video):

    source  first frame repeated and resized to the canvas (the clip Meridian would see)
    render  the depth-aligned texture re-captured along the authored camera path

Depth-Anything-V2-Small runs at ~7 ms per frame in FP16 on a Blackwell GPU; the whole
node typically completes in ~1-2 s, versus minutes for the VGGT subprocess.
"""

import json
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS, CAMERA_SIGNAL_TYPE

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

_GLOBAL_DEPTH_MODEL = None
_GLOBAL_MODEL_ID = None


def _get_depth_model(model_name: str, device: torch.device):
    """Cache Depth-Anything-V2 in FP16 on the GPU across runs; reload only if the variant changed."""
    global _GLOBAL_DEPTH_MODEL, _GLOBAL_MODEL_ID
    repo_id = model_name if "/" in model_name else f"depth-anything/{model_name}"
    if _GLOBAL_DEPTH_MODEL is None or _GLOBAL_MODEL_ID != repo_id:
        from transformers.models.depth_anything.modeling_depth_anything import DepthAnythingForDepthEstimation
        print(f"[Enndee] FastDepthSplat: loading {repo_id} (fp16)...", flush=True)
        _GLOBAL_DEPTH_MODEL = DepthAnythingForDepthEstimation.from_pretrained(repo_id).to(device).half().eval()
        _GLOBAL_MODEL_ID = repo_id
    return _GLOBAL_DEPTH_MODEL


def _bucket_480(width: int, height: int) -> Tuple[int, int]:
    """Meridian 480-class ladder entry nearest in log-aspect (recam/h3.py bucket())."""
    return min(LADDER_480, key=lambda c: abs(math.log((c[0] / c[1]) * (height / width))))


def _parse_camera_signal(signal: str) -> Tuple[dict, int]:
    """Parse the custom_camera JSON emitted by Meridian Camera Path Configurator (Enndee)."""
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
                          dolly: float, sweep: bool, ease: bool, aim: bool, device: torch.device
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Meridian's parametric camera offsets (inference/sample.py) for a static source frame.

    `piv` is the world-space pivot (frame-0 camera coordinates); the source camera sits at the
    origin with identity rotation, so the moved camera is exactly `delta`, as `c2w[t] = I @ delta`."""
    ramp = torch.linspace(0.0, 1.0, frames, device=device) if sweep else torch.ones(frames, device=device)
    if ease:
        ramp = (1.0 - torch.cos(math.pi * ramp)) / 2.0

    c2w = torch.eye(4, device=device, dtype=torch.float32).repeat(frames, 1, 1)
    focals = torch.ones(frames, device=device, dtype=torch.float32)
    for ti in range(frames):
        alpha = float(ramp[ti])
        th = math.radians(yaw * alpha)
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
            b = piv - delta[:3, 3]
            b = b / b.norm()
            v = torch.cross(a, b, dim=0)
            c = float(a @ b)
            sn = float(v.norm())
            if sn > 1e-8:
                K = torch.zeros(3, 3, device=device)
                K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -v[2], v[1], v[2], -v[0], -v[1], v[0]
                delta[:3, :3] = torch.eye(3, device=device) + K + K @ K * ((1.0 - c) / sn ** 2)
        c2w[ti] = delta
        focals[ti] = r  # Meridian: `f = r` when --zoom is not used
    return c2w, focals


class EnndeeMeridianPseudoRender:
    """Depth-aligned camera-flight condition renderer (Depth-Anything-V2 + GPU point splat).

    A VGGT-free, in-process replacement for the Enndee_MeridianGeometry preview pass that
    works from a single starting frame: predict its relative depth, unproject frame + depth
    into a 3D point cloud, then re-render that cloud along the authored camera path.

    Returns exactly what the geometry node returns - (source, render, width, height, length) -
    so the pair drops straight into MeridianRefConditioning as `<Video 1>` / `<Video 2>`.

    Connect custom_camera from Meridian Camera Path Configurator (Enndee) for an authored
    flight, or leave it unconnected and drive the built-in yaw/truck/boom/dolly offsets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Starting frame to unproject (image-to-video); the first frame of a batch is used."}),
                "model_size": (["Depth-Anything-V2-Small-hf", "Depth-Anything-V2-Base-hf", "Depth-Anything-V2-Large-hf"],
                               {"default": "Depth-Anything-V2-Small-hf",
                                "tooltip": "Depth-Anything-V2 variant. Small is ~7 ms per frame and plenty for a splat cloud; Base/Large are finer but slower and need a download."}),
                "frames": (CAMERA_FRAME_OPTIONS, {"default": "73",
                                                  "tooltip": "Camera-flight length (MiniMax-H3 17k+5 grid). Ignored (the path's own count wins) when custom_camera is connected."}),
                "canvas_mode": (["auto_meridian480", "custom"], {"default": "auto_meridian480",
                                                                 "tooltip": "Render canvas: 'auto_meridian480' picks the Meridian 480-class ladder entry nearest the frame's aspect; 'custom' uses the two fields below."}),
                "custom_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 32,
                                         "tooltip": "'custom' canvas width."}),
                "custom_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 32,
                                          "tooltip": "'custom' canvas height."}),
                "cloud_scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1,
                                        "tooltip": "Unprojection-grid upscale over the input frame: 2 doubles the point count (denser silhouette fill), 1 keeps the frame's own resolution."}),
                "splat_size": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1,
                                       "tooltip": "Point footprint: 0=1x1, 1=3x3, 2=5x5, 3=7x7. Larger fills holes where the cloud is sparse after a big camera move."}),
                "edge_cull": ("BOOLEAN", {"default": True,
                                          "tooltip": "Drop points on steep depth edges so silhouette borders cannot smear into flying spikes."}),
                "edge_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 2.0, "step": 0.01,
                                             "tooltip": "Cull points whose 3x3 relative depth spread exceeds this ratio (Meridian's EDGE_RTOL = 0.30)."}),
            },
            "optional": {
                "custom_camera": (CAMERA_SIGNAL_TYPE, {"forceInput": True,
                                                       "tooltip": "Connect custom_camera from Meridian Camera Path Configurator (Enndee). Its keyframes and frame count replace the parametric offsets below."}),
                "yaw": ("FLOAT", {"default": 10.0, "min": -180.0, "max": 180.0, "step": 0.5,
                                  "tooltip": "Orbit about the pivot, degrees (Meridian --yaw). Ignored when custom_camera is connected."}),
                "truck": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                                    "tooltip": "Sideways drift in pivot units (Meridian --truck)."}),
                "boom": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                                   "tooltip": "Vertical lift in pivot units (Meridian --boom)."}),
                "dolly": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05,
                                    "tooltip": "Distance to the pivot multiplier (Meridian --dolly): 1 = start, 0.5 = half-way in."}),
                "sweep": ("BOOLEAN", {"default": True,
                                      "tooltip": "Ramp the offsets from identity at frame 0 (Meridian --sweep); off, the full offset is applied at every frame."}),
                "ease": ("BOOLEAN", {"default": True,
                                     "tooltip": "Cosine ease of the ramp (Meridian --ease)."}),
                "aim": ("BOOLEAN", {"default": False,
                                    "tooltip": "Re-point the camera at the pivot after truck/boom so the subject stays centered (Meridian --aim)."}),
                "pivot": ("STRING", {"default": "",
                                     "tooltip": "Pivot sample point 'x,y' in 0..1 frame coordinates (Meridian --pivot; empty = median depth of the whole frame)."}),
                "pivot_lock": ("BOOLEAN", {"default": False,
                                           "tooltip": "Orbit about the unprojected pivot pixel instead of the line-of-sight point (Meridian --pivot-lock)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("source", "render", "width", "height", "length")
    FUNCTION = "generate"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Fast VGGT-free geometry preview for MiniMax-H3: Depth-Anything-V2 + GPU point splat render "
        "the authored camera flight from one starting frame. Returns (source, render, width, height, length) "
        "at the Meridian 480-class canvas, ready for MeridianRefConditioning."
    )

    def generate(self, image: torch.Tensor, model_size: str, frames: str, canvas_mode: str,
                 custom_width: int, custom_height: int, cloud_scale: int, splat_size: int,
                 edge_cull: bool, edge_threshold: float, custom_camera: Optional[str] = None,
                 yaw: float = 10.0, truck: float = 0.0, boom: float = 0.0, dolly: float = 1.0,
                 sweep: bool = True, ease: bool = True, aim: bool = False,
                 pivot: str = "", pivot_lock: bool = False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError("The image input must be a ComfyUI IMAGE batch [frames, height, width, channels].")
        first = image[0:1].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        _, src_h, src_w, _ = first.shape

        # canvas: Meridian's 480-class condition ladder (the trained reference canvas), or explicit
        if canvas_mode == "custom":
            out_w, out_h = int(custom_width), int(custom_height)
        else:
            out_w, out_h = _bucket_480(src_w, src_h)

        # --- depth: Depth-Anything-V2 at its native 518 px input ------------------------------------
        depth_model = _get_depth_model(model_size, device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x518 = F.interpolate(first.permute(0, 3, 1, 2), size=(518, 518), mode="bilinear", align_corners=False)
        with torch.no_grad():
            pred = depth_model(pixel_values=((x518 - mean) / std).half()).predicted_depth[0].float()

        # cloud grid: the frame (and the depth under it) upscaled so silhouettes get finer points
        scale = max(1, int(cloud_scale))
        while scale > 1 and (src_w * scale) * (src_h * scale) > MAX_CLOUD_PIXELS:
            scale -= 1
        cloud_w, cloud_h = src_w * scale, src_h * scale

        depth_cloud = F.interpolate(pred[None, None], size=(cloud_h, cloud_w),
                                    mode="bilinear", align_corners=False)[0, 0]
        d_lo = torch.quantile(depth_cloud.flatten(), DEPTH_PCT_LO)
        d_hi = torch.quantile(depth_cloud.flatten(), DEPTH_PCT_HI)
        depth_map = (DEPTH_NEAR + (DEPTH_FAR - DEPTH_NEAR)
                     * (depth_cloud - d_lo) / (d_hi - d_lo).clamp(min=1e-6)).clamp(min=0.05)

        # keep: Meridian's 3x3 depth-edge rule (no confidence head exists here, so nothing else is culled)
        keep = torch.ones((cloud_h, cloud_w), dtype=torch.bool, device=device)
        if edge_cull:
            mx = F.max_pool2d(depth_map[None, None], 3, 1, 1)[0, 0]
            mn = -F.max_pool2d(-depth_map[None, None], 3, 1, 1)[0, 0]
            keep = ((mx - mn) / depth_map.clamp(min=1e-6)) <= float(edge_threshold)

        # --- pivot: median depth in the +-5 % window, plus the pivot-lock unprojection --------------
        f_cloud = 0.5 * cloud_h / math.tan(math.radians(VFOV_DEGREES) / 2.0)
        cxc, cyc = cloud_w / 2.0, cloud_h / 2.0
        px = py = None
        if pivot and pivot.strip():
            try:
                fx, fy = (float(value) for value in pivot.replace(" ", "").split(","))
            except ValueError as exc:
                raise ValueError("pivot must be 'x,y' in 0..1 frame coordinates, or empty for the frame median.") from exc
            px, py = min(max(fx, 0.0), 1.0) * cloud_w, min(max(fy, 0.0), 1.0) * cloud_h
            rw, rh = PIVOT_WINDOW * cloud_w, PIVOT_WINDOW * cloud_h
            win = (slice(max(0, int(py - rh)), min(cloud_h, int(py + rh) + 1)),
                   slice(max(0, int(px - rw)), min(cloud_w, int(px + rw) + 1)))
            sample = depth_map[win][keep[win]]
            zm = float(sample.median()) if sample.numel() else float(depth_map[keep].median())
        else:
            zm = float(depth_map[keep].median())
        piv = torch.tensor([0.0, 0.0, zm], device=device)
        if pivot_lock and px is not None:   # orbit about the picked pixel so it holds its screen position
            piv = torch.tensor([(px - cxc) / f_cloud * zm, (py - cyc) / f_cloud * zm, zm], device=device)

        # --- cloud: frame + depth unprojected in frame-0 camera coordinates (the source c2w is I) ----
        yy, xx = torch.meshgrid(torch.arange(cloud_h, device=device, dtype=torch.float32),
                                torch.arange(cloud_w, device=device, dtype=torch.float32), indexing="ij")
        pts = torch.stack([(xx - cxc) / f_cloud * depth_map,
                           (yy - cyc) / f_cloud * depth_map, depth_map], dim=-1).reshape(-1, 3)
        colors_hi = F.interpolate(first.permute(0, 3, 1, 2), size=(cloud_h, cloud_w),
                                  mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        flat_keep = keep.reshape(-1)
        pts = pts[flat_keep]
        cols = (colors_hi.reshape(-1, 3)[flat_keep] * 255.0).round().to(torch.uint8)

        # --- camera: the authored path wins; otherwise Meridian's parametric offsets -----------------
        if custom_camera is not None:
            path_data, num_frames = _parse_camera_signal(custom_camera)
            c2w_np, focal_np = _evaluate_camera_path(path_data, num_frames, zm)
            c2w = torch.from_numpy(c2w_np).to(device)
            focal = torch.from_numpy(focal_np).to(device)
        else:
            num_frames = int(frames)
            c2w, focal = _build_parametric_c2w(num_frames, piv, zm, float(yaw), float(truck), float(boom),
                                               float(dolly), bool(sweep), bool(ease), bool(aim), device)

        # target intrinsics at the canvas: same vertical FOV, focal multipliers from the path applied
        # to both axes exactly as `intr_t[ti, 0, 0] *= f; intr_t[ti, 1, 1] *= f` in sample.py
        f_canvas = 0.5 * out_h / math.tan(math.radians(VFOV_DEGREES) / 2.0)
        focal_px = f_canvas * focal.to(device)      # [F]

        # --- render: z-buffered point splat along the flight (recam/geometry.py render_hw) -----------
        cxc_out, cyc_out = out_w / 2.0, out_h / 2.0
        splat = max(0, int(splat_size))
        offs = [(dx, dy) for dy in range(-splat, splat + 1) for dx in range(-splat, splat + 1)]
        hole = torch.full((out_h * out_w, 3), HOLE_COLOR, dtype=torch.uint8, device=device)
        rendered = []
        with torch.no_grad():
            for ti in range(num_frames):
                w2c_R = c2w[ti, :3, :3].transpose(0, 1)
                w2c_t = -w2c_R @ c2w[ti, :3, 3]
                cam = pts @ w2c_R.T + w2c_t
                front = cam[:, 2] > 1e-6
                cam_f, col_f = cam[front], cols[front]
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
        print(f"[Enndee] FastDepthSplat: {pts.shape[0]} points -> {out_w}x{out_h}, {num_frames} frames "
              f"(zm {zm:.3f}, cloud {cloud_w}x{cloud_h}, {model_size})", flush=True)
        return source, render, out_w, out_h, num_frames


NODE_CLASS_MAPPINGS = {"Enndee_MeridianPseudoRender": EnndeeMeridianPseudoRender}
NODE_DISPLAY_NAME_MAPPINGS = {"Enndee_MeridianPseudoRender": "Meridian Fast Depth Splat (Enndee)"}







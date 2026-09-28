"""Meridian Geometry (Enndee): one node over both geometry-condition backends.

    mode = "VGGT preview (subprocess)"          shells out to Meridian's inference/sample.py
                                                (VGGT-Omega reconstruction, --preview-only)
    mode = "Fast depth (Depth-Anything-V2)"     runs the in-process Depth-Anything point-cloud
                                                flight (enndee_meridian_fast_depth.py), V2 or V3

Both backends consume the same widgets - `image`/`video`, the `args` string (or the Meridian
Parameter Picker's `args_override`) and the optional `custom_camera` signal - and both return
(source, render, width, height, length) at the Meridian condition canvas, so the mode is a
drop-in swap. The fast-depth widgets (model size, the Depth-Anything-3 resolution cap, canvas
override, cloud/point density and the two cull rules) only apply to the fast mode and are shown
for it alone; the VGGT settings
(repo, python, cache, cache_dir plus the canvas, source-size and VGGT-path overrides the
picker also knows) hide while the fast mode is active. The node's own VGGT overrides only fill
flags the args string did not set, so a connected picker always wins.
web/js/enndee_meridian_geometry.js drives that visibility, the same way the parameter picker's
extension does.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import av
import numpy as np
import torch

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS, CAMERA_SIGNAL_TYPE
from enndee_meridian_fast_depth import DA3_RES, parse_camera_settings, render_depth_aligned

VGGT_MODE = "VGGT preview (subprocess)"
FAST_DEPTH_MODE = "Fast depth (Depth-Anything-V2)"
MODE_OPTIONS = (VGGT_MODE, FAST_DEPTH_MODE)



def _add_default_vggt_paths(args, repo):
    """Append the adjacent VGGT code/checkpoint when either path was omitted."""
    has_repo = any(token == "--vggt-repo" or token.startswith("--vggt-repo=") for token in args)
    has_checkpoint = any(token == "--vggt" or token.startswith("--vggt=") for token in args)
    if has_repo and has_checkpoint:
        return args

    candidate_roots = [os.path.dirname(os.path.abspath(repo))]
    configured_repo = next(
        (os.path.abspath(token.split("=", 1)[1]) for token in args if token.startswith("--vggt-repo=")),
        None,
    )
    if configured_repo:
        candidate_roots.insert(0, os.path.dirname(configured_repo))
    if "--vggt-repo" in args:
        index = args.index("--vggt-repo")
        if index + 1 < len(args):
            candidate_roots.insert(0, os.path.dirname(os.path.abspath(args[index + 1])))
    for root in candidate_roots:
        vggt_repo = os.path.join(root, "vggt-omega-fp16-version")
        checkpoint = os.path.join(root, "vggt-omega", "checkpoints", "vggt_omega_1b_512.pt")
        if os.path.isfile(os.path.join(vggt_repo, "vggt_omega", "models", "vggt_omega.py")):
            if not has_repo:
                args.extend(["--vggt-repo", vggt_repo])
            if not has_checkpoint:
                args.extend(["--vggt", checkpoint])
            break
    return args


def _apply_vggt_source_settings(cmd, canvas_enabled, canvas_width, canvas_height,
                                full_enabled, full_size, vggt_repo, vggt_checkpoint):
    """Append the node's VGGT-panel options, but never over an args string that set them.

    The Meridian Parameter Picker owns the same flags (`--canvas`, `--full`, `--vggt-repo`,
    `--vggt`); its args win whenever they are present, so these widgets only matter for
    picker-free graphs - the rule `_add_default_vggt_paths` already uses for the detected
    installation. Sizes are validated exactly like the picker validates them.
    """
    present = {token.split("=", 1)[0] for token in cmd}

    def add(flag, value):
        if flag not in present:
            cmd.extend([flag, value])
            present.add(flag)

    if canvas_enabled:
        width, height = int(canvas_width), int(canvas_height)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError("Geometry canvas width and height must be positive multiples of 32.")
        add("--canvas", f"{width}x{height}")
    if full_enabled:
        size = int(full_size)
        if size < 128 or size % 32:
            raise ValueError("VGGT source square size must be at least 128 and a multiple of 32.")
        add("--full", str(size))
    if str(vggt_repo).strip():
        add("--vggt-repo", str(vggt_repo).strip())
    if str(vggt_checkpoint).strip():
        add("--vggt", str(vggt_checkpoint).strip())
    return cmd


def _frames(path):
    with av.open(path) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not decoded:
        raise ValueError(f"Meridian produced an empty video: {path}")
    return torch.from_numpy(np.stack(decoded)).float().div_(255.0)


def _tensor_rgb8(frame):
    if isinstance(frame, torch.Tensor):
        pixels = frame.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    else:
        pixels = np.asarray(frame)
        if pixels.dtype != np.uint8:
            pixels = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] < 3:
        raise ValueError("A Meridian video frame must have shape (height, width, RGB[A]).")
    return np.ascontiguousarray(pixels[..., :3], dtype=np.uint8)


def _write_rgb_frames_video(rgb_frames, path, fps=24):
    """Write RGB uint8 frames losslessly with the portable PyAV libx264rgb encoder."""
    frame_iterator = iter(rgb_frames)
    try:
        first = _tensor_rgb8(next(frame_iterator))
    except StopIteration as exc:
        raise ValueError("Cannot write a Meridian video without frames.") from exc

    height, width = first.shape[:2]
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "fast"}

        video_frame = av.VideoFrame.from_ndarray(first, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
        for pixels in frame_iterator:
            rgb = _tensor_rgb8(pixels)
            if rgb.shape != first.shape:
                raise ValueError("All Meridian video frames must have identical dimensions.")
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_image_batch_video(frames, path, fps=24):
    """Encode a BHWC ComfyUI image batch as a lossless RGB video."""
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] < 1:
        raise ValueError("Meridian image batches must have shape (frames, height, width, channels).")
    _write_rgb_frames_video((frame for frame in frames), path, fps=fps)


def _write_repeated_frame_video(first_frame, frame_count, path, fps=24):
    """Create exactly frame_count copies of one input image for a custom camera path."""
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("A custom camera path must contain at least one frame.")
    rgb = _tensor_rgb8(first_frame)
    _write_rgb_frames_video((rgb for _ in range(frame_count)), path, fps=fps)


def _read_first_video_frame(video_path):
    """Decode only the first RGB frame from an existing video path."""
    with av.open(video_path) as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise ValueError(f"Could not decode a first frame from the video input: {video_path}")
    pixels = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(pixels.copy()).float().div_(255.0)


_CACHE_VERSION = "v1"


def _cache_root(override=""):
    return override.strip() or os.path.join(tempfile.gettempdir(), "enndee_meridian_geometry")


def _hash_frames(image):
    """Cheap content fingerprint: first/middle/last frame as uint8, plus shape."""
    sample = (torch.stack([image[0], image[len(image) // 2], image[-1]])
              .detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu())
    digest = hashlib.sha256()
    digest.update(str(tuple(image.shape)).encode())
    digest.update(sample.numpy().tobytes())
    return digest.hexdigest()


def _cache_key_for(cache, image, video, effective_args, camera_data, frame_count):
    """Stable key for a geometry pass: source content, options, camera path, frame count."""
    if not cache:
        return None
    tokens = _parse_cli_tokens(effective_args)
    if camera_data is not None:
        tokens = tokens + ["--camera-path",
                           json.dumps(camera_data, sort_keys=True, ensure_ascii=False)]
    digest = hashlib.sha256()
    digest.update(_CACHE_VERSION.encode())
    digest.update(str(frame_count).encode())
    digest.update("\x00".join(tokens).encode("utf-8", "replace"))
    if image is not None:
        digest.update(_hash_frames(image).encode())
    elif video and os.path.isfile(video):
        stat = os.stat(video)
        digest.update(f"{os.path.abspath(video)}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    else:
        return None
    return digest.hexdigest()[:24]


def _cache_load(directory):
    """Return (source, render, width, height, length) for a complete cache entry, else None."""
    meta_path = os.path.join(directory, "meta.json")
    source_path = os.path.join(directory, "cond_source.mp4")
    render_path = os.path.join(directory, "cond_render.mp4")
    if not (os.path.isfile(meta_path) and os.path.isfile(source_path) and os.path.isfile(render_path)):
        return None
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    source = _frames(source_path)
    render = _frames(render_path)
    return source, render, int(meta["width"]), int(meta["height"]), int(meta["length"])


def _cache_store(directory, source_path, render_path, width, height, length):
    os.makedirs(directory, exist_ok=True)
    shutil.copy2(source_path, os.path.join(directory, "cond_source.mp4"))
    shutil.copy2(render_path, os.path.join(directory, "cond_render.mp4"))
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"width": int(width), "height": int(height), "length": int(length)}, handle)


_CAMERA_OPTION_WITH_VALUE = {
    "--camera-path",
    "--frames",
    "--start",
    "--freeze",
    "--yaw",
    "--yaw-from",
    "--truck",
    "--boom",
    "--dolly",
    "--zoom",
    "--pivot",
    "--pivot-to",
    "--live-speed",
    "--fast-back",
}
_CAMERA_OPTION_FLAGS = {
    "--follow",
    "--sweep",
    "--bounce",
    "--swing",
    "--ease",
    "--aim",
    "--pivot-lock",
}


def _args_for_custom_camera(args_text, camera_path_file, frames):
    """Keep unrelated Meridian settings, removing motion args overridden by the path signal."""
    tokens = _parse_cli_tokens(args_text)
    result = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        option = token.split("=", 1)[0]
        if option in _CAMERA_OPTION_WITH_VALUE:
            if "=" not in token:
                skip_next = True
            continue
        if option in _CAMERA_OPTION_FLAGS:
            continue
        result.append(token)
    result.extend(["--start", "0", "--frames", str(frames), "--camera-path", camera_path_file])
    return result


def _parse_cli_tokens(args_text):
    """Split CLI text while preserving quoted paths containing Windows separators."""
    portable_text = (args_text or "").replace("\\", "/")
    return shlex.split(portable_text)


def _parse_custom_camera(signal):
    try:
        data = json.loads(signal)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("custom_camera must be a valid signal from Meridian Camera Path Configurator (Enndee).") from exc
    if not isinstance(data, dict) or not isinstance(data.get("path"), list):
        raise ValueError("custom_camera is missing its camera-path keyframes.")
    frames = int(data.get("frames", 0))
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(f"Custom camera path has unsupported frame count {frames}.")
    path = data["path"]
    if len(path) < 2 or not all(isinstance(key, dict) for key in path):
        raise ValueError("Camera path must contain at least two valid keyframe objects.")
    if path[0].get("t") != 0 or path[-1].get("t") != frames - 1:
        raise ValueError("Camera-path keys must start at frame 0 and end at the configured frame count minus one.")
    try:
        times = [int(key["t"]) for key in path]
        source_indices = [int(key["src"]) for key in path]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every camera-path key must have integer t and src frame indices.") from exc
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Camera-path key times must be strictly increasing.")
    if source_indices != times:
        raise ValueError("Custom camera paths must use real-time source indexing: every key must have src equal to t.")
    for key in path:
        for field in ("pos", "look"):
            vector = key.get(field)
            if not isinstance(vector, list) or len(vector) != 3:
                raise ValueError(f"Camera-path key {field} must be a 3D vector.")
            try:
                finite = all(np.isfinite(float(value)) for value in vector)
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"Camera-path key {field} values must be finite.")
    return data, frames


class EnndeeMeridianGeometry:
    """Run Meridian geometry preview from a video/still and optional generated camera path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("STRING", {"default": "clip.mp4", "tooltip": "Source video path. Ignored when a ComfyUI image/batch is connected; with a custom camera path and no image input, the first frame of this video is repeated."}),
                "args": ("STRING", {"default": "--boom 0.35 --pivot 0.5,0.55 --ease --sweep", "multiline": True, "tooltip": "Additional Meridian options. When custom_camera is connected, its path and frame count take precedence over camera-motion, freeze, follow, start, and frame-count flags."}),
                "repo": ("STRING", {"default": "/path/to/release_recam", "tooltip": "Meridian checkout containing inference/sample.py."}),
                "python": ("STRING", {"default": "python", "tooltip": "Python interpreter with Meridian/VGGT installed."}),
                "cache": ("BOOLEAN", {"default": True, "tooltip": "Reuse a previous geometry pass when the picture, camera path and options are unchanged - skips the whole VGGT subprocess, the repeated-frame encode and the render."}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Cache folder; empty = %TEMP%\\enndee_meridian_geometry. Delete it any time to force fresh renders."}),
                "mode": (list(MODE_OPTIONS), {"default": VGGT_MODE,
                                              "tooltip": "Geometry backend. 'VGGT preview' runs Meridian's own subprocess reconstruction (repo/python/cache plus the VGGT canvas/source settings). 'Fast depth' runs the in-process Depth-Anything point-cloud flight for a single still - pick V2 or V3 in `model_size`, no VGGT, no subprocess."}),
                "model_size": (["Depth-Anything-V2-Small-hf", "Depth-Anything-V2-Base-hf", "Depth-Anything-V2-Large-hf",
                                "Depth-Anything-3-Small", "Depth-Anything-3-Base", "Depth-Anything-3-Large",
                                "Depth-Anything-3-Mono-Large"],
                               {"default": "Depth-Anything-V2-Small-hf",
                                "tooltip": "Fast depth only: the depth model. The V2 trio predicts inverted disparity at ~7 ms per frame (needs `transformers`); the V3 series predicts depth directly and is markedly more accurate (needs `python -m pip install --no-deps depth-anything-3` in the ComfyUI python_embeded) - Mono-Large is tuned for single stills, Small is the fast one. Downloads land in the Hugging Face cache; all variants are Apache-2.0."}),
                "canvas_mode": (["auto_meridian480", "custom"], {"default": "auto_meridian480",
                                                                 "tooltip": "Fast depth only: 'auto_meridian480' picks the Meridian 480-class ladder entry nearest the frame's aspect (the trained condition canvas); 'custom' uses the two fields below."}),
                "custom_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 32,
                                         "tooltip": "Fast depth only: 'custom' canvas width."}),
                "custom_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 32,
                                          "tooltip": "Fast depth only: 'custom' canvas height."}),
                "cloud_scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1,
                                        "tooltip": "Fast depth only: unprojection-grid upscale over the input frame: 2 doubles the point count (denser silhouette fill), 1 keeps the frame's own resolution."}),
                "point_size": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1,
                                       "tooltip": "Fast depth only: point footprint 0=1x1, 1=3x3, 2=5x5, 3=7x7. Larger fills holes where the cloud is sparse after a big camera move."}),
                "edge_cull": ("BOOLEAN", {"default": True,
                                          "tooltip": "Fast depth only: drop points on steep depth edges (Meridian's 3x3 EDGE_RTOL rule, applied on the model's own depth grid) so silhouette borders cannot smear into flying spikes."}),
                "edge_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 2.0, "step": 0.01,
                                             "tooltip": "Fast depth only: cull points whose 3x3 relative depth spread exceeds this ratio (Meridian's EDGE_RTOL = 0.30)."}),
                "back_face_cull": ("BOOLEAN", {"default": False,
                                               "tooltip": "Fast depth only: mirror Meridian's --cull - drop the splats the target camera sees from behind, so a 180-degree view is a hole, not the mirrored front. The parameter picker's Cull option turns this on through the args string."}),
                "canvas_enabled": ("BOOLEAN", {"default": False,
                                               "tooltip": "VGGT preview only: override Meridian's automatic 768-class geometry canvas with the two sizes below. Ignored whenever the args string already carries a --canvas flag (a connected parameter picker owns one)."}),
                "canvas_width": ("INT", {"default": 864, "min": 32, "max": 4096, "step": 32,
                                         "tooltip": "VGGT preview only: custom geometry render width; used while Custom Canvas is enabled. Multiples of 32."}),
                "canvas_height": ("INT", {"default": 1184, "min": 32, "max": 4096, "step": 32,
                                          "tooltip": "VGGT preview only: custom geometry render height; used while Custom Canvas is enabled. Multiples of 32."}),
                "full_enabled": ("BOOLEAN", {"default": False,
                                             "tooltip": "VGGT preview only: override VGGT's 1280-pixel square source reconstruction size with the value below. Ignored whenever the args string already carries a --full flag."}),
                "full_size": ("INT", {"default": 1280, "min": 128, "max": 4096, "step": 32,
                                      "tooltip": "VGGT preview only: VGGT square source reconstruction side, in pixels; used while Custom VGGT Source Size is enabled."}),
                "vggt_repo": ("STRING", {"default": "", "tooltip": "VGGT preview only: VGGT-Omega source folder. Empty = the installation auto-detected next to `repo`, or the picker's --vggt-repo when its args string carries one."}),
                "vggt_checkpoint": ("STRING", {"default": "", "tooltip": "VGGT preview only: VGGT-Omega checkpoint (.pt). Empty = the checkpoint auto-detected next to `repo`, or the picker's --vggt when its args string carries one."}),
                "depth_res": ("INT", {"default": DA3_RES, "min": 0, "max": 4096, "step": 1,
                                      "tooltip": "Fast depth only, Depth-Anything-3 models only: the depth model's longest-side cap in pixels (aspect preserved, then rounded to multiples of 14). 0 = run the still at its own resolution for maximum depth detail - the attention cost grows with the square of the pixel count, so a full-resolution still takes seconds instead of tenths. Values above the still's own side make DA3 upscale. The V2 models ignore it and keep their native 518 square."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "args_override": ("STRING", {"forceInput": True, "tooltip": "Optional Meridian Parameter Picker argument override."}),
                "custom_camera": (CAMERA_SIGNAL_TYPE, {"forceInput": True, "tooltip": "Connect custom_camera from Meridian Camera Path Configurator (Enndee). Its frame count controls an automatic repeated-first-frame source clip."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("source", "render", "width", "height", "length")
    FUNCTION = "build"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Meridian geometry condition pass with two interchangeable backends, selected by `mode`. "
        "VGGT preview runs Meridian's VGGT-Omega subprocess (repo/python/cache and the canvas, "
        "source-size and VGGT-path overrides live on this node as well); Fast depth runs the "
        "in-process Depth-Anything point-cloud flight for a single still (V2 or V3 via "
        "`model_size`, no VGGT, no external environment; `depth_res` trades time for depth "
        "detail, 0 = the still's own resolution). Both accept the same args string and "
        "custom_camera signal - the Meridian Parameter Picker configures either - and both return "
        "(source, render, width, height, length) at the 480-class condition canvas."
    )

    def _build_fast_depth(self, video, effective_args, image, custom_camera, model_size, canvas_mode,
                          custom_width, custom_height, cloud_scale, point_size, edge_cull,
                          edge_threshold, back_face_cull, depth_res=DA3_RES):
        """In-process Depth-Anything pass (V2 or V3 by `model_size`) with the same contract as the VGGT preview.

        The `args` string is parsed for the sample.py camera flags the fast backend honours
        (`parse_camera_settings`), so one Meridian Parameter Picker output configures either
        mode. `--follow` cannot be replayed without VGGT poses and is reported instead of
        silently ignored; `--freeze`, `--start` and `--canvas` do not affect a single still.
        """
        settings = parse_camera_settings(_parse_cli_tokens(effective_args))
        if image is not None:
            if image.ndim != 4 or image.shape[0] < 1:
                raise ValueError("The connected image input must contain at least one frame.")
            if image.shape[0] > 1:
                print("Meridian geometry (Enndee): fast depth uses the first frame of the image batch "
                      "(single-still image-to-video mode).", flush=True)
            first_frame = image[0:1]
        else:
            if not video or not os.path.isfile(video):
                raise ValueError("Connect a still/image batch or provide an existing video path for the fast depth mode.")
            first_frame = _read_first_video_frame(video).unsqueeze(0)
        if custom_camera is not None:
            _camera_data, frame_count = _parse_custom_camera(custom_camera)
        else:
            if settings["follow"]:
                raise ValueError("Fast depth mode cannot replay a source video's own camera path (--follow); "
                                 "use the VGGT preview mode or turn Follow off in the parameter picker.")
            frame_count = 73 if settings["frames"] is None else int(settings["frames"])
            if frame_count not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
                raise ValueError("Fast depth mode needs a Meridian output length "
                                 f"({', '.join(CAMERA_FRAME_OPTIONS)}), got {frame_count}.")
        source, render, width, height, length = render_depth_aligned(
            first_frame, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_size=model_size, frames=frame_count, canvas_mode=canvas_mode,
            custom_width=custom_width, custom_height=custom_height, cloud_scale=cloud_scale,
            point_size=point_size, edge_cull=edge_cull, edge_threshold=edge_threshold,
            back_face_cull=back_face_cull, camera=settings, custom_camera=custom_camera,
            depth_res=depth_res,
        )
        print(f"Meridian geometry (Enndee): fast depth -> {width}x{height}, {length} frames.", flush=True)
        return source, render, width, height, length

    def build(self, video, args, repo, python, cache=True, cache_dir="", image=None, args_override=None,
              custom_camera=None, mode=VGGT_MODE, model_size="Depth-Anything-V2-Small-hf",
              canvas_mode="auto_meridian480", custom_width=832, custom_height=480, cloud_scale=2,
              point_size=1, edge_cull=True, edge_threshold=0.30, back_face_cull=False,
              canvas_enabled=False, canvas_width=864, canvas_height=1184, full_enabled=False,
              full_size=1280, vggt_repo="", vggt_checkpoint="", depth_res=DA3_RES):
        effective_args = args_override if args_override is not None else args
        if mode == FAST_DEPTH_MODE:
            return self._build_fast_depth(video, effective_args, image, custom_camera, model_size,
                                          canvas_mode, custom_width, custom_height, cloud_scale,
                                          point_size, edge_cull, edge_threshold, back_face_cull,
                                          depth_res)
        out = tempfile.mkdtemp(prefix="enndee_meridian_")
        image_input_path = None
        try:
            camera_data, frame_count = None, None
            if custom_camera is not None:
                camera_data, frame_count = _parse_custom_camera(custom_camera)
            else:
                tokens = _parse_cli_tokens(effective_args)
                if "--frames" in tokens and tokens.index("--frames") + 1 < len(tokens):
                    frame_count = int(tokens[tokens.index("--frames") + 1])

            entry = None
            cache_key = _cache_key_for(cache, image, video, effective_args, camera_data, frame_count)
            if cache_key:
                entry = os.path.join(_cache_root(cache_dir), cache_key)
                hit = _cache_load(entry)
                if hit is not None:
                    print(f"Meridian geometry (Enndee): cache hit -> {entry}", flush=True)
                    return hit
                print(f"Meridian geometry (Enndee): cache miss -> {entry}", flush=True)

            if custom_camera is not None:
                path_file = os.path.join(out, "custom_camera_path.json")
                with open(path_file, "w", encoding="utf-8") as handle:
                    json.dump(camera_data, handle, ensure_ascii=False)

                if image is not None:
                    if image.ndim != 4 or image.shape[0] < 1:
                        raise ValueError("The connected image input must contain at least one frame.")
                    first_frame = image[0]
                else:
                    if not video or not os.path.isfile(video):
                        raise ValueError("Connect a still/image batch or provide an existing video path when custom_camera is connected.")
                    first_frame = _read_first_video_frame(video)

                image_input_path = os.path.join(out, "custom_camera_repeated_first_frame.mp4")
                _write_repeated_frame_video(first_frame, frame_count, image_input_path)
                video = image_input_path
                cmd = [python, f"{repo}/inference/sample.py", "--video", video, "--out", out, "--preview-only"]
                cmd += _args_for_custom_camera(effective_args, path_file, frame_count)
            else:
                if image is not None:
                    if image.ndim != 4 or image.shape[0] < 1:
                        raise ValueError("The connected image input must contain at least one frame.")
                    if image.shape[0] > 1:
                        image_input_path = os.path.join(out, "image_batch.mp4")
                        _write_image_batch_video(image, image_input_path)
                    else:
                        from PIL import Image

                        image_input_path = os.path.join(out, "still.png")
                        Image.fromarray(_tensor_rgb8(image[0])).save(image_input_path)
                    video = image_input_path

                cmd = [python, f"{repo}/inference/sample.py", "--video", video, "--out", out, "--preview-only"]
                cmd += _parse_cli_tokens(effective_args)

            cmd = _apply_vggt_source_settings(cmd, canvas_enabled, canvas_width, canvas_height,
                                              full_enabled, full_size, vggt_repo, vggt_checkpoint)
            cmd = _add_default_vggt_paths(cmd, repo)
            print("Meridian geometry (Enndee):", " ".join(cmd), flush=True)
            result = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout, flush=True)
            result.check_returncode()

            match = re.search(r"canvas \((\d+), (\d+)\)", result.stdout)
            if not match:
                raise RuntimeError("Meridian completed without reporting its render canvas dimensions.")
            width, height = map(int, match.groups())
            source = _frames(os.path.join(out, "cond_source.mp4"))
            render = _frames(os.path.join(out, "cond_render.mp4"))
            if entry:
                try:
                    _cache_store(entry, os.path.join(out, "cond_source.mp4"),
                                 os.path.join(out, "cond_render.mp4"), width, height, source.shape[0])
                    print(f"Meridian geometry (Enndee): cached -> {entry}", flush=True)
                except OSError as exc:
                    print(f"Meridian geometry (Enndee): cache store failed: {exc}", flush=True)
            return source, render, width, height, source.shape[0]
        finally:
            shutil.rmtree(out, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"Enndee_MeridianGeometry": EnndeeMeridianGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"Enndee_MeridianGeometry": "Meridian Geometry (Enndee)"}
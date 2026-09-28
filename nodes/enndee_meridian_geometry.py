"""Meridian Geometry (Enndee): the in-process fast-depth geometry condition pass.

The node runs the Depth-Anything point-cloud flight (`enndee_meridian_fast_depth.py`) for a
single still - V2 or V3 via `model_size`, no VGGT, no external environment, no subprocess. It
consumes the `args` string (or **Meridian Parameters and Camera**'s `args_override`) and the
optional `custom_camera` signal from that same node, and returns
(source, render, width, height, length) at the Meridian condition canvas.

The VGGT-Omega subprocess backend, its cache and the canvas/source-size/VGGT-path overrides
were removed on 2026-09-28 - every supported workflow now drives this backend, and the picker
only offers its two camera modes (manual path or automatic estimate).
"""

import json
import os
import shlex

import av
import numpy as np
import torch

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS, CAMERA_SIGNAL_TYPE
from enndee_meridian_fast_depth import DA3_RES, parse_camera_settings, render_depth_aligned

def _read_first_video_frame(video_path):
    """Decode only the first RGB frame from an existing video path."""
    with av.open(video_path) as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise ValueError(f"Could not decode a first frame from the video input: {video_path}")
    pixels = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(pixels.copy()).float().div_(255.0)


def _parse_cli_tokens(args_text):
    """Split CLI text while preserving quoted paths containing Windows separators."""
    portable_text = (args_text or "").replace("\\", "/")
    return shlex.split(portable_text)


def _parse_custom_camera(signal):
    try:
        data = json.loads(signal)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("custom_camera must be a valid signal from Meridian Parameters and Camera (Enndee).") from exc
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
                "args": ("STRING", {"default": "--boom 0.35 --pivot 0.5,0.55 --ease --sweep", "multiline": True, "tooltip": "Additional Meridian options. When custom_camera is connected, its path and frame count take precedence over camera-motion and frame-count flags."}),
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
                                        "tooltip": "Fast depth only: unprojection-grid upscale over the working still: 2 doubles the point count (denser silhouette fill), 1 keeps the working still's own resolution."}),
                "point_size": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1,
                                       "tooltip": "Fast depth only: point footprint 0=1x1, 1=3x3, 2=5x5, 3=7x7. Larger fills holes where the cloud is sparse after a big camera move."}),
                "edge_cull": ("BOOLEAN", {"default": True,
                                          "tooltip": "Fast depth only: drop points on steep depth edges (Meridian's 3x3 EDGE_RTOL rule, applied on the model's own depth grid) so silhouette borders cannot smear into flying spikes."}),
                "edge_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 2.0, "step": 0.01,
                                             "tooltip": "Fast depth only: cull points whose 3x3 relative depth spread exceeds this ratio (Meridian's EDGE_RTOL = 0.30)."}),
                "back_face_cull": ("BOOLEAN", {"default": False,
                                               "tooltip": "Fast depth only: mirror Meridian's --cull - drop the splats the target camera sees from behind, so a 180-degree view is a hole, not the mirrored front. This widget is the one place to set it: Meridian Parameters and Camera no longer emits --cull (a hand-written --cull in the args string still enables it, but never use both - the renderer reads them as one OR-ed switch, and the widget cannot turn an args-driven cull back off)."}),
                "depth_res": ("INT", {"default": DA3_RES, "min": 0, "max": 4096, "step": 1,
                                      "tooltip": "Fast depth only: working-resolution cap in pixels on the still's longest side (aspect preserved). A bigger input picture is resized down to it *before* the depth model, the colours and the point cloud are built, so huge photos stay fast and can never overflow the percentile clip; the same number is Depth-Anything-V3's `process_res` (rounded to multiples of 14 by the library; a value above the still's own side makes the model upscale). 0 = keep the still's own resolution for maximum depth detail - only a 16.7 Mpx safety ceiling still applies, and a full-resolution still costs seconds per frame on the depth model. The V2 models keep their native 518 depth grid, but the still and the cloud follow this cap for them too."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "args_override": ("STRING", {"forceInput": True, "tooltip": "Optional Meridian Parameters and Camera argument override."}),
                "custom_camera": (CAMERA_SIGNAL_TYPE, {"forceInput": True, "tooltip": "Connect custom_camera from Meridian Parameters and Camera (Enndee) - the manual path or the automatic estimate. Its frame count sets the flight length."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("source", "render", "width", "height", "length")
    FUNCTION = "build"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Meridian geometry condition pass for a single still: the in-process Depth-Anything "
        "point-cloud flight (V2 or V3 via `model_size`, no VGGT, no subprocess). `depth_res` "
        "caps the still's working resolution and the V3 depth grid (0 = the picture's own "
        "resolution); `model_size`, the canvas override, the cloud/point density and the two "
        "cull rules shape the flight. Feed `args` / `args_override` and the `custom_camera` "
        "signal from Meridian Parameters and Camera (Enndee) - manual path or automatic "
        "estimate - and the node returns (source, render, width, height, length) at the "
        "480-class condition canvas."
    )

    def _build_fast_depth(self, video, effective_args, image, custom_camera, model_size, canvas_mode,
                          custom_width, custom_height, cloud_scale, point_size, edge_cull,
                          edge_threshold, back_face_cull, depth_res=DA3_RES):
        """In-process Depth-Anything pass (V2 or V3 by `model_size`) with the same contract as the VGGT preview.

        The `args` string is parsed for the sample.py camera flags the fast backend honours
        (`parse_camera_settings`), so one Meridian Parameters and Camera output configures the
        flight. `--follow` cannot be replayed without VGGT poses and is reported instead of
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
                raise ValueError("Fast depth cannot replay a source video's own camera path (--follow); "
                                 "remove the flag from the args or the parameter picker.")
            frame_count = 73 if settings["frames"] is None else int(settings["frames"])
            if frame_count not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
                raise ValueError("Fast depth needs a Meridian output length "
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

    def build(self, video, args, image=None, args_override=None, custom_camera=None,
              model_size="Depth-Anything-V2-Small-hf", canvas_mode="auto_meridian480",
              custom_width=832, custom_height=480, cloud_scale=2, point_size=1, edge_cull=True,
              edge_threshold=0.30, back_face_cull=False, depth_res=DA3_RES):
        """Run the fast-depth flight; `args`/`args_override` and `custom_camera` configure it."""
        effective_args = args_override if args_override is not None else args
        return self._build_fast_depth(video, effective_args, image, custom_camera, model_size,
                                      canvas_mode, custom_width, custom_height, cloud_scale,
                                      point_size, edge_cull, edge_threshold, back_face_cull,
                                      depth_res)


NODE_CLASS_MAPPINGS = {"Enndee_MeridianGeometry": EnndeeMeridianGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"Enndee_MeridianGeometry": "Meridian Geometry (Enndee)"}
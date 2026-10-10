"""Meridian Geometry (Enndee): the in-process fast-depth geometry condition pass.

The node runs the Depth-Anything point-cloud flight (`enndee_meridian_fast_depth.py`) for a
single still - V2 or V3 via `model_size`, no VGGT, no external environment, no subprocess. It
consumes the `args` string (or **Meridian Parameters and Camera**'s `args_override`) and the
optional `custom_camera` signal from that same node, and returns
(source, render, width, height, length) at the Meridian condition canvas.

The VGGT-Omega subprocess backend, its cache and the canvas/source-size/VGGT-path overrides
were removed on 2026-09-28 - every supported workflow now drives this backend, and the picker
only offers its two camera modes (manual path or automatic estimate).

`model_size` offers exactly two depth sources (2026-10-10): **Depth Anything 3 Mono Large**, or
**external**, which loads no model and consumes the `external_depth` map instead. The external
path is the answer to drift: a per-frame depth *estimate* is only self-consistent frame by
frame, so a scene shot in several takes gets a slightly different geometry each time, while one
map (or several shots rendered against one shared map) reprojects identically. The renderer only
needs relative depth with the right sign - it re-gauges the map through the same percentile clip
the model path uses - so `external_depth_polarity` is the only thing that has to be right.
"""

import json
import os
import shlex

import av
import numpy as np
import torch

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS, CAMERA_SIGNAL_TYPE
from enndee_meridian_fast_depth import DA3_RES, parse_camera_settings, render_depth_aligned

#: `model_size` choices. The picker is deliberately tiny: Depth-Anything-3 Mono-Large is the
#: best single-still depth model here, and `external` hands the depth to the connected
#: `external_depth` map instead of running a model at all. The V2 trio and the V3 any-view
#: trio were dropped on 2026-10-10 - neither beat Mono-Large on a still, and every extra
#: choice is one the user has to reason about. They still ROUTE if a saved workflow or a
#: script passes one by name (LEGACY_DEPTH_MODELS), so nothing silently changes meaning.
MODEL_OPTIONS = ["Depth Anything 3 Mono Large", "external"]

#: Picker label -> the backend's own model id.
DEPTH_MODEL_IDS = {
    "Depth Anything 3 Mono Large": "Depth-Anything-3-Mono-Large",
}

#: Older picker values, still accepted programmatically, so a workflow saved before the
#: picker shrank keeps working if its value reaches the node unchanged.
LEGACY_DEPTH_MODELS = {
    "Depth-Anything-V2-Small-hf": "Depth-Anything-V2-Small-hf",
    "Depth-Anything-V2-Base-hf": "Depth-Anything-V2-Base-hf",
    "Depth-Anything-V2-Large-hf": "Depth-Anything-V2-Large-hf",
    "Depth-Anything-3-Small": "Depth-Anything-3-Small",
    "Depth-Anything-3-Base": "Depth-Anything-3-Base",
    "Depth-Anything-3-Large": "Depth-Anything-3-Large",
    "Depth-Anything-3-Mono-Large": "Depth-Anything-3-Mono-Large",
}

#: `external_depth_polarity` -> the backend's `external_depth_invert`.
#: The backend wants "larger value = farther" (it re-gauges the map through a percentile
#: clip, so the absolute scale is irrelevant - only the ordering and the sign matter).
#: Depth-Anything and most 16-bit depth exports are already in that convention; MiDaS/DPT
#: disparity maps and many "depth preview" PNGs are not.
EXTERNAL_DEPTH_POLARITIES = {
    "white is far (depth)": False,
    "white is near (disparity)": True,
}

EXTERNAL_MODEL = "external"


def resolve_depth_model(model_size):
    """Picker label (or legacy id) -> ``(backend model id, use_external)``."""
    choice = str(model_size or "").strip()
    if choice == EXTERNAL_MODEL:
        return DEPTH_MODEL_IDS[MODEL_OPTIONS[0]], True
    if choice in DEPTH_MODEL_IDS:
        return DEPTH_MODEL_IDS[choice], False
    if choice in LEGACY_DEPTH_MODELS:
        return LEGACY_DEPTH_MODELS[choice], False
    raise ValueError(f"Unsupported depth model {model_size!r}; expected one of "
                     f"{', '.join(MODEL_OPTIONS)}.")

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
                "args": ("STRING", {"default": "", "multiline": True, "tooltip": "Additional Meridian options. When custom_camera is connected, its path and frame count take precedence over camera-motion and frame-count flags. Empty by default: the Parameters node emits its own arguments, and hand-written flags here are OR-ed with the widgets below."}),
                "model_size": (list(MODEL_OPTIONS),
                               {"default": MODEL_OPTIONS[0],
                                "tooltip": "Where the depth comes from. 'Depth Anything 3 Mono Large' runs the depth model in-process (needs `python -m pip install --no-deps depth-anything-3` in the ComfyUI python_embeded; weights land in the Hugging Face cache, Apache-2.0) - it is the best single-still depth model here. 'external' loads NO model and uses the connected external_depth map instead, which is the only way to make several shots of one scene share a single consistent geometry (see external_depth). The former V2 and V3-Small/Base/Large choices were removed: none of them beat Mono-Large on a still, and a shorter list is a shorter decision."}),
                "canvas_mode": (["auto_meridian480", "custom"], {"default": "custom",
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
                "edge_threshold": ("FLOAT", {"default": 0.10, "min": 0.05, "max": 2.0, "step": 0.01,
                                             "tooltip": "Fast depth only: cull points whose 3x3 relative depth spread exceeds this ratio (Meridian's own EDGE_RTOL is 0.30; a tighter value keeps more of the silhouette and drops fewer fine details)."}),
                "back_face_cull": ("BOOLEAN", {"default": True,
                                               "tooltip": "Fast depth only: mirror Meridian's --cull - drop the splats the target camera sees from behind, so a 180-degree view is a hole, not the mirrored front. This widget is the one place to set it: Meridian Parameters and Camera no longer emits --cull (a hand-written --cull in the args string still enables it, but never use both - the renderer reads them as one OR-ed switch, and the widget cannot turn an args-driven cull back off)."}),
                # 1920 matches the Meridian_Splatting_1.0 example workflow: the render depth model
                # runs at (about) the still's own side, so the reprojection keeps full detail.
                "depth_res": ("INT", {"default": 1920, "min": 0, "max": 4096, "step": 1,
                                      "tooltip": "Fast depth only: working-resolution cap in pixels on the still's longest side (aspect preserved). A bigger input picture is resized down to it *before* the depth model, the colours and the point cloud are built, so huge photos stay fast and can never overflow the percentile clip; the same number is Depth-Anything-V3's `process_res` (rounded to multiples of 14 by the library; a value above the still's own side makes the model upscale). 0 = keep the still's own resolution for maximum depth detail - only a 16.7 Mpx safety ceiling still applies, and a full-resolution still costs seconds per frame on the depth model. The V2 models keep their native 518 depth grid, but the still and the cloud follow this cap for them too."}),
                # Appended LAST on purpose: ComfyUI maps a saved workflow's widget_values by
                # insertion order, so a widget inserted in the middle would silently remap
                # every later widget of existing workflows.
                "external_depth_polarity": (list(EXTERNAL_DEPTH_POLARITIES),
                                            {"default": list(EXTERNAL_DEPTH_POLARITIES)[0],
                                             "tooltip": "External depth only: which way round the connected depth map is. The renderer needs 'larger value = farther' and re-gauges the map itself, so the absolute scale never matters - only the ordering and the sign. 'white is far (depth)' is what Depth-Anything and most 16-bit depth exports produce; 'white is near (disparity)' is the MiDaS/DPT convention and what many 'depth preview' PNGs show. Getting it wrong mirrors the scene: the background becomes the foreground and the flight flies backwards."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "args_override": ("STRING", {"forceInput": True, "tooltip": "Optional Meridian Parameters and Camera argument override."}),
                "custom_camera": (CAMERA_SIGNAL_TYPE, {"forceInput": True, "tooltip": "Connect custom_camera from Meridian Parameters and Camera (Enndee) - the manual path or the automatic estimate. Its frame count sets the flight length."}),
                "external_depth": ("IMAGE", {"tooltip": "External depth only (model_size='external'): the depth map for the still, as a ComfyUI IMAGE (one frame; a colour map is averaged to luminance). It replaces the depth model entirely and is the way to get CONSISTENT geometry - a single map, or several shots rendered against one shared map, always reprojects identically, while a per-frame model estimate drifts. It must be aligned with the image input (same framing); the node warns when the aspect ratios disagree. Relative depth is enough (the renderer re-gauges it), and external_depth_polarity sets the sign."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("source", "render", "width", "height", "length")
    FUNCTION = "build"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Meridian geometry condition pass for a single still: the in-process Depth-Anything-3 "
        "Mono-Large point-cloud flight (no VGGT, no subprocess), or an EXTERNAL depth map on "
        "`external_depth` when `model_size` is 'external' - which is how several shots are "
        "given one consistent geometry. `depth_res` caps the still's working resolution and "
        "the depth grid (0 = the picture's own resolution); the canvas override, the "
        "cloud/point density and the two cull rules shape the flight. Feed `args` / "
        "`args_override` and the `custom_camera` signal from Meridian Parameters and Camera "
        "(Enndee) - manual path or automatic estimate - and the node returns (source, render, "
        "width, height, length) at the 480-class condition canvas."
    )

    def _build_fast_depth(self, video, effective_args, image, custom_camera, model_size, canvas_mode,
                          custom_width, custom_height, cloud_scale, point_size, edge_cull,
                          edge_threshold, back_face_cull, depth_res=DA3_RES,
                          external_depth=None, external_depth_polarity=None):
        """Depth pass - the model by `model_size`, or a connected map - with the VGGT contract.

        The `args` string is parsed for the sample.py camera flags the fast backend honours
        (`parse_camera_settings`), so one Meridian Parameters and Camera output configures the
        flight. `--follow` cannot be replayed without VGGT poses and is reported instead of
        silently ignored; `--freeze`, `--start` and `--canvas` do not affect a single still.
        """
        resolved_model, use_external = resolve_depth_model(model_size)
        polarity = str(external_depth_polarity or list(EXTERNAL_DEPTH_POLARITIES)[0])
        if polarity not in EXTERNAL_DEPTH_POLARITIES:
            raise ValueError(f"Unsupported external depth polarity {external_depth_polarity!r}; "
                             f"expected one of {', '.join(EXTERNAL_DEPTH_POLARITIES)}.")
        if use_external and external_depth is None:
            raise ValueError(
                "model_size='external' needs a depth map on the external_depth input: connect "
                "one, or switch the model back to 'Depth Anything 3 Mono Large'."
            )
        if external_depth is not None and not use_external:
            print("Meridian geometry (Enndee): a depth map is connected on external_depth but "
                  "model_size is not 'external' - the map is IGNORED and the depth model runs. "
                  "Set model_size to 'external' to use the connected map.", flush=True)
            external_depth = None
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
        if external_depth is not None:
            print("Meridian geometry (Enndee): external depth map in use - no depth model is "
                  "loaded and the map's own geometry drives the flight.", flush=True)
        source, render, width, height, length = render_depth_aligned(
            first_frame, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_size=resolved_model, frames=frame_count, canvas_mode=canvas_mode,
            custom_width=custom_width, custom_height=custom_height, cloud_scale=cloud_scale,
            point_size=point_size, edge_cull=edge_cull, edge_threshold=edge_threshold,
            back_face_cull=back_face_cull, camera=settings, custom_camera=custom_camera,
            depth_res=depth_res, external_depth=external_depth,
            external_depth_invert=EXTERNAL_DEPTH_POLARITIES[polarity],
        )
        print(f"Meridian geometry (Enndee): fast depth -> {width}x{height}, {length} frames.", flush=True)
        return source, render, width, height, length

    def build(self, video, args, image=None, args_override=None, custom_camera=None,
              model_size=MODEL_OPTIONS[0], canvas_mode="auto_meridian480",
              custom_width=832, custom_height=480, cloud_scale=2, point_size=1, edge_cull=True,
              edge_threshold=0.30, back_face_cull=False, depth_res=DA3_RES,
              external_depth_polarity=None, external_depth=None):
        """Run the flight; `args`/`args_override` and `custom_camera` configure it.

        `model_size` picks the depth source (see MODEL_OPTIONS) and `external_depth` carries
        the map for the `external` choice.
        """
        effective_args = args_override if args_override is not None else args
        return self._build_fast_depth(video, effective_args, image, custom_camera, model_size,
                                      canvas_mode, custom_width, custom_height, cloud_scale,
                                      point_size, edge_cull, edge_threshold, back_face_cull,
                                      depth_res, external_depth, external_depth_polarity)


NODE_CLASS_MAPPINGS = {"Enndee_MeridianGeometry": EnndeeMeridianGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"Enndee_MeridianGeometry": "Meridian Geometry (Enndee)"}
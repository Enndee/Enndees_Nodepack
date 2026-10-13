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
import sys
from pathlib import Path

import av
import numpy as np
import torch

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS, CAMERA_SIGNAL_TYPE
from enndee_meridian_fast_depth import (DA3_RES, SURFACE_DEFAULT, SURFACE_MODES,
                                        parse_camera_settings, render_depth_aligned,
                                        resolve_depth_relief)

# `enndee_moge` lives at the pack root. This node only needs its constants and its validator -
# the `moge` package itself is imported lazily by the backend, so MoGe-3 stays optional and
# every other depth source keeps working without it.
_PACK_DIR = Path(__file__).resolve().parent.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

from enndee_moge import (  # noqa: E402  (path setup must run first)
    DEFAULT_REFINE_STEPS as MOGE_REFINE_STEPS,
    MAX_REFINE_STEPS as MOGE_MAX_REFINE_STEPS,
    resolve_refine_steps,
)

#: `model_size` choices - MoGe 3 ONLY.
#:
#: Depth-Anything-3 and `external` were REMOVED on 2026-10-12, and deliberately so. The
#: auto-camera's face orbit registers the SAM3 face box onto the depth cloud with the SAME
#: focal the cloud was built with, and that focal comes from the source camera's measured
#: intrinsics. MoGe-3 is the only depth source here that reports them (`infer` returns a
#: 3x3 intrinsics whose `fy` gives the vertical FOV). DA3 reports nothing, so
#: `LAST_VFOV_DEGREES` stays None and every projection silently falls back to the
#: hard-coded `VFOV_DEGREES = 55` - a 1.509x error in f on the Hanks still (real value
#: 38.07) - which is exactly the mis-registration that put the face orbit "slightly left of
#: and behind his head". `external` is worse still: it hands in a depth map with no camera
#: at all.
#:
#: The choice is a hard dependency now, so it is not offered. Keep `MODEL_OPTIONS` a
#: one-element list rather than deleting the widget - removing an INPUT shifts every later
#: widget's index and silently re-maps the saved workflow's `widgets_values`. Every other
#: value - the old DA3 label, the V2/V3 ids, `external` - is REJECTED by
#: `resolve_depth_model` rather than silently degraded at the wrong focal.
MODEL_OPTIONS = ["MoGe 3"]

#: Picker label -> the backend's own model id. Strictly MoGe-3: see the note above.
DEPTH_MODEL_IDS = {
    "MoGe 3": "MoGe-3-ViT-L",
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
MOGE_MODEL = "MoGe 3"


def resolve_depth_model(model_size):
    """Picker label -> ``(backend model id, use_external)``.

    MoGe-3 only. DA3 and ``external`` are REJECTED here rather than silently degraded: a
    depth map with no camera intrinsics makes the face orbit project at the wrong focal,
    which is a subtle wrong answer rather than a failure. See ``MODEL_OPTIONS``.
    """
    choice = str(model_size or "").strip()
    if choice in DEPTH_MODEL_IDS:
        return DEPTH_MODEL_IDS[choice], False
    raise ValueError(
        f"Unsupported depth model {model_size!r}. This node now requires MoGe-3 "
        f"({MOGE_MODEL!r}) - it is the only depth source that reports the camera "
        f"intrinsics the face orbit needs in order to register the SAM3 face box onto the "
        f"cloud. Depth-Anything-3 and 'external' were removed on 2026-10-12.")

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
    portable_text = str(args_text if args_text is not None else "").replace("\\", "/")
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
                # Appended LAST again (see the note above `external_depth_polarity`): a widget
                # inserted anywhere earlier would shift every later widget of a saved workflow.
                "moge_refine_steps": ("INT", {"default": MOGE_REFINE_STEPS, "min": 0,
                                              "max": MOGE_MAX_REFINE_STEPS, "step": 1,
                                              "tooltip": "MoGe 3 only: how many steps of MoGe-3's self-guided sparse volumetric refinement to run. MoGe-3 voxelises its coarse point map into a sparse 3D grid and walks it with a sparse 3D-UNet, which is what sharpens the fine geometry. 0 disables the stage (a single-pass, MoGe-2-style result) and is the fastest; 3 is MoGe's own default; past ~8 the extra time buys little. Ignored by every other depth source. Requires the OPTIONAL `moge` package (plus FlexGEMM): `git clone https://github.com/microsoft/MoGe.git` then `python -m pip install ./MoGe` in the ComfyUI python_embeded."}),
                # Appended LAST again (see the note above `external_depth_polarity`): a widget
                # inserted anywhere earlier would shift every later widget of a saved workflow.
                "surface_mode": (list(SURFACE_MODES),
                                 {"default": SURFACE_DEFAULT,
                                  "tooltip": "How the depth grid becomes the render. 'points' (default) is the original square-footprint splat: fast and proven. 'mesh' triangulates the depth grid instead, so the flight is a closed surface with real occlusion, no holes between points and no square point footprints - which makes Point Size and Back Face Cull unnecessary (they only apply to 'points'). Edge Cull and Edge Threshold still apply to both: they decide which parts of the grid are trusted, and for the mesh that is exactly what cuts the sheet at a silhouette - or at a depth step inside it - instead of bridging it with long stretched triangles. A triangle that crosses the camera plane is clipped against it rather than dropped, so geometry is only lost when the whole of it is behind the camera. The mesh is still opt-in for a different reason: when the camera flies inside the cloud a solid surface fills the view where the point splat still shows the scene, and the auto-camera spiral does fly through the backdrop at the end. Use it to compare the two renderers on the same clip."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "args_override": ("STRING", {"forceInput": True, "tooltip": "Optional Meridian Parameters and Camera argument override."}),
                # The LAST widget (see the note above `surface_mode`): 1.0 = identity,
                # `z' = median + relief * (z - median)` scales the depth spread about the
                # median, leaving `zm` - and therefore the whole camera rig - untouched and
                # keeping `x/z` so the identity frame reprojects exactly. Meridian Parameters
                # and Camera carries its own depth_relief in custom_camera and that value wins.
                "depth_relief": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                            "tooltip": "Fallback depth relief for camera paths that carry no depth_relief metadata. 1.0 = identity; ~1.3-1.5 deepens a shallow subject. The Meridian Parameters and Camera value in custom_camera takes precedence so the face pivot and the rendered surface see the same relief. Appended LAST to preserve saved widget indices."}),

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
                          external_depth=None, external_depth_polarity=None,
                          moge_refine_steps=MOGE_REFINE_STEPS, surface_mode=SURFACE_DEFAULT,
                 depth_relief=1.0):
        """Depth pass - the model by `model_size`, or a connected map - with the VGGT contract.

        The `args` string is parsed for the sample.py camera flags the fast backend honours
        (`parse_camera_settings`), so one Meridian Parameters and Camera output configures the
        flight. `--follow` cannot be replayed without VGGT poses and is reported instead of
        silently ignored; `--freeze`, `--start` and `--canvas` do not affect a single still.
        """
        resolved_model, _use_external = resolve_depth_model(model_size)
        polarity = str(external_depth_polarity or list(EXTERNAL_DEPTH_POLARITIES)[0])
        if polarity not in EXTERNAL_DEPTH_POLARITIES:
            raise ValueError(f"Unsupported external depth polarity {external_depth_polarity!r}; "
                             f"expected one of {', '.join(EXTERNAL_DEPTH_POLARITIES)}.")
        # Validated here (not in the backend) so a bad widget value is reported with the node's
        # own wording; the backend re-validates, since the auto-camera calls it directly.
        refine_steps = resolve_refine_steps(moge_refine_steps)
        depth_relief = resolve_depth_relief(depth_relief)
        surface = str(surface_mode or SURFACE_DEFAULT)
        if surface not in SURFACE_MODES:
            raise ValueError(f"Unsupported surface mode {surface_mode!r}; "
                             f"expected one of {', '.join(SURFACE_MODES)}.")
        if external_depth is not None:
            print("Meridian geometry (Enndee): a depth map is connected on external_depth but "
                  "external depth was removed on 2026-10-12 (this node is MoGe-3 only) - the "
                  "map is IGNORED and MoGe-3 runs.", flush=True)
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
            # The camera-path node's relief is authoritative: its face pivot and measured
            # keys were computed on the relieved surface, so the render must use it too.
            camera_relief = _camera_data.get("depth_relief")
            if camera_relief is not None:
                depth_relief = resolve_depth_relief(camera_relief)
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
            moge_refine_steps=refine_steps,
            surface_mode=surface,
            depth_relief=depth_relief,
        )
        print(f"Meridian geometry (Enndee): fast depth -> {width}x{height}, {length} frames.", flush=True)
        return source, render, width, height, length

    def build(self, video, args, image=None, args_override=None, custom_camera=None,
              model_size=MODEL_OPTIONS[0], canvas_mode="auto_meridian480",
              custom_width=832, custom_height=480, cloud_scale=2, point_size=1, edge_cull=True,
              edge_threshold=0.30, back_face_cull=False, depth_res=DA3_RES,
              external_depth_polarity=None, external_depth=None,
              moge_refine_steps=MOGE_REFINE_STEPS, surface_mode=SURFACE_DEFAULT,
              depth_relief=1.0):
        """Run the flight; `args`/`args_override` and `custom_camera` configure it.

        `model_size` picks the depth source (see MODEL_OPTIONS), `external_depth` carries the
        map for the `external` choice, `moge_refine_steps` is MoGe-3's refinement-step count
        (ignored by the other two) and `surface_mode` picks mesh or point-splat rendering.
        """
        effective_args = args_override if args_override is not None else args
        return self._build_fast_depth(video, effective_args, image, custom_camera, model_size,
                                      canvas_mode, custom_width, custom_height, cloud_scale,
                                      point_size, edge_cull, edge_threshold, back_face_cull,
                                      depth_res, external_depth, external_depth_polarity,
                                      moge_refine_steps, surface_mode, depth_relief)


NODE_CLASS_MAPPINGS = {"Enndee_MeridianGeometry": EnndeeMeridianGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"Enndee_MeridianGeometry": "Meridian Geometry (Enndee)"}
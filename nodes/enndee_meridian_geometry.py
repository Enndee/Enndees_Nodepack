"""Meridian Geometry runner with custom-camera-path and repeated-first-frame support."""

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
        "Run Meridian VGGT geometry preview. Connect a generated custom_camera signal to automatically "
        "repeat the source's first frame to the exact camera-path length and apply its path."
    )

    def build(self, video, args, repo, python, image=None, args_override=None, custom_camera=None):
        out = tempfile.mkdtemp(prefix="enndee_meridian_")
        image_input_path = None
        try:
            effective_args = args_override if args_override is not None else args

            if custom_camera is not None:
                camera_data, frame_count = _parse_custom_camera(custom_camera)
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
            return source, render, width, height, source.shape[0]
        finally:
            shutil.rmtree(out, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"Enndee_MeridianGeometry": EnndeeMeridianGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"Enndee_MeridianGeometry": "Meridian Geometry (Enndee)"}
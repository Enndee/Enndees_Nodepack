# type: ignore
"""
Enndee_VideoFrameExtractorWithAudio
─────────────────────────────────────────────────────────────────────────────────
Copy of `comfyui-video-frame-extractor` (VideoFrameExtractor) extended with an
AUDIO output: the node clips the *video* frames from the timeline loop region
AND extracts the matching *audio* segment of the source video
(start_frame .. end_frame → time range via fps).

New output vs. the original node:
  • "Clipped Audio" (AUDIO) or None if the video has no audio stream /
    ffmpeg is unavailable. Compatible with core SaveAudio, AudioPreview etc.

Audio extraction uses ffmpeg (resolved via ENNDEE_FORCE_FFMPEG_PATH env var,
imageio_ffmpeg, or PATH).

Timeline widget + API routes:
  • JS widget:  web/js/enndee_video_frame_extractor.js
    (served from WEB_DIRECTORY below — no other custom node needed)
  • API routes: /enndee_video_frame_extractor/info
                /enndee_video_frame_extractor/thumbnail
"""

import os
import io
import re
import sys
import shutil
import hashlib
import subprocess
import folder_paths
import numpy as np
import torch
from PIL import Image
import cv2


# ---------------------------------------------------------------------------
# Helper utilities (identical to the original node)
# ---------------------------------------------------------------------------

# Safety limits for memory estimation
DEFAULT_MAX_MEMORY_BYTES = 8 * 1024**3    # hard cap if psutil unavailable
SYSTEM_RAM_FRACTION_CAP  = 0.75           # never exceed 75% of available RAM


def estimate_peak_memory(width: int, height: int,
                         num_frames: int, custom_num_frames: int) -> int:
    """
    Estimate peak RAM in bytes used during frame extraction.

    Breakdown per call to extract_frames:
      • uint8 staging buffer      N × H × W × 3 × 1
      • float32 output tensor     N × H × W × 3 × 4
    Both live in memory simultaneously at peak.

    Plus we keep around:
      • frames               (float32, N frames)
      • frames_reversed      (float32, same N — torch.flip allocates new storage)
      • frames_at_fps        (float32, custom_N frames)
    """
    bpp_u8  = width * height * 3          # bytes per frame, uint8
    bpp_f32 = bpp_u8 * 4                  # bytes per frame, float32

    # Peak during the first extract_frames() call
    first_call_peak = (num_frames * bpp_u8) + (num_frames * bpp_f32)

    # After first call completes, frames (f32) + frames_reversed (f32) are resident
    resident_after_first = (num_frames * bpp_f32) * 2

    # Peak during the second extract_frames() call (target-fps)
    second_call_peak = (
        resident_after_first
        + (custom_num_frames * bpp_u8)
        + (custom_num_frames * bpp_f32)
    )

    return max(first_call_peak, second_call_peak)


def get_memory_limit() -> int:
    """Return the max allowed peak memory in bytes."""
    try:
        import psutil
        available = psutil.virtual_memory().available
        return int(available * SYSTEM_RAM_FRACTION_CAP)
    except Exception:
        return DEFAULT_MAX_MEMORY_BYTES


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps":          fps,
        "total_frames": total,
        "width":        width,
        "height":       height,
        "duration_seconds": total / fps if fps > 0 else 0,
        "bytes_per_frame_f32": width * height * 3 * 4,
        "memory_limit_bytes": get_memory_limit(),
    }


def read_single_frame(cap: cv2.VideoCapture, frame_index: int,
                      width: int, height: int) -> np.ndarray:
    """Read one frame; return a black frame on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_frames(video_path: str, start_frame: int, end_frame: int,
                   num_frames: int) -> torch.Tensor:
    """
    Extract *num_frames* evenly-spaced frames from [start_frame, end_frame].
    Returns a BHWC float32 tensor in [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, min(start_frame, total - 1))
    end_frame   = max(start_frame, min(end_frame, total - 1))
    num_frames  = max(1, num_frames)

    if start_frame == end_frame:
        indices = [start_frame]
    else:
        indices = [
            int(round(start_frame + i * (end_frame - start_frame) / (num_frames - 1)))
            for i in range(num_frames)
        ]

    arr = np.empty((len(indices), h, w, 3), dtype=np.uint8)
    for i, idx in enumerate(indices):
        arr[i] = read_single_frame(cap, idx, w, h)
    cap.release()

    float_arr = arr.astype(np.float32) / 255.0
    del arr
    return torch.from_numpy(float_arr)   # BHWC


# ---------------------------------------------------------------------------
# Audio helpers (EXTRA — not part of the original node)
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> str:
    """
    Resolve an ffmpeg executable:
      1. env var ENNDEE_FORCE_FFMPEG_PATH (absolute path)
      2. imageio_ffmpeg.get_ffmpeg_exe()  (bundled static build, if installed)
      3. shutil.which("ffmpeg")           (PATH)
    Raises RuntimeError if nothing suitable is found.
    """
    env = os.environ.get("ENNDEE_FORCE_FFMPEG_PATH")
    if env:
        if os.path.isfile(env):
            return env
        raise RuntimeError(
            f"ENNDEE_FORCE_FFMPEG_PATH is set but the file does not exist: {env}"
        )

    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        p = get_ffmpeg_exe()
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass

    p = shutil.which("ffmpeg")
    if p:
        return p

    raise RuntimeError(
        "No ffmpeg found for audio extraction. Install imageio_ffmpeg "
        "(`pip install imageio-ffmpeg`), put ffmpeg on PATH, or set "
        "ENNDEE_FORCE_FFMPEG_PATH to an absolute ffmpeg.exe location."
    )


def _channels_from_layout(layout: str) -> int:
    layout = (layout or "").strip().lower()
    map_ = {
        "mono": 1,
        "stereo": 2,
        "quad": 4,
        "2.1": 3,
        "5.1": 6,
        "7.1": 8,
        "1.0": 1,
        "2.0": 2,
    }
    if layout in map_:
        return map_[layout]
    # ffmpeg sometimes reports pure channel counts like "3", "6"
    if layout.isdigit():
        return int(layout)
    return 2  # default — most common case


def clip_audio(video_path: str, start_time: float, duration: float):
    """
    Extract the audio segment [start_time, start_time + duration] of a video
    file. Returns a ComfyUI AUDIO mapping
        {"waveform": (1, channels, samples) float32 in [-1, 1],
         "sample_rate": int}
    or None when the file has no audio stream / ffmpeg is unavailable.
    """
    if duration <= 0:
        return None
    try:
        ffmpeg = _resolve_ffmpeg()
    except Exception as exc:
        print(f"\033[93m[Enndee VFE+] {exc:}\033[0m")
        return None

    # 1) Probe for an audio stream + its native sample rate/layout
    try:
        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", video_path],
            capture_output=True, timeout=60,
        )
    except Exception as exc:
        print(f"\033[93m[Enndee VFE+] Audio probe failed: {exc}\033[0m")
        return None
    probe_err = probe.stderr.decode("utf-8", errors="ignore")

    audio_lines = [ln for ln in probe_err.splitlines() if "Audio:" in ln]
    if not audio_lines:
        print("\033[93m[Enndee VFE+] No audio stream in video — outputting None.\033[0m")
        return None

    sample_rate = 44100
    channels    = 2
    m = re.search(r"(\d+)\s*Hz", audio_lines[0])
    if m:
        sample_rate = int(m.group(1))
    m = re.search(r"Hz,\s*([\w.]+)", audio_lines[0])
    if m:
        channels = _channels_from_layout(m.group(1))

    # 2) Extract raw float32 audio for the requested window.
    #    -ss before -i = fast input seek; -t counts from the seeked start.
    try:
        res = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-ss", f"{max(0.0, start_time):.6f}",
             "-t", f"{duration:.6f}",
             "-i", video_path,
             "-vn", "-f", "f32le", "-"],
            capture_output=True, timeout=300, check=True,
        )
    except Exception as exc:
        err = ""
        if isinstance(exc, subprocess.CalledProcessError):
            err = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        else:
            err = str(exc)
        print(f"\033[93m[Enndee VFE+] Audio extraction failed: {err:}\033[0m")
        return None

    raw = res.stdout
    if len(raw) < 4:
        print("\033[93m[Enndee VFE+] Audio stream produced no data — outputting None.\033[0m")
        return None

    samples = np.frombuffer(raw, dtype=np.float32)
    # Trim to complete channel frames (defensive; should already be aligned)
    length = int(len(samples) // channels) * channels
    if length == 0:
        return None
    waveform = (
        torch.from_numpy(samples[:length].copy())
        .reshape(-1, channels)
        .transpose(0, 1)
        .unsqueeze(0)                        # (1, C, N)
    )
    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Node definition
# ---------------------------------------------------------------------------

class Enndee_VideoFrameExtractorWithAudio:
    """
    ComfyUI node: Enndee_VideoFrameExtractorWithAudio
    ─────────────────────────────────────────────────
    Copy of the Video Frame Extractor node, additionally outputting the
    audio track of the selected loop region:
      • first_frame   — single IMAGE of the loop start frame
      • last_frame    — single IMAGE of the loop end frame
      • frames        — all extracted frames in forward order (BHWC)
      • frames_reversed — same frames in reverse order (BHWC)
      • filename      — video filename (STRING)
      • width         — video width in pixels (INT)
      • height        — video height in pixels (INT)
      • fps           — source frame rate (FLOAT)
      • clipped_audio — AUDIO of [start_frame .. end_frame] (or None)
      • frames (target fps) / target fps — resampled batch (IMAGE / FLOAT)
    """

    @classmethod
    def INPUT_TYPES(cls):
        video_files = []
        input_dir = folder_paths.get_input_directory()
        for fname in sorted(os.listdir(input_dir)):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                video_files.append(fname)
        if not video_files:
            video_files = [""]

        return {
            "required": {
                "video": (video_files, {
                    "video_upload": True,
                }),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 999999, "step": 1,
                    "display": "number",
                }),
                "end_frame": ("INT", {
                    "default": 0, "min": 0, "max": 999999, "step": 1,
                    "display": "number",
                }),
                "num_frames": ("INT", {
                    "default": 0, "min": 0, "max": 999999, "step": 1,
                    "display": "number",
                }),
                "target_fps": ("FLOAT", {
                    "default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT", "FLOAT", "IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES  = ("Clipped Frames", "Reversed Clipped Frames", "First Frame", "Last Frame",
                     "Filename Prefix (no extension)", "Width", "Height", "Original FPS",
                     "Clipped Frames at Target FPS", "Target FPS",
                     "Clipped Audio")
    FUNCTION      = "extract"
    CATEGORY      = "video"
    OUTPUT_NODE   = True

    @classmethod
    def IS_CHANGED(cls, video, start_frame, end_frame, num_frames, target_fps):
        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            return float("nan")
        m = hashlib.md5()
        m.update(f"{video}{start_frame}{end_frame}{num_frames}{target_fps}".encode())
        return m.hexdigest()

    def extract(self, video: str, start_frame: int, end_frame: int,
                num_frames: int, target_fps: float):
        video_path = folder_paths.get_annotated_filepath(video)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        info   = get_video_info(video_path)
        total  = info["total_frames"]
        width  = info["width"]
        height = info["height"]
        fps    = info["fps"]

        # Clamp loop region
        start_frame = max(0, min(start_frame, total - 1))
        end_frame   = max(start_frame, min(end_frame, total - 1))
        if end_frame == 0 and start_frame == 0:
            end_frame = total - 1

        # num_frames == 0 → auto (JS widget sets this to loop span)
        if num_frames <= 0:
            num_frames = end_frame - start_frame + 1

        # ── Memory safety check ──────────────────────────────────────────
        clip_duration     = (end_frame - start_frame + 1) / fps if fps > 0 else 0
        custom_num_frames = max(1, round(clip_duration * target_fps))
        peak_bytes        = estimate_peak_memory(width, height, num_frames,
                                                 custom_num_frames)
        limit_bytes       = get_memory_limit()

        if peak_bytes > limit_bytes:
            raise RuntimeError(
                f"Refusing to run: estimated peak memory "
                f"{format_bytes(peak_bytes)} exceeds the safe limit of "
                f"{format_bytes(limit_bytes)}.\n\n"
                f"Clip: {num_frames} frames @ {width}x{height} "
                f"(+{custom_num_frames} frames at {target_fps:.1f} fps).\n\n"
                f"Reduce the loop region, shrink the video resolution, or "
                f"lower target_fps. A shorter clip uses proportionally less RAM."
            )

        # Extract the frame batch
        frames          = extract_frames(video_path, start_frame, end_frame, num_frames)
        frames_reversed = torch.flip(frames, dims=[0])

        # Single-frame outputs for the loop boundaries — slice from already-loaded batch
        first_frame = frames[0:1]
        last_frame  = frames[-1:]

        # Filename prefix — basename without extension
        filename_prefix = os.path.splitext(os.path.basename(video_path))[0]

        # Custom FPS batch: resample the loop region to target_fps
        # (custom_num_frames was computed above for the memory check)
        frames_at_fps     = extract_frames(video_path, start_frame, end_frame, custom_num_frames)

        # ── EXTRA: clipped audio of the same region ──────────────────────
        clipped_audio = None
        if fps and fps > 0:
            start_time = start_frame / fps
            clipped_audio = clip_audio(video_path, start_time, clip_duration)

        return (frames, frames_reversed, first_frame, last_frame,
                filename_prefix, width, height, fps, frames_at_fps,
                float(target_fps), clipped_audio)


# ─── API routes (own namespace — does NOT collide with the original
#     comfyui-video-frame-extractor package if that is still installed) ─────
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/enndee_video_frame_extractor/info")
    async def video_info_handler(request):
        video_name = request.rel_url.query.get("video", "")
        if not video_name:
            return web.json_response({"error": "no video specified"}, status=400)
        try:
            video_path = folder_paths.get_annotated_filepath(video_name)
            info = get_video_info(video_path)
            return web.json_response(info)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/enndee_video_frame_extractor/thumbnail")
    async def video_thumbnail_handler(request):
        """Return a single JPEG frame at the requested position."""
        video_name  = request.rel_url.query.get("video", "")
        frame_index = int(request.rel_url.query.get("frame", 0))
        if not video_name:
            return web.Response(status=400)
        try:
            video_path = folder_paths.get_annotated_filepath(video_name)
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return web.Response(status=404)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((640, 360))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            body = buf.getvalue()
            buf.close()
            return web.Response(body=body, content_type="image/jpeg")
        except Exception as exc:
            return web.Response(status=500, text=str(exc))

except Exception:
    pass


# ─── Exports ─────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Enndee_VideoFrameExtractorWithAudio": Enndee_VideoFrameExtractorWithAudio,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_VideoFrameExtractorWithAudio": "Video Frame Extractor + Audio (Enndee)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
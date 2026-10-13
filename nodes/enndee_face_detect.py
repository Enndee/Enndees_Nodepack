"""Face detection for the face-detail orbit - SAM3 behind one small, optional API.

Why this module exists
----------------------
A 180 deg turntable of a full-body figurine shows the face at roughly **0.4-0.8 % of
the frame** (measured on the Lara 2x180 capture: a 127x211 px box in a 2464x3232
frame). The splat therefore has almost no pixels to learn the face from, and a
head orbit - a short zoom onto the face during the top band - is the only way to
buy more. That orbit is only worth flying when there *is* a face, so the capture
needs a gate: "look for a face, and if one is there, fly the head orbit".

SAM3 is the only open-vocabulary detector in this install that can answer "where is
the face?" from a text prompt alone (SAM2 is prompt-based and needs a point or a box
to start from, which is exactly what we do not have). It ships in the
``comfyui-rmbg`` pack as ``sam3/`` plus ``models/sam3/sam3.pt``.

Measured behaviour on the Lara figurine (``_sam3_face_probe``, 175 frames):

======================  =========  ==========  ==================
prompt                  boxes      score       box area
======================  =========  ==========  ==================
``face`` / ``the face`` 1          0.90-0.92   0.34-0.57 % frame
``head``                1          0.80-0.90   0.80-1.61 % frame
``person``              1          0.75-0.87   15-34 % frame
======================  =========  ==========  ==================

``face`` and ``head`` both track the same feature (the ``head`` box is the larger
one - it includes the hair), and both stay at exactly one detection on every
front-facing frame. Turning away from the camera the detector starts inventing
low-confidence boxes elsewhere (0.51-0.73, near the frame edge), which is why the
default threshold here is **0.75** rather than SAM3's own 0.5: at 0.75 the real face
survives and the invented boxes are gone.

Everything is optional on purpose
---------------------------------
ComfyUI imports every custom node at startup, so this module must stay importable
when ``comfyui-rmbg`` is missing, when the checkpoint is absent, or when the box is
a CPU-only machine. The heavy import and the model build happen *inside*
:func:`load_processor`, and every failure path returns "no faces found" plus a
report string that says why - never an exception. Callers can then treat "no face"
and "no detector" the same way, which is the safe direction: no gate signal means no
head orbit, so a broken detector degrades to today's behaviour instead of breaking
the run.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

_TAG = "[Enndee/FaceDetect]"

#: The prompt the face gate uses unless the user says otherwise.
FACE_PROMPT = "face"

#: Confidence floor. SAM3's own default is 0.5, which on this capture also accepts
#: low-score boxes invented on frames that turned away from the face (0.51-0.73).
#: The real face scores 0.80-0.92, so 0.75 keeps it and drops the inventions.
FACE_THRESHOLD = 0.75

#: ComfyUI's model folder for the checkpoint (``models/sam3/sam3.pt``).
SAM3_MODEL_DIR = "sam3"
SAM3_CHECKPOINT = "sam3.pt"


def _package_candidates() -> List[str]:
    """Every plausible ``comfyui-rmbg`` root (the parent that owns ``sam3/``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes = os.path.dirname(os.path.dirname(here))       # .../ComfyUI/custom_nodes
    roots = [
        os.path.join(custom_nodes, "comfyui-rmbg"),
        os.path.join(custom_nodes, "ComfyUI-NunchakuFluxLoraStacker"),
    ]
    return [path for path in roots if os.path.isdir(path)]


def sam3_available() -> Tuple[bool, str]:
    """(usable, reason) - whether the detector could be imported at all."""
    for root in _package_candidates():
        if os.path.isdir(os.path.join(root, "sam3")):
            return True, f"sam3 package found at {root}"
    return False, ("no 'sam3' package found - install/enable comfyui-rmbg "
                   "(it ships sam3/ and models/sam3/sam3.pt)")


def _checkpoint_path() -> Optional[str]:
    """Absolute path of ``sam3.pt``, or None when it is nowhere to be found."""
    try:
        import folder_paths
        found = folder_paths.get_full_path(SAM3_MODEL_DIR, SAM3_CHECKPOINT)
        if found and os.path.isfile(found):
            return found
    except Exception:
        pass
    for root in _package_candidates():
        candidate = os.path.join(root, "models", SAM3_MODEL_DIR, SAM3_CHECKPOINT)
        if os.path.isfile(candidate):
            return candidate
    here = os.path.dirname(os.path.abspath(__file__))
    models = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))),
                          "models", SAM3_MODEL_DIR, SAM3_CHECKPOINT)
    return models if os.path.isfile(models) else None


_PROCESSOR_CACHE: Dict[str, object] = {}


def load_processor(device: str = "cuda"):
    """(processor, device_str) - the cached SAM3 image processor.

    Raises on any failure; :func:`detect_faces` is the forgiving entry point.
    """
    for root in _package_candidates():
        if root not in sys.path:
            sys.path.insert(0, root)
        inner = os.path.join(root, "sam3")
        if os.path.isdir(inner) and inner not in sys.path:
            sys.path.insert(0, inner)

    key = str(device)
    if key in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[key], key

    from sam3.model_builder import build_sam3_image_model            # noqa: PLC0415
    from sam3.model.sam3_image_processor import Sam3Processor         # noqa: PLC0415

    root = next((r for r in _package_candidates()
                 if os.path.isdir(os.path.join(r, "sam3"))), None)
    if root is None:
        raise RuntimeError("sam3 package not found")
    bpe = os.path.join(root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.isfile(bpe):
        raise RuntimeError(f"SAM3 assets missing: {bpe}")
    checkpoint = _checkpoint_path()
    if checkpoint is None:
        raise RuntimeError(f"{SAM3_CHECKPOINT} not found under models/{SAM3_MODEL_DIR}")

    model = build_sam3_image_model(
        bpe_path=bpe,
        device=key,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model, device=key)
    _PROCESSOR_CACHE[key] = processor
    return processor, key


def unload_processor() -> None:
    """Drop the cached processor and its VRAM (the node's unload widget)."""
    _PROCESSOR_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_device(choice: str = "Auto") -> str:
    """'cuda' when it is wanted and present, else 'cpu' - never raises."""
    choice = str(choice or "Auto").strip().lower()
    if choice == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _faces_from_state(state, width: int, height: int, threshold: float) -> List[Dict]:
    """Turn one SAM3 prompt state into normalised face records, largest first."""
    import torch                                                   # noqa: PLC0415

    boxes = state.get("boxes")
    scores = state.get("scores")
    if boxes is None or len(boxes) == 0:
        return []
    boxes = boxes.detach().float().cpu().reshape(-1, 4)
    scores = (scores.detach().float().cpu().reshape(-1) if scores is not None
              else torch.ones(len(boxes)))
    faces: List[Dict] = []
    for index in range(len(boxes)):
        score = float(scores[index])
        if score < float(threshold):
            continue
        x0, y0, x1, y1 = (float(value) for value in boxes[index])
        # SAM3 returns xyxy in the *original* image pixels; rescale defensively in
        # case a build ever hands back the 1008-normalised frame instead.
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        width_px, height_px = x1 - x0, y1 - y0
        if width_px <= 0 or height_px <= 0:
            continue
        faces.append({
            "x0": x0 / width, "y0": y0 / height, "x1": x1 / width, "y1": y1 / height,
            "cx": (x0 + x1) / (2.0 * width), "cy": (y0 + y1) / (2.0 * height),
            "width": width_px / width, "height": height_px / height,
            "width_px": width_px, "height_px": height_px,
            "area": (width_px * height_px) / (width * height),
            "score": score,
        })
    faces.sort(key=lambda face: face["area"], reverse=True)
    return faces


def detect_faces(image, prompt: str = FACE_PROMPT, threshold: float = FACE_THRESHOLD,
                 device: str = "Auto", max_faces: int = 8,
                 verbose: bool = True) -> Tuple[List[Dict], str]:
    """(faces, report) for one ComfyUI IMAGE batch - never raises.

    `image` is a ``[B, H, W, C]`` float tensor (ComfyUI's IMAGE). Each returned face
    is a normalised record: ``x0/y0/x1/y1`` and ``cx/cy`` in 0..1, ``area`` as the
    share of the frame, plus the pixel sizes and the SAM3 score. The list is sorted
    largest first, so ``faces[0]`` is "the biggest face" the gate should use.

    An empty list means *either* "no face" or "no detector"; `report` says which,
    and callers are expected to treat both the same way (no gate -> no head orbit).
    """
    lines: List[str] = []
    try:
        import numpy as np
        from PIL import Image as PILImage
    except Exception as exc:                                        # pragma: no cover
        return [], f"numpy/Pillow unavailable: {exc}"

    tensor = image
    if hasattr(tensor, "ndim") and getattr(tensor, "ndim", 0) == 4:
        tensor = tensor[0]
    try:
        array = (tensor.detach().cpu().numpy() if hasattr(tensor, "detach")
                 else np.asarray(tensor, dtype=np.float32))
        # Anything that is not a real numeric image (a dict, a plain object, an empty
        # tensor) must come back as "no faces + why", never as a TypeError out of
        # numpy's clip - the callers rely on this function not raising.
        array = np.asarray(array, dtype=np.float32)
    except Exception as exc:
        return [], f"image could not be read: {type(exc).__name__}: {exc}"
    if array.ndim < 2 or array.size == 0:
        return [], f"image could not be read: shape {tuple(array.shape)}"
    array = np.clip(array, 0.0, 1.0)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    height, width = int(array.shape[0]), int(array.shape[1])
    # No `mode=` here: Pillow 13 removes it (it warns from Pillow 12 on), and a 3-channel
    # uint8 array already infers RGB.
    pil = PILImage.fromarray((array * 255.0).astype(np.uint8))

    ok, reason = sam3_available()
    if not ok:
        return [], f"no face check: {reason}"
    try:
        processor, device_str = load_processor(resolve_device(device))
    except Exception as exc:
        return [], f"no face check: SAM3 could not load ({type(exc).__name__}: {exc})"

    text = str(prompt or FACE_PROMPT).strip() or FACE_PROMPT
    try:
        state = processor.set_image(pil)
        processor.reset_all_prompts(state)
        processor.set_confidence_threshold(float(threshold), state)
        state = processor.set_text_prompt(text, state)
        faces = _faces_from_state(state, width, height, threshold)
    except Exception as exc:
        return [], f"no face check: SAM3 failed on this image ({type(exc).__name__}: {exc})"

    faces = faces[:max(1, int(max_faces))] if max_faces else faces
    lines.append(f"prompt {text!r} on {width}x{height} ({device_str}), "
                 f"threshold {float(threshold):.2f}: {len(faces)} face(s)")
    for index, face in enumerate(faces):
        lines.append(f"  #{index} area={face['area'] * 100.0:.2f}% frame "
                     f"box={face['width_px']:.0f}x{face['height_px']:.0f}px "
                     f"centre=({face['cx'] * 100.0:.0f}%, {face['cy'] * 100.0:.0f}%) "
                     f"score={face['score']:.3f}")
    if not faces:
        lines.append("  (nothing above the threshold - the gate stays closed)")
    report = "\n".join(lines)
    if verbose:
        print(f"{_TAG} {report}", flush=True)
    return faces, report


def largest_face(faces: List[Dict]) -> Optional[Dict]:
    """The biggest detection, or None - the one the face orbit should follow."""
    return faces[0] if faces else None


def face_bbox_string(face: Optional[Dict]) -> str:
    """``"cx,cy,w,h"`` (normalised) for a face record, or ``""`` for none.

    The string form is what travels between nodes: it keeps the face orbit's inputs
    plain STRINGs, so a saved workflow needs no new ComfyUI type registered.
    """
    if not face:
        return ""
    return ",".join(f"{float(face[key]):.6f}" for key in ("cx", "cy", "width", "height"))


def parse_face_bbox(text: Optional[str]) -> Optional[Dict]:
    """Inverse of :func:`face_bbox_string` - a normalised face record, or None."""
    if not text:
        return None
    try:
        parts = [float(value) for value in str(text).replace(";", ",").split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    cx, cy, width, height = parts
    if width <= 0 or height <= 0:
        return None
    return {"cx": cx, "cy": cy, "width": width, "height": height,
            "x0": cx - width / 2.0, "y0": cy - height / 2.0,
            "x1": cx + width / 2.0, "y1": cy + height / 2.0,
            "area": width * height, "score": float("nan")}


def face_frame_indices(areas: List[float], multiple: float = 4.0) -> List[int]:
    """Which frames of a capture are face close-ups, from their face areas.

    A face orbit shows the face far larger than the body frames do - measured on this
    scene, ~0.5 % of the frame on the body frames against ~20 % in the orbit, a factor
    of 40. So the close-ups are an *outlier* group, and the test here is a
    scale-free one: **at least `multiple` x the median of the frames that show a face
    at all**.

    That median rule is the one that survives contact with real data. A ratio-of-the-
    maximum cut ("within 50 % of the biggest face") looks reasonable but is useless on
    a plain 180 deg capture: the face area then varies smoothly from 0.30 % to 0.60 %
    across the front arc, so half the maximum is 0.30 % and 130 of 175 frames qualify -
    verified on the Lara set. The median rule returns **nothing** there, which is the
    honest answer: no orbit was flown, so no frame is a close-up. Feed the camera
    node's `face_frames` output instead when you have it; it is authoritative and
    skips this heuristic entirely.

    Frames with no detection at all (0.0) are never selected and never set the median.
    """
    shown = [float(area) for area in areas if float(area) > 0.0]
    if not shown:
        return []
    shown.sort()
    median = shown[len(shown) // 2]
    if median <= 0.0:
        return []
    cut = float(multiple) * median
    return [index for index, area in enumerate(areas) if float(area) >= cut]


class Enndee_FaceDetect:
    """Where is the face? - the SAM3 gate in front of the face-detail orbit.

    Standalone node, so the gate can be inspected (and its numbers logged) without
    running a whole capture. The Meridian camera node runs the same code internally
    when its own face orbit is armed; this node is for checking what it will see.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": FACE_PROMPT,
                    "tooltip": "Open-vocabulary prompt SAM3 looks for. 'face' (or "
                               "'the face') tracks the figurine's face and stays at "
                               "one detection per front-facing frame; 'head' returns "
                               "a larger box that includes the hair.",
                }),
                "threshold": ("FLOAT", {
                    "default": FACE_THRESHOLD, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Confidence floor. SAM3's own default (0.5) also "
                               "accepts boxes invented on frames that turned away "
                               "from the face (0.51-0.73); the real face scores "
                               "0.80-0.92, so 0.75 keeps it and drops the rest.",
                }),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
                "max_faces": ("INT", {
                    "default": 8, "min": 1, "max": 64,
                    "tooltip": "How many detections to report (largest first). The "
                               "face orbit always follows #0.",
                }),
            },
            "optional": {
                "unload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free SAM3 from VRAM after this node - useful when a "
                               "video model runs right after it.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("face_bbox", "face_found", "face_area_percent", "faces_json", "report")
    FUNCTION = "detect"
    CATEGORY = "Enndee/camera"
    DESCRIPTION = ("Runs SAM3 with a text prompt (default 'face') and reports the "
                   "detections. face_bbox is 'cx,cy,w,h' normalised - empty when "
                   "nothing was found, which is what the face orbit's gate reads.")

    def detect(self, image, prompt, threshold, device, max_faces, unload_model=False):
        import json
        faces, report = detect_faces(image, prompt=prompt, threshold=threshold,
                                     device=device, max_faces=max_faces)
        best = largest_face(faces)
        if unload_model:
            unload_processor()
        print(f"{_TAG} {report}", flush=True)
        return (face_bbox_string(best), bool(best),
                float(best["area"] * 100.0) if best else 0.0,
                json.dumps(faces), report)


NODE_CLASS_MAPPINGS = {
    "Enndee_FaceDetect": Enndee_FaceDetect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_FaceDetect": "Enndee Face Detect (SAM3 gate)",
}


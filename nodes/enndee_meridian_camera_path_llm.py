"""Meridian Camera Path LLM (Enndee): a local vision LLM authors the camera path.

A ComfyUI node that hands a *local* vision LLM (LM Studio via the OpenAI-compatible endpoint, or
Ollama) the still, its depth map and a short instruction, and asks it to author an orbit-based
camera trajectory. The node renders that plan into a valid `MERIDIAN_CAMERA_PATH` signal using the
real geometry (the subject's pivot + framing orbit radius from the depth profile, in the same
median-depth units the auto-camera emits), so it plugs straight into Meridian Geometry.

The LLM does the *creative* work - reading the scene and translating "rotate 180 around the subject,
lift up a bit, rotate back" into a trajectory - while the node does the *precise* 3D placement. The
LLM never emits raw xyz; it emits orbit-space keyframes (azimuth / elevation / distance) that are
robust to scene scale and always keep the subject framed.

Provider + image handling follow the MiniMax-H3-Promptor (Enndee) conventions.
"""

import base64
import io
import json
import math
import re

import numpy as np
import requests
import torch

from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS, CAMERA_SIGNAL_TYPE
from enndee_meridian_auto_camera import _place, probe_surface, subject_framing

PROVIDERS = ["lmstudio", "ollama"]
REQUEST_TIMEOUT = 300

# ---------------------------------------------------------------------------
# The system prompt - everything static the LLM needs. The per-run bits (the
# instruction, frame count, depth convention, the images) ride in the user turn.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert virtual cinematographer. You author a 3D camera path for a single
still image that will be turned into a short camera-move video (an image-to-video "Meridian"
reconstruction). You are given the still, its depth map, and a short instruction. You output ONE JSON
object describing an orbit-based camera trajectory that flies around the subject of the still.

# WHAT YOU RECEIVE
1. The still image (RGB).
2. The depth map of the same still (grayscale). The message states whether brighter means closer or
   farther. Use it to understand the scene's 3D layout: where the subject is, how deep the scene is,
   and what a rear or side view would reveal.
3. A text message with: `instruction` (the requested move in plain language), `frames` (the exact
   number of output frames), and `fill_percent` (how large the subject should appear).

# THE CAMERA MODEL (orbit space)
The camera ALWAYS aims at the subject's pivot point (its volumetric center), so the subject stays in
the middle of the frame. There is NO roll: the horizon stays as level as the aim allows and the
camera never flips over the top (the renderer enforces this). You only choose where the camera sits on
a sphere around the subject. For each keyframe give three numbers:

- `azimuth` (degrees): orbit angle around the subject, measured from the source still's viewpoint.
  0 = the front (the original camera position). +90 = the subject's RIGHT side. 180 = directly
  behind. 360 = a full loop back to the front. Positive orbits toward the subject's right; negative
  orbits left. You may exceed 360 (e.g. 540 = 1.5 loops) or go negative.
- `elevation` (degrees): camera height around the subject. 0 = level with the pivot. Positive = above
  the subject looking down; negative = below looking up. Clamp to [-70, 70] - beyond that the look
  direction goes vertical and the framing becomes unstable.
- `distance` (multiple of the framing orbit radius): 1.0 = the distance that frames the subject at
  `fill_percent`. 0.5 = push in to half (tighter). 2.0 = pull back to double (wider). Keep within
  [0.5, 3.0]; prefer near 1.0 and only push/pull when asked or to keep the subject framed.

# OUTPUT FORMAT (STRICT)
Output ONLY a single valid JSON object. No prose before or after, no markdown fences, no comments, no
trailing commas. Schema:
{
  "notes": "<= 2 sentences of reasoning about the scene and the move",
  "frames": <int, exactly the requested frame count>,
  "keys": [
    {"t": <int frame index>, "azimuth": <float>, "elevation": <float>, "distance": <float>},
    ...
  ]
}

RULES FOR `keys`:
- At least 3 keys. First key `t` = 0. Last key `t` = `frames` - 1. `t` values are strictly increasing
  integers.
- Put a key wherever the motion changes: start, the beginning of a lift, an apex, a direction change,
  and the end. The renderer smooths between keys with a spline, so 4-12 well-placed keys are enough.
- Keep motion continuous: do not make one huge jump between adjacent keys unless the instruction asks
  for a hard cut. For a constant-speed orbit, space keys evenly in `t`; for an ease-in/out move, put
  keys a little closer to the start and end.

# HOW TO READ THE SCENE (use the depth map)
- The subject is usually the prominent near object; the depth map shows how far everything is.
- Prefer orbits that REVEAL dimensionality (parallax): going around a subject with depth shows new
  sides and reads as a real camera move.
- If the background is flat or distant (a wall, sky, an empty field), a full rear orbit to azimuth 180
  reveals little and can look empty - prefer shallow orbits or stay in the front hemisphere.
- Keep the subject in frame: at high elevation the subject can drift or shrink; a small `distance`
  pullback helps. Never push the camera through the subject (distance too small) or into the background.

# TRANSLATING THE INSTRUCTION
- "rotate / orbit N degrees around the subject" -> azimuth changes by N (sign = direction; "to the
  right" = positive, "to the left" = negative).
- "lift up / crane up / raise" -> elevation increases (e.g. 0 -> 15). "a bit" = 10-20 deg.
- "go over the top" -> elevation toward +70 (never beyond).
- "push in / dolly in / closer" -> distance decreases (e.g. 1.0 -> 0.7).
- "pull back / wider" -> distance increases (e.g. 1.0 -> 1.4).
- "rotate back toward the starting side / return" -> azimuth heads back toward 0, or continues to 360
  to end facing the front again.

# WORKED EXAMPLE
Instruction: "rotate 180 around the subject, lift up a bit, then rotate back toward the starting side."
(frames = 141). Reasoning: start front, orbit right to the back while lifting a little, then keep
orbiting around to 360 so the clip ends facing the front again.
{
  "notes": "Front start; a 180 right-orbit to the back with a gentle lift, then continue to 360 to end on the front.",
  "frames": 141,
  "keys": [
    {"t": 0,   "azimuth": 0.0,   "elevation": 0.0,  "distance": 1.0},
    {"t": 70,  "azimuth": 180.0, "elevation": 18.0, "distance": 1.08},
    {"t": 105, "azimuth": 270.0, "elevation": 18.0, "distance": 1.08},
    {"t": 140, "azimuth": 360.0, "elevation": 0.0,  "distance": 1.0}
  ]
}

Now output only the JSON object for the request you are given."""
# ---------------------------------------------------------------------------
# Images -> base64 (JPEG) for the vision API
# ---------------------------------------------------------------------------

def _tensor_to_b64(tensor) -> str:
    """First frame of a ComfyUI IMAGE [B,H,W,C]/[H,W,C] float 0..1 -> base64 JPEG string."""
    from PIL import Image

    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    image = Image.fromarray(array[..., :3])
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _depth_to_array(depth_image, invert: bool):
    """ComfyUI IMAGE depth map -> a [1, H, W] float torch tensor (larger = farther)."""
    array = depth_image.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    grid = array.mean(axis=-1) if array.ndim == 3 else array
    grid = grid.astype(np.float32)
    if invert:  # source map had brighter = farther; flip so larger = farther
        grid = grid.max() - grid
    return torch.from_numpy(grid).unsqueeze(0)


# ---------------------------------------------------------------------------
# Local LLM calls - LM Studio (OpenAI-compatible) and Ollama
# ---------------------------------------------------------------------------

LMSTUDIO_DEFAULT = "http://localhost:1234/v1"
OLLAMA_DEFAULT = "http://localhost:11434"
_ENDPOINT_SUFFIXES = ("/chat/completions", "/api/chat", "/api/generate", "/api/tags", "/models")


def _normalize_base_url(provider, base_url):
    """Provider-aware server base URL: strips a pasted endpoint and fixes a cross-provider default.

    The two local servers have different shapes - LM Studio is OpenAI-compatible at
    http://localhost:1234/v1, Ollama's native API is http://localhost:11434 with no /v1 - so a URL
    left on the *other* provider's default (the common mistake: provider 'ollama' with the LM Studio
    URL, which produced a bogus /v1/api/chat on the wrong port) is corrected to this provider's own
    default.
    """
    url = (base_url or "").strip().rstrip("/")
    for suffix in _ENDPOINT_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    if provider == "ollama":
        if url.endswith("/v1"):
            url = url[:-3].rstrip("/")
        if not url or url in (LMSTUDIO_DEFAULT, "http://localhost:1234"):
            url = OLLAMA_DEFAULT
    else:
        if not url or url == OLLAMA_DEFAULT:
            url = LMSTUDIO_DEFAULT
    return url


def _reachable(url, timeout=2.0):
    """True if something answers at `url` (any non-5xx response counts)."""
    try:
        return requests.get(url, timeout=timeout).status_code < 500
    except Exception:  # noqa: BLE001
        return False


def _connection_hint(provider, url):
    """A concrete, actionable message when the local server cannot be reached."""
    lm_up = _reachable(f"{LMSTUDIO_DEFAULT}/models")
    ol_up = _reachable(f"{OLLAMA_DEFAULT}/api/tags")
    lines = [f"Could not reach the local LLM server at {url} (connection refused)."]
    if lm_up:
        lines.append("LM Studio IS reachable at http://localhost:1234/v1 - set provider=lmstudio "
                     "and base_url=http://localhost:1234/v1.")
    if ol_up:
        lines.append("Ollama IS reachable at http://localhost:11434 - set provider=ollama "
                     "and base_url=http://localhost:11434.")
    if not lm_up and not ol_up:
        lines.append("Nothing answered on port 1234 (LM Studio) or 11434 (Ollama). Start the server: "
                     "LM Studio -> Developer/Server tab -> Start Server; Ollama -> run 'ollama serve'.")
    if provider == "ollama" and "1234" in url:
        lines.append("The URL uses port 1234 (LM Studio) but the provider is Ollama - did you mean "
                     "provider=lmstudio?")
    if provider == "lmstudio" and "11434" in url:
        lines.append("The URL uses port 11434 (Ollama) but the provider is lmstudio - did you mean "
                     "provider=ollama?")
    return " ".join(lines)


def _call_llm(provider, base_url, api_key, model, system_prompt, user_message, base64_images,
              temperature, max_tokens):
    """Send a vision chat request to a local server. Returns (content, error)."""
    if not model:
        return "", ("No model name set. Load a vision model in your server and put its name here "
                    "(e.g. qwen2.5-vl, llava, gemma3).")
    base_url = _normalize_base_url(provider, base_url)

    if provider == "ollama":
        url = f"{base_url}/api/chat"
        user_payload = {"role": "user", "content": user_message}
        if base64_images:
            user_payload["images"] = base64_images
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, user_payload],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        headers = {}
    else:  # lmstudio -> OpenAI-compatible /chat/completions
        url = f"{base_url}/chat/completions"
        user_content = [{"type": "text", "text": user_message}]
        for img in base64_images:
            user_content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return "", _connection_hint(provider, url)
    except requests.exceptions.Timeout:
        return "", (f"The server at {url} did not answer within {REQUEST_TIMEOUT}s "
                    "(a large model may still be loading - try again).")
    except Exception as exc:  # noqa: BLE001
        return "", f"Request to {url} failed: {exc}"

    if not response.ok:
        return "", f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
    try:
        data = response.json()
    except ValueError:
        return "", f"{url} returned a non-JSON response: {response.text[:200]}"
    if provider == "ollama":
        content = data.get("message", {}).get("content", "")
    else:
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
    if not content:
        return "", (f"The model returned an empty reply from {url}. Is '{model}' a vision model "
                    "that is currently loaded?")
    return content, None
# ---------------------------------------------------------------------------
# Plan parsing + rendering to a MERIDIAN_CAMERA_PATH signal
# ---------------------------------------------------------------------------

def _extract_json(text):
    """Pull the first JSON object out of an LLM reply (tolerates ```json fences and prose)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == "{":
                depth += 1
            elif candidate[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _parse_plan(text, frames):
    """LLM reply -> a sanitized keyframe plan. Returns (plan, error)."""
    raw = _extract_json(text)
    if not isinstance(raw, dict):
        return None, "No JSON object found in the model output."
    keys = raw.get("keys")
    if not isinstance(keys, list) or len(keys) < 2:
        return None, "Plan has no usable 'keys' array (need at least 2)."

    cleaned = []
    for key in keys:
        if not isinstance(key, dict):
            continue
        try:
            t = int(round(float(key.get("t"))))
            azimuth = float(key.get("azimuth", 0.0))
            elevation = max(-70.0, min(70.0, float(key.get("elevation", 0.0))))
            distance = max(0.5, min(3.0, float(key.get("distance", 1.0))))
        except (TypeError, ValueError):
            continue
        cleaned.append({"t": t, "azimuth": azimuth, "elevation": elevation, "distance": distance})
    if len(cleaned) < 2:
        return None, "Plan keys could not be parsed into numbers."

    cleaned.sort(key=lambda k: k["t"])
    last = frames - 1
    cleaned[0]["t"] = 0
    cleaned[-1]["t"] = max(last, len(cleaned) - 1)
    dedup = []
    for key in cleaned:
        if dedup and key["t"] <= dedup[-1]["t"]:
            key["t"] = dedup[-1]["t"] + 1
        if key["t"] > last:
            key["t"] = last
        if dedup and key["t"] <= dedup[-1]["t"]:
            continue
        dedup.append(key)
    if dedup[0]["t"] != 0:
        dedup.insert(0, dict(dedup[0], t=0))
    if dedup[-1]["t"] != last:
        dedup.append(dict(dedup[-1], t=last))

    return {"notes": str(raw.get("notes", ""))[:400], "frames": frames, "keys": dedup}, None


def _render_plan(plan, pivot, distance, depth_unit):
    """Orbit-space plan -> MERIDIAN_CAMERA_PATH keys (median-depth units), aiming at the pivot."""
    keys = []
    look = [round(float(value) / depth_unit, 6) for value in pivot]
    for key in plan["keys"]:
        radius = float(distance) * float(key["distance"])
        position = _place(pivot, radius, float(key["azimuth"]), float(key["elevation"]))
        keys.append({
            "pos": [round(float(value) / depth_unit, 6) for value in position],
            "look": look,
            "src": int(key["t"]),
            "t": int(key["t"]),
        })
    return keys
# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class Enndee_MeridianCameraPathLLM:
    """Author a Meridian camera path with a local vision LLM (LM Studio / Ollama)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "rotate 180 around the subject, lift up a bit, and rotate back toward the starting side",
                    "tooltip": "The camera move in plain language. The vision LLM translates it into an orbit trajectory.",
                }),
                "frames": (CAMERA_FRAME_OPTIONS, {
                    "default": "141",
                    "tooltip": "Output frame count (must match the Meridian Geometry / sampler).",
                }),
                "provider": (PROVIDERS, {
                    "default": "lmstudio",
                    "tooltip": "lmstudio = OpenAI-compatible server (LM Studio, default http://localhost:1234/v1). ollama = Ollama (http://localhost:11434).",
                }),
                "base_url": ("STRING", {
                    "default": "http://localhost:1234/v1",
                    "tooltip": "Server base URL. LM Studio: http://localhost:1234/v1 | Ollama: http://localhost:11434. It is auto-corrected to the selected provider's default if left on the other one's.",
                }),
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "Loaded vision model name (e.g. qwen2.5-vl-7b-instruct, llava, gemma3).",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional bearer token (LM Studio usually ignores it; Ollama does not use it).",
                }),
                "fill_percent": ("FLOAT", {
                    "default": 40.0, "min": 5.0, "max": 95.0, "step": 1.0,
                    "tooltip": "How large the subject is framed at distance 1.0. The plan's 'distance' multiplies this framing radius.",
                }),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The still (RGB). Sent to the LLM and used for geometry."}),
                "depth": ("IMAGE", {"tooltip": "The depth map of the still (grayscale). Sent to the LLM and used to place the camera."}),
                "subject_mask": ("MASK", {"tooltip": "Optional subject mask (white = subject) for a more accurate pivot."}),
                "depth_convention": (["brighter = closer", "brighter = farther"], {
                    "default": "brighter = closer",
                    "tooltip": "How to read the depth map image. Affects both the geometry and what the LLM is told.",
                }),
            },
        }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("custom_camera", "plan", "raw", "system_prompt")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎬 Meridian"

    def generate(self, instruction, frames, provider, base_url, model, api_key, fill_percent,
                 temperature, max_tokens, image=None, depth=None, subject_mask=None,
                 depth_convention="brighter = closer"):
        frames = int(frames)
        # ---- geometry from the depth profile (pivot + framing orbit radius + median-depth unit) ----
        pivot = [0.0, 0.0, 1.0]
        distance = 1.0
        depth_unit = 1.0
        geom_error = None
        if depth is not None and image is not None:
            try:
                # depth arrays want larger = farther; invert when the map is brighter = closer
                invert = depth_convention.startswith("brighter = closer")
                depth_array = _depth_to_array(depth, invert)
                surface = probe_surface(image, subject_mask=subject_mask,
                                        depth_fn=lambda _ref: depth_array)
                try:
                    distance, pivot, _metrics = subject_framing(surface, float(fill_percent))
                except Exception:  # fall back to the raw subject/scene profile
                    pivot = list(surface.get("pivot") or surface.get("scene_pivot") or pivot)
                    distance = float(surface.get("subject_radius") or surface.get("scene_radius") or 1.0)
                points = surface.get("scene_points")
                if points is not None and getattr(points, "numel", lambda: 0)():
                    med = float(points[:, 2].median())
                    if med > 1e-6:
                        depth_unit = med
            except Exception as exc:  # noqa: BLE001
                geom_error = f"geometry: {exc}"
        else:
            geom_error = "geometry: connect BOTH image and depth for a real pivot (using defaults)."

        # ---- images for the vision LLM ----
        base64_images = []
        if image is not None:
            base64_images.append(_tensor_to_b64(image))
        if depth is not None:
            base64_images.append(_tensor_to_b64(depth))

        convention_note = ("brighter = closer to the camera" if depth_convention.startswith("brighter = closer")
                           else "brighter = farther from the camera")
        user_message = (
            f"Request: {instruction}\n\n"
            f"frames: {frames}\n"
            f"fill_percent: {fill_percent:.0f}\n"
            f"Depth map convention: {convention_note}.\n"
            f"Image 1 is the still (RGB); Image 2 is its depth map.\n\n"
            f"Output the single JSON camera plan now."
        )
        # ---- call the local vision LLM ----
        content, error = _call_llm(provider, base_url, api_key, model, SYSTEM_PROMPT,
                                   user_message, base64_images, temperature, max_tokens)
        if error:
            message = f"[Meridian Camera Path LLM] {error}"
            return (message, "{}", "", SYSTEM_PROMPT)

        # ---- parse the plan and render it to a MERIDIAN_CAMERA_PATH signal ----
        plan, plan_error = _parse_plan(content, frames)
        if plan_error:
            message = f"[Meridian Camera Path LLM] {plan_error}"
            return (message, "{}", content, SYSTEM_PROMPT)

        keys = _render_plan(plan, pivot, distance, depth_unit)
        description = (f"LLM-authored camera path ({provider}/{model or 'default'}): "
                       f"{len(keys)} keys over {frames} frames.")
        if geom_error:
            description += f" NOTE: {geom_error}"
        signal = {
            "name": "LLM camera path",
            "description": description,
            "frames": frames,
            "notes": plan.get("notes", ""),
            "depth_model": f"vision-llm:{provider}/{model or 'default'}",
            "path": keys,
        }
        return (json.dumps(signal), json.dumps(plan, indent=2), content, SYSTEM_PROMPT)


NODE_CLASS_MAPPINGS = {
    "Enndee_MeridianCameraPathLLM": Enndee_MeridianCameraPathLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_MeridianCameraPathLLM": "Meridian Camera Path LLM (Enndee)",
}

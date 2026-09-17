"""
ComfyUI-Minimax-H3-Promptor
This custom node for ComfyUI provides automation suite for generating MiniMax H3 prompts.

This integration script follows GPL-3.0 License.
When using or modifying this code, please respect both the original model licenses
and this integration's license terms.

Source: https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor
"""

from .config_manager import get_config_manager
from .task_detector import TASK_TYPE_OPTIONS, TaskDetector
from .prompt_builder import PromptBuilder
from .post_processor import PostProcessor
from .provider_openai import OpenAIProvider
from .provider_ollama import OllamaProvider
from .provider_gemini import GeminiProvider
from .provider_claude import ClaudeProvider
from .utils import log_info, log_error, tensor_to_base64


# Provider options
PROVIDERS = ["openai", "ollama", "gemini", "claude"]


def _get_ollama_model_options() -> list:
    """Fetch the registered Ollama models (like 'ollama list') for the model dropdown."""
    try:
        config = get_config_manager()
        ollama_cfg = config.get_provider_config("ollama") or {}
        api_base = ollama_cfg.get("api_base", "http://localhost:11434")
        default_model = ollama_cfg.get("default_model", "")
        provider = OllamaProvider(api_base=api_base)
        models = provider.list_models()
        # Always include the configured default model if it's not already listed
        if default_model and default_model not in models:
            models.insert(0, default_model)
        return models or ["Select a model..."]
    except Exception:
        return ["Select a model..."]


# Build the model dropdown once at import time (refresh requires ComfyUI restart).
OLLAMA_MODEL_OPTIONS = _get_ollama_model_options()


def _create_provider(provider_name: str, config_manager, api_key_override: str = ""):
    """Create an LLM provider instance from config."""
    provider_config = config_manager.get_provider_config(provider_name)
    if not provider_config:
        raise ValueError(f"Provider '{provider_name}' not configured.")

    api_base = provider_config.get("api_base", "")
    api_key = api_key_override or provider_config.get("api_key", "")
    model = provider_config.get("default_model", "")

    if provider_name == "ollama":
        return OllamaProvider(api_base=api_base, model=model)
    elif provider_name == "gemini":
        return GeminiProvider(api_base=api_base, api_key=api_key, model=model)
    elif provider_name == "claude":
        return ClaudeProvider(api_base=api_base, api_key=api_key, model=model)
    else:
        return OpenAIProvider(api_base=api_base, api_key=api_key, model=model)


class H3_Multimodal_Promptor_Enndee:
    """
    MiniMax H3 Direct Multimodal Prompt Generator

    Unified "direct" node: feeds the reference images directly to a vision LLM
    in a single call, together with the task type, template documentation, and the
    user's creative description. This avoids the information loss that occurs when
    images are first transcribed to text by a separate vision-analyzer node.

    The LLM sees the actual pixels (via base64) and produces the final structured
    H3 prompt directly, using the official <Picture N> / <Subject N> / <Video N> /
    <Audio N> labels and the three-field / six-section output format.
    """

    def __init__(self):
        self.prompt_builder = PromptBuilder()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_type": (TASK_TYPE_OPTIONS, {
                    "default": TASK_TYPE_OPTIONS[0],
                    "tooltip": "Forces the H3 prompt format (Text-to-Video, Image-to-Video, etc.)"
                }),
                "description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your main creative description of the scene.",
                }),
                "duration": ("FLOAT", {
                    "default": 5.0, "min": 4.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Video length in seconds (4-15). Accepts INT or FLOAT.",
                }),
            },
            "optional": {
                "image_ref_1": ("IMAGE", {}),
                "image_ref_2": ("IMAGE", {}),
                "image_ref_3": ("IMAGE", {}),
                "image_ref_4": ("IMAGE", {}),
                "image_ref_5": ("IMAGE", {}),
                "image_ref_6": ("IMAGE", {}),
                "image_ref_7": ("IMAGE", {}),
                "image_ref_8": ("IMAGE", {}),
                "video_ref": ("IMAGE", {"tooltip": "Video batch input. Its extracted keyframes are sent to the model."}),
                "output_language": (["English", "Chinese"], {
                    "default": "English",
                    "tooltip": "The language the MiniMax H3 system will receive the prompt in."
                }),
                "provider": (PROVIDERS, {
                    "default": "openai",
                    "tooltip": "Vision LLM provider to use. Must support image input.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "API key override.",
                }),
                "model_name": (OLLAMA_MODEL_OPTIONS, {
                    "default": OLLAMA_MODEL_OPTIONS[0] if OLLAMA_MODEL_OPTIONS else "",
                    "tooltip": "Model to use. When Ollama is selected, this lists your registered Ollama models (like 'ollama list').",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "top_k": ("INT", {
                    "default": 64, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Ollama top_k sampling (0 = disabled).",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Ollama top_p nucleus sampling.",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ollama min_p minimum probability threshold.",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Ollama repeat penalty.",
                }),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 16384, "step": 256
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "🧪AILab/🎬 MiniMax H3-Promptor"

    def generate_prompt(
        self,
        task_type: str,
        description: str,
        duration: float,
        image_ref_1=None,
        image_ref_2=None,
        image_ref_3=None,
        image_ref_4=None,
        image_ref_5=None,
        image_ref_6=None,
        image_ref_7=None,
        image_ref_8=None,
        video_ref=None,
        output_language: str = "English",
        provider: str = "openai",
        api_key: str = "",
        model_name: str = "",
        temperature: float = 0.6,
        top_k: int = 64,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        max_tokens: int = 4096,
    ):
        """Generate a MiniMax H3 structured prompt directly from images + text."""
        try:
            # 1. Collect connected media -> base64 images + label mapping
            base64_images = []
            media_notes = []          # tells the LLM which image maps to which Picture/Video label
            image_count = 0
            has_video = False
            video_frames = 0

            def _add_image(tensor, index):
                nonlocal image_count
                image_count += 1
                frames = tensor_to_base64(tensor, max_frames=1)
                base64_images.extend(frames)
                media_notes.append(
                    f"Image {image_count} (a single reference image) -> <Picture {image_count}>"
                )

            if image_ref_1 is not None:
                _add_image(image_ref_1, 1)
            if image_ref_2 is not None:
                _add_image(image_ref_2, 2)
            if image_ref_3 is not None:
                _add_image(image_ref_3, 3)
            if image_ref_4 is not None:
                _add_image(image_ref_4, 4)
            if image_ref_5 is not None:
                _add_image(image_ref_5, 5)
            if image_ref_6 is not None:
                _add_image(image_ref_6, 6)
            if image_ref_7 is not None:
                _add_image(image_ref_7, 7)
            if image_ref_8 is not None:
                _add_image(image_ref_8, 8)

            if video_ref is not None:
                has_video = True
                frames = tensor_to_base64(video_ref, max_frames=4)
                video_frames = len(frames)
                base64_images.extend(frames)
                media_notes.append(
                    f"Video (represented by {video_frames} extracted keyframes) -> <Video 1>"
                )

            if not base64_images:
                # No images: fall back to a pure text-to-video style generation.
                log_info("No visual media provided; generating from text only.")
                media_notes.append("No reference images or videos are provided (text-to-video).")

            # 2. Detect task type from connected media + user override
            detected_type = TaskDetector.detect(
                image_count=image_count,
                has_video=has_video,
                user_override=task_type,
            )
            log_info(f"Task type: {TaskDetector.get_task_description(detected_type)}")

            # 3. Build system prompt from templates (system_base + task template)
            system_prompt = self.prompt_builder.build_system_prompt(detected_type)

            # 4. Build the user message that pairs the media labels with the description
            lang_constraint = "English"
            if output_language.lower() == "chinese":
                lang_constraint = "Simplified Chinese (简体中文)"

            media_header = "\n".join(f"- {note}" for note in media_notes) or "- None"

            # Normalize duration to a float and compute approximate frames (24 FPS)
            duration_f = float(duration)
            approx_frames = int(round(duration_f * 24))

            user_message = (
                f"Generate a MiniMax {detected_type} prompt.\n\n"
                f"Reference media mapping (use these labels exactly as shown):\n"
                f"{media_header}\n\n"
                f"Primary Target User Description:\n{description}\n\n"
                f"Constraint: The video will be {duration_f:.2f} seconds long "
                f"(approx. {approx_frames} frames). Pace the [SCENE]/[Shot] descriptions accordingly.\n\n"
                f"CRITICAL LANGUAGE CONSTRAINT:\n"
                f"You MUST write the ENTIRE OUTPUT PROMPT in {lang_constraint}."
            )

            # 5. Get LLM provider
            config_manager = get_config_manager()
            llm = _create_provider(provider, config_manager, api_key)

            model_override = model_name if model_name.strip() else None
            log_info(
                f"Calling {provider} (direct multimodal) | "
                f"model={model_override or llm.model} | "
                f"temp={temperature} | images={len(base64_images)}"
            )

            # Build provider-specific sampling options (used by Ollama)
            extra_options = {
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
            }

            response = llm.chat(
                system_prompt=system_prompt,
                user_message=user_message,
                base64_images=base64_images,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model_override,
                extra_options=extra_options,
            )

            if not response.success:
                error_msg = f"[H3-Multimodal-Promptor Error] {response.error or 'Unknown error'}"
                log_error(response.error or "Unknown error")
                return (error_msg,)

            # 6. Log raw response to console for user debugging
            print(f"\n{'-'*20} RAW MULTIMODAL PROMPTOR OUTPUT {'-'*20}")
            print(response.content)
            print(f"{'-'*60}\n")

            # 7. Post-process
            task_desc = TaskDetector.get_task_description(detected_type)
            cleaned_prompt = PostProcessor.clean(response.content, detected_type, full_task_desc=task_desc)
            log_info(
                f"Prompt generated: {len(cleaned_prompt)} chars | "
                f"model={response.model} | "
                f"tokens={response.usage.get('total_tokens', '?')}"
            )

            return (cleaned_prompt,)

        except Exception as e:
            error_msg = f"[H3-Multimodal-Promptor Error] {str(e)}"
            log_error(str(e))
            return (error_msg,)


NODE_CLASS_MAPPINGS = {
    "H3_Multimodal_Promptor_Enndee": H3_Multimodal_Promptor_Enndee,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_Multimodal_Promptor_Enndee": "MiniMax H3 Direct Promptor (Enndee)",
}
"""Image loader with optional resize, mask extraction and original-size output.

The widget layout mirrors the classic WAS "Load & Resize Image" node (file combo
with upload button, ``resize`` toggle, width/height, repeat, keep_proportion,
divisible_by, mask_channel, background_color). Additionally it offers the
``original_image`` output, the nine resize types of ComfyUI core's
``ResizeImageMaskNode`` (see ``enndee_resize_modes``) and a ``match`` input for
the "match size" type.
"""

import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

import enndee_resize_modes as resize_modes


MAX_RESOLUTION = resize_modes.MAX_RESOLUTION
MASK_CHANNELS = ("alpha", "red", "green", "blue")
_COLOR_CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


def list_input_images():
    """Sorted image files from ComfyUI's input folder for the combo widget."""
    input_directory = folder_paths.get_input_directory()
    if not input_directory or not os.path.isdir(input_directory):
        return []
    files = [
        name for name in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, name))
    ]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


def load_image_frames(image_path):
    """Decode every same-size frame as RGB with its alpha channel.

    Returns ``(image, alpha)`` where image is ``(batch, height, width, 3)`` and
    alpha is ``(batch, height, width)``. Files without an alpha channel get an
    opaque alpha (all ones), matching ComfyUI's LoadImage mask convention.
    """
    decoded = []
    alphas = []
    width = height = None
    with Image.open(image_path) as opened:
        for frame in ImageSequence.Iterator(opened):
            frame = ImageOps.exif_transpose(frame)
            if width is None:
                width, height = frame.size
            if frame.size != (width, height):
                continue
            if "A" in frame.getbands():
                alpha = np.asarray(frame.getchannel("A"), dtype=np.float32) / 255.0
            else:
                alpha = np.ones((height, width), dtype=np.float32)
            alphas.append(torch.from_numpy(alpha))
            rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
            decoded.append(torch.from_numpy(rgb))
    if not decoded:
        raise ValueError(f"Could not decode any image frames from {image_path}.")
    return torch.stack(decoded, dim=0), torch.stack(alphas, dim=0)


def extract_mask(image, alpha, channel):
    """Mask from the selected channel using ComfyUI load conventions.

    ``alpha`` returns ``1 - alpha`` like LoadImage; color channels return the
    raw channel value so bright pixels become high mask values.
    """
    channel = str(channel).lower()
    if channel == "alpha":
        return 1.0 - alpha
    index = _COLOR_CHANNEL_INDEX.get(channel)
    if index is None or index >= image.shape[-1]:
        return torch.zeros(image.shape[:3], dtype=image.dtype, device=image.device)
    return image[..., index].clone()


class ImageLoaderResizeEnndee:
    """Load an image file, resize it by any core resize type, and expose the original."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (list_input_images(), {
                    "image_upload": True,
                    "tooltip": "Image from ComfyUI's input folder. Use 'choose file to upload' to add files.",
                }),
                "resize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Resize the loaded image; the untouched file stays available on 'original_image'.",
                }),
                "resize_type": (list(resize_modes.RESIZE_TYPES), {
                    "default": "scale dimensions",
                    "tooltip": "Resize behaviour copied from ComfyUI's Resize Image/Mask node.",
                }),
                "width": ("INT", {
                    "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "'scale dimensions' / 'scale width': target width. 0 derives it from the height.",
                }),
                "height": ("INT", {
                    "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "'scale dimensions' / 'scale height': target height. 0 derives it from the width.",
                }),
                "repeat": ("INT", {
                    "default": 1, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Repeat the result in the batch this many times (still-image video helpers).",
                }),
                "keep_proportion": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the aspect ratio and fill the remainder with 'background_color' when the target size disagrees; off stretches to the exact size. Applies to 'scale dimensions' and 'match size'.",
                }),
                "divisible_by": ("INT", {
                    "default": 2, "min": 1, "max": 1024, "step": 1,
                    "tooltip": "Round the final width/height down to this multiple (2 keeps even sizes).",
                }),
                "mask_channel": (list(MASK_CHANNELS), {
                    "default": "alpha",
                    "tooltip": "Channel for the mask output: alpha (1 - alpha, like LoadImage), red, green or blue.",
                }),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Fill color used when keep_proportion pads the image ('#rrggbb', '#rgb' or a color name).",
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01,
                    "tooltip": "'scale by multiplier': 2.0 doubles the size, 0.5 halves it.",
                }),
                "longer_size": ("INT", {
                    "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "'scale longer dimension': the longer edge becomes this many pixels; 0 keeps the size.",
                }),
                "shorter_size": ("INT", {
                    "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "'scale shorter dimension': the shorter edge becomes this many pixels; 0 keeps the size.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01,
                    "tooltip": "'scale total pixels': target megapixels (1.0 is about 1024x1024).",
                }),
                "multiple": ("INT", {
                    "default": 8, "min": 1, "max": 512, "step": 1,
                    "tooltip": "'scale to multiple': floor width/height to this step, then cover-resize and center-crop.",
                }),
                "scale_method": (list(resize_modes.SCALE_METHODS), {
                    "default": "lanczos",
                    "tooltip": "Interpolation for the resized image and mask. 'area' is best for downscaling, 'lanczos' for upscaling.",
                }),
                "no_upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Never enlarge: when the resize target is bigger than the loaded image, keep the image at its current size ('keep_proportion' still fills the target canvas with 'background_color').",
                }),
            },
            "optional": {
                "match": ("IMAGE", {
                    "tooltip": "Reference image for the 'match size' resize type.",
                }),
            },
        }


    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "original_image", "mask", "width", "height", "image_path")
    OUTPUT_TOOLTIPS = (
        "The loaded image, resized when 'resize' is enabled.",
        "The loaded image at its original file size, never resized.",
        "Mask from 'mask_channel' (alpha inverts like LoadImage), resized with the image.",
        "Width of the resized image (source width when 'resize' is off).",
        "Height of the resized image (source height when 'resize' is off).",
        "Resolved path of the loaded file.",
    )
    FUNCTION = "load"
    CATEGORY = "Enndee/image"
    DESCRIPTION = (
        "Load an image from ComfyUI's input folder with the classic load-and-resize "
        "widget set, the nine resize types of ComfyUI core's Resize Image/Mask node, "
        "an alpha/color mask output, and the untouched original image."
    )

    def load(self, image, resize, resize_type, width, height, repeat, keep_proportion,
             divisible_by, mask_channel, background_color, multiplier, longer_size,
             shorter_size, megapixels, multiple, scale_method, no_upscale=False, match=None):
        image_path = folder_paths.get_annotated_filepath(image)
        loaded, alpha = load_image_frames(image_path)
        mask = extract_mask(loaded, alpha, mask_channel)
        original = loaded
        result = loaded

        if resize:
            step = max(1, int(divisible_by))
            mode = resize_type
            match_size = None
            if mode == "match size":
                match_size = resize_modes.source_size(match) if match is not None else None
                if match_size is None:
                    print("[Image Loader (Enndee)] 'match size' has no 'match' input - "
                          "using width/height instead.")
                    mode = "scale dimensions"

            if mode == "scale to multiple":
                result = resize_modes.match_multiple(result, multiple, scale_method)
                mask = resize_modes.match_multiple(mask, multiple, scale_method)
            else:
                source_width, source_height = int(result.shape[2]), int(result.shape[1])
                target_width, target_height = resize_modes.resolve_size(
                    source_width, source_height, mode,
                    width=width, height=height, multiplier=multiplier,
                    longer_size=longer_size, shorter_size=shorter_size,
                    megapixels=megapixels, multiple=multiple, match_size=match_size)
                target_width = max(step, (int(target_width) // step) * step)
                target_height = max(step, (int(target_height) // step) * step)

                if mode in ("scale dimensions", "match size") and keep_proportion:
                    fit_width, fit_height = resize_modes.fit_within(
                        source_width, source_height, target_width, target_height)
                    if no_upscale:
                        fit_width = min(fit_width, source_width)
                        fit_height = min(fit_height, source_height)
                    result = resize_modes.resize_like(result, fit_width, fit_height, scale_method, "disabled")
                    mask = resize_modes.resize_like(mask, fit_width, fit_height, scale_method, "disabled")
                    result = resize_modes.pad_to_size(
                        result, target_width, target_height, resize_modes.parse_color(background_color))
                    mask = resize_modes.pad_to_size(mask, target_width, target_height)
                else:
                    if no_upscale:
                        target_width = min(target_width, source_width)
                        target_height = min(target_height, source_height)
                    result = resize_modes.resize_like(result, target_width, target_height, scale_method, "disabled")
                    mask = resize_modes.resize_like(mask, target_width, target_height, scale_method, "disabled")

        repeat = max(1, int(repeat))
        if repeat > 1:
            result = result.repeat(repeat, 1, 1, 1)
            original = original.repeat(repeat, 1, 1, 1)
            mask = mask.repeat(repeat, 1, 1)

        return (result, original, mask, result.shape[2], result.shape[1], image_path)

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        image_path = folder_paths.get_annotated_filepath(image)
        digest = hashlib.sha256()
        with open(image_path, "rb") as handle:
            digest.update(handle.read())
        return digest.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


NODE_CLASS_MAPPINGS = {
    "Enndee_ImageLoaderResize": ImageLoaderResizeEnndee,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_ImageLoaderResize": "Load & Resize Image (Enndee)",
}

__all__ = [
    "ImageLoaderResizeEnndee",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]



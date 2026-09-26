"""Compact resolution calculator with optional image resizing via core modes."""

import math

import enndee_resize_modes as resize_modes

# Aspect ratio presets: label -> (width_ratio, height_ratio)
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}

ASPECT_RATIO_OPTIONS = list(ASPECT_RATIOS.keys())

# 1 megapixel is defined as 1024 x 1024 pixels
MEGAPIXEL_BASE = 1024 * 1024


def snap_to_multiple(value, multiple):
    """Round ``value`` to the nearest multiple (never below one multiple)."""
    multiple = max(1, int(multiple))
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def dimensions_from_ratio(width_ratio, height_ratio, megapixels, multiple):
    """
    Scale a width:height ratio to a megapixel target and snap to ``multiple``.

    Returns:
        tuple: (width, height) in pixels
    """
    total_pixels = float(megapixels) * MEGAPIXEL_BASE
    scale = math.sqrt(total_pixels / float(width_ratio * height_ratio))
    return (
        snap_to_multiple(width_ratio * scale, multiple),
        snap_to_multiple(height_ratio * scale, multiple),
    )


def dimensions_from_source(source_width, source_height, megapixels, multiple):
    """
    Keep the aspect ratio of a source image and scale it up/down to the
    requested megapixels, snapped to ``multiple``.

    Returns:
        tuple: (width, height) in pixels
    """
    source_width = max(1.0, float(source_width))
    source_height = max(1.0, float(source_height))
    return dimensions_from_ratio(source_width, source_height, megapixels, multiple)


def image_size(image):
    """Return (width, height) of a ComfyUI IMAGE batch, or None."""
    if image is None:
        return None

    shape = getattr(image, "shape", None)
    if shape is None:
        return None

    # ComfyUI IMAGE tensors are [Batch, Height, Width, Channels]
    if len(shape) == 4:
        return int(shape[2]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[0])
    return None


class ResolutionSelectorEnndee:
    """
    Small resolution calculator: aspect ratio + megapixels -> width/height.

    With ``keep_source_aspect_ratio`` enabled the ratio of the connected image
    is used instead of the ``aspect_ratio`` preset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {
                    "default": ASPECT_RATIO_OPTIONS[0],
                    "tooltip": "Aspect ratio for the selected dimensions. Ignored "
                               "while 'keep_source_aspect_ratio' is enabled.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": "Target total megapixels for the selected dimensions "
                               "and for the 'scale total pixels' resize type. "
                               "1.0 MP is about 1024x1024 for a square image.",
                }),
                "multiple": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 4,
                    "tooltip": "Round the selected width/height to this multiple "
                               "(8 is required by most latent models). Also the "
                               "step used by the 'scale to multiple' resize type.",
                }),
                "keep_source_aspect_ratio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the connected image's aspect ratio for the "
                               "selected dimensions; otherwise use the preset.",
                }),
                "resize_type": (list(resize_modes.RESIZE_TYPES), {
                    "default": "scale dimensions",
                    "tooltip": "How a connected image is resized, copied from "
                               "ComfyUI's Resize Image/Mask node. 'scale dimensions' "
                               "uses the selected width/height; the other types "
                               "resize from their own widget.",
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01,
                    "tooltip": "'scale by multiplier': 2.0 doubles the image, 0.5 halves it.",
                }),
                "longer_size": ("INT", {
                    "default": 512, "min": 1, "max": resize_modes.MAX_RESOLUTION, "step": 1,
                    "tooltip": "'scale longer dimension': the longer edge becomes this many pixels.",
                }),
                "shorter_size": ("INT", {
                    "default": 512, "min": 1, "max": resize_modes.MAX_RESOLUTION, "step": 1,
                    "tooltip": "'scale shorter dimension': the shorter edge becomes this many pixels.",
                }),
                "crop": (list(resize_modes.CROP_METHODS), {
                    "default": "center",
                    "tooltip": "Aspect handling for 'scale dimensions' and 'match size': "
                               "'center' crops, 'disabled' stretches.",
                }),
                "scale_method": (list(resize_modes.SCALE_METHODS), {
                    "default": "area",
                    "tooltip": "Interpolation for the resized image. 'area' is best "
                               "for downscaling, 'lanczos' for upscaling.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image to resize and to measure the aspect ratio for "
                               "'keep_source_aspect_ratio'.",
                }),
                "match": ("IMAGE", {
                    "tooltip": "Reference image for the 'match size' resize type.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "IMAGE")
    RETURN_NAMES = ("width", "height", "image")
    OUTPUT_TOOLTIPS = (
        "Selected width in pixels, or the resized image width when an image is connected.",
        "Selected height in pixels, or the resized image height when an image is connected.",
        "The connected image resized by the chosen resize type (None without an image input).",
    )
    FUNCTION = "select"
    CATEGORY = "Enndee/utils"
    DESCRIPTION = ("Calculate width and height from an aspect ratio (or from a "
                   "connected source image) and a megapixel target, and resize a "
                   "connected image with the nine resize types of ComfyUI core's "
                   "Resize Image/Mask node.")

    def select(self, aspect_ratio, megapixels, multiple, keep_source_aspect_ratio,
               resize_type, multiplier, longer_size, shorter_size, crop, scale_method,
               image=None, match=None):
        """Return the selected (width, height) and the resized image output."""
        multiple = max(1, int(multiple))
        megapixels = float(megapixels)

        # Selected dimensions: aspect-ratio preset (or source aspect) + megapixels.
        size = image_size(image)
        if keep_source_aspect_ratio and size is None:
            print("[Resolution Selector (Enndee)] 'keep_source_aspect_ratio' "
                  "is enabled but no image is connected - using the "
                  f"'{aspect_ratio}' preset instead.")
        if keep_source_aspect_ratio and size is not None:
            selected_width, selected_height = dimensions_from_source(
                size[0], size[1], megapixels, multiple)
        else:
            width_ratio, height_ratio = ASPECT_RATIOS.get(
                aspect_ratio, ASPECT_RATIOS["1:1 (Square)"])
            selected_width, selected_height = dimensions_from_ratio(
                width_ratio, height_ratio, megapixels, multiple)

        if size is None:
            if resize_type != "scale dimensions":
                print("[Resolution Selector (Enndee)] Connect an image to use "
                      f"'{resize_type}'; only the selected dimensions are reported.")
            return (selected_width, selected_height, None)

        source_width, source_height = size
        match_size = None
        if resize_type == "match size":
            match_size = image_size(match)
            if match_size is None:
                print("[Resolution Selector (Enndee)] 'match size' has no 'match' "
                      "input - using the selected dimensions instead.")
                resize_type = "scale dimensions"

        if resize_type == "scale to multiple":
            resized = resize_modes.match_multiple(image, multiple, scale_method)
        else:
            target_width, target_height = resize_modes.resolve_size(
                source_width, source_height, resize_type,
                width=selected_width, height=selected_height,
                multiplier=multiplier, longer_size=longer_size,
                shorter_size=shorter_size, megapixels=megapixels,
                multiple=multiple, match_size=match_size)
            mode_crop = crop if resize_type in ("scale dimensions", "match size") else "disabled"
            resized = resize_modes.resize_like(
                image, target_width, target_height, scale_method, mode_crop)

        result_width, result_height = resize_modes.source_size(resized)
        return (result_width, result_height, resized)


NODE_CLASS_MAPPINGS = {
    "Enndee_ResolutionSelector": ResolutionSelectorEnndee,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_ResolutionSelector": "Resolution Selector (Enndee)",
}

__all__ = [
    "ResolutionSelectorEnndee",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
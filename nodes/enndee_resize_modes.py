"""Resize modes shared by the Enndee image tools.

The nine resize types and their size math mirror ComfyUI core's
``ResizeImageMaskNode`` (``comfy_extras/nodes_post_processing.py`` upstream), so
these nodes resize exactly like the core node:

- ``scale dimensions``:        explicit width x height, 0 derives one side
- ``scale by multiplier``:     source size x multiplier
- ``scale longer dimension``:  longer edge becomes ``longer_size``, aspect kept
- ``scale shorter dimension``: shorter edge becomes ``shorter_size``, aspect kept
- ``scale width``:             width, height follows the source aspect
- ``scale height``:            height, width follows the source aspect
- ``scale total pixels``:      megapixels * 1024 * 1024 total pixels
- ``match size``:              copy the size of a reference image or mask
- ``scale to multiple``:       floor to a multiple, then cover + center crop

Tensors follow ComfyUI conventions: IMAGE is ``(batch, height, width, channels)``
and MASK is ``(batch, height, width)``.
"""

import math

import torch

import comfy.utils


MAX_RESOLUTION = 16384

MEGAPIXEL_BASE = 1024 * 1024

RESIZE_TYPES = (
    "scale dimensions",
    "scale by multiplier",
    "scale longer dimension",
    "scale shorter dimension",
    "scale width",
    "scale height",
    "scale total pixels",
    "match size",
    "scale to multiple",
)

SCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")

CROP_METHODS = ("disabled", "center")

_NAMED_COLORS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def is_image(tensor):
    """True for IMAGE tensors (4D), False for MASK tensors (3D)."""
    return tensor.dim() == 4


def source_size(tensor):
    """Return ``(width, height)`` of an IMAGE or MASK tensor.

    IMAGE tensors are ``(batch, height, width, channels)`` and MASK tensors are
    ``(batch, height, width)``, so width and height always live at index 2 and 1.
    """
    return int(tensor.shape[2]), int(tensor.shape[1])


def parse_color(text, label="background color"):
    """Convert ``#rgb``/``#rrggbb``/common color names to RGB floats in 0..1."""
    value = str(text or "").strip().lower()
    if not value:
        return (0.0, 0.0, 0.0)
    if value in _NAMED_COLORS:
        return _NAMED_COLORS[value]
    digits = value[1:] if value.startswith("#") else value
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    try:
        if len(digits) != 6:
            raise ValueError
        return tuple(int(digits[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    except ValueError as error:
        raise ValueError(
            f"Unsupported {label} {text!r}; use '#rrggbb', '#rgb' or a color name."
        ) from error


def resolve_size(source_width, source_height, resize_type, *, width=0, height=0,
                 multiplier=1.0, longer_size=0, shorter_size=0, megapixels=1.0,
                 multiple=8, match_size=None):
    """Return the ``(width, height)`` target the given resize type computes.

    The math matches ``ResizeImageMaskNode`` in ComfyUI core. ``match_size`` is
    a ``(width, height)`` tuple for the "match size" type; sizes never drop
    below one pixel.
    """
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))

    if resize_type == "scale by multiplier":
        factor = max(0.01, float(multiplier))
        return (
            max(1, round(source_width * factor)),
            max(1, round(source_height * factor)),
        )

    if resize_type == "scale longer dimension":
        longer = int(longer_size)
        if longer <= 0:
            return source_width, source_height
        if source_height > source_width:
            return max(1, round(source_width / source_height * longer)), longer
        if source_width > source_height:
            return longer, max(1, round(source_height / source_width * longer))
        return longer, longer

    if resize_type == "scale shorter dimension":
        shorter = int(shorter_size)
        if shorter <= 0:
            return source_width, source_height
        if source_height < source_width:
            return max(1, round(source_width / source_height * shorter)), shorter
        if source_width < source_height:
            return shorter, max(1, round(source_height / source_width * shorter))
        return shorter, shorter

    if resize_type == "scale width":
        target_width = int(width)
        if target_width <= 0:
            return source_width, source_height
        return target_width, max(1, round(source_height * target_width / source_width))

    if resize_type == "scale height":
        target_height = int(height)
        if target_height <= 0:
            return source_width, source_height
        return max(1, round(source_width * target_height / source_height)), target_height

    if resize_type == "scale total pixels":
        total = int(float(megapixels) * MEGAPIXEL_BASE)
        factor = math.sqrt(total / (source_width * source_height))
        return (
            max(1, round(source_width * factor)),
            max(1, round(source_height * factor)),
        )

    if resize_type == "match size":
        if match_size is None:
            return source_width, source_height
        return max(1, int(match_size[0])), max(1, int(match_size[1]))

    if resize_type == "scale to multiple":
        step = max(1, int(multiple))
        target_width = (source_width // step) * step
        target_height = (source_height // step) * step
        if target_width == 0 or target_height == 0:
            return source_width, source_height
        return target_width, target_height

    # "scale dimensions": 0 on one side derives it from the source aspect.
    target_width = int(width)
    target_height = int(height)
    if target_width <= 0 and target_height <= 0:
        return source_width, source_height
    if target_width <= 0:
        return max(1, round(source_width * target_height / source_height)), target_height
    if target_height <= 0:
        return target_width, max(1, round(source_height * target_width / source_width))
    return target_width, target_height


def resize_like(tensor, target_width, target_height, scale_method="area", crop="disabled"):
    """Resize an IMAGE or MASK to an exact size with core crop rules.

    ``crop`` only matters when the target aspect differs from the source aspect:
    ``"center"`` crops, ``"disabled"`` stretches. Sizes are clamped to at least
    one pixel, and already-matching sizes return the input untouched.
    """
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))
    image = is_image(tensor)
    if source_size(tensor) == (target_width, target_height):
        return tensor
    samples = tensor.movedim(-1, 1) if image else tensor.unsqueeze(1)
    samples = comfy.utils.common_upscale(samples, target_width, target_height, scale_method, crop)
    return samples.movedim(1, -1) if image else samples.squeeze(1)


def fit_within(source_width, source_height, target_width, target_height):
    """Largest size inside ``(target_width, target_height)`` that keeps the aspect."""
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    scale = min(target_width / source_width, target_height / source_height)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))


def pad_to_size(tensor, target_width, target_height, background_color=(0.0, 0.0, 0.0)):
    """Center an IMAGE (on background color) or MASK (on zeros) in an exact canvas."""
    target_width = max(int(target_width), 1)
    target_height = max(int(target_height), 1)
    image = is_image(tensor)
    height = int(tensor.shape[1])
    width = int(tensor.shape[2])
    target_width = max(target_width, width)
    target_height = max(target_height, height)
    if (width, height) == (target_width, target_height):
        return tensor
    if image:
        canvas = tensor.new_zeros((tensor.shape[0], target_height, target_width, tensor.shape[3]))
        color = torch.tensor(background_color, dtype=tensor.dtype, device=tensor.device)
        canvas[..., :color.numel()] = color
    else:
        canvas = tensor.new_zeros((tensor.shape[0], target_height, target_width))
    y = (target_height - height) // 2
    x = (target_width - width) // 2
    canvas[:, y:y + height, x:x + width] = tensor
    return canvas


def _cover_size(source_width, source_height, target_width, target_height):
    """Smallest size covering the target that keeps the source aspect."""
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    if scale_x >= scale_y:
        return target_width, max(target_height, math.ceil(source_height * scale_x))
    return max(target_width, math.ceil(source_width * scale_y)), target_height


def match_multiple(tensor, multiple, scale_method="area"):
    """Core "scale to multiple": floor to a multiple, cover-resize, center-crop."""
    step = max(1, int(multiple))
    image = is_image(tensor)
    width, height = source_size(tensor)
    target_width = (width // step) * step
    target_height = (height // step) * step
    if target_width == 0 or target_height == 0:
        return tensor
    if (target_width, target_height) == (width, height):
        return tensor
    scaled_width, scaled_height = _cover_size(width, height, target_width, target_height)
    resized = resize_like(tensor, scaled_width, scaled_height, scale_method, "disabled")
    x = (scaled_width - target_width) // 2
    y = (scaled_height - target_height) // 2
    if image:
        return resized[:, y:y + target_height, x:x + target_width, :]
    return resized[:, y:y + target_height, x:x + target_width]



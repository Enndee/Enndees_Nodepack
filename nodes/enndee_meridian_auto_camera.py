"""Estimate an optimal, speed-limited Meridian camera path from one still's surface data.

The Meridian Parameter Picker's automatic camera mode asks this module for a path instead of
letting the user hand-place one. Two targets:

    subject  orbit the main subject - an optional MASK input, else the near depth layer
    scene    a wider scan that keeps the whole reconstructed surface in frame

Both keep the per-frame camera travel under a speed cap (`max_speed` x content radius per
frame): too fast a camera - too much new surface per frame - is what makes Meridian's depth
reprojections smear and flicker. When the frame count cannot cover the desired swing at that
speed, the swing is shortened and the summary says so.

The surface is unprojected exactly like the fast-depth renderer (frame-0 camera coordinates,
`VFOV_DEGREES`, `x = (u - cx) / f * z`), and the emitted document is the same
`MERIDIAN_CAMERA_PATH` JSON the Camera Path Configurator produces, so the Geometry node and
the fast-depth backend consume it unchanged.
"""

import json
import math

import torch
import torch.nn.functional as F

import enndee_meridian_fast_depth as fast_depth
from enndee_meridian_camera_path import CAMERA_FRAME_OPTIONS

SUBJECT_TARGET = "subject"
SCENE_TARGET = "scene"
AUTO_TARGETS = (SUBJECT_TARGET, SCENE_TARGET)

AUTO_DEPTH_RES = 504              # DA3 process_res for the surface estimate (stats need no more)
MIN_SUBJECT_PIXELS = 64           # absolute floor for a near layer / mask to count as a subject
MIN_SUBJECT_SHARE = 0.01          # ... plus 1 % of the cloud, so big frames need a real subject
SUBJECT_FILL = 2.2                # orbit radius / subject radius (~45 % frame fill at 55 deg vfov)
SCENE_FILL = 1.5                  # orbit radius / scene radius
SUBJECT_ELEVATION = 12.0          # deg, the orbit rises/falls this much over the path
SCENE_ELEVATION = 8.0
SUBJECT_SPAN = 360.0              # deg of azimuth a subject orbit would like to cover
SCENE_SPAN = 270.0
KEY_TARGET = 17                   # path keys (Catmull-Rom control points)
MAX_AZIMUTH_PER_FRAME = 6.0       # hard cap: 360 deg then needs at least 60 frames
MIN_ORBIT_RADIUS = 0.05           # never place the camera on the content
DEFAULT_MAX_SPEED = 0.12          # fraction of the content radius the camera may travel per frame


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return value


def _percentile(values, q):
    """`torch.quantile` percentile of a 1-D tensor, with a bounded pool."""
    flat = values.reshape(-1).float()
    if flat.numel() > fast_depth.QUANTILE_MAX:
        flat = flat[:: -(-flat.numel() // fast_depth.QUANTILE_MAX)]
    return float(torch.quantile(flat, float(q)))


def _fit_mask(mask, height, width):
    """Bool selection on the depth grid: threshold at >0.5, nearest-resize to height x width."""
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask.float().mean(dim=0)
    if mask.ndim != 2:
        raise ValueError("The subject mask must be a 2D MASK tensor.")
    if tuple(mask.shape) != (height, width):
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                             size=(height, width), mode="nearest")[0, 0]
    return mask.float() > 0.5


def surface_points(depth, mask=None):
    """Unproject a depth map into frame-0 camera coordinates ([N, 3] float tensor).

    Same camera model as the fast-depth renderer: origin at the source camera, +z along the
    view axis, `f = 0.5 * H / tan(vfov / 2)`. `mask` selects a subset (the subject). Relative
    depth units are kept as-is - every decision below uses ratios, so the gauge is irrelevant.
    """
    if depth.ndim == 3:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError("The auto camera needs a 2D depth map from the fast-depth backend.")
    depth = depth.float()
    height, width = depth.shape
    focal = 0.5 * height / math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0)
    yy, xx = torch.meshgrid(torch.arange(height, device=depth.device, dtype=torch.float32),
                            torch.arange(width, device=depth.device, dtype=torch.float32),
                            indexing="ij")
    points = torch.stack([(xx - width / 2.0) / focal * depth,
                          (yy - height / 2.0) / focal * depth,
                          depth], dim=-1).reshape(-1, 3)
    if mask is not None:
        selection = _fit_mask(mask, height, width).reshape(-1)
        points = points[selection]
    if points.numel() == 0:
        raise ValueError("The subject mask is empty - nothing to orbit.")
    return points


def _minimum_subject_pixels(total_points):
    """A subject needs at least `MIN_SUBJECT_PIXELS` and `MIN_SUBJECT_SHARE` of the cloud."""
    return max(MIN_SUBJECT_PIXELS, MIN_SUBJECT_SHARE * int(total_points))


def _otsu_threshold(values, bins=64):
    """Otsu split of a 1-D value set: the histogram cut with the largest between-class variance.

    A fixed percentile only works when the subject fills a known share of the frame; Otsu
    separates the near layer from the background no matter how small the subject is.
    """
    values = values.reshape(-1).float()
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return high
    histogram = torch.histc(values, bins=bins, min=low, max=high)
    centres = low + (high - low) * (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
    weights = histogram / histogram.sum()
    omega = torch.cumsum(weights, dim=0)
    means = torch.cumsum(weights * centres, dim=0)
    between = (means[-1] * omega - means) ** 2 / (omega * (1.0 - omega)).clamp(min=1e-12)
    return float(centres[int(torch.argmax(between))])


def split_subject(points):
    """(subject points, label): the near layer found by Otsu, else the whole cloud.

    The subject is the near class of the depth split. A split that collapses (a sliver or
    almost everything) is not a subject, so "subject" then means the scene instead of failing.
    """
    depth_values = points[:, 2]
    threshold = _otsu_threshold(depth_values)
    near = points[depth_values <= threshold]
    share = near.shape[0] / points.shape[0]
    if near.shape[0] >= _minimum_subject_pixels(points.shape[0]) and 0.03 <= share <= 0.70:
        return near, f"near depth layer ({near.shape[0]} of {points.shape[0]} points)"
    return points, "whole surface (no clear near layer)"


def subject_points(depth, mask=None):
    """Subject points plus the label describing their source (a mask wins over the layer)."""
    if mask is not None:
        selected = surface_points(depth, mask=mask)
        if selected.shape[0] >= _minimum_subject_pixels(surface_points(depth).shape[0]):
            return selected, "input mask"
    return split_subject(surface_points(depth))


def sphere_of(points):
    """Robust (centre, radius) of a point set: median centre, 90th-percentile distance."""
    centre = torch.stack([points[:, axis].median() for axis in range(3)])
    radius = _percentile((points - centre).norm(dim=-1), 0.90)
    return centre, max(radius, 1e-6)


def speed_limited_span(desired_span, radius, content_radius, frames, max_speed):
    """(span_deg, travel_per_frame, limited, azimuth_per_frame) for one orbit.

    The camera travels `radius * azimuth` along the arc; its budget is `max_speed *
    content_radius` per frame, and the azimuth never exceeds `MAX_AZIMUTH_PER_FRAME`.
    `content_radius` (subject or scene radius) makes the cap gauge- and resolution-independent:
    every number is a ratio, not a metre.
    """
    frames = int(frames)
    if frames < 2:
        raise ValueError("A camera path needs at least two frames.")
    budget_per_frame = max(1e-6, _finite(max_speed, "Max camera speed")) * max(1e-6, content_radius)
    azimuth_cap = MAX_AZIMUTH_PER_FRAME * (frames - 1)
    span = min(_finite(desired_span, "Desired swing"), azimuth_cap)
    span = min(span, math.degrees(budget_per_frame / max(radius, 1e-6)) * (frames - 1))
    span = max(0.0, span)
    per_frame = span / (frames - 1)
    return span, math.radians(per_frame) * radius, span < desired_span - 1e-9, per_frame


def _place(centre, radius, yaw_degrees, elevation_degrees):
    """Camera position on the orbit sphere around `centre`.

    Yaw 0 sits *between* the centre and the source camera (the origin), positive yaw moves to the
    subject's right; elevation 0 is level with the centre, positive is above it (the OpenCV frame
    has y pointing down). That is the Camera Path Configurator's convention (`_yaw_direction`):
    the source camera at the origin is the closest possible viewpoint of the subject.
    """
    yaw = math.radians(yaw_degrees)
    elevation = math.radians(elevation_degrees)
    offset = (math.cos(elevation) * math.sin(yaw),
              -math.sin(elevation),
              -math.cos(elevation) * math.cos(yaw))
    return [float(centre[axis]) + offset[axis] * radius for axis in range(3)]


def _key_frames(frames, target=KEY_TARGET):
    """Evenly spaced key frames covering 0..frames-1, each a whole frame index."""
    frames = int(frames)
    step = max(1, round((frames - 1) / max(1, target - 1)))
    ticks = list(range(0, frames, step))
    if ticks[-1] != frames - 1:
        ticks.append(frames - 1)
    return ticks


def build_path_document(frames, centre, radius, span, elevation, name, description):
    """The MERIDIAN_CAMERA_PATH JSON for one speed-limited orbit plus its parsed keys."""
    frames = int(frames)
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(
            f"Unsupported Meridian frame count {frames}; choose {', '.join(CAMERA_FRAME_OPTIONS)}."
        )
    keys = []
    for tick in _key_frames(frames):
        fraction = tick / (frames - 1)
        position = _place(centre, radius, span * fraction, -elevation + 2.0 * elevation * fraction)
        keys.append({
            "pos": [round(value, 6) for value in position],
            "look": [round(float(value), 6) for value in centre],
            "src": int(tick),
            "t": int(tick),
        })
    document = {"name": name, "description": description, "frames": frames,
                "stations": ["Auto"], "path": keys}
    return json.dumps(document, separators=(",", ":"), allow_nan=False), keys


def estimate_camera_path(reference, frames, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED,
                         subject_mask=None, model_size="", depth_res=AUTO_DEPTH_RES,
                         device=None, depth_fn=None):
    """(signal JSON, summary) for one still: probe the surface, size the orbit, cap the speed.

    `target` is "subject" (the mask input, else the near depth layer) or "scene" (the whole
    reconstructed surface). The camera starts in front of the centre - the source camera's side,
    yaw 0 - swings away from it and rises/falls by the target's elevation range while always
    looking at the centre. The summary carries every number behind the decision, for the node's
    console line. `depth_fn` injects a depth map instead of running a depth model (tests).
    """
    frames = int(frames)
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(
            f"Unsupported Meridian frame count {frames}; choose {', '.join(CAMERA_FRAME_OPTIONS)}."
        )
    target = str(target or SUBJECT_TARGET).strip().lower()
    if target not in AUTO_TARGETS:
        raise ValueError(f"Automatic camera target must be one of {', '.join(AUTO_TARGETS)}.")

    depth = (depth_fn(reference) if depth_fn is not None else
             depth_from_reference(reference, model_size=model_size, depth_res=depth_res,
                                  device=device))
    cloud = surface_points(depth)
    scene_centre, scene_radius = sphere_of(cloud)
    if target == SUBJECT_TARGET:
        points, source_label = subject_points(depth, mask=subject_mask)
        centre, content_radius = sphere_of(points)
        fill, desired_span, elevation = SUBJECT_FILL, SUBJECT_SPAN, SUBJECT_ELEVATION
        mode_label = "subject orbit"
    else:
        points, source_label = cloud, "whole surface (scene)"
        centre, content_radius = scene_centre, scene_radius
        fill, desired_span, elevation = SCENE_FILL, SCENE_SPAN, SCENE_ELEVATION
        mode_label = "scene scan"

    orbit_radius = max(fill * content_radius, MIN_ORBIT_RADIUS)
    span, travel, limited, azimuth_per_frame = speed_limited_span(
        desired_span, orbit_radius, content_radius, frames, max_speed)
    description = (
        f"Estimated from the still's surface ({source_label}): {mode_label} at {fill:g}x the "
        f"content radius ({orbit_radius:.3g} vs {content_radius:.3g} units), {span:.1f} deg of "
        f"azimuth over {frames} frames ({azimuth_per_frame:.2f} deg/frame, {travel:.3g} "
        f"units/frame; budget {float(max_speed) * 100:.0f} % of the content radius per frame"
        + (", swing shortened to fit the budget" if limited else "") + "). The camera starts in "
        f"front of the centre, always looks at it and rises/falls by {elevation:g} deg. "
        "Non-front views are synthetic depth reprojections, not observed geometry."
    )
    document, keys = build_path_document(frames, centre, orbit_radius, span, elevation,
                                         f"Auto {mode_label} ({frames} frames)", description)
    summary = {
        "target": target, "frames": frames, "source": source_label,
        "points": int(points.shape[0]), "scene_radius": float(scene_radius),
        "centre": [float(value) for value in centre], "content_radius": float(content_radius),
        "orbit_radius": float(orbit_radius), "swing": span, "desired_swing": desired_span,
        "elevation": elevation, "azimuth_per_frame": azimuth_per_frame,
        "travel_per_frame": travel, "speed_limited": limited, "keys": len(keys),
        "max_speed": float(max_speed),
    }
    return document, summary


def depth_from_reference(reference, model_size="", depth_res=AUTO_DEPTH_RES, device=None):
    """Depth for the estimator: the requested DA3 pick, else DA3-Small, else Depth-Anything-V2.

    Reuses the fast-depth backend's loaders (its VRAM cache included), so the automatic camera
    costs one depth pass per run. Every fallback goes through the same "larger = farther"
    conversion, and the caller's `model_size` is tried first so the estimate matches what the
    Geometry node will render with.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = reference[0:1].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    attempts = [str(model_size).strip()] if str(model_size).strip() else []
    attempts += ["Depth-Anything-3-Small", "Depth-Anything-V2-Small-hf"]
    last_error = None
    for candidate in attempts:
        try:
            if candidate.startswith(fast_depth.DA3_PREFIX):
                height, width = int(first.shape[1]), int(first.shape[2])
                return fast_depth._predict_da3_depth(
                    candidate, first, device,
                    fast_depth._da3_process_res(depth_res, width, height))
            model = fast_depth._get_depth_model(candidate, device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x518 = F.interpolate(first.permute(0, 3, 1, 2),
                                 size=(fast_depth.DEPTH_RES, fast_depth.DEPTH_RES),
                                 mode="bilinear", align_corners=False)
            with torch.no_grad():
                prediction = model(pixel_values=((x518 - mean) / std).half())
            return fast_depth._invert_disparity(prediction.predicted_depth[0].float())
        except Exception as exc:      # missing weights / no network: try the next family
            last_error = exc
    raise RuntimeError("No depth model is available for the automatic camera "
                       f"(tried {', '.join(attempts)}): {last_error}")


def format_summary(summary):
    """One console-friendly line describing an estimate (used by the picker node)."""
    line = (
        f"auto camera: {summary['target']} orbit around "
        f"[{summary['centre'][0]:.3g}, {summary['centre'][1]:.3g}, {summary['centre'][2]:.3g}] "
        f"- {summary['source']}, radius {summary['orbit_radius']:.3g} "
        f"({summary['swing']:.0f}/{summary['desired_swing']:.0f} deg over {summary['frames']} "
        f"frames, {summary['azimuth_per_frame']:.2f} deg + {summary['travel_per_frame']:.3g} "
        f"units per frame, {summary['keys']} keys)"
    )
    if summary["speed_limited"]:
        line += f" - swing shortened to stay under {summary['max_speed'] * 100:.0f} %/frame"
    return line

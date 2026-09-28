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
PIVOT_PERCENTILE_LOW = 2.0        # per-axis clip before the bounding-box midpoint (robust pivot)
PIVOT_PERCENTILE_HIGH = 98.0
SUBJECT_FILL = 2.2                # orbit radius / subject radius (~45 % frame fill at 55 deg vfov)
SCENE_FILL = 2.0                  # the scene gets a deliberately big oval
FRONT_ORBIT_SHARE = 0.5           # share of the frames spent on the front O-orbit
FRONT_YAW_AMPLITUDE = 62.0        # deg, the front O swings this far to either side
FRONT_ELEVATION = 30.0            # deg, the front O reaches this high/low
REST_YAW_SPAN = 270.0             # deg, the height orbit laps around the rest of the subject
REST_ELEVATION_HIGH = 38.0        # where the height orbit ends: a new height, no surface repeat
SCENE_SPAN = 350.0                # deg, the scene oval stops just short of a full lap
SCENE_ELEVATION_LOW = -12.0
SCENE_ELEVATION_HIGH = 28.0
KEY_TARGET = 17                   # path keys (Catmull-Rom control points)
AMPLITUDE_STEPS = 40              # lambda ladder: 1.0, 0.975, ... 0.025 (2.5 % rungs)
MIN_ORBIT_RADIUS = 0.05           # never place the camera on the content
DEFAULT_MAX_SPEED = 0.12          # fraction of the content radius the camera may travel per frame
COLLISION_MARGIN = 0.15           # of the content radius: how close a camera key may come to geometry
COLLISION_ITERATIONS = 4
COLLISION_MAX_POINTS = 120_000    # stride bigger clouds down for the distance checks
MAX_PIVOT_OFFSET = 1.0            # pivot offset widgets, in content radii



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
        selection = _fit_mask(mask, height, width).reshape(-1).to(device=points.device)
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
    centres = low + (high - low) * (torch.arange(bins, dtype=torch.float32,
                                                device=values.device) + 0.5) / bins
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


def geometric_pivot(points):
    """Robust 3D bounding-box midpoint of a point set: the *volumetric* centre.

    Per-axis 2 %/98 % percentiles clip stray points, then the midpoint of that box is taken.
    Unlike a plain median (which hugs the dense front face, because that is where most samples
    sit) this sits in the middle of the depth profile, so an orbit around it keeps the subject
    centred from every side. Returns (midpoint [3], extents [3]).
    """
    low = torch.tensor([_percentile(points[:, axis], PIVOT_PERCENTILE_LOW / 100.0)
                        for axis in range(3)], dtype=points.dtype, device=points.device)
    high = torch.tensor([_percentile(points[:, axis], PIVOT_PERCENTILE_HIGH / 100.0)
                         for axis in range(3)], dtype=points.dtype, device=points.device)
    midpoint = (low + high) / 2.0
    extents = (high - low).clamp(min=1e-6)
    return midpoint, extents


def pivot_radius(points, pivot):
    """90th-percentile distance from a pivot: the radius that encloses the content."""
    return max(_percentile((points - pivot).norm(dim=-1), 0.90), 1e-6)


def validate_frames(frames):
    """The frame count as an int, checked against the node's CAMERA_FRAME_OPTIONS."""
    frames = int(frames)
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(
            f"Unsupported Meridian frame count {frames}; choose {', '.join(CAMERA_FRAME_OPTIONS)}."
        )
    return frames


def _fit_amplitude(samples_of, budget_per_frame, ladder=AMPLITUDE_STEPS):
    """Largest amplitude scale whose path keeps every per-frame step within the speed budget.

    The path is sampled at full amplitude first, the true camera travel between consecutive
    frames is measured, then the amplitudes are scaled down the ladder until the step fits. Every
    number stays a *ratio* to the content radius, so the estimate is scale-, resolution- and
    gauge-independent. Returns (scale, samples, travel_per_frame).
    """
    scale, samples, travel = 1.0, [], 0.0
    for step in range(ladder):
        scale = round(1.0 - 0.025 * step, 3)
        samples = samples_of(scale)
        travel = max((math.dist(samples[index - 1], samples[index])
                      for index in range(1, len(samples))), default=0.0)
        if travel <= budget_per_frame:
            break
    return scale, samples, travel


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


def subject_samples(frames, pivot, radius, scale=1.0):
    """Per-frame positions of the subject path: a big front O-orbit, then a height lap.

    Phase 1 (FRONT_ORBIT_SHARE of the frames) traces a big closed loop in front of the subject -
    the azimuth swings +/-FRONT_YAW_AMPLITUDE while the elevation goes low, level, high, level
    and back - so the front is shown from below, right, above and left in one move. Phase 2 laps
    the rest (REST_YAW_SPAN) while the elevation eases from the loop's low point up to
    REST_ELEVATION_HIGH: the back gets *new heights* instead of a flat ring, and the path ends on
    surface the front loop did not show. `scale` is the speed-fit amplitude.
    """
    frames = int(frames)
    yaw_amplitude = FRONT_YAW_AMPLITUDE * scale
    elevation_amplitude = FRONT_ELEVATION * scale
    rest_span = REST_YAW_SPAN * scale
    rest_high = REST_ELEVATION_HIGH * scale
    split = max(1.0, (frames - 1) * FRONT_ORBIT_SHARE)
    positions = []
    for index in range(frames):
        if index <= split:
            phase = index / split
            yaw = yaw_amplitude * math.sin(2.0 * math.pi * phase)
            elevation = -elevation_amplitude * math.cos(2.0 * math.pi * phase)
        else:
            phase = (index - split) / max(1.0, (frames - 1) - split)
            yaw = rest_span * phase
            elevation = -elevation_amplitude + (rest_high + elevation_amplitude) * (
                1.0 - math.cos(math.pi * phase)) / 2.0
        positions.append(_place(pivot, radius, yaw, elevation))
    return positions


def scene_samples(frames, pivot, radius, scale=1.0):
    """Per-frame positions of the scene path: one big oval lap around the whole scene.

    A near-full azimuth lap (SCENE_SPAN) at SCENE_FILL x the scene radius whose elevation eases
    from just below the horizon to well above it: the camera rises while it goes round, so the
    move reads as a big oval orbit *over* the scene instead of a flat ring. `scale` is the
    speed-fit amplitude.
    """
    frames = int(frames)
    span = SCENE_SPAN * scale
    low = SCENE_ELEVATION_LOW * scale
    high = SCENE_ELEVATION_HIGH * scale
    positions = []
    for index in range(frames):
        phase = index / max(1, frames - 1)
        yaw = span * phase
        elevation = low + (high - low) * (1.0 - math.cos(math.pi * phase)) / 2.0
        positions.append(_place(pivot, radius, yaw, elevation))
    return positions


def automatic_keys(frames, pivot, radius, content_radius, target, max_speed=DEFAULT_MAX_SPEED):
    """(keys, info) for the automatic path of `target`: subject composite or scene oval.

    The path is fitted to the speed budget by scaling its amplitudes; the keys are the
    Catmull-Rom control points the Geometry node samples (evenly spaced, whole frame indices).
    """
    frames = validate_frames(frames)
    budget = max(1e-6, _finite(max_speed, "Max camera speed")) * max(1e-6, content_radius)
    if str(target).strip().lower() == SUBJECT_TARGET:
        samples_of = lambda scale: subject_samples(frames, pivot, radius, scale)
        style = "front O-orbit + height lap"
    else:
        samples_of = lambda scale: scene_samples(frames, pivot, radius, scale)
        style = "big scene oval"
    scale, samples, travel = _fit_amplitude(samples_of, budget)
    keys = [{
        "pos": [round(value, 6) for value in samples[tick]],
        "look": [round(float(value), 6) for value in pivot],
        "src": int(tick),
        "t": int(tick),
    } for tick in _key_frames(frames)]
    info = {"style": style, "amplitude_scale": scale, "travel_per_frame": travel,
            "budget_per_frame": budget, "keys": len(keys)}
    return keys, info


def document_from_keys(frames, keys, name, description):
    """The MERIDIAN_CAMERA_PATH JSON for a finished key list (used by both camera paths)."""
    frames = validate_frames(frames)
    document = {"name": name, "description": description, "frames": frames,
                "stations": ["Auto"], "path": keys}
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def probe_surface(reference, target=SUBJECT_TARGET, subject_mask=None, model_size="",
                  depth_res=AUTO_DEPTH_RES, device=None, depth_fn=None):
    """One depth pass -> everything the automatic camera mode needs to know.

    Returns the scene cloud, the target's *geometric pivot* (the midpoint of its depth profile's
    bounding box, not the dense-surface median), the enclosing radius and the labels for the
    summary. Both automatic paths start here: the automatic one builds its orbit from it, the
    manual one only takes the pivot (plus the user's offset) - and both run the collision guard
    against this cloud. `depth_fn` injects a depth map instead of running a depth model (tests).
    """
    target = str(target or SUBJECT_TARGET).strip().lower()
    if target not in AUTO_TARGETS:
        raise ValueError(f"Automatic camera target must be one of {', '.join(AUTO_TARGETS)}.")
    depth = (depth_fn(reference) if depth_fn is not None else
             depth_from_reference(reference, model_size=model_size, depth_res=depth_res,
                                  device=device))
    cloud = surface_points(depth)
    scene_pivot, scene_extents = geometric_pivot(cloud)
    scene_radius = pivot_radius(cloud, scene_pivot)
    if target == SUBJECT_TARGET:
        points, source_label = subject_points(depth, mask=subject_mask)
        pivot, extents = geometric_pivot(points)
        radius = pivot_radius(points, pivot)
    else:
        points, source_label = cloud, "whole surface (scene)"
        pivot, extents, radius = scene_pivot, scene_extents, scene_radius
    return {
        "target": target, "source": source_label,
        "scene_points": cloud, "scene_radius": float(scene_radius),
        "scene_pivot": [float(value) for value in scene_pivot],
        "scene_extents": [float(value) for value in scene_extents],
        "points": int(cloud.shape[0]),
        "content_points": int(points.shape[0]),
        "pivot": [float(value) for value in pivot],
        "extents": [float(value) for value in extents],
        "content_radius": float(radius),
    }


def offset_pivot(surface, offsets):
    """The surface's pivot shifted by the user's offset (fractions of the content radius).

    The automatic pivot is the algorithm's answer; this is how the user nudges it without
    touching the estimate - the same three numbers the manual path would otherwise set, only
    relative to what the depth profile says.
    """
    radius = max(1e-6, float(surface["content_radius"]))
    return [float(surface["pivot"][axis]) + _finite(offsets[axis], "Pivot offset") * radius
            for axis in range(3)]


def guard_collisions(keys, surface, margin=COLLISION_MARGIN):
    """Push camera keys out of the scene until none sits closer than `margin` to any point.

    The distance to the scene points is checked for every key (the cloud is strided down to a
    bounded pool); offending keys move *away from the pivot* - the direction that keeps the
    subject framed - and are re-checked, up to COLLISION_ITERATIONS times. The margin is measured
    in content radii, so it scales with the subject (or scene) that is being orbited instead of
    with the reconstruction's absolute units. Returns (keys, fixed, worst_before, worst_after),
    the two clearances as content-radius fractions.
    """
    cloud = surface["scene_points"]
    unit = max(1e-6, float(surface["content_radius"]))
    clearance = max(0.0, _finite(margin, "Collision margin")) * unit
    if cloud.shape[0] == 0 or clearance <= 0.0 or not keys:
        return keys, 0, math.inf, math.inf
    stride = max(1, int(cloud.shape[0]) // COLLISION_MAX_POINTS)
    pool = cloud[::stride]
    pivot = torch.tensor(surface["pivot"], dtype=pool.dtype, device=pool.device)
    positions = torch.tensor([key["pos"] for key in keys], dtype=pool.dtype, device=pool.device)
    distances = torch.cdist(positions, pool).amin(dim=1)
    worst_before = float(distances.min()) / unit
    fixed = set()
    for _ in range(COLLISION_ITERATIONS):
        offenders = torch.nonzero(distances < clearance, as_tuple=False).flatten().tolist()
        if not offenders:
            break
        for index in offenders:
            fixed.add(index)
            direction = positions[index] - pivot
            length = float(direction.norm())
            if length < 1e-9:
                direction, length = positions.new_tensor([0.0, 0.0, -1.0]), 1.0
            push = (clearance - float(distances[index]) + 1e-3) * 1.15
            positions[index] = positions[index] + (direction / length) * push
        distances = torch.cdist(positions, pool).amin(dim=1)
    for index in sorted(fixed):
        keys[index]["pos"] = [round(value, 6) for value in positions[index].tolist()]
    return keys, len(fixed), worst_before, float(distances.min()) / unit


def estimate_camera_path(reference, frames, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED,
                         subject_mask=None, model_size="", depth_res=AUTO_DEPTH_RES,
                         device=None, depth_fn=None, pivot_offset=(0.0, 0.0, 0.0)):
    """(signal JSON, summary) for one still: geometric pivot, automatic path, collision guard.

    The pivot is the *geometric midpoint* of the target's depth profile - the robust bounding-box
    midpoint, so the depth of the subject is what places it, not the picture centre - optionally
    shifted by `pivot_offset` (in content radii). The path is the subject composite (front
    O-orbit + height lap) or the scene's big oval, fitted to the speed budget; the camera never
    comes closer to the scene than COLLISION_MARGIN x the content radius. The summary carries every
    number behind the decision, for the node's console line. `depth_fn` injects a depth map
    instead of running a depth model (tests).
    """
    frames = validate_frames(frames)
    surface = probe_surface(reference, target=target, subject_mask=subject_mask,
                            model_size=model_size, depth_res=depth_res, device=device,
                            depth_fn=depth_fn)
    target = surface["target"]
    max_speed = _finite(max_speed, "Max camera speed")
    pivot = offset_pivot(surface, pivot_offset)
    content_radius = max(1e-6, float(surface["content_radius"]))
    fill = SUBJECT_FILL if target == SUBJECT_TARGET else SCENE_FILL
    orbit_radius = max(fill * content_radius, MIN_ORBIT_RADIUS)
    keys, info = automatic_keys(frames, pivot, orbit_radius, content_radius, target, max_speed)
    keys, fixed, worst_before, worst_after = guard_collisions(keys, surface)
    description = (
        f"Estimated from the still's surface ({surface['source']}): pivot "
        f"[{pivot[0]:.3g}, {pivot[1]:.3g}, {pivot[2]:.3g}] is the geometric midpoint of the "
        f"depth profile ({surface['extents'][0]:.3g} x {surface['extents'][1]:.3g} x "
        f"{surface['extents'][2]:.3g} units), {info['style']} at {fill:g}x the content radius "
        f"({orbit_radius:.3g} vs {content_radius:.3g} units), amplitudes at "
        f"{info['amplitude_scale']:.2f} of full ({info['travel_per_frame']:.3g} units per frame, "
        f"budget {max_speed * 100:.0f} % of the content radius per frame)"
        + (f"; {fixed} key(s) pushed clear of the scene geometry" if fixed else "")
        + ". Non-front views are synthetic depth reprojections, not observed geometry."
    )
    document = document_from_keys(frames, keys, f"Auto {info['style']} ({frames} frames)",
                                  description)
    summary = {
        "target": target, "frames": frames, "source": surface["source"],
        "points": surface["points"], "content_points": surface["content_points"],
        "pivot": pivot, "extents": surface["extents"],
        "pivot_offset": [float(value) for value in pivot_offset],
        "content_radius": content_radius, "orbit_radius": float(orbit_radius),
        "style": info["style"], "amplitude_scale": info["amplitude_scale"],
        "travel_per_frame": info["travel_per_frame"],
        "budget_per_frame": info["budget_per_frame"], "keys": info["keys"],
        "max_speed": max_speed, "scene_radius": float(surface["scene_radius"]),
        "scene_pivot": surface["scene_pivot"], "collision_fixes": fixed,
        "collision_margin": COLLISION_MARGIN, "clearance_before": worst_before,
        "clearance_after": worst_after,
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
    """One console-friendly line describing an estimate (used by the node)."""
    line = (
        f"auto camera: {summary['style']} around "
        f"[{summary['pivot'][0]:.3g}, {summary['pivot'][1]:.3g}, {summary['pivot'][2]:.3g}] "
        f"- the geometric midpoint of the {summary['target']} depth profile "
        f"({summary['extents'][0]:.3g} x {summary['extents'][1]:.3g} x "
        f"{summary['extents'][2]:.3g} units, {summary['source']}), radius "
        f"{summary['orbit_radius']:.3g} over {summary['frames']} frames "
        f"({summary['travel_per_frame']:.3g} units/frame at {summary['amplitude_scale']:.2f}x "
        f"amplitude, budget {summary['max_speed'] * 100:.0f} %/frame, {summary['keys']} keys)"
    )
    if summary["collision_fixes"]:
        line += (f" - {summary['collision_fixes']} key(s) pushed out of the scene "
                 f"(closest approach {summary['clearance_before'] * 100:.1f} % -> "
                 f"{summary['clearance_after'] * 100:.1f} % of the content radius)")
    return line

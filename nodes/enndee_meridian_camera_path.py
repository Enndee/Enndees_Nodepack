"""Build Meridian camera paths: multi-station vertical O-orbits and alternating-height sweeps."""

import json
import math


CAMERA_FRAME_OPTIONS = ["73", "90", "107", "124", "141", "158", "175", "243"]
CAMERA_SIGNAL_TYPE = "MERIDIAN_CAMERA_PATH"
LOOP_KEY_PHASES = tuple(range(0, 361, 45))
CONNECTOR_STEPS = 4
ORBIT_OPTIONS = ("Front", "Left", "Right", "Back", "Up", "Down", "LeftBack", "RightBack")
START_STATION_OPTIONS = ("Visit order",) + ORBIT_OPTIONS

# Camera-path styles. "O Orbits" closes vertical O loops at the named stations;
# "Alternating Height" sweeps the azimuth while the elevation pendulum-swings
# between a lower and an upper arc. The picker's path_camera_mode combo and the
# web/js extension reuse these labels verbatim.
CAMERA_MODE_OPTIONS = ("O Orbits", "Alternating Height", "Spiral Sweep")
HEIGHT_SWEEP_MODE = CAMERA_MODE_OPTIONS[1]
SPIRAL_SWEEP_MODE = CAMERA_MODE_OPTIONS[2]
ARC_OPTIONS = ("Low arc", "High arc")
# Analytic keys per alternating sweep segment: enough to follow the eased
# pendulum inside the Catmull-Rom interpolation without dense key lists.
SWEEP_SUBSTEPS = 3
# Keys along a spiral sweep; the curve is smooth and nearly straight, so a
# handful of evenly spaced keys is both faithful and cheap.
SPIRAL_KEY_COUNT = 9
# Elevation cap shared with the Up/Down station convention (70 degrees). At
# +/-90 the look direction is parallel to Meridian's zero-roll up vector and the
# orientation degenerates (vertical gimbal lock), so the arcs stop short of the
# geometric zenith/nadir.
MAX_ELEVATION_DEGREES = 70.0
VISIT_ORDER = ("Front", "Left", "LeftBack", "Back", "RightBack", "Right", "Up", "Down")
ORBIT_WIDGET_NAMES = {
    "Front": "orbit_front",
    "Left": "orbit_left",
    "Right": "orbit_right",
    "Back": "orbit_back",
    "Up": "orbit_up",
    "Down": "orbit_down",
    "LeftBack": "orbit_left_back",
    "RightBack": "orbit_right_back",
}

# LeftBack/RightBack form a symmetric 120-degree triple with the front
# direction (the three directions sum to zero), matching the hand-authored
# Meridian 3-orbit 120-degree camera paths.

# 70-degree elevation pitch avoids vertical gimbal lock with OpenCV UP (0, -1, 0)
# while orienting the camera to look down onto (or up at) the target pivot.
UP_ELEVATION_DEGREES = 70.0
DOWN_ELEVATION_DEGREES = 70.0
_UP_RAD = math.radians(UP_ELEVATION_DEGREES)
_DOWN_RAD = math.radians(DOWN_ELEVATION_DEGREES)

STATION_DIRECTIONS = {
    "Left": (-1.0, 0.0, 0.0),
    "Right": (1.0, 0.0, 0.0),
    "Back": (0.0, 0.0, 1.0),
    "Up": (0.0, -math.sin(_UP_RAD), -math.cos(_UP_RAD)),
    "Down": (0.0, math.sin(_DOWN_RAD), -math.cos(_DOWN_RAD)),
}


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return value


def _unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-8:
        raise ValueError("Orbit station direction cannot be zero-length.")
    return tuple(value / length for value in vector)


def _front_direction(pivot):
    """Point from the look pivot toward the source camera at the origin."""
    return _unit((-pivot[0], -pivot[1], -pivot[2]))


def _rear_station_directions(pivot):
    """Directions exactly 120 degrees from the front direction and each other.

    The symmetric triple satisfies ``front + left_back + right_back = 0``, so
    each rear station is ``-front/2`` plus/minus ``sqrt(3)/2`` of the horizontal
    axis perpendicular to the front direction (a 120-degree rotation of the
    front direction about that axis). For a horizontal front this reproduces the
    hand-authored Meridian 3-orbit 120-degree stations; a tilted front spreads
    half of its elevation symmetrically across both rear stations.
    """
    front = _front_direction(pivot)
    up = (0.0, -1.0, 0.0)
    side = (
        up[1] * front[2] - up[2] * front[1],
        up[2] * front[0] - up[0] * front[2],
        up[0] * front[1] - up[1] * front[0],
    )
    side_length = math.sqrt(sum(value * value for value in side))
    if side_length < 1e-8:
        side = (1.0, 0.0, 0.0)
    else:
        side = tuple(value / side_length for value in side)

    base = tuple(-value * 0.5 for value in front)
    offset = tuple(value * (math.sqrt(3.0) / 2.0) for value in side)
    left_back = _unit(tuple(base[index] - offset[index] for index in range(3)))
    right_back = _unit(tuple(base[index] + offset[index] for index in range(3)))
    return left_back, right_back


def _station_direction(name, pivot):
    if name == "Front":
        return _front_direction(pivot)
    if name in ("LeftBack", "RightBack"):
        left_back, right_back = _rear_station_directions(pivot)
        return left_back if name == "LeftBack" else right_back
    return _unit(STATION_DIRECTIONS[name])


def _station_position(direction, pivot, station_radius):
    return tuple(pivot[index] + direction[index] * station_radius for index in range(3))


def _station_tangent_axes(direction):
    """Return orthonormal (cam_right, cam_down) in camera viewing plane.

    Meridian uses OpenCV camera conventions where UP = (0, -1, 0).
    cam_look = -direction (points toward pivot).
    cam_right = unit(UP x cam_look) = unit(UP x -direction).
    cam_down = unit(cam_right x direction).
    """
    up = (0.0, -1.0, 0.0)
    tangent = (
        up[1] * direction[2] - up[2] * direction[1],
        up[2] * direction[0] - up[0] * direction[2],
        up[0] * direction[1] - up[1] * direction[0],
    )
    t_len = math.sqrt(sum(value * value for value in tangent))
    if t_len < 1e-8:
        cam_right = (1.0, 0.0, 0.0)
    else:
        cam_right = tuple(value / t_len for value in tangent)

    down = (
        cam_right[1] * direction[2] - cam_right[2] * direction[1],
        cam_right[2] * direction[0] - cam_right[0] * direction[2],
        cam_right[0] * direction[1] - cam_right[1] * direction[0],
    )
    d_len = math.sqrt(sum(value * value for value in down))
    if d_len < 1e-8:
        cam_down = (0.0, 1.0, 0.0)
    else:
        cam_down = tuple(value / d_len for value in down)

    return cam_right, cam_down


def _station_tangent(direction):
    """Horizontal tangent (camera-right) for the O plane."""
    cam_right, _ = _station_tangent_axes(direction)
    return cam_right


def _interpolate_direction(start, end, fraction):
    """Spherical interpolation, including a deterministic route for antipodal stations."""
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(start, end))))
    if dot > 0.999999:
        return _unit(tuple(a + fraction * (b - a) for a, b in zip(start, end)))
    if dot < -0.999999:
        basis = min(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), key=lambda v: abs(sum(a * b for a, b in zip(start, v))))
        perpendicular = _unit((
            start[1] * basis[2] - start[2] * basis[1],
            start[2] * basis[0] - start[0] * basis[2],
            start[0] * basis[1] - start[1] * basis[0],
        ))
        angle = math.pi * fraction
        return _unit(tuple(math.cos(angle) * a + math.sin(angle) * b for a, b in zip(start, perpendicular)))

    angle = math.acos(dot)
    denominator = math.sin(angle)
    wa = math.sin((1.0 - fraction) * angle) / denominator
    wb = math.sin(fraction * angle) / denominator
    return _unit(tuple(wa * a + wb * b for a, b in zip(start, end)))


def _orbit_position(direction, phase_degrees, pivot, diameter, station_radius):
    """Project the O-orbit onto the station camera's viewing plane (cam_right and cam_down)."""
    center = _station_position(direction, pivot, station_radius)
    cam_right, cam_down = _station_tangent_axes(direction)
    radius = diameter * 0.5
    phase = math.radians(phase_degrees)
    cos_p = math.cos(phase)
    sin_p = math.sin(phase)
    return [
        center[0] + radius * (cos_p * cam_right[0] + sin_p * cam_down[0]),
        center[1] + radius * (cos_p * cam_right[1] + sin_p * cam_down[1]),
        center[2] + radius * (cos_p * cam_right[2] + sin_p * cam_down[2]),
    ]


def _dolly_radius(station_radius, dolly):
    """Apply the path dolly to the source-camera-to-pivot radius.

    The dolly shifts the whole path along the view axis in median-depth units:
    positive values move the camera further away from the pivot (zoom out),
    negative values closer. The result must stay safely outside the pivot so
    the depth reprojection keeps valid source views.
    """
    dolly = _finite(dolly, "Path Dolly")
    radius = station_radius + dolly
    if radius < 0.15:
        raise ValueError("Path Dolly must keep the camera at least 0.15 median-depth units from the pivot.")
    return radius


def _yaw_direction(pivot, yaw_degrees):
    """Horizontal direction from the pivot toward a camera at ``yaw_degrees``.

    Azimuth convention (degrees): 0 lies in front of the subject - on the line
    from the pivot back to the source camera - +90 is the subject's right, +180
    behind it, and negative angles swing to the subject's left. The direction is
    the source-camera "front" direction rotated about the world Y (down) axis
    and projected onto the horizontal plane.
    """
    yaw = math.radians(_finite(yaw_degrees, "Yaw"))
    front = _front_direction(pivot)
    angle = -yaw  # rotate the front direction toward the subject's right
    axis = (0.0, 1.0, 0.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cross = (
        axis[1] * front[2] - axis[2] * front[1],
        axis[2] * front[0] - axis[0] * front[2],
        axis[0] * front[1] - axis[1] * front[0],
    )
    dot = sum(axis[index] * front[index] for index in range(3))
    rotated = tuple(
        front[index] * cos_a + cross[index] * sin_a + axis[index] * dot * (1.0 - cos_a)
        for index in range(3)
    )
    horizontal = (rotated[0], 0.0, rotated[2])
    if math.hypot(horizontal[0], horizontal[2]) < 1e-8:
        # A straight up/down front (never valid for a pivot) keeps a
        # deterministic azimuth reference.
        horizontal = (math.sin(yaw), 0.0, -math.cos(yaw))
    return _unit(horizontal)


def _sweep_direction(pivot, yaw_degrees, elevation_degrees):
    """Direction from the pivot toward a camera on the given azimuth/elevation arc.

    Elevation is the standard altitude angle measured from the pivot's
    horizontal plane: 0 is level with the pivot, positive values place the
    camera above it (looking down), negative values below it (looking up).
    """
    elevation = _finite(elevation_degrees, "Elevation")
    if abs(elevation) > MAX_ELEVATION_DEGREES:
        raise ValueError(
            f"Elevation angles must stay within +/-{MAX_ELEVATION_DEGREES:g} degrees; beyond that "
            "the look direction is vertical gimbal lock for Meridian's zero-roll up vector."
        )
    horizontal = _yaw_direction(pivot, yaw_degrees)
    up = (0.0, -1.0, 0.0)  # world up in the OpenCV frame (y points down)
    cos_e, sin_e = math.cos(math.radians(elevation)), math.sin(math.radians(elevation))
    return _unit(tuple(cos_e * horizontal[index] + sin_e * up[index] for index in range(3)))


def _pendulum_elevation(first, second, fraction):
    """Cosine-eased elevation blend: the camera eases out at every arc apex."""
    return first + (second - first) * (1.0 - math.cos(math.pi * fraction)) / 2.0


def _order_selected_stations(selected_orbits):
    """Determine the optimal trajectory station visitation order.

    The horizontal sweep turns clockwise around the object (Front, Left,
    LeftBack, Back, RightBack, Right) before moving up and down. Rules:
    - Exactly {'Front', 'Left', 'Right'}: ['Right', 'Front', 'Left'].
    - When both 'Left' and 'Right' are selected:
      - If 'Back' is present: the full sweep Front, Left, LeftBack, Back,
        RightBack, Right, Up, Down (filtered).
      - Else if 'Up' is present: Front, Left, LeftBack, Up, RightBack, Right,
        Down (filtered).
      - Else: Front, Left, LeftBack, RightBack, Right, Down (filtered).
    - Default canonical order: the full clockwise sweep (filtered).
    """
    selected_set = set(selected_orbits)
    if selected_set == {"Front", "Left", "Right"}:
        return ["Right", "Front", "Left"]
    if "Left" in selected_set and "Right" in selected_set:
        if "Back" in selected_set:
            primary = ["Front", "Left", "LeftBack", "Back", "RightBack", "Right", "Up", "Down"]
        elif "Up" in selected_set:
            primary = ["Front", "Left", "LeftBack", "Up", "RightBack", "Right", "Down"]
        else:
            primary = ["Front", "Left", "LeftBack", "RightBack", "Right", "Down"]
        return [station for station in primary if station in selected_set]
    return [station for station in VISIT_ORDER if station in selected_set]


def _rotate_to_start(ordered, start_station):
    """Rotate the cyclic visit order so ``start_station`` runs first."""
    if start_station in (None, "", "Visit order"):
        return ordered
    if start_station in ordered:
        index = ordered.index(start_station)
        return ordered[index:] + ordered[:index]
    print(
        f"[Meridian Camera Path (Enndee)] Start station '{start_station}' is not "
        "selected - using the visit order instead."
    )
    return ordered


def _round_position(position):
    result = []
    for value in position:
        value = round(float(value), 6)
        result.append(0.0 if abs(value) < 0.0000005 else value)
    return result


def _allocate_segment_frames(frames, distances, minimum_intervals):
    """Allocate integer frame durations by distance while reserving a frame per key interval."""
    minimum_total = sum(minimum_intervals)
    intervals_available = int(frames) - 1
    if intervals_available < minimum_total:
        raise ValueError(
            f"{frames} frames are not enough for {len(distances)} smooth path segments; "
            f"use at least {minimum_total + 1} frames."
        )

    extra = intervals_available - minimum_total
    total_distance = sum(distances)
    if total_distance <= 0:
        raise ValueError("The generated camera path must have positive travel distance.")

    exact = [extra * distance / total_distance for distance in distances]
    allocated = [math.floor(value) for value in exact]
    remainder = extra - sum(allocated)
    order = sorted(range(len(exact)), key=lambda index: exact[index] - allocated[index], reverse=True)
    for index in order[:remainder]:
        allocated[index] += 1
    return [minimum + value for minimum, value in zip(minimum_intervals, allocated)]


def _key_time_offsets(duration, divisions):
    """Integer frame offsets for evenly timed keys, one per phase division.

    Meridian keys must land on whole frames, so a segment whose duration is not
    divisible by its division count alternates short and long gaps. Scaling each
    key's phase by its rounded offset keeps camera speed even in real time;
    keying fixed 45-degree phases onto alternating gaps makes the short gaps
    sprint at roughly double speed, which reads as a jump in the preview.
    """
    offsets = []
    for division in range(1, divisions + 1):
        offset = round(division * duration / divisions)
        previous = offsets[-1] if offsets else 0
        offsets.append(max(offset, previous + 1))
    return offsets


def build_meridian_custom_camera(number_of_frames, selected_orbits, orbit_diameter, pivot_x, pivot_y, pivot_z,
                                 start_station="Visit order", dolly=0.0):
    """Return JSON text for O loops at the selected named camera stations.

    ``start_station`` rotates the optimized cyclic visit order so the chosen
    station runs the first O loop; "Visit order" (or a station that is not
    selected) keeps the default order. ``dolly`` shifts the whole path along
    the view axis (positive = further from the pivot = zoom out); the O loops
    keep their diameter on the shifted sphere.
    """
    frames = int(number_of_frames)
    if frames not in {int(option) for option in CAMERA_FRAME_OPTIONS}:
        raise ValueError(f"Unsupported Meridian frame count {frames}; choose {', '.join(CAMERA_FRAME_OPTIONS)}.")

    if isinstance(selected_orbits, str):
        selected_orbits = [value.strip() for value in selected_orbits.split(",") if value.strip()]
    if not isinstance(selected_orbits, (list, tuple, set)):
        raise ValueError(
            f"Choose at least one orbit station from {', '.join(ORBIT_OPTIONS[:-1])} and {ORBIT_OPTIONS[-1]}."
        )
    unknown = set(selected_orbits) - set(ORBIT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown orbit station(s): {', '.join(sorted(unknown))}.")
    selected = _rotate_to_start(_order_selected_stations(selected_orbits), start_station)
    orbit_count = len(selected)
    if not selected:
        raise ValueError("Select at least one O-orbit station.")

    diameter = _finite(orbit_diameter, "O-Orbit Diameter")
    if diameter <= 0:
        raise ValueError("O-Orbit Diameter must be greater than zero.")
    if diameter > 2.0:
        raise ValueError("O-Orbit Diameter must not exceed 2 median-depth units.")

    pivot = [
        _finite(pivot_x, "Pivot X"),
        _finite(pivot_y, "Pivot Y"),
        _finite(pivot_z, "Pivot Z"),
    ]
    if any(abs(value) > 5.0 for value in pivot):
        raise ValueError("Pivot X, Y, and Z must each be between -5 and 5 median-depth units.")
    if math.hypot(pivot[0], pivot[2]) < 0.15:
        raise ValueError("Pivot X/Z must place the subject in front of the source camera; try Pivot Z between 0.15 and 5.")

    station_radius = math.sqrt(sum(value * value for value in pivot))
    if station_radius < 0.15:
        raise ValueError("Pivot must be at least 0.15 median-depth units from the source camera.")
    station_radius = _dolly_radius(station_radius, dolly)
    directions = [_station_direction(name, pivot) for name in selected]
    loop_distance = math.pi * diameter
    distances = []
    for index in range(orbit_count):
        distances.append(loop_distance)
        if index < orbit_count - 1:
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(directions[index], directions[index + 1]))))
            distances.append(station_radius * math.acos(dot))
    minimum_intervals = [8 if index % 2 == 0 else CONNECTOR_STEPS for index in range(len(distances))]
    segment_frames = _allocate_segment_frames(frames, distances, minimum_intervals)

    keys = []

    def add_key(position, t):
        key = {
            "pos": _round_position(position),
            "look": [round(value, 6) for value in pivot],
            "src": int(t),
            "t": int(t),
        }
        if keys and key["t"] <= keys[-1]["t"]:
            raise ValueError("Camera-path key times must be strictly increasing; increase the frame count or reduce orbit count.")
        keys.append(key)

    t = 0
    add_key(_orbit_position(directions[0], LOOP_KEY_PHASES[0], pivot, diameter, station_radius), t)

    loop_divisions = len(LOOP_KEY_PHASES) - 1
    for station_index, (station_name, direction) in enumerate(zip(selected, directions)):
        loop_duration = segment_frames[2 * station_index]
        segment_start = keys[-1]["t"]
        for offset in _key_time_offsets(loop_duration, loop_divisions):
            phase_degrees = 360.0 * offset / loop_duration
            t = segment_start + offset
            add_key(_orbit_position(direction, phase_degrees, pivot, diameter, station_radius), t)

        if station_index == orbit_count - 1:
            continue

        segment_start = keys[-1]["t"]
        transfer_duration = segment_frames[2 * station_index + 1]
        next_direction = directions[station_index + 1]
        for offset in _key_time_offsets(transfer_duration, CONNECTOR_STEPS):
            direction_between = _interpolate_direction(direction, next_direction, offset / transfer_duration)
            t = segment_start + offset
            add_key(_orbit_position(direction_between, 0.0, pivot, diameter, station_radius), t)

    if keys[-1]["t"] != frames - 1:
        raise ValueError(f"Generated path ends at frame {keys[-1]['t']}, expected {frames - 1}.")

    document = {
        "name": f"Meridian {orbit_count} evenly spaced vertical O-orbits ({frames} frames)",
        "description": (
            f"Vertical O-orbits at the selected stations in order: {', '.join(selected)}. LeftBack "
            "and RightBack sit 120 degrees from the front direction and from each other. The camera "
            "moves between selected stations and stops after the last O. Unchecked stations are "
            "skipped. O diameter "
            f"{diameter:g} median-depth units. Look pivot [{pivot[0]:g},{pivot[1]:g},{pivot[2]:g}]. "
            "Station radius is the source-camera-to-pivot distance, keeping Front at the source "
            "camera. Path uses real-time source indexing (src=t). With a single repeated still, "
            "non-front views are synthetic depth reprojections, not observed geometry."
        ),
        "frames": frames,
        "stations": selected,
        "path": keys,
    }
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def build_meridian_height_sweep(number_of_frames, start_yaw, target_yaw, low_elevation,
                                high_elevation, arc_switches, pivot_x, pivot_y, pivot_z,
                                start_arc="Low arc", dolly=0.0):
    """Return JSON text for an alternating-height sweep around the look pivot.

    The camera sweeps its azimuth from ``start_yaw`` to ``target_yaw`` while the
    elevation oscillates between the ``low_elevation`` and ``high_elevation``
    arcs - a pendulum that eases out at every apex - switching sides
    ``arc_switches`` times on the way. The camera keeps the
    source-camera-to-pivot radius (the same sphere the O loops use) and always
    looks at the pivot. ``dolly`` shifts the whole sweep away from the pivot
    along the view axis (positive = zoom out).
    """
    frames = int(number_of_frames)
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(f"Unsupported Meridian frame count {frames}; choose {', '.join(CAMERA_FRAME_OPTIONS)}.")

    start = _finite(start_yaw, "Start Yaw")
    target = _finite(target_yaw, "Target Yaw")
    if abs(start) > 360.0 or abs(target) > 360.0:
        raise ValueError("Start Yaw and Target Yaw must stay between -360 and 360 degrees.")
    if abs(target - start) < 1e-6:
        raise ValueError("Start Yaw and Target Yaw must differ; the sweep has no travel.")
    span = target - start

    low = _finite(low_elevation, "Low Arc Elevation")
    high = _finite(high_elevation, "High Arc Elevation")
    if abs(low) > MAX_ELEVATION_DEGREES or abs(high) > MAX_ELEVATION_DEGREES:
        raise ValueError(
            f"Low and High Arc Elevation must stay within +/-{MAX_ELEVATION_DEGREES:g} degrees; "
            "vertical gimbal lock makes Meridian's zero-roll up vector unstable beyond that."
        )
    if low >= high:
        raise ValueError("Low Arc Elevation must sit below High Arc Elevation.")

    switches = int(arc_switches)
    if switches < 1:
        raise ValueError("Arc Switches must be at least 1.")

    pivot = [
        _finite(pivot_x, "Pivot X"),
        _finite(pivot_y, "Pivot Y"),
        _finite(pivot_z, "Pivot Z"),
    ]
    if any(abs(value) > 5.0 for value in pivot):
        raise ValueError("Pivot X, Y, and Z must each be between -5 and 5 median-depth units.")
    if math.hypot(pivot[0], pivot[2]) < 0.15:
        raise ValueError("Pivot X/Z must place the subject in front of the source camera; try Pivot Z between 0.15 and 5.")

    station_radius = math.sqrt(sum(value * value for value in pivot))
    if station_radius < 0.15:
        raise ValueError("Pivot must be at least 0.15 median-depth units from the source camera.")
    station_radius = _dolly_radius(station_radius, dolly)

    if start_arc not in ARC_OPTIONS:
        raise ValueError(f"Start Arc must be {ARC_OPTIONS[0]} or {ARC_OPTIONS[1]}.")
    first_elevation = low if start_arc == ARC_OPTIONS[0] else high
    second_elevation = high if first_elevation == low else low

    # Every switch segment covers the same azimuth span and the same elevation
    # delta, so equal distances share the frame budget; SWEEP_SUBSTEPS analytic
    # keys per segment keep the eased pendulum inside Catmull-Rom.
    segment_frames = _allocate_segment_frames(frames, [1.0] * switches, [SWEEP_SUBSTEPS] * switches)

    keys = []

    def add_key(yaw_degrees, elevation_degrees, t):
        direction = _sweep_direction(pivot, yaw_degrees, elevation_degrees)
        position = tuple(pivot[index] + direction[index] * station_radius for index in range(3))
        key = {
            "pos": _round_position(position),
            "look": [round(value, 6) for value in pivot],
            "src": int(t),
            "t": int(t),
        }
        if keys and key["t"] <= keys[-1]["t"]:
            raise ValueError(
                "Camera-path key times must be strictly increasing; increase the frame count "
                "or reduce the arc switch count."
            )
        keys.append(key)

    add_key(start, first_elevation, 0)
    elevation = first_elevation
    for segment_index in range(switches):
        duration = segment_frames[segment_index]
        segment_start = keys[-1]["t"]
        next_elevation = second_elevation if elevation == first_elevation else first_elevation
        for offset in _key_time_offsets(duration, SWEEP_SUBSTEPS):
            fraction = offset / duration
            yaw = start + span * (segment_start + offset) / (frames - 1)
            add_key(yaw, _pendulum_elevation(elevation, next_elevation, fraction), segment_start + offset)
        elevation = next_elevation

    if keys[-1]["t"] != frames - 1:
        raise ValueError(f"Generated path ends at frame {keys[-1]['t']}, expected {frames - 1}.")

    document = {
        "name": f"Meridian alternating-height sweep ({frames} frames, {switches} arc switches)",
        "description": (
            f"Azimuth sweep from {start:g} to {target:g} degrees (0 front, +90 the subject's right, "
            f"180 back) while the camera elevation alternates between the {low:g} and {high:g} degree "
            f"arcs {switches} time(s), starting on the {start_arc.lower()}. Elevation 0 is level with "
            "the look pivot, positive looks down from above, negative looks up from below. The camera "
            "keeps the source-camera-to-pivot radius and always looks at the pivot. Path uses "
            "real-time source indexing (src=t). With a single repeated still, non-front views are "
            "synthetic depth reprojections, not observed geometry."
        ),
        "frames": frames,
        "stations": ["HeightSweep"],
        "path": keys,
    }
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def build_meridian_spiral_sweep(number_of_frames, start_yaw, target_yaw, start_elevation,
                                end_elevation, pivot_x, pivot_y, pivot_z, dolly=0.0):
    """Return JSON text for a monotone spiral sweep around the look pivot.

    Coverage- and speed-optimised alternative to the alternating-height
    pendulum: the azimuth and the elevation both advance monotonically from the
    start to the target, so every part of the path shows new surface (no
    re-covered azimuth bands), the route stays the shortest possible across the
    azimuth/elevation envelope, and the camera keeps a near-constant angular
    speed (no apex accelerations -> fewer synthesis artefacts). Choose Start and
    Target Yaw far enough apart (and/or different elevations) so the first and
    last frames do not show the same surface - the front region drifts slightly
    over a longer run and should only be covered once, early. ``dolly`` shifts
    the sweep away from the pivot along the view axis (positive = zoom out).
    """
    frames = int(number_of_frames)
    if frames not in {int(value) for value in CAMERA_FRAME_OPTIONS}:
        raise ValueError(f"Unsupported Meridian frame count {frames}; choose {', '.join(CAMERA_FRAME_OPTIONS)}.")

    start = _finite(start_yaw, "Start Yaw")
    target = _finite(target_yaw, "Target Yaw")
    if abs(start) > 360.0 or abs(target) > 360.0:
        raise ValueError("Start Yaw and Target Yaw must stay between -360 and 360 degrees.")
    if abs(target - start) < 1e-6:
        raise ValueError("Start Yaw and Target Yaw must differ; the sweep has no travel.")
    span = target - start

    start_angle = _finite(start_elevation, "Start Elevation")
    end_angle = _finite(end_elevation, "End Elevation")
    if abs(start_angle) > MAX_ELEVATION_DEGREES or abs(end_angle) > MAX_ELEVATION_DEGREES:
        raise ValueError(
            f"Start and End Elevation must stay within +/-{MAX_ELEVATION_DEGREES:g} degrees; vertical "
            "gimbal lock makes Meridian's zero-roll up vector unstable beyond that."
        )
    rise = end_angle - start_angle

    pivot = [
        _finite(pivot_x, "Pivot X"),
        _finite(pivot_y, "Pivot Y"),
        _finite(pivot_z, "Pivot Z"),
    ]
    if any(abs(value) > 5.0 for value in pivot):
        raise ValueError("Pivot X, Y, and Z must each be between -5 and 5 median-depth units.")
    if math.hypot(pivot[0], pivot[2]) < 0.15:
        raise ValueError("Pivot X/Z must place the subject in front of the source camera; try Pivot Z between 0.15 and 5.")

    station_radius = math.sqrt(sum(value * value for value in pivot))
    if station_radius < 0.15:
        raise ValueError("Pivot must be at least 0.15 median-depth units from the source camera.")
    station_radius = _dolly_radius(station_radius, dolly)

    keys = []
    # The last key lands on frames - 1; _key_time_offsets keeps every key on a
    # whole frame, so the even key spacing gives a constant angular speed.
    for t in [0] + _key_time_offsets(frames - 1, SPIRAL_KEY_COUNT - 1):
        fraction = t / (frames - 1)
        direction = _sweep_direction(pivot, start + span * fraction, start_angle + rise * fraction)
        position = tuple(pivot[index] + direction[index] * station_radius for index in range(3))
        keys.append({
            "pos": _round_position(position),
            "look": [round(value, 6) for value in pivot],
            "src": int(t),
            "t": int(t),
        })

    document = {
        "name": f"Meridian spiral sweep ({frames} frames)",
        "description": (
            f"Monotone spiral from yaw {start:g} to {target:g} degrees and elevation {start_angle:g} to "
            f"{end_angle:g} degrees (0 front, +90 the subject's right, 180 back; elevation 0 is level with "
            "the look pivot, positive looks down from above). Azimuth and height both advance in one "
            "direction, so the camera never re-covers a band, keeps a near-constant speed (lowest artefact "
            "risk) and ends away from the start surface. The camera keeps the source-camera-to-pivot radius "
            "and always looks at the pivot. Path uses real-time source indexing (src=t). With a single "
            "repeated still, non-front views are synthetic depth reprojections, not observed geometry."
        ),
        "frames": frames,
        "stations": ["SpiralSweep"],
        "path": keys,
    }
    return json.dumps(document, separators=(",", ":"), allow_nan=False)

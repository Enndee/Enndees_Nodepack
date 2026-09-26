"""ComfyUI parameter builder for Meridian's VGGT geometry node."""

import math
import shlex

from enndee_meridian_camera_path import (
    ARC_OPTIONS,
    CAMERA_MODE_OPTIONS,
    CAMERA_SIGNAL_TYPE,
    HEIGHT_SWEEP_MODE,
    ORBIT_OPTIONS,
    SPIRAL_SWEEP_MODE,
    START_STATION_OPTIONS,
    build_meridian_custom_camera,
    build_meridian_height_sweep,
    build_meridian_spiral_sweep,
)


OUTPUT_FRAME_OPTIONS = ["73", "90", "107", "124", "141", "158", "175", "243"]

# Camera-path (configurator) widgets; every other widget is forwarded to
# build_meridian_arguments. Note that "path_mode" belongs to the argument
# builder (camera ramp mode) and is deliberately not part of this tuple.
PATH_WIDGET_NAMES = (
    "path_orbit_front",
    "path_orbit_left",
    "path_orbit_right",
    "path_orbit_back",
    "path_orbit_up",
    "path_orbit_down",
    "path_orbit_left_back",
    "path_orbit_right_back",
    "path_start_station",
    "path_orbit_diameter",
    "path_pivot_x",
    "path_pivot_y",
    "path_pivot_z",
    "path_camera_mode",
    "path_start_yaw",
    "path_target_yaw",
    "path_low_elevation",
    "path_high_elevation",
    "path_arc_switches",
    "path_first_arc",
    "path_spiral_start_elevation",
    "path_spiral_end_elevation",
    "path_dolly",
)


def _float_text(value):
    """Format a float compactly while avoiding negative zero and scientific notation."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Meridian camera values must be finite numbers.")
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _portable_path(value):
    """Use forward slashes so Windows paths survive shlex.split on Windows."""
    return str(value).strip().replace("\\", "/")


def build_meridian_arguments(
    output_frames,
    source_start,
    freeze_source,
    freeze_frame,
    freeze_full_output,
    freeze_length,
    yaw_from,
    yaw_to,
    truck,
    boom,
    dolly,
    zoom,
    pivot_enabled,
    pivot_x,
    pivot_y,
    pivot_lock,
    aim,
    pivot_to_enabled,
    pivot_to_x,
    pivot_to_y,
    path_mode,
    ease,
    live_speed,
    fast_back,
    follow,
    smooth,
    cull,
    diagnostics_only,
    seed,
    camera_path,
    canvas_enabled,
    canvas_width,
    canvas_height,
    full_enabled,
    full_size,
    vggt_repo,
    vggt_checkpoint,
    custom_camera=False,
):
    """Build the argument string consumed by ``MeridianGeometry.args``.

    With ``custom_camera`` enabled the freeze window, source start, follow and
    the authored camera move / pivot options are left out: the Meridian Geometry
    node repeats the first frame and strips those flags whenever a custom camera
    path is connected, so they would have no effect.
    """
    frames = int(output_frames)
    if frames not in {int(value) for value in OUTPUT_FRAME_OPTIONS}:
        raise ValueError(f"Unsupported Meridian output length {frames}; choose one of {', '.join(OUTPUT_FRAME_OPTIONS)}.")

    if custom_camera:
        source_start = 0
        freeze_source = False
        freeze_frame = 0
        freeze_full_output = True
        freeze_length = 1
        follow = False
        camera_path = ""

    source_start = int(source_start)
    freeze_frame = int(freeze_frame)
    if source_start < 0 or freeze_frame < 0:
        raise ValueError("Source start and freeze frame must be zero or greater.")

    if freeze_source:
        if freeze_frame < source_start:
            raise ValueError("Freeze frame must be greater than or equal to Source Start.")
        # Source frames before the freeze remain as live lead-in output frames.
        freeze_length = frames - (freeze_frame - source_start) if freeze_full_output else int(freeze_length)
        if freeze_length < 1:
            raise ValueError("Freeze length must be at least one frame.")
        tail = frames - freeze_length - (freeze_frame - source_start)
        if tail < 0:
            raise ValueError(
                f"Freeze {freeze_frame}:{freeze_length} does not fit in a {frames}-frame output "
                f"starting at source frame {source_start}."
            )

    camera_path = _portable_path(camera_path)
    if camera_path and (freeze_source or follow):
        raise ValueError("A camera-path JSON requires Freeze Source and Follow Source Camera to be off.")
    if follow and freeze_source:
        raise ValueError("Follow Source Camera requires Freeze Source to be off and real source footage.")
    if path_mode not in ("Sweep", "Bounce", "Swing", "No ramp"):
        raise ValueError(f"Unknown camera ramp mode: {path_mode}")
    if pivot_to_enabled and not aim and not custom_camera:
        raise ValueError("Enable Aim Camera before using the secondary pivot target.")

    args = ["--frames", str(frames)]
    if source_start:
        args.extend(["--start", str(source_start)])
    if freeze_source:
        args.extend(["--freeze", f"{freeze_frame}:{freeze_length}"])

    # Camera-path and follow modes supply their own camera motion, and a custom
    # camera path overrides all authored motion. Do not add offsets those modes
    # ignore.
    if not custom_camera and not follow and not camera_path:
        args.extend(["--yaw-from", _float_text(yaw_from), "--yaw", _float_text(yaw_to)])
        args.extend(["--truck", _float_text(truck), "--boom", _float_text(boom)])
        args.extend(["--dolly", _float_text(dolly)])
        if float(zoom) != 0.0:
            args.extend(["--zoom", _float_text(zoom)])

        if freeze_source and path_mode == "No ramp":
            raise ValueError("A freeze window always ramps camera offsets. Choose Sweep, Bounce, or Swing.")
        ramp_flags = {"Sweep": "--sweep", "Bounce": "--bounce", "Swing": "--swing"}
        if path_mode in ramp_flags:
            if freeze_source and path_mode != "Sweep":
                # With a freeze, --sweep enables camera motion over any live
                # lead-in/tail; bounce/swing then shape the camera ramp.
                args.append("--sweep")
            args.append(ramp_flags[path_mode])
        if ease:
            args.append("--ease")
        args.extend(["--live-speed", _float_text(live_speed)])
        if float(fast_back) > 1.0:
            args.extend(["--fast-back", _float_text(fast_back)])

    if not custom_camera:
        if pivot_enabled:
            args.extend(["--pivot", f"{_float_text(pivot_x)},{_float_text(pivot_y)}"])
        if pivot_enabled and pivot_lock:
            args.append("--pivot-lock")
        if aim:
            args.append("--aim")
        if pivot_to_enabled:
            args.extend(["--pivot-to", f"{_float_text(pivot_to_x)},{_float_text(pivot_to_y)}"])

        if follow:
            args.append("--follow")
            args.extend(["--smooth", _float_text(smooth)])
    if cull:
        args.append("--cull")
    if diagnostics_only:
        args.append("--gauge-only")

    args.extend(["--seed", str(int(seed))])

    if camera_path:
        args.extend(["--camera-path", camera_path])
    if canvas_enabled:
        width, height = int(canvas_width), int(canvas_height)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError("Geometry canvas width and height must be positive multiples of 32.")
        args.extend(["--canvas", f"{width}x{height}"])
    if full_enabled:
        size = int(full_size)
        if size < 128 or size % 32:
            raise ValueError("VGGT source square size must be at least 128 and a multiple of 32.")
        args.extend(["--full", str(size)])

    if str(vggt_repo).strip():
        args.extend(["--vggt-repo", _portable_path(vggt_repo)])
    if str(vggt_checkpoint).strip():
        args.extend(["--vggt", _portable_path(vggt_checkpoint)])

    # MeridianGeometry uses shlex.split() to pass this text to sample.py.
    # shlex.join preserves Windows paths containing spaces through that roundtrip.
    return shlex.join(args)


class MeridianParameterPickerEnndee:
    """Pick Meridian geometry settings and emit a ready-to-connect args string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_frames": (OUTPUT_FRAME_OPTIONS, {
                    "default": "73",
                    "tooltip": "Total reference/output frames. Meridian has matching prompt assets only for these lengths.",
                }),
                "source_start": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "First source-video frame used. Leave at 0 for a still image.",
                }),
                "freeze_source": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Hold one source frame while the virtual camera moves. Enable for a still image.",
                }),
                "freeze_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Source frame index to hold. A still image uses frame 0.",
                }),
                "freeze_full_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, hold the chosen source frame for all output frames remaining after any live lead-in. Disable to use Freeze Length.",
                }),
                "freeze_length": ("INT", {
                    "default": 73, "min": 1, "max": 243, "step": 1,
                    "tooltip": "Number of output frames to hold when Freeze Full Output is disabled. For stills, normally match Output Frames.",
                }),
                "yaw_from": ("FLOAT", {
                    "default": -15.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Starting orbit angle in degrees. Positive yaw moves the camera to the left.",
                }),
                "yaw_to": ("FLOAT", {
                    "default": 15.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "Ending/held orbit angle in degrees. Example: 90 to -90 moves from the left side to the right through the front.",
                }),
                "truck": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Sideways camera shift, in units of pivot depth. Positive moves the camera right.",
                }),
                "boom": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Vertical camera shift, in units of pivot depth. Positive raises the camera.",
                }),
                "dolly": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05,
                    "tooltip": "Final orbit radius as a fraction of pivot depth. Default 1 keeps the radius unchanged; focal length compensates to keep pivot size similar.",
                }),
                "zoom": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Optional final focal multiplier. 0 leaves Meridian's automatic dolly focal scaling in charge; 1 means no optical zoom.",
                }),
                "pivot_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Choose a point in the crop as the 3D orbit pivot. If disabled, Meridian uses the scene's median depth.",
                }),
                "pivot_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Horizontal pivot position in normalized crop coordinates: 0 is left, 0.5 center, 1 right.",
                }),
                "pivot_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical pivot position in normalized crop coordinates: 0 is top, 0.5 center, 1 bottom.",
                }),
                "pivot_lock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Orbit around the selected 3D pivot so it stays approximately fixed on screen. Requires Pivot Enabled.",
                }),
                "aim": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-aim after boom/truck/dolly so the pivot remains on screen.",
                }),
                "pivot_to_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "With Aim Camera enabled, pan/tilt toward a second point that will land at frame center.",
                }),
                "pivot_to_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Horizontal secondary look-at point in normalized crop coordinates.",
                }),
                "pivot_to_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical secondary look-at point in normalized crop coordinates.",
                }),
                "path_mode": (["Sweep", "Bounce", "Swing", "No ramp"], {
                    "default": "Sweep",
                    "tooltip": "Sweep moves start-to-end; Bounce goes out and returns; Swing oscillates to both sides; No ramp holds the final offsets.",
                }),
                "ease": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Ease camera motion at the beginning/end of the ramp so it starts and stops gently.",
                }),
                "live_speed": ("FLOAT", {
                    "default": 0.33, "min": 0.01, "max": 5.0, "step": 0.01,
                    "tooltip": "With Freeze + Sweep, speed of camera motion over live lead-in/tail footage relative to the frozen section.",
                }),
                "fast_back": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Values above 1 move faster through the middle half (typically the unobserved back) of a yaw sweep. 1 disables this timing change.",
                }),
                "follow": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Replay the camera movement estimated from a real source video instead of authored yaw/truck/boom offsets. Not for still-image orbiting.",
                }),
                "smooth": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "With Follow enabled, Gaussian smoothing sigma in frames for the estimated camera path. 0 disables smoothing.",
                }),
                "cull": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove surfaces viewed from behind. Unseen areas become holes rather than mirrored surfaces.",
                }),
                "diagnostics_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run geometry/camera diagnostics and stop before reference videos are exported. Enable only for diagnostics; downstream H3 nodes will not receive images.",
                }),
                "seed": ("INT", {
                    "default": 1234, "min": 0, "max": 2147483647, "step": 1,
                    "tooltip": "Random seed used by Meridian's geometry/inference process.",
                }),
                "camera_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional JSON camera-path file. Advanced: requires real-time source footage; cannot be combined with Freeze or Follow.",
                }),
                "canvas_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Override the automatic 768-class geometry canvas. Use multiples of 32; larger canvases need more memory.",
                }),
                "canvas_width": ("INT", {
                    "default": 864, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Custom geometry render width. Used only when Custom Canvas is enabled.",
                }),
                "canvas_height": ("INT", {
                    "default": 1184, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Custom geometry render height. Used only when Custom Canvas is enabled.",
                }),
                "full_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Override VGGT's 1280-pixel square source reconstruction size. Higher values use more memory; raise it when also increasing the custom canvas.",
                }),
                "full_size": ("INT", {
                    "default": 1280, "min": 128, "max": 4096, "step": 32,
                    "tooltip": "VGGT square source reconstruction side length, in pixels. Used only when Custom VGGT Source Size is enabled.",
                }),
                "vggt_repo": ("STRING", {
                    "default": "",
                    "tooltip": "Optional VGGT-Omega source-code folder. Leave empty to use the Meridian Geometry node's configured/default installation.",
                }),
                "vggt_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "Optional VGGT-Omega checkpoint file (.pt). Leave empty to use the Meridian Geometry node's configured/default checkpoint.",
                }),
                "use_custom_camera": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Drive Meridian Geometry with the generated camera path below (O orbits, an alternating-height pendulum or a spiral sweep). While enabled, the freeze/start/follow and authored camera-move options are hidden and left out of the args (the Geometry node overrides them), and the custom_camera output is filled.",
                }),
                "path_orbit_front": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Camera path: do an O loop on the front/source-camera side of the object.",
                }),
                "path_orbit_left": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Camera path: do an O loop on the object's left side.",
                }),
                "path_orbit_right": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Camera path: do an O loop on the object's right side.",
                }),
                "path_orbit_back": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Camera path: do an O loop behind the object, opposite the front/source camera.",
                }),
                "path_orbit_up": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Camera path: do an O loop above the object looking down onto the target pivot.",
                }),
                "path_orbit_down": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Camera path: do an O loop below the object looking up at the target pivot.",
                }),
                "path_orbit_left_back": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Camera path: do an O loop behind-left, 120 degrees around from the front direction.",
                }),
                "path_orbit_right_back": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Camera path: do an O loop behind-right, 120 degrees from the front direction and from LeftBack.",
                }),
                "path_start_station": (list(START_STATION_OPTIONS), {
                    "default": START_STATION_OPTIONS[0],
                    "tooltip": "Camera path: which selected station starts the path. 'Visit order' keeps the optimized sweep; picking a station rotates the cycle so it runs first.",
                }),
                "path_orbit_diameter": ("FLOAT", {
                    "default": 0.28, "min": 0.02, "max": 2.0, "step": 0.02,
                    "tooltip": "Camera path: full diameter of each O loop in median-depth units. Used by O Orbits only; the sweep styles keep the source-camera-to-pivot radius.",
                }),
                "path_pivot_x": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Camera path look-pivot X in frame-0 camera coordinates (0 is image center).",
                }),
                "path_pivot_y": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Camera path look-pivot Y in frame-0 camera coordinates (positive is down).",
                }),
                "path_pivot_z": ("FLOAT", {
                    "default": 1.0, "min": 0.15, "max": 5.0, "step": 0.01,
                    "tooltip": "Camera path look-pivot depth in median-depth units; must be in front of the source camera.",
                }),
                "path_camera_mode": (list(CAMERA_MODE_OPTIONS), {
                    "default": CAMERA_MODE_OPTIONS[0],
                    "tooltip": "Camera-path style. 'O Orbits': closed vertical O loops at the selected stations. 'Alternating Height': sweep the azimuth while the elevation pendulum-swings between the Low and High Arc (zig-zag styling). 'Spiral Sweep': monotone azimuth + elevation sweep - the shortest route across the full envelope, so it moves slowest and steadiest and shows the most new surface per frame.",
                }),
                "path_start_yaw": ("FLOAT", {
                    "default": -90.0, "min": -360.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Sweep start azimuth in degrees (used by both sweep styles). 0 = in front of the subject (toward the source camera), +90 = the subject's right, 180 = behind, 270 = the subject's left; values are unwrapped multiples of the angle, so 270 and -90 point the same way.",
                }),
                "path_target_yaw": ("FLOAT", {
                    "default": 90.0, "min": -360.0, "max": 360.0, "step": 5.0,
                    "tooltip": "Sweep end azimuth in degrees. The camera travels from Start Yaw to this angle; the sign of the difference sets the turn direction, and up to 360 degrees means up to a full extra turn (e.g. 0 -> 270 covers three quarters of the subject without returning to the start surface).",
                }),
                "path_low_elevation": ("FLOAT", {
                    "default": -30.0, "min": -70.0, "max": 70.0, "step": 5.0,
                    "tooltip": "Alternating Height: lower arc elevation (altitude angle). 0 = level with the pivot, negative = below it looking up, positive = above it looking down. Mathematically the minimum is -90 (straight below), but the arcs stop at -70 like the Up/Down O stations, because a fully vertical look direction is gimbal lock for Meridian's zero-roll orientation.",
                }),
                "path_high_elevation": ("FLOAT", {
                    "default": 30.0, "min": -70.0, "max": 70.0, "step": 5.0,
                    "tooltip": "Alternating Height: upper arc elevation (altitude angle). 0 = level with the pivot, positive = above it looking down, negative = below it looking up. Mathematically the maximum is +90 (straight above), but the arcs stop at +70 like the Up/Down O stations, because a fully vertical look direction is gimbal lock for Meridian's zero-roll orientation.",
                }),
                "path_arc_switches": ("INT", {
                    "default": 3, "min": 1, "max": 12, "step": 1,
                    "tooltip": "Alternating Height: how many times the camera flips between the low and high arc on its way from the start to the target yaw. An odd count finishes on the opposite arc, an even count returns to the starting arc.",
                }),
                "path_first_arc": (list(ARC_OPTIONS), {
                    "default": ARC_OPTIONS[0],
                    "tooltip": "Alternating Height: which arc the sweep starts on.",
                }),
                "path_spiral_start_elevation": ("FLOAT", {
                    "default": -20.0, "min": -70.0, "max": 70.0, "step": 5.0,
                    "tooltip": "Spiral Sweep: elevation (altitude angle) at the start of the sweep. 0 = level with the pivot, positive = above it looking down, negative = below it looking up. May sit above or below the end elevation - the spiral descends when it does.",
                }),
                "path_spiral_end_elevation": ("FLOAT", {
                    "default": 45.0, "min": -70.0, "max": 70.0, "step": 5.0,
                    "tooltip": "Spiral Sweep: elevation at the end of the sweep. Give it a different height than the start so the last frames show the subject from a new angle instead of repeating the start surface.",
                }),
                "path_dolly": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 3.0, "step": 0.05,
                    "tooltip": "Camera path: shifts the whole path along the view axis, in median-depth units. Positive values pull the camera back (zoom out a bit), negative push it closer. Applies to every path style - the sweeps orbit at the new distance and the O loops keep their diameter on the shifted sphere. The args Dolly is ignored while a custom path is connected, so use this one.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", CAMERA_SIGNAL_TYPE)
    RETURN_NAMES = ("args", "custom_camera")
    OUTPUT_TOOLTIPS = (
        "Meridian sample.py arguments for the Geometry node's args_override input.",
        "Custom O-orbit camera path for the Geometry node's custom_camera input; filled while Use Custom Camera is on (empty string otherwise) and always as long as Output Frames.",
    )
    FUNCTION = "build"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Unified Meridian configurator: builds the Geometry arguments AND an optional custom camera path "
        "(vertical O orbits at named stations, an alternating-height pendulum, or a monotone spiral sweep). "
        "With 'Use Custom Camera' enabled the path group appears and the freeze/author-move options disappear, "
        "matching what Meridian Geometry actually applies. The path length always follows Output Frames."
    )

    def build(self, **kwargs):
        """Emit the argument string plus the camera-path signal while enabled."""
        use_custom_camera = bool(kwargs["use_custom_camera"])
        args_values = {
            name: value for name, value in kwargs.items()
            if name not in PATH_WIDGET_NAMES and name != "use_custom_camera"
        }
        signal = ""
        if use_custom_camera:
            mode = kwargs["path_camera_mode"]
            if mode == HEIGHT_SWEEP_MODE:
                signal = build_meridian_height_sweep(
                    kwargs["output_frames"],
                    kwargs["path_start_yaw"],
                    kwargs["path_target_yaw"],
                    kwargs["path_low_elevation"],
                    kwargs["path_high_elevation"],
                    kwargs["path_arc_switches"],
                    kwargs["path_pivot_x"],
                    kwargs["path_pivot_y"],
                    kwargs["path_pivot_z"],
                    kwargs["path_first_arc"],
                    dolly=kwargs["path_dolly"],
                )
            elif mode == SPIRAL_SWEEP_MODE:
                signal = build_meridian_spiral_sweep(
                    kwargs["output_frames"],
                    kwargs["path_start_yaw"],
                    kwargs["path_target_yaw"],
                    kwargs["path_spiral_start_elevation"],
                    kwargs["path_spiral_end_elevation"],
                    kwargs["path_pivot_x"],
                    kwargs["path_pivot_y"],
                    kwargs["path_pivot_z"],
                    dolly=kwargs["path_dolly"],
                )
            else:
                selected_orbits = [
                    name
                    for name, enabled in zip(
                        ORBIT_OPTIONS,
                        (kwargs["path_orbit_front"], kwargs["path_orbit_left"], kwargs["path_orbit_right"],
                         kwargs["path_orbit_back"], kwargs["path_orbit_up"], kwargs["path_orbit_down"],
                         kwargs["path_orbit_left_back"], kwargs["path_orbit_right_back"]),
                    )
                    if enabled
                ]
                signal = build_meridian_custom_camera(
                    kwargs["output_frames"],
                    selected_orbits,
                    kwargs["path_orbit_diameter"],
                    kwargs["path_pivot_x"],
                    kwargs["path_pivot_y"],
                    kwargs["path_pivot_z"],
                    kwargs["path_start_station"],
                    dolly=kwargs["path_dolly"],
                )
        return (build_meridian_arguments(**args_values, custom_camera=use_custom_camera), signal)
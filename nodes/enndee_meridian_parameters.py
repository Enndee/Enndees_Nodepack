"""Meridian Parameters and Camera (Enndee): the geometry arguments plus the camera path.

One node replaces the old split between the parameter picker and the standalone camera-path
configurator. `camera_mode` decides where the path on `custom_camera` comes from:

    Manual      the builders of `enndee_meridian_camera_path` - vertical O orbits at named
                stations, an alternating-height pendulum or a monotone spiral sweep, with the
                look pivot and dolly underneath
    Automatic   `enndee_meridian_auto_camera` estimates the *geometric pivot* of the subject (or
                of the whole scene) from the connected reference image - the midpoint of its depth
                profile, so the depth decides where the camera looks, not the picture centre. The
                pivot is what `auto_pivot_*` offsets, and `auto_path_mode` then picks how to fly:

                  Automatic   a big front O orbit followed by a height lap around the rest (or a
                              big oval over the whole scene), fitted to the camera-speed budget
                  Manual      the manual path widgets below, aimed at the estimated pivot instead
                              of the absolute look-pivot

                Both automatic paths run the collision guard, so no camera key ends up inside the
                scene geometry.

VGGT is gone from this pack: the Meridian Geometry node runs its fast-depth backend only, so the
args string carries nothing but the output length and culling. The freeze/source-start/follow
and authored camera-move options went with it - every run now uses exactly one of the two camera
modes above, and the Geometry node always repeats the first still to the frame count.
"""

import json
import shlex

import torch

from enndee_meridian_auto_camera import (
    AUTO_TARGETS,
    COLLISION_MARGIN,
    DEFAULT_MAX_SPEED,
    SUBJECT_TARGET,
    document_from_keys,
    estimate_camera_path,
    format_summary,
    guard_collisions,
    offset_pivot,
    probe_surface,
)
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

MANUAL_MODE = "Manual"
AUTOMATIC_MODE = "Automatic"
CAMERA_MODES = (MANUAL_MODE, AUTOMATIC_MODE)

AUTOMATIC_PATH = "Automatic"
MANUAL_PATH = "Manual"
AUTO_PATH_MODES = (AUTOMATIC_PATH, MANUAL_PATH)

# The manual path widgets, in widget order; everything else belongs to the argument builder or
# to the automatic mode. Must match the "path_*" names in web/js/enndee_meridian_parameters.js.
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
AUTO_WIDGET_NAMES = ("auto_target", "auto_max_speed", "auto_path_mode",
                     "auto_pivot_x", "auto_pivot_y", "auto_pivot_z")


def build_meridian_arguments(output_frames, cull=False):
    """Build the argument string consumed by ``MeridianGeometry.args_override``.

    The fast-depth backend reads `--frames` (the path always carries the same count) and
    `--cull`; the VGGT-era flags (`--freeze`, `--start`, `--canvas`, `--full`, `--vggt*`, the
    authored camera move and `--seed`) are gone - Meridian Geometry ignores or strips them now.
    """
    frames = int(output_frames)
    if frames not in {int(value) for value in OUTPUT_FRAME_OPTIONS}:
        raise ValueError(
            f"Unsupported Meridian output length {frames}; "
            f"choose one of {', '.join(OUTPUT_FRAME_OPTIONS)}."
        )
    args = ["--frames", str(frames)]
    if cull:
        args.append("--cull")
    # MeridianGeometry uses shlex.split() to pass this text to the fast-depth backend.
    return shlex.join(args)


class MeridianParametersAndCamera:
    """Pick Meridian geometry settings and emit the args string plus the camera path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_frames": (OUTPUT_FRAME_OPTIONS, {
                    "default": "73",
                    "tooltip": "Total reference/output frames and the camera-path length. Meridian has matching prompt assets only for these lengths.",
                }),
                "camera_mode": (list(CAMERA_MODES), {
                    "default": MANUAL_MODE,
                    "tooltip": "Where the camera path comes from. 'Manual' flies the path widgets below around the absolute look-pivot. 'Automatic' estimates the geometric pivot of the subject (or of the whole scene) from the connected reference image - the midpoint of its depth profile, so depth decides where the camera looks instead of the picture centre - and then flies either the estimated path or the manual one around that pivot (see Auto Path Mode).",
                }),
                "auto_target": (list(AUTO_TARGETS), {
                    "default": SUBJECT_TARGET,
                    "tooltip": "Automatic camera: 'subject' pivots and orbits the main subject - the connected subject mask if present, else the near depth layer of the still - while 'scene' treats the whole reconstructed surface as the subject, pivot and all. The estimate prints its pivot, extents and per-frame speed to the console.",
                }),
                "auto_max_speed": ("FLOAT", {
                    "default": round(DEFAULT_MAX_SPEED * 100.0, 1), "min": 1.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Automatic camera: speed cap in percent of the content radius per frame (12 % = the camera may travel 12 % of the subject/scene radius each frame). Lower is safer - too much new surface per frame is what makes the reprojections smear. If the frame count cannot cover the path at this speed, its amplitudes are scaled down automatically and the console line reports the factor.",
                }),
                "auto_path_mode": (list(AUTO_PATH_MODES), {
                    "default": AUTOMATIC_PATH,
                    "tooltip": "Automatic camera: how to fly around the estimated pivot. 'Automatic' builds the estimated path - a big front O orbit that emphasises the front, then a height lap around the rest of the subject (or one big oval over the whole scene), fitted to the speed budget. 'Manual' flies the manual path widgets below - same styles, same settings - but aimed at the estimated pivot instead of the absolute look-pivot. Either way the collision guard keeps the camera out of the scene geometry.",
                }),
                "auto_pivot_x": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Automatic camera: pivot offset X in content radii (frame-0 camera axes, 0 = image centre). The estimate already puts the pivot in the middle of the subject's depth profile - this nudges it, e.g. -0.3 moves the look target a third of the subject radius to the left.",
                }),
                "auto_pivot_y": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Automatic camera: pivot offset Y in content radii (positive is down). Raise or lower the framing midpoint the camera keeps looking at.",
                }),
                "auto_pivot_z": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Automatic camera: pivot offset Z in content radii along the view axis (positive is farther away). Negative pulls the look target toward the source camera, positive pushes it into the subject - useful when the depth profile's middle sits inside a hollow subject.",
                }),
                "cull": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove surfaces viewed from behind. Unseen areas become holes rather than mirrored surfaces.",
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
                    "tooltip": "Camera path look-pivot X in frame-0 camera coordinates (0 is image center). Automatic camera mode replaces this with the estimated pivot plus Auto Pivot X.",
                }),
                "path_pivot_y": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Camera path look-pivot Y in frame-0 camera coordinates (positive is down). Automatic camera mode replaces this with the estimated pivot plus Auto Pivot Y.",
                }),
                "path_pivot_z": ("FLOAT", {
                    "default": 1.0, "min": 0.15, "max": 5.0, "step": 0.01,
                    "tooltip": "Camera path look-pivot depth in median-depth units; must be in front of the source camera. Automatic camera mode replaces this with the estimated pivot plus Auto Pivot Z.",
                }),
                "path_camera_mode": (list(CAMERA_MODE_OPTIONS), {
                    "default": CAMERA_MODE_OPTIONS[0],
                    "tooltip": "Manual camera-path style. 'O Orbits': closed vertical O loops at the selected stations. 'Alternating Height': sweep the azimuth while the elevation pendulum-swings between the Low and High Arc (zig-zag styling). 'Spiral Sweep': monotone azimuth + elevation sweep - the shortest route across the full envelope, so it moves slowest and steadiest and shows the most new surface per frame.",
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
                    "tooltip": "Camera path: shifts the whole path along the view axis, in median-depth units. Positive values pull the camera back (zoom out a bit), negative push it closer. Applies to every path style - the sweeps orbit at the new distance and the O loops keep their diameter on the shifted sphere.",
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Automatic camera mode: the still the path is estimated from - connect the same image the Geometry node receives. The estimate runs once per queue and reuses the fast-depth backend's model cache.",
                }),
                "subject_mask": ("MASK", {
                    "tooltip": "Automatic camera mode (optional): a subject mask from any segmentation node (white = subject). Without it the estimator splits the near depth layer off the background itself.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", CAMERA_SIGNAL_TYPE)
    RETURN_NAMES = ("args", "custom_camera")
    OUTPUT_TOOLTIPS = (
        "Meridian sample.py arguments for the Geometry node's args_override input.",
        "Camera path for the Geometry node's custom_camera input: the manual path widgets or the "
        "estimated automatic orbit, always as long as Output Frames.",
    )
    FUNCTION = "build"
    CATEGORY = "Enndee/Meridian"
    DESCRIPTION = (
        "Meridian Parameters and Camera (Enndee) - one node for the geometry arguments AND the "
        "camera path. Camera Mode picks between a hand-authored path (vertical O orbits at named "
        "stations, an alternating-height pendulum, or a monotone spiral sweep) and an automatic "
        "mode that estimates the *geometric pivot* of the subject - or of the whole scene - from "
        "the reference image's depth profile, then flies either the estimated path (a big front O "
        "orbit plus a height lap around the rest, or one big oval over the scene) or the manual "
        "path around that pivot, always fitted to a camera-speed budget and guarded against "
        "colliding with the scene geometry. VGGT is gone: Meridian Geometry runs its in-process "
        "fast-depth backend only."
    )

    def build(self, reference_image=None, subject_mask=None, **kwargs):
        """Emit the argument string plus the camera-path signal for the selected camera mode."""
        frames = int(kwargs["output_frames"])
        if frames not in {int(value) for value in OUTPUT_FRAME_OPTIONS}:
            raise ValueError(
                f"Unsupported Meridian output length {frames}; "
                f"choose one of {', '.join(OUTPUT_FRAME_OPTIONS)}."
            )
        camera_mode = kwargs["camera_mode"]
        if camera_mode == AUTOMATIC_MODE:
            if reference_image is None:
                raise ValueError(
                    "Automatic camera mode needs the reference image: connect the same still the "
                    "Geometry node receives to the reference_image input."
                )
            signal = self._build_automatic_path(reference_image, subject_mask, kwargs)
        elif camera_mode == MANUAL_MODE:
            signal = self._build_manual_path(kwargs)
        else:
            raise ValueError(f"Unknown Meridian camera mode {camera_mode!r}.")
        return (build_meridian_arguments(frames, kwargs["cull"]), signal)

    @classmethod
    def _build_automatic_path(cls, reference_image, subject_mask, kwargs):
        """Estimate the pivot, then fly the estimated path or the manual one around it."""
        path_mode = kwargs["auto_path_mode"]
        if path_mode not in AUTO_PATH_MODES:
            raise ValueError(f"Unknown automatic Meridian path mode {path_mode!r}.")
        target = kwargs["auto_target"]
        offset = (float(kwargs["auto_pivot_x"]), float(kwargs["auto_pivot_y"]),
                  float(kwargs["auto_pivot_z"]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path_mode == AUTOMATIC_PATH:
            signal, summary = estimate_camera_path(
                reference_image, int(kwargs["output_frames"]), target=target,
                max_speed=float(kwargs["auto_max_speed"]) / 100.0,
                subject_mask=subject_mask, device=device, pivot_offset=offset,
            )
            print(f"[Enndee] Meridian {format_summary(summary)}", flush=True)
            return signal
        frame_count = int(kwargs["output_frames"])
        surface = probe_surface(reference_image, target=target, subject_mask=subject_mask,
                                device=device)
        pivot = offset_pivot(surface, offset)
        aimed = dict(kwargs, path_pivot_x=pivot[0], path_pivot_y=pivot[1], path_pivot_z=pivot[2])
        keys = json.loads(cls._build_manual_path(aimed))["path"]
        keys, fixed, worst_before, worst_after = guard_collisions(keys, surface)
        description = (
            f"Manual path '{kwargs['path_camera_mode']}' around the estimated pivot "
            f"[{pivot[0]:.3g}, {pivot[1]:.3g}, {pivot[2]:.3g}] - the geometric midpoint of the "
            f"{target} depth profile ({surface['source']}), plus the Auto Pivot X/Y/Z offset "
            f"({offset[0]:g}, {offset[1]:g}, {offset[2]:g} content radii). Collision guard: no key "
            f"closer than {COLLISION_MARGIN * 100:.0f} % of the content radius"
            + (f"; {fixed} key(s) pushed clear of the geometry" if fixed else "") + "."
        )
        print(
            f"[Enndee] Meridian manual path on the estimated pivot "
            f"[{pivot[0]:.3g}, {pivot[1]:.3g}, {pivot[2]:.3g}] ({target}, {surface['source']}), "
            f"closest approach {worst_before * 100:.1f} % -> {worst_after * 100:.1f} % of the "
            f"content radius, {fixed} collision fix(es)",
            flush=True,
        )
        return document_from_keys(
            frame_count, keys,
            f"{kwargs['path_camera_mode']} on the estimated pivot ({frame_count} frames)",
            description,
        )

    @staticmethod
    def _build_manual_path(kwargs):
        """The Camera Path Configurator's builders, driven by the path_* widgets."""
        mode = kwargs["path_camera_mode"]
        if mode == HEIGHT_SWEEP_MODE:
            return build_meridian_height_sweep(
                kwargs["output_frames"],
                kwargs["path_start_yaw"], kwargs["path_target_yaw"],
                kwargs["path_low_elevation"], kwargs["path_high_elevation"],
                kwargs["path_arc_switches"], kwargs["path_pivot_x"], kwargs["path_pivot_y"],
                kwargs["path_pivot_z"], kwargs["path_first_arc"],
                dolly=kwargs["path_dolly"],
            )
        if mode == SPIRAL_SWEEP_MODE:
            return build_meridian_spiral_sweep(
                kwargs["output_frames"],
                kwargs["path_start_yaw"], kwargs["path_target_yaw"],
                kwargs["path_spiral_start_elevation"], kwargs["path_spiral_end_elevation"],
                kwargs["path_pivot_x"], kwargs["path_pivot_y"], kwargs["path_pivot_z"],
                dolly=kwargs["path_dolly"],
            )
        selected_orbits = [
            name for name, enabled in zip(
                ORBIT_OPTIONS,
                (kwargs["path_orbit_front"], kwargs["path_orbit_left"], kwargs["path_orbit_right"],
                 kwargs["path_orbit_back"], kwargs["path_orbit_up"], kwargs["path_orbit_down"],
                 kwargs["path_orbit_left_back"], kwargs["path_orbit_right_back"]))
            if enabled
        ]
        return build_meridian_custom_camera(
            kwargs["output_frames"], selected_orbits, kwargs["path_orbit_diameter"],
            kwargs["path_pivot_x"], kwargs["path_pivot_y"], kwargs["path_pivot_z"],
            kwargs["path_start_station"], dolly=kwargs["path_dolly"],
        )

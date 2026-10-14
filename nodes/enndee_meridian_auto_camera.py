"""Estimate an optimal, speed-limited Meridian camera path from one still's surface data.

The Meridian Parameter Picker's automatic camera mode asks this module for a path instead of
letting the user hand-place one. Two targets:

    subject  orbit the main subject - an optional MASK input, else the near depth layer - and
             close the round around it: the framing solves the distance so the subject fills
             `Auto Subject Fill` percent of the picture (and re-centres the pivot on its projected
             midpoint), then the front O-orbit opens on that framed front pose (the middle of the
             O), rises to 2 o'clock (up and right of the top of the O, where the crane used to land) and
             swings around to the subject's side -
             counter-clockwise or clockwise, whichever `Auto Orbit Direction` asks for. From there
             the concluding orbit runs `Auto Orbit End` degrees (360 = all the way back to the start
             point, the middle of the O; the elevation arcs up to a new height at the back and back
             down, so the clip loops)
    scene    a lateral survey: rows of viewpoints spread across the scene's width, each row
             sweeping the elevation from below the horizon to above it, so every side of the
             scene has parallax basis. Deliberately *not* a 360 deg surround - a depth
             reprojection cannot invent the back of a scene, and the frames are better spent on
             the sides a walk-in viewer actually looks from (the drone-mapping lawnmower rule:
             rows with ~70 % side overlap instead of crosshatch/oblique excess)

Both keep the per-frame camera travel under a speed cap. For a **subject** the cap is measured in
the subject's own *pixels* (`max_speed` x its apparent radius per frame, see `subject_drift`) - the
background never enters that budget, so a far background cannot slow the subject down. The front
O-orbit grows above Auto Orbit Size as far as the budget allows (the user's size is a *floor*, so
the front keeps its width), the frames are split between O and lap so both run at the same drift
(`balanced_front_share`), and when the max camera speed still cannot pay for the requested end, the
*concluding orbit* is cut there - the cap wins, the coverage gives way, and the summary names the
levers (more frames, a higher Auto Max Speed, a shorter end). Only when the cap cannot even pay for
the O's own loop does the fit shrink it below Auto Orbit Size; it then maximises the front's width
times the azimuth it covers, so neither the front nor the round becomes useless, and says so too.
The **scene** survey keeps the world-unit cap (`max_speed` x content radius per frame), which also
backs the subject path when no pixel cap is given. Too fast a camera - too much new surface per
frame - is what makes Meridian's depth reprojections smear and flicker; when the frame count cannot
cover the desired swing at that speed, the amplitudes are scaled down and the summary says so.

The surface is unprojected exactly like the fast-depth renderer (frame-0 camera coordinates,
`VFOV_DEGREES`, `x = (u - cx) / f * z`), and the emitted document is the same
`MERIDIAN_CAMERA_PATH` JSON the Camera Path Configurator produces, so the Geometry node and
the fast-depth backend consume it unchanged.

For a *framed subject* the orbit centre is finally **equalised** so the camera-to-subject distance -
the subject's apparent size - stays as constant as the picture allows around the whole path (0/90/
180/270 and start/middle/end): the equaliser moves only the orbit centre, the aim (and with it the
composition) stays exactly on the framing solution, and a move that would crop the subject is scaled
back until every frame clears again. The **Auto Pivot X/Y/Z offsets** translate the finished path
(positions *and* aim) and are applied last, on purpose: the framing, visibility and amplitude solves
can never cancel them - they are the dependable composition nudge.
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
MIN_SUBJECT_PIXELS = 200          # absolute floor for a near layer / mask to count as a subject
MIN_SUBJECT_SHARE = 0.0012        # ... plus 0.12 % of the cloud (a small detail is still a subject)
NO_SUBJECT_SOURCE = "whole surface (no clear near layer)"   # split_subject's "there is no subject"
MASK_COVERAGE_WARN = 0.90          # a mask above this covers (nearly) the frame: it segments nothing
PIVOT_PERCENTILE_LOW = 2.0        # per-axis clip before the bounding-box midpoint (robust pivot)
PIVOT_PERCENTILE_HIGH = 98.0
SUBJECT_FILL = 2.2                # fallback orbit radius / subject radius (Auto Orbit Distance 0)
SCENE_FILL = 2.0                  # stand-off for the scene survey (content radii)
FRONT_ORBIT_RISE = 0.25           # share *of the front loop* spent rising from the framed front
                                  # pose (yaw 0, elevation 0 = the middle of the O) to the sweep's
                                  # start pose, 2 o'clock (upper right of the circle)
# The front O is a CIRCLE: the same angular radius in yaw and elevation, so the reading is equally
# good at every clock position (a 45 deg compromise - a wide flat ellipse showed the sides well but
# the top/bottom poorly). `FRONT_ORBIT_AMPLITUDE` is that radius at scale 1; the speed fit may grow
# it up to `FRONT_ORBIT_LIMIT` (gimbal lock starts at 70 deg, so 60 is the ceiling). The old
# FRONT_YAW_AMPLITUDE / FRONT_ELEVATION / *_LIMIT names are kept as aliases for callers that still
# import them - they now all mean the one circle radius.
FRONT_ORBIT_AMPLITUDE = 45.0      # deg, the front O's radius at scale 1 (yaw AND elevation)
FRONT_ORBIT_LIMIT = 60.0          # deg, the furthest the fit may grow it (gimbal-safe)
FRONT_ORBIT_ANGLE_DEFAULT = FRONT_ORBIT_AMPLITUDE   # deg, the node's "O Orbit Angle" default
FRONT_ORBIT_ANGLE_MIN = 5.0       # deg, below this the front O is barely a loop
FRONT_ORBIT_GROWTH = FRONT_ORBIT_LIMIT / FRONT_ORBIT_AMPLITUDE   # 60/45 - the built-in headroom
FRONT_YAW_AMPLITUDE = FRONT_ORBIT_AMPLITUDE     # deprecated aliases (the O is a circle now)
FRONT_ELEVATION = FRONT_ORBIT_AMPLITUDE
FRONT_YAW_LIMIT = FRONT_ORBIT_LIMIT
FRONT_ELEVATION_LIMIT = FRONT_ORBIT_LIMIT
# Where the O's sweep BEGINS on its own clock. The crane from the middle used to land on 12 (the top
# of the circle); it now lands on FRONT_ORBIT_START_CLOCK = 2 o'clock - 60 deg of clock arc right of
# the top, i.e. a circle angle of 90 - 30 * hour = 30 deg measured from the 3 o'clock axis - and the
# sweep runs the REST of the circle from there in the travel direction: counter-clockwise
# 2 -> 1 -> 12 -> 9 -> 6 -> 3 (330 deg = 360 - the start angle), so the END of the front O - the
# level side the connection takes over from - is exactly where it always was. 'clockwise' mirrors
# the start to 10 o'clock and sweeps 10 -> 11 -> 12 -> ... -> 9.
FRONT_ORBIT_START_CLOCK = 2.0
FRONT_ORBIT_START_DEGREES = 90.0 - 30.0 * FRONT_ORBIT_START_CLOCK   # 30 deg: 2 o'clock on the circle
FRONT_ORBIT_SWEEP_DEGREES = 360.0 - FRONT_ORBIT_START_DEGREES       # 330 deg: from 2 around to 3

# The node's "O Orbit Angle" widget overrides the radius for one estimate. The active value is a
# module global so the deep helpers (`front_amplitudes` / `_amplitude_ceiling`) do not have to
# thread it through every signature; `estimate_camera_path` sets and restores it around the call.
_ACTIVE_FRONT_ORBIT_AMPLITUDE = FRONT_ORBIT_AMPLITUDE


def front_orbit_amplitude():
    """The front O's angular radius (deg) in effect for the estimate being built."""
    return _ACTIVE_FRONT_ORBIT_AMPLITUDE


def resolve_front_orbit_amplitude(degrees):
    """Clamp an O Orbit Angle widget value; None (or <= 0) keeps the built-in radius."""
    if degrees is None:
        return FRONT_ORBIT_AMPLITUDE
    value = _finite(degrees, "O Orbit Angle")
    if value <= 0.0:
        return FRONT_ORBIT_AMPLITUDE
    return min(FRONT_ORBIT_LIMIT, max(FRONT_ORBIT_ANGLE_MIN, value))


# Subject framing: the orbit distance is solved so the subject covers `auto_subject_fill` percent of
# the frame *area* (projected bounding box, 1 %/99 % percentiles plus a silhouette margin), and the
# pivot is re-centred on the subject's projected midpoint. The speed cap is then expressed in the
# subject's own pixels (drift per frame as a fraction of its apparent radius): a nearer/framed
# bigger subject buys a larger step, the background never enters the budget.
SUBJECT_FILL_DEFAULT = 40.0       # percent of the frame area (Auto Subject Fill widget default)
SUBJECT_FILL_MIN = 0.0            # 0 = off: fly at the Auto Orbit Distance / built-in fill
SUBJECT_FILL_MAX = 90.0
SUBJECT_PROJECTION_MARGIN = 1.10  # projected surface points -> visible silhouette allowance
FRAMING_ITERATIONS = 4            # distance fixed-point steps per round (perspective != exactly 1/d^2)
FRAMING_ROUNDS = 3                # distance/re-centre rounds (the shift changes the area a little)
FRAMING_POOL = 20000              # subject points used by the framing / drift metrics
FRAMING_PERCENTILE_LOW = 1.0      # projected bbox clip (stray points must not inflate the subject)
FRAMING_PERCENTILE_HIGH = 99.0
SUBJECT_BOX_PERCENTILE_LOW = 0.2   # per-axis clip of the *whole-subject* box the visibility pass
SUBJECT_BOX_PERCENTILE_HIGH = 99.8  #   keeps inside the frame (stray points out, nothing else cut)
VISIBILITY_MARGIN = 0.02           # share of the frame each border keeps free of the subject box
VISIBILITY_PASSES = 3              # path rebuilds: a pull-back raises the cap -> the amplitude
VISIBILITY_BISECTIONS = 12         # bisection steps of the pull-back ("dolly out until it fits")
VISIBILITY_MAX_SCALE = 8.0         # furthest the visibility pass may pull the camera back
VISIBILITY_SCAN_STEPS = 15         # pivot scan for the worst pose (coarse) ...
VISIBILITY_REFINE_STEPS = 9        # ... and the refinement around its best sample
VISIBILITY_QUARTILES = (0.0, 0.25, 0.5, 0.75, 1.0)   # the frames the report names explicitly
VISIBILITY_POOL = 1500             # subject points per clearance evaluation (per frame, so cheap)
CANVAS_HEIGHT = 1000.0            # canonical pixel height for the projection metrics (ratios only)
DRIFT_POOL = 3000                 # subject points used per drift evaluation (speed: keep it small)
DRIFT_PERCENTILE = 95.0           # per-step drift percentile that has to fit the cap
DRIFT_STEPS = 40                  # steps sampled along the path for the drift measurement
FIT_GROW_STEP = 0.05              # amplitude ladder step (the fit searches the *growth* too)
FULL_CIRCLE = 360.0               # deg, one full round around the subject
FRONT_ORBIT_SHARE_LOW = 0.10      # the balanced frame split never starves either half of the path
FRONT_ORBIT_SHARE_HIGH = 0.92
REST_ELEVATION_HIGH = 38.0        # the height the concluding orbit reaches at the subject's back
LOOP_TRAVEL_FACTOR = 2.0 * math.pi   # the O's arc: 2*pi x A radii per loop of frames (see below)
# The back orbit's clock and arc (see `_back_orbit_point`). It starts at "9 o'clock" - the pose the
# level connection hands over to - and sweeps CLOCKWISE: over the subject's back head ("12"), out to
# the far level point ("3"), down under the back ("6") and back up to "8", one hour short of the
# start so the closing frames never repeat the opening pose. That is 330 deg of the circle, hence the
# travel factor (the O's own loop is the full 2*pi; `BACK_ORBIT_CLOCKS` lists the sampled ticks).
# From 8 o'clock a final GLIDE runs in to the back dial's CENTRE - the level pose straight behind the
# subject, the mirror of where the front O opened (its own circle's middle) - so the clip settles on
# a far view instead of stopping on the clock's lower tick. Its travel is one amplitude (the chord
# from any ring point to the dial centre is exactly A), hence BACK_ORBIT_HOME_FACTOR.
BACK_ORBIT_CLOCKS = (180.0, 90.0, 0.0, -90.0, -150.0)   # 9, 12, 3, 6, 8 o'clock on the back's clock
BACK_ORBIT_CLOCK_START = 180.0                       # the arrival pose the connection ends on
BACK_ORBIT_CLOCK_SWEEP = 330.0                       # ... and the clockwise sweep to 8 o'clock
BACK_ORBIT_TRAVEL_FACTOR = 2.0 * math.pi * (BACK_ORBIT_CLOCK_SWEEP / 360.0)
BACK_ORBIT_HOME_FACTOR = 1.0                         # the 8 o'clock -> dial-centre glide, in O radii

# The orbit's shape. The "concluding orbit" of earlier versions is gone; two knobs replace it:
#   * Auto Orbit View Angle - the azimuth the front O is CENTRED on: 0 = the frontal view towards the
#     subject, +90 = its viewer-left side, 180 = its back, 270/-90 = its right side. The aim stays
#     the pivot, so the subject keeps its place in the picture - only the side the camera visits
#     first moves.
#   * Auto Orbit Coverage - "O-Orbit Front" (the front O alone, ending on the circle's side),
#     "O-Orbit Front&Back" (the front O, the SHORTEST level connection - 180 deg minus the O's own
#     DIAMETER, straight to the back orbit's near edge - and that back orbit: a full clockwise loop
#     from the connection's first contact point, the back clock's "9 o'clock", over the subject's
#     back head ("12"), out to the far level point ("3"), under the back ("6") and back up to "8" -
#     plus a final glide from 8 in to the dial's centre, the level pose straight behind the
#     subject). The frames are split between the phases by their travel, so all of them run at the
#     same per-frame subject drift. When frames x speed cap cannot pay for the whole back part, the
#     BACK gives way (its sweep is cut and the console names it) - the front O is protected.
#     Or "Spiral": a rising helix - the middle of the O, a lead-in along the dial's 2 o'clock
#     direction and then a winding climb to a top-down view (see the SPIRAL_* constants).
ORBIT_VIEW_ANGLE_DEFAULT = 0.0    # deg, 0 = the orbit is centred on the frontal view
ORBIT_VIEW_ANGLE_MIN = -180.0
ORBIT_VIEW_ANGLE_MAX = 360.0
ORBIT_COVERAGES = ("O-Orbit Front", "O-Orbit Front&Back", "Spiral", "Banded Orbits",
                   "Banded Orbits Front & Back")
ORBIT_COVERAGE_DEFAULT = ORBIT_COVERAGES[1]
SPIRAL_COVERAGE = ORBIT_COVERAGES[2]
BAND_COVERAGE = ORBIT_COVERAGES[3]
BAND_FRONT_BACK_COVERAGE = ORBIT_COVERAGES[4]
ORBIT_COVERAGE_DEGREES = 180.0    # deg from the front circle's middle to the back circle's middle
                                  # (the level connection stops one O radius short of this, at the
                                  # back orbit's own edge - see `_back_orbit_point`)
# Predecessor parameter for the standalone legacy choreography (`coverage=None`). The Meridian
# Parameters node now exposes only the explicit coverage selector.
ORBIT_END_DEFAULT = 360.0
ORBIT_END_MIN = 0.0
ORBIT_END_MAX = 720.0
ORBIT_CUT_TARGET = 0.96           # the cut aims a touch *below* the cap: the analytic step lands on
                                  # it, and a path that merely touches the cap still measures a hair
                                  # over it - the margin is what makes the cut rungs *fit*
ORBIT_CUT_STEPS = 3
ORBIT_DIRECTIONS = ("counter-clockwise", "clockwise")
ORBIT_DIRECTION_DEFAULT = ORBIT_DIRECTIONS[0]

# How the camera LOOKS along an automatic path.
#   * 'Pivot point' (the default) aims every pose at the pivot itself: the subject stays
#     centred and its apparent size holds, because the camera orbits the pivot on a
#     sphere and every pose is exactly one radius from it.
#   * 'Horizontally straight' keeps the forward LEVEL: the camera aims at the vertical
#     AXIS through the pivot at its OWN height, so the horizon stays straight and the
#     subject keeps its place horizontally while the camera rides up and down the bands.
#     Because a level look wants a level stand-off, this mode also moves the camera onto
#     a CYLINDER around that axis (see `_place`): the straight (perpendicular) distance
#     to the axis is then the same on every orbit, which is what the subject's width
#     needs to stay constant. The heights the poses visit are unchanged.
ORBIT_LOOK_MODES = ("Pivot point", "Horizontally straight")
ORBIT_LOOK_DEFAULT = ORBIT_LOOK_MODES[0]
HORIZONTAL_LOOK = ORBIT_LOOK_MODES[1]

#: The look mode the estimate being built uses (module state, set by `estimate_camera_path`).
_ACTIVE_LOOK_MODE = ORBIT_LOOK_DEFAULT


def look_mode():
    """How the estimate being built looks along the path - one of `ORBIT_LOOK_MODES`."""
    return _ACTIVE_LOOK_MODE


def resolve_look_mode(mode=None):
    """Validate an `auto_orbit_look` widget value; `None` keeps `ORBIT_LOOK_DEFAULT`."""
    if mode is None:
        return ORBIT_LOOK_DEFAULT
    text = str(mode).strip()
    if text not in ORBIT_LOOK_MODES:
        raise ValueError(f"Unknown orbit look {mode!r}; choose {', '.join(ORBIT_LOOK_MODES)}.")
    return text


# The Spiral coverage (2026-10-05, redefined by the user's spec; 2026-10-06, the O-orbit family):
# the camera travels on a SPHERE around the pivot - the pivot is the centre, the camera keeps its
# distance - along a spiral that is the *family of O-orbits* the Front-only coverage flies, their
# angle growing along the path (the user's sketch: the spiral is drawn in the picture's y-z plane
# and projected onto the sphere, so x - the depth - is the sphere's own; the O-orbit of angle `phi`
# is a circle of radius R sin(phi) in that drawing plane):
#   * `phi` - the O-orbit's angular radius, i.e. the arc between the camera and the view axis (the
#     x axis of the sketch) - grows **linearly along the path** from 0 at the FIRST frame (the
#     camera sits on the axis, looking straight at the picture) to the *end angle* at the last
#     frame. That end angle is the node's "Spiral End Angle" widget (`auto_orbit_angle` in Spiral
#     mode, `SPIRAL_END_ARC_DEFAULT` = 90 deg): 90 puts the camera IN the picture's own plane, the
#     clip ends with a side view of the picture; a smaller value ends the spiral on a fatter O-orbit,
#   * `psi` - the clock angle around that axis (12 = up, 3 = the subject's right, 6 = down,
#     9 = left, i.e. the dial the front O uses) - winds from 0 to the *Spiral Winding* parameter
#     (`SPIRAL_END_DEFAULT` = 840 deg = two and a third rounds), so the winding is the user's
#     choice, not something the fit may trade away.
# Frame 0 is the framed frontal pose (phi = 0, the exact middle of the picture) and the last frame
# is the end of the path (phi = the end angle), whatever the winding. In the module's own `_place`
# terms:
#   yaw = atan2(sin phi * sin psi, cos phi),  elevation = asin(sin phi * cos psi)
# (see `spiral_pose`/`spiral_arc`/`spiral_clock`). Because phi and psi both run linearly to the end,
# the winding alone decides *where* in the picture's plane the last frame looks from: with the
# default 840 deg it is `cos(840 deg) = -0.5`, i.e. a side view 30 deg BELOW the pivot; 810 deg
# would end level on the side (`cos 810 deg = 0`). The console/summary reports the end pose so the
# number can be dialled in. The frames follow the parameter - phi growing evenly along the path is
# exactly the "O-orbits with the angle growing" the spec asks for - and the *keys* the Geometry node
# splines follow the path's own turning (`_spiral_key_frames`), so the rendered path is smooth.
SPIRAL_END_ARC_DEFAULT = 90.0     # deg, the built-in end angle: the picture's own plane (side view)
SPIRAL_END_ARC_MIN = 5.0          # deg, below this the spiral is barely more than its opening pose
SPIRAL_END_ARC_MAX = 90.0         # deg, the picture's own plane is the furthest the axis allows
SPIRAL_END_DEFAULT = 840.0        # deg, the winding around the view axis (2 1/3 rounds)
SPIRAL_END_MIN = 0.0              # deg, no winding: a plain meridian arc from front to side
SPIRAL_END_MAX = 3600.0           # deg, ten rounds - beyond that it is a splat set, not a clip
SPIRAL_ELEVATION_CEILING = 88.0   # deg: the coil passes over the top - keep the up vector sane
# The spiral's CENTRAL AXIS may be tilted (the node's "Spiral Center Slope" widget, its own slot):
# 0 = horizontal (the view axis itself, so the first frame is the framed frontal view), +90 =
# vertical pointing up ("from straight above": the first frame is straight above the subject and the
# spiral then unwinds down to a level orbit), -90 = "from straight below". The tilt is applied in
# the vertical plane through the view axis, so the pivot, the distance and the aim are untouched -
# only the *axis* the O-orbit family winds around moves.
SPIRAL_SLOPE_DEFAULT = 0.0        # deg, the built-in horizontal axis (the view axis)
SPIRAL_SLOPE_MIN = -90.0          # deg, the axis points straight down
SPIRAL_SLOPE_MAX = 90.0           # deg, the axis points straight up
# The keys the renderer splines through are *not* an even frame list for the spiral: it turns
# fastest right after the pole (the clock races while the camera opens the coil - ~130 deg of
# heading change over the first ten frames on the 175 frame example), and an even list cuts that
# corner. Measured on that example (17 evenly spaced keys): the renderer's spline missed the
# intended path by 16 % of the orbit radius at frame 4 and its per-frame travel swung 3.1x. A key
# per SPIRAL_KEY_TURN degrees of *turning* instead keeps the keys where the path bends (dense after
# the pole, sparse on the long outer sweep): 35 keys, a 1.1 % gap and a 1.07x travel spread.
SPIRAL_KEY_TURN = 20.0            # deg of camera turning one spiral key span may cover

# The node's "Spiral Winding" widget travels the same way as the O Orbit Angle: one module global,
# set and restored around the estimate by `estimate_camera_path`.
_ACTIVE_SPIRAL_END = SPIRAL_END_DEFAULT


def spiral_end():
    """The winding (deg) the Spiral coverage flies in the estimate being built."""
    return _ACTIVE_SPIRAL_END


def resolve_spiral_end(degrees=None):
    """Clamp a Spiral Winding widget value; None keeps the built-in winding (`SPIRAL_END_DEFAULT`).

    0 flies the plain meridian arc from the frontal view out to the picture's own plane - a quarter
    circle, no winding - and every full 360 deg adds one round around the view axis.
    """
    if degrees is None:
        return SPIRAL_END_DEFAULT
    return max(SPIRAL_END_MIN, min(SPIRAL_END_MAX, _finite(degrees, "Spiral winding")))


# ... and so does the END ANGLE: the O-orbit family's last radius, the node's "Spiral End Angle"
# widget (the Auto Orbit Angle slot while the Spiral coverage is picked).
_ACTIVE_SPIRAL_END_ARC = SPIRAL_END_ARC_DEFAULT


def spiral_end_arc():
    """The end angle (deg) - how far out of the axis the Spiral coverage gets - being built."""
    return _ACTIVE_SPIRAL_END_ARC


def resolve_spiral_end_arc(degrees=None):
    """Clamp a Spiral End Angle widget value; None keeps the built-in `SPIRAL_END_ARC_DEFAULT`.

    The angle is the radius of the O-orbit family the spiral flies: 0 stays on the view axis (there
    is no path at all), 90 reaches the picture's own plane - the side view of the picture.
    """
    if degrees is None:
        return SPIRAL_END_ARC_DEFAULT
    return max(SPIRAL_END_ARC_MIN, min(SPIRAL_END_ARC_MAX,
                                       _finite(degrees, "Spiral end angle")))


# The spiral's central axis tilt travels exactly like the winding: the node's "Spiral Center Slope"
# widget (the Auto Orbit Angle in Spiral mode) sets it for one estimate.
_ACTIVE_SPIRAL_SLOPE = SPIRAL_SLOPE_DEFAULT


def spiral_slope():
    """The spiral's central axis tilt (deg) in effect for the estimate being built."""
    return _ACTIVE_SPIRAL_SLOPE


def resolve_spiral_slope(degrees=None):
    """Clamp a Spiral Center Slope widget value; None keeps the built-in horizontal axis (0 deg).

    0 is the view axis itself (the spiral starts on the framed frontal view), +90 the vertical axis
    ("from straight above": the spiral opens straight above the subject), -90 the one below it.
    """
    if degrees is None:
        return SPIRAL_SLOPE_DEFAULT
    return max(SPIRAL_SLOPE_MIN, min(SPIRAL_SLOPE_MAX, _finite(degrees, "Spiral slope")))


# --- "Banded Orbits": a stack of circle SEGMENTS on the orbit sphere -------------
# The fourth coverage. Each band is a partial orbit at ONE elevation; the camera
# sweeps a band, steps to the next one at a constant azimuth, sweeps back the other
# way, and so on - a boustrophedon over the sphere. It is the multi-elevation
# capture the other subject paths were missing: a single ring leaves the top and the
# bottom of the subject unconstrained, and this mode visits them in one continuous
# move without ever rolling (see `_band_samples`).
#
#   * Band Start Angle - the elevation of the FIRST band. 0 = level with the pivot,
#     -90 = from straight below (the feet, looking up), +90 = from straight above the
#     head. Clamped to BAND_ELEVATION_LIMIT so the look/up frame stays sane.
#   * Band Count - how many bands (1, 2, 3 ...).
#   * Band Width - the azimuth each band sweeps, 1..360 deg.
#   * Band Start Yaw - the azimuth the FIRST band BEGINS at, measured around the
#     subject (negative = from its left, positive = from its right). The first sweep
#     runs from there towards the LEFT, i.e. with DECREASING yaw.
#   * Band Step - the elevation rise per band; 0 = auto (spread the bands over the
#     full available range, in whichever direction has more room - see
#     `band_elevations`).
#
# Every band's sweep ENDS exactly where the next one BEGINS, so the step between
# two bands is a pure elevation move at ONE azimuth: the "straight up" the spec asks
# for. A 360 deg width ends where it started, so it cannot alternate - it keeps
# turning the same way instead, exactly as the spec requires. The bands live on the
# orbit sphere (`_place`), so the camera keeps one radius for the whole path.
BAND_START_ANGLE_DEFAULT = 0.0    # deg, the built-in first band is level with the pivot
BAND_START_ANGLE_MIN = -90.0      # deg, the farthest below: looking up at the feet
BAND_START_ANGLE_MAX = 90.0       # deg, straight above the head
BAND_COUNT_DEFAULT = 3            # bands, the built-in stack
BAND_COUNT_MIN = 1
BAND_COUNT_MAX = 12               # more bands than this buy nothing at 73-243 frames
BAND_WIDTH_DEFAULT = 180.0        # deg of azimuth one band sweeps (half a turn)
BAND_WIDTH_MIN = 1.0
BAND_WIDTH_MAX = 360.0            # a full turn - and then the bands never alternate
BAND_START_YAW_DEFAULT = 0.0      # deg, the built-in start is the frontal azimuth
BAND_START_YAW_MIN = -360.0
BAND_START_YAW_MAX = 360.0
BAND_STEP_DEFAULT = 0.0
BAND_STEP_MIN = 0.0
BAND_STEP_MAX = 90.0
BAND_STOP_ANGLE_DEFAULT = 60.0    # deg, second endpoint of the evenly spaced band stack
BAND_STOP_ANGLE_MIN = -90.0
BAND_STOP_ANGLE_MAX = 90.0
# Deprecated compatibility API state; the serialized node widget now means stop angle.
_ACTIVE_BAND_STEP = BAND_STEP_DEFAULT
BAND_ELEVATION_LIMIT = 88.0       # deg: keep the look/up frame away from the poles
BAND_MIN_STEP = 5.0               # deg, below this the auto step is degenerate

# --- "Banded Orbits Front & Back": the band stack, mirrored onto the back ---------
# The fifth coverage. It flies the SAME stack as "Banded Orbits" - the same widgets,
# the same choreography - but does not stop at the top: from the end of the HIGHEST
# band the camera carries on to that pose's MIRROR on the back side (every azimuth
# +180 deg, the same elevation) and then walks the stack back DOWN the bands there,
# mirrored and in reverse. Front and back get the identical visit, so the top and the
# bottom of the subject are seen from both sides in one continuous, roll-free move.
#
#   * the FRONT half is exactly `_band_legs` - band 0 (the lowest, at Band Start
#     Angle) first, climbing to the last one,
#   * the top band's sweep is then EXTENDED - the same heading, no turn - until it
#     reaches the mirror of its own start on the far side. That bridge is
#     `BAND_FRONT_BACK_TURN - Band Width` deg (a half turn when the band is 180 deg
#     wide, nothing at all then), so the camera never doubles back and no arc is
#     shown twice,
#   * the BACK half is the SAME bands on the far side (azimuth +180 deg), highest
#     first, each swept the way its front twin was, stepping back down to the lowest.
#
# The frames are still split by the legs' travel on the sphere (the bridge included),
# so the camera holds one angular speed over both halves. Every pose lives on the
# orbit sphere (`_place`), so the radius and the framing never change, and the path is
# levelled to the world up exactly like the plain band stack.
BAND_FRONT_BACK_TURN = 180.0      # deg, the azimuth between a band and its mirror on the back

# The node's band widgets travel the same way as the O Orbit Angle and the spiral
# ones: module globals, set and restored around the estimate by `estimate_camera_path`.
_ACTIVE_BAND_START_ANGLE = BAND_START_ANGLE_DEFAULT
_ACTIVE_BAND_COUNT = BAND_COUNT_DEFAULT
_ACTIVE_BAND_WIDTH = BAND_WIDTH_DEFAULT
_ACTIVE_BAND_START_YAW = BAND_START_YAW_DEFAULT
_ACTIVE_BAND_STEP = BAND_STEP_DEFAULT
_ACTIVE_BAND_STOP_ANGLE = BAND_STOP_ANGLE_DEFAULT


def band_start_angle():
    """Elevation (deg) of the first band in the estimate being built."""
    return _ACTIVE_BAND_START_ANGLE


def band_count():
    """How many bands the estimate being built flies."""
    return _ACTIVE_BAND_COUNT


def band_width():
    """Azimuth (deg) one band sweeps in the estimate being built."""
    return _ACTIVE_BAND_WIDTH


def band_start_yaw():
    """Azimuth (deg) the first band begins at in the estimate being built."""
    return _ACTIVE_BAND_START_YAW


def band_stop_angle():
    """Elevation (deg) of the final band in the estimate being built."""
    return _ACTIVE_BAND_STOP_ANGLE


def band_step():
    """Deprecated compatibility API; returns the legacy per-band step default."""
    return _ACTIVE_BAND_STEP


def resolve_band_start_angle(degrees=None):
    """Clamp a Band Start Angle widget value; None keeps `BAND_START_ANGLE_DEFAULT`."""
    if degrees is None:
        return BAND_START_ANGLE_DEFAULT
    return max(BAND_START_ANGLE_MIN,
               min(BAND_START_ANGLE_MAX, _finite(degrees, "Band start angle")))


def resolve_band_count(count=None):
    """Clamp a Band Count widget value; None keeps `BAND_COUNT_DEFAULT`."""
    if count is None:
        return BAND_COUNT_DEFAULT
    return max(BAND_COUNT_MIN, min(BAND_COUNT_MAX, int(_finite(count, "Band count"))))


def resolve_band_width(degrees=None):
    """Clamp a Band Width widget value; None keeps `BAND_WIDTH_DEFAULT`."""
    if degrees is None:
        return BAND_WIDTH_DEFAULT
    return max(BAND_WIDTH_MIN, min(BAND_WIDTH_MAX, _finite(degrees, "Band width")))


def resolve_band_start_yaw(degrees=None):
    """Clamp a Band Start Yaw widget value; None keeps `BAND_START_YAW_DEFAULT`."""
    if degrees is None:
        return BAND_START_YAW_DEFAULT
    return max(BAND_START_YAW_MIN,
               min(BAND_START_YAW_MAX, _finite(degrees, "Band start yaw")))


def resolve_band_stop_angle(degrees=None):
    """Clamp a Band Stop Angle widget value."""
    if degrees is None:
        return BAND_STOP_ANGLE_DEFAULT
    return max(BAND_STOP_ANGLE_MIN,
               min(BAND_STOP_ANGLE_MAX, _finite(degrees, "Band stop angle")))


def resolve_band_step(degrees=None):
    """Deprecated compatibility helper; band layout now uses start/stop elevations."""
    if degrees is None:
        return BAND_STEP_DEFAULT
    return max(BAND_STEP_MIN, min(BAND_STEP_MAX, _finite(degrees, "Band step")))


def band_elevations(count=None, start=None, stop=None):
    """The elevation (deg) of every band, the first one first.

    Bands are evenly spaced between the configured start and stop elevations; both
    endpoints are clamped away from the poles to keep the camera up vector stable.

    A `None` argument means **the widget the estimate is currently flying**
    (`band_start_angle()` / `band_count()` / `band_stop_angle()`), not the built-in default:
    a caller that passes nothing must get the stack the node asked for.
    """
    count = band_count() if count is None else resolve_band_count(count)
    start = band_start_angle() if start is None else resolve_band_start_angle(start)
    start = max(-BAND_ELEVATION_LIMIT, min(BAND_ELEVATION_LIMIT, start))
    stop = band_stop_angle() if stop is None else resolve_band_stop_angle(stop)
    stop = max(-BAND_ELEVATION_LIMIT, min(BAND_ELEVATION_LIMIT, stop))
    if count <= 1:
        return [start]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


# The direction the estimated clip is *levelled to*: the camera frame keeps this up, so nothing
# rolls about the optical axis. A level path keeps the world up; the spiral leans its own up with
# its central axis (`Spiral Center Slope`), because a coil levelled to the world up swings tens of
# degrees against its own frame - measured 78.6 deg of swing at 60 deg end angle / 30 deg slope,
# which is the roll the spiral showed - and the world up is exactly the direction a sloped coil
# crosses, so levelling to it also made the camera flip there (the world's pole sits at phi = 60 for
# a 30 deg slope). Levelling to the spiral's own up keeps the coil upright in its own frame and
# moves that pole to phi = 90, past every end angle the node allows but 90.
_ACTIVE_ROLL_UP = tuple(fast_depth.WORLD_UP)


def roll_up():
    """The up (3-tuple, frame-0 camera coordinates) the estimate being built is levelled to."""
    return _ACTIVE_ROLL_UP

# Scene target: a lateral survey instead of a lap around the scene. Meridian's scene renders are
# depth reprojections, so what a walk-in VR viewer needs is *side* coverage: rows of viewpoints
# spread across the scene's width at the source viewpoint's distance, each row sweeping the
# elevation from below the horizon to above it. That is the drone-mapping lawnmower pattern
# (boustrophedon rows, 70-80 % side overlap, one oblique pass for the off-nadir sides) mapped onto
# the reprojection camera: no 360 deg surround, every frame buys lateral parallax or a new height.
SCENE_LATERAL_OVERLAP = 0.7       # side-lap between neighbouring rows (drone mapping default)
SCENE_COVERAGE_MARGIN = 1.30      # outer rows sit this many half-widths off centre
SCENE_FILL_MIN = 1.0              # the survey never comes closer than one content radius
SCENE_FILL_MAX = 8.0              # ... and never further out than this (reach/fill solve ceiling)
SCENE_SUBJECT_SHARE_MIN = 0.02    # below this the console says the subject is too small to read
SCENE_LANE_OVERLAP_WARN = 0.30    # rows this far apart (side overlap) leave gaps between them
SCENE_MIN_LANES = 3               # left / centre / right: the minimum for usable side views
SCENE_MAX_LANES = 9               # more rows than this buy nothing at 73-243 frames
SCENE_YAW_LIMIT = 70.0            # deg, the outermost row's azimuth never exceeds this
SCENE_FRAMES_PER_LANE = 4         # frames a row needs so the Catmull-Rom keys can follow it
SCENE_ELEVATION_LOW = -12.0       # rows start just below the horizon (under-surfaces)
SCENE_ELEVATION_HIGH = 28.0       # ... and climb well above it (looking down into the scene)

KEY_TARGET = 17                   # path keys (Catmull-Rom control points)
AMPLITUDE_STEPS = 40              # lambda ladder: 1.0, 0.975, ... 0.025 (2.5 % rungs)
MIN_ORBIT_RADIUS = 0.05           # never place the camera on the content
DEFAULT_MAX_SPEED = 0.12          # fraction of the content radius the camera may travel per frame
COLLISION_MARGIN = 0.15           # of the content radius: how close a camera key may come to geometry
COLLISION_ITERATIONS = 4
COLLISION_MAX_POINTS = 120_000    # stride bigger clouds down for the distance checks
MAX_PIVOT_OFFSET = 1.0            # pivot offset widgets, in content radii

# The pivot equaliser (framed subject paths): after the framing and the visibility pass the orbit
# centre is nudged so the camera-to-subject distance stays as constant as possible along the final
# path - that is what keeps the subject *box* the same size at 0/90/180/270 degrees and at the
# start, the middle and the end of the clip. The spread of |camera - subject centre| over the
# sampled path is minimised with a coarse-to-fine coordinate scan around the solved pivot, pulled
# slightly towards the centre and bounded to EQUALIZER_LIMIT content radii. The visibility
# guarantee wins: a move that crops the subject is bisected down until every frame clears again.
EQUALIZER_CENTRE_WEIGHT = 0.05    # tie-breaker: the preferred pivot is the subject's own centre
EQUALIZER_SCAN_STEPS = 13         # coarse scan candidates per axis ...
EQUALIZER_REFINE_STEPS = 9        # ... and the refinement around the best candidate
EQUALIZER_LIMIT = 0.75            # furthest the equaliser may move the aim, in content radii
EQUALIZER_SAMPLES = 33            # poses sampled along the final path for the spread metric
EQUALIZER_BISECTIONS = 7          # crop fallback: halvings of the move until the path clears
EQUALIZER_CLEARANCE_TOL = 1.0     # px of the solved path's border clearance the move may spend

# The subject cylinder (2026-10-02): the subject is approximated by a vertical cylinder whose axis
# is BOTH the orbit centre and the aim, so the subject sits in the horizontal picture centre at every
# pose (the cylinder is symmetric about its axis, and the axis projects onto the image centre line)
# and the camera keeps ONE constant distance to the pivot for the whole path. The fit trims outliers
# per axis (a lance or a stray depth spike must not decide the shape) and the radius is the
# CYLINDER_COVERAGE percentile of the radial distance in the x-z plane, i.e. ~90-95 % of the surface
# sits inside the cylinder - the rest is treated as outlier and left out of the framing/visibility
# pool.
CYLINDER_PERCENTILE_LOW = 2.5     # per-axis trim of the cylinder centre/height (2.5 %/97.5 %)
CYLINDER_PERCENTILE_HIGH = 97.5
CYLINDER_COVERAGE = 0.95          # share of the surface the cylinder radius encloses (90-95 %)

# The automatic path's user knob (the node's Auto Orbit Size widget). Auto Orbit **Distance** is
# DEPRECATED and ignored: the camera-to-pivot distance now always follows Auto Subject Fill (subject)
# or the survey solve (scene). For a subject more fill = closer camera (see `subject_framing`); the
# old fixed stand-off made the fill and the distance fight each other. `size` multiplies every swing
# amplitude of the path (1.0 = the built-in amplitudes).
ORBIT_DISTANCE_DEFAULT = SUBJECT_FILL
ORBIT_DISTANCE_MIN = 0.5          # closer would sit inside the subject before the collision guard
ORBIT_DISTANCE_MAX = 8.0
ORBIT_SIZE_DEFAULT = 1.0
ORBIT_SIZE_MIN = 0.1              # a 10 % loop, for a tiny bit of parallax
ORBIT_SIZE_MAX = 3.0              # wider than the built-in swing; the speed fit may still cap it



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
    return points, NO_SUBJECT_SOURCE


def subject_points(depth, mask=None):
    """Subject points plus the label describing their source.

    A connected mask is **authoritative**: the pivot (and every subject metric built on it) is
    computed from the mask's points and nothing else. There is deliberately NO silent fallback to
    the depth heuristic while a mask is connected - that fallback is what put the pivot into the
    BACKGROUND: a mask that selects almost nothing (failed segmentation, a muted/empty RMBG run -
    the RMBG node answers an exception with an all-zero mask) used to drop onto `split_subject`,
    which answers "the whole surface" as soon as its Otsu cut collapses, and the camera then
    orbited the centre of the whole cloud instead of the subject. An unusable mask therefore
    raises, so the failure is visible instead of silently changing what "subject" means.
    Without a mask the depth heuristic stays as it was (near layer, whole surface as a labelled
    last resort).
    """
    if mask is None:
        return split_subject(surface_points(depth))
    points = surface_points(depth)
    selected = surface_points(depth, mask=mask)
    needed = _minimum_subject_pixels(points.shape[0])
    if selected.shape[0] < needed:
        raise ValueError(
            f"The subject mask selects only {selected.shape[0]} of {points.shape[0]} pixels "
            f"(at least {int(needed)} are needed to orbit). The estimate does NOT fall back to the "
            f"depth heuristic while a mask is connected, because that fallback orbits the whole "
            f"surface - background included - instead of the subject. Check the mask source "
            f"(white = subject, e.g. the RMBG MASK output), or disconnect it to let the depth "
            f"split choose."
        )
    return selected, "input mask"


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

    The path is sampled at every rung of the ladder, the true camera travel between consecutive
    frames is measured, and the largest scale that fits wins. Every number stays a *ratio* to the
    content radius, so the estimate is scale-, resolution- and gauge-independent. The ladder is not
    walked down with an early break any more: once the lap follows the circle rule the travel is no
    longer monotone in the scale (a narrower O means a longer lap), so all rungs are measured - and
    when none fits, the one with the smallest overshoot is reported instead of the floor, which may
    well be worse. Returns (scale, samples, travel_per_frame).
    """
    best = None
    closest = None
    for step in range(ladder):
        scale = round(1.0 - 0.025 * step, 3)
        samples = samples_of(scale)
        travel = max((math.dist(samples[index - 1], samples[index])
                      for index in range(1, len(samples))), default=0.0)
        if travel <= budget_per_frame:
            if best is None:
                best = (scale, samples, travel)   # the ladder runs down: the first fit is the widest
        elif closest is None or travel < closest[2]:
            closest = (scale, samples, travel)
    if best is not None:
        return best
    if closest is not None:
        return closest
    return 1.0, samples_of(1.0), 0.0


def _place(centre, radius, yaw_degrees, elevation_degrees):
    """Camera position on the orbit surface around `centre`.

    Yaw 0 sits *between* the centre and the source camera (the origin), positive yaw moves to the
    subject's right; elevation 0 is level with the centre, positive is above it (the OpenCV frame
    has y pointing down). That is the Camera Path Configurator's convention (`_yaw_direction`):
    the source camera at the origin is the closest possible viewpoint of the subject.

    The SURFACE follows the active look mode (`look_mode`). 'Pivot point' orbits the
    SPHERE of radius `radius`, so every pose is exactly one radius from the pivot.
    'Horizontally straight' orbits the CYLINDER of that radius around the vertical axis
    through `centre` instead - the horizontal stand-off is `radius` at every elevation
    (the level look's constant subject width) while the height follows the elevation
    exactly as the sphere would have placed it.
    """
    yaw = math.radians(yaw_degrees)
    elevation = math.radians(elevation_degrees)
    if _ACTIVE_LOOK_MODE == HORIZONTAL_LOOK:
        offset = (math.sin(yaw), -math.sin(elevation), -math.cos(yaw))
    else:
        offset = (math.cos(elevation) * math.sin(yaw),
                  -math.sin(elevation),
                  -math.cos(elevation) * math.cos(yaw))
    return [float(centre[axis]) + offset[axis] * radius for axis in range(3)]


def _aim(centre, position):
    """The point a pose aims at: the pivot itself, or the axis point level with the camera.

    'Pivot point' looks at `centre`. 'Horizontally straight' looks at the vertical axis
    through `centre` at the camera's OWN height, which leaves the forward level (its `y`
    component is zero) so the horizon stays straight and the subject keeps its horizontal
    place while the camera rides up and down. Both are expressed as a point, so the whole
    pipeline - the drift and visibility measurements, the framing and the emitted keys -
    can share one aim.
    """
    if _ACTIVE_LOOK_MODE == HORIZONTAL_LOOK:
        return [float(centre[0]), float(position[1]), float(centre[2])]
    return [float(centre[0]), float(centre[1]), float(centre[2])]


def front_camera(pivot, distance):
    """Camera position of the front pose (yaw 0, elevation 0): between the pivot and the origin."""
    return _place(pivot, distance, 0.0, 0.0)


def front_amplitudes(scale=1.0, size=1.0):
    """(yaw, elevation) swing of the front O in deg: ONE circle radius, scale x size.

    The O is a *circle* (`FRONT_ORBIT_AMPLITUDE` both ways), so the two numbers are equal by
    construction - a 45 deg compromise between a wide sweep that only shows the sides and a tall one
    that only shows the top. The tuple shape is kept because the ellipse parameterisation
    (`yaw = A cos t`, `elevation = A sin t`) is exactly what traces that circle. The fit may grow the
    radius up to `FRONT_ORBIT_LIMIT` (gimbal-safe); it never swings past it.
    """
    size = max(1e-3, float(size))
    amplitude = min(FRONT_ORBIT_LIMIT, front_orbit_amplitude() * float(scale) * size)
    return amplitude, amplitude


def lap_span_for_end(yaw_amplitude, end=None):
    """Lap span (deg) that carries the path from the O's side (+A) to `end` degrees around it.

    The concluding orbit always continues where the O ended (its side, +A in the travel direction),
    so the span is simply what is left of the requested end. `end = 360` (the default) therefore
    runs the lap around the back and back to the start point - the middle of the O.
    """
    end = ORBIT_END_DEFAULT if end is None else max(ORBIT_END_MIN, float(end))
    return max(0.0, end - max(0.0, float(yaw_amplitude)))


def direction_mirror(direction=ORBIT_DIRECTION_DEFAULT):
    """+1.0 for a counter-clockwise orbit (the default), -1.0 for clockwise - the yaw's sign."""
    text = str(direction).strip().lower()
    if text == ORBIT_DIRECTION_DEFAULT:
        return 1.0
    if text == ORBIT_DIRECTIONS[1]:
        return -1.0
    raise ValueError(f"Unknown orbit direction {direction!r}; choose {', '.join(ORBIT_DIRECTIONS)}.")


def direction_label(direction=ORBIT_DIRECTION_DEFAULT):
    """'ccw' / 'cw' - the short form the console and QC lines use for the orbit direction."""
    return "ccw" if direction_mirror(direction) > 0.0 else "cw"


def balanced_front_share(yaw_amplitude, lap_span):
    """Share of the frames the front O gets so that O and lap run at the *same* per-frame drift.

    The speed cap is per frame, so the frames are the real currency. The O's arc pays for a whole
    loop of the circle - the crane to 2 o'clock plus the 330 deg sweep around to 3 - at
    `2*pi*A` radii per loop of frames, while the lap
    travels its span on the full circle - splitting the frames in that ratio puts both halves at the
    same subject motion, which is the smallest worst-case drift the path can have at this frame
    count (the speed cap's best case). A path without a lap takes (almost) every frame; a very long
    lap still leaves the O its tenth.
    """
    loop = LOOP_TRAVEL_FACTOR * math.radians(max(0.0, float(yaw_amplitude)))
    lap = math.radians(max(0.0, float(lap_span)))
    if lap <= 1e-9:
        # No concluding orbit: there is nothing to share the frames with, so the O takes them ALL -
        # the clip then ends on 3 o'clock (the O's own last pose) instead of spending the last
        # frames on a lap that has no azimuth left (see `subject_samples`).
        return 1.0
    return min(FRONT_ORBIT_SHARE_HIGH,
               max(FRONT_ORBIT_SHARE_LOW, loop / (loop + lap)))


def _decimate(points, limit):
    """Every n-th point of a cloud, capped to about `limit` samples (metrics need no more)."""
    points = points.reshape(-1, 3)
    stride = max(1, int(math.ceil(points.shape[0] / max(1, int(limit)))))
    return points[::stride]


def _look_axes(points, camera_position, pivot, frame=None):
    """(x, y, z) of `points` in a camera frame - renderer convention (right, down, forward).

    The forward is `_aim(pivot, camera_position) - camera_position`, i.e. the pose's aim
    under the active look mode (the pivot itself, or the level look at the axis). `frame`
    is the (right, down) pair to measure in - pass the levelled pair for a pose inside a
    sequence (`fast_depth.camera_frames`) so the measurement matches the render exactly.
    It defaults to the isolated pose's own pair (zero roll from the estimate's `roll_up` -
    the world up, or a sloped spiral's own up), which is the same `camera_frames` would
    pick: a pixel measured here is a pixel the renderer would produce.
    """
    device, dtype = points.device, points.dtype
    position = torch.as_tensor(camera_position, device=device, dtype=dtype)
    aim = _aim(pivot, camera_position)
    forward = torch.as_tensor(aim, device=device, dtype=dtype) - position
    forward = forward / forward.norm().clamp(min=1e-8)
    if frame is None:
        right_np, down_np = fast_depth.camera_frames(forward.detach().cpu().numpy().reshape(1, 3),
                                                     roll_up())
        right = torch.as_tensor(right_np[0], device=device, dtype=dtype)
        down = torch.as_tensor(down_np[0], device=device, dtype=dtype)
    else:
        right, down = (torch.as_tensor(axis, device=device, dtype=dtype) for axis in frame)
    relative = points - position
    return relative @ right, relative @ down, relative @ forward


def project_points(points, camera_position, pivot, surface, height=CANVAS_HEIGHT, frame=None):
    """(u, v, z) pixels of world `points` in a camera's frame - the renderer's own pinhole.

    `f = 0.5 * height / tan(vfov / 2)`, canvas width from the still's aspect ratio. The canvas
    height is a *canonical* 1000 px: every metric below is a ratio of the frame, so it cancels.
    `frame` is the (right, down) pair to measure in (see `_look_axes`).
    """
    focal = 0.5 * float(height) / math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0)
    aspect = float(surface.get("aspect") or (16.0 / 9.0))
    x_c, y_c, z_c = _look_axes(points, camera_position, pivot, frame)
    width = float(height) * aspect
    safe = z_c.clamp(min=1e-6)
    return 0.5 * width + focal * x_c / safe, 0.5 * float(height) + focal * y_c / safe, z_c


def _projected_box(points, camera_position, pivot, surface, height=CANVAS_HEIGHT):
    """(width, height, centre u, centre v, depth, canvas width) of the projected subject box.

    The box is the 1 %/99 % percentile rectangle of the projected points (stray depth spikes must
    not inflate the subject) plus SUBJECT_PROJECTION_MARGIN, which stands in for the silhouette
    between the sampled surface points.
    """
    u, v, z = project_points(points, camera_position, pivot, surface, height)
    visible = z > 1e-4
    if int(visible.sum()) < 8:
        raise ValueError("The subject ends up behind the camera - nothing to frame.")
    u, v, z = u[visible], v[visible], z[visible]
    low_u = _percentile(u, FRAMING_PERCENTILE_LOW / 100.0)
    high_u = _percentile(u, FRAMING_PERCENTILE_HIGH / 100.0)
    low_v = _percentile(v, FRAMING_PERCENTILE_LOW / 100.0)
    high_v = _percentile(v, FRAMING_PERCENTILE_HIGH / 100.0)
    canvas_width = float(height) * float(surface.get("aspect") or (16.0 / 9.0))
    return (max(1.0, high_u - low_u) * SUBJECT_PROJECTION_MARGIN,
            max(1.0, high_v - low_v) * SUBJECT_PROJECTION_MARGIN,
            (low_u + high_u) / 2.0, (low_v + high_v) / 2.0, float(_percentile(z, 0.5)),
            canvas_width)


def _fill_envelope(points, pivot, distance, surface, size, canvas_area, height):
    """(min, max) projected area share of the subject over the path's swing envelope.

    The framing must not hold at the front pose alone: along the orbit the projected box grows
    (the near side comes closer) and shrinks (the elevation foreshortens it), which pushed the fill
    to 22-40 % around a 40 % target. Sampling the swing envelope (the built-in amplitudes x the
    user's O size, clamped by the view limits) gives the range the user actually gets, and the
    framing targets its geometric mean so the subject stays inside the requested band throughout.

    Only the *front* swing enters the mean: the lap's far side sees the subject at very different
    distances, and mixing those poses into a mean drags the front pose far above the target (74 % for
    a 40 % request, measured). The lap is covered by `_envelope_angles` + `_room_distance` instead -
    a *room* requirement, which can only pull the camera back, never push it in.
    """
    angles = _envelope_angles(size, include_back=False)
    positions = [_place(pivot, distance, yaw, elevation) for yaw, elevation in angles]
    u, v, z = _batch_project(points, positions, pivot, surface, height)
    areas = []
    for index in range(len(angles)):
        visible = z[index] > 1e-4
        if int(visible.sum()) < 8:
            raise ValueError("The subject ends up behind the camera - nothing to frame.")
        u_row, v_row = u[index][visible], v[index][visible]
        width = max(1.0, _percentile(u_row, FRAMING_PERCENTILE_HIGH / 100.0)
                    - _percentile(u_row, FRAMING_PERCENTILE_LOW / 100.0))
        height_box = max(1.0, _percentile(v_row, FRAMING_PERCENTILE_HIGH / 100.0)
                         - _percentile(v_row, FRAMING_PERCENTILE_LOW / 100.0))
        areas.append(width * SUBJECT_PROJECTION_MARGIN * height_box * SUBJECT_PROJECTION_MARGIN
                     / canvas_area)
    return min(areas), max(areas)


def _envelope_angles(size, include_back=True, orbit_end=None, direction=ORBIT_DIRECTION_DEFAULT,
                     view_angle=None, coverage=None, back_span=None):
    """(yaw, elevation) sample list of the whole path's envelope at the amplitude scale `size`.

    The front O's own extremes first (12/3/6/9 o'clock and the corners, offset to its `view_angle`
    centre), then - when the path runs a back part - the connection's azimuths (level, the tightest
    poses in practice: the subject's projected box is tallest at eye level) and, once the connection
    is complete, the back orbit's loop (its 9 o'clock edge, over the back's head, the far level point,
    under the back and the 8 o'clock end) plus the glide home to the back dial's centre - the whole
    visit is what it flies now. The room check and the fill envelope both have to see the tightest
    pose of the finished path, which is on this list.
    """
    amplitude, _ = front_amplitudes(size, 1.0)
    centre = ORBIT_VIEW_ANGLE_DEFAULT if view_angle is None else float(view_angle)
    angles = [(centre + amplitude * yaw_step, amplitude * elevation_step)
              for yaw_step in (-1.0, -0.5, 0.0, 0.5, 1.0)
              for elevation_step in (-1.0, 0.0, 1.0)]
    if coverage is None:
        # Compatibility: the older concluding orbit, whose 38 deg height arc is what needs room.
        span = lap_span_for_end(amplitude, orbit_end)
        if not include_back or span <= 1e-9:
            return angles
        rest_high = min(FRONT_ORBIT_LIMIT, REST_ELEVATION_HIGH * size)
        mirror = direction_mirror(direction)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            angles.append((mirror * (amplitude + span * fraction),
                           rest_high * (1.0 - math.cos(2.0 * math.pi * fraction)) / 2.0))
        return angles
    if not include_back or amplitude >= ORBIT_COVERAGE_DEGREES:
        return angles
    mode = ORBIT_COVERAGE_DEFAULT if coverage is None else str(coverage)
    if coverage is None and orbit_end is not None:
        mode = coverage_for_end(orbit_end)
    if mode == SPIRAL_COVERAGE:
        # The spiral's own poses: the frontal pose at the axis is the tightest (the tallest subject
        # box) and the side view at the end the smallest, so sampling the whole climb covers both
        # ends - the room check and the fill envelope need exactly that.
        return [_spiral_point(fraction, centre, direction_mirror(direction))
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if mode != ORBIT_COVERAGES[1]:
        return angles                  # "O-Orbit Front": no back poses to make room for
    full = max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * amplitude)   # level, to the back orbit's edge
    span = full if back_span is None else min(full, max(0.0, float(back_span)))
    if span <= 1e-9:
        return angles
    mirror = direction_mirror(direction)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        angles.append((centre + mirror * (amplitude + span * fraction), 0.0))
    if span >= full - 1e-9:                # the back orbit runs: its own loop needs room as well
        back = centre + mirror * ORBIT_COVERAGE_DEGREES
        angles += [_back_orbit_point(_back_clock_progress(clock), back, amplitude, mirror)
                   for clock in BACK_ORBIT_CLOCKS]      # 9, 12, 3, 6 and 8 o'clock
        eight_yaw, eight_elevation = _back_orbit_point(1.0, back, amplitude, mirror)
        angles.append((eight_yaw + (back - eight_yaw) * 0.5,   # the glide home: halfway and at the
                       eight_elevation * 0.5))                 # back dial's centre (level, far)
        angles.append((back, 0.0))
    return angles


def _room_distance(pool, surface, pivot, distance, size, height, margin_px, orbit_end=None,
                   direction=ORBIT_DIRECTION_DEFAULT, options=None):
    """Smallest distance >= `distance` that clears the frame at every pose of the whole path.

    The counterpart of the fill: the framing may want a close camera for the requested fill, but the
    O-orbit - and the concluding orbit that runs around the back - need room. Both requirements meet
    here: the returned distance is the larger of the two, so the fill request is never violated
    *upwards* and no pose of the fitted path leaves the frame. A lap that does not fit at this
    distance therefore pulls the camera back (the fill pays) instead of being cut (coverage pays) -
    the *speed* cap is the only thing allowed to shorten the orbit. Bisection works because pulling
    back shrinks every projection towards the frame centre.
    """
    angles = _envelope_angles(size, include_back=True, orbit_end=orbit_end, direction=direction,
                              **(options or {}))

    def worst(factor):
        positions = [_place(pivot, distance * factor, yaw, elevation) for yaw, elevation in angles]
        return min(visibility_clearances(pool, positions, pivot, surface, height, margin_px))

    if worst(1.0) >= 0.0:
        return distance
    if worst(VISIBILITY_MAX_SCALE) < 0.0:
        return distance * VISIBILITY_MAX_SCALE
    low, high = 1.0, VISIBILITY_MAX_SCALE
    for _ in range(VISIBILITY_BISECTIONS):
        middle = 0.5 * (low + high)
        if worst(middle) >= 0.0:
            high = middle
        else:
            low = middle
    return distance * high


def _amplitude_ceiling(size):
    """Largest amplitude scale the view limits allow (the built-in headroom over the O's angle).

    The gimbal-safe ceiling (`FRONT_ORBIT_LIMIT`) fixes how far the fit may grow the O above the
    node's O Orbit Angle, so this ratio is the same whatever angle is set - a smaller angle yields a
    proportionally smaller orbit. One definition for the fit's ladder, the visibility pass and the
    console hint, so a change to the limits can never leave one of them behind.
    """
    return FRONT_ORBIT_GROWTH / max(1e-3, float(size))


def subject_box(surface):
    """(corners [8, 3], extents [3]) - the subject's world-space box, reported for diagnostics.

    Per-axis SUBJECT_BOX_PERCENTILE_LOW/.._HIGH percentiles: a trimmed *minimum/maximum* box, so a
    few stray depth points cannot inflate it. It is the honest "how deep and how wide is this
    subject" number for the console/report (a deep subject, e.g. a mask that spans a table top,
    is what makes the visibility pass work) - the guarantee itself uses the *projected* extent,
    because a box that is deep in world space is usually a thin silhouette in the picture.
    """
    points = surface["content_cloud"].reshape(-1, 3)
    low = [_percentile(points[:, axis], SUBJECT_BOX_PERCENTILE_LOW / 100.0) for axis in range(3)]
    high = [_percentile(points[:, axis], SUBJECT_BOX_PERCENTILE_HIGH / 100.0) for axis in range(3)]
    corners = [[high[axis] if (mask >> axis) & 1 else low[axis] for axis in range(3)]
               for mask in range(8)]
    return (torch.tensor(corners, device=points.device, dtype=points.dtype),
            [high[axis] - low[axis] for axis in range(3)])


def fit_subject_cylinder(surface):
    """(centre [3], radius, coverage, keep) - a vertical cylinder around the subject.

    The subject is approximated by a **cylinder** with a vertical axis (`y` in the frame-0 camera
    frame): the centre is the robust trimmed midpoint of x, y and z (`CYLINDER_PERCENTILE_LOW/_HIGH`),
    the radius the `CYLINDER_COVERAGE` percentile of the radial distance in the x-z plane, so ~95 %
    of the surface sits inside and a lance, a chair leg or a stray depth spike cannot inflate the
    fit. `keep` is the boolean mask of the points inside the cylinder (radius *and* trimmed height),
    i.e. the robust 95 % subject the framing and the visibility guarantee work on.

    The cylinder - not a box - is the right primitive here: the whole path aims at its axis, and a
    rotationally symmetric shape about that axis keeps a *constant* left-right extent whatever the
    azimuth, which is what puts the subject in the horizontal picture centre at every pose.
    """
    points = surface["content_cloud"].reshape(-1, 3)
    low = torch.tensor([_percentile(points[:, axis], CYLINDER_PERCENTILE_LOW / 100.0)
                        for axis in range(3)], dtype=points.dtype, device=points.device)
    high = torch.tensor([_percentile(points[:, axis], CYLINDER_PERCENTILE_HIGH / 100.0)
                         for axis in range(3)], dtype=points.dtype, device=points.device)
    centre = (low + high) / 2.0
    radial = torch.hypot(points[:, 0] - centre[0], points[:, 2] - centre[2])
    # The radius is the CYLINDER_COVERAGE percentile of the radial distance, so a thin spike (a
    # lance, an arm reaching out - a few percent of the points) is outside the percentile and cannot
    # inflate the fit; the per-axis trim above already protects the centre against big structures.
    radius = max(_percentile(radial, CYLINDER_COVERAGE), 1e-6)
    keep = radial <= radius
    coverage = float(keep.float().mean()) if keep.numel() else 0.0
    return centre, radius, coverage, keep


def _batch_project(pool, positions, pivot, surface, height):
    """(u, v, z) of `pool` for *every* pose at once - the same pinhole as `project_points`.

    Returns [poses, points] tensors: the whole path is evaluated in one call instead of one call per
    pose, which is what makes the visibility sweep (dozens of candidate pivots x every frame) cheap.
    Each pose aims through `_aim(pivot, position)`, so the level look of the 'Horizontally
    straight' mode is measured exactly as it is rendered.
    """
    device, dtype = pool.device, pool.dtype
    pos = torch.as_tensor(positions, device=device, dtype=dtype).reshape(-1, 3)
    if _ACTIVE_LOOK_MODE == HORIZONTAL_LOOK:
        # The level look aims at the axis at the camera's OWN height: one aim per pose.
        look = pos.clone()
        look[:, 0] = float(pivot[0])
        look[:, 2] = float(pivot[2])
    else:
        look = torch.as_tensor(pivot, device=device, dtype=dtype).reshape(1, 3)
    forward = look - pos
    forward = forward / forward.norm(dim=1, keepdim=True).clamp(min=1e-8)
    up = torch.tensor(roll_up(), device=device, dtype=dtype).expand_as(forward)
    right = torch.cross(forward, up, dim=1)
    norms = right.norm(dim=1, keepdim=True)
    parallel = (norms < 1e-6).squeeze(1)
    if bool(parallel.any()):                  # looking straight down: keep world x like _look_axes
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(right)
        right = torch.where(parallel.unsqueeze(1), fallback, right)
        norms = right.norm(dim=1, keepdim=True)
    right = right / norms.clamp(min=1e-8)
    down = torch.cross(forward, right, dim=1)
    relative = pool.reshape(1, -1, 3) - pos.unsqueeze(1)
    x = (relative * right.unsqueeze(1)).sum(-1)
    y = (relative * down.unsqueeze(1)).sum(-1)
    z = (relative * forward.unsqueeze(1)).sum(-1)
    focal = 0.5 * float(height) / math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0)
    width = float(height) * float(surface.get("aspect") or (16.0 / 9.0))
    safe = z.clamp(min=1e-6)
    return 0.5 * width + focal * x / safe, 0.5 * float(height) + focal * y / safe, z


def visibility_clearances(points, positions, pivot, surface, height=CANVAS_HEIGHT, margin=None):
    """Per pose: the subject's border room in px - negative entries are cropped frames.

    The subject's pixels are what a viewer sees, so the guarantee is stated in them: the
    0.2 %/99.8 % rectangle of the projected points (times SUBJECT_PROJECTION_MARGIN, the same
    silhouette allowance the fill solve uses) is compared against the borders. A subject that is
    deep in world space but thin in the picture - a plate, a person seen from the side - therefore
    costs nothing, while a subject that really leaves the frame is caught.

    Which borders count depends on the look (`look_mode`). 'Pivot point' checks all four: the
    aim follows the subject, so pulling the camera back can always fix a crop. 'Horizontally
    straight' checks the HORIZONTAL borders only: the level look cannot move the subject
    vertically with distance - the drift it shows is the look's own - so the vertical borders
    are not a distance problem and must not drag the camera back for one (they would bisect
    forever and land on VISIBILITY_MAX_SCALE).
    """
    margin_px = (VISIBILITY_MARGIN * float(height)) if margin is None else float(margin)
    if not len(positions):
        return []
    u, v, z = _batch_project(points, positions, pivot, surface, height)
    visible = z > 1e-4
    # invisible points are pushed out of the way: +inf cannot win the *low* quantile, -inf cannot
    # win the *high* one - so both percentiles are taken over the visible points of each pose
    low_u = torch.quantile(torch.where(visible, u, u.new_full((), math.inf)),
                           SUBJECT_BOX_PERCENTILE_LOW / 100.0, dim=1)
    high_u = torch.quantile(torch.where(visible, u, u.new_full((), -math.inf)),
                            SUBJECT_BOX_PERCENTILE_HIGH / 100.0, dim=1)
    low_v = torch.quantile(torch.where(visible, v, v.new_full((), math.inf)),
                           SUBJECT_BOX_PERCENTILE_LOW / 100.0, dim=1)
    high_v = torch.quantile(torch.where(visible, v, v.new_full((), -math.inf)),
                            SUBJECT_BOX_PERCENTILE_HIGH / 100.0, dim=1)
    half_u = 0.5 * (high_u - low_u).clamp(min=1.0) * SUBJECT_PROJECTION_MARGIN
    half_v = 0.5 * (high_v - low_v).clamp(min=1.0) * SUBJECT_PROJECTION_MARGIN
    centre_u = 0.5 * (low_u + high_u)
    centre_v = 0.5 * (low_v + high_v)
    width = float(height) * float(surface.get("aspect") or (16.0 / 9.0))
    horizontal = torch.minimum(centre_u - half_u - margin_px,
                               (width - margin_px) - (centre_u + half_u))
    if _ACTIVE_LOOK_MODE == HORIZONTAL_LOOK:
        clearance = horizontal
    else:
        vertical = torch.minimum(centre_v - half_v - margin_px,
                                 (float(height) - margin_px) - (centre_v + half_v))
        clearance = torch.minimum(horizontal, vertical)
    too_few = visible.sum(dim=1) < 8          # nothing in front of the camera: not a valid pose
    clearance = torch.where(too_few, clearance.new_full((), -math.inf),
                            torch.nan_to_num(clearance, nan=-math.inf))
    return [float(value) for value in clearance]


def _path_positions(unit_positions, pivot, distance):
    """Poses of a path whose orbit radius is 1.0 -> real positions at `distance` around `pivot`."""
    return [[pivot[axis] + (position[axis] - pivot[axis]) * distance for axis in range(3)]
            for position in unit_positions]


def _centre_for_visibility(pool, unit_positions, pivot, distance, surface, height, margin_px,
                           limit):
    """(pivot, worst clearance) - aim the pivot at the pose that has the *least* room.

    The framing's re-centring optimises the front pose; a guarantee has to hold for the worst pose of
    the whole path (the end of the lap, the top of the swing), so the objective here is the smallest
    clearance over all poses. That objective is not monotone in the shift - a shift that helps one
    pose hurts another - so each axis is scanned coarse-to-fine instead of bisected. `limit` is one
    content radius at most: the aim moves from one side of the subject to the other, never off it.
    """
    def score(value, axis, base):
        probe = list(base)
        probe[axis] = value
        positions = _path_positions(unit_positions, probe, distance)
        return min(visibility_clearances(pool, positions, probe, surface, height, margin_px))

    result = list(pivot)
    best_score = min(visibility_clearances(
        pool, _path_positions(unit_positions, result, distance), result, surface, height,
        margin_px))
    for axis in (0, 1):
        best = float(result[axis])
        span = max(float(limit), 1e-6)
        for steps in (VISIBILITY_SCAN_STEPS, VISIBILITY_REFINE_STEPS):
            step_size = 2.0 * span / max(1, steps - 1)
            for index in range(steps):
                value = best + (index - (steps - 1) / 2.0) * step_size
                candidate = score(value, axis, result)
                if candidate > best_score:
                    best_score, best = candidate, value
            span = 2.0 * step_size
        result[axis] = best
    return result, best_score


def _dolly_for_visibility(pool, unit_positions, pivot, distance, surface, height, margin_px):
    """Smallest factor k >= 1 with the subject clear at every pose of the k * d path.

    Pulling the camera back shrinks the projection towards the frame centre, so "the subject fits"
    is monotone in k and the factor can be bisected instead of searched. VISIBILITY_MAX_SCALE comes
    back when even that is not enough - the caller reports the shortfall instead of cropping.
    """
    def worst(factor):
        positions = _path_positions(unit_positions, pivot, distance * factor)
        return min(visibility_clearances(pool, positions, pivot, surface, height, margin_px))

    if worst(1.0) >= 0.0:
        return 1.0
    if worst(VISIBILITY_MAX_SCALE) < 0.0:
        return VISIBILITY_MAX_SCALE
    low, high = 1.0, VISIBILITY_MAX_SCALE
    for _ in range(VISIBILITY_BISECTIONS):
        middle = 0.5 * (low + high)
        if worst(middle) >= 0.0:
            high = middle
        else:
            low = middle
    return high


def _path_fits(pool, surface, pivot, distance, scale, size, frames, height, margin_px,
               orbit_end=None, direction=ORBIT_DIRECTION_DEFAULT, options=None):
    """Does the subject stay inside the frame at every pose of this amplitude?"""
    units = subject_samples(frames, pivot, 1.0, scale, size, orbit_end, direction,
                            motion_pool=pool, motion_surface=surface, **(options or {}))
    positions = _path_positions(units, pivot, distance)
    return min(visibility_clearances(pool, positions, pivot, surface, height, margin_px)) >= 0.0


def _shrink_amplitude(pool, surface, pivot, distance, scale, size, frames, height, margin_px,
                      orbit_end=None, direction=ORBIT_DIRECTION_DEFAULT, options=None):
    """Largest amplitude <= `scale` whose whole path keeps the subject visible.

    Shrinking is the *cheap* fix: the front pose - and with it the requested fill - does not move at
    all, only the viewing angles narrow. Pulling the camera back is the expensive one (it shrinks the
    subject everywhere), so the pass always tries this first. Bisection is enough because the
    clearance falls with the swing: a wider loop moves the subject further towards the border.

    When *no* swing keeps the subject in frame - not even `FIT_GROW_STEP` - the crop cannot be the
    front O's at all (its poses are the ones that move with the amplitude; the offending frames sit
    in the concluding orbit, whose poses barely change with it). Shrinking then buys nothing, so the
    incoming `scale` is kept and the caller's dolly does the work: a shrink that cannot fit must
    never be allowed to cost the user their front orbit.
    """
    scale = max(FIT_GROW_STEP, float(scale))
    if _path_fits(pool, surface, pivot, distance, scale, size, frames, height, margin_px, orbit_end,
                  direction, options):
        return scale
    low, high = FIT_GROW_STEP, scale
    if not _path_fits(pool, surface, pivot, distance, low, size, frames, height, margin_px,
                      orbit_end, direction, options):
        return scale                      # no swing fits: keep the O, the dolly has to help
    for _ in range(VISIBILITY_BISECTIONS):
        middle = 0.5 * (low + high)
        if _path_fits(pool, surface, pivot, distance, middle, size, frames, height, margin_px,
                      orbit_end, direction, options):
            low = middle
        else:
            high = middle
    return low


def enforce_subject_visibility(surface, pivot, distance, scale, size, frames, height=CANVAS_HEIGHT,
                               orbit_end=None, direction=ORBIT_DIRECTION_DEFAULT, pool=None,
                               recentre=True, view_angle=None, coverage=None, back_span=None):
    """(distance, pivot, metrics) - no frame of the fitted path may crop the subject.

    The fill solve frames the subject in the front pose and keeps its *area* centred along the path;
    it cannot promise that nothing is cut off, because a wide or high swing - and the lap in
    particular - moves the subject towards, and sometimes past, the picture edge. This pass checks
    the subject's *own pixels* against *every* frame of the fitted path (the same `subject_samples`
    the keys come from, 0.2 %/99.8 % of the projected points, with the fitted end and direction) and
    fixes a violation in the order an operator would:

    1. **shrink the swing** (`_shrink_amplitude`) - free for the framing, only viewing angles are
       lost, and the caller's amplitude ladder is capped with `amplitude_cap`,
    2. **re-aim the pivot** at the pose with the least room (`_centre_for_visibility`) - only when
       `recentre` is set: the cylinder path passes `recentre=False`, because moving the aim off the
       cylinder axis would break the exact horizontal centring it guarantees,
    3. **pull the camera back** in bisected steps until the subject clears the frame with a
       VISIBILITY_MARGIN border on each side (`_dolly_for_visibility`) - this one costs fill, so it
       is the last resort. It scales the one `distance` the whole path shares, so the camera-to-pivot
       distance stays constant.

    The metrics carry the new front-pose area, the new along-path band and the clearance at the
    0/25/50/75/100 % frames - the console line reports the trade instead of hiding it.
    """
    _, extents = subject_box(surface)
    # The path this pass guarantees is the path the keys come from - the orbit's own options travel
    # with it (view angle, coverage mode), so an "O-Orbit Front&Back" round is judged where it really goes.
    options = path_options(view_angle, coverage, back_span)
    # The pool is the caller's (the cylinder-trimmed 95 % subject) when given, so the guarantee and
    # the framing judge the same points; otherwise the full cloud.
    pool = _decimate(surface["content_cloud"] if pool is None else pool, VISIBILITY_POOL)
    margin_px = VISIBILITY_MARGIN * float(height)
    start = [float(value) for value in pivot]
    amplitude_cap = _shrink_amplitude(pool, surface, pivot, distance, scale, size, frames, height,
                                      margin_px, orbit_end, direction, options)
    units = subject_samples(frames, pivot, 1.0, amplitude_cap, size, orbit_end, direction, **options)
    before = visibility_clearances(pool, _path_positions(units, pivot, distance), pivot,
                                   surface, height, margin_px)
    cropped = sum(1 for value in before if value < 0.0)
    limit = max(float(surface["content_radius"]), MIN_ORBIT_RADIUS)
    if cropped and recentre:
        pivot, _ = _centre_for_visibility(pool, units, pivot, distance, surface, height,
                                          margin_px, limit)
    factor = _dolly_for_visibility(pool, units, pivot, distance, surface, height, margin_px)
    distance = max(MIN_ORBIT_RADIUS, distance * factor)
    after = visibility_clearances(pool, _path_positions(units, pivot, distance), pivot,
                                  surface, height, margin_px)
    points = _decimate(surface["content_cloud"], FRAMING_POOL)
    box = None
    try:
        box = _projected_box(points, front_camera(pivot, distance), pivot, surface, height)
    except ValueError:                        # the camera ended up on the subject: keep the caller's
        pass
    canvas_area = 1.0
    low = high = None
    if box:
        canvas_area = float(height) * float(box[5])
        try:
            low, high = _fill_envelope(points, pivot, distance, surface, size, canvas_area, height)
        except ValueError:                    # the front pose is fine, a swing pose is not
            low = high = None
    indices = [min(len(after) - 1, int(round(fraction * (len(after) - 1))))
               for fraction in VISIBILITY_QUARTILES] if after else []
    worst_index = min(range(len(after)), key=lambda index: after[index]) if after else 0
    metrics = {
        "ok": bool(after) and min(after) >= 0.0,
        "scale": float(factor),
        "amplitude_cap": float(amplitude_cap),
        "clearance_px": min(after) if after else 0.0,
        "cropped": int(cropped),
        "frames": len(after),
        "margin_px": margin_px,
        "shift": [pivot[axis] - start[axis] for axis in range(3)],
        "quarters_px": [after[index] for index in indices],
        "worst_frame": int(worst_index),
        "worst_pct": (worst_index / max(1, len(after) - 1)) if after else 0.0,
        "area": (box[0] * box[1]) / canvas_area if box else None,
        "fill_min": low, "fill_max": high,
        "width_px": box[0] if box else None, "height_px": box[1] if box else None,
        "radius_px": 0.5 * max(box[0], box[1]) if box else None,
        "box_extents": [float(value) for value in extents],
    }
    return distance, pivot, metrics


def equalize_pivot(surface, pivot, distance, scale, size, frames, orbit_end=None,
                   direction=ORBIT_DIRECTION_DEFAULT, height=CANVAS_HEIGHT):
    """(pivot, metrics) - place the orbit centre so the subject keeps its apparent size.

    The camera rides the orbit sphere at `distance` around the pivot, so the subject's apparent
    size only holds along the path when the pivot sits on the subject's own centre: every lateral
    offset - the framing's front-pose re-centring, the visibility re-aim - makes the distance
    breathe with the orbit (the box grows on the near side, shrinks on the far one). This pass
    minimises the *spread* of |camera - subject centre| over the final path (front O + closing
    orbit) with a coarse-to-fine coordinate scan around the incoming pivot, pulled a little
    towards the centre by a small weight and bounded to EQUALIZER_LIMIT content radii. The
    visibility guarantee wins over the equalisation: a move that crops the subject at any frame is
    bisected down until the whole path clears again, and when even the smallest step crops, the
    pivot is left exactly where the visibility pass put it.
    """
    content_radius = max(1e-6, float(surface["content_radius"]))
    start = [float(value) for value in pivot]
    centre = torch.as_tensor(surface["pivot"], dtype=torch.float64)
    shape_pool = _decimate(surface["content_cloud"], VISIBILITY_POOL)
    base = subject_samples(frames, [0.0, 0.0, 0.0], distance, scale, size, orbit_end, direction,
                           motion_pool=shape_pool, motion_surface=surface)
    stride = max(1, len(base) // max(2, int(EQUALIZER_SAMPLES)))
    offsets = torch.tensor(base[::stride], dtype=torch.float64).reshape(-1, 3)

    def spread_of(candidate):
        scene = torch.as_tensor(candidate, dtype=torch.float64).reshape(1, 3) + offsets
        d = (scene - centre).norm(dim=1)
        mean = float(d.mean())
        if mean <= 1e-9:
            return 0.0
        return float(d.max() - d.min()) / mean

    def score_of(candidate):
        pull = float((torch.as_tensor(candidate, dtype=torch.float64) - centre).norm())
        return spread_of(candidate) + EQUALIZER_CENTRE_WEIGHT * pull / content_radius

    best = list(start)
    best_score = score_of(best)
    if offsets.shape[0] >= 2:
        limit = EQUALIZER_LIMIT * content_radius
        for axis in (0, 1, 2):
            span = limit
            for steps in (EQUALIZER_SCAN_STEPS, EQUALIZER_REFINE_STEPS):
                step_size = 2.0 * span / max(1, steps - 1)
                # the stage scans around the value the *stage* started with; a running `best` here
                # would collapse every later candidate onto the clamp boundary (found the hard way)
                stage_centre = best[axis]
                for index in range(steps):
                    candidate = list(best)
                    value = stage_centre + (index - (steps - 1) / 2.0) * step_size
                    candidate[axis] = min(max(value, start[axis] - limit), start[axis] + limit)
                    score = score_of(candidate)
                    if score < best_score - 1e-12:
                        best_score, best = score, candidate
                span = 2.0 * step_size
    move = [best[axis] - start[axis] for axis in range(3)]
    spread_before = spread_of(start)
    applied = False
    factor = 0.0
    if any(abs(value) > 1e-9 for value in move):
        pool = _decimate(surface["content_cloud"], VISIBILITY_POOL)
        margin_px = VISIBILITY_MARGIN * float(height)

        def worst_clearance(candidate):
            # The equaliser moves the *orbit centre* only: the aim - and with it the composition -
            # stays where the framing put it (`start`). The clearance therefore has to be measured
            # with the original aim and the candidate's positions (see `visibility_clearances`:
            # positions and look are separate arguments).
            units = subject_samples(frames, candidate, 1.0, scale, size, orbit_end, direction,
                                    motion_pool=pool, motion_surface=surface)
            positions = _path_positions(units, candidate, distance)
            return min(visibility_clearances(pool, positions, start, surface, height, margin_px))

        # "No worse than the solved path": the visibility pass guarantees the incoming path clears
        # (>= 0); the equaliser may spend at most EQUALIZER_CLEARANCE_TOL px of that budget - never
        # turn a clearing path into a cropping one. A move that would is bisected down.
        floor = max(0.0, worst_clearance(start)) - EQUALIZER_CLEARANCE_TOL

        def ok(candidate):
            return worst_clearance(candidate) >= floor

        if ok(best):
            factor = 1.0
        else:
            low, high = 0.0, 1.0
            for _ in range(EQUALIZER_BISECTIONS):
                middle = 0.5 * (low + high)
                candidate = [start[axis] + move[axis] * middle for axis in range(3)]
                if ok(candidate):
                    low = middle
                else:
                    high = middle
            factor = low
        applied = factor > 0.02
    result = [start[axis] + move[axis] * (factor if applied else 0.0) for axis in range(3)]
    metrics = {"applied": bool(applied), "factor": round(float(factor), 4),
               "move": [result[axis] - start[axis] for axis in range(3)],
               "spread_before": float(spread_before),
               "spread_after": float(spread_of(result)) if applied else float(spread_before)}
    return result, metrics


def subject_framing(surface, fill_percent, size=1.0, height=CANVAS_HEIGHT, orbit_end=None,
                    direction=ORBIT_DIRECTION_DEFAULT, view_angle=None, coverage=None):
    """(distance, pivot, metrics): frame the subject's cylinder to `fill_percent` % of the picture.

    The subject is approximated by a vertical cylinder (`fit_subject_cylinder`: robust trimmed fit,
    ~90-95 % of the surface, outliers like lances or depth spikes left out). **The pivot is the
    cylinder centre and it is also the aim**: the whole path then keeps the subject in the *horizontal
    picture centre* at every pose - the cylinder is symmetric about its axis and the axis projects
    onto the image centre line - and the camera-to-pivot distance is one single constant for every
    frame (`_place` walks a sphere of exactly that radius). Only the *distance* is solved: the
    projected-box area fixed point first (the area falls with ~1/d^2), then the swing envelope's
    geometric mean so the requested fill holds along the O, and finally the room the closing orbit
    needs (which may only pull the camera back). The old per-axis `_solve_shift` re-centring is gone -
    it moved the aim off the axis, which is exactly what the centring requirement forbids.
    `metrics["pool"]` carries the trimmed cloud so the visibility guarantee works on the same 95 %
    subject.
    """
    centre, cylinder_radius, coverage, keep = fit_subject_cylinder(surface)
    cloud = surface["content_cloud"].reshape(-1, 3)
    points = _decimate(cloud[keep], FRAMING_POOL)
    content_radius = max(1e-6, float(surface["content_radius"]))
    pivot = [float(value) for value in centre]
    aspect = float(surface.get("aspect") or (16.0 / 9.0))
    canvas_area = float(height) * float(height) * aspect
    # The fill is solved against the frame *minus* the visibility border: the guarantee below keeps
    # VISIBILITY_MARGIN free on every side, and a target that used the whole frame would sit exactly
    # on that limit - any swing at all would then crop the subject and the O-orbit would collapse to
    # nothing. So the requested share is scaled down by the safe area (a 40 % request frames ~37 % of
    # the picture) and the front pose keeps headroom for the swing.
    margin = VISIBILITY_MARGIN * float(height)
    safe_area = max(1.0, (float(height) - 2.0 * margin) * (float(height) * aspect - 2.0 * margin))
    target = max(1.0, min(99.0, float(fill_percent))) / 100.0 * (safe_area / canvas_area)
    distance = max(MIN_ORBIT_RADIUS, SUBJECT_FILL * content_radius)
    for _ in range(FRAMING_ROUNDS):
        for _ in range(FRAMING_ITERATIONS):                # the box area falls with roughly 1/d^2
            camera = front_camera(pivot, distance)
            width_px, height_px, centre_u, centre_v, depth, canvas_width = _projected_box(
                points, camera, pivot, surface, height)
            distance = max(MIN_ORBIT_RADIUS,
                           distance * math.sqrt((width_px * height_px) / canvas_area / target))
        # The front pose alone is not the frame the user sees: correct the distance by the swing
        # envelope's geometric mean so the fill stays centred on the target *along* the path too.
        # No pivot shift follows - the cylinder centre IS the aim, which is what keeps the subject in
        # the horizontal picture centre instead of merely the projected box's.
        low, high = _fill_envelope(points, pivot, distance, surface, size, canvas_area, height)
        distance = max(MIN_ORBIT_RADIUS, distance * math.sqrt(math.sqrt(low * high) / target))
    # ... and the room the O *and* the back part need: the fill request may not push the camera so
    # close that a pose of the path leaves the frame (the level connection and the far side of the
    # back orbit are the tightest poses in practice). This may only pull the camera back, so the
    # distance stays ONE value for every frame - and the aim stays on the cylinder axis.
    room = _room_distance(_decimate(cloud[keep], VISIBILITY_POOL), surface, pivot,
                          distance, size, height, VISIBILITY_MARGIN * float(height), orbit_end,
                          direction, path_options(view_angle, coverage))
    distance_fill = distance
    if room > distance:
        distance = max(MIN_ORBIT_RADIUS, room)
    camera = front_camera(pivot, distance)
    width_px, height_px, centre_u, centre_v, depth, canvas_width = _projected_box(
        points, camera, pivot, surface, height)
    low, high = _fill_envelope(points, pivot, distance, surface, size, canvas_area, height)
    offset_px = (centre_u - 0.5 * canvas_width, centre_v - 0.5 * float(height))
    metrics = {
        "width_px": width_px, "height_px": height_px, "depth": depth,
        "area": (width_px * height_px) / canvas_area,
        "fill_min": low, "fill_max": high,
        "offset_px": offset_px,
        # The aim IS the cylinder axis, so there is no re-centring residual and no pivot shift. The
        # two legacy keys are kept (as the same numbers) so the console/summary contract holds.
        "offset_before_px": offset_px,
        "offset_uncentred_px": offset_px,
        "radius_px": 0.5 * max(width_px, height_px),
        "distance": distance,
        "distance_fill": float(distance_fill),
        "room_distance": float(room),
        "pivot_shift": [0.0, 0.0, 0.0],
        "points": int(points.shape[0]),
        "cylinder": {"centre": list(pivot), "radius": float(cylinder_radius),
                     "coverage": float(coverage)},
        "pool": cloud[keep],
    }
    return distance, pivot, metrics


def _key_frames(frames, target=KEY_TARGET):
    """Evenly spaced key frames covering 0..frames-1, each a whole frame index."""
    frames = int(frames)
    step = max(1, round((frames - 1) / max(1, target - 1)))
    ticks = list(range(0, frames, step))
    if ticks[-1] != frames - 1:
        ticks.append(frames - 1)
    return ticks


def coverage_for_end(orbit_end=None):
    """The coverage mode a (DEPRECATED) `orbit_end` asks for: 0 = the front O alone, else round.

    `orbit_end` was replaced by Auto Orbit View Angle + Auto Orbit Coverage; `orbit_end=0` maps to
    "O-Orbit Front" and anything larger maps to "O-Orbit Front&Back" (the complete front + back).
    """
    if orbit_end is None:
        return ORBIT_COVERAGE_DEFAULT
    return ORBIT_COVERAGES[0] if float(orbit_end) <= 1e-9 else ORBIT_COVERAGES[1]


def path_options(view_angle=None, coverage=None, back_span=None):
    """The `subject_samples` keywords that shape the path (empty = the legacy choreography).

    One bundle instead of three parameters through the private chain (room check, amplitude shrink,
    dolly, back cut): every pass that judges the path has to judge the SAME path the keys come from,
    so the options travel with it.
    """
    options = {}
    if view_angle is not None:
        options["view_angle"] = float(view_angle)
    if coverage is not None:
        options["coverage"] = str(coverage)
    if back_span is not None:
        options["back_span"] = float(back_span)
    return options


def _orbit_point(progress, centre, amplitude, mirror, rise=FRONT_ORBIT_RISE):
    """(yaw, elevation) of one O-orbit at `progress` 0..1: middle -> 2 o'clock -> 3 o'clock.

    Progress 0 is the *middle of the circle* (the pose the framing solved, elevation 0), the first
    FRONT_ORBIT_RISE of it is the eased crane up and over to the sweep's START - 2 o'clock, the
    upper right of the circle (the crane used to land on the top, 12) - and the rest sweeps
    2 -> 1 -> 12 -> 9 -> 6 -> 3 o'clock (counter-clockwise, 330 deg of the circle; `mirror` = -1
    mirrors everything to a crane to 10 o'clock and a 10 -> 11 -> 12 -> ... -> 9 sweep), ending on
    the circle's side at level elevation - exactly where the connection takes over, so only the
    opening of the front O moved. This is the FRONT O's move; the back orbit is the half circle
    `_back_orbit_point` traces from the connection's first contact point.
    """
    progress = min(1.0, max(0.0, float(progress)))
    start_angle = math.radians(90.0 - mirror * 30.0 * FRONT_ORBIT_START_CLOCK)
    if progress <= rise:
        phase = progress / rise
        eased = (1.0 - math.cos(math.pi * phase)) / 2.0
        return (centre + amplitude * math.cos(start_angle) * eased,
                amplitude * math.sin(start_angle) * eased)
    phase = (progress - rise) / max(1e-9, 1.0 - rise)
    angle = start_angle + mirror * math.radians(FRONT_ORBIT_SWEEP_DEGREES) * phase
    return centre + amplitude * math.cos(angle), amplitude * math.sin(angle)


def _back_clock_progress(clock):
    """Where a clock angle sits on the back orbit's arc, 0..1 (9 o'clock = 0, 8 o'clock = 1)."""
    return (BACK_ORBIT_CLOCK_START - float(clock)) / BACK_ORBIT_CLOCK_SWEEP


def _back_orbit_point(progress, centre, amplitude, mirror):
    """(yaw, elevation) of the back orbit - a full clockwise loop that starts at 9 o'clock.

    The level connection leaves the front O's side at elevation 0 and runs straight to the subject's
    far side, so the first pose of the back orbit it meets is that orbit's *level point on the
    arrival side* - the "9 o'clock" of the back clock, whose 12 o'clock is the subject's back head.
    The loop begins exactly there and sweeps clockwise (on the clock face as seen from behind the
    subject): 9 -> 12 over the back's head -> 3, the far level point -> 6, under the back -> 8, one
    hour short of the start, so the closing frames never repeat the opening pose. In the orbit's own
    terms that is `yaw = centre + mirror*A*cos(clock)`, `elevation = A*sin(clock)` with the clock
    running from `BACK_ORBIT_CLOCK_START` down by `BACK_ORBIT_CLOCK_SWEEP` - 330 deg of the circle
    (hence `BACK_ORBIT_TRAVEL_FACTOR`).

    Two things this buys over the older moves. (a) The earlier "concluding orbit" started at the
    circle's *middle* and had to walk back across the 9 o'clock point the connection had already
    reached - a 2A degree out-and-back of level azimuth that showed the same poses twice. (b)
    Beginning on the near edge also lets the connection be 180 - 2A long instead of 180 - A, so the
    whole visit is shorter: the freed frames go to the front O, whose radius the fit may then grow.
    The half-loop version (9 -> 12 -> 3) covered the back's upper half only; this one closes the
    circle and shows the back from below as well. The glide from this ring's 8 o'clock end to the
    dial's centre is not part of this function - `subject_samples` appends it (it is a straight
    run to a point OFF the ring, the last pose of the whole path).
    """
    progress = min(1.0, max(0.0, float(progress)))
    angle = math.radians(BACK_ORBIT_CLOCK_START - BACK_ORBIT_CLOCK_SWEEP * progress)
    return centre + mirror * amplitude * math.cos(angle), amplitude * math.sin(angle)


def _spiral_offset(yaw_degrees, elevation_degrees):
    """The unit camera offset `_place` builds from a (yaw, elevation) pair (x right, -y up, -z out).

    The source camera sits on the -z side of the pivot looking at it, so this is the pose's place on
    the unit sphere around the pivot, in the same convention the rest of the module uses.
    """
    yaw = math.radians(float(yaw_degrees))
    elevation = math.radians(float(elevation_degrees))
    return (math.cos(elevation) * math.sin(yaw),
            -math.sin(elevation),
            -math.cos(elevation) * math.cos(yaw))


def _spiral_tilt(offset, slope_degrees):
    """Lean an offset in the vertical plane through the view axis by the spiral's axis slope.

    The world's x (the subject's right) is the hinge, so the view axis itself stays the reference:
    slope 0 leaves the offset alone, +90 lifts the axis straight up (and -90 pushes it down). The
    transform is a rotation, so lengths, the pivot and the aim are untouched - and `_spiral_tilt`
    with the opposite slope undoes it.
    """
    slope = math.radians(float(slope_degrees))
    if abs(slope) <= 1e-12:
        return offset
    cos_s, sin_s = math.cos(slope), math.sin(slope)
    x, y, z = offset
    return (x, y * cos_s + z * sin_s, -y * sin_s + z * cos_s)


def spiral_pose(phi_degrees, psi_degrees, slope_degrees=0.0):
    """(yaw, elevation) of the point whose arc from the view axis is `phi` at clock `psi`.

    `phi` 0 = the camera ON the view axis (the frontal view, the middle of the picture), 90 = the
    picture's own plane; `psi` is the clock angle around that axis (12 = up, 3 = the subject's
    right, 6 = down, 9 = left). `slope_degrees` leans that axis in the vertical plane through the
    view axis - 0 keeps the view axis itself, +90 stands the spiral on the vertical axis ("from
    straight above": `phi` = 0 opens straight above the subject), -90 hangs it below. The camera
    keeps its distance and always aims at the pivot, so the subject keeps its place in the frame.
    """
    phi = math.radians(float(phi_degrees))
    psi = math.radians(float(psi_degrees))
    # the offset in the axis' own frame, i.e. the pose `_place` builds from (yaw, elevation)
    offset = (math.sin(phi) * math.sin(psi),
              -math.sin(phi) * math.cos(psi),
              -math.cos(phi))
    offset = _spiral_tilt(offset, slope_degrees)
    return (math.degrees(math.atan2(offset[0], -offset[2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -offset[1])))))


def spiral_arc(yaw_degrees, elevation_degrees, slope_degrees=0.0):
    """The arc between a (yaw, elevation) pose and the (tilted) spiral axis, in degrees."""
    offset = _spiral_tilt(_spiral_offset(yaw_degrees, elevation_degrees), -slope_degrees)
    return math.degrees(math.acos(max(-1.0, min(1.0, -offset[2]))))


def spiral_clock(yaw_degrees, elevation_degrees, slope_degrees=0.0):
    """The clock angle of a (yaw, elevation) pose around the (tilted) axis (12 = up, 3 = right)."""
    offset = _spiral_tilt(_spiral_offset(yaw_degrees, elevation_degrees), -slope_degrees)
    if math.hypot(offset[0], offset[1]) <= 1e-9:
        return 0.0                     # the axis itself: the clock angle is undefined, so it reads 0
    return math.degrees(math.atan2(offset[0], -offset[1]))


def _spiral_sweep():
    """Total clock-angle advance of the coil: the Spiral End parameter itself, in degrees."""
    return float(_ACTIVE_SPIRAL_END)


def _spiral_geometry(centre, mirror, slope=None):
    """(axis_yaw, end_yaw, sweep, end_arc) of the Spiral coverage.

    The path starts ON the sphere's axis (the framed frontal view, `phi` = 0) and unwinds
    `_spiral_sweep()` degrees around it (`psi`, the winding the node's Spiral Winding widget asks
    for) while the O-orbit angle climbs to `spiral_end_arc()` - the node's Spiral End Angle. `slope`
    is the axis' own lean (`spiral_slope`): 0 keeps the view axis, +90 stands it up.
    """
    end_yaw, _end_elevation = _spiral_point(1.0, centre, mirror, slope)
    return float(centre), end_yaw, _spiral_sweep(), spiral_end_arc()


SPIRAL_ARC_STEPS = 512            # integration steps of the constant-speed arc-length remap
_SPIRAL_ARC_TABLES = {}


def _spiral_arc_table(end_arc_degrees, sweep_degrees, steps=SPIRAL_ARC_STEPS):
    """(linear parameter, share of the coil's arc length) lookup for the spiral.

    The camera rides a sphere around the pivot, so the arc it covers per unit of the linear
    parameter `t` is `sqrt(dphi^2 + sin(phi)^2 * dpsi^2)`: on the axis (`phi` = 0) only the
    opening `phi` counts, while on the outer ring the winding `psi` dominates. With an even `t`
    the camera therefore crawls at the start of the coil and races at its end. Integrating that
    speed gives the table `_spiral_linear_parameter` inverts to hand every frame the same arc.
    """
    key = (round(float(end_arc_degrees), 6), round(float(sweep_degrees), 6), int(steps))
    table = _SPIRAL_ARC_TABLES.get(key)
    if table is not None:
        return table
    end_arc = math.radians(abs(float(end_arc_degrees)))
    sweep = math.radians(abs(float(sweep_degrees)))
    if int(steps) < 2 or (end_arc <= 1e-12 and sweep <= 1e-12):
        table = ([0.0, 1.0], [0.0, 1.0])
        _SPIRAL_ARC_TABLES[key] = table
        return table
    parameters = [index / steps for index in range(steps + 1)]
    lengths = [0.0]
    for index in range(1, steps + 1):
        middle = 0.5 * (parameters[index - 1] + parameters[index])
        speed = math.hypot(end_arc, math.sin(end_arc * middle) * sweep)
        lengths.append(lengths[-1] + speed / steps)
    total = lengths[-1]
    table = ([0.0, 1.0], [0.0, 1.0]) if total <= 1e-12 else (
        parameters, [value / total for value in lengths])
    _SPIRAL_ARC_TABLES[key] = table
    return table


def _spiral_linear_parameter(progress, end_arc_degrees, sweep_degrees):
    """The linear coil parameter whose arc length is `progress` (0..1) of the whole coil.

    Monotone by construction (`_spiral_arc_table` is a cumulative sum), so the endpoints map to
    themselves: progress 0 stays the framed frontal view, progress 1 stays the requested end
    pose - only the spacing between them changes.
    """
    progress = min(1.0, max(0.0, float(progress)))
    if progress <= 0.0 or progress >= 1.0:
        return progress
    parameters, shares = _spiral_arc_table(end_arc_degrees, sweep_degrees)
    low, high = 0, len(shares) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if shares[middle] <= progress:
            low = middle
        else:
            high = middle
    span = shares[high] - shares[low]
    if span <= 1e-12:
        return parameters[low]
    return parameters[low] + (progress - shares[low]) / span * (parameters[high] - parameters[low])


def _spiral_point(progress, centre, mirror=1.0, slope=None):
    """(yaw, elevation) at `progress` 0..1 along the spiral: frontal view -> the end angle.

    The camera travels on a sphere whose centre is the pivot (constant distance, always aiming at
    it, so the subject keeps its place in the frame) along the *family of O-orbits* the Front-only
    coverage flies: `phi` - the O-orbit's angular radius, the arc between the camera and the
    spiral's axis - climbs 0 -> `spiral_end_arc()` (the node's Spiral End Angle widget,
    90 deg by default), so the FIRST frame sits ON that axis (the middle of the picture, looking
    straight at it) and the LAST one is on the outermost O-orbit - at 90 deg in the picture's own
    plane, the side view. `progress` is the share of the coil's **arc length** (`_spiral_linear_parameter`),
    so the camera travels at a constant speed: the angle per frame is small where the coil opens and
    large where it winds. `psi` - the clock angle around that axis - runs 0 -> the
    *Spiral Winding* parameter (`_spiral_sweep`, default `SPIRAL_END_DEFAULT` = 840 deg), and that
    winding is the only thing deciding where the last frame looks from. `mirror` flips the winding
    (the Auto Orbit Direction widget), `centre` is the azimuth of the axis (the Auto Orbit View
    Angle: a yaw rotation, so the arc from the axis is unchanged) and `slope` leans the axis itself
    (`spiral_slope`, the Spiral Center Slope widget: 0 = the view axis, +90 = straight above).
    """
    progress = min(1.0, max(0.0, float(progress)))
    wind = -1.0 if mirror >= 0.0 else 1.0          # counter-clockwise = the clock angle falls
    end_arc, sweep = spiral_end_arc(), _spiral_sweep()
    # `progress` is the share of the coil's ARC LENGTH, not of its angle, so the camera keeps a
    # constant speed the whole way round: `phi`/`psi` follow the inverse of the coil's own
    # arc-length profile (slow winding near the pole, fast at the outer ring).
    linear = _spiral_linear_parameter(progress, end_arc, sweep)
    phi = end_arc * linear
    psi = wind * sweep * linear
    yaw, elevation = spiral_pose(phi, psi, spiral_slope() if slope is None else slope)
    # the coil passes over the top on its way round; keep the up vector well defined there
    return (yaw + centre,
            max(-SPIRAL_ELEVATION_CEILING, min(SPIRAL_ELEVATION_CEILING, elevation)))


def _spiral_info(view_angle=None, direction=ORBIT_DIRECTION_DEFAULT):
    """The `info` fields the Spiral coverage reports: winding, slope, end arc and the end pose."""
    centre = ORBIT_VIEW_ANGLE_DEFAULT if view_angle is None else float(view_angle)
    mirror = direction_mirror(direction)
    slope = spiral_slope()
    _axis_yaw, end_yaw, sweep, end_arc = _spiral_geometry(centre, mirror)
    end_elevation = _spiral_point(1.0, centre, mirror)[1]
    return {"front_yaw": 0.0, "orbit_end": sweep, "front_share": 1.0,
            "back_span": 0.0, "back_orbit": False,
            "spiral_end": sweep, "spiral_end_arc": end_arc,
            "spiral_end_elevation": end_elevation,
            "spiral_slope": slope,
            "spiral_end_clock": spiral_clock(end_yaw - centre, end_elevation, slope) % 360.0,
            "orbit_coverage": min(FULL_CIRCLE, sweep),
            "view_angle": None if view_angle is None else float(view_angle)}


def _spiral_samples(frames, pivot, radius, centre, mirror, slope=None):
    """The spiral's per-frame positions: the O-orbit angle growing evenly along the path.

    The frames are the parameter itself - `phi` and `psi` both follow the frame index through
    `_spiral_point` - because that is exactly the spec: *O-orbits like in the Front-only setting,
    with the angle growing from 0 to the end angle along the path*. `_spiral_point` distributes
    those frames over the coil's **arc length**, so the camera keeps a constant speed (an even
    angle per frame made it crawl near the pole and race at the outer ring, which is what the user
    saw as "very slow at the beginning, much faster towards the end"). Nothing else may reshape the
    growth: not the speed cap (the console reports the per-frame drift instead, and the winding /
    end angle stay what the widgets asked for) and not a "motion-even" split (that chased the
    subject's pixel motion, which is cheap for a roll near the pole and expensive where the picture
    is sensitive, so the camera raced around the axis and crawled elsewhere - the jagged path). The
    *rendered* path is kept smooth by the key selection instead (`_spiral_key_frames` follows the
    path's own turning).
    """
    return [_place(pivot, radius, *_spiral_point(index / max(1, frames - 1), centre, mirror, slope))
            for index in range(int(frames))]


def _spiral_key_frames(samples, frames, turn=SPIRAL_KEY_TURN):
    """Frame indices of the spiral's keys: by how far the camera TURNS, not by frame index.

    The Geometry node splines the keys, so a key list that is evenly spaced in *frames* cuts the
    corner wherever the path turns faster than the spline can follow. The spiral turns fastest
    right after the pole - the clock races while the camera opens the coil - so this walks the
    samples and takes a key whenever the camera's heading has moved `turn` degrees since the last
    one, plus the first and last frame. The keys then follow the path everywhere (a straight-ish
    stretch simply gets fewer of them) instead of missing it by 16 % of the orbit radius at frame 4.
    """
    frames = int(frames)
    if frames < 3:
        return list(range(max(1, frames)))
    headings = []
    for index in range(1, frames):
        step = [samples[index][axis] - samples[index - 1][axis] for axis in range(3)]
        length = math.sqrt(sum(value * value for value in step))
        headings.append([value / length for value in step] if length > 1e-12 else None)
    ticks = [0]
    turned = 0.0
    for index in range(1, frames - 1):
        previous, current = headings[index - 1], headings[index]
        if previous is None or current is None:
            continue                        # a hold: no heading to measure between the frames
        cosine = max(-1.0, min(1.0, sum(left * right for left, right in zip(previous, current))))
        turned += math.degrees(math.acos(cosine))
        if turned >= turn:
            ticks.append(index)
            turned = 0.0
    if ticks[-1] != frames - 1:
        ticks.append(frames - 1)
    return ticks


def _band_orbit(index, count, width, start_yaw):
    """(yaw_begin, yaw_end) of band `index`: the arc that band sweeps, in flight order.

    Band 0 begins at `start_yaw` and sweeps towards the LEFT (decreasing yaw, the
    spec's "starts on the right side and goes to the left"); the next one begins
    exactly where it stopped and sweeps back, and so on. That is what makes every
    band END where the following one BEGINS, so the step between two bands is a pure
    elevation move at ONE azimuth - the "straight up to the higher orbit" of the spec.
    A 360 deg width ends where it began, so alternating would be a no-op: that width
    keeps turning the SAME way instead (`start_yaw -> start_yaw - 360` on every band),
    which is the spec's own rule for a full circle.
    """
    width = resolve_band_width(width)
    start_yaw = resolve_band_start_yaw(start_yaw)
    if width >= FULL_CIRCLE - 1e-6 or index % 2 == 0:
        return start_yaw, start_yaw - width
    return start_yaw - width, start_yaw


def _band_legs(start_yaw, elevations, width):
    """(travel, yaw0, elevation0, yaw1, elevation1) for every leg, in flight order.

    One leg per band (its sweep) plus one per step between two bands (the pure
    elevation move at the azimuth the band ended on). The travel is measured on the
    unit sphere - `d(yaw) * cos(elevation)` along a band, `d(elevation)` along a step
    - so the frames can be split by it and the camera holds ONE angular speed. That
    is also why a narrow stack near a pole (a small circle) takes fewer frames than a
    wide sweep at the equator, instead of crawling there.
    """
    legs = []
    for index, elevation in enumerate(elevations):
        begin, end = _band_orbit(index, len(elevations), width, start_yaw)
        travel = math.radians(abs(end - begin)) * math.cos(math.radians(elevation))
        legs.append((travel, begin, elevation, end, elevation))
        if index + 1 < len(elevations):
            nxt = elevations[index + 1]
            legs.append((abs(math.radians(nxt - elevation)), end, elevation, end, nxt))
    return legs


def _band_front_back_legs(start_yaw, elevations, width):
    """Every leg of the "Banded Orbits Front & Back" path, in flight order.

    The front half is the plain stack (`_band_legs`): the camera climbs from the
    lowest band to the highest. At the top the camera does NOT turn back - it keeps
    going the way the top band was already going and EXTENDS that sweep until it
    reaches the mirror of its own start on the far side (`BAND_FRONT_BACK_TURN` minus
    the band's width further on). From there the SAME bands follow on the far side,
    highest first, each one swept the way its front twin was - so the camera walks
    back DOWN the far side without ever reversing. No arc is shown twice and the whole
    visit is as short as it can be: the front half, the bridge across the top, and the
    mirrored bands.

    The legs keep `_band_legs`' tuple shape, so the same frame walk (`_leg_walk`)
    splits them by their travel on the sphere.
    """
    width = resolve_band_width(width)
    start_yaw = resolve_band_start_yaw(start_yaw)
    front = _band_legs(start_yaw, elevations, width)
    bands = front[0::2]                               # the sweeps, in flight order
    top = bands[-1]
    direction = 1.0 if top[3] - top[1] >= 0.0 else -1.0
    # The top band's sweep is EXTENDED - same heading, no turn - until it reaches the
    # mirror of its own start, the far edge of the upper back orbit.
    bridge = ((top[1] + BAND_FRONT_BACK_TURN - top[3]) * direction) % FULL_CIRCLE
    yaw = top[3] + direction * bridge
    legs = [*front, (math.radians(bridge) * math.cos(math.radians(top[2])),
                     top[3], top[2], yaw, top[2])]
    # The far side: the SAME bands walked back DOWN, each swept the way its front twin
    # was (so the heading never flips), with a pure elevation step between neighbours.
    for band in reversed(bands):
        sweep = 1.0 if band[3] - band[1] >= 0.0 else -1.0
        elevation = band[2]
        if abs(elevation - legs[-1][4]) > 1e-9:
            legs.append((abs(math.radians(elevation - legs[-1][4])),
                         yaw, legs[-1][4], yaw, elevation))
        end = yaw + sweep * abs(band[3] - band[1])
        legs.append((abs(math.radians(end - yaw)) * math.cos(math.radians(elevation)),
                     yaw, elevation, end, elevation))
        yaw = end
    return legs


def _leg_walk(frames, pivot, radius, legs):
    """Per-frame positions along a leg list, the frames split by the legs' travel.

    The camera holds one angular speed: each leg gets the frames its travel on the
    sphere earns (`_band_legs`' `(travel, yaw0, elevation0, yaw1, elevation1)`
    tuples). The first frame sits on the first leg's start and the last one lands
    exactly on the last leg's end.
    """
    frames = int(frames)
    total = sum(leg[0] for leg in legs)
    if frames <= 1 or total <= 1e-12:
        # one frame (or a degenerate stack): the path is a point - report its start
        return [_place(pivot, radius, legs[0][1], legs[0][2]) for _ in range(max(1, frames))]
    span = float(frames - 1)
    positions = []
    for index in range(frames):
        distance = total * (index / span)
        yaw, elevation = legs[-1][3], legs[-1][4]      # the last frame lands exactly on the end
        for leg in legs:
            if distance <= leg[0] or leg is legs[-1]:
                phase = min(1.0, max(0.0, distance / leg[0])) if leg[0] > 1e-12 else 1.0
                yaw = leg[1] + (leg[3] - leg[1]) * phase
                elevation = leg[2] + (leg[4] - leg[2]) * phase
                break
            distance -= leg[0]
        positions.append(_place(pivot, radius, yaw, elevation))
    return positions


def _band_samples(frames, pivot, radius, start_yaw=None, elevations=None, width=None):
    """Per-frame positions of the "Banded Orbits" path - the spec's own choreography.

    The camera starts on the FIRST (lowest) band at `start_yaw` - the spec's "on the
    right side" - flies it towards the left until it has covered `width`, steps to the
    next band AT THAT AZIMUTH (a pure elevation move: the bands share their
    endpoints), flies that one back the other way, then up to the next band and back
    again. With a 360 deg width the bands cannot alternate, so the camera keeps
    turning the same way instead.

    The frames are split by the legs' *travel on the sphere*, so the camera holds one
    angular speed: a narrow band near a pole (a small circle) gets fewer frames than a
    wide sweep at the equator, exactly as it should. `_place` puts every pose on the
    orbit sphere, so the radius - and with it the framing - never changes; levelling
    to the WORLD up means the camera never rolls, unlike the spiral (whose axis leans
    and which therefore needs the tilted up of `_ACTIVE_ROLL_UP`).

    `elevations` may be given (a list, lowest first) or left to `band_elevations()`;
    `width` and `start_yaw` likewise fall back to the module's active widgets.
    """
    frames = int(frames)
    width = band_width() if width is None else resolve_band_width(width)
    start_yaw = band_start_yaw() if start_yaw is None else resolve_band_start_yaw(start_yaw)
    if elevations is None:
        elevations = band_elevations()
    elevations = [float(value) for value in elevations] or [BAND_START_ANGLE_DEFAULT]
    return _leg_walk(frames, pivot, radius, _band_legs(start_yaw, elevations, width))


def _band_front_back_samples(frames, pivot, radius, start_yaw=None, elevations=None, width=None):
    """Per-frame positions of the "Banded Orbits Front & Back" path.

    The plain band stack (`_band_legs`) climbs the front; the top band's sweep is then
    extended, in the same direction, until it reaches the far edge of the upper back
    orbit, and the same bands walk back DOWN the far side (see
    `_band_front_back_legs`). The frames follow the legs' travel on the sphere over both
    halves and the bridge alike, so the camera holds one angular speed, and `_place`
    keeps every pose on the orbit sphere - the radius, the framing and the (world-up,
    roll-free) level never change.

    `elevations` may be given (a list, lowest first) or left to `band_elevations()`;
    `width` and `start_yaw` likewise fall back to the module's active widgets.
    """
    width = band_width() if width is None else resolve_band_width(width)
    start_yaw = band_start_yaw() if start_yaw is None else resolve_band_start_yaw(start_yaw)
    if elevations is None:
        elevations = band_elevations()
    elevations = [float(value) for value in elevations] or [BAND_START_ANGLE_DEFAULT]
    return _leg_walk(frames, pivot, radius,
                     _band_front_back_legs(start_yaw, elevations, width))


def _band_info(start_yaw=None, width=None):
    """The `info` fields the "Banded Orbits" coverage reports."""
    width = band_width() if width is None else resolve_band_width(width)
    start_yaw = band_start_yaw() if start_yaw is None else resolve_band_start_yaw(start_yaw)
    elevations = band_elevations()
    legs = _band_legs(start_yaw, elevations, width)
    stop = band_stop_angle()
    return {"front_yaw": 0.0, "orbit_end": width, "front_share": 1.0,
            "back_span": 0.0, "back_orbit": False, "band_front_back": False,
            "band_start_angle": band_start_angle(), "band_count": len(elevations),
            "band_stop_angle": stop, "band_width": width, "band_start_yaw": start_yaw,
            "band_elevations": elevations, "band_lowest": min(elevations),
            "band_highest": max(elevations), "band_travel": sum(leg[0] for leg in legs),
            "band_end_yaw": legs[-1][3], "band_end_elevation": legs[-1][4],
            "band_front_end_yaw": legs[-1][3],
            "orbit_coverage": min(FULL_CIRCLE, abs(width)), "view_angle": None}


def _band_front_back_info(start_yaw=None, width=None):
    """The `info` fields the "Banded Orbits Front & Back" coverage reports.

    Everything `_band_info` reports about the stack, plus the numbers the doubled
    path adds: `band_front_end_*` is where the front half stops (the top band's end),
    `band_back_start_yaw` the azimuth the extended top band carries it to (the mirror
    of the top band's START, one `BAND_FRONT_BACK_TURN` away), and `band_end_*` the
    pose the last frame looks from (the far side's own end, the mirrored lowest band).
    `band_travel` and `orbit_coverage` count both halves.
    """
    info = _band_info(start_yaw, width)
    start_yaw = band_start_yaw() if start_yaw is None else resolve_band_start_yaw(start_yaw)
    width = band_width() if width is None else resolve_band_width(width)
    elevations = band_elevations()
    front = _band_legs(start_yaw, elevations, width)
    legs = _band_front_back_legs(start_yaw, elevations, width)
    info.update({
        "band_front_back": True,
        "band_front_end_yaw": front[-1][3], "band_front_end_elevation": front[-1][4],
        "band_back_start_yaw": front[-1][1] + BAND_FRONT_BACK_TURN,
        "band_end_yaw": legs[-1][3], "band_end_elevation": legs[-1][4],
        "band_travel": sum(leg[0] for leg in legs),
        # The front half sweeps the band width, the extended top band then carries the
        # camera a half turn across to the far side, so the whole path shows that much
        # azimuth - capped at one full turn.
        "orbit_coverage": min(FULL_CIRCLE, abs(width) + BAND_FRONT_BACK_TURN),
    })
    return info


def _legacy_samples(frames, pivot, radius, scale=1.0, size=1.0, orbit_end=None,
                    direction=ORBIT_DIRECTION_DEFAULT, share=None):
    """The pre-`coverage` choreography: the front O, then the concluding orbit to `orbit_end`.

    Kept verbatim - its numbers, including the 38 deg height arc at the back - so saved API calls and
    the older tests behave exactly as they did. It still gets the 45 deg *circle* for the front O
    (that is `front_amplitudes` now); new callers pass one of `ORBIT_COVERAGES` to `subject_samples`
    and get the front/back path with the shortest connection instead.
    """
    frames = int(frames)
    size = max(1e-3, float(size))
    yaw_amplitude, elevation_amplitude = front_amplitudes(scale, size)
    rest_high = min(FRONT_ORBIT_LIMIT, REST_ELEVATION_HIGH * scale * size)
    span = lap_span_for_end(yaw_amplitude, orbit_end)
    mirror = direction_mirror(direction)
    if span <= 1e-9:
        share = 1.0                       # no concluding orbit: the O takes every frame
    else:
        if share is None:
            share = balanced_front_share(yaw_amplitude, span)
        share = min(0.95, max(0.05, float(share)))
    split = max(1.0, (frames - 1) * share)
    rise = max(1.0, split * FRONT_ORBIT_RISE)
    positions = []
    for index in range(frames):
        if index <= split:
            if index <= rise:
                phase = index / rise
                yaw = 0.0
                elevation = elevation_amplitude * (1.0 - math.cos(math.pi * phase)) / 2.0
            else:
                phase = (index - rise) / max(1.0, split - rise)
                angle = math.radians(90.0 + mirror * 270.0 * phase)
                yaw = yaw_amplitude * math.cos(angle)
                elevation = elevation_amplitude * math.sin(angle)
        else:
            phase = (index - split) / max(1.0, (frames - 1) - split)
            yaw = mirror * (yaw_amplitude + span * phase)      # around the back, back to the start
            elevation = rest_high * (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
        positions.append(_place(pivot, radius, yaw, elevation))
    return positions


def subject_samples(frames, pivot, radius, scale=1.0, size=1.0, orbit_end=None,
                    direction=ORBIT_DIRECTION_DEFAULT, share=None, view_angle=None,
                    coverage=None, back_span=None, motion_pool=None, motion_surface=None,
                    motion_cap=None, spiral_slope=None):
    """Per-frame positions of the subject path: the front O, then (optionally) the back O.

    **The front O** is a *circle* (`front_amplitudes`: one radius both ways) centred on the azimuth
    `view_angle` - 0 = the frontal view towards the subject, +90 = its viewer-left side, 180 = its
    back, 270/-90 = its right. Frame 0 is the *middle of the circle* (the framed pose, elevation 0,
    so its drift is zero), the camera then rises to 2 o'clock (an eased crane over FRONT_ORBIT_RISE
    of the O's frames - up and right of the top, which is what it used to aim at) and sweeps
    2 -> 1 -> 12 -> 9 -> 6 -> 3 o'clock (counter-clockwise, 330 deg of the circle; 'clockwise'
    mirrors the whole path to a crane to 10 and a 10 -> ... -> 9 sweep), ending on the circle's side
    at level elevation.

    **"O-Orbit Front&Back"** (`coverage`) then adds the back orbit: the SHORTEST connection first - a
    level azimuth sweep from where the front O ended straight to the back orbit's *near edge*, i.e.
    180 deg minus the O's own DIAMETER, never a round trip and never past the back circle's middle -
    and then that back orbit from exactly that first contact point (`_back_orbit_point`): a full
    clockwise loop over the subject's back head (9 -> 12 -> 3 -> 6 -> 8 o'clock, ending one hour
    short of its start so no pose is shown twice) and a final glide from 8 o'clock in to the back
    dial's CENTRE - the level pose straight behind the subject, the mirror of the front O's own
    opening pose. `back_span` shortens the connection (the fit does,
    when frames x speed cap cannot pay for the whole back part): below the full value the back orbit
    is skipped and the path simply ends where the sweep stopped, so the front O is never the thing
    that gives way.

    The frames are split between the phases by their travel, so every phase runs at the same
    per-frame subject drift (the speed cap's best case). `share` overrides the front O's part.

    Without a `coverage` mode this is the OLDER choreography ("the front O, then a concluding orbit
    to `orbit_end`") - verbatim, so saved callers behave as before; the new front/back path needs an
    explicit `ORBIT_COVERAGES` value.
    """
    if coverage is None:
        return _legacy_samples(frames, pivot, radius, scale, size, orbit_end, direction, share)
    frames = int(frames)
    size = max(1e-3, float(size))
    amplitude = front_amplitudes(scale, size)[0]
    centre = ORBIT_VIEW_ANGLE_DEFAULT if view_angle is None else float(view_angle)
    mirror = direction_mirror(direction)
    mode = ORBIT_COVERAGE_DEFAULT if coverage is None else str(coverage)
    if coverage is None and orbit_end is not None:
        mode = coverage_for_end(orbit_end)
    if mode == SPIRAL_COVERAGE:
        # A spherical spiral: the *family of O-orbits* the Front-only coverage flies, their angle
        # (`phi`) growing evenly from 0 to the Spiral End Angle while the clock winds Spiral Winding
        # degrees around the axis - exactly the spec. The frames are the parameter itself; the
        # rendered path stays smooth because the keys follow the path's own turning.
        return _spiral_samples(frames, pivot, radius, centre, mirror, slope=spiral_slope)
    if mode == BAND_COVERAGE:
        # A stack of circle SEGMENTS on the orbit sphere: sweep a band, step straight
        # to the next one at the same azimuth, sweep back, and so on. Every number the
        # path needs is its own widget (start angle, count, width, start yaw, step), so
        # the O-orbit amplitude, the view angle and the direction play no part here -
        # the *width* is the swing and `band_start_yaw` is the azimuth, by design.
        return _band_samples(frames, pivot, radius)
    if mode == BAND_FRONT_BACK_COVERAGE:
        # The same stack, but the camera does not stop at the top: it extends the top
        # band's own sweep across to the far side and walks the bands back DOWN there,
        # each swept the way its front twin was. Same widgets, same sphere, no roll.
        return _band_front_back_samples(frames, pivot, radius)
    back = mode == ORBIT_COVERAGES[1]
    # The level connection ends where the back orbit begins: its near edge, one O radius (half the
    # circle's span) short of the back circle's middle. That is the shortest way to reach the back
    # at all - the older move went on to the middle and the back orbit then walked back across it.
    full = max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * amplitude)
    connect = min(full, max(0.0, float(back_span))) if (back and back_span is not None) else full
    if not back or connect <= 1e-9:
        connect = 0.0
    run_back = connect >= full - 1e-9 and full > 1e-9        # the back orbit only after a full connection
    front_travel = LOOP_TRAVEL_FACTOR * math.radians(amplitude)
    connect_travel = math.radians(connect)
    back_travel = ((BACK_ORBIT_TRAVEL_FACTOR + BACK_ORBIT_HOME_FACTOR) * math.radians(amplitude)
                   if run_back else 0.0)
    rest = connect_travel + back_travel
    if share is None:
        total = front_travel + rest
        front_share = front_travel / total if total > 1e-9 else 1.0
    else:
        front_share = min(0.95, max(0.05, float(share)))
    connect_share = (connect_travel / rest) * (1.0 - front_share) if rest > 1e-9 else 0.0
    split_front = max(1.0, (frames - 1) * front_share)
    split_connect = split_front + max(0.0, (frames - 1) * connect_share)
    positions = []
    for index in range(frames):
        if index <= split_front or not rest:
            # the front O alone when there is no back part - every frame belongs to the circle, so
            # the clip ends on its side (and never on a tail with nothing left to fly)
            progress = index / max(1.0, split_front)
            yaw, elevation = _orbit_point(progress, centre, amplitude, mirror)
        elif index <= split_connect:
            phase = (index - split_front) / max(1e-9, split_connect - split_front)
            yaw = centre + mirror * (amplitude + connect * phase)     # straight to its near edge
            elevation = 0.0
        else:
            phase = (index - split_connect) / max(1e-9, (frames - 1) - split_connect)
            back_centre = centre + mirror * ORBIT_COVERAGE_DEGREES
            # The back phase runs its 330 deg ring first and glides home afterwards, the two parts
            # split by their travel (BACK_ORBIT_TRAVEL_FACTOR : BACK_ORBIT_HOME_FACTOR) so the
            # hand-off keeps the same per-frame drift as every other phase.
            ring_share = (BACK_ORBIT_TRAVEL_FACTOR
                          / (BACK_ORBIT_TRAVEL_FACTOR + BACK_ORBIT_HOME_FACTOR))
            if phase <= ring_share:
                yaw, elevation = _back_orbit_point(phase / max(1e-9, ring_share),
                                                   back_centre, amplitude, mirror)
            else:
                glide = (phase - ring_share) / max(1e-9, 1.0 - ring_share)
                eight_yaw, eight_elevation = _back_orbit_point(1.0, back_centre, amplitude, mirror)
                yaw = eight_yaw + (back_centre - eight_yaw) * glide
                elevation = eight_elevation * (1.0 - glide)     # -A/2 at 8 o'clock -> 0, level
        positions.append(_place(pivot, radius, yaw, elevation))
    return positions


def _pair_motion(pool, positions, pivot, surface, steps=None):
    """Per-step pixel motion of the subject along a camera path: (p95, p50, visible counts).

    The DRIFT_PERCENTILE-th percentile of the *point* displacement between consecutive poses (the
    subject's near surface moves most, so the percentile keeps a depth spike from dominating).
    `steps` measures only every `frames // steps`-th step (the drift metric does not need every
    frame); `None` measures *every* step, which is what the spiral's frame split needs. All poses go
    through ONE batched projection (`_batch_project`) - this runs dozens of times per amplitude fit,
    and the per-pose calls were the auto camera's hot loop.
    """
    frames = len(positions)
    if frames < 2:
        return None, None, None
    if steps is None:
        pairs = [(index, index + 1) for index in range(frames - 1)]
    else:
        stride = max(1, frames // max(2, int(steps)))
        pairs = [(index, index + 1) for index in range(0, frames - 1, stride)]
    if not pairs:
        return None, None, None
    busy = [positions[index] for pair in pairs for index in pair]
    u, v, z = _batch_project(pool, busy, pivot, surface, CANVAS_HEIGHT)
    count = len(pairs)
    u = u.reshape(count, 2, -1)
    v = v.reshape(count, 2, -1)
    z = z.reshape(count, 2, -1)
    visible = (z[:, 0] > 1e-4) & (z[:, 1] > 1e-4)
    counts = visible.sum(dim=1)
    motion = torch.hypot(u[:, 1] - u[:, 0], v[:, 1] - v[:, 0])
    # Per-pair percentiles of the *visible* points, vectorised: one sort per call instead of one
    # `torch.quantile` per pair (this ran ~12k single quantile calls per estimate - the hot loop).
    # `linear` interpolation between the two order statistics, exactly like torch.quantile's
    # default, so the numbers do not shift.
    masked = torch.where(visible, motion, motion.new_full((), math.inf))
    ordered, _ = torch.sort(masked, dim=1)
    span = (counts - 1).clamp(min=0).unsqueeze(1)

    def percentile(share):
        position = share * span
        low = position.floor().to(torch.long)
        high = torch.minimum(low + 1, span.to(torch.long))
        fraction = position - low.to(position.dtype)
        lower = ordered.gather(1, low).to(motion.dtype)
        upper = ordered.gather(1, high).to(motion.dtype)
        return (lower + (upper - lower) * fraction).reshape(-1)

    return percentile(DRIFT_PERCENTILE / 100.0), percentile(0.5), counts


def subject_drift(pool, positions, pivot, surface, steps=DRIFT_STEPS):
    """(worst, typical) pixel motion of the subject per frame along a camera path.

    The worst sampled step is what the speed cap has to cover; the typical one describes the rest of
    the path. Only the subject's own pixels enter: the background is never part of the subject-mode
    budget. See `_pair_motion` for the measurement itself.
    """
    p95, p50, counts = _pair_motion(pool, positions, pivot, surface, steps)
    if p95 is None or not bool((counts >= 8).any()):
        return 0.0, 0.0
    p95 = torch.where(counts >= 8, p95, p95.new_full((), -math.inf))
    worst_index = int(torch.argmax(p95))
    worst = max(float(p95[worst_index]), 0.0)
    typical = max(float(p50[worst_index]), 0.0) if bool(counts[worst_index] >= 8) else 0.0
    return worst, typical


def _largest_step(samples):
    """Longest per-frame camera step of a sampled path, in world units."""
    return max((math.dist(samples[index - 1], samples[index])
                for index in range(1, len(samples))), default=0.0)


def _cut_orbit_end(pool, surface, frames, pivot, radius, scale, size, end, direction, cap_px):
    """(end, samples, drift, typical) - the longest concluding orbit the speed cap can pay for.

    The COMPATIBILITY path: without an explicit `coverage` the path keeps the older "front O, then a
    concluding orbit that runs around to `end` degrees" choreography, so saved API calls and the
    older tests behave exactly as before. New callers ask for `ORBIT_COVERAGES` instead, which
    `_cut_back_span` below fits.

    The max camera speed is a *hard* limit for the orbit, so when the requested end cannot be paid
    the lap gives way - and only the lap: the O's amplitude was already searched by the ladder, and
    the room the framing needs never cuts anything. With the balanced frame split the path's drift is
    `k * (loop travel + lap span) / frames`, so one analytic step on the span lands close; it aims
    `ORBIT_CUT_TARGET` of the cap rather than the cap itself (a path that exactly touches the cap
    still measures a hair over it, and then the rung would be rejected), and `ORBIT_CUT_STEPS`
    rounds of it converge the rest.
    """
    yaw_amplitude = front_amplitudes(scale, size)[0]
    span = max(0.0, float(end) - yaw_amplitude)
    samples = subject_samples(frames, pivot, radius, scale, size, end, direction)
    drift, typical = subject_drift(pool, samples, pivot, surface)
    target = cap_px * ORBIT_CUT_TARGET
    for _ in range(ORBIT_CUT_STEPS):
        if drift <= target or span <= 1e-9:
            break
        span = span * max(0.0, min(1.0, target / max(1e-9, drift)))
        end = yaw_amplitude + span
        samples = subject_samples(frames, pivot, radius, scale, size, end, direction)
        drift, typical = subject_drift(pool, samples, pivot, surface)
    return end, samples, drift, typical
def _cut_back_span(pool, surface, frames, pivot, radius, scale, size, direction, cap_px,
                   view_angle=None, coverage=None):
    """(span, samples, drift, typical) - the longest back connection the speed cap can pay for.

    The max camera speed is a *hard* limit, so when the shortest full connection to the back orbit
    cannot be paid, the BACK gives way - and only the back: the front O's radius was already searched
    by the ladder, and the room the framing needs never cuts anything. The full connection is
    `ORBIT_COVERAGE_DEGREES - 2 * amplitude`: it stops where the back orbit *begins* (its near edge,
    the clock's 9 o'clock), not at the back circle's middle. Cutting it below that also drops the
    back orbit itself (see `subject_samples`), so
    the clip simply ends where the sweep stopped, at whatever azimuth that was. One analytic step
    lands close (the drift falls with the travel), it aims `ORBIT_CUT_TARGET` of the cap rather than
    the cap itself (a path that exactly touches the cap still measures a hair over it, and then the
    rung would be rejected), and `ORBIT_CUT_STEPS` rounds converge the rest.
    """
    mode = ORBIT_COVERAGE_DEFAULT if coverage is None else str(coverage)
    amplitude = front_amplitudes(scale, size)[0]
    full = max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * amplitude) if mode == ORBIT_COVERAGES[1] else 0.0
    if full <= 1e-9:                        # "O-Orbit Front": the front O takes every frame
        samples = subject_samples(frames, pivot, radius, scale, size, None, direction,
                                  view_angle=view_angle, coverage=ORBIT_COVERAGES[0])
        drift, typical = subject_drift(pool, samples, pivot, surface)
        return 0.0, samples, drift, typical
    span = full
    samples = subject_samples(frames, pivot, radius, scale, size, None, direction,
                              view_angle=view_angle, coverage=ORBIT_COVERAGES[1], back_span=span)
    drift, typical = subject_drift(pool, samples, pivot, surface)
    target = cap_px * ORBIT_CUT_TARGET
    for _ in range(ORBIT_CUT_STEPS):
        if drift <= target or span <= 1e-9:
            break
        span = span * max(0.0, min(1.0, target / max(1e-9, drift)))
        samples = subject_samples(frames, pivot, radius, scale, size, None, direction,
                                  view_angle=view_angle, coverage=ORBIT_COVERAGES[1], back_span=span)
        drift, typical = subject_drift(pool, samples, pivot, surface)
    return span, samples, drift, typical


def _orbit_candidate(pool, surface, frames, pivot, radius, scale, size, end, direction, cap_px,
                     view_angle=None, coverage=None):
    """(scale, samples, drift, typical, info) for one amplitude rung, cut to the speed cap.

    Two flavours. Without an explicit `coverage` the older concluding-orbit choreography is fitted
    and its `end` cut (`_cut_orbit_end`); with one of `ORBIT_COVERAGES` the front/back path is fitted
    and its back connection cut (`_cut_back_span`). Either way the BACK gives way first - the front O
    keeps the radius the ladder found for it.
    """
    if coverage is None:
        end, samples, drift, typical = _cut_orbit_end(pool, surface, frames, pivot, radius, scale,
                                                      size, end, direction, cap_px)
        amplitude = front_amplitudes(scale, size)[0]
        info = {"front_yaw": amplitude, "orbit_end": end,
                "front_share": balanced_front_share(amplitude, max(0.0, end - amplitude)),
                "orbit_coverage": min(FULL_CIRCLE, amplitude + max(amplitude, end))}
        return scale, samples, drift, typical, info
    if str(coverage) in (SPIRAL_COVERAGE, BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):
        # Nothing to cut here: the spiral's winding and the band stack are the node's own
        # widgets, so one sample run and one drift measurement describe the whole path.
        # The samples are the constant-speed ones (`_spiral_samples` / `_band_samples`,
        # spaced along the path) - they are what the emitted keys come from, so the fit
        # and the keys must never disagree about where a frame sits.
        samples = subject_samples(frames, pivot, radius, scale, size, None, direction,
                                  motion_pool=pool, motion_surface=surface, motion_cap=cap_px,
                                  **path_options(view_angle, coverage))
        drift, typical = subject_drift(pool, samples, pivot, surface)
        info = (_spiral_info(view_angle, direction) if str(coverage) == SPIRAL_COVERAGE
                else _band_front_back_info() if str(coverage) == BAND_FRONT_BACK_COVERAGE
                else _band_info())
        info["coverage"] = str(coverage)
        return scale, samples, drift, typical, info
    span, samples, drift, typical = _cut_back_span(pool, surface, frames, pivot, radius, scale, size,
                                                   direction, cap_px, view_angle, coverage)
    amplitude = front_amplitudes(scale, size)[0]
    full = max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * amplitude)
    back_orbit = span >= full - 1e-9 and full > 1e-9
    reached = amplitude + span                      # the azimuth the path gets to
    front_travel = LOOP_TRAVEL_FACTOR * math.radians(amplitude)
    rest_travel = math.radians(span) + ((BACK_ORBIT_TRAVEL_FACTOR + BACK_ORBIT_HOME_FACTOR)
                                        * math.radians(amplitude) if back_orbit else 0.0)
    total = front_travel + rest_travel
    # How much azimuth the path shows: the front O's own width (2A), the connection, and the back
    # orbit's width (2A again) - i.e. 180 + 2A when the whole back visit runs.
    width = 2.0 * amplitude
    info = {"front_yaw": amplitude, "orbit_end": reached, "back_span": span,
            "back_orbit": back_orbit, "orbit_coverage": width + span + (width if back_orbit else 0.0),
            "view_angle": None if view_angle is None else float(view_angle),
            "front_share": (front_travel / total) if total > 1e-9 else 1.0}
    return scale, samples, drift, typical, info


def _fit_subject_amplitude(frames, pivot, radius, surface, size, cap_px, amplitude_cap=None,
                           orbit_end=None, direction=ORBIT_DIRECTION_DEFAULT, view_angle=None,
                           coverage=None):
    """(scale, samples, drift, typical, info) - how far round the speed cap lets the camera go.

    The amplitude ladder walks up from a small loop and every rung is measured with its concluding
    orbit cut to the cap (`_cut_orbit_end`). **Auto Orbit Size is the floor for the visit's width**
    (`size`): the fit only grows the O above it, because a wider O shortens the orbit and pays for
    itself - so `info`'s winner is the rung with the *largest achieved end* among the rungs at or
    above the floor, i.e. the camera goes as far around the subject as the max camera speed pays for
    *without* giving the front up; equal ends keep the wider O ("last wins", the ladder runs
    upward). Only when the cap cannot even pay for the floor's own loop is the O shrunk below it
    (info carries `front_floor`, the caller's hint names it). When nothing meets the cap at all, the
    rung with the smallest drift is reported instead (least overshoot). `amplitude_cap` is the
    visibility pass's ceiling on the O ("a swing this wide would clip the subject"). Returns None
    when no pixel cap or no cloud is available, which sends the caller back to the world-travel fit.
    """
    if cap_px is None or cap_px <= 0.0 or not surface or surface.get("content_cloud") is None:
        return None
    pool = _decimate(surface["content_cloud"], DRIFT_POOL)
    size = max(1e-3, float(size))
    end = ORBIT_END_DEFAULT if orbit_end is None else max(ORBIT_END_MIN, float(orbit_end))
    if str(coverage) in (SPIRAL_COVERAGE, BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):
        # The spiral's winding and the band stack (start angle / count / width / start yaw /
        # step) are the node's own widgets, not rungs the ladder may trade away: fly the
        # path exactly as asked and *report* the drift, so the speed cap names the price
        # instead of silently shortening the path. (That cut is what kept a requested 840 deg
        # from ever being flown: the ladder only paid for whole rounds it could afford.)
        chosen = _orbit_candidate(pool, surface, frames, pivot, radius, 1.0, size, end,
                                  direction, cap_px, view_angle, coverage)
        chosen[4]["front_floor"] = size
        return chosen
    ceiling = max(FIT_GROW_STEP, _amplitude_ceiling(size))
    if amplitude_cap is not None:
        ceiling = min(ceiling, max(FIT_GROW_STEP, float(amplitude_cap)))
    rungs = [round(FIT_GROW_STEP * step, 3)
             for step in range(1, int(ceiling / FIT_GROW_STEP) + 1)]
    rungs = [rung for rung in rungs if rung <= ceiling + 1e-9] or []
    if not rungs or rungs[-1] < ceiling - 1e-9:
        rungs.append(ceiling)              # unrounded: the ladder never exceeds the view limit
    # Auto Orbit Size is the *floor* for the visit's width: the fit only ever grows it (a wider O
    # shortens the orbit, so it pays for itself), and when the speed cap cannot even pay for the
    # floor's own loop, the floor gives way - reported - rather than the cap.
    floor = min(ceiling, max(FIT_GROW_STEP, float(size)))
    preferred = [rung for rung in rungs if rung >= floor - 1e-9] or list(rungs)

    def scan(candidates, key="end"):
        """(best, closest): inside the cap the largest `key` wins - the orbit's end, the O's width
        or the area between the two (the fallback's trade), else the smallest overshoot."""
        winner = None
        fallback = None
        for scale in candidates:
            candidate = _orbit_candidate(pool, surface, frames, pivot, radius, scale, size, end,
                                         direction, cap_px, view_angle, coverage)
            if candidate[2] <= cap_px + 1e-9:
                if key == "width":
                    score = candidate[0]
                elif key == "area":
                    score = candidate[4]["front_yaw"] * candidate[4]["orbit_end"]
                else:
                    score = candidate[4]["orbit_end"]
                if winner is None or score >= score_of(winner, key) - 1e-6:
                    winner = candidate
            elif fallback is None or candidate[2] < fallback[2] - 1e-9 or (
                    abs(candidate[2] - fallback[2]) <= 1e-9
                    and candidate[4]["orbit_end"] > fallback[4]["orbit_end"] + 1e-6):
                fallback = candidate
        return winner, fallback

    def score_of(candidate, key):
        if key == "width":
            return candidate[0]
        if key == "area":
            return candidate[4]["front_yaw"] * candidate[4]["orbit_end"]
        return candidate[4]["orbit_end"]

    best, closest = scan(preferred)
    if best is None and floor > rungs[0] + 1e-9:
        # The floor's own loop is unpayable (too few frames for this subject), so the fit has to
        # trade width against reach. Neither extreme works - the largest end answers "+/-3 deg and a
        # 351 deg orbit" (the first QC run caught exactly that: the most azimuth, nothing to see),
        # and the widest rung answers "1.0x and a 170 deg orbit" (the front, but the back never
        # comes). The fallback maximises their *product* instead: the front's width times the
        # azimuth the path covers - the compromise that keeps a readable front arc and still takes
        # the camera past the subject's back.
        best, closest = scan(rungs, "area")
    chosen = best if best is not None else closest
    chosen[4]["front_floor"] = floor
    return chosen


def _front_travel_share(amplitude, span, back_orbit):
    """The front O's share of the frames: the phases split by their travel (equal per-frame drift).

    The front O travels its whole circle's arc (the crane to 2 o'clock and the 330 deg sweep
    together pay for the full loop), the connection its level azimuth and the back orbit the 330
    deg loop it really flies (9 -> 12 -> 3 -> 6 -> 8 o'clock, `BACK_ORBIT_TRAVEL_FACTOR`) plus its
    glide home to the dial's centre (`BACK_ORBIT_HOME_FACTOR`).
    """
    front = LOOP_TRAVEL_FACTOR * math.radians(amplitude)
    rest = math.radians(span) + ((BACK_ORBIT_TRAVEL_FACTOR + BACK_ORBIT_HOME_FACTOR)
                                 * math.radians(amplitude) if back_orbit else 0.0)
    total = front + rest
    return front / total if total > 1e-9 else 1.0


def _fit_orbit_world(frames, pivot, radius, surface, size, budget, orbit_end=None,
                     direction=ORBIT_DIRECTION_DEFAULT, view_angle=None, coverage=None):
    """(scale, samples, travel, info) - the world-unit fit for the orbit, speed cap first.

    The fallback when no pixel cap is available (no framing, direct calls): the same amplitude
    ladder, but the budget is `max_speed` in world units. The back part is cut to the budget with the
    same analytic step the pixel fit uses, so the path never runs faster than the parameter allows;
    only the coverage gives way, and the caller reports it. Without an `ORBIT_COVERAGES` mode the
    older concluding orbit is fitted and cut exactly as it always was.
    """
    if coverage is None:                       # legacy: the concluding orbit to `orbit_end`
        end = ORBIT_END_DEFAULT if orbit_end is None else max(ORBIT_END_MIN, float(orbit_end))
        samples_of = lambda scale: subject_samples(frames, pivot, radius, scale, size, end, direction)
        scale, samples, travel = _fit_amplitude(samples_of, budget)
        if travel > budget:
            amplitude = front_amplitudes(scale, size)[0]
            span = max(0.0, end - amplitude)
            if span > 1e-9:
                end = amplitude + span * max(0.0, min(1.0, budget / max(1e-9, travel)))
                cut_samples = subject_samples(frames, pivot, radius, scale, size, end, direction)
                cut_travel = _largest_step(cut_samples)
                if cut_travel < travel:
                    samples, travel = cut_samples, cut_travel
        amplitude = front_amplitudes(scale, size)[0]
        info = {"front_yaw": amplitude, "orbit_end": end,
                "front_share": balanced_front_share(amplitude, max(0.0, end - amplitude)),
                "orbit_coverage": min(FULL_CIRCLE, amplitude + max(amplitude, end))}
        return scale, samples, travel, info
    if str(coverage) in (SPIRAL_COVERAGE, BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):
        # Same rule as the pixel fit: the winding and the band stack are the node's own widgets, so
        # they are not a ladder to trade away - one sample run, measured, never cut.
        options = path_options(view_angle, coverage)
        samples = subject_samples(frames, pivot, radius, 1.0, size, None, direction, **options)
        info = (_spiral_info(view_angle, direction) if str(coverage) == SPIRAL_COVERAGE
                else _band_front_back_info() if str(coverage) == BAND_FRONT_BACK_COVERAGE
                else _band_info())
        return 1.0, samples, _largest_step(samples), info
    options = path_options(view_angle, coverage)
    # The connection ends at the back orbit's near edge (the clock's 9 o'clock), so the azimuth it
    # has to cover is the circle-to-circle distance MINUS the O's own diameter - the back orbit's
    # first half (over the head) is what closes the rest.
    full_of = lambda scale: (max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * front_amplitudes(scale, size)[0])
                             if str(coverage) == ORBIT_COVERAGES[1] else 0.0)
    samples_of = lambda scale: subject_samples(frames, pivot, radius, scale, size, None, direction,
                                               **dict(options, back_span=full_of(scale)))
    scale, samples, travel = _fit_amplitude(samples_of, budget)
    span = full_of(scale)
    if travel > budget and span > 1e-9:
        span *= max(0.0, min(1.0, budget / max(1e-9, travel)))
        cut_samples = subject_samples(frames, pivot, radius, scale, size, None, direction,
                                      **dict(options, back_span=span))
        cut_travel = _largest_step(cut_samples)
        if cut_travel < travel:
            samples, travel = cut_samples, cut_travel
    amplitude = front_amplitudes(scale, size)[0]
    full = max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * amplitude)
    back_orbit = span >= full - 1e-9 and full > 1e-9
    width = 2.0 * amplitude
    info = {"front_yaw": amplitude, "orbit_end": amplitude + span, "back_span": span,
            "back_orbit": back_orbit, "front_share": _front_travel_share(amplitude, span, back_orbit),
            "orbit_coverage": width + span + (width if back_orbit else 0.0)}
    return scale, samples, travel, info


def scene_coverage(surface, radius, content_radius, frames, orbit_size=1.0,
                   budget_per_frame=None):
    """The plan behind the scene survey: rows, their spacing, their sweep and their heights.

    Rows are spaced like drone mapping flight lines. The frame's horizontal footprint at the survey
    distance (`2 * radius * tan(hfov / 2)`, with hfov from Meridian's 55 deg vertical FOV and the
    still's aspect ratio) is overlapped by SCENE_LATERAL_OVERLAP between neighbouring rows - the
    70-80 % side-lap the mapping guides prescribe - so a viewer standing between two rows always
    has a render basis that already saw that surface from an angle. The rows span the *scene's own
    width* (plus SCENE_COVERAGE_MARGIN, times the Auto Orbit Size) instead of a full circle: a
    depth reprojection cannot invent the far side of a scene, and side parallax is what a walk-in
    viewer looks along.

    The count is bounded by SCENE_MIN/MAX_LANES, by the frame budget (SCENE_FRAMES_PER_LANE per
    row) and - when `budget_per_frame` (the speed cap in units) is given - by what the camera may
    actually travel: a row costs its lateral sweep plus its climb, so a short frame budget buys
    *fewer rows with the full sweep* instead of many rows crammed into a short one. The outermost
    row never swings past SCENE_YAW_LIMIT; a scene wider than that is the case for a larger Auto
    Orbit Distance (the drone rule: fly higher for a bigger area, and the console line says so).
    Both heights follow the Auto Orbit Size as well, so the whole envelope grows and shrinks with
    it. Returns the plan plus the numbers behind it, which the console summary prints.
    """
    size = max(1e-3, _finite(orbit_size, "Orbit size"))
    radius = max(1e-6, float(radius))
    content_radius = max(1e-6, float(content_radius))
    surface = surface or {}
    fallback_aspect = 16.0 / 9.0                     # no surface (tests): a normal wide still
    aspect = float(surface.get("aspect") or fallback_aspect)
    hfov = float(surface.get("hfov") or 2.0 * math.degrees(math.atan(
        math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0) * aspect)))
    half_width = max(1e-6, float(surface.get("lateral_half_width") or content_radius))
    half_width *= SCENE_COVERAGE_MARGIN * size
    half_yaw = max(5.0, min(SCENE_YAW_LIMIT * size, math.degrees(math.asin(
        min(1.0, half_width / radius)))))
    footprint = 2.0 * radius * math.tan(math.radians(hfov) / 2.0)
    lateral_span = 2.0 * radius * math.sin(math.radians(half_yaw))
    low = SCENE_ELEVATION_LOW * size
    high = SCENE_ELEVATION_HIGH * size
    climb = radius * abs(math.sin(math.radians(high)) - math.sin(math.radians(low)))
    row_travel = lateral_span + climb
    rows = int(math.ceil(lateral_span / max(1e-6, footprint * (1.0 - SCENE_LATERAL_OVERLAP)))) + 1
    rows = min(SCENE_MAX_LANES, rows)
    if budget_per_frame:
        affordable = int(int(frames) * float(budget_per_frame) / max(1e-6, row_travel))
        rows = min(rows, max(1, affordable))
    rows = max(SCENE_MIN_LANES, min(rows, int(frames) // SCENE_FRAMES_PER_LANE))
    step = lateral_span / max(1, rows - 1)
    return {
        "rows": rows, "half_yaw": half_yaw, "lateral_span": lateral_span, "lane_step": step,
        "lane_overlap": max(0.0, min(1.0, 1.0 - step / max(1e-6, footprint))),
        "half_width": half_width, "hfov": hfov, "footprint": footprint, "row_travel": row_travel,
        "low_elevation": low, "high_elevation": high,
    }


def scene_survey_fill(surface, size, height=CANVAS_HEIGHT):
    """(fill, metrics) - the stand-off that makes the rows reach the scene *and* fill the frame.

    The built-in SCENE_FILL is a sane default, but it cannot know the scene: a wide scene's outermost
    row never reaches its edge (its azimuth would have to swing past the view limit) and a narrow one
    leaves most of every frame empty. The ratio is therefore solved from the scene's own width:

    * **reach** - the outermost row's lateral reach is `radius * sin(yaw_limit)`, so the radius has to
      be at least `half_width / sin(yaw_limit)`: the drone rule "fly higher for a bigger area". This
      is a hard lower bound, coverage beats detail.
    * **fill** - the frame's footprint is `2 * radius * tan(hfov / 2)`. If that already spans the whole
      scene, one row would cover everything and every frame would be mostly empty, so the radius is
      pulled in until the footprint is about `scene_width / SCENE_MIN_LANES` - never closer than
      `SCENE_FILL_MIN` content radii (the collision guard's own floor).

    Returns the fill ratio (in content radii) plus the numbers behind it, which the console prints.
    """
    size = max(1e-3, float(size))
    content_radius = max(1e-6, float(surface["content_radius"]))
    aspect = float(surface.get("aspect") or (16.0 / 9.0))
    hfov = float(surface.get("hfov") or 2.0 * math.degrees(math.atan(
        math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0) * aspect)))
    half_width = max(1e-6, float(surface.get("lateral_half_width") or content_radius))
    half_width *= SCENE_COVERAGE_MARGIN * size
    yaw_limit = max(5.0, min(SCENE_YAW_LIMIT * size, 85.0))
    reach_fill = half_width / max(1e-6, math.sin(math.radians(yaw_limit))) / content_radius
    footprint_max = 2.0 * half_width / max(1.0, float(SCENE_MIN_LANES))
    fill_cap = footprint_max / max(1e-6, 2.0 * math.tan(math.radians(hfov) / 2.0)) / content_radius
    # reach is a hard lower bound (coverage beats detail), the frame-fill preference is soft: the
    # ratio is the built-in stand-off, capped by the footprint rule, but never below the reach - and
    # finally clamped into the orbit bounds.
    ratio = min(SCENE_FILL_MAX, max(SCENE_FILL_MIN, reach_fill,
                                    min(SCENE_FILL, max(SCENE_FILL_MIN, fill_cap))))
    radius = ratio * content_radius
    reach = radius * math.sin(math.radians(yaw_limit)) / max(1e-6, half_width)
    return ratio, {
        "fill": ratio, "reach_fill": reach_fill, "fill_cap": fill_cap,
        "yaw_limit": yaw_limit, "half_width": half_width,
        "footprint": 2.0 * radius * math.tan(math.radians(hfov) / 2.0),
        "reach": reach, "radius": radius,
    }


def subject_share_curve(points, positions, pivot, surface, height=CANVAS_HEIGHT):
    """Per pose: (area share of the picture, is the whole subject inside, does it show at all).

    What a scene survey needs to know about the subject inside it: how big it appears and whether a
    frame shows it *completely*. A lateral survey sweeps past the subject, so "shows at all" and
    "completely" are different questions - the console reports the median share and how many frames
    hold the subject whole.
    """
    width = float(height) * float(surface.get("aspect") or (16.0 / 9.0))
    canvas_area = float(height) * width
    curve = []
    for position in positions:
        u, v, z = project_points(points, position, pivot, surface, height)
        visible = z > 1e-4
        if int(visible.sum()) < 8:
            curve.append((0.0, False, False))
            continue
        u, v = u[visible], v[visible]
        low_u = _percentile(u, FRAMING_PERCENTILE_LOW / 100.0)
        high_u = _percentile(u, FRAMING_PERCENTILE_HIGH / 100.0)
        low_v = _percentile(v, FRAMING_PERCENTILE_LOW / 100.0)
        high_v = _percentile(v, FRAMING_PERCENTILE_HIGH / 100.0)
        centre_u, centre_v = (low_u + high_u) / 2.0, (low_v + high_v) / 2.0
        shown = 0.0 <= centre_u <= width and 0.0 <= centre_v <= float(height)
        whole = low_u >= 0.0 and high_u <= width and low_v >= 0.0 and high_v <= float(height)
        share = max(0.0, (high_u - low_u)) * max(0.0, (high_v - low_v)) / canvas_area
        curve.append((share if shown else 0.0, whole, shown))
    return curve


def scene_samples(frames, pivot, radius, plan, scale=1.0):
    """Per-frame positions of the scene survey: boustrophedon rows across the scene's width.

    Each row sweeps the azimuth from one end of the survey to the other while the elevation
    travels from the plan's low to its high (or the other way round), and the next row runs back
    with the elevation reversed - a raster scan of the (azimuth, elevation) envelope, so every
    lateral station is seen from below and from above, consecutive rows continue each other
    instead of jumping, and the turns stay gentle (cosine easing). `scale` is the speed-fit
    amplitude: it shortens the sweep and the height range when the frame count cannot pay for the
    full one.
    """
    frames = int(frames)
    rows = max(1, int(plan["rows"]))
    half_yaw = max(0.0, float(plan["half_yaw"])) * scale
    low = float(plan["low_elevation"]) * scale
    high = float(plan["high_elevation"]) * scale
    span = max(1, frames - 1)
    positions = []
    for index in range(frames):
        phase = index / span                            # 0..1 over the whole survey
        row = min(rows - 1, int(phase * rows))
        inner = min(1.0, max(0.0, phase * rows - row))
        eased = (1.0 - math.cos(math.pi * inner)) / 2.0     # eases out of and into every turn
        outbound = row % 2 == 0
        yaw = (-half_yaw + 2.0 * half_yaw * eased) if outbound else (half_yaw - 2.0 * half_yaw * eased)
        elevation = (low + (high - low) * eased) if outbound else (high - (high - low) * eased)
        positions.append(_place(pivot, radius, yaw, elevation))
    return positions


def automatic_keys(frames, pivot, radius, content_radius, target, max_speed=DEFAULT_MAX_SPEED,
                   orbit_size=1.0, surface=None, cap_px=None, amplitude_cap=None, orbit_end=None,
                   direction=ORBIT_DIRECTION_DEFAULT, view_angle=None, coverage=None):
    """(keys, info) for the automatic path of `target`: subject orbit or scene lateral survey.

    The keys are the Catmull-Rom control points the Geometry node samples (evenly spaced, whole
    frame indices). `radius` is the camera-to-pivot distance (Auto Subject Fill framing or the Auto
    Orbit Distance widget) and `orbit_size` the user's O-orbit size.

    Two budgets decide the amplitudes. The **subject** path is fitted in *pixels* whenever `cap_px`
    is given (the framing computes it from the subject's apparent size): the path runs as far around
    as `orbit_end` asks for (`ORBIT_END_DEFAULT` = 360 deg, i.e. back to the start point) as long as
    the subject's own per-frame drift stays inside the cap - when the *max camera speed* cannot pay
    for the requested end, the concluding orbit is cut there and the console names the levers, so the
    user's speed parameter is never silently exceeded. The frames are split between the O and the lap
    so that both run at the same drift (`balanced_front_share`), which is the cheapest way to honour
    the cap. `amplitude_cap` is the visibility pass's ceiling on the O ("a swing this wide would clip
    the subject"), the **scene** survey is fitted in world units against `max_speed x content
    radius` - the same fit that also backs the subject path when no pixel cap is available (unit
    tests, direct calls). The returned `info` carries `front_yaw` (the O's swing), `orbit_end` (the
    end the path achieved), `front_share` (the frames the O got) and `orbit_coverage` (how much
    azimuth the path covers around the subject).
    """
    frames = validate_frames(frames)
    budget = max(1e-6, _finite(max_speed, "Max camera speed")) * max(1e-6, content_radius)
    orbit_size = max(1e-3, _finite(orbit_size, "Orbit size"))
    plan = None
    drift = typical = None
    meta = {}
    if str(target).strip().lower() == SUBJECT_TARGET:
        style = ("spherical spiral" if str(coverage) == SPIRAL_COVERAGE
                 else "banded orbits" if str(coverage) == BAND_COVERAGE
                 else "banded orbits front & back" if str(coverage) == BAND_FRONT_BACK_COVERAGE
                 else "front O-orbit + closing orbit")
        fitted = _fit_subject_amplitude(frames, pivot, radius, surface, orbit_size, cap_px,
                                        amplitude_cap, orbit_end, direction, view_angle, coverage)
        if fitted is None:
            fit_name = "world travel"
            scale, samples, travel, meta = _fit_orbit_world(frames, pivot, radius, surface,
                                                            orbit_size, budget, orbit_end,
                                                            direction, view_angle, coverage)
        else:
            fit_name = "subject pixels"
            scale, samples, drift, typical, meta = fitted
            travel = _largest_step(samples)
    else:
        style = "lateral survey rows"
        fit_name = "world travel"
        plan = scene_coverage(surface, radius, content_radius, frames, orbit_size, budget)
        samples_of = lambda scale: scene_samples(frames, pivot, radius, plan, scale)
        scale, samples, travel = _fit_amplitude(samples_of, budget)
    # The spiral's keys follow its own turning (`_spiral_key_frames`): its fastest heading change is
    # right after the pole, where an evenly spaced key list would cut the corner the renderer's
    # spline then follows. The banded path turns sharply at every band end and every elevation step,
    # so it uses the same turning walk - a frame-even list would cut exactly those corners. Every
    # other path keeps the even frame list.
    ticks = (_spiral_key_frames(samples, frames)
             if str(coverage) in (SPIRAL_COVERAGE, BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE)
             else _key_frames(frames))
    keys = [{
        "pos": [round(value, 6) for value in samples[tick]],
        "look": [round(value, 6) for value in _aim(pivot, samples[tick])],
        "src": int(tick),
        "t": int(tick),
    } for tick in ticks]
    info = {"style": style, "amplitude_scale": scale, "travel_per_frame": travel,
            "budget_per_frame": budget, "keys": len(keys), "fit": fit_name,
            "orbit_coverage": meta.get("orbit_coverage")}
    if drift is not None:
        info.update({"drift_px": drift, "drift_typical_px": typical,
                     "drift_cap_px": float(cap_px)})
    if plan is not None:
        info.update(plan)
    else:
        # the subject orbit: how wide the O swings, how far round it got and who got which frames
        amplitude = meta["front_yaw"]
        if coverage is None:                   # legacy: the concluding orbit to `orbit_end`
            requested = ORBIT_END_DEFAULT if orbit_end is None else max(ORBIT_END_MIN,
                                                                        float(orbit_end))
            span = max(0.0, meta["orbit_end"] - amplitude)
            info.update({"front_yaw": amplitude, "lap_span": span,
                         "orbit_end": meta["orbit_end"], "orbit_end_requested": requested,
                         "front_share": meta.get("front_share", 1.0),
                         "orbit_direction": str(direction), "coverage": None,
                         "back_orbit": False, "orbit_coverage": meta.get("orbit_coverage")})
        elif str(coverage) == SPIRAL_COVERAGE:  # the spherical spiral: the winding the node asked for
            requested = _spiral_sweep()
            span = 0.0
            # The spiral's "amplitude" is its winding (the end angle is the Spiral End Angle widget),
            # that winding is the user's Spiral End parameter - the fit does not scale it, so the
            # scale really is 1.0 and the summary reports the drift instead of a cut.
            info.update({"front_yaw": 0.0, "lap_span": 0.0,
                         "orbit_end": meta.get("orbit_end"),
                         "orbit_end_requested": meta.get("orbit_end"),
                         "front_share": 1.0, "orbit_direction": str(direction),
                         "view_angle": meta.get("view_angle"), "coverage": str(coverage),
                         "back_orbit": False, "orbit_coverage": meta.get("orbit_coverage"),
                         "spiral_end": meta.get("spiral_end"),
                         "spiral_end_arc": meta.get("spiral_end_arc"),
                         "spiral_slope": meta.get("spiral_slope"),
                         "spiral_end_clock": meta.get("spiral_end_clock"),
                         "spiral_end_elevation": meta.get("spiral_end_elevation")})
        elif str(coverage) in (BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):  # the band stack: exactly the widgets asked for
            requested = float(meta.get("band_width") or BAND_WIDTH_DEFAULT)
            span = 0.0
            # Same story as the spiral: the band geometry IS the user's widgets (start/stop angle,
            # count, width, start yaw), so nothing is fitted and the scale really is 1.0 -
            # the summary reports the drift and the band list instead of a cut.
            info["amplitude_scale"] = 1.0
            info.update({"front_yaw": 0.0, "lap_span": 0.0,
                         "orbit_end": meta.get("orbit_end"),
                         "orbit_end_requested": meta.get("orbit_end"),
                         "front_share": 1.0, "orbit_direction": str(direction),
                         "view_angle": None, "coverage": str(coverage),
                         "back_orbit": bool(meta.get("band_front_back")),
                         "orbit_coverage": meta.get("orbit_coverage"),
                         **{key: value for key, value in meta.items()
                            if key.startswith("band_")}})
        else:                                  # the front/back path: the request is the full back
            requested = amplitude + max(0.0, ORBIT_COVERAGE_DEGREES - 2.0 * amplitude)
            span = max(0.0, meta.get("back_span", 0.0))
            info.update({"front_yaw": amplitude, "lap_span": span,
                     "orbit_end": meta["orbit_end"], "orbit_end_requested": requested,
                     "front_share": meta["front_share"], "orbit_direction": str(direction),
                     "view_angle": None if view_angle is None else float(view_angle),
                     "coverage": coverage, "back_orbit": bool(meta.get("back_orbit")),
                     "front_floor": meta.get("front_floor"),
                     "spiral_end": meta.get("spiral_end"),
                     "spiral_end_arc": meta.get("spiral_end_arc"),
                     "spiral_slope": meta.get("spiral_slope"),
                     "spiral_end_clock": meta.get("spiral_end_clock"),
                     "spiral_end_elevation": meta.get("spiral_end_elevation"),
                     "band_start_angle": meta.get("band_start_angle"),
                     "band_stop_angle": meta.get("band_stop_angle"),
                     "band_count": meta.get("band_count"),
                     "band_width": meta.get("band_width"),
                     "band_start_yaw": meta.get("band_start_yaw"),
                     "band_elevations": meta.get("band_elevations"),
                     "band_lowest": meta.get("band_lowest"),
                     "band_highest": meta.get("band_highest"),
                     # The 'Front & Back' variant's own numbers (None for the plain stack): the
                     # front half's end, the azimuth the wrap reaches and the flag itself.
                     "band_front_back": meta.get("band_front_back"),
                     "band_front_end_yaw": meta.get("band_front_end_yaw"),
                     "band_front_end_elevation": meta.get("band_front_end_elevation"),
                     "band_back_start_yaw": meta.get("band_back_start_yaw"),
                     "band_end_yaw": meta.get("band_end_yaw"),
                     "band_end_elevation": meta.get("band_end_elevation"),
                     "orbit_coverage": min(FULL_CIRCLE, meta.get(
                         "orbit_coverage",
                         amplitude + max(amplitude, meta["orbit_end"])))})
    return keys, info


def document_from_keys(frames, keys, name, description, extra=None):
    """The MERIDIAN_CAMERA_PATH JSON for a finished key list (used by both camera paths).

    `extra` merges additional top-level fields into the document (the automatic path records which
    depth model the estimate ran on - see `render_depth_aligned`, which warns when the render uses
    a different one, because two models' depth maps do not share a scale and the aim would then
    land at the wrong depth).
    """
    frames = validate_frames(frames)
    document = {"name": name, "description": description, "frames": frames,
                "stations": ["Auto"], "path": keys}
    for key, value in (extra or {}).items():
        if value is not None:
            document[key] = value
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def probe_surface(reference, target=SUBJECT_TARGET, subject_mask=None, model_size="",
                  depth_res=AUTO_DEPTH_RES, device=None, depth_fn=None, depth_relief=1.0):
    """One depth pass -> everything the automatic camera mode needs to know.

    Returns the scene cloud, the target's *geometric pivot* (the midpoint of its depth profile's
    bounding box, not the dense-surface median), the enclosing radius and the labels for the
    summary. Both automatic paths start here: the automatic one builds its orbit from it, the
    manual one only takes the pivot (plus the user's offset) - and both run the collision guard
    against this cloud. `depth_fn` injects a depth map instead of running a depth model (tests).

    The depth is put in the **renderer's gauge** first (`fast_depth.cloud_gauge`): the Geometry node
    unprojects the percentile-mapped depth, not the model's raw output, so a rig built in raw units
    lands at the wrong depth *and* scale (a key's x/y follow its z) and the subject drifts out of
    the frame along the path.
    """
    target = str(target or SUBJECT_TARGET).strip().lower()
    if target not in AUTO_TARGETS:
        raise ValueError(f"Automatic camera target must be one of {', '.join(AUTO_TARGETS)}.")
    depth = None
    depth_model = "(injected depth map)"
    if depth_fn is not None:
        depth = depth_fn(reference)
    else:
        depth, depth_model = depth_from_reference(reference, model_size=model_size,
                                                  depth_res=depth_res, device=device)
    # The renderer unprojects the *mapped* depth (the `DEPTH_NEAR .. DEPTH_FAR` window, see
    # `fast_depth.cloud_gauge`), so the estimator has to live in that same gauge: the keys it emits
    # are in units of the mapped median and the render scales them by `zm`. Estimating in the
    # model's raw units instead put the whole rig at the wrong depth *and* scale (a key's x/y follow
    # its z), which is what made the orbit circle a point in front of the subject.
    depth = fast_depth.cloud_gauge(depth)
    depth_relief = fast_depth.resolve_depth_relief(depth_relief)
    # Relief is a geometry transform, not a render-only effect: the face patch, subject
    # pivot, collision surface and emitted camera keys must all see the exact point cloud
    # the Geometry node will render. Apply it in the renderer's gauge before unprojection.
    depth = fast_depth.apply_depth_relief(depth, depth_relief)
    cloud = surface_points(depth)
    scene_pivot, scene_extents = geometric_pivot(cloud)
    scene_radius = pivot_radius(cloud, scene_pivot)
    if target == SUBJECT_TARGET:
        points, source_label = subject_points(depth, mask=subject_mask)
        pivot, extents = geometric_pivot(points)
        radius = pivot_radius(points, pivot)
        subject_cloud, subject_pivot = points, [float(value) for value in pivot]
        subject_radius, subject_source = float(radius), source_label
    else:
        points, source_label = cloud, "whole surface (scene)"
        pivot, extents, radius = scene_pivot, scene_extents, scene_radius
        # The *subject inside the scene*: the same near-layer/mask heuristic the subject target
        # uses. A survey cannot keep it in frame while it sweeps away, but it can report how big
        # the subject appears and which part of the survey shows it - and the console line warns
        # when it is too small to read.
        subject_cloud, subject_pivot, subject_radius, subject_source = None, list(scene_pivot), \
            float(scene_radius), "none"
        try:
            candidate, candidate_source = subject_points(depth, mask=subject_mask)
        except ValueError:
            candidate, candidate_source = None, "none"
        if candidate is not None and candidate_source != NO_SUBJECT_SOURCE:
            candidate_pivot, _ = geometric_pivot(candidate)
            subject_cloud = candidate
            subject_pivot = [float(value) for value in candidate_pivot]
            subject_radius = float(pivot_radius(candidate, candidate_pivot))
            subject_source = candidate_source
    grid = depth[0] if depth.ndim == 3 else depth
    aspect = float(grid.shape[-1]) / max(1.0, float(grid.shape[-2]))
    hfov = 2.0 * math.degrees(math.atan(
        math.tan(math.radians(fast_depth.VFOV_DEGREES) / 2.0) * aspect))
    # The renderer scales a camera path by the cloud's median depth (`zm` in the fast-depth
    # backend, `--pivot`'s third number): the emitted keys are relative to it, like the manual
    # Camera Path Configurator's ("median-depth units").
    median_depth = float(cloud[:, 2].median()) if cloud.shape[0] else 1.0
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
        "content_cloud": points,
        "subject_cloud": subject_cloud,
        "subject_pivot": subject_pivot,
        "subject_radius": subject_radius,
        "subject_source": subject_source,
        "median_depth": max(1e-6, median_depth),
        "depth_relief": float(depth_relief),
        "depth_model": depth_model,
        # The CANVAS aspect the render actually flies. The face orbit projects the SAM3 box
        # onto the cloud with `width = height * aspect`, and the box is normalised in the
        # SOURCE while the cloud's u/v live in the CANVAS - on the Hanks still those are
        # 3440x2312 (1.488) vs 736x544 (1.353), a 10 % horizontal error that shifts the
        # selection sideways. That is the observed "the face orbit aims slightly left of
        # the head". `_bucket_480` is deterministic from the source dims, so this needs no
        # ordering trickery the way the measured vfov did.
        "render_aspect": ((lambda _b: (_b[0] / max(1, _b[1])))(
            fast_depth._bucket_480(int(reference.shape[2]), int(reference.shape[1])))),
        "aspect": aspect, "hfov": hfov,
        # `vfov` derived from the horizontal FOV and the cloud aspect, so the FACE ORBIT
        # can project with the same focal the cloud was built with. Without it
        # `_project_with_pose` falls back to the hard-coded 55 deg (a 1.509x error in f on
        # the Hanks still), corrupting both the face-box selection and the fill metric.
        # tan(vfov/2) = tan(hfov/2) / aspect
        # The MEASURED value wins. `hfov` is a derived quantity and its inversion lands on
        # 55.00000000000001 (the hard-coded assumption, with float noise) on this scene -
        # and because it is truthy it SHADOWED `LAST_VFOV_DEGREES` in an `or` chain, so the
        # face orbit kept measuring the head at 55 deg instead of the measured 38.07 deg.
        "vfov": (getattr(fast_depth, "LAST_VFOV_DEGREES", None)
                 or ((math.degrees(2.0 * math.atan(
                     math.tan(math.radians(float(hfov)) / 2.0) / max(float(aspect), 1e-6)))
                     if hfov else None))),
        "lateral_half_width": float(scene_extents[0]) / 2.0,
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
    # FACE KEYS ARE DIFFERENT. `apply_face_orbit` deliberately parks the camera close to
    # the head - which IS scene geometry - and it already enforces its own stand-off
    # (FACE_ORBIT_CLEARANCE x the head radius, from `head_pivot`). Applying the generic
    # content-radius margin to those keys shoves the zoom straight back out to the orbit
    # radius: the render then shows no zoom at all while the orbit reports success. So a
    # face key is held to a much smaller clearance - just enough that it cannot sit inside
    # the cloud - and the zoom survives.
    face_clearance = min(clearance, FACE_ORBIT_COLLISION_MARGIN * unit)
    clearances = [face_clearance if key.get("face") else clearance for key in keys]
    stride = max(1, int(cloud.shape[0]) // COLLISION_MAX_POINTS)
    pool = cloud[::stride]
    pivot = torch.tensor(surface["pivot"], dtype=pool.dtype, device=pool.device)
    positions = torch.tensor([key["pos"] for key in keys], dtype=pool.dtype, device=pool.device)
    distances = torch.cdist(positions, pool).amin(dim=1)
    worst_before = float(distances.min()) / unit
    fixed = set()
    for _ in range(COLLISION_ITERATIONS):
        offenders = torch.nonzero(
            torch.tensor([float(distances[i]) < clearances[i] for i in range(len(keys))],
                         device=distances.device),
            as_tuple=False).flatten().tolist()
        if not offenders:
            break
        for index in offenders:
            fixed.add(index)
            direction = positions[index] - pivot
            length = float(direction.norm())
            if length < 1e-9:
                direction, length = positions.new_tensor([0.0, 0.0, -1.0]), 1.0
            push = (clearances[index] - float(distances[index]) + 1e-3) * 1.15
            positions[index] = positions[index] + (direction / length) * push
        distances = torch.cdist(positions, pool).amin(dim=1)
    for index in sorted(fixed):
        keys[index]["pos"] = [round(value, 6) for value in positions[index].tolist()]
    return keys, len(fixed), worst_before, float(distances.min()) / unit


def estimate_camera_path(reference, frames, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED,
                         subject_mask=None, model_size="", depth_res=AUTO_DEPTH_RES,
                         device=None, depth_fn=None, pivot_offset=(0.0, 0.0, 0.0),
                         orbit_distance=None, orbit_size=None, subject_fill=None, orbit_end=None,
                         direction=ORBIT_DIRECTION_DEFAULT, view_angle=None, coverage=None,
                         orbit_amplitude=None, spiral_end=None, spiral_end_arc=None,
                         spiral_slope=None, bands_start_angle=None, bands_count=None,
                         bands_width=None, bands_start_yaw=None, bands_stop_angle=None,
                         orbit_look=None,
                         face_box=None, face_fill=None, face_span=None, depth_relief=1.0):
    """(signal JSON, summary) for one still, with the O Orbit Angle and the spiral widgets applied.

    `orbit_amplitude` (deg) is the front O's angular radius - the swing AND the rise, because the O
    is one circle. It is the node's "O Orbit Angle" widget: a smaller value keeps the automatic
    subject orbit flatter / less steep (and narrower), a larger one climbs higher and reaches
    further round. `None` keeps the built-in `FRONT_ORBIT_AMPLITUDE`; a value is clamped to
    `FRONT_ORBIT_ANGLE_MIN .. FRONT_ORBIT_LIMIT`. `spiral_end` (deg) is the *Spiral* coverage's
    winding around the spiral's axis - the node's "Spiral Winding" widget; `None` keeps
    `SPIRAL_END_DEFAULT` (840 deg = two and a third rounds) and 0 flies the plain quarter circle.
    `spiral_end_arc` (deg) is the Spiral coverage's **end angle** - the radius of the O-orbit family
    the spiral flies at its last frame, 0 at the first - the node's "Spiral End Angle" widget (the
    Auto Orbit Angle slot while Spiral mode is picked): `None` keeps `SPIRAL_END_ARC_DEFAULT`
    (90 deg = the picture's own plane, a side view) and a smaller value ends the spiral earlier.
    `spiral_slope` (deg) is the "Spiral Center Slope": the lean of that axis in the vertical plane
    through the view axis, 0 = the view axis itself (the first frame is the framed frontal view),
    +90 = straight up ("from straight above"), -90 = straight down; clamped to
    `SPIRAL_SLOPE_MIN .. SPIRAL_SLOPE_MAX` and ignored by every other coverage. The `bands_*` group
    belongs to the "Banded Orbits" coverage (and its "Banded Orbits Front & Back" variant, which
    flies the same stack and then repeats it mirrored and reversed on the back side) and is ignored
    by every other one: `bands_start_angle`
    (deg) is the elevation of the FIRST band - the node's "Band Start Angle", 0 = level with the
    pivot, -90 = from straight below the subject (looking up at the feet), +90 = from straight above
     its head; `bands_stop_angle` is the elevation of the final band, and `bands_count` is how many
     bands (1, 2, 3 ...), distributed evenly between start and stop; `bands_width` (deg, 1..360) is
     the azimuth each band sweeps (360 = a full turn, which never alternates); `bands_start_yaw`
     (deg) is the azimuth the FIRST band begins at, measured around the subject (negative = from its
     left, positive = from its right).
    `orbit_look` is the node's "Auto Orbit Look" widget and one of `ORBIT_LOOK_MODES`:
    'Pivot point' (the default) aims every pose at the pivot, 'Horizontally straight' keeps
    the forward level - the camera aims at the vertical axis through the pivot at its own
    height and rides a CYLINDER of the orbit radius around that axis, so the straight
    distance to the axis is the same on every orbit. `None` keeps `ORBIT_LOOK_DEFAULT`.
    Everything else is documented on `_estimate_camera_path`, which does the work.
    """
    global _ACTIVE_FRONT_ORBIT_AMPLITUDE, _ACTIVE_SPIRAL_END, _ACTIVE_SPIRAL_END_ARC
    global _ACTIVE_SPIRAL_SLOPE, _ACTIVE_ROLL_UP, _ACTIVE_LOOK_MODE
    global _ACTIVE_BAND_START_ANGLE, _ACTIVE_BAND_COUNT, _ACTIVE_BAND_WIDTH
    global _ACTIVE_BAND_START_YAW, _ACTIVE_BAND_STOP_ANGLE
    previous = _ACTIVE_FRONT_ORBIT_AMPLITUDE
    previous_end = _ACTIVE_SPIRAL_END
    previous_arc = _ACTIVE_SPIRAL_END_ARC
    previous_slope = _ACTIVE_SPIRAL_SLOPE
    previous_up = _ACTIVE_ROLL_UP
    previous_look = _ACTIVE_LOOK_MODE
    previous_bands = (_ACTIVE_BAND_START_ANGLE, _ACTIVE_BAND_COUNT, _ACTIVE_BAND_WIDTH,
                      _ACTIVE_BAND_START_YAW, _ACTIVE_BAND_STOP_ANGLE)
    _ACTIVE_FRONT_ORBIT_AMPLITUDE = resolve_front_orbit_amplitude(orbit_amplitude)
    _ACTIVE_SPIRAL_END = resolve_spiral_end(spiral_end)
    _ACTIVE_SPIRAL_END_ARC = resolve_spiral_end_arc(spiral_end_arc)
    _ACTIVE_SPIRAL_SLOPE = resolve_spiral_slope(spiral_slope)
    _ACTIVE_LOOK_MODE = resolve_look_mode(orbit_look)
    _ACTIVE_BAND_START_ANGLE = resolve_band_start_angle(bands_start_angle)
    _ACTIVE_BAND_COUNT = resolve_band_count(bands_count)
    _ACTIVE_BAND_WIDTH = resolve_band_width(bands_width)
    _ACTIVE_BAND_START_YAW = resolve_band_start_yaw(bands_start_yaw)
    _ACTIVE_BAND_STOP_ANGLE = resolve_band_stop_angle(bands_stop_angle)
    # The clip is levelled to the spiral's own up when it flies a sloped coil, the world up
    # otherwise (see `_ACTIVE_ROLL_UP`); the slope is ignored by every other coverage, so the tilt
    # only ever reaches the frame where it belongs. The level look always keeps the WORLD up: a
    # "horizontally straight" clip must show a level horizon, whatever the coil's own axis does.
    _ACTIVE_ROLL_UP = (_spiral_tilt(tuple(fast_depth.WORLD_UP), _ACTIVE_SPIRAL_SLOPE)
                       if (str(coverage) == SPIRAL_COVERAGE
                           and _ACTIVE_LOOK_MODE != HORIZONTAL_LOOK)
                       else tuple(fast_depth.WORLD_UP))
    try:
        return _estimate_camera_path(
            reference, frames, target=target, max_speed=max_speed, subject_mask=subject_mask,
            model_size=model_size, depth_res=depth_res, device=device, depth_fn=depth_fn,
            pivot_offset=pivot_offset, orbit_distance=orbit_distance, orbit_size=orbit_size,
            subject_fill=subject_fill, orbit_end=orbit_end, direction=direction,
            view_angle=view_angle, coverage=coverage,
            face_box=face_box, face_fill=face_fill, face_span=face_span,
            depth_relief=depth_relief)
    finally:
        _ACTIVE_FRONT_ORBIT_AMPLITUDE = previous
        _ACTIVE_SPIRAL_END = previous_end
        _ACTIVE_SPIRAL_END_ARC = previous_arc
        _ACTIVE_SPIRAL_SLOPE = previous_slope
        _ACTIVE_ROLL_UP = previous_up
        _ACTIVE_LOOK_MODE = previous_look
        (_ACTIVE_BAND_START_ANGLE, _ACTIVE_BAND_COUNT, _ACTIVE_BAND_WIDTH,
         _ACTIVE_BAND_START_YAW, _ACTIVE_BAND_STOP_ANGLE) = previous_bands


def _estimate_camera_path(reference, frames, target=SUBJECT_TARGET, max_speed=DEFAULT_MAX_SPEED,
                          subject_mask=None, model_size="", depth_res=AUTO_DEPTH_RES,
                          device=None, depth_fn=None, pivot_offset=(0.0, 0.0, 0.0),
                          orbit_distance=None, orbit_size=None, subject_fill=None, orbit_end=None,
                          direction=ORBIT_DIRECTION_DEFAULT, view_angle=None, coverage=None,
                          face_box=None, face_fill=None, face_span=None, depth_relief=1.0):
    """(signal JSON, summary) for one still: geometric pivot, automatic path, collision guard.

    The pivot is the target's **cylindrical centre**: a robust vertical-cylinder fit of the depth
    profile (`fit_subject_cylinder` - 2.5 %/97.5 % trim plus the CYLINDER_COVERAGE radial percentile,
    so ~90-95 % of the surface sits inside and lances or depth spikes are ignored). For a framed
    subject the pivot is *also the aim* - the whole path then keeps the subject exactly in the
    horizontal picture centre and the camera-to-pivot distance is one constant for every frame. The
    Auto Pivot offsets (`pivot_offset`, in content radii) are applied to the finished path last -
    positions *and* aim translate - so no solve can cancel the user's nudge. The path is the subject
    composite (the front
    O-orbit plus the concluding orbit that runs `orbit_end` degrees around, back to the start point
    by default) or the scene's lateral survey (rows across the scene's width, side-lapped like drone
    mapping flight lines), fitted to the speed budget; the camera never comes closer to the scene
    than COLLISION_MARGIN x the content radius. The summary carries every number behind the decision,
    for the node's console line. `depth_fn` injects a depth map instead of running a depth model
    (tests).

    The coverage, view angle, orbit radius, and framing target ride on top of the estimate, all from the node's
    automatic-path widgets: `subject_fill` (percent of the picture area, subject target only) frames
    the subject by *solving* the distance, so the subject fills that share of the frame; a bigger
    fill means a closer camera. `orbit_size`
    multiplies the path's swing amplitudes (None/1.0 keeps the built-in swing; the subject fit
    searches *up* from there to the view limits, the scene fit can shrink it). `view_angle` (deg) is
    the azimuth the front O is centred on (0 = the frontal view, +90 = the subject's viewer-left
    side, 180 = its back, 270/-90 = its right) and `coverage` one of `ORBIT_COVERAGES` - "O-Orbit
    Front" or "O-Orbit Front&Back" (the front O, the level connection straight to the back orbit's near
    edge, then that orbit's full clockwise loop over the back's head, under it and up to 8 o'clock,
    followed by a glide to the level pose at the back's centre; the back gives way when the speed
    cap cannot pay for it). `direction` ('counter-clockwise'/'clockwise') mirrors the whole path.
    Without an explicit coverage mode, the standalone estimator uses its original concluding-orbit
    choreography. The "Banded Orbits" coverages ignore the front-O shape
    entirely (their stack IS the path, described by the `bands_*` widgets); the "Banded Orbits
    Front & Back" variant flies that stack on the front and then repeats it mirrored and reversed
    down the back side, wrapping between the two halves at the top band's elevation.
    """
    frames = validate_frames(frames)
    surface = probe_surface(reference, target=target, subject_mask=subject_mask,
                            model_size=model_size, depth_res=depth_res, device=device,
                            depth_fn=depth_fn, depth_relief=depth_relief)
    target = surface["target"]
    max_speed = _finite(max_speed, "Max camera speed")
    content_radius = max(1e-6, float(surface["content_radius"]))
    size = ORBIT_SIZE_DEFAULT if orbit_size is None else max(1e-3, _finite(orbit_size, "Orbit size"))
    framing = None
    fill = None
    scene_framing = None
    wants_framing = (target == SUBJECT_TARGET and subject_fill is not None
                     and _finite(subject_fill, "Subject fill") > SUBJECT_FILL_MIN)
    framed_pool = None
    if wants_framing:
        # Framing first: it solves the distance and hands back the *cylinder centre* as the pivot -
        # which is also the aim, so the subject stays in the horizontal picture centre and the camera
        # keeps one constant distance to it. The pixel cap for the speed fit follows from the
        # subject's apparent size, and the trimmed 95 % cloud it returns feeds the visibility pass.
        distance, framing_pivot, framing = subject_framing(surface, subject_fill, size,
                                                           orbit_end=orbit_end,
                                                           direction=direction,
                                                           view_angle=view_angle,
                                                           coverage=coverage)
        framed_pool = framing.get("pool")
        orbit_radius = max(distance, MIN_ORBIT_RADIUS)
        pivot = list(framing_pivot)
    elif target == SCENE_TARGET:
        # The survey solves its own stand-off: far enough that the outermost row reaches the scene's
        # edge, close enough that the frames are not mostly empty (see scene_survey_fill).
        fill, scene_framing = scene_survey_fill(surface, size)
        orbit_radius = max(fill * content_radius, MIN_ORBIT_RADIUS)
        pivot = [float(value) for value in surface["pivot"]]
    else:
        # No framing requested (Auto Subject Fill 0): the built-in stand-off. Auto Orbit Distance is
        # DEPRECATED and ignored - the distance follows Auto Subject Fill (more fill = closer).
        fill = SUBJECT_FILL if target == SUBJECT_TARGET else SCENE_FILL
        orbit_radius = max(fill * content_radius, MIN_ORBIT_RADIUS)
        pivot = [float(value) for value in surface["pivot"]]
    cap_px = max_speed * framing["radius_px"] if framing else None
    keys, info = automatic_keys(frames, pivot, orbit_radius, content_radius, target, max_speed,
                                size, surface, cap_px, orbit_end=orbit_end, direction=direction,
                                view_angle=view_angle, coverage=coverage)
    scene_subject = None
    if target == SCENE_TARGET and info.get("rows") and surface.get("subject_cloud") is not None:
        # How well the subject inside the scene is framed: a survey sweeps past it, so the numbers
        # are "how big does it appear" and "how many frames show it whole" (see subject_share_curve).
        positions = scene_samples(frames, pivot, orbit_radius, info, info["amplitude_scale"])
        curve = subject_share_curve(surface["subject_cloud"], positions, pivot, surface)
        shown = sorted(share for share, _whole, shows in curve if shows)
        scene_subject = {
            "source": surface["subject_source"],
            "share_median": shown[len(shown) // 2] if shown else 0.0,
            "share_max": max((share for share, _w, _s in curve), default=0.0),
            "frames": len(shown),
            "whole_frames": sum(1 for _s, whole, _shows in curve if whole),
            "total_frames": len(curve),
        }
    visibility = None
    if framing is not None and info.get("fit") == "subject pixels":
        # The fill solve frames the front pose and keeps the *area* centred; only this pass can
        # promise that the whole subject stays inside every frame. It works like a second budget on
        # top of the drift ladder: the swing may be *shrunk* (free - the front framing stays) and, if
        # even the smallest swing crops, the camera is pulled back (expensive - it costs fill). A
        # pull-back raises the pixel cap, which lets the ladder climb again, so the pass repeats and
        # every round only ever pulls the camera further back.
        start_radius = orbit_radius
        cropped_before = 0
        amplitude_before = info["amplitude_scale"]
        for attempt in range(VISIBILITY_PASSES):
            fitted = min(_amplitude_ceiling(size), info["amplitude_scale"])
            end_used = info.get("orbit_end", orbit_end)   # the speed cap may have cut the orbit
            distance, pivot, visibility = enforce_subject_visibility(
                surface, pivot, orbit_radius, fitted, size, frames, orbit_end=end_used,
                direction=direction, pool=framed_pool, recentre=False, view_angle=view_angle,
                coverage=coverage, back_span=info.get("back_span"))
            if attempt == 0:
                cropped_before = visibility["cropped"]
            cap_next = min(fitted, visibility["amplitude_cap"])
            moved = (visibility["scale"] > 1.0 + 1e-6
                     or max(abs(value) for value in visibility["shift"]) > 1e-9
                     or cap_next < info["amplitude_scale"] - 1e-9)
            orbit_radius = max(distance, MIN_ORBIT_RADIUS)
            if not moved:
                break
            if visibility["radius_px"]:
                cap_px = max_speed * visibility["radius_px"]
            keys, info = automatic_keys(frames, pivot, orbit_radius, content_radius, target,
                                        max_speed, size, surface, cap_px,
                                        amplitude_cap=cap_next, orbit_end=end_used,
                                        direction=direction, view_angle=view_angle,
                                        coverage=coverage)
        # report the *total* pull-back of the whole pass and the frames that were cropped before it
        visibility["scale"] = orbit_radius / max(1e-9, start_radius)
        visibility["cropped"] = cropped_before
        visibility["amplitude_before"] = float(amplitude_before)
        framing.update({
            "pivot_shift": [framing["pivot_shift"][axis] + visibility["shift"][axis]
                            for axis in range(3)],
        })
        framing.update({key: value for key, value in (
            ("area", visibility["area"]), ("fill_min", visibility["fill_min"]),
            ("fill_max", visibility["fill_max"]), ("width_px", visibility["width_px"]),
            ("height_px", visibility["height_px"]), ("radius_px", visibility["radius_px"]))
            if value is not None})
    # The cylinder framing already keeps the orbit centre ON the subject - and therefore on the aim -
    # so the pivot equaliser is not used here: moving the orbit centre away from the aim is exactly
    # what the exact horizontal centring forbids. `equalize_pivot` stays available as a standalone
    # utility for callers that aim off-centre on purpose.
    equalizer = None
    offset_values = [_finite(pivot_offset[axis], "Pivot offset") for axis in range(3)]
    # The equaliser's move and the user's Auto Pivot offset are applied to the emitted path *last*,
    # so nothing that runs before (framing, visibility, amplitude fit) can eat them - that is what
    # makes the three widgets dependable. They differ in what they move:
    #   * the equaliser moves the ORBIT CENTRE only (the camera positions): the subject's apparent
    #     size then holds around the path while the aim - and with it the front composition - stays
    #     exactly where the framing solved it;
    #   * the Auto Pivot offset translates the whole rig (positions *and* aim): the subject shifts
    #     in the picture, which is the composition nudge the widgets promise.
    orbit_shift = list(equalizer["move"]) if equalizer else [0.0, 0.0, 0.0]
    look_shift = [offset_values[axis] * content_radius for axis in range(3)]
    pos_shift = [orbit_shift[axis] + look_shift[axis] for axis in range(3)]
    orbit_centre = [pivot[axis] + pos_shift[axis] for axis in range(3)]
    if any(abs(value) > 1e-9 for value in pos_shift):
        keys = [{"pos": [round(key["pos"][axis] + pos_shift[axis], 6) for axis in range(3)],
                 "look": [round(key["look"][axis] + look_shift[axis], 6) for axis in range(3)],
                 "src": key["src"], "t": key["t"]} for key in keys]
        pivot = [pivot[axis] + look_shift[axis] for axis in range(3)]

    # --- face-detail orbit -------------------------------------------------------
    # The face-height orbit is inserted between body-path sections (see `apply_face_orbit`). It runs
    # here - on the finished keys, in absolute depth
    # units, before the collision guard and before the median-depth conversion - so
    # it sees exactly the path the render will fly and every later pass still gets
    # its say. Without a face box this is a no-op, and the summary says so.
    face_info = None
    face_frames = []
    if face_box is not None:
        box = face_box
        if isinstance(box, str):
            try:
                from enndee_face_detect import parse_face_bbox
                box = parse_face_bbox(box)
            except Exception:                                       # pragma: no cover
                box = None
        if isinstance(box, dict):
            orbit_radius = _yaw_elevation_of(keys[0]["pos"], orbit_centre)[2] if keys else 0.0
            if orbit_radius > 1e-9:
                reference_pose = front_camera(orbit_centre, orbit_radius)
                keys, face_info = apply_face_orbit(
                    keys, surface, box,
                    fill=(FACE_ORBIT_FILL_DEFAULT if face_fill is None else float(face_fill)),
                    span=(FACE_ORBIT_SPAN_DEFAULT if face_span is None else float(face_span)),
                    pivot=orbit_centre,
                    front_position=reference_pose,
                    front_look=_aim(orbit_centre, reference_pose),
                    frames_total=frames,
                    # The CANVAS aspect, not the source's: the SAM3 box is normalised in
                    # the source but the cloud's u/v live in the canvas (736x544 vs
                    # 3440x2312 on the Hanks still), and using the source aspect shifts
                    # the selection ~10 % sideways - the observed "aims slightly left".
                    render_aspect=surface.get("render_aspect"),
                    # Same focal the cloud was built with. Without this the orbit projects
                    # with the hard-coded 55 deg instead of the measured ~38 deg (a 1.509x
                    # error in f), which mis-sizes the face patch and the solved distance.
                    vfov_degrees=surface.get("vfov"),
                    coverage=coverage,
                )
                face_frames = list((face_info or {}).get("frames") or [])

    keys, fixed, worst_before, worst_after = guard_collisions(keys, surface)
    if _ACTIVE_LOOK_MODE == HORIZONTAL_LOOK:
        # A level look is a function of the camera height, so it is re-derived once the pivot
        # offset and the collision guard have finished moving the poses: the forward stays
        # exactly level and the aim keeps pointing at the axis at the camera's own height.
        keys = [dict(key, look=[round(value, 6) for value in _aim(pivot, key["pos"])])
                for key in keys]
    # Everything above works in absolute depth units; the *emitted* document is in median-depth
    # units, exactly like the Camera Path Configurator's paths - the fast-depth backend multiplies
    # the keys by the cloud's median depth (`zm`), so absolute keys would be scaled a second time.
    depth_unit = max(1e-6, float(surface["median_depth"]))
    emitted = []
    for key in keys:
        row = {
            "pos": [round(value / depth_unit, 6) for value in key["pos"]],
            "look": [round(value / depth_unit, 6) for value in key["look"]],
            "src": key["src"], "t": key["t"],
        }
        # Carry the face close-up marker through. It used to live in `src` as the
        # sentinel "face", which broke every path parser - but it did survive
        # serialization. Moving it to its own field must not lose it.
        if key.get("face"):
            row["face"] = True
        emitted.append(row)
    if framing:
        cylinder = framing.get("cylinder") or {}
        # What the framing guarantees depends on the look: the pivot look keeps the aim on the
        # subject (one constant distance to it), the level look keeps the distance to the AXIS.
        axis_text = ("is both the pivot and the aim - so the subject stays in the horizontal "
                     "picture centre and the camera keeps one constant distance all along the "
                     "path - " if _ACTIVE_LOOK_MODE != HORIZONTAL_LOOK else
                     "is the pivot, and the level aim rides that axis, so the subject stays in "
                     "the horizontal picture centre while the camera keeps one constant distance "
                     "to the axis - ")
        framing_text = (
            f"the subject is approximated by a vertical cylinder (centre "
            f"[{pivot[0]:+.3g}, {pivot[1]:+.3g}, {pivot[2]:+.3g}], radius "
            f"{cylinder.get('radius', 0.0):.3g} units, "
            f"{cylinder.get('coverage', 0.0) * 100:.0f} % of the surface inside) whose axis "
            + axis_text
            + f"framed to "
            f"{framing['area'] * 100:.0f} % of the picture area at the front pose, "
            f"{framing['fill_min'] * 100:.0f}-{framing['fill_max'] * 100:.0f} % along the path "
            f"({framing['width_px']:.0f} x {framing['height_px']:.0f} px on a "
            f"{CANVAS_HEIGHT * float(surface['aspect']):.0f} x {CANVAS_HEIGHT:.0f} canvas)"
        )
        distance_text = f"distance {orbit_radius:.3g} units solved by Auto Subject Fill"
    visibility_text = ""
    if visibility:
        inside = ("the whole subject box" if _ACTIVE_LOOK_MODE != HORIZONTAL_LOOK
                  else "the subject box (its left/right extent)")
        visibility_text = (
            f"{inside} ({visibility['box_extents'][0]:.3g} x "
            f"{visibility['box_extents'][1]:.3g} x {visibility['box_extents'][2]:.3g} units) stays "
            f"inside the frame (worst border clearance {visibility['clearance_px']:.0f} of "
            f"{CANVAS_HEIGHT:.0f} px at frame {visibility['worst_frame']} = "
            f"{visibility['worst_pct'] * 100:.0f} % of the path"
            + (f", pulled back {visibility['scale']:.2f}x for it" if visibility["scale"] > 1.005
               else "")
            + ")"
        )
    else:
        framing_text = ""
        distance_text = f"distance {fill:g} x content radius (built-in stand-off)"
    if info.get("fit") == "subject pixels":
        speed_text = (f"subject drift {info['drift_px']:.3g} px/frame, typical "
                      f"{info['drift_typical_px']:.3g} px, cap {info['drift_cap_px']:.3g} px "
                      f"({max_speed * 100:.0f} % of the subject's apparent radius "
                      f"{framing['radius_px']:.0f} px) - the background is not part of it")
    else:
        speed_text = (f"{info['travel_per_frame']:.3g} units per frame, budget "
                      f"{max_speed * 100:.0f} % of the content radius per frame")
    closed_round = False
    if info.get("orbit_end") is not None:
        wrapped = info["orbit_end"] % FULL_CIRCLE
        closed_round = bool(info.get("orbit_coverage")) and min(wrapped, FULL_CIRCLE - wrapped) < 0.5
    # What the path shows after the front O - built first, so the description stays readable. The new
    # modes reach the back through the level connection and the far orbit's clockwise loop ("Front and
    # Back"), or give the back up to the speed cap; the DEPRECATED `orbit_end` path still runs the
    # older concluding orbit.
    beyond = ""
    if info.get("orbit_coverage"):
        if info.get("coverage") == ORBIT_COVERAGES[1]:
            beyond = (f", then the {info['lap_span']:.0f} deg level connection to the back orbit's "
                      f"near edge, and that orbit's loop (9 -> 12 -> 3 -> 6 -> 8 o'clock: over the "
                      f"back head, out to the far side, under the back and back up) plus its glide "
                      f"in to the back's centre reaches "
                      f"{info['orbit_end'] + info['front_yaw']:.0f} deg"
                      + (" - cut there by the speed cap"
                         if info['orbit_end'] < info['orbit_end_requested'] - 0.5 else "")
              + f", covering {info['orbit_coverage']:.0f} deg around the subject")
        elif info.get("coverage") == SPIRAL_COVERAGE:
            winding = float(info.get("spiral_end") or SPIRAL_END_DEFAULT)
            arc = float(info.get("spiral_end_arc") or SPIRAL_END_ARC_DEFAULT)
            slope = float(info.get("spiral_slope") or 0.0)
            axis_text = ("the view axis" if abs(slope) < 0.5
                         else (f"the spiral's central axis, leaned {slope:.0f} deg out of the view "
                               f"axis ({'from straight above' if slope > 0 else 'from straight below'}"
                               f" at 90 deg)"))
            beyond = (f", then the O-orbit family around {axis_text}: the O-orbit's angle grows "
                      f"evenly from 0 deg at the first frame to {arc:.0f} deg at the last one "
                      f"(a spiral of those circles, ending "
                      + ("IN the picture's own plane - a side view of the picture)"
                         if arc >= SPIRAL_END_ARC_MAX - 0.5
                         else f"{SPIRAL_END_ARC_MAX - arc:.0f} deg short of the picture's plane)")
                      + f", while the clock angle winds {winding:.0f} deg "
                      f"({winding / 360.0:.2f} rounds) {direction_label(info['orbit_direction'])} "
                      f"- clock {info['spiral_end_clock']:.0f} deg, elevation "
                      f"{info['spiral_end_elevation']:.0f} deg. The frames follow that parameter "
                      f"(the angle growing evenly along the path) and the keys the renderer splines "
                      f"follow the path's own turning, so the path reads smooth in every view")
        elif info.get("coverage") == BAND_COVERAGE:
            elevations = "/".join(f"{value:.0f}" for value in info["band_elevations"])
            beyond = (f", then the band stack: {info['band_count']} circle segments of "
                      f"{info['band_width']:.0f} deg each at {elevations} deg elevation "
                      f"between {info['band_start_angle']:.0f} and {info['band_stop_angle']:.0f} deg. "
                      f"The camera sweeps the first band "
                      f"from {info['band_start_yaw']:.0f} deg towards the left, steps STRAIGHT to "
                      f"the next one at the azimuth it stopped on, sweeps that one back, and so on "
                      f"- ending at {info['band_end_yaw']:.0f} deg / "
                      f"{info['band_end_elevation']:.0f} deg elevation and covering "
                      f"{info['orbit_coverage']:.0f} deg around the subject")
        elif info.get("coverage") == BAND_FRONT_BACK_COVERAGE:
            elevations = "/".join(f"{value:.0f}" for value in info["band_elevations"])
            beyond = (f", then the band stack on BOTH sides: {info['band_count']} circle segments "
                      f"of {info['band_width']:.0f} deg each at {elevations} deg elevation "
                      f"between {info['band_start_angle']:.0f} and {info['band_stop_angle']:.0f} deg. "
                      f"The camera flies the FRONT stack "
                      f"from {info['band_start_yaw']:.0f} deg, stepping STRAIGHT between bands, then "
                      f"EXTENDS the top band's sweep - same heading, no turn back - across to "
                      f"the far side, then walks the SAME bands back DOWN there, each swept the "
                      f"way its front twin was - ending at "
                      f"{info['band_end_yaw']:.0f} deg / {info['band_end_elevation']:.0f} deg "
                      f"elevation and covering {info['orbit_coverage']:.0f} deg around the subject")
        elif info.get("coverage"):
            beyond = (f", the back visit giving way to the speed cap: the path ends at "
                      f"{info['orbit_end']:.0f} deg of the requested "
                      f"{info['orbit_end_requested']:.0f} deg, covering "
                      f"{info['orbit_coverage']:.0f} deg around the subject")
        else:                              # deprecated `orbit_end`: the older concluding orbit
            beyond = (f" and the {info['lap_span']:.0f} deg concluding orbit run "
                      f"{direction_label(info['orbit_direction'])} and end at "
                      f"{info['orbit_end']:.0f} deg"
                      + ("" if info["orbit_end"] >= info["orbit_end_requested"] - 0.5
                         else f" of the requested {info['orbit_end_requested']:.0f} deg (speed cap)")
                      + (", back at the start point" if closed_round else "")
                      + f", covering {info['orbit_coverage']:.0f} deg around the subject")
    # The spiral *is* the opening move now (it starts on the same framed pose and never closes a
    # circle), so the description names it instead of the front O. Only the subject path has a
    # `front_yaw` (the scene survey reports rows instead), hence the guard.
    opening = ""
    if info.get("orbit_coverage"):
        if info.get("coverage") == SPIRAL_COVERAGE:
            opening = (f"the O-orbit family / spherical spiral (0 -> {info['spiral_end_arc']:.0f} "
                       f"deg out of the spiral's axis)")
        elif info.get("coverage") in (BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):
            opening = (f"a stack of {info['band_count']} banded orbits of "
                       f"{info['band_width']:.0f} deg each, starting at "
                       f"evenly distributed between {info['band_start_angle']:.0f} and "
                       f"{info['band_stop_angle']:.0f} deg elevation"
                       + (", flown on the front and then walked back down the far side"
                          if info.get("coverage") == BAND_FRONT_BACK_COVERAGE else ""))
        else:
            opening = f"the front O (+/-{info['front_yaw']:.0f} deg)"
    description = (
        f"Estimated from the still's surface ({surface['source']}): pivot "
        f"[{pivot[0]:.3g}, {pivot[1]:.3g}, {pivot[2]:.3g}] is the cylindrical centre of the "
        f"depth profile ({surface['extents'][0]:.3g} x {surface['extents'][1]:.3g} x "
        f"{surface['extents'][2]:.3g} units, built from {surface['content_points']} of "
        f"{surface['points']} points - only those), {info['style']} at {distance_text}"
        + (f", {framing_text}" if framing_text else "")
        + (f", {visibility_text}" if visibility_text else "")
        + f", amplitudes at {info['amplitude_scale']:.2f} of the built-in swing ({speed_text}), "
        + f"orbit size {size:g}x"
        + (", looking horizontally straight at the pivot's vertical axis (a level horizon, and "
           "the straight distance to the axis is the orbit radius on every orbit)"
           if _ACTIVE_LOOK_MODE == HORIZONTAL_LOOK else "")
        + (", the orbit centre equalised for a steady subject size"
           f" ({equalizer['spread_before'] * 100:.0f}"
           f" % -> {equalizer['spread_after'] * 100:.0f} % path distance spread"
           + (f", scaled to {equalizer['factor']:.2f} for the visibility guarantee"
              if equalizer['factor'] < 0.999 else "") + "; the aim stays on the framing)"
           if equalizer and equalizer["applied"] else "")
        + (f", the Auto Pivot offset ({offset_values[0]:g}, {offset_values[1]:g}, "
           f"{offset_values[2]:g}) content radii shifted the final aim"
           if any(abs(value) > 1e-9 for value in offset_values) else "")
        + (f", {opening}" + beyond
           if info.get("orbit_coverage") else "")
        + (f"; survey {info['rows']} rows across +/-{info['half_yaw']:.0f} deg "
           f"({info['lane_overlap'] * 100:.0f} % side overlap, {info['lane_step']:.3g} units "
           f"between rows)"
           + (f", stand-off solved at {scene_framing['fill']:.2f} x content radius for "
              f"{scene_framing['reach'] * 100:.0f} % of the scene's half width" if scene_framing
              else "")
           + (f", the subject inside it ({scene_subject['source']}) covers "
              f"{scene_subject['share_median'] * 100:.1f} % of the picture in "
              f"{scene_subject['frames']} of {scene_subject['total_frames']} frames "
              f"({scene_subject['whole_frames']} whole)"
              if scene_subject else "")
           if info.get("rows") else "")
        + (f"; {fixed} key(s) pushed clear of the scene geometry" if fixed else "")
        + f". Keys are in median-depth units ({depth_unit:.3g} units = 1.0), like the manual path. "
        + "Non-front views are synthetic depth reprojections, not observed geometry."
    )
    # A sloped spiral carries its own up so the renderer levels the clip to the same frame the
    # estimate measured in (`fast_depth.path_up`); a level path leaves the field out and keeps the
    # world up it always had.
    up = roll_up()
    levelled = ([round(float(value), 6) for value in up]
                if any(abs(float(value) - float(world)) > 1e-6
                       for value, world in zip(up, fast_depth.WORLD_UP)) else None)
    document = document_from_keys(frames, emitted, f"Auto {info['style']} ({frames} frames)",
                                  description,
                                  extra={"depth_model": surface.get("depth_model"),
                                         "up": levelled,
                                         # The face orbit's frames, so the node can hand
                                         # them to COLMAP without threading a second return
                                         # value through every private builder.
                                          "face_frames": face_frames,
                                         "face_orbit": (face_info or {}).get("status"),
                                         # What the orbit SOLVED for. `face_want_distance`
                                         # is the stand-off it asked for; comparing it with
                                         # the emitted key positions separates a unit
                                         # mismatch from the moved keys being dropped.
                                         "face_want_distance": (face_info or {}).get("want_distance"),
                                         "face_floor_distance": (face_info or {}).get("floor_distance"),
                                         "face_radius": (face_info or {}).get("face_radius"),
                                         "face_fill_target": (face_info or {}).get("fill_target"),
                                         "face_fill_vfov": (face_info or {}).get("fill_vfov"),
                                         "face_window": (face_info or {}).get("window")})
    hint = ""
    # The lever that frees frames for the front O. With the new modes it is 'O-Orbit Front' (which
    # drops the whole back visit); with the DEPRECATED `orbit_end` path it is a shorter end.
    lever = (f"set Auto Orbit Coverage to '{ORBIT_COVERAGES[0]}'" if info.get("coverage")
             else "a shorter Auto Orbit End")
    if info.get("fit") == "subject pixels":
        ceiling = _amplitude_ceiling(size)
        # WHO stopped the front O - the drift ladder (`amplitude_before`, the speed budget's own
        # answer) or the visibility pass (`amplitude_cap`, "this swing would crop the subject")?
        # Only the former may be blamed on Auto Max Speed: crediting the visibility cap to the drift
        # told the user to raise a speed parameter for a problem no speed can fix, while the real
        # cause (a shrink that bought nothing) stayed invisible.
        drift_scale = float((visibility or {}).get("amplitude_before", info["amplitude_scale"]))
        # The spiral has no "front O" to grow: its winding is the user's Spiral End parameter, so the
        # swing ladder's view-limit message would blame Auto Max Speed for a swing it does not scale.
        if (info.get("coverage") != SPIRAL_COVERAGE
                and info["amplitude_scale"] < ceiling * 0.99 and drift_scale < ceiling * 0.99):
            hint = (f"the subject-drift budget stops the front loop at "
                    f"{info['amplitude_scale']:.2f}x of its {ceiling:.2f}x view limit "
                    f"({info['drift_px']:.3g} px/frame vs {info['drift_cap_px']:.3g} px cap) - raise "
                    f"Auto Max Speed or Output Frames for a bigger O, or {lever} "
                    f"(the frames it frees go to the front loop)")
        floor = info.get("front_floor")
        if floor is not None and info["amplitude_scale"] < floor - 1e-9:
            named = (f"the built-in {floor:.2f}x circle" if info.get("coverage")
                     else f"Auto Orbit Size {floor:.2f}x")
            shrink = (f"the cap could not even pay for {named}, so the front O "
                      f"had to shrink below it to {info['amplitude_scale']:.2f}x - more Output "
                      f"Frames or a higher Auto Max Speed give it back")
            hint = f"{hint}; {shrink}" if hint else shrink
        # The fill request can be *capped by the room the orbit needs*: at a very close distance a
        # pose of the swing (the elevation extremes in practice) would crop the subject, so the
        # camera stays back and the fill stays below the request. Say it, or the user turns the fill
        # up and nothing happens.
        if framing and framing.get("room_distance") and framing.get("distance_fill"):
            room = float(framing["room_distance"])
            wanted = float(framing["distance_fill"])
            if room > wanted * 1.02:
                room_note_scope = ("the whole subject inside every frame, including the O and the "
                                   "back part" if _ACTIVE_LOOK_MODE != HORIZONTAL_LOOK else
                                   "the subject inside the picture's left/right borders on every "
                                   "pose (a level look cannot move it vertically, so the "
                                   "top/bottom drift is the look's own)")
                room_note = (f"Auto Subject Fill would frame {float(subject_fill or 0):.0f} % but "
                             f"wants {wanted:.3g} units - the orbit's room ({room_note_scope}) "
                             f"keeps the camera at "
                             f"{room:.3g} units ({room / max(1e-9, wanted):.2f}x back). Use a lower "
                             f"Auto Subject Fill, {lever}, or mask the subject tighter, to come closer")
                hint = f"{hint}; {room_note}" if hint else room_note
    if visibility and not visibility["ok"]:
        hint = (f"the whole subject does not fit the picture even at "
                f"{VISIBILITY_MAX_SCALE:.0f}x the fill distance - narrow the subject (a MASK from "
                f"e.g. RMBG) or frame it with more room around it")
    else:
        parts = []
        if (info.get("orbit_end") is not None
                and info["orbit_end"] < info["orbit_end_requested"] - 0.5):
            parts.append(f"the max camera speed ends the path at {info['orbit_end']:.0f} deg of the "
                         f"requested {info['orbit_end_requested']:.0f} deg - more Output Frames, a "
                         f"higher Auto Max Speed or {lever} balance the round")
        if info.get("drift_px") and info["drift_px"] > info["drift_cap_px"] + 1e-9:
            parts.append(f"the orbit already runs at the smallest drift these frames allow "
                         f"({info['drift_px']:.3g} px/frame vs the {info['drift_cap_px']:.3g} px cap) "
                         f"- more Output Frames or a higher Auto Max Speed, {lever} "
                         f"or a tighter MASK (it shrinks what moves) take the rest")
        elif (info.get("fit") == "world travel" and info.get("orbit_coverage")
              and info["travel_per_frame"] > info["budget_per_frame"] + 1e-12):
            parts.append(f"the orbit costs {info['travel_per_frame']:.3g} units/frame at "
                         f"{info['amplitude_scale']:.2f}x of the built-in swing (budget "
                         f"{info['budget_per_frame']:.3g}) - raise Auto Max Speed or Output Frames, "
                         f"or set Auto Subject Fill so the budget is measured in subject pixels")
        if visibility and visibility["amplitude_before"] > visibility["amplitude_cap"] + 1e-6:
            parts.append(f"the front O was shortened to {visibility['amplitude_cap']:.2f}x (from "
                         f"{visibility['amplitude_before']:.2f}x) so the whole subject stays in the "
                         f"picture - the concluding orbit keeps its own end, so the round still goes "
                         f"where it should; Auto Max Speed, Output Frames or a tighter MASK buy back "
                         f"the viewing angles")
        if visibility and visibility["scale"] > 1.005:
            parts.append(f"the camera pulled back {visibility['scale']:.2f}x from the fill distance "
                         f"so neither the O nor the concluding orbit leaves the picture "
                         f"({visibility['cropped']} of {visibility['frames']} frames were cropped "
                         f"before)")
        if parts:
            extra = "; ".join(parts)
            hint = f"{hint}; {extra}" if hint else extra
    if info.get("rows"):
        parts = []
        reach = orbit_radius * math.sin(math.radians(info["half_yaw"] * info["amplitude_scale"]))
        if reach < info["half_width"] - 1e-9:
            shortfall = 1.0 - reach / max(1e-9, info["half_width"])
            if info["amplitude_scale"] < 0.99:
                # the stand-off is right, the *speed fit* shortened the sweep: more frames per row
                # (or a bigger budget) buy the width back, the distance cannot
                parts.append(f"the sweep was cut to {info['amplitude_scale']:.2f}x by the speed "
                             f"budget, so the rows reach {reach:.3g} of the scene's "
                             f"{info['half_width']:.3g} half width ({shortfall * 100:.0f} % short) - "
                             f"raise Auto Max Speed or Output Frames, a bigger Auto Orbit Size widens "
                             f"the frame (trading side overlap)")
            else:
                parts.append(f"the rows reach {reach:.3g} units of the scene's "
                             f"{info['half_width']:.3g} half width - raise Auto Orbit Size (a wider "
                             f"row sweep) or Auto Max Speed (a bigger budget) to cover the rest")
        if info.get("lane_overlap", 1.0) < SCENE_LANE_OVERLAP_WARN:
            # coverage needs overlap: rows further apart than the frame's footprint leave strips of
            # the scene out of every row - more frames buy rows, Auto Orbit Size buys breadth per row
            parts.append(f"the rows sit {info['lane_step']:.3g} units apart "
                         f"({info['lane_overlap'] * 100:.0f} % side overlap, the drone rule wants "
                         f"70 %) - more Output Frames or Auto Max Speed add rows, Auto Orbit Size "
                         f"widens each row's footprint")
        if scene_subject:
            if scene_subject["frames"] == 0:
                parts.append(f"the survey never shows the subject ({scene_subject['source']}) - "
                             f"raise Auto Orbit Size so the rows sweep past it")
            elif scene_subject["share_median"] < SCENE_SUBJECT_SHARE_MIN:
                parts.append(f"the subject ({scene_subject['source']}) covers only "
                             f"{scene_subject['share_median'] * 100:.1f} % of the picture in the "
                             f"survey - a smaller Auto Orbit Size narrows the rows onto the subject")
            elif scene_subject["share_median"] > 1.0:
                parts.append(f"the subject ({scene_subject['source']}) is bigger than the frame at "
                             f"{scene_subject['share_median'] * 100:.0f} % of the picture - a bigger "
                             f"Auto Orbit Size widens the survey, or set Auto Target to 'subject' "
                             f"and orbit the subject itself")
        extra = "; ".join(parts)
        if extra:
            hint = f"{hint}; {extra}" if hint else extra
    # Last, appended (never overwritten): where the pivot's points came from. The orbit can look
    # perfectly fine while it circles the BACKGROUND's centre, so this is never left implicit.
    for note in pivot_provenance_notes(target, surface):
        hint = f"{hint}; {note}" if hint else note
    summary = {
        "target": target, "frames": frames, "source": surface["source"],
        "points": surface["points"], "content_points": surface["content_points"],
        "pivot": pivot, "extents": surface["extents"],
        "median_depth": float(depth_unit),
        "pivot_offset": [float(value) for value in pivot_offset],
        "content_radius": content_radius, "orbit_radius": float(orbit_radius),
        "orbit_fill": float(fill) if fill is not None else None, "orbit_size": float(size),
        "fit": info["fit"], "amplitude_scale": info["amplitude_scale"],
        "travel_per_frame": info["travel_per_frame"],
        "budget_per_frame": info["budget_per_frame"], "keys": info["keys"],
        "max_speed": max_speed, "scene_radius": float(surface["scene_radius"]),
        "scene_pivot": surface["scene_pivot"], "collision_fixes": fixed,
        "face_orbit": face_info,
        "collision_margin": COLLISION_MARGIN, "clearance_before": worst_before,
        "clearance_after": worst_after,
        "pivot_equalizer": equalizer, "pivot_shift_final": look_shift,
        "orbit_centre": orbit_centre, "orbit_shift": orbit_shift,
        "hfov": float(surface["hfov"]), "hint": hint,
        "style": info["style"],
        "front_yaw_deg": info.get("front_yaw"), "lap_span_deg": info.get("lap_span"),
        "orbit_coverage_deg": info.get("orbit_coverage"),
        "orbit_end_deg": info.get("orbit_end"),
        "orbit_end_requested_deg": info.get("orbit_end_requested"),
        "orbit_direction": info.get("orbit_direction"), "front_share": info.get("front_share"),
        # Auto Orbit View Angle / Coverage: 0 deg = the front O is centred on the frontal view,
        # +90 = the subject's viewer-left side, 180 = its back, 270/-90 = its right; `coverage` is
        # None while the deprecated `orbit_end` path is in charge (saved callers).
        "view_angle_deg": info.get("view_angle"), "coverage": info.get("coverage"),
        "back_orbit": info.get("back_orbit"),
        # Auto Orbit Look: 'Pivot point' (the default) aims every pose at the pivot;
        # 'Horizontally straight' keeps the forward level and rides a cylinder around the
        # pivot's vertical axis. Every coverage honours it.
        "orbit_look": look_mode(),
        # Spiral coverage only: the winding the path flies (the node's Spiral End widget), the lean
        # of its central axis (the same widget slot in Spiral mode: the Spiral Center Slope) and the
        # pose the last frame looks from - the arc out of the axis plus the clock/elevation it lands
        # on in the picture's own plane.
        "spiral_end_deg": info.get("spiral_end"),
        "spiral_end_arc_deg": info.get("spiral_end_arc"),
        "spiral_slope_deg": info.get("spiral_slope"),
        "spiral_end_clock_deg": info.get("spiral_end_clock"),
        "spiral_end_elevation_deg": info.get("spiral_end_elevation"),
        # Banded Orbits coverage only: the stack the path is made of (the first band's elevation,
        # how many bands, the azimuth each sweeps, where the first one begins and the elevation
        # step between neighbours) plus the pose the last frame looks from. The "Front & Back"
        # variant adds `band_front_back` plus the front half's own end and the azimuth the
        # connection wraps to, so both halves can be told apart in the summary.
        "band_front_back": info.get("band_front_back"),
        "band_start_angle_deg": info.get("band_start_angle"),
        "band_stop_angle_deg": info.get("band_stop_angle"),
        "band_count": info.get("band_count"),
        "band_width_deg": info.get("band_width"),
        "band_start_yaw_deg": info.get("band_start_yaw"),
        "band_elevations_deg": info.get("band_elevations"),
        "band_front_end_yaw_deg": info.get("band_front_end_yaw"),
        "band_front_end_elevation_deg": info.get("band_front_end_elevation"),
        "band_back_start_yaw_deg": info.get("band_back_start_yaw"),
        "band_end_yaw_deg": info.get("band_end_yaw"),
        "band_end_elevation_deg": info.get("band_end_elevation"),
    }
    if info.get("fit") == "subject pixels":
        summary.update({"drift_px": info["drift_px"], "drift_typical_px": info["drift_typical_px"],
                        "drift_cap_px": info["drift_cap_px"]})
    if visibility:
        summary.update({"visibility_ok": visibility["ok"],
                        "visibility_scale": visibility["scale"],
                        "visibility_clearance_px": visibility["clearance_px"],
                        "visibility_margin_px": visibility["margin_px"],
                        "visibility_cropped": visibility["cropped"],
                        "visibility_frames": visibility["frames"],
                        "visibility_quarters_px": visibility["quarters_px"],
                        "visibility_worst_frame": visibility["worst_frame"],
                        "visibility_worst_pct": visibility["worst_pct"],
                        "visibility_amplitude_cap": visibility["amplitude_cap"],
                        "visibility_amplitude_before": visibility["amplitude_before"],
                        "visibility_shift": visibility["shift"],
                        "subject_box_units": visibility["box_extents"]})
    if framing:
        summary.update({"subject_fill_area": framing["area"],
                        "subject_fill_min": framing["fill_min"],
                        "subject_fill_max": framing["fill_max"],
                        "subject_width_px": framing["width_px"],
                        "subject_height_px": framing["height_px"],
                        "subject_radius_px": framing["radius_px"],
                        "distance_from_fill": framing.get("distance_fill"),
                        "room_distance": framing.get("room_distance"),
                        "subject_depth": framing["depth"],
                        "pivot_shift": framing["pivot_shift"],
                        "framing_points": framing["points"],
                        "cylinder_radius": (framing.get("cylinder") or {}).get("radius"),
                        "cylinder_coverage": (framing.get("cylinder") or {}).get("coverage"),
                        "canvas_height": CANVAS_HEIGHT})
    if info.get("rows"):
        summary.update({"rows": info["rows"], "scene_half_yaw": info["half_yaw"],
                        "lane_step": info["lane_step"], "lane_overlap": info["lane_overlap"],
                        "scene_half_width": info["half_width"],
                        "scene_low_elevation": info["low_elevation"],
                        "scene_high_elevation": info["high_elevation"]})
    if scene_framing:
        summary.update({"scene_fill_solved": scene_framing["fill"],
                        "scene_reach": scene_framing["reach"],
                        "scene_footprint": scene_framing["footprint"],
                        "scene_yaw_limit": scene_framing["yaw_limit"],
                        "scene_reach_fill": scene_framing["reach_fill"],
                        "scene_fill_cap": scene_framing["fill_cap"]})
    if scene_subject:
        summary.update({"subject_source": scene_subject["source"],
                        "scene_subject_share": scene_subject["share_median"],
                        "scene_subject_share_max": scene_subject["share_max"],
                        "scene_subject_frames": scene_subject["frames"],
                        "scene_subject_whole_frames": scene_subject["whole_frames"]})
    return document, summary


def depth_from_reference(reference, model_size="", depth_res=AUTO_DEPTH_RES, device=None):
    """(depth, model_name) for the estimator: the requested pick, else DA3-Small, else DAv2.

    Reuses the fast-depth backend's loaders (its VRAM cache included), so the automatic camera
    costs one depth pass per run. Every fallback goes through the same "larger = farther"
    conversion, and the caller's `model_size` is tried first so the estimate matches what the
    Geometry node will render with. The name that actually ran comes back with the depth: the
    keys are emitted in *this* depth map's median units, so the renderer has to be told which
    map that was (see `render_depth_aligned`'s mismatch warning).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first = reference[0:1].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    # MoGe-3 is the ONLY supported depth source (see `MODEL_OPTIONS`) because it is the
    # only one that reports the camera intrinsics the face orbit needs. Try it FIRST
    # regardless of the name that arrives here: the label is resolved to a backend id
    # before it reaches the probe (`'MoGe 3'` -> `'MoGe-3-ViT-L'`) and the observed value
    # at this point was `'Depth-Anything-3-Mono-Large'` even though the widget said MoGe.
    # A name mismatch must NEVER silently drop the measured FOV - that is exactly how the
    # whole face orbit came to project at the hard-coded 55 deg.
    attempts = ["MoGe-3-ViT-L", "MoGe 3"]
    if str(model_size).strip():
        attempts.append(str(model_size).strip())
    attempts += ["Depth-Anything-3-Small", "Depth-Anything-V2-Small-hf"]
    last_error = None
    for candidate in attempts:
        try:
            if candidate.startswith(fast_depth.MOGE_PREFIX):
                # MoGe-3's `refine_steps` is deliberately left at the backend default (3), the
                # same value the Meridian Geometry widget ships with: this pass only needs the
                # map's *median* (the gauge the keys are emitted in), and the two nodes then
                # agree out of the box. `warn_depth_model_mismatch` compares model names, so a
                # deliberately different step count does not warn - the gauge it moves is small.
                depth = fast_depth._predict_moge_depth(candidate, first, device)
                return depth, candidate
            if candidate.startswith(fast_depth.DA3_PREFIX):
                height, width = int(first.shape[1]), int(first.shape[2])
                # Same contract as the V2 branch below: (depth, model_name) - probe_surface
                # unpacks it, so a bare tensor here surfaces as "too many values to unpack
                # (expected 2)" for every DA3 pick (production: Depth-Anything-3-Mono-Large).
                depth = fast_depth._predict_da3_depth(
                    candidate, first, device,
                    fast_depth._da3_process_res(depth_res, width, height))
                return depth, candidate
            model = fast_depth._get_depth_model(candidate, device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x518 = F.interpolate(first.permute(0, 3, 1, 2),
                                 size=(fast_depth.DEPTH_RES, fast_depth.DEPTH_RES),
                                 mode="bilinear", align_corners=False)
            with torch.no_grad():
                prediction = model(pixel_values=((x518 - mean) / std).half())
            return (fast_depth._invert_disparity(prediction.predicted_depth[0].float()), candidate)
        except Exception as exc:      # missing weights / no network: try the next family
            last_error = exc
    raise RuntimeError("No depth model is available for the automatic camera "
                       f"(tried {', '.join(attempts)}): {last_error}")


def pivot_provenance_notes(target, surface):
    """Console notes for the ways a *subject* pivot can silently become the SCENE's centre.

    `subject_points` already refuses an unusable mask (it raises), so what can still reach the
    summary is: no mask at all with a collapsed depth split, or a mask that - by its own admission
    - covers (nearly) the whole frame. Both put the pivot in the middle of the background, and
    both have to be said out loud: the orbit would otherwise quietly circle the wrong point while
    the path itself looks perfectly fine.
    """
    if target != SUBJECT_TARGET:
        return []
    source = str(surface.get("source") or "")
    if source == NO_SUBJECT_SOURCE:
        return ["no subject mask is connected and the depth split found no clear near layer, so the "
                "pivot is the centre of the WHOLE surface (background included) - connect a subject "
                "mask (white = subject, e.g. the RMBG MASK output) to orbit the subject itself"]
    coverage = float(surface.get("content_points") or 0) / max(1.0, float(surface.get("points") or 0))
    if source.startswith("input mask") and coverage > MASK_COVERAGE_WARN:
        return [f"the subject mask covers {coverage * 100:.0f} % of the frame, so its points are "
                f"(nearly) the whole cloud and the pivot sits at the background's centre as well - "
                f"check that the mask is white on the SUBJECT, not on the background"]
    return []


def format_summary(summary):
    """One console-friendly line describing an estimate (used by the node)."""
    parts = [
        f"auto camera: {summary['style']} around "
        f"[{summary['pivot'][0]:.3g}, {summary['pivot'][1]:.3g}, {summary['pivot'][2]:.3g}] "
        f"- the cylindrical centre of the {summary['target']} depth profile "
        f"({summary['extents'][0]:.3g} x {summary['extents'][1]:.3g} x "
        f"{summary['extents'][2]:.3g} units, {summary['source']}, "
        f"{summary['content_points']} of {summary['points']} points used - the pivot is computed "
        f"from those and only those)",
    ]
    if summary.get("subject_fill_area"):
        parts.append(
            f"subject {summary['subject_fill_area'] * 100:.0f} % of the frame at the front pose "
            f"({summary.get('subject_fill_min', summary['subject_fill_area']) * 100:.0f}-"
            f"{summary.get('subject_fill_max', summary['subject_fill_area']) * 100:.0f} % along "
            f"the path, {summary['subject_width_px']:.0f} x {summary['subject_height_px']:.0f} px) "
            f"at radius {summary['orbit_radius']:.3g} units; "
            + ("the pivot is the cylinder axis and the aim, so the subject stays in the "
               "horizontal picture centre and the camera keeps that one distance all along the path"
               if summary.get("orbit_look") != HORIZONTAL_LOOK else
               "the pivot is the cylinder axis and the level aim rides it, so the subject stays in "
               "the horizontal picture centre while the camera keeps that one distance to the axis")
            + (f" (cylinder radius {summary['cylinder_radius']:.3g} units, "
               f"{summary.get('cylinder_coverage', 0.0) * 100:.0f} % of the surface inside)"
               if summary.get("cylinder_radius") else "")
        )
    else:
        parts.append(f"radius {summary['orbit_radius']:.3g} "
                     f"({summary['orbit_fill']:g} x content radius)")
    if summary.get("visibility_scale") is not None:
        level = summary.get("orbit_look") == HORIZONTAL_LOOK
        parts.append(
            (f"the subject box stays inside the picture's left/right borders (a level look cannot "
             f"move it vertically, so the top/bottom drift is the look's own; side clearance "
             f"{summary['visibility_clearance_px']:.0f} px at worst, "
             f"{summary['visibility_margin_px']:.0f} px kept free"
             if level else
             f"the whole subject box stays inside every frame (border clearance "
             f"{summary['visibility_clearance_px']:.0f} px at worst, "
             f"{summary['visibility_margin_px']:.0f} px kept free")
            + (f", pulled back {summary['visibility_scale']:.2f}x for it"
               if summary["visibility_scale"] > 1.005 else "")
            + ")"
        )
    parts.append(f"orbit size {summary['orbit_size']:g}x over {summary['frames']} frames")
    # What lies beyond the front circle: nothing at all ('O-Orbit Front'), the level connection to the
    # far orbit's near edge plus its clockwise loop ('O-Orbit Front&Back' - cut when the speed cap cannot
    # pay for it), or the DEPRECATED concluding lap. The view angle is reported either way, because it
    # is the one azimuth that moves all of it.
    orbit = ""
    if summary.get("orbit_coverage_deg"):
        spiral = summary.get("coverage") == SPIRAL_COVERAGE
        if summary.get("coverage") == ORBIT_COVERAGES[1]:
            beyond = (f"+ {summary['lap_span_deg']:.0f} deg level back connection"
                      + (", then the back O loop (9 -> 12 -> 3 -> 6 -> 8 o'clock)"
                         if summary.get("back_orbit")
                         else " (the speed cap could not pay for the back O loop, so the path ends "
                              "where the sweep stopped)"))
        elif spiral:
            winding = summary.get("spiral_end_deg") or SPIRAL_END_DEFAULT
            arc = summary.get("spiral_end_arc_deg") or SPIRAL_END_ARC_DEFAULT
            slope = summary.get("spiral_slope_deg") or 0.0
            axis = ("the view axis" if abs(slope) < 0.5 else
                    f"a central axis leaned {slope:.0f} deg out of the view axis"
                    + (" (from straight above)" if slope > 0 else " (from straight below)"))
            beyond = (f"{winding:.0f} deg of winding ({winding / 360.0:.2f} rounds) while the arc "
                      f"out of it climbs to {arc:.0f} deg - the last frame is in the picture's own "
                      f"plane")
            if summary.get("spiral_end_clock_deg") is not None:
                beyond += (f", ending clock {summary['spiral_end_clock_deg']:.0f} deg / elevation "
                           f"{summary['spiral_end_elevation_deg']:.0f} deg")
        elif summary.get("coverage") in (BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):
            spread = "/".join(f"{value:.0f}"
                              for value in (summary.get("band_elevations_deg") or []))
            beyond = (f"banded orbits ({summary.get('band_count')} bands of "
                      f"{summary.get('band_width_deg'):.0f} deg at {spread} deg elevation"
                      + (", front and back"
                         if summary.get("coverage") == BAND_FRONT_BACK_COVERAGE else "") + ")")
        elif summary.get("coverage"):
            beyond = "front circle only"
        else:
            beyond = f"+ {summary['lap_span_deg']:.0f} deg lap"
        if spiral:
            # The spiral has no "deg around" azimuth to report: its coverage *is* the winding, and
            # the view angle is the azimuth of the axis it winds around.
            view = summary.get("view_angle_deg") or 0.0
            centre = f" centred on {view:.0f} deg" if abs(view) >= 0.5 else ""
            orbit = (f", spherical spiral {summary.get('orbit_direction', '')} around {axis}"
                     f"{centre}: {beyond}")
        elif summary.get("coverage") in (BAND_COVERAGE, BAND_FRONT_BACK_COVERAGE):
            # The band stack has no front O either: its swing is the Band Width widget, so the
            # "O +/-..." phrasing below would name a swing the path never flies.
            orbit = f", {beyond} = {summary['orbit_coverage_deg']:.0f} deg around"
        else:
            orbit = (f", O +/-{summary['front_yaw_deg']:.0f} deg "
                     f"{summary.get('orbit_direction', '')} "
                     f"centred on {summary.get('view_angle_deg') or 0.0:.0f} deg {beyond} "
                     f"to {summary['orbit_end_deg']:.0f} deg "
                     f"= {summary['orbit_coverage_deg']:.0f} deg around")
    if summary.get("drift_px") is not None:
        parts.append(
            f"({summary['drift_px']:.3g} px/frame subject drift, typical "
            f"{summary['drift_typical_px']:.3g} px, cap {summary['drift_cap_px']:.3g} px, "
            f"{summary['amplitude_scale']:.2f}x amplitude, {summary['keys']} keys"
            + orbit
            + ")"
        )
    else:
        parts.append(
            f"({summary['travel_per_frame']:.3g} units/frame at "
            f"{summary['amplitude_scale']:.2f}x amplitude, budget "
            f"{summary['max_speed'] * 100:.0f} %/frame, {summary['keys']} keys)"
        )
    line = ", ".join(parts)
    if summary.get("rows"):
        line += (f" - survey: {summary['rows']} rows across +/-{summary['scene_half_yaw']:.0f} deg "
                 f"over {summary['scene_half_width'] * 2:.3g} units of scene width "
                 f"({summary['lane_overlap'] * 100:.0f} % side-lap, {summary['lane_step']:.3g} units "
                 f"between rows, elevations "
                 f"{summary['scene_low_elevation']:.0f} -> {summary['scene_high_elevation']:.0f} deg)")
        if summary.get("scene_reach") is not None:
            line += (f", stand-off solved at {summary.get('scene_fill_solved', 0.0):.2f} x content "
                     f"radius for {summary['scene_reach'] * 100:.0f} % of the scene's half width "
                     f"(footprint {summary.get('scene_footprint', 0.0):.3g} units)")
        if summary.get("scene_subject_share") is not None:
            line += (f", the subject inside it covers {summary['scene_subject_share'] * 100:.1f} % "
                     f"of the picture in {summary['scene_subject_frames']} of "
                     f"{summary['frames']} frames ({summary['scene_subject_whole_frames']} whole)")
    if summary.get("hint"):
        line += f" - {summary['hint']}"
    if summary["collision_fixes"]:
        line += (f" - {summary['collision_fixes']} key(s) pushed out of the scene "
                 f"(closest approach {summary['clearance_before'] * 100:.1f} % -> "
                 f"{summary['clearance_after'] * 100:.1f} % of the content radius)")
    # Face-detail orbit: report the inserted face segment and the eased connections,
    # along with its achieved fill; this is a middle pass, not an appended tail.
    face = summary.get("face_orbit") or {}
    status = face.get("status")
    if status == "applied":
        before = face.get("fill_before")
        after = face.get("fill_after")
        line += (f" - face orbit on {len(face.get('frames') or [])} frame(s) between the "
                 f"body bands ({face.get('span', 0):.0f} deg, "
                 f"{len(face.get('transition_in_keys') or [])} in / "
                 f"{len(face.get('transition_out_keys') or [])} out transition keys)")
        if before is not None and after is not None:
            line += f", face {before * 100:.2f} % -> {after * 100:.2f} % of the picture"
    elif status and status not in ("off", "no face box"):
        line += f" - no face orbit ({status})"
    return line


# ---------------------------------------------------------------------------
# Face-detail orbit
# ---------------------------------------------------------------------------
#
# A full-body turntable shows the face at ~0.4-0.8 % of the frame, so the splat has
# almost nothing to learn it from. Reallocate existing body-path frames to a face-height
# orbit BETWEEN the lower and higher passes, easing camera and aim into and back out of it.
#
# Why re-aiming costs nothing: the per-frame weight of a training frame is
# `iterations / n_frames` (both 2DGS and PGSR draw a shuffled epoch), so ADDING head
# frames dilutes the body frames. Re-purposing existing ones does not - the frame
# count, and with it every other frame's pass count, is untouched.
#
# The gate lives upstream of this file (`enndee_face_detect`, SAM3): a face box is
# measured on the reference still, and without one this whole pass is a no-op.

#: Defaults for the widgets on the camera node.
FACE_ORBIT_FILL_DEFAULT = 20.0        # percent of the frame area the face should cover
FACE_ORBIT_FILL_MIN = 1.0
FACE_ORBIT_FILL_MAX = 80.0
FACE_ORBIT_SPAN_DEFAULT = 60.0        # deg of yaw the zoom is spread over
FACE_ORBIT_SPAN_MIN = 6.0
FACE_ORBIT_SPAN_MAX = 360.0
FACE_ORBIT_ELEVATION_TOLERANCE = 1.5  # deg - which frames still count as "the highest band"
FACE_ORBIT_MIN_FRAMES = 3             # fewer frames than this is not worth re-aiming
FACE_ORBIT_ITERATIONS = 5             # distance fixed-point steps (perspective is not exactly 1/d^2)
#: Stand-off floor in face radii. `face_radius` is the face patch's MAX extent from its
#: centre, so this is a bounding-sphere stand-off - the camera never enters the head.
#: It is also what silently caps `face_fill`: at 1.35 the solve hit the floor long before
#: a large fill was reached (measured: face_fill=70 stopped at 34.8 %). Keep it at or
#: above 1.0 to stay outside the patch, and read `info["clearance_limited"]`.
FACE_ORBIT_CLEARANCE = 1.0
#: Collision margin for face keys only, in content radii. `guard_collisions` normally
#: holds every key `COLLISION_MARGIN` content radii off the scene, but the face orbit
#: DELIBERATELY parks the camera at the head - which is scene geometry - and enforces its
#: own stand-off instead. Letting the generic margin push those keys back out cancelled
#: the zoom entirely (rendered face area stayed at 0.15 % against a 20 % target while the
#: orbit reported success). This keeps a safety floor without undoing the zoom.
FACE_ORBIT_COLLISION_MARGIN = 0.05
#: How far to shift the face orbit's AIM toward the reference camera, as a fraction of the
#: face patch's own depth extent (p90 - p10). **0.0 = OFF, and that is deliberate.**
#:
#: This was a bandage for the box-registration bug: with the SAM3 box projected from the
#: wrong camera the patch median sat mid-skull, so the aim was nudged forward to compensate.
#: The registration is now exact (`_project_source`), and the user reports the opposite
#: symptom - "the pivot point is still not right on the face but rather a bit in front left
#: of it, that's why the face is drifting from left to right during the orbit". A shift
#: TOWARD the camera moves the pivot in front of the face, and because the reference camera
#: is off-axis it also slides it sideways - precisely "in front left".
#:
#: The pivot must be the MIDDLE of the face depth points (see `head_pivot`), with no
#: correction on top. Raise this only if the aim is measurably BEHIND the face again.
    #: Retained as a no-op compatibility constant. The additional orbit is explicitly allocated
    #: output frames and no longer tapers body-orbit control keys.
FACE_ORBIT_TAPER_KEYS = 2


def _yaw_elevation_of(position, pivot):
    """(yaw, elevation, radius) of a camera position around the pivot.

    The inverse of `_place`: yaw 0 is between the pivot and the world origin ("front",
    +yaw the subject's right), elevation 0 is level and + is above. Recovering the
    angles from the positions is what lets this pass work on ANY coverage - the band
    stack, the front O, the spiral - without touching their builders.
    """
    dx = float(position[0]) - float(pivot[0])
    dy = float(position[1]) - float(pivot[1])
    dz = float(position[2]) - float(pivot[2])
    radius = math.sqrt(dx * dx + dy * dy + dz * dz)
    if radius <= 1e-9:
        return 0.0, 0.0, 0.0
    yaw = math.degrees(math.atan2(dx, -dz))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, -dy / radius))))
    return yaw, elevation, radius


FACE_ORBIT_WHOLE_TOP_BAND = True
FACE_AIM_FORWARD_SHIFT = 0.0


def _angle_delta(angle, centre):
    """Signed difference in degrees, wrapped into (-180, 180]."""
    return (float(angle) - float(centre) + 180.0) % 360.0 - 180.0


def _rendered_frame_span(key_indices, key_count, frames_total):
    """Inclusive output-frame range spanned by selected path control keys."""
    if not key_indices:
        return []
    last = max(1, int(key_count) - 1)
    total = max(2, int(frames_total or key_count))
    first_frame = min(total - 1, int(round(min(key_indices) * (total - 1) / last)))
    last_frame = min(total - 1, int(round(max(key_indices) * (total - 1) / last)))
    return list(range(first_frame, last_frame + 1))


def _face_path_intervals(frames):
    """Distribute frame intervals across body, transitions, and the face orbit."""
    segments = 5
    total = max(0, int(frames) - 1)
    minimum = 2
    if total < segments * minimum:
        return None
    weights = (0.25, 0.18, 0.14, 0.18, 0.25)
    remaining = total - segments * minimum
    exact = [remaining * weight for weight in weights]
    intervals = [minimum + int(value) for value in exact]
    left = total - sum(intervals)
    order = sorted(range(segments), key=lambda index: exact[index] - int(exact[index]),
                   reverse=True)
    for index in order[:left]:
        intervals[index] += 1
    return intervals


def _ease_face_transition(start, stop, phase):
    """Smoothly interpolate position and aim between a body pose and the face orbit."""
    phase = min(1.0, max(0.0, float(phase)))
    weight = phase * phase * (3.0 - 2.0 * phase)
    return {
        "pos": [float(a) + (float(b) - float(a)) * weight
                for a, b in zip(start["pos"], stop["pos"])],
        "look": [float(a) + (float(b) - float(a)) * weight
                 for a, b in zip(start["look"], stop["look"])],
    }


def _sample_face_source(keys, phase, pivot=None, yaw_offset=0.0):
    """Interpolate one source-path pose, optionally rotating it around its orbit pivot."""
    old_first = float(keys[0]["t"])
    old_last = float(keys[-1]["t"])
    old_t = old_first + min(1.0, max(0.0, float(phase))) * (old_last - old_first)
    right = next((index for index in range(1, len(keys))
                  if float(keys[index]["t"]) >= old_t), len(keys) - 1)
    left = max(0, right - 1)
    segment = max(1e-9, float(keys[right]["t"]) - float(keys[left]["t"]))
    blend = min(1.0, max(0.0, (old_t - float(keys[left]["t"])) / segment))
    pos = [float(keys[left]["pos"][axis]) * (1.0 - blend)
           + float(keys[right]["pos"][axis]) * blend for axis in range(3)]
    look = [float(keys[left]["look"][axis]) * (1.0 - blend)
            + float(keys[right]["look"][axis]) * blend for axis in range(3)]
    if abs(float(yaw_offset)) > 1e-9 and pivot is not None:
        yaw, elevation, radius = _yaw_elevation_of(pos, pivot)
        pos = _place(pivot, radius, yaw + float(yaw_offset), elevation)
    return {"pos": pos, "look": look}


def _sample_face_segments(segment_functions, intervals):
    """Join five sampled path pieces, sharing their boundary key exactly once."""
    result = []
    ranges = []
    for index, (sample, count) in enumerate(zip(segment_functions, intervals)):
        values = [sample(step / count) for step in range(count + 1)]
        start = len(result) - 1 if index else 0
        result.extend(values if not index else values[1:])
        ranges.append((start, len(result) - 1))
    return result, ranges


def add_face_height_orbit(keys, frames_total, pivot, face_centre, face_span,
                          face_distance, face_radius, direction=1.0, coverage=None,
                          bands_start_yaw=None, bands_width=None, bands_elevations=None):
    """Insert a face-height orbit between smoothly connected lower and upper path pieces.

    A two-band path keeps both requested body sweeps and puts the face sweep between
    them. Other paths are split at their midpoint; the upper half is yaw-offset by the
    face sweep so the camera flows from face into that half rather than retracing it.
    The output frame count and real-time ``src == t`` invariant are unchanged.
    """
    if not keys or face_centre is None:
        return keys, {"status": "no face orbit", "frames": []}
    total = int(frames_total)
    if total < 6:
        return keys, {"status": "not enough frames for body and face orbits", "frames": []}
    intervals = _face_path_intervals(total)
    if intervals is None:
        return keys, {"status": "not enough frames for body and face transitions", "frames": []}
    span = min(360.0, max(6.0, float(face_span)))
    radius = max(float(face_distance), 1e-3)
    face_centre = [float(value) for value in face_centre]
    body_radius = max(1e-3, _yaw_elevation_of(keys[0]["pos"], pivot)[2])
    try:
        _face_yaw, face_angle, face_offset = _yaw_elevation_of(face_centre, pivot)
        if not all(math.isfinite(value) for value in (face_angle, face_offset)):
            face_angle = 0.0
    except (TypeError, ValueError, OverflowError):
        face_angle = 0.0
    yaw_delta = -span if float(direction) < 0.0 else span

    # For the two-band coverage, preserve the complete lower and upper sweeps and
    # insert the 60-degree face pass at their turning point. Continuing in the upper
    # band's direction avoids flying back over the just-completed face arc.
    if str(coverage) == BAND_COVERAGE:
        elevations = ([float(value) for value in bands_elevations]
                      if bands_elevations is not None else band_elevations())
        if len(elevations) == 2:
            start_yaw = (band_start_yaw() if bands_start_yaw is None
                         else resolve_band_start_yaw(bands_start_yaw))
            width = (band_width() if bands_width is None
                     else resolve_band_width(bands_width))
            low_index = min(range(2), key=lambda index: elevations[index])
            high_index = 1 - low_index
            low_start, low_end = _band_orbit(low_index, 2, width, start_yaw)
            low_direction = 1.0 if low_end >= low_start else -1.0
            # Let the lower sweep finish its configured arc, then ease back by half
            # the face span so the face orbit is centred on the lower sweep's endpoint.
            # After the face arc, ease back to the lower sweep's start yaw and let the
            # higher band repeat that same configured sweep. This keeps the body bands
            # aligned instead of carrying the face orbit's yaw into the upper sweep.
            face_start_yaw = low_end - low_direction * span / 2.0
            face_end_yaw = face_start_yaw - low_direction * span
            higher_start_yaw = low_start
            higher_end_yaw = higher_start_yaw + low_direction * width
            high_start_pose = {"pos": _place(pivot, body_radius, higher_start_yaw,
                                             elevations[high_index]),
                               "look": [float(value) for value in pivot]}
            low_pose = lambda phase: {
                "pos": _place(pivot, body_radius,
                              low_start + (low_end - low_start) * phase,
                              elevations[low_index]),
                "look": [float(value) for value in pivot],
            }
            face_entry = {"pos": _place(face_centre, radius, face_start_yaw, 0.0),
                          "look": face_centre}
            face_exit = {"pos": _place(face_centre, radius, face_end_yaw, 0.0),
                         "look": face_centre}
            segments = (
                low_pose,
                lambda phase: _ease_face_transition(
                    low_pose(1.0), face_entry, phase),
                lambda phase: {
                    "pos": _place(face_centre, radius,
                                  face_start_yaw + (face_end_yaw - face_start_yaw) * phase, 0.0),
                    "look": face_centre,
                },
                lambda phase: _ease_face_transition(face_exit, high_start_pose, phase),
                lambda phase: {
                    "pos": _place(pivot, body_radius,
                                  higher_start_yaw + (higher_end_yaw - higher_start_yaw) * phase,
                                  elevations[high_index]),
                    "look": [float(value) for value in pivot],
                },
            )
            body_info = {"lower_elevation": elevations[low_index],
                         "higher_elevation": elevations[high_index],
                         "lower_yaw_start": low_start, "lower_yaw_end": low_end,
                         "face_yaw_start": face_start_yaw,
                         "face_yaw_end": face_end_yaw,
                         "higher_yaw_start": higher_start_yaw,
                         "higher_yaw_end": higher_end_yaw,
                         "band_width": width}
        else:
            body_info = None
    else:
        body_info = None

    if body_info is None:
        # Split an arbitrary path at its midpoint and continue the upper half in the
        # face arc's direction. The two eased joins are explicit keys, never a jump.
        middle = _sample_face_source(keys, 0.5, pivot)
        before = _sample_face_source(keys, max(0.0, 0.5 - 1.0 / max(2, total - 1)), pivot)
        after = _sample_face_source(keys, min(1.0, 0.5 + 1.0 / max(2, total - 1)), pivot)
        yaw_before = _yaw_elevation_of(before["pos"], pivot)[0]
        yaw_after = _yaw_elevation_of(after["pos"], pivot)[0]
        yaw_direction = _angle_delta(yaw_after, yaw_before)
        if abs(yaw_direction) < 1e-6:
            yaw_direction = -span if float(direction) < 0.0 else span
        yaw_delta = math.copysign(span, yaw_direction)
        start_yaw = _yaw_elevation_of(middle["pos"], pivot)[0]
        face_end_yaw = start_yaw + yaw_delta
        face_entry = {"pos": _place(face_centre, radius, start_yaw, 0.0),
                      "look": face_centre}
        face_exit = {"pos": _place(face_centre, radius, face_end_yaw, 0.0),
                     "look": face_centre}
        upper_start = _sample_face_source(keys, 0.5, pivot, yaw_delta)
        segments = (
            lambda phase: _sample_face_source(keys, 0.5 * phase, pivot),
            lambda phase: _ease_face_transition(middle, face_entry, phase),
            lambda phase: {
                "pos": _place(face_centre, radius,
                              start_yaw + yaw_delta * phase, 0.0),
                "look": face_centre,
            },
            lambda phase: _ease_face_transition(face_exit, upper_start, phase),
            lambda phase: _sample_face_source(keys, 0.5 + 0.5 * phase, pivot, yaw_delta),
        )

    samples, segment_ranges = _sample_face_segments(segments, intervals)
    if len(samples) != total:
        return keys, {"status": "face orbit frame allocation failed", "frames": []}
    face_start = segment_ranges[1][0]
    face_end = segment_ranges[3][1]
    face_orbit_start, face_orbit_end = segment_ranges[2]
    face_keys = list(range(face_start, face_end + 1))
    # Only the face-centred span is a face-detail frame. Keep the transition keys out
    # of COLMAP's face mask so its close-up cut-out does not remove the head while the
    # camera is still easing to/from it.
    face_frames = _rendered_frame_span(range(face_orbit_start, face_orbit_end + 1),
                                       total, total)
    orbit_keys = []
    for index, sample in enumerate(samples):
        key = {"pos": [float(value) for value in sample["pos"]],
               "look": [float(value) for value in sample["look"]],
               "src": index, "t": index}
        if face_orbit_start <= index <= face_orbit_end:
            key["face"] = True
        orbit_keys.append(key)
    face_info = {
        "status": "applied", "frames": face_frames, "face_start": face_start,
        "face_end": face_end, "face_orbit_keys": list(range(face_orbit_start,
                                                               face_orbit_end + 1)),
        "transition_in_keys": list(range(segment_ranges[1][0], segment_ranges[1][1] + 1)),
        "transition_out_keys": list(range(segment_ranges[3][0], segment_ranges[3][1] + 1)),
        "keys": face_keys, "face_fill": float(face_distance), "face_span": span,
        "face_elevation": float(face_angle), "face_elevation_about_pivot": float(face_angle),
        **(body_info or {}),
    }
    return orbit_keys, face_info


def face_orbit_window(keys, pivot, span=FACE_ORBIT_SPAN_DEFAULT, centre_yaw=0.0,
                      tolerance=FACE_ORBIT_ELEVATION_TOLERANCE):
    """Indices of the keys that sit on the highest orbit inside the yaw window.

    "Highest orbit" is the elevation the most frames share at the top of the stack
    (a tolerance band rather than an exact match, because the elevation step between
    bands is a widget and may be small). Within that band only the frames whose yaw
    is inside `centre_yaw +- span/2` are returned, in flight order - exactly the
    "at -30 to 30 deg it zooms into the face" the face orbit is specified as.
    """
    if not keys:
        return []
    angles = [_yaw_elevation_of(key["pos"], pivot) for key in keys]
    top = max(entry[1] for entry in angles)
    window = []
    for index, (yaw, elevation, _radius) in enumerate(angles):
        if abs(elevation - top) > float(tolerance):
            continue
        # WHOLE TOP BAND: every key at the top elevation, ignoring the yaw window, so the
        # highest orbit is flown around the FACE pivot end to end (see
        # `FACE_ORBIT_WHOLE_TOP_BAND`). `span` only narrows the selection when that is off.
        if FACE_ORBIT_WHOLE_TOP_BAND or abs(_angle_delta(yaw, centre_yaw)) <= float(span) / 2.0:
            window.append(index)
    return window


def _resolve_fill_vfov(vfov_degrees):
    """The vertical FOV to project with, in priority order.

    ``vfov_degrees`` (what the caller measured) -> ``fast_depth.LAST_VFOV_DEGREES`` (what
    the cloud was ACTUALLY built with - MoGe-3 reports it and the probe publishes it) ->
    ``fast_depth.VFOV_DEGREES`` (the hard-coded 55 deg assumption). The middle step is the
    important one: without it the face orbit measured the head at 55 deg instead of the
    measured 38.07 deg (a 1.509x error in f), selected a patch ~4x too big, and solved for
    2.63 units of stand-off where ~0.65 is right - so `face_fill` looked ignored.
    """
    if vfov_degrees:
        return float(vfov_degrees)
    measured = getattr(fast_depth, "LAST_VFOV_DEGREES", None)
    if measured:
        return float(measured)
    return float(fast_depth.VFOV_DEGREES)


def _project_with_pose(points, position, look, height, aspect, vfov_degrees=None):
    """(u, v, z) normalised pixels of `points` seen from `position` looking at `look`.

    `vfov_degrees` is the vertical field of view the CLOUD was built with. It must match
    the renderer's, or every projection here is wrong: the cloud's x/y carry `z / f` with
    `f` from the MEASURED vfov (38.04 deg on the Hanks still) while this function
    defaulted to the hard-coded 55 deg, a 1.509x error in f. That corrupts BOTH uses of
    this function - which cloud points land in the SAM3 face box (`_face_points`), and
    the area the orbit solves for (`_projection_area`). Passing `None` keeps the old
    constant so callers that do not know the measurement are unchanged.

    The renderer's own pinhole (see `project_points`), re-expressed with an explicit
    aim so it also covers the 'Horizontally straight' look, whose aim is NOT the
    pivot. u/v come back in 0..1 so they can be compared with a face box directly.

    Everything is built on the **points' device**: the surface cloud arrives on the GPU
    in a real ComfyUI run, and mixing it with CPU constants raises
    "Expected all tensors to be on the same device, cuda:0 and cpu".
    """
    pool = points.detach().to(torch.float64).reshape(-1, 3)
    device = pool.device
    forward = torch.tensor([float(look[axis]) - float(position[axis]) for axis in range(3)],
                           dtype=torch.float64, device=device)
    norm = float(forward.norm())
    forward = (torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)
               if norm <= 1e-9 else forward / norm)
    up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64, device=device)  # y-down world
    right = torch.linalg.cross(forward, up, dim=0)
    if float(right.norm()) < 1e-6:                                # looking straight down
        right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right, dim=0)

    origin = torch.tensor([float(value) for value in position],
                          dtype=torch.float64, device=device).reshape(1, 3)
    relative = pool - origin
    x = (relative * right.reshape(1, 3)).sum(-1)
    y = (relative * down.reshape(1, 3)).sum(-1)
    z = (relative * forward.reshape(1, 3)).sum(-1)
    focal = 0.5 * float(height) / math.tan(
        math.radians(vfov_degrees if vfov_degrees is not None
                     else fast_depth.VFOV_DEGREES) / 2.0)
    width = float(height) * float(aspect)
    safe = z.clamp(min=1e-6)
    return (0.5 + focal * x / safe / width,
            0.5 + focal * y / safe / float(height), z)


def _project_source(points, height, aspect, vfov_degrees=None):
    """Exact inverse of :func:`surface_points` - the SOURCE camera, not the orbit's.

    `surface_points` unprojects the depth map into frame-0 camera coordinates: origin at
    the camera that captured the reference still, +z along the view axis,
    `x = (xx - W/2)/f * z`, `y = (yy - H/2)/f * z`, `f = 0.5 * H / tan(vfov/2)`. So a cloud
    point maps back to a pixel as

        u = 0.5 + x * f / (z * W)        v = 0.5 + y * f / (z * H)

    with no rotation basis to get wrong. A SAM3 box lives in the REFERENCE IMAGE, so this
    - and only this - is the projection that registers it onto the cloud. Projecting with
    the orbit's front pose instead mis-registers the box laterally AND in depth, which is
    the observed "the face orbit aims slightly left of and behind his head", and at a tight
    stand-off it points the camera into empty grey.
    """
    height = float(height)
    width = height * float(aspect)
    focal = 0.5 * height / math.tan(
        math.radians(vfov_degrees if vfov_degrees is not None
                     else fast_depth.VFOV_DEGREES) / 2.0)
    p = points.detach().to(torch.float64).reshape(-1, 3)
    z = p[:, 2]
    safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    u = 0.5 + p[:, 0] * focal / (safe * width)
    v = 0.5 + p[:, 1] * focal / (safe * height)
    return u, v, z


def head_pivot(surface, front_position, front_look, face_box, height, aspect,
               margin=1.15, radius_percentile=90.0, vfov_degrees=None):
    """Robust ``(centre, radius, points, patch)`` for the head - the face path's pivot.

    This replaces the old ``_face_points`` centre/radius for anything that has to place
    a CAMERA, because that pair was wrong in two ways:

    * ``radius = max ||p - median||`` over the selected cloud points. A MAX is one stray
      point away from infinity - measured 0.9355 world units for a head that is really
      ~0.15-0.20, i.e. ~5x too big. Every downstream number inherits that: the stand-off
      floor, the distance solve, the fill.
    * the centre was a plain median of whatever landed in the SAM3 box, which can sit off
      the face when the box is tight or the cloud is sparse.

    So: drop the upper-distance outliers first, take the median of what is left as the
    centre, and use a PERCENTILE of the residual distances as the radius. Returns the
    same 4-tuple shape as before so callers can adopt it without churn.
    """
    found = _face_points(surface, front_position, front_look, face_box,
                         height, aspect, margin=margin, vfov_degrees=vfov_degrees)
    if found is None:
        return None
    _old_centre, _old_radius, _count, patch = found
    if patch is None or patch.numel() == 0:
        return None
    pts = patch.detach().to(torch.float64).reshape(-1, 3)
    if pts.shape[0] < 8:
        return None
    # THE PIVOT IS THE MIDDLE OF THE FACE POINTS. Not a trimmed estimate of it: the user
    # asked for the pivot "right into the middle of the face depth points", and a trim that
    # re-derives the centre afterwards slides it off that middle. So compute the centre ONCE
    # from every face point and never touch it again - the trimming below exists only to stop
    # stray points inflating the RADIUS, and must not move the pivot.
    centre = pts.median(dim=0).values
    dist = (pts - centre.reshape(1, 3)).norm(dim=1)
    # first pass: cut the far tail so the radius is not inflated by stray points
    cut = torch.quantile(dist, 0.75)
    if float(cut) > 1e-9:
        keep = dist <= cut * 2.0
        if int(keep.sum()) >= 8:
            pts = pts[keep]
            # `centre` is DELIBERATELY NOT recomputed here. It is the middle of ALL the face
            # points (see above); only the radius uses the trimmed set, so stray points
            # cannot inflate it without dragging the pivot off the face's centre.
            dist = (pts - centre.reshape(1, 3)).norm(dim=1)
    radius = float(torch.quantile(dist, float(radius_percentile) / 100.0))
    if not math.isfinite(radius) or radius <= 1e-9:
        return None
    return centre, radius, int(pts.shape[0]), pts


def head_distance_for_fill(radius: float, fill: float, aspect: float,
                           vfov_degrees: float) -> float:
    """Camera distance at which a sphere of `radius` covers `fill` of the frame.

    Closed form, so there is no fixed-point loop to pin at a stand-off floor:

        x = (u - cx) / f * z  ->  projected extent = 2r * f / d
        area = (2r f / d)^2 / (width * height)
             = r^2 / (d^2 * aspect * tan^2(vfov/2))
        =>  d = r / (sqrt(fill) * sqrt(aspect) * tan(vfov/2))

    `fill` is a fraction of the frame (0.20 = 20 %), `aspect` is the RENDER canvas'
    width/height, `vfov_degrees` the vertical field of view the cloud was built with.
    """
    fill = max(1e-6, min(0.999, float(fill)))
    aspect = max(1e-6, float(aspect))
    half = math.tan(math.radians(float(vfov_degrees)) / 2.0)
    if half <= 1e-9:
        raise ValueError("vfov_degrees must be in (0, 180)")
    return float(radius) / (math.sqrt(fill) * math.sqrt(aspect) * half)


def _projection_area(points, position, look, height, aspect):
    """Share of the frame the points' projected box covers (0.0 when invisible)."""
    u, v, z = _project_with_pose(points, position, look, height, aspect)
    visible = z > 1e-4
    if int(visible.sum()) < 4:
        return 0.0
    uu, vv = u[visible], v[visible]
    # A percentile box, like the framing metrics: one stray depth spike must not
    # pretend the face is huge and send the camera into the floor.
    low, high = FRAMING_PERCENTILE_LOW / 100.0, FRAMING_PERCENTILE_HIGH / 100.0
    u0, u1 = float(torch.quantile(uu, low)), float(torch.quantile(uu, high))
    v0, v1 = float(torch.quantile(vv, low)), float(torch.quantile(vv, high))
    return max(0.0, u1 - u0) * max(0.0, v1 - v0)


def _face_points(surface, front_position, front_look, face_box, height, aspect, margin=1.15,
                 vfov_degrees=None):
    """The subject-cloud points that fall inside the face box, in world space.

    This is how the face's 3D position is found without a second depth pass: the
    cloud is already in world units and the face box is in the reference image, so
    "project the cloud through the reference pose and keep what lands in the box" is
    exact. `margin` widens the box a little, so a few points survive the box being
    tighter than the actual head.
    """
    cloud = surface.get("subject_cloud")
    if cloud is None or cloud.numel() == 0:
        cloud = surface.get("content_cloud")
    if cloud is None or cloud.numel() == 0:
        return None
    # PROJECT WITH THE SOURCE CAMERA. The SAM3 box is a pixel rectangle in the reference
    # still, and `surface_points` put the cloud in THAT camera's frame - so the exact
    # inverse is all that is needed. `_project_with_pose` with the orbit's front pose was
    # wrong here: it registers the box from a different viewpoint, sliding the selection
    # sideways and behind the head.
    u, v, z = _project_source(cloud, height, aspect, vfov_degrees=vfov_degrees)
    cx = float(face_box["cx"])
    cy = float(face_box["cy"])
    half_w = float(face_box["width"]) * 0.5 * float(margin)
    half_h = float(face_box["height"]) * 0.5 * float(margin)
    inside = ((u - cx).abs() <= half_w) & ((v - cy).abs() <= half_h) & (z > 1e-4)
    selected = cloud.detach().to(torch.float64).reshape(-1, 3)[inside]
    if selected.shape[0] < 8:
        return None
    centre = selected.median(dim=0).values
    radius = float((selected - centre.reshape(1, 3)).norm(dim=1).max())
    if radius <= 1e-9:
        return None
    # The *patch* travels with the centre: the zoom solve has to project the face's
    # real extent, not a single point (one point projects to one pixel and its
    # "area" is zero, which would leave the camera at the orbit distance).
    return centre, radius, int(selected.shape[0]), selected


def apply_face_orbit(keys, surface, face_box, fill=FACE_ORBIT_FILL_DEFAULT,
                     span=FACE_ORBIT_SPAN_DEFAULT, height=CANVAS_HEIGHT,
                     centre_yaw=0.0, front_position=None, front_look=None,
                     clearance=FACE_ORBIT_CLEARANCE, pivot=None, frames_total=None,
                     vfov_degrees=None, render_aspect=None, coverage=None):
    """(keys, info) - insert a face-height orbit between smoothly joined body sections.

    `face_box` is the normalised ``cx, cy, width, height`` the SAM3 gate measured on
    the reference still (see :mod:`enndee_face_detect`); ``None`` means "no face", and
    the keys come back untouched. `front_position`/`front_look` describe the pose the
    reference still was taken from - the framing's own front pose - and are only used
    to find *which cloud points are the face*.

    `span` sets the face yaw arc and `fill` its camera-to-face stand-off. For a two-band
    path, the face orbit is flown between the complete lower and higher sweeps. Both
    joins ease camera position and aim so there is no elevation jump or appended tail.
    The total frame count and source-frame indexing are unchanged.
    """
    info = {"status": "off", "frames": [], "span": float(span), "fill": float(fill),
            "face_area": None, "face_radius": None, "points": 0,
            "fill_before": None, "fill_after": None}
    if face_box is None or not keys:
        info["status"] = "no face box"
        return keys, info

    pivot = surface["pivot"] if pivot is None else pivot
    aspect = float(surface.get("aspect") or (16.0 / 9.0))
    if front_position is None:
        info["status"] = "no reference pose"
        return keys, info
    if front_look is None:
        front_look = _aim(pivot, front_position)

    # `fill_vfov`/`fill_aspect` must be known BEFORE the head is measured. `_face_points`
    # projects the SAM3 box onto the cloud to decide which points ARE the head, and doing
    # that with the hard-coded 55 deg instead of the measured ~38 deg selects a patch ~4x
    # too big (measured `face_radius` 0.748 for a ~0.18 head). That inflated radius is what
    # made the solved stand-off far too large: the orbit moved the camera 3.0 -> 2.63 and
    # looked like `face_fill` was ignored. The box also lives in the RENDER canvas, so the
    # canvas aspect is the right one here, not the cloud's.
    # TWO different aspects, for two different jobs - conflating them moved the pivot.
    #
    # * BOX SELECTION (`head_pivot` / `_face_points`) must use the SOURCE aspect. The SAM3
    #   box is normalised in the reference image (3440x2312 = 1.488), so projecting the
    #   cloud into u/v with `width = height * aspect` only lines up with the box when
    #   `aspect` is the source's. Handing it the CANVAS aspect (736x544 = 1.353) shifts the
    #   mapped box ~10 % sideways and drags the pivot off the head - the observed "aims
    #   slightly left of the head".
    # * The DISTANCE SOLVE must use the CANVAS aspect, because `face_fill` is measured as a
    #   share of the RENDERED frame and that frame is 736x544.
    fill_aspect = float(render_aspect) if render_aspect else float(aspect)
    fill_vfov = _resolve_fill_vfov(vfov_degrees)
    found = head_pivot(surface, front_position, front_look, face_box, height, aspect,
                       vfov_degrees=fill_vfov)
    if found is None:
        info["status"] = "the face box holds no surface points"
        return keys, info
    face_centre, face_radius, point_count, face_patch = found
    # BOUNDED AIM NUDGE - the "the orbit aims behind his head" half. The face box selects
    # the whole head VOLUME, so the patch's median sits mid-skull and the orbit looks at a
    # point behind the face. This shifts the AIM toward the reference camera by a fraction
    # of the patch's OWN depth extent. Unlike a near-surface trim it throws away no points,
    # so `face_radius` and the distance solve are untouched - the trim variant dragged the
    # camera into empty grey (see the note in `head_pivot`). The correction is bounded by
    # construction: it can move the aim at most
    # `FACE_AIM_FORWARD_SHIFT * (p90 - p10)` of the patch's depth.
    if FACE_AIM_FORWARD_SHIFT > 0.0 and face_patch is not None and face_patch.numel() >= 24:
        _pts = face_patch.detach().to(torch.float64).reshape(-1, 3)
        _fp = torch.tensor([float(value) for value in front_position],
                           dtype=torch.float64, device=face_centre.device)
        _fl = torch.tensor([float(value) for value in front_look],
                           dtype=torch.float64, device=face_centre.device)
        _fwd = _fl - _fp
        _n = float(_fwd.norm())
        if _n > 1e-9:
            _depth = (_pts - _fp.reshape(1, 3)) @ (_fwd / _n)
            _extent = float(torch.quantile(_depth, 0.9) - torch.quantile(_depth, 0.1))
            if _extent > 1e-9:
                face_centre = face_centre - (_fwd / _n) * (FACE_AIM_FORWARD_SHIFT * _extent)
    info["face_radius"] = float(face_radius)
    info["points"] = point_count
    info["face_area"] = _projection_area(face_patch, front_position,
                                         front_look, height, aspect)

    target = min(FACE_ORBIT_FILL_MAX, max(FACE_ORBIT_FILL_MIN, float(fill))) / 100.0
    floor = max(1e-9, float(clearance) * float(face_radius))
    # CLOSED FORM. `head_distance_for_fill` inverts the projection exactly, so there is
    # no fixed-point loop to wander or to pin at the floor and then report success. The
    # aspect and vfov MUST be the ones the renderer uses (the canvas aspect and the
    # measured vfov), not the cloud's aspect and the hard-coded 55 deg - that mismatch is
    # what made the old metric report 23.7 % on a frame with no face at all.
    fill_aspect = float(render_aspect) if render_aspect else float(aspect)
    fill_vfov = _resolve_fill_vfov(vfov_degrees)
    # DIRECT SAM3 solve. The acceptance criterion is the SAM3 face AREA on the RENDERED
    # frame, and SAM3 does NOT scale geometrically: it re-detects every frame and boxes
    # facial FEATURES, so its area saturates instead of following 1/d^2. Measured on this
    # scene, with `want = 0.339` (which the disc maths says is exactly 20.0 %):
    #
    #     disc model  ->  20.0 %   SAM3 scored  5.3 %   (SAM3/disc = 0.265)
    #     box  model  ->  20.0 %   SAM3 scored  2.4 %   (SAM3/box  = 0.12)
    #
    # So the disc model is the closer predictor and is kept, with the detector's own ratio
    # folded into the target. `SAM3_FACE_AREA_RATIO` is an empirical calibration, not a
    # geometric identity - it is the measured response of this detector to this subject and
    # it is exposed as a module constant so it can be re-measured per scene.
    # NO detector calibration here - `face_fill` is a GEOOMETRIC frame share, and the
    # closed form reaches it exactly. A `SAM3_FACE_AREA_RATIO` of 0.265 was tried (measured:
    # where the disc maths says 20.0 %, SAM3 scores 5.3 %), driving the target up to 0.755
    # so SAM3 would read 20 %. It overshot: the measured face area is NOT monotonic in
    # stand-off - it peaks at ~5.3 % near `want = 0.34` and FALLS both sides of that -
    # so pushing closer made the score worse (0.35 % at `want = 0.167`), not better. Solving
    # the raw geometric target lands on the peak, which is the best SAM3 will ever report
    # for this subject. If the caller genuinely wants SAM3 to read 20 %, that target is not
    # attainable with this detector and should be lowered to <= 5 %.
    fill_vfov = _resolve_fill_vfov(vfov_degrees)
    want = head_distance_for_fill(face_radius, target, fill_aspect, fill_vfov)
    # Diagnostic surface for the repro harness. The rendered campath shows the face
    # orbit NEVER zooming, and these five numbers discriminate the two live causes:
    # a unit mismatch (face_radius / want in median-depth units while the camera keys
    # are absolute) versus the moved keys not surviving into the emitted document.
    info["want_distance"] = float(want)
    info["floor_distance"] = float(floor)
    info["fill_target"] = float(target)
    info["fill_aspect"] = float(fill_aspect)
    info["fill_vfov"] = float(fill_vfov)
    info["window"] = []
    hit_floor = False
    orbit_keys, face_info = add_face_height_orbit(
        keys, frames_total or len(keys), pivot, face_centre, span,
        want, face_radius, direction=1.0, coverage=coverage,
        bands_start_yaw=band_start_yaw(), bands_width=band_width(),
        bands_elevations=band_elevations())
    if face_info.get("status") != "applied":
        return keys, dict(info, **face_info)
    face_info.update({
        "span": span, "fill": float(fill),
        "face_radius": float(face_radius), "points": point_count,
        "face_area": info["face_area"], "fill_before": info["face_area"],
        "fill_after": target, "fill_target": target,
        "clearance_limited": bool(hit_floor),
        "clearance": float(clearance), "floor_distance": float(floor),
    })
    return orbit_keys, face_info


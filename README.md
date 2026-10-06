# Enndees Nodepack

Custom nodes for **ComfyUI**: one-shot **Lichtfeld / 3DGS dataset creation** with
global Structure-from-Motion, plus a video frame extractor that also delivers the
matching audio.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-informational)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-informational)

---

## Why this pack

Most camera-tracking nodes expect you to install COLMAP (and GLOMAP) by hand and
to type in absolute paths. This pack is **all-in-one**: the tested COLMAP and
GLOMAP builds are downloaded automatically into the pack's own `bin/` folder -
either by ComfyUI-Manager running `install.py` or lazily on the very first run.
Nothing is added to the system `PATH`, nothing has to be configured manually.

Combined with the integrated RMBG background removal, the `GLOMAP Lichtfeld
Tracker (Enndee)` node turns a frame sequence into a complete, ready-to-train
Lichtfeld Studio dataset in a single step.

---

## Nodes

| Node | ID | Purpose |
|------|----|---------|
| **GLOMAP Lichtfeld Tracker (Enndee)** | `Enndee_GLOMAPLichtfeldTracker` | Global SfM camera tracking + Lichtfeld dataset export |
| **COLMAP for Lichtfeld (Enndee)** | `Enndee_ColmapLichtfeldTracker` | The same tracker through COLMAP's **native Python API** (pycolmap) - GLOMAP is part of COLMAP >= 3.12, so no GLOMAP binary is needed; installs/repairs `pycolmap` + `onnxruntime-gpu` on demand and, on Windows (no CUDA pycolmap wheel), downloads the CUDA COLMAP build so the SIFT stages still run on the GPU |
| **Video Frame Extractor + Audio (Enndee)** | `Enndee_VideoFrameExtractorWithAudio` | Frame/audio extraction with an in-node timeline widget |
| **MiniMax H3 Direct Promptor (Enndee)** | `H3_Multimodal_Promptor_Enndee` | Official-format MiniMax H3 prompts from reference images (vision LLM) |
| **Resolution Selector (Enndee)** | `Enndee_ResolutionSelector` | Aspect-ratio + megapixel sizing plus the nine core resize types and a resized image output |
| **Load & Resize Image (Enndee)** | `Enndee_ImageLoaderResize` | Load an image with the classic load-and-resize widgets, core resize types, mask channel, and original-size output |
| **Meridian Parameters and Camera (Enndee)** | `Enndee_MeridianParametersAndCamera` | Meridian geometry arguments plus the camera path: hand-authored O orbits / alternating-height pendulum / spiral sweeps, or an automatic mode that estimates the subject's (or scene's) geometric pivot from the still's depth profile and flies a speed-capped, collision-guarded path around it (subject: almost a full circle; scene: lateral survey rows for side coverage) |
| **Meridian Geometry (Enndee)** | `Enndee_MeridianGeometry` | Run VGGT geometry preview; optionally repeat the first frame to a connected custom path's required length |
| **Meridian Camera Path LLM (Enndee)** | `Enndee_MeridianCameraPathLLM` | Author a custom camera path with a local vision LLM (LM Studio / Ollama): describe the move in plain language, the LLM sees the still + depth map and outputs an orbit trajectory, and the node renders it into a `MERIDIAN_CAMERA_PATH` signal using the real pivot/framing geometry |
| **Lichtfeld Headless Trainer (Enndee)** | `Enndee_LichtfeldHeadlessTrainer` | Start configurable Lichtfeld Studio Gaussian-splat training from a tracker dataset and export the result as .ply, .sog or .spz |
| **Standby On Signal (Enndee)** | `Enndee_StandbyOnSignal` | Puts the PC into S3 standby when the workflow reaches the node and the ComfyUI queue is empty (last queued prompt) |
| **Sharpness Analyzer (Enndee)** | `Enndee_SharpnessAnalyzer` | Laplacian-variance sharpness score for every frame of an IMAGE batch |
| **Sharp Frame Selector Top-N (Enndee)** | `Enndee_SharpFrameSelector` | Reduce an IMAGE batch to its sharpest frames - top N per chunk (`batched_topn`, e.g. 3 of every 4), one per chunk, or the global top N |
| **Meridian Prompt Composer (conditional pictures)** | `MeridianPromptComposer` | The Meridian example workflow's conditional per-picture prompt blocks |

---

## Node: Standby On Signal (Enndee)

Puts the PC into **S3 standby** once the workflow reaches the node - the
overnight-batch helper. It is the ComfyUI side of the `Wakeup_from_Sleep`
toolkit (`Standby_Timer.ps1` / `Standby_Guard.ps1`) and uses the identical
mechanism (`SetSuspendState`, `bHibernate = FALSE`), so it always sleeps -
even when Windows' own idle standby is blocked by something.

### Inputs (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | BOOLEAN | false | Safety switch - only True really triggers standby |
| `mode` | COMBO | `Standby (S3)` | `Standby (S3)` suspends the PC, `DryRun (log only)` just prints |
| `delay_seconds` | INT | 30 | Grace period before sleeping (0-3600 s); abortable with the ComfyUI Cancel button |
| `check_queue` | BOOLEAN | true | Sleep only when the ComfyUI queue is empty - with many queued prompts only the last one sleeps |
| `server_url` | STRING | `http://127.0.0.1:8188` | ComfyUI API base URL for the queue check |

### Inputs (optional) / Outputs

| Name | Type | Description |
|------|------|-------------|
| `signal` | ANY | optional passthrough signal; connect the last node of your graph so the standby node runs after it |

As an output node it is executed even without a connected input - simply place
it at the end of the graph.

### Notes

- While `Standby_Guard.ps1` runs, start ComfyUI through `Keep-AwakeDuring.ps1`
  (or pause the guard): ComfyUI holds no power requests, so the idle guard
  would otherwise sleep mid-generation. This node then sleeps at the end.
- After the wake, ComfyUI keeps running; new prompts are picked up again.
- The queue is re-checked after the grace period; an unreachable server means
  "stay awake" (safe default).

---

## Node: Meridian Parameters and Camera (Enndee)

**Meridian Parameters and Camera (Enndee)** replaces the old parameter picker and
the standalone camera-path configurator. It emits both halves of a Meridian
geometry pass: connect its `args` output to **Meridian Geometry**'s
`args_override` input and its `custom_camera` output to that node's
`custom_camera` input. The path always matches `Output Frames` (73, 90, 107,
124, 141, 158, 175 or 243), and the node deliberately does not expose `--video`,
`--out` or `--preview-only` - the Geometry node supplies those itself.

**Camera Mode** decides where the path comes from:

- **Manual** - the hand-authored paths below (O orbits, alternating-height
  pendulum, monotone spiral sweep), built by the same code the retired
  configurator used, around the absolute look-pivot.
- **Automatic** - the node looks at the connected `reference_image` (and the
  optional `subject_mask`), estimates the subject's - or the scene's -
  **geometric pivot** from its depth profile and then flies either the
  estimated path or the manual one around that pivot. Nothing else needs to be
  set.

### Manual path

The path widgets follow the selected **Camera Mode** style:

- **O Orbits**: closed vertical O loops at the selected stations (Front, Left,
  Right, Back, Up, Down, LeftBack, RightBack) with a start-station choice. The
  two rear stations sit exactly 120 degrees from the front direction and from
  each other, so `Front + LeftBack + RightBack` is a balanced tripod route.
  Selected loops run in the builder's own sweep order and stop at the final
  selection instead of wrapping to the front. **Orbit Diameter** (median-depth
  units) sets the loop size.
- **Alternating Height**: the camera sweeps its **azimuth (yaw)** from **Start
  Yaw** to **Target Yaw** while the **elevation** alternates between the **Low
  Arc** and **High Arc** - a pendulum that eases out at every apex. Yaw is in
  degrees: 0 = in front of the subject (toward the source camera, i.e. between
  it and the pivot), +90 = the subject's right, 180 = behind it; values accept
  up to +/-360 and wrap (270 and -90 point the same way). Elevation is the
  altitude angle: 0 = level with the pivot, positive = above it looking down,
  negative = below looking up - capped at +/-70 like the Up/Down O stations,
  because a fully vertical look direction is gimbal lock for Meridian's
  zero-roll orientation. **Arc Switches** sets how often the camera flips sides
  (odd = ends on the opposite arc, even = returns to the starting one) and
  **Start Arc** picks the first arc.
- **Spiral Sweep**: the coverage/artefact-optimised sweep. Azimuth and elevation
  both advance monotonically (Start Yaw -> Target Yaw, Spiral Start Elevation ->
  Spiral End Elevation), so the camera never re-covers a band, follows the
  shortest route across the whole envelope and keeps a nearly constant angular
  speed - less velocity means fewer synthesis artefacts, and every frame shows
  new surface. The elevations may rise or fall freely.

All styles orbit the shared **Pivot** (`x`, `y`, `z` in frame-0 camera
coordinates) and honour **Path Dolly** (median-depth units): positive values
shift the whole path along the view axis to zoom out, negative to push in.

For the most surface at the lowest camera speed prefer **Spiral Sweep** with a
long sweep (e.g. Start Yaw 0 -> Target Yaw 270), a rising elevation (-20 -> 45),
a slightly wider framing (**Path Dolly** 0.2-0.5) and a large frame count. The
pendulum zig-zags across the same envelope, so it needs roughly 1.4-2x the
travel length for the same covered surface - and that extra travel is exactly
the per-frame velocity that creates artefacts. Keep the end at least one field
of view (~60-90 degrees) away from the start so the slightly drifted front
surface never reappears at the end. Watch the key-frame budget: each O loop
needs 8 key intervals and each transfer 4, so 73/90-frame paths fit up to 6/7
stations while the full eight-station sweep needs 107 frames or more; the node
raises a clear error when a selection does not fit.
### Automatic camera

Select **Automatic** and connect the still to `reference_image` - the same
image the Geometry node receives. The estimator:

1. runs a depth pass on the still (Depth-Anything-3-Small by default; it reuses
   the fast-depth backend's loaded model, so it costs one extra forward pass
   per queue);
2. places the **pivot** in the middle of the target's depth profile - the
   *geometric* midpoint of its robust bounding box (per-axis 2 %/98 %
   percentiles), not the picture centre and not the median, which would hug the
   dense front face. The **target** decides which geometry is used:
   - **subject** - the connected `subject_mask`, else the near-depth layer that
     an Otsu split separates from the background. Any segmentation node can
     drive the mask (the GLOMAP tracker's `use_rmbg` output works well); without
     one the depth split does the job;
   - **scene** - the whole reconstructed surface, whose pivot sits in the middle
     of all depth points. Its path is a **lateral survey**, deliberately *not* a
     360-degree lap: rows of viewpoints are spread across the scene's own width,
     and each row sweeps the elevation from -12 up to +28 degrees, so every
     lateral station is seen from below, level and above. This is the drone
     mapping pattern (boustrophedon flight lines with 70-80 % side overlap)
     mapped onto Meridian's reprojection camera, because what a walk-in VR viewer
     needs is *side* parallax - a depth reprojection cannot invent the far side of
     a scene, so a surround would only spend frames on invented surface. The row
     spacing follows the frame's footprint at the survey distance (55 deg vertical
     FOV and the still's aspect decide it); three rows (left/centre/right) are the
     minimum, nine the maximum, and a short frame or speed budget buys *fewer rows
     with the full sweep* instead of many cramped ones. A scene wider than the rows
     reach is what **Auto Orbit Distance** is for - the console line says how far
     they currently reach ("fly higher for a bigger area", as the mapping guides
     put it).
     The survey also **solves its own stand-off** (`scene_survey_fill`), because a
     fixed fill cannot fit every scene: the camera is pulled out until the outermost
     row really reaches the scene's edge (`radius >= half_width / sin(yaw limit)` -
     coverage is the hard requirement) and pulled in until the frame's footprint is
     about a third of the scene's width, so the frames are not mostly empty - never
     closer than one content radius. The console line reports the solved ratio, the
     reach ("reaching 100 % of the scene's half width") and the footprint. **Auto
     Orbit Distance** still overrides the solve.
     The **subject inside the scene** is measured too: the estimator keeps the same
     near-layer/mask layer the subject target would use and reports how big it
     appears along the survey ("the subject inside it covers 6.4 % of the picture in
     24 of 73 frames (18 whole)"). A lateral survey sweeps past a subject, so the
     useful question is not "is it always in frame" but "does a row show it well":
     the console warns when the subject covers less than 2 % of the picture ("a
     lower Auto Orbit Distance brings the rows closer, a smaller Auto Orbit Size
     narrows them onto the subject") or when no frame shows it at all. Connect a
     **MASK** to make that number mean the person rather than the near depth layer.
   **Auto Pivot X/Y/Z** shifts the estimated pivot by up to one content radius
   per axis (frame-0 camera axes: +X right, +Y down, +Z away) - for subjects
   whose depth midpoint is not the point you want framed.
3. picks the path with **Auto Path Mode**:
   - **Automatic** - the estimated path. For a subject: first the front
     **O orbit** (azimuth and elevation swings starting at +/-62 / +/-30 degrees
     and *growing* towards +/-85 / +/-60 as far as the pixel budget allows). It
     opens on the framed front pose - frame 0 is the view the fill solve targets -
     rises to the top of the O (12 o'clock), swings around past 9 and 6 to
     3 o'clock (the front from the middle, above, the left, below and the right in
     one move) and hands over to the **concluding orbit** there. **Auto Orbit
     Direction** picks the way: `counter-clockwise` (the default) turns 12 -> 9 ->
     6 -> 3 o'clock, `clockwise` mirrors the whole path to 12 -> 3 -> 6 -> 9.
     **Auto Orbit End** says where the orbit stops, in degrees measured from the
     start - the *middle of the O*: `360` (the default) is the complete round, back
     at the start point, its height arc rising to +38 degrees at the back and
     easing back to 0 so the last frame sits on the first one and the clip loops;
     `180` stops behind the subject, `270` on its other side, `0` flies the front O
     alone (up to `720`, two rounds). **The max camera speed is a hard limit**, and
     **Auto Orbit Size is the floor for the front**: the fit grows the O above it
     when the budget allows (a wider O keeps the front's angle variety *and*
     shortens the orbit), and when the budget cannot pay for the requested end at
     that width, the *orbit* is cut - the coverage gives way, never the front O and
     never the speed cap. The console names the levers ("the max camera speed ends
     the orbit at 265 deg of the requested 360 deg - more Output Frames, a higher
     Auto Max Speed or a shorter Auto Orbit End balance the round"; and only if the
     cap cannot even pay for the O itself: "the cap could not even pay for Auto
     Orbit Size 1.00x, so the front O had to shrink below it to 0.42x"). The frames
     are split between O and orbit so that both run at the same drift
     (`balanced_front_share`) - the cheapest way to honour the cap, because whatever
     one half gives up the other half uses. For a scene: the **lateral survey** of
     rows described at `scene` above (no surround).
     **Auto Orbit Coverage** (`auto_orbit_coverage`) picks *how much* of the subject
     that visit covers:
     - `Front only` - the front O alone, the cheapest visit; all of it runs at the
       Auto Subject Fill distance and the framing is solved for exactly that path.
     - `Front and Back` (the default) - the front O, the shortest level connection
       (180 deg minus the O's own diameter, stopping at the far orbit's near edge,
       its "9 o'clock") and that far orbit's clockwise loop 9 -> 12 -> 3 -> 6 -> 8
       plus a glide in to the far dial's centre: both sides without a full lap.
       When the budget cannot pay for the back part, *it* is cut - the front O keeps
       the radius the fit found (the console names the cut).
     - `Spiral` - the **O-orbit family** flown as a spherical spiral around the pivot.
       The camera travels on a sphere whose centre is the pivot (constant distance,
       always aiming at it, so the subject keeps its place in the frame) along the
       same circles the **Front-only** coverage flies, their angle growing along the
       path (the spiral is drawn in the picture's y-z plane and projected onto the
       sphere, so x - the depth - is the sphere's own):
       * `phi` - the O-orbit's angular radius, i.e. the arc between the camera and the
         spiral's axis - grows **evenly along the path** from **0 at the first frame**
         (the camera sits *on* the axis, looking straight at the picture: the middle of
         the frame) to the **Spiral End Angle** at the last frame. That end angle is the
         Auto Orbit Angle slot while this coverage is picked (`spiral_end_arc`, degrees,
         default **90**): at 90 the camera ends *in the picture's own plane* - the clip
         finishes with a **side view** of the picture; a smaller value stops the spiral
         earlier, on a fatter O-orbit,
       * `psi` - the clock angle around that axis (12 = up, 3 = the subject's right,
         6 = down, 9 = left) - winds from 0 to the **Spiral Winding** widget
         (`spiral_end`, degrees, default **840** = two and a third rounds), in Auto
         Orbit Direction (counter-clockwise by default).
       The winding alone decides *where* the last frame looks from: with the default
       840 deg it is `cos 840 deg = -0.5`, i.e. a side view 30 degrees *below* the
       pivot; **810 deg** ends level on the side (`cos 810 deg = 0`), and 0 flies the
       plain quarter circle (front view -> straight up). The console line and the path
       description name the end pose (clock/elevation), so the number can be dialled in.
       **The axis itself leans with the `Spiral Center Slope`** - its own widget: the
       angle of the spiral's central rotational axis in the vertical plane through the
       view axis, **0 (the default) horizontal** = the view axis itself (the spiral opens
       on the framed frontal view), **+90 "from straight above"** (it opens straight above
       the subject and unwinds down to a level orbit), **-90 "from straight below"**.
       Everything else - pivot, distance, aim - is untouched.
       The frames follow the parameter itself - the O-orbit angle growing evenly along
       the path, exactly "O-orbits with the angle growing" - and the **keys** the
       Geometry node splines follow the path's own *turning* rather than every Nth frame
       (dense right after the pole, sparse on the long outer sweep), so the path reads
       smooth in every view. **Auto Max Speed never reshapes the spiral** (the winding
       and the end angle are your parameters, not the fit's): the console line reports
       the per-frame drift against the cap instead, so more Output Frames or a higher
       Auto Max Speed are the levers.
       **Camera roll: none.** The camera keeps one distance and always aims at the pivot,
       and its orientation is the **level-horizon (world-up) look-at**: `right =
       cross(forward, world-up)`, so the horizon stays level and there is **no roll
       about the optical (x) axis** at any pose - the frame does not accumulate the
       holonomy a spiralling path would otherwise twist into it. The opening frame is
       the plain world-up zero-roll look-at, so a clip starts exactly as it always has.
       The one pose without a horizon is the **pole** (the look straight up/down the
       world axis): there the previous frame is held (parallel transport) and walked
       back to level over the next 24 frames, so a coil that grazes the top re-locks
       smoothly instead of flipping ~180 deg. This is the renderer's own basis
       (`camera_frames`/`_evaluate_camera_path` in the fast-depth backend), and the
       auto-camera's measurements use the same level-horizon basis, so a pixel
       measured is a pixel rendered.
   - **Manual** - the manual styles below, but aimed at the estimated pivot
     instead of the absolute look-pivot (the `path_pivot_*` widgets are ignored
     and hidden then; every other path widget applies as usual).
4. fits the path to **Max Speed** (`auto_max_speed`, percent of the content
   radius per frame, 12 % default): the path is built at full amplitude, the
   true per-frame travel is measured and every amplitude is scaled down the
   ladder until it fits - the console line reports the factor ("0.28x
   amplitude"). Slow and steady beats fast: too much new surface per frame is
   what makes the depth reprojections smear.
   **Auto Orbit Distance** (`auto_orbit_distance`, in content radii) sets how far
   the camera flies from the estimated pivot: 2.2 is the built-in stand-off (2.0
   for a scene), lower values push in for more parallax and a fuller frame, higher
   values pull back for a wider, calmer view. **Auto Orbit Size**
   (`auto_orbit_size`) multiplies the path's swing amplitudes: 1.0 is the built-in
   O orbit described above, 0.5 a tight loop that only grazes the front, 2.0 a far
   sweep. Both shape the *automatic* path only - the manual path keeps its own pivot
   and dolly - and the speed fit and the collision guard below still have the last
   word.
   **Auto Subject Fill** (`auto_subject_fill`, default 40 %) is the framing target: the
   node solves the camera distance so the subject's projected bounding box covers that
   share of the picture *area*, and re-centres the pivot on the subject's projected
   midpoint (the perspective residual a plain 3D midpoint would leave). 30-50 % is the
   sweet spot for depth reprojections; `0` switches the framing off and Auto Orbit
   Distance applies again. **Auto Orbit Distance** (`auto_orbit_distance`, in content
   radii) overrides the framing with a fixed stand-off.
   **The whole subject stays in the picture**: the fill solve already reserves the room for the
   whole path - the camera is pulled back until *every* pose of the O *and* the concluding orbit
   keeps the subject inside the frame with a 2 % border (`_room_distance`, and the fill target is
   solved against that safe area rather than the bare frame). Room is never bought by cutting the
   orbit any more: the stand-off (with it the fill) pays instead - the console names the pull-back.
   On top of that the estimator checks the subject's own pixels
   (0.2 %/99.8 % of its projected points) against every single frame of the fitted path and repairs
   any leftover: first the swing is **shortened** (free - the front framing stays, and the orbit
   keeps its own end), then the pivot is **re-aimed** at the pose with the least room, and only
   if that is still not enough is the camera **pulled back** in bisected steps. What comes out is
   *as much as the geometry and the speed cap allow*: the O at whatever the drift and the room
   permit, the orbit out to its end or as far as the budget pays for, the fill whatever the room
   leaves - all reported, never hidden. The only budget that may cut the *coverage* is the *speed*
   cap (the user's own parameter), and the console says so with the numbers
   ("the max camera speed ends the orbit at 265 deg of the requested 360 deg - more Output Frames,
   a higher Auto Max Speed or a shorter Auto Orbit End balance the round"). A subject that fills
   40 % of the frame at ~1.5 radii simply cannot be circled at +/-62 deg within 12 %/frame: that is
   physics, not a setting. Remedies, in the order that helps most: more `output_frames` or a higher
   **Auto Max Speed**, a shorter **Auto Orbit End** (the freed budget goes to the front O), a
   **MASK input** (e.g. the installed RMBG-2.0 node) that hugs the subject, a lower **Auto Subject
   Fill** (further out = room for a wider loop), a smaller **Auto Orbit Size**. The QC plot carries
   the per-frame border clearance with the 0/25/50/75/100 % marks and the worst frame.
5. the **speed cap** (`auto_max_speed`) is measured in *pixels* for a subject: the
   node projects the subject's own surface points along the path and keeps the
   per-frame drift of the 95th percentile inside that percentage of the subject's
   apparent radius - the background is never part of the budget, so a distant
   background cannot slow the subject down. The front **O orbit grows toward its view
   limits** (+/-85 deg azimuth, +/-60 deg elevation) as far as that budget allows, so
   the front side is covered as far round as the frame count permits; a tight budget
   shrinks the loop instead and the console line says so ("raise Auto Max Speed or
   Output Frames for a bigger O"). **Auto Orbit Size** multiplies the amplitudes if
   you want to override the fit. The saved path is written in **median-depth units**
   (1.0 = the scene's median depth), exactly like the manual Camera Path Configurator,
   because the Geometry node multiplies the keys by that median before rendering.
6. runs the **collision guard**: every path key is checked against the whole
   scene cloud and pushed away from the pivot until it is at least 15 % of the
   content radius clear of the nearest point, so the camera never ends up inside
   the geometry it orbits (the console line reports how many keys were moved).

Every estimate prints a one-line summary to the ComfyUI console (pivot, depth
extents, source label, path style, units per frame, amplitude factor, collision
fixes). The `subject_mask` input is optional; the summary says which source was
used.

### What this node no longer has

VGGT is not part of this pack any more: the Geometry node runs the in-process
fast-depth backend only, so the VGGT/canvas/source-size options, the
freeze/start/follow window, the authored camera move (yaw, truck, boom, dolly,
zoom, pivot, aim), diagnostics, seed and the camera-path JSON widget are gone.
Use **Camera Mode** instead - manual path or automatic estimate. The retired
**Meridian Parameter Picker** and **Meridian Camera Path Configurator** nodes
were replaced by this one; re-add it in workflows that still reference them.

### Meridian Geometry IMAGE input

The `image` socket on **Meridian Geometry (Enndee)** accepts one still or a
multi-frame ComfyUI IMAGE batch; the fast-depth backend renders from the first
frame and repeats it for every flight frame, so a batch only decides which
picture is used. A `video` path works the same way when no image is connected:
the node decodes its first frame. Repeated stills do not add observed backside
geometry - the back-facing views are depth reprojections - so inspect the render
for holes or stretching before feeding it to H3. Restart ComfyUI after updating
the node pack to register the new node.

## Node: Meridian Camera Path LLM (Enndee)

Author a **custom camera path** with a **local vision LLM** (LM Studio or Ollama). You describe the
move in plain language ("rotate 180 around the subject, lift up a bit, and rotate back toward the
starting side"), and a vision LLM - shown the still and its depth map - turns it into an orbit
trajectory. The node then renders that plan into a valid `MERIDIAN_CAMERA_PATH` signal using the real
geometry (the subject's pivot + framing orbit radius from the depth profile, in the same median-depth
units the auto-camera emits), so its `custom_camera` output plugs straight into **Meridian Geometry**.

The LLM does the creative work (reading the scene, translating the instruction); the node does the
precise 3D placement. The LLM never emits raw xyz - it emits **orbit-space keyframes** (`azimuth` /
`elevation` / `distance` around the subject), which are robust to scene scale and always keep the
subject framed and aimed at.

### Inputs (required)
- **instruction** - the camera move in plain language.
- **frames** - output frame count (73/90/107/124/141/158/175/243; match the sampler).
- **provider** - `lmstudio` (OpenAI-compatible, default `http://localhost:1234/v1`) or `ollama`
  (`http://localhost:11434`).
- **base_url** / **model** / **api_key** - the local server endpoint and a loaded **vision** model
  (e.g. `qwen2.5-vl`, `llava`, `gemma3`). `api_key` is optional (Ollama ignores it).
- **fill_percent** - how large the subject is framed at `distance` 1.0; the plan's `distance`
  multiplies this framing radius.
- **temperature** / **max_tokens** - sampling.

### Inputs (optional)
- **image** - the still (RGB); sent to the LLM and used for geometry.
- **depth** - the depth map of the still (grayscale); sent to the LLM and used to place the camera.
- **subject_mask** - optional subject mask (white = subject) for a more accurate pivot.
- **depth_convention** - how to read the depth image (`brighter = closer` / `brighter = farther`).

### Outputs
- **custom_camera** - the `MERIDIAN_CAMERA_PATH` signal → Meridian Geometry's `custom_camera`.
- **plan** - the sanitized JSON keyframe plan the LLM produced (for inspection / editing).
- **raw** - the raw model reply (debugging).
- **system_prompt** - the full system prompt used (inspect or copy it).

### The system prompt
The node embeds a complete, reusable system prompt (also exposed on the `system_prompt` output) that
teaches the LLM: its role as a virtual cinematographer; the orbit-space camera model (azimuth /
elevation / distance, aim at the pivot, **no roll / no flip**); hard constraints (elevation ±70° for
gimbal safety, distance 0.5–3×, exact frame count, key spacing, continuous motion); how to read the
depth map (prefer orbits that reveal dimensionality, avoid empty rear orbits on flat backgrounds);
how to translate verbs ("orbit N degrees", "lift up", "push in", "rotate back"); a strict JSON schema;
and a worked example. The per-run bits (instruction, frame count, depth convention, the two images)
ride in the user turn.


## Node: Lichtfeld Headless Trainer (Enndee)

The trainer runs the installed LichtFeld Studio CLI with its supported
`--headless --train` options. It does not automate the GUI, patch Studio, or
install a separate CUDA/Torch stack. Training output is streamed to the ComfyUI
console and saved to a log file.

### Wiring

```text
GLOMAP Lichtfeld Tracker (Enndee): dataset_path
                    -> Lichtfeld Headless Trainer (Enndee): dataset_path
```

The tracker's `STRING` output `dataset_path` points to its `images/`, `masks/`,
`masks_GLOMAP/`, and `sparse/0/` files. This output was added *after* the
existing trajectory, point-cloud, and confidence sockets so previous workflows
retain their output connections. You may also enter a dataset path manually.

### Training, masks, and outputs

- **Studio executable:** detects `LICHTFELD_STUDIO_EXE`, a PATH entry, or the
  installed Studio on this machine; enter the executable path if necessary.
- **Iterations, Strategy, SH Degree, Max Gaussians, Steps Scaler:** set the
  corresponding Lichtfeld CLI training options. `max_gaussians` is the training
  Gaussian capacity; it is distinct from the tracker's SIFT `max_features`.
- **Centralize Dataset:** optionally center the scene origin `by_pointcloud` or
  `by_cameras`; `off` preserves the exported coordinates.
- **Image Resize Factor / Max Image Width:** configure Studio's loader (`auto`,
  1×/2×/4×/8× and an optional width cap). **Disable Downscaling** forces factor
  1 and removes the width cap (equivalent to unlimited width); expect more VRAM
  and compute use at full resolution.
- **Grow Until Iter / Stop Refine:** optional explicit refinement cutoffs. Zero
  leaves Lichtfeld's selected strategy/config defaults untouched. These are
  applied through Studio's supported `--python-script` iteration-start callback.
- **Save Steps / Eval Steps:** optional comma-separated iteration lists (e.g.
  `5000,10000,20000`). Empty Save Steps preserves Studio/config defaults. Eval
  Steps automatically enable evaluation; if evaluation is enabled but Eval Steps
  are empty, evaluations mirror Save Steps.
- **Output Name:** optionally overrides Studio's default `splat_ITER` filename.
- **Export Format:** the splat file left in the output folder. `ply` (default) is Studio's
  own training result. `sog` (SuperSplat) and `spz` (Niantic) additionally run Studio's
  own `convert` subcommand on the finished `.ply` right after training, so the compressed
  file appears next to it (the `.ply` stays in the folder and can be re-exported any time).
  The node picks the finished splat itself: an explicit Output Name wins, otherwise the
  highest iteration number - checkpoint saves never win over the final model. A build
  without the `convert` subcommand keeps the `.ply` and logs a warning instead of failing.
- **Config File:** optionally supplies a Lichtfeld Studio JSON config for
  additional settings supported by your installed Studio build. The selected
  file must exist and contain valid JSON.
- **Mask Mode / Invert Masks:** configure Lichtfeld's segmentation handling.
  The tracker's `masks/` convention is **white = keep the subject**;
  `masks_GLOMAP/` is a separate feature-extraction mask. The default mode is
  `segment`; inspect the masks and training previews and toggle **Invert Masks**
  only if the foreground/background polarity is reversed.
- **Background Mode / Color:** defaults to a **solid white (`#FFFFFF`)**
  training background. This changes Lichtfeld's training background, not the
  exported image pixels, and does not compensate for a bad mask.
- **Mip, Bilateral Grid, Evaluation, Sparsity, Log Level:** control optional
  settings accepted by the installed Lichtfeld CLI.
- **Output Path:** an empty value **or the dataset root itself** creates a
  fresh timestamped folder inside `<dataset>/output/`. This accommodates
  workflows which link the tracker dataset output to both the trainer's
  `dataset_path` and `output_path` sockets. A connected Comfy socket takes
  precedence over that input's text widget. Relative custom output paths resolve
  below the dataset. The node saves a separate timestamped log alongside the
  training outputs. Non-empty output folders are protected unless **Overwrite
  Output** is explicitly enabled.
- **Preview Only:** validates the image folder, all three COLMAP sparse files,
  Studio executable, and training arguments, then displays the command without
  launching Studio or creating output files. Turn this **off** to start a real
  training run.
- **Allow Concurrent Studio Process:** by default the node refuses to launch
  while a Lichtfeld Studio GUI is open. It never closes or modifies existing
  sessions. Enabling this option explicitly allows GPU contention from separate
  Studio instances.

Training is a synchronous Comfy job: the queue item runs until Lichtfeld Studio
finishes. The node outputs `output_path`, `command`, `log_file`, and `summary`;
a `.sog`/`.spz` export adds its converted path to the summary as a `Splat:` line
and streams the converter's own progress to the console.

### Older and free Studio builds

The node asks the executable itself what it supports - `--version`, `--help`, and
`convert --help` when the build advertises that subcommand - and adapts the run instead of
trusting a version number. Older free builds print "unknown" for `--version` and reject
unknown flags outright (`Error: Parse error: Flag could not be matched: bg-mode`), so
capability probing is the only reliable trigger. Read-only and cached per executable.

- Flags the build does not know (`--bg-mode`, `--bg-color`, `--centralize`, `--output-name`,
  ...) are left out and listed in the node summary; the build's own defaults apply.
- `sog`/`spz` without a `convert` subcommand fall back to the `.ply` export, with a warning in
  the console and in the summary. (LichtFeld Studio 0.5.0 already ships `convert` including
  SOG/SPZ - the fallback is for builds that predate it.)
- Strategy and mask mode are checked against the build's own list (an older build may only
  offer `mcmc`/`adc`/`igs+`). An unsupported choice raises with that list; a log level the
  build does not know falls back to `info` with a note.
- A build without `--python-script` logs that Grow Until / Stop Refine / Save Steps /
  Eval Steps are ignored for this run.

### Tracker features vs. trained splats

The GLOMAP tracker's `max_features` controls the maximum number of SIFT keypoints
extracted per image for matching and camera/point-cloud reconstruction. It does
not cap or directly select the number of trained Gaussians. Fewer features can
indirectly lead to fewer or less stable SfM points and weaker camera estimates,
which can affect Lichtfeld's initialization and final learned splat count. The
trainer's **Max Gaussians** (`--max-cap`) is the explicit Gaussian ceiling; the
chosen strategy and its pruning/growth behavior, masks, view coverage, and
training settings determine the actual count below that ceiling.

### Temporally stable source-frame cleanup

For noisy rendered/video frames that should keep a consistent, smooth texture,
use a video denoiser on the **source frames before reconstruction**, then feed
that cleaned sequence into the tracker. Avoid frame-independent image denoisers
or aggressive sharpening: they can make grain/detail change from frame to frame
and put artificial high-frequency texture into the point cloud. Preview a short
representative clip first and compare frames side-by-side; keep the original
sequence in case the denoiser waxes out thin geometry, hair, and fabric.

Options:

- **Topaz Video Denoise / Nyx High-Fidelity:** Topaz's Denoise filter is explicitly
  for removing grain without additional enhancement and has no tuning controls.
  Topaz cautions that standard Nyx variants may also sharpen/modify source content;
  Nyx XL suppresses more noise but trades away texture retention. This is a good
  quality-first external preprocessing choice when paid software is acceptable.
- **ComfyUI / SeedVR2 already installed here:** SeedVR2 is a temporal video
  upscaler/restoration model, not a dedicated denoiser. Its node docs require
  `batch_size >= 5` (4n+1 values) for its temporal consistency path; temporal
  overlap helps continuity across chunks. For a softer A/B, try a *small*
  `latent_noise_scale` such as 0.05–0.10 (the node documents it as softening
  excessive detail). It injects diffusion noise; it is not guaranteed to remove
  noise and may hallucinate/shift detail, so compare a short test before tracker
  export. The installed VideoHelperSuite handles video I/O, not denoising. I did
  not find a dedicated temporal denoising-only node/model already installed.
- **Frame interpolation is not denoising:** interpolation can synthesize smoother
  motion, but it does not consistently remove noise and may introduce motion
  artifacts. Do not add interpolated frames solely to improve splatting.

If staying fully local/free is important, evaluate SeedVR2 conservatively on a
short clip before adding another model stack; for strict denoising, prefer a
motion-aligned video denoiser that explicitly targets grain without generating
replacement texture. For strongly “plasticky” style, gentle temporally stable
denoising plus controlled desaturation/low-frequency cleanup is safer than forcing
a generative restyle, which can hallucinate different detail and break geometry
consistency.

---

## Node: GLOMAP Lichtfeld Tracker (Enndee)

Runs the complete pipeline and writes a Lichtfeld Studio dataset. COLMAP and
GLOMAP command output is streamed to the ComfyUI console while each stage runs;
the node also reports image export progress and sends a heartbeat every 30
seconds if an external command is quiet.

```text
<lichtfeld_export_path>/
|-- images/           0001.png or 0001.jpeg ... (full resolution)
|-- masks/            Lichtfeld / splat masks   (white = keep)
|-- masks_GLOMAP/     feature extraction masks  (white = excluded)
`-- sparse/0/         cameras.txt, images.txt, points3D.txt
```

### Inputs (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `colmap_path` | STRING | auto | COLMAP.bat / colmap.exe. Empty = auto detect / auto install |
| `glomap_path` | STRING | auto | glomap.exe (unused with `mapper_backend=colmap_global`) |
| `camera_model` | COMBO | `SIMPLE_PINHOLE` | intrinsics model (`PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`) |
| `matcher` | COMBO | `sequential` | `sequential` (video) or `exhaustive` (unordered images) |
| `max_features` | INT | 24000 | SIFT features per image |
| `images_path` | STRING | - | image folder, used when no `images` input is connected |
| `masks_path` | STRING | - | GLOMAP mask folder, used when no `masks_glomap` input |
| `lichtfeld_export_path` | STRING | - | export target; relative paths go below ComfyUI's `output/` |

### Inputs (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | - | frame batch (preferred source) |
| `masks_glomap` | MASK | - | masks for feature extraction (**white = excluded**) |
| `masks_lichtfeld` | MASK | - | splat masks for Lichtfeld (**white = keep**) |
| `use_rmbg` | BOOLEAN | true | built-in RMBG background removal |
| `rmbg_mode` | COMBO | `base` | `base`, `fast`, `base-nightly` |
| `rmbg_threshold` | FLOAT | 0.5 | segmentation sensitivity |
| `rmbg_resize` | COMBO | `static` | `static` or `dynamic` (up to 1280 px) |
| `use_gpu` | BOOLEAN | true | GPU for SIFT/RMBG |
| `keep_workspace` | BOOLEAN | false | keep the temporary COLMAP workspace |
| `auto_align` | BOOLEAN | true | align to the ground plane (Y up, floor at Y=0) |
| `sequential_overlap` | INT | 15 | neighbouring frames compared by the matcher |
| `max_image_size` | INT | 5120 | longest edge for feature extraction (export stays full res) |
| `frame_step` | INT | 2 | use every n-th frame |
| `downscale_factor` | FLOAT | 1.0 | SfM input scale only |
| `offset_glomap` | INT | 4 | erode (+) / dilate (-) the GLOMAP masks in px |
| `offset_splat` | INT | 12 | erode (+) / dilate (-) the Lichtfeld masks in px |
| `mapper_backend` | COMBO | `glomap` | `glomap` (GLOMAP 1.2.0) or `colmap_global` (COLMAP >= 3.12) |
| `auto_install_binaries` | BOOLEAN | true | download missing binaries on demand |
| `binary_flavor` | COMBO | `auto` | `auto`, `cuda`, `nocuda` |
| `embed_alpha_in_images` | BOOLEAN | false | also store the RMBG alpha in `images/` (RGBA PNGs) |
| `image_format` | COMBO | `PNG` | Export `images/` as lossless PNG or JPEG |
| `jpeg_quality` | INT | 90 | JPEG quality (1-100); used only when image format is JPEG |

PNG is the default and retains image alpha. JPEG exports RGB `.jpg` files;
available alpha is retained in the separate Lichtfeld `masks/` export because
JPEG cannot contain an alpha channel. The JPEG quality setting affects only
`images/`; masks remain PNG. Re-running an export into an existing `images/`
folder removes old numbered image files first so mixed-format duplicates do not
enter the SfM dataset.

### Outputs

| Name | Type |
|------|------|
| `trajectory` | `CAMERA_TRAJECTORY` |
| `point_cloud` | `POINTCLOUD` |
| `confidence` | FLOAT |

### Mask conventions (important)

* `masks_glomap` / `masks_path`: **white (1) = region that is excluded from
  feature extraction** (typically the moving subject), black = features allowed.
  The masks are inverted internally because COLMAP expects white = valid.
* `masks_lichtfeld`: **white (1) = region to keep** in the splat. It is stored
  as-is in `masks/` (after `offset_splat`).
* RGBA images: the alpha channel is used automatically for `masks/` when no
  explicit Lichtfeld mask is connected.

### Recommended settings

* 24 fps 360 degree orbit of a subject: `matcher=sequential`,
  `sequential_overlap=15`, `frame_step=2`, `max_features=24000`,
  `max_image_size=5120`, `offset_glomap=+4`, `offset_splat=+12`.
* Unordered photo set: `matcher=exhaustive`, `frame_step=1`,
  `sequential_overlap` is ignored.
* Fast camera motion / shaky footage: increase `sequential_overlap` (20-30) and
  `max_features`, lower `frame_step` to 1.
* Very large images (> 4K): `downscale_factor=0.5` - the export stays full res.

---

## Node: COLMAP for Lichtfeld (Enndee)

The same tracker as above, but the Structure-from-Motion backend is **COLMAP's own
Python API** (`pycolmap`) instead of downloaded executables. GLOMAP was merged into
COLMAP (COLMAP >= 3.12 ships the global mapper), so `pycolmap.global_mapping()` *is*
the GLOMAP pipeline - no GLOMAP binary is needed, pinned or searched for.

| binary tracker | native tracker |
| --- | --- |
| `colmap feature_extractor` | `pycolmap.extract_features` |
| `colmap sequential_matcher` | `pycolmap.match_sequential` |
| `colmap exhaustive_matcher` | `pycolmap.match_exhaustive` |
| `colmap global_mapper` / `glomap mapper` | `pycolmap.global_mapping` |

With `use_gpu` (default) and a pycolmap build that has no CUDA - every Windows wheel -
the two **SIFT** stages are delegated to the downloaded CUDA COLMAP build instead, so
the GPU is used where it pays off while the mapper stays in-process (see
[Speed / GPU](#speed-the-gpu-bridge-on-windows)).

Everything else - RMBG background removal, masks, frame stepping, the complete
Lichtfeld Studio dataset export (`images/`, `masks/`, `sparse/` incl. the TXT model)
and the trajectory / point-cloud outputs - is shared verbatim with the binary node.

### Widgets

Identical to the GLOMAP tracker **except**:

* `colmap_path`, `glomap_path` and `binary_flavor` are gone - there is no binary to
  point at.
* `mapper_backend` is `global` (default) or `incremental` (COLMAP's classic mapper,
  slower but sometimes more forgiving); the old names `glomap` and `colmap_global`
  are still accepted.
* `auto_install_binaries` now means "install/repair what the node needs": the python
  accelerators (see below) **and** the CUDA COLMAP build that runs the SIFT stages on
  the GPU (~154 MB, once).

### Environment / accelerators

With `auto_install_binaries=True` (default) the node checks and, if needed, installs

* `pycolmap` - this node's backend, **the CUDA build whenever one is available** (see
  below), and
* `onnxruntime-gpu` - the CUDA ONNX runtime for the RMBG / ONNX nodes, matched to
  the CUDA version of your torch (CUDA 13 -> `>= 1.30`, CUDA 12 -> `1.19 .. 1.29`).
  A shadowing CPU wheel (`onnxruntime`) is removed, because it silently makes
  `get_available_providers()` lose `CUDAExecutionProvider`.

`python install.py` does the same at setup time (`--skip-accelerators` opts out), and
`ENNDEE_AUTO_DOWNLOAD=0` disables every automatic download/install.

### Speed: the GPU bridge on Windows

The expensive part of SfM is SIFT - feature extraction and matching. The binary tracker
downloads *CUDA* builds of COLMAP/GLOMAP (`colmap.exe` links `cudart64_12.dll`) and runs
those stages on the GPU. The official `pycolmap` wheel for **Windows** is compiled without
CUDA (`pycolmap.has_cuda == False`; the `pycolmap-cuda12` wheels exist for Linux and macOS
only - see `https://pypi.org/project/pycolmap-cuda12/`).

So the native node **borrows the GPU** where it matters: with `use_gpu=True` (default) it
downloads the pinned CUDA COLMAP build (the same one the binary tracker uses, ~154 MB, once,
into `<pack>/bin`) and runs `feature_extractor`, `sequential_matcher` and `exhaustive_matcher`
through it with `SiftExtraction.use_gpu 1` / `SiftMatching.use_gpu 1`. Both sides read and
write the *same* COLMAP database, and everything else stays native: the global mapper runs
in-process through `pycolmap.global_mapping`, the status label and progress bar stay live
(COLMAP's own `Processed file [n/m]` records feed the bar, so nothing is printed).

Measured on this machine (RTX 5090, 12 real 2048 px photos, ~10k features each, sequential
overlap 5, identical settings):

| stage | binary tracker (COLMAP CUDA) | native, no bridge (pycolmap CPU) | native + GPU bridge |
| --- | --- | --- | --- |
| feature extraction, 12 frames | 1.0 s | 4.8 s | **1.5 s** |
| feature matching, 33-50 pairs | 0.6 s | 22.1 s | **0.8 s** |

That is the difference between "121.8 s (native, CPU) vs 65.8 s (binary)" on a real 113
frame workflow and **native ≈ binary** with the bridge. The mapping stage itself is
CPU-bound in *both* nodes (Ceres without CUDA), which is why it is not bridged.

Controls:

* `use_gpu=False` - no download, no bridge, SIFT on the CPU.
* `ENNDEE_PYCOLMAP_GPU_BRIDGE=0` - same, without touching the workflow.
* `ENNDEE_AUTO_DOWNLOAD=0` / `auto_install_binaries=False` - never download; an already
  installed CUDA COLMAP build (or `ENNDEE_COLMAP_PATH`) is still used.
* a CUDA-enabled pycolmap build (self-built, or via `ENNDEE_PYCOLMAP_CUDA_WHEEL`) always
  wins over the bridge and runs everything in-process.
* the console says which one is active:

      COLMAP for Lichtfeld (Enndee) - native pycolmap backend
      pycolmap : 4.2.1 [cpu-fallback]
      gpu      : CUDA COLMAP bridge -> ...\Enndees_Nodepack\bin\colmap-3.11.1-cuda\COLMAP.bat
                 feature extraction + matching run on the GPU, the global mapper in-process

### Console output

COLMAP logs through glog, and at INFO level that is a *lot*: every SIFT thread setup,
every processed image, every pairing step. A 113 frame run printed **2455** such lines
around the 4 warnings that actually mattered. The node therefore runs COLMAP at
**WARNING** level:

* warnings and errors still appear (missing focal priors, real failures) - the two
  "Requested to use GPU for bundle adjustment, but COLMAP was compiled without CUDA
  support" lines are gone, because a CPU-only build is now *told* not to ask for the
  GPU solvers (they only repeated what `pycolmap.has_cuda` already says - see the
  speed section below),
* the progress chatter is gone - the node's own status label and progress bar carry that
  information now,
* `ENNDEE_COLMAP_VERBOSE=1` restores the full output for debugging.

The binary tracker passes the same setting to the COLMAP/GLOMAP executables, so
`glomap mapper` is quiet too (verified: `COLMAP feature_extractor` prints 3 INFO lines
without it and 0 with it).

The one warning that stays is worth decoding:

    W... global_pipeline.cc:62] Less than 50% of cameras have prior focal lengths.
    The global mapper depends on reasonably good focal length priors to perform well. ...

It comes from the global mapper (`pycolmap`/COLMAP >= 3.12 - the GLOMAP 1.2.0 binary
does not even contain that text) and it is a **quality hint, not an error**: frames that
are generated rather than photographed carry no EXIF focal length, so COLMAP seeds the
camera with its default guess (`f = 1.2 * max(width, height)`) and does not mark it as a
*prior*. The mapper then estimates the focal length itself - which is exactly what the
bundle adjustment is for. Nothing to fix unless you know the real focal length (then a
calibrated `camera_params` removes the warning and makes the reconstruction better).

The chunked feature extraction imports the frames **once** and pins that camera for every
chunk: `extract_features` imports what it is handed, and COLMAP's "single camera" mode is
per call - so a chunked run used to produce one camera per chunk (113 frames came out
with 13 cameras, each with its own intrinsics block, and the global mapper warned about
missing focal priors). The Lichtfeld export now always contains a single camera.

### CUDA first, the GPU bridge, CPU only as the last resort

The node always tries to run on the GPU and only then falls back - and it **shows
which one it is using** (see below). The order is:

1. an installed CUDA pycolmap (`pycolmap.has_cuda`) - everything in-process,
2. `ENNDEE_PYCOLMAP_CUDA_WHEEL=<path|url>` - a wheel you built or downloaded yourself,
3. `pycolmap-cuda12` - when pip finds a wheel for this platform (Linux/macOS so far),
4. **the GPU bridge** - the downloaded CUDA COLMAP build runs extraction + matching
   (`SiftExtraction.use_gpu 1`), the mapper stays in-process (see the speed section).

The CPU build is only kept when there is no CUDA device/driver, no CUDA wheel for
this platform *and* no bridge (switched off, or no download allowed and nothing
installed). The verdict is cached per session (the first check costs about a second,
later ones are free).

Reality check: the official **Windows** wheels of `pycolmap` have no CUDA and the
CUDA wheels are Linux/macOS only - so on Windows step 4 is the normal case and the
SIFT work still lands on the GPU. A CUDA build can be created from source (COLMAP +
vcpkg + CUDA SDK) and then either installed normally or dropped in via
`ENNDEE_PYCOLMAP_CUDA_WHEEL`; it takes over from the bridge automatically.

### Live status and progress

* **Status label** - a read-only text area on the node
  (`web/js/enndee_colmap_status.js`) that is updated while the node runs: the
  environment check (backend, CUDA device, torch, onnxruntime, attention
  accelerators) plus the current stage. The final summary also goes through
  ComfyUI's built-in `ui.text` output.
* **Progress bar** - `comfy.utils.ProgressBar`: 0-90 % for the reconstruction and the
  rest for the dataset export and the model parsing. Both paths feed it: the pycolmap
  CPU/CUDA path reports per chunk of frames / batch of image pairs (the pairs come from
  COLMAP's own pair generator, so chunking does not change the result), and the GPU
  bridge parses COLMAP's own `Processed file [n/m]` / `Matching block [n/m]` records.

While it runs the label looks like this:

    COLMAP for Lichtfeld (Enndee) - native pycolmap backend
    pycolmap : 4.2.1 [cpu-fallback]
    gpu      : CUDA COLMAP bridge -> ...\Enndees_Nodepack\bin\colmap-3.11.1-cuda\COLMAP.bat
               feature extraction + matching run on the GPU, the global mapper in-process
    torch    : 2.14.1+cu130 (CUDA 13.0, NVIDIA GeForce RTX 5090)
    onnx     : 1.30.0 [Tensorrt, CUDA, CPU]
    attention: flash_attn=yes, sageattention=yes
    status   : feature extraction 42/113 images

---

## Node: Resolution Selector (Enndee)

A compact rebuild of ComfyUI's built-in **Resolution Selector** (category
`utilities`) - the same three sizing controls, plus a visible *keep source aspect
ratio* toggle and an optional `image` socket.

Give it an **aspect ratio**, a **megapixel target** and a **multiple**, and it
returns the matching `width` and `height`. The node shows a live preview of the
result (`2048 × 2048   4.00 MP`) that updates while you drag the sliders.

New in this pack:

* **`image` input** - connect a source image.
* **`keep_source_aspect_ratio`** - when enabled, use the connected image's ratio
  and scale up or down to the megapixel target. Without a connected image, the
  node falls back to the selected preset and logs a note.
* **`resize_type`** - the nine resize types of ComfyUI core's *Resize Image/Mask*
  node (`scale dimensions`, `scale by multiplier`, `scale longer dimension`,
  `scale shorter dimension`, `scale width`, `scale height`, `scale total pixels`,
  `match size`, `scale to multiple`). `scale dimensions` resizes the connected
  image to the selected width/height; the other types use their own widget.
* **`image` output** - the connected image resized by the chosen resize type
  (`None` while no image is connected).

The calculation itself is the same short formula: `scale = sqrt(megapixels *
1024^2 / (w_ratio * h_ratio))`, then both axes are rounded to the nearest
`multiple` (8 = valid latent size, 32/64 for some models). In image-ratio mode the
preview shows the megapixel target; the final dimensions are available from the
output sockets. The resize math is shared with the *Load & Resize Image* node
(`nodes/enndee_resize_modes.py`) and mirrors the core node exactly.

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aspect_ratio` | COMBO | `1:1 (Square)` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9` |
| `megapixels` | FLOAT | 4.0 | target total megapixels (0.1 - 16.0) |
| `multiple` | INT | 32 | snap the result to this multiple (8 - 128) |
| `keep_source_aspect_ratio` | BOOLEAN | false | visible mode toggle: take the ratio from the image |
| `resize_type` | COMBO | `scale dimensions` | one of the nine core resize types (see above) |
| `multiplier` | FLOAT | 1.0 | `scale by multiplier`: 2.0 doubles, 0.5 halves |
| `longer_size` | INT | 512 | `scale longer dimension`: target of the longer edge |
| `shorter_size` | INT | 512 | `scale shorter dimension`: target of the shorter edge |
| `crop` | COMBO | `center` | `scale dimensions`/`match size`: `center` crops, `disabled` stretches |
| `scale_method` | COMBO | `area` | `nearest-exact`, `bilinear`, `area`, `bicubic`, `lanczos` |
| `image` | IMAGE | - | optional source image for keep-source-aspect-ratio mode and the `image` output |
| `match` | IMAGE | - | reference image for the `match size` resize type |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `width` | INT | selected width, or the resized image width when an image is connected |
| `height` | INT | selected height, or the resized image height when an image is connected |
| `image` | IMAGE | the connected image resized by `resize_type` (`None` without an image) |

### Examples

| Input | Settings | Result |
|-------|----------|--------|
| preset | `1:1 (Square)`, 4.0 MP, multiple 32 | 2048x2048 |
| preset | `16:9 (Widescreen)`, 1.0 MP, multiple 8 | 1368x768 |
| image 1920x1080 | `keep_source_aspect_ratio`, 2.0 MP, multiple 8 | 1928x1088 |
| image 1080x1350 | `keep_source_aspect_ratio`, 1.0 MP, multiple 8 | 912x1144 |
| image 640x480 | `keep_source_aspect_ratio`, 4.0 MP, multiple 64 | 2368x1792 |
| image 200x100 | `scale by multiplier`, multiplier 1.5 | `image` 300x150, width/height 300/150 |

> `multiple` snapping can shift the exact aspect ratio by a few pixels - that is
> intentional, it keeps the size valid for latent models. Use `multiple=8` for
> the smallest deviation.

---

## Node: Load & Resize Image (Enndee)

An image loader with the classic load-and-resize widget set (file combo with
**choose file to upload**, `resize` toggle, width/height, repeat,
keep_proportion, divisible_by, mask_channel, background_color), extended with:

* the nine **resize types** of ComfyUI core's *Resize Image/Mask* node (shared
  math in `nodes/enndee_resize_modes.py`), including a `match` input for
  `match size`;
* an **`original_image`** output - the file exactly as loaded, never resized;
* a `scale_method` combo for the interpolation quality.

Only the options that matter are visible: while `resize` is off, the resize
widgets collapse; with `resize` on, the node shows the parameters of the
selected `resize_type` plus `scale_method` (`background_color` appears when
`keep_proportion` pads, and `width`/`height` return for `match size` until a
`match` image is connected). Hidden widgets keep their values for the workflow
and the python node.

With `no_upscale` enabled the image is never enlarged: when the computed target
is bigger than the loaded image, the content keeps its source size (downscaling
and the `keep_proportion` padding still work, so the target canvas can still be
filled).

The node reads from ComfyUI's `input/` folder and also accepts annotated paths
(`file.png [input]`). `width`/`height` report the final size of the `image`
socket (the source size with `resize` off). `keep_proportion` pads with
`background_color` (`#rrggbb`, `#rgb` or a color name) instead of stretching, and
`divisible_by` rounds the final size down to a multiple.

### Inputs (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | COMBO | - | file from the input folder; upload button included |
| `resize` | BOOLEAN | false | resize the loaded image; the original stays on its socket |
| `resize_type` | COMBO | `scale dimensions` | one of the nine core resize types |
| `width` / `height` | INT | 512 | `scale dimensions` (+ `scale width`/`scale height`) target; 0 derives the other side |
| `repeat` | INT | 1 | repeat the output in the batch (still-image video helpers) |
| `keep_proportion` | BOOLEAN | true | fit + pad with `background_color` instead of stretching (explicit-target types) |
| `divisible_by` | INT | 2 | round the final size down to this multiple |
| `mask_channel` | COMBO | `alpha` | `alpha` (1 - alpha, like LoadImage), `red`, `green`, `blue` |
| `background_color` | STRING | `#000000` | fill color for the `keep_proportion` padding |
| `multiplier` | FLOAT | 1.0 | `scale by multiplier` factor |
| `longer_size` / `shorter_size` | INT | 512 | `scale longer/shorter dimension` targets |
| `megapixels` | FLOAT | 1.0 | `scale total pixels` target (1.0 ~ 1024x1024). Slider with a FLOAT input slot - drop a primitive float signal onto the parameter to drive it (the slot's input format is FLOAT) |
| `multiple` | INT | 8 | `scale to multiple` step (cover-resize + center crop) |
| `scale_method` | COMBO | `lanczos` | `nearest-exact`, `bilinear`, `area`, `bicubic`, `lanczos` |
| `no_upscale` | BOOLEAN | false | never enlarge: larger targets keep the source size (padding still fills the canvas) |

### Inputs (optional)

| Parameter | Type | Description |
|-----------|------|-------------|
| `match` | IMAGE | reference image for the `match size` resize type |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | loaded and (optionally) resized image |
| `original_image` | IMAGE | loaded image at its original file size |
| `mask` | MASK | mask from `mask_channel`, resized with the image |
| `width` / `height` | INT | final image size |
| `image_path` | STRING | resolved path of the loaded file |
| `megapixels` | FLOAT | echo of the effective `megapixels` value (socket → legacy `megapixels_in` → default 1.0, clamped 0.01 - 16.0) - chain it into other nodes |

Animated images (WebP/GIF) load every same-size frame; `IS_CHANGED` hashes the
file so edits re-trigger the graph.

---

## Node: MiniMax H3 Direct Promptor (Enndee)

Generates **official-format MiniMax H3 prompts** directly from up to 8 reference
images (+ optional video keyframes) and your description - in a single vision-LLM
call (OpenAI, Ollama, Gemini or Claude). Vendored here from the standalone
`ComfyUI-MiniMax-H3-Promptor-Enndee` (GPL-3.0, fork of
[1038lab/ComfyUI-Minimax-H3-Promptor](https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor)).

### Inputs (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | COMBO | T2V | Forces the H3 format: T2V, I2V, I2VA, V2V, V2VA, A2V, FL2VA, Ref2VA |
| `description` | STRING | - | your creative scene description |
| `duration` | FLOAT | 5.0 | video length in seconds (4-15) |

### Inputs (optional)

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_ref_1..8` | IMAGE | up to 8 reference images (mapped to `<Picture N>` labels) |
| `video_ref` | IMAGE | video batch - keyframes are extracted and mapped to `<Video 1>` |
| `output_language` | COMBO | `English` / `Chinese` |
| `provider` | COMBO | `openai`, `ollama`, `gemini`, `claude` |
| `api_key` | STRING | per-run override (config.json stays untouched) |
| `model_name` | COMBO | selected vision model (Ollama: lists your installed models) |
| `temperature` / `top_k` / `top_p` / `min_p` / `repeat_penalty` | - | sampling options (Ollama sent on every call) |
| `max_tokens` | INT | 4096 | LLM output budget (256-16384) |

### Output

| Name | Type |
|------|------|
| `prompt` | STRING |

### Configuration & models

* The selected model must be **vision-capable** (`gpt-4o`, Claude, Qwen-VL, Gemma
  vision, ...).
* On first use the node auto-creates `minimax_h3_promptor/config.json` from
  `config.example.json` - open it and add your API keys. The file is
  **git-ignored** and never shipped; you can also pass `api_key` per node run.
* Output is capped at MiniMax's official **7,000 character** limit
  (`max_tokens` is the generating LLM's budget, not H3's limit).

---

## Node: Sharpness Analyzer + Sharp Frame Selector Top-N (Enndee)

A self-contained pair for **filtering the sharpest frames** out of an IMAGE
batch, based on the MIT-licensed
[ComfyUI-Sharp-Selector](https://github.com/ethanfel/ComfyUI-Sharp-Selector)
duo. Every frame is scored with the **Laplacian variance** (higher = sharper),
then the selector reduces the batch.

The customized selector adds a **`batched_topn`** mode the original node does
not have: it keeps the **top `num_frames` frames of every `batch_size` chunk** -
e.g. the **3 sharpest of every 4 frames** of a 73-243 frame clip - so a long
clip keeps its sharp frames evenly distributed instead of only its global best.

### Node: Sharpness Analyzer (Enndee)

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | IMAGE | frame batch to score (one Laplacian-variance score per frame) |

Output: `scores` (`SHARPNESS_SCORES`) - feed into the selector.

### Node: Sharp Frame Selector Top-N (Enndee)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | - | the full frame batch to reduce |
| `scores` | SHARPNESS_SCORES | - | scores from a Sharpness Analyzer (Enndee or ComfyUI-Sharp-Selector) |
| `selection_method` | COMBO | `batched_topn` | `batched_topn` = top `num_frames` of every `batch_size` chunk; `batched` = single sharpest per chunk (original); `best_n` = global top N |
| `batch_size` | INT | 24 | chunk length in frames (use **4** to group into batches of four) |
| `batch_buffer` | INT | 0 | frames skipped between chunks (stride = `batch_size + batch_buffer`); keep **0** for gap-free coverage |
| `num_frames` | INT | 10 | frames kept **per chunk** in `batched_topn` (e.g. **3**) and globally in `best_n`; ignored by `batched` |
| `min_sharpness` | FLOAT | 0.0 | drop frames scoring below this value (0.0 keeps everything) |

Outputs: `selected_images` (IMAGE) - the reduced batch; `count` (INT) - how many
frames were kept.

**Recipe - "3 sharpest of every 4 frames"**: `selection_method=batched_topn`,
`batch_size=4`, `batch_buffer=0`, `num_frames=3`, `min_sharpness=0.0`. A
73-frame clip yields **55** frames, a 243-frame clip yields **183** (about 75 %,
evenly distributed).

The `SHARPNESS_SCORES` type is shared with ComfyUI-Sharp-Selector, so the
Enndee and the original analyzer/selector are interchangeable.

---

## Global: save without the running counter

ComfyUI appends a running number to **every** saved file
(`ComfyUI_00001_.png`, `MiniMax_H3_00001_.mp4`, `Preview_00025.mp4`, ...).
There is no core option to disable this - each save node formats the counter
unconditionally (`folder_paths.get_save_image_path` only computes it).

This pack therefore ships a small global hook (`nodes/enndee_unique_filenames.py`,
loaded automatically, no node appears in the UI): after a save finishes the
file is renamed so the number only stays when the plain name is already taken:

| written by ComfyUI | ends up on disk |
|---|---|
| `MyClip_20261002_143022_00001_.png` | `MyClip_20261002_143022.png` |
| ... saved again with the same name | `MyClip_20261002_143022_1.png` |

- Covered: core **Save Image**, **Save Latent**, **Save Video**, **Save WEBM**
  and **VHS Video Combine** (including `-audio` sidecars and poster frames).
  The UI previews keep working because the result entries are updated in
  place; batch saves number from the second file on (`X.png`, `X_1.png`, ...).
- Not touched: temp previews (`PreviewImage`), output files whose name has no
  5-digit counter block.
- Opt out: set the environment variable `ENNDEE_KEEP_FILE_COUNTER=1` before
  starting ComfyUI.

Works perfectly together with `DateTimeToString` prefixes: a per-second unique
prefix means the number never shows up at all.

## License

* This pack's own code: **MIT** (see `LICENSE`).
* The copied `Enndee_ResolutionSelector` node is based on BRADSEC's MIT licensed
  [ComfyUI_ResolutionSelector](https://github.com/BRADSEC/ComfyUI_ResolutionSelector)
  (MIT, Copyright (c) 2023 BRADSEC) - MIT is compatible with this pack's license.
* The `Sharpness Analyzer (Enndee)` / `Sharp Frame Selector Top-N (Enndee)`
  nodes are based on ethanfel's MIT licensed
  [ComfyUI-Sharp-Selector](https://github.com/ethanfel/ComfyUI-Sharp-Selector) -
  MIT is compatible with this pack's license.
* The vendored `minimax_h3_promptor/` component: **GPL-3.0** (see
  `minimax_h3_promptor/LICENSE`) - a fork of the
  [1038lab/ComfyUI-Minimax-H3-Promptor](https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor)
  project.
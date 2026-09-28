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
| **Video Frame Extractor + Audio (Enndee)** | `Enndee_VideoFrameExtractorWithAudio` | Frame/audio extraction with an in-node timeline widget |
| **MiniMax H3 Direct Promptor (Enndee)** | `H3_Multimodal_Promptor_Enndee` | Official-format MiniMax H3 prompts from reference images (vision LLM) |
| **Resolution Selector (Enndee)** | `Enndee_ResolutionSelector` | Aspect-ratio + megapixel sizing plus the nine core resize types and a resized image output |
| **Load & Resize Image (Enndee)** | `Enndee_ImageLoaderResize` | Load an image with the classic load-and-resize widgets, core resize types, mask channel, and original-size output |
| **Meridian Parameters and Camera (Enndee)** | `Enndee_MeridianParametersAndCamera` | Meridian geometry arguments plus the camera path: hand-authored O orbits / alternating-height pendulum / spiral sweeps, or an automatic mode that estimates the subject's (or scene's) geometric pivot from the still's depth profile and flies a speed-capped, collision-guarded path around it |
| **Meridian Geometry (Enndee)** | `Enndee_MeridianGeometry` | Run VGGT geometry preview; optionally repeat the first frame to a connected custom path's required length |
| **Lichtfeld Headless Trainer (Enndee)** | `Enndee_LichtfeldHeadlessTrainer` | Start configurable Lichtfeld Studio Gaussian-splat training from a tracker dataset and export the result as .ply, .sog or .spz |

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
     of all depth points and whose path is wider.
   **Auto Pivot X/Y/Z** shifts the estimated pivot by up to one content radius
   per axis (frame-0 camera axes: +X right, +Y down, +Z away) - for subjects
   whose depth midpoint is not the point you want framed.
3. picks the path with **Auto Path Mode**:
   - **Automatic** - the estimated path. For a subject: first a big front
     **O orbit** (azimuth +/-62 degrees, elevation +/-30) that shows the front
     from below, right, above and left in one loop, then a 270-degree **height
     lap** around the rest of the subject that eases from -30 up to +38 degrees
     elevation, so the last frames show new surface at a new height instead of
     repeating the start. For a scene: one big **oval** - a 350-degree lap at
     twice the scene radius that rises from -12 to +28 degrees while it goes
     round.
   - **Manual** - the manual styles below, but aimed at the estimated pivot
     instead of the absolute look-pivot (the `path_pivot_*` widgets are ignored
     and hidden then; every other path widget applies as usual).
4. fits the path to **Max Speed** (`auto_max_speed`, percent of the content
   radius per frame, 12 % default): the path is built at full amplitude, the
   true per-frame travel is measured and every amplitude is scaled down the
   ladder until it fits - the console line reports the factor ("0.28x
   amplitude"). Slow and steady beats fast: too much new surface per frame is
   what makes the depth reprojections smear.
5. runs the **collision guard**: every path key is checked against the whole
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
| `megapixels` | FLOAT | 1.0 | `scale total pixels` target (1.0 ~ 1024x1024) |
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

## License

* This pack's own code: **MIT** (see `LICENSE`).
* The copied `Enndee_ResolutionSelector` node is based on BRADSEC's MIT licensed
  [ComfyUI_ResolutionSelector](https://github.com/BRADSEC/ComfyUI_ResolutionSelector)
  (MIT, Copyright (c) 2023 BRADSEC) - MIT is compatible with this pack's license.
* The vendored `minimax_h3_promptor/` component: **GPL-3.0** (see
  `minimax_h3_promptor/LICENSE`) - a fork of the
  [1038lab/ComfyUI-Minimax-H3-Promptor](https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor)
  project.
# Enndees Nodepack: project context

This repository is the Enndees custom-node pack for ComfyUI, installed in the
portable workspace under `ComfyUI/custom_nodes/Enndees_Nodepack`. It provides
camera/reconstruction, image and video utilities, face-detail mask helpers,
sharp-frame selection, and integrations that export datasets for Lichtfeld
Studio and other reconstruction/training backends. The package is Python-based;
ComfyUI frontend extensions live under `web/js/`, and automated tests are in
`tests/` (standard-library `unittest`, with PyTorch used by tensor logic).

## COLMAP/GLOMAP tracker architecture

The supported COLMAP workflow is the binary **GLOMAP Lichtfeld Tracker
(Enndee)** node (`Enndee_GLOMAPLichtfeldTracker`), registered by `__init__.py`.
It invokes the bundled/downloadable COLMAP and GLOMAP executables. The previous
pycolmap node implementation and tests remain in `nodes/colmap_lichtfeld_node.py`
and `tests/test_colmap_lichtfeld_node.py` for migration reference, but that node
is intentionally not registered. Existing workflows using the pycolmap node
must be migrated to the binary tracker.

The tracker can derive feature masks from explicit GLOMAP masks, RMBG, RGBA
alpha, or Lichtfeld masks. GLOMAP masks use white as excluded; Lichtfeld splat
masks use white as retained. Explicit GLOMAP masks take precedence over
automatically derived subject masks. Sharp-frame selection is integrated into
the tracker so it can keep image, alpha, and mask batches aligned and remap
`face_frames` from original input indices to the selected frame batch. The
`sharp_selection.json` sidecar records retained/dropped original indices and
the final mapping after `frame_step`. Dependent widget visibility and display
labels are implemented in `web/js/enndee_colmap_widgets.js`.

## Validation and operational notes

Run the package tests from the portable workspace with its embedded interpreter:

```powershell
& 'D:\ComfyUI_Windows_portable\ComfyUI_windows_portable\python_embeded\python.exe' `
  'D:\ComfyUI_Windows_portable\ComfyUI_windows_portable\_run_tests.py'
```

The runner writes a UTF-8 report to `_probe_out/tests.txt` and must return a
nonzero status when tests fail. Focused module reports are written by
`_run_focused_tests.py`. Some renderer tests may be sensitive to the installed
graphics/PyTorch environment; report the exact test names and traceback rather
than attributing those failures to unrelated tracker code.

Full-suite verification before the latest Meridian UI schema cleanup (2026-10-14): 790 tests passed.
The current focused Parameters, Geometry, and face-detail suites pass 31/31, 32/32, and 55/55;
the full suite has not yet been rerun. Meridian point splats intentionally keep their uncovered
pixels visible because `point_size` and back-face culling are point-renderer controls;
mesh mode alone applies the
hole-inpainting post-process. Applying that fill to point mode hides point-size
coverage and culling results.

ComfyUI loads node registrations and the JavaScript extension at startup. After
changing `__init__.py` or `web/js/enndee_colmap_widgets.js`, restart ComfyUI to
observe registration and frontend-widget changes. The tracker feature tests are
unit tests; they do not replace visual validation against the face-detail
comparison datasets.

## Ground-truth extrinsics pipeline (2026-10-15)

Two nodes export Lichtfeld datasets with the camera rig the Meridian render flew
instead of solving SfM (added for camera-path-driven videos where the path IS the
pose source):

- `Enndee_MeridianGeometryExtrinsics` (`nodes/enndee_meridian_geometry_extrinsics.py`)
  - a copy of Meridian Geometry with a sixth output `extrinsics`
    (`MERIDIAN_EXTRINSICS`, JSON): per-frame c2w [F,4,4] (columns right, down,
    forward - the renderer's OpenCV frame, `zm` gauge), focal multipliers [F],
    `vfov_degrees`, canvas size and frame count. The rig is published by
    `render_depth_aligned` through `fast_depth.LAST_CAMERA_EVAL` (same one-shot
    pattern as `LAST_VFOV_DEGREES`), so the signal always matches the render.
- `Enndee_PathPoseDataset` (`nodes/enndee_path_pose_dataset.py`, display
  "Path Pose Dataset - no COLMAP (Enndee)") - the GLOMAP tracker minus SfM. It
  reuses the tracker's helpers (sharp selection, frame_step, Lichtfeld masks from
  explicit masks/RGBA alpha/RMBG, image export) and assigns each exported frame
  the extrinsics of its ORIGINAL input index (`final_output_to_input`, recorded
  in `sharp_selection.json` alongside the pose mapping). Writes the tracker's
  dataset layout with `sparse/0/{cameras,images,points3D}.txt`; `points3D.txt` is
  empty (no SfM ran), `images.txt` carries the ground-truth poses (COLMAP
  world-to-camera, `c2w_to_w2c`, qvec via a local `rotmat_to_qvec`).
  Frame mapping (`extrinsics_mapping`): 'auto' uniform resample (identity when
  counts match), 'manual' index+offset for chunked clips. Focal (`focal_mode`):
  'vfov from signal' (the renderer's model, `0.5*H/tan(vFOV/2)` at export res),
  'manual focal px', 'manual hfov' - use the manual modes when the generated
  video's lens differs from the planning vFOV.

Tests: `tests/test_enndee_path_pose_dataset.py` (18 tests, no binaries/models:
signal round trip, qvec round trip through COLMAPParser, mapping math, the
geometry node's appended output with a stubbed render, end-to-end dataset export
including sharp-selection pose assignment and `validate_dataset`). Known caveat
(from the 20261014__23_43_33 comparison): diffusion-generated videos only
approximate the planned path (~5-7 deg rotation deviation measured), so path
poses are exact for warp-rendered flights but an approximation for AI videos -
the `focal_mode` manual options and the GLOMAP tracker remain the fallback.
QC helper: `tools/compare_path_vs_colmap.py <dataset_folder>` measures the
planned-path-vs-SfM deviation for a clip (Umeyama fit + per-frame errors).
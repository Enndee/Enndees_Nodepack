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

Recent full-suite verification (2026-10-14): 787 tests passed. Meridian point
splats intentionally keep their uncovered pixels visible because `point_size`
and back-face culling are point-renderer controls; mesh mode alone applies the
hole-inpainting post-process. Applying that fill to point mode hides point-size
coverage and culling results.

ComfyUI loads node registrations and the JavaScript extension at startup. After
changing `__init__.py` or `web/js/enndee_colmap_widgets.js`, restart ComfyUI to
observe registration and frontend-widget changes. The tracker feature tests are
unit tests; they do not replace visual validation against the face-detail
comparison datasets.
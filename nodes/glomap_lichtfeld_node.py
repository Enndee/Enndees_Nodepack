"""
GLOMAP Lichtfeld Tracker (Enndee) - global Structure-from-Motion camera tracking
and one-shot Lichtfeld / 3DGS dataset export for ComfyUI.
================================================================================

The node takes a frame sequence (connected ``IMAGE`` batch or a folder on disk),
optionally removes the background, runs COLMAP (SIFT) + GLOMAP (global mapper)
and writes a complete, Lichtfeld-Studio ready dataset:

    <export>/
    ├── images/         0001.png ... or 0001.jpeg ...
    ├── masks/          Lichtfeld / splat masks (0001.png ...)
    ├── masks_GLOMAP/   masks used for feature extraction (0001.png ...)
    └── sparse/0/       cameras.txt, images.txt, points3D.txt

Key properties
--------------
* **All-in-one**: the tested COLMAP + GLOMAP builds are downloaded into
  ``<pack>/bin`` automatically (``install.py`` / ComfyUI-Manager or lazily on
  the first run).  See ``enndee_bin.py``.
* **Portable**: no hard coded developer paths.  Both path widgets may stay
  empty - resolution order is widget -> env var -> ``bin/enndee_binaries.json``
  -> pack ``bin/`` -> auto detection.
* **Two mask paths** (as before): ``masks_glomap`` only constrains feature
  extraction, ``masks_lichtfeld`` is the mask that Lichtfeld Studio uses for
  splatting.  RGBA input images contribute their alpha channel automatically.
* **Optional COLMAP >= 3.12 global mapper** (``mapper_backend``) because
  upstream GLOMAP is deprecated and frozen at 1.2.0.
* **Live console progress**: COLMAP/GLOMAP stdout and stderr are streamed to
  ComfyUI's console as each stage runs instead of being buffered until exit.
* **PNG/JPEG dataset export** with a user-selectable JPEG quality; masks stay
  lossless PNG and preserve alpha when JPEG is selected.

Widget order of the original node is preserved, all new options are appended at
the end so existing workflows keep their settings.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from the pack (vendored COLMAP/GLOMAP library + binary manager)
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

from enndee_bin import (  # noqa: E402  (path setup must run first)
    ensure_binaries,
    resolve_binary,
    source_of,
)
from enndee_rmbg import (  # noqa: E402  (path setup must run first)
    RMBG_DEFAULT_MODE,
    RMBG_MODES,
    RMBG_MODE_TOOLTIP,
    remove_background,
)

try:
    from enndee_colmap import (  # noqa: E402
        COLMAPParser,
        GLOMAPWrapper,
        fix_image_names_in_sparse,
    )
    _LIB_SOURCE = "vendored enndee_colmap"
except Exception:  # pragma: no cover - fallback for users with comfyui_colmap
    _fallback = _PACK_DIR.parent / "comfyui_colmap"
    if _fallback.is_dir() and str(_fallback) not in sys.path:
        sys.path.append(str(_fallback))
    from lib.colmap_parser import COLMAPParser  # type: ignore
    from lib.glomap_wrapper import GLOMAPWrapper, fix_image_names_in_sparse  # type: ignore
    _LIB_SOURCE = "external comfyui_colmap"

try:
    import folder_paths

    COMFY_OUTPUT_DIR: Optional[str] = folder_paths.get_output_directory()
except Exception:  # pragma: no cover - running outside ComfyUI
    folder_paths = None  # type: ignore
    COMFY_OUTPUT_DIR = None

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MASK_GLOMAP_DIR = "masks_GLOMAP"
MASK_LICHTFELD_DIR = "masks"
#: sub folder Lichtfeld scans for depth maps; files are ``<image stem>.depth.png``
#: (``depths/`` is accepted by Studio as well, but ``depth/`` is what we write).
#: Shared with the VGGT node so both nodes produce a byte-identical layout.
DEPTH_DIR = "depth"
SPLAT_SUBDIR = "Splat_Frames"

# Cached RMBG models live in :mod:`enndee_rmbg` (shared by every tracker node).


def log(message: str) -> None:
    """Single logging helper so the prefix stays consistent."""
    print(f"[Enndee] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[Enndee] WARNING: {message}", flush=True)


def auto_download_enabled() -> bool:
    """ENNDEE_AUTO_DOWNLOAD=0 disables the on demand binary download."""
    value = (os.environ.get("ENNDEE_AUTO_DOWNLOAD") or "").strip().lower()
    return value not in ("0", "false", "no", "off")


def tooltip(text: str) -> Dict[str, str]:
    """Small helper for readable INPUT_TYPES."""
    return {"tooltip": text}


#: One-time hint: the mapping is CPU-bound, so the widgets are the only lever.
_SLOW_MAPPING_HINT_SHOWN = False


def maybe_warn_slow_mapping(matcher: str, frame_step: int, max_features: int) -> None:
    """Warn *once* when the widget combination makes the CPU-bound mapping slow.

    The mapping (global positioning + bundle adjustment) runs on the CPU on every build:
    a CUDA pycolmap only accelerates SIFT extraction and matching (81 images: 8.85 s ->
    1.65 s and 47.42 s -> 0.89 s; the mapping was 87.6/70.2 s on the CPU wheel vs
    115.7/67.0 s on the CUDA wheel - noise). Exhaustive matching, every frame and ~24k
    features is the combination that costs the most, so it gets one hint per session.
    """
    global _SLOW_MAPPING_HINT_SHOWN
    if _SLOW_MAPPING_HINT_SHOWN:
        return
    if str(matcher).lower() != "exhaustive":
        return
    if int(frame_step) != 1 or int(max_features) < 24000:
        return
    _SLOW_MAPPING_HINT_SHOWN = True
    log_warn("exhaustive matching + frame_step=1 + "
             f"{int(max_features)} features makes the CPU-bound mapping (and the "
             "matching) slow - the fast path is matcher=sequential, frame_step=2 "
             "and about 10000 features")


def default_tool_path(kind: str) -> str:
    """
    Best known path for a tool, used as widget default.

    Never downloads anything - returns "" when nothing was found, so the user
    immediately sees that the node will fall back to auto install/detection.
    """
    try:
        found = resolve_binary(kind)
        return str(found) if found else ""
    except Exception:
        return ""


class GLOMAPLichtfeldTracker:
    """
    Global SfM camera tracking + Lichtfeld dataset export.

    Runs COLMAP feature extraction/matching and a global mapper (GLOMAP or
    COLMAP's own ``global_mapper``), then exports poses, point cloud and a
    ready-to-use Lichtfeld Studio dataset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "colmap_path": ("STRING", {
                    "default": default_tool_path("colmap"),
                    **tooltip(
                        "COLMAP.bat or colmap.exe used for SIFT features and "
                        "matching. Leave EMPTY for auto detection; if nothing "
                        "is found the tested COLMAP build is downloaded into "
                        "<pack>/bin automatically."
                    ),
                }),
                "glomap_path": ("STRING", {
                    "default": default_tool_path("glomap"),
                    **tooltip(
                        "glomap.exe used by mapper_backend='glomap'. Leave EMPTY "
                        "for auto detection / automatic download. Ignored when "
                        "mapper_backend='colmap_global'."
                    ),
                }),
                "camera_model": (["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL",
                                  "RADIAL", "OPENCV"], {
                    "default": "SIMPLE_PINHOLE",
                    **tooltip(
                        "Intrinsic camera model. SIMPLE_PINHOLE (f, cx, cy) is "
                        "ideal for AI generated / distortion free footage. "
                        "PINHOLE adds a second focal length, OPENCV models real "
                        "lens distortion."
                    ),
                }),
                "matcher": (["sequential", "exhaustive"], {
                    "default": "sequential",
                    **tooltip(
                        "sequential = match neighbouring frames (fast, perfect "
                        "for video orbits). exhaustive = every image against "
                        "every image (slow but robust for shuffled image sets). "
                        "This is the main lever for mapping time - the mapper "
                        "is CPU-bound, and exhaustive multiplies its input."
                    ),
                }),
                "max_features": ("INT", {
                    "default": 24000, "min": 1000, "max": 32768, "step": 1000,
                    **tooltip(
                        "SIFT features per image. More features = more stable "
                        "reconstruction but slower and more RAM. 24000 works "
                        "well for high resolution 360 deg orbits. This is the "
                        "main lever for mapping time - the mapper is CPU-bound "
                        "(about 10000 features is the fast path)."
                    ),
                }),
                "images_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Folder with the input images (JPG/PNG/RGBA). Only used "
                        "when no 'images' input is connected. May point at a "
                        "dataset root containing an 'images' sub folder."
                    ),
                }),
                "masks_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Folder with GLOMAP (feature) masks, same convention as "
                        "'masks_glomap': WHITE = region to exclude from feature "
                        "extraction (e.g. the moving subject), BLACK = features "
                        "allowed. Only used when no 'masks_glomap' input is "
                        "connected."
                    ),
                }),
                "lichtfeld_export_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Target folder for the complete Lichtfeld export "
                        "(images/, masks/, masks_GLOMAP/, sparse/0/). Relative "
                        "paths are resolved below the ComfyUI output folder. "
                        "Empty = <output>/Splat_Frames/<timestamp>."
                    ),
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks_glomap": ("MASK", {
                    **tooltip(
                        "Masks for feature extraction. WHITE (1) = region to "
                        "ignore (usually the moving subject), BLACK (0) = "
                        "features allowed. Gets inverted internally because "
                        "COLMAP expects white = valid."
                    ),
                }),
                "masks_lichtfeld": ("MASK", {
                    **tooltip(
                        "Splat masks for Lichtfeld Studio: WHITE (1) = keep. "
                        "Overrides the alpha channel of RGBA images. Saved to "
                        "masks/ (offset_splat is applied)."
                    ),
                }),
                "use_rmbg": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Run the built in RMBG background removal (RMBG-2.0 "
                        "by default, see rmbg_mode). The alpha channel is "
                        "reused as the Lichtfeld splat mask. Disable when "
                        "masks come from other nodes."
                    ),
                }),
                "rmbg_mode": (list(RMBG_MODES), {
                    "default": RMBG_DEFAULT_MODE,
                    **tooltip(RMBG_MODE_TOOLTIP),
                }),
                "rmbg_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    **tooltip(
                        "RMBG segmentation sensitivity. Lower (0.3-0.4) keeps "
                        "more foreground, higher (0.6-0.7) keeps less."
                    ),
                }),
                "rmbg_resize": (["static", "dynamic"], {
                    "default": "static",
                    **tooltip(
                        "RMBG-1.4 modes only (RMBG-2.0 always runs at its "
                        "native 1024 px). static = resize to the fixed model "
                        "resolution (consistent). dynamic = up to 1280 px for "
                        "better large image results (slower, base models only)."
                    ),
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "GPU acceleration for COLMAP SIFT extraction/matching "
                        "and for RMBG (torch CUDA)."
                    ),
                }),
                "keep_workspace": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Keep the temporary COLMAP/GLOMAP workspace (database, "
                        "frames, raw sparse output) for debugging."
                    ),
                }),
                "auto_align": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Align the reconstruction to the detected ground plane "
                        "(Y up, floor at Y=0) - recommended for clean camera "
                        "paths in Lichtfeld / 3DGS."
                    ),
                }),
                "sequential_overlap": ("INT", {
                    "default": 15, "min": 2, "max": 50, "step": 1,
                    **tooltip(
                        "Neighbouring frames compared by the sequential matcher. "
                        "15 is reliable for 360 deg orbits, raise it for fast "
                        "camera motions, lower it for speed. This is the main "
                        "lever for mapping time - the mapper is CPU-bound."
                    ),
                }),
                "max_image_size": ("INT", {
                    "default": 5120, "min": 1024, "max": 8192, "step": 256,
                    **tooltip(
                        "Longest image edge used by COLMAP's feature extractor. "
                        "The exported dataset always keeps the original "
                        "resolution."
                    ),
                }),
                "frame_step": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    **tooltip(
                        "Use only every n-th frame. 1 = all frames (best "
                        "quality), 2 = half the frames (recommended for 24 fps "
                        "orbits, roughly half the runtime). This is the main "
                        "lever for mapping time - the mapper is CPU-bound."
                    ),
                }),
                "downscale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1,
                    **tooltip(
                        "Resolution factor for the SfM input only. 1.0 = full "
                        "resolution. 0.5 = 4x less RAM. Exported images and "
                        "masks always stay at full resolution."
                    ),
                }),
                "offset_glomap": ("INT", {
                    "default": 4, "min": -50, "max": 50, "step": 1,
                    **tooltip(
                        "Morphological offset for the GLOMAP masks. Positive = "
                        "erode (shrink) to drop soft RMBG edges, negative = "
                        "dilate (grow). 0 = untouched."
                    ),
                }),
                "offset_splat": ("INT", {
                    "default": 12, "min": -100, "max": 100, "step": 1,
                    **tooltip(
                        "Morphological offset for the Lichtfeld/splat masks. "
                        "Positive = erode to remove alpha halos, negative = "
                        "dilate to keep loose hair or cloth. 0 = untouched."
                    ),
                }),
                # --- new in v2.0 (appended so existing workflows keep working) ---
                "mapper_backend": (["glomap", "colmap_global"], {
                    "default": "glomap",
                    **tooltip(
                        "glomap = use GLOMAP 1.2.0 (bundled/downloaded, very "
                        "fast; upstream project is discontinued and frozen). "
                        "colmap_global = use 'colmap global_mapper' from "
                        "COLMAP >= 3.12 - no GLOMAP download needed."
                    ),
                }),
                "auto_install_binaries": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Download the pinned COLMAP/GLOMAP builds into "
                        "<pack>/bin when no usable binary is found. Can be "
                        "disabled globally with ENNDEE_AUTO_DOWNLOAD=0."
                    ),
                }),
                "binary_flavor": (["auto", "cuda", "nocuda"], {
                    "default": "auto",
                    **tooltip(
                        "Flavor of the prebuilt binaries. auto = CUDA when a "
                        "CUDA GPU is detected, otherwise the CPU build."
                    ),
                }),
                "embed_alpha_in_images": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Also write the RMBG alpha channel into images/ "
                        "(RGBA PNGs) instead of only into masks/. Off keeps the "
                        "classic layout: RGB images + masks/."
                    ),
                }),
                "image_format": (["PNG", "JPEG"], {
                    "default": "PNG",
                    **tooltip(
                        "File format for exported images/. PNG is lossless and "
                        "preserves alpha. JPEG is smaller; alpha is kept in the "
                        "separate masks/ folder instead."
                    ),
                }),
                "jpeg_quality": ("INT", {
                    "default": 90, "min": 1, "max": 100, "step": 1,
                    **tooltip(
                        "JPEG quality from 1 (smallest/most artifacts) to 100 "
                        "(best/largest). Used only when Image Format is JPEG."
                    ),
                }),
            },
        }

    # Append outputs rather than reordering the established tracker sockets so
    # existing saved workflows keep their original output connections.
    RETURN_TYPES = ("CAMERA_TRAJECTORY", "POINTCLOUD", "FLOAT", "STRING")
    RETURN_NAMES = ("trajectory", "point_cloud", "confidence", "dataset_path")
    FUNCTION = "track"
    CATEGORY = "Enndee/3D"
    OUTPUT_NODE = True
    DESCRIPTION = ("Global SfM camera tracking (COLMAP + GLOMAP/global mapper) "
                   "with automatic binary setup and a complete Lichtfeld Studio "
                   "dataset export.")

    # The native pycolmap node ("COLMAP for Lichtfeld (Enndee)") overrides these
    # two hooks; everything below the backend setup is shared verbatim.
    BACKEND_READY_MESSAGE = "Binary setup complete"

    # =======================================================================
    # Main entry point
    # =======================================================================

    def track(self, colmap_path, glomap_path, camera_model, matcher, max_features,
              images_path="", masks_path="", lichtfeld_export_path="",
              images=None, masks_glomap=None, masks_lichtfeld=None,
              use_rmbg=True, rmbg_mode=RMBG_DEFAULT_MODE, rmbg_threshold=0.5,
              rmbg_resize="static",
              use_gpu=True, keep_workspace=False, auto_align=True,
              sequential_overlap=15, max_image_size=5120, frame_step=2,
              downscale_factor=1.0, offset_glomap=4, offset_splat=12,
               mapper_backend="glomap", auto_install_binaries=True,
               binary_flavor="auto", embed_alpha_in_images=False,
               image_format="PNG", jpeg_quality=90,
               export_depth_maps=False, dense_max_image_size=1024,
               dense_geom_consistency=True, fuse_dense_cloud=False,
               dense_cloud_max_points=400000, mesh_dense_surface=False,
               mesh_method="poisson"):
        """Run the complete tracking + export pipeline for one frame batch."""

        # ---------- 1. Make sure COLMAP / GLOMAP are available ------------
        colmap_exe, glomap_exe = self._setup_binaries(
            colmap_path, glomap_path, mapper_backend,
            auto_install_binaries, binary_flavor,
        )
        if colmap_exe is None:
            log_warn("COLMAP is not available - aborting. Run "
                     "'python install.py' inside the Enndees-Nodepack folder "
                     "or set ENNDEE_COLMAP_PATH.")
            return self._empty(1)
        log(self.BACKEND_READY_MESSAGE)
        maybe_warn_slow_mapping(matcher, frame_step, max_features)

        # ---------- 2. Load the input images -----------------------------
        export_images, sf_images, images_from_path = self._load_input_images(
            images, images_path, downscale_factor,
        )
        if sf_images is None:
            return self._empty(1)

        frame_total = int(export_images.shape[0])
        log(f"Input: {frame_total} frames "
            f"({export_images.shape[2]}x{export_images.shape[1]}), "
            f"source={'folder' if images_from_path else 'IMAGE input'}")

        # ---------- 3. Background removal (optional) ---------------------
        alpha_images = None  # RGBA batch used for alpha based splat masks
        if isinstance(export_images, torch.Tensor) and export_images.shape[-1] == 4:
            alpha_images = export_images
            log("RGBA input detected - alpha channel will be used as mask")

        if use_rmbg and alpha_images is None:
            log("Starting background removal")
            alpha_images = self._run_rmbg(
                export_images, use_gpu, rmbg_mode, rmbg_threshold, rmbg_resize,
            )
            if alpha_images is not None:
                log(f"RMBG done: {tuple(alpha_images.shape)}")

        # ---------- 4. Masks --------------------------------------------
        masks_glomap = self._prepare_masks(
            masks_glomap, masks_path, downscale_factor, offset_glomap, "GLOMAP")
        masks_lichtfeld = self._prepare_masks(
            masks_lichtfeld, "", 1.0, offset_splat, "Lichtfeld")

        # ---------- 5. Frame stepping -----------------------------------
        step = max(1, int(frame_step))
        if step > 1:
            export_images = export_images[::step]
            sf_images = sf_images[::step]
            if alpha_images is not None:
                alpha_images = alpha_images[::step]
            if masks_glomap is not None:
                masks_glomap = masks_glomap[::step]
            if masks_lichtfeld is not None:
                masks_lichtfeld = masks_lichtfeld[::step]
            log(f"frame_step={step}: {sf_images.shape[0]} frames kept")

        sf_images, masks_glomap = self._match_lengths(
            sf_images, masks_glomap, "GLOMAP masks")
        export_images, alpha_images = self._match_lengths(
            export_images, alpha_images, "alpha masks")

        # ---------- 6. Export folder ------------------------------------
        export_dir = self._resolve_export_dir(lichtfeld_export_path)
        has_alpha = bool(alpha_images is not None and alpha_images.shape[-1] == 4)

        if export_dir is not None:
            self._export_dataset_images(
                export_dir, export_images, alpha_images,
                embed_alpha=bool(embed_alpha_in_images),
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            if masks_glomap is not None:
                self._save_masks(masks_glomap, export_dir / MASK_GLOMAP_DIR,
                                 "GLOMAP")
            if masks_lichtfeld is not None:
                self._save_masks(masks_lichtfeld, export_dir / MASK_LICHTFELD_DIR,
                                 "Lichtfeld")
            elif has_alpha:
                log("Extracting alpha channel as Lichtfeld splat masks")
                self._save_alpha_masks(alpha_images, export_dir / MASK_LICHTFELD_DIR)

        # ---------- 7. Run the SfM pipeline -----------------------------
        wrapper = None
        frames_out = int(export_images.shape[0])
        try:
            wrapper = self._create_wrapper(colmap_exe, glomap_exe, mapper_backend)
            wrapper.progress_callback = log

            log(f"Starting SfM: {int(sf_images.shape[0])} frames; "
                f"feature extraction -> {matcher} matching -> "
                f"{mapper_backend} mapping")
            sparse_path = wrapper.run_pipeline(
                images=self._to_numpy_batch(sf_images),
                camera_model=camera_model,
                matcher=matcher,
                max_features=int(max_features),
                use_gpu=bool(use_gpu),
                keep_workspace=bool(keep_workspace),
                masks=self._to_numpy_masks(masks_glomap),
                sequential_overlap=int(sequential_overlap),
                max_image_size=int(max_image_size),
                mapper_backend=mapper_backend,
            )
            if not sparse_path:
                log_warn("Reconstruction failed - no sparse model was produced")
                return self._empty(frames_out)

            # ---------- 8. Parse the reconstruction ----------------------
            parser = COLMAPParser(sparse_path)
            parser.parse_all()
            poses, names = parser.get_camera_poses(convention="opengl")
            intrinsics = parser.get_intrinsics()
            confidence = float(parser.get_reconstruction_quality())
            points, colors = parser.get_point_cloud(convention="opengl")

            if auto_align and len(points) > 0 and len(poses) > 0:
                transform = parser.compute_alignment_transform(
                    points, poses, align_to_ground=True, recenter=False,
                )
                points, poses = parser.apply_transform(points, poses, transform)
                log("Scene aligned to the ground plane")

            trajectory = {
                "matrices": poses.astype(np.float32),
                "poses": poses.astype(np.float32),
                "translations": (poses[:, :3, 3].astype(np.float32)
                                 if len(poses) else np.zeros((1, 3), np.float32)),
                "intrinsics": intrinsics.astype(np.float32),
                "confidence": confidence,
                "format": "opengl",
                "source": f"glomap:{mapper_backend}",
                "num_frames": frames_out,
                "reconstructed_frames": len(poses),
                "image_names": names,
            }
            point_cloud = {
                "points": points.astype(np.float32),
                "colors": colors.astype(np.float32),
                "num_points": len(points),
                "source": "glomap",
            }

            # ---------- 9. Sparse model for Lichtfeld --------------------
            if export_dir is not None:
                self._export_sparse(sparse_path, export_dir, parser)

            # ---------- 10. Dense MVS products (optional) ----------------
            self._export_dense_geometry(
                wrapper, sparse_path, export_dir, export_images, parser,
                bool(export_depth_maps), int(dense_max_image_size),
                bool(dense_geom_consistency), bool(fuse_dense_cloud),
                int(dense_cloud_max_points), bool(mesh_dense_surface),
                str(mesh_method),
            )

            log(f"Done: {len(poses)}/{frames_out} poses registered, "
                f"{len(points)} 3D points, confidence={confidence:.2f}")
            return (trajectory, point_cloud, confidence,
                    str(export_dir) if export_dir is not None else "")

        except Exception as exc:  # noqa: BLE001
            log_warn(f"Pipeline error: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            return self._empty(frames_out)

        finally:
            if wrapper is not None and not keep_workspace:
                wrapper.cleanup_workspace()
            self._free_memory()

    # =======================================================================
    # Helpers: binaries
    # =======================================================================

    def _create_wrapper(self, colmap_exe, glomap_exe, mapper_backend):
        """Create the SfM wrapper and log which executables are used.

        The native pycolmap node overrides this (no executables at all).
        """
        log(f"COLMAP : {colmap_exe}  [{source_of('colmap')}]")
        if mapper_backend == "glomap":
            log(f"GLOMAP : {glomap_exe}  [{source_of('glomap')}]")
        else:
            log("Mapper : COLMAP global_mapper")
        return GLOMAPWrapper(
            colmap_path=str(colmap_exe),
            glomap_path=str(glomap_exe) if glomap_exe else None,
        )

    def _setup_binaries(self, colmap_path, glomap_path, mapper_backend,
                        auto_install_binaries, binary_flavor):
        """
        Resolve the COLMAP / GLOMAP executables, downloading them if allowed.

        Returns ``(colmap_path, glomap_path)``; ``glomap_path`` is None for the
        ``colmap_global`` backend.
        """
        needs_glomap = mapper_backend == "glomap"
        colmap_exe = resolve_binary("colmap", colmap_path)
        glomap_exe = resolve_binary("glomap", glomap_path) if needs_glomap else None

        missing = [name for name, path in (("colmap", colmap_exe),
                                          ("glomap", glomap_exe))
                   if path is None]
        if missing and auto_install_binaries and auto_download_enabled():
            log(f"Missing binaries: {', '.join(missing)} - downloading the "
                f"pinned builds into the pack folder (happens only once)")
            try:
                ensure_binaries(kinds=missing, flavor=binary_flavor, log=log)
            except Exception as exc:  # noqa: BLE001
                log_warn(f"Automatic binary install failed: {exc}")
            colmap_exe = resolve_binary("colmap", colmap_path)
            glomap_exe = resolve_binary("glomap", glomap_path) if needs_glomap else None

        if needs_glomap and glomap_exe is None:
            log_warn("GLOMAP not found. Run 'python install.py' inside the "
                     "Enndees-Nodepack folder, set ENNDEE_GLOMAP_PATH, or use "
                     "mapper_backend='colmap_global' with COLMAP >= 3.12.")
        return colmap_exe, glomap_exe

    # =======================================================================
    # Helpers: image / mask input
    # =======================================================================

    def _load_input_images(self, images, images_path, downscale_factor):
        """
        Return ``(export_images, sfm_images, loaded_from_path)``.

        ``export_images`` keeps the original resolution (dataset export),
        ``sfm_images`` is the optionally downscaled batch for COLMAP/GLOMAP.
        """
        factor = float(downscale_factor)

        if images is not None:
            batch = self._as_image_batch(images)
            if batch is None or batch.shape[0] == 0:
                log_warn("'images' input is empty - nothing to do")
                return None, None, False
            return batch, self._down(batch, factor), False

        folder = (images_path or "").strip()
        if not folder:
            log_warn("No 'images' input connected and 'images_path' is empty")
            return None, None, False

        export_images = self._load_images_from_path(folder, 1.0)
        if export_images is None or export_images.shape[0] == 0:
            log_warn(f"No images found in '{folder}'")
            return None, None, False

        return export_images, self._down(export_images, factor), True

    @staticmethod
    def _as_image_batch(images):
        """Normalise an IMAGE input to a float32 tensor [N,H,W,C] in [0, 1].

        ComfyUI hands ``IMAGE`` over as float32 in ``[0, 1]``, but a caller that
        drives :meth:`track` from a script (or another custom node) may hand over
        8-bit ``0..255`` data instead.  Without the rescale below the pipeline
        still runs and writes *silently* wrong files: the dataset export clamps
        to ``[0, 1]``, so every non-zero pixel becomes 1 and the dataset images
        come out as binary masks - and the depth model is fed that same mask.
        This is the same guard :meth:`_as_mask_tensor` already applies.
        """
        if isinstance(images, torch.Tensor):
            batch = images.detach()
        else:
            batch = torch.from_numpy(np.asarray(images, dtype=np.float32))

        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        if batch.dim() != 4:
            return None
        batch = batch.to("cpu", dtype=torch.float32).contiguous()

        if batch.numel() and float(batch.max()) > 1.0:
            log_warn("images arrived outside the ComfyUI [0, 1] float range "
                     "(8-bit 0..255?) - rescaling by 1/255; without this the "
                     "dataset export would clamp them into a binary mask")
            batch = batch / 255.0
        return batch.clamp(0.0, 1.0).contiguous()

    @staticmethod
    def _as_mask_tensor(masks):
        """Normalise a MASK input to a float32 tensor [N,1,H,W] in [0, 1]."""
        if isinstance(masks, torch.Tensor):
            tensor = masks.detach().to("cpu", dtype=torch.float32)
        else:
            tensor = torch.from_numpy(np.asarray(masks, dtype=np.float32))

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() == 4 and tensor.shape[-1] == 1:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.dim() != 4:
            raise ValueError(f"Unsupported mask shape {tuple(tensor.shape)}")

        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0).contiguous()

    @staticmethod
    def _to_numpy_batch(images):
        """IMAGE batch -> numpy array for the COLMAP/GLOMAP wrapper."""
        if images is None:
            return None
        if isinstance(images, torch.Tensor):
            return images.detach().cpu().numpy()
        return np.asarray(images)

    @staticmethod
    def _to_numpy_masks(masks):
        """Mask tensor [N,1,H,W] -> numpy [N,H,W] for the wrapper."""
        if masks is None:
            return None
        if isinstance(masks, torch.Tensor):
            array = masks.detach().cpu().numpy()
        else:
            array = np.asarray(masks, dtype=np.float32)

        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        elif array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim == 2:
            array = array[None, ...]
        return array.astype(np.float32)

    @staticmethod
    def _down(batch, factor):
        """Antialiased downscale of an image batch [N,H,W,C] (factor < 1)."""
        factor = float(factor)
        if batch is None or factor >= 1.0 or batch.shape[0] == 0:
            return batch

        height = max(32, int(round(batch.shape[1] * factor)))
        width = max(32, int(round(batch.shape[2] * factor)))
        data = batch.permute(0, 3, 1, 2).float()
        data = torch.nn.functional.interpolate(
            data, size=(height, width), mode="bilinear",
            align_corners=False, antialias=True,
        )
        return data.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _down_m(masks, factor):
        """Antialiased downscale of a mask tensor [N,1,H,W] (factor < 1)."""
        factor = float(factor)
        if masks is None or factor >= 1.0 or masks.shape[0] == 0:
            return masks

        height = max(32, int(round(masks.shape[2] * factor)))
        width = max(32, int(round(masks.shape[3] * factor)))
        data = torch.nn.functional.interpolate(
            masks.float(), size=(height, width), mode="bilinear",
            align_corners=False, antialias=True,
        )
        return data.clamp(0.0, 1.0).contiguous()

    @staticmethod
    def _match_lengths(images, masks, label):
        """Keep images and masks aligned when their counts differ."""
        if images is None or masks is None:
            return images, masks
        if int(images.shape[0]) == int(masks.shape[0]):
            return images, masks

        count = min(int(images.shape[0]), int(masks.shape[0]))
        log_warn(f"{label}: {int(masks.shape[0])} masks for "
                 f"{int(images.shape[0])} frames - truncating to {count}")
        return images[:count], masks[:count]

    @staticmethod
    def _free_memory():
        """Release CPU/GPU memory after large batches."""
        import gc

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # =======================================================================
    # Helpers: mask preparation
    # =======================================================================

    def _prepare_masks(self, masks, masks_path, downscale_factor, offset, label):
        """
        Normalise one optional mask source and apply the morphological offset.

        ``masks`` may be a MASK input (tensor) or None; in the latter case the
        folder in ``masks_path`` is loaded.  Returns a [N,1,H,W] tensor or None.
        """
        if masks is None:
            folder = (masks_path or "").strip()
            if folder:
                masks = self._load_masks_from_path(folder)
            if masks is not None and float(downscale_factor) < 1.0:
                masks = self._down_m(masks, downscale_factor)
        else:
            masks = self._as_mask_tensor(masks)
            if float(downscale_factor) < 1.0:
                masks = self._down_m(masks, downscale_factor)

        if masks is None or masks.shape[0] == 0:
            return None

        if int(offset) != 0:
            masks = self._offset_masks(masks, int(offset), label)

        log(f"{label} masks ready: {tuple(masks.shape)}")
        return masks

    # =======================================================================
    # Helpers: folders
    # =======================================================================

    @staticmethod
    def _list_images(folder: Path):
        """All image files in a folder, sorted by name (never raises)."""
        try:
            return sorted(
                entry for entry in folder.iterdir()
                if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
            )
        except OSError:
            return []

    @classmethod
    def _resolve_images_folder(cls, path):
        """
        Accept either a folder full of images or a dataset root containing
        ``images/``.  Returns None when the path does not exist.
        """
        folder = Path(str(path))
        if not folder.is_dir():
            return None
        if not cls._list_images(folder):
            nested = folder / "images"
            if nested.is_dir():
                return nested
        return folder

    def _load_images_from_path(self, path, downscale_factor=1.0):
        """Load all images of a folder as a float tensor [N,H,W,C]."""
        from PIL import Image

        folder = self._resolve_images_folder(path)
        if folder is None:
            log_warn(f"Image folder not found: {path}")
            return None

        files = self._list_images(folder)
        if not files:
            return None

        frames = []
        for file in files:
            with Image.open(file) as img:
                if img.mode == "RGBA":
                    frames.append(np.asarray(img, dtype=np.float32) / 255.0)
                else:
                    frames.append(
                        np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)

        batch = torch.from_numpy(np.stack(frames))
        log(f"Loaded {len(files)} images from {folder}")
        return self._down(batch, downscale_factor)

    def _load_masks_from_path(self, path):
        """Load all masks of a folder as a float tensor [N,1,H,W]."""
        from PIL import Image

        folder = Path(str(path))
        if not folder.is_dir():
            log_warn(f"Mask folder not found: {folder}")
            return None

        files = self._list_images(folder)
        if not files:
            log_warn(f"No masks found in {folder}")
            return None

        masks = []
        for file in files:
            with Image.open(file) as img:
                masks.append(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)

        tensor = torch.from_numpy(np.stack(masks)).unsqueeze(1)
        log(f"Loaded {len(files)} masks from {folder}")
        return tensor

    @staticmethod
    def _resolve_export_dir(lichtfeld_export_path):
        """
        Resolve the export folder.

        Relative paths are placed below the ComfyUI output folder, an empty
        value falls back to ``<output>/Splat_Frames/<timestamp>``.
        """
        raw = (lichtfeld_export_path or "").strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute() and COMFY_OUTPUT_DIR:
                path = Path(COMFY_OUTPUT_DIR) / path
                log(f"Relative export path resolved to {path}")
            return path

        if not COMFY_OUTPUT_DIR:
            log_warn("No export path given and ComfyUI output folder unknown - "
                     "the dataset will not be written to disk")
            return None

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = Path(COMFY_OUTPUT_DIR) / SPLAT_SUBDIR / stamp
        log(f"No export path given - using {path}")
        return path

    # =======================================================================
    # Helpers: dataset export
    # =======================================================================

    def _export_dataset_images(self, export_dir, export_images, alpha_images,
                              embed_alpha=False, image_format="PNG",
                              jpeg_quality=90):
        """
        Write dataset images as numbered PNG or JPEG files at full resolution.

        PNG preserves RGBA. JPEG is necessarily RGB; any available alpha remains
        in the separate Lichtfeld masks export.
        """
        from PIL import Image

        image_format = str(image_format).strip().upper()
        if image_format not in ("PNG", "JPEG"):
            raise ValueError(f"Unsupported dataset image format: {image_format!r}; choose PNG or JPEG")
        jpeg_quality = int(jpeg_quality)
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")

        if isinstance(export_images, torch.Tensor):
            batch = export_images.detach().to("cpu", dtype=torch.float32)
        else:
            batch = torch.from_numpy(np.asarray(export_images, dtype=np.float32))

        if (embed_alpha and alpha_images is not None
                and batch.shape[-1] == 3 and alpha_images.shape[-1] == 4):
            alpha = torch.as_tensor(alpha_images[..., 3:4], dtype=batch.dtype,
                                    device=batch.device)
            batch = torch.cat([batch, alpha], dim=-1)

        target = Path(export_dir) / "images"
        target.mkdir(parents=True, exist_ok=True)
        extension = ".png" if image_format == "PNG" else ".jpeg"

        # Re-running into an existing dataset folder with another format must
        # not leave old numbered images beside the new ones: COLMAP would see
        # both sets as separate views and the sparse TXT name mapping can drift.
        for existing in target.iterdir():
            if (existing.is_file()
                    and existing.suffix.lower() in IMAGE_SUFFIXES
                    and existing.stem.isdigit()):
                existing.unlink()

        if image_format == "JPEG" and batch.shape[-1] == 4:
            log("JPEG does not support alpha; exporting RGB images and preserving alpha in masks/")

        report_every = max(1, (len(batch) + 9) // 10)
        for index, frame in enumerate(batch):
            if index == 0 or (index + 1) % report_every == 0 or index + 1 == len(batch):
                log(f"Saving dataset image {index + 1}/{len(batch)} ({image_format})")
            frame_uint8 = (torch.clamp(frame, 0.0, 1.0) * 255.0).round() \
                .to(torch.uint8).cpu().numpy()
            mode = "RGBA" if frame_uint8.shape[-1] == 4 else "RGB"
            image = Image.fromarray(frame_uint8, mode)
            output_path = target / f"{index + 1:04d}{extension}"
            if image_format == "JPEG":
                if mode == "RGBA":
                    image = image.convert("RGB")
                image.save(output_path, format="JPEG", quality=jpeg_quality,
                           subsampling=0, optimize=True)
            else:
                image.save(output_path, format="PNG")

        detail = f", quality={jpeg_quality}" if image_format == "JPEG" else ""
        log(f"{len(batch)} {image_format} images -> {target} "
            f"(0001{extension}, ...){detail}")

    def _save_masks(self, masks, masks_dir, label):
        """Save masks as 0001.png ... (white = 255)."""
        from PIL import Image

        array = self._to_numpy_masks(masks)
        if array is None or len(array) == 0:
            return

        masks_dir = Path(masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for index, mask in enumerate(array):
            mask_uint8 = (np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            Image.fromarray(mask_uint8, "L").save(masks_dir / f"{index + 1:04d}.png")

        log(f"{len(array)} {label} masks -> {masks_dir} (0001.png, ...)")

    def _save_alpha_masks(self, alpha_images, masks_dir):
        """Save the alpha channel of RGBA images as masks."""
        if alpha_images is None or alpha_images.shape[-1] != 4:
            log_warn("No RGBA images available to derive alpha masks from")
            return
        self._save_masks(alpha_images[..., 3:4], masks_dir, "Lichtfeld (alpha)")

    def _export_dense_geometry(self, wrapper, sparse_path, export_dir, export_images,
                               parser, export_depth_maps, max_image_size,
                               geom_consistency, fuse_dense_cloud,
                               dense_cloud_max_points, mesh_dense_surface,
                               mesh_method):
        """The dense MVS products: depth maps, one fused cloud, one surface.

        All three fall out of the SAME PatchMatch pass, so it runs at most once. Each is
        an independent switch and each degrades to a warning - a dense product must never
        take the poses down with it.

        Why the cloud is the safe half of "consistent geometry": it lives in the sparse
        model's own world frame (``stereo_fusion`` works in that frame) and only *seeds*
        the Gaussians, so it can neither contradict the poses nor freeze an error in.
        The mesh is the opposite - a hard constraint - which is why it is the one that
        gets frozen only on request.
        """
        wanted = [name for name, enabled in (("export_depth_maps", export_depth_maps),
                                             ("fuse_dense_cloud", fuse_dense_cloud),
                                             ("mesh_dense_surface", mesh_dense_surface))
                  if enabled]
        if not wanted:
            return
        if export_dir is None:
            log_warn(f"{', '.join(wanted)} is on but no Lichtfeld export folder was "
                     "given - nothing written")
            return

        runner = getattr(wrapper, "dense_depth_maps", None)
        if runner is None:
            log_warn(f"{type(wrapper).__name__} has no dense MVS stage - the dense "
                     "products need the COLMAP node's native pycolmap backend on a "
                     "CUDA build")
            return

        try:
            maps = runner(sparse_path, max_image_size=int(max_image_size),
                          geom_consistency=bool(geom_consistency), log=log)
        except Exception as exc:  # noqa: BLE001 - dense products are optional
            log_warn(f"Dense MVS failed: {type(exc).__name__}: {exc}")
            return
        if not maps:
            log_warn("Dense MVS produced no depth maps - nothing exported")
            return

        if export_depth_maps:
            self._write_dense_depth(wrapper, export_dir, export_images, maps)
        if fuse_dense_cloud:
            self._write_fused_cloud(wrapper, export_dir, parser,
                                    int(dense_cloud_max_points))
        if mesh_dense_surface:
            self._write_dense_mesh(wrapper, export_dir, str(mesh_method))

    def _write_fused_cloud(self, wrapper, export_dir, parser, max_points):
        """Replace the dataset's sparse SIFT points with the fused dense cloud.

        Only ``points3D.txt`` is rewritten - the poses in ``images.txt`` stay exactly as
        exported, and the cloud is in that same model's world frame, so the two cannot
        disagree. Lichtfeld reads ``points3D.txt`` purely to seed Gaussians, so the empty
        per-point tracks are harmless: the VGGT node has always written them that way.

        ``parser`` MUST be the one that read the original *binary* sparse model. Re-parsing
        the exported folder does NOT work: ``parse_all`` only reads BIN models, the export
        deletes those, and the re-write then emits an EMPTY ``images.txt`` - which Lichtfeld
        rejects with "File is empty" (measured).

        The cloud is subsampled to ``max_points`` because the real budget is the
        rasterizer's 32-bit (primitive x tile) counter, not the point count.
        """
        fuse = getattr(wrapper, "dense_fused_cloud", None)
        if fuse is None:
            log_warn(f"{type(wrapper).__name__} has no dense fusion stage - keeping the "
                     "sparse initialisation")
            return
        if parser is None:
            log_warn("No parsed sparse model available - keeping the sparse initialisation")
            return

        import shutil

        try:
            fused = fuse(max_points=int(max_points), log=log)
        except Exception as exc:  # noqa: BLE001
            log_warn(f"Dense fusion failed: {type(exc).__name__}: {exc}")
            return
        points = fused.get("points") if fused else None
        colors = fused.get("colors") if fused else None
        if points is None or len(points) == 0:
            log_warn("Dense fusion produced no points - keeping the sparse initialisation")
            return

        target = Path(export_dir) / "sparse" / "0"
        images_dir = Path(export_dir) / "images"
        if not target.is_dir():
            log_warn(f"No sparse model at {target} - the fused cloud has nowhere to go")
            return

        try:
            parser.points3d = {}
            rgb = np.clip(np.asarray(colors, dtype=np.float64) * 255.0, 0, 255)
            for index in range(len(points)):
                parser.points3d[index + 1] = {
                    "xyz": np.asarray(points[index], dtype=np.float64),
                    "rgb": rgb[index].round().astype(np.int64),
                    "error": 0.0,
                    "track": [],
                }
            parser.write_txt(target, images_dir if images_dir.is_dir() else None)
        except Exception as exc:  # noqa: BLE001
            log_warn(f"Could not write the fused cloud into points3D.txt: "
                     f"{type(exc).__name__}: {exc}")
            return

        # keep the cloud itself: it is a useful artefact on its own, and the only copy
        # otherwise lives in the SfM workspace, which is deleted at the end of the run
        source = Path(fused.get("ply")) if fused.get("ply") else None
        if source is not None and source.is_file():
            try:
                mesh_dir = Path(export_dir) / "mesh"
                mesh_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, mesh_dir / "fused_points.ply")
            except Exception as exc:  # noqa: BLE001
                log_warn(f"Could not keep fused_points.ply: {type(exc).__name__}: {exc}")

        log(f"Dense cloud initialisation: {len(points)} points -> "
            f"sparse/0/points3D.txt (fused total {int(fused.get('total', len(points)))})")

    def _write_dense_mesh(self, wrapper, export_dir, mesh_method):
        """Write the dense surface to ``mesh/dense_mesh.ply`` for the trainer's anchors.

        The name matters: the Lichtfeld Headless Trainer looks for exactly
        ``mesh/dense_mesh.ply`` (or a ``.obj``) once its surface anchors are switched on.
        """
        import shutil

        build = getattr(wrapper, "dense_mesh", None)
        if build is None:
            log_warn(f"{type(wrapper).__name__} has no dense meshing stage - no surface")
            return
        try:
            mesh = build(method=str(mesh_method), log=log)
        except Exception as exc:  # noqa: BLE001
            log_warn(f"Dense meshing failed: {type(exc).__name__}: {exc}")
            return
        if mesh is None or not Path(mesh).is_file():
            log_warn(f"Dense meshing ({mesh_method}) produced no surface")
            return

        target_dir = Path(export_dir) / "mesh"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "dense_mesh.ply"
        try:
            shutil.copy2(mesh, target)
        except Exception as exc:  # noqa: BLE001
            log_warn(f"Could not copy the surface to {target}: {type(exc).__name__}: {exc}")
            return
        log(f"Dense surface: {mesh_method} -> mesh/dense_mesh.ply "
            f"({Path(mesh).stat().st_size / 1e6:.1f} MB)")

    def _write_dense_depth(self, wrapper, export_dir, export_images, maps):
        """Dense MVS depth maps -> ``depth/<stem>.depth.png`` in the VGGT node's layout.

        COLMAP's PatchMatch depth is fitted *photometrically to the sparse model's
        own poses*, so it cannot disagree with them - which is what makes it safe
        for Lichtfeld's depth loss **and** for the Meridian Geometry node's
        ``external_depth`` socket (one geometry shared by several shots).

        The wrapper returns one map per view PatchMatch covered, already warped
        onto the original image grid.  Views the sparse model never registered -
        or that had too little texture - stay all-zero, which Lichtfeld reads as
        "no depth": the same convention the VGGT node uses.
        """
        from enndee_feedforward import (depth_to_uint16, upsample_depth_maps,
                                        write_depth_maps)

        count = int(export_images.shape[0])
        image_hw = (int(export_images.shape[1]), int(export_images.shape[2]))

        # the workspace frames are frame_<i>.jpg, the dataset images are <i+1:04d>
        try:
            workspace_names = sorted(p.name for p in Path(wrapper.image_dir).iterdir()
                                     if p.is_file())
        except Exception:  # noqa: BLE001
            workspace_names = []
        if not workspace_names:
            log_warn("Cannot list the SfM workspace frames - no depth maps written")
            return

        found = {}
        for index in range(min(count, len(workspace_names))):
            name = workspace_names[index]
            depth = maps.get(name)
            if depth is None:
                depth = maps.get(Path(name).stem)
            if depth is not None and depth.size:
                found[index] = depth
        if not found:
            log_warn("Dense MVS depth maps match none of the exported frames - "
                     "nothing written")
            return

        shape = next(iter(found.values())).shape
        depth_small = np.zeros((count, int(shape[0]), int(shape[1])), np.float32)
        for index, depth in found.items():
            if depth.shape == shape:
                depth_small[index] = depth
            else:
                log_warn(f"Depth map {index} has an unexpected shape {depth.shape} "
                         f"(expected {shape}) - left empty")

        depth_full = upsample_depth_maps(depth_small, depth_small > 0, image_hw)
        stems = [f"{index + 1:04d}" for index in range(count)]
        written = write_depth_maps(export_dir / DEPTH_DIR, stems,
                                   depth_to_uint16(depth_full, depth_full > 0),
                                   log=log)
        log(f"Dense depth: {len(found)}/{count} frames covered, {written} written")

    def _export_sparse(self, sparse_path, export_dir, parser):
        """Copy the sparse model and write the TXT files Lichtfeld expects."""
        import shutil

        source = Path(sparse_path)
        if not source.exists():
            log_warn(f"Sparse model not found: {source}")
            return

        target = Path(export_dir) / "sparse" / "0"
        target.mkdir(parents=True, exist_ok=True)

        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, target / name)

        images_dir = Path(export_dir) / "images"
        if images_dir.exists():
            if parser is not None:
                parser.write_txt(target, images_dir)
            else:
                fix_image_names_in_sparse(target, images_dir)

        # Lichtfeld reads the TXT model only
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            leftover = target / name
            if leftover.exists():
                leftover.unlink()

        log(f"Sparse export -> {target} (cameras.txt, images.txt, points3D.txt)")

    @staticmethod
    def _empty(count):
        """Fallback outputs used when the reconstruction is not available."""
        frames = max(1, int(count))
        identity = np.tile(np.eye(4), (frames, 1, 1)).astype(np.float32)
        trajectory = {
            "matrices": identity,
            "poses": identity,
            "translations": np.zeros((frames, 3), np.float32),
            "intrinsics": np.array([1, 1, 0.5, 0.5], np.float32),
            "confidence": 0.0,
            "format": "opengl",
            "source": "glomap",
            "num_frames": frames,
            "reconstructed_frames": 0,
            "image_names": [],
        }
        point_cloud = {
            "points": np.zeros((0, 3), np.float32),
            "colors": np.zeros((0, 3), np.float32),
            "num_points": 0,
            "source": "glomap",
        }
        return trajectory, point_cloud, 0.0, ""

    # =======================================================================
    # Helpers: mask offsets
    # =======================================================================

    @staticmethod
    def _offset_masks(masks, offset_pixels, label="mask"):
        """
        Erode (positive) or dilate (negative) masks with a disk kernel.

        Positive values shrink the mask (good to drop soft RMBG edges), negative
        values grow it (good to keep hair or semi transparent cloth).  The input
        and the result are ``[N,1,H,W]`` float tensors in [0, 1].
        """
        if masks is None or int(offset_pixels) == 0:
            return masks

        try:
            import cv2
        except ImportError:
            log_warn(f"OpenCV not available - skipping the {label} offset")
            return masks

        array = GLOMAPLichtfeldTracker._to_numpy_masks(masks)
        if array is None or array.size == 0:
            return masks

        pixels = abs(int(offset_pixels))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
        operation = cv2.erode if offset_pixels > 0 else cv2.dilate

        result = np.empty_like(array, dtype=np.uint8)
        for index in range(array.shape[0]):
            source = (np.clip(array[index], 0.0, 1.0) * 255.0) \
                .round().astype(np.uint8)
            result[index] = operation(source, kernel, iterations=1)

        mode = "erode" if offset_pixels > 0 else "dilate"
        log(f"{label} masks: {mode} {pixels} px")
        return torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(1)

    # =======================================================================
    # Helpers: RMBG background removal
    # =======================================================================

    def _run_rmbg(self, images, use_gpu=True, mode=RMBG_DEFAULT_MODE,
                  threshold=0.5, resize="static"):
        """
        Remove the background and return an RGBA float tensor [N,H,W,4].

        Delegates to :mod:`enndee_rmbg`, which serves **RMBG-2.0**
        (``briaai/RMBG-2.0``, BiRefNet through ``transformers``) as well as the
        older **RMBG-1.4** modes (``transparent_background`` / InSPyReNet).
        Returns ``None`` when nothing could be loaded - the caller then keeps the
        frames untouched.
        """
        batch = self._as_image_batch(images)
        if batch is None or batch.shape[0] == 0:
            return None

        # Both generations are plain PyTorch models - only torch decides whether
        # the pass can use the GPU.  (The old onnxruntime check here forced the
        # CPU whenever onnxruntime-gpu was missing, which made a 113 frame run
        # take ~10 minutes instead of ~25 seconds.)
        want_gpu = bool(use_gpu) and torch.cuda.is_available()
        if bool(use_gpu) and not want_gpu:
            log_warn("torch reports no CUDA device - RMBG runs on the CPU")
        device = "cuda" if want_gpu else "cpu"

        frames = []
        for frame in batch:
            image_uint8 = (torch.clamp(frame, 0.0, 1.0) * 255.0) \
                .round().to(torch.uint8).cpu().numpy()
            if image_uint8.shape[-1] == 4:
                image_uint8 = image_uint8[..., :3]
            frames.append(image_uint8)

        mattes = remove_background(
            frames, mode=str(mode), device=device, threshold=threshold,
            resize=resize, log=log, log_warn=log_warn,
        )
        if mattes is None:
            return None

        rgb = np.stack([frame.astype(np.float32) / 255.0 for frame in frames])
        alpha = np.stack(mattes).astype(np.float32)[..., None]
        return torch.from_numpy(np.concatenate([rgb, alpha], axis=-1))


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Enndee_GLOMAPLichtfeldTracker": GLOMAPLichtfeldTracker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_GLOMAPLichtfeldTracker": "GLOMAP Lichtfeld Tracker (Enndee)",
}

__all__ = [
    "GLOMAPLichtfeldTracker",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

"""
GLOMAP Lichtfeld Tracker (Enndee) - global Structure-from-Motion camera tracking
and one-shot Lichtfeld / 3DGS dataset export for ComfyUI.
================================================================================

The node takes a frame sequence (connected ``IMAGE`` batch or a folder on disk),
optionally removes the background, runs COLMAP (SIFT) + GLOMAP (global mapper)
and writes a complete, Lichtfeld-Studio ready dataset:

    <export>/
    ├── images/         0001.png ...   (RGBA when masks/alpha are available)
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

Widget order of the original node is preserved, all new options are appended at
the end so existing workflows keep their settings.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports from the pack (vendored COLMAP/GLOMAP library + binary manager)
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

from enndee_bin import (  # noqa: E402  (path setup must run first)
    KINDS,
    ensure_binaries,
    resolve_binary,
    source_of,
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
SPLAT_SUBDIR = "Splat_Frames"

# Cached RMBG remover (transparent_background)
_rmbg_remover = None


def log(message: str) -> None:
    """Single logging helper so the prefix stays consistent."""
    print(f"[Enndee] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[Enndee] WARNING: {message}", flush=True)


def auto_download_enabled() -> bool:
    """ENNDEE_AUTO_DOWNLOAD=0 disables the on demand binary download."""
    value = (os.environ.get("ENNDEE_AUTO_DOWNLOAD") or "").strip().lower()
    return value not in ("0", "false", "no", "off")


def onnx_cuda_available() -> bool:
    """
    True when the installed onnxruntime can really use CUDA.

    ``transparent_background`` (RMBG) runs through onnxruntime, so asking for a
    CUDA device without onnxruntime-gpu would fail - this check keeps the node
    working with a CPU-only onnxruntime.
    """
    try:
        import onnxruntime  # type: ignore

        providers = onnxruntime.get_available_providers()
        return any("CUDA" in provider for provider in providers)
    except Exception:
        return False


def tooltip(text: str) -> Dict[str, str]:
    """Small helper for readable INPUT_TYPES."""
    return {"tooltip": text}

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
                        "every image (slow but robust for shuffled image sets)."
                    ),
                }),
                "max_features": ("INT", {
                    "default": 24000, "min": 1000, "max": 32768, "step": 1000,
                    **tooltip(
                        "SIFT features per image. More features = more stable "
                        "reconstruction but slower and more RAM. 24000 works "
                        "well for high resolution 360 deg orbits."
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
                        "Run the built in RMBG background removal "
                        "(transparent_background). The alpha channel is reused "
                        "as the Lichtfeld splat mask. Disable when masks come "
                        "from other nodes."
                    ),
                }),
                "rmbg_mode": (["base", "fast", "base-nightly"], {
                    "default": "base",
                    **tooltip(
                        "RMBG model. base = best quality, fast = quicker with "
                        "lower quality, base-nightly = newest base build."
                    ),
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
                        "static = resize to the fixed model resolution "
                        "(consistent). dynamic = up to 1280 px for better large "
                        "image results (slower, base models only)."
                    ),
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "GPU acceleration for COLMAP SIFT extraction/matching "
                        "and for RMBG (when onnxruntime-gpu is available)."
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
                        "camera motions, lower it for speed."
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
                        "orbits, roughly half the runtime)."
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
            },
        }

    RETURN_TYPES = ("CAMERA_TRAJECTORY", "POINTCLOUD", "FLOAT")
    RETURN_NAMES = ("trajectory", "point_cloud", "confidence")
    FUNCTION = "track"
    CATEGORY = "Enndee/3D"
    OUTPUT_NODE = True
    DESCRIPTION = ("Global SfM camera tracking (COLMAP + GLOMAP/global mapper) "
                   "with automatic binary setup and a complete Lichtfeld Studio "
                   "dataset export.")

    # =======================================================================
    # Main entry point
    # =======================================================================

    def track(self, colmap_path, glomap_path, camera_model, matcher, max_features,
              images_path="", masks_path="", lichtfeld_export_path="",
              images=None, masks_glomap=None, masks_lichtfeld=None,
              use_rmbg=True, rmbg_mode="base", rmbg_threshold=0.5,
              rmbg_resize="static",
              use_gpu=True, keep_workspace=False, auto_align=True,
              sequential_overlap=15, max_image_size=5120, frame_step=2,
              downscale_factor=1.0, offset_glomap=4, offset_splat=12,
              mapper_backend="glomap", auto_install_binaries=True,
              binary_flavor="auto", embed_alpha_in_images=False):
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
            wrapper = GLOMAPWrapper(
                colmap_path=str(colmap_exe),
                glomap_path=str(glomap_exe) if glomap_exe else None,
            )
            log(f"COLMAP : {colmap_exe}  [{source_of('colmap')}]")
            if mapper_backend == "glomap":
                log(f"GLOMAP : {glomap_exe}  [{source_of('glomap')}]")
            else:
                log("Mapper : COLMAP global_mapper")

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

            log(f"Done: {len(poses)}/{frames_out} poses registered, "
                f"{len(points)} 3D points, confidence={confidence:.2f}")
            return (trajectory, point_cloud, confidence)

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
        """Normalise an IMAGE input to a float32 tensor [N,H,W,C] on the CPU."""
        if isinstance(images, torch.Tensor):
            batch = images.detach()
        else:
            batch = torch.from_numpy(np.asarray(images, dtype=np.float32))

        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        if batch.dim() != 4:
            return None
        return batch.to("cpu", dtype=torch.float32).contiguous()

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
                              embed_alpha=False):
        """
        Write the dataset images as 0001.png ... in full resolution.

        Images stay RGBA when the input already carried an alpha channel.  With
        ``embed_alpha`` the RMBG alpha (or whatever produced the mask) is added
        to RGB images so ``images/`` is RGBA as well.
        """
        from PIL import Image

        if isinstance(export_images, torch.Tensor):
            batch = export_images.detach().to("cpu", dtype=torch.float32)
        else:
            batch = torch.from_numpy(np.asarray(export_images, dtype=np.float32))

        if (embed_alpha and alpha_images is not None
                and batch.shape[-1] == 3 and alpha_images.shape[-1] == 4):
            batch = torch.cat([batch, alpha_images[..., 3:4]], dim=-1)

        target = Path(export_dir) / "images"
        target.mkdir(parents=True, exist_ok=True)

        for index, frame in enumerate(batch):
            frame_uint8 = (torch.clamp(frame, 0.0, 1.0) * 255.0).round() \
                .to(torch.uint8).cpu().numpy()
            mode = "RGBA" if frame_uint8.shape[-1] == 4 else "RGB"
            Image.fromarray(frame_uint8, mode).save(target / f"{index + 1:04d}.png")

        log(f"{len(batch)} images -> {target} (0001.png, ...)")

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
        return trajectory, point_cloud, 0.0

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

    def _run_rmbg(self, images, use_gpu=True, mode="base", threshold=0.5,
                  resize="static"):
        """
        Remove the background with ``transparent_background`` (RMBG).

        Returns an RGBA float tensor [N,H,W,4] where alpha = foreground, or None
        when the optional dependency is missing / the model fails to load.
        """
        global _rmbg_remover

        try:
            from transparent_background import Remover
        except ImportError:
            log("transparent_background is not installed - RMBG disabled "
                "(pip install transparent_background)")
            return None

        want_gpu = bool(use_gpu) and torch.cuda.is_available()
        if want_gpu and not onnx_cuda_available():
            log_warn("onnxruntime has no CUDA provider - RMBG runs on the CPU "
                     "(install onnxruntime-gpu for GPU acceleration)")
            want_gpu = False
        device = "cuda" if want_gpu else "cpu"

        cache_key = (mode, resize, device)
        if _rmbg_remover is None or getattr(_rmbg_remover, "_enndee_key", None) != cache_key:
            try:
                _rmbg_remover = Remover(mode=mode, device=device, resize=resize)
                _rmbg_remover._enndee_key = cache_key
                log(f"RMBG model loaded (mode={mode}, device={device}, "
                    f"resize={resize})")
            except Exception as exc:  # noqa: BLE001
                log_warn(f"Could not load the RMBG model: {exc}")
                return None

        batch = self._as_image_batch(images)
        if batch is None or batch.shape[0] == 0:
            return None

        frames = []
        total = int(batch.shape[0])
        with torch.inference_mode():
            for index, frame in enumerate(batch):
                image_uint8 = (torch.clamp(frame, 0.0, 1.0) * 255.0) \
                    .round().to(torch.uint8).cpu().numpy()
                if image_uint8.shape[-1] == 4:
                    image_uint8 = image_uint8[..., :3]

                try:
                    rgba = _rmbg_remover.process(image_uint8, threshold=threshold)
                    frames.append(np.asarray(rgba, dtype=np.uint8))
                except Exception as exc:  # noqa: BLE001
                    log_warn(f"RMBG failed for frame {index}: {exc}")
                    height, width = image_uint8.shape[:2]
                    frames.append(np.dstack([
                        image_uint8,
                        np.full((height, width), 255, dtype=np.uint8),
                    ]))

                if (index + 1) % 10 == 0 or index + 1 == total:
                    log(f"RMBG {index + 1}/{total} frames")

        stacked = np.stack(frames).astype(np.float32) / 255.0
        if stacked.shape[-1] == 3:
            stacked = np.concatenate(
                [stacked, np.ones_like(stacked[..., :1])], axis=-1)
        return torch.from_numpy(stacked)


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

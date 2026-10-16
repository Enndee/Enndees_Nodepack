"""Path Pose Dataset (Enndee): the GLOMAP tracker minus SfM, plus ground-truth poses.

The reduced counterpart of ``GLOMAPLichtfeldTracker``: it runs the tracker's exact
frame bookkeeping (sharp-frame selection, ``frame_step``, Lichtfeld masks from
explicit masks / RGBA alpha / RMBG, dataset image export) but does **not** run
COLMAP/GLOMAP. The camera poses come from the ``MERIDIAN_EXTRINSICS`` signal of
**Meridian Geometry + Extrinsics (Enndee)** - the rig the video was rendered with -
assigned to the frames that survive selection by their ORIGINAL input index.

Output: the same Lichtfeld-Studio dataset layout as the tracker

    <export>/
    ├── images/                 0001.png ... or 0001.jpeg ...
    ├── masks/                  Lichtfeld / splat masks (0001.png ...)
    ├── sharp_selection.json    kept/dropped ORIGINAL frame numbers + pose mapping
    └── sparse/0/               cameras.txt, images.txt, points3D.txt

with ``images.txt`` carrying the ground-truth extrinsics (COLMAP world-to-camera,
the renderer's c2w inverted). ``points3D.txt`` is written empty: no SfM ran, so
there is no sparse cloud (the trainer only requires the file to exist).

Pose assignment: exported image k corresponds to ORIGINAL input frame
``final_output_to_input[k]`` (sharp selection + ``frame_step``, the tracker's
``sharp_selection.json`` schema). That frame maps onto the extrinsics timeline by
``extrinsics_mapping``: 'auto' resamples uniformly (identity when the counts
match), 'manual' is ``index + extrinsics_offset`` (chunked videos).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_PACK_DIR = Path(__file__).resolve().parent.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

from enndee_rmbg import (  # noqa: E402  (path setup must run first)
    RMBG_DEFAULT_MODE,
    RMBG_MODES,
)

try:
    from enndee_colmap import COLMAPParser  # noqa: E402
except Exception:  # pragma: no cover - fallback like the tracker's
    _fallback = _PACK_DIR.parent / "comfyui_colmap"
    if _fallback.is_dir() and str(_fallback) not in sys.path:
        sys.path.append(str(_fallback))
    from lib.colmap_parser import COLMAPParser  # type: ignore

from glomap_lichtfeld_node import (  # noqa: E402
    GLOMAPLichtfeldTracker,
    log,
    log_warn,
    tooltip,
)

#: Must match ``enndee_meridian_geometry_extrinsics.EXTRINSICS_SIGNAL_TYPE``.
EXTRINSICS_SIGNAL_TYPE = "MERIDIAN_EXTRINSICS"
EXTRINSICS_FORMAT = "meridian_extrinsics_v1"

FOCAL_MODES = ("vfov from signal", "manual focal px", "manual hfov", "from cameras.txt")


def read_reference_intrinsics(reference, width, height):
    """(fx, fy) at the EXPORT resolution from a reference dataset's cameras.txt.

    ``reference`` may be a dataset root (sparse/0/cameras.txt is picked) or the
    cameras.txt itself. Focals are rescaled by the resolution ratio so a
    reference measured at another size (e.g. before an upscale) still fits.
    """
    path = Path(str(reference or "").strip())
    if path.is_dir():
        for candidate in (path / "sparse" / "0" / "cameras.txt", path / "cameras.txt"):
            if candidate.is_file():
                path = candidate
                break
    if not path.is_file():
        raise ValueError(
            "focal_mode='from cameras.txt' needs focal_reference_path pointing at "
            "a dataset folder (or its cameras.txt).")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        model = fields[1]
        ref_w, ref_h = int(fields[2]), int(fields[3])
        params = [float(value) for value in fields[4:]]
        if model == "PINHOLE":
            fx, fy = params[0], params[1]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"):
            fx = fy = params[0]
        else:
            raise ValueError(f"unsupported camera model in reference: {model}")
        return fx * (float(width) / ref_w), fy * (float(height) / ref_h)
    raise ValueError(f"no camera line found in {path}")


def parse_extrinsics_signal(signal):
    """``MERIDIAN_EXTRINSICS`` JSON -> validated dict (c2w [F,4,4], focal [F], ...)."""
    try:
        data = json.loads(signal) if isinstance(signal, str) else dict(signal)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "extrinsics must be the MERIDIAN_EXTRINSICS signal of Meridian Geometry "
            "+ Extrinsics (Enndee).") from exc
    if not isinstance(data, dict) or data.get("format") != EXTRINSICS_FORMAT:
        raise ValueError(
            "extrinsics signal has an unknown format - connect the 'extrinsics' "
            "output of Meridian Geometry + Extrinsics (Enndee).")
    frames = int(data.get("frames", 0))
    c2w = np.asarray(data.get("c2w", []), dtype=np.float64)
    focal = np.asarray(data.get("focal", []), dtype=np.float64).reshape(-1)
    if frames <= 0 or c2w.shape != (frames, 4, 4):
        raise ValueError(
            f"extrinsics signal carries no camera frames (frames={frames}, "
            f"c2w shape {c2w.shape}) - run the geometry node first.")
    if focal.shape[0] != frames:
        raise ValueError("extrinsics signal: one focal multiplier per frame required")
    return {
        "frames": frames,
        "c2w": c2w,
        "focal": focal,
        "vfov_degrees": float(data.get("vfov_degrees", 0.0)),
        "zm": float(data.get("zm", 1.0)),
        "width": int(data.get("width", 0)),
        "height": int(data.get("height", 0)),
    }


def map_input_to_extrinsics(index, input_count, extr_count, mode, offset=0):
    """ORIGINAL input frame index -> extrinsics timeline index (clamped).

    'auto' spreads the input timeline uniformly over the extrinsics timeline, so
    identical counts give the identity and a re-timed/interpolated video still
    lands proportionally. 'manual' is ``index + offset`` for chunked clips.
    """
    index, offset = int(index), int(offset)
    if int(extr_count) <= 0:
        raise ValueError("no extrinsics frames to map onto")
    if int(input_count) <= 1 or str(mode) == "manual":
        mapped = index + offset
    else:
        mapped = int(round(index * (extr_count - 1) / float(input_count - 1))) + offset
    return max(0, min(int(extr_count) - 1, mapped))


def c2w_to_w2c(c2w):
    """[4,4] camera-to-world -> [3,4] world-to-camera (COLMAP/OpenCV).

    The rig's c2w rotation columns are (right, down, forward) - the renderer's
    OpenCV frame - so the world-to-camera rotation is exactly its transpose.
    """
    matrix = np.asarray(c2w, dtype=np.float64)
    rotation = matrix[:3, :3].T
    translation = -rotation @ matrix[:3, 3]
    return np.concatenate([rotation, translation[:, None]], axis=1)


def rotmat_to_qvec(rotation):
    """3x3 rotation -> ``[qw, qx, qy, qz]`` (COLMAP's eigen trick, same as
    ``enndee_feedforward.rotmat_to_qvec`` - kept local to avoid that module's
    model-stack imports)."""
    r = np.asarray(rotation, dtype=np.float64)
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = r.flat
    k = np.array([
        [rxx - ryy - rzz, 0, 0, 0],
        [ryx + rxy, ryy - rxx - rzz, 0, 0],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
        [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class EnndeePathPoseDataset(GLOMAPLichtfeldTracker):
    """Lichtfeld dataset export with ground-truth extrinsics - no COLMAP/GLOMAP run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extrinsics": (EXTRINSICS_SIGNAL_TYPE, {
                    "forceInput": True,
                    **tooltip(
                        "The 'extrinsics' output of Meridian Geometry + Extrinsics "
                        "(Enndee) - the exact rig the render flew. Its per-frame "
                        "c2w matrices are assigned to the selected frames and "
                        "written as sparse/0/images.txt."
                    ),
                }),
                "images_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Folder with the input frames (JPG/PNG/RGBA). Only used "
                        "when no 'images' input is connected."
                    ),
                }),
                "use_rmbg": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Derive the Lichtfeld masks with background removal "
                        "(RMBG). Runs AFTER sharp selection, so discarded frames "
                        "cost nothing. RGBA images already carry their alpha."
                    ),
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    **tooltip("Run RMBG on the GPU."),
                }),
                "rmbg_mode": (list(RMBG_MODES), {
                    "default": RMBG_DEFAULT_MODE,
                    **tooltip("Background-removal model/mode (same as the tracker)."),
                }),
                "rmbg_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    **tooltip("Alpha threshold for the RMBG matte."),
                }),
                "rmbg_resize": (["static", "dynamic"], {
                    "default": "static",
                    **tooltip("RMBG resize strategy (same as the tracker)."),
                }),
                "offset_splat": ("INT", {
                    "default": 12, "min": -64, "max": 64, "step": 1,
                    **tooltip(
                        "Morphological offset for the Lichtfeld masks in px: "
                        "positive erodes, negative dilates (same as the tracker)."
                    ),
                }),
                "frame_step": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    **tooltip("Use every n-th selected frame (poses follow)."),
                }),
                "sharp_selection": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Sharp Frame Selection (the tracker's folded-in Sharp "
                        "Frame Selector Top-N). Only the sharpest frames are "
                        "exported - the poses follow via the ORIGINAL frame "
                        "indices recorded in sharp_selection.json."
                    ),
                }),
                "sharp_method": (["batched_topn", "best_n"], {
                    "default": "batched_topn",
                    "label": "Selection Method",
                    **tooltip(
                        "batched_topn: keep the top 'Sharp Num Frames' of every "
                        "'Sharp Batch' chunk. best_n: the globally sharpest."
                    ),
                }),
                "sharp_batch": ("INT", {
                    "default": 4, "min": 2, "max": 64, "step": 1,
                    "label": "Batch Size",
                    **tooltip("Chunk size for the batched_topn method."),
                }),
                "sharp_num_frames": ("INT", {
                    "default": 3, "min": 1, "max": 64, "step": 1,
                    "label": "Num Frames",
                    **tooltip("Frames to keep per chunk (batched_topn) or in "
                              "total (best_n)."),
                }),
                "extrinsics_mapping": (["auto", "manual"], {
                    "default": "auto",
                    **tooltip(
                        "How input frames map onto the extrinsics timeline. "
                        "'auto' spreads uniformly (identity when the counts "
                        "match - the normal case; also covers interpolated/"
                        "retimed videos). 'manual' is index + Extrinsics Offset "
                        "for chunked videos."
                    ),
                }),
                "extrinsics_offset": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 1,
                    **tooltip("Offset added to the mapped extrinsics index "
                              "(manual mode, or a trim correction in auto)."),
                }),
                "focal_mode": (list(FOCAL_MODES), {
                    "default": FOCAL_MODES[0],
                    **tooltip(
                        "'vfov from signal': f = 0.5*H/tan(vFOV/2) at the EXPORT "
                        "resolution, times the path's focal multiplier - the "
                        "renderer's own model, but ONLY correct when the video "
                        "keeps the still's lens (generated videos usually do NOT "
                        "- a wrong focal makes the frustum symbols oversized in "
                        "Lichtfeld and distorts training). 'manual focal px' / "
                        "'manual hfov': the video's true lens. 'from cameras.txt': "
                        "copy the focal from a GLOMAP-tracked reference dataset "
                        "(focal_reference_path), rescaled to the export size."
                    ),
                }),
                "focal_px": ("FLOAT", {
                    "default": 1000.0, "min": 1.0, "max": 100000.0, "step": 1.0,
                    **tooltip("Focal length in px at the export resolution "
                              "(focal_mode='manual focal px')."),
                }),
                "hfov_deg": ("FLOAT", {
                    "default": 45.0, "min": 1.0, "max": 179.0, "step": 0.1,
                    **tooltip("Horizontal field of view in degrees "
                              "(focal_mode='manual hfov')."),
                }),
                "focal_reference_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "focal_mode='from cameras.txt': a dataset folder with "
                        "sparse/0/cameras.txt (or the cameras.txt itself), e.g. "
                        "the same clip tracked with the GLOMAP tracker. Its "
                        "focal is rescaled to the export resolution and used."
                    ),
                }),
                "lichtfeld_export_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Dataset output folder (images/, masks/, sparse/0/). "
                        "Relative paths resolve below the ComfyUI output folder; "
                        "empty = <output>/Splat_Frames/<timestamp>."
                    ),
                }),
                "embed_alpha_in_images": ("BOOLEAN", {
                    "default": False,
                    **tooltip("Write the alpha channel into images/ (RGBA PNGs) "
                              "instead of only into masks/."),
                }),
                "image_format": (["PNG", "JPEG"], {
                    "default": "PNG",
                    **tooltip("Dataset image format (masks stay PNG)."),
                }),
                "jpeg_quality": ("INT", {
                    "default": 90, "min": 1, "max": 100, "step": 1,
                    **tooltip("JPEG quality (image_format='JPEG')."),
                }),
            },
            "optional": {
                # Sockets live OPTIONAL on purpose: ComfyUI demands a connection
                # for every socket in "required", but both of these have working
                # fallbacks (images_path folder / alpha / RMBG / no masks).
                "images": ("IMAGE", {
                    **tooltip(
                        "The frames to train on (e.g. the generated video's "
                        "frames). Their ORIGINAL batch indices drive the pose "
                        "assignment, so sharp selection and frame_step can drop "
                        "frames without losing the pose mapping. Unconnected: "
                        "loaded from images_path instead."
                    ),
                }),
                "masks": ("MASK", {
                    "forceInput": True,
                    **tooltip(
                        "Optional Lichtfeld / splat masks (white = keep). When "
                        "absent, RGBA alpha or RMBG supplies them; with none of "
                        "those no masks/ folder is written (full-frame splatting)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dataset_path", "pose_report")
    FUNCTION = "export_dataset"
    CATEGORY = "Enndee/3D"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Lichtfeld / 3DGS dataset export with ground-truth camera extrinsics from "
        "the Meridian camera path: the tracker's sharp-frame selection and masking, "
        "but no COLMAP/GLOMAP run - the poses of the selected frames are taken "
        "straight from the rendered rig.")

    def export_dataset(self, extrinsics, images=None, images_path="", masks=None,
                       use_rmbg=False, use_gpu=True, rmbg_mode=RMBG_DEFAULT_MODE,
                       rmbg_threshold=0.5, rmbg_resize="static", offset_splat=12,
                       frame_step=1, sharp_selection=False,
                       sharp_method="batched_topn", sharp_batch=4,
                       sharp_num_frames=3, extrinsics_mapping="auto",
                       extrinsics_offset=0, focal_mode=FOCAL_MODES[0],
                       focal_px=1000.0, hfov_deg=45.0, focal_reference_path="",
                       lichtfeld_export_path="",
                       embed_alpha_in_images=False, image_format="PNG",
                       jpeg_quality=90):
        """Sharp selection + masking + dataset export, poses from the signal."""
        pose_data = parse_extrinsics_signal(extrinsics)

        # ---------- 1. Input frames (tracker's loader) --------------------
        export_images, _sf_images, images_from_path = self._load_input_images(
            images, images_path, 1.0)
        if export_images is None:
            raise ValueError("No input frames - connect 'images' or set images_path.")
        frame_total = int(export_images.shape[0])
        log(f"Path pose dataset: {frame_total} frames "
            f"({export_images.shape[2]}x{export_images.shape[1]}), "
            f"source={'folder' if images_from_path else 'IMAGE input'}, "
            f"{pose_data['frames']} extrinsics frames")

        alpha_images = None
        if isinstance(export_images, torch.Tensor) and export_images.shape[-1] == 4:
            alpha_images = export_images
            log("RGBA input detected - alpha channel will be used as mask")

        # ---------- 2. Sharp selection (tracker's, ORIGINAL indices) ------
        kept_indices = None
        sharp_record = None
        if sharp_selection:
            kept_indices = self._select_sharp_indices(
                export_images, sharp_method, sharp_batch, sharp_num_frames)
            kept_set = set(kept_indices)
            dropped = [index for index in range(frame_total) if index not in kept_set]
            sharp_record = {
                "selection_method": str(sharp_method),
                "batch_size": int(sharp_batch),
                "num_frames": int(sharp_num_frames),
                "kept_input_indices": list(kept_indices),
                "dropped_input_indices": dropped,
                "input_frame_count": frame_total,
            }
            export_images = self._subset_frame_batch(export_images, kept_indices)
            if alpha_images is not None:
                alpha_images = self._subset_frame_batch(alpha_images, kept_indices)
            log(f"sharp selection ({sharp_method}): kept {len(kept_indices)} of "
                f"{frame_total} frames (dropped {len(dropped)})")

        # ---------- 3. Masks (explicit -> alpha -> RMBG), like the tracker --
        if use_rmbg and alpha_images is None:
            log("Starting background removal")
            alpha_images = self._run_rmbg(
                export_images, use_gpu, rmbg_mode, rmbg_threshold, rmbg_resize)
            if alpha_images is not None:
                log(f"RMBG done: {tuple(alpha_images.shape)}")

        masks_lichtfeld = self._prepare_masks(masks, "", 1.0, offset_splat, "Lichtfeld")
        if kept_indices is not None:
            masks_lichtfeld = self._subset_prepared_masks(
                masks_lichtfeld, kept_indices, frame_total, "Lichtfeld")

        # ---------- 4. Frame stepping ------------------------------------
        step = max(1, int(frame_step))
        if step > 1:
            export_images = export_images[::step]
            if alpha_images is not None:
                alpha_images = alpha_images[::step]
            if masks_lichtfeld is not None:
                masks_lichtfeld = masks_lichtfeld[::step]
            log(f"frame_step={step}: {export_images.shape[0]} frames kept")
        export_images, alpha_images = self._match_lengths(
            export_images, alpha_images, "alpha masks")

        if kept_indices is not None:
            final_input_indices = self._final_frame_indices(kept_indices, step)
        else:
            final_input_indices = list(range(frame_total))[::step]
        if len(final_input_indices) != int(export_images.shape[0]):
            count = min(len(final_input_indices), int(export_images.shape[0]))
            log_warn(f"pose mapping: {len(final_input_indices)} frame indices for "
                     f"{int(export_images.shape[0])} images - truncating to {count}")
            final_input_indices = final_input_indices[:count]
            export_images = export_images[:count]

        # ---------- 5. Export folder: images + masks (tracker's writers) ---
        export_dir = self._resolve_export_dir(lichtfeld_export_path)
        has_alpha = bool(alpha_images is not None and alpha_images.shape[-1] == 4)
        if export_dir is None:
            raise ValueError("No dataset folder - set lichtfeld_export_path or run "
                             "inside ComfyUI.")
        if sharp_record is not None:
            sharp_record["frame_step"] = step
            sharp_record["final_input_indices"] = list(final_input_indices)
            sharp_record["final_output_to_input"] = {
                str(output_index): int(input_index)
                for output_index, input_index in enumerate(final_input_indices)}
            self._write_sharp_selection_sidecar(export_dir, sharp_record)
        self._export_dataset_images(
            export_dir, export_images, alpha_images,
            embed_alpha=bool(embed_alpha_in_images),
            image_format=image_format, jpeg_quality=jpeg_quality)
        if masks_lichtfeld is not None:
            self._save_masks(masks_lichtfeld, export_dir / "masks", "Lichtfeld")
        elif has_alpha:
            log("Extracting alpha channel as Lichtfeld splat masks")
            self._save_alpha_masks(alpha_images, export_dir / "masks")

        # ---------- 6. Ground-truth poses for the exported frames ---------
        height, width = int(export_images.shape[1]), int(export_images.shape[2])
        extrinsics_w2c, intrinsics, extr_indices, focal_used = self._assign_poses(
            pose_data, final_input_indices, frame_total,
            extrinsics_mapping, extrinsics_offset,
            focal_mode, focal_px, hfov_deg, width, height, focal_reference_path)
        extension = ".png" if str(image_format).upper() == "PNG" else ".jpeg"
        names = [f"{index + 1:04d}{extension}"
                 for index in range(len(final_input_indices))]
        self._write_sparse_model(export_dir, extrinsics_w2c, intrinsics, names,
                                 (height, width))
        log(f"ground-truth poses -> sparse/0/images.txt "
            f"({len(names)} views, focal {float(intrinsics[0, 0, 0]):.1f} px "
            f"at {width}x{height})")

        report = {
            "views": len(names),
            "input_frames": frame_total,
            "extrinsics_frames": int(pose_data["frames"]),
            "mapping": str(extrinsics_mapping),
            "offset": int(extrinsics_offset),
            "final_output_to_input": [int(value) for value in final_input_indices],
            "final_output_to_extrinsics": [int(value) for value in extr_indices],
            "focal_px": [round(float(intrinsics[i, 0, 0]), 3)
                         for i in range(intrinsics.shape[0])],
        }
        return (str(export_dir), json.dumps(report))

    def _assign_poses(self, pose_data, final_input_indices, frame_total,
                      mapping_mode, mapping_offset, focal_mode, focal_px,
                      hfov_deg, width, height, focal_reference_path=""):
        """Selected frames -> (w2c [N,3,4], K [N,3,3], extrinsics indices, focal list)."""
        extr_count = int(pose_data["frames"])
        extr_indices = [
            map_input_to_extrinsics(index, frame_total, extr_count,
                                    mapping_mode, mapping_offset)
            for index in final_input_indices]

        if str(focal_mode) == FOCAL_MODES[1]:
            base_fx = base_fy = float(focal_px)
        elif str(focal_mode) == FOCAL_MODES[2]:
            base_fx = base_fy = 0.5 * float(width) / math.tan(math.radians(float(hfov_deg)) / 2.0)
        elif str(focal_mode) == FOCAL_MODES[3]:
            base_fx, base_fy = read_reference_intrinsics(
                focal_reference_path, width, height)
        else:
            vfov = float(pose_data["vfov_degrees"])
            if not (0.0 < vfov < 180.0):
                raise ValueError(
                    "extrinsics signal carries no usable vFOV - pick another "
                    "focal_mode or fix the geometry node's signal.")
            base_fx = base_fy = 0.5 * float(height) / math.tan(math.radians(vfov) / 2.0)

        w2c_rows, intrinsics_rows, focal_used = [], [], []
        for index in extr_indices:
            w2c = c2w_to_w2c(pose_data["c2w"][index])
            mult = float(pose_data["focal"][index])
            fx, fy = base_fx * mult, base_fy * mult
            intrinsic = np.array([[fx, 0.0, width / 2.0],
                                  [0.0, fy, height / 2.0],
                                  [0.0, 0.0, 1.0]])
            w2c_rows.append(w2c)
            intrinsics_rows.append(intrinsic)
            focal_used.append(fx)
        return (np.stack(w2c_rows), np.stack(intrinsics_rows),
                extr_indices, focal_used)

    def _write_sparse_model(self, export_dir, extrinsics, intrinsics, names,
                            image_hw):
        """sparse/0/{cameras,images,points3D}.txt from the ground-truth poses.

        Same files as ``enndee_feedforward.write_colmap_model`` (kept local so this
        node never imports the feed-forward model stack): one shared PINHOLE camera
        - or one per view when the path's focal multipliers differ - and an empty
        ``points3D.txt`` (no SfM ran; the trainer only requires the file to exist).
        """
        height, width = int(image_hw[0]), int(image_hw[1])
        target = Path(export_dir) / "sparse" / "0"
        target.mkdir(parents=True, exist_ok=True)
        parser = COLMAPParser(str(target))
        parser.cameras = {}
        parser.images = {}
        parser.points3d = {}

        focal_values = [float(intrinsic[0, 0]) for intrinsic in intrinsics]
        constant_focal = max(focal_values) - min(focal_values) < 1e-6
        if constant_focal:
            camera_of = {index: 1 for index in range(len(names))}
            parser.cameras[1] = {
                "model_id": 1, "model_name": "PINHOLE", "width": width,
                "height": height,
                "params": np.array([intrinsics[0][0, 0], intrinsics[0][1, 1],
                                    intrinsics[0][0, 2], intrinsics[0][1, 2]]),
            }
        else:
            camera_of = {}
            for index, intrinsic in enumerate(intrinsics):
                camera_id = index + 1
                parser.cameras[camera_id] = {
                    "model_id": 1, "model_name": "PINHOLE", "width": width,
                    "height": height,
                    "params": np.array([intrinsic[0, 0], intrinsic[1, 1],
                                        intrinsic[0, 2], intrinsic[1, 2]]),
                }
                camera_of[index] = camera_id

        for index, name in enumerate(names):
            pose = np.asarray(extrinsics[index], dtype=np.float64)
            parser.images[index + 1] = {
                "qvec": rotmat_to_qvec(pose[:3, :3]),
                "tvec": pose[:3, 3].copy(),
                "camera_id": camera_of[index],
                "name": str(name),
                "points2d": np.zeros((0, 2)),
                "point3d_ids": np.zeros((0,), dtype=np.int64),
            }

        images_dir = Path(export_dir) / "images"
        parser.write_txt(target, images_dir if images_dir.is_dir() else None)
        log(f"Sparse export -> {target} (cameras.txt, images.txt, points3D.txt; "
            f"{len(names)} views, ground-truth poses, 0 points)")


NODE_CLASS_MAPPINGS = {"Enndee_PathPoseDataset": EnndeePathPoseDataset}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_PathPoseDataset": "Path Pose Dataset - no COLMAP (Enndee)"}



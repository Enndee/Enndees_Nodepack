"""
VGGT for Lichtfeld (Enndee) - feed-forward multi-view reconstruction and
one-shot Lichtfeld / 3DGS dataset export.
============================================================================

A drop-in **alternative to the COLMAP / GLOMAP tracker**: instead of SIFT
features plus a global mapper it runs a feed-forward pointmap model (VGGT-Omega
or VGG-T^3) over the whole frame set in **one forward pass** and exports

    <export>/
    ├── images/         0001.png ...            (unchanged, shared with COLMAP node)
    ├── masks/          Lichtfeld splat masks   (optional)
    ├── depth/          0001.depth.png ...      (16-bit, image resolution)  <- NEW
    └── sparse/0/       cameras.txt, images.txt, points3D.txt               <- from VGGT

Why use this instead of COLMAP
------------------------------
* **One source for everything.**  Poses, intrinsics and depth come out of a
  single forward pass, so they are in one gauge by construction.  That is what
  makes Lichtfeld's depth loss (``--use-depth-loss``) *safe*: a depth prior that
  disagrees with the poses fights the reconstruction, a prior in the same gauge
  can only reinforce it.
* **No incremental drift.**  COLMAP registers frames one by one; on AI generated
  footage that accumulates error.  A global-attention model predicts all views
  jointly, so there is no accumulation stage to drift.
* **Dense initialisation.**  ``points3D.txt`` is built by unprojecting the
  predicted depth (confidence filtered), which is far denser than a SIFT cloud.
* **Speed.**  One forward pass replaces extraction + matching + mapping.

The dataset layout is otherwise identical to the COLMAP node, so the two nodes
are interchangeable for the trainer.

.. note::
   The ``depth/`` folder is only meaningful together with Lichtfeld's
   ``use_depth_loss`` setting; enable it in the **Lichtfeld Headless Trainer**
   node.  Verify the result with ``LFS_DEPTH_LOSS_DIAG=1`` (the trainer log then
   prints ``corr`` / ``valid_fraction`` per iteration).
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PACK_DIR = Path(__file__).resolve().parent.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

from enndee_feedforward import (  # noqa: E402  (path setup must run first)
    BACKEND_NAMES,
    VGGT_OMEGA,
    backend_report,
    depth_to_uint16,
    depth_valid_mask,
    make_backend,
    resolve_checkpoint,
    sample_point_cloud,
    scale_intrinsics,
    unproject_depth,
    upsample_depth_maps,
    write_colmap_model,
    write_depth_maps,
)
from glomap_lichtfeld_node import (  # noqa: E402
    DEPTH_DIR,
    GLOMAPLichtfeldTracker,
    MASK_LICHTFELD_DIR,
    log,
    log_warn,
    tooltip,
)
from enndee_rmbg import (  # noqa: E402  (path setup must run first)
    RMBG_DEFAULT_MODE,
    RMBG_MODES,
    RMBG_MODE_TOOLTIP,
)

# ``DEPTH_DIR`` (the ``depth/`` sub folder Lichtfeld scans for
# ``<image stem>.depth.png``) is imported from the base node so the VGGT, COLMAP
# and GLOMAP nodes cannot drift apart on the layout.


class VGGTLichtfeldTracker(GLOMAPLichtfeldTracker):
    """Feed-forward multi-view reconstruction + Lichtfeld dataset export.

    Reuses the COLMAP node's image loading, background removal, mask handling,
    frame stepping and dataset image/mask export verbatim (it subclasses it) and
    replaces only the reconstruction backend.
    """

    BACKEND_READY_MESSAGE = "Feed-forward backend ready"

    #: last pipeline error, surfaced in the node's text preview (see ``_failed``)
    _last_error = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(BACKEND_NAMES), {
                    "default": VGGT_OMEGA,
                    **tooltip(
                        "Feed-forward reconstruction model - all three take the "
                        "whole frame set in one pass and return poses AND depth "
                        "in the same gauge. VGGT-Omega (Meta/Oxford, CVPR 2026) "
                        "is the most accurate pointmap model and the default; it "
                        "needs a local checkpoint. DA3-AnyView (ByteDance Depth "
                        "Anything 3, ICLR 2026, Apache-2.0) predicts a depth-ray "
                        "field with a single plain transformer and its report "
                        "puts it ahead of VGGT by +35.7% on camera pose and "
                        "+23.6% on geometry; weights come from Hugging Face. "
                        "VGG-T3 (NVIDIA, CVPR 2026) swaps the quadratic global "
                        "attention for a linear test-time-training one, so cost "
                        "grows linearly with the frame count (~1000 images in "
                        "under a minute) at a small accuracy cost - pick it for "
                        "very long frame sets; also downloaded from Hugging Face. "
                        "For DA3-AnyView, 'model_path' selects the variant "
                        "(depth-anything/DA3-SMALL is Apache-2.0 and fastest; "
                        "DA3-LARGE-1.1 is the Apache-2.0 LARGE default)."
                    ),
                }),
                "model_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Optional checkpoint. VGGT-Omega: the .pt file "
                        "(vggt_omega_1b_512.pt) or a checkout folder. "
                        "DA3-AnyView / VGG-T3: a Hugging Face repo id "
                        "('depth-anything/DA3-SMALL', 'depth-anything/DA3-LARGE', "
                        "'nvidia/vgg-ttt') or a local model folder - leave EMPTY "
                        "for the default repo, which is downloaded on first run. "
                        "Auto detection searches ENNDEE_VGGT_OMEGA_PATH / "
                        "ENNDEE_DA3_REPO / ENNDEE_VGGT_T3_PATH, then the usual "
                        "Tools/ and models/ folders next to the portable root and "
                        "inside this pack."
                    ),
                }),
                "image_resolution": ("INT", {
                    "default": 512, "min": 256, "max": 1536, "step": 64,
                    **tooltip(
                        "Resolution the reconstruction runs at (both models are "
                        "trained around 512). This is the main lever for VRAM: "
                        "VGGT-Omega needs roughly 10 GB at 50 frames and 21 GB at "
                        "200 frames at 512 px - lower it when you run out of "
                        "memory, raise it for more depth detail. The exported "
                        "depth maps are always upsampled to the image "
                        "resolution, which Lichtfeld requires."
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
                "lichtfeld_export_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Target folder for the complete Lichtfeld export "
                        "(images/, masks/, depth/, sparse/0/). Relative paths "
                        "are resolved below the ComfyUI output folder. "
                        "Empty = <output>/Splat_Frames/<timestamp>."
                    ),
                }),
                "export_depth": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Write the depth/ folder (<image stem>.depth.png, "
                        "16-bit, same resolution as the images) that Lichtfeld's "
                        "depth loss consumes. Turn this on together with "
                        "'use_depth_loss' in the Lichtfeld Headless Trainer. "
                        "Disable to write poses + point cloud only."
                    ),
                }),
                "depth_confidence_percentile": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    **tooltip(
                        "Pixels whose model confidence is below this percentile "
                        "of the same image are written as 'no depth' (0). The "
                        "gate is relative per image, so it adapts to each view. "
                        "Raise it to throw away more of the unreliable depth, "
                        "lower it to keep everything."
                    ),
                }),
                "point_cloud_source": (["depth", "none"], {
                    "default": "depth",
                    **tooltip(
                        "depth = write sparse/0/points3D.txt from the "
                        "unprojected, confidence filtered depth (dense "
                        "initialisation, far more points than SIFT). none = write "
                        "an empty points3D.txt and let Lichtfeld initialise "
                        "randomly."
                    ),
                }),
                "max_points": ("INT", {
                    "default": 400000, "min": 0, "max": 4000000, "step": 10000,
                    **tooltip(
                        "Upper bound for the exported point cloud (0 = keep every "
                        "confident pixel). Dense depth produces millions of "
                        "points; a few hundred thousand is plenty as a splat "
                        "initialisation and keeps the TXT readable."
                    ),
                }),
                "per_view_intrinsics": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Off = one shared PINHOLE camera with the averaged "
                        "predicted intrinsics (what COLMAP/GLOMAP produce for a "
                        "video, and the safest for Lichtfeld). On = one camera per "
                        "view, preserving the per-view prediction."
                    ),
                }),
                "auto_align": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Align the returned trajectory / point cloud to the "
                        "detected ground plane (Y up, floor at Y=0). Only affects "
                        "the node outputs, not the written dataset - identical to "
                        "the COLMAP node."
                    ),
                }),
                "frame_step": ("INT", {
                    "default": 2, "min": 1, "max": 16, "step": 1,
                    **tooltip(
                        "Keep every Nth frame. Reconstruction cost grows with the "
                        "frame count, and 4-8 MP footage does not need every "
                        "frame - step 2 is the usual sweet spot."
                    ),
                }),
                "downscale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05,
                    **tooltip(
                        "Downscale the input images before reconstruction and "
                        "export (1.0 = full resolution). Lower it to save VRAM "
                        "and time; the exported dataset keeps the downscaled size."
                    ),
                }),
                "masks_path": ("STRING", {
                    "default": "",
                    **tooltip(
                        "Folder with Lichtfeld splat masks (WHITE = keep), same "
                        "convention as 'masks_lichtfeld'. Only used when no mask "
                        "input is connected."
                    ),
                }),
                "offset_splat": ("INT", {
                    "default": 12, "min": -50, "max": 50, "step": 1,
                    **tooltip(
                        "Erode (+) or dilate (-) the Lichtfeld splat masks by "
                        "this many pixels, in the exported image resolution."
                    ),
                }),
                "use_rmbg": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Run the built in RMBG background removal (RMBG-2.0 "
                        "by default, see rmbg_mode). The alpha channel is "
                        "reused as the Lichtfeld splat mask. Disable when "
                        "masks come from other nodes - and for unmasked "
                        "scenes you usually want this OFF so the background "
                        "stays in the splat."
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
                        "Run RMBG and the reconstruction model on the GPU. The "
                        "feed-forward models have no practical CPU path - keep "
                        "this on."
                    ),
                }),
                "autocast": ("BOOLEAN", {
                    "default": True,
                    **tooltip(
                        "Run the model in bfloat16 (recommended, roughly halves "
                        "VRAM). Disable only for numerical troubleshooting."
                    ),
                }),
                "keep_workspace": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Keep intermediate files (the raw predicted tensors) for "
                        "debugging."
                    ),
                }),
                "embed_alpha_in_images": ("BOOLEAN", {
                    "default": False,
                    **tooltip(
                        "Write the background removal alpha into the exported "
                        "PNGs (RGBA) instead of only into masks/."
                    ),
                }),
                "image_format": (["PNG", "JPEG"], {
                    "default": "PNG",
                    **tooltip(
                        "Format of the exported dataset images. PNG is lossless "
                        "and supports alpha, JPEG is much smaller. Masks and "
                        "depth maps are always PNG."
                    ),
                }),
                "jpeg_quality": ("INT", {
                    "default": 90, "min": 1, "max": 100, "step": 1,
                    **tooltip("JPEG quality for the exported dataset images."),
                }),
                "subject_focus": (["full frame", "subject only"], {
                    "default": "full frame",
                    **tooltip(
                        "How far the DEPTH maps and the initial point cloud follow "
                        "the splat mask - this is the priority control for the "
                        "subject. 'full frame' keeps depth and points everywhere "
                        "(pair it with mask_mode='none' for the complete picture, or "
                        "with 'segment' + a low mask opacity penalty to merely "
                        "prioritise the subject). 'subject only' writes 'no depth' "
                        "(0) outside the mask and keeps only subject points, so the "
                        "prior agrees with a hard cut-out (mask_mode='segment' / "
                        "'alpha_consistent'). The masks/ folder is written either "
                        "way - the trainer decides how strictly to use it."
                    ),
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks_lichtfeld": ("MASK", {
                    **tooltip(
                        "Splat masks for Lichtfeld Studio: WHITE (1) = keep. "
                        "Overrides the alpha channel of RGBA images. Saved to "
                        "masks/ (offset_splat is applied)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("CAMERA_TRAJECTORY", "POINTCLOUD", "FLOAT", "STRING")
    RETURN_NAMES = ("trajectory", "point_cloud", "confidence", "dataset_path")
    FUNCTION = "track"
    CATEGORY = "Enndee/3D"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Feed-forward multi-view reconstruction (VGGT-Omega or VGG-T3) as an "
        "alternative to the COLMAP/GLOMAP tracker: one forward pass over the "
        "whole frame set yields camera poses, intrinsics AND per-view depth maps "
        "in a single, consistent gauge, plus a dense point cloud. Exports the "
        "same Lichtfeld Studio dataset layout as the COLMAP node, extended with "
        "a depth/ folder (<image stem>.depth.png, 16-bit, image resolution) for "
        "Lichtfeld's depth loss. Because poses and depth come from the same "
        "prediction, the depth prior cannot inject drift - it can only reinforce "
        "the reconstruction. Expect ~10-21 GB VRAM for 50-200 frames at 512 px."
    )

    # =======================================================================
    # Entry point
    # =======================================================================

    def track(self, model, model_path, image_resolution, images_path,
              lichtfeld_export_path, export_depth, depth_confidence_percentile,
              point_cloud_source, max_points, per_view_intrinsics, auto_align,
              frame_step, downscale_factor, masks_path, offset_splat,
              use_rmbg, rmbg_mode, rmbg_threshold, rmbg_resize, use_gpu,
              autocast, keep_workspace, embed_alpha_in_images, image_format,
              jpeg_quality, subject_focus="full frame",
              images=None, masks_lichtfeld=None):
        """Run one feed-forward reconstruction and export the Lichtfeld dataset."""

        # ---------- 1. Backend + checkpoint ------------------------------
        self._last_error = ""
        if not torch.cuda.is_available():
            return self._failed("CUDA is not available - the feed-forward "
                                "backends have no practical CPU path.")

        checkpoint = resolve_checkpoint(model, model_path)
        # model_path doubles as the hub repo id for the hub based backends
        backend = make_backend(model, checkpoint=checkpoint, repo=model_path,
                               autocast=autocast)
        usable, reason = backend.status()
        if not usable:
            return self._failed(reason, model_path=model_path)
        log(self.BACKEND_READY_MESSAGE)
        log(f"Backend : {reason}")

        # ---------- 2. Load the input images -----------------------------
        export_images, sf_images, images_from_path = self._load_input_images(
            images, images_path, downscale_factor,
        )
        if sf_images is None:
            return self._empty(1)

        log(f"Input: {int(export_images.shape[0])} frames "
            f"({export_images.shape[2]}x{export_images.shape[1]}), "
            f"source={'folder' if images_from_path else 'IMAGE input'}")

        # ---------- 3. Background removal (optional) ---------------------
        alpha_images = None
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
        masks_lichtfeld = self._prepare_masks(
            masks_lichtfeld, masks_path, 1.0, offset_splat, "Lichtfeld")

        # ---------- 5. Frame stepping -----------------------------------
        step = max(1, int(frame_step))
        if step > 1:
            export_images = export_images[::step]
            sf_images = sf_images[::step]
            if alpha_images is not None:
                alpha_images = alpha_images[::step]
            if masks_lichtfeld is not None:
                masks_lichtfeld = masks_lichtfeld[::step]
            log(f"frame_step={step}: {sf_images.shape[0]} frames kept")

        export_images, alpha_images = self._match_lengths(
            export_images, alpha_images, "alpha masks")
        frames_out = int(export_images.shape[0])

        # ---------- 6. Export folder, images, masks ---------------------
        export_dir = self._resolve_export_dir(lichtfeld_export_path)
        has_alpha = bool(alpha_images is not None and alpha_images.shape[-1] == 4)

        if export_dir is not None:
            self._export_dataset_images(
                export_dir, export_images, alpha_images,
                embed_alpha=bool(embed_alpha_in_images),
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            if masks_lichtfeld is not None:
                self._save_masks(masks_lichtfeld, export_dir / MASK_LICHTFELD_DIR,
                                 "Lichtfeld")
            elif has_alpha:
                log("Extracting alpha channel as Lichtfeld splat masks")
                self._save_alpha_masks(alpha_images, export_dir / MASK_LICHTFELD_DIR)

        result = self._reconstruct_and_export(
            backend=backend, model=model, export_images=export_images,
            export_dir=export_dir, image_resolution=int(image_resolution),
            export_depth=bool(export_depth),
            depth_confidence_percentile=float(depth_confidence_percentile),
            point_cloud_source=str(point_cloud_source),
            max_points=int(max_points),
            per_view_intrinsics=bool(per_view_intrinsics),
            auto_align=bool(auto_align), image_format=str(image_format),
            frames_out=frames_out, subject_focus=str(subject_focus),
            masks_lichtfeld=masks_lichtfeld,
        )
        # ``ui.text`` is ComfyUI's built-in text preview, so a failure the node
        # swallowed into an empty dataset is still visible on the node
        return {"ui": {"text": [self._result_text(result)]}, "result": result}

    def _failed(self, reason: str, model_path: str = ""):
        """Report a startup failure on the node itself, not only in the console.

        Returning an empty dataset with nothing but a console warning is what made
        VGG-T3 look like it "did nothing at all" - the reason now also reaches
        ComfyUI's text preview, together with the availability of every backend.
        """
        lines = [reason, "", "Available backends:"]
        lines.extend(backend_report(model_path).splitlines())
        log_warn(reason)
        for line in lines[2:]:
            log(line)
        return {"ui": {"text": ["\n".join(lines)]}, "result": self._empty(1)}

    def _result_text(self, result) -> str:
        """One line for the node's text preview (carries a swallowed failure)."""
        if self._last_error:
            return f"failed - {self._last_error}"
        try:
            trajectory, point_cloud, confidence = result[0], result[1], result[2]
            return (f"done - {trajectory.get('reconstructed_frames')}/"
                    f"{trajectory.get('num_frames')} views, "
                    f"{point_cloud.get('num_points')} points, "
                    f"confidence {float(confidence):.2f}")
        except Exception:  # noqa: BLE001 - the summary must never break the run
            return "done"

    # =======================================================================
    # Reconstruction + dataset writing
    # =======================================================================

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, model_hw) -> np.ndarray:
        """Resize ``(N, H, W, 3)`` in [0, 1] to the model resolution (colours)."""
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
        target = (int(model_hw[0]), int(model_hw[1]))
        if (int(rgb.shape[1]), int(rgb.shape[2])) != target:
            tensor = torch.nn.functional.interpolate(
                tensor, size=target, mode="bilinear", align_corners=False)
        return tensor.permute(0, 2, 3, 1).contiguous().numpy()

    @classmethod
    def _apply_subject_focus(cls, valid, masks_lichtfeld, subject_focus,
                             model_hw, count):
        """Optionally restrict the depth validity to the splat mask.

        The mask (``[N, 1, H, W]``, **white = keep = subject**) is resampled to
        the model resolution with nearest neighbour and ANDed into the depth
        validity gate. Masked-out pixels then become "no depth" (``0``) in the
        written maps *and* drop out of the initial point cloud, so the depth
        prior agrees with a hard cut-out instead of pushing geometry into a
        background the trainer ignores.

        ``full frame`` (the default) leaves everything untouched, which is what
        you want when the background should stay reconstructable.
        """
        if str(subject_focus) != "subject only":
            log("Subject focus: full frame - depth and points cover the whole image")
            return valid
        if masks_lichtfeld is None:
            log_warn("subject_focus='subject only' but no mask is available - "
                     "the depth maps stay full frame")
            return valid

        masks = masks_lichtfeld
        if not torch.is_tensor(masks):
            masks = torch.from_numpy(np.asarray(masks, dtype=np.float32))
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        if masks.dim() != 4:
            log_warn(f"Unexpected mask shape {tuple(masks.shape)} - "
                     f"the depth maps stay full frame")
            return valid
        if int(masks.shape[0]) != int(count):
            log_warn(f"Mask count {int(masks.shape[0])} != {int(count)} frames - "
                     f"the depth maps stay full frame")
            return valid

        target = (int(model_hw[0]), int(model_hw[1]))
        masks = masks.detach().to("cpu", dtype=torch.float32)
        if (int(masks.shape[2]), int(masks.shape[3])) != target:
            masks = torch.nn.functional.interpolate(masks, size=target, mode="nearest")
        keep = masks[:, 0].numpy() >= 0.5

        focused = valid & keep
        before = float(valid.mean()) if valid.size else 0.0
        after = float(focused.mean()) if focused.size else 0.0
        log(f"Subject focus: subject only - depth valid fraction "
            f"{before:.3f} -> {after:.3f} (mask keeps {float(keep.mean()):.3f} of the frame)")
        return focused

    def _reconstruct_and_export(self, backend, model, export_images, export_dir,
                                image_resolution, export_depth,
                                depth_confidence_percentile, point_cloud_source,
                                max_points, per_view_intrinsics, auto_align,
                                image_format, frames_out,
                                subject_focus="full frame", masks_lichtfeld=None):
        """Run the model once, then write depth + sparse model + outputs."""
        try:
            image_hw = (int(export_images.shape[1]), int(export_images.shape[2]))
            rgb = self._to_numpy_batch(export_images)
            if rgb is None or len(rgb) == 0:
                log_warn("No images to reconstruct")
                return self._empty(frames_out)
            rgb = rgb[..., :3]

            log(f"Reconstructing {frames_out} frames "
                f"({image_hw[1]}x{image_hw[0]}) at {image_resolution} px with {model}")
            reconstruction = backend.run(rgb, image_resolution=image_resolution,
                                         device="cuda")
            count = min(int(reconstruction.num_views), frames_out)
            if count < frames_out:
                log_warn(f"{model} returned {count} views for {frames_out} frames - "
                         f"the dataset keeps the first {count}")
            if count <= 0:
                return self._empty(frames_out)

            intrinsics = scale_intrinsics(reconstruction.intrinsics[:count],
                                          reconstruction.model_hw, image_hw)
            extrinsics = reconstruction.extrinsics[:count]

            # ---- depth maps for Lichtfeld's depth loss -----------------
            valid_small = depth_valid_mask(
                reconstruction.depth[:count], reconstruction.depth_conf[:count],
                conf_percentile=depth_confidence_percentile)
            # the priority control: optionally keep the depth prior inside the mask
            valid_small = self._apply_subject_focus(
                valid_small, masks_lichtfeld, subject_focus,
                reconstruction.model_hw, count)
            fraction = float(valid_small.mean()) if valid_small.size else 0.0
            log(f"Depth: valid fraction {fraction:.3f} "
                f"(confidence percentile {depth_confidence_percentile:.0f})")

            if export_dir is not None and export_depth:
                depth_full = upsample_depth_maps(reconstruction.depth[:count],
                                                 valid_small, image_hw)
                stems = [f"{index + 1:04d}" for index in range(count)]
                write_depth_maps(export_dir / DEPTH_DIR, stems,
                                 depth_to_uint16(depth_full, depth_full > 0), log=log)

            # ---- dense point cloud from the same depth -----------------
            points = np.zeros((0, 3), np.float32)
            colors = np.zeros((0, 3), np.float32)
            if point_cloud_source == "depth":
                colors_small = self._resize_rgb(rgb[:count], reconstruction.model_hw)
                points, colors = sample_point_cloud(
                    unproject_depth(reconstruction.depth[:count],
                                    reconstruction.intrinsics[:count], extrinsics),
                    colors_small, valid_small, max_points=max_points, seed=0)
                log(f"Point cloud: {len(points)} points (unprojected depth)")

            # ---- sparse/0: cameras.txt, images.txt, points3D.txt -------
            extension = ".png" if str(image_format).upper() == "PNG" else ".jpeg"
            names = [f"{index + 1:04d}{extension}" for index in range(count)]
            if export_dir is not None:
                write_colmap_model(export_dir, extrinsics, intrinsics, names,
                                   image_hw, points=points, colors=colors,
                                   per_view_intrinsics=per_view_intrinsics, log=log)
                if (export_dir / MASK_LICHTFELD_DIR).is_dir():
                    log("Trainer hint: masks/ written - mask_mode='none' splats the full "
                        "picture, 'segment' cuts to the subject, and 'segment' with a low "
                        "'mask opacity penalty' only prioritises it.")

            return self._build_outputs(
                extrinsics=extrinsics, intrinsics=intrinsics, points=points,
                colors=colors, names=names, count=count, frames_out=frames_out,
                auto_align=auto_align, export_dir=export_dir,
                source=reconstruction.source,
                confidence=float(reconstruction.confidences()[:count].mean()))
        except Exception as exc:  # noqa: BLE001
            log_warn(f"Pipeline error: {type(exc).__name__}: {exc}")
            self._last_error = f"{type(exc).__name__}: {exc}"
            import traceback

            traceback.print_exc()
            return self._empty(frames_out)
        finally:
            backend.release()
            self._free_memory()

    # =======================================================================
    # Node outputs (same structure as the COLMAP node)
    # =======================================================================

    @staticmethod
    def _build_outputs(extrinsics, intrinsics, points, colors, names, count,
                       frames_out, auto_align, export_dir, source, confidence):
        """Trajectory, point cloud, confidence and dataset path.

        The trajectory uses the same OpenGL camera-to-world convention as
        :meth:`COLMAPParser.get_camera_poses`, so downstream nodes cannot tell
        which tracker produced it.
        """
        from enndee_colmap.colmap_parser import COLMAPParser

        # world-to-camera (COLMAP/OpenCV) -> camera-to-world in OpenGL
        cv2gl = np.diag([1.0, -1.0, -1.0, 1.0])
        poses = []
        for index in range(count):
            pose = np.asarray(extrinsics[index], dtype=np.float64)
            rotation_c2w = pose[:3, :3].T
            translation_c2w = -rotation_c2w @ pose[:3, 3]
            pose_cv = np.eye(4)
            pose_cv[:3, :3] = rotation_c2w
            pose_cv[:3, 3] = translation_c2w
            poses.append(cv2gl @ pose_cv @ cv2gl)
        poses = np.array(poses, dtype=np.float64) if poses else np.zeros((0, 4, 4))

        # the node previews the cloud in the OpenGL convention, the dataset
        # keeps the raw COLMAP/OpenCV coordinates (as in the COLMAP node)
        preview_points = np.asarray(points, dtype=np.float32).copy()
        if len(preview_points):
            preview_points[:, 1] *= -1
            preview_points[:, 2] *= -1

        if auto_align and len(preview_points) and len(poses):
            parser = COLMAPParser(str(export_dir) if export_dir else ".")
            transform = parser.compute_alignment_transform(
                preview_points, poses, align_to_ground=True, recenter=False)
            preview_points, poses = parser.apply_transform(
                preview_points, poses, transform)
            log("Scene aligned to the ground plane")

        mean = np.asarray(intrinsics, dtype=np.float64).mean(axis=0)
        trajectory = {
            "matrices": poses.astype(np.float32),
            "poses": poses.astype(np.float32),
            "translations": (poses[:, :3, 3].astype(np.float32) if len(poses)
                             else np.zeros((1, 3), np.float32)),
            "intrinsics": np.array([mean[0, 0], mean[1, 1], mean[0, 2], mean[1, 2]],
                                   np.float32),
            "confidence": float(confidence),
            "format": "opengl",
            "source": source,
            "num_frames": int(frames_out),
            "reconstructed_frames": int(count),
            "image_names": list(names),
        }
        point_cloud = {
            "points": preview_points,
            "colors": np.asarray(colors, dtype=np.float32),
            "num_points": int(len(preview_points)),
            "source": source,
        }
        log(f"Done: {count}/{frames_out} views, {len(preview_points)} points, "
            f"confidence={float(confidence):.3f}")
        return (trajectory, point_cloud, float(confidence),
                str(export_dir) if export_dir is not None else "")

"""
COLMAP for Lichtfeld (Enndee) - the GLOMAP tracker through COLMAP's **native**
Python API instead of the downloaded executables.

Why this node exists
--------------------
GLOMAP was merged into COLMAP: since COLMAP 3.12 the global mapper ships with
COLMAP itself (``colmap global_mapper``), and the Python bindings expose exactly
the same pipeline:

============================  =========================================
downloaded binaries           native (pycolmap)
============================  =========================================
``colmap feature_extractor``  ``pycolmap.extract_features``
``colmap sequential_matcher`` ``pycolmap.match_sequential``
``colmap exhaustive_matcher`` ``pycolmap.match_exhaustive``
``colmap global_mapper``      ``pycolmap.global_mapping``  (GLOMAP)
``glomap mapper``             ``pycolmap.global_mapping``  (same thing)
============================  =========================================

So there is nothing to download, nothing to pin and no executable to find - just
``pip install pycolmap`` (which this node does for you, together with the CUDA ONNX
runtime, see :mod:`enndee_accelerators`).

Everything else - RMBG background removal, mask handling, frame stepping, the
Lichtfeld Studio dataset export (images/, masks/, sparse/) and the trajectory /
point-cloud outputs - is **shared verbatim** with the binary node: this class only
overrides the backend hooks.

.. note::
   The official pycolmap wheels for Windows are built **without CUDA** (CUDA wheels
   are Linux only), so SIFT extraction/matching and the bundle adjustment run on the
   CPU here. ``use_gpu`` switches to the GPU automatically if a CUDA-enabled
   pycolmap is installed (``pycolmap.has_cuda``).
"""

from enndee_accelerators import ensure_accelerators
from enndee_colmap.pycolmap_wrapper import PyColmapWrapper, pycolmap_info
from glomap_lichtfeld_node import GLOMAPLichtfeldTracker, log, log_warn, tooltip

#: ``mapper_backend`` names of the binary node, mapped to the native options.
LEGACY_MAPPER_BACKENDS = {"glomap": "global", "colmap_global": "global"}


class ColmapLichtfeldTracker(GLOMAPLichtfeldTracker):
    """Global SfM + Lichtfeld dataset export through pycolmap (no binaries)."""

    BACKEND_READY_MESSAGE = "Native pycolmap backend ready"

    @classmethod
    def INPUT_TYPES(cls):
        spec = GLOMAPLichtfeldTracker.INPUT_TYPES()
        required = spec["required"]
        optional = spec["optional"]

        # The binary paths (and their download flavour) are meaningless here.
        for name in ("colmap_path", "glomap_path"):
            required.pop(name, None)
        optional.pop("binary_flavor", None)

        # Same widgets, native option lists (the legacy names stay accepted).
        optional["mapper_backend"] = (["global", "incremental"], {
            "default": "global",
            **tooltip(
                "global = COLMAP's global mapper (GLOMAP, merged into COLMAP >= "
                "3.12; 'glomap' / 'colmap_global' are accepted as aliases). "
                "incremental = COLMAP's classic incremental mapper, slower but "
                "sometimes more forgiving on difficult image sets."
            ),
        })
        optional["auto_install_binaries"] = ("BOOLEAN", {
            "default": True,
            **tooltip(
                "Install/repair the python accelerators in ComfyUI's environment on "
                "demand: pycolmap (this node's backend) and onnxruntime-gpu (CUDA "
                "ONNX runtime for the RMBG/ONNX nodes, matched to your torch CUDA "
                "version, with the shadowing CPU wheel removed). "
                "ENNDEE_AUTO_DOWNLOAD=0 disables it."
            ),
        })
        return spec

    DESCRIPTION = (
        "Global SfM camera tracking through COLMAP's native Python API (pycolmap) - "
        "GLOMAP is part of COLMAP >= 3.12, so nothing is downloaded. Same widgets, "
        "same Lichtfeld Studio dataset export as the binary tracker; installs/repairs "
        "pycolmap and the CUDA ONNX runtime on demand."
    )

    # =======================================================================
    # Backend hooks (the only differences to the binary node)
    # =======================================================================

    def _setup_binaries(self, colmap_path, glomap_path, mapper_backend,
                        auto_install_binaries, binary_flavor):
        """No executables: make sure the python accelerators are installed."""
        report = ensure_accelerators(auto_install=bool(auto_install_binaries), log=log)
        colmap = report["pycolmap"]
        onnx = report["onnxruntime"]
        attention = report.get("optional_attention") or {}

        log(f"pycolmap    : {colmap['version'] or 'missing'}"
            f"{'' if colmap['cuda'] else ' (CPU build - Windows wheels have no CUDA)'}")
        log(f"onnxruntime : {onnx['version'] or 'missing'} "
            f"[{', '.join(onnx['providers']) or 'no providers'}]")
        log("attention   : " + ", ".join(
            f"{name}={'yes' if state else 'no'}" for name, state in attention.items()))

        if not colmap["available"]:
            log_warn("pycolmap is not available - aborting. Run 'python install.py' "
                     "inside the Enndees-Nodepack folder or 'pip install pycolmap'.")
            return None, None
        if not onnx["cuda"] and report.get("cuda_major"):
            log("note: the ONNX nodes (e.g. BRIA RMBG) still run on the CPU - restart "
                "ComfyUI after a fresh onnxruntime-gpu install")
        # ``colmap_exe`` is only a truthy marker here - the native wrapper needs no
        # executable, and ``_create_wrapper`` ignores it.
        return "pycolmap", None

    def _create_wrapper(self, colmap_exe, glomap_exe, mapper_backend):
        """Build the native pycolmap wrapper instead of the CLI wrapper."""
        info = pycolmap_info()
        log(f"COLMAP : pycolmap {info['version']} (native, "
            f"{'CUDA' if info['cuda'] else 'CPU'} build)")
        log("Mapper : " + ("incremental_mapping" if mapper_backend == "incremental"
                           else "global_mapping (GLOMAP)"))
        return PyColmapWrapper()

    # =======================================================================
    # Entry point: same widgets as the binary node, minus the two paths
    # =======================================================================

    def track(self, camera_model, matcher, max_features, images_path="",
              masks_path="", lichtfeld_export_path="", images=None,
              masks_glomap=None, masks_lichtfeld=None, use_rmbg=True,
              rmbg_mode="base", rmbg_threshold=0.5, rmbg_resize="static",
              use_gpu=True, keep_workspace=False, auto_align=True,
              sequential_overlap=15, max_image_size=5120, frame_step=2,
              downscale_factor=1.0, offset_glomap=4, offset_splat=12,
              mapper_backend="global", auto_install_binaries=True,
              embed_alpha_in_images=False, image_format="PNG", jpeg_quality=90):
        """Run the shared pipeline with the native backend."""
        backend = LEGACY_MAPPER_BACKENDS.get(str(mapper_backend).lower(),
                                             str(mapper_backend).lower())
        return GLOMAPLichtfeldTracker.track(
            self,
            colmap_path=None,
            glomap_path=None,
            camera_model=camera_model,
            matcher=matcher,
            max_features=max_features,
            images_path=images_path,
            masks_path=masks_path,
            lichtfeld_export_path=lichtfeld_export_path,
            images=images,
            masks_glomap=masks_glomap,
            masks_lichtfeld=masks_lichtfeld,
            use_rmbg=use_rmbg,
            rmbg_mode=rmbg_mode,
            rmbg_threshold=rmbg_threshold,
            rmbg_resize=rmbg_resize,
            use_gpu=use_gpu,
            keep_workspace=keep_workspace,
            auto_align=auto_align,
            sequential_overlap=sequential_overlap,
            max_image_size=max_image_size,
            frame_step=frame_step,
            downscale_factor=downscale_factor,
            offset_glomap=offset_glomap,
            offset_splat=offset_splat,
            mapper_backend=backend,
            auto_install_binaries=auto_install_binaries,
            binary_flavor="native",
            embed_alpha_in_images=embed_alpha_in_images,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )


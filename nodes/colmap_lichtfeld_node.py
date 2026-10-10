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
   are Linux only), so SIFT extraction/matching fall back to the **downloaded CUDA
   COLMAP build**; a CUDA-enabled pycolmap (``pycolmap.has_cuda``, self-built or set
   via ``ENNDEE_PYCOLMAP_CUDA_WHEEL``) is used in-process instead. The **mapping**
   (global positioning + bundle adjustment) runs on the **CPU on every build** - it is
   not a misconfiguration: COLMAP's global mapper pins ``SPARSE_SCHUR`` with
   ``auto_select_solver_type = false``, so only a COLMAP built with cuDSS/Caspar would
   use a GPU solver (see :meth:`~enndee_colmap.pycolmap_wrapper.PyColmapWrapper.mapper`).
"""

from typing import List, Optional

from enndee_accelerators import ensure_accelerators
from enndee_colmap.pycolmap_wrapper import PyColmapWrapper, pycolmap_info, resolve_gpu_bridge
from glomap_lichtfeld_node import (  # noqa: E402
    RMBG_DEFAULT_MODE,
    GLOMAPLichtfeldTracker,
    log,
    log_warn,
    tooltip,
)

#: ``mapper_backend`` names of the binary node, mapped to the native options.
LEGACY_MAPPER_BACKENDS = {"glomap": "global", "colmap_global": "global"}

#: custom websocket event carrying the live status text to the node
STATUS_EVENT = "enndee-colmap-status"


class NodeStatus:
    """Live status text plus ComfyUI's progress bar for one node execution.

    * the text ends up on the node twice: live through ``STATUS_EVENT`` (see
      ``web/js/enndee_colmap_status.js``) and, when the node finishes, through the
      built-in ``{"ui": {"text": [...]}}`` output;
    * the bar is ``comfy.utils.ProgressBar`` - 0-90 % for the SfM pipeline (which
      reports per chunk of images / pairs) and the rest for export + parsing.
    """

    def __init__(self, node_id=None, total: int = 100):
        self.node_id: Optional[str] = None if node_id in (None, "") else str(node_id)
        self.header: List[str] = []
        self.stage: str = ""
        self.text: str = ""
        self._last_percent: float = 0.0
        try:
            import comfy.utils

            self.pbar = comfy.utils.ProgressBar(total)
        except Exception:  # noqa: BLE001 - running outside ComfyUI (tests)
            self.pbar = None

    # ------------------------------------------------------------------ text
    def set_header(self, lines) -> None:
        self.header = [str(line) for line in lines]
        self._publish()

    def set_stage(self, text: str) -> None:
        self.stage = str(text)
        self._publish()

    def _publish(self) -> None:
        parts = list(self.header)
        if self.stage:
            parts.append(f"status  : {self.stage}")
        self.text = "\n".join(parts)
        if self.node_id is None:
            return
        try:
            from server import PromptServer

            PromptServer.instance.send_sync(STATUS_EVENT,
                                            {"node": self.node_id, "text": self.text})
        except Exception:  # noqa: BLE001 - no server (tests) or client gone
            pass

    # -------------------------------------------------------------- progress
    def set_percent(self, percent: float) -> None:
        """Advance the bar - it never moves backwards (stage markers emit 0)."""
        percent = max(self._last_percent, float(percent))
        self._last_percent = percent
        if self.pbar is None:
            return
        try:
            self.pbar.update_absolute(max(0, min(100, int(round(percent)))))
        except Exception:  # noqa: BLE001
            self.pbar = None

    def progress(self, value: int, total: int, label: str) -> None:
        """Progress hook of the wrapper: the pipeline owns the first 90 %."""
        fraction = (float(value) / float(total)) if total else 0.0
        self.set_stage(label)
        self.set_percent(fraction * 90.0)

    def finish(self, label: str = "finished") -> None:
        self.set_percent(100.0)
        self.set_stage(label)


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
                "Install/repair what the node needs, on demand: pycolmap (this node's backend), "
                "onnxruntime-gpu (CUDA ONNX runtime for the RMBG/ONNX nodes, matched to your torch "
                "CUDA version, with the shadowing CPU wheel removed) and - with use_gpu on - the "
                "CUDA COLMAP build that runs the SIFT stages on the GPU (~154 MB, one time; "
                "pycolmap has no CUDA wheel for Windows). ENNDEE_AUTO_DOWNLOAD=0 disables it, "
                "ENNDEE_PYCOLMAP_GPU_BRIDGE=0 keeps the node on pycolmap alone."
            ),
        })
        # Hidden: the node id routes the live status events to this node.
        spec.setdefault("hidden", {})["unique_id"] = "UNIQUE_ID"

        # --- dense MVS depth export ---------------------------------------
        # Appended LAST (after jpeg_quality) so every existing workflow keeps its
        # widget_values positions.
        optional["export_depth_maps"] = ("BOOLEAN", {
            "default": False,
            **tooltip(
                "Also run COLMAP's dense MVS (image undistorter + PatchMatch "
                "stereo) and write depth/<image stem>.depth.png (16-bit), the "
                "exact layout the VGGT node writes. This depth is fitted "
                "photometrically to the sparse model's own poses, so it cannot "
                "disagree with them - which is what makes it safe for "
                "Lichtfeld's use_depth_loss and for the Meridian Geometry node's "
                "external_depth socket (one geometry shared by several shots). "
                "Costs a CUDA pass over the whole frame set."
            ),
        })
        optional["dense_max_image_size"] = ("INT", {
            "default": 1024, "min": 320, "max": 4096, "step": 128,
            **tooltip(
                "Longest edge the dense MVS stage runs at; the maps are "
                "upsampled to the image resolution afterwards. Lichtfeld's "
                "depth loss only needs the relative ordering inside one image, "
                "so 1024 is usually plenty. This is the main runtime lever - "
                "cost grows with its square (measured: ~9 s per frame at 640 px "
                "and ~22 s at 1024 px for 3456x2304 frames on a 5090)."
            ),
        })
        optional["dense_geom_consistency"] = ("BOOLEAN", {
            "default": True,
            **tooltip(
                "COLMAP's cross-view consistency filter for PatchMatch. ON "
                "(recommended) also writes the filtered .geometric maps, and "
                "that is what gets exported: fewer valid pixels, no outliers. "
                "OFF exports the raw .photometric maps, which fill every pixel "
                "but contain wild values."
            ),
        })
        # --- dense geometry products (appended LAST, same rule as above) -----
        optional["fuse_dense_cloud"] = ("BOOLEAN", {
            "default": False,
            **tooltip(
                "Fuse the dense MVS depth maps into ONE multi-view consistent "
                "point cloud (colmap stereo_fusion) and write it into the "
                "dataset's sparse/0/points3D.txt as the training "
                "initialisation, instead of the sparse SIFT points. 3DGS is "
                "initialisation-sensitive and a dense, globally consistent "
                "cloud places far more primitives in the right places than SIFT "
                "ever can - and unlike a mesh it constrains nothing, so it "
                "cannot freeze an error in. Needs export_depth_maps (fusion "
                "runs on those maps). The cloud is subsampled to "
                "dense_cloud_max_points."
            ),
        })
        optional["dense_cloud_max_points"] = ("INT", {
            "default": 400000, "min": 10000, "max": 5000000, "step": 10000,
            **tooltip(
                "Upper bound on the fused cloud written to points3D.txt. The "
                "trainer's max_gaussians default is 1,000,000 and it is the "
                "rasterizer's 32-bit (primitive x tile) counter that actually "
                "overflows, so 400k is a safe initialisation budget. Raise it "
                "only together with max_gaussians."
            ),
        })
        optional["mesh_dense_surface"] = ("BOOLEAN", {
            "default": False,
            **tooltip(
                "Also reconstruct a SURFACE (triangle mesh) from the dense "
                "geometry and write mesh/dense_mesh.ply into the dataset. This "
                "is the anchor geometry: the Lichtfeld Headless Trainer can "
                "rasterise it into surface-aligned Gaussians (mesh2splat) and "
                "freeze them as scaffolding. It is a HARD constraint, unlike "
                "the depth loss - a wrong surface stays wrong. Needs "
                "export_depth_maps."
            ),
        })
        optional["mesh_method"] = (["poisson", "delaunay"], {
            "default": "poisson",
            **tooltip(
                "poisson = in-process Poisson surface reconstruction from the "
                "fused cloud: smooth and watertight, but it rounds off thin "
                "structures (hair, wires, leaves). delaunay = the bundled "
                "COLMAP 'delaunay_mesher' (visibility based, runs as a "
                "subprocess): keeps depth discontinuities and finer detail, "
                "noisier and less forgiving of outliers. Poisson needs no "
                "binary; delaunay needs the COLMAP build under <pack>/bin."
            ),
        })
        return spec

    DESCRIPTION = (
        "Global SfM camera tracking through COLMAP's native Python API (pycolmap) - "
        "GLOMAP is part of COLMAP >= 3.12, so nothing is downloaded. Same widgets, "
        "same Lichtfeld Studio dataset export as the binary tracker; installs/repairs "
        "pycolmap and the CUDA ONNX runtime on demand. GPU: SIFT extraction + matching "
        "run on the GPU - a pycolmap CUDA build in-process when present (Linux/macOS or "
        "self-built), otherwise the downloaded CUDA COLMAP build (the pycolmap wheels "
        "have no CUDA on Windows). The mapping (global positioning + bundle adjustment) "
        "stays on the CPU on every build: COLMAP's global mapper uses a CPU sparse "
        "solver (SPARSE_SCHUR) unless COLMAP is built with cuDSS. That is expected and "
        "not a misconfiguration - the mapping is the stage to shrink with frame_step, "
        "max_features, max_image_size, matcher and sequential_overlap."
    )

    # =======================================================================
    # Backend hooks (the only differences to the binary node)
    # =======================================================================

    def _setup_binaries(self, colmap_path, glomap_path, mapper_backend,
                        auto_install_binaries, binary_flavor):
        """No executables: make sure the **CUDA** python accelerators are installed."""
        status = getattr(self, "_status", None)
        if status is not None:
            status.set_stage("checking the python environment")
        report = ensure_accelerators(auto_install=bool(auto_install_binaries), log=log)
        colmap = report["pycolmap"]
        onnx = report["onnxruntime"]
        cuda = report.get("cuda") or {}
        attention = report.get("optional_attention") or {}
        self._backend_note = f"pycolmap {colmap['version'] or 'missing'} [{colmap['mode']}]"

        # ---- the GPU bridge: SIFT on the GPU through the downloaded CUDA COLMAP build -----
        # pycolmap has no CUDA build for Windows (PyPI wheels are CPU only, the CUDA ones are
        # Linux/macOS), so the node borrows the GPU where it pays off - extraction and matching -
        # and keeps the global mapper in-process. Downloading is fine: the user asked for GPU.
        bridge = None
        if not colmap["cuda"] and bool(getattr(self, "_use_gpu", True)):
            bridge = resolve_gpu_bridge(auto_install=bool(auto_install_binaries), log=log)
        self._gpu_bridge = bridge

        lines = ["COLMAP for Lichtfeld (Enndee) - native pycolmap backend",
                 f"pycolmap : {colmap['version'] or 'missing'} [{colmap['mode']}]"]
        if bridge is not None:
            label = "CUDA COLMAP bridge" if bridge.cuda else "COLMAP bridge"
            self._backend_note += " + " + ("CUDA bridge" if bridge.cuda else "bridge")
            lines.append(f"gpu      : {label} -> {bridge.executable}")
            lines.append("           feature extraction + matching run "
                         + ("on the GPU" if bridge.cuda else "in the COLMAP process")
                         + ", the global mapper in-process")
        elif colmap["cuda"]:
            lines.append("gpu      : native pycolmap CUDA build - everything runs in-process")
        elif not bool(getattr(self, "_use_gpu", True)):
            lines.append("gpu      : off (use_gpu is False) - SIFT runs on the CPU")
        else:
            lines.append("gpu      : none - SIFT runs on the CPU "
                         f"(ENNDEE_PYCOLMAP_GPU_BRIDGE=0 or no CUDA COLMAP build)")
        if colmap["mode"] == "cpu-fallback":
            lines.append(f"           CUDA {cuda.get('version') or '?'} "
                         f"({cuda.get('device') or 'GPU'}) is available, but pycolmap "
                         f"has to run on the CPU here:")
            lines.append(f"           {colmap['reason']}")
        elif colmap["mode"] == "cpu":
            lines.append("           no CUDA device/driver detected - the CPU build is correct")
        try:
            import torch

            torch_line = f"torch    : {torch.__version__}"
            torch_line += (f" (CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)})"
                           if torch.cuda.is_available() else " (no CUDA device)")
        except Exception:  # noqa: BLE001
            torch_line = "torch    : unavailable"
        lines.append(torch_line)
        lines.append(f"onnx     : {onnx['version'] or 'missing'} "
                     f"[{', '.join(onnx['providers']) or 'no providers'}]")
        lines.append("attention: " + ", ".join(
            f"{name}={'yes' if state else 'no'}" for name, state in attention.items()))
        for line in lines:
            log(line)
        if status is not None:
            status.set_header(lines)

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
        """Build the native pycolmap wrapper (plus the CUDA bridge for the SIFT stages)."""
        info = pycolmap_info()
        bridge = getattr(self, "_gpu_bridge", None)
        log(f"COLMAP : pycolmap {info['version']} (native, "
            f"{'CUDA' if info['cuda'] else 'CPU'} build)")
        if bridge is not None:
            log(f"SIFT   : {'GPU via the CUDA COLMAP build' if bridge.cuda else 'COLMAP build'} "
                f"{bridge.executable}")
        log("Mapper : " + ("incremental_mapping" if mapper_backend == "incremental"
                           else "global_mapping (GLOMAP)"))
        wrapper = PyColmapWrapper(bridge=bridge)
        status = getattr(self, "_status", None)
        if status is not None:
            wrapper.set_progress_hook(status.progress)
        return wrapper

    # =======================================================================
    # Entry point: same widgets as the binary node, minus the two paths
    # =======================================================================

    def track(self, camera_model, matcher, max_features, images_path="",
              masks_path="", lichtfeld_export_path="", images=None,
              masks_glomap=None, masks_lichtfeld=None, use_rmbg=True,
              rmbg_mode=RMBG_DEFAULT_MODE, rmbg_threshold=0.5, rmbg_resize="static",
              use_gpu=True, keep_workspace=False, auto_align=True,
              sequential_overlap=15, max_image_size=5120, frame_step=2,
              downscale_factor=1.0, offset_glomap=4, offset_splat=12,
              mapper_backend="global", auto_install_binaries=True,
              embed_alpha_in_images=False, image_format="PNG", jpeg_quality=90,
              export_depth_maps=False, dense_max_image_size=1024,
              dense_geom_consistency=True, fuse_dense_cloud=False,
              dense_cloud_max_points=400000, mesh_dense_surface=False,
              mesh_method="poisson", unique_id=None):
        """Run the shared pipeline with the native backend (plus live status)."""
        backend = LEGACY_MAPPER_BACKENDS.get(str(mapper_backend).lower(),
                                             str(mapper_backend).lower())
        self._status = NodeStatus(unique_id)
        self._backend_note = "pycolmap"
        self._use_gpu = bool(use_gpu)          # decides whether the CUDA bridge is fetched
        self._status.set_stage("starting")
        result = GLOMAPLichtfeldTracker.track(
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
            export_depth_maps=export_depth_maps,
            dense_max_image_size=dense_max_image_size,
            dense_geom_consistency=dense_geom_consistency,
            fuse_dense_cloud=fuse_dense_cloud,
            dense_cloud_max_points=dense_cloud_max_points,
            mesh_dense_surface=mesh_dense_surface,
            mesh_method=mesh_method,
        )
        self._status.finish(self._summary(result))
        # ``ui.text`` is ComfyUI's built-in text preview (see PreviewAny); the live
        # updates during the run go through the STATUS_EVENT websocket message.
        return {"ui": {"text": [self._status.text]}, "result": result}

    def _summary(self, result) -> str:
        """One line describing what the run produced (shown on the node)."""
        try:
            trajectory, point_cloud = result[0], result[1]
            frames = trajectory.get("reconstructed_frames")
            if not frames:
                return f"no reconstruction - {self._backend_note}"
            return (f"done - {frames}/{trajectory.get('num_frames')} frames registered, "
                    f"{point_cloud.get('num_points')} points, {self._backend_note}")
        except Exception:  # noqa: BLE001
            return f"done - {self._backend_note}"


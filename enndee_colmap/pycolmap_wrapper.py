"""
Native pycolmap backend - COLMAP's own Python bindings instead of CLI binaries.

COLMAP >= 3.12 absorbed **GLOMAP**: the ``colmap global_mapper`` command and
``pycolmap.global_mapping()`` are the same global SfM pipeline. This wrapper is a
drop-in replacement for :class:`~enndee_colmap.glomap_wrapper.GLOMAPWrapper`: it
inherits the whole workspace / frame / mask handling from
:class:`~enndee_colmap.colmap_wrapper.COLMAPWrapper` and only overrides the
pipeline stages so they run **in-process** through pycolmap:

============================  =========================================
CLI (downloaded binaries)     native (pycolmap)
============================  =========================================
``colmap feature_extractor``  ``pycolmap.extract_features``
``colmap sequential_matcher`` ``pycolmap.match_sequential``
``colmap exhaustive_matcher`` ``pycolmap.match_exhaustive``
``colmap global_mapper``      ``pycolmap.global_mapping``   (GLOMAP)
``glomap mapper``             ``pycolmap.global_mapping``   (same thing)
============================  =========================================

Nothing is downloaded and no executable is needed - only ``pip install pycolmap``.

.. note::
   The official PyPI wheels are built **without CUDA on Windows** (the CUDA wheels are Linux and
   macOS only). Feature extraction and matching are therefore delegated to the **downloaded CUDA
   COLMAP build** when one is available (``resolve_gpu_bridge``, ~154 MB, fetched on demand) - the
   global mapper still runs in-process through ``pycolmap.global_mapping``, and both sides read and
   write the same COLMAP database. A CUDA-enabled pycolmap build (self-built, or dropped in via
   ``ENNDEE_PYCOLMAP_CUDA_WHEEL``) is used in-process instead, so ``use_gpu`` really means GPU.
   Without either, the SIFT work runs on the CPU - and the run is noticeably slower.
"""

from pathlib import Path
import os
import re
import sys
from typing import Callable, List, Optional, Sequence, Tuple

from .colmap_wrapper import (_timeout_from_env, colmap_child_env, colmap_verbose,
                             run_streaming_command)
from .glomap_wrapper import GLOMAPWrapper

#: ``callable(value, total, label)`` - drives ComfyUI's progress bar.
ProgressHook = Callable[[int, int, str], None]

#: extraction runs in a handful of chunks: a call per image would be ~5x slower
#: (SIFT thread setup + database reopen) while one call gives no progress at all.
EXTRACTION_UPDATES = 12
#: matching chunks - roughly this many progress updates per matcher run.
MATCHING_UPDATES = 40

#: ``mapper_backend`` values that mean "global SfM" (GLOMAP).
GLOBAL_BACKENDS = ("glomap", "colmap_global", "global", "global_mapper")
#: ``mapper_backend`` value for COLMAP's classic incremental mapper.
INCREMENTAL_BACKEND = "incremental"

#: Set this to get COLMAP's full INFO flood back (its default is warnings only).
VERBOSE_ENV = "ENNDEE_COLMAP_VERBOSE"
#: glog level 1 = WARNING: keep the warnings, drop the "I2026... " progress spam.
GLOG_LEVEL_WARNING = "1"
#: glog level 0 = INFO: where COLMAP's own "Processed file [n/m]" progress records live.
GLOG_LEVEL_INFO = "0"
#: ``ENNDEE_PYCOLMAP_GPU_BRIDGE=0`` keeps the native node on pycolmap alone (no download).
BRIDGE_ENV = "ENNDEE_PYCOLMAP_GPU_BRIDGE"
#: ``ENNDEE_PYCOLMAP_GPU_BA=1`` opts back into the GPU bundle adjustment solvers. Only a
#: COLMAP built with cuDSS/Caspar can use them - see :meth:`PyColmapWrapper.mapper`.
GPU_BA_ENV = "ENNDEE_PYCOLMAP_GPU_BA"


def gpu_bridge_enabled() -> bool:
    """False when the user switched the CUDA bridge off via ``ENNDEE_PYCOLMAP_GPU_BRIDGE=0``."""
    return not str(os.environ.get(BRIDGE_ENV) or "").strip() in ("0", "false", "no", "off")


def gpu_ba_requested() -> bool:
    """True when the user forced the GPU solvers back on via ``ENNDEE_PYCOLMAP_GPU_BA=1``.

    COLMAP's global mapper pins ``linear_solver_type = SPARSE_SCHUR`` together with
    ``auto_select_solver_type = false`` (``src/colmap/sfm/global_mapper.h``), so its GPU
    solver is unreachable unless COLMAP was built with cuDSS/Caspar - which the standard
    builds are not. This switch exists for such a build; on everything else it only brings
    back the misleading "Falling back to CPU" warnings.
    """
    return str(os.environ.get(GPU_BA_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


class GpuBridge:
    """The downloaded COLMAP build, used for the SIFT stages (see `resolve_gpu_bridge`)."""

    def __init__(self, executable, source: str = "", cuda: bool = False):
        self.executable = Path(executable)
        self.source = str(source or "")
        #: True when the resolved build is the CUDA flavor (the one we download by default).
        self.cuda = bool(cuda)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GpuBridge({self.executable}, {self.source!r}, cuda={self.cuda})"


def _bin_manager():
    """The pack's binary manager (`enndee_bin`), or None when it cannot be imported."""
    for root in (Path(__file__).resolve().parent.parent,):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        import enndee_bin  # type: ignore
    except Exception:  # noqa: BLE001 - standalone use without the pack
        return None
    return enndee_bin


def resolve_gpu_bridge(auto_install: bool = True, log: Callable[[str], None] = print,
                       flavor: str = "cuda") -> Optional[GpuBridge]:
    """The **CUDA** COLMAP executable for the SIFT bridge, downloading it when allowed.

    pycolmap has no CUDA build for Windows (the PyPI wheels are CPU only and the CUDA wheels are
    Linux/macOS), so the native node borrows the GPU where it actually pays off: feature extraction
    and matching. Everything else stays native - the global mapper still runs in-process through
    ``pycolmap.global_mapping``, and the database both sides read and write is the same COLMAP
    database file.

    Resolution order: ``ENNDEE_COLMAP_PATH`` (explicit), the pin file ``bin/enndee_binaries.json``,
    an already downloaded build in ``<pack>/bin``, then - when `auto_install` is set - the pinned
    CUDA download (COLMAP 3.11.1, ~154 MB, done once).
    """
    if not gpu_bridge_enabled():
        return None
    if auto_install:
        # ENNDEE_AUTO_DOWNLOAD=0 is the pack's documented global escape hatch (the binary node and
        # the accelerator installer honour it as well).
        value = (os.environ.get("ENNDEE_AUTO_DOWNLOAD") or "").strip().lower()
        if value in ("0", "false", "no", "off"):
            auto_install = False
    manager = _bin_manager()
    if manager is None:
        return None
    try:
        explicit = os.environ.get("ENNDEE_COLMAP_PATH") or ""
        executable = manager.resolve_binary("colmap", explicit)
        if executable is None and auto_install:
            log("[pycolmap] GPU bridge: downloading the CUDA COLMAP build "
                "(one time, ~154 MB) - it runs SIFT on the GPU")
            try:
                manager.ensure_binaries(kinds=["colmap"], flavor=flavor, log=log)
            except Exception as exc:  # noqa: BLE001 - no download, no bridge
                log(f"[pycolmap] GPU bridge download failed: {exc}")
            executable = manager.resolve_binary("colmap", explicit)
    except Exception as exc:  # noqa: BLE001 - never break the node over the bridge
        log(f"[pycolmap] GPU bridge unavailable: {exc}")
        return None
    if executable is None:
        return None
    try:
        source = manager.source_of("colmap", explicit)
    except Exception:  # noqa: BLE001
        source = ""
    # The pack installs into <pack>/bin/<kind>-<version>-<flavor>, so the folder says which
    # build we ended up with - do not claim CUDA for a CPU build (or a user's own COLMAP).
    detected = Path(executable).parent.name.lower().rsplit("-", 1)[-1] == "cuda"
    return GpuBridge(executable, source, cuda=detected)


#: the "no CUDA build" situation is a property of the build, not of a stage - one
#: clear line per session is enough (extraction + matching + mapping would repeat it).
_CPU_NOTICE_SHOWN = False
#: same for the "a CUDA build still maps on the CPU" explanation - it is a property of
#: COLMAP's global mapper, not of this run.
_MAPPING_NOTICE_SHOWN = False


def _cpu_notice() -> None:
    """Explain *once* that this pycolmap build cannot use the GPU.

    The official Windows wheels are CPU only, so ``use_gpu`` is a no-op here. Without
    this line the user sees "running on the CPU" three times per run and wonders why
    the same settings are faster in the binary node - which runs SIFT on the GPU.
    """
    global _CPU_NOTICE_SHOWN
    if _CPU_NOTICE_SHOWN:
        return
    _CPU_NOTICE_SHOWN = True
    print("[pycolmap] this build has no CUDA support (the official Windows wheels are "
          "CPU only) - SIFT extraction, matching and the bundle adjustment run on the "
          "CPU here; the binary tracker (COLMAP CUDA build) does the same work on the "
          "GPU and is faster")


def _mapping_notice() -> None:
    """Explain *once* that even a CUDA build maps on the CPU (see `mapper`).

    SIFT extraction and matching are the stages the GPU really accelerates (81 images:
    8.85 s -> 1.65 s and 47.42 s -> 0.89 s). The mapping (global positioning + bundle
    adjustment) is CPU-bound on both builds, so a CUDA build that maps on the CPU is
    not a broken install - only a COLMAP with cuDSS/Caspar would change that.
    """
    global _MAPPING_NOTICE_SHOWN
    if _MAPPING_NOTICE_SHOWN:
        return
    _MAPPING_NOTICE_SHOWN = True
    print("[pycolmap] SIFT extraction + matching run on the GPU; the mapping (global "
          "positioning + bundle adjustment) stays on the CPU - COLMAP's global mapper "
          "uses a CPU sparse solver unless COLMAP is built with cuDSS")


def silence_colmap_logging(module=None) -> None:
    """Keep COLMAP's INFO flood out of the ComfyUI console.

    COLMAP logs through glog: every SIFT thread setup, every processed image and every
    pairing step lands on stderr as an ``I<date>`` line - hundreds of lines per run that
    bury the actual node output. The glog flag is read when the library initialises, so
    the environment variable has to be set *before* the import (``import_pycolmap`` does
    that) and the runtime level is set as well for good measure.
    ``ENNDEE_COLMAP_VERBOSE=1`` restores everything.
    """
    if colmap_verbose():
        return
    os.environ.setdefault("GLOG_minloglevel", GLOG_LEVEL_WARNING)
    module = module if module is not None else sys.modules.get("pycolmap")
    if module is None:
        return
    try:
        module.logging.minloglevel = module.logging.WARNING
    except Exception:  # noqa: BLE001 - older/newer bindings may differ
        pass


def import_pycolmap():
    """Return the ``pycolmap`` module or None (never raises)."""
    if not colmap_verbose():
        # must happen before the import: glog reads the flag at initialisation
        os.environ.setdefault("GLOG_minloglevel", GLOG_LEVEL_WARNING)
    try:
        import pycolmap  # type: ignore
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return None
    silence_colmap_logging(pycolmap)
    return pycolmap


def _cuda_device_name(module=None) -> str:
    """Best effort CUDA device name (torch knows it, pycolmap only counts devices)."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - torch is optional and may have no CUDA device
        pass
    try:
        if module is not None and int(module.get_num_cuda_devices()) > 0:
            return "CUDA device"
    except Exception:  # noqa: BLE001 - older/newer bindings may not expose it
        pass
    return ""


def pycolmap_info() -> dict:
    """What the installed pycolmap can do - for the node's header and the reports.

    Keys: ``available``, ``version``, ``cuda`` (the *build* has CUDA), ``device_name``,
    ``sift_on_gpu`` (this build can run SIFT on the GPU; the node additionally reports
    its CUDA COLMAP bridge, which covers the SIFT stages for a CPU-only build) and
    ``mapping_on_gpu`` - that stays False unless ``ENNDEE_PYCOLMAP_GPU_BA=1`` is set,
    because COLMAP's global mapper only reaches its GPU solver with cuDSS/Caspar
    (see :meth:`PyColmapWrapper.mapper`).
    """
    module = import_pycolmap()
    if module is None:
        return {"available": False, "version": "", "cuda": False, "device_name": "",
                "sift_on_gpu": False, "mapping_on_gpu": False}
    cuda = bool(getattr(module, "has_cuda", False))
    return {
        "available": True,
        "version": str(getattr(module, "__version__", "?")),
        "cuda": cuda,
        "device_name": _cuda_device_name(module) if cuda else "",
        "sift_on_gpu": cuda,
        "mapping_on_gpu": bool(cuda and gpu_ba_requested()),
    }


# ---------------------------------------------------------------------------
# Dense geometry products: fusion + surface
# ---------------------------------------------------------------------------


def points_from_reconstruction(module, reconstruction, ply_path=None):
    """``(points [N,3] float64, colors [N,3] float32 in [0,1])`` from a fusion result.

    ``stereo_fusion`` returns a :class:`pycolmap.Reconstruction` whose ``points3D`` carry
    ``xyz`` and an 8-bit ``color``.  If a build ever hands one back without points, the
    PLY the call wrote is read again through ``Reconstruction.import_PLY`` - verified to
    work on COLMAP's fused PLY.
    """
    import numpy as np

    entries = []
    if reconstruction is not None and hasattr(reconstruction, "points3D"):
        entries = list(reconstruction.points3D.values())
    if not entries and ply_path is not None and Path(ply_path).is_file():
        try:
            loaded = module.Reconstruction()
            loaded.import_PLY(str(ply_path))
            entries = list(loaded.points3D.values())
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] dense: cannot read {Path(ply_path).name}: {exc}")
            entries = []
    if not entries:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.float32)

    points = np.array([np.asarray(entry.xyz, dtype=np.float64) for entry in entries])
    colors = np.array([np.asarray(entry.color, dtype=np.float64) for entry in entries])
    if colors.size and float(colors.max()) > 1.5:      # 8-bit colours
        colors = colors / 255.0
    return points, np.clip(colors, 0.0, 1.0).astype(np.float32)


def stride_subset(count: int, cap: int):
    """Indices keeping at most ``cap`` of ``count`` entries, evenly and deterministically."""
    import numpy as np

    if cap <= 0 or count <= cap:
        return np.arange(count, dtype=np.int64)
    keep = np.linspace(0, count - 1, int(cap)).round().astype(np.int64)
    return np.unique(keep)


class PyColmapWrapper(GLOMAPWrapper):
    """Global SfM through COLMAP's native Python API (no executables at all)."""

    def __init__(self, device: Optional[str] = None,
                 progress_hook: Optional[ProgressHook] = None,
                 bridge: Optional[GpuBridge] = None):
        # No COLMAP/GLOMAP binaries of our own: set up the base attributes by hand because
        # COLMAPWrapper.__init__ insists on a real colmap_path. A `bridge` (the downloaded CUDA
        # COLMAP build) only serves the SIFT stages - see `resolve_gpu_bridge`.
        self.gpu_bridge = bridge
        self.colmap_path = str(bridge.executable) if bridge is not None else None
        self.workspace = None
        self.image_dir = None
        self.mask_dir = None
        self.database_path = None
        self.sparse_dir = None
        self._commands_cache = None
        self.progress_callback = None
        self.glomap_path = None
        self.native_device = device
        self.native_cuda = False
        self.progress_hook: Optional[ProgressHook] = progress_hook

    # -------------------------------------------------------------- progress
    def set_progress_hook(self, hook: Optional[ProgressHook]) -> None:
        """Install the ``callable(value, total, label)`` used for progress."""
        self.progress_hook = hook

    def _emit(self, value: int, total: int, label: str) -> None:
        """Report progress - a broken hook must never break a reconstruction."""
        if self.progress_hook is None:
            return
        try:
            self.progress_hook(int(value), max(1, int(total)), str(label))
        except Exception:  # noqa: BLE001
            self.progress_hook = None

    def _image_names(self) -> List[str]:
        """The staged frames, sorted (the order COLMAP would use)."""
        try:
            return sorted(p.name for p in Path(self.image_dir).iterdir() if p.is_file())
        except Exception:  # noqa: BLE001
            return []

    def _import_once(self, module, reader_options: dict) -> Optional[int]:
        """Import every frame up front so all extraction chunks share ONE camera.

        ``extract_features`` imports whatever images it is handed, and COLMAP's
        ``CameraMode.SINGLE`` means "one camera for the images of *this* call" - so the
        chunked extraction created one camera per chunk (113 frames came out with 13
        cameras, each with its own intrinsics block, and the global mapper then warned
        about missing focal priors). Importing the whole folder once and pinning that
        camera for every chunk keeps the single shared camera a video orbit needs.

        Returns the camera id to pin, or None when the import was not possible (the
        caller then keeps the old per-chunk behaviour).
        """
        try:
            module.Database.open(self.database_path).close()   # import_images wants the file
            module.import_images(self.database_path, self.image_dir,
                                 camera_mode=module.CameraMode.SINGLE,
                                 options=reader_options)
            database = module.Database.open(self.database_path)
            images = database.read_all_images()
            if hasattr(images, "values"):        # older builds hand out a dict
                images = list(images.values())
            camera_id = int(images[0].camera_id) if images else None
            database.close()
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] single-camera import failed ({exc}) - every chunk will "
                  f"create its own camera")
            return None
        return camera_id

    # ------------------------------------------------------------------ utils
    def _module(self):
        module = import_pycolmap()
        if module is None:
            print("[pycolmap] not installed - run 'python install.py' inside the "
                  "Enndees-Nodepack folder or 'pip install pycolmap'")
        return module

    def _effective_gpu(self, module, use_gpu: bool) -> bool:
        """True when this pycolmap build can really use the GPU.

        Also the place where the one-time explanations fire: a build without CUDA gets
        `_cpu_notice`, a CUDA build gets `_mapping_notice` (its mapping still runs on
        the CPU - see `mapper`).
        """
        has_cuda = bool(getattr(module, "has_cuda", False))
        if use_gpu and has_cuda:
            _mapping_notice()
        elif use_gpu and not has_cuda:
            _cpu_notice()
        return bool(use_gpu) and has_cuda

    @staticmethod
    def _device(module, effective_gpu: bool):
        device = getattr(module, "Device", None)
        if device is None:
            return None
        return device.cuda if effective_gpu else device.cpu

    # ------------------------------------------------------------ GPU bridge
    def _bridge_active(self, use_gpu: bool) -> bool:
        """True when the SIFT stages should run on the GPU through the CUDA COLMAP build.

        A pycolmap build *with* CUDA always wins (everything stays in-process); the bridge is for
        the builds that have none - which is every Windows wheel.
        """
        if not use_gpu or self.gpu_bridge is None:
            return False
        module = import_pycolmap()
        return not bool(getattr(module, "has_cuda", False)) if module is not None else False

    def _bridge_run(self, args: List[str], label: str) -> bool:
        """Run one bridge command, turning COLMAP's "[n/m]" progress records into the progress bar.

        The child runs at glog INFO (that is where the records live) and its output goes to the
        callback instead of the console - so the console stays as quiet as with the CPU path while
        the bar still moves.
        """
        bridge = self.gpu_bridge
        if bridge is None:
            return False
        progress = {"value": 0, "total": max(1, len(self._image_names()))}

        def on_output(message: str) -> None:
            if colmap_verbose():
                # run_streaming_command already prefixed the record with the label
                print(message, flush=True)
            found = re.findall(r"\[(\d+)/(\d+)\]", message)
            if not found:
                return                              # INFO chatter: neither console nor bar
            value, total = int(found[-1][0]), int(found[-1][1])
            progress["value"] = max(progress["value"], value)
            progress["total"] = max(1, total)
            self._emit(progress["value"], progress["total"], f"{label} {value}/{total}")

        self._emit(0, 1, label)
        try:
            return_code, output = run_streaming_command(
                [str(bridge.executable)] + [str(argument) for argument in args],
                label,
                _timeout_from_env("ENNDEE_COLMAP_TIMEOUT", 3600),
                progress_callback=on_output,
                env=colmap_child_env(GLOG_LEVEL_INFO),
            )
        except Exception as exc:  # noqa: BLE001 - the CPU path is the fallback
            print(f"[pycolmap] GPU bridge {label} failed: {exc}")
            return False
        if return_code != 0:
            print(f"[pycolmap] GPU bridge {label} failed (exit {return_code}) - "
                  f"falling back to the CPU path")
            for line in (output or "").strip().splitlines()[-4:]:
                print(f"    {line}")
            return False
        self._emit(1, 1, label)
        return True

    def _bridge_extract(self, camera_model: str, single_camera: bool, max_image_size: int,
                        max_num_features: int, mask_path: Optional[str],
                        estimate_affine_shape: bool, domain_size_pooling: bool) -> bool:
        """``colmap feature_extractor`` with ``SiftExtraction.use_gpu 1`` - the whole folder at once.

        One call means COLMAP's own single-camera mode spans every frame (the chunked pycolmap path
        needs `_import_once` for that), and its "Processed file [n/m]" records drive the bar.
        """
        args = ["feature_extractor",
                "--database_path", str(self.database_path),
                "--image_path", str(self.image_dir),
                "--ImageReader.camera_model", str(camera_model),
                "--ImageReader.single_camera", "1" if single_camera else "0",
                "--SiftExtraction.max_image_size", str(int(max_image_size)),
                "--SiftExtraction.max_num_features", str(int(max_num_features)),
                "--SiftExtraction.estimate_affine_shape",
                "1" if estimate_affine_shape else "0",
                "--SiftExtraction.domain_size_pooling", "1" if domain_size_pooling else "0",
                "--SiftExtraction.use_gpu", "1"]
        if mask_path:
            args.extend(["--ImageReader.mask_path", str(mask_path)])
            print(f"[pycolmap] Using masks from: {mask_path}")
        return self._bridge_run(args, "feature extraction (GPU)")

    def _bridge_match(self, kind: str, overlap: int = 10) -> bool:
        """``colmap sequential_matcher`` / ``exhaustive_matcher`` with ``SiftMatching.use_gpu 1``."""
        if kind == "sequential":
            args = ["sequential_matcher",
                    "--database_path", str(self.database_path),
                    "--SequentialMatching.overlap", str(int(overlap)),
                    "--SequentialMatching.loop_detection", "0",
                    "--SiftMatching.use_gpu", "1"]
        else:
            args = ["exhaustive_matcher",
                    "--database_path", str(self.database_path),
                    "--SiftMatching.use_gpu", "1"]
        return self._bridge_run(args, f"{kind} matching (GPU)")

    # ------------------------------------------------------- pipeline stages
    def feature_extractor(self, camera_model: str = "SIMPLE_RADIAL",
                          single_camera: bool = True, max_image_size: int = 3200,
                          max_num_features: int = 8192, use_gpu: bool = True,
                          mask_path: Optional[str] = None,
                          estimate_affine_shape: bool = False,
                          domain_size_pooling: bool = False) -> bool:
        """``colmap feature_extractor`` -> ``pycolmap.extract_features``.

        With `use_gpu` set and a pycolmap build that has no CUDA (every Windows wheel), the work is
        delegated to the downloaded CUDA COLMAP build - see `resolve_gpu_bridge`.
        """
        if self._bridge_active(use_gpu) and self._bridge_extract(
                camera_model, single_camera, max_image_size, max_num_features, mask_path,
                estimate_affine_shape, domain_size_pooling):
            return True
        module = self._module()
        if module is None:
            return False

        reader_options = {"camera_model": str(camera_model)}
        if mask_path:
            reader_options["mask_path"] = str(mask_path)
            print(f"[pycolmap] Using masks from: {mask_path}")
        extraction_options = {
            "max_image_size": int(max_image_size),
            "use_gpu": bool(use_gpu),
            "sift": {
                "max_num_features": int(max_num_features),
                "estimate_affine_shape": bool(estimate_affine_shape),
                "domain_size_pooling": bool(domain_size_pooling),
            },
        }
        camera_mode = (module.CameraMode.SINGLE if single_camera
                       else module.CameraMode.AUTO)
        device = self._device(module, self._effective_gpu(module, use_gpu))

        # Extract in chunks: one call per image is ~5x slower (thread setup +
        # database reopen per call), one call for everything shows no progress.
        names = self._image_names()
        total = max(1, len(names))
        chunk = max(4, total // EXTRACTION_UPDATES)
        batches: List[Optional[Sequence[str]]] = [
            names[start:start + chunk] for start in range(0, len(names), chunk)
        ] or [None]
        # ONE camera for the whole set: chunked extraction would otherwise create a camera
        # per chunk (COLMAP's SINGLE camera mode applies to the images of each call).
        if single_camera and len(batches) > 1:
            camera_id = self._import_once(module, reader_options)
            if camera_id is not None:
                reader_options = dict(reader_options, existing_camera_id=camera_id)
        done = 0
        try:
            for batch in batches:
                module.extract_features(
                    self.database_path, self.image_dir,
                    camera_mode=camera_mode,
                    reader_options=reader_options,
                    extraction_options=extraction_options,
                    device=device,
                    **({"image_names": list(batch)} if batch else {}),
                )
                done += len(batch) if batch else 1
                self._emit(done, total, f"feature extraction {done}/{total} images")
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] feature extraction failed: {exc}")
            return False
        return True

    def _pair_names(self, module, kind: str,
                    overlap: int) -> Optional[List[Tuple[str, str]]]:
        """COLMAP's own pairing, as image-name pairs (None when unavailable)."""
        try:
            database = module.Database.open(self.database_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] cannot open the database for pairing: {exc}")
            return None
        try:
            if kind == "sequential":
                options = module.SequentialPairingOptions()
                options.overlap = int(overlap)
                options.loop_detection = False
                generator = module.SequentialPairGenerator(options, database)
            else:
                generator = module.ExhaustivePairGenerator(
                    module.ExhaustivePairingOptions(), database)
            pairs = list(generator.all_pairs())
            images = database.read_all_images()
            # pycolmap 4.x returns a *list* of Image objects here (older builds a dict).
            if hasattr(images, "items"):
                names = {int(key): value.name for key, value in images.items()}
            else:
                names = {int(getattr(image, "image_id", index + 1)): image.name
                         for index, image in enumerate(images)}
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] pair generation failed ({exc})")
            return None
        finally:
            try:
                database.close()
            except Exception:  # noqa: BLE001
                pass

        converted: List[Tuple[str, str]] = []
        for pair in pairs:
            try:
                first, second = int(pair.image_id1), int(pair.image_id2)
            except Exception:  # noqa: BLE001 - tolerate other binding shapes
                try:
                    first, second = int(pair[0]), int(pair[1])
                except Exception:  # noqa: BLE001
                    continue
            name1, name2 = names.get(first), names.get(second)
            if name1 and name2:
                converted.append((name1, name2))
        return converted or None

    def _call_matcher(self, label: str, function, required: tuple, optional: dict) -> None:
        """Call a pycolmap matcher, tolerating binding differences in the keyword form.

        ``matching_options`` / ``pairing_options`` / ``device`` are passed as plain dicts,
        which every 4.x build accepts - but not every *option class* is exposed at the top
        level (pycolmap 4.2.1 has no ``SequentialMatchingOptions``, for example), and a
        keyword this build does not know must not kill the run. A rejected keyword is
        therefore retried once with the required arguments only. ``ENNDEE_COLMAP_VERBOSE=1``
        says which form was used; if the bare form fails as well the exception propagates
        to the caller's fallback.
        """
        try:
            function(*required, **optional)
        except Exception as exc:  # noqa: BLE001 - a binding difference, not a fatal error
            print(f"[pycolmap] {label}: option keyword rejected ({exc}) - retrying with "
                  f"the required arguments only")
            function(*required)
            if colmap_verbose():
                print(f"[pycolmap] {label}: bare form accepted")
            return
        if colmap_verbose():
            print(f"[pycolmap] {label}: full option form accepted")

    def _match(self, kind: str, use_gpu: bool, overlap: int = 10) -> bool:
        module = self._module()
        if module is None:
            return False
        matching_options = {"use_gpu": bool(use_gpu)}
        device = self._device(module, self._effective_gpu(module, use_gpu))

        # Matching in batches of pairs keeps COLMAP's pairing exactly (the pair
        # list comes from pycolmap's own generator) while feeding the progress
        # bar; the built-in matcher stays as the fallback.
        pairs = self._pair_names(module, kind, overlap)
        if pairs:
            total = len(pairs)
            batch_size = max(1, total // MATCHING_UPDATES)
            done = 0
            try:
                for start in range(0, total, batch_size):
                    chunk = pairs[start:start + batch_size]
                    listing = Path(self.workspace) / f"match_list_{kind}_{start:06d}.txt"
                    listing.write_text("\n".join(f"{a} {b}" for a, b in chunk),
                                       encoding="utf-8")
                    self._call_matcher(
                        f"chunked {kind} matching", module.match_image_pairs,
                        (self.database_path,),
                        {"matching_options": matching_options,
                         "pairing_options": {"match_list_path": str(listing)},
                         "device": device},
                    )
                    done += len(chunk)
                    self._emit(done, total, f"{kind} matching {done}/{total} pairs")
                return True
            except Exception as exc:  # noqa: BLE001
                print(f"[pycolmap] chunked {kind} matching failed ({exc}) - "
                      f"falling back to the built-in matcher")
                self._emit(0, 1, f"{kind} matching (built-in)")

        try:
            if kind == "sequential":
                self._call_matcher(
                    "sequential matching", module.match_sequential,
                    (self.database_path,),
                    {"matching_options": matching_options,
                     "pairing_options": {"overlap": int(overlap),
                                         "loop_detection": False},
                     "device": device},
                )
            else:
                self._call_matcher(
                    "exhaustive matching", module.match_exhaustive,
                    (self.database_path,),
                    {"matching_options": matching_options, "device": device},
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] {kind} matching failed: {exc}")
            return False
        self._emit(1, 1, f"{kind} matching")
        return True

    def sequential_matcher(self, use_gpu: bool = True, overlap: int = 10) -> bool:
        """``colmap sequential_matcher`` -> ``pycolmap.match_sequential`` (or the GPU bridge)."""
        if self._bridge_active(use_gpu) and self._bridge_match("sequential", overlap):
            return True
        return self._match("sequential", use_gpu, overlap)

    def exhaustive_matcher(self, use_gpu: bool = True) -> bool:
        """``colmap exhaustive_matcher`` -> ``pycolmap.match_exhaustive`` (or the GPU bridge)."""
        if self._bridge_active(use_gpu) and self._bridge_match("exhaustive"):
            return True
        return self._match("exhaustive", use_gpu)

    def mapper(self, backend: str = "glomap", min_num_matches: int = 15,
               num_threads: Optional[int] = None, **kwargs) -> bool:
        """``colmap global_mapper`` / ``glomap mapper`` -> ``pycolmap.global_mapping``.

        ``backend`` accepts the legacy names (``glomap``, ``colmap_global``) and the
        new ones (``global``, ``global_mapper``); ``incremental`` runs COLMAP's
        classic incremental mapper instead.

        The global pipeline always gets explicit solver flags, and both stages default
        to *false* regardless of ``has_cuda``. Both default to *true* inside COLMAP, but
        COLMAP's global mapper pins ``linear_solver_type = SPARSE_SCHUR`` together with
        ``auto_select_solver_type = false`` (``src/colmap/sfm/global_mapper.h``), so its
        GPU solver is never selected - a CUDA/Ceres build only reaches it with
        cuDSS/Caspar, which the standard builds do not have. Leaving the defaults on
        therefore buys nothing and logs two misleading "Requested to use GPU for bundle
        adjustment ... Falling back to CPU" warnings per run.

        Measured on the same 81-image set (mapping = global positioning + bundle
        adjustment; two runs per wheel):

        ==================  ==============  ================
        stage               CPU wheel       CUDA wheel
        ==================  ==============  ================
        mapping total       87.6 / 70.2 s   115.7 / 67.0 s
        ==================  ==============  ================

        The two runs of *one* build differ by more than the builds differ from each
        other (the positioning stage alone varied 21.7 s <-> 44.2 s for a single config),
        so the mapping is CPU-bound either way - the only lever is the problem size
        (``frame_step``, ``max_features``, ``max_image_size``, ``matcher``,
        ``sequential_overlap``). The GPU's win is the SIFT work: on the same set
        extraction went 8.85 s -> 1.65 s and sequential matching 47.42 s -> 0.89 s.

        Set ``ENNDEE_PYCOLMAP_GPU_BA=1`` to restore ``use_gpu = True`` for both stages -
        only meaningful for a COLMAP built with cuDSS/Caspar. ``auto_select_solver_type``
        is deliberately never touched (it changes the solver and does not help the
        default global-SfM BA). The ``incremental`` backend takes no ``mapper`` sub-tree
        and is left untouched.
        """
        module = self._module()
        if module is None:
            return False

        options = {"min_num_matches": int(min_num_matches)}
        if num_threads:
            options["num_threads"] = int(num_threads)
        if backend in GLOBAL_BACKENDS:
            # the sub-tree is ``GlobalPipelineOptions.mapper.<stage>`` - see
            # ``mapper.global_positioning`` / ``mapper.bundle_adjustment.ceres``.
            gpu_ba = gpu_ba_requested()
            options["mapper"] = {
                "global_positioning": {"use_gpu": gpu_ba},
                "bundle_adjustment": {"ceres": {"use_gpu": gpu_ba}},
            }
            if gpu_ba:
                print(f"[pycolmap] mapping solver: GPU requested via {GPU_BA_ENV}=1 - "
                      f"the GPU sparse solver needs a cuDSS-enabled COLMAP build")
            else:
                print("[pycolmap] mapping solver: CPU (SPARSE_SCHUR) - the GPU bundle "
                      "adjustment needs a cuDSS-enabled COLMAP build")
        self._emit(0, 1, f"{backend} mapping")
        try:
            if backend == INCREMENTAL_BACKEND:
                models = module.incremental_mapping(self.database_path, self.image_dir,
                                                    self.sparse_dir, options=options)
            elif backend in GLOBAL_BACKENDS:
                models = module.global_mapping(self.database_path, self.image_dir,
                                               self.sparse_dir, options=options)
            else:
                print(f"[pycolmap] unknown mapper_backend {backend!r} - use one of "
                      f"{GLOBAL_BACKENDS + (INCREMENTAL_BACKEND,)}")
                return False
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] {backend} mapping failed: {exc}")
            return False

        if not models:
            print(f"[pycolmap] {backend} mapping produced no reconstruction")
            return False
        self._emit(1, 1, f"{backend} mapping")
        # pycolmap writes the models into <sparse>/<index> itself; make sure a
        # readable model directory exists either way.
        model_path = self.get_sparse_model_path()
        if model_path is None:
            for index, model in models.items():
                model.write(self.sparse_dir / str(index))
            model_path = self.get_sparse_model_path()
        return model_path is not None


    # ==================================================================
    # Dense MVS: depth the sparse model itself is consistent with
    # ==================================================================

    def dense_depth_maps(self, sparse_path, dense_dir=None, *,
                         max_image_size: int = 1600,
                         geom_consistency: bool = True,
                         window_radius: int = 5,
                         num_samples: int = 15,
                         cache_size: int = 32,
                         log: Callable[[str], None] = print) -> dict:
        """Run COLMAP's dense MVS and read the depth maps back.

        This is the ``colmap image_undistorter`` -> ``colmap patch_match_stereo``
        pair, run in-process: the sparse model is undistorted into
        ``<workspace>/dense``, PatchMatch stereo fills that workspace with
        ``stereo/depth_maps/*.bin``, and every map is read through
        :class:`pycolmap.DepthMap`.

        Why this is worth the CUDA pass: unlike a feed-forward model, this depth
        is *photometrically fitted to the very poses the sparse model produced*.
        It therefore cannot disagree with them, which is what Lichtfeld's depth
        loss needs - a depth prior in another gauge fights the reconstruction, a
        prior in the same gauge can only reinforce it.  ``geom_consistency`` is
        the cross-view consistency filter: on it also writes the filtered
        ``.geometric`` maps (used when both exist), off only the raw
        ``.photometric`` ones.

        Returns ``{image_name: float32[H, W]}`` for every view PatchMatch wrote a
        map for, or ``{}`` when the stage could not run at all (no CUDA pycolmap,
        no undistorter, no sparse model).
        """
        module = self._module()
        if module is None:
            return {}
        if not self.workspace or not self.image_dir:
            print("[pycolmap] dense: no workspace - run the SfM stages first")
            return {}

        import shutil

        workspace = Path(self.workspace)
        dense = Path(dense_dir) if dense_dir else workspace / "dense"
        # a stale dense/ folder from an earlier run would be mixed into this one
        if dense.exists():
            shutil.rmtree(dense, ignore_errors=True)
        dense.mkdir(parents=True, exist_ok=True)

        sparse = Path(sparse_path)
        if not sparse.exists():
            print(f"[pycolmap] dense: sparse model not found: {sparse}")
            return {}

        self._emit(0, 3, "dense: undistort")
        log(f"Dense MVS: undistorting {sparse.name} -> {dense}")
        try:
            undistort = module.UndistortCameraOptions()
            if int(max_image_size) > 0:
                undistort.max_image_size = int(max_image_size)
            module.undistort_images(str(dense), str(sparse), str(self.image_dir),
                                    undistort_options=undistort)
        except Exception as exc:  # noqa: BLE001 - dense is optional, never fatal
            print(f"[pycolmap] dense: undistortion failed: {exc}")
            return {}

        options = module.PatchMatchOptions()
        try:
            options.max_image_size = int(max_image_size) if int(max_image_size) > 0 else -1
            options.geom_consistency = bool(geom_consistency)
            options.window_radius = int(window_radius)
            options.num_samples = int(num_samples)
            options.cache_size = int(cache_size)
        except Exception:  # noqa: BLE001 - binding differences must not kill the run
            print("[pycolmap] dense: some PatchMatch options were rejected - "
                  "using COLMAP's defaults for those")

        self._emit(1, 3, "dense: patch match")
        log("Dense MVS: PatchMatch stereo on the GPU - this is the slow stage")
        try:
            module.patch_match_stereo(str(dense), options=options)
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] dense: patch_match_stereo failed: {exc}")
            return {}

        self._emit(2, 3, "dense: reading depth maps")
        return self.read_dense_depth_maps(dense, sparse, log=log)

    @staticmethod
    def _warp_to_original(depth, undistorted_camera, original_camera, target_hw):
        """Resample an undistorted depth map onto the original (distorted) grid.

        PatchMatch only runs on the undistorted workspace, so its maps live on a
        different pixel grid *and* in a different projection than the frames the
        dataset exports.  A plain resize would be wrong: for the SIMPLE_RADIAL
        model of a 3456x2304 orbit the radial term moves a corner pixel by more
        than a dozen pixels, which is exactly the kind of misalignment a
        per-pixel depth loss notices.

        The warp is therefore inverted exactly.  Every target pixel becomes a ray
        through the **original** camera, that ray is projected with the
        **undistorted** camera, and the depth is sampled there.  Depth is the
        z-coordinate along the camera axis, which undistortion does not change, so
        the value transfers unchanged.  Target pixels whose ray falls outside the
        undistorted image get ``0`` - Lichtfeld's "no depth".
        """
        import numpy as np
        import torch

        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        grid_y, grid_x = np.meshgrid(np.arange(target_h, dtype=np.float64),
                                     np.arange(target_w, dtype=np.float64),
                                     indexing="ij")
        pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        # original pixel -> ray, ray -> undistorted pixel (both vectorised)
        rays = np.asarray(original_camera.cam_from_img(pixels), dtype=np.float64)
        rays = np.concatenate([rays, np.ones((rays.shape[0], 1))], axis=1)
        projected = np.asarray(undistorted_camera.img_from_cam(rays),
                               dtype=np.float64)

        width = max(int(undistorted_camera.width) - 1, 1)
        height = max(int(undistorted_camera.height) - 1, 1)
        sample_x = (projected[:, 0] / width) * 2.0 - 1.0
        sample_y = (projected[:, 1] / height) * 2.0 - 1.0

        source = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))
        # grid_sample wants [N, H_out, W_out, 2] - a flat [N, 2] grid would be read
        # as a 1xN output image
        grid = np.stack([sample_x, sample_y], axis=1).reshape(target_h, target_w, 2)
        warped = torch.nn.functional.grid_sample(
            source[None, None], torch.from_numpy(grid.astype(np.float32))[None],
            mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped[0, 0].numpy().astype(np.float32)

    @staticmethod
    def read_dense_depth_maps(dense_dir, sparse_path=None,
                              log: Callable[[str], None] = print) -> dict:
        """Read ``<dense>/stereo/depth_maps/*.bin`` into ``{image_name: [H, W]}``.

        COLMAP names each file after the image and appends the map kind, so one
        view can be ``0001.png.photometric.bin`` *and* ``0001.png.geometric.bin``.
        The filtered geometric map wins when both are present.

        With ``sparse_path`` (the distorted model the SfM stage produced) the maps
        are warped back onto the original image grid - see
        :meth:`_warp_to_original`.  Without it they stay in the undistorted frame,
        where they do **not** line up with the exported images.
        """
        import numpy as np

        depth_maps_dir = Path(dense_dir) / "stereo" / "depth_maps"
        index = {}
        if depth_maps_dir.is_dir():
            for path in sorted(depth_maps_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() != ".bin":
                    continue
                name = path.name[: -len(".bin")]
                for kind in (".geometric", ".photometric"):
                    if name.endswith(kind):
                        name = name[: -len(kind)]
                        break
                index.setdefault(name, path)

        if not index:
            print(f"[pycolmap] dense: no depth maps in {depth_maps_dir}")
            return {}

        module = import_pycolmap()
        original_cameras, undistorted_cameras = {}, {}
        if sparse_path is not None:
            try:
                original_model = module.Reconstruction()
                original_model.read(str(sparse_path))
                original_cameras = {image.name: image.camera
                                    for image in original_model.images.values()}
                dense_model = module.Reconstruction()
                dense_model.read(str(Path(dense_dir) / "sparse"))
                undistorted_cameras = {image.name: image.camera
                                       for image in dense_model.images.values()}
            except Exception as exc:  # noqa: BLE001 - fall back to the raw maps
                print(f"[pycolmap] dense: cannot read the camera models ({exc}) - "
                      "depth maps stay in the undistorted frame")

        results, warped = {}, 0
        for name, path in index.items():
            depth_map = module.DepthMap()
            try:
                depth_map.read(str(path))
                depth = depth_map.to_array().astype(np.float32)
            except Exception as exc:  # noqa: BLE001 - skip the odd broken map
                print(f"[pycolmap] dense: cannot read {path.name}: {exc}")
                continue

            original_camera = original_cameras.get(name)
            undistorted_camera = undistorted_cameras.get(name)
            if (original_camera is not None and undistorted_camera is not None
                    and not original_camera.is_undistorted()):
                depth = PyColmapWrapper._warp_to_original(
                    depth, undistorted_camera, original_camera,
                    (original_camera.height, original_camera.width))
                warped += 1
            results[name] = depth

        detail = f", {warped} warped back onto the original camera model" if warped else ""
        log(f"Dense MVS: {len(results)} depth maps read from {depth_maps_dir}{detail}")
        return results

    # ------------------------------------------------------------------ A: fusion
    def dense_fused_cloud(self, dense_dir=None, max_points: int = 400000, *,
                          log: Callable[[str], None] = print) -> dict:
        """``colmap stereo_fusion`` -> one multi-view consistent point cloud.

        Every PatchMatch depth map is fused into a single cloud **in the sparse model's
        world frame**, with the per-view outliers rejected. Unlike a mesh this constrains
        nothing downstream - it only seeds the Gaussians - so it cannot freeze an error
        in, which makes it the safer of the two dense products.

        ``output_type="ply"`` is load-bearing: the default ``"bin"`` writes a COLMAP
        model *folder* at ``output_path`` and fails outright when handed a file path
        (``Check failed: ExistsDir(path_val)``).

        Returns ``{"points": [N,3], "colors": [N,3] in [0,1], "total": int, "ply": Path}``
        or ``{}`` when the fusion could not run.
        """
        module = self._module()
        if module is None:
            return {}
        if not self.workspace:
            print("[pycolmap] dense: no workspace - run the SfM stages first")
            return {}

        dense = Path(dense_dir) if dense_dir else Path(self.workspace) / "dense"
        if not dense.is_dir():
            print(f"[pycolmap] dense: no dense workspace at {dense}")
            return {}
        ply = dense / "fused.ply"

        self._emit(0, 1, "dense: fusing")
        log("Dense MVS: fusing the depth maps into one point cloud")
        try:
            fused = module.stereo_fusion(str(ply), str(dense), input_type="geometric",
                                         output_type="ply",
                                         options=module.StereoFusionOptions())
        except Exception as exc:  # noqa: BLE001 - fusion is optional, never fatal
            print(f"[pycolmap] dense: stereo_fusion failed: {exc}")
            return {}

        points, colors = points_from_reconstruction(module, fused, ply)
        if len(points) == 0:
            print("[pycolmap] dense: stereo_fusion produced no points")
            return {}

        total = int(len(points))
        keep = stride_subset(total, int(max_points))
        if len(keep) != total:
            points, colors = points[keep], colors[keep]
        self._emit(1, 1, "dense: fused")
        log(f"Dense MVS: fused {total} points -> {len(keep)} kept "
            f"(cap {int(max_points)}) -> {ply.name}")
        return {"points": points, "colors": colors, "total": total, "ply": ply}

    # ------------------------------------------------------------- B: surface
    def _colmap_executable(self):
        """A ``colmap.exe`` for the CLI-only stages, taken from the GPU bridge.

        The bridge path is often the ``COLMAP.bat`` launcher, whose real binary sits in
        the sibling ``bin/`` folder, so both shapes are resolved.
        """
        raw = str(getattr(self, "colmap_path", "") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if path.suffix.lower() == ".exe" and path.is_file():
            return path
        for candidate in (path.parent / "colmap.exe", path.parent / "bin" / "colmap.exe"):
            if candidate.is_file():
                return candidate
        return None

    def dense_mesh(self, dense_dir=None, method: str = "poisson",
                   fused_ply=None, *, log: Callable[[str], None] = print):
        """A triangle surface for the dense geometry, or ``None``.

        ``poisson`` runs in-process from the fused cloud. ``trim`` MUST be overridden:
        this COLMAP build defaults it to 10, which crops the result to **12 vertices /
        20 faces** on a real scene - measured. At ``trim=0`` the same input yields
        ~167k vertices / ~334k faces. ``depth`` keeps COLMAP's default 13.

        ``delaunay`` shells out to the bundled COLMAP because pycolmap 4.2.1 exposes
        ``DelaunayMeshingOptions`` but **no** ``delaunay_meshing`` function (verified:
        ``AttributeError``). Its input is the *dense workspace*, not the fused PLY.
        """
        module = self._module()
        if module is None:
            return None
        dense = Path(dense_dir) if dense_dir else (
            Path(self.workspace) / "dense" if self.workspace else None)
        if dense is None or not dense.is_dir():
            print(f"[pycolmap] dense: no dense workspace at {dense}")
            return None

        if str(method).lower() == "delaunay":
            return self._delaunay_mesh(dense, log=log)

        source = Path(fused_ply) if fused_ply else dense / "fused.ply"
        if not source.is_file():
            print(f"[pycolmap] dense: no fused cloud at {source} - run the fusion first")
            return None
        target = dense / "mesh_poisson.ply"

        self._emit(0, 1, "dense: meshing")
        log("Dense MVS: Poisson surface reconstruction from the fused cloud")
        try:
            options = module.PoissonMeshingOptions()
            options.trim = 0.0          # see the docstring - the default kills the mesh
            module.poisson_meshing(str(source), str(target), options=options)
        except Exception as exc:  # noqa: BLE001 - meshing is optional, never fatal
            print(f"[pycolmap] dense: poisson_meshing failed: {exc}")
            return None
        self._emit(1, 1, "dense: meshed")
        if not target.is_file():
            print("[pycolmap] dense: poisson_meshing wrote no file")
            return None
        log(f"Dense MVS: Poisson surface -> {target.name}")
        return target

    def _delaunay_mesh(self, dense: Path, *, log: Callable[[str], None] = print):
        """``colmap delaunay_mesher --input_path <dense> --input_type dense``."""
        import subprocess

        executable = self._colmap_executable()
        if executable is None:
            print("[pycolmap] dense: delaunay needs the bundled COLMAP binary "
                  "(none resolved) - use mesh_method='poisson'")
            return None
        target = dense / "mesh_delaunay.ply"
        log(f"Dense MVS: Delaunay surface reconstruction via {executable.name}")
        try:
            proc = subprocess.run(
                [str(executable), "delaunay_mesher",
                 "--input_path", str(dense), "--input_type", "dense",
                 "--output_path", str(target)],
                capture_output=True, text=True, timeout=1800)
        except Exception as exc:  # noqa: BLE001
            print(f"[pycolmap] dense: delaunay_mesher failed to start: {exc}")
            return None
        if not target.is_file():
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-4:]
            print(f"[pycolmap] dense: delaunay_mesher produced no mesh "
                  f"(rc={proc.returncode}): {' | '.join(tail)}")
            return None
        log(f"Dense MVS: Delaunay surface -> {target.name}")
        return target


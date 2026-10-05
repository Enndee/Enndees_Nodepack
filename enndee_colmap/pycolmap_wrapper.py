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

from .colmap_wrapper import (COLMAPWrapper, GLOG_LEVEL_WARNING, VERBOSE_ENV, _timeout_from_env,
                             colmap_child_env, colmap_verbose, run_streaming_command)
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


def gpu_bridge_enabled() -> bool:
    """False when the user switched the CUDA bridge off via ``ENNDEE_PYCOLMAP_GPU_BRIDGE=0``."""
    return not str(os.environ.get(BRIDGE_ENV) or "").strip() in ("0", "false", "no", "off")


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


def pycolmap_info() -> dict:
    """``{'available': bool, 'version': str, 'cuda': bool}`` for reporting."""
    module = import_pycolmap()
    if module is None:
        return {"available": False, "version": "", "cuda": False}
    return {
        "available": True,
        "version": str(getattr(module, "__version__", "?")),
        "cuda": bool(getattr(module, "has_cuda", False)),
    }


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
        """True when this pycolmap build can really use the GPU."""
        has_cuda = bool(getattr(module, "has_cuda", False))
        if use_gpu and not has_cuda:
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
                    module.match_image_pairs(
                        self.database_path,
                        matching_options=matching_options,
                        pairing_options={"match_list_path": str(listing)},
                        device=device,
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
                module.match_sequential(
                    self.database_path,
                    matching_options=matching_options,
                    pairing_options={"overlap": int(overlap),
                                     "loop_detection": False},
                    device=device,
                )
            else:
                module.match_exhaustive(self.database_path,
                                        matching_options=matching_options,
                                        device=device)
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

        Without CUDA in the build the global pipeline is told not to ask for the GPU
        solvers: ``global_positioning.use_gpu`` and ``bundle_adjustment.ceres.use_gpu``
        default to *true*, and every run then logs two "Requested to use GPU for bundle
        adjustment, but COLMAP was compiled without CUDA support - falling back to the
        CPU" warnings that look like errors but only repeat what ``has_cuda`` already
        says. Nothing is lost: there is no GPU to fall back *from*.
        """
        module = self._module()
        if module is None:
            return False

        options = {"min_num_matches": int(min_num_matches)}
        if num_threads:
            options["num_threads"] = int(num_threads)
        if backend in GLOBAL_BACKENDS and not getattr(module, "has_cuda", False):
            # the sub-tree is ``GlobalPipelineOptions.mapper.<stage>`` - see
            # ``mapper.global_positioning`` / ``mapper.bundle_adjustment.ceres``.
            options["mapper"] = {
                "global_positioning": {"use_gpu": False},
                "bundle_adjustment": {"ceres": {"use_gpu": False}},
            }
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


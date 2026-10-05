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
   The official PyPI wheels are built **without CUDA on Windows** (CUDA wheels are
   Linux-only, see the pycolmap docs). Feature extraction/matching and the bundle
   adjustment therefore run on the CPU here; ``use_gpu`` is honoured automatically
   as soon as a CUDA-enabled pycolmap build is installed (``pycolmap.has_cuda``).
"""

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .colmap_wrapper import COLMAPWrapper
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


def import_pycolmap():
    """Return the ``pycolmap`` module or None (never raises)."""
    try:
        import pycolmap  # type: ignore
    except Exception:  # noqa: BLE001 - any import problem means "not available"
        return None
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
                 progress_hook: Optional[ProgressHook] = None):
        # No COLMAP/GLOMAP binaries: set up the base attributes by hand because
        # COLMAPWrapper.__init__ insists on a real colmap_path.
        self.colmap_path = None
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
            print("[pycolmap] this build has no CUDA support (the official Windows "
                  "wheels are CPU only) - running on the CPU")
        return bool(use_gpu) and has_cuda

    @staticmethod
    def _device(module, effective_gpu: bool):
        device = getattr(module, "Device", None)
        if device is None:
            return None
        return device.cuda if effective_gpu else device.cpu

    # ------------------------------------------------------- pipeline stages
    def feature_extractor(self, camera_model: str = "SIMPLE_RADIAL",
                          single_camera: bool = True, max_image_size: int = 3200,
                          max_num_features: int = 8192, use_gpu: bool = True,
                          mask_path: Optional[str] = None,
                          estimate_affine_shape: bool = False,
                          domain_size_pooling: bool = False) -> bool:
        """``colmap feature_extractor`` -> ``pycolmap.extract_features``."""
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
            names = {int(key): value.name
                     for key, value in database.read_all_images().items()}
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
        """``colmap sequential_matcher`` -> ``pycolmap.match_sequential``."""
        return self._match("sequential", use_gpu, overlap)

    def exhaustive_matcher(self, use_gpu: bool = True) -> bool:
        """``colmap exhaustive_matcher`` -> ``pycolmap.match_exhaustive``."""
        return self._match("exhaustive", use_gpu)

    def mapper(self, backend: str = "glomap", min_num_matches: int = 15,
               num_threads: Optional[int] = None, **kwargs) -> bool:
        """``colmap global_mapper`` / ``glomap mapper`` -> ``pycolmap.global_mapping``.

        ``backend`` accepts the legacy names (``glomap``, ``colmap_global``) and the
        new ones (``global``, ``global_mapper``); ``incremental`` runs COLMAP's
        classic incremental mapper instead.
        """
        module = self._module()
        if module is None:
            return False

        options = {"min_num_matches": int(min_num_matches)}
        if num_threads:
            options["num_threads"] = int(num_threads)
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


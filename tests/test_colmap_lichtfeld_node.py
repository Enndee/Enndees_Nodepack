"""Tests for the native pycolmap node (COLMAP for Lichtfeld) and its helpers.

Everything runs against a *fake* pycolmap module, so the suite needs neither the
real bindings nor a GPU - it checks the option/argument mapping, the backend hooks
and the accelerator logic.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR))

import colmap_lichtfeld_node as native          # noqa: E402
import enndee_accelerators as accelerators      # noqa: E402
import glomap_lichtfeld_node as binary          # noqa: E402
from enndee_colmap import pycolmap_wrapper      # noqa: E402


class FakePycolmap:
    """Records every call the wrapper makes and returns fake models."""

    def __init__(self, has_cuda=False, models=(0,), image_names=(), pair_error=False,
                 camera_id=1):
        self.has_cuda = has_cuda
        self.__version__ = "9.9.9-test"
        self.models = models
        self.image_names = list(image_names)
        self.pair_error = pair_error
        self.camera_id = camera_id
        self.import_error = False
        #: True = the matcher raises TypeError when any option keyword is passed (a
        #: binding that does not know ``device`` / ``pairing_options``).
        self.reject_kwargs = False
        self.calls = []
        #: calls that were made without the optional keywords (the retry form).
        self.bare_calls = []
        self.closed_databases = []
        self.pair_options = []
        self.pair_databases = []

        class Device:
            cpu = "cpu"
            cuda = "cuda"

        class CameraMode:
            SINGLE = "SINGLE"
            AUTO = "AUTO"

        self.Device = Device
        self.CameraMode = CameraMode
        # pycolmap's pairing API (options are plain objects, the generators are
        # constructed from (options, database) and expose all_pairs()).
        self.SequentialPairingOptions = lambda: types.SimpleNamespace(
            overlap=0, loop_detection=True, kind="sequential")
        self.ExhaustivePairingOptions = lambda: types.SimpleNamespace(kind="exhaustive")
        self.SequentialPairGenerator = self._pair_generator
        self.ExhaustivePairGenerator = self._pair_generator
        self.Database = types.SimpleNamespace(open=self._open_database)

    # ------------------------------------------------------------ database
    def _open_database(self, path):
        fake = self

        class Database:
            def read_all_images(self):
                class Image:
                    def __init__(self, name, camera_id, index):
                        self.name = name
                        self.camera_id = camera_id
                        self.image_id = index

                return [Image(name, fake.camera_id, index + 1)
                        for index, name in enumerate(fake.image_names)]

            def close(self):
                fake.closed_databases.append(str(path))

        return Database()

    def import_images(self, database, images, **kwargs):
        self.calls.append(("import_images", str(database), str(images), kwargs))
        if self.import_error:
            raise RuntimeError("import exploded")

    # ------------------------------------------------------------- pairing
    def _pair_generator(self, options, database):
        fake = self
        fake.pair_options.append(options)
        fake.pair_databases.append(database)

        class Generator:
            def all_pairs(self):
                if fake.pair_error:
                    raise RuntimeError("pair generator exploded")

                class ImagePair:
                    def __init__(self, first, second):
                        self.image_id1 = first
                        self.image_id2 = second

                return [ImagePair(index + 1, index + 2)
                        for index in range(len(fake.image_names) - 1)]

        return Generator()

    def extract_features(self, database, images, **kwargs):
        self.calls.append(("extract", str(database), str(images), kwargs))

    def _record_match(self, name, database, kwargs):
        """Record one matcher call - and reject the keyword form when asked to."""
        if self.reject_kwargs and kwargs:
            raise TypeError(f"{name}() got an unexpected keyword argument 'device'")
        if kwargs:
            self.calls.append((name, str(database), kwargs))
        else:
            self.bare_calls.append((name, str(database)))

    def match_sequential(self, database, **kwargs):
        self._record_match("sequential", database, kwargs)

    def match_exhaustive(self, database, **kwargs):
        self._record_match("exhaustive", database, kwargs)

    def match_image_pairs(self, database, **kwargs):
        self._record_match("match_image_pairs", database, kwargs)

    def _models(self):
        class Model:
            def __init__(self, index):
                self.index = index

            def write(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "cameras.bin").write_bytes(b"")

        return {index: Model(index) for index in self.models}

    def global_mapping(self, database, images, output, **kwargs):
        self.calls.append(("global", str(database), str(images), str(output), kwargs))
        return self._models()

    def incremental_mapping(self, database, images, output, **kwargs):
        self.calls.append(("incremental", str(database), str(images), str(output), kwargs))
        return self._models()

    def call(self, name):
        for entry in self.calls:
            if entry[0] == name:
                return entry
        raise AssertionError(f"{name} was not called (calls={self.calls})")


class PyColmapWrapperTests(unittest.TestCase):
    """The wrapper must map the CLI flags onto the native option dicts."""

    IMAGES = ("frame_0001.png", "frame_0002.png", "frame_0003.png",
              "frame_0004.png", "frame_0005.png")

    def setUp(self):
        self.fake = FakePycolmap(image_names=self.IMAGES)
        self.patch = mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                       return_value=self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.progress = []
        self.wrapper = pycolmap_wrapper.PyColmapWrapper(progress_hook=self._progress)
        self.wrapper.setup_workspace()
        self.addCleanup(self.wrapper.cleanup_workspace)
        for name in self.IMAGES:
            (self.wrapper.image_dir / name).write_bytes(b"")
        (self.wrapper.sparse_dir / "0").mkdir(exist_ok=True)
        (self.wrapper.sparse_dir / "0" / "cameras.bin").write_bytes(b"")

    def _progress(self, value, total, label):
        self.progress.append((value, total, label))

    def test_no_binaries_are_needed(self):
        self.assertIsNone(self.wrapper.colmap_path)
        self.assertIsNone(self.wrapper.glomap_path)

    def test_feature_extractor_maps_every_option(self):
        ok = self.wrapper.feature_extractor(camera_model="SIMPLE_PINHOLE",
                                           max_num_features=10000,
                                           max_image_size=1024,
                                           mask_path="C:/masks",
                                           use_gpu=False,
                                           estimate_affine_shape=True,
                                           domain_size_pooling=True)
        self.assertTrue(ok)
        _, database, images, kwargs = self.fake.call("extract")
        self.assertEqual(kwargs["reader_options"]["camera_model"], "SIMPLE_PINHOLE")
        self.assertEqual(kwargs["reader_options"]["mask_path"], "C:/masks")
        self.assertEqual(kwargs["extraction_options"]["max_image_size"], 1024)
        self.assertEqual(kwargs["extraction_options"]["sift"]["max_num_features"], 10000)
        self.assertTrue(kwargs["extraction_options"]["sift"]["estimate_affine_shape"])
        self.assertTrue(kwargs["extraction_options"]["sift"]["domain_size_pooling"])
        self.assertEqual(kwargs["camera_mode"], "SINGLE")
        self.assertEqual(kwargs["device"], "cpu")
        self.assertTrue(database.endswith("database.db"))
        self.assertTrue(images.endswith("images"))
        # chunked extraction: 5 frames in batches of 4 -> 2 calls, same options
        extracts = [entry for entry in self.fake.calls if entry[0] == "extract"]
        self.assertEqual(len(extracts), 2)
        self.assertEqual([len(entry[3]["image_names"]) for entry in extracts], [4, 1])
        for entry in extracts:
            self.assertEqual(entry[3]["reader_options"], kwargs["reader_options"])
            self.assertEqual(entry[3]["extraction_options"], kwargs["extraction_options"])
        # progress: increasing, measured in frames
        self.assertEqual([value for value, _total, _label in self.progress], [4, 5])
        self.assertTrue(all(total == 5 for _value, total, _label in self.progress))

    def test_gpu_is_only_requested_when_the_build_has_cuda(self):
        self.wrapper.feature_extractor(use_gpu=True)
        self.assertEqual(self.fake.call("extract")[3]["device"], "cpu")

        self.fake.calls.clear()
        self.fake.has_cuda = True
        self.wrapper.feature_extractor(use_gpu=True)
        self.assertEqual(self.fake.call("extract")[3]["device"], "cuda")

    def test_chunked_extraction_pins_one_camera(self):
        """113 frames came out with 13 cameras before: import once, pin that camera."""
        self.assertTrue(self.wrapper.feature_extractor(camera_model="SIMPLE_PINHOLE"))
        _, _, _, kwargs = self.fake.call("import_images")
        self.assertEqual(kwargs["camera_mode"], "SINGLE")
        self.assertEqual(kwargs["options"]["camera_model"], "SIMPLE_PINHOLE")
        extracts = [entry for entry in self.fake.calls if entry[0] == "extract"]
        self.assertEqual(len(extracts), 2)
        for entry in extracts:                     # every chunk reuses the imported camera
            self.assertEqual(entry[3]["reader_options"]["existing_camera_id"], 1)

    def test_one_chunk_needs_no_extra_import(self):
        for name in self.IMAGES:
            (self.wrapper.image_dir / name).unlink()
        (self.wrapper.image_dir / "only.png").write_bytes(b"")
        self.assertTrue(self.wrapper.feature_extractor())
        self.assertEqual([entry for entry in self.fake.calls if entry[0] == "import_images"], [])
        self.assertNotIn("existing_camera_id", self.fake.call("extract")[3]["reader_options"])

    def test_import_failure_falls_back_to_per_chunk_cameras(self):
        self.fake.import_error = True
        self.assertTrue(self.wrapper.feature_extractor())
        self.assertNotIn("existing_camera_id", self.fake.call("extract")[3]["reader_options"])

    def test_sequential_matching_uses_colmaps_own_pairing(self):
        self.assertTrue(self.wrapper.sequential_matcher(use_gpu=False, overlap=15))
        self.assertEqual(self.fake.pair_options[0].overlap, 15)
        self.assertFalse(self.fake.pair_options[0].loop_detection)
        _, _, kwargs = self.fake.call("match_image_pairs")
        self.assertEqual(kwargs["matching_options"]["use_gpu"], False)
        listings = [Path(entry[2]["pairing_options"]["match_list_path"])
                    for entry in self.fake.calls if entry[0] == "match_image_pairs"]
        # one batch per pair here (4 pairs / 40 updates -> batch size 1)
        self.assertEqual(len(listings), len(self.IMAGES) - 1)
        written = "\n".join(path.read_text(encoding="utf-8").strip() for path in listings)
        self.assertEqual(written,
                         "\n".join(f"{self.IMAGES[i]} {self.IMAGES[i + 1]}"
                                   for i in range(len(self.IMAGES) - 1)))
        # the database is opened for the pairing and closed again
        self.assertEqual(self.fake.closed_databases, [str(self.wrapper.database_path)])
        self.assertEqual(self.progress[-1],
                         (4, 4, "sequential matching 4/4 pairs"))

    def test_exhaustive_matcher(self):
        self.assertTrue(self.wrapper.exhaustive_matcher(use_gpu=False))
        self.assertEqual(self.fake.pair_options[0].kind, "exhaustive")
        self.fake.call("match_image_pairs")

    def test_matching_falls_back_to_the_builtin_matcher(self):
        self.fake.pair_error = True
        self.assertTrue(self.wrapper.sequential_matcher(use_gpu=False, overlap=7))
        _, _, kwargs = self.fake.call("sequential")
        self.assertEqual(kwargs["pairing_options"]["overlap"], 7)
        self.assertEqual(kwargs["matching_options"]["use_gpu"], False)
        self.assertEqual(self.progress[-1], (1, 1, "sequential matching"))

    def test_builtin_matching_retries_without_rejected_option_keywords(self):
        """A binding that rejects ``device``/``pairing_options`` must not kill the run."""
        self.fake.pair_error = True                 # force the built-in matcher path
        self.fake.reject_kwargs = True
        self.assertTrue(self.wrapper.sequential_matcher(use_gpu=False, overlap=5))
        self.assertEqual(self.fake.calls, [])       # the keyword form was refused
        self.assertEqual(self.fake.bare_calls,
                         [("sequential", str(self.wrapper.database_path))])
        self.assertEqual(self.progress[-1], (1, 1, "sequential matching"))

    def test_chunked_matching_retries_without_rejected_option_keywords(self):
        self.fake.reject_kwargs = True
        self.assertTrue(self.wrapper.exhaustive_matcher(use_gpu=False))
        self.assertEqual([entry[0] for entry in self.fake.bare_calls],
                         ["match_image_pairs"] * (len(self.IMAGES) - 1))
        self.assertEqual(self.progress[-1],
                         (4, 4, "exhaustive matching 4/4 pairs"))

    def test_matching_still_fails_when_even_the_bare_form_fails(self):
        self.fake.pair_error = True
        self.fake.reject_kwargs = True

        def boom(*args, **kwargs):
            raise RuntimeError("no matcher at all")

        self.fake.match_sequential = boom
        self.assertFalse(self.wrapper.sequential_matcher(use_gpu=False))

    def test_mapper_reports_progress(self):
        self.assertTrue(self.wrapper.mapper(backend="global"))
        self.assertEqual(self.progress[0], (0, 1, "global mapping"))
        self.assertEqual(self.progress[-1], (1, 1, "global mapping"))

    def test_mapper_global_is_the_glomap_pipeline(self):
        for backend in ("global", "glomap", "colmap_global", "global_mapper"):
            self.fake.calls.clear()
            self.assertTrue(self.wrapper.mapper(backend=backend), backend)
            self.fake.call("global")

    def test_mapper_incremental(self):
        self.assertTrue(self.wrapper.mapper(backend="incremental"))
        self.fake.call("incremental")

    def test_mapper_rejects_unknown_backends(self):
        self.assertFalse(self.wrapper.mapper(backend="nonsense"))

    def test_mapper_reports_failure_without_a_model(self):
        self.fake.models = ()
        self.assertFalse(self.wrapper.mapper(backend="global"))

    def test_mapper_disables_the_gpu_solvers_without_cuda(self):
        """A CPU-only build must not ask for the GPU solvers (those two warnings)."""
        self.assertTrue(self.wrapper.mapper(backend="global"))
        options = self.fake.call("global")[4]["options"]
        self.assertEqual(options["mapper"]["global_positioning"]["use_gpu"], False)
        self.assertEqual(options["mapper"]["bundle_adjustment"]["ceres"]["use_gpu"], False)

    def test_mapper_also_disables_the_gpu_solvers_with_cuda(self):
        """A CUDA build maps on the CPU too - the GPU solver needs cuDSS, not CUDA."""
        self.fake.has_cuda = True
        self.assertTrue(self.wrapper.mapper(backend="global"))
        options = self.fake.call("global")[4]["options"]
        self.assertEqual(options["mapper"]["global_positioning"]["use_gpu"], False)
        self.assertEqual(options["mapper"]["bundle_adjustment"]["ceres"]["use_gpu"], False)

    def test_mapper_never_touches_auto_select_solver_type(self):
        """It changes the solver and does not help the default global-SfM BA."""
        self.assertTrue(self.wrapper.mapper(backend="global"))
        ceres = self.fake.call("global")[4]["options"]["mapper"]["bundle_adjustment"]["ceres"]
        self.assertNotIn("auto_select_solver_type", ceres)

    def test_mapper_reports_the_cpu_solver_once(self):
        """One clear line instead of the two misleading COLMAP warnings."""
        with mock.patch("builtins.print") as printed:
            self.assertTrue(self.wrapper.mapper(backend="global"))
        lines = [call.args[0] for call in printed.call_args_list if call.args]
        solver = [line for line in lines if "mapping solver" in line]
        self.assertEqual(len(solver), 1)
        self.assertIn("CPU (SPARSE_SCHUR)", solver[0])
        self.assertIn("cuDSS", solver[0])

    def test_mapper_gpu_ba_switch_restores_the_gpu_solvers(self):
        """``ENNDEE_PYCOLMAP_GPU_BA=1`` is the escape hatch for a cuDSS/Caspar build."""
        with mock.patch.dict("os.environ", {pycolmap_wrapper.GPU_BA_ENV: "1"}), \
                mock.patch("builtins.print") as printed:
            self.assertTrue(self.wrapper.mapper(backend="global"))
        options = self.fake.call("global")[4]["options"]
        self.assertEqual(options["mapper"]["global_positioning"]["use_gpu"], True)
        self.assertEqual(options["mapper"]["bundle_adjustment"]["ceres"]["use_gpu"], True)
        lines = [call.args[0] for call in printed.call_args_list if call.args]
        solver = [line for line in lines if "mapping solver" in line]
        self.assertEqual(len(solver), 1)
        self.assertIn("GPU requested", solver[0])

    def test_mapper_incremental_is_left_alone(self):
        """``IncrementalPipelineOptions`` has no ``mapper`` sub-tree - don't send one."""
        self.assertTrue(self.wrapper.mapper(backend="incremental"))
        self.assertNotIn("mapper", self.fake.call("incremental")[4]["options"])

    def test_missing_pycolmap_is_reported_not_raised(self):
        with mock.patch.object(pycolmap_wrapper, "import_pycolmap", return_value=None):
            self.assertFalse(self.wrapper.feature_extractor())
            self.assertFalse(self.wrapper.sequential_matcher())
            self.assertFalse(self.wrapper.exhaustive_matcher())
            self.assertFalse(self.wrapper.mapper())

    def test_pycolmap_info_shape(self):
        info = pycolmap_wrapper.pycolmap_info()
        self.assertEqual(info, {"available": True, "version": "9.9.9-test",
                                "cuda": False, "device_name": "",
                                "sift_on_gpu": False, "mapping_on_gpu": False})

    def test_pycolmap_info_reports_a_cuda_build(self):
        self.fake.has_cuda = True
        with mock.patch.object(pycolmap_wrapper, "_cuda_device_name",
                               return_value="RTX 5090"):
            info = pycolmap_wrapper.pycolmap_info()
        self.assertTrue(info["cuda"])
        self.assertEqual(info["device_name"], "RTX 5090")
        self.assertTrue(info["sift_on_gpu"])
        self.assertFalse(info["mapping_on_gpu"])       # the GPU solver needs cuDSS

    def test_pycolmap_info_reports_the_gpu_ba_switch(self):
        self.fake.has_cuda = True
        with mock.patch.dict("os.environ", {pycolmap_wrapper.GPU_BA_ENV: "1"}):
            info = pycolmap_wrapper.pycolmap_info()
        self.assertTrue(info["mapping_on_gpu"])

    def test_pycolmap_info_without_pycolmap(self):
        with mock.patch.object(pycolmap_wrapper, "import_pycolmap", return_value=None):
            info = pycolmap_wrapper.pycolmap_info()
        self.assertFalse(info["available"])
        self.assertIn("sift_on_gpu", info)
        self.assertIn("mapping_on_gpu", info)


class GpuBridgeTests(unittest.TestCase):
    """The CUDA bridge: the SIFT stages go to the downloaded CUDA COLMAP build on the GPU."""

    def setUp(self):
        self.fake = FakePycolmap(image_names=("a.png", "b.png", "c.png"))
        self.patch = mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                       return_value=self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.bridge = pycolmap_wrapper.GpuBridge("C:/bin/colmap-3.11.1-cuda/COLMAP.bat", "download",
                                                 cuda=True)
        self.runs = []

    def _wrapper(self, bridge=None):
        wrapper = pycolmap_wrapper.PyColmapWrapper(bridge=bridge)
        wrapper.setup_workspace()
        self.addCleanup(wrapper.cleanup_workspace)
        for name in ("a.png", "b.png", "c.png"):
            (wrapper.image_dir / name).write_bytes(b"")
        return wrapper

    def _fake_cli(self, return_code=0, output="I... feature_extractor.cc] Processed file [2/3]"):
        """Stand in for `run_streaming_command`: record the command and replay COLMAP's output."""
        def run(command, desc, timeout, progress_callback=None, env=None):
            self.runs.append({"command": [str(part) for part in command], "desc": desc,
                              "env": env, "callback": progress_callback})
            if progress_callback is not None:
                for line in output.splitlines():
                    progress_callback(line)
            return return_code, output
        return run

    def test_extraction_and_matching_go_to_the_cuda_build(self):
        wrapper = self._wrapper(self.bridge)
        progress = []
        wrapper.set_progress_hook(lambda value, total, label: progress.append((value, total, label)))
        with mock.patch.object(pycolmap_wrapper, "run_streaming_command",
                               side_effect=self._fake_cli()):
            self.assertTrue(wrapper.feature_extractor(camera_model="SIMPLE_PINHOLE",
                                                      max_num_features=10000,
                                                      max_image_size=2048, use_gpu=True,
                                                      mask_path="C:/masks"))
            self.assertTrue(wrapper.sequential_matcher(use_gpu=True, overlap=15))
        extract = self.runs[0]["command"]
        self.assertEqual(extract[0], str(self.bridge.executable))
        self.assertEqual(extract[1], "feature_extractor")
        self.assertEqual(extract[extract.index("--SiftExtraction.use_gpu") + 1], "1")
        self.assertEqual(extract[extract.index("--ImageReader.single_camera") + 1], "1")
        self.assertEqual(extract[extract.index("--SiftExtraction.max_num_features") + 1], "10000")
        self.assertEqual(extract[extract.index("--ImageReader.mask_path") + 1], "C:/masks")
        # COLMAP's own progress record drives the bar, and the child runs at INFO to have it
        self.assertIn((2, 3, "feature extraction (GPU) 2/3"), progress)
        self.assertEqual(self.runs[0]["env"].get("GLOG_minloglevel"),
                         pycolmap_wrapper.GLOG_LEVEL_INFO)
        self.assertEqual(self.fake.calls, [])          # ... the CPU path was not used
        match = self.runs[1]["command"]
        self.assertEqual(match[1], "sequential_matcher")
        self.assertEqual(match[match.index("--SequentialMatching.overlap") + 1], "15")
        self.assertEqual(match[match.index("--SiftMatching.use_gpu") + 1], "1")

    def test_native_cuda_pycolmap_wins_over_the_bridge(self):
        self.fake.has_cuda = True
        wrapper = self._wrapper(self.bridge)
        with mock.patch.object(pycolmap_wrapper, "run_streaming_command",
                               side_effect=self._fake_cli()) as cli:
            self.assertTrue(wrapper.feature_extractor(use_gpu=True))
        cli.assert_not_called()
        self.fake.call("extract")                      # pycolmap did the work, in-process

    def test_use_gpu_false_keeps_the_cpu_path(self):
        wrapper = self._wrapper(self.bridge)
        with mock.patch.object(pycolmap_wrapper, "run_streaming_command",
                               side_effect=self._fake_cli()) as cli:
            self.assertTrue(wrapper.feature_extractor(use_gpu=False))
        cli.assert_not_called()
        self.fake.call("extract")

    def test_no_bridge_keeps_the_cpu_path(self):
        wrapper = self._wrapper(None)
        self.assertIsNone(wrapper.colmap_path)
        self.assertFalse(wrapper._bridge_active(True))
        self.assertTrue(wrapper.feature_extractor(use_gpu=True))
        self.fake.call("extract")

    def test_a_failed_bridge_falls_back_to_pycolmap(self):
        wrapper = self._wrapper(self.bridge)
        with mock.patch.object(pycolmap_wrapper, "run_streaming_command",
                               side_effect=self._fake_cli(return_code=1, output="ERROR: no GPU")):
            self.assertTrue(wrapper.feature_extractor(use_gpu=True))
        self.fake.call("extract")                      # the CPU path took over


    def test_the_switch_disables_the_bridge(self):
        with mock.patch.dict("os.environ", {pycolmap_wrapper.BRIDGE_ENV: "0"}):
            self.assertFalse(pycolmap_wrapper.gpu_bridge_enabled())
            self.assertIsNone(pycolmap_wrapper.resolve_gpu_bridge(auto_install=False))
        with mock.patch.dict("os.environ", {pycolmap_wrapper.BRIDGE_ENV: "1"}):
            self.assertTrue(pycolmap_wrapper.gpu_bridge_enabled())

    def test_resolution_downloads_once_and_reports_the_source(self):
        manager = types.SimpleNamespace(
            resolve_binary=mock.Mock(side_effect=[None, "C:/bin/colmap-3.11.1-cuda/COLMAP.bat"]),
            ensure_binaries=mock.Mock(),
            source_of=mock.Mock(return_value="download"))
        with mock.patch.dict("os.environ", {"ENNDEE_AUTO_DOWNLOAD": "1"}), \
                mock.patch.object(pycolmap_wrapper, "_bin_manager", return_value=manager):
            bridge = pycolmap_wrapper.resolve_gpu_bridge(auto_install=True, log=lambda text: None)
        self.assertIsNotNone(bridge)
        self.assertEqual(bridge.executable, Path("C:/bin/colmap-3.11.1-cuda/COLMAP.bat"))
        self.assertEqual(bridge.source, "download")
        self.assertTrue(bridge.cuda)                   # <kind>-<version>-<flavor> folder
        manager.ensure_binaries.assert_called_once()
        self.assertEqual(manager.ensure_binaries.call_args.kwargs["flavor"], "cuda")

    def test_a_cpu_build_is_reported_as_such(self):
        manager = types.SimpleNamespace(resolve_binary=mock.Mock(
            return_value="C:/bin/colmap-3.11.1-nocuda/COLMAP.bat"),
            ensure_binaries=mock.Mock(), source_of=mock.Mock(return_value="pack bin/ folder"))
        with mock.patch.object(pycolmap_wrapper, "_bin_manager", return_value=manager):
            bridge = pycolmap_wrapper.resolve_gpu_bridge(auto_install=False, log=lambda text: None)
        self.assertIsNotNone(bridge)
        self.assertFalse(bridge.cuda)                  # never claim a GPU we do not have
        manager.ensure_binaries.assert_not_called()

    def test_auto_download_zero_blocks_the_download(self):
        manager = types.SimpleNamespace(resolve_binary=mock.Mock(return_value=None),
                                        ensure_binaries=mock.Mock(),
                                        source_of=mock.Mock(return_value=""))
        with mock.patch.dict("os.environ", {"ENNDEE_AUTO_DOWNLOAD": "0"}), \
                mock.patch.object(pycolmap_wrapper, "_bin_manager", return_value=manager):
            self.assertIsNone(pycolmap_wrapper.resolve_gpu_bridge(auto_install=True,
                                                                  log=lambda text: None))
        manager.ensure_binaries.assert_not_called()

    def test_resolution_without_download_uses_what_is_there(self):
        manager = types.SimpleNamespace(resolve_binary=mock.Mock(return_value=None),
                                        ensure_binaries=mock.Mock(),
                                        source_of=mock.Mock(return_value=""))
        with mock.patch.object(pycolmap_wrapper, "_bin_manager", return_value=manager):
            self.assertIsNone(pycolmap_wrapper.resolve_gpu_bridge(auto_install=False,
                                                                  log=lambda text: None))
        manager.ensure_binaries.assert_not_called()


class CpuNoticeTests(unittest.TestCase):
    """The "no CUDA build" explanation is a build property - one line, not one per stage."""

    def test_notice_is_printed_once(self):
        fake = types.SimpleNamespace(has_cuda=False)
        wrapper = pycolmap_wrapper.PyColmapWrapper()
        with mock.patch.object(pycolmap_wrapper, "_CPU_NOTICE_SHOWN", False), \
                mock.patch("builtins.print") as printed:
            self.assertFalse(wrapper._effective_gpu(fake, True))
            self.assertFalse(wrapper._effective_gpu(fake, True))
        self.assertEqual(printed.call_count, 1)
        self.assertIn("no CUDA support", printed.call_args.args[0])
        self.assertIn("GPU", printed.call_args.args[0])

    def test_no_notice_without_use_gpu(self):
        wrapper = pycolmap_wrapper.PyColmapWrapper()
        with mock.patch.object(pycolmap_wrapper, "_CPU_NOTICE_SHOWN", False), \
                mock.patch.object(pycolmap_wrapper, "_MAPPING_NOTICE_SHOWN", False), \
                mock.patch("builtins.print") as printed:
            self.assertFalse(wrapper._effective_gpu(types.SimpleNamespace(has_cuda=False),
                                                    False))
            self.assertFalse(wrapper._effective_gpu(types.SimpleNamespace(has_cuda=True),
                                                    False))
        printed.assert_not_called()


class MappingNoticeTests(unittest.TestCase):
    """A CUDA build explains *once* that its mapping still runs on the CPU."""

    def test_mapping_notice_is_printed_once(self):
        fake = types.SimpleNamespace(has_cuda=True)
        wrapper = pycolmap_wrapper.PyColmapWrapper()
        with mock.patch.object(pycolmap_wrapper, "_MAPPING_NOTICE_SHOWN", False), \
                mock.patch("builtins.print") as printed:
            self.assertTrue(wrapper._effective_gpu(fake, True))
            self.assertTrue(wrapper._effective_gpu(fake, True))
        self.assertEqual(printed.call_count, 1)
        self.assertIn("stays on the CPU", printed.call_args.args[0])
        self.assertIn("cuDSS", printed.call_args.args[0])

    def test_the_two_notices_are_independent(self):
        wrapper = pycolmap_wrapper.PyColmapWrapper()
        with mock.patch.object(pycolmap_wrapper, "_CPU_NOTICE_SHOWN", False), \
                mock.patch.object(pycolmap_wrapper, "_MAPPING_NOTICE_SHOWN", False), \
                mock.patch("builtins.print") as printed:
            wrapper._effective_gpu(types.SimpleNamespace(has_cuda=False), True)
            wrapper._effective_gpu(types.SimpleNamespace(has_cuda=True), True)
        self.assertEqual(printed.call_count, 2)
        self.assertIn("no CUDA support", printed.call_args_list[0].args[0])
        self.assertIn("stays on the CPU", printed.call_args_list[1].args[0])


class NativeNodeTests(unittest.TestCase):
    """The node keeps every widget of the binary tracker except the paths."""

    REMOVED_REQUIRED = ("colmap_path", "glomap_path")
    REMOVED_OPTIONAL = ("binary_flavor",)
    OVERRIDDEN = ("mapper_backend", "auto_install_binaries")
    #: the native node's dense-MVS products - appended LAST so existing
    #: workflows keep their widget_values positions
    ADDED_OPTIONAL = ("export_depth_maps", "dense_max_image_size",
                      "dense_geom_consistency", "fuse_dense_cloud",
                      "dense_cloud_max_points", "mesh_dense_surface", "mesh_method")

    def test_widget_parity_with_the_binary_node(self):
        binary_types = binary.GLOMAPLichtfeldTracker.INPUT_TYPES()
        native_types = native.ColmapLichtfeldTracker.INPUT_TYPES()
        # the native node adds exactly one thing: the hidden node id
        self.assertEqual(list(native_types), list(binary_types) + ["hidden"])
        self.assertEqual(native_types["hidden"], {"unique_id": "UNIQUE_ID"})

        expected_required = [name for name in binary_types["required"]
                             if name not in self.REMOVED_REQUIRED]
        self.assertEqual(list(native_types["required"]), expected_required)
        for name in expected_required:
            self.assertEqual(native_types["required"][name],
                             binary_types["required"][name], name)

        expected_optional = [name for name in binary_types["optional"]
                             if name not in self.REMOVED_OPTIONAL]
        # ... plus the depth-export widgets, appended last
        self.assertEqual(list(native_types["optional"]),
                         expected_optional + list(self.ADDED_OPTIONAL))
        for name in expected_optional:
            if name in self.OVERRIDDEN:
                continue
            self.assertEqual(native_types["optional"][name],
                             binary_types["optional"][name], name)

    def test_mapper_backend_defaults_to_the_native_global_mapper(self):
        widget = native.ColmapLichtfeldTracker.INPUT_TYPES()["optional"]["mapper_backend"]
        self.assertEqual(list(widget[0]), ["global", "incremental"])
        self.assertEqual(widget[1]["default"], "global")

    def test_track_signature_dropped_only_the_paths(self):
        import inspect

        binary_params = list(inspect.signature(
            binary.GLOMAPLichtfeldTracker.track).parameters)
        native_params = list(inspect.signature(
            native.ColmapLichtfeldTracker.track).parameters)
        expected = [p for p in binary_params
                    if p not in self.REMOVED_REQUIRED + self.REMOVED_OPTIONAL]
        # the native node adds only the hidden node id for the live status events;
        # the depth-export parameters already exist on the shared base signature
        # (the native node is simply the only one that exposes them as widgets -
        # see test_widget_parity_with_the_binary_node)
        self.assertEqual(native_params, expected + ["unique_id"])
        self.assertEqual(native.ColmapLichtfeldTracker.FUNCTION, "track")

    def test_setup_reports_the_native_backend(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False,
                               "mode": "cpu-fallback",
                               "reason": "no 'pycolmap-cuda12' wheel exists for this platform"},
                  "onnxruntime": {"version": "1.30.0", "providers": ["CUDAExecutionProvider"],
                                  "cuda": True},
                  "cuda": {"available": True, "version": "13.0", "device": "RTX 5090"},
                  "cuda_major": 13, "optional_attention": {"flash_attn": False,
                                                           "sageattention": True}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report) as call, \
                mock.patch.object(native, "resolve_gpu_bridge", return_value=None) as bridge:
            colmap_exe, glomap_exe = native.ColmapLichtfeldTracker()._setup_binaries(
                "", "", "global", True, "native")
        self.assertEqual((colmap_exe, glomap_exe), ("pycolmap", None))
        self.assertTrue(call.call_args.kwargs["auto_install"])
        self.assertTrue(bridge.call_args.kwargs["auto_install"])   # may fetch the CUDA build

    def test_the_bridge_reaches_the_header_and_the_wrapper(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False,
                               "mode": "cpu-fallback", "reason": "Windows wheel has no CUDA"},
                  "onnxruntime": {"version": "1.30.0", "providers": ["CUDAExecutionProvider"],
                                  "cuda": True},
                  "cuda": {"available": True, "version": "13.0", "device": "RTX 5090"},
                  "cuda_major": 13, "optional_attention": {}}
        bridge = pycolmap_wrapper.GpuBridge("C:/bin/colmap-3.11.1-cuda/COLMAP.bat", "download",
                                            cuda=True)
        node = native.ColmapLichtfeldTracker()
        header = []
        node._status = types.SimpleNamespace(set_stage=lambda *a: None,
                                             set_header=lambda lines: header.extend(lines),
                                             progress=lambda *a: None)
        node._use_gpu = True
        with mock.patch.object(native, "ensure_accelerators", return_value=report), \
                mock.patch.object(native, "resolve_gpu_bridge", return_value=bridge) as resolve:
            node._setup_binaries("", "", "global", True, "native")
        resolve.assert_called_once()
        self.assertIs(node._gpu_bridge, bridge)
        self.assertIn("+ CUDA bridge", node._backend_note)
        self.assertTrue(any("CUDA COLMAP bridge" in line for line in header))
        self.assertTrue(any("feature extraction + matching run on the GPU" in line
                            for line in header))
        self.assertIs(node._create_wrapper("pycolmap", None, "global").gpu_bridge, bridge)

    def test_the_bridge_is_not_fetched_without_use_gpu(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False,
                               "mode": "cpu", "reason": "no CUDA device"},
                  "onnxruntime": {"version": "1.30.0", "providers": [], "cuda": False},
                  "cuda_major": 13, "optional_attention": {}}
        node = native.ColmapLichtfeldTracker()
        node._use_gpu = False
        with mock.patch.object(native, "ensure_accelerators", return_value=report), \
                mock.patch.object(native, "resolve_gpu_bridge") as resolve:
            node._setup_binaries("", "", "global", True, "native")
        resolve.assert_not_called()
        self.assertIsNone(node._gpu_bridge)

    def test_setup_aborts_without_pycolmap(self):
        report = {"pycolmap": {"available": False, "version": "", "cuda": False,
                               "mode": "missing", "reason": "not installed"},
                  "onnxruntime": {"version": "", "providers": [], "cuda": False},
                  "cuda_major": 13, "optional_attention": {}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report), \
                mock.patch.object(native, "resolve_gpu_bridge", return_value=None):
            self.assertEqual(
                native.ColmapLichtfeldTracker()._setup_binaries("", "", "global", False,
                                                                "native"),
                (None, None))

    def test_setup_honours_the_node_switch(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False,
                               "mode": "cpu", "reason": "no CUDA device"},
                  "onnxruntime": {"version": "1.30.0", "providers": [], "cuda": False},
                  "cuda_major": 13, "optional_attention": {}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report) as call, \
                mock.patch.object(native, "resolve_gpu_bridge", return_value=None):
            native.ColmapLichtfeldTracker()._setup_binaries("", "", "global", False,
                                                            "native")
        self.assertFalse(call.call_args.kwargs["auto_install"])

    def test_wrapper_is_the_native_one(self):
        wrapper = native.ColmapLichtfeldTracker()._create_wrapper("pycolmap", None, "global")
        self.assertIsInstance(wrapper, pycolmap_wrapper.PyColmapWrapper)

    def test_track_forwards_legacy_backend_names(self):
        calls = {}

        def fake_track(self, *args, **kwargs):
            calls.update(kwargs)
            return ("trajectory", "points", 1.0, "")

        with mock.patch.object(binary.GLOMAPLichtfeldTracker, "track", fake_track):
            node = native.ColmapLichtfeldTracker()
            for given, expected in (("glomap", "global"), ("colmap_global", "global"),
                                    ("incremental", "incremental")):
                returned = node.track("SIMPLE_PINHOLE", "sequential", 10000,
                                      mapper_backend=given, use_rmbg=False)
                self.assertEqual(calls["mapper_backend"], expected, given)
                self.assertIsNone(calls["colmap_path"])
                self.assertIsNone(calls["glomap_path"])
                self.assertEqual(calls["binary_flavor"], "native")
                # the node reports its status through ComfyUI's ui output
                self.assertEqual(returned["result"],
                                 ("trajectory", "points", 1.0, ""))
                self.assertTrue(returned["ui"]["text"][0])

    def test_ready_message_differs_from_the_binary_node(self):
        self.assertNotEqual(native.ColmapLichtfeldTracker.BACKEND_READY_MESSAGE,
                            binary.GLOMAPLichtfeldTracker.BACKEND_READY_MESSAGE)


class AcceleratorTests(unittest.TestCase):
    """The env helper picks the right wheel and never installs silently."""

    def test_onnx_spec_matches_the_cuda_major(self):
        self.assertEqual(accelerators._onnx_spec(13), "onnxruntime-gpu>=1.30")
        self.assertTrue(accelerators._onnx_spec(12).startswith("onnxruntime-gpu>=1.19"))
        self.assertEqual(accelerators._onnx_spec(None), "onnxruntime-gpu")

    def test_auto_install_switch(self):
        with mock.patch.dict("os.environ", {"ENNDEE_AUTO_DOWNLOAD": "0"}):
            self.assertFalse(accelerators.auto_install_enabled())
        with mock.patch.dict("os.environ", {"ENNDEE_AUTO_DOWNLOAD": "1"}):
            self.assertTrue(accelerators.auto_install_enabled())

    def test_disabled_auto_install_runs_no_pip(self):
        state = {"cuda_major": 13,
                 "onnxruntime": {"version": "1.30.0", "providers": [], "cuda": False,
                                 "cpu_wheel": True},
                 "pycolmap": {"available": False, "version": "", "cuda": False},
                 "optional_attention": {}}
        with mock.patch.object(accelerators, "accelerator_report", return_value=state), \
                mock.patch.object(accelerators, "pip") as pip:
            accelerators.ensure_accelerators(auto_install=False)
        pip.assert_not_called()

    def test_cpu_wheel_is_removed_and_the_gpu_wheel_installed(self):
        state = {"cuda_major": 13,
                 "onnxruntime": {"version": "1.24.3",
                                 "providers": ["CPUExecutionProvider"], "cuda": False,
                                 "cpu_wheel": True, "gpu_wheel": True},
                 "pycolmap": {"available": False, "version": "", "cuda": False},
                 "optional_attention": {}}
        calls = []

        def fake_pip(args, dry=False):
            calls.append(args)
            return 0, ""

        with mock.patch.object(accelerators, "accelerator_report", return_value=state), \
                mock.patch.object(accelerators, "pip", side_effect=fake_pip), \
                mock.patch.object(accelerators, "pycolmap_state",
                                  return_value={"available": False, "version": "",
                                                "cuda": False}), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": False, "version": "",
                                                "device": ""}), \
                mock.patch.object(accelerators, "onnxruntime_state",
                                  return_value={"providers": ["CUDAExecutionProvider"],
                                                "cuda": True, "version": "1.30.0"}):
            report = accelerators.ensure_accelerators(auto_install=True)
        self.assertIn(["uninstall", "-y", "onnxruntime"], calls)
        self.assertIn(["install", "--upgrade", "onnxruntime-gpu>=1.30"], calls)
        self.assertIn(["install", "pycolmap"], calls)
        self.assertTrue(report["onnxruntime"]["cuda"])


class NodeStatusTests(unittest.TestCase):
    """The live label and the progress bar (no running ComfyUI needed)."""

    def test_progress_maps_the_pipeline_onto_0_90_percent(self):
        status = native.NodeStatus()
        seen = []
        status.set_percent = seen.append
        status.progress(5, 10, "feature extraction 5/10 images")
        self.assertEqual(seen, [45.0])
        self.assertEqual(status.stage, "feature extraction 5/10 images")

    def test_finish_reports_100_percent(self):
        status = native.NodeStatus()
        seen = []
        status.set_percent = seen.append
        status.finish("done")
        self.assertEqual(seen, [100.0])
        self.assertEqual(status.stage, "done")

    def test_text_contains_header_and_stage(self):
        status = native.NodeStatus()
        status.set_header(["pycolmap : 4.2.1 [cpu-fallback]", "           no CUDA wheel"])
        status.set_stage("feature extraction 1/5 images")
        lines = status.text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[-1], "status  : feature extraction 1/5 images")

    def test_status_is_sent_to_the_node(self):
        sent = []
        server = types.ModuleType("server")

        class FakePromptServer:
            def send_sync(self, event, payload):
                sent.append((event, payload))

        server.PromptServer = types.SimpleNamespace(instance=FakePromptServer())
        with mock.patch.dict(sys.modules, {"server": server}):
            status = native.NodeStatus(node_id="42")
            status.set_stage("working")
        self.assertEqual(sent[-1][0], native.STATUS_EVENT)
        self.assertEqual(sent[-1][1]["node"], "42")
        self.assertIn("working", sent[-1][1]["text"])

    def test_without_a_node_id_only_the_text_is_kept(self):
        status = native.NodeStatus()
        status.set_stage("working")
        self.assertIn("working", status.text)

    def test_percent_never_moves_backwards(self):
        status = native.NodeStatus()
        seen = []
        status.pbar = types.SimpleNamespace(
            update_absolute=lambda value, total=None, preview=None: seen.append(value))
        status.set_percent(60.0)
        status.set_percent(0.0)   # stage marker of the next stage
        status.set_percent(90.0)
        self.assertEqual(seen, [60, 60, 90])

    def test_progress_bar_is_driven_when_comfy_is_available(self):
        comfy = types.ModuleType("comfy")
        utils = types.ModuleType("comfy.utils")
        updates = []

        class FakeBar:
            def __init__(self, total):
                updates.append(("init", total))

            def update_absolute(self, value, total=None, preview=None):
                updates.append(("update", value))

        utils.ProgressBar = FakeBar
        comfy.utils = utils
        with mock.patch.dict(sys.modules, {"comfy": comfy, "comfy.utils": utils}):
            status = native.NodeStatus()
            status.set_percent(42.4)
        self.assertEqual(updates, [("init", 100), ("update", 42)])


class PycolmapCudaTests(unittest.TestCase):
    """A CUDA build is preferred; the CPU fallback always says why."""

    def setUp(self):
        accelerators._CUDA_ATTEMPT.clear()
        self.addCleanup(accelerators._CUDA_ATTEMPT.clear)

    def test_cuda_build_is_used_when_present(self):
        with mock.patch.object(accelerators, "pycolmap_state",
                               return_value={"available": True, "version": "4.2.1",
                                             "cuda": True}), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": True, "version": "13.0",
                                                "device": "RTX 5090"}), \
                mock.patch.object(accelerators, "pip") as pip:
            result = accelerators.ensure_pycolmap()
        self.assertEqual(result["mode"], "cuda")
        self.assertTrue(result["cuda"])
        pip.assert_not_called()

    def test_no_cuda_device_means_the_cpu_build_is_correct(self):
        with mock.patch.object(accelerators, "pycolmap_state",
                               return_value={"available": True, "version": "4.2.1",
                                             "cuda": False}), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": False, "version": "",
                                                "device": ""}):
            result = accelerators.ensure_pycolmap()
        self.assertEqual(result["mode"], "cpu")
        self.assertIn("no CUDA device", result["reason"])

    def test_missing_cuda_wheel_is_reported_as_a_fallback(self):
        with mock.patch.object(accelerators, "pycolmap_state",
                               return_value={"available": True, "version": "4.2.1",
                                             "cuda": False}), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": True, "version": "13.0",
                                                "device": "RTX 5090"}), \
                mock.patch.object(accelerators, "_pip_can_install", return_value=False), \
                mock.patch.object(accelerators, "pip") as pip:
            result = accelerators.ensure_pycolmap()
        self.assertEqual(result["mode"], "cpu-fallback")
        self.assertIn("pycolmap-cuda12", result["reason"])
        self.assertEqual(result["cuda_device"], "RTX 5090")
        pip.assert_not_called()  # nothing installable -> no pointless pip run

    def test_env_wheel_is_used_for_the_cuda_build(self):
        # the install is verified with pycolmap_info(), so the state is read a third time
        # (start, verify, re-check after the install)
        states = [{"available": True, "version": "4.2.1", "cuda": False},
                  {"available": True, "version": "4.2.1", "cuda": True},
                  {"available": True, "version": "4.2.1", "cuda": True}]
        with mock.patch.object(accelerators, "pycolmap_state", side_effect=states), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": True, "version": "13.0",
                                                "device": "RTX 5090"}), \
                mock.patch.dict("os.environ",
                                {accelerators.PYCOLMAP_CUDA_WHEEL_ENV:
                                 "D:/wheels/pycolmap_cuda.whl"}), \
                mock.patch.object(accelerators, "pip", return_value=(0, "")) as pip:
            result = accelerators.ensure_pycolmap()
        self.assertEqual(result["mode"], "cuda")
        pip.assert_called_once()
        self.assertIn("D:/wheels/pycolmap_cuda.whl", pip.call_args.args[0])
        # the reason names the source and the verified version
        self.assertIn(accelerators.PYCOLMAP_CUDA_WHEEL_ENV, result["reason"])
        self.assertIn("4.2.1", result["reason"])

    def test_cuda_specs_follow_the_torch_cuda_major(self):
        self.assertEqual(accelerators.pycolmap_cuda_specs(13),
                         ("pycolmap-cuda13", "pycolmap-cuda12"))
        self.assertEqual(accelerators.pycolmap_cuda_specs(12), ("pycolmap-cuda12",))
        with mock.patch.object(accelerators, "torch_cuda_major", return_value=13):
            self.assertEqual(accelerators.pycolmap_cuda_specs()[0], "pycolmap-cuda13")

    def test_a_cpu_wheel_never_masquerades_as_a_cuda_build(self):
        """The install is verified: ``cuda`` must really be True afterwards."""
        with mock.patch.object(accelerators, "pycolmap_state",
                               return_value={"available": True, "version": "4.2.1",
                                             "cuda": False}), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": True, "version": "13.0",
                                                "device": "RTX 5090"}), \
                mock.patch.object(accelerators, "_pip_can_install", return_value=True), \
                mock.patch.object(accelerators, "pip", return_value=(0, "")):
            result = accelerators.ensure_pycolmap()
        self.assertEqual(result["mode"], "cpu-fallback")
        self.assertIn("without CUDA support", result["reason"])
        self.assertIn("4.2.1", result["reason"])

    def test_a_broken_cuda_wheel_is_reported_as_a_failure(self):
        with mock.patch.object(accelerators, "pycolmap_state",
                               return_value={"available": True, "version": "4.2.1",
                                             "cuda": False}), \
                mock.patch.object(accelerators, "cuda_state",
                                  return_value={"available": True, "version": "13.0",
                                                "device": "RTX 5090"}), \
                mock.patch.dict("os.environ",
                                {accelerators.PYCOLMAP_CUDA_WHEEL_ENV: "D:/wheels/bad.whl"}), \
                mock.patch.object(accelerators, "_pip_can_install", return_value=False), \
                mock.patch.object(accelerators, "pip", return_value=(1, "boom")):
            result = accelerators.ensure_pycolmap()
        self.assertEqual(result["mode"], "cpu-fallback")
        self.assertIn("could not be installed", result["reason"])

    def test_cuda_detection_prefers_torch(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True,
                                       get_device_name=lambda index: "RTX 5090"),
            version=types.SimpleNamespace(cuda="13.0"))
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            state = accelerators.cuda_state()
        self.assertTrue(state["available"])
        self.assertEqual(state["source"], "torch")
        self.assertEqual(state["device"], "RTX 5090")


class ColmapLoggingTests(unittest.TestCase):
    """COLMAP's INFO flood is silenced unless ENNDEE_COLMAP_VERBOSE is set."""

    def test_import_sets_the_glog_level_before_importing(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            module = pycolmap_wrapper.import_pycolmap()
            level = os.environ.get("GLOG_minloglevel")
        self.assertIsNotNone(module)                      # pycolmap is installed here
        self.assertEqual(level, pycolmap_wrapper.GLOG_LEVEL_WARNING)

    def test_child_env_is_quiet_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(pycolmap_wrapper.colmap_verbose())
            env = pycolmap_wrapper.colmap_child_env()
        values = {key.upper(): value for key, value in env.items()}
        self.assertEqual(values.get("GLOG_MINLOGLEVEL"), "1")

    def test_existing_glog_level_is_left_alone(self):
        with mock.patch.dict("os.environ", {"GLOG_minloglevel": "0"}, clear=True):
            env = pycolmap_wrapper.colmap_child_env()
        matching = {key: value for key, value in env.items()
                    if key.upper() == "GLOG_MINLOGLEVEL"}
        self.assertEqual(list(matching.values()), ["0"])

    def test_verbose_env_keeps_the_full_output(self):
        with mock.patch.dict("os.environ", {"ENNDEE_COLMAP_VERBOSE": "1"}, clear=True):
            self.assertTrue(pycolmap_wrapper.colmap_verbose())
            env = pycolmap_wrapper.colmap_child_env()
        # verbose adds nothing of its own - whatever the user set stays untouched
        self.assertFalse(any(key.upper() == "GLOG_MINLOGLEVEL" for key in env))

    def test_runtime_level_is_set_on_the_module(self):
        fake = types.SimpleNamespace(logging=types.SimpleNamespace(WARNING=1, minloglevel=0))
        with mock.patch.dict("os.environ", {}, clear=True):
            pycolmap_wrapper.silence_colmap_logging(fake)
        self.assertEqual(fake.logging.minloglevel, 1)

    def test_verbose_skips_the_runtime_level(self):
        fake = types.SimpleNamespace(logging=types.SimpleNamespace(WARNING=1, minloglevel=0))
        with mock.patch.dict("os.environ", {"ENNDEE_COLMAP_VERBOSE": "1"}, clear=True):
            pycolmap_wrapper.silence_colmap_logging(fake)
        self.assertEqual(fake.logging.minloglevel, 0)

class FakePinholeCamera:
    """Duck-typed camera carrying only what ``_warp_to_original`` touches.

    The real ``pycolmap.Camera`` cannot be used in this suite (it is the reason the
    whole file fakes pycolmap), and the warp only ever calls ``cam_from_img``,
    ``img_from_cam``, ``is_undistorted``, ``width`` and ``height``.
    """

    def __init__(self, width, height, focal, cx=None, cy=None, undistorted=True):
        self.width = int(width)
        self.height = int(height)
        self.focal = float(focal)
        self.cx = float((width - 1) / 2.0 if cx is None else cx)
        self.cy = float((height - 1) / 2.0 if cy is None else cy)
        self._undistorted = bool(undistorted)

    def is_undistorted(self):
        return self._undistorted

    def cam_from_img(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float64)
        return (pixels - np.array([self.cx, self.cy])) / self.focal

    def img_from_cam(self, rays):
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 3:
            raise ValueError("img_from_cam wants [N, 3] camera points")
        return rays[:, :2] * self.focal + np.array([self.cx, self.cy])


class DenseMvsTests(unittest.TestCase):
    """The dense-MVS depth read + the un-warp onto the original camera grid."""

    def test_warp_is_exact_for_a_scaled_pinhole_pair(self):
        """A half-resolution undistorted camera must map target pixel 2k -> k."""
        original = FakePinholeCamera(8, 8, focal=4.0, cx=3.5, cy=3.5)
        undistorted = FakePinholeCamera(4, 4, focal=2.0, cx=1.75, cy=1.75)
        source = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0)

        warped = pycolmap_wrapper.PyColmapWrapper._warp_to_original(
            source, undistorted, original, (8, 8))

        self.assertEqual(warped.shape, (8, 8))
        # every even pixel samples the source exactly - no shift, no flip
        np.testing.assert_allclose(warped[::2, ::2], source, rtol=0, atol=1e-4)
        # odd pixels are the bilinear midpoint between their two neighbours
        np.testing.assert_allclose(warped[0, 1], (source[0, 0] + source[0, 1]) / 2,
                                   rtol=0, atol=1e-4)

    def test_warp_zeroes_pixels_outside_the_undistorted_view(self):
        """A ray leaving the undistorted image has no depth, so it must be 0."""
        # same focal as the original but a 2x2 grid -> only the 4 central target
        # pixels project inside it, everything else falls outside the view
        original = FakePinholeCamera(8, 8, focal=4.0)
        undistorted = FakePinholeCamera(2, 2, focal=4.0)
        source = np.ones((2, 2), dtype=np.float32)

        warped = pycolmap_wrapper.PyColmapWrapper._warp_to_original(
            source, undistorted, original, (8, 8))

        self.assertEqual(warped.shape, (8, 8))
        valid = warped > 0
        self.assertGreater(int(valid.sum()), 0)
        self.assertLess(float(valid.mean()), 1.0)
        self.assertEqual(float(warped.min()), 0.0)
        # the survivors are the ones whose ray lands inside the small view
        np.testing.assert_allclose(warped[3:5, 3:5], 1.0, rtol=0, atol=1e-4)

    def test_img_from_cam_needs_the_padded_ray(self):
        """The fake mirrors pycolmap: [N, 2] must be rejected, [N, 3] accepted."""
        camera = FakePinholeCamera(4, 4, focal=2.0)
        with self.assertRaises(ValueError):
            camera.img_from_cam(np.zeros((3, 2)))
        self.assertEqual(np.asarray(camera.img_from_cam(np.zeros((3, 3)))).shape,
                         (3, 2))


class DenseDepthMapReadTests(unittest.TestCase):
    """``read_dense_depth_maps`` must index COLMAP's files and pick the right kind."""

    class FakeDepthMap:
        def __init__(self, recorder):
            self._recorder = recorder
            self._value = 0.0

        def read(self, path):
            self._recorder.append(str(path))
            # the two kinds are told apart by their value so the test can prove
            # which one was picked
            self._value = 2.0 if "geometric" in str(path) else 1.0

        def to_array(self):
            return np.full((2, 3), self._value, dtype=np.float32)

    def _module(self, recorder):
        fake = types.SimpleNamespace()
        fake.DepthMap = lambda: DenseDepthMapReadTests.FakeDepthMap(recorder)
        return fake

    def test_prefers_geometric_and_strips_the_kind(self):
        import tempfile

        recorder = []
        with tempfile.TemporaryDirectory() as folder:
            depth_maps = Path(folder) / "stereo" / "depth_maps"
            depth_maps.mkdir(parents=True)
            # deliberately written photometric FIRST: sorted() must still win
            for name in ("frame_00.jpg.photometric.bin", "frame_00.jpg.geometric.bin",
                         "frame_01.jpg.geometric.bin", "notes.txt"):
                (depth_maps / name).write_bytes(b"")

            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module(recorder)):
                maps = pycolmap_wrapper.PyColmapWrapper.read_dense_depth_maps(
                    Path(folder), None, log=lambda _message: None)

        self.assertEqual(sorted(maps), ["frame_00.jpg", "frame_01.jpg"])
        self.assertEqual(len(recorder), 2)
        # 2.0 is the geometric value - the filtered map wins
        np.testing.assert_allclose(maps["frame_00.jpg"], 2.0)
        self.assertNotIn("notes.txt", recorder)

    def test_missing_folder_is_not_an_error(self):
        with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                               return_value=self._module([])):
            maps = pycolmap_wrapper.PyColmapWrapper.read_dense_depth_maps(
                Path("does-not-exist"), None, log=lambda _message: None)
        self.assertEqual(maps, {})


class DenseFusionTests(unittest.TestCase):
    """``dense_fused_cloud``: the fused cloud that replaces the sparse initialisation."""

    class FakePoint:
        def __init__(self, xyz, colour):
            self.xyz = np.asarray(xyz, dtype=np.float64)
            self.color = np.asarray(colour, dtype=np.int64)

    class FakeReconstruction:
        def __init__(self, points):
            self.points3D = {index + 1: point for index, point in enumerate(points)}

    def _module(self, recorder, points, fail=False):
        fake = types.SimpleNamespace()

        def stereo_fusion(output_path, workspace_path, **kwargs):
            recorder.append(("stereo_fusion", output_path, workspace_path, kwargs))
            if fail:
                raise RuntimeError("fusion exploded")
            # the fused PLY is what the node copies out of the workspace
            Path(output_path).write_bytes(b"ply")
            return DenseFusionTests.FakeReconstruction(points)

        fake.stereo_fusion = stereo_fusion
        fake.StereoFusionOptions = lambda: types.SimpleNamespace()
        fake.Reconstruction = lambda: types.SimpleNamespace(
            import_PLY=lambda _path: None)
        return fake

    def _wrapper(self, folder):
        wrapper = pycolmap_wrapper.PyColmapWrapper()
        wrapper.workspace = str(folder)
        return wrapper

    def test_uses_output_type_ply_and_subsamples_deterministically(self):
        recorder = []
        points = [DenseFusionTests.FakePoint([index, 0.0, 0.0], [255, 128, 0])
                  for index in range(10)]
        with tempfile.TemporaryDirectory() as folder:
            dense = Path(folder) / "dense"
            dense.mkdir()
            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module(recorder, points)):
                fused = self._wrapper(folder).dense_fused_cloud(
                    max_points=4, log=lambda _message: None)

        self.assertEqual(len(recorder), 1)
        name, output_path, workspace_path, kwargs = recorder[0]
        self.assertEqual(name, "stereo_fusion")
        # "bin" (the default) would treat the output path as a DIRECTORY and fail
        self.assertEqual(kwargs["output_type"], "ply")
        self.assertEqual(kwargs["input_type"], "geometric")
        self.assertEqual(Path(output_path).name, "fused.ply")
        self.assertEqual(Path(workspace_path), Path(folder) / "dense")

        self.assertEqual(fused["total"], 10)
        self.assertEqual(len(fused["points"]), 4)
        # 8-bit colours come back as 0..1 floats
        self.assertAlmostEqual(float(fused["colors"][0][0]), 1.0)
        self.assertAlmostEqual(float(fused["colors"][0][1]), 128 / 255)
        # the same input gives the same subset, so two runs agree
        self.assertTrue(np.all(np.diff(fused["points"][:, 0]) > 0))

    def test_no_workspace_and_fusion_failure_are_not_fatal(self):
        with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                               return_value=self._module([], [])):
            self.assertEqual(pycolmap_wrapper.PyColmapWrapper().dense_fused_cloud(
                log=lambda _message: None), {})

        recorder = []
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "dense").mkdir()
            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module(recorder, [], fail=True)):
                self.assertEqual(self._wrapper(folder).dense_fused_cloud(
                    log=lambda _message: None), {})

    def test_stride_subset_keeps_everything_below_the_cap(self):
        self.assertEqual(list(pycolmap_wrapper.stride_subset(5, 10)), [0, 1, 2, 3, 4])
        self.assertEqual(list(pycolmap_wrapper.stride_subset(5, 0)), [0, 1, 2, 3, 4])
        self.assertEqual(len(pycolmap_wrapper.stride_subset(1000, 10)), 10)

    def test_points_from_reconstruction_falls_back_to_the_ply(self):
        module = types.SimpleNamespace()
        points = [DenseFusionTests.FakePoint([1, 2, 3], [10, 20, 30])]
        xyz, rgb = pycolmap_wrapper.points_from_reconstruction(
            module, DenseFusionTests.FakeReconstruction(points))
        np.testing.assert_allclose(xyz, [[1, 2, 3]])
        self.assertAlmostEqual(float(rgb[0][0]), 10 / 255)

        # an empty reconstruction with no PLY is simply empty, never an exception
        xyz, rgb = pycolmap_wrapper.points_from_reconstruction(module, None, None)
        self.assertEqual(xyz.shape, (0, 3))
        self.assertEqual(rgb.shape, (0, 3))


class DenseMeshTests(unittest.TestCase):
    """``dense_mesh``: Poisson in-process, Delaunay through the bundled COLMAP binary."""

    def _module(self, recorder, fail=False):
        fake = types.SimpleNamespace()

        def poisson_meshing(source, target, options=None):
            recorder.append(("poisson_meshing", source, target, options))
            if fail:
                raise RuntimeError("poisson exploded")
            Path(target).write_bytes(b"mesh")

        def options():
            return types.SimpleNamespace(trim=10.0, depth=13)

        fake.poisson_meshing = poisson_meshing
        fake.PoissonMeshingOptions = options
        return fake

    def _wrapper(self, folder):
        wrapper = pycolmap_wrapper.PyColmapWrapper()
        wrapper.workspace = str(folder)
        return wrapper

    def test_poisson_overrides_trim_because_the_default_kills_the_mesh(self):
        recorder = []
        with tempfile.TemporaryDirectory() as folder:
            dense = Path(folder) / "dense"
            dense.mkdir()
            (dense / "fused.ply").write_bytes(b"ply")
            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module(recorder)):
                mesh = self._wrapper(folder).dense_mesh(log=lambda _message: None)

        self.assertEqual(len(recorder), 1)
        _name, source, target, options = recorder[0]
        self.assertEqual(Path(source).name, "fused.ply")
        self.assertEqual(Path(target).name, "mesh_poisson.ply")
        # MEASURED: trim=10 (the COLMAP default) crops a real scene to 12 vertices /
        # 20 faces; trim=0 yields ~167k vertices. depth keeps COLMAP's 13.
        self.assertEqual(options.trim, 0.0)
        self.assertEqual(options.depth, 13)
        self.assertEqual(Path(mesh), Path(target))

    def test_poisson_without_a_fused_cloud_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "dense").mkdir()
            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module([])):
                self.assertIsNone(self._wrapper(folder).dense_mesh(
                    log=lambda _message: None))

    def test_delaunay_shells_out_with_the_documented_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            dense = Path(folder) / "dense"
            dense.mkdir()
            executable = Path(folder) / "colmap.exe"
            executable.write_bytes(b"exe")
            wrapper = self._wrapper(folder)
            wrapper.colmap_path = str(executable)

            def fake_run(command, **_kwargs):
                Path(command[command.index("--output_path") + 1]).write_bytes(b"mesh")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module([])), \
                 mock.patch("subprocess.run", side_effect=fake_run) as runner:
                mesh = wrapper.dense_mesh(method="delaunay", log=lambda _message: None)

            command = runner.call_args[0][0]
            self.assertEqual(command[1], "delaunay_mesher")
            # --workspace_path is NOT a valid flag on this build
            self.assertIn("--input_path", command)
            self.assertNotIn("--workspace_path", command)
            self.assertEqual(command[command.index("--input_type") + 1], "dense")
            self.assertEqual(Path(mesh).name, "mesh_delaunay.ply")

    def test_delaunay_without_a_binary_reports_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "dense").mkdir()
            wrapper = self._wrapper(folder)
            wrapper.colmap_path = None
            with mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                   return_value=self._module([])):
                self.assertIsNone(wrapper.dense_mesh(method="delaunay",
                                                     log=lambda _message: None))

    def test_colmap_executable_resolves_the_bat_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "bin").mkdir()
            real = root / "bin" / "colmap.exe"
            real.write_bytes(b"exe")
            wrapper = pycolmap_wrapper.PyColmapWrapper()

            wrapper.colmap_path = str(root / "COLMAP.bat")
            self.assertEqual(wrapper._colmap_executable(), real)

            wrapper.colmap_path = str(real)
            self.assertEqual(wrapper._colmap_executable(), real)

            wrapper.colmap_path = ""
            self.assertIsNone(wrapper._colmap_executable())


if __name__ == "__main__":
    unittest.main()


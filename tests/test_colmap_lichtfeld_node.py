"""Tests for the native pycolmap node (COLMAP for Lichtfeld) and its helpers.

Everything runs against a *fake* pycolmap module, so the suite needs neither the
real bindings nor a GPU - it checks the option/argument mapping, the backend hooks
and the accelerator logic.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR))

import colmap_lichtfeld_node as native          # noqa: E402
import enndee_accelerators as accelerators      # noqa: E402
import glomap_lichtfeld_node as binary          # noqa: E402
from enndee_colmap import pycolmap_wrapper      # noqa: E402


class FakePycolmap:
    """Records every call the wrapper makes and returns fake models."""

    def __init__(self, has_cuda=False, models=(0,), image_names=(), pair_error=False):
        self.has_cuda = has_cuda
        self.__version__ = "9.9.9-test"
        self.models = models
        self.image_names = list(image_names)
        self.pair_error = pair_error
        self.calls = []
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
                    def __init__(self, name):
                        self.name = name

                return {index + 1: Image(name)
                        for index, name in enumerate(fake.image_names)}

            def close(self):
                fake.closed_databases.append(str(path))

        return Database()

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

    def match_sequential(self, database, **kwargs):
        self.calls.append(("sequential", str(database), kwargs))

    def match_exhaustive(self, database, **kwargs):
        self.calls.append(("exhaustive", str(database), kwargs))

    def match_image_pairs(self, database, **kwargs):
        self.calls.append(("match_image_pairs", str(database), kwargs))

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

    def test_missing_pycolmap_is_reported_not_raised(self):
        with mock.patch.object(pycolmap_wrapper, "import_pycolmap", return_value=None):
            self.assertFalse(self.wrapper.feature_extractor())
            self.assertFalse(self.wrapper.sequential_matcher())
            self.assertFalse(self.wrapper.exhaustive_matcher())
            self.assertFalse(self.wrapper.mapper())

    def test_pycolmap_info_shape(self):
        info = pycolmap_wrapper.pycolmap_info()
        self.assertEqual(info, {"available": True, "version": "9.9.9-test",
                                "cuda": False})


class NativeNodeTests(unittest.TestCase):
    """The node keeps every widget of the binary tracker except the paths."""

    REMOVED_REQUIRED = ("colmap_path", "glomap_path")
    REMOVED_OPTIONAL = ("binary_flavor",)
    OVERRIDDEN = ("mapper_backend", "auto_install_binaries")

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
        self.assertEqual(list(native_types["optional"]), expected_optional)
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
        # the native node adds the hidden node id for the live status events
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
        with mock.patch.object(native, "ensure_accelerators", return_value=report) as call:
            colmap_exe, glomap_exe = native.ColmapLichtfeldTracker()._setup_binaries(
                "", "", "global", True, "native")
        self.assertEqual((colmap_exe, glomap_exe), ("pycolmap", None))
        self.assertTrue(call.call_args.kwargs["auto_install"])

    def test_setup_aborts_without_pycolmap(self):
        report = {"pycolmap": {"available": False, "version": "", "cuda": False,
                               "mode": "missing", "reason": "not installed"},
                  "onnxruntime": {"version": "", "providers": [], "cuda": False},
                  "cuda_major": 13, "optional_attention": {}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report):
            self.assertEqual(
                native.ColmapLichtfeldTracker()._setup_binaries("", "", "global", False,
                                                                "native"),
                (None, None))

    def test_setup_honours_the_node_switch(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False,
                               "mode": "cpu", "reason": "no CUDA device"},
                  "onnxruntime": {"version": "1.30.0", "providers": [], "cuda": False},
                  "cuda_major": 13, "optional_attention": {}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report) as call:
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
        states = [{"available": True, "version": "4.2.1", "cuda": False},
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


if __name__ == "__main__":
    unittest.main()


"""Tests for the native pycolmap node (COLMAP for Lichtfeld) and its helpers.

Everything runs against a *fake* pycolmap module, so the suite needs neither the
real bindings nor a GPU - it checks the option/argument mapping, the backend hooks
and the accelerator logic.
"""

import sys
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

    def __init__(self, has_cuda=False, models=(0,)):
        self.has_cuda = has_cuda
        self.__version__ = "9.9.9-test"
        self.models = models
        self.calls = []

        class Device:
            cpu = "cpu"
            cuda = "cuda"

        class CameraMode:
            SINGLE = "SINGLE"
            AUTO = "AUTO"

        self.Device = Device
        self.CameraMode = CameraMode

    def extract_features(self, database, images, **kwargs):
        self.calls.append(("extract", str(database), str(images), kwargs))

    def match_sequential(self, database, **kwargs):
        self.calls.append(("sequential", str(database), kwargs))

    def match_exhaustive(self, database, **kwargs):
        self.calls.append(("exhaustive", str(database), kwargs))

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

    def setUp(self):
        self.fake = FakePycolmap()
        self.patch = mock.patch.object(pycolmap_wrapper, "import_pycolmap",
                                       return_value=self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.wrapper = pycolmap_wrapper.PyColmapWrapper()
        self.wrapper.setup_workspace()
        self.addCleanup(self.wrapper.cleanup_workspace)
        (self.wrapper.sparse_dir / "0").mkdir(exist_ok=True)
        (self.wrapper.sparse_dir / "0" / "cameras.bin").write_bytes(b"")

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

    def test_gpu_is_only_requested_when_the_build_has_cuda(self):
        self.wrapper.feature_extractor(use_gpu=True)
        self.assertEqual(self.fake.call("extract")[3]["device"], "cpu")

        self.fake.calls.clear()
        self.fake.has_cuda = True
        self.wrapper.feature_extractor(use_gpu=True)
        self.assertEqual(self.fake.call("extract")[3]["device"], "cuda")

    def test_sequential_matcher_passes_the_overlap(self):
        self.assertTrue(self.wrapper.sequential_matcher(use_gpu=False, overlap=15))
        _, _, kwargs = self.fake.call("sequential")
        self.assertEqual(kwargs["pairing_options"]["overlap"], 15)
        self.assertFalse(kwargs["pairing_options"]["loop_detection"])
        self.assertEqual(kwargs["matching_options"]["use_gpu"], False)

    def test_exhaustive_matcher(self):
        self.assertTrue(self.wrapper.exhaustive_matcher(use_gpu=False))
        self.fake.call("exhaustive")

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
        self.assertEqual(list(native_types), list(binary_types))

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
        self.assertEqual(native_params,
                         [p for p in binary_params
                          if p not in self.REMOVED_REQUIRED + self.REMOVED_OPTIONAL])
        self.assertEqual(native.ColmapLichtfeldTracker.FUNCTION, "track")

    def test_setup_reports_the_native_backend(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False},
                  "onnxruntime": {"version": "1.30.0", "providers": ["CUDAExecutionProvider"],
                                  "cuda": True},
                  "cuda_major": 13, "optional_attention": {"flash_attn": False,
                                                           "sageattention": True}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report) as call:
            colmap_exe, glomap_exe = native.ColmapLichtfeldTracker()._setup_binaries(
                "", "", "global", True, "native")
        self.assertEqual((colmap_exe, glomap_exe), ("pycolmap", None))
        self.assertTrue(call.call_args.kwargs["auto_install"])

    def test_setup_aborts_without_pycolmap(self):
        report = {"pycolmap": {"available": False, "version": "", "cuda": False},
                  "onnxruntime": {"version": "", "providers": [], "cuda": False},
                  "cuda_major": 13, "optional_attention": {}}
        with mock.patch.object(native, "ensure_accelerators", return_value=report):
            self.assertEqual(
                native.ColmapLichtfeldTracker()._setup_binaries("", "", "global", False,
                                                                "native"),
                (None, None))

    def test_setup_honours_the_node_switch(self):
        report = {"pycolmap": {"available": True, "version": "4.2.1", "cuda": False},
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
                node.track("SIMPLE_PINHOLE", "sequential", 10000,
                           mapper_backend=given, use_rmbg=False)
                self.assertEqual(calls["mapper_backend"], expected, given)
                self.assertIsNone(calls["colmap_path"])
                self.assertIsNone(calls["glomap_path"])
                self.assertEqual(calls["binary_flavor"], "native")

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
                mock.patch.object(accelerators, "onnxruntime_state",
                                  return_value={"providers": ["CUDAExecutionProvider"],
                                                "cuda": True, "version": "1.30.0"}), \
                mock.patch.object(accelerators, "pycolmap_state",
                                  return_value={"available": True, "version": "4.2.1"}):
            report = accelerators.ensure_accelerators(auto_install=True)
        self.assertIn(["uninstall", "-y", "onnxruntime"], calls)
        self.assertIn(["install", "--upgrade", "onnxruntime-gpu>=1.30"], calls)
        self.assertIn(["install", "pycolmap"], calls)
        self.assertTrue(report["onnxruntime"]["cuda"])


if __name__ == "__main__":
    unittest.main()


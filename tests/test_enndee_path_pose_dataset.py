"""Ground-truth extrinsics pipeline: Meridian Geometry + Extrinsics and the Path Pose Dataset.

Pure logic - no MoGe, no COLMAP/GLOMAP binaries: the extrinsics signal round trip,
the geometry node's appended output (render stubbed at the module symbol, the rig
published through `fast_depth.LAST_CAMERA_EVAL`), the reduced dataset node's pose
assignment (sharp selection + frame_step + mapping), its sparse/0 export, and the
`validate_dataset` contract of the Lichtfeld trainer.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR))

import enndee_meridian_fast_depth as fast_depth  # noqa: E402
import enndee_meridian_geometry as geometry  # noqa: E402
from enndee_meridian_geometry_extrinsics import (  # noqa: E402
    EnndeeMeridianGeometryExtrinsics,
    build_extrinsics_signal,
)
from enndee_path_pose_dataset import (  # noqa: E402
    EnndeePathPoseDataset,
    c2w_to_w2c,
    map_input_to_extrinsics,
    parse_extrinsics_signal,
    rotmat_to_qvec,
)
from glomap_lichtfeld_node import GLOMAPLichtfeldTracker  # noqa: E402


def _two_key_path(frames=90):
    return {"frames": frames, "path": [
        {"t": 0, "src": 0, "pos": [0.0, 0.0, 0.0], "look": [0.0, 0.0, 1.0]},
        {"t": frames - 1, "src": frames - 1, "pos": [0.6, 0.1, 0.6],
         "look": [0.0, 0.0, 1.0]},
    ]}


def _rig(frames, vfov=55.0, zm=1.5, width=112, height=64):
    """A camera rig with identity rotations and translation (i, 0, 0)."""
    c2w = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    for index in range(frames):
        c2w[index, 0, 3] = float(index)
    return {"c2w": c2w, "focal": np.ones(frames, dtype=np.float32),
            "vfov_degrees": vfov, "zm": zm, "width": width, "height": height,
            "frames": frames}


class _StubRenderWithRig:
    """render_depth_aligned stub that publishes a camera rig like the real one."""

    def __init__(self, rig):
        self.rig = rig
        self.kwargs = None

    def __call__(self, first, device, **kwargs):
        self.kwargs = kwargs
        fast_depth.LAST_CAMERA_EVAL = self.rig
        length = int(self.rig["frames"])
        frame = torch.zeros(length, 64, 112, 3)
        return frame, frame.clone(), 112, 64, length


def _checkerboard(size=16):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return ((xx + yy) % 2).float().unsqueeze(-1).expand(size, size, 3).clone()


def _flat(size=16):
    return torch.zeros(size, size, 3)


def _read_images_txt(path):
    """images.txt -> [(qvec[4], tvec[3], camera_id, name)] in file order."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10:
            continue
        entries.append((np.array([float(v) for v in fields[1:5]]),
                        np.array([float(v) for v in fields[5:8]]),
                        int(fields[8]), fields[9]))
    return entries


def _read_cameras_txt(path):
    fields = Path(path).read_text(encoding="utf-8").strip().splitlines()[-1].split()
    return {"model": fields[1], "width": int(fields[2]), "height": int(fields[3]),
            "params": [float(v) for v in fields[4:]]}


class ExtrinsicsSignalTests(unittest.TestCase):
    def test_signal_round_trips_the_rig(self):
        signal = build_extrinsics_signal(_rig(4))
        data = parse_extrinsics_signal(signal)
        self.assertEqual(data["frames"], 4)
        self.assertEqual(data["c2w"].shape, (4, 4, 4))
        np.testing.assert_allclose(data["c2w"][3][:3, 3], [3.0, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(data["vfov_degrees"], 55.0, places=5)
        self.assertAlmostEqual(data["zm"], 1.5, places=5)

    def test_empty_rig_writes_a_valid_document(self):
        document = json.loads(build_extrinsics_signal(None))
        self.assertEqual(document["frames"], 0)
        self.assertEqual(document["c2w"], [])

    def test_foreign_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown format"):
            parse_extrinsics_signal(json.dumps({"frames": 1}))
        with self.assertRaisesRegex(ValueError, "MERIDIAN_EXTRINSICS"):
            parse_extrinsics_signal("not json at all")


class PoseMathTests(unittest.TestCase):
    def test_c2w_to_w2c_inverts_the_rig(self):
        c2w = np.eye(4)
        c2w[:3, 3] = [1.0, 2.0, 3.0]
        w2c = c2w_to_w2c(c2w)
        self.assertEqual(w2c.shape, (3, 4))
        np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(w2c[:3, 3], [-1.0, -2.0, -3.0], atol=1e-12)

    def test_qvec_round_trips_through_the_colmap_parser(self):
        from enndee_colmap import COLMAPParser

        parser = COLMAPParser(".")
        rng = np.random.default_rng(7)
        for _ in range(8):
            rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1.0
            qvec = rotmat_to_qvec(rotation)
            self.assertAlmostEqual(float(np.linalg.norm(qvec)), 1.0, places=6)
            np.testing.assert_allclose(parser.qvec_to_rotmat(qvec), rotation,
                                       atol=1e-8)

    def test_mapping_is_identity_when_the_counts_match(self):
        for index in range(5):
            self.assertEqual(map_input_to_extrinsics(index, 5, 5, "auto", 0), index)

    def test_mapping_resamples_uniformly_on_retimed_video(self):
        got = [map_input_to_extrinsics(index, 5, 9, "auto", 0) for index in range(5)]
        self.assertEqual(got, [0, 2, 4, 6, 8])

    def test_mapping_manual_offset_for_chunked_video(self):
        got = [map_input_to_extrinsics(index, 4, 8, "manual", 4) for index in range(4)]
        self.assertEqual(got, [4, 5, 6, 7])

    def test_mapping_clamps_out_of_range(self):
        self.assertEqual(map_input_to_extrinsics(-3, 4, 8, "manual", 0), 0)
        self.assertEqual(map_input_to_extrinsics(99, 4, 8, "manual", 0), 7)


class GeometryExtrinsicsNodeTests(unittest.TestCase):
    def test_the_node_is_the_geometry_node_plus_one_output(self):
        self.assertEqual(
            EnndeeMeridianGeometryExtrinsics.RETURN_TYPES,
            geometry.EnndeeMeridianGeometry.RETURN_TYPES + ("MERIDIAN_EXTRINSICS",))
        self.assertEqual(
            EnndeeMeridianGeometryExtrinsics.RETURN_NAMES,
            ("source", "render", "width", "height", "length", "extrinsics"))
        inputs = EnndeeMeridianGeometryExtrinsics.INPUT_TYPES()
        self.assertEqual(list(inputs["optional"]),
                         ["image", "args_override", "custom_camera"])

    def test_build_appends_the_rendered_rig(self):
        rig = _rig(90)
        stub = _StubRenderWithRig(rig)
        with mock.patch.object(geometry, "render_depth_aligned", stub):
            outputs = EnndeeMeridianGeometryExtrinsics().build(
                video="unused.mp4", args="",
                image=torch.zeros(1, 64, 112, 3),
                custom_camera=json.dumps(_two_key_path(90)))
        self.assertEqual(len(outputs), 6)
        self.assertEqual(int(outputs[4]), 90)
        data = parse_extrinsics_signal(outputs[5])
        self.assertEqual(data["frames"], 90)
        np.testing.assert_allclose(data["c2w"][9][:3, 3], [9.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual(data["width"], 112)
        self.assertEqual(data["height"], 64)
class PathPoseDatasetTests(unittest.TestCase):
    """End-to-end export of the reduced tracker into a temp dataset folder."""

    def _export(self, images, frames, tmp, **overrides):
        signal = build_extrinsics_signal(_rig(frames))
        settings = dict(
            extrinsics=signal, images=images, lichtfeld_export_path=tmp,
            offset_splat=0, focal_mode="vfov from signal")
        settings.update(overrides)
        return EnndeePathPoseDataset().export_dataset(**settings)

    def test_dataset_passes_the_trainer_contract_with_selected_poses(self):
        from lichtfeld_training_node import validate_dataset

        images = torch.stack([_flat(), _checkerboard(), _flat(), _checkerboard()])
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, report_text = self._export(
                images, 4, tmp, sharp_selection=True, sharp_method="best_n",
                sharp_num_frames=2)
            dataset = validate_dataset(dataset_path)
            entries = _read_images_txt(dataset / "sparse" / "0" / "images.txt")
            # sharp selection keeps the two checkerboards: input frames 1 and 3,
            # identity mapping -> extrinsics 1 and 3 with t = (-i, 0, 0).
            self.assertEqual([name for *_, name in entries], ["0001.png", "0002.png"])
            for entry, (input_index, tvec) in zip(entries, [(1, [-1, 0, 0]),
                                                            (3, [-3, 0, 0])]):
                qvec, translation, camera_id, _name = entry
                np.testing.assert_allclose(qvec, [1, 0, 0, 0], atol=1e-6)
                np.testing.assert_allclose(translation, tvec, atol=1e-6)
                self.assertEqual(camera_id, 1)
            report = json.loads(report_text)
            self.assertEqual(report["final_output_to_input"], [1, 3])
            self.assertEqual(report["final_output_to_extrinsics"], [1, 3])

    def test_focal_follows_the_signal_vfov_at_the_export_resolution(self):
        images = torch.zeros(2, 32, 48, 3)
        with tempfile.TemporaryDirectory() as tmp:
            self._export(images, 2, tmp)
            camera = _read_cameras_txt(Path(tmp) / "sparse" / "0" / "cameras.txt")
        expected = 0.5 * 32 / math.tan(math.radians(55.0) / 2.0)
        self.assertEqual(camera["model"], "PINHOLE")
        self.assertEqual((camera["width"], camera["height"]), (48, 32))
        self.assertAlmostEqual(camera["params"][0], expected, places=3)
        self.assertAlmostEqual(camera["params"][1], expected, places=3)
        self.assertAlmostEqual(camera["params"][2], 24.0, places=6)
        self.assertAlmostEqual(camera["params"][3], 16.0, places=6)

    def test_manual_focal_and_hfov_modes(self):
        images = torch.zeros(1, 32, 48, 3)
        with tempfile.TemporaryDirectory() as tmp:
            self._export(images, 1, tmp, focal_mode="manual focal px", focal_px=777.0)
            camera = _read_cameras_txt(Path(tmp) / "sparse" / "0" / "cameras.txt")
            self.assertAlmostEqual(camera["params"][0], 777.0, places=3)
        with tempfile.TemporaryDirectory() as tmp:
            self._export(images, 1, tmp, focal_mode="manual hfov", hfov_deg=60.0)
            camera = _read_cameras_txt(Path(tmp) / "sparse" / "0" / "cameras.txt")
            self.assertAlmostEqual(camera["params"][0],
                                   0.5 * 48 / math.tan(math.radians(30.0)),
                                   places=3)
    def test_frame_step_composes_with_the_selection_and_keeps_poses(self):
        images = torch.stack([_flat(), _flat(), _checkerboard(), _flat(),
                              _checkerboard()])
        with tempfile.TemporaryDirectory() as tmp:
            report_text = self._export(
                images, 5, tmp, sharp_selection=True, sharp_method="best_n",
                sharp_num_frames=2, frame_step=2)[1]
            entries = _read_images_txt(Path(tmp) / "sparse" / "0" / "images.txt")
        # selection keeps [2, 4]; frame_step=2 then keeps input frame 2 only.
        report = json.loads(report_text)
        self.assertEqual(report["final_output_to_input"], [2])
        self.assertEqual(report["final_output_to_extrinsics"], [2])
        self.assertEqual(len(entries), 1)

    def test_mapping_offset_shifts_the_pose_assignment(self):
        images = torch.zeros(3, 16, 16, 3)
        with tempfile.TemporaryDirectory() as tmp:
            self._export(images, 6, tmp, extrinsics_mapping="manual",
                         extrinsics_offset=3)
            entries = _read_images_txt(Path(tmp) / "sparse" / "0" / "images.txt")
        for entry, index in zip(entries, [3, 4, 5]):
            np.testing.assert_allclose(entry[1], [-index, 0, 0], atol=1e-6)

    def test_explicit_masks_and_alpha_masks_both_reach_masks_folder(self):
        images = torch.stack([_flat(), _flat()])
        masks = torch.stack([torch.ones(16, 16), torch.zeros(16, 16)])
        with tempfile.TemporaryDirectory() as tmp:
            self._export(images, 2, tmp, masks=masks)
            self.assertTrue((Path(tmp) / "masks" / "0001.png").is_file())
            self.assertTrue((Path(tmp) / "masks" / "0002.png").is_file())

        rgba = torch.zeros(2, 16, 16, 4)
        rgba[..., 3] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            self._export(rgba, 2, tmp)
            self.assertTrue((Path(tmp) / "masks" / "0001.png").is_file())

    def test_sharp_selection_sidecar_records_the_pose_mapping(self):
        images = torch.stack([_flat(), _checkerboard(), _flat(), _checkerboard()])
        with tempfile.TemporaryDirectory() as tmp:
            self._export(images, 4, tmp, sharp_selection=True,
                         sharp_method="best_n", sharp_num_frames=2)
            record = json.loads(
                (Path(tmp) / "sharp_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(record["kept_input_indices"], [1, 3])
        self.assertEqual(record["final_output_to_input"], {"0": 1, "1": 3})

    def test_focal_from_reference_cameras_txt_is_rescaled_to_export_size(self):
        images = torch.zeros(1, 48, 64, 3)
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref"
            (ref / "sparse" / "0").mkdir(parents=True)
            (ref / "sparse" / "0" / "cameras.txt").write_text(
                "# comment\n1 SIMPLE_PINHOLE 3456 2304 4075.5 1728.0 1152.0\n",
                encoding="utf-8")
            self._export(images, 1, tmp, focal_mode="from cameras.txt",
                         focal_reference_path=str(ref))
            camera = _read_cameras_txt(Path(tmp) / "sparse" / "0" / "cameras.txt")
        # fx scales by the WIDTH ratio (HFOV preserved), fy by the HEIGHT ratio
        # (VFOV preserved) - identical values when the aspects match, as for real
        # video exports; the synthetic 4:3-vs-3:2 case pins the policy.
        self.assertAlmostEqual(camera["params"][0], 4075.5 * 64 / 3456.0, places=3)
        self.assertAlmostEqual(camera["params"][1], 4075.5 * 48 / 2304.0, places=3)

    def test_focal_reference_errors_are_clear(self):
        from enndee_path_pose_dataset import read_reference_intrinsics

        with self.assertRaisesRegex(ValueError, "focal_reference_path"):
            read_reference_intrinsics("", 64, 48)
        with self.assertRaisesRegex(ValueError, "focal_reference_path"):
            read_reference_intrinsics(r"C:\does\not\exist", 64, 48)


class DatasetNodeSchemaTests(unittest.TestCase):
    def test_only_extrinsics_is_a_required_socket(self):
        """ComfyUI demands a connection for every socket in "required" ("Missing
        connection" at queue time) - images and masks have working fallbacks and
        must stay OPTIONAL."""
        inputs = EnndeePathPoseDataset.INPUT_TYPES()
        self.assertIn("extrinsics", inputs["required"])
        for socket in ("images", "masks"):
            self.assertNotIn(socket, inputs["required"],
                             f"{socket} has a fallback and must be optional")
        self.assertEqual(list(inputs["optional"]), ["images", "masks"])

    def test_every_required_entry_is_a_widget_with_a_default(self):
        widget_types = ("STRING", "INT", "FLOAT", "BOOLEAN")
        inputs = EnndeePathPoseDataset.INPUT_TYPES()
        for name, (kind, metadata) in inputs["required"].items():
            if name == "extrinsics":
                continue
            self.assertTrue(kind in widget_types or isinstance(kind, list),
                            f"{name} must be a widget, not a connection socket")
            self.assertTrue("tooltip" in metadata or isinstance(kind, list),
                            f"{name} needs a tooltip")


if __name__ == "__main__":
    unittest.main()


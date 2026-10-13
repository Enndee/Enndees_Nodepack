"""Train a Gaussian splat with gsplat, as a drop-in for the LichtFeld Studio node.

Why this exists
---------------
The measured comparison that motivated it: a gsplat run on 175 COLMAP-posed frames reached
**PSNR 30.31 / SSIM 0.913** in **113 s**, while the LichtFeld node's own tuning notes record
**~13 dB** on a comparable subject set. Neither trainer solves SfM - both consume the same
`images/ + sparse/0/` dataset - so the interesting difference is *how the dataset reaches the
trainer*, and that is what this node controls end to end.

The outputs match :class:`LichtfeldHeadlessTrainer` exactly
(``output_path, command, log_file, summary``), so this node can replace it in a workflow
without rewiring anything downstream.

Two things this node does that a bare trainer invocation does not:

* **prepares the dataset** - COLMAP's undistorter needs a model, and the *uniform crop* is
  required by anything SuGaR-shaped downstream (see the tooltip on ``uniform_crop``);
* **pins the two flags that decide whether the result is usable** -
  ``--no-normalize-world-space`` and ``--data-factor 1``. Getting the first one wrong does
  not fail, it silently puts the Gaussians in a different frame from the poses.

The trainer runs as a subprocess under the environment that has gsplat (torch + CUDA
extension), because that is not ComfyUI's interpreter. Nothing is compiled here: gsplat is a
wheel install, and the only constraint is that the wheel matches the torch/CUDA/python it
runs against.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: interpreter that owns gsplat + torch. Not ComfyUI's.
DEFAULT_PYTHON = r"D:\Gaussian_Splatting\SuGaR\.conda\python.exe"
#: gsplat's own example trainer - the CLI this node drives.
DEFAULT_TRAINER = (r"D:\Gaussian_Splatting\Gsplat\pt212cu130-cp312-cp312"
                   r"\examples\simple_trainer.py")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def _first_existing(*candidates) -> str:
    """First candidate that exists, else the first one (so the tooltip still shows a path)."""
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return str(candidates[0]) if candidates else ""


def _find_model_dir(dataset: Path):
    """The COLMAP model inside a dataset folder: ``sparse/0``, ``sparse`` or ``sparse_norm/0``."""
    for candidate in (dataset / "sparse" / "0", dataset / "sparse",
                      dataset / "sparse_norm" / "0"):
        if (candidate / "images.bin").is_file() or (candidate / "images.txt").is_file():
            return candidate
    for candidate in sorted(dataset.rglob("images.bin")):
        return candidate.parent
    return None


def _find_images_dir(dataset: Path):
    for name in ("images", "Images", "rgb", "input"):
        if (dataset / name).is_dir():
            return dataset / name
    return None


def _log(lines, text, log_file=None):
    lines.append(text)
    print(text, flush=True)
    if log_file is not None:
        Path(log_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_dataset(dataset: Path, work_dir: Path, max_image_size: int,
                     uniform_crop: bool, lines, log_file):
    """Undistort a COLMAP dataset, optionally to one image size, and return what to train on.

    Undistortion matters because a 3DGS trainer treats the camera as a pinhole; feeding it
    distorted frames means every pixel's ray is slightly wrong, which shows up as softness
    that no amount of training fixes.

    The uniform crop matters because COLMAP's undistorter emits a **different width per
    frame** whenever the model carries one camera per frame (which is what a per-image
    bundle adjustment produces): measured on the Lara scenes, 12 distinct widths from 1210
    to 1224, with the per-frame focals spread over 22 %. SuGaR's coarse stage requires a
    single size and dies with ``shape '[-1, 1600, 1223, 3]' is invalid``. Cropping to the
    common minimum is *exact* rather than a compromise: the undistorted principal point is
    the image centre and SuGaR assumes exactly that, so a centred crop preserves
    ``cx = w/2`` with no focal rescaling.
    """
    import pycolmap
    from PIL import Image

    model = _find_model_dir(dataset)
    images = _find_images_dir(dataset)
    if model is None:
        raise ValueError(f"no COLMAP model under {dataset} "
                         "(expected sparse/0/images.bin or images.txt)")
    if images is None:
        raise ValueError(f"no images folder under {dataset}")

    undistorted = work_dir / "_undistorted"
    options = pycolmap.UndistortCameraOptions()
    if max_image_size:
        options.max_image_size = int(max_image_size)
    _log(lines, f"undistort: {len(list(images.iterdir()))} frames, "
                f"max_image_size={max_image_size} -> {undistorted}", log_file)
    started = time.time()
    if undistorted.exists():
        shutil.rmtree(undistorted)
    pycolmap.undistort_images(str(undistorted), str(model), str(images),
                              undistort_options=options)
    _log(lines, f"undistort done in {time.time() - started:.0f}s", log_file)

    if not uniform_crop:
        return undistorted

    target = work_dir / "dataset"
    if target.exists():
        shutil.rmtree(target)
    (target / "images").mkdir(parents=True, exist_ok=True)

    source_images = undistorted / "images"
    sizes = {}
    for path in sorted(source_images.glob("*")):
        with Image.open(path) as handle:
            sizes[path] = handle.size
    widths = [size[0] for size in sizes.values()]
    heights = [size[1] for size in sizes.values()]
    if not widths:
        raise ValueError(f"the undistorter produced no images in {source_images}")
    target_w, target_h = min(widths), min(heights)
    distinct = len({s for s in sizes.values()})
    _log(lines, f"uniform crop: {distinct} distinct size(s) -> {target_w}x{target_h}",
         log_file)

    for path, (width, height) in sizes.items():
        with Image.open(path) as handle:
            image = handle.convert(handle.mode)
            if (width, height) != (target_w, target_h):
                left = (width - target_w) // 2
                top = (height - target_h) // 2
                image = image.crop((left, top, left + target_w, top + target_h))
            image.save(target / "images" / path.name)

    # The model must be TXT *only*: pycolmap prefers cameras.bin whenever both exist, so
    # writing the correction next to a copied binary silently does nothing at all.
    sparse = target / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.Reconstruction(str(undistorted / "sparse"))
    reconstruction.write_text(str(sparse))

    cameras_txt = sparse / "cameras.txt"
    if cameras_txt.is_file():
        fixed = []
        for line in cameras_txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if line.startswith("#") or len(parts) < 5:
                fixed.append(line)
                continue
            model_name = parts[1]
            params = [float(v) for v in parts[4:]]
            if model_name == "PINHOLE" and len(params) >= 4:
                params[2] = target_w / 2.0
                params[3] = target_h / 2.0
            fixed.append(f"{parts[0]} {model_name} {target_w} {target_h} "
                         + " ".join(f"{v:.6f}" for v in params))
        cameras_txt.write_text("\n".join(fixed) + "\n", encoding="utf-8")

    check = pycolmap.Reconstruction(str(sparse))
    sample = next(iter(check.cameras.values()))
    _log(lines, f"dataset ready: {len(check.images)} images, "
                f"{sample.model_name} {sample.width}x{sample.height} "
                f"params={[round(float(v), 3) for v in sample.params]}", log_file)
    return target


def _read_psnr(ckpt: Path):
    """gsplat writes its own validation metrics; surface them instead of guessing."""
    import json

    stats = ckpt / "stats"
    if not stats.is_dir():
        return None
    for path in sorted(stats.glob("val_step*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "psnr" in data:
            return (float(data["psnr"]), float(data.get("ssim", float("nan"))),
                    float(data.get("lpips", float("nan"))), int(data.get("num_GS", 0)))
    return None


class GsplatHeadlessTrainer:
    """Train a 3DGS splat with gsplat's own trainer. Drop-in for the Lichtfeld node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Folder with images/ and sparse/0/. Connect the COLMAP or GLOMAP "
                        "tracker's dataset_path output. The model may be distorted - this "
                        "node undistorts it (see prepare_dataset)."
                    ),
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Where to write the splat and the log. Empty uses "
                        "<dataset>/output/gsplat_<timestamp>."
                    ),
                }),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Filename stem for the splat. Empty uses splat_<iterations>.",
                }),
                "iterations": ("INT", {
                    "default": 7000, "min": 100, "max": 1000000, "step": 500,
                    "tooltip": (
                        "Training steps. MEASURED: 7000 steps over 175 frames at 1600 px "
                        "reached PSNR 30.31 / SSIM 0.913 in 113 s. 7000 is also what "
                        "SuGaR's coarse and refine stages expect, so keep it at 7000 if a "
                        "SuGaR mesh or refined splat comes next. 30000 is the 3DGS paper "
                        "default and typically buys about +1 dB for roughly 4x the time."
                    ),
                }),
                "sh_degree": ("INT", {
                    "default": 3, "min": 0, "max": 3, "step": 1,
                    "tooltip": "Maximum spherical-harmonics degree. 3 is standard.",
                }),
                "data_factor": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": (
                        "Downscale images by this factor before training. Keep at 1 - the "
                        "measured 30.31 dB run used full resolution, and detail removed "
                        "here cannot be recovered by more iterations."
                    ),
                }),
                "normalize_world_space": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Leave this OFF. When the trainer normalises world space it "
                        "rescales the scene, so the Gaussians no longer live in the frame "
                        "the cameras.json poses describe. The render is then subtly wrong "
                        "rather than obviously broken - the worst kind of failure to "
                        "diagnose. (The flag passed is --no-normalize-world-space.)"
                    ),
                }),
                "prepare_dataset": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Undistort the dataset before training. A 3DGS trainer treats the "
                        "camera as a pinhole, so feeding it distorted frames makes every "
                        "pixel's ray slightly wrong - softness no amount of training fixes."
                    ),
                }),
                "uniform_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Crop every frame to one size after undistortion. Required if a "
                        "SuGaR stage comes next, harmless otherwise. COLMAP emits a "
                        "different width per frame when the model has one camera per frame "
                        "(measured: 12 widths from 1210 to 1224) and SuGaR's coarse stage "
                        "requires a single size."
                    ),
                }),
                "max_image_size": ("INT", {
                    "default": 1600, "min": 0, "max": 8192, "step": 160,
                    "tooltip": (
                        "Cap for the undistorted images (0 = native). The measured runs "
                        "used 1600 against 1248x1632 sources, so nearly all detail is kept."
                    ),
                }),
                "test_every": ("INT", {
                    "default": 8, "min": 2, "max": 1000, "step": 1,
                    "tooltip": (
                        "Every Nth frame is held out for validation (gsplat's own "
                        "--test-every). 8 is the default and is what produces the PSNR/SSIM "
                        "in the summary and the comparison renders - keep it on, or there "
                        "is no way to tell whether a run is good. There is no 'off' value: "
                        "a large number simply means almost nothing is held out."
                    ),
                }),
                "python_executable": ("STRING", {
                    "default": DEFAULT_PYTHON,
                    "tooltip": (
                        "Interpreter that owns gsplat + torch. NOT ComfyUI's interpreter - "
                        "gsplat is a compiled CUDA extension and needs the env it was "
                        "installed into."
                    ),
                }),
                "trainer_script": ("STRING", {
                    "default": DEFAULT_TRAINER,
                    "tooltip": "Path to gsplat's examples/simple_trainer.py.",
                }),
                "overwrite_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear an existing result directory before training.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_path", "command", "log_file", "summary")
    FUNCTION = "train"
    CATEGORY = "Enndee/3D"

    def train(self, dataset_path, output_path, output_name, iterations, sh_degree,
              data_factor, normalize_world_space, prepare_dataset, uniform_crop,
              max_image_size, test_every, python_executable, trainer_script,
              overwrite_output):
        lines = []
        stamp = time.strftime("%Y%m%d_%H%M%S")

        dataset = Path(dataset_path).expanduser()
        if not dataset.is_dir():
            raise ValueError(f"dataset folder does not exist: {dataset}")
        python = _first_existing(python_executable, DEFAULT_PYTHON)
        trainer = _first_existing(trainer_script, DEFAULT_TRAINER)
        if not Path(python).is_file():
            raise ValueError(
                f"no interpreter at {python}. gsplat needs the environment it was "
                "installed into (see the python_executable tooltip).")
        if not Path(trainer).is_file():
            raise ValueError(f"no gsplat trainer at {trainer}")

        out = Path(output_path).expanduser() if str(output_path).strip() \
            else dataset / "output" / f"gsplat_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        log_file = out / "gsplat_train.log"
        ckpt = out / "ckpt"
        if overwrite_output and ckpt.exists():
            shutil.rmtree(ckpt)

        _log(lines, "=" * 74, log_file)
        _log(lines, f"gsplat trainer: {dataset.name}", log_file)
        _log(lines, "=" * 74, log_file)

        train_dir = dataset
        if prepare_dataset:
            train_dir = _prepare_dataset(dataset, out, max_image_size, uniform_crop,
                                         lines, log_file)
        else:
            _log(lines, "prepare_dataset off - training on the folder as given", log_file)

        # Flags verified against the trainer's tyro config: it parses a dataclass, so the
        # flag names come from the field names (max_steps -> --max-steps,
        # normalize_world_space -> --no-normalize-world-space). There is deliberately no
        # --no-eval: the holdout is --test-every.
        command = [
            python, trainer, "default",
            "--data-dir", str(train_dir),
            "--data-factor", str(int(data_factor)),
            "--result-dir", str(ckpt),
            "--max-steps", str(int(iterations)),
            "--save-steps", str(int(iterations)),
            "--eval-steps", str(int(iterations)),
            "--ply-steps", str(int(iterations)),
            "--sh-degree", str(int(sh_degree)),
            "--test-every", str(int(test_every)),
            "--save-ply", "--disable-viewer",
        ]
        if normalize_world_space:
            _log(lines, "WARNING: normalize_world_space is ON - the Gaussians will end up in "
                        "a different frame from the poses. Leave it off.", log_file)
        else:
            command.append("--no-normalize-world-space")

        _log(lines, "$ " + " ".join(command), log_file)
        started = time.time()
        proc = subprocess.run(command, cwd=str(Path(trainer).parent.parent),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        elapsed = time.time() - started
        for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines()[-25:]:
            _log(lines, "   " + line, log_file)
        _log(lines, f"[gsplat] exit={proc.returncode} in {elapsed:.0f}s", log_file)
        if proc.returncode != 0:
            _log(lines, "training failed - see the log above", log_file)
            return (str(out), " ".join(command), str(log_file),
                    f"FAILED after {elapsed:.0f}s (exit {proc.returncode})")

        # gsplat writes ply/point_cloud_<last step>.ply; a stable name is what downstream
        # nodes can actually consume.
        plys = sorted((ckpt / "ply").glob("*.ply")) if (ckpt / "ply").is_dir() else []
        if not plys:
            _log(lines, f"no ply under {ckpt / 'ply'} - was --save-ply honoured?", log_file)
            return (str(out), " ".join(command), str(log_file),
                    f"FAILED: no ply produced ({elapsed:.0f}s)")

        stem = str(output_name).strip() or f"splat_{int(iterations)}"
        splat = out / f"{stem}.ply"
        shutil.copy2(plys[-1], splat)
        _log(lines, f"splat: {splat} ({splat.stat().st_size / 1e6:.1f} MB)", log_file)

        # SuGaR expects point_cloud/iteration_<n>/point_cloud.ply - the name is just a label,
        # but the path shape is what its loader looks for.
        sugar_ckpt = ckpt / "point_cloud" / f"iteration_{int(iterations)}"
        sugar_ckpt.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plys[-1], sugar_ckpt / "point_cloud.ply")

        metrics = _read_psnr(ckpt)
        if metrics:
            psnr, ssim, lpips, num_gs = metrics
            summary = (f"ok - {num_gs:,} Gaussians, PSNR {psnr:.2f} dB, SSIM {ssim:.3f}, "
                       f"LPIPS {lpips:.3f}, {elapsed:.0f}s")
        else:
            summary = (f"ok - {splat.stat().st_size / 1e6:.1f} MB in {elapsed:.0f}s "
                       "(no validation metrics found)")
        _log(lines, "", log_file)
        _log(lines, "SUMMARY: " + summary, log_file)
        _log(lines, f"splat -> {splat}", log_file)
        _log(lines, f"SuGaR checkpoint layout -> {sugar_ckpt}", log_file)
        return (str(out), " ".join(command), str(log_file), summary)



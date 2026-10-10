"""
Feed-forward multi-view geometry backends for the Lichtfeld dataset export.
============================================================================

This module is the *non-COLMAP* alternative for the Lichtfeld / 3DGS dataset
export: instead of SIFT + a global mapper it runs a **feed-forward pointmap
model** over the whole frame set in one forward pass and returns

    * per-view camera extrinsics (world-to-camera, OpenCV) + intrinsics,
    * per-view depth maps **in the same gauge as those poses**,
    * a dense point cloud (unprojected depth, confidence filtered).

Why one model instead of COLMAP + a monocular depth estimator
-------------------------------------------------------------
The depth supervision Lichtfeld consumes (``--use-depth-loss``) is only useful
when the depth prior agrees with the camera poses the trainer optimises.  A
per-frame monocular depth model (Depth Anything 2/3 style) produces an
**independent** scale and shift for every frame, so the prior fights the
geometry instead of reinforcing it.  A feed-forward multi-view model predicts
all views *jointly* (global attention over the whole set), so poses and depth
come out of one bundle, in one gauge - the prior can then only reinforce the
reconstruction, never inject drift.

Supported backends
------------------
``VGGT-Omega``
    Meta / Oxford VGGT-Omega (CVPR 2026).  Current state of the art for
    pointmap accuracy.  Needs a **local checkpoint file**; installed as the
    importable ``vggt_omega`` package.
``DA3-AnyView``
    ByteDance **Depth Anything 3**, any-view series (``depth-anything/DA3-*``,
    ICLR 2026, Apache-2.0).  Depth + confidence + extrinsics + intrinsics from
    one pass, exactly like VGGT, and the report puts it ahead of VGGT on camera
    pose and geometric accuracy.  This is the *any-view* DA3 - its mono sibling
    (single still, depth only) is what the Meridian node uses.
``VGG-T3``
    NVIDIA VGG-T^3 (CVPR 2026).  Replaces the quadratic softmax global
    attention with a linear test-time-training attention, so cost grows
    *linearly* with the view count (~1k images in under a minute).  Slightly
    less accurate than VGGT-Omega on standard benchmarks - pick it when the
    frame count gets large.

``DA3-AnyView`` and ``VGG-T3`` are **hub backends**: each has a default repo id
(see :data:`HUB_REPOS`) and pulls its weights with ``from_pretrained`` on first
use, so an empty ``model_path`` is fine and a missing local file must never be
read as "not installed".  Only ``VGGT-Omega`` requires a local checkpoint.  All
backends expose the same :class:`Reconstruction` result, so the dataset writer
below is backend agnostic.

.. note::
   The model checkpoints are **research licensed** (VGGT lineage is
   non-commercial; VGG-T^3 ships under the NVIDIA OneWay non-commercial
   license).  Check the licence before shipping anything commercial.

No hard coded developer paths: a checkpoint is resolved from the node widget,
then an environment variable, then a few generic locations relative to the
ComfyUI portable root / this pack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from enndee_da3 import (
    DA3_ANYVIEW_REPO,
    INSTALL_HINT as DA3_INSTALL_HINT,
    load_da3_api,
    local_da3_repo,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

#: selectable backend names (node widget order)
VGGT_OMEGA = "VGGT-Omega"
DA3_ANYVIEW = "DA3-AnyView"
VGG_T3 = "VGG-T3"
BACKEND_NAMES: Tuple[str, ...] = (VGGT_OMEGA, DA3_ANYVIEW, VGG_T3)

#: Backends whose weights live on the Hugging Face hub and are pulled by
#: ``from_pretrained`` - they need **no** local checkpoint file, so a missing
#: ``model_path`` must not be treated as "not installed".  The value is the
#: default repo id; ``ENV_REPO`` and the ``model_path`` widget override it.
HUB_REPOS = {
    DA3_ANYVIEW: DA3_ANYVIEW_REPO,
    VGG_T3: "nvidia/vgg-ttt",
}

#: env vars honoured for the checkpoint / repository of each backend
ENV_CHECKPOINT = {
    VGGT_OMEGA: "ENNDEE_VGGT_OMEGA_PATH",
    DA3_ANYVIEW: "ENNDEE_DA3_PATH",
    VGG_T3: "ENNDEE_VGGT_T3_PATH",
}

#: env vars that override the *hub repo id* of a hub based backend
ENV_REPO = {
    DA3_ANYVIEW: "ENNDEE_DA3_REPO",
    VGG_T3: "ENNDEE_VGG_T3_REPO",
}

#: environment variable that carries the pack root (set by the pack __init__)
PACK_ROOT_ENV = "ENNDEE_PACK_ROOT"

#: checkpoint file names searched inside the candidate folders
CHECKPOINT_NAMES = {
    VGGT_OMEGA: ("vggt_omega_1b_512.pt", "model.pt", "vggt_omega_1b_512.safetensors"),
    DA3_ANYVIEW: ("model.safetensors", "model.pt"),
    # a VGG-T^3 download is a *folder* handed straight to ``from_pretrained``
    VGG_T3: ("model.safetensors", "model.pt"),
}

#: folder names searched below the portable root / pack
SEARCH_FOLDERS = {
    VGGT_OMEGA: ("vggt-omega", "vggt_omega", "VGGT-Omega", "vggt-omega-fp16-version"),
    DA3_ANYVIEW: ("DA3-LARGE", "DA3-SMALL", "da3-large", "depth-anything-3"),
    VGG_T3: ("vgg-ttt", "vggttt", "VGG-T3", "vgg-ttt-main"),
}


def pack_root() -> Optional[Path]:
    """Folder of this module's pack (``.../custom_nodes/Enndees_Nodepack``)."""
    env = os.environ.get(PACK_ROOT_ENV, "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    here = Path(__file__).resolve().parent
    return here if (here / "enndee_bin.py").is_file() else None


def portable_root() -> Optional[Path]:
    """The ComfyUI **portable** root (the folder holding ``python_embeded``).

    Derived from the running interpreter instead of being hard coded, so the
    generic search below works on any machine layout.
    """
    try:
        exe = Path(sys.executable).resolve()
    except Exception:  # noqa: BLE001 - embedded interpreters can be odd
        return None
    for candidate in (exe.parent, exe.parent.parent):
        if (candidate / "python_embeded").is_dir() or (candidate / "ComfyUI").is_dir():
            return candidate
    return None


def candidate_folders(backend: str) -> List[Path]:
    """Generic folders that may contain a backend checkout, best guess first."""
    folders: List[Path] = []

    def add(path: Optional[Path]) -> None:
        if path is not None and path not in folders:
            folders.append(path)

    root = portable_root()
    if root is not None:
        # <portable>/../Tools/<model>  (the layout used on this machine) and
        # <portable>/Tools/<model> / <portable>/models/<model> as fallbacks.
        for parent in (root.parent, root):
            for name in SEARCH_FOLDERS.get(backend, ()):  # type: ignore[arg-type]
                add(parent / "Tools" / name)
                add(parent / "models" / name)
    pack = pack_root()
    if pack is not None:
        for name in SEARCH_FOLDERS.get(backend, ()):  # type: ignore[arg-type]
            add(pack / "models" / name)
            add(pack / "bin" / name)

    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        add(Path(hf_home))
    add(Path.home() / ".cache" / "huggingface" / "hub")
    return folders


def resolve_checkpoint(backend: str, explicit: str = "") -> Optional[Path]:
    """Locate a backend checkpoint / repository folder.

    Order: node widget -> environment variable -> generic folders (see
    :func:`candidate_folders`).  Returns ``None`` when nothing was found.
    """
    explicit = (explicit or "").strip()
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        # a widget value that does not exist is reported by the caller
        return None

    env = os.environ.get(ENV_CHECKPOINT.get(backend, ""), "").strip()
    if env:
        path = Path(env)
        if path.exists():
            return path

    if backend in HUB_REPOS:
        # Hub backends fetch their weights with ``from_pretrained``.  Scanning the
        # folders below would happily match some *other* model's model.safetensors
        # (it sits in the same Hugging Face cache), which reads as "checkpoint
        # found" and is worse than no answer at all - so only an explicit path or
        # the env var counts as a local copy here.
        return None

    names = CHECKPOINT_NAMES.get(backend, ())
    for folder in candidate_folders(backend):
        if not folder.is_dir():
            continue
        for name in names:
            candidate = folder / name
            if candidate.exists():
                return candidate
        for name in names:
            for hit in folder.glob(f"**/{name}"):
                if hit.exists():
                    return hit
    return None


def resolve_hub_repo(backend: str, explicit: str = "") -> str:
    """Hub repo id for a backend whose weights come from the Hugging Face hub.

    Order: the ``model_path`` widget -> the backend's env var -> :data:`HUB_REPOS`.
    A widget value that is **not** an existing path is read as a repo id (that is
    how ``model_path`` doubles as the variant picker: ``depth-anything/DA3-SMALL``,
    ``nvidia/vgg-ttt``).  Returns ``""`` for a backend with no hub repo, which is
    the signal that a local checkpoint is mandatory (VGGT-Omega).
    """
    explicit = (explicit or "").strip()
    if explicit and not Path(explicit).exists():
        return explicit
    env = (os.environ.get(ENV_REPO.get(backend, ""), "") or "").strip()
    if env:
        return env
    return HUB_REPOS.get(backend, "")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class Reconstruction:
    """Backend independent output of one feed-forward reconstruction.

    All arrays are numpy; poses are **world-to-camera in OpenCV** coordinates
    (exactly what COLMAP stores), depth is positive forward-z ("larger =
    farther"), and every quantity lives in the *same* gauge.
    """

    def __init__(self, extrinsics: np.ndarray, intrinsics: np.ndarray,
                 depth: np.ndarray, depth_conf: np.ndarray,
                 model_hw: Tuple[int, int], source: str):
        self.extrinsics = np.asarray(extrinsics, dtype=np.float64)   # [N, 3, 4]
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)   # [N, 3, 3]
        self.depth = np.asarray(depth, dtype=np.float32)             # [N, h, w]
        self.depth_conf = np.asarray(depth_conf, dtype=np.float32)   # [N, h, w]
        self.model_hw = (int(model_hw[0]), int(model_hw[1]))
        self.source = str(source)

    @property
    def num_views(self) -> int:
        return int(self.extrinsics.shape[0])

    def confidences(self) -> np.ndarray:
        """Mean confidence per view (used as the node's quality score)."""
        if self.depth_conf.size == 0:
            return np.ones(self.num_views, np.float32)
        per_view = self.depth_conf.reshape(self.depth_conf.shape[0], -1)
        return per_view.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared image preprocessing
# ---------------------------------------------------------------------------

def _composite_on_white(rgb: np.ndarray) -> np.ndarray:
    """Drop an alpha channel by compositing on white (upstream behaviour)."""
    if rgb.shape[-1] == 3:
        return rgb
    alpha = rgb[..., 3:4].astype(np.float32)
    return (rgb[..., :3].astype(np.float32) * alpha + (1.0 - alpha)).astype(np.float32)


def balanced_target_shape(height: int, width: int, image_resolution: int,
                          patch_size: int = 16) -> Tuple[int, int]:
    """Patch aligned size whose token count is close to ``image_resolution**2``.

    Mirrors the ``balanced`` mode of the upstream loaders, but **without** the
    centre crop they apply to extreme aspect ratios - the crop would make the
    depth -> image mapping non invertible, and our frames share one aspect
    ratio anyway.
    """
    aspect = float(height) / max(float(width), 1.0)
    tokens = float((image_resolution // patch_size) ** 2)
    w_patches = max(1.0, float(np.sqrt(tokens / max(aspect, 1e-6))))
    h_patches = max(1.0, tokens / w_patches)
    target_h = max(patch_size, int(round(h_patches)) * patch_size)
    target_w = max(patch_size, int(round(w_patches)) * patch_size)
    return target_h, target_w


def preprocess_images(images: np.ndarray, image_resolution: int = 512,
                      patch_size: int = 16, device: str = "cuda"):
    """``(N, H, W, 3|4)`` in [0, 1] -> ``(1, N, 3, h, w)`` float tensor + ``(h, w)``.

    Returns the model input plus the model resolution so the caller can map the
    predicted depth back to the original image size.
    """
    import torch

    batch = np.asarray(images)
    if batch.ndim != 4:
        raise ValueError(f"expected (N, H, W, C) images, got {batch.shape}")
    batch = _composite_on_white(batch.astype(np.float32, copy=False))

    height, width = int(batch.shape[1]), int(batch.shape[2])
    target_h, target_w = balanced_target_shape(height, width, int(image_resolution), patch_size)

    tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
    if (target_h, target_w) != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor, size=(target_h, target_w), mode="bicubic", align_corners=False,
        )
    tensor = tensor.clamp_(0.0, 1.0).unsqueeze(0).to(device)
    return tensor, (target_h, target_w)


def squeeze_channel_dim(tensor):
    """Drop trailing singleton axes so depth/confidence come out as ``[N, H, W]``.

    The backends disagree about the shape they hand back: VGGT-Omega returns
    ``[N, H, W, 1, 1]`` (hence the historical ``ndim == 5`` squeeze) while VGG-T^3
    returns ``[N, H, W, 1]``.  Everything downstream - the confidence gate, the
    upsampler, the 16-bit writer - expects a plain ``[N, H, W]``, so a trailing
    ``1`` has to go either way.  Leaving the VGG-T^3 form in place made the depth
    maps 4-D and the confidence maps 3-D in the *same* result.
    """
    while tensor.ndim > 3 and int(tensor.shape[-1]) == 1:
        tensor = tensor[..., 0]
    return tensor


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _Backend:
    """Common backend interface."""

    name = ""

    def __init__(self, checkpoint: Optional[Path] = None, repo: str = "",
                 autocast: bool = True):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.repo = (repo or "").strip()
        self.autocast = bool(autocast)
        self._model = None

    # -- availability ------------------------------------------------------
    def missing_module(self) -> str:
        raise NotImplementedError

    def install_hint(self) -> str:
        """One line telling the user how to make this backend usable."""
        return (f"python package '{self.missing_module()}' is not installed for the "
                f"ComfyUI interpreter. Install it into python_embeded, then restart "
                f"ComfyUI.")

    @property
    def hub_repo(self) -> str:
        """Hub repo id this backend pulls from (``""`` when it has none)."""
        return resolve_hub_repo(self.name, self.repo)

    def needs_local_checkpoint(self) -> bool:
        """True when the backend cannot run without a local weight file.

        The hub based backends (``DA3-AnyView``, ``VGG-T3``) fetch their weights
        with ``from_pretrained`` on first use, so an empty ``model_path`` is
        perfectly fine for them.  Requiring a local checkpoint unconditionally is
        what made VGG-T3 report "no checkpoint found" and quietly return nothing
        even though it never needed one.
        """
        return not self.hub_repo

    def status(self) -> Tuple[bool, str]:
        """``(usable, human readable reason)`` - never raises."""
        if self.needs_local_checkpoint() and self.checkpoint is None:
            return False, (f"{self.name}: no checkpoint found. Set the 'model_path' "
                           f"widget or {ENV_CHECKPOINT.get(self.name, 'the env var')}.")
        try:
            import importlib

            importlib.import_module(self.missing_module())
        except Exception:  # noqa: BLE001 - optional dependency
            if not self._repo_candidates():
                return False, f"{self.name}: {self.install_hint()}"
        if self.checkpoint is not None:
            return True, f"{self.name}: {self.checkpoint}"
        return True, f"{self.name}: {self.hub_repo} (hub, downloaded on first run)"

    def _repo_candidates(self) -> List[Path]:
        candidates: List[Path] = []
        if self.repo:
            candidates.append(Path(self.repo))
        if self.checkpoint is not None:
            candidates.extend([self.checkpoint.parent, self.checkpoint.parent.parent])
            if self.checkpoint.is_dir():
                candidates.insert(0, self.checkpoint)
        return [c for c in candidates if c.is_dir()]

    def _ensure_importable(self) -> None:
        """Import the backend package, adding a local checkout to ``sys.path``."""
        try:
            import importlib

            importlib.import_module(self.missing_module())
            return
        except Exception:  # noqa: BLE001 - fall through to the path search
            pass
        for folder in self._repo_candidates():
            for entry in (folder, folder.parent):
                if str(entry) not in sys.path:
                    sys.path.insert(0, str(entry))
        import importlib

        importlib.import_module(self.missing_module())

    # -- inference ---------------------------------------------------------
    def load(self):
        raise NotImplementedError

    def run(self, images: np.ndarray, image_resolution: int = 512,
            device: str = "cuda") -> Reconstruction:
        raise NotImplementedError

    def release(self) -> None:
        """Free the model and the CUDA cache."""
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - torch always present in ComfyUI
            pass


class VGGTOmegaBackend(_Backend):
    """Meta / Oxford VGGT-Omega (``vggt_omega``)."""

    name = VGGT_OMEGA
    PATCH_SIZE = 16

    def missing_module(self) -> str:
        return "vggt_omega"

    def load(self):
        if self._model is not None:
            return self._model
        self._ensure_importable()
        import torch

        from vggt_omega.models import VGGTOmega

        model = VGGTOmega(autocast=self.autocast).eval()
        try:
            state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
        except TypeError:  # older torch without weights_only
            state = torch.load(str(self.checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model" in state and not any(
            k.startswith("aggregator") for k in state
        ):
            state = state["model"]
        model.load_state_dict(state)
        self._model = model.to("cuda")
        return self._model

    def run(self, images: np.ndarray, image_resolution: int = 512,
            device: str = "cuda") -> Reconstruction:
        import torch

        # load() first: it puts a local checkout on sys.path, which the
        # pose_enc import below needs when the package is not pip installed.
        model = self.load()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        tensor, model_hw = preprocess_images(
            images, image_resolution, self.PATCH_SIZE, device)
        with torch.inference_mode():
            predictions = model(tensor)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:])
        depth = squeeze_channel_dim(predictions["depth"])
        depth_conf = predictions.get("depth_conf")
        if depth_conf is None:
            depth_conf = torch.ones_like(depth)
        depth_conf = squeeze_channel_dim(depth_conf)

        return Reconstruction(
            extrinsics=extrinsics[0].float().cpu().numpy(),
            intrinsics=intrinsics[0].float().cpu().numpy(),
            depth=depth[0].float().cpu().numpy(),
            depth_conf=depth_conf[0].float().cpu().numpy(),
            model_hw=model_hw,
            source=f"{self.name}:{self.checkpoint.name}",
        )


class DA3AnyViewBackend(_Backend):
    """Depth Anything 3, **any-view** models (``depth-anything/DA3-*``).

    The any-view DA3 is the same class of model as VGGT-Omega: depth, confidence,
    extrinsics and intrinsics all come out of **one forward pass over the whole
    frame set**, so the depth it writes is in the gauge of the poses it also
    wrote.  That is what makes it safe for Lichtfeld's depth loss, and it is why
    this backend lives here and not next to COLMAP.

    Why it is offered at all: the DA3 report (arXiv 2511.10647, ByteDance Seed,
    ICLR 2026) puts the any-view models **+35.7 % on camera pose accuracy and
    +23.6 % on geometric accuracy over VGGT** on their visual-geometry benchmark,
    and the weights are Apache-2.0.  Its mono sibling is what the Meridian node
    uses - single still, depth only, no poses.

    ``model_path`` / ``ENNDEE_DA3_REPO`` select the variant: any of
    ``depth-anything/DA3-SMALL``, ``-BASE`` or ``-LARGE`` (the default), or a
    local folder.  Nothing is downloaded until the first run.
    """

    name = DA3_ANYVIEW
    #: DINOv2 patch size - DA3 rounds its processed size to a multiple of this
    PATCH_SIZE = 14

    def missing_module(self) -> str:
        return "depth_anything_3"

    def install_hint(self) -> str:
        return ("python package 'depth_anything_3' is not installed for the ComfyUI "
                f"interpreter. In python_embeded run: {DA3_INSTALL_HINT}")

    def load(self):
        if self._model is not None:
            return self._model
        DepthAnything3 = load_da3_api()
        reference = (local_da3_repo(self.repo) or self.hub_repo
                     or DA3_ANYVIEW_REPO)
        model = DepthAnything3.from_pretrained(reference).to("cuda").eval()
        self._model = model
        return self._model

    def run(self, images: np.ndarray, image_resolution: int = 512,
            device: str = "cuda") -> Reconstruction:
        import torch

        model = self.load()
        # DA3 preprocesses internally (aspect preserving, longest side capped at
        # process_res and rounded to a multiple of 14 - no padding, no crop), so
        # hand it plain uint8 frames and let it do the resizing
        frames = []
        for frame in np.asarray(images):
            rgb = _composite_on_white(np.asarray(frame, dtype=np.float32))
            frames.append((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8))

        with torch.inference_mode():
            prediction = model.inference(frames, process_res=int(image_resolution))

        depth = np.asarray(prediction.depth, dtype=np.float32)          # [N, h, w]
        conf = getattr(prediction, "conf", None)
        conf = (np.asarray(conf, dtype=np.float32) if conf is not None
                else np.ones_like(depth))
        # DA3 already returns world-to-camera [N, 3, 4] - no inversion needed
        extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
        intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)

        return Reconstruction(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            depth_conf=conf,
            model_hw=(int(depth.shape[-2]), int(depth.shape[-1])),
            source=f"{self.name}:{self.hub_repo or DA3_ANYVIEW_REPO}",
        )


class VGGT3Backend(_Backend):
    """NVIDIA VGG-T^3 (``vggttt``) - linear time in the view count.

    Same VGGT style API as upstream::

        model = VGGT.from_pretrained("nvidia/vgg-ttt").eval().cuda()
        preds = model.infer(images)   # pose, intrinsics, pts3d, conf, depth

    ``pose`` is camera-to-world, so it is inverted to the COLMAP convention.
    """

    name = VGG_T3
    PATCH_SIZE = 14

    def missing_module(self) -> str:
        return "vggttt"

    def install_hint(self) -> str:
        return ("python package 'vggttt' is not installed for the ComfyUI "
                "interpreter. Install it from the official repo WITHOUT its "
                "requirements - they pin torch==2.7.1 and would downgrade the CUDA "
                "build:  python -m pip install --no-deps "
                "git+https://github.com/nv-dvl/vgg-ttt")

    def load(self):
        if self._model is not None:
            return self._model
        self._ensure_importable()

        from vggttt.nets.vggt.models.vggt import VGGT

        # from_pretrained takes a repo id or a model *folder* - a bare checkpoint
        # file cannot be loaded that way, so fall back to the hub repo for it
        reference = self.hub_repo or "nvidia/vgg-ttt"
        if self.checkpoint is not None and self.checkpoint.is_dir():
            reference = str(self.checkpoint)
        elif self.checkpoint is not None:
            print(f"[Enndees] VGG-T3: {self.checkpoint.name} is a file, but "
                  f"from_pretrained needs a repo id or a model folder - "
                  f"using {reference}")
        model = VGGT.from_pretrained(reference).eval()
        self._model = model.to("cuda")
        return self._model

    def run(self, images: np.ndarray, image_resolution: int = 512,
            device: str = "cuda") -> Reconstruction:
        import torch

        tensor, model_hw = preprocess_images(
            images, image_resolution, self.PATCH_SIZE, device)
        model = self.load()
        # VGG-T^3's infer() takes [#images, 3, H, W] and adds the batch dimension
        # itself, while preprocess_images returns VGGT-Omega's [1, N, 3, H, W] -
        # passing the 5-D form straight through fails with
        # "ValueError: too many values to unpack (expected 4)"
        with torch.inference_mode():
            predictions = model.infer(tensor[0])

        poses = predictions["pose"]
        intrinsics = predictions["intrinsics"]
        depth = squeeze_channel_dim(predictions["depth"])
        conf = predictions.get("conf")
        if conf is None:
            conf = torch.ones_like(depth)
        conf = squeeze_channel_dim(conf)

        # camera-to-world -> world-to-camera (COLMAP stores w2c)
        rotation = poses[:, :3, :3]
        translation = poses[:, :3, 3:4]
        rotation_t = rotation.transpose(1, 2)
        extrinsics = torch.cat([rotation_t, -torch.bmm(rotation_t, translation)], dim=2)

        return Reconstruction(
            extrinsics=extrinsics.float().cpu().numpy(),
            intrinsics=intrinsics.float().cpu().numpy(),
            depth=depth.float().cpu().numpy(),
            depth_conf=conf.float().cpu().numpy(),
            model_hw=model_hw,
            source=(f"{self.name}:{self.checkpoint.name}" if self.checkpoint
                    else f"{self.name}:{self.hub_repo or 'nvidia/vgg-ttt'}"),
        )


#: name -> backend class
BACKENDS: Dict[str, type] = {
    VGGT_OMEGA: VGGTOmegaBackend,
    DA3_ANYVIEW: DA3AnyViewBackend,
    VGG_T3: VGGT3Backend,
}


def make_backend(name: str, checkpoint: Optional[Path] = None, repo: str = "",
                 autocast: bool = True) -> _Backend:
    """Instantiate the backend selected in the node."""
    key = str(name or "").strip()
    for known, backend in BACKENDS.items():
        if known.lower() == key.lower():
            return backend(checkpoint=checkpoint, repo=repo, autocast=autocast)
    raise ValueError(f"Unknown feed-forward backend {name!r}; "
                     f"expected one of {', '.join(BACKEND_NAMES)}")


def backend_report(explicit: str = "") -> str:
    """One line per backend describing whether it can run right now.

    ``explicit`` is the node's ``model_path`` widget, so the report reflects the
    repo id the user actually typed instead of only the default.
    """
    lines = []
    for name in BACKEND_NAMES:
        checkpoint = resolve_checkpoint(name, explicit)
        backend = BACKENDS[name](checkpoint=checkpoint, repo=explicit)
        usable, reason = backend.status()
        lines.append(f"  {'OK ' if usable else '-- '} {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def scale_intrinsics(intrinsics: np.ndarray, model_hw: Tuple[int, int],
                     image_hw: Tuple[int, int]) -> np.ndarray:
    """Rescale intrinsics from the model resolution to the image resolution.

    The preprocessing is a pure resize, so ``fx``/``fy``/``cx``/``cy`` scale by
    the width/height ratio.  Lichtfeld (and COLMAP) expect intrinsics that match
    the *written* image size.
    """
    intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    model_h, model_w = int(model_hw[0]), int(model_hw[1])
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    sx = float(image_w) / float(max(model_w, 1))
    sy = float(image_h) / float(max(model_h, 1))
    intrinsics[..., 0, 0] *= sx
    intrinsics[..., 0, 2] *= sx
    intrinsics[..., 1, 1] *= sy
    intrinsics[..., 1, 2] *= sy
    return intrinsics


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> ``[qw, qx, qy, qz]`` (COLMAP order).

    Exact inverse of :meth:`COLMAPParser.qvec_to_rotmat` (Hamilton convention).
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qvec = np.array([0.25 * scale,
                         (rotation[2, 1] - rotation[1, 2]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale,
                         (rotation[1, 0] - rotation[0, 1]) / scale])
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qvec = np.array([(rotation[2, 1] - rotation[1, 2]) / scale,
                         0.25 * scale,
                         (rotation[0, 1] + rotation[1, 0]) / scale,
                         (rotation[0, 2] + rotation[2, 0]) / scale])
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qvec = np.array([(rotation[0, 2] - rotation[2, 0]) / scale,
                         (rotation[0, 1] + rotation[1, 0]) / scale,
                         0.25 * scale,
                         (rotation[1, 2] + rotation[2, 1]) / scale])
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qvec = np.array([(rotation[1, 0] - rotation[0, 1]) / scale,
                         (rotation[0, 2] + rotation[2, 0]) / scale,
                         (rotation[1, 2] + rotation[2, 1]) / scale,
                         0.25 * scale])
    norm = np.linalg.norm(qvec)
    return qvec / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray,
                    extrinsics: np.ndarray) -> np.ndarray:
    """Depth maps -> world points, one cloud per view.

    ``depth`` is ``[N, h, w]``, ``intrinsics`` ``[N, 3, 3]`` (matching ``h``/``w``)
    and ``extrinsics`` ``[N, 3, 4]`` world-to-camera.  Returns ``[N, h, w, 3]``
    in COLMAP/OpenCV world coordinates.
    """
    depth = np.asarray(depth, dtype=np.float64)
    count, height, width = depth.shape
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid_x = np.broadcast_to(grid_x[None], (count, height, width)).astype(np.float64)
    grid_y = np.broadcast_to(grid_y[None], (count, height, width)).astype(np.float64)

    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    camera_points = np.stack([(grid_x - cx) / fx * depth,
                              (grid_y - cy) / fy * depth,
                              depth], axis=-1)
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(rotation, (0, 2, 1)),
                     camera_points - translation[:, None, None, :])


# ---------------------------------------------------------------------------
# Depth maps (Lichtfeld depth loss)
# ---------------------------------------------------------------------------

def depth_valid_mask(depth: np.ndarray, confidence: np.ndarray,
                     conf_percentile: float = 20.0) -> np.ndarray:
    """Pixels that carry a usable depth: finite, positive and confident enough.

    ``conf_percentile`` is a *relative* gate - the threshold is the given
    percentile of the confidence of the same image, so it adapts to each view
    instead of assuming an absolute confidence scale.
    """
    depth = np.asarray(depth, dtype=np.float64)
    finite = np.isfinite(depth) & (depth > 0)
    if confidence is None or np.size(confidence) == 0:
        return finite
    confidence = np.asarray(confidence, dtype=np.float64)
    threshold = np.percentile(confidence, float(conf_percentile))
    return finite & (confidence >= threshold)


def upsample_depth_maps(depth: np.ndarray, valid: np.ndarray,
                        image_hw: Tuple[int, int]) -> np.ndarray:
    """Resample ``[N, h, w]`` depth (plus validity) to the image resolution.

    Bicubic for the values, nearest for the mask, then invalid pixels are set to
    ``0`` - Lichtfeld's depth loader reports ``valid_pixels``/``valid_fraction``,
    i.e. it treats ``0`` as "no depth" (a zero depth is physically impossible).
    """
    import torch

    target_h, target_w = int(image_hw[0]), int(image_hw[1])
    values = torch.from_numpy(np.asarray(depth, dtype=np.float32))[:, None]
    mask = torch.from_numpy(np.asarray(valid, dtype=np.float32))[:, None]
    if (int(depth.shape[1]), int(depth.shape[2])) != (target_h, target_w):
        values = torch.nn.functional.interpolate(
            values, size=(target_h, target_w), mode="bicubic", align_corners=False)
        mask = torch.nn.functional.interpolate(
            mask, size=(target_h, target_w), mode="nearest")
    values = values[:, 0].numpy()
    mask = mask[:, 0].numpy() >= 0.5
    values[~mask] = 0.0
    return values.astype(np.float32)


def depth_to_uint16(depth: np.ndarray, valid: Optional[np.ndarray] = None,
                    near_percentile: float = 0.5,
                    far_percentile: float = 99.5) -> np.ndarray:
    """Normalise depth maps to 16-bit, ``larger = farther``, ``0 = invalid``.

    The normalisation is per image and uses robust percentiles.  This is safe:
    both Lichtfeld depth loss modes are invariant to a per-image scale and shift
    (``pearson`` is a correlation, ``adaptive-warped-l1`` fits a per-image affine
    warp), so only the *relative* ordering inside one image has to survive.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3:
        raise ValueError(f"expected (N, H, W) depth, got {depth.shape}")
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)

    out = np.zeros(depth.shape, dtype=np.uint16)
    for index in range(depth.shape[0]):
        frame = depth[index]
        mask = valid[index]
        if not mask.any():
            continue
        values = frame[mask]
        near = float(np.percentile(values, float(near_percentile)))
        far = float(np.percentile(values, float(far_percentile)))
        if not np.isfinite(near) or not np.isfinite(far) or far <= near:
            near, far = float(values.min()), float(values.max())
        if far <= near:
            far = near + 1e-6
        scaled = np.clip((frame - near) / (far - near), 0.0, 1.0)
        frame_out = (scaled * 65535.0).round().astype(np.uint16)
        # keep at least one unit of headroom so "far" never collides with invalid
        frame_out[frame_out == 0] = 1
        frame_out[~mask] = 0
        out[index] = frame_out
    return out


def write_depth_maps(depth_dir, stems: Sequence[str], depth_uint16: np.ndarray,
                     log=None) -> int:
    """Write ``<stem>.depth.png`` (16-bit grayscale) for every view.

    Lichtfeld scans a ``depth/`` (or ``depths/``) folder next to ``images/``,
    matches ``<image stem>.depth.png`` and requires the **same resolution** as
    the image.  Returns the number of files written.
    """
    from PIL import Image

    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # stale maps from an earlier run would be matched to the wrong frames
    for existing in depth_dir.iterdir():
        if existing.is_file() and existing.name.endswith(".depth.png"):
            existing.unlink()

    written = 0
    for index, stem in enumerate(stems):
        if index >= len(depth_uint16):
            break
        array = np.ascontiguousarray(depth_uint16[index], dtype=np.uint16)
        # no ``mode=`` argument: Pillow >= 11 deprecates it and infers "I;16"
        # from the uint16 array (verified lossless round trip, 16-bit PNG)
        Image.fromarray(array).save(depth_dir / f"{stem}.depth.png")
        written += 1
    if log is not None:
        log(f"{written} depth maps -> {depth_dir} (<stem>.depth.png, 16-bit)")
    return written


# ---------------------------------------------------------------------------
# Point cloud + COLMAP TXT model
# ---------------------------------------------------------------------------

def sample_point_cloud(points: np.ndarray, colors: np.ndarray, valid: np.ndarray,
                       max_points: int = 400000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten the per-view clouds into one subsampled ``(points, colors)`` pair.

    ``points``/``colors`` are ``[N, h, w, 3]`` at the model resolution and
    ``valid`` is ``[N, h, w]``.  Colours are returned as floats in ``[0, 1]``
    (the same convention as :meth:`COLMAPParser.get_point_cloud`).
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    flat_points = points[mask].reshape(-1, 3)
    flat_colors = colors[mask].reshape(-1, 3)
    if len(flat_points) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    limit = int(max_points)
    if limit > 0 and len(flat_points) > limit:
        generator = np.random.default_rng(int(seed))
        keep = generator.choice(len(flat_points), size=limit, replace=False)
        keep.sort()
        flat_points = flat_points[keep]
        flat_colors = flat_colors[keep]
    return flat_points, np.clip(flat_colors, 0.0, 1.0)


def write_colmap_model(export_dir, extrinsics: np.ndarray, intrinsics: np.ndarray,
                       image_names: Sequence[str], image_hw: Tuple[int, int],
                       points: Optional[np.ndarray] = None,
                       colors: Optional[np.ndarray] = None,
                       per_view_intrinsics: bool = False, log=None) -> Path:
    """Write ``sparse/0/{cameras,images,points3D}.txt`` from a reconstruction.

    The TXT files are produced by the vendored :class:`COLMAPParser`, i.e. by
    exactly the same writer the COLMAP node uses - the two nodes are
    interchangeable for Lichtfeld.

    ``extrinsics`` is ``[N, 3, 4]`` world-to-camera (COLMAP convention) and
    ``intrinsics`` ``[N, 3, 3]`` already scaled to the image resolution.
    """
    from enndee_colmap.colmap_parser import COLMAPParser

    export_dir = Path(export_dir)
    target = export_dir / "sparse" / "0"
    target.mkdir(parents=True, exist_ok=True)

    height, width = int(image_hw[0]), int(image_hw[1])
    parser = COLMAPParser(str(target))
    parser.cameras = {}
    parser.images = {}
    parser.points3d = {}

    if per_view_intrinsics:
        camera_of = {}
        for index in range(len(image_names)):
            camera_id = index + 1
            params = intrinsics[index]
            parser.cameras[camera_id] = {
                "model_id": 1,
                "model_name": "PINHOLE",
                "width": width,
                "height": height,
                "params": np.array([params[0, 0], params[1, 1], params[0, 2], params[1, 2]]),
            }
            camera_of[index] = camera_id
    else:
        # one shared camera (what COLMAP / GLOMAP produce for a video): the
        # average of the predicted intrinsics keeps the model clean and avoids
        # relying on Lichtfeld honouring per-image cameras.
        mean = np.asarray(intrinsics, dtype=np.float64).mean(axis=0)
        parser.cameras[1] = {
            "model_id": 1,
            "model_name": "PINHOLE",
            "width": width,
            "height": height,
            "params": np.array([mean[0, 0], mean[1, 1], mean[0, 2], mean[1, 2]]),
        }
        camera_of = {index: 1 for index in range(len(image_names))}

    for index, name in enumerate(image_names):
        pose = np.asarray(extrinsics[index], dtype=np.float64)
        parser.images[index + 1] = {
            "qvec": rotmat_to_qvec(pose[:3, :3]),
            "tvec": pose[:3, 3].copy(),
            "camera_id": camera_of[index],
            "name": str(name),
            "points2d": np.zeros((0, 2)),
            "point3d_ids": np.zeros((0,), dtype=np.int64),
        }

    if points is not None and colors is not None and len(points) > 0:
        rgb = np.clip(np.asarray(colors, dtype=np.float64) * 255.0, 0, 255)
        for index in range(len(points)):
            parser.points3d[index + 1] = {
                "xyz": np.asarray(points[index], dtype=np.float64),
                "rgb": rgb[index].round().astype(np.int64),
                "error": 0.0,
                "track": [],
            }

    images_dir = export_dir / "images"
    parser.write_txt(target, images_dir if images_dir.is_dir() else None)
    if log is not None:
        log(f"Sparse export -> {target} (cameras.txt, images.txt, points3D.txt; "
            f"{len(parser.images)} views, {len(parser.points3d)} points)")
    return target

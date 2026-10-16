"""Shared-pose complementary subject/background Lichtfeld training views."""

from pathlib import Path
import json
import re
import shutil

import numpy as np


BACKGROUND_SUFFIX = "_background"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def prepare_subject_mask(masks, images):
    """Normalize a white=subject keep mask to [N,H,W] at image resolution."""
    import torch

    if masks is None:
        raise ValueError(
            "Split training requires masks_lichtfeld, RGBA alpha, or a successful RMBG matte."
        )
    if isinstance(masks, torch.Tensor):
        subject = masks.detach().to(device="cpu", dtype=torch.float32)
    else:
        subject = torch.as_tensor(np.asarray(masks), dtype=torch.float32)
    if subject.dim() == 2:
        subject = subject.unsqueeze(0).unsqueeze(0)
    elif subject.dim() == 3:
        subject = subject.unsqueeze(1)
    elif subject.dim() == 4 and subject.shape[-1] == 1:
        subject = subject.permute(0, 3, 1, 2)
    if subject.dim() != 4:
        raise ValueError(f"Unsupported split mask shape {tuple(subject.shape)}")
    if int(subject.shape[0]) != int(images.shape[0]):
        raise ValueError(
            f"Split mask has {int(subject.shape[0])} frames for "
            f"{int(images.shape[0])} selected images."
        )
    if float(subject.max()) > 1.0:
        subject = subject / 255.0
    subject = subject.clamp(0.0, 1.0)
    if int(subject.shape[1]) != 1:
        subject = subject[:, :1]
    size = (int(images.shape[1]), int(images.shape[2]))
    if tuple(subject.shape[-2:]) != size:
        subject = torch.nn.functional.interpolate(
            subject, size=size, mode="bilinear", align_corners=False)
    return subject[:, 0].contiguous()


def export_complementary_training_views(export_dir, parser, subject_masks,
                                         image_format="PNG", jpeg_quality=90,
                                         log=print):
    """Export background copies with exact inverse masks and source camera poses.

    COLMAP/GLOMAP runs only on the original images. For every registered exported
    image, this writes a second training image, the bytewise inverse mask, and a
    pose-identical sparse camera row with an empty POINTS2D line. Thus both views
    share one coordinate frame and there is no second alignment.
    """
    from PIL import Image

    root = Path(export_dir)
    image_dir = root / "images"
    mask_dir = root / "masks"
    depth_dir = root / "depth"
    images_txt = root / "sparse" / "0" / "images.txt"
    if not image_dir.is_dir() or not mask_dir.is_dir() or not images_txt.is_file():
        raise ValueError("Split export needs images/, masks/, and sparse/0/images.txt.")

    files = sorted(
        (path for path in image_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.isdigit()),
        key=lambda path: path.name,
    )
    masks = np.asarray(subject_masks, dtype=np.float32)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    # _export_sparse calls COLMAPParser.write_txt, which sorts registered names,
    # then remaps source trailing indices onto the sorted exported files. Mirror
    # that mapping so partial registration does not shift masks to the wrong image.
    registered = sorted(parser.images.items(), key=lambda pair: pair[1]["name"])
    files_by_name = {path.name: index for index, path in enumerate(files)}
    next_image_id = max((int(image_id) for image_id in parser.images), default=0) + 1
    pairs = []
    for source_id, record in registered:
        source_name = Path(record["name"]).name
        frame_index = files_by_name.get(source_name)
        if frame_index is None:
            match = re.search(r"(\d+)$", Path(source_name).stem)
            if match is None:
                continue
            frame_index = int(match.group(1))
        if frame_index >= len(files):
            continue
        image_path = files[frame_index]
        subject_u8 = np.rint(np.clip(masks[frame_index], 0.0, 1.0) * 255.0).astype(np.uint8)
        background_u8 = 255 - subject_u8

        Image.fromarray(subject_u8).save(mask_dir / f"{image_path.stem}.png")
        Image.fromarray(background_u8).save(
            mask_dir / f"{image_path.stem}{BACKGROUND_SUFFIX}.png")

        background_name = f"{image_path.stem}{BACKGROUND_SUFFIX}{image_path.suffix}"
        with Image.open(image_path) as source_image:
            background_image = source_image.convert("RGB")
        if str(image_format).upper() == "JPEG" or image_path.suffix.lower() in (".jpg", ".jpeg"):
            background_image.save(image_dir / background_name, format="JPEG",
                                  quality=int(jpeg_quality), subsampling=0, optimize=True)
        else:
            background_image.save(image_dir / background_name, format="PNG")

        depth_path = depth_dir / f"{image_path.stem}.depth.png"
        if depth_path.is_file():
            shutil.copy2(depth_path, depth_dir /
                         f"{image_path.stem}{BACKGROUND_SUFFIX}.depth.png")

        qvec = np.asarray(record["qvec"], dtype=np.float64).reshape(4)
        tvec = np.asarray(record["tvec"], dtype=np.float64).reshape(3)
        pairs.append({
            "source_image": image_path.name,
            "background_image": background_name,
            "source_image_id": int(source_id),
            "background_image_id": next_image_id,
            "camera_id": int(record["camera_id"]),
            "qvec": qvec.tolist(),
            "tvec": tvec.tolist(),
        })
        next_image_id += 1

    if not pairs:
        raise ValueError("COLMAP/GLOMAP registered no exported images for the split.")

    # A COLMAP image entry has a pose/name line followed by a POINTS2D line.
    # Copies were not used by SfM, so their feature line is intentionally empty.
    sparse_text = images_txt.read_text(encoding="utf-8")
    sparse_lines = sparse_text.splitlines(keepends=True)
    original_count = 0
    for line_index, line in enumerate(sparse_lines):
        match = re.match(r"#\s*Number of images:\s*(\d+)", line)
        if match:
            original_count = int(match.group(1))
            sparse_lines[line_index] = f"# Number of images: {original_count + len(pairs)}\n"
            break
    images_txt.write_text("".join(sparse_lines), encoding="utf-8", newline="\n")
    with images_txt.open("a", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            qvec, tvec = pair["qvec"], pair["tvec"]
            stream.write(
                f"{pair['background_image_id']} "
                f"{qvec[0]:.6f} {qvec[1]:.6f} {qvec[2]:.6f} {qvec[3]:.6f} "
                f"{tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f} "
                f"{pair['camera_id']} {pair['background_image']}\n\n")

    metadata = [{key: value for key, value in pair.items() if key not in {"qvec", "tvec"}}
                for pair in pairs]
    (root / "subject_background_split.json").write_text(json.dumps({
        "mode": "shared_camera_complementary_masks",
        "camera_solve": "single shared COLMAP/GLOMAP reconstruction",
        "mask_polarity": {"original": "subject keep",
                          "background_copy": "exact byte complement"},
        "pairs": metadata,
    }, indent=2), encoding="utf-8")
    log(f"Complementary subject/background views exported: {len(pairs)} registered frames")
    return metadata


def remove_background_views(export_dir):
    """Remove stale split copies before re-exporting into a reused dataset folder."""
    root = Path(export_dir)
    for folder in (root / "images", root / "masks", root / "depth"):
        if folder.is_dir():
            for path in folder.glob(f"*{BACKGROUND_SUFFIX}*"):
                if path.is_file():
                    path.unlink()
    metadata = root / "subject_background_split.json"
    if metadata.is_file():
        metadata.unlink()


# Descriptive alias retained for call sites that emphasize stale-file cleanup.
remove_stale_background_views = remove_background_views
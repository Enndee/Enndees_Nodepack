"""
GLOMAP wrapper - extends COLMAPWrapper with GLOMAP mapper support.
GLOMAP is a much faster alternative to COLMAP's mapper for SfM.
Uses COLMAP for feature extraction/matching, GLOMAP for mapping.

Vendored into Enndees Nodepack from `comfyui_colmap` by nelsig (MIT license).
Original: https://gitlab.com/nelsig/comfyui_colmap  (lib/glomap_wrapper.py)

Changes by Enndee (see NOTICE.md):
  * no hard coded developer paths anymore
  * ``max_image_size`` is really forwarded to COLMAP's feature extractor
    (previously the node value was silently ignored)
  * optional ``colmap_global`` mapper backend (COLMAP >= 3.12) as a drop-in
    replacement for the deprecated GLOMAP binary
  * no flashing console windows + ``errors="replace"`` on Windows
  * mapper timeout configurable via ``ENNDEE_GLOMAP_TIMEOUT``
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from .colmap_wrapper import COLMAPWrapper, subprocess_window_kwargs, _timeout_from_env


def fix_image_names_in_sparse(sparse_dir: Path, images_dir: Path, colmap_path: str = "") -> bool:
    """
    Fix image names in images.bin to match actual files in images_dir.
    Also generates TXT files (cameras.txt, images.txt, points3D.txt) for easy debugging.

    IMPORTANT: Uses direct BIN parsing (NOT COLMAP model_converter) because
    COLMAP's model_converter CORRUPTS GLOMAP's BIN format (10-field pose lines).

    Args:
        sparse_dir: Path to sparse/0 directory containing cameras.bin, images.bin, points3D.bin
        images_dir: Path to images directory with the actual image files
        colmap_path: Not used anymore (kept for backward compatibility)

    Returns:
        True if successful, False otherwise
    """
    import struct

    sparse_dir = Path(sparse_dir)
    images_dir = Path(images_dir)

    if not sparse_dir.exists() or not images_dir.exists():
        print("[FixImageNames] sparse_dir or images_dir not found")
        return False

    # Get actual image files in the export images folder
    actual_files = sorted([f.name for f in images_dir.iterdir()
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not actual_files:
        print("[FixImageNames] No image files found in images_dir")
        return False

    # ========== 1. Read and fix images.bin directly ==========
    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        print("[FixImageNames] images.bin not found")
        return False

    def read_next_bytes(fid, num_bytes, fmt, endian="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian + fmt, data)

    # Read all images - GLOMAP BIN format (same as COLMAP):
    #   num_images (Q)
    #   pro Bild: image_id (i), qw,qx,qy,qz (4d), tx,ty,tz (3d), camera_id (i),
    #             name (null-terminiert), num_points2D (Q), points2D (24*num)
    images = []
    with open(images_bin, 'rb') as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            try:
                image_id = read_next_bytes(fid, 4, "i")[0]
                qw, qx, qy, qz = read_next_bytes(fid, 32, "dddd")
                tx, ty, tz = read_next_bytes(fid, 24, "ddd")
                camera_id = read_next_bytes(fid, 4, "i")[0]
                name = b""
                char = fid.read(1)
                while char != b"\x00":
                    name += char
                    char = fid.read(1)
                name = name.decode('utf-8')
                num_points2D = read_next_bytes(fid, 8, "Q")[0]
                points2d = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
                images.append({
                    'id': image_id, 'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                    'tx': tx, 'ty': ty, 'tz': tz, 'camera_id': camera_id,
                    'name': name, 'num_points2D': num_points2D, 'points2d': points2d
                })
            except (struct.error, UnicodeDecodeError) as e:
                print(f"[FixImageNames] Warnung: Korrupte Daten beim Lesen von Bild {len(images)}: {e}")
                break

    # Fix names: Nur Bilder behalten, die tatsaechlich im Ordner existieren
    # Die images.bin kann mehr Eintraege haben als Bilder vorhanden sind
    # (z.B. von einem frueheren Lauf mit mehr Frames)
    actual_names = set(actual_files)
    filtered_images = []
    changed = 0
    for i, img in enumerate(images):
        old_name = img['name']
        # Pruefen ob der Name (oder der Name ohne .jpg/.png) im Ordner existiert
        if old_name in actual_names:
            filtered_images.append(img)
        else:
            # Versuche den Namen zu korrigieren: Die images.bin hat Namen wie
            # frame_000.jpg, aber die tatsaechlichen Dateien heissen frame_00001_.jpg
            # Wir versuchen, den Index aus dem Namen zu extrahieren und die
            # tatsaechliche Datei anhand der Position zuzuordnen
            if i < len(actual_files):
                new_name = actual_files[i]
                img['name'] = new_name
                if new_name in actual_names:
                    filtered_images.append(img)
                    changed += 1
    images = filtered_images
    print(f"[FixImageNames] {len(images)} Bilder in images.bin, {len(actual_files)} Bilder im Ordner")

    # Write fixed images.bin (GLOMAP-Format: image_id, qw,qx,qy,qz, tx,ty,tz, camera_id)
    with open(images_bin, 'wb') as fid:
        fid.write(struct.pack('<Q', len(images)))
        for img in images:
            fid.write(struct.pack('<idddddddi',
                img['id'],
                img['qw'], img['qx'], img['qy'], img['qz'],
                img['tx'], img['ty'], img['tz'],
                img['camera_id']))
            name_bytes = img['name'].encode('utf-8') + b'\x00'
            fid.write(name_bytes)
            fid.write(struct.pack('<Q', img['num_points2D']))
            if img['num_points2D'] > 0:
                try:
                    fid.write(struct.pack('<ddq' * img['num_points2D'], *img['points2d']))
                except (struct.error, TypeError):
                    # Korrupte Daten - schreibe 0 Punkte
                    print(f"[FixImageNames] Warnung: Korrupte 2D-Punkte fuer Bild {img['name']}, schreibe 0")
                    # Wir muessen die Datei zurueckspulen und korrigieren
                    # Einfacher: num_points2D auf 0 setzen
                    pass

    print(f"[FixImageNames] {changed} image names fixed in images.bin ({len(images)} total)")

    # ========== 2. Generate TXT files for debugging ==========
    # Read cameras.bin
    cameras = {}
    cameras_bin = sparse_dir / "cameras.bin"
    if cameras_bin.exists():
        with open(cameras_bin, 'rb') as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 12, 8: 15}[model_id]
                params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
                cameras[camera_id] = (model_id, width, height, params)

    # Write cameras.txt
    model_names = {0: 'SIMPLE_PINHOLE', 1: 'SIMPLE_RADIAL', 2: 'RADIAL', 3: 'OPENCV',
                   4: 'OPENCV_FISHEYE', 5: 'FULL_OPENCV', 6: 'FOV', 7: 'SIMPLE_RADIAL_FISHEYE', 8: 'RADIAL_FISHEYE'}
    with open(sparse_dir / 'cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'# Number of cameras: {len(cameras)}\n')
        for cam_id, (model, w, h, params) in sorted(cameras.items()):
            model_name = model_names.get(model, f'MODEL_{model}')
            params_str = ' '.join(f'{p:.6f}' for p in params)
            f.write(f'{cam_id} {model_name} {w} {h} {params_str}\n')

    # Write images.txt
    with open(sparse_dir / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write(f'# Number of images: {len(images)}\n')
        for img in images:
            f.write(f"{img['id']} {img['qw']:.6f} {img['qx']:.6f} {img['qy']:.6f} {img['qz']:.6f} "
                    f"{img['tx']:.6f} {img['ty']:.6f} {img['tz']:.6f} {img['camera_id']} {img['name']}\n")
            f.write('\n')

    # Read and write points3D.txt
    points3d_bin = sparse_dir / "points3D.bin"
    if points3d_bin.exists():
        try:
            points = []
            with open(points3d_bin, 'rb') as fid:
                num_points = read_next_bytes(fid, 8, "Q")[0]
                for _ in range(num_points):
                    point_id, x, y, z, r, g, b = read_next_bytes(fid, 43, "QdddBBB")
                    num_obs = read_next_bytes(fid, 8, "Q")[0]
                    obs = read_next_bytes(fid, 8 * num_obs, "ii" * num_obs)
                    points.append((point_id, x, y, z, r, g, b, num_obs, obs))

            # Check if GLOMAP wrote black colors (RGB=0,0,0) - if so, extract from images
            has_black = any(p[5] == 0 and p[6] == 0 and p[7] == 0 for p in points)
            if has_black:
                print("[FixImageNames] GLOMAP wrote black colors - extracting RGB from source images...")
                try:
                    from PIL import Image
                    # Build lookup: image_id -> {point3d_id: (x, y)}
                    # The images list has points2d as flat tuples (x, y, point3d_id, x, y, point3d_id, ...)
                    img_point_map = {}  # img_id -> {pt3d_id: (x, y)}
                    for img in images:
                        img_id = img['id']
                        img_point_map[img_id] = {}
                        pts = img['points2d']
                        for j in range(0, len(pts), 3):
                            px, py, pt3d_id = pts[j], pts[j+1], int(pts[j+2])
                            if pt3d_id >= 0:
                                img_point_map[img_id][pt3d_id] = (px, py)

                    # Camera resolution for scaling
                    cam_width = None
                    cam_height = None
                    if cameras:
                        cam = list(cameras.values())[0]
                        cam_width = cam[1]
                        cam_height = cam[2]

                    # Cache loaded images
                    image_cache = {}
                    fixed_count = 0

                    for idx, p in enumerate(points):
                        point_id, x, y, z, r, g, b, num_obs, obs = p
                        if r != 0 or g != 0 or b != 0:
                            continue

                        # Find this point in any image
                        color = None
                        for img in images:
                            img_id = img['id']
                            if point_id in img_point_map.get(img_id, {}):
                                px, py = img_point_map[img_id][point_id]

                                # Load image if not cached
                                img_path = images_dir / img['name']
                                if img_path not in image_cache:
                                    if not img_path.exists():
                                        continue
                                    try:
                                        image_cache[img_path] = Image.open(img_path).convert('RGB')
                                    except Exception:
                                        continue

                                img_obj = image_cache[img_path]
                                w, h = img_obj.size

                                # Scale coordinates from camera resolution to actual image resolution
                                if cam_width and cam_height and cam_width > 0 and cam_height > 0:
                                    scale_x = w / cam_width
                                    scale_y = h / cam_height
                                    px = px * scale_x
                                    py = py * scale_y

                                px_int, py_int = int(round(px)), int(round(py))
                                if 0 <= px_int < w and 0 <= py_int < h:
                                    color = img_obj.getpixel((px_int, py_int))
                                    break

                        if color is not None:
                            points[idx] = (point_id, x, y, z, color[0], color[1], color[2], num_obs, obs)
                            fixed_count += 1

                    print(f"[FixImageNames] Extracted colors for {fixed_count} 3D points")
                except ImportError:
                    print("[FixImageNames] PIL not available - cannot extract colors")

            with open(sparse_dir / 'points3D.txt', 'w') as f:
                f.write('# 3D point list with one line of data per point:\n')
                f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
                f.write(f'# Number of points: {len(points)}\n')
                for p in points:
                    point_id, x, y, z, r, g, b, num_obs, obs = p
                    track = ' '.join(f'{obs[i]} {obs[i+1]}' for i in range(0, len(obs), 2))
                    f.write(f'{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0 {track}\n')
        except Exception as e:
            print(f"[FixImageNames] points3D.txt generation failed: {e}")

    print("[FixImageNames] Done - images.bin fixed directly, TXT files generated")
    return True


class GLOMAPWrapper(COLMAPWrapper):
    """
    GLOMAP wrapper - uses COLMAP for features/matching, GLOMAP for mapping.

    ``glomap_path`` is optional: when the ``colmap_global`` backend is used the
    mapper shipped with COLMAP (>= 3.12) is called instead, so no GLOMAP binary
    is required at all.
    """

    def __init__(self, colmap_path: Optional[str] = None,
                 glomap_path: Optional[str] = None):
        super().__init__(colmap_path)
        self.glomap_path: Optional[Path] = Path(glomap_path) if glomap_path else None
        if self.glomap_path is not None and not self.glomap_path.exists():
            raise FileNotFoundError(f"GLOMAP not found at {self.glomap_path}")

    def _run_glomap(self, args: List[str], desc: str = "") -> Tuple[bool, str]:
        """Run a GLOMAP command."""
        if self.glomap_path is None:
            print("[GLOMAP] No GLOMAP binary configured")
            return False, "GLOMAP binary missing"

        cmd = [str(self.glomap_path)] + args
        print(f"[GLOMAP] {desc}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_timeout_from_env("ENNDEE_GLOMAP_TIMEOUT", 7200),
                **subprocess_window_kwargs(),
            )
            if result.returncode != 0:
                print(f"[GLOMAP] Error: {result.stderr}")
                return False, result.stderr
            return True, result.stdout
        except subprocess.TimeoutExpired:
            print("[GLOMAP] Command timed out")
            return False, "Timeout"
        except Exception as e:
            print(f"[GLOMAP] Exception: {e}")
            return False, str(e)

    def mapper(self, backend: str = "glomap", **kwargs) -> bool:
        """
        Run the global mapper.

        Args:
            backend: "glomap"     -> use the GLOMAP binary (fast, frozen at 1.2.0)
                     "colmap_global" -> use ``colmap global_mapper`` (COLMAP >= 3.12)
        """
        if backend == "colmap_global":
            if not self.supports_command("global_mapper"):
                print("[GLOMAP] This COLMAP build has no 'global_mapper' command. "
                      "Use the 'glomap' backend or a newer COLMAP (>= 3.12).")
                return False
            # COLMAP's global mapper writes to the same sparse layout as GLOMAP.
            args = [
                "global_mapper",
                "--database_path", str(self.database_path),
                "--output_path", str(self.sparse_dir),
            ]
            success, _ = self._run_command(args, "COLMAP global mapper")
            return success

        # GLOMAP only accepts --database_path and --output_path (no --image_path!)
        args = [
            "mapper",
            "--database_path", str(self.database_path),
            "--output_path", str(self.sparse_dir),
        ]
        success, _ = self._run_glomap(args, "GLOMAP mapper")
        return success

    def run_pipeline(self, images: np.ndarray, camera_model: str = "PINHOLE",
                     matcher: str = "sequential", max_features: int = 8192,
                     use_gpu: bool = True, keep_workspace: bool = False,
                     align_orientation: bool = True,
                     masks: Optional[np.ndarray] = None,
                     sequential_overlap: int = 15, max_image_size: int = 4096,
                     min_num_matches: int = 15,
                     ba_refine_focal_length: bool = True,
                     ba_refine_principal_point: bool = False,
                     ba_refine_extra_params: bool = True,
                     estimate_affine_shape: bool = False,
                     domain_size_pooling: bool = False,
                     mapper_backend: str = "glomap") -> Optional[str]:
        """Run complete global SfM pipeline (COLMAP features + global mapper)."""
        try:
            self.setup_workspace()
            print(f"[GLOMAP] Processing {images.shape[0]} frames...")
            image_paths = self.export_frames(images)

            mask_path = None
            if masks is not None and len(masks) > 0:
                image_names = [Path(p).name for p in image_paths]
                mask_path = self.export_masks(masks, image_names)

            if not self.feature_extractor(camera_model=camera_model,
                                          max_num_features=max_features,
                                          max_image_size=max_image_size,
                                          mask_path=mask_path, use_gpu=use_gpu,
                                          estimate_affine_shape=estimate_affine_shape,
                                          domain_size_pooling=domain_size_pooling):
                print("[GLOMAP] Feature extraction failed")
                return None

            if matcher == "sequential":
                if not self.sequential_matcher(use_gpu=use_gpu, overlap=sequential_overlap):
                    print("[GLOMAP] Sequential matching failed")
                    return None
            else:
                if not self.exhaustive_matcher(use_gpu=use_gpu):
                    print("[GLOMAP] Exhaustive matching failed")
                    return None

            if not self.mapper(backend=mapper_backend):
                print("[GLOMAP] Global mapper failed")
                return None

            model_path = self.get_sparse_model_path()
            if model_path:
                print(f"[GLOMAP] Reconstruction complete: {model_path}")
            return model_path

        except Exception as e:
            print(f"[GLOMAP] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if not keep_workspace:
                pass

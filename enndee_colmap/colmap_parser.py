"""
Parser for COLMAP binary output files.
Reads cameras.bin, images.bin, and points3D.bin files.

Vendored into Enndees Nodepack from `comfyui_colmap` by nelsig (MIT license).
Original: https://gitlab.com/nelsig/comfyui_colmap  (lib/colmap_parser.py)
Unmodified except for this notice -- see enndee_colmap/NOTICE.md.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


class COLMAPParser:
    """Parser for COLMAP binary reconstruction files."""

    # Camera model IDs from COLMAP
    CAMERA_MODELS = {
        0: ("SIMPLE_PINHOLE", 3),    # f, cx, cy
        1: ("PINHOLE", 4),            # fx, fy, cx, cy
        2: ("SIMPLE_RADIAL", 4),      # f, cx, cy, k
        3: ("RADIAL", 5),             # f, cx, cy, k1, k2
        4: ("OPENCV", 8),             # fx, fy, cx, cy, k1, k2, p1, p2
        5: ("OPENCV_FISHEYE", 8),     # fx, fy, cx, cy, k1, k2, k3, k4
        6: ("FULL_OPENCV", 12),       # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        7: ("FOV", 5),                # fx, fy, cx, cy, omega
        8: ("SIMPLE_RADIAL_FISHEYE", 4),  # f, cx, cy, k
        9: ("RADIAL_FISHEYE", 5),     # f, cx, cy, k1, k2
        10: ("THIN_PRISM_FISHEYE", 12),  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
    }

    def __init__(self, sparse_dir: str):
        """
        Initialize parser with path to COLMAP sparse reconstruction directory.

        Args:
            sparse_dir: Path to directory containing cameras.bin, images.bin, points3D.bin
        """
        self.sparse_dir = Path(sparse_dir)
        self.cameras = {}
        self.images = {}
        self.points3d = {}

    def parse_all(self) -> Tuple[Dict, Dict, Dict]:
        """
        Parse all COLMAP binary files.

        Returns:
            Tuple of (cameras, images, points3d) dictionaries
        """
        self.cameras = self.read_cameras_binary()
        self.images = self.read_images_binary()
        self.points3d = self.read_points3D_binary()
        return self.cameras, self.images, self.points3d

    def read_cameras_binary(self) -> Dict[int, Dict]:
        """
        Read cameras.bin file.

        Returns:
            Dictionary mapping camera_id to camera parameters
        """
        cameras = {}
        cameras_file = self.sparse_dir / "cameras.bin"

        if not cameras_file.exists():
            print(f"[COLMAP Parser] cameras.bin not found at {cameras_file}")
            return cameras

        with open(cameras_file, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<I", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]

                model_name, num_params = self.CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
                params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))

                cameras[camera_id] = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "width": width,
                    "height": height,
                    "params": np.array(params),
                }

        print(f"[COLMAP Parser] Loaded {len(cameras)} camera(s)")
        return cameras

    def read_images_binary(self) -> Dict[int, Dict]:
        """
        Read images.bin file.

        Returns:
            Dictionary mapping image_id to image data (pose, camera_id, name, etc.)
        """
        images = {}
        images_file = self.sparse_dir / "images.bin"

        if not images_file.exists():
            print(f"[COLMAP Parser] images.bin not found at {images_file}")
            return images

        with open(images_file, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]

                # Quaternion (qw, qx, qy, qz)
                qvec = struct.unpack("<4d", f.read(32))

                # Translation (tx, ty, tz)
                tvec = struct.unpack("<3d", f.read(24))

                camera_id = struct.unpack("<I", f.read(4))[0]

                # Image name (null-terminated string)
                name = ""
                while True:
                    char = f.read(1)
                    if char == b"\x00":
                        break
                    name += char.decode("utf-8")

                # 2D points in image
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                points2d = []
                point3d_ids = []

                for _ in range(num_points2d):
                    x, y = struct.unpack("<2d", f.read(16))
                    point3d_id = struct.unpack("<q", f.read(8))[0]  # -1 if no 3D point
                    points2d.append((x, y))
                    point3d_ids.append(point3d_id)

                images[image_id] = {
                    "qvec": np.array(qvec),
                    "tvec": np.array(tvec),
                    "camera_id": camera_id,
                    "name": name,
                    "points2d": np.array(points2d),
                    "point3d_ids": np.array(point3d_ids),
                }

        print(f"[COLMAP Parser] Loaded {len(images)} image(s)")
        return images

    def read_points3D_binary(self) -> Dict[int, Dict]:
        """
        Read points3D.bin file.

        Returns:
            Dictionary mapping point3d_id to 3D point data
        """
        points3d = {}
        points_file = self.sparse_dir / "points3D.bin"

        if not points_file.exists():
            print(f"[COLMAP Parser] points3D.bin not found at {points_file}")
            return points3d

        with open(points_file, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_points):
                point3d_id = struct.unpack("<Q", f.read(8))[0]
                xyz = struct.unpack("<3d", f.read(24))
                rgb = struct.unpack("<3B", f.read(3))
                error = struct.unpack("<d", f.read(8))[0]

                # Track: list of (image_id, point2d_idx) pairs
                track_length = struct.unpack("<Q", f.read(8))[0]
                track = []
                for _ in range(track_length):
                    image_id = struct.unpack("<I", f.read(4))[0]
                    point2d_idx = struct.unpack("<I", f.read(4))[0]
                    track.append((image_id, point2d_idx))

                points3d[point3d_id] = {
                    "xyz": np.array(xyz),
                    "rgb": np.array(rgb),
                    "error": error,
                    "track": track,
                }

        print(f"[COLMAP Parser] Loaded {len(points3d)} 3D point(s)")
        return points3d

    def _extract_colors_from_images(self, images_dir: Optional[Path]) -> Dict[int, np.ndarray]:
        """
        Extract RGB colors for 3D points from source images.
        GLOMAP writes RGB=0,0,0 for all points, so we need to recover colors
        by sampling the pixel colors at the 2D projections in the images.

        The 2D point coordinates are in the resolution GLOMAP processed (downscaled),
        while the images in images_dir may be in original resolution.
        We scale coordinates using the camera width/height from the reconstruction.

        Args:
            images_dir: Directory containing the source images

        Returns:
            Dictionary mapping point3d_id to RGB color array [3]
        """
        try:
            from PIL import Image
        except ImportError:
            print("[COLMAP Parser] PIL/Pillow not available - cannot extract colors")
            return {}

        if images_dir is None:
            return {}
        images_dir = Path(images_dir)
        if not images_dir.exists():
            print(f"[COLMAP Parser] Images directory not found: {images_dir}")
            return {}

        # Determine scale factor from camera resolution vs actual image resolution
        # GLOMAP processes images at the resolution stored in cameras.bin
        # The images in images_dir may be at original (higher) resolution
        cam_width = None
        cam_height = None
        if self.cameras:
            cam = list(self.cameras.values())[0]
            cam_width = cam['width']
            cam_height = cam['height']

        # Cache loaded images
        image_cache = {}
        colors = {}

        # For each 3D point, sample color from first valid track observation
        for pt_id, pt in self.points3d.items():
            # Skip if point already has valid (non-zero) color
            if np.any(pt['rgb'] > 0):
                colors[pt_id] = pt['rgb']
                continue

            # Try each track observation until we find a valid color
            for img_id, point2d_idx in pt['track']:
                if img_id not in self.images:
                    continue

                img_data = self.images[img_id]
                if point2d_idx >= len(img_data['points2d']):
                    continue

                # Load image if not cached
                img_path = images_dir / img_data['name']
                if img_path not in image_cache:
                    if not img_path.exists():
                        continue
                    try:
                        image_cache[img_path] = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"[COLMAP Parser] Failed to load image {img_path}: {e}")
                        continue

                # Get 2D point coordinates (in GLOMAP's processed resolution)
                x, y = img_data['points2d'][point2d_idx]

                img = image_cache[img_path]
                w, h = img.size

                # Scale coordinates from camera resolution to actual image resolution
                if cam_width and cam_height and cam_width > 0 and cam_height > 0:
                    scale_x = w / cam_width
                    scale_y = h / cam_height
                    x = x * scale_x
                    y = y * scale_y

                x_int, y_int = int(round(x)), int(round(y))

                # Clamp to image bounds
                if 0 <= x_int < w and 0 <= y_int < h:
                    rgb = np.array(img.getpixel((x_int, y_int)))
                    colors[pt_id] = rgb
                    break

        print(f"[COLMAP Parser] Extracted colors for {len(colors)} 3D points")
        return colors

    def write_txt(self, output_dir: Path, images_dir: Optional[Path] = None) -> None:
        """
        Write COLMAP TXT files (cameras.txt, images.txt, points3D.txt) from parsed data.
        This avoids the need to parse BIN files again - uses already-parsed data.
        Also extracts RGB colors from source images if GLOMAP wrote black colors.

        Args:
            output_dir: Directory to write TXT files to
            images_dir: Optional images directory to filter image names
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter images to only those that exist in images_dir.
        # If GLOMAP's image names (frame_00.jpg) don't match the actual
        # files (0001.png), remap them by index rather than discarding.
        filtered_images = {}
        if images_dir is not None:
            images_dir = Path(images_dir)
            actual_files = sorted([f for f in images_dir.iterdir()
                                   if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            actual_names = set(f.name for f in actual_files)
            # Sort self.images by name to get deterministic order
            sorted_images = sorted(self.images.items(), key=lambda kv: kv[1]['name'])
            for img_id, img in sorted_images:
                if img['name'] in actual_names:
                    filtered_images[img_id] = img
                else:
                    # Extract index from GLOMAP name like "frame_12.jpg" -> 12
                    # If not parseable, fall back to position-based mapping
                    idx = None
                    base = Path(img['name']).stem
                    # Try to extract trailing digits from frame_12 or 0001
                    import re
                    m = re.search(r'(\d+)$', base)
                    if m:
                        idx = int(m.group(1))
                    if idx is not None and idx < len(actual_files):
                        new_name = actual_files[idx].name
                        img['name'] = new_name
                        filtered_images[img_id] = img
        else:
            filtered_images = self.images

        # Check if GLOMAP wrote black colors - if so, extract from images
        extracted_colors = {}
        has_black_colors = any(
            np.all(pt['rgb'] == 0) for pt in self.points3d.values()
        ) if self.points3d else False

        if has_black_colors and images_dir is not None:
            print("[COLMAP Parser] GLOMAP wrote black colors - extracting RGB from source images...")
            extracted_colors = self._extract_colors_from_images(images_dir)

        # Write cameras.txt
        with open(output_dir / 'cameras.txt', 'w') as f:
            f.write('# Camera list with one line of data per camera:\n')
            f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
            f.write(f'# Number of cameras: {len(self.cameras)}\n')
            for cam_id, cam in sorted(self.cameras.items()):
                params_str = ' '.join(f'{p:.6f}' for p in cam['params'])
                f.write(f'{cam_id} {cam["model_name"]} {cam["width"]} {cam["height"]} {params_str}\n')

        # Write images.txt
        with open(output_dir / 'images.txt', 'w') as f:
            f.write('# Image list with two lines of data per image:\n')
            f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
            f.write(f'# Number of images: {len(filtered_images)}\n')
            for img_id, img in sorted(filtered_images.items()):
                q = img['qvec']
                t = img['tvec']
                f.write(f"{img_id} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                        f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {img['camera_id']} {img['name']}\n")
                f.write('\n')

        # Write points3D.txt
        with open(output_dir / 'points3D.txt', 'w') as f:
            f.write('# 3D point list with one line of data per point:\n')
            f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
            f.write(f'# Number of points: {len(self.points3d)}\n')
            for pt_id, pt in sorted(self.points3d.items()):
                xyz = pt['xyz']
                rgb = pt['rgb']

                # Use extracted color if available
                if pt_id in extracted_colors:
                    rgb = extracted_colors[pt_id]

                track = ' '.join(f'{img_id} {idx}' for img_id, idx in pt['track'])
                f.write(f"{pt_id} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                        f"{rgb[0]} {rgb[1]} {rgb[2]} {pt['error']:.6f} {track}\n")

        print(f"[COLMAP Parser] TXT files written to {output_dir}")

    def qvec_to_rotmat(self, qvec: np.ndarray) -> np.ndarray:
        """
        Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.

        Args:
            qvec: Quaternion as [qw, qx, qy, qz]

        Returns:
            3x3 rotation matrix
        """
        qw, qx, qy, qz = qvec

        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
        ])

        return R

    def get_camera_poses(self, convention: str = "opengl") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera poses as 4x4 matrices (camera-to-world).

        COLMAP coordinate system: X-right, Y-down, Z-forward (OpenCV convention)
        OpenGL coordinate system: X-right, Y-up, Z-backward

        Args:
            convention: "opencv" or "opengl" coordinate system

        Returns:
            Tuple of (poses, image_names) where poses is [N, 4, 4] array
        """
        if not self.images:
            self.parse_all()

        # Sort images by name to maintain frame order
        sorted_images = sorted(self.images.values(), key=lambda x: x["name"])

        print(f"[COLMAP Parser] get_camera_poses: {len(sorted_images)} images in reconstruction")
        if len(sorted_images) > 0:
            print(f"[COLMAP Parser] First image: {sorted_images[0]['name']}, Last: {sorted_images[-1]['name']}")

        # Coordinate conversion matrix from OpenCV to OpenGL
        # OpenCV: X-right, Y-down, Z-forward
        # OpenGL: X-right, Y-up, Z-backward
        # So we flip Y and Z: new_y = -old_y, new_z = -old_z
        cv2gl = np.diag([1.0, -1.0, -1.0, 1.0])

        poses = []
        names = []

        for img_data in sorted_images:
            qvec = img_data["qvec"]
            tvec = img_data["tvec"]

            # COLMAP stores world-to-camera transform (w2c)
            R_w2c = self.qvec_to_rotmat(qvec)
            t_w2c = tvec

            # Convert to camera-to-world (c2w)
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c

            # Build 4x4 matrix in OpenCV convention
            pose_cv = np.eye(4)
            pose_cv[:3, :3] = R_c2w
            pose_cv[:3, 3] = t_c2w

            if convention == "opengl":
                # Convert entire pose from OpenCV world to OpenGL world
                # pose_gl = cv2gl @ pose_cv @ cv2gl.T
                # This transforms: world coords (left multiply) and camera local coords (right multiply)
                pose = cv2gl @ pose_cv @ cv2gl
            else:
                pose = pose_cv

            poses.append(pose)
            names.append(img_data["name"])

        return np.array(poses), np.array(names)

    def get_intrinsics(self) -> np.ndarray:
        """
        Get camera intrinsics as [fx, fy, cx, cy].

        Returns:
            Intrinsics array [4]
        """
        if not self.cameras:
            self.parse_all()

        if not self.cameras:
            return np.array([1.0, 1.0, 0.5, 0.5])

        # Get first camera (usually there's only one for video)
        camera = list(self.cameras.values())[0]
        params = camera["params"]
        model_name = camera["model_name"]

        # Extract fx, fy, cx, cy based on model
        if model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"]:
            # f, cx, cy, ...
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model_name in ["PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                           "FULL_OPENCV", "FOV", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"]:
            # fx, fy, cx, cy, ...
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            # Default fallback
            fx = fy = params[0] if len(params) > 0 else 1.0
            cx = params[1] if len(params) > 1 else 0.5
            cy = params[2] if len(params) > 2 else 0.5

        return np.array([fx, fy, cx, cy])

    def get_point_cloud(self, convention: str = "opengl") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 3D point cloud with colors.

        Args:
            convention: "opencv" or "opengl" coordinate system

        Returns:
            Tuple of (points, colors) where points is [N, 3] and colors is [N, 3]
        """
        if not self.points3d:
            self.parse_all()

        if not self.points3d:
            return np.zeros((0, 3)), np.zeros((0, 3))

        points = []
        colors = []

        for point_data in self.points3d.values():
            points.append(point_data["xyz"])
            colors.append(point_data["rgb"] / 255.0)  # Normalize to [0, 1]

        points = np.array(points)
        colors = np.array(colors)

        # Convert from COLMAP/OpenCV to OpenGL convention if needed
        # COLMAP: X-right, Y-down, Z-forward
        # OpenGL: X-right, Y-up, Z-backward
        if convention == "opengl" and len(points) > 0:
            points[:, 1] *= -1  # Flip Y
            points[:, 2] *= -1  # Flip Z

        return points, colors

    def get_reconstruction_quality(self) -> float:
        """
        Estimate reconstruction quality based on various metrics.

        Returns:
            Quality score between 0 and 1
        """
        if not self.images or not self.points3d:
            self.parse_all()

        if not self.images:
            return 0.0

        # Metrics for quality estimation
        num_images = len(self.images)
        num_points = len(self.points3d)

        # Average reprojection error
        if self.points3d:
            errors = [p["error"] for p in self.points3d.values()]
            avg_error = np.mean(errors)
        else:
            avg_error = float('inf')

        # Average track length (how many images see each point)
        if self.points3d:
            track_lengths = [len(p["track"]) for p in self.points3d.values()]
            avg_track_length = np.mean(track_lengths)
        else:
            avg_track_length = 0

        # Compute confidence score
        # Low error is good (< 1 pixel is excellent)
        error_score = max(0, 1 - avg_error / 2.0)

        # More points per image is good
        points_per_image = num_points / max(num_images, 1)
        points_score = min(1.0, points_per_image / 1000.0)

        # Higher track length is good (means consistent matching)
        track_score = min(1.0, avg_track_length / 10.0)

        # Combine scores
        confidence = (error_score * 0.4 + points_score * 0.3 + track_score * 0.3)

        return float(np.clip(confidence, 0, 1))

    def detect_ground_plane(self, points: np.ndarray, threshold: float = 0.02,
                            min_inliers_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect dominant ground plane using RANSAC.

        Args:
            points: Point cloud [N, 3]
            threshold: Distance threshold for inliers
            min_inliers_ratio: Minimum ratio of points that should be inliers

        Returns:
            Tuple of (plane_normal, plane_point, inlier_ratio)
        """
        if len(points) < 10:
            return np.array([0, 1, 0]), np.array([0, 0, 0]), 0.0

        n_points = len(points)
        best_normal = np.array([0, 1, 0])
        best_point = np.mean(points, axis=0)
        best_inliers = 0

        # RANSAC iterations
        n_iterations = 500

        for _ in range(n_iterations):
            # Sample 3 random points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm

            # Count inliers
            distances = np.abs(np.dot(points - p1, normal))
            inliers = np.sum(distances < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_point = p1

                # Early exit if we found a good plane
                if inliers > n_points * 0.5:
                    break

        inlier_ratio = best_inliers / n_points

        # Ensure normal points upward (positive Y in final coord system)
        # We'll assume the camera is generally above the ground plane
        # Check the average camera height relative to the plane

        print(f"[COLMAP Parser] Ground plane detected: {best_inliers}/{n_points} inliers ({inlier_ratio:.1%})")

        return best_normal, best_point, inlier_ratio

    def compute_alignment_transform(self, points: np.ndarray, poses: np.ndarray,
                                    align_to_ground: bool = True,
                                    recenter: bool = False) -> np.ndarray:
        """
        Compute transformation matrix to align scene to world axes.

        - Y-up (ground plane becomes XZ plane at Y=0)
        - Camera trajectory roughly along Z or X axis
        - Origin at centroid of the scene (if recenter=True)

        Args:
            points: Point cloud [N, 3] in OpenGL convention (Y-up)
            poses: Camera poses [T, 4, 4] in OpenGL convention
            align_to_ground: Whether to detect and align to ground plane
            recenter: Whether to recenter the scene (False keeps COLMAP's centering)

        Returns:
            4x4 transformation matrix
        """
        transform = np.eye(4)

        if len(points) == 0:
            return transform

        # Get centroid for potential recentering
        centroid = np.mean(points, axis=0)

        # Detect ground plane if requested
        if align_to_ground and len(points) > 10:
            plane_normal, plane_point, inlier_ratio = self.detect_ground_plane(points)

            if inlier_ratio > 0.05:  # Valid plane detected (lowered threshold)
                # We want Y-up, so we need to rotate plane_normal to [0, 1, 0]
                target_up = np.array([0, 1, 0])

                # Check if cameras are above or below the plane
                # In OpenGL convention, Y is up, so cameras should have higher Y than ground
                if len(poses) > 0:
                    camera_positions = poses[:, :3, 3]
                    avg_camera_pos = np.mean(camera_positions, axis=0)
                    # Vector from plane to camera
                    to_camera = avg_camera_pos - plane_point
                    # If dot product is negative, cameras are "below" the plane normal
                    # We want the normal to point toward the cameras (up)
                    if np.dot(to_camera, plane_normal) < 0:
                        plane_normal = -plane_normal
                        print(f"[COLMAP Parser] Flipped plane normal to point toward cameras")

                print(f"[COLMAP Parser] Plane normal: {plane_normal}")
                print(f"[COLMAP Parser] Target up: {target_up}")

                # Compute rotation from plane_normal to target_up
                rotation = self._rotation_between_vectors(plane_normal, target_up)

                # Apply rotation
                transform[:3, :3] = rotation

                if recenter:
                    # Build transform: rotate first, then translate
                    # T = R * (p - centroid) = R*p - R*centroid
                    rotated_centroid = rotation @ centroid
                    transform[:3, 3] = -rotated_centroid

                    # Also compute height offset to put ground at Y=0
                    # After rotation, find the lowest Y coordinate of points on the plane
                    rotated_plane_point = rotation @ plane_point - rotated_centroid
                    # Shift so ground is at Y=0
                    transform[1, 3] -= rotated_plane_point[1]
                    print(f"[COLMAP Parser] Scene rotated to ground plane and recentered (Y-up, ground at Y=0)")
                else:
                    # Just rotate, keep COLMAP's centering
                    # But still shift Y so ground is at Y=0
                    rotated_plane_point = rotation @ plane_point
                    transform[1, 3] = -rotated_plane_point[1]
                    print(f"[COLMAP Parser] Scene rotated to ground plane (Y-up, ground at Y=0), keeping COLMAP centering")
            else:
                if recenter:
                    transform[:3, 3] = -centroid
                    print(f"[COLMAP Parser] No clear ground plane ({inlier_ratio:.1%} inliers), scene centered only")
                else:
                    print(f"[COLMAP Parser] No clear ground plane ({inlier_ratio:.1%} inliers), no changes applied")
        elif recenter:
            # Just center the scene
            transform[:3, 3] = -centroid

        return transform

    def _rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix that rotates v1 to v2.

        Args:
            v1: Source vector (normalized)
            v2: Target vector (normalized)

        Returns:
            3x3 rotation matrix
        """
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Handle parallel vectors
        dot = np.dot(v1, v2)
        if dot > 0.9999:
            return np.eye(3)
        if dot < -0.9999:
            # 180 degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v1, perp)
            axis = axis / np.linalg.norm(axis)
            # Rodrigues formula for 180 degree rotation
            return 2 * np.outer(axis, axis) - np.eye(3)

        # Rodrigues rotation formula
        cross = np.cross(v1, v2)
        skew = np.array([
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0]
        ])

        R = np.eye(3) + skew + skew @ skew * (1 / (1 + dot))
        return R

    def apply_transform(self, points: np.ndarray, poses: np.ndarray,
                       transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transformation to points and poses.

        Args:
            points: Point cloud [N, 3]
            poses: Camera poses [T, 4, 4]
            transform: 4x4 transformation matrix

        Returns:
            Tuple of (transformed_points, transformed_poses)
        """
        # Transform points
        if len(points) > 0:
            points_h = np.hstack([points, np.ones((len(points), 1))])
            transformed_points = (transform @ points_h.T).T[:, :3]
        else:
            transformed_points = points

        # Transform poses
        transformed_poses = []
        for pose in poses:
            # pose is camera-to-world, we need to apply world transform
            # new_pose = transform @ pose
            new_pose = transform @ pose
            transformed_poses.append(new_pose)

        return transformed_points, np.array(transformed_poses)

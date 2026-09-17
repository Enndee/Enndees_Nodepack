"""
Wrapper for COLMAP CLI commands.
Handles running COLMAP binaries and managing temporary files.

Vendored into Enndees Nodepack from `comfyui_colmap` by nelsig (MIT license).
Original: https://gitlab.com/nelsig/comfyui_colmap  (lib/colmap_wrapper.py)

Changes by Enndee (see NOTICE.md):
  * no hard coded developer paths anymore - ``colmap_path`` is mandatory
  * child processes run without a flashing console window on Windows
  * stdout/stderr are decoded with ``errors="replace"``
  * ``SiftMatching.use_gpu`` is also passed to the sequential matcher
  * command timeouts can be overridden via ``ENNDEE_COLMAP_TIMEOUT``
  * helper to detect the COLMAP >= 3.12 ``global_mapper`` command
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np


def subprocess_window_kwargs() -> dict:
    """Return subprocess kwargs that suppress console windows on Windows."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _timeout_from_env(name: str, default: int) -> int:
    """Read an integer timeout override (seconds) from the environment."""
    try:
        value = int(os.environ.get(name, "") or default)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


class COLMAPWrapper:
    """Wrapper for calling COLMAP CLI commands."""

    def __init__(self, colmap_path: Optional[str] = None):
        """
        Initialize COLMAP wrapper.

        Args:
            colmap_path: Path to COLMAP.bat or the colmap executable.
                Mandatory - resolving a suitable binary is the job of
                ``enndee_bin.resolve_binary("colmap")``.
        """
        if not colmap_path:
            raise ValueError(
                "No COLMAP path given. Run 'python install.py' inside the "
                "Enndees-Nodepack folder or set ENNDEE_COLMAP_PATH."
            )

        self.colmap_path = Path(colmap_path)
        self.workspace = None
        self.image_dir = None
        self.mask_dir = None
        self.database_path = None
        self.sparse_dir = None
        self._commands_cache: Optional[set] = None

        if not self.colmap_path.exists():
            raise FileNotFoundError(f"COLMAP not found at {self.colmap_path}")

    def setup_workspace(self, base_dir: Optional[str] = None) -> str:
        """
        Create temporary workspace for COLMAP processing.

        Args:
            base_dir: Optional base directory. If None, uses system temp.

        Returns:
            Path to workspace directory
        """
        if base_dir:
            self.workspace = Path(base_dir)
            self.workspace.mkdir(parents=True, exist_ok=True)
        else:
            self.workspace = Path(tempfile.mkdtemp(prefix="colmap_"))

        self.image_dir = self.workspace / "images"
        self.image_dir.mkdir(exist_ok=True)

        self.mask_dir = self.workspace / "masks"
        # Don't create mask_dir yet - only if masks are provided

        self.database_path = self.workspace / "database.db"
        self.sparse_dir = self.workspace / "sparse"
        self.sparse_dir.mkdir(exist_ok=True)

        print(f"[COLMAP] Workspace: {self.workspace}")
        return str(self.workspace)

    def cleanup_workspace(self):
        """Remove temporary workspace."""
        if self.workspace and self.workspace.exists():
            try:
                shutil.rmtree(self.workspace)
                print(f"[COLMAP] Cleaned up workspace: {self.workspace}")
            except Exception as e:
                print(f"[COLMAP] Warning: Could not cleanup workspace: {e}")

    def export_frames(self, images: np.ndarray, prefix: str = "frame") -> List[str]:
        """
        Export image tensor to JPG files.

        Args:
            images: Image tensor [T, H, W, C] in float [0, 1] or uint8 [0, 255]
            prefix: Filename prefix

        Returns:
            List of saved image paths
        """
        from PIL import Image

        if self.image_dir is None:
            self.setup_workspace()

        saved_paths = []
        num_frames = images.shape[0]

        # Determine number of digits needed for naming
        num_digits = len(str(num_frames))

        for i in range(num_frames):
            frame = images[i]

            # Convert to uint8 if needed
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            # Handle different channel formats
            if len(frame.shape) == 2:
                # Grayscale
                img = Image.fromarray(frame, mode='L')
            elif frame.shape[2] == 4:
                # RGBA -> RGB
                img = Image.fromarray(frame[:, :, :3], mode='RGB')
            else:
                # RGB
                img = Image.fromarray(frame, mode='RGB')

            filename = f"{prefix}_{str(i).zfill(num_digits)}.jpg"
            filepath = self.image_dir / filename
            img.save(filepath, quality=95)
            saved_paths.append(str(filepath))

        print(f"[COLMAP] Exported {len(saved_paths)} frames to {self.image_dir}")
        return saved_paths

    def export_masks(self, masks: np.ndarray, image_names: List[str]) -> Optional[str]:
        """
        Export mask tensor to PNG files for COLMAP.

        COLMAP mask convention:
        - White (255) = valid regions (features will be extracted)
        - Black (0) = masked regions (features will be ignored)

        Args:
            masks: Mask tensor [T, H, W] with values 0-1 or 0-255
                   Where 1/255 = regions to EXCLUDE (e.g., moving objects)
            image_names: List of corresponding image filenames

        Returns:
            Path to mask directory or None if no masks provided
        """
        from PIL import Image

        if masks is None or len(masks) == 0:
            return None

        if self.mask_dir is None:
            if self.workspace is None:
                self.setup_workspace()
            self.mask_dir = self.workspace / "masks"

        self.mask_dir.mkdir(exist_ok=True)

        num_masks = masks.shape[0]

        for i in range(min(num_masks, len(image_names))):
            mask = masks[i]

            # Convert to numpy if needed
            if hasattr(mask, 'numpy'):
                mask = mask.numpy()

            # Normalize to 0-255
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

            # COLMAP expects: white = valid, black = masked
            # Input convention: white = regions to exclude (people, etc.)
            # So we need to INVERT the mask
            mask_inverted = 255 - mask

            # COLMAP mask filename: image_name.png.png or image_name.geometric.png
            # We use the simpler format: same name as image but in masks folder
            image_name = Path(image_names[i]).stem
            mask_filename = f"{image_name}.png"
            mask_path = self.mask_dir / mask_filename

            img = Image.fromarray(mask_inverted, mode='L')
            img.save(mask_path)

        print(f"[COLMAP] Exported {num_masks} masks to {self.mask_dir}")
        return str(self.mask_dir)

    def _run_command(self, args: List[str], desc: str = "") -> Tuple[bool, str]:
        """
        Run a COLMAP command.

        Args:
            args: Command arguments
            desc: Description for logging

        Returns:
            Tuple of (success, output)
        """
        cmd = [str(self.colmap_path)] + args
        print(f"[COLMAP] {desc}: {' '.join(args[:2])}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_timeout_from_env("ENNDEE_COLMAP_TIMEOUT", 3600),
                **subprocess_window_kwargs(),
            )

            if result.returncode != 0:
                print(f"[COLMAP] Error: {result.stderr}")
                return False, result.stderr

            return True, result.stdout

        except subprocess.TimeoutExpired:
            print("[COLMAP] Command timed out")
            return False, "Timeout"
        except Exception as e:
            print(f"[COLMAP] Exception: {e}")
            return False, str(e)

    def available_commands(self) -> set:
        """
        Return the set of COLMAP sub commands supported by this binary.

        Used to detect newer features (e.g. ``global_mapper`` on COLMAP >= 3.12)
        without crashing on older releases.  The result is cached.
        """
        if self._commands_cache is not None:
            return self._commands_cache

        commands = set()
        for arg in ("help", "--help", "-h"):
            try:
                result = subprocess.run(
                    [str(self.colmap_path), arg],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=120,
                    **subprocess_window_kwargs(),
                )
            except Exception:
                continue

            text = f"{result.stdout or ''}\n{result.stderr or ''}"
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("-") or " " in line:
                    continue
                commands.add(line)
            if commands:
                break

        self._commands_cache = commands
        return commands

    def supports_command(self, name: str) -> bool:
        """Return True if this COLMAP build knows the given sub command."""
        commands = self.available_commands()
        if not commands:
            return False
        return name in commands or f"{name}," in commands

    def feature_extractor(
        self,
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
        max_image_size: int = 3200,
        max_num_features: int = 8192,
        use_gpu: bool = True,
        mask_path: Optional[str] = None,
        estimate_affine_shape: bool = False,
        domain_size_pooling: bool = False,
    ) -> bool:
        """
        Extract features from images.

        Args:
            camera_model: Camera model (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV)
            single_camera: Assume all images from same camera
            max_image_size: Maximum image dimension
            max_num_features: Maximum number of features per image
            use_gpu: Use GPU for feature extraction
            mask_path: Optional path to mask directory. White=valid, Black=masked.
            estimate_affine_shape: Estimate affine shape of SIFT features (more robust)
            domain_size_pooling: Use Domain Size Pooling for more features

        Returns:
            Success status
        """
        args = [
            "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1" if single_camera else "0",
            "--SiftExtraction.max_image_size", str(max_image_size),
            "--SiftExtraction.max_num_features", str(max_num_features),
            "--SiftExtraction.estimate_affine_shape", "1" if estimate_affine_shape else "0",
            "--SiftExtraction.domain_size_pooling", "1" if domain_size_pooling else "0",
            "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        ]

        # Add mask path if provided
        if mask_path:
            args.extend(["--ImageReader.mask_path", str(mask_path)])
            print(f"[COLMAP] Using masks from: {mask_path}")

        success, _ = self._run_command(args, "Feature extraction")
        return success

    def sequential_matcher(
        self,
        use_gpu: bool = True,
        overlap: int = 10,
    ) -> bool:
        """
        Match features sequentially (good for video sequences).

        Args:
            use_gpu: Use GPU for matching
            overlap: Number of neighboring frames to match

        Returns:
            Success status
        """
        args = [
            "sequential_matcher",
            "--database_path", str(self.database_path),
            "--SequentialMatching.overlap", str(overlap),
            "--SequentialMatching.loop_detection", "0",
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        ]

        success, _ = self._run_command(args, "Sequential matching")
        return success

    def exhaustive_matcher(self, use_gpu: bool = True) -> bool:
        """
        Match all image pairs exhaustively (slower but more robust).

        Args:
            use_gpu: Use GPU for matching

        Returns:
            Success status
        """
        args = [
            "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        ]

        success, _ = self._run_command(args, "Exhaustive matching")
        return success

    def mapper(
        self,
        min_num_matches: int = 15,
        ba_refine_focal_length: bool = True,
        ba_refine_principal_point: bool = False,
        ba_refine_extra_params: bool = True,
    ) -> bool:
        """
        Run Structure-from-Motion reconstruction.

        Args:
            min_num_matches: Minimum number of matches for image registration
            ba_refine_focal_length: Refine focal length during bundle adjustment
            ba_refine_principal_point: Refine principal point
            ba_refine_extra_params: Refine distortion parameters

        Returns:
            Success status
        """
        args = [
            "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir),
            "--Mapper.min_num_matches", str(min_num_matches),
            "--Mapper.ba_refine_focal_length", "1" if ba_refine_focal_length else "0",
            "--Mapper.ba_refine_principal_point", "1" if ba_refine_principal_point else "0",
            "--Mapper.ba_refine_extra_params", "1" if ba_refine_extra_params else "0",
        ]

        success, _ = self._run_command(args, "Sparse reconstruction")
        return success

    def model_orientation_aligner(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """
        Align model orientation using Manhattan world assumption.

        Uses vanishing point detection in images to find gravity axis and
        major horizontal axis for proper scene alignment.

        Args:
            input_path: Path to input sparse model
            output_path: Path for aligned output (if None, overwrites input)

        Returns:
            Success status
        """
        if output_path is None:
            output_path = input_path

        args = [
            "model_orientation_aligner",
            "--image_path", str(self.image_dir),
            "--input_path", input_path,
            "--output_path", output_path,
        ]

        success, output = self._run_command(args, "Model orientation alignment (Manhattan world)")

        if success:
            print("[COLMAP] Model aligned using Manhattan world assumption")
        else:
            print("[COLMAP] Model orientation alignment failed (may not have enough vertical/horizontal lines)")

        return success

    def color_extractor(self, input_path: str, output_path: Optional[str] = None) -> bool:
        """
        Extract mean colors for all 3D points from the images.

        This updates the point colors to be the mean color of all observations
        across images, giving more accurate colors than single-image extraction.

        Args:
            input_path: Path to input sparse model
            output_path: Path for output (if None, overwrites input)

        Returns:
            Success status
        """
        if output_path is None:
            output_path = input_path

        args = [
            "color_extractor",
            "--image_path", str(self.image_dir),
            "--input_path", input_path,
            "--output_path", output_path,
        ]

        success, output = self._run_command(args, "Color extraction")

        if success:
            print("[COLMAP] Point colors extracted from images")
        else:
            print("[COLMAP] Color extraction failed")

        return success

    def get_sparse_model_path(self) -> Optional[str]:
        """
        Get path to the best sparse reconstruction model.

        COLMAP creates numbered subdirectories (0, 1, 2, ...) for each model.
        Returns the path to model 0 (usually the largest/best).

        Returns:
            Path to sparse model directory or None
        """
        if not self.sparse_dir or not self.sparse_dir.exists():
            return None

        # Look for model directories
        model_dirs = sorted([d for d in self.sparse_dir.iterdir() if d.is_dir()])

        if not model_dirs:
            print("[COLMAP] No sparse models found")
            return None

        # Return first (usually best) model
        model_path = model_dirs[0]
        print(f"[COLMAP] Using sparse model: {model_path}")
        return str(model_path)

    def run_pipeline(
        self,
        images: np.ndarray,
        camera_model: str = "SIMPLE_RADIAL",
        matcher: str = "sequential",
        max_features: int = 8192,
        use_gpu: bool = True,
        keep_workspace: bool = False,
        align_orientation: bool = True,
        masks: Optional[np.ndarray] = None,
        # Advanced parameters
        sequential_overlap: int = 10,
        max_image_size: int = 4096,
        min_num_matches: int = 15,
        ba_refine_focal_length: bool = True,
        ba_refine_principal_point: bool = False,
        ba_refine_extra_params: bool = True,
        # SIFT parameters
        estimate_affine_shape: bool = False,
        domain_size_pooling: bool = False,
    ) -> Optional[str]:
        """
        Run complete COLMAP pipeline.

        Args:
            images: Image tensor [T, H, W, C]
            camera_model: Camera model type
            matcher: "sequential" or "exhaustive"
            max_features: Max features per image
            use_gpu: Use GPU acceleration
            keep_workspace: Keep temporary files after completion
            align_orientation: Run Manhattan world alignment after reconstruction
            masks: Optional mask tensor [T, H, W] where white=regions to exclude
            sequential_overlap: Number of neighboring frames to match (sequential mode)
            min_num_matches: Minimum matches to register an image
            ba_refine_focal_length: Refine focal length during bundle adjustment
            ba_refine_principal_point: Refine principal point during bundle adjustment
            ba_refine_extra_params: Refine distortion params during bundle adjustment
            estimate_affine_shape: Estimate affine shape of SIFT features (more robust)
            domain_size_pooling: Use Domain Size Pooling for more features

        Returns:
            Path to sparse reconstruction directory or None on failure
        """
        try:
            # Setup
            self.setup_workspace()

            # Export frames
            print(f"[COLMAP] Processing {images.shape[0]} frames...")
            image_paths = self.export_frames(images)

            # Export masks if provided
            mask_path = None
            if masks is not None and len(masks) > 0:
                # Get image filenames for mask naming
                image_names = [Path(p).name for p in image_paths]
                mask_path = self.export_masks(masks, image_names)

            # Feature extraction
            if not self.feature_extractor(
                camera_model=camera_model,
                max_num_features=max_features,
                max_image_size=max_image_size,
                mask_path=mask_path,
                use_gpu=use_gpu,
                estimate_affine_shape=estimate_affine_shape,
                domain_size_pooling=domain_size_pooling,
            ):
                print("[COLMAP] Feature extraction failed")
                return None

            # Matching
            if matcher == "sequential":
                if not self.sequential_matcher(
                    use_gpu=use_gpu,
                    overlap=sequential_overlap,
                ):
                    print("[COLMAP] Sequential matching failed")
                    return None
            else:
                if not self.exhaustive_matcher(use_gpu=use_gpu):
                    print("[COLMAP] Exhaustive matching failed")
                    return None

            # Reconstruction
            if not self.mapper(
                min_num_matches=min_num_matches,
                ba_refine_focal_length=ba_refine_focal_length,
                ba_refine_principal_point=ba_refine_principal_point,
                ba_refine_extra_params=ba_refine_extra_params,
            ):
                print("[COLMAP] Reconstruction failed")
                return None

            # Get result path
            model_path = self.get_sparse_model_path()

            # Run Manhattan world alignment if requested
            if model_path and align_orientation:
                # Try to align, but don't fail if it doesn't work
                # (e.g., outdoor scenes may not have clear Manhattan structure)
                self.model_orientation_aligner(model_path, model_path)

            # Extract accurate colors for 3D points from images
            if model_path:
                self.color_extractor(model_path, model_path)

            if model_path and not keep_workspace:
                # Copy results before cleanup
                print(f"[COLMAP] Reconstruction complete: {model_path}")

            return model_path

        except Exception as e:
            print(f"[COLMAP] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            if not keep_workspace:
                # Don't cleanup yet - caller needs to read results first
                pass

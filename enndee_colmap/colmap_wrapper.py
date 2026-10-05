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
import queue
import subprocess
import tempfile
import shutil
import threading
import time
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


#: Set this to get COLMAP's/GLOMAP's full INFO flood back (their default here: warnings only).
VERBOSE_ENV = "ENNDEE_COLMAP_VERBOSE"
#: glog level 1 = WARNING: keep the warnings, drop the "I2026... " progress spam.
GLOG_LEVEL_WARNING = "1"


def colmap_verbose() -> bool:
    """True when the user asked for COLMAP's/GLOMAP's raw INFO output."""
    return bool((os.environ.get(VERBOSE_ENV) or "").strip())


def colmap_child_env() -> dict:
    """Environment for COLMAP/GLOMAP child processes: quiet unless asked otherwise.

    COLMAP and GLOMAP log through glog; at INFO level they print every SIFT thread setup,
    every processed image and every pairing step - hundreds of lines per run that bury the
    node's own output in ComfyUI's console. glog reads ``GLOG_minloglevel`` when the
    process starts, so the flag goes into the child's environment; warnings and errors
    (missing focal priors, "compiled without CUDA support", failures) still come through.
    ``ENNDEE_COLMAP_VERBOSE=1`` restores the full output.
    """
    env = dict(os.environ)
    # Windows' os.environ upper-cases its keys, and a *copy* is case sensitive again - so
    # look for the flag without caring about case before adding it.
    if not colmap_verbose() and not any(key.upper() == "GLOG_MINLOGLEVEL" for key in env):
        env["GLOG_minloglevel"] = GLOG_LEVEL_WARNING
    return env


def run_streaming_command(cmd, desc: str, timeout: int,
                          progress_callback=None) -> Tuple[int, str]:
    """Run a CLI process while forwarding its combined output as it arrives.

    COLMAP/GLOMAP can run for a long time. Capturing output with
    ``subprocess.run(capture_output=True)`` hides all native progress until a
    phase finishes. This helper relays both newline- and carriage-return-
    terminated progress records to the ComfyUI console while retaining the
    complete output for the wrapper's existing return contract.
    """
    prefix = f"[{desc}]" if desc else "[SfM]"

    def emit(text):
        if not text:
            return
        message = f"{prefix} {text}"
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message, flush=True)

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=colmap_child_env(),
            **subprocess_window_kwargs(),
        )
    except Exception:
        raise

    chunks = []
    pending = bytearray()
    output_queue = queue.Queue()

    def read_output():
        try:
            pipe = process.stdout
            while True:
                # os.read returns bytes currently available instead of waiting
                # for the full requested buffer (essential for live progress).
                chunk = os.read(pipe.fileno(), 4096)
                if not chunk:
                    break
                output_queue.put(chunk)
        except Exception as exc:  # relay reader failures to the main thread
            output_queue.put(exc)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, name="enndee-cli-output", daemon=True)
    reader.start()
    started = time.monotonic()
    next_heartbeat = started + 30.0
    reader_finished = False

    def consume_lines(final=False):
        while pending:
            boundary = next((i for i, byte in enumerate(pending) if byte in (10, 13)), None)
            if boundary is None:
                if final:
                    record = bytes(pending)
                    pending.clear()
                    emit(record.decode("utf-8", errors="replace").strip())
                return
            record = bytes(pending[:boundary])
            delimiter = pending[boundary]
            del pending[:boundary + 1]
            # Treat CRLF as one line break, not an additional empty record.
            if delimiter == 13 and pending[:1] == b"\n":
                del pending[:1]
            emit(record.decode("utf-8", errors="replace").strip())

    try:
        while not reader_finished or process.poll() is None:
            now = time.monotonic()
            if timeout and now - started >= timeout and process.poll() is None:
                process.kill()
                process.wait()
                reader.join(timeout=2.0)
                consume_lines(final=True)
                raise subprocess.TimeoutExpired(cmd, timeout, output=b"".join(chunks))

            try:
                item = output_queue.get(timeout=0.25)
            except queue.Empty:
                item = "__timeout__"

            if item is None:
                reader_finished = True
                consume_lines(final=True)
            elif isinstance(item, Exception):
                emit(f"output reader warning: {item}")
                reader_finished = True
            elif item != "__timeout__":
                chunks.append(item)
                pending.extend(item)
                next_heartbeat = time.monotonic() + 30.0
                consume_lines()
            elif process.poll() is None and now >= next_heartbeat:
                emit(f"still running ({int(now - started)}s elapsed)")
                next_heartbeat = now + 30.0

        return_code = process.wait()
        reader.join(timeout=2.0)
        consume_lines(final=True)
        return return_code, b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


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
        self.progress_callback = None

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
        report_every = max(1, num_frames // 10)

        for i in range(num_frames):
            if i == 0 or (i + 1) % report_every == 0 or i + 1 == num_frames:
                print(f"[COLMAP] Preparing SfM image {i + 1}/{num_frames}", flush=True)
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
            return_code, output = run_streaming_command(
                cmd,
                desc or "COLMAP",
                _timeout_from_env("ENNDEE_COLMAP_TIMEOUT", 3600),
                progress_callback=self.progress_callback,
            )
            if return_code != 0:
                print(f"[COLMAP] {desc} failed (exit code {return_code})", flush=True)
                return False, output

            return True, output

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

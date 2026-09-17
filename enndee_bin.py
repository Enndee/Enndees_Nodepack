"""
enndee_bin - portable COLMAP / GLOMAP binary manager for Enndees Nodepack.
==========================================================================

Makes the node pack a true all-in-one package: the required SfM binaries are
downloaded automatically into ``<Enndees-Nodepack>/bin/`` - either by
ComfyUI-Manager running ``install.py`` or lazily on the first node execution.

Everything is pinned to a *tested* combination (see ``ASSETS``) so that the
Lichtfeld dataset export is reproducible:

    COLMAP 3.11.1   (feature extraction + matching)
    GLOMAP 1.2.0    (global mapper - last release, upstream is deprecated)

Path resolution order used by the node::

    1. explicit path from the node widget
    2. ENNDEE_COLMAP_PATH / ENNDEE_GLOMAP_PATH environment variables
    3. bin/enndee_binaries.json        (written by install.py or manual --pin)
    4. <pack>/bin/<kind>-<version>-<flavor>/<executable>   (auto download)
    5. auto detection (PATH, C:\\Tools\\colmap*, ...)

CLI (also used by ``install.py``)::

    python enndee_bin.py --check                 # show what is available
    python enndee_bin.py                         # install missing binaries
    python enndee_bin.py --force --nocuda        # re-download CPU builds
    python enndee_bin.py --pin colmap="C:\\Tools\\colmap-x64-windows-cuda\\COLMAP.bat"

Environment variables::

    ENNDEE_COLMAP_PATH      explicit COLMAP.bat / colmap.exe
    ENNDEE_GLOMAP_PATH      explicit glomap.exe
    ENNDEE_BIN_FLAVOR       "cuda" | "nocuda"  (overrides auto detection)
    ENNDEE_AUTO_DOWNLOAD    "0" disables the automatic download
    ENNDEE_GLOMAP_VERSION   override the pinned GLOMAP tag (advanced)
    ENNDEE_COLMAP_VERSION   override the pinned COLMAP tag (advanced)
"""

import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

PACK_DIR = Path(__file__).resolve().parent
BIN_DIR = PACK_DIR / "bin"
CONFIG_PATH = BIN_DIR / "enndee_binaries.json"

COLMAP_VERSION = os.environ.get("ENNDEE_COLMAP_VERSION", "3.11.1")
GLOMAP_VERSION = os.environ.get("ENNDEE_GLOMAP_VERSION", "1.2.0")

_GITHUB = "https://github.com"


@dataclass(frozen=True)
class Asset:
    """A downloadable prebuilt binary archive."""

    kind: str          # "colmap" | "glomap"
    flavor: str        # "cuda" | "nocuda"
    version: str
    file_name: str
    url: str
    size: int          # expected size in bytes (0 = unknown)
    sha256: str        # published checksum ("" = not published by upstream)
    exe: str           # executable path relative to the install folder

    @property
    def install_dir(self) -> str:
        return f"{self.kind}-{self.version}-{self.flavor}"


ASSETS: Tuple[Asset, ...] = (
    # --- COLMAP -----------------------------------------------------------
    Asset(
        kind="colmap",
        flavor="cuda",
        version=COLMAP_VERSION,
        file_name="colmap-x64-windows-cuda.zip",
        url=f"{_GITHUB}/colmap/colmap/releases/download/{COLMAP_VERSION}/"
            "colmap-x64-windows-cuda.zip",
        size=153_987_261,
        sha256="",  # upstream published no digest for this tag
        exe="COLMAP.bat",
    ),
    Asset(
        kind="colmap",
        flavor="nocuda",
        version=COLMAP_VERSION,
        file_name="colmap-x64-windows-nocuda.zip",
        url=f"{_GITHUB}/colmap/colmap/releases/download/{COLMAP_VERSION}/"
            "colmap-x64-windows-nocuda.zip",
        size=64_416_015,
        sha256="",
        exe="COLMAP.bat",
    ),
    # --- GLOMAP (global mapper, last release 1.2.0) -----------------------
    Asset(
        kind="glomap",
        flavor="cuda",
        version=GLOMAP_VERSION,
        file_name="glomap-x64-windows-cuda.zip",
        url=f"{_GITHUB}/colmap/glomap/releases/download/{GLOMAP_VERSION}/"
            "glomap-x64-windows-cuda.zip",
        size=73_572_171,
        sha256="7b32a8b0ecfaec28b82d5e0b7d40e38198259849ad1af8ba3e34dab301a3a773",
        exe="bin/glomap.exe",
    ),
    Asset(
        kind="glomap",
        flavor="nocuda",
        version=GLOMAP_VERSION,
        file_name="glomap-x64-windows-nocuda.zip",
        url=f"{_GITHUB}/colmap/glomap/releases/download/{GLOMAP_VERSION}/"
            "glomap-x64-windows-nocuda.zip",
        size=18_151_227,
        sha256="f46ea8acbf1b4afc97f2a46329ac9d9decc228e2077330f5117708dc0962c0d0",
        exe="bin/glomap.exe",
    ),
)

# Accepted executables per kind (first existing one wins)
_EXE_FALLBACKS = {
    "colmap": ("COLMAP.bat", "bin/colmap.exe", "colmap.exe"),
    "glomap": ("bin/glomap.exe", "glomap.exe"),
}

# Env vars that pin a path explicitly
_ENV_VAR = {"colmap": "ENNDEE_COLMAP_PATH", "glomap": "ENNDEE_GLOMAP_PATH"}

# Folders searched by the auto detection (portable, no user specific paths)
_AUTODETECT_DIRS = (
    r"C:\Tools",
    r"C:\Program Files\COLMAP",
    r"C:\Program Files",
    r"D:\Tools",
    str(Path.home() / "Tools"),
    str(Path.home() / "Downloads"),
)

KINDS = ("colmap", "glomap")


def log_default(message: str) -> None:
    """Default logger."""
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Flavor / CUDA detection
# ---------------------------------------------------------------------------

def has_cuda() -> bool:
    """Best effort CUDA detection (torch first, then nvidia-smi)."""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        return shutil.which("nvidia-smi") is not None
    except Exception:
        return False


def resolve_flavor(flavor: str = "auto") -> str:
    """
    Return "cuda" or "nocuda".

    ``auto`` honours ENNDEE_BIN_FLAVOR and otherwise auto detects the GPU.
    """
    flavor = (flavor or "auto").strip().lower()
    if flavor in ("cuda", "nocuda"):
        return flavor

    env = (os.environ.get("ENNDEE_BIN_FLAVOR") or "").strip().lower()
    if env in ("cuda", "nocuda"):
        return env

    return "cuda" if has_cuda() else "nocuda"


def asset_for(kind: str, flavor: str) -> Asset:
    """Return the pinned asset for a kind/flavor combination."""
    for asset in ASSETS:
        if asset.kind == kind and asset.flavor == flavor:
            return asset
    raise KeyError(f"No asset pinned for {kind}/{flavor}")


def compatible_assets(kind: str, flavor: Optional[str] = None) -> List[Asset]:
    """Assets for a kind, preferred flavor first (cuda > nocuda)."""
    preferred = resolve_flavor(flavor) if flavor else resolve_flavor()
    order = [preferred] + [f for f in ("cuda", "nocuda") if f != preferred]
    return [asset_for(kind, f) for f in order]


# ---------------------------------------------------------------------------
# Config file (bin/enndee_binaries.json)
# ---------------------------------------------------------------------------

def load_config() -> Dict:
    """Read the optional user configuration, never raises."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(update: Optional[Dict] = None, replace: bool = False) -> Dict:
    """Merge ``update`` into the config file and write it back."""
    data = {} if replace else load_config()
    if update:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
    try:
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        log_default(f"[Enndee] Could not write {CONFIG_PATH}: {exc}")
    return data


def pin_path(kind: str, path: str) -> Dict:
    """Pin an explicit binary path in the config file."""
    return save_config({kind: {"path": str(path), "source": "manual"}})


def config_path(kind: str) -> Optional[Path]:
    """Path from the config file, if it exists on disk."""
    entry = load_config().get(kind)
    if isinstance(entry, dict):
        candidate = entry.get("path")
    else:
        candidate = entry
    if not candidate:
        return None
    candidate_path = Path(str(candidate))
    return candidate_path if candidate_path.exists() else None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _exe_names(kind: str) -> Tuple[str, ...]:
    return _EXE_FALLBACKS.get(kind, ("",))


def local_candidates(kind: str) -> List[Path]:
    """
    Executables inside <pack>/bin, preferred flavor first.

    Both the CUDA and the CPU flavor are considered so a GPU-less machine can
    still use the pack after a CPU download.
    """
    candidates: List[Path] = []
    if not BIN_DIR.is_dir():
        return candidates

    ordered_dirs: List[Path] = []
    for asset in compatible_assets(kind):
        ordered_dirs.append(BIN_DIR / asset.install_dir)
    # any other version/flavor folders that happen to be present
    for folder in sorted(BIN_DIR.glob(f"{kind}-*")):
        if folder not in ordered_dirs:
            ordered_dirs.append(folder)

    for folder in ordered_dirs:
        for rel in _exe_names(kind):
            candidate = folder / rel
            if candidate.is_file():
                candidates.append(candidate)
    return candidates


def autodetect(kind: str) -> Optional[Path]:
    """Search PATH and a few common install folders - never raises."""
    exe = "glomap" if kind == "glomap" else "colmap"
    found = shutil.which(exe)
    if found:
        return Path(found)

    patterns = (
        ("glomap*/bin/glomap.exe", "glomap*/glomap.exe", "glomap.exe")
        if kind == "glomap"
        else ("colmap*/COLMAP.bat", "colmap*/bin/colmap.exe", "colmap.exe")
    )

    for base in _AUTODETECT_DIRS:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for pattern in patterns:
            try:
                matches = sorted(base_path.glob(pattern))
            except Exception:
                continue
            if matches:
                return matches[0]
    return None


def installed_executable(kind: str) -> Optional[Path]:
    """Executable installed inside the pack's own bin/ folder."""
    candidates = local_candidates(kind)
    return candidates[0] if candidates else None


def source_of(kind: str, explicit: str = "") -> str:
    """Human readable description where a resolved path comes from."""
    if explicit and str(explicit).strip() and Path(explicit.strip().strip('"')).is_file():
        return "node widget"
    if os.environ.get(_ENV_VAR.get(kind, ""), ""):
        return "environment variable"
    if config_path(kind):
        return "bin/enndee_binaries.json"
    if installed_executable(kind):
        return "pack bin/ folder"
    if autodetect(kind):
        return "auto detected"
    return "not found"


def is_installed(kind: str) -> bool:
    """True if a usable binary for ``kind`` lives in the pack's bin/ folder."""
    return installed_executable(kind) is not None


def resolve_binary(kind: str, explicit: str = "") -> Optional[Path]:
    """
    Resolve an executable path using the documented priority chain.

    Returns None when nothing is available (caller should offer a download).
    """
    if explicit and str(explicit).strip():
        candidate = Path(str(explicit).strip().strip('"'))
        if candidate.is_file():
            return candidate

    env_value = os.environ.get(_ENV_VAR.get(kind, ""), "")
    if env_value:
        candidate = Path(env_value.strip().strip('"'))
        if candidate.is_file():
            return candidate

    pinned = config_path(kind)
    if pinned:
        return pinned

    local = installed_executable(kind)
    if local:
        return local

    return autodetect(kind)


# ---------------------------------------------------------------------------
# Download / install
# ---------------------------------------------------------------------------

_USER_AGENT = "Enndees-Nodepack-Installer/1.0 (+https://github.com/)"


def human_size(size: int) -> str:
    """Format a byte count for log output."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Streamed SHA256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, asset: Asset, log=log_default) -> bool:
    """Check size (if known) and checksum (if published by upstream)."""
    try:
        size = path.stat().st_size
    except OSError:
        return False

    if asset.size and size != asset.size:
        log(f"[Enndee] Size mismatch for {path.name}: "
            f"{human_size(size)} != {human_size(asset.size)}")
        return False

    if asset.sha256:
        digest = sha256_of_file(path)
        if digest != asset.sha256:
            log(f"[Enndee] Checksum mismatch for {path.name}")
            return False

    return True


def download_file(asset: Asset, dest_dir: Path, log=log_default,
                  retries: int = 3) -> Path:
    """
    Download an asset into ``dest_dir`` (resumable, verified).

    Raises RuntimeError when the download could not be completed/verified.
    """
    import time
    import urllib.error
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / asset.file_name
    partial = target.with_name(target.name + ".part")

    if target.is_file() and verify_file(target, asset, log):
        log(f"[Enndee] Already downloaded: {target.name}")
        return target

    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": _USER_AGENT}
        if existing:
            headers["Range"] = f"bytes={existing}-"

        log(f"[Enndee] Downloading {asset.file_name} "
            f"({human_size(asset.size)}) - attempt {attempt}/{retries}")
        try:
            request = urllib.request.Request(asset.url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = existing > 0 and response.status == 206
                mode = "ab" if resumed else "wb"
                done = existing if resumed else 0
                total = asset.size or 0
                if not total:
                    length = response.headers.get("Content-Length")
                    total = (int(length) + done) if length else 0

                with open(partial, mode) as fh:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        fh.write(block)
                        done += len(block)
                        progress(done, total, log)

            if total:
                log("")
            if not verify_file(partial, asset, log):
                partial.unlink(missing_ok=True)
                last_error = "verification failed"
                continue

            partial.replace(target)
            log(f"[Enndee] Saved {target} ({human_size(target.stat().st_size)})")
            return target

        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            log(f"[Enndee] Download failed ({type(exc).__name__}: {exc})")
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"Could not download {asset.file_name}: {last_error}")


def progress(done: int, total: int, log=log_default) -> None:
    """Print a single line progress indicator."""
    if not total:
        log(f"\r[Enndee]   {human_size(done)}")
        return
    percent = min(100.0, done * 100.0 / total)
    log(f"\r[Enndee]   {percent:5.1f}%  {human_size(done)} / {human_size(total)}")


def extract_zip(zip_path: Path, dest_dir: Path, log=log_default) -> None:
    """Extract a zip archive with zip-slip protection."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"Unsafe path inside archive: {name}")
            target = (root / name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path inside archive: {name}")
        archive.extractall(dest_dir)
    log(f"[Enndee] Extracted {zip_path.name} -> {dest_dir}")


def _flatten_single_folder(dest_dir: Path) -> None:
    """
    Lift the contents up when the archive has a single top level folder.

    Keeps the on-disk layout predictable (``bin/glomap.exe``, ``COLMAP.bat``)
    regardless of how upstream packed the release.
    """
    try:
        entries = list(dest_dir.iterdir())
    except OSError:
        return
    folders = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file() and not e.name.endswith(".part")]
    if len(folders) != 1 or files:
        return

    inner = folders[0]
    for item in list(inner.iterdir()):
        item.replace(dest_dir / item.name)
    try:
        inner.rmdir()
    except OSError:
        pass


def find_executable(root: Path, exe_rel: str) -> Optional[Path]:
    """Find an executable below ``root``, tolerating one extra folder level."""
    name = Path(exe_rel).name
    direct = root / exe_rel
    if direct.is_file():
        return direct

    for child in sorted(root.rglob(name)):
        if child.is_file():
            return child
    return None


def install_asset(asset: Asset, log=log_default, force: bool = False,
                  keep_archive: bool = False) -> Path:
    """
    Download + extract a single asset into ``<pack>/bin/<install_dir>``.

    Returns the executable path. Raises RuntimeError on failure.
    """
    target_dir = BIN_DIR / asset.install_dir

    if not force:
        existing = find_executable(target_dir, asset.exe)
        if existing is not None:
            log(f"[Enndee] {asset.kind} {asset.version} ({asset.flavor}) already installed")
            return existing

    if force and target_dir.exists():
        log(f"[Enndee] Removing old {target_dir.name}")
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    archive = download_file(asset, BIN_DIR / "downloads", log)
    extract_zip(archive, target_dir, log)
    _flatten_single_folder(target_dir)

    executable = find_executable(target_dir, asset.exe)
    if executable is None:
        raise RuntimeError(
            f"{asset.kind}: {asset.exe} not found inside {target_dir} "
            f"after extracting {archive.name}"
        )

    if not keep_archive:
        archive.unlink(missing_ok=True)

    log(f"[Enndee] {asset.kind} {asset.version} ({asset.flavor}) -> {executable}")
    return executable


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ensure_binaries(kinds=KINDS, flavor: str = "auto", force: bool = False,
                    allow_download: bool = True,
                    log=log_default) -> Dict[str, Optional[Path]]:
    """
    Resolve the required binaries, downloading them if necessary.

    Returns a mapping ``{kind: executable path or None}``.
    """
    from datetime import datetime

    result: Dict[str, Optional[Path]] = {}
    metadata: Dict[str, Dict[str, str]] = {}

    for kind in kinds:
        resolved = resolve_binary(kind)
        if resolved is not None and not force:
            log(f"[Enndee] {kind}: using {resolved}  [{source_of(kind)}]")
            result[kind] = resolved
            continue

        if not allow_download:
            result[kind] = resolved
            continue

        log(f"[Enndee] {kind}: no usable binary found - installing prebuilt version")
        installed: Optional[Path] = None
        for asset in compatible_assets(kind, flavor):
            try:
                installed = install_asset(asset, log, force=force)
                metadata[kind] = {
                    "version": asset.version,
                    "flavor": asset.flavor,
                    "source": "download",
                    "installed_at": datetime.now().isoformat(timespec="seconds"),
                }
                break
            except Exception as exc:  # noqa: BLE001
                log(f"[Enndee] {kind} ({asset.flavor}) install failed: {exc}")

        result[kind] = installed if installed is not None else resolved
        if result[kind] is None:
            log(f"[Enndee] WARNING: {kind} is still unavailable - "
                f"the node cannot run until it is installed.")

    if metadata:
        save_config(metadata)
    return result


def status_report(log=log_default) -> Dict[str, Optional[str]]:
    """Log a summary of the current binary situation and return it."""
    log("[Enndee] Binary status")
    log(f"[Enndee]   pack folder : {PACK_DIR}")
    log(f"[Enndee]   bin folder  : {BIN_DIR}")
    log(f"[Enndee]   platform    : {platform.system()} {platform.release()} / "
        f"python {sys.version.split()[0]}")
    log(f"[Enndee]   cuda        : {'yes' if has_cuda() else 'no'} "
        f"(flavor={resolve_flavor()})")

    result: Dict[str, Optional[str]] = {}
    for kind in KINDS:
        path = resolve_binary(kind)
        result[kind] = str(path) if path else None
        shown = path if path else "MISSING"
        log(f"[Enndee]   {kind:<7}: {shown}  [{source_of(kind)}]")
    return result


def main(argv=None) -> int:
    """CLI entry point - also used by install.py and ComfyUI-Manager."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="enndee_bin",
        description="Download / locate COLMAP and GLOMAP for Enndees Nodepack",
    )
    parser.add_argument("--check", action="store_true",
                        help="only report the current status, never download")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a binary is already present")
    flavor_group = parser.add_mutually_exclusive_group()
    flavor_group.add_argument("--cuda", action="store_true",
                              help="force the CUDA build")
    flavor_group.add_argument("--nocuda", action="store_true",
                              help="force the CPU build")
    parser.add_argument("--only", choices=list(KINDS), action="append",
                        help="install only this component (repeatable)")
    parser.add_argument("--pin", action="append", metavar="KIND=PATH",
                        help="pin an existing installation, e.g. "
                             "--pin colmap=C:\\Tools\\colmap-x64-windows-cuda\\COLMAP.bat")
    parser.add_argument("--no-download", action="store_true",
                        help="resolve only, never download")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = (lambda *_a, **_kw: None) if args.quiet else log_default

    pinned = False
    for item in args.pin or []:
        if "=" not in item:
            log(f"[Enndee] Ignoring invalid --pin value: {item}")
            continue
        kind, _, value = item.partition("=")
        kind = kind.strip().lower()
        value = value.strip().strip('"')
        if kind not in KINDS:
            log(f"[Enndee] Unknown component in --pin: {kind}")
            continue
        if not Path(value).is_file():
            log(f"[Enndee] Pinned file does not exist: {value}")
            continue
        pin_path(kind, value)
        log(f"[Enndee] Pinned {kind} -> {value}")
        pinned = True

    if args.check:
        resolved = status_report(log)
        return 0 if all(resolved.values()) else 1

    flavor = "cuda" if args.cuda else ("nocuda" if args.nocuda else "auto")
    kinds = tuple(args.only) if args.only else KINDS

    if pinned:
        log("[Enndee] Pinned paths win over freshly downloaded binaries.")

    resolved = ensure_binaries(kinds=kinds, flavor=flavor, force=args.force,
                               allow_download=not args.no_download, log=log)
    status_report(log)
    return 0 if all(resolved.get(k) for k in kinds) else 1


if __name__ == "__main__":
    sys.exit(main())
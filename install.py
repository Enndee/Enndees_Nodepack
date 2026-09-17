#!/usr/bin/env python
"""
Enndees Nodepack - installer
============================

ComfyUI-Manager automatically executes an ``install.py`` placed in a custom node
folder (after installing ``requirements.txt``).  This script therefore makes the
pack a real *all-in-one* package: it downloads the tested COLMAP + GLOMAP
binaries into ``<Enndees-Nodepack>/bin/`` so the "GLOMAP Lichtfeld Tracker
(Enndee)" node works right away - no manual COLMAP/GLOMAP installation needed.

Manual usage::

    python install.py                  # download what is missing
    python install.py --check          # only report the current status
    python install.py --force          # re-download everything
    python install.py --nocuda         # CPU builds (no NVIDIA GPU)
    python install.py --only glomap    # single component
    python install.py --pin glomap="D:\\Tools\\glomap-1.2.0\\bin\\glomap.exe"
    python install.py --skip-binaries  # do not download anything

Pinned versions (see ``enndee_bin.py`` for the URLs and checksums):

    COLMAP 3.11.1   - SIFT features + matching
    GLOMAP 1.2.0    - global mapper (last upstream release)

Everything is installed below this folder, so the pack stays self contained and
nothing has to be added to the system PATH.  ``bin/`` is git-ignored.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
BANNER = "=" * 72


def step(message: str) -> None:
    """Print a formatted progress line."""
    print(f"[Enndees-Nodepack] {message}", flush=True)


def install_requirements(verbose: bool = True) -> int:
    """
    Install the python requirements with the interpreter running this script.

    ComfyUI-Manager already does this, so failures are not fatal.
    """
    requirements = PACK_DIR / "requirements.txt"
    if not requirements.is_file():
        return 0

    step(f"Installing python requirements from {requirements.name}")
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    try:
        result = subprocess.run(command, capture_output=not verbose, text=True)
    except Exception as exc:  # noqa: BLE001
        step(f"Could not run pip: {exc}")
        return 1

    if result.returncode != 0:
        step("pip returned an error - the pack still works without the "
             "optional features (RMBG / audio extraction)")
    return result.returncode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="install.py",
        description="Install the COLMAP/GLOMAP binaries for Enndees Nodepack",
    )
    parser.add_argument("--check", action="store_true",
                        help="only report the current status")
    parser.add_argument("--force", action="store_true",
                        help="re-download binaries that are already present")
    flavor = parser.add_mutually_exclusive_group()
    flavor.add_argument("--cuda", action="store_true", help="force the CUDA build")
    flavor.add_argument("--nocuda", action="store_true", help="force the CPU build")
    parser.add_argument("--only", choices=("colmap", "glomap"), action="append",
                        help="install only this component (repeatable)")
    parser.add_argument("--pin", action="append", metavar="KIND=PATH",
                        help="use an already installed binary, e.g. "
                             "--pin colmap=C:\\Tools\\colmap-x64-windows-cuda\\COLMAP.bat")
    parser.add_argument("--skip-binaries", action="store_true",
                        help="do not download or resolve any binary")
    parser.add_argument("--with-requirements", action="store_true",
                        help="also run pip install -r requirements.txt")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print(BANNER)
    step("Installer for Enndees Nodepack")
    step(f"pack folder: {PACK_DIR}")
    step(f"python     : {sys.version.split()[0]}  ({sys.executable})")
    print(BANNER)

    if args.with_requirements:
        install_requirements(verbose=True)

    if args.skip_binaries:
        step("Skipping COLMAP/GLOMAP setup (--skip-binaries)")
        print(BANNER)
        return 0

    sys.path.insert(0, str(PACK_DIR))
    try:
        import enndee_bin
    except Exception as exc:  # noqa: BLE001
        step(f"ERROR: could not import enndee_bin.py: {exc}")
        return 1

    forwarded = ["--check"] if args.check else []
    if args.force:
        forwarded.append("--force")
    if args.cuda:
        forwarded.append("--cuda")
    if args.nocuda:
        forwarded.append("--nocuda")
    for kind in args.only or []:
        forwarded.extend(["--only", kind])
    for item in args.pin or []:
        forwarded.extend(["--pin", item])

    exit_code = enndee_bin.main(forwarded)

    print(BANNER)
    if exit_code == 0:
        step("Setup finished - COLMAP and GLOMAP are ready.")
        step("The node auto-detects them; leave colmap_path/glomap_path empty "
             "unless you want to override them.")
    else:
        step("Setup incomplete - at least one binary is missing.")
        step("Run 'python install.py' again, pass --pin <KIND>=<PATH> for an "
             "existing installation, or set ENNDEE_COLMAP_PATH / "
             "ENNDEE_GLOMAP_PATH.")
    print(BANNER)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
"""
Make sure the optional python accelerators live in ComfyUI's environment.

A plain ComfyUI install ships none of these, and the ONNX based nodes (BRIA RMBG,
ComfyUI-RMBG, ...) plus the native COLMAP node need them:

* ``pycolmap`` - COLMAP's native Python API (used by **COLMAP for Lichtfeld**).
* ``onnxruntime-gpu`` - the CUDA ONNX runtime. Two traps make this fail silently:

  1. the **CPU wheel** (``onnxruntime``) **shadows** the GPU one - both install the
     same ``onnxruntime`` package and the CPU build wins when it is installed last,
     so ``get_available_providers()`` no longer lists ``CUDAExecutionProvider``.
     Never keep both.
  2. the wheel must match the **CUDA major version** of the installed torch:
     CUDA 13 needs ``onnxruntime-gpu >= 1.30`` (older CUDA 12 builds cannot even
     load their provider DLL on a CUDA 13 machine).

* optional: it *reports* flash-attn / sageattention (attention accelerators) but
  never installs them - they are build specific and already handled by the
  ComfyUI launcher.

Auto install is on by default and honours the same switch as the COLMAP/GLOMAP
binaries (``ENNDEE_AUTO_DOWNLOAD=0`` disables it) plus the node's
``auto_install_binaries`` widget.
"""

import os
import subprocess
import sys
from typing import Callable, Dict, Optional

#: pip requirement per torch CUDA major version.
ONNX_GPU_SPECS = {13: "onnxruntime-gpu>=1.30", 12: "onnxruntime-gpu>=1.19,<1.30"}
ONNX_GPU_FALLBACK = "onnxruntime-gpu"
PYCOLMAP_SPEC = "pycolmap"


def auto_install_enabled() -> bool:
    """``ENNDEE_AUTO_DOWNLOAD=0`` disables the on-demand package installs."""
    value = (os.environ.get("ENNDEE_AUTO_DOWNLOAD") or "").strip().lower()
    return value not in ("0", "false", "no", "off")


def _run(args, dry: bool = False, timeout: int = 3600):
    """Run pip / the interpreter; returns ``(returncode, combined output)``."""
    printable = " ".join(str(part) for part in args)
    print(f"[accelerators] $ {printable}", flush=True)
    if dry:
        return 0, ""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        result = subprocess.run([str(part) for part in args], capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return 1, repr(exc)
    output = (result.stdout or "") + (result.stderr or "")
    tail = "\n".join(line for line in output.strip().splitlines()[-4:])
    if tail:
        print(f"[accelerators]   {tail}", flush=True)
    return result.returncode, output


def pip(args, dry: bool = False):
    return _run([sys.executable, "-m", "pip"] + list(args), dry=dry)


def torch_cuda_major() -> Optional[int]:
    """CUDA major version of the installed torch (13 for ``2.14.1+cu130``)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        version = getattr(torch.version, "cuda", None)
        if not version:
            return None
        return int(str(version).split(".")[0])
    except Exception:  # noqa: BLE001
        return None


def onnxruntime_state() -> Dict[str, object]:
    """What the *resolved* ``onnxruntime`` package can do right now."""
    state: Dict[str, object] = {"available": False, "version": "", "providers": [],
                                "cuda": False, "cpu_wheel": False, "gpu_wheel": False}
    try:
        import onnxruntime  # type: ignore

        state["available"] = True
        state["version"] = str(getattr(onnxruntime, "__version__", "?"))
        providers = list(onnxruntime.get_available_providers())
        state["providers"] = providers
        state["cuda"] = any("CUDA" in name for name in providers)
    except Exception:  # noqa: BLE001
        pass
    try:
        import importlib.metadata as metadata

        installed = {dist.metadata["Name"].lower().replace("_", "-")
                     for dist in metadata.distributions()}
        state["cpu_wheel"] = "onnxruntime" in installed
        state["gpu_wheel"] = "onnxruntime-gpu" in installed
    except Exception:  # noqa: BLE001
        pass
    return state


def pycolmap_state() -> Dict[str, object]:
    from enndee_colmap.pycolmap_wrapper import pycolmap_info

    return pycolmap_info()


def accelerator_report() -> Dict[str, object]:
    """Everything the node logs before it starts (no installs, never raises)."""
    report: Dict[str, object] = {
        "cuda_major": torch_cuda_major(),
        "onnxruntime": onnxruntime_state(),
        "pycolmap": pycolmap_state(),
    }
    optional = {}
    for name in ("flash_attn", "sageattention"):
        try:
            __import__(name)
            optional[name] = True
        except Exception:  # noqa: BLE001
            optional[name] = False
    report["optional_attention"] = optional
    return report


def _onnx_spec(cuda_major: Optional[int]) -> str:
    return ONNX_GPU_SPECS.get(cuda_major or 0, ONNX_GPU_FALLBACK)


def ensure_accelerators(auto_install: bool = True, dry: bool = False,
                        log: Optional[Callable[[str], None]] = None,
                        install_onnx: bool = True,
                        install_pycolmap: bool = True) -> Dict[str, object]:
    """Install/repair ``pycolmap`` and a CUDA-matching ``onnxruntime-gpu``.

    Returns the report **after** the work. Idempotent: a healthy environment costs
    two imports and nothing else.
    """
    def say(message: str) -> None:
        if log is not None:
            log(message)
        else:
            print(f"[accelerators] {message}", flush=True)

    report = accelerator_report()
    cuda_major = report["cuda_major"]
    ort = report["onnxruntime"]
    colmap = report["pycolmap"]

    say(f"CUDA {cuda_major or 'none'} | onnxruntime {ort['version'] or 'missing'} "
        f"({', '.join(ort['providers']) or 'no providers'}) | "
        f"pycolmap {colmap['version'] or 'missing'}")

    allowed = bool(auto_install) and auto_install_enabled()
    if not allowed:
        if not colmap["available"] or not ort["cuda"]:
            say("auto install disabled (ENNDEE_AUTO_DOWNLOAD=0 or the node switch) - "
                "install manually: pip install " + _onnx_spec(cuda_major) + " " +
                PYCOLMAP_SPEC)
        return report

    # ---- onnxruntime: the CPU wheel shadows the GPU one -----------------
    if install_onnx and cuda_major is not None and not ort["cuda"]:
        if ort.get("cpu_wheel"):
            say("removing the CPU onnxruntime wheel - it shadows onnxruntime-gpu")
            pip(["uninstall", "-y", "onnxruntime"], dry=dry)
        spec = _onnx_spec(cuda_major)
        say(f"installing {spec} (CUDA {cuda_major})")
        code, _ = pip(["install", "--upgrade", spec], dry=dry)
        if code != 0:
            say(f"WARNING: could not install {spec} - the ONNX nodes stay on the CPU")
        report["onnxruntime"] = onnxruntime_state()
        say("onnxruntime providers now: %s"
            % (", ".join(report["onnxruntime"]["providers"]) or "none"))

    # ---- pycolmap -------------------------------------------------------
    if install_pycolmap and not colmap["available"]:
        say(f"installing {PYCOLMAP_SPEC} (COLMAP's native Python API)")
        code, _ = pip(["install", PYCOLMAP_SPEC], dry=dry)
        if code != 0:
            say("WARNING: could not install pycolmap - the native COLMAP node "
                "cannot run")
        report["pycolmap"] = pycolmap_state()

    return report


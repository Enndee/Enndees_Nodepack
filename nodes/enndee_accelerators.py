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
#: packages that ship a CUDA enabled pycolmap (platform dependent, see docs)
PYCOLMAP_CUDA_SPECS = ("pycolmap-cuda12",)
#: point at a self built / downloaded CUDA pycolmap wheel (path or URL)
PYCOLMAP_CUDA_WHEEL_ENV = "ENNDEE_PYCOLMAP_CUDA_WHEEL"
#: one CUDA install attempt per session is enough
_CUDA_ATTEMPT: Dict[str, object] = {}


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


def cuda_state() -> Dict[str, object]:
    """Is a usable NVIDIA CUDA environment present?

    torch is asked first (it knows its own build), ``nvidia-smi`` second - a
    CPU-only torch must not hide a perfectly usable CUDA driver from pycolmap.
    """
    state: Dict[str, object] = {"available": False, "version": "", "device": "",
                                "source": ""}
    try:
        import torch

        if torch.cuda.is_available():
            state.update({"available": True, "source": "torch",
                          "version": str(getattr(torch.version, "cuda", "") or ""),
                          "device": torch.cuda.get_device_name(0)})
            return state
    except Exception:  # noqa: BLE001
        pass

    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=30, **kwargs)
        first = (result.stdout or "").strip().splitlines()
        if result.returncode == 0 and first:
            name, _, driver = first[0].partition(",")
            state.update({"available": True, "source": "nvidia-smi",
                          "version": driver.strip(), "device": name.strip()})
    except Exception:  # noqa: BLE001
        pass
    return state


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


def _pycolmap_result(state: Dict[str, object], cuda: Dict[str, object], mode: str,
                     reason: str) -> Dict[str, object]:
    """Uniform pycolmap status: mode is "cuda", "cpu", "cpu-fallback" or "missing"."""
    return {
        "available": bool(state.get("available")),
        "version": state.get("version", ""),
        "cuda": bool(state.get("cuda")),
        "mode": mode,
        "reason": reason,
        "cuda_available": bool(cuda.get("available")),
        "cuda_device": cuda.get("device", ""),
        "cuda_version": cuda.get("version", ""),
    }


def _pip_can_install(spec: str, dry: bool = False) -> bool:
    """Does pip see an installable wheel for this requirement on *this* platform?"""
    code, _ = pip(["install", "--dry-run", spec], dry=dry)
    return code == 0


def _install_pycolmap_cuda(say, dry: bool = False) -> str:
    """Try every known CUDA pycolmap source; returns the reason when none worked."""
    reasons = []
    wheel = (os.environ.get(PYCOLMAP_CUDA_WHEEL_ENV) or "").strip()
    if wheel:
        say(f"{PYCOLMAP_CUDA_WHEEL_ENV} is set - installing the CUDA build from {wheel}")
        code, _ = pip(["install", "--force-reinstall", "--no-deps", wheel], dry=dry)
        if code == 0:
            return f"installed from {PYCOLMAP_CUDA_WHEEL_ENV}"
        reasons.append(f"{PYCOLMAP_CUDA_WHEEL_ENV} could not be installed")

    for spec in PYCOLMAP_CUDA_SPECS:
        if not _pip_can_install(spec, dry=dry):
            reasons.append(f"no '{spec}' wheel exists for this platform")
            continue
        say(f"installing the CUDA build: {spec}")
        code, _ = pip(["install", "--force-reinstall", "--no-deps", spec], dry=dry)
        if code == 0:
            return f"installed {spec}"
        reasons.append(f"{spec} could not be installed")

    reasons.append(f"build pycolmap from source with CUDA, or set "
                   f"{PYCOLMAP_CUDA_WHEEL_ENV}=<wheel|url>")
    return "; ".join(reasons)


def ensure_pycolmap(auto_install: bool = True, prefer_cuda: bool = True, dry: bool = False,
                    log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Make sure pycolmap is installed - the **CUDA** build whenever possible.

    The CPU build is only the fallback, and the returned ``reason`` always says
    *why* (no CUDA device, no CUDA wheel for this platform, install disabled, ...).
    The CUDA attempt runs at most once per session.
    """
    def say(message: str) -> None:
        if log is not None:
            log(message)
        else:
            print(f"[accelerators] {message}", flush=True)

    cuda = cuda_state()
    state = pycolmap_state()
    allowed = bool(auto_install) and auto_install_enabled()

    # ---- 1. installed at all? -------------------------------------------
    if not state.get("available"):
        if not allowed:
            return _pycolmap_result(state, cuda, "missing",
                                    "pycolmap is not installed and auto install is off "
                                    f"(pip install {PYCOLMAP_SPEC})")
        say(f"installing {PYCOLMAP_SPEC} (COLMAP's native Python API)")
        pip(["install", PYCOLMAP_SPEC], dry=dry)
        state = pycolmap_state()
        if not state.get("available"):
            return _pycolmap_result(state, cuda, "missing",
                                    "pycolmap could not be installed")

    # ---- 2. already a CUDA build? ---------------------------------------
    if state.get("cuda"):
        return _pycolmap_result(state, cuda, "cuda", "pycolmap was built with CUDA support")

    # ---- 3. no CUDA on this machine -> CPU is the right answer ----------
    if not cuda.get("available"):
        return _pycolmap_result(state, cuda, "cpu",
                                "no CUDA device/driver detected - the CPU build is correct")

    if not prefer_cuda:
        return _pycolmap_result(state, cuda, "cpu-fallback", "prefer_cuda is disabled")

    if not allowed:
        return _pycolmap_result(
            state, cuda, "cpu-fallback",
            "CUDA is available but auto install is off - install "
            f"{PYCOLMAP_CUDA_SPECS[0]} or set {PYCOLMAP_CUDA_WHEEL_ENV}=<wheel|url>")

    # ---- 4. try to get a CUDA build (once per session) ------------------
    cache_key = f"{state.get('version')}|{cuda.get('version')}"
    if cache_key not in _CUDA_ATTEMPT:
        say(f"CUDA {cuda.get('version') or '?'} ({cuda.get('device') or 'GPU'}) is available, "
            f"but pycolmap has no CUDA support - looking for a CUDA build")
        _CUDA_ATTEMPT[cache_key] = _install_pycolmap_cuda(say, dry=dry)
        state = pycolmap_state()

    if state.get("cuda"):
        return _pycolmap_result(state, cuda, "cuda", "CUDA build installed")

    reason = str(_CUDA_ATTEMPT[cache_key])
    say(f"staying on the CPU: {reason}")
    return _pycolmap_result(state, cuda, "cpu-fallback", reason)


def accelerator_report() -> Dict[str, object]:
    """Everything the node logs before it starts (no installs, never raises)."""
    report: Dict[str, object] = {
        "cuda": cuda_state(),
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
                        install_pycolmap: bool = True,
                        prefer_cuda: bool = True) -> Dict[str, object]:
    """Install/repair ``pycolmap`` (CUDA build first) and a CUDA ``onnxruntime-gpu``.

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

    cuda = report.get("cuda") or {}
    say(f"CUDA {cuda_major or cuda.get('version') or 'none'} "
        f"({cuda.get('device') or 'no GPU detected'}) | "
        f"onnxruntime {ort['version'] or 'missing'} "
        f"({', '.join(ort['providers']) or 'no providers'}) | "
        f"pycolmap {colmap['version'] or 'missing'}")

    allowed = bool(auto_install) and auto_install_enabled()

    # ---- onnxruntime: the CPU wheel shadows the GPU one -----------------
    if install_onnx and allowed and cuda_major is not None and not ort["cuda"]:
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

    # ---- pycolmap: CUDA build first, CPU only as the fallback -----------
    if install_pycolmap:
        report["pycolmap"] = colmap = ensure_pycolmap(
            auto_install=allowed, prefer_cuda=prefer_cuda, dry=dry, log=log)
        say(f"pycolmap {colmap['version'] or 'missing'} [{colmap['mode']}] "
            f"{colmap['reason']}")

    if not allowed and (not colmap["available"] or not ort["cuda"]):
        say("auto install is off (ENNDEE_AUTO_DOWNLOAD=0 or the node switch) - manual: "
            f"pip install {_onnx_spec(cuda_major)} {PYCOLMAP_SPEC}")
    return report


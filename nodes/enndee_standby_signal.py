"""Standby On Signal: suspend the PC (S3) once the workflow reaches this node.

Part of the Wakeup_from_Sleep toolkit (see the workspace's CONTEXT.md): the
Windows idle standby is unreliable on this machine, so this node forces standby
itself. It uses the same proven mechanism as Standby_Timer.ps1 /
Standby_Guard.ps1: powrprof!SetSuspendState(bHibernate = FALSE) -> suspend /
sleep (S3), no admin rights required, rundll32 fallback when the API reports
an error.

Safety: "enabled" defaults to False (DryRun-style no-op until you opt in),
DryRun mode only logs, and by default the node sleeps only when the ComfyUI
queue is empty - for several queued prompts only the last one actually sleeps.

Note for Standby_Guard users: ComfyUI holds no power requests, so run ComfyUI
through Keep-AwakeDuring.ps1 (or pause the guard) to keep the idle guard from
sleeping mid-generation - this node then takes over the sleep at the end.
"""

import ctypes
import json
import subprocess
import time
import urllib.request

# Optional: ComfyUI interrupt support. The module stays importable (and so
# testable) without ComfyUI present - falls back to "never interrupted".
try:
    import comfy.model_management as _mm
except Exception:
    _mm = None

_TAG = "[Enndee/StandbySignal]"


def _log(message):
    print("{} {}".format(_TAG, message), flush=True)


def _processing_interrupted():
    """True while the user pressed Cancel/Interrupt in the ComfyUI UI."""
    if _mm is None:
        return False
    try:
        return bool(_mm.processing_interrupted())
    except Exception:
        return False


def _queue_is_empty(server_url, timeout=3.0):
    """True = queue empty, False = jobs pending/running, None = not checkable."""
    url = server_url.rstrip("/") + "/queue"
    try:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        running = data.get("queue_running", [])
        pending = data.get("queue_pending", [])
        return (len(running) == 0) and (len(pending) == 0)
    except Exception as exc:
        _log("Queue check failed ({}): {}".format(url, exc))
        return None


def _trigger_standby():
    """Suspend (S3) via powrprof SetSuspendState(bHibernate=FALSE, ...)."""
    ok = 0
    try:
        ok = ctypes.windll.powrprof.SetSuspendState(False, False, False)
    except Exception as exc:
        _log("SetSuspendState call failed: {}".format(exc))
    if ok:
        return True
    _log("SetSuspendState reported an error - falling back to "
         "rundll32 powrprof.dll,SetSuspendState 0,1,0")
    try:
        subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        return True
    except Exception as exc:
        _log("Fallback failed: {}".format(exc))
        return False


class Enndee_StandbyOnSignal:
    """Sleeps the PC when the workflow reaches this node and the queue is empty."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Safety switch: while False the node only logs a "
                               "hint and nothing else happens.",
                }),
                "mode": (["Standby (S3)", "DryRun (log only)"], {
                    "tooltip": "Standby (S3) really suspends the PC; DryRun only "
                               "writes to the console - use it for testing.",
                }),
                "delay_seconds": ("INT", {
                    "default": 30, "min": 0, "max": 3600,
                    "tooltip": "Grace period after this node ran before sleeping. "
                               "The ComfyUI Cancel/Interrupt button aborts during "
                               "this window.",
                }),
                "check_queue": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Sleep only when the ComfyUI queue is empty - with "
                               "several queued prompts only the last one sleeps. "
                               "The queue is re-checked after the grace period; "
                               "on unreachable servers nothing sleeps (safe).",
                }),
                "server_url": ("STRING", {
                    "default": "http://127.0.0.1:8188",
                    "tooltip": "Base URL of the ComfyUI API used for the queue "
                               "check (adjust when ComfyUI runs on another port).",
                }),
            },
            "optional": {
                "signal": ("*", ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("signal",)
    OUTPUT_NODE = True
    FUNCTION = "trigger"
    CATEGORY = "Enndee/utils"

    def trigger(self, enabled, mode, delay_seconds, check_queue, server_url, signal=None):
        if not enabled:
            _log("Node is disabled (enabled=False) - nothing to do.")
            return (signal,)

        if check_queue:
            state = _queue_is_empty(server_url)
            if state is False:
                _log("Queue is not empty (more prompts waiting) - no standby.")
                return (signal,)
            if state is None:
                _log("Queue not checkable - staying awake for safety.")
                return (signal,)
            _log("Queue is empty - this was the last queued prompt.")

        for _ in range(int(delay_seconds)):
            if _processing_interrupted():
                _log("Cancelled (interrupt) - no standby.")
                return (signal,)
            time.sleep(1)

        if check_queue:
            state = _queue_is_empty(server_url)
            if state is not True:
                _log("New jobs queued after the grace period (or not checkable) "
                     "- no standby.")
                return (signal,)

        if str(mode).startswith("DryRun"):
            _log("DryRun: standby would trigger now (nothing happens).")
            return (signal,)

        _log("Triggering standby - good night. "
             "(ComfyUI keeps running and resumes after wake.)")
        _trigger_standby()
        _log("Awake again - ComfyUI is ready.")
        return (signal,)

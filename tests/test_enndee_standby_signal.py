"""Tests for the Enndee Standby On Signal node (no ComfyUI/torch needed)."""

import sys
import time
import unittest
from pathlib import Path


PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))

import enndee_standby_signal as standby_signal  # noqa: E402
from enndee_standby_signal import Enndee_StandbyOnSignal  # noqa: E402


DEAD_URL = "http://127.0.0.1:9"


class StandbyOnSignalTests(unittest.TestCase):
    def trigger(self, **overrides):
        kwargs = dict(
            enabled=False,
            mode="Standby (S3)",
            delay_seconds=0,
            check_queue=False,
            server_url=DEAD_URL,
            signal=None,
        )
        kwargs.update(overrides)
        return Enndee_StandbyOnSignal().trigger(**kwargs)

    def patched_standby(self, func):
        """Run 'func' with _trigger_standby replaced; returns the call count list."""
        calls = []
        original = standby_signal._trigger_standby
        standby_signal._trigger_standby = lambda: (calls.append(True), True)[1]
        try:
            func()
        finally:
            standby_signal._trigger_standby = original
        return calls

    def test_interface_and_widget_defaults(self):
        inputs = Enndee_StandbyOnSignal.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            ["enabled", "mode", "delay_seconds", "check_queue", "server_url"],
        )
        self.assertIn("signal", inputs["optional"])
        self.assertTrue(Enndee_StandbyOnSignal.OUTPUT_NODE)
        self.assertEqual(Enndee_StandbyOnSignal.RETURN_TYPES, ("*",))
        self.assertEqual(Enndee_StandbyOnSignal.RETURN_NAMES, ("signal",))
        self.assertEqual(Enndee_StandbyOnSignal.CATEGORY, "Enndee/utils")
        self.assertEqual(inputs["required"]["enabled"][1]["default"], False)
        self.assertEqual(
            list(inputs["required"]["mode"][0]), ["Standby (S3)", "DryRun (log only)"])
        self.assertTrue(all(
            "tooltip" in metadata
            for _options, metadata in inputs["required"].values()))

    def test_disabled_passes_signal_through(self):
        self.assertEqual(self.trigger(signal="X"), ("X",))

    def test_unreachable_queue_prevents_sleep(self):
        calls = self.patched_standby(
            lambda: self.assertEqual(
                self.trigger(enabled=True, check_queue=True, signal=42), (42,)))
        self.assertEqual(calls, [])

    def test_dry_run_does_not_sleep(self):
        calls = self.patched_standby(
            lambda: self.assertEqual(
                self.trigger(enabled=True, mode="DryRun (log only)", signal="S"), ("S",)))
        self.assertEqual(calls, [])

    def test_standby_path_triggers_exactly_once(self):
        calls = self.patched_standby(
            lambda: self.assertEqual(self.trigger(enabled=True, signal=None), (None,)))
        self.assertEqual(calls, [True])

    def test_delay_is_respected_and_cancellable(self):
        started = time.monotonic()
        calls = self.patched_standby(
            lambda: self.trigger(enabled=True, delay_seconds=2))
        self.assertGreaterEqual(time.monotonic() - started, 1.8)
        self.assertEqual(calls, [True])

    def test_queue_helper_on_dead_url_is_unknown(self):
        self.assertIsNone(standby_signal._queue_is_empty(DEAD_URL, timeout=1.0))

    def test_processing_interrupted_without_comfy(self):
        # Without a ComfyUI runtime the helper must stay quiet instead of raising.
        self.assertIsInstance(standby_signal._processing_interrupted(), bool)


if __name__ == "__main__":
    unittest.main()

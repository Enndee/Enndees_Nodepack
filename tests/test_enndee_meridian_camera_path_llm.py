import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

NODES = Path(__file__).resolve().parent.parent / "nodes"
if str(NODES) not in sys.path:
    sys.path.insert(0, str(NODES))

import enndee_meridian_camera_path_llm as llm  # noqa: E402
from enndee_meridian_camera_path_llm import Enndee_MeridianCameraPathLLM  # noqa: E402

PLAN_JSON = json.dumps({
    "notes": "front start, orbit to the back while lifting, return to front",
    "frames": 141,
    "keys": [
        {"t": 0,   "azimuth": 0.0,   "elevation": 0.0,  "distance": 1.0},
        {"t": 70,  "azimuth": 180.0, "elevation": 18.0, "distance": 1.08},
        {"t": 140, "azimuth": 360.0, "elevation": 0.0,  "distance": 1.0},
    ],
})


def _still_and_depth(size=64):
    """A synthetic still + a depth map image (brighter = a near subject block in the centre)."""
    image = torch.rand(1, size, size, 3)
    depth = torch.full((1, size, size, 3), 0.8)   # bright-ish background
    quarter = size // 4
    depth[:, quarter: size - quarter, quarter: size - quarter, :] = 0.2  # near subject (darker)
    return image, depth


class MeridianCameraPathLLMTests(unittest.TestCase):
    def test_system_prompt_covers_the_camera_model_and_schema(self):
        prompt = llm.SYSTEM_PROMPT
        for needle in ("azimuth", "elevation", "distance", "aim", "NO roll", "JSON",
                       "elevation", "frames", "worked", "depth map"):
            self.assertIn(needle.lower(), prompt.lower())
        # the strict output schema is spelled out
        self.assertIn('"keys"', prompt)
        self.assertIn('"notes"', prompt)

    def test_parse_plan_extracts_json_and_sanitizes(self):
        # fenced JSON wrapped in prose, with an out-of-range elevation and a far-past last key
        messy = ('Here you go:\n```json\n'
                 '{"notes":"x","frames":141,"keys":['
                 '{"t":0,"azimuth":0,"elevation":0,"distance":1},'
                 '{"t":200,"azimuth":180,"elevation":95,"distance":9}]}\n```')
        plan, error = llm._parse_plan(messy, 141)
        self.assertIsNone(error)
        self.assertEqual(plan["frames"], 141)
        self.assertEqual(plan["keys"][0]["t"], 0)          # starts at 0
        self.assertEqual(plan["keys"][-1]["t"], 140)       # ends at frames-1
        self.assertLessEqual(plan["keys"][-1]["elevation"], 70.0)   # clamped
        self.assertLessEqual(plan["keys"][-1]["distance"], 3.0)     # clamped

    def test_render_plan_aims_at_the_pivot_in_median_depth_units(self):
        plan = {"frames": 141, "keys": [
            {"t": 0, "azimuth": 0.0, "elevation": 0.0, "distance": 1.0},
            {"t": 140, "azimuth": 180.0, "elevation": 0.0, "distance": 1.0},
        ]}
        pivot = [0.2, 0.1, 2.0]
        depth_unit = 2.0
        keys = llm._render_plan(plan, pivot, 1.5, depth_unit)
        self.assertEqual([k["t"] for k in keys], [0, 140])
        for key in keys:
            self.assertEqual(key["look"], [0.1, 0.05, 1.0])   # pivot / depth_unit
            self.assertEqual(key["src"], key["t"])
        # azimuth 0 sits between the pivot and the origin; azimuth 180 is behind the pivot
        self.assertLess(keys[0]["pos"][2], pivot[2] / depth_unit)
        self.assertGreater(keys[1]["pos"][2], pivot[2] / depth_unit)

    def test_generate_end_to_end_with_a_mocked_llm(self):
        image, depth = _still_and_depth()
        node = Enndee_MeridianCameraPathLLM()
        with mock.patch.object(llm, "_call_llm", return_value=(PLAN_JSON, None)) as called:
            signal, plan, raw, system_prompt = node.generate(
                instruction="rotate 180 around the subject, lift up a bit, and rotate back",
                frames="141", provider="lmstudio", base_url="http://localhost:1234/v1",
                model="qwen2.5-vl", api_key="", fill_percent=40.0, temperature=0.5,
                max_tokens=2048, image=image, depth=depth,
                depth_convention="brighter = closer")
        self.assertTrue(called.called)                       # the LLM was actually invoked
        data = json.loads(signal)                            # a valid MERIDIAN_CAMERA_PATH document
        self.assertEqual(data["frames"], 141)
        path = data["path"]
        self.assertEqual(path[0]["t"], 0)
        self.assertEqual(path[-1]["t"], 140)
        self.assertGreaterEqual(len(path), 3)
        for key in path:
            self.assertEqual(len(key["pos"]), 3)
            self.assertEqual(len(key["look"]), 3)
            self.assertIsInstance(key["src"], int)
        self.assertEqual(json.loads(raw), json.loads(PLAN_JSON))
        self.assertIn("cinematographer", system_prompt)

    def test_the_signal_is_accepted_by_the_geometry_node(self):
        from enndee_meridian_geometry import _parse_custom_camera
        image, depth = _still_and_depth()
        node = Enndee_MeridianCameraPathLLM()
        with mock.patch.object(llm, "_call_llm", return_value=(PLAN_JSON, None)):
            signal, _p, _r, _s = node.generate(
                instruction="orbit", frames="141", provider="ollama",
                base_url="http://localhost:11434", model="llava", api_key="",
                fill_percent=40.0, temperature=0.5, max_tokens=2048,
                image=image, depth=depth, depth_convention="brighter = closer")
        parsed = _parse_custom_camera(signal)   # raises if the signal is malformed
        self.assertIsNotNone(parsed)

    def test_base_url_is_normalized_per_provider(self):
        norm = llm._normalize_base_url
        # the exact bug: provider=ollama with the LM Studio URL produced /v1/api/chat on the wrong port
        self.assertEqual(norm("ollama", "http://localhost:1234/v1"), "http://localhost:11434")
        self.assertEqual(norm("ollama", "http://localhost:11434/v1"), "http://localhost:11434")
        self.assertEqual(norm("ollama", ""), "http://localhost:11434")
        self.assertEqual(norm("ollama", "http://localhost:11434/"), "http://localhost:11434")
        self.assertEqual(norm("lmstudio", "http://localhost:11434"), "http://localhost:1234/v1")
        self.assertEqual(norm("lmstudio", ""), "http://localhost:1234/v1")
        # a pasted endpoint suffix is stripped
        self.assertEqual(norm("lmstudio", "http://localhost:1234/v1/chat/completions"),
                         "http://localhost:1234/v1")
        self.assertEqual(norm("ollama", "http://localhost:11434/api/chat"),
                         "http://localhost:11434")

    def test_a_connection_error_is_diagnosed(self):
        with mock.patch.object(llm, "_reachable",
                               side_effect=lambda url, timeout=2.0: "1234" in url):
            hint = llm._connection_hint("ollama", "http://localhost:11434/api/chat")
        self.assertIn("Could not reach", hint)
        self.assertIn("LM Studio IS reachable", hint)          # names the server that IS up
        self.assertIn("provider=lmstudio", hint)               # and how to fix it

    def test_a_missing_model_reports_clearly(self):
        node = Enndee_MeridianCameraPathLLM()
        signal, plan, raw, _sp = node.generate(
            instruction="orbit", frames="73", provider="lmstudio",
            base_url="http://localhost:1234/v1", model="",
            api_key="", fill_percent=40.0, temperature=0.5, max_tokens=256)
        self.assertIn("Meridian Camera Path LLM", signal)
        self.assertIn("No model name", signal)
        self.assertEqual(plan, "{}")


if __name__ == "__main__":
    unittest.main()

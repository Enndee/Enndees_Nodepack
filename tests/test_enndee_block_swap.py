"""Tests for the Enndee Block Swap node (LoRA-safe block offloading).

The suite runs against a tiny synthetic DiT on the real GPU (when one is
available) and asserts the three properties that matter:

* **LoRA safe** - a weight delta applied before the node survives it, and the
  output is identical to the un-swapped run.
* **Device placement** - after a forward pass only ``blocks_to_keep`` blocks are
  still on the GPU; every streamed block is back on the CPU.
* **Fail safe** - no CUDA, no block container or a double application must
  never change the model.
"""

import sys
import types
import unittest
from pathlib import Path

import torch

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR.parents[1]))          # the ComfyUI root (comfy.*)

import enndee_block_swap as bs  # noqa: E402


class Block(torch.nn.Module):
    """One transformer block with a single weight matrix."""

    def __init__(self, dim):
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return x + self.lin(x)


class TinyDiT(torch.nn.Module):
    """A miniature diffusion transformer: in_proj -> blocks -> out."""

    def __init__(self, dim=64, count=8):
        super().__init__()
        self.in_proj = torch.nn.Linear(dim, dim, bias=False)
        self.blocks = torch.nn.ModuleList([Block(dim) for _ in range(count)])
        self.out = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


class FakeModel:
    """Stand-in for a ComfyUI MODEL wrapper (``clone()`` + ``model.diffusion_model``)."""

    def __init__(self, diffusion):
        self.model = types.SimpleNamespace(diffusion_model=diffusion)

    def clone(self):
        return FakeModel(self.model.diffusion_model)


def _devices(module):
    """The set of devices used by a module's parameters."""
    return {parameter.data.device.type for parameter in module.parameters(recurse=True)}


class BlockSwapTests(unittest.TestCase):
    """The node must stream blocks without changing the result."""

    def setUp(self):
        torch.manual_seed(0)
        self.has_cuda = torch.cuda.is_available()
        self.dim = 64
        self.model = TinyDiT(self.dim, 8)
        self.node = bs.EnndeeBlockSwap()

    def _input(self, device):
        return torch.randn(2, 4, self.dim, device=device)

    def test_registered_as_a_model_node(self):
        self.assertIn("Enndee_BlockSwap", bs.NODE_CLASS_MAPPINGS)
        spec = bs.EnndeeBlockSwap.INPUT_TYPES()
        self.assertEqual(spec["required"]["model"], ("MODEL",))
        self.assertIn("blocks_to_keep", spec["required"])
        self.assertEqual(bs.EnndeeBlockSwap.RETURN_TYPES, ("MODEL",))

    def test_block_container_is_detected(self):
        name, container = bs._pick_block_container(self.model)
        self.assertEqual(name, "blocks")
        self.assertEqual(len(container), 8)

    def test_named_container_wins_over_the_heuristic(self):
        class Weird(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(8) for _ in range(4)])
                self.other = torch.nn.ModuleList([Block(8) for _ in range(6)])

        name, container = bs._pick_block_container(Weird())
        self.assertEqual(name, "blocks")
        self.assertEqual(len(container), 4)

    def test_heuristic_fallback_finds_an_unnamed_container(self):
        class Odd(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(*[Block(8) for _ in range(5)])

        name, container = bs._pick_block_container(Odd())
        self.assertEqual(name, "layers")
        self.assertEqual(len(container), 5)

    def test_no_container_passes_the_model_through(self):
        class Flat(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(8, 8)

        result, = self.node.apply(FakeModel(Flat()), blocks_to_keep=1, verbose=False)
        self.assertIsNone(getattr(result.model.diffusion_model, "_enndee_blockswap_plan", None))

    def test_second_application_reuses_the_plan(self):
        model = FakeModel(self.model)
        self.node.apply(model, blocks_to_keep=2, verbose=False)
        plan = self.model._enndee_blockswap_plan
        handles = len(plan["handles"])
        self.node.apply(model, blocks_to_keep=2, verbose=False)
        self.assertEqual(len(self.model._enndee_blockswap_plan["handles"]), handles)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class BlockSwapGpuTests(unittest.TestCase):
    """On a real GPU: identical numbers, LoRA kept, blocks actually offloaded."""

    def setUp(self):
        torch.manual_seed(1)
        self.dim = 64
        self.model = TinyDiT(self.dim, 8).to("cuda", dtype=torch.float32).eval()
        self.x = torch.randn(2, 4, self.dim, device="cuda")
        self.node = bs.EnndeeBlockSwap()

    def test_streaming_matches_the_unswapped_run_and_keeps_the_lora(self):
        with torch.no_grad():
            plain = self.model(self.x)

        # a "LoRA": add a delta to one block's weight, like a patch would
        delta = torch.randn_like(self.model.blocks[5].lin.weight)
        with torch.no_grad():
            self.model.blocks[5].lin.weight.add_(delta)
            patched = self.model(self.x)
        self.assertFalse(torch.allclose(plain, patched),
                         "the injected delta has to change the output")

        with torch.no_grad():
            weight_before = self.model.blocks[5].lin.weight.detach().clone()

        self.node.apply(FakeModel(self.model), blocks_to_keep=2, verbose=False)
        with torch.no_grad():
            swapped = self.model(self.x)

        self.assertTrue(torch.equal(patched, swapped),
                        "the swapped run must be identical to the un-swapped run")
        weight_after = self.model.blocks[5].lin.weight.detach().to(weight_before.device)
        self.assertTrue(torch.equal(weight_before, weight_after),
                        "the node must not modify any weight")

    def test_only_kept_blocks_stay_on_the_gpu(self):
        self.node.apply(FakeModel(self.model), blocks_to_keep=2, verbose=False)
        with torch.no_grad():
            self.model(self.x)
        for index in range(8):
            devices = _devices(self.model.blocks[index])
            if index < 2:
                self.assertEqual(devices, {"cuda"}, f"block {index} should stay resident")
            else:
                self.assertEqual(devices, {"cpu"}, f"block {index} should be offloaded")
        # parts outside the block container are never touched
        self.assertEqual(_devices(self.model.in_proj), {"cuda"})
        self.assertEqual(_devices(self.model.out), {"cuda"})

    def test_a_second_forward_reloads_the_blocks_correctly(self):
        self.node.apply(FakeModel(self.model), blocks_to_keep=1, verbose=False)
        with torch.no_grad():
            first = self.model(self.x)
            second = self.model(self.x)
        self.assertTrue(torch.equal(first, second),
                        "reloading a block must reproduce its result exactly")

    def test_blocks_to_keep_zero_streams_everything(self):
        self.node.apply(FakeModel(self.model), blocks_to_keep=0, verbose=False)
        with torch.no_grad():
            out = self.model(self.x)
        self.assertEqual(_devices(self.model.blocks[0]), {"cpu"})
        self.assertEqual(_devices(self.model.blocks[7]), {"cpu"})
        self.assertTrue(torch.isfinite(out).all())

    def test_plan_reports_the_expected_geometry(self):
        self.node.apply(FakeModel(self.model), blocks_to_keep=3, verbose=False)
        plan = self.model._enndee_blockswap_plan
        self.assertEqual(plan["total"], 8)
        self.assertEqual(plan["keep"], 3)
        self.assertEqual(plan["streamed"], [3, 4, 5, 6, 7])
        self.assertEqual(len(plan["handles"]), 2 * 5)


if __name__ == "__main__":
    unittest.main()


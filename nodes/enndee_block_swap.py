"""
Block Swap (Enndee) - stream a diffusion model's transformer blocks through VRAM.

Why this node exists
--------------------
The generic "UniBlockSwap" node frees blocks to the ``meta`` device and then
*bypasses* ComfyUI's LoRA path for them::

    def _skip_swap_patch(key, ...):
        if _is_swap_key(key):
            return                      # LoRA never reaches the swapped block
    def _purge_swap_from_backup(p):
        for k in [k for k in p.backup if _is_swap_key(k)]:
            p.backup.pop(k, None)       # ...and the original weight is dropped

Every LoRA (``LoraLoaderModelOnly``, ``MiniMaxH3TurboLoRA``, ...) is applied
through exactly the code that gets skipped, so a swapped model silently loses
its LoRAs - the run completes and the video comes out broken/black.

This node takes the opposite approach: **it never touches the patcher**. All
LoRA/patch bookkeeping (``patch_weight_to_device``, ``backup``, ``_load_list``,
``unpatch_model``) runs untouched, so LoRA deltas land in the weights exactly
as ComfyUI intends. Only afterwards does this node move each transformer
block's already-patched tensors between GPU and CPU around its forward call::

    UNETLoader -> LoraLoaderModelOnly -> [Block Swap (Enndee)] -> sampler

Consequences of that design:

* **LoRA safe by construction** - it moves the final, patched weights; there is
  no second copy that can get out of sync.
* **Placement independent** - the hooks live on the nn.Module, which every
  ``ModelPatcher.clone()`` shares, so before or after a LoRA loader both work.
* **No dependency on the minimax-h3 pack** - a plain ComfyUI ``MODEL`` node, so
  it drops into the stock ``UNETLoader``/``SamplerCustomAdvanced`` pipeline used
  by ``Meridian_Splatting_Benchmark.json``.
* **Parameter identity preserved** - the move writes ``param.data`` in place
  instead of replacing ``nn.Parameter`` objects, so ComfyUI's ``backup``
  references stay valid.

How it saves VRAM
-----------------
A 50-block MiniMax H3 DiT is ~18.5 GB in bf16 (~371 MB/block). With
``blocks_to_keep = 2`` only two blocks plus the block currently executing live
on the GPU (~1.1 GB instead of ~18.5 GB). Every other block is copied H2D
before its forward and back to the CPU right after it. The trade-off is PCIe
traffic, not compute: raise ``blocks_to_keep`` as far as your free VRAM allows
to cut the number of copies. Nothing is quantized, cast or recomputed, so the
result is the same as an un-swapped run.
"""

from __future__ import annotations

import torch

#: attribute names that hold a diffusion transformer's repeated blocks
_BLOCK_CONTAINER_NAMES = (
    "blocks", "block", "transformer_blocks", "double_blocks", "single_blocks",
    "encoder_layers", "decoder_layers", "layers", "h", "resblocks",
)

#: a container must hold at least this many parameter-bearing children to count
_MIN_BLOCKS = 4



def _torch_device():
    """The device the model computes on (ComfyUI's pick, else CUDA/CPU)."""
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:  # noqa: BLE001 - running outside ComfyUI (tests)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _offload_device():
    """Where swapped blocks wait - ComfyUI's UNet offload device, else CPU."""
    try:
        import comfy.model_management as mm

        return mm.unet_offload_device()
    except Exception:  # noqa: BLE001
        return torch.device("cpu")


def _diffusion_module(model):
    """The ``nn.Module`` that holds the blocks (a MODEL or a ModelPatcher)."""
    patcher = getattr(model, "model", model)
    diffusion = getattr(patcher, "diffusion_model", None)
    if isinstance(diffusion, torch.nn.Module):
        return diffusion
    if isinstance(patcher, torch.nn.Module):
        return patcher
    return None


def _count_modules(container):
    """How many children of ``container`` carry parameters of their own."""
    count = 0
    for child in container:
        if isinstance(child, torch.nn.Module) and any(
            True for _ in child.parameters(recurse=True)
        ):
            count += 1
    return count


def _pick_block_container(diffusion):
    """Find the repeated-block container inside a diffusion transformer.

    Named containers win (``blocks`` first, which is what the MiniMax H3 DiT
    uses); otherwise the direct child that is a list-like module with the most
    parameter-bearing entries is taken. Returns ``(name, module)`` or
    ``(None, None)``.
    """
    for name in _BLOCK_CONTAINER_NAMES:
        candidate = getattr(diffusion, name, None)
        if isinstance(candidate, (torch.nn.ModuleList, torch.nn.Sequential)):
            if _count_modules(candidate) >= _MIN_BLOCKS:
                return name, candidate

    best_name, best_module, best_count = None, None, 0
    for name, child in diffusion.named_children():
        if not isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
            continue
        count = _count_modules(child)
        if count > best_count:
            best_name, best_module, best_count = name, child, count
    if best_count >= _MIN_BLOCKS:
        return best_name, best_module
    return None, None


def _module_bytes(module):
    """Bytes held by a module's parameters and buffers."""
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        total += tensor.numel() * tensor.element_size()
    return total


def _move_module(module, device):
    """Move a module's tensors to ``device`` in place, keeping Parameter identity.

    ``module.to()`` is deliberately avoided: it can replace ``nn.Parameter``
    objects (``torch.__future__.set_overwrite_module_params_on_conversion``),
    which would invalidate the references ComfyUI keeps in ``backup``. Writing
    ``param.data`` on the existing object is always safe.
    """
    moved = 0
    with torch.no_grad():
        for param in module.parameters(recurse=True):
            if param.data.device != device:
                param.data = param.data.to(device)
                moved += 1
        for buffer in module.buffers(recurse=True):
            if buffer.device != device:
                buffer.data = buffer.data.to(device)
                moved += 1
    return moved


class EnndeeBlockSwap:
    """Stream a diffusion model's transformer blocks between CPU and GPU.

    Works with any ComfyUI ``MODEL`` (``UNETLoader``, ``CheckpointLoaderSimple``,
    ...) and leaves LoRA/patch handling entirely to ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_keep": ("INT", {
                    "default": 2, "min": 0, "max": 512, "step": 1,
                    "tooltip": (
                        "How many leading blocks stay on the GPU permanently. "
                        "Every other block is copied to the GPU for its forward "
                        "pass and back to the CPU right after. Higher = faster "
                        "(fewer copies), lower = less VRAM. The currently "
                        "executing block is always resident, so the real "
                        "footprint is blocks_to_keep + 1."),
                }),
                "verbose": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Log the detected block layout and the resulting "
                               "VRAM footprint.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Enndee/VRAM"
    DESCRIPTION = (
        "Block-swap VRAM saver for any ComfyUI diffusion model. Streams each "
        "transformer block from CPU RAM to the GPU around its forward pass, so a "
        "50-block DiT runs in a fraction of its weight VRAM. LoRA safe: it never "
        "touches ComfyUI's patch/backup bookkeeping, so LoRA deltas are applied "
        "exactly as usual (unlike UniBlockSwap, which skips and purges them)."
    )

    def apply(self, model, blocks_to_keep=2, verbose=True):
        patcher = model.clone() if hasattr(model, "clone") else model
        diffusion = _diffusion_module(patcher)
        if diffusion is None:
            if verbose:
                print("[Enndee BlockSwap] no diffusion module found - "
                      "passing the model through")
            return (patcher,)

        compute = _torch_device()
        offload = _offload_device()
        if torch.device(compute).type != "cuda":
            if verbose:
                print(f"[Enndee BlockSwap] compute device is {compute} - nothing to swap")
            return (patcher,)
        if torch.device(offload) == torch.device(compute):
            if verbose:
                print("[Enndee BlockSwap] offload device equals the compute device "
                      "(GPU-only mode) - nothing to swap")
            return (patcher,)

        name, container = _pick_block_container(diffusion)
        if container is None:
            if verbose:
                print("[Enndee BlockSwap] no transformer block container found in "
                      f"{type(diffusion).__name__} - passing the model through")
            return (patcher,)

        blocks = list(container)
        total = len(blocks)
        keep = max(0, min(int(blocks_to_keep), total - 1))
        streamed = list(range(keep, total))
        if not streamed:
            if verbose:
                print(f"[Enndee BlockSwap] blocks_to_keep={keep} covers all {total} "
                      "blocks - nothing to swap")
            return (patcher,)

        # One plan per module instance: a second application must not stack hooks.
        if getattr(diffusion, "_enndee_blockswap_plan", None) is not None:
            if verbose:
                print(f"[Enndee BlockSwap] '{name}' is already swapped - reusing the plan")
            return (patcher,)

        per_block = _module_bytes(blocks[0]) if blocks else 0
        plan = {
            "total": total,
            "keep": keep,
            "streamed": streamed,
            "blocks": blocks,
            "compute": compute,
            "offload": offload,
            "primed": False,
            "handles": [],
        }
        diffusion._enndee_blockswap_plan = plan

        def _prime(current):
            """Push every streamed block except the current one to the CPU.

            ComfyUI loads the whole model onto the GPU before sampling, so
            without this the very first forward pass would still peak at the
            full model size.
            """
            for index in plan["streamed"]:
                if index != current:
                    _move_module(blocks[index], plan["offload"])

        def _pre_hook(module, args, index):
            # Always make sure the block that is about to run is resident: the
            # setup below may already have evicted it, and ComfyUI may have left
            # it on the CPU.
            _move_module(module, plan["compute"])
            if not plan["primed"]:
                plan["primed"] = True
                _prime(index)
            return None

        def _post_hook(module, args, output, index):
            _move_module(blocks[index], plan["offload"])
            return None

        for index in streamed:
            plan["handles"].append(blocks[index].register_forward_pre_hook(
                lambda module, args, index=index: _pre_hook(module, args, index)))
            plan["handles"].append(blocks[index].register_forward_hook(
                lambda module, args, output, index=index:
                _post_hook(module, args, output, index)))

        # Already on the GPU (re-queued workflow): free the streamed blocks now.
        first_param = next(blocks[keep].parameters(recurse=True), None)
        if first_param is not None and first_param.data.device.type == "cuda":
            _prime(None)

        if verbose:
            resident_mb = (keep + 1) * per_block / 2 ** 20
            full_mb = total * per_block / 2 ** 20
            print(f"[Enndee BlockSwap] '{name}': {total} blocks, {keep} kept + 1 "
                  f"active resident, {len(streamed)} streamed "
                  f"({per_block / 2 ** 20:.0f} MB/block) - ~{resident_mb:.0f} MB "
                  f"instead of ~{full_mb:.0f} MB on {compute}")
        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "Enndee_BlockSwap": EnndeeBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_BlockSwap": "Block Swap (Enndee)",
}

__all__ = [
    "EnndeeBlockSwap",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]



"""LoRA heads and the helpers that apply merged LoRA deltas to a model.

A *head* is what one task-tuned model contributes to merging: its LoRA matrices and
its classifier. On disk it is the pair of safetensors files that ``save_head`` writes:

- ``<country>_lora.safetensors`` with ``w_a_{i:03d}`` / ``w_b_{i:03d}`` for LoRA target i
- ``<country>_fc.safetensors`` with ``fc_{in}in_{out}out`` (weight) and ``head_bias``

In memory a head is a dict with ``"A"`` and ``"B"`` (lists over targets), or ``"delta"``
(a list of full delta matrices, used for merge results), plus ``"fc"``
(``{"weight": W, "bias": b or None}``).

LoRA targets are ordered the way the LoRA wrappers create them: block 0 q, block 0 v,
block 1 q, block 1 v, ... ``lora_targets`` checks this order against the wrapper's
``w_As`` list, so saved key i is always applied to the projection it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.nn.utils import parametrize

from .lora_vit import _LoRA_qkv_timm, _LoRALayer

Head = dict


@dataclass
class LoRATarget:
    """One LoRA-adapted projection.

    Attributes:
        linear: The frozen base projection the LoRA delta is added to.
        rows: Output rows of ``linear`` that this target covers (fused qkv), or None.
        lora: The LoRA module that owns the A/B pair for this target.
        scale: LoRA scaling factor alpha / r.
    """

    linear: nn.Linear
    rows: slice | None
    lora: nn.Module
    scale: float


def find_lora_wrapper(model: nn.Module) -> nn.Module:
    """Return the LoRA wrapper module (the one that holds ``w_As`` and ``w_Bs``)."""
    for module in model.modules():
        if hasattr(module, "w_As") and hasattr(module, "w_Bs"):
            return module
    raise ValueError("Model has no LoRA wrapper with w_As / w_Bs")


def lora_targets(model: nn.Module) -> list[LoRATarget]:
    """List the LoRA targets of a LoRA-wrapped model in saved-key order.

    Raises:
        RuntimeError: If the module order does not match the wrapper's ``w_As`` order.
    """
    targets: list[LoRATarget] = []
    for module in model.modules():
        if isinstance(module, _LoRALayer):
            targets.append(LoRATarget(module.w, None, module, module.alpha / module.r))
        elif isinstance(module, _LoRA_qkv_timm):
            d, scale = module.dim, module.alpha / module.r
            targets.append(LoRATarget(module.qkv, slice(0, d), module, scale))
            targets.append(LoRATarget(module.qkv, slice(2 * d, 3 * d), module, scale))

    wrapper = find_lora_wrapper(model)
    if len(targets) != len(wrapper.w_As):
        raise RuntimeError(f"Found {len(targets)} LoRA targets but {len(wrapper.w_As)} A matrices")
    for i, t in enumerate(targets):
        if isinstance(t.lora, _LoRALayer):
            a = t.lora.w_a
        else:
            a = t.lora.linear_a_q if i % 2 == 0 else t.lora.linear_a_v
        if a is not wrapper.w_As[i]:
            raise RuntimeError(f"LoRA target {i} does not match saved key w_a_{i:03d}")
    return targets


def classifier_of(model: nn.Module) -> nn.Linear:
    """Return the classification head of a LoRA-wrapped backbone."""
    if isinstance(model, nn.Sequential) and isinstance(model[-1], nn.Linear):
        return model[-1]  # SoftCon: Sequential(LoRA_ViT_timm, classifier)
    inner = getattr(model, "lora_vit", model)
    for name in ("fc", "head"):
        head = getattr(inner, name, None)
        if isinstance(head, nn.Linear):
            return head
    raise ValueError("Could not find a linear classification head")


def save_head(model: nn.Module, lora_path: str | Path, fc_path: str | Path) -> None:
    """Save the LoRA matrices and classifier of a trained model as a head."""
    wrapper = find_lora_wrapper(model)
    tensors = {}
    for i, (a, b) in enumerate(zip(wrapper.w_As, wrapper.w_Bs)):
        tensors[f"w_a_{i:03d}"] = a.weight.detach().cpu().contiguous()
        tensors[f"w_b_{i:03d}"] = b.weight.detach().cpu().contiguous()
    save_file(tensors, str(lora_path))

    clf = classifier_of(model)
    out_f, in_f = clf.weight.shape
    fc = {f"fc_{in_f}in_{out_f}out": clf.weight.detach().cpu().contiguous()}
    if clf.bias is not None:
        fc["head_bias"] = clf.bias.detach().cpu().contiguous()
    save_file(fc, str(fc_path))


def load_head(lora_path: str | Path, fc_path: str | Path) -> Head:
    """Load a head saved by ``save_head`` (or by ``save_lora_parameters``)."""
    sd = load_file(str(lora_path))
    n = sum(1 for k in sd if k.startswith("w_a_"))
    head: Head = {
        "A": [sd[f"w_a_{i:03d}"] for i in range(n)],
        "B": [sd[f"w_b_{i:03d}"] for i in range(n)],
    }
    fc_sd = load_file(str(fc_path))
    weight = next(v for k, v in fc_sd.items() if k.startswith("fc_"))
    head["fc"] = {"weight": weight, "bias": fc_sd.get("head_bias")}
    return head


def head_deltas(head: Head, targets: list[LoRATarget]) -> list[Tensor]:
    """Return the full delta matrix (scale * B @ A) of a head for every target."""
    if "delta" in head:
        return list(head["delta"])
    return [t.scale * (b @ a) for t, a, b in zip(targets, head["A"], head["B"])]


def reset_lora_branch(model: nn.Module) -> None:
    """Zero every LoRA B matrix and freeze all parameters.

    After this the LoRA branch adds nothing, so merged deltas can be applied to the
    base projections through ``attach_deltas`` without double counting.
    """
    wrapper = find_lora_wrapper(model)
    for b in wrapper.w_Bs:
        nn.init.zeros_(b.weight)
    for p in model.parameters():
        p.requires_grad_(False)


class _AddDeltas(nn.Module):
    """Parametrization: W -> W + delta for each target that lives in this weight."""

    def __init__(self, merger: nn.Module, entries: list[tuple[int, slice | None]]):
        super().__init__()
        self.merger = merger
        self.entries = entries

    def forward(self, weight: Tensor) -> Tensor:
        for i, rows in self.entries:
            delta = self.merger.delta(i).to(weight.dtype)
            if rows is None:
                weight = weight + delta
            else:
                index = torch.arange(rows.start, rows.stop, device=weight.device)
                weight = weight.index_add(0, index, delta)
        return weight


def _group_by_linear(targets: list[LoRATarget]) -> list[tuple[nn.Linear, list]]:
    groups: dict[int, tuple[nn.Linear, list]] = {}
    for i, t in enumerate(targets):
        groups.setdefault(id(t.linear), (t.linear, []))[1].append((i, t.rows))
    return list(groups.values())


def attach_deltas(targets: list[LoRATarget], merger: nn.Module) -> None:
    """Add ``merger.delta(i)`` to the weight of every target i (differentiably)."""
    for linear, entries in _group_by_linear(targets):
        parametrize.register_parametrization(linear, "weight", _AddDeltas(merger, entries))


def detach_deltas(targets: list[LoRATarget]) -> None:
    """Remove the parametrizations added by ``attach_deltas`` (restores base weights)."""
    for linear, _ in _group_by_linear(targets):
        if parametrize.is_parametrized(linear, "weight"):
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)

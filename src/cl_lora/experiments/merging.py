"""LoRA merging strategies for continual learning.

Three strategies merge the heads of several task-tuned models into one model:

- ZipLoRA: one learnable merger vector per head and input column of every delta,
  trained with the task loss plus a penalty on the overlap of the merger vectors
  (Shah et al. 2023, https://arxiv.org/abs/2311.13600; here with a supervised task
  loss instead of reconstructing each LoRA's outputs).
- LoRA Soups: one learnable mixing weight per head and target, softmax-normalised,
  applied to the full deltas (CAT-style soups).
- LoRAHub: one scalar weight per head, shared by all targets, applied to A and B
  separately and searched without gradients with Nevergrad
  (Huang et al. 2023, https://arxiv.org/abs/2307.13269).

Each strategy also builds a classifier: a weighted average of the heads' classifiers,
then trained on the merge data.

The functions take a fresh LoRA-wrapped model, apply the merged deltas to its base
projections (see ``cl_lora.models.lora_heads``) and return the merged head together
with the model, ready for evaluation.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.lora_heads import (
    Head,
    LoRATarget,
    attach_deltas,
    classifier_of,
    head_deltas,
    lora_targets,
    reset_lora_branch,
)

log = logging.getLogger(__name__)

MERGE_TYPES = ("ZipLoRA", "LoRASoups", "LoRAHub")


# ---------------------------------------------------------------------------
# Mergers: each holds the heads and returns the merged delta of target i
# ---------------------------------------------------------------------------
class ZipMerger(nn.Module):
    """ZipLoRA: delta_i = sum_h m_{h,i} (column-wise) * Delta_{h,i}."""

    def __init__(self, deltas: list[list[Tensor]], init: float = 0.5):
        super().__init__()
        self.n_targets = len(deltas[0])
        for i in range(self.n_targets):
            self.register_buffer(f"d{i}", torch.stack([d[i] for d in deltas]))  # [H, out, in]
        self.m = nn.ParameterList(
            nn.Parameter(torch.full((len(deltas), deltas[0][i].shape[1]), init))
            for i in range(self.n_targets)
        )

    def delta(self, i: int) -> Tensor:
        return (getattr(self, f"d{i}") * self.m[i].unsqueeze(1)).sum(0)

    def overlap(self) -> Tensor:
        """Mean |cosine similarity| between the merger vectors of different heads."""
        terms = [
            F.cosine_similarity(m[a], m[b], dim=0).abs()
            for m in self.m
            for a, b in itertools.combinations(range(m.shape[0]), 2)
        ]
        return torch.stack(terms).mean()


class SoupMerger(nn.Module):
    """LoRA Soups: delta_i = sum_h softmax(alpha_i)_h * Delta_{h,i}."""

    def __init__(self, deltas: list[list[Tensor]]):
        super().__init__()
        self.n_targets = len(deltas[0])
        for i in range(self.n_targets):
            self.register_buffer(f"d{i}", torch.stack([d[i] for d in deltas]))
        self.alpha = nn.Parameter(torch.zeros(self.n_targets, len(deltas)))

    def delta(self, i: int) -> Tensor:
        weights = torch.softmax(self.alpha[i], dim=0)
        return torch.einsum("h,hoi->oi", weights, getattr(self, f"d{i}"))


class HubMerger(nn.Module):
    """LoRAHub: delta_i = scale_i * (sum_h w_h B_{h,i}) @ (sum_h w_h A_{h,i})."""

    def __init__(self, heads: list[Head], targets: list[LoRATarget]):
        super().__init__()
        self.n_targets = len(targets)
        self.scales = [t.scale for t in targets]
        for i in range(self.n_targets):
            self.register_buffer(f"a{i}", torch.stack([h["A"][i] for h in heads]))  # [H, r, in]
            self.register_buffer(f"b{i}", torch.stack([h["B"][i] for h in heads]))  # [H, out, r]
        self.register_buffer("w", torch.zeros(len(heads)))

    def merged_ab(self, i: int) -> tuple[Tensor, Tensor]:
        a = torch.einsum("h,hri->ri", self.w, getattr(self, f"a{i}"))
        b = torch.einsum("h,hor->or", self.w, getattr(self, f"b{i}"))
        return a, b

    def delta(self, i: int) -> Tensor:
        a, b = self.merged_ab(i)
        return self.scales[i] * (b @ a)


# ---------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------
def _init_classifier(clf: nn.Linear, heads: list[Head], weights: list[float]) -> None:
    """Set the classifier to the weighted average of the heads' classifiers."""
    with torch.no_grad():
        clf.weight.copy_(sum(w * h["fc"]["weight"].to(clf.weight) for w, h in zip(weights, heads)))
        if clf.bias is not None:
            biases = [h["fc"]["bias"] for h in heads]
            if all(b is not None for b in biases):
                clf.bias.copy_(sum(w * b.to(clf.bias) for w, b in zip(weights, biases)))
            else:
                clf.bias.zero_()
    for p in clf.parameters():
        p.requires_grad_(True)


def _batches(loaders: list[DataLoader]) -> Iterable[list[tuple[Tensor, Tensor]]]:
    """Yield one batch from every loader per step (stops at the shortest loader)."""
    yield from zip(*loaders)


def _train(model, params, loaders, epochs, lr, device, extra_loss=None) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params, lr=lr)
    model.train()
    for epoch in range(epochs):
        total, steps = 0.0, 0
        for batches in _batches(loaders):
            loss = sum(criterion(model(x.to(device)), y.to(device).float()) for x, y in batches)
            if extra_loss is not None:
                loss = loss + extra_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + loss.item(), steps + 1
        log.info(f"  merge epoch {epoch + 1}/{epochs}: loss {total / max(steps, 1):.4f}")


@torch.no_grad()
def _mean_loss(model, loader, device) -> float:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    losses = [criterion(model(x.to(device)), y.to(device).float()).item() for x, y in loader]
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def merge_heads(
    merge_type: str,
    model: nn.Module,
    heads: list[Head],
    classifier_weights: list[float],
    loaders: list[DataLoader],
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    zip_cos_weight: float = 1.0,
    zip_init: float = 0.5,
    hub_loader: DataLoader | None = None,
    hub_budget: int = 40,
    hub_l1: float = 0.05,
) -> tuple[Head, nn.Module]:
    """Merge several heads into ``model`` and return (merged head, model).

    Args:
        merge_type: One of ``MERGE_TYPES``.
        model: A freshly loaded LoRA-wrapped model (its own LoRA branch is reset).
        heads: Heads to merge, oldest first.
        classifier_weights: Weights for averaging the heads' classifiers.
        loaders: Training loaders; one batch of each is used per step.
        epochs: Training epochs for the merge parameters and the classifier.
        lr: Learning rate.
        device: Device to train on.
        zip_cos_weight: ZipLoRA weight of the merger-vector overlap penalty.
        zip_init: ZipLoRA initial value of the merger vectors.
        hub_loader: LoRAHub data for the gradient-free search (defaults to loaders[0]).
        hub_budget: LoRAHub number of Nevergrad evaluations.
        hub_l1: LoRAHub L1 penalty on the head weights.
    """
    if merge_type not in MERGE_TYPES:
        raise ValueError(f"Unknown merge type {merge_type}; use one of {MERGE_TYPES}")
    model.to(device)
    reset_lora_branch(model)
    targets = lora_targets(model)
    clf = classifier_of(model)
    _init_classifier(clf, heads, classifier_weights)

    if merge_type == "LoRAHub":
        merger = HubMerger(heads, targets).to(device)
    else:
        deltas = [[d.to(device) for d in head_deltas(h, targets)] for h in heads]
        merger = (
            ZipMerger(deltas, zip_init) if merge_type == "ZipLoRA" else SoupMerger(deltas)
        ).to(device)
    attach_deltas(targets, merger)

    if merge_type == "LoRAHub":
        import nevergrad as ng

        data = hub_loader if hub_loader is not None else loaders[0]

        def objective(weights: np.ndarray) -> float:
            merger.w.copy_(torch.as_tensor(weights, dtype=merger.w.dtype, device=device))
            return _mean_loss(model, data, device) + hub_l1 * float(np.mean(np.abs(weights)))

        parametrization = ng.p.Array(init=np.zeros(len(heads)), lower=-1.5, upper=1.5)
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=hub_budget)
        best = optimizer.minimize(objective).value
        merger.w.copy_(torch.as_tensor(best, dtype=merger.w.dtype, device=device))
        log.info(f"  LoRAHub weights: {np.round(best, 3).tolist()}")
        _train(model, list(clf.parameters()), loaders, epochs, lr, device)
        merged_lora = [merger.merged_ab(i) for i in range(len(targets))]
        merged: Head = {
            "A": [a.detach().cpu() for a, _ in merged_lora],
            "B": [b.detach().cpu() for _, b in merged_lora],
        }
    else:
        extra = (lambda: zip_cos_weight * merger.overlap()) if merge_type == "ZipLoRA" else None
        params = list(merger.parameters()) + list(clf.parameters())
        _train(model, params, loaders, epochs, lr, device, extra_loss=extra)
        merged = {"delta": [merger.delta(i).detach().cpu() for i in range(len(targets))]}

    merged["fc"] = {
        "weight": clf.weight.detach().cpu(),
        "bias": None if clf.bias is None else clf.bias.detach().cpu(),
    }
    model.eval()
    return merged, model

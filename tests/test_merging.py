"""Tests for LoRA heads and merging, on tiny random models (no pretrained weights)."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cl_lora.experiments.merging import ZipMerger, merge_heads
from cl_lora.models.lora_heads import (
    attach_deltas,
    classifier_of,
    head_deltas,
    load_head,
    lora_targets,
    reset_lora_branch,
    save_head,
)
from cl_lora.models.lora_vit import LoRA_SViT, LoRA_ViT_timm
from cl_lora.models.video_vit import VisionTransformer
from cl_lora.training import forgetting_and_intransigence

N_CLASSES = 3


def tiny_svit():
    torch.manual_seed(0)
    return VisionTransformer(
        num_frames=4,
        t_patch_size=2,
        img_size=16,
        patch_size=8,
        in_chans=1,
        num_classes=N_CLASSES,
        embed_dim=16,
        depth=2,
        num_heads=2,
        sep_pos_embed=True,
        drop_path_rate=0.0,
    )


def tiny_timm():
    from timm.models.vision_transformer import VisionTransformer as TimmViT

    torch.manual_seed(0)
    return TimmViT(
        img_size=16,
        patch_size=8,
        in_chans=3,
        num_classes=N_CLASSES,
        embed_dim=16,
        depth=2,
        num_heads=2,
    )


BUILDERS = {
    "svit": (
        lambda base: LoRA_SViT(base, r=2, alpha=4, num_classes=N_CLASSES),
        tiny_svit,
        (4, 16, 16),
    ),
    "timm": (
        lambda base: LoRA_ViT_timm(base, r=2, alpha=4, num_classes=N_CLASSES),
        tiny_timm,
        (3, 16, 16),
    ),
}


def trained_like(kind, base, seed):
    """A LoRA model whose A, B and classifier are random (as if trained)."""
    wrap, _, _ = BUILDERS[kind]
    model = wrap(copy.deepcopy(base))
    g = torch.Generator().manual_seed(seed)
    wrapper = model
    with torch.no_grad():
        for m in list(wrapper.w_As) + list(wrapper.w_Bs):
            m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
        clf = classifier_of(model)
        clf.weight.copy_(torch.randn(clf.weight.shape, generator=g))
        clf.bias.copy_(torch.randn(clf.bias.shape, generator=g))
    return model.eval()


class FixedMerger(nn.Module):
    def __init__(self, deltas):
        super().__init__()
        self.deltas = deltas

    def delta(self, i):
        return self.deltas[i]


@pytest.mark.parametrize("kind", ["svit", "timm"])
def test_target_order_matches_saved_keys(kind):
    wrap, make_base, _ = BUILDERS[kind]
    model = wrap(make_base())
    targets = lora_targets(model)
    assert len(targets) == 4  # 2 blocks x (q, v)
    if kind == "svit":
        blocks = model.lora_vit.blocks
        assert targets[0].linear is blocks[0].attn.q.w
        assert targets[1].linear is blocks[0].attn.v.w
        assert targets[2].linear is blocks[1].attn.q.w
    else:
        assert targets[0].rows == slice(0, 16) and targets[1].rows == slice(32, 48)


@pytest.mark.parametrize("kind", ["svit", "timm"])
def test_head_deltas_reproduce_the_trained_model(kind, tmp_path):
    """Saving a head and applying its deltas to a fresh model gives the same outputs."""
    wrap, make_base, shape = BUILDERS[kind]
    base = make_base()
    trained = trained_like(kind, base, seed=1)
    save_head(trained, tmp_path / "lora.safetensors", tmp_path / "fc.safetensors")
    head = load_head(tmp_path / "lora.safetensors", tmp_path / "fc.safetensors")

    fresh = wrap(copy.deepcopy(base)).eval()
    reset_lora_branch(fresh)
    targets = lora_targets(fresh)
    attach_deltas(targets, FixedMerger(head_deltas(head, targets)))
    clf = classifier_of(fresh)
    with torch.no_grad():
        clf.weight.copy_(head["fc"]["weight"])
        clf.bias.copy_(head["fc"]["bias"])

    x = torch.randn(2, *shape)
    with torch.no_grad():
        assert torch.allclose(fresh(x), trained(x), atol=1e-5)


def test_zip_merger_starts_at_the_average():
    d1 = [torch.randn(5, 4)]
    d2 = [torch.randn(5, 4)]
    merger = ZipMerger([d1, d2], init=0.5)
    assert torch.allclose(merger.delta(0), 0.5 * (d1[0] + d2[0]))


@pytest.mark.parametrize("merge_type", ["ZipLoRA", "LoRASoups", "LoRAHub"])
def test_merge_heads_runs_and_returns_a_head(merge_type, tmp_path):
    wrap, make_base, shape = BUILDERS["svit"]
    base = make_base()
    heads = []
    for seed in (1, 2):
        m = trained_like("svit", base, seed)
        save_head(m, tmp_path / f"{seed}_lora.safetensors", tmp_path / f"{seed}_fc.safetensors")
        heads.append(
            load_head(tmp_path / f"{seed}_lora.safetensors", tmp_path / f"{seed}_fc.safetensors")
        )

    x = torch.randn(16, *shape)
    y = (torch.rand(16, N_CLASSES) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    merged, model = merge_heads(
        merge_type,
        wrap(copy.deepcopy(base)),
        heads,
        [0.5, 0.5],
        [loader, loader],
        epochs=1,
        lr=1e-2,
        device=torch.device("cpu"),
        hub_loader=loader,
        hub_budget=4,
    )
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, N_CLASSES)
    assert "fc" in merged
    if merge_type == "LoRAHub":
        assert len(merged["A"]) == 4 and merged["A"][0].shape == heads[0]["A"][0].shape
    else:
        assert len(merged["delta"]) == 4 and merged["delta"][0].shape == (16, 16)


def test_forgetting_and_intransigence():
    acc = np.array([[0.9, np.nan], [0.7, 0.8]])
    f, i, _ = forgetting_and_intransigence(acc, joint_acc=[0.95, 0.85])
    assert f[0] == 0 and np.isclose(f[1], 0.2)
    assert np.allclose(i, [0.05, 0.05])

"""End-to-end smoke test: task tuning, then every merge strategy and approach.

Uses synthetic saved datasets and a tiny SpectralGPT-style model instead of the
pretrained backbone, so the whole pipeline runs on CPU in seconds.
"""

import pytest
import torch

import cl_lora.models.backbones as backbones
from cl_lora.experiments.baseline import run_baseline_experiment
from cl_lora.experiments.merge import run_merge_experiment
from cl_lora.models.lora_vit import LoRA_SViT
from cl_lora.models.video_vit import VisionTransformer

COUNTRIES = ["A", "B", "C"]
N_CLASSES = 3


def tiny_loader(lora_r=2, lora_alpha=4, num_classes=N_CLASSES, device=None, **_):
    torch.manual_seed(0)  # same "pretrained" backbone every time
    base = VisionTransformer(num_frames=4, t_patch_size=2, img_size=16, patch_size=8, in_chans=1,
                             num_classes=num_classes, embed_dim=16, depth=2, num_heads=2,
                             sep_pos_embed=True, drop_path_rate=0.0)
    return LoRA_SViT(base, r=lora_r, alpha=lora_alpha, num_classes=num_classes).to(device or "cpu")


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(0)
    for i, c in enumerate(COUNTRIES):
        (tmp_path / "data" / c).mkdir(parents=True)
        for split, n in (("train", 40), ("test", 16)):
            x = torch.randn(n, 4, 16, 16, generator=g) + i
            y = (torch.rand(n, N_CLASSES, generator=g) > 0.5).float()
            torch.save((x, y), tmp_path / "data" / c / f"{split}_data.pt")
    monkeypatch.setattr(backbones, "load_spectral_gpt", tiny_loader)
    monkeypatch.setenv("PATHS_SAVE_DIR", str(tmp_path / "out"))
    config = {
        "model": {"backbone": "spectral_gpt", "lora_r": 2, "lora_alpha": 4, "num_classes": N_CLASSES},
        "data": {"batch_size": 8, "num_workers": 0, "saved_datasets_dir": str(tmp_path / "data")},
        "training": {"epochs": 1, "lr": 1e-3, "patience": None},
        "experiment": {"countries": COUNTRIES, "permutation": [2, 0, 1], "memory_size": 16, "seed": 0,
                       "train_frac": 0.75},
        "merge": {"epochs": 1, "hub_budget": 3,
                  "task_heads_dir": str(tmp_path / "out" / "task_tuning" / "permutation_2_0_1")},
    }
    return config


def test_task_tuning_then_merging(setup):
    df = run_baseline_experiment(setup, "task_tuning")
    assert list(df["countries"]) == [("C",), ("A",), ("B",)]  # permutation order
    for merge_type in ("ZipLoRA", "LoRASoups", "LoRAHub"):
        for approach in ("continual", "from_scratch"):
            res = run_merge_experiment(setup, merge_type, approach)
            assert list(res["step"]) == [2, 3]
            assert res.iloc[-1]["countries"] == ("C", "A", "B")
            assert 0.0 <= res.iloc[-1]["mean_micro_ap"] <= 1.0


def test_no_replay_evaluates_by_position(setup):
    df = run_baseline_experiment(setup, "no_replay")
    last = df.iloc[-1]
    assert last["countries"] == ("C", "A", "B")
    assert all(f"{c}_micro_ap" in df.columns for c in COUNTRIES)

"""SpectralGPT backbone model loading and configuration.

SpectralGPT is a spectral remote sensing foundation model based on
masked autoencoder pretraining on satellite imagery.

Reference:
    Hong, D., et al. "SpectralGPT: Spectral Remote Sensing Foundation Model."
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
    https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...data.constants import get_pretrained_path
from ..lora_vit import LoRA_SViT
from ..pos_embed import interpolate_pos_embed
from ..video_vit import vit_base_patch8_128


def load_mae_encoder(
    model: nn.Module,
    ckpt_path: str,
) -> nn.Module:
    """Load MAE pretrained weights into SpectralGPT model.

    Handles the weight loading process including:
    - Removing decoder and mask token weights
    - Handling shape mismatches in patch embedding
    - Interpolating positional embeddings if needed

    Args:
        model: SpectralGPT VisionTransformer model
        ckpt_path: Path to pretrained checkpoint

    Returns:
        Model with loaded pretrained weights
    """
    state_dict = model.state_dict()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_model = ckpt.get("model", ckpt)

    # Remove decoder & mask token weights
    ckpt_model = {
        k: v
        for k, v in ckpt_model.items()
        if not (k.startswith("decoder") or k.startswith("mask_token"))
    }

    # Delete mismatched weights
    keys_to_check = [
        "patch_embed.0.proj.weight",
        "patch_embed.1.proj.weight",
        "patch_embed.2.proj.weight",
        "patch_embed.2.proj.bias",
        "head.weight",
        "head.bias",
    ]
    for k in keys_to_check:
        if k in ckpt_model and ckpt_model[k].shape != state_dict.get(k, torch.tensor([])).shape:
            print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
            del ckpt_model[k]

    # Positional embedding interpolation
    if "pos_embed_spatial" in ckpt_model:
        interpolate_pos_embed(model, ckpt_model)

    # Load with strict=False to ignore missing head weights
    msg = model.load_state_dict(ckpt_model, strict=False)
    print("SpectralGPT weights loaded:", msg)

    return model


def load_spectral_gpt(
    pretrained_path: str | None = None,
    num_classes: int = 19,
    lora_r: int = 4,
    lora_alpha: int = 16,
    device: str | torch.device | None = None,
) -> LoRA_SViT:
    """Load SpectralGPT model with LoRA adapter.

    Creates a SpectralGPT ViT-Base model, loads pretrained weights,
    and wraps it with Low-Rank Adaptation layers.

    Args:
        pretrained_path: Path to pretrained weights. If None, uses
                        PATHS_SPECTRAL_PRETRAINED from environment.
        num_classes: Number of output classes (default: 19 for BigEarthNet)
        lora_r: LoRA rank (default: 4)
        lora_alpha: LoRA alpha scaling factor (default: 16)
        device: Device to load model to. If None, auto-detects GPU.

    Returns:
        LoRA-wrapped SpectralGPT model ready for fine-tuning.

    Example:
        >>> model = load_spectral_gpt(num_classes=19, lora_r=4)
        >>> output = model(images)  # images: [B, 12, 128, 128]
    """
    # Get pretrained path
    if pretrained_path is None:
        pretrained_path = get_pretrained_path("spectral_gpt")

    # Create base model
    model = vit_base_patch8_128(sep_pos_embed=True, num_classes=num_classes)

    # Load pretrained weights
    model = load_mae_encoder(model, pretrained_path)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpectralGPT base model trainable parameters: {num_params:,}")

    # Wrap with LoRA
    lora_model = LoRA_SViT(model, r=lora_r, alpha=lora_alpha, num_classes=num_classes)

    num_params_lora = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"SpectralGPT + LoRA trainable parameters: {num_params_lora:,}")

    # Move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)

    return lora_model

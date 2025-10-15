"""SoftCon (DINOv2-based) backbone model loading and configuration.

SoftCon is a self-supervised learning model for Earth observation based on
DINOv2 architecture, pretrained on Sentinel-2 imagery.

Reference:
    Wang, Y., et al. "SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal
    Dataset for Self-Supervised Learning in Earth Observation." CVPRW, 2023.
    https://github.com/zhu-xlab/softcon

Note:
    Requires the softcon package to be installed:
    pip install git+https://github.com/zhu-xlab/softcon.git
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...data.constants import get_pretrained_path


def load_softcon(
    pretrained_path: str | None = None,
    num_classes: int = 19,
    lora_r: int = 4,
    lora_alpha: int = 16,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Load SoftCon (DINOv2) model with LoRA adapter.

    Creates a DINOv2 ViT-Base model from the softcon package,
    loads pretrained weights, and wraps it with LoRA layers.

    Args:
        pretrained_path: Path to pretrained weights. If None, uses
                        PATHS_SOFTCON_PRETRAINED from environment.
        num_classes: Number of output classes (default: 19 for BigEarthNet)
        lora_r: LoRA rank (default: 4)
        lora_alpha: LoRA alpha scaling factor (default: 16)
        device: Device to load model to. If None, auto-detects GPU.

    Returns:
        LoRA-wrapped SoftCon model with classification head, ready for fine-tuning.

    Raises:
        ImportError: If softcon package is not installed.

    Example:
        >>> model = load_softcon(num_classes=19, lora_r=4)
        >>> output = model(images)  # images: [B, 13, 224, 224]
    """
    # Import softcon - requires external package
    try:
        from models.dinov2 import vision_transformer as dinov2_vitb
    except ImportError:
        raise ImportError(
            "SoftCon backbone requires the softcon package. "
            "Install with: pip install git+https://github.com/zhu-xlab/softcon.git"
        )

    # Import LoRA wrapper for timm-style models
    try:
        from lora import LoRA_ViT_timm
    except ImportError:
        # Fallback to local implementation
        from ..lora_vit import LoRA_ViT_timm

    # Get pretrained path
    if pretrained_path is None:
        pretrained_path = get_pretrained_path("softcon")

    # Create base model
    model_vitb14 = dinov2_vitb.__dict__["vit_base"](
        img_size=224,
        patch_size=14,
        in_chans=13,  # 13 Sentinel-2 bands (including zero-initialized B10)
        block_chunks=0,
        init_values=1e-4,
        num_register_tokens=0,
    )

    # Load pretrained weights
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    model_vitb14.load_state_dict(ckpt)
    print("SoftCon pretrained weights loaded")

    num_params = sum(p.numel() for p in model_vitb14.parameters() if p.requires_grad)
    print(f"SoftCon base model trainable parameters: {num_params:,}")

    # Wrap with LoRA (num_classes=0 means no classification head in LoRA wrapper)
    lora_model = LoRA_ViT_timm(model_vitb14, num_classes=0, r=lora_r, alpha=lora_alpha)

    # Add classification head
    classifier = nn.Linear(model_vitb14.embed_dim, num_classes)
    lora_with_head = nn.Sequential(lora_model, classifier)

    # Ensure classification head is trainable
    for param in classifier.parameters():
        param.requires_grad = True

    num_params_lora = sum(p.numel() for p in lora_with_head.parameters() if p.requires_grad)
    print(f"SoftCon + LoRA trainable parameters: {num_params_lora:,}")

    # Move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_with_head.to(device)

    return lora_with_head

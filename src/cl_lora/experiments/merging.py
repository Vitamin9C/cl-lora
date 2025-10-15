"""LoRA merging strategies for continual learning.

This module implements various LoRA weight merging techniques:
- ZipLoRA: Gradient-based importance weighting for merging LoRA adapters
- LoRAHub: Coefficient optimization for combining multiple LoRA adapters

References:
    - ZipLoRA: https://arxiv.org/abs/2311.13600
    - LoRAHub: https://arxiv.org/abs/2307.13269
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def get_lora_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA parameters (w_A and w_B weights) from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        Dictionary mapping parameter names to tensors
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if (
            "w_a" in name.lower()
            or "w_b" in name.lower()
            or "linear_a" in name.lower()
            or "linear_b" in name.lower()
        ):
            lora_params[name] = param.detach().clone()
    return lora_params


def set_lora_parameters(model: nn.Module, lora_params: dict[str, torch.Tensor]) -> None:
    """Set LoRA parameters in a model.

    Args:
        model: Model with LoRA layers
        lora_params: Dictionary mapping parameter names to tensors
    """
    state_dict = model.state_dict()
    for name, param in lora_params.items():
        if name in state_dict:
            state_dict[name] = param
    model.load_state_dict(state_dict)


def compute_gradient_importance(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_batches: int = 10,
) -> dict[str, torch.Tensor]:
    """Compute gradient-based importance scores for LoRA parameters.

    Uses Fisher Information approximation: importance = E[grad^2]

    Args:
        model: Model with LoRA layers
        dataloader: DataLoader for computing gradients
        criterion: Loss function
        device: Device to run computation on
        num_batches: Number of batches to use for estimation

    Returns:
        Dictionary mapping parameter names to importance scores
    """
    model.train()
    importance = {}

    # Initialize importance accumulator
    for name, param in model.named_parameters():
        if param.requires_grad and (
            "w_a" in name.lower()
            or "w_b" in name.lower()
            or "linear_a" in name.lower()
            or "linear_b" in name.lower()
        ):
            importance[name] = torch.zeros_like(param)

    # Accumulate squared gradients
    batch_count = 0
    for imgs, labels in dataloader:
        if batch_count >= num_batches:
            break

        imgs = imgs.to(device)
        labels = labels.to(device).float()

        model.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if name in importance and param.grad is not None:
                importance[name] += param.grad.pow(2)

        batch_count += 1

    # Normalize by batch count
    for name in importance:
        importance[name] /= batch_count

    return importance


class ZipLoRAMerger:
    """ZipLoRA: Merging LoRA adapters using gradient-based importance weighting.

    ZipLoRA merges two LoRA adapters by computing importance weights for each
    parameter based on gradient information, then performing weighted averaging.

    Reference: https://arxiv.org/abs/2311.13600
    """

    def __init__(
        self,
        device: torch.device | None = None,
        num_importance_batches: int = 10,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_importance_batches = num_importance_batches
        self.criterion = nn.BCEWithLogitsLoss()

    def merge(
        self,
        model_old: nn.Module,
        model_new: nn.Module,
        dataloader_old: DataLoader,
        dataloader_new: DataLoader,
    ) -> nn.Module:
        """Merge two LoRA-adapted models using ZipLoRA.

        Args:
            model_old: Model with LoRA adapter for old tasks
            model_new: Model with LoRA adapter for new task
            dataloader_old: DataLoader for old task data (for importance estimation)
            dataloader_new: DataLoader for new task data (for importance estimation)

        Returns:
            Merged model with combined LoRA weights
        """
        log.info("Computing importance scores for old model...")
        importance_old = compute_gradient_importance(
            model_old, dataloader_old, self.criterion, self.device, self.num_importance_batches
        )

        log.info("Computing importance scores for new model...")
        importance_new = compute_gradient_importance(
            model_new, dataloader_new, self.criterion, self.device, self.num_importance_batches
        )

        # Get LoRA parameters
        params_old = get_lora_parameters(model_old)
        params_new = get_lora_parameters(model_new)

        # Merge using importance-weighted averaging
        merged_params = {}
        for name in params_old:
            if name in params_new:
                imp_old = importance_old.get(name, torch.ones_like(params_old[name]))
                imp_new = importance_new.get(name, torch.ones_like(params_new[name]))

                # Normalize importance scores
                total_imp = imp_old + imp_new + 1e-8
                weight_old = imp_old / total_imp
                weight_new = imp_new / total_imp

                # Weighted average
                merged_params[name] = weight_old * params_old[name] + weight_new * params_new[name]
            else:
                merged_params[name] = params_old[name]

        # Add any new parameters not in old model
        for name in params_new:
            if name not in merged_params:
                merged_params[name] = params_new[name]

        # Create merged model (use new model as base)
        merged_model = copy.deepcopy(model_new)
        set_lora_parameters(merged_model, merged_params)

        log.info("ZipLoRA merge complete")
        return merged_model


class LoRAHubMerger:
    """LoRAHub: Combining multiple LoRA adapters with optimized coefficients.

    LoRAHub learns optimal linear combination coefficients for multiple LoRA
    adapters by minimizing loss on a validation set.

    Reference: https://arxiv.org/abs/2307.13269
    """

    def __init__(
        self,
        device: torch.device | None = None,
        num_optimization_steps: int = 100,
        learning_rate: float = 0.1,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_optimization_steps = num_optimization_steps
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()

    def merge(
        self,
        base_model: nn.Module,
        lora_models: list[nn.Module],
        val_dataloader: DataLoader,
        initial_coefficients: list[float] | None = None,
    ) -> tuple[nn.Module, list[float]]:
        """Merge multiple LoRA adapters using LoRAHub.

        Args:
            base_model: Base model without LoRA (or with base LoRA to merge into)
            lora_models: List of models with different LoRA adapters
            val_dataloader: Validation DataLoader for coefficient optimization
            initial_coefficients: Initial coefficients (default: uniform)

        Returns:
            Tuple of (merged model, optimized coefficients)
        """
        n_adapters = len(lora_models)

        # Initialize coefficients
        if initial_coefficients is None:
            coefficients = torch.ones(n_adapters, device=self.device) / n_adapters
        else:
            coefficients = torch.tensor(initial_coefficients, device=self.device)
        coefficients.requires_grad = True

        # Extract LoRA parameters from all models
        all_lora_params = [get_lora_parameters(m) for m in lora_models]
        param_names = list(all_lora_params[0].keys())

        optimizer = torch.optim.Adam([coefficients], lr=self.learning_rate)

        # Get a batch for optimization
        batch_iter = iter(val_dataloader)

        log.info(f"Optimizing LoRAHub coefficients for {n_adapters} adapters...")

        for step in range(self.num_optimization_steps):
            try:
                imgs, labels = next(batch_iter)
            except StopIteration:
                batch_iter = iter(val_dataloader)
                imgs, labels = next(batch_iter)

            imgs = imgs.to(self.device)
            labels = labels.to(self.device).float()

            # Normalize coefficients to sum to 1
            normalized_coef = torch.softmax(coefficients, dim=0)

            # Merge LoRA parameters with current coefficients
            merged_params = {}
            for name in param_names:
                merged_params[name] = sum(
                    normalized_coef[i] * all_lora_params[i][name].to(self.device)
                    for i in range(n_adapters)
                )

            # Apply merged parameters to base model
            merged_model = copy.deepcopy(base_model)
            set_lora_parameters(merged_model, merged_params)
            merged_model.to(self.device)
            merged_model.eval()

            # Compute loss
            with torch.enable_grad():
                outputs = merged_model(imgs)
                loss = self.criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                log.info(
                    f"  Step {step}: loss={loss.item():.4f}, coefficients={normalized_coef.detach().cpu().tolist()}"
                )

        # Final merge with optimized coefficients
        final_coef = torch.softmax(coefficients, dim=0).detach()
        log.info(f"Final coefficients: {final_coef.cpu().tolist()}")

        merged_params = {}
        for name in param_names:
            merged_params[name] = sum(
                final_coef[i] * all_lora_params[i][name].to(self.device) for i in range(n_adapters)
            )

        final_model = copy.deepcopy(base_model)
        set_lora_parameters(final_model, merged_params)

        log.info("LoRAHub merge complete")
        return final_model, final_coef.cpu().tolist()


def simple_average_merge(models: list[nn.Module]) -> nn.Module:
    """Simple averaging of LoRA parameters (LoRA Soups baseline).

    Args:
        models: List of models with LoRA adapters

    Returns:
        Model with averaged LoRA parameters
    """
    all_params = [get_lora_parameters(m) for m in models]
    param_names = list(all_params[0].keys())

    merged_params = {}
    for name in param_names:
        merged_params[name] = sum(p[name] for p in all_params) / len(all_params)

    merged_model = copy.deepcopy(models[-1])
    set_lora_parameters(merged_model, merged_params)

    return merged_model

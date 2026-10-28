"""Evaluation metrics for continual learning experiments.

Provides functions for computing and aggregating metrics across
multi-label classification tasks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


def default_metrics(
    y_true: list[Any] | torch.Tensor,
    y_pred: list[Any] | torch.Tensor,
) -> dict[str, float]:
    """Compute default evaluation metrics.

    Wraps sklearn.metrics.average_precision_score with micro averaging.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities or scores

    Returns:
        Dictionary with micro average precision score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    return {"micro_AP": average_precision_score(y_true, y_pred, average="micro")}


def eval_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Evaluate model on a test set.

    Computes micro AP, macro AP, and accuracy for multi-label classification.

    Args:
        model: Model to evaluate (should be nn.Module or LightningModule)
        test_loader: DataLoader for test data
        device: Device to run evaluation on. If None, auto-detects.

    Returns:
        Dictionary with metrics:
        - micro_ap: Micro-averaged average precision
        - macro_ap: Macro-averaged average precision
        - accuracy: Per-label accuracy

    Example:
        >>> metrics = eval_model(model, test_loader)
        >>> print(f"Micro AP: {metrics['micro_ap']:.4f}")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_labels = []
    all_scores = []
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()

            # Forward pass
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)

            # Accumulate for AP computation
            all_scores.append(probs.cpu())
            all_labels.append(labels.cpu())

            # Accuracy computation
            preds = (probs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.numel()

    # Stack all predictions and labels
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute metrics
    micro_ap = average_precision_score(all_labels, all_scores, average="micro")
    macro_ap = average_precision_score(all_labels, all_scores, average="macro")
    accuracy = 100.0 * val_correct / val_total

    return {
        "micro_ap": micro_ap,
        "macro_ap": macro_ap,
        "accuracy": accuracy,
    }


def compute_forgetting(
    accuracies_after_each_task: list[dict[str, float]],
) -> dict[str, float]:
    """Compute forgetting metrics for continual learning.

    Forgetting is measured as the difference between peak performance
    on a task and the performance after learning subsequent tasks.

    Args:
        accuracies_after_each_task: List of dictionaries containing
            per-task metrics after each training step.

    Returns:
        Dictionary with average and maximum forgetting metrics.
    """
    if len(accuracies_after_each_task) < 2:
        return {"avg_forgetting": 0.0, "max_forgetting": 0.0}

    # Track peak performance for each task
    peak_perf: dict[str, float] = {}
    forgetting_per_task: dict[str, float] = {}

    for step_metrics in accuracies_after_each_task:
        for task, metric in step_metrics.items():
            if task not in peak_perf:
                peak_perf[task] = metric
            else:
                peak_perf[task] = max(peak_perf[task], metric)

    # Compute forgetting (peak - final)
    final_metrics = accuracies_after_each_task[-1]
    for task in peak_perf:
        if task in final_metrics:
            forgetting_per_task[task] = peak_perf[task] - final_metrics[task]

    forgetting_values = list(forgetting_per_task.values())

    return {
        "avg_forgetting": sum(forgetting_values) / len(forgetting_values)
        if forgetting_values
        else 0.0,
        "max_forgetting": max(forgetting_values) if forgetting_values else 0.0,
        "per_task_forgetting": forgetting_per_task,
    }


def forgetting_and_intransigence(acc_matrix, joint_acc=None):
    """Forgetting and intransigence of a continual learning run (Chaudhry et al. 2018).

    Args:
        acc_matrix: (K, K) array; entry [k, j] is the metric on task j after training
            through task k (0-based). Entries with j > k may be NaN.
        joint_acc: Optional length-K array; entry k is the metric on task k of the
            joint-training reference. Needed for intransigence.

    Returns:
        F_per_k: Length-K array; mean forgetting after task k (0 for k = 0).
        I_per_k: Length-K array; intransigence on task k (NaN without joint_acc).
        f_taskwise: (K, K) array of per-task forgetting (defined for j < k).
    """
    import numpy as np

    acc_matrix = np.asarray(acc_matrix, dtype=float)
    K = acc_matrix.shape[0]
    f_taskwise = np.full((K, K), np.nan)
    F_per_k = np.zeros(K)
    I_per_k = np.full(K, np.nan)

    for k in range(1, K):
        for j in range(k):
            f_taskwise[k, j] = np.nanmax(acc_matrix[:k, j]) - acc_matrix[k, j]
        F_per_k[k] = np.nanmean(f_taskwise[k, :k])

    if joint_acc is not None:
        joint_acc = np.asarray(joint_acc, dtype=float)
        I_per_k = joint_acc - np.diag(acc_matrix)

    return F_per_k, I_per_k, f_taskwise

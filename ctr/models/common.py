"""Shared PyTorch building blocks and training utilities."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Dense ReLU stack mapping ``input_dim`` -> ``output_dim``."""

    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int], output_dim: int
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve the training device.

    ``"auto"`` selects CUDA when available and CPU otherwise; an explicit
    ``"cuda"`` on a machine without GPUs falls back to CPU with a warning so
    training never hard-fails on hardware differences.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


def make_optimizer(name: str, params: Iterable, lr: float):
    """Build the configured optimizer (adam / adagrad).

    Adagrad is the classic choice for sparse embedding tables; it is kept
    configurable because dense-only models often prefer Adam.
    """
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr)
    raise ValueError(f"unknown optimizer {name!r} (expected adam or adagrad)")


def binary_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary cross-entropy between labels and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_score, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC, NaN when a label class is absent."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

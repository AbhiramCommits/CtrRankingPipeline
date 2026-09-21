"""Training loop for the DLRM ranker.

BCEWithLogits loss, configurable Adam/Adagrad optimizer, gradient clipping,
per-epoch train/val AUC and logloss, early stopping on validation logloss
and checkpointing to ``artifacts/ranker/``. CUDA is used when available
with an automatic CPU fallback (see :func:`ctr.models.common.resolve_device`).
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ctr.models.common import (
    binary_auc,
    binary_logloss,
    make_optimizer,
    resolve_device,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for :func:`train_dlrm`."""

    device: str = "auto"
    optimizer: str = "adagrad"
    lr: float = 0.01
    batch_size: int = 4096
    epochs: int = 8
    patience: int = 3
    grad_clip: float = 1.0
    seed: int = 42


def _predict(
    model: nn.Module, dataset: TensorDataset, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probs = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for cats, nums, ys in loader:
            logits = model(cats.to(device), nums.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(ys.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    device: str = "auto",
    batch_size: int = 4096,
) -> dict[str, float]:
    """AUC and logloss of the model over a dataset."""
    dev = resolve_device(device)
    y, p = _predict(model, dataset, dev, batch_size)
    return {"auc": binary_auc(y, p), "logloss": binary_logloss(y, p)}


def train_dlrm(
    model: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    cfg: TrainConfig,
    artifact_dir: str,
) -> dict[str, Any]:
    """Train ``model`` and persist the best checkpoint + metrics.

    Returns a dict with the history and the final (best-checkpoint)
    validation metrics. The checkpoint is ``artifact_dir/model.pt``.
    """
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = make_optimizer(cfg.optimizer, model.parameters(), cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    os.makedirs(artifact_dir, exist_ok=True)

    best_val_logloss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for cats, nums, ys in loader:
            cats, nums, ys = cats.to(device), nums.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(cats, nums)
            loss = criterion(logits, ys)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(ys)
            count += len(ys)

        train_metrics = evaluate(model, train_dataset, device, cfg.batch_size)
        val_metrics = evaluate(model, val_dataset, device, cfg.batch_size)
        epoch_loss = total_loss / count if count else float("nan")
        logger.info(
            "epoch=%d train_loss=%.5f train_auc=%.4f train_logloss=%.5f "
            "val_auc=%.4f val_logloss=%.5f",
            epoch,
            epoch_loss,
            train_metrics["auc"],
            train_metrics["logloss"],
            val_metrics["auc"],
            val_metrics["logloss"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_auc": train_metrics["auc"],
                "train_logloss": train_metrics["logloss"],
                "val_auc": val_metrics["auc"],
                "val_logloss": val_metrics["logloss"],
            }
        )

        if val_metrics["logloss"] < best_val_logloss - 1e-4:
            best_val_logloss = val_metrics["logloss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping after %d epochs", epoch)
                break

    model.load_state_dict(best_state)
    checkpoint = {"state_dict": best_state, "config": asdict(cfg)}
    if hasattr(model, "config"):
        checkpoint["model_config"] = model.config()
    torch.save(checkpoint, os.path.join(artifact_dir, "model.pt"))
    metrics = {
        "best_val_logloss": best_val_logloss,
        "final_val": evaluate(model, val_dataset, device, cfg.batch_size),
        "history": history,
    }
    with open(os.path.join(artifact_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(
        "Checkpointed best model (val_logloss=%.5f) to %s",
        best_val_logloss,
        os.path.join(artifact_dir, "model.pt"),
    )
    return metrics

"""Unit tests for the DLRM training loop."""

import json
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ctr.models.common import resolve_device
from ctr.models.dlrm import DLRM
from ctr.models.train import TrainConfig, evaluate, train_dlrm


def _synthetic_dataset(n=800, seed=0):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, 30, (n, 4))
    numeric = rng.normal(size=(n, 5)).astype(np.float32)
    z = 1.0 * numeric[:, 0] + 0.5 * (cats[:, 1] > 15) - 0.4 * numeric[:, 2]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    return TensorDataset(
        torch.from_numpy(cats),
        torch.from_numpy(numeric),
        torch.from_numpy(y),
    )


def test_train_loop_improves_and_checkpoints(tmp_path):
    dataset = _synthetic_dataset()
    model = DLRM(
        field_vocab_sizes=[35] * 4,
        num_numeric=5,
        embedding_dim=8,
        bottom_mlp_dims=(32,),
        top_mlp_dims=(32,),
    )
    cfg = TrainConfig(
        device="cpu",
        optimizer="adam",
        lr=0.02,
        batch_size=128,
        epochs=4,
        patience=3,
        grad_clip=1.0,
        seed=0,
    )
    metrics = train_dlrm(model, dataset, dataset, cfg, str(tmp_path / "ranker"))

    assert os.path.isfile(os.path.join(tmp_path, "ranker", "model.pt"))
    assert os.path.isfile(os.path.join(tmp_path, "ranker", "metrics.json"))
    assert metrics["final_val"]["auc"] > 0.65

    history_losses = [entry["train_loss"] for entry in metrics["history"]]
    assert history_losses[-1] < history_losses[0]

    checkpoint = torch.load(os.path.join(tmp_path, "ranker", "model.pt"))
    assert "state_dict" in checkpoint

    with open(os.path.join(tmp_path, "ranker", "metrics.json")) as handle:
        json.load(handle)  # valid JSON

    final = evaluate(model, dataset, device="cpu", batch_size=256)
    assert 0.0 < final["logloss"] < 0.7


def test_resolve_device_falls_back_to_cpu():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"

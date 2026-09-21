"""Two-tower retrieval model trained with in-batch (sampled softmax) negatives.

Architecture
------------
* **Context tower**: embeddings over the context categorical fields plus the
  numeric feature block (transformed integers + point-in-time aggregates),
  fed through an MLP and L2-normalized to ``output_dim`` (default 64).
* **Ad tower**: embeddings over the ad identity fields only (``C1``
  advertiser, ``C6`` campaign, ``C9`` creative), same MLP + normalization.

Training
--------
Click labels are implicit feedback: we train only on positive pairs and take
negatives *in-batch* -- for every context embedding the remaining ads in the
batch are negatives, giving the sampled-softmax / cross-entropy objective
without an explicit negative sampler. Ranking quality is logged as in-batch
recall@k on a held-out (validation) set of positives.

Note on PIT features: the rolling CTR/impression aggregates are
pair-specific (they depend on the ad identity). Folding them into the
context tower is a deliberate simplification for this exercise; in a
production two-tower, pair-specific features belong to the ranking stage.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ctr.data.synthetic import CATEGORICAL_FEATURES
from ctr.models.common import MLP, resolve_device

logger = logging.getLogger(__name__)

# Ad identity fields, as specified for the ad tower.
AD_IDENTITY_FIELDS: tuple[str, ...] = ("C1", "C6", "C9")
CONTEXT_FIELDS: tuple[str, ...] = tuple(
    field for field in CATEGORICAL_FEATURES if field not in AD_IDENTITY_FIELDS
)
TOWER_OUTPUT_DIM = 64


class Tower(nn.Module):
    """Embeddings -> MLP -> L2-normalized vector (one tower of the model)."""

    def __init__(
        self,
        field_names: Sequence[str],
        vocab_sizes: dict[str, int],
        embedding_dim: int,
        mlp_dims: Sequence[int],
        output_dim: int,
        numeric_dim: int = 0,
    ):
        super().__init__()
        self.field_names = list(field_names)
        self.numeric_dim = numeric_dim
        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(vocab_sizes[field], embedding_dim, padding_idx=0)
                for field in self.field_names
            }
        )
        mlp_input = embedding_dim * len(self.field_names) + numeric_dim
        self.mlp = MLP(mlp_input, mlp_dims, output_dim)

    def forward(
        self,
        categorical_indices: torch.Tensor,
        numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            self.embeddings[field](categorical_indices[:, i])
            for i, field in enumerate(self.field_names)
        ]
        if self.numeric_dim:
            parts.append(numeric)
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTower(nn.Module):
    """Two-tower retriever: scores are inner products of normalized vectors."""

    def __init__(
        self,
        context_fields: Sequence[str],
        context_vocab_sizes: dict[str, int],
        numeric_dim: int,
        ad_fields: Sequence[str],
        ad_vocab_sizes: dict[str, int],
        embedding_dim: int = 32,
        tower_mlp_dims: Sequence[int] = (256, 128),
        output_dim: int = TOWER_OUTPUT_DIM,
    ):
        super().__init__()
        self.context_fields = list(context_fields)
        self.context_vocab_sizes = dict(context_vocab_sizes)
        self.numeric_dim = int(numeric_dim)
        self.ad_fields = list(ad_fields)
        self.ad_vocab_sizes = dict(ad_vocab_sizes)
        self.embedding_dim = int(embedding_dim)
        self.tower_mlp_dims = tuple(tower_mlp_dims)
        self.output_dim = int(output_dim)
        self.context_tower = Tower(
            context_fields,
            context_vocab_sizes,
            embedding_dim,
            tower_mlp_dims,
            output_dim,
            numeric_dim=numeric_dim,
        )
        self.ad_tower = Tower(
            ad_fields, ad_vocab_sizes, embedding_dim, tower_mlp_dims, output_dim
        )

    def config(self) -> dict:
        """Structural hyperparameters (enough to reconstruct the model)."""
        return {
            "context_fields": self.context_fields,
            "context_vocab_sizes": self.context_vocab_sizes,
            "numeric_dim": self.numeric_dim,
            "ad_fields": self.ad_fields,
            "ad_vocab_sizes": self.ad_vocab_sizes,
            "embedding_dim": self.embedding_dim,
            "tower_mlp_dims": self.tower_mlp_dims,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> TwoTower:
        """Load a checkpoint produced by :func:`train_two_tower`."""
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(torch.device(device))
        model.eval()
        return model

    def forward(
        self,
        context_cats: torch.Tensor,
        context_numeric: torch.Tensor,
        ad_cats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.context_tower(context_cats, context_numeric),
            self.ad_tower(ad_cats),
        )

    @torch.no_grad()
    def embed_context(self, context_cats: torch.Tensor, context_numeric: torch.Tensor):
        return self.context_tower(context_cats, context_numeric)

    @torch.no_grad()
    def embed_ad(self, ad_cats: torch.Tensor):
        return self.ad_tower(ad_cats)


def in_batch_loss(context_vecs: torch.Tensor, ad_vecs: torch.Tensor) -> torch.Tensor:
    """Sampled-softmax loss: each context's true ad vs every batch ad."""
    logits = context_vecs @ ad_vecs.t()  # [B, B]
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def recall_at_k(
    context_vecs: torch.Tensor, ad_vecs: torch.Tensor, k: int
) -> float:
    """In-batch recall@k: fraction of contexts whose true ad ranks <= k."""
    logits = context_vecs @ ad_vecs.t()
    ranks = (logits >= logits.diag().unsqueeze(1)).sum(dim=1)
    return float((ranks <= k).float().mean().item())


@dataclass(frozen=True)
class TwoTowerConfig:
    """Training configuration for the two-tower retriever."""

    embedding_dim: int = 32
    tower_mlp_dims: tuple[int, ...] = (256, 128)
    output_dim: int = TOWER_OUTPUT_DIM
    lr: float = 0.001
    epochs: int = 10
    batch_size: int = 1024
    eval_batch_size: int = 2048
    recall_ks: tuple[int, ...] = (1, 10, 50)
    seed: int = 42
    device: str = "auto"


def evaluate_recall(
    model: TwoTower,
    dataset: TensorDataset,
    ks: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, float]:
    """Weighted in-batch recall@k over a dataset of positive pairs."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {k: 0.0 for k in ks}
    n_rows = 0
    with torch.no_grad():
        for context_cats, context_numeric, ad_cats in loader:
            u, a = model(
                context_cats.to(device),
                context_numeric.to(device),
                ad_cats.to(device),
            )
            batch = u.shape[0]
            if batch < 2:
                continue
            for k in ks:
                # in-batch recall is only meaningful for k < batch size
                kk = min(k, batch - 1)
                totals[k] += recall_at_k(u, a, kk) * batch
            n_rows += batch
    return {k: totals[k] / n_rows if n_rows else float("nan") for k in ks}


def train_two_tower(
    model: TwoTower,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    cfg: TwoTowerConfig,
    artifact_dir: str,
) -> dict[str, Any]:
    """Train the two-tower with in-batch negatives; log recall@k on val.

    Persists the checkpoint with the best validation recall to
    ``artifact_dir/model.pt`` and the metrics history to
    ``artifact_dir/metrics.json``.
    """
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    os.makedirs(artifact_dir, exist_ok=True)

    best_recall = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for context_cats, context_numeric, ad_cats in loader:
            context_cats = context_cats.to(device)
            context_numeric = context_numeric.to(device)
            ad_cats = ad_cats.to(device)
            optimizer.zero_grad()
            u, a = model(context_cats, context_numeric, ad_cats)
            loss = in_batch_loss(u, a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(context_cats)
            count += len(context_cats)

        recalls = evaluate_recall(
            model, val_dataset, cfg.recall_ks, cfg.eval_batch_size, device
        )
        logger.info(
            "epoch=%d train_loss=%.5f val_recall=%s",
            epoch,
            total_loss / count if count else float("nan"),
            {k: round(v, 4) for k, v in recalls.items()},
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / count if count else float("nan"),
                "val_recall": recalls,
            }
        )
        if recalls[max(cfg.recall_ks)] > best_recall:
            best_recall = recalls[max(cfg.recall_ks)]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    checkpoint = {"state_dict": best_state, "config": asdict(cfg)}
    if hasattr(model, "config"):
        checkpoint["model_config"] = model.config()
    torch.save(checkpoint, os.path.join(artifact_dir, "model.pt"))
    metrics = {"history": history, "final_val_recall": recalls}
    with open(os.path.join(artifact_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Best validation recall@%d: %.4f", max(cfg.recall_ks), best_recall)
    return metrics

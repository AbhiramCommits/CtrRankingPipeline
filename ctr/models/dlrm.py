"""DLRM-style deep ranking model.

Layout (mirrors the original DLRM paper):

* one embedding table per categorical field (dimension configurable,
  default 16; index 0 is the padding/OOV slot),
* a bottom MLP over the numeric features producing a dense vector,
* an explicit second-order interaction: the pairwise dot products of all
  embedding vectors -- ``n_fields * (n_fields - 1) / 2`` terms -- with the
  dense vector concatenated (not dotted), following the DLRM convention,
* a top MLP mapping ``[dense_vector | interaction_terms]`` to a single
  logit.

Embedding dimension and both MLP shapes are configurable from YAML (see
``configs/ranker.yaml``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ctr.models.common import MLP


class InteractionLayer(nn.Module):
    """Explicit second-order interaction between all embedding vectors.

    Given ``[B, n_fields, D]`` embeddings, outputs ``[B, n_fields *
    (n_fields - 1) / 2]`` pairwise dot products, in (i, j) upper-triangular
    order. The dense (bottom-MLP) vector is intentionally *not* part of the
    pairwise products; it is concatenated to them downstream, matching the
    DLRM design.
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        n_fields = embeddings.shape[1]
        gram = torch.bmm(embeddings, embeddings.transpose(1, 2))  # [B, n, n]
        rows, cols = torch.triu_indices(n_fields, n_fields, offset=1)
        return gram[:, rows, cols]  # [B, n*(n-1)/2]


class DLRM(nn.Module):
    """DLRM ranker mapping categorical indices + numeric features to logits."""

    def __init__(
        self,
        field_vocab_sizes: Sequence[int],
        num_numeric: int,
        embedding_dim: int = 16,
        bottom_mlp_dims: Sequence[int] = (128, 64),
        top_mlp_dims: Sequence[int] = (128, 64),
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                for vocab_size in field_vocab_sizes
            ]
        )
        self.bottom_mlp = MLP(num_numeric, bottom_mlp_dims[:-1], bottom_mlp_dims[-1])
        self.interaction = InteractionLayer()
        n_fields = len(field_vocab_sizes)
        top_input_dim = bottom_mlp_dims[-1] + n_fields * (n_fields - 1) // 2
        self.top_mlp = MLP(top_input_dim, top_mlp_dims[:-1], 1)

    def forward(
        self, categorical_indices: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        """categorical_indices: [B, n_fields] int64; numeric: [B, D] float."""
        embeddings = torch.stack(
            [
                embedding(categorical_indices[:, i])
                for i, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )  # [B, n_fields, E]
        interactions = self.interaction(embeddings)  # [B, n*(n-1)/2]
        dense = self.bottom_mlp(numeric)  # [B, bottom_out]
        combined = torch.cat([dense, interactions], dim=-1)
        return self.top_mlp(combined).squeeze(-1)  # [B]

"""Unit tests for the DLRM ranker and its interaction layer."""

import torch
from torch import nn

from ctr.models.dlrm import DLRM, InteractionLayer


def test_interaction_layer_produces_expected_pairwise_terms():
    layer = InteractionLayer()
    batch, n_fields, dim = 4, 5, 8
    embeddings = torch.randn(batch, n_fields, dim)
    out = layer(embeddings)
    expected_terms = n_fields * (n_fields - 1) // 2
    assert out.shape == (batch, expected_terms)

    # The upper-triangular term (i, j) must equal dot(e_i, e_j).
    i, j = 1, 3
    term_index = i * (2 * n_fields - i - 1) // 2 + (j - i - 1)
    assert torch.allclose(
        out[:, term_index],
        (embeddings[:, i] * embeddings[:, j]).sum(dim=-1),
        atol=1e-5,
    )


def test_dlrm_output_shape():
    torch.manual_seed(0)
    model = DLRM(
        field_vocab_sizes=[30, 40, 50],
        num_numeric=6,
        embedding_dim=8,
        bottom_mlp_dims=(16,),
        top_mlp_dims=(16,),
    )
    cats = torch.randint(0, 30, (8, 3))
    numeric = torch.randn(8, 6)
    logits = model(cats, numeric)
    assert logits.shape == (8,)


def test_dlrm_loss_decreases_over_50_steps():
    torch.manual_seed(0)
    n = 256
    model = DLRM(
        field_vocab_sizes=[40] * 6,
        num_numeric=8,
        embedding_dim=8,
        bottom_mlp_dims=(32,),
        top_mlp_dims=(32,),
    )
    cats = torch.randint(0, 40, (n, 6))
    numeric = torch.randn(n, 8)
    # Separable labels: a deterministic logistic function of the inputs.
    z = 0.8 * numeric[:, 0] + 0.5 * (cats[:, 0] > 20).float() - 0.3 * numeric[:, 1]
    labels = (torch.rand(n) < torch.sigmoid(z)).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(cats, numeric)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9

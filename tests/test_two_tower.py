"""Unit tests for the two-tower retrieval model."""

import torch
import torch.nn.functional as F

from ctr.retrieval.two_tower import TwoTower, in_batch_loss, recall_at_k


def make_model():
    return TwoTower(
        context_fields=["C2", "C3"],
        context_vocab_sizes={"C2": 30, "C3": 20},
        numeric_dim=5,
        ad_fields=["C1"],
        ad_vocab_sizes={"C1": 25},
        embedding_dim=8,
        tower_mlp_dims=(16,),
        output_dim=64,
    )


def test_two_tower_output_shapes_and_normalization():
    torch.manual_seed(0)
    model = make_model()
    ctx_cats = torch.randint(0, 20, (7, 2))
    ctx_numeric = torch.randn(7, 5)
    ad_cats = torch.randint(0, 25, (7, 1))
    u, a = model(ctx_cats, ctx_numeric, ad_cats)
    assert u.shape == (7, 64)
    assert a.shape == (7, 64)
    assert torch.allclose(u.norm(dim=-1), torch.ones(7), atol=1e-5)
    assert torch.allclose(a.norm(dim=-1), torch.ones(7), atol=1e-5)


def test_in_batch_loss_decreases_over_50_steps():
    torch.manual_seed(0)
    batch = 64
    model = TwoTower(
        context_fields=["C2"],
        context_vocab_sizes={"C2": 50},
        numeric_dim=4,
        ad_fields=["C1"],
        ad_vocab_sizes={"C1": 50},
        embedding_dim=8,
        tower_mlp_dims=(16,),
        output_dim=32,
    )
    ctx_cats = torch.randint(0, 50, (batch, 1))
    ctx_numeric = torch.randn(batch, 4)
    ad_cats = torch.randint(0, 50, (batch, 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        u, a = model(ctx_cats, ctx_numeric, ad_cats)
        loss = in_batch_loss(u, a)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9


def test_recall_at_k_is_perfect_for_identical_embeddings():
    u = F.normalize(torch.randn(8, 16), dim=-1)
    assert recall_at_k(u, u, k=1) == 1.0

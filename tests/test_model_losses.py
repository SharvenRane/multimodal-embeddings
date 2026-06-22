import torch

from src.model import ProjectionHead, MultimodalAligner
from src.losses import info_nce_loss


def test_projection_head_normalises():
    head = ProjectionHead(in_dim=20, embed_dim=8)
    x = torch.randn(5, 20)
    z = head(x)
    assert z.shape == (5, 8)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_aligner_outputs_two_embeddings():
    model = MultimodalAligner(dim_a=64, dim_b=48, embed_dim=16)
    a = torch.randn(7, 64)
    b = torch.randn(7, 48)
    ea, eb = model(a, b)
    assert ea.shape == (7, 16)
    assert eb.shape == (7, 16)


def test_info_nce_minimised_when_aligned():
    # Identical normalised embeddings: the diagonal dominates, loss is low.
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    aligned = info_nce_loss(z, z.clone(), temperature=0.07)

    # Shuffle one side so matched pairs no longer line up: loss should rise.
    perm = torch.randperm(16)
    misaligned = info_nce_loss(z, z[perm].clone(), temperature=0.07)

    assert aligned < misaligned


def test_info_nce_is_non_negative_scalar():
    z = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    loss = info_nce_loss(z, z.clone())
    assert loss.ndim == 0
    assert float(loss) >= 0.0

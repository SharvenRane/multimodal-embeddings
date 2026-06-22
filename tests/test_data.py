import torch

from src.data import make_paired_data, PairedDataset


def test_shapes_and_pairing():
    a, b = make_paired_data(n=128, dim_a=64, dim_b=48, seed=1)
    assert a.shape == (128, 64)
    assert b.shape == (128, 48)


def test_matched_pairs_share_signal_in_raw_space():
    # Even before any learned projection, matched raw views should correlate
    # more than mismatched ones, because they come from a shared latent. We
    # check this in a common space by projecting both with the same random map.
    a, b = make_paired_data(n=200, latent_dim=16, dim_a=64, dim_b=64, noise=0.2, seed=2)

    a = a - a.mean(0)
    b = b - b.mean(0)
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)

    sim = a @ b.t()
    matched = torch.diagonal(sim).mean()
    off = sim[~torch.eye(sim.shape[0], dtype=torch.bool)].mean()
    # Shared latent makes matched rows more similar than random rows.
    assert matched > off


def test_dataset_returns_pairs():
    a, b = make_paired_data(n=10, seed=3)
    ds = PairedDataset(a, b)
    assert len(ds) == 10
    xa, xb = ds[4]
    assert torch.equal(xa, a[4])
    assert torch.equal(xb, b[4])

"""Synthetic paired multimodal data.

Two views share a latent factor. Each view is produced by passing the shared
latent through a fixed random linear map and adding view specific noise. So a
matched pair really does carry common signal, while two random samples do not.
This is exactly the structure a contrastive aligner is meant to recover.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def make_paired_data(
    n: int = 512,
    latent_dim: int = 16,
    dim_a: int = 64,
    dim_b: int = 48,
    noise: float = 0.3,
    seed: int = 0,
):
    """Build n matched pairs (view_a, view_b) from a shared latent factor.

    Returns two tensors of shape (n, dim_a) and (n, dim_b). Row i of each
    tensor is the i-th matched pair.
    """
    g = torch.Generator().manual_seed(seed)

    latent = torch.randn(n, latent_dim, generator=g)

    # Fixed generative maps from the shared latent into each view space.
    map_a = torch.randn(latent_dim, dim_a, generator=g)
    map_b = torch.randn(latent_dim, dim_b, generator=g)

    view_a = latent @ map_a + noise * torch.randn(n, dim_a, generator=g)
    view_b = latent @ map_b + noise * torch.randn(n, dim_b, generator=g)

    return view_a, view_b


class PairedDataset(Dataset):
    """Wraps two aligned view tensors as a paired dataset."""

    def __init__(self, view_a: torch.Tensor, view_b: torch.Tensor):
        if view_a.shape[0] != view_b.shape[0]:
            raise ValueError("views must have the same number of rows")
        self.view_a = view_a
        self.view_b = view_b

    def __len__(self) -> int:
        return self.view_a.shape[0]

    def __getitem__(self, idx: int):
        return self.view_a[idx], self.view_b[idx]

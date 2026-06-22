"""Projection heads and the joint aligner.

Each view has its own encoder that maps the raw view into a shared embedding
space. Embeddings are L2 normalised so cosine similarity is just a dot product.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """A small MLP that projects one view into the shared embedding space."""

    def __init__(self, in_dim: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class MultimodalAligner(nn.Module):
    """Holds one projection head per view and exposes both embeddings."""

    def __init__(self, dim_a: int, dim_b: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.head_a = ProjectionHead(dim_a, embed_dim, hidden_dim)
        self.head_b = ProjectionHead(dim_b, embed_dim, hidden_dim)

    def encode_a(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_a(x)

    def encode_b(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_b(x)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor):
        return self.encode_a(view_a), self.encode_b(view_b)

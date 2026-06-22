"""Contrastive alignment loss (symmetric InfoNCE, the CLIP objective)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of matched embeddings.

    emb_a and emb_b are assumed L2 normalised, shape (batch, embed_dim). The
    diagonal of the similarity matrix holds the matched pairs, every off
    diagonal entry is a mismatched pair. We pull diagonals up and push the rest
    down in both the a to b and b to a directions.
    """
    if emb_a.shape != emb_b.shape:
        raise ValueError("emb_a and emb_b must have the same shape")

    logits = (emb_a @ emb_b.t()) / temperature
    targets = torch.arange(emb_a.shape[0], device=emb_a.device)

    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)

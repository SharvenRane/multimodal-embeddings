"""Training loop and alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from .data import PairedDataset
from .losses import info_nce_loss
from .model import MultimodalAligner


@dataclass
class TrainResult:
    model: MultimodalAligner
    loss_history: list = field(default_factory=list)
    gap_history: list = field(default_factory=list)


@torch.no_grad()
def mean_matched_minus_mismatched(model: MultimodalAligner, view_a: torch.Tensor, view_b: torch.Tensor) -> float:
    """Mean cosine sim of matched pairs minus mean over mismatched pairs.

    A positive value means matched pairs sit closer in the embedding space than
    random pairs, which is the property we want alignment to produce.
    """
    model.eval()
    emb_a, emb_b = model(view_a, view_b)
    sim = emb_a @ emb_b.t()  # (n, n), normalised embeddings so this is cosine

    n = sim.shape[0]
    diag = torch.diagonal(sim)
    matched = diag.mean()

    off_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    mismatched = sim[off_mask].mean()

    return float(matched - mismatched)


def train_aligner(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 0.07,
    seed: int = 0,
) -> TrainResult:
    """Train the aligner with symmetric InfoNCE and track the alignment gap."""
    torch.manual_seed(seed)

    dim_a = view_a.shape[1]
    dim_b = view_b.shape[1]
    model = MultimodalAligner(dim_a, dim_b, embed_dim=embed_dim, hidden_dim=hidden_dim)

    dataset = PairedDataset(view_a, view_b)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    result = TrainResult(model=model)

    # Record the alignment gap before any training so improvement is measurable.
    result.gap_history.append(mean_matched_minus_mismatched(model, view_a, view_b))

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_a, batch_b in loader:
            emb_a, emb_b = model(batch_a, batch_b)
            loss = info_nce_loss(emb_a, emb_b, temperature=temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach())
            n_batches += 1

        result.loss_history.append(epoch_loss / max(n_batches, 1))
        result.gap_history.append(mean_matched_minus_mismatched(model, view_a, view_b))

    return result

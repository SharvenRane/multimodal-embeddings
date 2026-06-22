import torch

from src.data import make_paired_data
from src.train import train_aligner, mean_matched_minus_mismatched


def test_alignment_gap_positive_after_training():
    a, b = make_paired_data(n=512, dim_a=64, dim_b=48, noise=0.3, seed=0)
    res = train_aligner(a, b, epochs=30, batch_size=64, lr=1e-3, seed=0)

    gap = mean_matched_minus_mismatched(res.model, a, b)
    # Matched pairs end up closer than mismatched pairs in the learned space.
    assert gap > 0.1


def test_alignment_improves_over_training():
    a, b = make_paired_data(n=512, dim_a=64, dim_b=48, noise=0.3, seed=0)
    res = train_aligner(a, b, epochs=30, batch_size=64, lr=1e-3, seed=0)

    start_gap = res.gap_history[0]
    end_gap = res.gap_history[-1]
    # Training should widen the gap between matched and mismatched similarity.
    assert end_gap > start_gap + 0.05


def test_loss_trends_down():
    a, b = make_paired_data(n=512, dim_a=64, dim_b=48, noise=0.3, seed=0)
    res = train_aligner(a, b, epochs=30, batch_size=64, lr=1e-3, seed=0)

    first = sum(res.loss_history[:3]) / 3.0
    last = sum(res.loss_history[-3:]) / 3.0
    assert last < first


def test_matched_pairs_rank_first():
    # For each view_a embedding, its true partner should be the most similar
    # view_b embedding more often than chance after training.
    a, b = make_paired_data(n=256, dim_a=64, dim_b=48, noise=0.3, seed=1)
    res = train_aligner(a, b, epochs=30, batch_size=64, lr=1e-3, seed=1)

    res.model.eval()
    with torch.no_grad():
        ea, eb = res.model(a, b)
        sim = ea @ eb.t()
        top1 = sim.argmax(dim=1)
        correct = (top1 == torch.arange(sim.shape[0])).float().mean()

    assert float(correct) > 0.5

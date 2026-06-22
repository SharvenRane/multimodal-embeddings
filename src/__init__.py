"""Multimodal embedding alignment package."""

from .data import make_paired_data, PairedDataset
from .model import ProjectionHead, MultimodalAligner
from .losses import info_nce_loss
from .train import train_aligner, mean_matched_minus_mismatched

__all__ = [
    "make_paired_data",
    "PairedDataset",
    "ProjectionHead",
    "MultimodalAligner",
    "info_nce_loss",
    "train_aligner",
    "mean_matched_minus_mismatched",
]

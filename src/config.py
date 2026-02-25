"""
TACNet Configuration
All hyperparameters and paths in one place.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    data_root: str = "./data"
    results_dir: str = "./results"
    checkpoints_dir: str = "./checkpoints"

    # ── Dataset ───────────────────────────────────────────────────────────────
    num_classes: int = 10
    image_size: int = 32
    num_workers: int = 4
    seed: int = 42

    # ── Classifier Training ───────────────────────────────────────────────────
    clf_epochs: int = 50
    clf_lr: float = 0.1
    clf_momentum: float = 0.9
    clf_weight_decay: float = 5e-4
    clf_batch_size: int = 128

    # ── Compressor Training ───────────────────────────────────────────────────
    cmp_epochs: int = 100
    cmp_lr: float = 1e-3
    cmp_batch_size: int = 128
    cmp_latent_channels: int = 8     # C: controls latent spatial depth

    # ── Loss Weights ──────────────────────────────────────────────────────────
    alpha: float = 1.0   # reconstruction loss weight
    beta: float = 0.5    # task (classification) loss weight
    gamma: float = 0.01  # rate proxy loss weight

    # ── Multi-Rate Experiments ────────────────────────────────────────────────
    # Four γ values → 4 operating points on the BPP-Accuracy curve
    gamma_values: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.1]
    )

    # ── Quick Mode (CPU-friendly reduced epochs) ──────────────────────────────
    quick_clf_epochs: int = 15
    quick_cmp_epochs: int = 30

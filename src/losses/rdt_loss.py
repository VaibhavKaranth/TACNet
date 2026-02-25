"""
Rate–Distortion–Task (RDT) Loss.

L = α · L_rec  +  β · L_task  +  γ · L_rate

  L_rec  = MSE(x_hat, x)                    — pixel reconstruction fidelity
  L_task = CrossEntropy(classifier(x_hat), y) — classification preservation
  L_rate = mean(|z|)                          — bitrate proxy (penalises large latents)

Setting β=0 reduces TACNet to a reconstruction-only baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDTLoss(nn.Module):
    """Rate–Distortion–Task loss.

    Args:
        alpha : reconstruction weight  (default 1.0)
        beta  : task weight            (default 0.5; set 0 for baseline)
        gamma : rate weight            (default 0.01; higher → lower BPP)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(
        self,
        x:      torch.Tensor,   # original images      [B, 3, H, W]
        x_hat:  torch.Tensor,   # reconstructed images [B, 3, H, W]
        z:      torch.Tensor,   # pre-quant latent      [B, C, Hz, Wz]
        logits: torch.Tensor,   # classifier output     [B, num_classes]
        labels: torch.Tensor,   # ground truth labels   [B]
    ):
        """Compute the joint RDT loss.

        Returns:
            total_loss : scalar tensor (backpropagate this)
            loss_dict  : dict of individual loss values (for logging)
        """
        # ── Reconstruction loss ────────────────────────────────────────────────
        L_rec = F.mse_loss(x_hat, x)

        # ── Task (classification) loss ─────────────────────────────────────────
        # CE on reconstructed images; gradients flow back through x_hat
        L_task = F.cross_entropy(logits, labels)

        # ── Rate proxy ─────────────────────────────────────────────────────────
        # Penalises large latent magnitudes → encourages sparse, compressible codes
        L_rate = z.abs().mean()

        # ── Joint loss ─────────────────────────────────────────────────────────
        total = self.alpha * L_rec + self.beta * L_task + self.gamma * L_rate

        loss_dict = {
            "total": total.item(),
            "rec":   L_rec.item(),
            "task":  L_task.item(),
            "rate":  L_rate.item(),
        }

        return total, loss_dict

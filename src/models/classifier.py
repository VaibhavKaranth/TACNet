"""
ResNet-18 adapted for CIFAR-10 (32×32 images).

Standard ResNet-18 uses a 7×7 stride-2 conv + max-pool which aggressively
downsamples 224×224 → 56×56.  For 32×32 inputs that would leave only 8×8
feature maps before the residual blocks — far too small.

Adaptation (standard practice for CIFAR):
  • Replace first conv: 7×7 stride-2  →  3×3 stride-1
  • Remove max-pool  (nn.Identity)
  • Replace FC head  : 1000-class → num_classes
"""

import torch.nn as nn
import torchvision.models as models


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Build a CIFAR-adapted ResNet-18.

    Args:
        num_classes: number of output classes (10 for CIFAR-10).

    Returns:
        nn.Module ready for training.
    """
    model = models.resnet18(weights=None)

    # ── Adapt first conv for 32×32 inputs ─────────────────────────────────────
    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64,
        kernel_size=3, stride=1, padding=1, bias=False,
    )

    # ── Remove max-pool (would halve 32→16 unnecessarily) ─────────────────────
    model.maxpool = nn.Identity()

    # ── Replace classifier head ───────────────────────────────────────────────
    model.fc = nn.Linear(512, num_classes)

    return model

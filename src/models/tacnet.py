"""
TACNet: Task-Aware Compression Network.

Combines a learned image compressor with a frozen classifier.
The classifier's gradient signal guides the encoder to retain
task-relevant (classification-useful) information.

Key design point:
  • Compressor works in [0, 1] pixel space.
  • Frozen classifier expects CIFAR-10-normalised inputs.
  • TACNet normalises x_hat internally before passing to the classifier.
  • During compressor training, gradients flow through normalize(x_hat)
    back into the decoder/encoder — the classifier parameters do NOT
    receive gradient updates (they are frozen).
"""

import os
import torch
import torch.nn as nn

from .compressor import ImageCompressor
from .classifier import build_resnet18_cifar

# CIFAR-10 normalisation constants
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)


class TACNet(nn.Module):
    """Task-Aware Compression Network.

    Args:
        latent_channels: latent feature-map depth (controls BPP).
        num_classes    : number of classification classes.
    """

    def __init__(self, latent_channels: int = 8, num_classes: int = 10):
        super().__init__()

        self.compressor = ImageCompressor(latent_channels)
        self.classifier = build_resnet18_cifar(num_classes)

        # Register normalisation tensors as buffers so they move with .to(device)
        self.register_buffer(
            "norm_mean",
            torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "norm_std",
            torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

        # Classifier is frozen by default
        self._freeze_classifier()

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise [0,1] images to CIFAR-10 classifier input space."""
        return (x - self.norm_mean) / self.norm_std

    # ── Freeze / unfreeze ─────────────────────────────────────────────────────

    def _freeze_classifier(self):
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

    def freeze_classifier(self):
        self._freeze_classifier()

    def unfreeze_classifier(self):
        for p in self.classifier.parameters():
            p.requires_grad = True
        self.classifier.train()

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def load_classifier(self, checkpoint_path: str, device: torch.device):
        """Load a pre-trained classifier checkpoint and freeze it.

        Args:
            checkpoint_path: path to classifier_best.pth
            device         : target device
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {checkpoint_path}\n"
                "Train the classifier first:  python main.py --mode train_classifier"
            )
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self._freeze_classifier()
        val_acc = ckpt.get("val_acc", "N/A")
        print(f"[TACNet] Classifier loaded from {checkpoint_path}  (val_acc={val_acc:.2f}%)")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """Full compression + task-prediction pass.

        Args:
            x: raw images [B, 3, 32, 32] in [0, 1]

        Returns:
            x_hat  : reconstructed images [B, 3, 32, 32] in [0, 1]
            z      : pre-quantisation latent [B, C, 8, 8]
            z_hat  : quantised latent [B, C, 8, 8]
            logits : classifier output on reconstructed images [B, num_classes]
        """
        # Step 1: compress and reconstruct
        x_hat, z, z_hat = self.compressor(x)

        # Step 2: normalise reconstructed image for the frozen classifier
        # Gradients flow through x_hat_norm → x_hat → decoder/encoder weights
        x_hat_norm = self._normalize(x_hat)

        # Step 3: classify (classifier params are frozen; gradients flow through input)
        logits = self.classifier(x_hat_norm)

        return x_hat, z, z_hat, logits

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_compressor_params(self):
        """Return only the compressor parameters (for the optimizer)."""
        return list(self.compressor.parameters())

    def train_mode(self):
        """Set compressor to train, classifier always stays in eval."""
        self.compressor.train()
        self.classifier.eval()

    def eval_mode(self):
        self.compressor.eval()
        self.classifier.eval()

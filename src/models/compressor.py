"""
Learned image compressor: Encoder → STE Quantizer → Decoder.

Architecture (CIFAR-10, 32×32):
  Encoder : 3×32×32 → C×8×8  (2× stride-2 downsampling)
  Decoder : C×8×8  → 3×32×32 (2× stride-2 upsampling)
  Quantizer: straight-through estimator (round in forward, identity in backward)
"""

import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

def conv_bn_lrelu(in_ch: int, out_ch: int, kernel: int = 3,
                  stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def deconv_bn_lrelu(in_ch: int, out_ch: int, kernel: int = 4,
                    stride: int = 2, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """CNN encoder: 3×32×32 → C×8×8.

    Two stride-2 convolutions halve spatial resolution twice (32→16→8).
    Final conv maps to `latent_channels` feature maps without further downsampling.
    """

    def __init__(self, latent_channels: int = 8):
        super().__init__()
        self.latent_channels = latent_channels

        self.net = nn.Sequential(
            # 3 × 32 × 32  →  64 × 32 × 32
            conv_bn_lrelu(3, 64, kernel=3, stride=1, padding=1),

            # 64 × 32 × 32  →  128 × 16 × 16
            conv_bn_lrelu(64, 128, kernel=3, stride=2, padding=1),

            # 128 × 16 × 16  →  256 × 8 × 8
            conv_bn_lrelu(128, 256, kernel=3, stride=2, padding=1),

            # 256 × 8 × 8  →  C × 8 × 8  (latent space; no activation)
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """CNN decoder: C×8×8 → 3×32×32.

    Two transposed convolutions double spatial resolution twice (8→16→32).
    Final conv + Sigmoid maps to RGB in [0, 1].
    """

    def __init__(self, latent_channels: int = 8):
        super().__init__()
        self.latent_channels = latent_channels

        self.net = nn.Sequential(
            # C × 8 × 8  →  256 × 8 × 8
            conv_bn_lrelu(latent_channels, 256, kernel=3, stride=1, padding=1),

            # 256 × 8 × 8  →  128 × 16 × 16
            deconv_bn_lrelu(256, 128, kernel=4, stride=2, padding=1),

            # 128 × 16 × 16  →  64 × 32 × 32
            deconv_bn_lrelu(128, 64, kernel=4, stride=2, padding=1),

            # 64 × 32 × 32  →  3 × 32 × 32  (output in [0, 1])
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)


# ── Straight-Through Estimator Quantizer ─────────────────────────────────────

class STEQuantizer(nn.Module):
    """Differentiable scalar quantizer via the Straight-Through Estimator.

    Forward  : z_hat = round(z)           — discrete, non-differentiable
    Backward : ∂loss/∂z ≈ ∂loss/∂z_hat   — gradient flows straight through

    Implementation trick:
        z_hat = z + (round(z) - z).detach()
    The detach() stops gradient from the (round(z) - z) correction term,
    so ∂z_hat/∂z = 1 during backprop.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_rounded = torch.round(z)
        # STE: forward uses rounded values, backward treats them as identity
        return z + (z_rounded - z).detach()


# ── Full Image Compressor ─────────────────────────────────────────────────────

class ImageCompressor(nn.Module):
    """Complete image compressor: Encoder → Quantizer → Decoder.

    Args:
        latent_channels: number of feature maps in the latent space (controls BPP).
    """

    def __init__(self, latent_channels: int = 8):
        super().__init__()
        self.encoder   = Encoder(latent_channels)
        self.quantizer = STEQuantizer()
        self.decoder   = Decoder(latent_channels)
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor):
        """Full encode → quantize → decode pass.

        Args:
            x: input images [B, 3, H, W] in [0, 1]

        Returns:
            x_hat : reconstructed images [B, 3, H, W] in [0, 1]
            z     : pre-quantisation latent [B, C, H/4, W/4]
            z_hat : quantised latent [B, C, H/4, W/4]
        """
        z     = self.encoder(x)
        z_hat = self.quantizer(z)
        x_hat = self.decoder(z_hat)
        return x_hat, z, z_hat

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + quantise only (no decode)."""
        return self.quantizer(self.encoder(x))

    def decompress(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Decode only from a quantised latent."""
        return self.decoder(z_hat)

"""
Evaluation metrics for TACNet.

  - PSNR   : Peak Signal-to-Noise Ratio (dB)
  - SSIM   : Structural Similarity Index
  - BPP    : Bits Per Pixel (empirical entropy estimate)
  - Accuracy: Top-1 classification on reconstructed images
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim


# ── PSNR ──────────────────────────────────────────────────────────────────────

def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    Args:
        x, x_hat: [B, C, H, W] float tensors in [0, 1]
    Returns:
        Average PSNR over the batch (dB).
    """
    x_hat = x_hat.clamp(0.0, 1.0)
    mse = F.mse_loss(x_hat, x).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


# ── SSIM ──────────────────────────────────────────────────────────────────────

def compute_ssim(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Mean Structural Similarity Index over a batch.

    Uses win_size=7 (appropriate for 32×32 CIFAR-10 images).

    Args:
        x, x_hat: [B, C, H, W] float tensors in [0, 1]
    Returns:
        Average SSIM score in [0, 1].
    """
    x_np     = x.detach().cpu().permute(0, 2, 3, 1).numpy()                  # B×H×W×C
    x_hat_np = x_hat.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()  # B×H×W×C

    scores = []
    for i in range(x_np.shape[0]):
        score = skimage_ssim(
            x_np[i], x_hat_np[i],
            channel_axis=2,
            data_range=1.0,
            win_size=7,        # must be odd and ≤ min(H, W); 7 ≤ 32 ✓
        )
        scores.append(score)

    return float(np.mean(scores))


# ── BPP ───────────────────────────────────────────────────────────────────────

def compute_bpp(z_hat: torch.Tensor, image_h: int, image_w: int) -> float:
    """Empirical bits-per-pixel estimate from quantised latent codes.

    Estimates the Shannon entropy of the quantised symbol distribution and
    multiplies by the number of symbols per image, divided by image pixels.

    BPP = (n_symbols × H(z_hat)) / (image_h × image_w)

    where H is the empirical entropy in bits.

    Args:
        z_hat   : quantised latent [B, C, Hz, Wz]
        image_h : original image height
        image_w : original image width
    Returns:
        Average BPP over the batch.
    """
    B, C, Hz, Wz = z_hat.shape
    n_symbols = C * Hz * Wz   # symbols per image
    n_pixels  = image_h * image_w

    bpp_list = []
    z_np = z_hat.detach().cpu().numpy()

    for b in range(B):
        symbols = z_np[b].flatten()
        unique_vals, counts = np.unique(symbols, return_counts=True)
        probs = counts / counts.sum()
        # Shannon entropy (bits)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        bpp_list.append(n_symbols * entropy / n_pixels)

    return float(np.mean(bpp_list))


def compute_bpp_theoretical(
    latent_channels: int, latent_h: int, latent_w: int,
    image_h: int, image_w: int, bits: int = 8,
) -> float:
    """Theoretical maximum BPP (assumes uniform 8-bit quantisation).

    BPP_max = (C × Hz × Wz × bits) / (H × W)
    """
    return (latent_channels * latent_h * latent_w * bits) / (image_h * image_w)


# ── Full model evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    test_loader,
    device: torch.device,
    image_size: int = 32,
) -> dict:
    """Evaluate a TACNet model on all metrics.

    Args:
        model      : TACNet instance (compressor + frozen classifier)
        test_loader: DataLoader returning (x, y) with x in [0, 1]
        device     : compute device
        image_size : spatial dimension of CIFAR-10 images (32)

    Returns:
        dict with keys: accuracy, bpp, psnr, ssim
    """
    model.eval_mode()

    all_preds  = []
    all_labels = []
    all_psnr   = []
    all_ssim   = []
    all_bpp    = []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        x_hat, z, z_hat, logits = model(x)

        # Classification accuracy
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        # Image quality metrics (computed on CPU numpy)
        all_psnr.append(compute_psnr(x, x_hat))
        all_ssim.append(compute_ssim(x, x_hat))

        # BPP
        all_bpp.append(compute_bpp(z_hat, image_size, image_size))

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy   = 100.0 * (all_preds == all_labels).mean()

    return {
        "accuracy": float(accuracy),
        "psnr":     float(np.mean(all_psnr)),
        "ssim":     float(np.mean(all_ssim)),
        "bpp":      float(np.mean(all_bpp)),
    }

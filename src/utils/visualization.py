"""
Visualization utilities for TACNet results.

Produces:
  1. Accuracy vs BPP curve (TACNet vs Baseline)
  2. PSNR vs BPP curve (tradeoff)
  3. Qualitative grid (original / baseline / TACNet at same bitrate)
  4. Training loss curves
  5. Console results table
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.data.dataset import CIFAR10_CLASSES

sns.set_style("whitegrid")
PALETTE = {"tacnet": "#2563EB", "baseline": "#DC2626"}


# ── Helper ────────────────────────────────────────────────────────────────────

def _sort_by_bpp(results):
    """Sort result dicts by ascending BPP."""
    return sorted(results, key=lambda r: r["bpp"])


# ── Plot 1: Accuracy vs BPP ───────────────────────────────────────────────────

def plot_accuracy_vs_bpp(tacnet_results, baseline_results, save_path):
    """Accuracy vs BPP curve — the primary publishable result."""
    fig, ax = plt.subplots(figsize=(8, 6))

    t = _sort_by_bpp(tacnet_results)
    b = _sort_by_bpp(baseline_results)

    ax.plot(
        [r["bpp"] for r in t], [r["accuracy"] for r in t],
        "o-", color=PALETTE["tacnet"], linewidth=2.5, markersize=9,
        label="TACNet  (α·Lrec + β·Ltask + γ·Lrate)",
    )
    ax.plot(
        [r["bpp"] for r in b], [r["accuracy"] for r in b],
        "s--", color=PALETTE["baseline"], linewidth=2.5, markersize=9,
        label="Baseline  (reconstruction only, β=0)",
    )

    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=13)
    ax.set_ylabel("Top-1 Classification Accuracy (%)", fontsize=13)
    ax.set_title("TACNet: Accuracy vs BPP", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Plot 2: PSNR vs BPP ───────────────────────────────────────────────────────

def plot_psnr_vs_bpp(tacnet_results, baseline_results, save_path):
    """PSNR vs BPP — shows the reconstruction quality tradeoff."""
    fig, ax = plt.subplots(figsize=(8, 6))

    t = _sort_by_bpp(tacnet_results)
    b = _sort_by_bpp(baseline_results)

    ax.plot(
        [r["bpp"] for r in t], [r["psnr"] for r in t],
        "o-", color=PALETTE["tacnet"], linewidth=2.5, markersize=9,
        label="TACNet",
    )
    ax.plot(
        [r["bpp"] for r in b], [r["psnr"] for r in b],
        "s--", color=PALETTE["baseline"], linewidth=2.5, markersize=9,
        label="Baseline",
    )

    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=13)
    ax.set_ylabel("PSNR (dB)", fontsize=13)
    ax.set_title("TACNet: PSNR vs BPP (Tradeoff)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Plot 3: Qualitative Grid ──────────────────────────────────────────────────

@torch.no_grad()
def plot_qualitative_grid(
    model_tacnet,
    model_baseline,
    test_loader,
    device,
    save_path,
    n_images: int = 8,
):
    """Side-by-side: original / baseline compressed / TACNet compressed."""
    model_tacnet.eval_mode()
    model_baseline.eval_mode()

    # Grab one batch
    x, y = next(iter(test_loader))
    x  = x[:n_images].to(device)
    y  = y[:n_images]

    x_hat_t, _, _, _ = model_tacnet(x)
    x_hat_b, _, _, _ = model_baseline(x)

    to_np = lambda t: t.cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    imgs_orig     = to_np(x)
    imgs_baseline = to_np(x_hat_b)
    imgs_tacnet   = to_np(x_hat_t)

    row_labels = ["Original", "Baseline\ncompressed", "TACNet\ncompressed"]
    rows       = [imgs_orig, imgs_baseline, imgs_tacnet]

    fig, axes = plt.subplots(3, n_images, figsize=(2.2 * n_images, 7.5))
    fig.patch.set_facecolor("#f8fafc")

    for row_idx, (label, row_imgs) in enumerate(zip(row_labels, rows)):
        for col_idx in range(n_images):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.clip(row_imgs[col_idx], 0, 1), interpolation="nearest")
            ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10, rotation=90,
                              labelpad=40, va="center")
            if row_idx == 0:
                cls_name = CIFAR10_CLASSES[y[col_idx].item()]
                ax.set_title(cls_name, fontsize=9, pad=4)

    plt.suptitle(
        "Qualitative Comparison: Original vs Compressed Images",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Plot 4: Training curves ───────────────────────────────────────────────────

def plot_training_history(history: dict, title: str, save_path: str):
    """Loss component curves over training epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["total"]) + 1)

    axes[0].plot(epochs, history["total"], color="#1e293b", linewidth=2)
    axes[0].set_title("Total Loss")

    axes[1].plot(epochs, history["rec"],  label="Reconstruction", color="#16a34a", linewidth=2)
    axes[1].plot(epochs, history["task"], label="Task (CE)",       color="#dc2626", linewidth=2)
    axes[1].set_title("Component Losses")
    axes[1].legend()

    axes[2].plot(epochs, history["rate"], color="#7c3aed", linewidth=2)
    axes[2].set_title("Rate Proxy  mean(|z|)")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.35)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Plot 5: SSIM vs BPP ───────────────────────────────────────────────────────

def plot_ssim_vs_bpp(tacnet_results, baseline_results, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    t = _sort_by_bpp(tacnet_results)
    b = _sort_by_bpp(baseline_results)

    ax.plot(
        [r["bpp"] for r in t], [r["ssim"] for r in t],
        "o-", color=PALETTE["tacnet"], linewidth=2.5, markersize=9, label="TACNet",
    )
    ax.plot(
        [r["bpp"] for r in b], [r["ssim"] for r in b],
        "s--", color=PALETTE["baseline"], linewidth=2.5, markersize=9, label="Baseline",
    )

    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=13)
    ax.set_ylabel("SSIM", fontsize=13)
    ax.set_title("TACNet: SSIM vs BPP", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Plot 6: Confusion Matrix ─────────────────────────────────────────────────

def plot_confusion_matrix(cm_tacnet, cm_baseline, gamma: float, save_path: str):
    """Side-by-side normalised confusion matrices for TACNet vs Baseline.

    Corresponds to PDF requirement: 'confusion matrix change vs baseline'.

    Args:
        cm_tacnet  : (10, 10) int array
        cm_baseline: (10, 10) int array
        gamma      : rate weight used (for title)
        save_path  : output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#f8fafc")

    def _norm(cm):
        row_sums = cm.sum(axis=1, keepdims=True).clip(1)
        return cm.astype(float) / row_sums

    for ax, cm, title, color in zip(
        axes,
        [_norm(cm_tacnet), _norm(cm_baseline)],
        [f"TACNet  (β=0.5, γ={gamma})", f"Baseline (β=0, γ={gamma}) — Ablation"],
        [PALETTE["tacnet"], PALETTE["baseline"]],
    ):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CIFAR10_CLASSES)))
        ax.set_yticks(range(len(CIFAR10_CLASSES)))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)

        # Annotate cells
        for i in range(len(CIFAR10_CLASSES)):
            for j in range(len(CIFAR10_CLASSES)):
                val = cm[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Confusion Matrices on Reconstructed Images\n"
        "(Ablation: removing task loss β=0 causes accuracy degradation)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Plot 7: Paper-ready results table as PNG ──────────────────────────────────

def save_results_table_image(tacnet_results, baseline_results, gamma_values, save_path: str):
    """Save the Accuracy/PSNR/SSIM results table as a PNG for the conference paper.

    Matches PDF requirement:
      'Table: Accuracy / PSNR / SSIM at 3 bitrate levels (low/med/high)'
    """
    level_labels = {0: "Low Compression", 1: "Medium Compression", 2: "High Compression"}
    rows = []
    for i, gamma in enumerate(gamma_values):
        label = level_labels.get(i, f"γ={gamma}")
        if i < len(tacnet_results) and i < len(baseline_results):
            t = tacnet_results[i]
            b = baseline_results[i]
            rows.append([label, "TACNet (β=0.5)",
                         f"{t['bpp']:.3f}", f"{t['accuracy']:.1f}%",
                         f"{t['psnr']:.2f}", f"{t['ssim']:.4f}"])
            rows.append(["",    "Baseline (β=0) — Ablation",
                         f"{b['bpp']:.3f}", f"{b['accuracy']:.1f}%",
                         f"{b['psnr']:.2f}", f"{b['ssim']:.4f}"])

    col_labels = ["Bitrate Level", "Method", "BPP ↓", "Accuracy ↑", "PSNR dB ↑", "SSIM ↑"]

    fig, ax = plt.subplots(figsize=(13, max(3, 1 + 0.55 * len(rows))))
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2563EB")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style TACNet rows (odd rows in each pair)
    for i in range(1, len(rows) + 1):
        is_tacnet = (i % 2 == 1)
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#dbeafe" if is_tacnet else "#fee2e2")

    plt.title(
        "TACNet vs Baseline — CIFAR-10 Compression Results\n"
        "(Baseline = Ablation: task loss β removed)",
        fontsize=13, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {save_path}")


# ── Console results table ─────────────────────────────────────────────────────

def print_results_table(tacnet_results, baseline_results, gamma_values):
    """Print a formatted ASCII comparison table to stdout."""
    W = 85
    print()
    print("=" * W)
    print("  RESULTS TABLE: TACNet vs Baseline Compression on CIFAR-10")
    print("=" * W)
    header = f"{'γ':>8}  {'Method':>10}  {'BPP':>7}  {'Acc (%)':>9}  {'PSNR (dB)':>11}  {'SSIM':>7}"
    print(header)
    print("-" * W)

    for i, gamma in enumerate(gamma_values):
        if i < len(tacnet_results) and i < len(baseline_results):
            t = tacnet_results[i]
            b = baseline_results[i]
            print(
                f"{gamma:>8.4f}  {'TACNet':>10}  {t['bpp']:>7.3f}  "
                f"{t['accuracy']:>9.2f}  {t['psnr']:>11.2f}  {t['ssim']:>7.4f}"
            )
            print(
                f"{gamma:>8.4f}  {'Baseline':>10}  {b['bpp']:>7.3f}  "
                f"{b['accuracy']:>9.2f}  {b['psnr']:>11.2f}  {b['ssim']:>7.4f}"
            )
            print("-" * W)

    print("=" * W)
    print()

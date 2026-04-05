"""
Full TACNet pipeline: classifier training → multi-rate compressor training → evaluation → plots.

Runs automatically when called:
    python main.py --mode run_all
    python experiments/run_all.py
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn

# Allow running from project root or experiments/ directory
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import Config
from src.utils.device import get_device
from src.data.dataset import get_cifar10_loaders, get_raw_cifar10_loaders
from src.models.classifier import build_resnet18_cifar
from src.models.tacnet import TACNet
from src.train.train_classifier import train_classifier, evaluate_classifier
from src.train.train_tacnet import train_tacnet
from src.evaluate.metrics import evaluate_model
from src.utils.visualization import (
    plot_accuracy_vs_bpp,
    plot_psnr_vs_bpp,
    plot_ssim_vs_bpp,
    plot_qualitative_grid,
    plot_training_history,
    print_results_table,
)


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_all(config: Config = None, quick: bool = False):
    """End-to-end TACNet experiment pipeline.

    Args:
        config : Config object (uses defaults if None)
        quick  : reduce epochs for CPU-friendly runs
    """
    if config is None:
        config = Config()

    if quick:
        config.clf_epochs = config.quick_clf_epochs
        config.cmp_epochs = config.quick_cmp_epochs
        print(f"[Quick mode] Epochs → classifier={config.clf_epochs}, compressor={config.cmp_epochs}")

    _set_seed(config.seed)
    device = get_device()

    os.makedirs(config.results_dir,     exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  PHASE 1: ResNet-18 Classifier Training")
    print("=" * 65)

    train_loader_norm, val_loader_norm, test_loader_norm = get_cifar10_loaders(
        config.data_root, config.clf_batch_size, config.num_workers,
    )
    classifier_path = os.path.join(config.checkpoints_dir, "classifier_best.pth")

    if os.path.exists(classifier_path):
        print(f"[Phase 1] Checkpoint found — loading: {classifier_path}")
        classifier = build_resnet18_cifar(config.num_classes).to(device)
        ckpt = torch.load(classifier_path, map_location=device)
        classifier.load_state_dict(ckpt["model_state_dict"])
        print(f"[Phase 1] Loaded (val_acc={ckpt.get('val_acc', 'N/A'):.2f}%)")
    else:
        classifier = build_resnet18_cifar(config.num_classes)
        classifier, clf_history = train_classifier(
            classifier, train_loader_norm, val_loader_norm, config, device,
        )
        # Plot classifier training curves
        plot_training_history(
            {"total": clf_history["train_loss"],
             "rec":   clf_history["train_loss"],
             "task":  clf_history["val_loss"],
             "rate":  [0.0] * len(clf_history["train_loss"])},
            "Classifier Training History",
            os.path.join(config.results_dir, "classifier_training.png"),
        )

    # Final test accuracy
    classifier = classifier.to(device)
    _, test_acc = evaluate_classifier(
        classifier, test_loader_norm, nn.CrossEntropyLoss(), device,
    )
    print(f"[Phase 1] Classifier test accuracy: {test_acc:.2f}%\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Multi-Rate Compressor Training
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  PHASE 2: Multi-Rate Compressor Training")
    print(f"  γ values : {config.gamma_values}")
    print(f"  Models   : TACNet (β={config.beta}) + Baseline (β=0.0) per γ")
    print("=" * 65)

    train_loader_raw, test_loader_raw = get_raw_cifar10_loaders(
        config.data_root, config.cmp_batch_size, config.num_workers,
    )

    trained_tacnets   = []   # list of (model, gamma)
    trained_baselines = []   # list of (model, gamma)

    for gamma in config.gamma_values:
        g_tag = f"{gamma:.4f}".replace(".", "_")

        # ── TACNet ────────────────────────────────────────────────────────────
        print(f"\n── TACNet  | γ={gamma:.4f}  β={config.beta} ──────────────────────")
        tacnet_name = f"tacnet_gamma{g_tag}"
        tacnet_ckpt = os.path.join(config.checkpoints_dir, f"{tacnet_name}.pth")

        tacnet = TACNet(config.cmp_latent_channels, config.num_classes).to(device)
        tacnet.load_classifier(classifier_path, device)

        if os.path.exists(tacnet_ckpt):
            print(f"  Loading: {tacnet_ckpt}")
            data = torch.load(tacnet_ckpt, map_location=device)
            tacnet.compressor.load_state_dict(data["compressor_state_dict"])
        else:
            tacnet, t_history = train_tacnet(
                tacnet, train_loader_raw, config, device,
                gamma=gamma, beta=config.beta,
                experiment_name=tacnet_name,
            )
            plot_training_history(
                t_history,
                f"TACNet Training  γ={gamma:.4f}",
                os.path.join(config.results_dir, f"history_{tacnet_name}.png"),
            )
        trained_tacnets.append((tacnet, gamma))

        # ── Baseline ──────────────────────────────────────────────────────────
        print(f"\n── Baseline | γ={gamma:.4f}  β=0.0 ──────────────────────────────")
        baseline_name = f"baseline_gamma{g_tag}"
        baseline_ckpt = os.path.join(config.checkpoints_dir, f"{baseline_name}.pth")

        baseline = TACNet(config.cmp_latent_channels, config.num_classes).to(device)
        baseline.load_classifier(classifier_path, device)

        if os.path.exists(baseline_ckpt):
            print(f"  Loading: {baseline_ckpt}")
            data = torch.load(baseline_ckpt, map_location=device)
            baseline.compressor.load_state_dict(data["compressor_state_dict"])
        else:
            baseline, b_history = train_tacnet(
                baseline, train_loader_raw, config, device,
                gamma=gamma, beta=0.0,
                experiment_name=baseline_name,
            )
            plot_training_history(
                b_history,
                f"Baseline Training  γ={gamma:.4f}",
                os.path.join(config.results_dir, f"history_{baseline_name}.png"),
            )
        trained_baselines.append((baseline, gamma))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  PHASE 3: Evaluation on CIFAR-10 Test Set")
    print("=" * 65)

    tacnet_results   = []
    baseline_results = []

    for (model, gamma) in trained_tacnets:
        print(f"\n  Evaluating TACNet  γ={gamma:.4f} ...")
        res = evaluate_model(model, test_loader_raw, device, config.image_size)
        res["gamma"] = gamma
        tacnet_results.append(res)
        print(f"    Acc={res['accuracy']:.2f}%  BPP={res['bpp']:.3f}  "
              f"PSNR={res['psnr']:.2f}dB  SSIM={res['ssim']:.4f}")

    for (model, gamma) in trained_baselines:
        print(f"\n  Evaluating Baseline γ={gamma:.4f} ...")
        res = evaluate_model(model, test_loader_raw, device, config.image_size)
        res["gamma"] = gamma
        baseline_results.append(res)
        print(f"    Acc={res['accuracy']:.2f}%  BPP={res['bpp']:.3f}  "
              f"PSNR={res['psnr']:.2f}dB  SSIM={res['ssim']:.4f}")

    # Save raw results as JSON
    results_json = os.path.join(config.results_dir, "results.json")
    with open(results_json, "w") as f:
        json.dump({"tacnet": tacnet_results, "baseline": baseline_results}, f, indent=2)
    print(f"\n[Phase 3] Results saved: {results_json}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Visualization
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  PHASE 4: Generating Plots")
    print("=" * 65)

    plot_accuracy_vs_bpp(
        tacnet_results, baseline_results,
        os.path.join(config.results_dir, "accuracy_vs_bpp.png"),
    )
    plot_psnr_vs_bpp(
        tacnet_results, baseline_results,
        os.path.join(config.results_dir, "psnr_vs_bpp.png"),
    )
    plot_ssim_vs_bpp(
        tacnet_results, baseline_results,
        os.path.join(config.results_dir, "ssim_vs_bpp.png"),
    )

    # Qualitative grid: pick the mid-γ model
    mid = len(trained_tacnets) // 2
    plot_qualitative_grid(
        trained_tacnets[mid][0], trained_baselines[mid][0],
        test_loader_raw, device,
        os.path.join(config.results_dir, "qualitative_grid.png"),
    )

    # Print ASCII table
    print_results_table(tacnet_results, baseline_results, config.gamma_values)

    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print(f"  All experiments complete!")
    print(f"  Results  : {os.path.abspath(config.results_dir)}")
    print(f"  Checkpts : {os.path.abspath(config.checkpoints_dir)}")
    print("=" * 65)

    return {"tacnet": tacnet_results, "baseline": baseline_results}


if __name__ == "__main__":
    run_all()

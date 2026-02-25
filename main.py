"""
TACNet — Task-Aware Image Compression
CLI entry point.

Usage examples:
  python main.py --mode run_all                          # full pipeline
  python main.py --mode run_all --quick                  # fast CPU run
  python main.py --mode train_classifier
  python main.py --mode train_tacnet --gamma 0.01 --beta 0.5
  python main.py --mode train_tacnet --gamma 0.01 --beta 0.0 --name baseline_g001
  python main.py --mode evaluate   --name tacnet_gamma0_0100
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="TACNet: Task-Aware Image Compression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["run_all", "train_classifier", "train_tacnet", "evaluate"],
        default="run_all",
        help="Operation mode",
    )
    p.add_argument("--gamma",   type=float, default=0.01,
                   help="Rate loss weight γ  (for train_tacnet / evaluate)")
    p.add_argument("--beta",    type=float, default=0.5,
                   help="Task loss weight β  (0 = baseline)")
    p.add_argument("--latent-channels", type=int, default=8,
                   help="Latent channels C in encoder/decoder")
    p.add_argument("--clf-epochs",  type=int, default=None,
                   help="Override classifier training epochs")
    p.add_argument("--cmp-epochs",  type=int, default=None,
                   help="Override compressor training epochs")
    p.add_argument("--data-root",       default="./data")
    p.add_argument("--results-dir",     default="./results")
    p.add_argument("--checkpoints-dir", default="./checkpoints")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--name",  type=str, default=None,
                   help="Experiment name (for train_tacnet / evaluate)")
    p.add_argument("--quick", action="store_true",
                   help="Reduced epochs for CPU-friendly testing")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    from src.config import Config
    config = Config(
        data_root=args.data_root,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        seed=args.seed,
        cmp_latent_channels=args.latent_channels,
        alpha=1.0,
        beta=args.beta,
        gamma=args.gamma,
    )
    if args.clf_epochs is not None:
        config.clf_epochs = args.clf_epochs
    if args.cmp_epochs is not None:
        config.cmp_epochs = args.cmp_epochs

    # ── Seed ──────────────────────────────────────────────────────────────────
    import torch, numpy as np
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    from src.utils.device import get_device
    device = get_device()

    os.makedirs(config.results_dir,     exist_ok=True)
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    if args.mode == "run_all":
        from experiments.run_all import run_all
        run_all(config=config, quick=args.quick)

    # ══════════════════════════════════════════════════════════════════════════
    elif args.mode == "train_classifier":
        from src.data.dataset import get_cifar10_loaders
        from src.models.classifier import build_resnet18_cifar
        from src.train.train_classifier import train_classifier

        train_loader, val_loader, _ = get_cifar10_loaders(
            config.data_root, config.clf_batch_size, config.num_workers,
        )
        if args.quick:
            config.clf_epochs = config.quick_clf_epochs

        model = build_resnet18_cifar(config.num_classes)
        train_classifier(model, train_loader, val_loader, config, device)

    # ══════════════════════════════════════════════════════════════════════════
    elif args.mode == "train_tacnet":
        from src.data.dataset import get_raw_cifar10_loaders
        from src.models.tacnet import TACNet
        from src.train.train_tacnet import train_tacnet

        classifier_path = os.path.join(config.checkpoints_dir, "classifier_best.pth")
        if not os.path.exists(classifier_path):
            print(f"ERROR: Classifier checkpoint not found at {classifier_path}")
            print("       Run first:  python main.py --mode train_classifier")
            sys.exit(1)

        train_loader, _ = get_raw_cifar10_loaders(
            config.data_root, config.cmp_batch_size, config.num_workers,
        )
        if args.quick:
            config.cmp_epochs = config.quick_cmp_epochs

        exp_name = args.name or (
            f"tacnet_gamma{args.gamma:.4f}_beta{args.beta:.2f}".replace(".", "_")
        )

        model = TACNet(config.cmp_latent_channels, config.num_classes).to(device)
        model.load_classifier(classifier_path, device)
        train_tacnet(
            model, train_loader, config, device,
            gamma=args.gamma, beta=args.beta,
            experiment_name=exp_name,
        )

    # ══════════════════════════════════════════════════════════════════════════
    elif args.mode == "evaluate":
        from src.data.dataset import get_raw_cifar10_loaders
        from src.models.tacnet import TACNet
        from src.evaluate.metrics import evaluate_model
        import json

        classifier_path = os.path.join(config.checkpoints_dir, "classifier_best.pth")
        exp_name = args.name or (
            f"tacnet_gamma{args.gamma:.4f}_beta{args.beta:.2f}".replace(".", "_")
        )
        ckpt_path = os.path.join(config.checkpoints_dir, f"{exp_name}.pth")

        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        _, test_loader = get_raw_cifar10_loaders(
            config.data_root, config.cmp_batch_size, config.num_workers,
        )

        model = TACNet(config.cmp_latent_channels, config.num_classes).to(device)
        model.load_classifier(classifier_path, device)
        data = torch.load(ckpt_path, map_location=device)
        model.compressor.load_state_dict(data["compressor_state_dict"])

        results = evaluate_model(model, test_loader, device, config.image_size)

        print("\n[Evaluation Results]")
        print(f"  Experiment : {exp_name}")
        print(f"  Accuracy   : {results['accuracy']:.2f}%")
        print(f"  BPP        : {results['bpp']:.4f}")
        print(f"  PSNR       : {results['psnr']:.2f} dB")
        print(f"  SSIM       : {results['ssim']:.4f}")

        out = os.path.join(config.results_dir, f"eval_{exp_name}.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()

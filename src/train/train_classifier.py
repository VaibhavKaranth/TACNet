"""
Phase 1: Train ResNet-18 classifier on CIFAR-10.

Saves best checkpoint to checkpoints/classifier_best.pth.
Returns trained model and training history dict.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_classifier(model, train_loader, val_loader, config, device):
    """Train CIFAR-10 classifier.

    Args:
        model        : ResNet-18 (CIFAR-adapted)
        train_loader : normalised training DataLoader
        val_loader   : normalised validation DataLoader
        config       : Config object
        device       : compute device

    Returns:
        model   : trained model (best weights loaded)
        history : dict with train/val loss and accuracy per epoch
    """
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.clf_lr,
        momentum=config.clf_momentum,
        weight_decay=config.clf_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.clf_epochs,
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(config.checkpoints_dir, "classifier_best.pth")

    for epoch in range(config.clf_epochs):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct    = 0
        total      = 0

        pbar = tqdm(
            train_loader,
            desc=f"Classifier [{epoch+1:3d}/{config.clf_epochs}]",
            ncols=90,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.1f}%",
            )

        scheduler.step()

        train_acc  = 100.0 * correct / total
        avg_loss   = train_loss / total

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch+1:3d} | "
            f"Train {train_acc:.2f}%  Val {val_acc:.2f}%  "
            f"LR {scheduler.get_last_lr()[0]:.5f}"
        )

        # ── Save best ─────────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                },
                ckpt_path,
            )

    # Reload best weights before returning
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\n[Classifier] Training complete — best val accuracy: {best_val_acc:.2f}%")
    print(f"[Classifier] Checkpoint saved: {ckpt_path}")
    return model, history


def _evaluate(model, loader, criterion, device):
    """Return (avg_loss, accuracy%) on a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total   += y.size(0)

    return total_loss / total, 100.0 * correct / total


# Public alias used by train_classifier.py and run_all.py
evaluate_classifier = _evaluate

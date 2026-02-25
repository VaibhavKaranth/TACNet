"""
Phase 2: Train TACNet image compressor.

Optimises only the encoder + decoder (classifier is frozen).
Joint RDT loss guides what information is retained.

Setting beta=0 trains the reconstruction-only baseline.
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from src.losses.rdt_loss import RDTLoss


def train_tacnet(
    model,
    train_loader,
    config,
    device: torch.device,
    gamma: float,
    beta: float,
    experiment_name: str = "tacnet",
):
    """Train the TACNet compressor with joint RDT loss.

    Args:
        model           : TACNet instance (classifier already loaded + frozen)
        train_loader    : raw [0,1] CIFAR-10 DataLoader
        config          : Config object
        device          : compute device
        gamma           : rate loss weight (controls BPP)
        beta            : task loss weight (0 → baseline, >0 → TACNet)
        experiment_name : used for checkpoint filename

    Returns:
        model   : trained TACNet
        history : dict of per-epoch loss components
    """
    model = model.to(device)

    # Only optimise compressor; classifier params have requires_grad=False
    optimizer = optim.Adam(model.get_compressor_params(), lr=config.cmp_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.cmp_epochs,
    )
    criterion = RDTLoss(alpha=config.alpha, beta=beta, gamma=gamma)

    history = {"loss": [], "rec": [], "task": [], "rate": []}
    os.makedirs(config.checkpoints_dir, exist_ok=True)

    epochs = config.cmp_epochs

    for epoch in range(epochs):
        model.train_mode()   # compressor train, classifier eval

        epoch_totals = {"loss": 0.0, "rec": 0.0, "task": 0.0, "rate": 0.0}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{experiment_name}] [{epoch+1:3d}/{epochs}]",
            ncols=100,
        )
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            x_hat, z, z_hat, logits = model(x)
            loss, loss_dict = criterion(x, x_hat, z, logits, y)
            loss.backward()
            optimizer.step()

            for k in epoch_totals:
                epoch_totals[k] += loss_dict[k]
            n_batches += 1

            pbar.set_postfix(
                L=f"{loss_dict['total']:.4f}",
                rec=f"{loss_dict['rec']:.4f}",
                task=f"{loss_dict['task']:.4f}",
                rate=f"{loss_dict['rate']:.4f}",
            )

        scheduler.step()

        for k in epoch_totals:
            history[k].append(epoch_totals[k] / n_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  [{experiment_name}] Epoch {epoch+1:3d} | "
                f"Loss={history['loss'][-1]:.4f}  "
                f"Rec={history['rec'][-1]:.4f}  "
                f"Task={history['task'][-1]:.4f}  "
                f"Rate={history['rate'][-1]:.4f}"
            )

    # ── Save checkpoint ────────────────────────────────────────────────────────
    ckpt_path = os.path.join(config.checkpoints_dir, f"{experiment_name}.pth")
    torch.save(
        {
            "compressor_state_dict": model.compressor.state_dict(),
            "gamma":   gamma,
            "beta":    beta,
            "history": history,
        },
        ckpt_path,
    )
    print(f"[{experiment_name}] Checkpoint saved: {ckpt_path}")
    return model, history

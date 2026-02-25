"""
CIFAR-10 data loading utilities.

Two loader families:
  - Normalized loaders  : for classifier training/evaluation
  - Raw [0,1] loaders   : for compressor training/evaluation
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# CIFAR-10 channel statistics (computed over training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ── Normalised loaders (for classifier) ───────────────────────────────────────

def get_cifar10_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
):
    """CIFAR-10 loaders with standard normalisation and augmentation.

    Returns train, val, test DataLoaders.
    Used for classifier training and evaluation.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Download on first run; cached afterwards
    full_train = datasets.CIFAR10(
        data_root, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        data_root, train=False, download=True, transform=test_transform
    )

    # Reproducible val split
    val_size   = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Val subset needs the test-time transform (no augmentation)
    val_set = datasets.CIFAR10(
        data_root, train=True, download=False, transform=test_transform
    )
    val_indices = val_subset.indices
    val_subset_clean = torch.utils.data.Subset(val_set, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset_clean, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ── Raw [0, 1] loaders (for compressor) ───────────────────────────────────────

def get_raw_cifar10_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
):
    """CIFAR-10 loaders with images in [0, 1] — no normalisation.

    Used for compressor training and evaluation.
    The TACNet model normalises internally before passing to the frozen classifier.
    """
    transform = transforms.ToTensor()   # Converts uint8 [0,255] → float [0,1]

    train_set = datasets.CIFAR10(
        data_root, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """Normalise a [0,1] CIFAR-10 image batch to classifier input space.

    Args:
        x: Tensor [B, 3, H, W] in [0, 1]
    Returns:
        Normalised tensor with CIFAR-10 mean/std subtracted.
    """
    mean = torch.tensor(CIFAR10_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

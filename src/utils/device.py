"""
Device auto-detection.
Priority: CUDA GPU > Apple MPS > CPU
"""

import multiprocessing
import torch


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Prints a clear summary so the user always knows what hardware is active.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print("[Device] ── GPU (CUDA) detected ──────────────────────────────")
        print(f"[Device]   Name  : {props.name}")
        print(f"[Device]   VRAM  : {props.total_memory / 1e9:.1f} GB")
        print(f"[Device]   CUDA  : {torch.version.cuda}")
        if torch.cuda.device_count() > 1:
            print(f"[Device]   Multi : {torch.cuda.device_count()} GPUs available (using cuda:0)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] ── Apple MPS detected ──────────────────────────────")
    else:
        device = torch.device("cpu")
        n_cores = multiprocessing.cpu_count()
        print("[Device] ── CPU only ────────────────────────────────────────")
        print(f"[Device]   Cores : {n_cores}")
        print("[Device]   Note  : Training will be slow. Use --quick to reduce epochs.")

    print(f"[Device]   PyTorch : {torch.__version__}")
    print(f"[Device]   Running on : {device}\n")
    return device


def is_gpu_available() -> bool:
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def device_summary() -> str:
    if torch.cuda.is_available():
        return f"GPU ({torch.cuda.get_device_name(0)})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"

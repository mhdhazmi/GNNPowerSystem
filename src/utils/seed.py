"""
Seed control utilities for reproducibility.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic algorithms in PyTorch
                      (may reduce performance but ensures reproducibility)
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        if deterministic:
            # Ensure deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_cuda: If True, prefer CUDA over MPS

    Returns:
        torch.device for computation
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get information about available compute devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info

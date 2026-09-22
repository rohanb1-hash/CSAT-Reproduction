"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Seed every RNG this project touches.

    Note on honesty: full bit-wise determinism on GPU also requires
    ``CUBLAS_WORKSPACE_CONFIG`` and can *change measured runtimes*, so we set
    deterministic algorithms for correctness experiments but the benchmarking
    module deliberately re-enables cuDNN autotuning (see utils/benchmarking.py).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA when available, otherwise CPU (Kaggle CPU-only fallback)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device | None = None) -> dict:
    """Environment fingerprint saved alongside every result file."""
    device = device or get_device()
    info = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        info.update(
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
            gpu_total_memory_GB=round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2
            ),
            gpu_capability=".".join(map(str, torch.cuda.get_device_capability(0))),
        )
    return info

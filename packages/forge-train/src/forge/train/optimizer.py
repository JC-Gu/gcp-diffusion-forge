"""Optimizer factory — selects the right optimizer for each backend.

CUDA:    AdamW8bit (bitsandbytes) when available, falls back to AdamW.
MPS/CPU: Standard torch.optim.AdamW (bitsandbytes is CUDA-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from forge.core.device import Backend


def build_optimizer(
    params,
    lr: float,
    backend: Backend,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.optim.Optimizer:
    """Return the best available optimizer for the given backend.

    On CUDA, tries bitsandbytes AdamW8bit first (halves optimizer memory).
    Falls back to standard AdamW if bitsandbytes is not installed.
    MPS and CPU always use standard AdamW.

    Args:
        params:       Iterable of parameters or param groups.
        lr:           Learning rate.
        backend:      Detected compute backend from forge.core.device.
        weight_decay: L2 regularisation weight.
        beta1:        Adam beta1 (first moment decay).
        beta2:        Adam beta2 (second moment decay).

    Returns:
        Configured optimizer instance.
    """
    betas = (beta1, beta2)

    if backend == Backend.CUDA:
        try:
            from bitsandbytes.optim import AdamW8bit  # optional dep

            return AdamW8bit(params, lr=lr, weight_decay=weight_decay, betas=betas)
        except ImportError:
            pass  # bitsandbytes not installed — fall through to AdamW

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)

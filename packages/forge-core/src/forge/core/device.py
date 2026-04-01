"""Device detection and backend-specific configuration.

All platform-specific logic lives here. Every other forge package imports
from this module — never call torch.cuda.is_available() elsewhere.

Supported backends:
- CUDA:  NVIDIA GPU (Ampere+ for bfloat16 + flash-attn; pre-Ampere uses xformers)
- MPS:   Apple Silicon (float16 only; no distributed, no flash-attn)
- CPU:   Fallback (float32 only)
"""

from __future__ import annotations

from enum import Enum

import torch


class Backend(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def get_device() -> tuple[Backend, torch.device]:
    """Detect the best available compute backend."""
    if torch.cuda.is_available():
        return Backend.CUDA, torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return Backend.MPS, torch.device("mps")
    return Backend.CPU, torch.device("cpu")


def get_dtype(backend: Backend) -> torch.dtype:
    """Return the recommended default dtype for the given backend.

    - CUDA (Ampere+): bfloat16 — preferred for diffusion models
    - MPS: float16 — bfloat16 is unreliable on Apple Silicon
    - CPU: float32
    """
    if backend == Backend.CUDA:
        return torch.bfloat16
    if backend == Backend.MPS:
        return torch.float16
    return torch.float32


def get_attn_backend(backend: Backend) -> str:
    """Return the recommended attention implementation for the given backend.

    Returns one of: "flash_attn", "xformers", "sdpa"
    - flash_attn: CUDA Ampere+ (compute capability >= 8.0)
    - xformers:   CUDA pre-Ampere (e.g. T4)
    - sdpa:       MPS and CPU (torch.nn.functional.scaled_dot_product_attention)
    """
    if backend == Backend.CUDA and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability()
        if cc >= (8, 0):  # Ampere and newer (A100, H100, A10, L4, ...)
            return "flash_attn"
        return "xformers"  # Pre-Ampere (T4, V100, ...)
    return "sdpa"


def is_distributed() -> bool:
    """Return True if multi-GPU distributed training is available.

    MPS and CPU are always single-device; CUDA requires NCCL.
    """
    return torch.cuda.is_available() and torch.distributed.is_available()


def get_recommended_compile(backend: Backend) -> bool:
    """Return whether torch.compile is recommended for the given backend.

    torch.compile on MPS is limited and often slower; disable by default.
    """
    return backend == Backend.CUDA

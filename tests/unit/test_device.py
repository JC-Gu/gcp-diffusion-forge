"""Unit tests for forge.core.device — all pass without GPU."""

from __future__ import annotations

import torch
import pytest

from forge.core.device import (
    Backend,
    get_attn_backend,
    get_device,
    get_dtype,
    get_recommended_compile,
    is_distributed,
)


def test_get_device_returns_valid_backend() -> None:
    backend, device = get_device()
    assert isinstance(backend, Backend)
    assert isinstance(device, torch.device)


def test_get_device_backend_matches_device_type() -> None:
    backend, device = get_device()
    assert device.type == backend.value


def test_get_device_is_deterministic() -> None:
    assert get_device() == get_device()


def test_get_dtype_cpu() -> None:
    assert get_dtype(Backend.CPU) == torch.float32


def test_get_dtype_mps() -> None:
    assert get_dtype(Backend.MPS) == torch.float16


def test_get_dtype_cuda() -> None:
    assert get_dtype(Backend.CUDA) == torch.bfloat16


def test_get_attn_backend_cpu() -> None:
    assert get_attn_backend(Backend.CPU) == "sdpa"


def test_get_attn_backend_mps() -> None:
    assert get_attn_backend(Backend.MPS) == "sdpa"


def test_is_distributed_returns_bool() -> None:
    result = is_distributed()
    assert isinstance(result, bool)


def test_compile_disabled_on_mps() -> None:
    assert get_recommended_compile(Backend.MPS) is False


def test_compile_disabled_on_cpu() -> None:
    assert get_recommended_compile(Backend.CPU) is False


def test_compile_enabled_on_cuda() -> None:
    assert get_recommended_compile(Backend.CUDA) is True


@pytest.mark.cuda
def test_attn_backend_cuda_ampere_or_newer() -> None:
    backend_str = get_attn_backend(Backend.CUDA)
    assert backend_str in ("flash_attn", "xformers")


@pytest.mark.cuda
def test_is_distributed_true_on_cuda() -> None:
    assert is_distributed() is True

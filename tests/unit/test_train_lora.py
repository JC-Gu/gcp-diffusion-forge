"""Unit tests for forge.train utility functions — no GPU, no model downloads.

Tests cover the pure utility functions: optimizer factory, LoRA config building,
model injection, precision resolution, and checkpoint I/O. The training loop
(LoRATrainer.run) is covered by integration tests that use the tiny SD model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig

from forge.core.config import TrainingMethod
from forge.core.device import Backend
from forge.train.checkpoint import load_lora_weights, save_lora_weights
from forge.train.lora import (
    _DEFAULT_TARGET_MODULES,
    build_lora_config,
    inject_lora,
    resolve_training_precision,
)
from forge.train.optimizer import build_optimizer
from forge.train.types import TrainResult


# ── Tiny model fixture for LoRA injection and checkpoint tests ────────────────


class _TinyAttention(nn.Module):
    """Minimal model that mirrors the to_q/to_k/to_v/to_out.0 naming
    used in UNet attention blocks so LoRA target_modules work correctly."""

    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8, bias=False)
        self.to_k = nn.Linear(8, 8, bias=False)
        self.to_v = nn.Linear(8, 8, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(8, 8, bias=False)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


@pytest.fixture
def tiny_model() -> _TinyAttention:
    return _TinyAttention()


@pytest.fixture
def lora_model(tiny_model: _TinyAttention) -> nn.Module:
    cfg = build_lora_config(rank=4, alpha=4, method=TrainingMethod.LORA)
    return inject_lora(tiny_model, cfg)


# ── build_optimizer ───────────────────────────────────────────────────────────


def test_build_optimizer_mps_returns_adamw() -> None:
    model = nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), lr=1e-4, backend=Backend.MPS)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_cpu_returns_adamw() -> None:
    model = nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), lr=1e-4, backend=Backend.CPU)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_cuda_falls_back_to_adamw_without_bnb() -> None:
    """When bitsandbytes is not installed, CUDA should fall back to AdamW."""
    model = nn.Linear(4, 4)
    with patch.dict("sys.modules", {"bitsandbytes": None, "bitsandbytes.optim": None}):
        opt = build_optimizer(model.parameters(), lr=1e-4, backend=Backend.CUDA)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_cuda_uses_adam8bit_when_available() -> None:
    mock_bnb = MagicMock()
    mock_adam8bit = MagicMock(return_value=MagicMock(spec=torch.optim.Optimizer))
    mock_bnb.optim.AdamW8bit = mock_adam8bit

    model = nn.Linear(4, 4)
    with patch.dict("sys.modules", {"bitsandbytes": mock_bnb, "bitsandbytes.optim": mock_bnb.optim}):
        build_optimizer(model.parameters(), lr=1e-4, backend=Backend.CUDA)

    mock_adam8bit.assert_called_once()


def test_build_optimizer_respects_lr() -> None:
    model = nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), lr=5e-5, backend=Backend.CPU)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_build_optimizer_respects_weight_decay() -> None:
    model = nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), lr=1e-4, backend=Backend.CPU, weight_decay=0.05)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)


def test_build_optimizer_respects_betas() -> None:
    model = nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), lr=1e-4, backend=Backend.CPU, beta1=0.95, beta2=0.98)
    assert opt.param_groups[0]["betas"] == (0.95, 0.98)


# ── build_lora_config ─────────────────────────────────────────────────────────


def test_build_lora_config_sets_rank() -> None:
    cfg = build_lora_config(rank=16, alpha=16, method=TrainingMethod.LORA)
    assert cfg.r == 16


def test_build_lora_config_sets_alpha() -> None:
    cfg = build_lora_config(rank=8, alpha=32, method=TrainingMethod.LORA)
    assert cfg.lora_alpha == 32


def test_build_lora_config_lora_use_dora_false() -> None:
    cfg = build_lora_config(rank=4, alpha=4, method=TrainingMethod.LORA)
    assert cfg.use_dora is False


def test_build_lora_config_dora_use_dora_true() -> None:
    cfg = build_lora_config(rank=4, alpha=4, method=TrainingMethod.DORA)
    assert cfg.use_dora is True


def test_build_lora_config_default_target_modules() -> None:
    cfg = build_lora_config(rank=4, alpha=4, method=TrainingMethod.LORA)
    assert set(cfg.target_modules) == set(_DEFAULT_TARGET_MODULES)


def test_build_lora_config_custom_target_modules() -> None:
    custom = ["to_q", "to_v"]
    cfg = build_lora_config(rank=4, alpha=4, method=TrainingMethod.LORA, target_modules=custom)
    assert set(cfg.target_modules) == set(custom)


def test_build_lora_config_bias_none() -> None:
    cfg = build_lora_config(rank=4, alpha=4, method=TrainingMethod.LORA)
    assert cfg.bias == "none"


# ── inject_lora ───────────────────────────────────────────────────────────────


def test_inject_lora_freezes_all_base_params(lora_model: nn.Module) -> None:
    base_params = [p for n, p in lora_model.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad for p in base_params), \
        "All base parameters should be frozen after LoRA injection"


def test_inject_lora_trainable_lora_params_exist(lora_model: nn.Module) -> None:
    lora_params = [p for n, p in lora_model.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0
    assert any(p.requires_grad for p in lora_params)


def test_inject_lora_reduces_trainable_param_count(tiny_model: _TinyAttention) -> None:
    total_before = sum(p.numel() for p in tiny_model.parameters())

    cfg = build_lora_config(rank=2, alpha=2, method=TrainingMethod.LORA)
    lora_model = inject_lora(tiny_model, cfg)
    trainable_after = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    assert trainable_after < total_before


# ── resolve_training_precision ────────────────────────────────────────────────


def test_resolve_precision_mps_returns_no_autocast() -> None:
    mode, _ = resolve_training_precision("bf16", Backend.MPS)
    assert mode == "no"


def test_resolve_precision_mps_returns_float16() -> None:
    _, dtype = resolve_training_precision("bf16", Backend.MPS)
    assert dtype == torch.float16


def test_resolve_precision_cpu_returns_no_and_float32() -> None:
    mode, dtype = resolve_training_precision("bf16", Backend.CPU)
    assert mode == "no"
    assert dtype == torch.float32


def test_resolve_precision_cuda_bf16() -> None:
    mode, dtype = resolve_training_precision("bf16", Backend.CUDA)
    assert mode == "bf16"
    assert dtype == torch.bfloat16


def test_resolve_precision_cuda_fp16() -> None:
    mode, dtype = resolve_training_precision("fp16", Backend.CUDA)
    assert mode == "fp16"
    assert dtype == torch.float16


def test_resolve_precision_cuda_no() -> None:
    mode, dtype = resolve_training_precision("no", Backend.CUDA)
    assert mode == "no"
    assert dtype == torch.float32


# ── save_lora_weights / load_lora_weights ─────────────────────────────────────


def test_save_lora_weights_creates_file(lora_model: nn.Module, tmp_path: Path) -> None:
    path = save_lora_weights(lora_model, str(tmp_path), "test.safetensors")
    assert Path(path).exists()


def test_save_lora_weights_extension_is_safetensors(
    lora_model: nn.Module, tmp_path: Path
) -> None:
    path = save_lora_weights(lora_model, str(tmp_path))
    assert path.endswith(".safetensors")


def test_save_lora_weights_creates_output_dir(lora_model: nn.Module, tmp_path: Path) -> None:
    new_dir = tmp_path / "nested" / "checkpoints"
    save_lora_weights(lora_model, str(new_dir))
    assert new_dir.exists()


def test_save_lora_weights_only_contains_lora_keys(
    lora_model: nn.Module, tmp_path: Path
) -> None:
    from safetensors import safe_open

    path = save_lora_weights(lora_model, str(tmp_path), "weights.safetensors")
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
    assert all("lora_" in k for k in keys), \
        f"Expected only LoRA keys, got: {keys}"


def test_load_lora_weights_roundtrip(tmp_path: Path) -> None:
    """Save then reload weights; values must be identical."""
    model_a = inject_lora(_TinyAttention(), build_lora_config(4, 4, TrainingMethod.LORA))
    path = save_lora_weights(model_a, str(tmp_path), "roundtrip.safetensors")

    model_b = inject_lora(_TinyAttention(), build_lora_config(4, 4, TrainingMethod.LORA))
    load_lora_weights(model_b, path)

    for (na, pa), (nb, pb) in zip(
        sorted(model_a.named_parameters()), sorted(model_b.named_parameters())
    ):
        if "lora_" in na:
            assert torch.allclose(pa, pb), f"Mismatch in {na}"


# ── TrainResult ───────────────────────────────────────────────────────────────


def test_train_result_to_dict_includes_loss() -> None:
    r = TrainResult(
        model_id="test/model",
        method="lora",
        steps_completed=100,
        checkpoint_path="/tmp/ckpt.safetensors",
        elapsed_sec=60.0,
        final_loss=0.042,
    )
    d = r.to_dict()
    assert "final_loss" in d
    assert d["final_loss"] == pytest.approx(0.042, abs=1e-6)


def test_train_result_to_dict_excludes_none_loss() -> None:
    r = TrainResult(
        model_id="test/model",
        method="lora",
        steps_completed=100,
        checkpoint_path="/tmp/ckpt.safetensors",
        elapsed_sec=60.0,
    )
    assert "final_loss" not in r.to_dict()


def test_train_result_to_dict_excludes_empty_metadata() -> None:
    r = TrainResult(
        model_id="test/model",
        method="lora",
        steps_completed=100,
        checkpoint_path="/tmp/ckpt.safetensors",
        elapsed_sec=10.0,
    )
    assert "metadata" not in r.to_dict()


def test_train_result_to_dict_includes_metadata_when_set() -> None:
    r = TrainResult(
        model_id="test/model",
        method="lora",
        steps_completed=100,
        checkpoint_path="/tmp/ckpt.safetensors",
        elapsed_sec=10.0,
        metadata={"lora_rank": 16, "backend": "mps"},
    )
    d = r.to_dict()
    assert d["metadata"]["lora_rank"] == 16

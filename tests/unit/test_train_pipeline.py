"""Unit tests for forge.train.pipeline.TrainPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.core.config import (
    DataSpec,
    EvaluationSpec,
    JobMetadata,
    ModelArchitecture,
    ModelSpec,
    OptimizerSpec,
    OutputSpec,
    ResourceSpec,
    TrainingJobSpec,
    TrainingMethod,
    TrainingSpec,
)
from forge.train.pipeline import TrainPipeline
from forge.train.types import TrainResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spec(
    architecture: str = "sd15",
    method: str = "lora",
    output_dir: str = "/tmp/ckpt",
    data_source: str = "/tmp/data",
) -> TrainingJobSpec:
    return TrainingJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "TrainingJob",
        "metadata": {"name": "test-job"},
        "model": {"architecture": architecture, "base": "test/model"},
        "training": {
            "method": method,
            "loraRank": 4,
            "loraAlpha": 4,
            "steps": 10,
            "batchSize": 1,
        },
        "data": {"source": data_source},
        "output": {"checkpointDir": output_dir},
    })


def _fake_result() -> TrainResult:
    return TrainResult(
        model_id="test/model",
        method="lora",
        steps_completed=10,
        checkpoint_path="/tmp/ckpt/lora_weights.safetensors",
        elapsed_sec=5.0,
        final_loss=0.05,
    )


# ── dry_run validation ────────────────────────────────────────────────────────


def test_dry_run_valid_sd15_lora_returns_none() -> None:
    spec = _make_spec(architecture="sd15", method="lora")
    result = TrainPipeline(spec).run(dry_run=True)
    assert result is None


def test_dry_run_valid_sdxl_lora_returns_none() -> None:
    spec = _make_spec(architecture="sdxl", method="lora")
    result = TrainPipeline(spec).run(dry_run=True)
    assert result is None


def test_dry_run_valid_dora_returns_none() -> None:
    spec = _make_spec(architecture="sd15", method="dora")
    result = TrainPipeline(spec).run(dry_run=True)
    assert result is None


def test_dry_run_unsupported_architecture_raises() -> None:
    spec = _make_spec(architecture="flux")
    with pytest.raises(ValueError, match="flux"):
        TrainPipeline(spec).run(dry_run=True)


def test_dry_run_unsupported_architecture_error_mentions_planned() -> None:
    spec = _make_spec(architecture="flux")
    with pytest.raises(ValueError, match="planned"):
        TrainPipeline(spec).run(dry_run=True)


def test_dry_run_unsupported_method_raises() -> None:
    spec = _make_spec(method="full")
    with pytest.raises(ValueError, match="full"):
        TrainPipeline(spec).run(dry_run=True)


def test_dry_run_does_not_instantiate_trainer() -> None:
    spec = _make_spec()
    with patch("forge.train.pipeline.LoRATrainer") as MockTrainer:
        TrainPipeline(spec).run(dry_run=True)
    MockTrainer.assert_not_called()


# ── dispatch ──────────────────────────────────────────────────────────────────


def test_run_dispatches_to_lora_trainer() -> None:
    spec = _make_spec()
    mock_trainer = MagicMock()
    mock_trainer.run.return_value = _fake_result()

    with patch("forge.train.pipeline.LoRATrainer", return_value=mock_trainer):
        result = TrainPipeline(spec).run()

    mock_trainer.run.assert_called_once()
    assert isinstance(result, TrainResult)


def test_run_returns_train_result() -> None:
    spec = _make_spec()
    expected = _fake_result()

    with patch("forge.train.pipeline.LoRATrainer") as MockTrainer:
        MockTrainer.return_value.run.return_value = expected
        result = TrainPipeline(spec).run()

    assert result is expected


def test_run_passes_spec_to_trainer() -> None:
    spec = _make_spec()

    with patch("forge.train.pipeline.LoRATrainer") as MockTrainer:
        MockTrainer.return_value.run.return_value = _fake_result()
        TrainPipeline(spec).run()

    call_args = MockTrainer.call_args
    assert call_args[0][0] is spec


def test_run_passes_device_override_to_trainer() -> None:
    import torch
    spec = _make_spec()
    device = torch.device("cpu")

    with patch("forge.train.pipeline.LoRATrainer") as MockTrainer:
        MockTrainer.return_value.run.return_value = _fake_result()
        TrainPipeline(spec, device=device).run()

    call_args = MockTrainer.call_args
    assert call_args.kwargs.get("device") is device or call_args[0][1] is device


# ── TrainingJobSpec YAML parsing ──────────────────────────────────────────────


def test_spec_parses_camel_case_lora_rank() -> None:
    spec = _make_spec()
    assert spec.training.lora_rank == 4


def test_spec_defaults_method_to_lora() -> None:
    spec = TrainingJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "TrainingJob",
        "metadata": {"name": "j"},
        "model": {"architecture": "sd15", "base": "test/m"},
        "data": {"source": "/d"},
        "output": {"checkpointDir": "/o"},
    })
    assert spec.training.method == TrainingMethod.LORA


def test_spec_default_target_modules_are_attention() -> None:
    spec = _make_spec()
    assert "to_q" in spec.training.lora_target_modules
    assert "to_v" in spec.training.lora_target_modules

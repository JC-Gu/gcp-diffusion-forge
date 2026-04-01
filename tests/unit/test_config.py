"""Unit tests for forge.core.config YAML spec models."""

from __future__ import annotations

import pytest

from forge.core.config import (
    DataJobSpec,
    GpuType,
    ModelArchitecture,
    TrainingJobSpec,
    TrainingMethod,
)

# ---------------------------------------------------------------------------
# Fixtures: raw dicts as they would come from yaml.safe_load()
# ---------------------------------------------------------------------------

TRAINING_JOB_DICT = {
    "apiVersion": "forge/v1",
    "kind": "TrainingJob",
    "metadata": {"name": "test-flux-lora"},
    "model": {
        "architecture": "flux",
        "base": "black-forest-labs/FLUX.1-dev",
        "variant": "bf16",
    },
    "data": {
        "source": "gs://forge-data/portraits-v2",
        "format": "webdataset",
    },
    "output": {
        "checkpointDir": "gs://forge-checkpoints/flux-lora-test",
    },
}

DATA_JOB_DICT = {
    "apiVersion": "forge/v1",
    "kind": "DataJob",
    "metadata": {"name": "portrait-curation", "tags": ["portraits", "v2"]},
    "stages": [
        {"name": "download", "type": "img2dataset", "params": {"imageSize": 1024}},
        {"name": "filter", "type": "quality_filter", "params": {"aestheticScoreThreshold": 4.5}},
    ],
}

# ---------------------------------------------------------------------------
# TrainingJobSpec tests
# ---------------------------------------------------------------------------


def test_training_job_parses_metadata() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.metadata.name == "test-flux-lora"


def test_training_job_parses_model() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.model.architecture == ModelArchitecture.FLUX
    assert spec.model.base == "black-forest-labs/FLUX.1-dev"
    assert spec.model.variant == "bf16"


def test_training_job_default_method_is_lora() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.training.method == TrainingMethod.LORA


def test_training_job_default_lora_rank() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.training.lora_rank == 32


def test_training_job_default_steps() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.training.steps == 2000


def test_training_job_default_resources_no_spot() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.resources.spot is False
    assert spec.resources.gpu_type == GpuType.CPU


def test_training_job_output_checkpoint_dir() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert spec.output.checkpoint_dir == "gs://forge-checkpoints/flux-lora-test"


def test_training_job_output_default_format() -> None:
    spec = TrainingJobSpec.model_validate(TRAINING_JOB_DICT)
    assert "safetensors" in spec.output.export_formats


def test_training_job_invalid_architecture_raises() -> None:
    bad = {**TRAINING_JOB_DICT, "model": {"architecture": "dall-e", "base": "openai/dall-e"}}
    with pytest.raises(Exception):
        TrainingJobSpec.model_validate(bad)


def test_training_job_missing_data_raises() -> None:
    no_data = {k: v for k, v in TRAINING_JOB_DICT.items() if k != "data"}
    with pytest.raises(Exception):
        TrainingJobSpec.model_validate(no_data)


# ---------------------------------------------------------------------------
# DataJobSpec tests
# ---------------------------------------------------------------------------


def test_data_job_parses_metadata() -> None:
    spec = DataJobSpec.model_validate(DATA_JOB_DICT)
    assert spec.metadata.name == "portrait-curation"
    assert "portraits" in spec.metadata.tags


def test_data_job_parses_stages() -> None:
    spec = DataJobSpec.model_validate(DATA_JOB_DICT)
    assert len(spec.stages) == 2
    assert spec.stages[0].type == "img2dataset"
    assert spec.stages[1].type == "quality_filter"


def test_data_job_stage_params_preserved() -> None:
    spec = DataJobSpec.model_validate(DATA_JOB_DICT)
    # params keys remain as-is (not snake_cased by the model)
    assert spec.stages[0].params["imageSize"] == 1024

"""Pydantic models for forge workflow YAML specs.

Job specs are YAML files with apiVersion: forge/v1. Load them with load_job_spec().

Example:
    spec = load_job_spec("workflows/train_flux_lora.yaml")
    assert isinstance(spec, TrainingJobSpec)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class ForgeModel(BaseModel):
    """Base model: accepts both camelCase (YAML) and snake_case (Python) keys."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelArchitecture(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"
    SD3 = "sd3"
    SD15 = "sd15"


class TrainingMethod(str, Enum):
    LORA = "lora"
    DORA = "dora"
    FULL = "full"
    DREAMBOOTH = "dreambooth"


class GpuType(str, Enum):
    T4 = "t4"
    L4 = "l4"
    A100_40 = "a100-40gb"
    A100_80 = "a100-80gb"
    H100 = "h100-80gb"
    CPU = "cpu"
    MPS = "mps"


# ---------------------------------------------------------------------------
# Shared sub-specs
# ---------------------------------------------------------------------------


class JobMetadata(ForgeModel):
    name: str
    tags: list[str] = Field(default_factory=list)
    description: str = ""


class ResourceSpec(ForgeModel):
    gpu_type: GpuType = GpuType.CPU
    gpu_count: int = 1
    memory: str = "16Gi"
    spot: bool = False


# ---------------------------------------------------------------------------
# Training job
# ---------------------------------------------------------------------------


class ModelSpec(ForgeModel):
    architecture: ModelArchitecture
    base: str  # HuggingFace model ID or gs:// path
    variant: str | None = None  # e.g. "bf16"


class OptimizerSpec(ForgeModel):
    type: str = "adamw8bit"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999


class TrainingSpec(ForgeModel):
    method: TrainingMethod = TrainingMethod.LORA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )
    steps: int = 2000
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine_with_restarts"
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)


class DataSpec(ForgeModel):
    source: str  # gs:// path or HuggingFace dataset ID
    format: str = "webdataset"
    resolution: int = 1024
    center_crop: bool = True
    caption_field: str = "json.caption"


class EvaluationSpec(ForgeModel):
    every_n_steps: int = 500
    metrics: list[str] = Field(default_factory=lambda: ["clip_score"])
    sample_prompts: list[str] = Field(default_factory=list)


class OutputSpec(ForgeModel):
    checkpoint_dir: str  # gs:// path
    push_to_hub: bool = False
    export_formats: list[str] = Field(default_factory=lambda: ["safetensors"])


class TrainingJobSpec(ForgeModel):
    api_version: Literal["forge/v1"] = "forge/v1"
    kind: Literal["TrainingJob"] = "TrainingJob"
    metadata: JobMetadata
    model: ModelSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    data: DataSpec
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    output: OutputSpec


# ---------------------------------------------------------------------------
# Data curation job
# ---------------------------------------------------------------------------


class DataStageSpec(ForgeModel):
    name: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class DataJobSpec(ForgeModel):
    api_version: Literal["forge/v1"] = "forge/v1"
    kind: Literal["DataJob"] = "DataJob"
    metadata: JobMetadata
    stages: list[DataStageSpec]
    resources: ResourceSpec = Field(default_factory=ResourceSpec)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_job_spec(yaml_path: str) -> TrainingJobSpec | DataJobSpec:
    """Load and validate a workflow YAML spec file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    kind = data.get("kind")
    if kind == "TrainingJob":
        return TrainingJobSpec.model_validate(data)
    if kind == "DataJob":
        return DataJobSpec.model_validate(data)
    raise ValueError(f"Unknown job kind: {kind!r}. Expected TrainingJob or DataJob.")

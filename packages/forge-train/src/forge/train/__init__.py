"""forge-train: LoRA and DoRA fine-tuning for diffusion models.

Supported architectures (v1): sd15, sdxl
Supported methods:            lora, dora
Supported backends:           CUDA (Tier 1), MPS (Tier 2), CPU (Tier 3 / CI only)

Public API::

    from forge.train import TrainPipeline, TrainResult, LoRATrainer
    from forge.train import build_lora_config, inject_lora, build_optimizer
    from forge.train import save_lora_weights, load_lora_weights

CLI::

    forge-train workflows/train_sd15_lora.yaml
    forge-train workflows/train_sd15_lora.yaml --dry-run
"""

from forge.train.checkpoint import load_lora_weights, save_lora_weights
from forge.train.lora import build_lora_config, inject_lora, resolve_training_precision
from forge.train.optimizer import build_optimizer
from forge.train.pipeline import TrainPipeline
from forge.train.trainer import LoRATrainer
from forge.train.types import TrainResult

__all__ = [
    "LoRATrainer",
    "TrainPipeline",
    "TrainResult",
    "build_lora_config",
    "build_optimizer",
    "inject_lora",
    "load_lora_weights",
    "resolve_training_precision",
    "save_lora_weights",
]

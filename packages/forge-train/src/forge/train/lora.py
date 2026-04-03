"""LoRA / DoRA utilities: config building, model injection, precision resolution.

LoRA and DoRA are handled by the same code path via PEFT's LoraConfig.
DoRA is enabled by setting use_dora=True.

References:
    LoRA:  https://arxiv.org/abs/2106.09685
    DoRA:  https://arxiv.org/abs/2402.09353
    PEFT:  https://github.com/huggingface/peft
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from forge.core.config import TrainingMethod
from forge.core.device import Backend

# Default UNet attention projection layers targeted by LoRA.
# Covers Q/K/V projections and the output projection of each attention block.
_DEFAULT_TARGET_MODULES: list[str] = ["to_q", "to_k", "to_v", "to_out.0"]


def build_lora_config(
    rank: int,
    alpha: int,
    method: TrainingMethod,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """Build a PEFT LoraConfig from training spec parameters.

    Args:
        rank:            LoRA rank (number of low-rank decomposition dimensions).
                         Higher rank = more parameters, more expressive.
        alpha:           LoRA scaling factor. Conventionally set equal to rank.
        method:          LORA or DORA. DoRA adds a weight decomposition step.
        target_modules:  Module names to inject adapters into.
                         Defaults to attention projection layers.

    Returns:
        LoraConfig ready to pass to inject_lora().
    """
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        use_dora=(method == TrainingMethod.DORA),
        target_modules=target_modules or _DEFAULT_TARGET_MODULES,
        bias="none",
    )


def inject_lora(model: nn.Module, config: LoraConfig) -> nn.Module:
    """Freeze all base parameters and attach LoRA adapters.

    After injection:
    - All original parameters have requires_grad=False.
    - LoRA adapter parameters (lora_A, lora_B) have requires_grad=True.
    - Total trainable parameter count is drastically reduced.

    Args:
        model:  Base model (UNet, transformer, or any nn.Module).
        config: LoraConfig produced by build_lora_config().

    Returns:
        PEFT-wrapped model with LoRA adapters attached.
    """
    model.requires_grad_(False)
    return get_peft_model(model, config)


def resolve_training_precision(
    spec_mixed_precision: str,
    backend: Backend,
) -> tuple[str, torch.dtype]:
    """Determine Accelerate mixed_precision mode and model dtype for a backend.

    Accelerate's autocast on MPS is unreliable across PyTorch versions.
    The safe strategy for MPS is to load the model in float16 directly
    and use mixed_precision="no" (no autocast, no GradScaler).

    Args:
        spec_mixed_precision: Value from TrainingSpec.mixed_precision ("bf16",
                              "fp16", or "no"). Only respected on CUDA.
        backend:              Detected backend from forge.core.device.

    Returns:
        (accelerate_mixed_precision, model_dtype) tuple.
    """
    if backend == Backend.CPU:
        return "no", torch.float32

    if backend == Backend.MPS:
        # Load model in float16; skip Accelerate autocast (MPS stability)
        return "no", torch.float16

    # CUDA: honour the spec value
    if spec_mixed_precision == "bf16":
        return "bf16", torch.bfloat16
    if spec_mixed_precision == "fp16":
        return "fp16", torch.float16
    return "no", torch.float32

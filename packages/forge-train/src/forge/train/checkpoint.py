"""LoRA checkpoint I/O — always writes safetensors, never pickle.

Only the LoRA adapter weights are saved (not the full base model).
The base model is re-loaded at inference time and the adapter weights
are merged or loaded on top.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

_log = logging.getLogger(__name__)


def save_lora_weights(
    model: nn.Module,
    output_dir: str,
    filename: str = "lora_weights.safetensors",
) -> str:
    """Save LoRA adapter weights to a safetensors file.

    Only LoRA parameters are saved — the base model weights are not
    included, keeping checkpoint files small (typically 1–50 MB).

    Args:
        model:      PEFT-wrapped model (output of inject_lora()).
        output_dir: Directory to write the checkpoint to (created if needed).
        filename:   File name within output_dir. Must end in .safetensors.

    Returns:
        Absolute path of the written checkpoint file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    state_dict = get_peft_model_state_dict(model)
    full_path = str(output_path / filename)
    save_file(state_dict, full_path)
    _log.info("Saved LoRA weights → %s (%d tensors)", full_path, len(state_dict))
    return full_path


def load_lora_weights(model: nn.Module, weights_path: str) -> None:
    """Load LoRA adapter weights from a safetensors file into a PEFT model.

    The model must already have LoRA adapters injected (via inject_lora).
    Weights are loaded in-place; the function returns None.

    Args:
        model:        PEFT-wrapped model to load weights into.
        weights_path: Path to a .safetensors file produced by save_lora_weights.
    """
    state_dict = load_file(weights_path)
    incompatible = set_peft_model_state_dict(model, state_dict)
    if incompatible and incompatible.unexpected_keys:
        _log.warning(
            "Unexpected keys when loading LoRA weights: %s",
            incompatible.unexpected_keys,
        )
    _log.info("Loaded LoRA weights ← %s", weights_path)

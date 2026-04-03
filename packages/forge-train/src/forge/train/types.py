"""Output types for forge-train."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainResult:
    """Structured output from a completed training run.

    Attributes:
        model_id:          HuggingFace model ID used as base.
        method:            "lora" or "dora".
        steps_completed:   Total gradient steps taken.
        checkpoint_path:   Absolute path to the saved .safetensors file.
        elapsed_sec:       Wall-clock time for the full run.
        final_loss:        Noise-prediction MSE loss at the last step.
        metadata:          Free-form dict (rank, lr, backend, …).
    """

    model_id: str
    method: str
    steps_completed: int
    checkpoint_path: str
    elapsed_sec: float
    final_loss: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "method": self.method,
            "steps_completed": self.steps_completed,
            "checkpoint_path": self.checkpoint_path,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }
        if self.final_loss is not None:
            d["final_loss"] = round(self.final_loss, 6)
        if self.metadata:
            d["metadata"] = self.metadata
        return d

"""TrainPipeline: validate a TrainingJobSpec and dispatch to LoRATrainer.

CLI entrypoint: forge-train workflows/train_sd15_lora.yaml

The same pattern as DataPipeline and RunnerPipeline — YAML in, result out,
dry-run validates the spec without loading any model.
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from forge.core.config import (
    ModelArchitecture,
    TrainingJobSpec,
    TrainingMethod,
    load_job_spec,
)
from forge.train.trainer import LoRATrainer, _SUPPORTED_ARCHITECTURES, _SUPPORTED_METHODS
from forge.train.types import TrainResult

# These constants mirror what the trainer enforces — kept here so
# _validate() can give clear errors before loading any model weights.
_SUPPORTED_ARCH_NAMES = {a.value for a in _SUPPORTED_ARCHITECTURES}
_SUPPORTED_METHOD_NAMES = {m.value for m in _SUPPORTED_METHODS}


class TrainPipeline:
    """Validate a TrainingJobSpec and run the appropriate trainer.

    Args:
        spec:   Validated TrainingJobSpec (from forge.core.config.load_job_spec).
        device: Override the auto-detected torch.device (useful in tests).
    """

    def __init__(self, spec: TrainingJobSpec, device=None) -> None:
        self._spec = spec
        self._device = device

    def _validate(self) -> None:
        arch = self._spec.model.architecture
        method = self._spec.training.method

        if arch not in _SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Architecture {arch.value!r} is not yet supported by forge-train. "
                f"Supported: {sorted(_SUPPORTED_ARCH_NAMES)}. "
                f"FLUX and SD3 support is planned for a future release."
            )
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Training method {method.value!r} is not yet supported. "
                f"Supported: {sorted(_SUPPORTED_METHOD_NAMES)}. "
                f"Full fine-tuning (FSDP2) is planned after forge-infra lands."
            )

    def run(self, dry_run: bool = False) -> TrainResult | None:
        """Validate then execute training.

        Args:
            dry_run: If True, validate the spec and return None without
                     loading any model or writing any files.

        Returns:
            TrainResult on success; None on dry_run.

        Raises:
            ValueError: on unsupported architecture or training method.
        """
        self._validate()
        if dry_run:
            return None

        trainer = LoRATrainer(self._spec, device=self._device)
        return trainer.run()


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: D401
    """forge-train CLI: run a TrainingJob YAML spec."""
    parser = argparse.ArgumentParser(
        prog="forge-train",
        description="Fine-tune a diffusion model with LoRA or DoRA.",
    )
    parser.add_argument("spec", help="Path to TrainingJob YAML file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec without loading any model",
    )
    args = parser.parse_args()

    job = load_job_spec(args.spec)
    if not isinstance(job, TrainingJobSpec):
        print(f"Error: expected a TrainingJob spec, got {job.__class__.__name__}", file=sys.stderr)
        sys.exit(1)

    result = TrainPipeline(job).run(dry_run=args.dry_run)

    if result is None:
        print("dry-run: spec is valid", flush=True)
    else:
        print(json.dumps(result.to_dict(), indent=2), flush=True)

    sys.exit(0)

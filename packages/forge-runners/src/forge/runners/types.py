"""Types shared across forge-runners: RunnerResult and RunnerJobSpec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from forge.core.config import ForgeModel, JobMetadata
from forge.eval.result import EvalResult


class RunnerJobSpec(ForgeModel):
    """YAML spec for a RunnerJob (kind: RunnerJob).

    Example::

        apiVersion: forge/v1
        kind: RunnerJob
        metadata:
          name: smoke-sd-turbo
        runner: smoke
        params:
          modelId: stabilityai/sd-turbo
          outputDir: /tmp/smoke-out
          nImages: 4
    """

    api_version: str = Field(alias="apiVersion", default="forge/v1")
    kind: Literal["RunnerJob"] = "RunnerJob"
    metadata: JobMetadata
    runner: str  # "smoke" | "eval"
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RunnerResult:
    """Structured output from any runner.

    Attributes:
        runner_type:  "smoke" or "eval".
        model_id:     HuggingFace model ID used for generation.
        n_generated:  Number of images generated.
        output_dir:   Directory where generated images were written.
        eval_result:  forge-eval metrics; None if metrics were not requested.
        elapsed_sec:  Wall-clock time for the full run.
        metadata:     Free-form dict (e.g. num_inference_steps, guidance_scale).
    """

    runner_type: str
    model_id: str
    n_generated: int
    output_dir: str
    eval_result: EvalResult | None = None
    elapsed_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict; None fields are excluded."""
        d: dict[str, Any] = {
            "runner_type": self.runner_type,
            "model_id": self.model_id,
            "n_generated": self.n_generated,
            "output_dir": self.output_dir,
            "elapsed_sec": round(self.elapsed_sec, 3),
        }
        if self.eval_result is not None:
            d["eval"] = self.eval_result.to_dict()
        if self.metadata:
            d["metadata"] = self.metadata
        return d

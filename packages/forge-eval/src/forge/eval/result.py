"""EvalResult: structured container for evaluation metric outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Holds the output of one evaluation run.

    Only metrics that were actually computed are present in `to_dict()`;
    uncomputed metrics remain None and are excluded from serialization.

    Attributes:
        fid:             Fréchet Inception Distance (lower is better).
        clip_score:      Mean cosine similarity between images and prompts in [-1, 1].
        aesthetic_score: Mean LAION aesthetic predictor score, roughly 0–10.
        n_generated:     Number of generated images evaluated.
        metadata:        Free-form dict for step, checkpoint path, timestamp, etc.
    """

    fid: float | None = None
    clip_score: float | None = None
    aesthetic_score: float | None = None
    n_generated: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dict with None fields excluded."""
        return self.model_dump(exclude_none=True)

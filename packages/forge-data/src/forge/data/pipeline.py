"""DataPipeline: orchestrates data curation stages from a DataJobSpec.

Reads a forge/v1 DataJobSpec YAML and dispatches each stage in order.
A single CLIPScorer is shared across embed and filter stages to avoid
loading the CLIP model twice.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from forge.core.config import DataJobSpec
from forge.core.device import get_device
from forge.core.scorers.clip import CLIPScorer
from forge.data.caption import run_caption
from forge.data.download import run_download
from forge.data.embed import run_embed
from forge.data.filter import run_filter
from forge.data.types import StageResult

# Maps YAML stage type strings → function names in this module's globals().
# Using string names (not direct references) so unittest.mock.patch() works.
_STAGE_REGISTRY: dict[str, str] = {
    "download":       "run_download",
    "img2dataset":    "run_download",
    "embed":          "run_embed",
    "clip_embed":     "run_embed",
    "filter":         "run_filter",
    "quality_filter": "run_filter",
    "caption":        "run_caption",
    "recaption":      "run_caption",
}

_CLIP_STAGE_TYPES = frozenset({"embed", "clip_embed", "filter", "quality_filter"})
_REQUIRED_PARAMS = {
    "download":       {"input_path", "output_dir"},
    "img2dataset":    {"input_path", "output_dir"},
    "embed":          {"input_dir", "output_dir"},
    "clip_embed":     {"input_dir", "output_dir"},
    "filter":         {"input_dir", "output_dir"},
    "quality_filter": {"input_dir", "output_dir"},
    "caption":        {"input_dir", "output_dir"},
    "recaption":      {"input_dir", "output_dir"},
}


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case for YAML param keys → Python kwargs."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    return {_camel_to_snake(k): v for k, v in params.items()}


class DataPipeline:
    """Orchestrates a sequence of data curation stages from a DataJobSpec.

    Usage::

        import yaml
        from forge.core.config import DataJobSpec
        from forge.data import DataPipeline

        spec = DataJobSpec.model_validate(yaml.safe_load(open("workflows/data_pipeline.yaml")))
        pipeline = DataPipeline(spec)
        results = pipeline.run()

    Args:
        spec:       Validated DataJobSpec (from forge.core.config).
        device:     torch.device for all ML stages. Auto-detected if None.
        batch_size: Batch size passed to embed and filter stages.
    """

    def __init__(
        self,
        spec: DataJobSpec,
        device: torch.device | None = None,
        batch_size: int = 256,
    ) -> None:
        self._spec = spec
        self._batch_size = batch_size

        if device is None:
            _, device = get_device()
        self._device = device

        self._clip_scorer: CLIPScorer | None = None
        self._scorers_built = False

    def _build_shared_scorers(self) -> None:
        """Create one CLIPScorer shared across all embed and filter stages."""
        if self._scorers_built:
            return
        stage_types = {s.type for s in self._spec.stages}
        if stage_types & _CLIP_STAGE_TYPES:
            self._clip_scorer = CLIPScorer(
                model_name="ViT-L-14",
                pretrained="openai",
                device=self._device,
                batch_size=self._batch_size,
            )
        self._scorers_built = True

    def _validate(self) -> None:
        """Raise ValueError if any stage has an unknown type or missing required params."""
        for stage in self._spec.stages:
            if stage.type not in _STAGE_REGISTRY:
                raise ValueError(
                    f"Stage {stage.name!r}: unknown type {stage.type!r}. "
                    f"Known types: {sorted(_STAGE_REGISTRY)}"
                )
            params = _normalize_params(stage.params)
            required = _REQUIRED_PARAMS.get(stage.type, set())
            missing = required - params.keys()
            if missing:
                raise ValueError(
                    f"Stage {stage.name!r} (type={stage.type!r}) "
                    f"is missing required params: {sorted(missing)}"
                )

    def run(self, dry_run: bool = False) -> list[StageResult]:
        """Execute all stages in order.

        Args:
            dry_run: If True, validate the spec and return an empty list
                     without executing any stage or touching the filesystem.

        Returns:
            List of StageResult objects, one per stage.

        Raises:
            ValueError: on unknown stage types or missing required params.
        """
        self._validate()

        if dry_run:
            return []

        self._build_shared_scorers()
        results: list[StageResult] = []

        for stage in self._spec.stages:
            fn = globals()[_STAGE_REGISTRY[stage.type]]
            kwargs = _normalize_params(stage.params)

            # Inject shared CLIPScorer into CLIP-dependent stages
            if stage.type in _CLIP_STAGE_TYPES:
                kwargs["clip_scorer"] = self._clip_scorer

            result = fn(**kwargs)
            # Attach the user-defined stage name from the spec
            result.stage_name = stage.name
            results.append(result)

        return results

"""EvalRunner: orchestrates multiple evaluation metrics in a single call.

Key optimization: when both clip_score and aesthetic_score are requested,
a single CLIPScorer(ViT-L-14/openai) instance is shared between them,
so the CLIP model is only loaded once.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forge.core.device import get_device
from forge.core.scorers.aesthetic import AestheticScorer
from forge.core.scorers.clip import CLIPScorer
from forge.eval.metrics import compute_aesthetic_score, compute_clip_score, compute_fid
from forge.eval.result import EvalResult

_KNOWN_METRICS = {"clip_score", "aesthetic_score", "fid"}


def _save_images_for_fid(images: list[Image.Image], directory: str) -> None:
    """Save PIL images as lossless PNGs into directory for FID computation."""
    for i, img in enumerate(images):
        img.convert("RGB").save(os.path.join(directory, f"{i:06d}.png"))


class EvalRunner:
    """Runs a configurable set of evaluation metrics in one call.

    Scorers are built lazily on the first call to run() and reused across
    subsequent calls, so constructing an EvalRunner is always cheap.

    Args:
        metrics:       Metrics to compute. Subset of {"clip_score", "aesthetic_score", "fid"}.
        reference_dir: Directory of reference images. Required when "fid" is in metrics.
        device:        torch.device. Auto-detected if None.
        batch_size:    Images per forward pass for all scorers.

    Raises:
        ValueError: if unknown metric names are provided.
        ValueError: if "fid" is in metrics but reference_dir is not provided.

    Example::

        runner = EvalRunner(
            metrics=["clip_score", "aesthetic_score"],
            device=torch.device("cuda"),
        )
        result = runner.run(images, prompts=prompts)
        print(result.clip_score, result.aesthetic_score)
    """

    def __init__(
        self,
        metrics: list[str],
        reference_dir: str | os.PathLike | None = None,
        device: torch.device | None = None,
        batch_size: int = 256,
    ) -> None:
        unknown = set(metrics) - _KNOWN_METRICS
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. Supported: {_KNOWN_METRICS}"
            )
        if "fid" in metrics and reference_dir is None:
            raise ValueError(
                "'fid' metric requires reference_dir to be provided."
            )

        self._metrics = set(metrics)
        self._reference_dir = str(reference_dir) if reference_dir is not None else None
        self._batch_size = batch_size

        if device is None:
            _, device = get_device()
        self._device = device

        # Scorers are populated lazily in _build_scorers()
        self._clip_scorer: CLIPScorer | None = None
        self._aesthetic_scorer: AestheticScorer | None = None
        self._scorers_built = False

    def _build_scorers(self) -> None:
        """Lazy-initialize scorers. Called once on the first run()."""
        if self._scorers_built:
            return

        needs_clip = "clip_score" in self._metrics
        needs_aesthetic = "aesthetic_score" in self._metrics

        if needs_clip or needs_aesthetic:
            # ViT-L-14/openai is the only variant compatible with AestheticScorer,
            # so always create one shared instance for both metrics.
            self._clip_scorer = CLIPScorer(
                model_name="ViT-L-14",
                pretrained="openai",
                device=self._device,
                batch_size=self._batch_size,
            )

        if needs_aesthetic:
            # Inject the shared CLIPScorer — AestheticScorer will not load CLIP again.
            self._aesthetic_scorer = AestheticScorer(
                clip_scorer=self._clip_scorer,
                device=self._device,
                batch_size=self._batch_size,
            )

        self._scorers_built = True

    def run(
        self,
        images: list[Image.Image],
        prompts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Compute all configured metrics and return an EvalResult.

        Args:
            images:   List of generated PIL Images to evaluate.
            prompts:  Text prompts paired with each image. Required for clip_score.
            metadata: Optional free-form dict stored in EvalResult.metadata.

        Returns:
            EvalResult with one field per requested metric.

        Raises:
            ValueError: if clip_score is requested but prompts is None or mismatched.
        """
        if "clip_score" in self._metrics:
            if prompts is None:
                raise ValueError("clip_score metric requires prompts.")
            if len(prompts) != len(images):
                raise ValueError(
                    f"clip_score: len(prompts)={len(prompts)} != len(images)={len(images)}"
                )

        self._build_scorers()

        clip_score: float | None = None
        aesthetic_score: float | None = None
        fid_score: float | None = None

        if "clip_score" in self._metrics:
            assert prompts is not None
            clip_score = compute_clip_score(images, prompts, scorer=self._clip_scorer)

        if "aesthetic_score" in self._metrics:
            aesthetic_score = compute_aesthetic_score(images, scorer=self._aesthetic_scorer)

        if "fid" in self._metrics:
            assert self._reference_dir is not None
            with tempfile.TemporaryDirectory(prefix="forge_eval_fid_") as tmpdir:
                _save_images_for_fid(images, tmpdir)
                fid_score = compute_fid(
                    tmpdir,
                    self._reference_dir,
                    device=self._device,
                    batch_size=self._batch_size,
                )

        return EvalResult(
            fid=fid_score,
            clip_score=clip_score,
            aesthetic_score=aesthetic_score,
            n_generated=len(images),
            metadata=metadata or {},
        )

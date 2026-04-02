"""Filter stage: quality-filter WebDataset shards.

Applies three independent thresholds in a single streaming pass:
  1. Minimum image resolution (cheap, per-sample).
  2. CLIP text-image cosine similarity >= threshold.
  3. LAION aesthetic score >= threshold.

The CLIPScorer is shared with AestheticScorer (same ViT-L-14 model) to
avoid loading CLIP twice — inject via clip_scorer= from DataPipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image

from forge.core.device import get_device
from forge.core.scorers.aesthetic import AestheticScorer
from forge.core.scorers.clip import CLIPScorer
from forge.data.io import iter_webdataset, write_webdataset
from forge.data.types import StageResult, WdsSample

_log = logging.getLogger(__name__)


def _process_batch(
    samples: list[WdsSample],
    images: list[Image.Image],
    clip_scorer: CLIPScorer,
    aesthetic_scorer: AestheticScorer,
    clip_threshold: float,
    aesthetic_threshold: float,
) -> list[WdsSample]:
    """Return samples from this batch that pass both quality thresholds."""
    passing_indices = list(range(len(samples)))

    # --- CLIP similarity filter ---
    if clip_threshold > 0 and passing_indices:
        captions = [samples[i].get("txt", "") for i in passing_indices]
        # Only score samples that have a caption; no caption → fail
        scorable = [(i, idx) for i, (idx, cap) in enumerate(zip(passing_indices, captions)) if cap]

        if scorable:
            pos, idx_list = zip(*scorable)
            scored_imgs = [images[idx] for idx in idx_list]
            scored_caps = [captions[pos_i] for pos_i in pos]
            scores = clip_scorer.score(scored_imgs, scored_caps)
            fail = {
                idx_list[pos_i]
                for pos_i, score in enumerate(scores)
                if score.item() < clip_threshold
            }
        else:
            fail = set(passing_indices)  # all fail: no captions

        # Also fail samples with no caption
        fail |= {passing_indices[i] for i, cap in enumerate(captions) if not cap}
        passing_indices = [i for i in passing_indices if i not in fail]

    # --- Aesthetic score filter ---
    if aesthetic_threshold > 0 and passing_indices:
        batch_images = [images[i] for i in passing_indices]
        ae_scores = aesthetic_scorer.score(batch_images)
        passing_indices = [
            idx
            for pos, idx in enumerate(passing_indices)
            if ae_scores[pos].item() >= aesthetic_threshold
        ]

    return [samples[i] for i in passing_indices]


def _iter_filtered_samples(
    url_pattern: str,
    clip_scorer: CLIPScorer,
    aesthetic_scorer: AestheticScorer,
    clip_similarity_threshold: float,
    aesthetic_score_threshold: float,
    min_resolution: int,
    batch_size: int,
) -> Iterator[WdsSample]:
    """Stream samples that pass all quality thresholds."""
    pending_samples: list[WdsSample] = []
    pending_images: list[Image.Image] = []

    def flush() -> list[WdsSample]:
        if not pending_samples:
            return []
        result = _process_batch(
            pending_samples, pending_images,
            clip_scorer, aesthetic_scorer,
            clip_similarity_threshold, aesthetic_score_threshold,
        )
        pending_samples.clear()
        pending_images.clear()
        return result

    for sample in iter_webdataset(url_pattern):
        img = sample.get("jpg") or sample.get("png")
        if not isinstance(img, Image.Image):
            continue

        # Resolution check — cheap, skip before any model inference
        if min_resolution > 0 and min(img.size) < min_resolution:
            continue

        pending_samples.append(sample)
        pending_images.append(img)

        if len(pending_samples) >= batch_size:
            yield from flush()

    yield from flush()


def run_filter(
    input_dir: str,
    output_dir: str,
    clip_similarity_threshold: float = 0.28,
    aesthetic_score_threshold: float = 4.5,
    min_resolution: int = 512,
    batch_size: int = 256,
    device: torch.device | None = None,
    clip_scorer: CLIPScorer | None = None,
) -> StageResult:
    """Filter WebDataset shards by resolution, CLIP similarity, and aesthetic score.

    Args:
        input_dir:                  Directory containing input .tar shards.
        output_dir:                 Directory for filtered output shards.
        clip_similarity_threshold:  Min cosine similarity between image and caption.
                                    Set to 0 to disable CLIP filtering.
        aesthetic_score_threshold:  Min LAION aesthetic predictor score (~0–10).
                                    Set to 0 to disable aesthetic filtering.
        min_resolution:             Minimum image dimension in pixels.
                                    Set to 0 to disable resolution filtering.
        batch_size:                 Images per scoring forward pass.
        device:                     torch.device. Auto-detected if None.
        clip_scorer:                Pre-loaded CLIPScorer to reuse. The same instance
                                    is passed to AestheticScorer to avoid loading
                                    CLIP twice. Inject from DataPipeline.

    Returns:
        StageResult with output_dir and elapsed time.

    Raises:
        ValueError: if no .tar shards are found in input_dir.
    """
    if not list(Path(input_dir).glob("*.tar")):
        raise ValueError(f"No .tar shards found in {input_dir!r}")

    if device is None:
        _, device = get_device()

    if clip_scorer is None:
        clip_scorer = CLIPScorer(device=device, batch_size=batch_size)

    # AestheticScorer receives the shared CLIPScorer — no second CLIP load.
    aesthetic_scorer = AestheticScorer(
        clip_scorer=clip_scorer,
        device=device,
        batch_size=batch_size,
    )

    start = time.time()
    url_pattern = str(Path(input_dir) / "*.tar")
    _log.info(
        "Filtering %s → %s (clip>=%.2f, aesthetic>=%.1f, res>=%d)",
        input_dir, output_dir,
        clip_similarity_threshold, aesthetic_score_threshold, min_resolution,
    )

    write_webdataset(
        _iter_filtered_samples(
            url_pattern, clip_scorer, aesthetic_scorer,
            clip_similarity_threshold, aesthetic_score_threshold,
            min_resolution, batch_size,
        ),
        output_dir,
    )

    return StageResult(
        stage_name="filter",
        stage_type="filter",
        output_dir=output_dir,
        elapsed_sec=time.time() - start,
    )

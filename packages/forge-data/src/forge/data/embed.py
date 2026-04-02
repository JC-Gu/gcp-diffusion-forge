"""Embed stage: CLIP-embed all images in WebDataset shards.

Adds a 'npy' field (float32 CLIP embedding) to each sample so the
filter stage can reuse embeddings instead of re-embedding from scratch.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image

from forge.core.device import get_device
from forge.core.scorers.clip import CLIPScorer
from forge.data.io import iter_webdataset, write_webdataset
from forge.data.types import StageResult, WdsSample

_log = logging.getLogger(__name__)


def _iter_embedded_samples(
    url_pattern: str,
    clip_scorer: CLIPScorer,
    batch_size: int,
) -> Iterator[WdsSample]:
    """Stream samples enriched with a 'npy' CLIP embedding field."""
    buffer_samples: list[WdsSample] = []
    buffer_images: list[Image.Image] = []

    def flush() -> Iterator[WdsSample]:
        if not buffer_images:
            return
        embeddings = clip_scorer.embed_images(buffer_images)  # [N, D] float32 CPU
        for sample, emb in zip(buffer_samples, embeddings):
            sample["npy"] = emb.numpy().astype("float32")
            yield sample
        buffer_samples.clear()
        buffer_images.clear()

    for sample in iter_webdataset(url_pattern):
        img = sample.get("jpg") or sample.get("png")
        if not isinstance(img, Image.Image):
            continue
        buffer_samples.append(sample)
        buffer_images.append(img)

        if len(buffer_images) >= batch_size:
            yield from flush()

    yield from flush()


def run_embed(
    input_dir: str,
    output_dir: str,
    batch_size: int = 256,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    device: torch.device | None = None,
    clip_scorer: CLIPScorer | None = None,
) -> StageResult:
    """Add CLIP embeddings to all samples in WebDataset shards.

    Streams input shards, embeds images in batches, and writes output shards
    with an additional 'npy' field per sample containing the float32 embedding.

    Args:
        input_dir:   Directory containing input .tar shards.
        output_dir:  Directory for output shards (may differ from input_dir).
        batch_size:  Images per CLIP forward pass.
        model_name:  OpenCLIP architecture (default: ViT-L-14).
        pretrained:  Pretrained weights name (default: openai).
        device:      torch.device. Auto-detected if None.
        clip_scorer: Pre-loaded CLIPScorer to reuse. When provided, model_name
                     and pretrained are ignored. Pass from DataPipeline to avoid
                     loading CLIP twice when embed and filter run together.

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
        clip_scorer = CLIPScorer(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
        )

    start = time.time()
    url_pattern = str(Path(input_dir) / "*.tar")
    _log.info("Embedding shards from %s → %s", input_dir, output_dir)

    write_webdataset(
        _iter_embedded_samples(url_pattern, clip_scorer, batch_size),
        output_dir,
    )

    return StageResult(
        stage_name="embed",
        stage_type="embed",
        output_dir=output_dir,
        elapsed_sec=time.time() - start,
    )

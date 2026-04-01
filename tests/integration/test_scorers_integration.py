"""Integration tests for forge.core.scorers — requires network + model downloads.

Run with:  uv run pytest tests/integration/ --run-slow
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from forge.core.scorers.aesthetic import AestheticScorer
from forge.core.scorers.clip import CLIPScorer


@pytest.fixture(scope="module")
def sample_images() -> list[Image.Image]:
    """Solid-color and random images for testing."""
    return [
        Image.new("RGB", (224, 224), color=(220, 40, 40)),    # red
        Image.new("RGB", (224, 224), color=(40, 180, 40)),    # green
        Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        ),  # noise
    ]


@pytest.fixture(scope="module")
def clip_scorer() -> CLIPScorer:
    return CLIPScorer(model_name="ViT-L-14", pretrained="openai")


@pytest.mark.slow
def test_clip_embeds_images(clip_scorer: CLIPScorer, sample_images: list[Image.Image]) -> None:
    emb = clip_scorer.embed_images(sample_images)
    assert emb.shape == (3, 768)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-4)


@pytest.mark.slow
def test_clip_embeds_texts(clip_scorer: CLIPScorer) -> None:
    emb = clip_scorer.embed_texts(["a red image", "a green image", "random noise"])
    assert emb.shape == (3, 768)


@pytest.mark.slow
def test_clip_score_red_image_vs_texts(
    clip_scorer: CLIPScorer, sample_images: list[Image.Image]
) -> None:
    """Red image should score higher with 'red' text than 'green' text."""
    red_img = [sample_images[0]]
    score_red = clip_scorer.score(red_img, ["a red image"])
    score_green = clip_scorer.score(red_img, ["a green image"])
    assert score_red[0] > score_green[0], (
        f"Expected red image to match 'red' better: {score_red[0]:.3f} vs {score_green[0]:.3f}"
    )


@pytest.mark.slow
def test_aesthetic_scorer_returns_scores(sample_images: list[Image.Image]) -> None:
    scorer = AestheticScorer()
    scores = scorer.score(sample_images)
    assert scores.shape == (3,)
    assert scores.dtype == torch.float32


@pytest.mark.slow
def test_aesthetic_scorer_shared_clip(
    clip_scorer: CLIPScorer, sample_images: list[Image.Image]
) -> None:
    """Passing a pre-loaded CLIPScorer avoids loading CLIP twice."""
    scorer = AestheticScorer(clip_scorer=clip_scorer)
    scores = scorer.score(sample_images)
    assert scores.shape == (3,)

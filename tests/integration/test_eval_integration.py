"""Integration tests for forge-eval — requires network + model downloads.

Run with:  uv run pytest tests/integration/ --run-slow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from forge.core.scorers.clip import CLIPScorer
from forge.eval import EvalResult, EvalRunner, compute_aesthetic_score, compute_clip_score
from forge.eval.metrics import compute_fid


@pytest.fixture(scope="module")
def sample_images() -> list[Image.Image]:
    return [
        Image.new("RGB", (224, 224), color=(200, 50, 50)),    # red
        Image.new("RGB", (224, 224), color=(50, 180, 50)),    # green
        Image.new("RGB", (224, 224), color=(50, 50, 200)),    # blue
        Image.fromarray(
            np.random.default_rng(42).integers(0, 255, (224, 224, 3), dtype=np.uint8)
        ),  # noise
    ]


@pytest.fixture(scope="module")
def sample_prompts() -> list[str]:
    return [
        "a red image",
        "a green image",
        "a blue image",
        "random noise texture",
    ]


@pytest.fixture(scope="module")
def shared_clip() -> CLIPScorer:
    return CLIPScorer(model_name="ViT-L-14", pretrained="openai")


# ── compute_clip_score ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_clip_score_returns_float(
    sample_images: list[Image.Image], sample_prompts: list[str]
) -> None:
    result = compute_clip_score(sample_images, sample_prompts)
    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


@pytest.mark.slow
def test_clip_score_with_shared_scorer(
    sample_images: list[Image.Image],
    sample_prompts: list[str],
    shared_clip: CLIPScorer,
) -> None:
    result = compute_clip_score(sample_images, sample_prompts, scorer=shared_clip)
    assert isinstance(result, float)
    # Verify the shared scorer was actually used (model loaded)
    assert shared_clip._model is not None


@pytest.mark.slow
def test_clip_score_semantic_alignment(
    sample_images: list[Image.Image], shared_clip: CLIPScorer
) -> None:
    """Red image should align better with 'red' than 'blue'."""
    red_img = [sample_images[0]]
    score_red = compute_clip_score(red_img, ["a red image"], scorer=shared_clip)
    score_blue = compute_clip_score(red_img, ["a blue image"], scorer=shared_clip)
    assert score_red > score_blue, (
        f"Expected red image to score higher with 'red': {score_red:.3f} vs {score_blue:.3f}"
    )


# ── compute_aesthetic_score ───────────────────────────────────────────────────


@pytest.mark.slow
def test_aesthetic_score_returns_float(sample_images: list[Image.Image]) -> None:
    result = compute_aesthetic_score(sample_images)
    assert isinstance(result, float)
    assert 0.0 < result < 10.0


@pytest.mark.slow
def test_aesthetic_score_with_shared_clip(
    sample_images: list[Image.Image], shared_clip: CLIPScorer
) -> None:
    from forge.core.scorers.aesthetic import AestheticScorer

    scorer = AestheticScorer(clip_scorer=shared_clip)
    result = compute_aesthetic_score(sample_images, scorer=scorer)
    assert isinstance(result, float)


# ── EvalRunner ────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_runner_clip_and_aesthetic_load_clip_once(
    sample_images: list[Image.Image], sample_prompts: list[str]
) -> None:
    """CLIPScorer should be instantiated exactly once for both metrics."""
    from unittest.mock import patch

    original_init = CLIPScorer.__init__
    init_call_count: list[int] = [0]

    def counting_init(self: CLIPScorer, **kwargs: object) -> None:
        init_call_count[0] += 1
        original_init(self, **kwargs)  # type: ignore[call-arg]

    with patch.object(CLIPScorer, "__init__", counting_init):
        runner = EvalRunner(metrics=["clip_score", "aesthetic_score"])
        runner.run(sample_images, prompts=sample_prompts)

    assert init_call_count[0] == 1, (
        f"CLIPScorer was instantiated {init_call_count[0]} times; expected 1"
    )


@pytest.mark.slow
def test_runner_full_result_all_metrics(
    sample_images: list[Image.Image],
    sample_prompts: list[str],
    tmp_path: Path,
) -> None:
    """End-to-end: all three metrics in one run()."""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    for i, img in enumerate(sample_images):
        img.save(ref_dir / f"{i:06d}.png")

    runner = EvalRunner(
        metrics=["clip_score", "aesthetic_score", "fid"],
        reference_dir=str(ref_dir),
    )
    result = runner.run(sample_images, prompts=sample_prompts)

    assert isinstance(result, EvalResult)
    assert isinstance(result.clip_score, float)
    assert isinstance(result.aesthetic_score, float)
    assert isinstance(result.fid, float) and result.fid >= 0.0
    assert result.n_generated == len(sample_images)


@pytest.mark.slow
def test_runner_fid_warns_small_dataset(
    sample_images: list[Image.Image], tmp_path: Path
) -> None:
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    for i, img in enumerate(sample_images):
        img.save(ref_dir / f"{i:06d}.png")

    runner = EvalRunner(metrics=["fid"], reference_dir=str(ref_dir))
    with pytest.warns(UserWarning, match="images"):
        runner.run(sample_images)

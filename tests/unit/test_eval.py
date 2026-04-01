"""Unit tests for forge-eval — no GPU, no network, no model downloads."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from forge.eval import EvalResult, EvalRunner, compute_aesthetic_score, compute_clip_score
from forge.eval.metrics import _FID_MIN_IMAGES, compute_fid
from forge.eval.runner import _save_images_for_fid


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_images() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        for _ in range(4)
    ]


@pytest.fixture
def dummy_prompts() -> list[str]:
    return ["a red circle", "a blue square", "a green triangle", "a yellow star"]


@pytest.fixture
def mock_clip_scorer() -> MagicMock:
    scorer = MagicMock(spec=["score", "embed_images", "embed_texts", "_load"])
    scorer.score.return_value = torch.tensor([0.25, 0.30, 0.20, 0.28])
    return scorer


@pytest.fixture
def mock_aesthetic_scorer() -> MagicMock:
    scorer = MagicMock(spec=["score", "_load"])
    scorer.score.return_value = torch.tensor([5.0, 6.2, 4.8, 5.5])
    return scorer


# ── EvalResult ────────────────────────────────────────────────────────────────


def test_eval_result_defaults_are_none() -> None:
    r = EvalResult(n_generated=10)
    assert r.fid is None
    assert r.clip_score is None
    assert r.aesthetic_score is None


def test_eval_result_metadata_defaults_empty() -> None:
    r = EvalResult(n_generated=5)
    assert r.metadata == {}


def test_eval_result_n_generated_required() -> None:
    with pytest.raises(ValidationError):
        EvalResult()  # type: ignore[call-arg]


def test_eval_result_to_dict_excludes_none() -> None:
    r = EvalResult(n_generated=4, clip_score=0.27)
    d = r.to_dict()
    assert "fid" not in d
    assert "aesthetic_score" not in d
    assert d["clip_score"] == pytest.approx(0.27)
    assert d["n_generated"] == 4


def test_eval_result_to_dict_includes_all_when_set() -> None:
    r = EvalResult(n_generated=4, fid=12.5, clip_score=0.27, aesthetic_score=5.3)
    d = r.to_dict()
    assert set(d.keys()) == {"fid", "clip_score", "aesthetic_score", "n_generated", "metadata"}


def test_eval_result_metadata_roundtrip() -> None:
    r = EvalResult(n_generated=1, metadata={"step": 500, "checkpoint": "gs://bucket/ckpt"})
    assert r.to_dict()["metadata"]["step"] == 500


# ── compute_clip_score ────────────────────────────────────────────────────────


def test_compute_clip_score_returns_float(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
    mock_clip_scorer: MagicMock,
) -> None:
    result = compute_clip_score(dummy_images, dummy_prompts, scorer=mock_clip_scorer)
    assert isinstance(result, float)


def test_compute_clip_score_correct_mean(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
    mock_clip_scorer: MagicMock,
) -> None:
    result = compute_clip_score(dummy_images, dummy_prompts, scorer=mock_clip_scorer)
    expected = float(torch.tensor([0.25, 0.30, 0.20, 0.28]).mean())
    assert result == pytest.approx(expected)


def test_compute_clip_score_uses_injected_scorer(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
    mock_clip_scorer: MagicMock,
) -> None:
    compute_clip_score(dummy_images, dummy_prompts, scorer=mock_clip_scorer)
    mock_clip_scorer.score.assert_called_once_with(dummy_images, dummy_prompts)


def test_compute_clip_score_creates_scorer_when_none(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
) -> None:
    mock_instance = MagicMock()
    mock_instance.score.return_value = torch.zeros(len(dummy_images))
    with patch("forge.eval.metrics.CLIPScorer", return_value=mock_instance) as MockCLIP:
        compute_clip_score(dummy_images, dummy_prompts, scorer=None)
        MockCLIP.assert_called_once()


# ── compute_aesthetic_score ───────────────────────────────────────────────────


def test_compute_aesthetic_score_returns_float(
    dummy_images: list[Image.Image],
    mock_aesthetic_scorer: MagicMock,
) -> None:
    result = compute_aesthetic_score(dummy_images, scorer=mock_aesthetic_scorer)
    assert isinstance(result, float)


def test_compute_aesthetic_score_correct_mean(
    dummy_images: list[Image.Image],
    mock_aesthetic_scorer: MagicMock,
) -> None:
    result = compute_aesthetic_score(dummy_images, scorer=mock_aesthetic_scorer)
    expected = float(torch.tensor([5.0, 6.2, 4.8, 5.5]).mean())
    assert result == pytest.approx(expected)


def test_compute_aesthetic_score_uses_injected_scorer(
    dummy_images: list[Image.Image],
    mock_aesthetic_scorer: MagicMock,
) -> None:
    compute_aesthetic_score(dummy_images, scorer=mock_aesthetic_scorer)
    mock_aesthetic_scorer.score.assert_called_once_with(dummy_images)


def test_compute_aesthetic_score_creates_scorer_when_none(
    dummy_images: list[Image.Image],
) -> None:
    mock_instance = MagicMock()
    mock_instance.score.return_value = torch.zeros(len(dummy_images))
    with patch("forge.eval.metrics.AestheticScorer", return_value=mock_instance) as MockAesthetic:
        compute_aesthetic_score(dummy_images, scorer=None)
        MockAesthetic.assert_called_once()


# ── compute_fid ───────────────────────────────────────────────────────────────


def test_compute_fid_calls_cleanfid(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    # Create enough images to skip the warning
    for i in range(_FID_MIN_IMAGES):
        (gen_dir / f"{i:06d}.png").touch()

    with patch("forge.eval.metrics._cleanfid_fid") as mock_fid:
        mock_fid.compute_fid.return_value = 42.5
        result = compute_fid(gen_dir, ref_dir)

    assert result == pytest.approx(42.5)
    assert isinstance(result, float)
    mock_fid.compute_fid.assert_called_once_with(
        fdir1=str(gen_dir),
        fdir2=str(ref_dir),
        mode="clean",
        device=ANY,
        batch_size=ANY,
        num_workers=0,
    )


def test_compute_fid_warns_on_small_dataset(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    # Only 10 images — below threshold
    for i in range(10):
        Image.new("RGB", (32, 32)).save(gen_dir / f"{i:06d}.png")

    with patch("forge.eval.metrics._cleanfid_fid") as mock_fid:
        mock_fid.compute_fid.return_value = 0.0
        with pytest.warns(UserWarning, match="10 images"):
            compute_fid(gen_dir, ref_dir)


def test_compute_fid_no_warning_at_threshold(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    for i in range(_FID_MIN_IMAGES):
        (gen_dir / f"{i:06d}.png").touch()

    with patch("forge.eval.metrics._cleanfid_fid") as mock_fid:
        mock_fid.compute_fid.return_value = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_fid(gen_dir, ref_dir)  # should not raise


# ── _save_images_for_fid ──────────────────────────────────────────────────────


def test_save_images_for_fid_writes_pngs(
    tmp_path: Path, dummy_images: list[Image.Image]
) -> None:
    _save_images_for_fid(dummy_images, str(tmp_path))
    saved = sorted(tmp_path.glob("*.png"))
    assert len(saved) == len(dummy_images)
    assert saved[0].name == "000000.png"


def test_save_images_for_fid_converts_to_rgb(tmp_path: Path) -> None:
    rgba_img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    _save_images_for_fid([rgba_img], str(tmp_path))
    loaded = Image.open(tmp_path / "000000.png")
    assert loaded.mode == "RGB"


# ── EvalRunner construction validation ───────────────────────────────────────


def test_runner_unknown_metric_raises() -> None:
    with pytest.raises(ValueError, match="Unknown metrics"):
        EvalRunner(metrics=["perceptual_hash"])


def test_runner_fid_without_reference_dir_raises() -> None:
    with pytest.raises(ValueError, match="reference_dir"):
        EvalRunner(metrics=["fid"])


def test_runner_fid_with_reference_dir_ok(tmp_path: Path) -> None:
    runner = EvalRunner(metrics=["fid"], reference_dir=str(tmp_path))
    assert runner is not None


def test_runner_empty_metrics_ok() -> None:
    runner = EvalRunner(metrics=[])
    assert runner is not None


# ── EvalRunner.run() validation ───────────────────────────────────────────────


def test_runner_clip_score_requires_prompts(dummy_images: list[Image.Image]) -> None:
    runner = EvalRunner(metrics=["clip_score"])
    with pytest.raises(ValueError, match="prompts"):
        runner.run(dummy_images, prompts=None)


def test_runner_clip_score_mismatched_prompts_raises(
    dummy_images: list[Image.Image],
) -> None:
    runner = EvalRunner(metrics=["clip_score"])
    with pytest.raises(ValueError, match="len"):
        runner.run(dummy_images, prompts=["only one"])


# ── EvalRunner scorer sharing ─────────────────────────────────────────────────


def test_runner_shares_clip_scorer_between_metrics(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
) -> None:
    """CLIPScorer must be instantiated exactly once when both clip and aesthetic requested."""
    mock_clip = MagicMock()
    mock_clip.score.return_value = torch.zeros(len(dummy_images))
    mock_aesthetic = MagicMock()
    mock_aesthetic.score.return_value = torch.zeros(len(dummy_images))

    with (
        patch("forge.eval.runner.CLIPScorer", return_value=mock_clip) as MockCLIP,
        patch("forge.eval.runner.AestheticScorer", return_value=mock_aesthetic),
    ):
        runner = EvalRunner(metrics=["clip_score", "aesthetic_score"])
        runner.run(dummy_images, prompts=dummy_prompts)
        MockCLIP.assert_called_once()


def test_runner_aesthetic_receives_shared_clip_scorer(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
) -> None:
    mock_clip = MagicMock()
    mock_clip.score.return_value = torch.zeros(len(dummy_images))
    mock_aesthetic = MagicMock()
    mock_aesthetic.score.return_value = torch.zeros(len(dummy_images))

    with (
        patch("forge.eval.runner.CLIPScorer", return_value=mock_clip),
        patch("forge.eval.runner.AestheticScorer", return_value=mock_aesthetic) as MockAesthetic,
    ):
        runner = EvalRunner(metrics=["clip_score", "aesthetic_score"])
        runner.run(dummy_images, prompts=dummy_prompts)

        # AestheticScorer must receive the shared CLIPScorer instance
        MockAesthetic.assert_called_once_with(
            clip_scorer=mock_clip,
            device=ANY,
            batch_size=ANY,
        )


def test_runner_aesthetic_only_no_clip_scorer_created(
    dummy_images: list[Image.Image],
) -> None:
    """When only aesthetic_score is requested, no standalone CLIPScorer is needed."""
    mock_clip = MagicMock()
    mock_clip.score.return_value = torch.zeros(len(dummy_images))
    mock_aesthetic = MagicMock()
    mock_aesthetic.score.return_value = torch.zeros(len(dummy_images))

    with (
        patch("forge.eval.runner.CLIPScorer", return_value=mock_clip) as MockCLIP,
        patch("forge.eval.runner.AestheticScorer", return_value=mock_aesthetic),
    ):
        runner = EvalRunner(metrics=["aesthetic_score"])
        runner.run(dummy_images)
        # CLIPScorer is still created (shared into AestheticScorer),
        # but it's not used for the clip_score metric
        MockCLIP.assert_called_once()


def test_runner_scorers_built_only_once(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
) -> None:
    mock_clip = MagicMock()
    mock_clip.score.return_value = torch.zeros(len(dummy_images))

    with patch("forge.eval.runner.CLIPScorer", return_value=mock_clip) as MockCLIP:
        runner = EvalRunner(metrics=["clip_score"])
        runner.run(dummy_images, prompts=dummy_prompts)
        runner.run(dummy_images, prompts=dummy_prompts)
        # Second run must not re-instantiate scorers
        MockCLIP.assert_called_once()


# ── EvalRunner.run() result correctness ──────────────────────────────────────


def test_runner_run_returns_eval_result(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
) -> None:
    mock_clip = MagicMock()
    mock_clip.score.return_value = torch.ones(len(dummy_images)) * 0.25

    with patch("forge.eval.runner.CLIPScorer", return_value=mock_clip):
        runner = EvalRunner(metrics=["clip_score"])
        result = runner.run(dummy_images, prompts=dummy_prompts)

    assert isinstance(result, EvalResult)
    assert result.n_generated == len(dummy_images)
    assert result.clip_score == pytest.approx(0.25)
    assert result.fid is None
    assert result.aesthetic_score is None


def test_runner_run_fid_uses_temp_dir_then_cleans_up(
    dummy_images: list[Image.Image],
    tmp_path: Path,
) -> None:
    captured_tmpdir: list[str] = []

    def fake_save(images: list, directory: str) -> None:
        captured_tmpdir.append(directory)

    with (
        patch("forge.eval.runner._save_images_for_fid", side_effect=fake_save),
        patch("forge.eval.runner.compute_fid", return_value=15.0),
    ):
        runner = EvalRunner(metrics=["fid"], reference_dir=str(tmp_path))
        result = runner.run(dummy_images)

    assert result.fid == pytest.approx(15.0)
    # Temp dir must have been created and then cleaned up
    assert len(captured_tmpdir) == 1
    assert not os.path.exists(captured_tmpdir[0])


def test_runner_metadata_forwarded_to_result(
    dummy_images: list[Image.Image],
    dummy_prompts: list[str],
) -> None:
    mock_clip = MagicMock()
    mock_clip.score.return_value = torch.zeros(len(dummy_images))

    with patch("forge.eval.runner.CLIPScorer", return_value=mock_clip):
        runner = EvalRunner(metrics=["clip_score"])
        result = runner.run(
            dummy_images,
            prompts=dummy_prompts,
            metadata={"step": 1000, "model": "flux-lora-v1"},
        )

    assert result.metadata["step"] == 1000
    assert result.metadata["model"] == "flux-lora-v1"

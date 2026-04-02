"""Unit tests for forge.runners — no GPU, no model downloads.

DiffusionWrapper.generate() is always mocked so tests run in milliseconds
on any machine (MPS, CPU, CI).  EvalRunner is also mocked so CLIP/aesthetic
model weights are never downloaded.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from forge.eval.result import EvalResult
from forge.runners.diffusion import DiffusionWrapper
from forge.runners.eval import _load_prompts, _resolve_prompts, run_eval
from forge.runners.smoke import _build_prompts, _is_valid_image, run_smoke
from forge.runners.types import RunnerResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _solid_image(color: tuple[int, int, int] = (128, 64, 32), size=(64, 64)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _noisy_image(size=(64, 64)) -> Image.Image:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def mock_diffusion() -> MagicMock:
    """DiffusionWrapper that returns 4 noisy images without loading any model."""
    wrapper = MagicMock(spec=DiffusionWrapper)
    wrapper.model_id = "test/model"
    wrapper.generate.return_value = [_noisy_image() for _ in range(4)]
    return wrapper


@pytest.fixture
def mock_eval_result() -> EvalResult:
    return EvalResult(clip_score=0.32, aesthetic_score=5.1, n_generated=4)


@pytest.fixture
def mock_eval_runner(mock_eval_result: EvalResult) -> MagicMock:
    runner = MagicMock()
    runner.run.return_value = mock_eval_result
    return runner


# ── _is_valid_image ───────────────────────────────────────────────────────────


def test_is_valid_image_rejects_solid_black() -> None:
    assert not _is_valid_image(Image.new("RGB", (64, 64), color=(0, 0, 0)))


def test_is_valid_image_rejects_solid_white() -> None:
    assert not _is_valid_image(Image.new("RGB", (64, 64), color=(255, 255, 255)))


def test_is_valid_image_accepts_noisy() -> None:
    assert _is_valid_image(_noisy_image())


def test_is_valid_image_accepts_varied_colour() -> None:
    assert _is_valid_image(_solid_image())  # non-zero std across channels


# ── _build_prompts ────────────────────────────────────────────────────────────


def test_build_prompts_returns_correct_count() -> None:
    assert len(_build_prompts(6)) == 6


def test_build_prompts_cycles_defaults() -> None:
    prompts = _build_prompts(8)
    assert len(set(prompts)) >= 2   # cycles, not all same


# ── run_smoke ─────────────────────────────────────────────────────────────────


def test_run_smoke_returns_runner_result(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    with patch("forge.runners.smoke.EvalRunner", return_value=mock_eval_runner):
        result = run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["a test prompt"] * 4,
            diffusion=mock_diffusion,
        )
    assert isinstance(result, RunnerResult)
    assert result.runner_type == "smoke"


def test_run_smoke_calls_generate(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    with patch("forge.runners.smoke.EvalRunner", return_value=mock_eval_runner):
        run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p1", "p2", "p3", "p4"],
            diffusion=mock_diffusion,
        )
    mock_diffusion.generate.assert_called_once_with(["p1", "p2", "p3", "p4"])


def test_run_smoke_writes_images_to_disk(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    with patch("forge.runners.smoke.EvalRunner", return_value=mock_eval_runner):
        run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_run_smoke_n_generated_matches_prompts(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    with patch("forge.runners.smoke.EvalRunner", return_value=mock_eval_runner):
        result = run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert result.n_generated == 4


def test_run_smoke_attaches_eval_result(
    tmp_path: Path, mock_diffusion: MagicMock,
    mock_eval_runner: MagicMock, mock_eval_result: EvalResult
) -> None:
    with patch("forge.runners.smoke.EvalRunner", return_value=mock_eval_runner):
        result = run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert result.eval_result is mock_eval_result


def test_run_smoke_skips_eval_when_metrics_empty(
    tmp_path: Path, mock_diffusion: MagicMock
) -> None:
    with patch("forge.runners.smoke.EvalRunner") as MockEval:
        run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            metrics=[],
            diffusion=mock_diffusion,
        )
    MockEval.assert_not_called()


def test_run_smoke_raises_on_degenerate_image(tmp_path: Path) -> None:
    bad_diffusion = MagicMock(spec=DiffusionWrapper)
    bad_diffusion.generate.return_value = [
        Image.new("RGB", (64, 64), (0, 0, 0))  # all black
    ]
    with pytest.raises(ValueError, match="degenerate"):
        run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"],
            metrics=[],
            diffusion=bad_diffusion,
        )


def test_run_smoke_raises_on_fid_metric(tmp_path: Path, mock_diffusion: MagicMock) -> None:
    with pytest.raises(ValueError, match="FID"):
        run_smoke(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"],
            metrics=["fid"],
            diffusion=mock_diffusion,
        )


def test_run_smoke_creates_output_dir(mock_diffusion: MagicMock, tmp_path: Path) -> None:
    new_dir = tmp_path / "a" / "b" / "smoke-out"
    with patch("forge.runners.smoke.EvalRunner", return_value=MagicMock(run=lambda *a, **k: EvalResult(n_generated=4))):
        run_smoke(
            model_id="test/model",
            output_dir=str(new_dir),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert new_dir.exists()


# ── _resolve_prompts / _load_prompts ──────────────────────────────────────────


def test_resolve_prompts_uses_inline(tmp_path: Path) -> None:
    result = _resolve_prompts(["a", "b"], None, None)
    assert result == ["a", "b"]


def test_resolve_prompts_reads_file(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("prompt one\nprompt two\n\nprompt three\n")
    result = _resolve_prompts(None, str(p), None)
    assert result == ["prompt one", "prompt two", "prompt three"]


def test_resolve_prompts_inline_takes_priority(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text("from file\n")
    result = _resolve_prompts(["from inline"], str(p), None)
    assert result == ["from inline"]


def test_resolve_prompts_n_images_truncates(tmp_path: Path) -> None:
    result = _resolve_prompts(["a", "b", "c", "d"], None, 2)
    assert result == ["a", "b"]


def test_resolve_prompts_fallback_uses_n_images() -> None:
    result = _resolve_prompts(None, None, 3)
    assert len(result) == 3


def test_resolve_prompts_raises_on_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("\n\n")
    with pytest.raises(ValueError, match="No prompts"):
        _resolve_prompts(None, str(p), None)


# ── run_eval ──────────────────────────────────────────────────────────────────


def test_run_eval_returns_runner_result(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    with patch("forge.runners.eval._MetricsRunner", return_value=mock_eval_runner):
        result = run_eval(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert isinstance(result, RunnerResult)
    assert result.runner_type == "eval"


def test_run_eval_generates_in_batches(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    """batch_size=2 with 4 prompts → generate called twice."""
    mock_diffusion.generate.return_value = [_noisy_image(), _noisy_image()]
    with patch("forge.runners.eval._MetricsRunner", return_value=mock_eval_runner):
        run_eval(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p1", "p2", "p3", "p4"],
            batch_size=2,
            diffusion=mock_diffusion,
        )
    assert mock_diffusion.generate.call_count == 2


def test_run_eval_adds_fid_when_reference_dir_given(
    tmp_path: Path, mock_diffusion: MagicMock
) -> None:
    """reference_dir present → fid is added to metrics automatically."""
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    captured: dict = {}

    def capture_metrics(metrics, reference_dir, device):
        captured["metrics"] = metrics
        return MagicMock(run=lambda *a, **k: EvalResult(n_generated=4))

    with patch("forge.runners.eval._MetricsRunner", side_effect=capture_metrics):
        run_eval(
            model_id="test/model",
            output_dir=str(tmp_path / "out"),
            prompts=["p"] * 4,
            reference_dir=str(ref_dir),
            diffusion=mock_diffusion,
        )
    assert "fid" in captured["metrics"]


def test_run_eval_does_not_add_fid_without_reference_dir(
    tmp_path: Path, mock_diffusion: MagicMock
) -> None:
    captured: dict = {}

    def capture_metrics(metrics, reference_dir, device):
        captured["metrics"] = metrics
        return MagicMock(run=lambda *a, **k: EvalResult(n_generated=4))

    with patch("forge.runners.eval._MetricsRunner", side_effect=capture_metrics):
        run_eval(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert "fid" not in captured["metrics"]


def test_run_eval_writes_images(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    with patch("forge.runners.eval._MetricsRunner", return_value=mock_eval_runner):
        run_eval(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p"] * 4,
            diffusion=mock_diffusion,
        )
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_run_eval_n_images_truncates_prompts(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    mock_diffusion.generate.return_value = [_noisy_image(), _noisy_image()]
    with patch("forge.runners.eval._MetricsRunner", return_value=mock_eval_runner):
        result = run_eval(
            model_id="test/model",
            output_dir=str(tmp_path),
            prompts=["p1", "p2", "p3", "p4"],
            n_images=2,
            diffusion=mock_diffusion,
        )
    assert result.n_generated == 2


def test_run_eval_loads_prompts_from_file(
    tmp_path: Path, mock_diffusion: MagicMock, mock_eval_runner: MagicMock
) -> None:
    pf = tmp_path / "prompts.txt"
    pf.write_text("apple\nbanana\norange\n")
    mock_diffusion.generate.return_value = [_noisy_image() for _ in range(3)]
    with patch("forge.runners.eval._MetricsRunner", return_value=mock_eval_runner):
        result = run_eval(
            model_id="test/model",
            output_dir=str(tmp_path / "out"),
            prompts_path=str(pf),
            diffusion=mock_diffusion,
        )
    assert result.n_generated == 3


# ── DiffusionWrapper ──────────────────────────────────────────────────────────


def test_diffusion_wrapper_lazy_load() -> None:
    """Model should not be loaded until generate() is called."""
    wrapper = DiffusionWrapper(model_id="test/model", device=torch.device("cpu"))
    assert wrapper._pipe is None


def test_diffusion_wrapper_generate_calls_pipe() -> None:
    wrapper = DiffusionWrapper(model_id="test/model", device=torch.device("cpu"))
    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [_noisy_image(), _noisy_image()]
    wrapper._pipe = mock_pipe  # bypass _load

    images = wrapper.generate(["p1", "p2"])
    assert len(images) == 2
    mock_pipe.assert_called_once()

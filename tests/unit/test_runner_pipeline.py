"""Unit tests for forge.runners.pipeline.RunnerPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.runners.pipeline import RunnerPipeline, _camel_to_snake, _normalize_params
from forge.runners.types import RunnerJobSpec, RunnerResult
from forge.eval.result import EvalResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spec(runner: str, params: dict) -> RunnerJobSpec:
    return RunnerJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "RunnerJob",
        "metadata": {"name": "test-runner"},
        "runner": runner,
        "params": params,
    })


def _smoke_spec(model_id: str = "test/model", output_dir: str = "/out") -> RunnerJobSpec:
    return _make_spec("smoke", {"modelId": model_id, "outputDir": output_dir})


def _eval_spec(model_id: str = "test/model", output_dir: str = "/out") -> RunnerJobSpec:
    return _make_spec("eval", {"modelId": model_id, "outputDir": output_dir})


def _fake_result(runner_type: str = "smoke") -> RunnerResult:
    return RunnerResult(
        runner_type=runner_type,
        model_id="test/model",
        n_generated=4,
        output_dir="/out",
        eval_result=EvalResult(n_generated=4),
    )


# ── _camel_to_snake / _normalize_params ──────────────────────────────────────


@pytest.mark.parametrize("camel,snake", [
    ("modelId", "model_id"),
    ("outputDir", "output_dir"),
    ("numInferenceSteps", "num_inference_steps"),
    ("batchSize", "batch_size"),
    ("promptsPath", "prompts_path"),
    ("referenceDir", "reference_dir"),
    ("model_id", "model_id"),   # already snake — unchanged
])
def test_camel_to_snake(camel: str, snake: str) -> None:
    assert _camel_to_snake(camel) == snake


def test_normalize_params_converts_all_keys() -> None:
    raw = {"modelId": "x", "outputDir": "/o", "batchSize": 8}
    assert _normalize_params(raw) == {"model_id": "x", "output_dir": "/o", "batch_size": 8}


# ── dry_run validation ────────────────────────────────────────────────────────


def test_dry_run_unknown_runner_raises() -> None:
    spec = _make_spec("benchmark", {"modelId": "x", "outputDir": "/o"})
    with pytest.raises(ValueError, match="Unknown runner"):
        RunnerPipeline(spec).run(dry_run=True)


def test_dry_run_missing_model_id_raises() -> None:
    spec = _make_spec("smoke", {"outputDir": "/o"})
    with pytest.raises(ValueError, match="model_id"):
        RunnerPipeline(spec).run(dry_run=True)


def test_dry_run_missing_output_dir_raises() -> None:
    spec = _make_spec("eval", {"modelId": "x"})
    with pytest.raises(ValueError, match="output_dir"):
        RunnerPipeline(spec).run(dry_run=True)


def test_dry_run_returns_none_for_valid_spec() -> None:
    result = RunnerPipeline(_smoke_spec()).run(dry_run=True)
    assert result is None


def test_dry_run_does_not_call_runner_functions() -> None:
    with (
        patch("forge.runners.pipeline.run_smoke") as mock_smoke,
        patch("forge.runners.pipeline.run_eval") as mock_eval,
    ):
        RunnerPipeline(_smoke_spec()).run(dry_run=True)
    mock_smoke.assert_not_called()
    mock_eval.assert_not_called()


# ── dispatch ──────────────────────────────────────────────────────────────────


def test_smoke_spec_dispatches_to_run_smoke() -> None:
    with patch("forge.runners.pipeline.run_smoke", return_value=_fake_result("smoke")) as mock:
        RunnerPipeline(_smoke_spec()).run()
    mock.assert_called_once()


def test_eval_spec_dispatches_to_run_eval() -> None:
    with patch("forge.runners.pipeline.run_eval", return_value=_fake_result("eval")) as mock:
        RunnerPipeline(_eval_spec()).run()
    mock.assert_called_once()


def test_params_normalized_before_dispatch() -> None:
    """camelCase YAML params must arrive as snake_case kwargs."""
    spec = _make_spec("smoke", {
        "modelId": "test/model",
        "outputDir": "/out",
        "nImages": 8,
        "numInferenceSteps": 4,
    })
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    with patch("forge.runners.pipeline.run_smoke", side_effect=capture):
        RunnerPipeline(spec).run()

    assert captured["model_id"] == "test/model"
    assert captured["output_dir"] == "/out"
    assert captured["n_images"] == 8
    assert captured["num_inference_steps"] == 4


def test_pipeline_returns_runner_result() -> None:
    with patch("forge.runners.pipeline.run_smoke", return_value=_fake_result("smoke")):
        result = RunnerPipeline(_smoke_spec()).run()
    assert isinstance(result, RunnerResult)


# ── RunnerJobSpec YAML validation ─────────────────────────────────────────────


def test_spec_parses_camel_case_runner_name() -> None:
    spec = RunnerJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "RunnerJob",
        "metadata": {"name": "my-job"},
        "runner": "smoke",
        "params": {},
    })
    assert spec.runner == "smoke"
    assert spec.metadata.name == "my-job"


def test_runner_result_to_dict_excludes_none_eval() -> None:
    r = RunnerResult(runner_type="smoke", model_id="x", n_generated=4, output_dir="/o")
    d = r.to_dict()
    assert "eval" not in d


def test_runner_result_to_dict_includes_eval_when_present() -> None:
    r = RunnerResult(
        runner_type="eval",
        model_id="x",
        n_generated=4,
        output_dir="/o",
        eval_result=EvalResult(clip_score=0.3, n_generated=4),
    )
    d = r.to_dict()
    assert "eval" in d
    assert d["eval"]["clip_score"] == pytest.approx(0.3)

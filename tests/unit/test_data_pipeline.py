"""Unit tests for forge.data.pipeline.DataPipeline."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import torch

from forge.core.config import DataJobSpec
from forge.data.pipeline import DataPipeline, _camel_to_snake, _normalize_params
from forge.data.types import StageResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spec(stages: list[dict]) -> DataJobSpec:
    return DataJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "DataJob",
        "metadata": {"name": "test-pipeline"},
        "stages": stages,
    })


def _embed_filter_spec(input_dir: str, embed_out: str, filter_out: str) -> DataJobSpec:
    return _make_spec([
        {
            "name": "embed",
            "type": "embed",
            "params": {"inputDir": input_dir, "outputDir": embed_out},
        },
        {
            "name": "filter",
            "type": "filter",
            "params": {"inputDir": embed_out, "outputDir": filter_out},
        },
    ])


# ── _camel_to_snake / _normalize_params ──────────────────────────────────────


@pytest.mark.parametrize("camel,snake", [
    ("inputDir", "input_dir"),
    ("outputDir", "output_dir"),
    ("clipSimilarityThreshold", "clip_similarity_threshold"),
    ("batchSize", "batch_size"),
    ("modelName", "model_name"),
    ("input_dir", "input_dir"),   # already snake — unchanged
])
def test_camel_to_snake(camel: str, snake: str) -> None:
    assert _camel_to_snake(camel) == snake


def test_normalize_params_converts_all_keys() -> None:
    raw = {"inputDir": "/in", "outputDir": "/out", "batchSize": 64}
    norm = _normalize_params(raw)
    assert norm == {"input_dir": "/in", "output_dir": "/out", "batch_size": 64}


# ── dry_run validation ────────────────────────────────────────────────────────


def test_dry_run_unknown_stage_type_raises() -> None:
    spec = _make_spec([{
        "name": "hash",
        "type": "perceptual_hash",
        "params": {"inputDir": "/in", "outputDir": "/out"},
    }])
    with pytest.raises(ValueError, match="unknown type"):
        DataPipeline(spec).run(dry_run=True)


def test_dry_run_missing_required_param_raises() -> None:
    spec = _make_spec([{
        "name": "embed",
        "type": "embed",
        "params": {"outputDir": "/out"},   # missing inputDir
    }])
    with pytest.raises(ValueError, match="input_dir"):
        DataPipeline(spec).run(dry_run=True)


def test_dry_run_does_not_call_stage_functions(tmp_path: Path) -> None:
    spec = _embed_filter_spec("/in", "/embed_out", "/filter_out")
    with (
        patch("forge.data.pipeline.run_embed") as mock_embed,
        patch("forge.data.pipeline.run_filter") as mock_filter,
    ):
        DataPipeline(spec).run(dry_run=True)
    mock_embed.assert_not_called()
    mock_filter.assert_not_called()


def test_dry_run_returns_empty_list(tmp_path: Path) -> None:
    spec = _embed_filter_spec("/in", "/embed_out", "/filter_out")
    results = DataPipeline(spec).run(dry_run=True)
    assert results == []


# ── CLIPScorer sharing ────────────────────────────────────────────────────────


def test_clip_scorer_created_once_for_embed_and_filter(tmp_path: Path) -> None:
    spec = _embed_filter_spec(
        str(tmp_path / "raw"), str(tmp_path / "embedded"), str(tmp_path / "filtered")
    )
    mock_embed_result = StageResult("embed", "embed", str(tmp_path / "embedded"))
    mock_filter_result = StageResult("filter", "filter", str(tmp_path / "filtered"))

    mock_clip = MagicMock()

    with (
        patch("forge.data.pipeline.CLIPScorer", return_value=mock_clip) as MockCLIP,
        patch("forge.data.pipeline.run_embed", return_value=mock_embed_result),
        patch("forge.data.pipeline.run_filter", return_value=mock_filter_result),
    ):
        DataPipeline(spec).run()
        MockCLIP.assert_called_once()


def test_clip_scorer_injected_into_embed_stage(tmp_path: Path) -> None:
    spec = _embed_filter_spec(
        str(tmp_path / "raw"), str(tmp_path / "embedded"), str(tmp_path / "filtered")
    )
    mock_clip = MagicMock()

    with (
        patch("forge.data.pipeline.CLIPScorer", return_value=mock_clip),
        patch("forge.data.pipeline.run_embed", return_value=StageResult("embed", "embed", "/out")) as mock_embed,
        patch("forge.data.pipeline.run_filter", return_value=StageResult("filter", "filter", "/out")),
    ):
        DataPipeline(spec).run()

    _, kwargs = mock_embed.call_args
    assert kwargs.get("clip_scorer") is mock_clip


def test_clip_scorer_injected_into_filter_stage(tmp_path: Path) -> None:
    spec = _embed_filter_spec(
        str(tmp_path / "raw"), str(tmp_path / "embedded"), str(tmp_path / "filtered")
    )
    mock_clip = MagicMock()

    with (
        patch("forge.data.pipeline.CLIPScorer", return_value=mock_clip),
        patch("forge.data.pipeline.run_embed", return_value=StageResult("embed", "embed", "/out")),
        patch("forge.data.pipeline.run_filter", return_value=StageResult("filter", "filter", "/out")) as mock_filter,
    ):
        DataPipeline(spec).run()

    _, kwargs = mock_filter.call_args
    assert kwargs.get("clip_scorer") is mock_clip


def test_no_clip_scorer_for_download_only() -> None:
    """Download stage doesn't need a CLIPScorer."""
    spec = _make_spec([{
        "name": "dl",
        "type": "download",
        "params": {"inputPath": "gs://bucket/urls.parquet", "outputDir": "/out"},
    }])

    with (
        patch("forge.data.pipeline.CLIPScorer") as MockCLIP,
        patch("forge.data.pipeline.run_download", return_value=StageResult("dl", "download", "/out")),
    ):
        DataPipeline(spec).run()
    MockCLIP.assert_not_called()


# ── Stage ordering and result collection ─────────────────────────────────────


def test_stages_called_in_spec_order(tmp_path: Path) -> None:
    spec = _embed_filter_spec(
        str(tmp_path / "raw"), str(tmp_path / "embedded"), str(tmp_path / "filtered")
    )
    call_order: list[str] = []

    def fake_embed(**kwargs):
        call_order.append("embed")
        return StageResult("embed", "embed", kwargs["output_dir"])

    def fake_filter(**kwargs):
        call_order.append("filter")
        return StageResult("filter", "filter", kwargs["output_dir"])

    with (
        patch("forge.data.pipeline.CLIPScorer"),
        patch("forge.data.pipeline.run_embed", side_effect=fake_embed),
        patch("forge.data.pipeline.run_filter", side_effect=fake_filter),
    ):
        DataPipeline(spec).run()

    assert call_order == ["embed", "filter"]


def test_run_returns_stage_results_list(tmp_path: Path) -> None:
    spec = _embed_filter_spec(
        str(tmp_path / "raw"), str(tmp_path / "embedded"), str(tmp_path / "filtered")
    )
    with (
        patch("forge.data.pipeline.CLIPScorer"),
        patch("forge.data.pipeline.run_embed", return_value=StageResult("embed", "embed", "/out")),
        patch("forge.data.pipeline.run_filter", return_value=StageResult("filter", "filter", "/out")),
    ):
        results = DataPipeline(spec).run()

    assert len(results) == 2
    assert all(isinstance(r, StageResult) for r in results)


def test_stage_name_overridden_from_spec(tmp_path: Path) -> None:
    """StageResult.stage_name should reflect the user-defined name, not the type."""
    spec = _make_spec([{
        "name": "my-custom-embed",
        "type": "embed",
        "params": {"inputDir": "/in", "outputDir": "/out"},
    }])
    with (
        patch("forge.data.pipeline.CLIPScorer"),
        patch("forge.data.pipeline.run_embed", return_value=StageResult("embed", "embed", "/out")),
    ):
        results = DataPipeline(spec).run()

    assert results[0].stage_name == "my-custom-embed"


def test_scorers_built_only_once_on_multiple_runs(tmp_path: Path) -> None:
    spec = _embed_filter_spec(
        str(tmp_path / "raw"), str(tmp_path / "embedded"), str(tmp_path / "filtered")
    )
    with (
        patch("forge.data.pipeline.CLIPScorer") as MockCLIP,
        patch("forge.data.pipeline.run_embed", return_value=StageResult("embed", "embed", "/out")),
        patch("forge.data.pipeline.run_filter", return_value=StageResult("filter", "filter", "/out")),
    ):
        pipeline = DataPipeline(spec)
        pipeline.run()
        pipeline.run()  # second run should NOT rebuild scorers
        MockCLIP.assert_called_once()

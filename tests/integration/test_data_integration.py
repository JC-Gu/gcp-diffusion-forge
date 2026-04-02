"""Integration tests for forge-data — requires network + model downloads.

Run with:  uv run pytest tests/integration/ --run-slow

Creates a small synthetic WebDataset (20 samples) and exercises the full
embed → filter pipeline with real CLIP model weights.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from forge.core.config import DataJobSpec
from forge.core.scorers.clip import CLIPScorer
from forge.data import DataPipeline, iter_webdataset, write_webdataset
from forge.data.embed import run_embed
from forge.data.filter import run_filter


# ── Synthetic dataset fixture ─────────────────────────────────────────────────


def _write_synthetic_shard(path: Path, start: int, count: int) -> None:
    """Write 'count' random 64×64 samples to a .tar shard."""
    rng = np.random.default_rng(start)
    with tarfile.open(path, "w") as tar:
        for i in range(count):
            key = f"{start + i:06d}"
            # JPEG
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG")
            _add(tar, f"{key}.jpg", buf.getvalue())
            # Caption
            caption = f"a synthetic image sample {start + i}"
            _add(tar, f"{key}.txt", caption.encode())
            # JSON
            meta = json.dumps({"idx": start + i, "split": "train"})
            _add(tar, f"{key}.json", meta.encode())


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture(scope="module")
def synthetic_wds_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Two shards of 10 samples each (20 total), at 64×64."""
    d = tmp_path_factory.mktemp("synthetic_wds")
    _write_synthetic_shard(d / "shard-000000.tar", start=0, count=10)
    _write_synthetic_shard(d / "shard-000001.tar", start=10, count=10)
    return str(d)


@pytest.fixture(scope="module")
def shared_clip() -> CLIPScorer:
    return CLIPScorer(model_name="ViT-L-14", pretrained="openai")


# ── iter_webdataset integration ───────────────────────────────────────────────


@pytest.mark.slow
def test_iter_webdataset_reads_all_samples(synthetic_wds_dir: str) -> None:
    samples = list(iter_webdataset(str(Path(synthetic_wds_dir) / "*.tar")))
    assert len(samples) == 20


@pytest.mark.slow
def test_iter_webdataset_sample_structure(synthetic_wds_dir: str) -> None:
    sample = next(iter_webdataset(str(Path(synthetic_wds_dir) / "*.tar")))
    assert "__key__" in sample
    assert isinstance(sample["jpg"], Image.Image)
    assert isinstance(sample["txt"], str)


# ── run_embed integration ─────────────────────────────────────────────────────


@pytest.mark.slow
def test_run_embed_adds_npy_embeddings(
    synthetic_wds_dir: str, tmp_path: Path, shared_clip: CLIPScorer
) -> None:
    out_dir = tmp_path / "embedded"
    result = run_embed(synthetic_wds_dir, str(out_dir), clip_scorer=shared_clip)

    assert result.output_dir == str(out_dir)
    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    assert len(samples) == 20
    for s in samples:
        assert "npy" in s
        assert isinstance(s["npy"], np.ndarray)
        assert s["npy"].shape == (768,)
        assert s["npy"].dtype == np.float32


@pytest.mark.slow
def test_run_embed_normalized_embeddings(
    synthetic_wds_dir: str, tmp_path: Path, shared_clip: CLIPScorer
) -> None:
    out_dir = tmp_path / "embedded_norm"
    run_embed(synthetic_wds_dir, str(out_dir), clip_scorer=shared_clip)
    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    for s in samples:
        norm = np.linalg.norm(s["npy"])
        assert abs(norm - 1.0) < 1e-3, f"Embedding not unit-normalized: norm={norm:.4f}"


# ── run_filter integration ────────────────────────────────────────────────────


@pytest.mark.slow
def test_run_filter_reduces_with_strict_aesthetic(
    synthetic_wds_dir: str, tmp_path: Path, shared_clip: CLIPScorer
) -> None:
    """Very strict aesthetic threshold should reduce sample count."""
    out_dir = tmp_path / "filtered_strict"
    run_filter(
        synthetic_wds_dir, str(out_dir),
        aesthetic_score_threshold=9.9,  # near-impossible threshold
        clip_similarity_threshold=0.0,
        min_resolution=0,
        clip_scorer=shared_clip,
    )
    samples = list(iter_webdataset(str(out_dir / "*.tar"))) if list(out_dir.glob("*.tar")) else []
    assert len(samples) < 20


@pytest.mark.slow
def test_run_filter_passes_all_with_zero_thresholds(
    synthetic_wds_dir: str, tmp_path: Path, shared_clip: CLIPScorer
) -> None:
    out_dir = tmp_path / "filtered_all_pass"
    run_filter(
        synthetic_wds_dir, str(out_dir),
        aesthetic_score_threshold=0.0,
        clip_similarity_threshold=0.0,
        min_resolution=0,
        clip_scorer=shared_clip,
    )
    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    assert len(samples) == 20


# ── DataPipeline end-to-end ───────────────────────────────────────────────────


@pytest.mark.slow
def test_pipeline_embed_filter_end_to_end(
    synthetic_wds_dir: str, tmp_path: Path
) -> None:
    """Full embed→filter pipeline returns two StageResults and writes output shards."""
    embed_out = str(tmp_path / "embedded")
    filter_out = str(tmp_path / "filtered")

    spec = DataJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "DataJob",
        "metadata": {"name": "integration-test"},
        "stages": [
            {
                "name": "embed",
                "type": "embed",
                "params": {"inputDir": synthetic_wds_dir, "outputDir": embed_out},
            },
            {
                "name": "filter",
                "type": "filter",
                "params": {
                    "inputDir": embed_out,
                    "outputDir": filter_out,
                    "aestheticScoreThreshold": 0.0,
                    "clipSimilarityThreshold": 0.0,
                    "minResolution": 0,
                },
            },
        ],
    })

    pipeline = DataPipeline(spec)
    results = pipeline.run()

    assert len(results) == 2
    assert results[0].stage_name == "embed"
    assert results[1].stage_name == "filter"
    assert list(Path(filter_out).glob("*.tar"))


@pytest.mark.slow
def test_pipeline_shares_clip_scorer(
    synthetic_wds_dir: str, tmp_path: Path
) -> None:
    """CLIPScorer must be instantiated exactly once for embed+filter pipeline."""
    from unittest.mock import patch

    original_init = CLIPScorer.__init__
    count: list[int] = [0]

    def counting_init(self, **kwargs):
        count[0] += 1
        original_init(self, **kwargs)

    spec = DataJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "DataJob",
        "metadata": {"name": "clip-sharing-test"},
        "stages": [
            {
                "name": "embed",
                "type": "embed",
                "params": {"inputDir": synthetic_wds_dir, "outputDir": str(tmp_path / "emb")},
            },
            {
                "name": "filter",
                "type": "filter",
                "params": {
                    "inputDir": str(tmp_path / "emb"),
                    "outputDir": str(tmp_path / "filt"),
                    "aestheticScoreThreshold": 0.0,
                    "clipSimilarityThreshold": 0.0,
                    "minResolution": 0,
                },
            },
        ],
    })

    with patch.object(CLIPScorer, "__init__", counting_init):
        DataPipeline(spec).run()

    assert count[0] == 1, f"CLIPScorer instantiated {count[0]} times; expected 1"


@pytest.mark.slow
def test_pipeline_dry_run_writes_nothing(
    synthetic_wds_dir: str, tmp_path: Path
) -> None:
    spec = DataJobSpec.model_validate({
        "apiVersion": "forge/v1",
        "kind": "DataJob",
        "metadata": {"name": "dry-run-test"},
        "stages": [
            {
                "name": "embed",
                "type": "embed",
                "params": {"inputDir": synthetic_wds_dir, "outputDir": str(tmp_path / "out")},
            },
        ],
    })
    results = DataPipeline(spec).run(dry_run=True)
    assert results == []
    assert not list(tmp_path.glob("**/*.tar"))

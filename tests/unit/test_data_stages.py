"""Unit tests for forge.data stage functions — no GPU, no network."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from forge.data.download import run_download
from forge.data.embed import run_embed
from forge.data.filter import run_filter
from forge.data.types import StageResult, WdsSample


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_jpeg_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 80, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def _write_shard(path: Path, n_samples: int, image_size=(128, 128), add_npy: bool = False) -> None:
    """Write a minimal WebDataset .tar shard to path."""
    with tarfile.open(path, "w") as tar:
        for i in range(n_samples):
            key = f"{i:06d}"
            # JPEG
            jpg = _make_jpeg_bytes(image_size)
            _add_file(tar, f"{key}.jpg", jpg)
            # Caption
            cap = f"a photo of object {i}".encode()
            _add_file(tar, f"{key}.txt", cap)
            # JSON metadata
            meta = json.dumps({"idx": i}).encode()
            _add_file(tar, f"{key}.json", meta)
            # Optional pre-computed embedding
            if add_npy:
                emb = np.random.randn(768).astype("float32")
                buf = io.BytesIO()
                np.save(buf, emb)
                _add_file(tar, f"{key}.npy", buf.getvalue())


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    """Directory with one 5-sample shard."""
    _write_shard(tmp_path / "shard-000000.tar", n_samples=5)
    return tmp_path


@pytest.fixture
def shard_dir_large_images(tmp_path: Path) -> Path:
    """Directory with 5-sample shard at 256×256."""
    _write_shard(tmp_path / "shard-000000.tar", n_samples=5, image_size=(256, 256))
    return tmp_path


@pytest.fixture
def mock_clip_scorer() -> MagicMock:
    scorer = MagicMock(spec=["score", "embed_images", "embed_texts", "_load"])
    scorer.embed_images.return_value = torch.randn(5, 768)
    scorer.score.return_value = torch.tensor([0.35, 0.40, 0.25, 0.30, 0.38])
    return scorer


@pytest.fixture
def mock_aesthetic_scorer() -> MagicMock:
    scorer = MagicMock(spec=["score", "_load"])
    scorer.score.return_value = torch.tensor([5.0, 6.0, 4.0, 5.5, 5.2])
    return scorer


# ── run_download ──────────────────────────────────────────────────────────────


def test_run_download_calls_img2dataset(tmp_path: Path) -> None:
    with patch("forge.data.download.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        result = run_download(
            input_path="gs://bucket/urls.parquet",
            output_dir=str(tmp_path / "output"),
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "img2dataset"


def test_run_download_passes_image_size(tmp_path: Path) -> None:
    with patch("forge.data.download.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        run_download("gs://bucket/urls.parquet", str(tmp_path), image_size=1024)
    cmd = mock_run.call_args[0][0]
    assert "--image_size=1024" in cmd


def test_run_download_passes_resize_mode(tmp_path: Path) -> None:
    with patch("forge.data.download.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        run_download("gs://bucket/urls.parquet", str(tmp_path), resize_mode="border")
    cmd = mock_run.call_args[0][0]
    assert "--resize_mode=border" in cmd


def test_run_download_returns_stage_result(tmp_path: Path) -> None:
    with patch("forge.data.download.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        result = run_download("gs://bucket/urls.parquet", str(tmp_path))
    assert isinstance(result, StageResult)
    assert result.stage_type == "download"
    assert result.output_dir == str(tmp_path)


def test_run_download_raises_on_nonzero_exit(tmp_path: Path) -> None:
    with patch("forge.data.download.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "img2dataset")
        with pytest.raises(subprocess.CalledProcessError):
            run_download("gs://bucket/urls.parquet", str(tmp_path))


# ── run_embed ─────────────────────────────────────────────────────────────────


def test_run_embed_writes_output_shards(
    shard_dir: Path, tmp_path: Path, mock_clip_scorer: MagicMock
) -> None:
    out_dir = tmp_path / "embedded"
    result = run_embed(str(shard_dir), str(out_dir), clip_scorer=mock_clip_scorer)
    assert list(out_dir.glob("*.tar")), "Expected .tar shards in output dir"
    assert isinstance(result, StageResult)


def test_run_embed_calls_embed_images(
    shard_dir: Path, tmp_path: Path, mock_clip_scorer: MagicMock
) -> None:
    run_embed(str(shard_dir), str(tmp_path / "out"), clip_scorer=mock_clip_scorer)
    assert mock_clip_scorer.embed_images.called


def test_run_embed_adds_npy_to_samples(
    shard_dir: Path, tmp_path: Path, mock_clip_scorer: MagicMock
) -> None:
    """Samples in output shards must have a 'npy' field."""
    from forge.data.io import iter_webdataset
    out_dir = tmp_path / "embedded"
    run_embed(str(shard_dir), str(out_dir), clip_scorer=mock_clip_scorer)
    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    assert all("npy" in s for s in samples)


def test_run_embed_npy_shape(
    shard_dir: Path, tmp_path: Path, mock_clip_scorer: MagicMock
) -> None:
    from forge.data.io import iter_webdataset
    out_dir = tmp_path / "embedded"
    run_embed(str(shard_dir), str(out_dir), clip_scorer=mock_clip_scorer)
    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    for s in samples:
        assert isinstance(s["npy"], np.ndarray)
        assert s["npy"].dtype == np.float32
        assert s["npy"].ndim == 1


def test_run_embed_raises_if_no_shards(tmp_path: Path, mock_clip_scorer: MagicMock) -> None:
    with pytest.raises(ValueError, match="No .tar shards"):
        run_embed(str(tmp_path / "empty"), str(tmp_path / "out"), clip_scorer=mock_clip_scorer)


def test_run_embed_creates_clip_scorer_when_none(
    shard_dir: Path, tmp_path: Path
) -> None:
    with patch("forge.data.embed.CLIPScorer") as MockCLIP:
        instance = MagicMock()
        instance.embed_images.return_value = torch.randn(5, 768)
        MockCLIP.return_value = instance
        run_embed(str(shard_dir), str(tmp_path / "out"), clip_scorer=None)
        MockCLIP.assert_called_once()


# ── run_filter ────────────────────────────────────────────────────────────────


def test_run_filter_returns_stage_result(
    shard_dir_large_images: Path,
    tmp_path: Path,
    mock_clip_scorer: MagicMock,
    mock_aesthetic_scorer: MagicMock,
) -> None:
    with patch("forge.data.filter.AestheticScorer", return_value=mock_aesthetic_scorer):
        result = run_filter(
            str(shard_dir_large_images),
            str(tmp_path / "filtered"),
            clip_similarity_threshold=0.0,
            aesthetic_score_threshold=0.0,
            min_resolution=0,
            clip_scorer=mock_clip_scorer,
        )
    assert isinstance(result, StageResult)
    assert result.stage_type == "filter"


def test_run_filter_resolution_removes_small_images(
    tmp_path: Path, mock_clip_scorer: MagicMock, mock_aesthetic_scorer: MagicMock
) -> None:
    """Images smaller than min_resolution must be filtered out."""
    _write_shard(tmp_path / "shard-000000.tar", n_samples=5, image_size=(64, 64))
    out_dir = tmp_path / "filtered"

    from forge.data.io import iter_webdataset
    with patch("forge.data.filter.AestheticScorer", return_value=mock_aesthetic_scorer):
        run_filter(
            str(tmp_path), str(out_dir),
            min_resolution=128,           # 64×64 images should all fail
            clip_similarity_threshold=0.0,
            aesthetic_score_threshold=0.0,
            clip_scorer=mock_clip_scorer,
        )

    samples = list(iter_webdataset(str(out_dir / "*.tar"))) if list(out_dir.glob("*.tar")) else []
    assert len(samples) == 0


def test_run_filter_clip_threshold_removes_low_scores(
    shard_dir_large_images: Path,
    tmp_path: Path,
    mock_aesthetic_scorer: MagicMock,
) -> None:
    """Samples with CLIP similarity below threshold must be removed."""
    from forge.data.io import iter_webdataset

    mock_clip = MagicMock(spec=["score", "embed_images", "embed_texts", "_load"])
    # All scores below a very strict threshold
    mock_clip.score.return_value = torch.tensor([0.10, 0.12, 0.09, 0.11, 0.08])

    out_dir = tmp_path / "filtered"
    with patch("forge.data.filter.AestheticScorer", return_value=mock_aesthetic_scorer):
        run_filter(
            str(shard_dir_large_images), str(out_dir),
            clip_similarity_threshold=0.50,  # strict — all fail
            aesthetic_score_threshold=0.0,
            min_resolution=0,
            clip_scorer=mock_clip,
        )

    samples = list(iter_webdataset(str(out_dir / "*.tar"))) if list(out_dir.glob("*.tar")) else []
    assert len(samples) == 0


def test_run_filter_aesthetic_threshold(
    shard_dir_large_images: Path,
    tmp_path: Path,
    mock_clip_scorer: MagicMock,
) -> None:
    """Samples with aesthetic score below threshold must be removed."""
    from forge.data.io import iter_webdataset

    # Scores: [5.0, 6.0, 4.0, 5.5, 5.2] → threshold 5.3 keeps indices 1 (6.0) and 3 (5.5)
    mock_ae = MagicMock(spec=["score", "_load"])
    mock_ae.score.return_value = torch.tensor([5.0, 6.0, 4.0, 5.5, 5.2])

    out_dir = tmp_path / "filtered"
    with patch("forge.data.filter.AestheticScorer", return_value=mock_ae):
        run_filter(
            str(shard_dir_large_images), str(out_dir),
            aesthetic_score_threshold=5.3,
            clip_similarity_threshold=0.0,
            min_resolution=0,
            clip_scorer=mock_clip_scorer,
        )

    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    assert len(samples) == 2


def test_run_filter_all_pass_when_thresholds_zero(
    shard_dir_large_images: Path,
    tmp_path: Path,
    mock_clip_scorer: MagicMock,
    mock_aesthetic_scorer: MagicMock,
) -> None:
    from forge.data.io import iter_webdataset
    out_dir = tmp_path / "filtered"
    with patch("forge.data.filter.AestheticScorer", return_value=mock_aesthetic_scorer):
        run_filter(
            str(shard_dir_large_images), str(out_dir),
            clip_similarity_threshold=0.0,
            aesthetic_score_threshold=0.0,
            min_resolution=0,
            clip_scorer=mock_clip_scorer,
        )
    samples = list(iter_webdataset(str(out_dir / "*.tar")))
    assert len(samples) == 5


def test_run_filter_creates_aesthetic_scorer_with_shared_clip(
    shard_dir_large_images: Path, tmp_path: Path, mock_clip_scorer: MagicMock
) -> None:
    """AestheticScorer must receive the injected CLIPScorer."""
    mock_ae = MagicMock(spec=["score", "_load"])
    mock_ae.score.return_value = torch.ones(5) * 6.0

    with patch("forge.data.filter.AestheticScorer", return_value=mock_ae) as MockAE:
        run_filter(
            str(shard_dir_large_images), str(tmp_path / "out"),
            aesthetic_score_threshold=0.0,
            clip_similarity_threshold=0.0,
            min_resolution=0,
            clip_scorer=mock_clip_scorer,
        )
    MockAE.assert_called_once()
    _, kwargs = MockAE.call_args
    assert kwargs.get("clip_scorer") is mock_clip_scorer


def test_run_filter_raises_if_no_shards(
    tmp_path: Path, mock_clip_scorer: MagicMock
) -> None:
    with pytest.raises(ValueError, match="No .tar shards"):
        run_filter(str(tmp_path / "empty"), str(tmp_path / "out"), clip_scorer=mock_clip_scorer)


# ── run_caption ───────────────────────────────────────────────────────────────


def test_run_caption_raises_without_transformers(
    shard_dir: Path, tmp_path: Path
) -> None:
    """ImportError with helpful message when transformers is not installed."""
    import sys
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        with patch("forge.data.caption._load_florence", side_effect=ImportError(
            "run_caption requires transformers>=4.47. Install with: uv add forge-data[caption]"
        )):
            with pytest.raises(ImportError, match="transformers"):
                from forge.data.caption import run_caption
                run_caption(str(shard_dir), str(tmp_path / "out"))

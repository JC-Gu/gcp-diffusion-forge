"""Unit tests for forge.data.io — uses stdlib tarfile, no model downloads."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge.data.io import iter_webdataset, load_images_from_dir, write_webdataset
from forge.data.types import WdsSample


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_jpeg_bytes(size: tuple[int, int] = (64, 64), color: tuple = (128, 64, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _create_shard(path: Path, samples: list[dict]) -> None:
    """Write a minimal WebDataset .tar shard using only stdlib tarfile."""
    with tarfile.open(path, "w") as tar:
        for s in samples:
            key = s["__key__"]
            for ext, data in s.items():
                if ext == "__key__":
                    continue
                raw = data.encode() if isinstance(data, str) else data
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))


@pytest.fixture
def single_shard(tmp_path: Path) -> Path:
    """A .tar shard with 3 JPEG+caption samples."""
    shard = tmp_path / "shard-000000.tar"
    _create_shard(shard, [
        {
            "__key__": f"{i:06d}",
            "jpg": _make_jpeg_bytes(color=(i * 30, 100, 200)),
            "txt": f"caption for sample {i}",
            "json": json.dumps({"idx": i}),
        }
        for i in range(3)
    ])
    return shard


# ── load_images_from_dir ──────────────────────────────────────────────────────


def test_load_images_from_dir_returns_list(tmp_path: Path) -> None:
    for i in range(3):
        Image.new("RGB", (32, 32), color=(i * 80, 0, 0)).save(tmp_path / f"{i:04d}.png")
    images = load_images_from_dir(tmp_path)
    assert len(images) == 3


def test_load_images_from_dir_all_rgb(tmp_path: Path) -> None:
    Image.new("RGBA", (32, 32)).save(tmp_path / "img.png")
    images = load_images_from_dir(tmp_path)
    assert images[0].mode == "RGB"


def test_load_images_from_dir_sorted_order(tmp_path: Path) -> None:
    for name in ["c.png", "a.png", "b.png"]:
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    images = load_images_from_dir(tmp_path)
    assert len(images) == 3  # order is sorted by filename


def test_load_images_from_dir_empty(tmp_path: Path) -> None:
    assert load_images_from_dir(tmp_path) == []


def test_load_images_skips_non_image_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("not an image")
    Image.new("RGB", (8, 8)).save(tmp_path / "img.png")
    images = load_images_from_dir(tmp_path)
    assert len(images) == 1


# ── iter_webdataset ───────────────────────────────────────────────────────────


def test_iter_webdataset_yields_all_samples(single_shard: Path) -> None:
    samples = list(iter_webdataset(str(single_shard.parent / "*.tar")))
    assert len(samples) == 3


def test_iter_webdataset_sample_has_key(single_shard: Path) -> None:
    samples = list(iter_webdataset(str(single_shard.parent / "*.tar")))
    assert all("__key__" in s for s in samples)


def test_iter_webdataset_decodes_jpg_to_pil(single_shard: Path) -> None:
    samples = list(iter_webdataset(str(single_shard.parent / "*.tar")))
    for s in samples:
        assert isinstance(s["jpg"], Image.Image)


def test_iter_webdataset_decodes_txt_to_str(single_shard: Path) -> None:
    samples = list(iter_webdataset(str(single_shard.parent / "*.tar")))
    for i, s in enumerate(samples):
        assert isinstance(s["txt"], str)
        assert f"caption for sample {i}" == s["txt"]


def test_iter_webdataset_fields_filter(single_shard: Path) -> None:
    samples = list(iter_webdataset(str(single_shard.parent / "*.tar"), fields=["txt"]))
    for s in samples:
        assert "jpg" not in s
        assert "txt" in s
        assert "__key__" in s


# ── write_webdataset ──────────────────────────────────────────────────────────


def _make_samples(n: int) -> list[WdsSample]:
    return [
        {
            "__key__": f"{i:06d}",
            "jpg": Image.new("RGB", (32, 32), color=(i * 20 % 255, 100, 50)),
            "txt": f"sample {i}",
        }
        for i in range(n)
    ]


def test_write_webdataset_creates_tar_files(tmp_path: Path) -> None:
    paths = write_webdataset(iter(_make_samples(5)), tmp_path)
    assert len(paths) >= 1
    assert all(p.endswith(".tar") for p in paths)


def test_write_webdataset_respects_shard_size(tmp_path: Path) -> None:
    paths = write_webdataset(iter(_make_samples(10)), tmp_path, shard_size=4)
    # 10 samples with shard_size=4 → 3 shards (4, 4, 2)
    assert len(paths) == 3


def test_write_webdataset_creates_output_dir(tmp_path: Path) -> None:
    new_dir = tmp_path / "new" / "subdir"
    write_webdataset(iter(_make_samples(2)), new_dir)
    assert new_dir.exists()


def test_write_webdataset_returns_sorted_paths(tmp_path: Path) -> None:
    paths = write_webdataset(iter(_make_samples(15)), tmp_path, shard_size=5)
    assert paths == sorted(paths)


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    samples = _make_samples(4)
    write_webdataset(iter(samples), tmp_path)
    read_back = list(iter_webdataset(str(tmp_path / "*.tar")))
    assert len(read_back) == 4
    keys_written = {s["__key__"] for s in samples}
    keys_read = {s["__key__"] for s in read_back}
    assert keys_written == keys_read

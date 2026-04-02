"""WebDataset I/O utilities.

Provides streaming read/write helpers for WebDataset .tar shards and
convenience loaders for small local image directories.
"""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path
from typing import Iterator

from PIL import Image

import webdataset as wds

from forge.data.types import WdsSample

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def load_images_from_dir(
    directory: str | os.PathLike,
    extensions: frozenset[str] = _IMAGE_EXTENSIONS,
) -> list[Image.Image]:
    """Load all images from a flat directory as RGB PIL Images.

    Not streaming — intended for small test sets or evaluation samples only.
    Use iter_webdataset for large datasets.

    Args:
        directory:  Path to directory containing image files.
        extensions: Set of lowercase file extensions to include.

    Returns:
        List of PIL Images in sorted filename order.
    """
    images: list[Image.Image] = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in extensions:
            images.append(Image.open(path).convert("RGB"))
    return images


def iter_webdataset(
    url_pattern: str,
    fields: list[str] | None = None,
    shardshuffle: bool = False,
) -> Iterator[WdsSample]:
    """Stream decoded samples from WebDataset .tar shards.

    Decodes image fields to PIL Images, text to str, JSON to dict, and
    .npy files to numpy arrays. Corrupt samples are skipped with a warning
    rather than aborting the pipeline.

    Args:
        url_pattern:  Glob for local shards (e.g. "/data/*.tar") or a
                      braceexpand pattern for GCS (e.g. "gs://b/{000..099}.tar").
        fields:       If set, only yield these keys (plus __key__ always).
        shardshuffle: Shuffle shard order. Use True only during training.

    Yields:
        WdsSample dicts with decoded field values.
    """
    # WebDataset doesn't expand shell globs — do it here for local paths.
    if "*" in url_pattern or "?" in url_pattern:
        urls: list[str] | str = sorted(_glob.glob(url_pattern))
        if not urls:
            return
    else:
        urls = url_pattern

    dataset = (
        wds.WebDataset(urls, shardshuffle=shardshuffle, handler=wds.warn_and_continue, empty_check=False)
        .decode("pil")
    )
    for sample in dataset:
        if fields is not None:
            sample = {k: v for k, v in sample.items() if k == "__key__" or k in fields}
        yield sample


def write_webdataset(
    samples: Iterator[WdsSample],
    output_dir: str | os.PathLike,
    shard_size: int = 10_000,
    shard_prefix: str = "shard",
) -> list[str]:
    """Write a stream of WdsSamples to numbered .tar shards.

    PIL Image values are re-encoded to JPEG. numpy arrays are stored as .npy.
    The output directory is created if it does not exist.

    Args:
        samples:      Iterator of WdsSample dicts.
        output_dir:   Directory where .tar files are written.
        shard_size:   Maximum samples per shard file.
        shard_prefix: Filename prefix. "shard" → "shard-000000.tar".

    Returns:
        Sorted list of absolute paths of written shard files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pattern = str(output_path / f"{shard_prefix}-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=shard_size) as writer:
        for sample in samples:
            writer.write(dict(sample))

    return sorted(str(p) for p in output_path.glob(f"{shard_prefix}-*.tar"))

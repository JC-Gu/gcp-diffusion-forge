"""Download stage: wraps img2dataset CLI to fetch images from URL lists."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from forge.data.types import StageResult

_log = logging.getLogger(__name__)


def run_download(
    input_path: str,
    output_dir: str,
    image_size: int = 512,
    resize_mode: str = "center_crop",
    processes_count: int = 4,
    url_col: str = "url",
    caption_col: str = "caption",
    output_format: str = "webdataset",
) -> StageResult:
    """Download images from a URL list using img2dataset.

    Wraps the img2dataset CLI via subprocess. img2dataset must be installed
    separately:  uv add forge-data[download]

    Args:
        input_path:      Local or GCS path to a parquet/CSV file with image URLs.
        output_dir:      Directory where WebDataset .tar shards are written.
        image_size:      Target dimension for both width and height after resize.
        resize_mode:     One of "center_crop", "border", "no", "keep_ratio".
        processes_count: Number of parallel download workers.
        url_col:         Column name in input file that contains image URLs.
        caption_col:     Column name containing captions (written to .txt files).
        output_format:   img2dataset output format; "webdataset" produces .tar shards.

    Returns:
        StageResult with output_dir and elapsed time.

    Raises:
        FileNotFoundError: if the img2dataset CLI is not on PATH.
        subprocess.CalledProcessError: if img2dataset exits with a non-zero code.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    start = time.time()

    cmd: list[str] = [
        "img2dataset",
        f"--url_list={input_path}",
        f"--output_folder={output_dir}",
        f"--image_size={image_size}",
        f"--resize_mode={resize_mode}",
        f"--processes_count={processes_count}",
        f"--url_col={url_col}",
        f"--caption_col={caption_col}",
        f"--output_format={output_format}",
        "--enable_wandb=False",
    ]

    _log.debug("Running: %s", " ".join(cmd))

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)

    for line in proc.stdout.splitlines():
        _log.debug("img2dataset: %s", line)
    for line in proc.stderr.splitlines():
        _log.debug("img2dataset stderr: %s", line)

    return StageResult(
        stage_name="download",
        stage_type="download",
        output_dir=output_dir,
        elapsed_sec=time.time() - start,
    )

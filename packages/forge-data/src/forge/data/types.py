"""Core data types for the forge-data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from typing import TypedDict


class WdsSample(TypedDict, total=False):
    """A single decoded sample from a WebDataset .tar shard.

    Keys match WebDataset file extensions within the shard.
    After iter_webdataset() decodes with mode="pil", image fields hold
    PIL Images rather than raw bytes.

    Fields added progressively by pipeline stages:
      npy  — float32 CLIP embedding [D], added by the embed stage.
    """

    __key__: str               # sample identifier (basename, e.g. "000123")
    jpg: Image.Image           # PIL Image decoded from .jpg
    png: Image.Image           # PIL Image decoded from .png
    txt: str                   # caption text (decoded from .txt)
    json: dict[str, Any]       # metadata (parsed from .json)
    npy: np.ndarray            # CLIP embedding float32 [D], added by embed stage


@dataclass
class StageResult:
    """Result returned by each data pipeline stage.

    Attributes:
        stage_name:   The user-defined name from the DataJobSpec (e.g. "embed").
        stage_type:   The registered stage type key (e.g. "embed", "filter").
        output_dir:   Path to the directory where output shards were written.
        n_samples:    Number of samples written. None when output is streaming
                      and the count was not tracked.
        elapsed_sec:  Wall-clock time for this stage.
    """

    stage_name: str
    stage_type: str
    output_dir: str
    n_samples: int | None = None
    elapsed_sec: float = 0.0

    def __str__(self) -> str:
        parts = [f"stage={self.stage_name!r}", f"output={self.output_dir!r}"]
        if self.n_samples is not None:
            parts.append(f"n_samples={self.n_samples}")
        parts.append(f"elapsed={self.elapsed_sec:.1f}s")
        return f"StageResult({', '.join(parts)})"

"""forge-data: Data curation pipeline for diffusion model training.

Stages: download → embed → filter → caption → WebDataset .tar shards

Public API::

    from forge.data import DataPipeline, WdsSample
    from forge.data import run_download, run_embed, run_filter, run_caption
    from forge.data import load_images_from_dir, iter_webdataset, write_webdataset
"""

from forge.data.caption import run_caption
from forge.data.download import run_download
from forge.data.embed import run_embed
from forge.data.filter import run_filter
from forge.data.io import iter_webdataset, load_images_from_dir, write_webdataset
from forge.data.pipeline import DataPipeline
from forge.data.types import StageResult, WdsSample

__all__ = [
    "DataPipeline",
    "StageResult",
    "WdsSample",
    "load_images_from_dir",
    "iter_webdataset",
    "write_webdataset",
    "run_download",
    "run_embed",
    "run_filter",
    "run_caption",
]

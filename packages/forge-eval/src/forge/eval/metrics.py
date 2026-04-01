"""Standalone metric functions for diffusion model evaluation.

Each function is independent — usable directly from a training loop or
via EvalRunner for multi-metric orchestration.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from PIL import Image

from forge.core.device import get_device
from forge.core.scorers.aesthetic import AestheticScorer
from forge.core.scorers.clip import CLIPScorer

# Lazy import: cleanfid pulls in torchvision; only load when FID is actually called.
try:
    from cleanfid import fid as _cleanfid_fid  # type: ignore[import-untyped]

    _CLEANFID_AVAILABLE = True
except ImportError:
    _CLEANFID_AVAILABLE = False

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_FID_MIN_IMAGES = 2048


def compute_clip_score(
    images: list[Image.Image],
    prompts: list[str],
    scorer: CLIPScorer | None = None,
) -> float:
    """Mean CLIP cosine similarity between each image and its paired prompt.

    Args:
        images:  List of PIL Images.
        prompts: List of text prompts, one per image.
        scorer:  Pre-loaded CLIPScorer to reuse. Created with ViT-L-14/openai if None.

    Returns:
        Mean cosine similarity as a plain Python float in [-1, 1].
    """
    if scorer is None:
        _, device = get_device()
        scorer = CLIPScorer(model_name="ViT-L-14", pretrained="openai", device=device)
    scores = scorer.score(images, prompts)
    return float(scores.mean().item())


def compute_aesthetic_score(
    images: list[Image.Image],
    scorer: AestheticScorer | None = None,
) -> float:
    """Mean LAION aesthetic score across all images.

    Args:
        images: List of PIL Images.
        scorer: Pre-loaded AestheticScorer to reuse. Created fresh if None.

    Returns:
        Mean aesthetic score as a plain Python float (~0–10).
    """
    if scorer is None:
        _, device = get_device()
        scorer = AestheticScorer(device=device)
    scores = scorer.score(images)
    return float(scores.mean().item())


def compute_fid(
    generated_dir: str | os.PathLike,
    reference_dir: str | os.PathLike,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> float:
    """FID between a directory of generated images and a reference directory.

    Uses clean-fid (mode='clean') which corrects for JPEG resizing artifacts
    present in the original FID implementation.

    Args:
        generated_dir:  Path to directory of generated images.
        reference_dir:  Path to directory of reference images.
        device:         torch.device. Auto-detected if None.
        batch_size:     Images per forward pass through the feature extractor.

    Returns:
        FID score as a plain Python float (lower is better; 0 = identical distributions).

    Warns:
        UserWarning if generated_dir contains fewer than 2048 images, as FID
        statistics are unreliable with small sample sizes.
    """
    if not _CLEANFID_AVAILABLE:
        raise ImportError(
            "clean-fid is required for compute_fid. "
            "Install it with: uv add clean-fid"
        )

    n = _count_images(generated_dir)
    if n < _FID_MIN_IMAGES:
        warnings.warn(
            f"compute_fid: only {n} images found in generated_dir "
            f"(recommend >= {_FID_MIN_IMAGES} for reliable FID statistics).",
            UserWarning,
            stacklevel=2,
        )

    if device is None:
        _, device = get_device()

    score = _cleanfid_fid.compute_fid(
        fdir1=str(generated_dir),
        fdir2=str(reference_dir),
        mode="clean",
        device=device,
        batch_size=batch_size,
        num_workers=0,
    )
    return float(score)


def _count_images(directory: str | os.PathLike) -> int:
    """Count image files in a directory (non-recursive)."""
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in _IMAGE_EXTS:
                count += 1
    return count

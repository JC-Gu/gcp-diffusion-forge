"""EvalRunner stage: generate images then score with forge-eval metrics.

Handles larger prompt sets (file or inline), batched generation, and the
full forge-eval suite (clip_score, aesthetic_score, fid).  FID is enabled
automatically when reference_dir is provided.

This is the recommended runner for:
- Local validation against a held-out prompt set
- Kubernetes eval jobs (same code, different outputDir / promptsPath)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from PIL import Image

from forge.core.device import get_device
from forge.eval.runner import EvalRunner as _MetricsRunner
from forge.runners.diffusion import DiffusionWrapper
from forge.runners.types import RunnerResult

_log = logging.getLogger(__name__)

_DEFAULT_METRICS: list[str] = ["clip_score", "aesthetic_score"]


def _load_prompts(prompts_path: str) -> list[str]:
    """Load one prompt per line from a plain-text file, skipping blank lines."""
    lines = Path(prompts_path).read_text().splitlines()
    return [l.strip() for l in lines if l.strip()]


def run_eval(
    model_id: str,
    output_dir: str,
    prompts: list[str] | None = None,
    prompts_path: str | None = None,
    n_images: int | None = None,
    reference_dir: str | None = None,
    batch_size: int = 4,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
    metrics: list[str] | None = None,
    device: torch.device | None = None,
    diffusion: DiffusionWrapper | None = None,
) -> RunnerResult:
    """Generate images and evaluate quality metrics.

    Prompt resolution order (first wins):
      1. ``prompts`` kwarg (inline list)
      2. ``prompts_path`` (one prompt per line text file)

    If neither is provided, n_images random-category prompts are used.

    FID is included automatically when ``reference_dir`` is given; it is
    silently omitted otherwise so the caller doesn't need to branch.

    Args:
        model_id:             HuggingFace model ID or local path.
        output_dir:           Directory to write generated images as PNGs.
        prompts:              Inline prompt list (takes priority over prompts_path).
        prompts_path:         Path to a plain-text file, one prompt per line.
        n_images:             Limit generation to the first n prompts. Useful
                              for quick local tests against a large prompt file.
        reference_dir:        Directory of reference images for FID. When
                              provided, "fid" is added to metrics automatically.
        batch_size:           Prompts per inference call (tune for VRAM).
        num_inference_steps:  Denoising steps.
        guidance_scale:       CFG scale.
        metrics:              Metrics to compute. Defaults to clip_score +
                              aesthetic_score (+ fid if reference_dir set).
        device:               torch.device. Auto-detected if None.
        diffusion:            Pre-built DiffusionWrapper (inject mock in tests).

    Returns:
        RunnerResult with eval_result containing all requested metrics.

    Raises:
        ValueError: if no prompts are available from any source.
    """
    active_prompts = _resolve_prompts(prompts, prompts_path, n_images)

    _metrics = list(metrics) if metrics is not None else list(_DEFAULT_METRICS)
    if reference_dir is not None and "fid" not in _metrics:
        _metrics.append("fid")

    if device is None:
        _, device = get_device()

    if diffusion is None:
        diffusion = DiffusionWrapper(
            model_id=model_id,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start = time.time()
    _log.info(
        "EvalRunner: generating %d images with %s (batch_size=%d)",
        len(active_prompts),
        model_id,
        batch_size,
    )

    all_images: list[Image.Image] = []
    for batch_start in range(0, len(active_prompts), batch_size):
        batch = active_prompts[batch_start : batch_start + batch_size]
        all_images.extend(diffusion.generate(batch))

    # Save to disk (enables FID via reference_dir, and inspection)
    for i, img in enumerate(all_images):
        img.save(out_path / f"{i:06d}.png")

    metrics_runner = _MetricsRunner(
        metrics=_metrics,
        reference_dir=reference_dir,
        device=device,
    )
    eval_result = metrics_runner.run(all_images, prompts=active_prompts)

    elapsed = time.time() - start
    _log.info("EvalRunner done in %.1fs", elapsed)

    return RunnerResult(
        runner_type="eval",
        model_id=model_id,
        n_generated=len(all_images),
        output_dir=output_dir,
        eval_result=eval_result,
        elapsed_sec=elapsed,
        metadata={
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "batch_size": batch_size,
            "prompts_path": prompts_path,
        },
    )


# ── Prompt resolution ─────────────────────────────────────────────────────────

_FALLBACK_PROMPTS = [
    "a photograph of a landscape",
    "a close-up of a flower",
    "a city street at night",
    "a portrait of a person smiling",
]


def _resolve_prompts(
    prompts: list[str] | None,
    prompts_path: str | None,
    n_images: int | None,
) -> list[str]:
    if prompts is not None:
        result = prompts
    elif prompts_path is not None:
        result = _load_prompts(prompts_path)
        if not result:
            raise ValueError(f"No prompts found in {prompts_path!r}")
    else:
        count = n_images or len(_FALLBACK_PROMPTS)
        result = [_FALLBACK_PROMPTS[i % len(_FALLBACK_PROMPTS)] for i in range(count)]

    if n_images is not None:
        result = result[:n_images]

    if not result:
        raise ValueError("No prompts available — provide prompts, prompts_path, or n_images.")

    return result

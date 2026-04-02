"""SmokeRunner: fast local sanity check for a diffusion model checkpoint.

Generates a small fixed batch of images, validates they are non-degenerate
(not all one colour), and optionally computes CLIP score + aesthetic score.
No reference dataset required — FID is intentionally excluded from smoke tests.

Typical use:
- Local dev: quick check after a training run
- CI: gate on every PR that touches model code
- k8s: smoke-test a newly deployed model pod before routing traffic

Recommended models for local/CI:
  hf-internal-testing/tiny-stable-diffusion-pipe  (~30 MB, random weights)
  stabilityai/sd-turbo                            (~2 GB, 1-step, production quality)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from forge.core.device import get_device
from forge.eval.runner import EvalRunner
from forge.runners.diffusion import DiffusionWrapper
from forge.runners.types import RunnerResult

_log = logging.getLogger(__name__)

_DEFAULT_PROMPTS: list[str] = [
    "a red apple on a white table",
    "a golden retriever running in a park",
    "the eiffel tower at sunset",
    "a bowl of colorful tropical fruits",
]

_DEFAULT_METRICS: list[str] = ["clip_score", "aesthetic_score"]


def _is_valid_image(img: Image.Image) -> bool:
    """Return False if the image is entirely one colour (failed generation)."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    return float(arr.std()) > 1.0


def run_smoke(
    model_id: str,
    output_dir: str,
    prompts: list[str] | None = None,
    n_images: int = 4,
    num_inference_steps: int = 1,
    guidance_scale: float = 0.0,
    metrics: list[str] | None = None,
    device: torch.device | None = None,
    diffusion: DiffusionWrapper | None = None,
) -> RunnerResult:
    """Run a smoke test: generate a tiny batch, validate images, score optionally.

    Args:
        model_id:             HuggingFace model ID or local path.
        output_dir:           Directory to write generated images as PNGs.
        prompts:              Text prompts. Uses built-in defaults when None.
        n_images:             Number of images. Used only when prompts is None
                              (truncates/repeats the default prompt list).
        num_inference_steps:  Denoising steps. Default 1 (Turbo-style models).
        guidance_scale:       CFG scale. Default 0.0 (disabled for Turbo).
        metrics:              Metrics to compute. Subset of {"clip_score",
                              "aesthetic_score"}. FID is not available in smoke.
                              Defaults to both clip_score + aesthetic_score.
                              Pass [] to skip metrics entirely.
        device:               torch.device. Auto-detected if None.
        diffusion:            Pre-built DiffusionWrapper; created internally if None.
                              Inject a mock here in tests.

    Returns:
        RunnerResult with eval_result populated if metrics were requested.

    Raises:
        ValueError: if any generated image is degenerate (all one colour).
        ValueError: if "fid" is requested (not supported in smoke runner).
    """
    _metrics = metrics if metrics is not None else _DEFAULT_METRICS
    if "fid" in _metrics:
        raise ValueError(
            "FID is not supported in the smoke runner — no reference dataset. "
            "Use run_eval with reference_dir instead."
        )

    active_prompts = prompts if prompts is not None else _build_prompts(n_images)

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
    _log.info("SmokeRunner: generating %d images with %s", len(active_prompts), model_id)

    images = diffusion.generate(active_prompts)

    # Validate — fail fast on degenerate output
    bad = [i for i, img in enumerate(images) if not _is_valid_image(img)]
    if bad:
        raise ValueError(
            f"SmokeRunner: {len(bad)} degenerate image(s) (all one colour) at "
            f"indices {bad}. Model may have failed to load correctly."
        )

    # Save to disk
    for i, img in enumerate(images):
        img.save(out_path / f"{i:04d}.png")

    # Metrics
    eval_result = None
    if _metrics:
        eval_runner = EvalRunner(metrics=_metrics, device=device)
        eval_result = eval_runner.run(images, prompts=active_prompts)

    elapsed = time.time() - start
    _log.info("SmokeRunner done in %.1fs — %s", elapsed, eval_result)

    return RunnerResult(
        runner_type="smoke",
        model_id=model_id,
        n_generated=len(images),
        output_dir=output_dir,
        eval_result=eval_result,
        elapsed_sec=elapsed,
        metadata={
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        },
    )


def _build_prompts(n: int) -> list[str]:
    """Return exactly n prompts by cycling through the defaults."""
    base = _DEFAULT_PROMPTS
    return [base[i % len(base)] for i in range(n)]

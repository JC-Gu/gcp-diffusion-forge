"""DiffusionWrapper: thin diffusers AutoPipeline wrapper.

Handles device placement, dtype selection, and lazy model loading.
Keeping this as a plain class (not a forge stage function) makes it
easy to inject a mock in tests without touching subprocess or network.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from forge.core.device import Backend, get_device, get_dtype

_log = logging.getLogger(__name__)


class DiffusionWrapper:
    """Wraps diffusers AutoPipelineForText2Image with forge device conventions.

    The model is loaded lazily on the first call to generate(), so construction
    is always cheap and safe to do at spec-parse time.

    Args:
        model_id:             HuggingFace model ID or local path.
        num_inference_steps:  Default steps per image. 1 works for Turbo-family
                              models; increase for standard schedulers.
        guidance_scale:       CFG scale. 0.0 disables classifier-free guidance
                              (correct for sd-turbo / sdxl-turbo).
        device:               torch.device. Auto-detected if None.
    """

    def __init__(
        self,
        model_id: str,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale

        if device is None:
            backend, device = get_device()
        else:
            _type = device.type
            backend = Backend.CUDA if _type == "cuda" else (
                Backend.MPS if _type == "mps" else Backend.CPU
            )
        self._device = device
        self._dtype = get_dtype(backend)
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        from diffusers import AutoPipelineForText2Image  # lazy — heavy dep

        _log.info(
            "Loading %s → device=%s dtype=%s", self.model_id, self._device, self._dtype
        )
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._pipe.set_progress_bar_config(disable=True)

    def generate(
        self,
        prompts: list[str],
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> list[Image.Image]:
        """Generate one image per prompt.

        Args:
            prompts:              Text prompts; one image is returned per entry.
            num_inference_steps:  Overrides the constructor default if provided.
            guidance_scale:       Overrides the constructor default if provided.

        Returns:
            List of PIL Images in the same order as prompts.
        """
        self._load()
        output = self._pipe(
            prompts,
            num_inference_steps=num_inference_steps or self._num_inference_steps,
            guidance_scale=(
                guidance_scale if guidance_scale is not None else self._guidance_scale
            ),
        )
        return output.images

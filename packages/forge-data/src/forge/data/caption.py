"""Caption stage: VLM recaptioning using Florence-2.

Replaces noisy alt-text captions with dense, model-generated descriptions.
Requires the [caption] extra:  uv add forge-data[caption]

transformers is imported lazily inside _load_florence() so that importing
forge.data does not require transformers to be installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
from PIL import Image

from forge.core.device import get_device
from forge.data.io import iter_webdataset, write_webdataset
from forge.data.types import StageResult, WdsSample

if TYPE_CHECKING:
    # Import only for type checking — not at runtime
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-untyped]

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "microsoft/Florence-2-base"
_DEFAULT_PROMPT = "<DETAILED_CAPTION>"


def _load_florence(
    model_id: str,
    device: torch.device,
) -> tuple["AutoModelForCausalLM", "AutoProcessor"]:
    """Lazy-load Florence-2 model and processor.

    Raises:
        ImportError: with an actionable message if transformers is not installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "run_caption requires transformers>=4.47. "
            "Install with:  uv add forge-data[caption]"
        ) from exc

    _log.info("Loading Florence-2 model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def _iter_recaptioned_samples(
    url_pattern: str,
    model: Any,
    processor: Any,
    device: torch.device,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
) -> Iterator[WdsSample]:
    """Stream samples with 'txt' fields replaced by Florence-2 captions."""
    buffer: list[WdsSample] = []

    def flush() -> Iterator[WdsSample]:
        if not buffer:
            return
        images = [s.get("jpg") or s.get("png") for s in buffer]
        images = [img for img in images if isinstance(img, Image.Image)]

        if not images:
            yield from buffer
            return

        inputs = processor(
            text=[prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        for sample, out_id, img in zip(buffer, output_ids, images):
            raw = processor.decode(out_id, skip_special_tokens=False)
            parsed = processor.post_process_generation(
                raw, task=prompt, image_size=img.size
            )
            # Florence-2 returns dict {task: caption_str}
            caption = parsed.get(prompt, raw)
            if isinstance(caption, str):
                sample["txt"] = caption.strip()
            yield sample

        buffer.clear()

    for sample in iter_webdataset(url_pattern):
        buffer.append(sample)
        if len(buffer) >= batch_size:
            yield from flush()

    yield from flush()


def run_caption(
    input_dir: str,
    output_dir: str,
    model_id: str = _DEFAULT_MODEL,
    batch_size: int = 8,
    prompt: str = _DEFAULT_PROMPT,
    device: torch.device | None = None,
    max_new_tokens: int = 256,
) -> StageResult:
    """Replace captions in WebDataset shards using Florence-2.

    Streams input shards, generates dense captions with Florence-2, and
    writes output shards with updated 'txt' fields.

    Args:
        input_dir:      Directory containing input .tar shards.
        output_dir:     Directory for output shards.
        model_id:       HuggingFace model ID (default: microsoft/Florence-2-base).
        batch_size:     Images per VLM forward pass. Keep small (4–16) to avoid OOM.
        prompt:         Florence-2 task prompt (default: <DETAILED_CAPTION>).
        device:         torch.device. Auto-detected if None.
        max_new_tokens: Maximum tokens to generate per caption.

    Returns:
        StageResult with output_dir and elapsed time.

    Raises:
        ImportError: if transformers is not installed (install forge-data[caption]).
        ValueError:  if no .tar shards are found in input_dir.
    """
    if not list(Path(input_dir).glob("*.tar")):
        raise ValueError(f"No .tar shards found in {input_dir!r}")

    if device is None:
        _, device = get_device()

    model, processor = _load_florence(model_id, device)

    start = time.time()
    url_pattern = str(Path(input_dir) / "*.tar")
    _log.info("Recaptioning %s → %s using %s", input_dir, output_dir, model_id)

    write_webdataset(
        _iter_recaptioned_samples(
            url_pattern, model, processor, device,
            prompt, batch_size, max_new_tokens,
        ),
        output_dir,
    )

    return StageResult(
        stage_name="caption",
        stage_type="caption",
        output_dir=output_dir,
        elapsed_sec=time.time() - start,
    )

"""LoRATrainer: core training loop for SD15 LoRA / DoRA fine-tuning.

Supported architectures:  sd15, sdxl  (SD3 / FLUX = NotImplementedError)
Supported methods:        lora, dora
Supported backends:       CUDA (Tier 1), MPS (Tier 2), CPU (Tier 3 / smoke only)

The training loop follows the standard diffusion LoRA recipe:
  1. Encode images → latents  (frozen VAE, no grad)
  2. Encode captions → embeddings  (frozen text encoder, no grad)
  3. Add noise to latents at a random timestep
  4. Predict the noise with the LoRA-adapted UNet
  5. MSE loss → backward → optimizer step

Only the LoRA adapter parameters receive gradients.
"""

from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from forge.core.config import ModelArchitecture, TrainingJobSpec, TrainingMethod
from forge.core.device import Backend, get_device
from forge.train.checkpoint import save_lora_weights
from forge.train.lora import build_lora_config, inject_lora, resolve_training_precision
from forge.train.optimizer import build_optimizer
from forge.train.types import TrainResult

_log = logging.getLogger(__name__)

# Architectures supported in this release.
_SUPPORTED_ARCHITECTURES = {ModelArchitecture.SD15, ModelArchitecture.SDXL}
_SUPPORTED_METHODS = {TrainingMethod.LORA, TrainingMethod.DORA}

# How often to emit a progress log (steps).
_LOG_EVERY = 50


class LoRATrainer:
    """Fine-tune a diffusion model UNet with LoRA or DoRA adapters.

    Args:
        spec:    Validated TrainingJobSpec from a forge/v1 YAML.
        device:  Override the auto-detected torch.device (useful in tests).
    """

    def __init__(
        self,
        spec: TrainingJobSpec,
        device: torch.device | None = None,
    ) -> None:
        self._spec = spec
        if device is not None:
            self._device = device
            self._backend = _backend_from_device(device)
        else:
            self._backend, self._device = get_device()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> TrainResult:
        """Execute the full training loop and return a TrainResult."""
        spec = self._spec
        backend = self._backend

        accel_precision, model_dtype = resolve_training_precision(
            spec.training.mixed_precision, backend
        )

        from accelerate import Accelerator
        from diffusers import DDPMScheduler
        from diffusers.optimization import get_scheduler

        accelerator = Accelerator(
            mixed_precision=accel_precision,
            gradient_accumulation_steps=spec.training.gradient_accumulation_steps,
        )

        # ── Load model components ──────────────────────────────────────
        _log.info("Loading model components from %s", spec.model.base)
        components = _load_components(spec.model.base, spec.model.architecture, model_dtype)

        unet = components["unet"]
        vae = components["vae"].to(accelerator.device)
        text_encoder = components["text_encoder"].to(accelerator.device)
        tokenizer = components["tokenizer"]

        # ── Inject LoRA ────────────────────────────────────────────────
        lora_cfg = build_lora_config(
            rank=spec.training.lora_rank,
            alpha=spec.training.lora_alpha,
            method=spec.training.method,
            target_modules=spec.training.lora_target_modules,
        )
        unet = inject_lora(unet, lora_cfg)

        if spec.training.gradient_checkpointing and backend != Backend.MPS:
            # gradient_checkpointing on MPS can cause segfaults in some torch versions
            unet.enable_gradient_checkpointing()

        # ── Optimizer ─────────────────────────────────────────────────
        optimizer = build_optimizer(
            unet.parameters(),
            lr=spec.training.learning_rate,
            backend=backend,
            weight_decay=spec.training.optimizer.weight_decay,
            beta1=spec.training.optimizer.beta1,
            beta2=spec.training.optimizer.beta2,
        )

        # ── Data ──────────────────────────────────────────────────────
        dataloader = _build_dataloader(spec, tokenizer)

        # ── LR scheduler ──────────────────────────────────────────────
        lr_scheduler = get_scheduler(
            spec.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=max(1, spec.training.steps // 20),
            num_training_steps=spec.training.steps,
        )

        # ── Accelerate prepare ────────────────────────────────────────
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            spec.model.base, subfolder="scheduler"
        )

        # ── Training loop ─────────────────────────────────────────────
        _log.info(
            "Starting %s training: %d steps, rank=%d, lr=%.2e, backend=%s",
            spec.training.method.value,
            spec.training.steps,
            spec.training.lora_rank,
            spec.training.learning_rate,
            backend.value,
        )

        start = time.time()
        global_step = 0
        last_loss: float | None = None
        checkpoint_every = max(1, spec.training.steps // 5)  # ~5 checkpoints

        for _epoch in itertools.count():
            for batch in dataloader:
                with accelerator.accumulate(unet):
                    # Latent encoding (frozen VAE)
                    with torch.no_grad():
                        latents = _encode_images(batch["pixel_values"], vae)
                        encoder_hidden_states = _encode_text(
                            batch["input_ids"], text_encoder
                        )

                    # Forward diffusion: add noise at random timestep
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                        dtype=torch.long,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Predict noise with LoRA-adapted UNet
                    noise_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                last_loss = loss.detach().item()
                global_step += 1

                if global_step % _LOG_EVERY == 0:
                    _log.info("step=%d loss=%.4f", global_step, last_loss)

                # Intermediate checkpoint
                if global_step % checkpoint_every == 0:
                    _checkpoint(accelerator, unet, spec, global_step)

                if global_step >= spec.training.steps:
                    break

            if global_step >= spec.training.steps:
                break

        # ── Final checkpoint ──────────────────────────────────────────
        final_path = _checkpoint(accelerator, unet, spec, global_step, final=True)

        elapsed = time.time() - start
        _log.info("Training complete in %.1fs — final loss=%.4f", elapsed, last_loss or 0)

        return TrainResult(
            model_id=spec.model.base,
            method=spec.training.method.value,
            steps_completed=global_step,
            checkpoint_path=final_path,
            elapsed_sec=elapsed,
            final_loss=last_loss,
            metadata={
                "lora_rank": spec.training.lora_rank,
                "learning_rate": spec.training.learning_rate,
                "backend": backend.value,
                "architecture": spec.model.architecture.value,
            },
        )


# ── Private helpers ────────────────────────────────────────────────────────────


def _backend_from_device(device: torch.device) -> Backend:
    t = device.type
    return Backend.CUDA if t == "cuda" else (Backend.MPS if t == "mps" else Backend.CPU)


def _load_components(base: str, arch: ModelArchitecture, dtype: torch.dtype) -> dict:
    """Load frozen VAE, text encoder, tokenizer, and trainable UNet."""
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
    vae.requires_grad_(False)

    # UNet starts unfrozen — LoRA injection will freeze base weights
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)

    return {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "unet": unet}


@torch.no_grad()
def _encode_images(pixel_values: torch.Tensor, vae) -> torch.Tensor:
    """VAE encode a batch of images to latent space."""
    latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
    return latents * vae.config.scaling_factor


@torch.no_grad()
def _encode_text(input_ids: torch.Tensor, text_encoder) -> torch.Tensor:
    """CLIP encode a batch of token IDs to embeddings."""
    return text_encoder(input_ids.to(text_encoder.device))[0]


def _build_dataloader(spec: TrainingJobSpec, tokenizer) -> torch.utils.data.DataLoader:
    """Build a DataLoader from WebDataset shards at spec.data.source."""
    import torchvision.transforms.functional as TF
    from forge.data.io import iter_webdataset

    resolution = spec.data.resolution
    center_crop = spec.data.center_crop
    caption_field = spec.data.caption_field
    url_pattern = str(Path(spec.data.source) / "*.tar")

    processed: list[dict] = []
    for sample in iter_webdataset(url_pattern):
        img = sample.get("jpg") or sample.get("png")
        if not isinstance(img, Image.Image):
            continue
        img = img.convert("RGB")

        # Resize + crop
        img = TF.resize(img, resolution, interpolation=TF.InterpolationMode.BILINEAR)
        if center_crop:
            img = TF.center_crop(img, resolution)

        pixel_values = TF.normalize(
            TF.to_tensor(img), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
        caption = _extract_caption(sample, caption_field)
        tokens = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        processed.append({
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        })

    if not processed:
        raise ValueError(f"No valid samples found at {spec.data.source!r}")

    class _ListDataset(torch.utils.data.Dataset):
        def __getitem__(self, i: int) -> dict:
            return processed[i]

        def __len__(self) -> int:
            return len(processed)

    return torch.utils.data.DataLoader(
        _ListDataset(),
        batch_size=spec.training.batch_size,
        shuffle=True,
        drop_last=len(processed) >= spec.training.batch_size,
    )


def _extract_caption(sample: dict, caption_field: str) -> str:
    """Extract caption from a WdsSample. Supports dot-notation (e.g. 'json.caption')."""
    if "." in caption_field:
        field, key = caption_field.split(".", 1)
        value = sample.get(field, {})
        return str(value.get(key, "")) if isinstance(value, dict) else ""
    return str(sample.get(caption_field, ""))


def _checkpoint(
    accelerator,
    unet,
    spec: TrainingJobSpec,
    step: int,
    final: bool = False,
) -> str:
    filename = "lora_weights.safetensors" if final else f"checkpoint-{step:06d}.safetensors"
    unwrapped = accelerator.unwrap_model(unet)
    return save_lora_weights(unwrapped, spec.output.checkpoint_dir, filename)

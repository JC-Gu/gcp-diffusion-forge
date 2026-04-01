"""CLIPScorer: OpenCLIP model wrapper for image and text embedding.

Used by:
  forge-data  — similarity filtering during data curation
  forge-eval  — CLIP score metric on generated images
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip

from forge.core.device import get_device


class CLIPScorer:
    """Wraps an OpenCLIP model for image and text embedding.

    The model is loaded lazily on first use. All public methods are
    decorated with @torch.no_grad() and return CPU tensors.

    Args:
        model_name: OpenCLIP architecture (default: ViT-L-14).
        pretrained: Pretrained weights name (default: openai).
        device: torch.device to run on. Auto-detected from forge-core if None.
        batch_size: Maximum images or texts per forward pass.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: torch.device | None = None,
        batch_size: int = 256,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size

        if device is None:
            _, device = get_device()
        self.device = device

        self._model: open_clip.CLIP | None = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the model. No-op if already loaded."""
        if self._model is not None:
            return
        # create_model_and_transforms returns (model, train_transforms, val_transforms)
        # We always use val_transforms (no augmentation) for scoring.
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the output embedding vectors."""
        self._load()
        assert self._model is not None
        return self._model.visual.output_dim  # type: ignore[return-value]

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed a list of PIL Images into unit-normalized vectors.

        Returns:
            Float tensor of shape [N, D] on CPU, unit-normalized (L2).
        """
        self._load()
        assert self._model is not None and self._preprocess is not None

        results: list[torch.Tensor] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            pixels = torch.stack([self._preprocess(img) for img in batch]).to(self.device)
            emb = self._model.encode_image(pixels)
            emb = F.normalize(emb.float(), dim=-1)
            results.append(emb.cpu())
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Embed a list of strings into unit-normalized vectors.

        Returns:
            Float tensor of shape [N, D] on CPU, unit-normalized (L2).
        """
        self._load()
        assert self._model is not None and self._tokenizer is not None

        results: list[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tokens = self._tokenizer(batch).to(self.device)
            emb = self._model.encode_text(tokens)
            emb = F.normalize(emb.float(), dim=-1)
            results.append(emb.cpu())
        return torch.cat(results, dim=0)

    def score(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        """Cosine similarity between each (image, text) pair.

        Returns:
            Float tensor of shape [N] in [-1, 1].

        Raises:
            ValueError: if images and texts have different lengths.
        """
        if len(images) != len(texts):
            raise ValueError(
                f"images and texts must have the same length, "
                f"got {len(images)} and {len(texts)}"
            )
        img_emb = self.embed_images(images)
        txt_emb = self.embed_texts(texts)
        # Both are unit-normalized, so dot product == cosine similarity
        return (img_emb * txt_emb).sum(dim=-1)

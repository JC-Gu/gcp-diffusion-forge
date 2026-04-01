"""AestheticScorer: LAION improved aesthetic predictor.

Predicts a perceptual aesthetic quality score for images using an MLP
trained on top of CLIP ViT-L/14 embeddings.

Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor

Used by:
  forge-data  — filter training data by aesthetic quality (keep score >= 4.5)
  forge-eval  — measure aesthetic quality of generated images
"""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download

from forge.core.device import get_device
from forge.core.scorers.clip import CLIPScorer

# ── MLP architecture ──────────────────────────────────────────────────────────

_WEIGHTS_REPO = "christophschuhmann/improved-aesthetic-predictor"
_WEIGHTS_FILE = "sac+logos+ava1-l14-linearMSE.pth"


class _AestheticMLP(nn.Module):
    """MLP head trained on CLIP ViT-L/14 embeddings to predict aesthetic scores.

    Input:  768-dim L2-normalized CLIP image embedding
    Output: scalar aesthetic score (roughly 0–10 in practice)

    This class is exported for direct testing of the architecture.
    """

    INPUT_DIM = 768

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ── AestheticScorer ───────────────────────────────────────────────────────────


class AestheticScorer:
    """Score images using the LAION improved aesthetic predictor.

    Weights are downloaded from HuggingFace Hub on first use and cached
    in ~/.cache/huggingface/hub/. Requires CLIP ViT-L/14 (openai) internally —
    this is fixed by the MLP training setup and cannot be changed.

    Args:
        clip_scorer: Optional pre-loaded CLIPScorer to reuse (avoids loading
            CLIP twice when both CLIPScorer and AestheticScorer are used together).
            Must use model_name="ViT-L-14" and pretrained="openai".
        device: torch.device. Auto-detected from forge-core if None.
        batch_size: Maximum images per forward pass.

    Raises:
        ValueError: if clip_scorer uses a model other than ViT-L-14/openai.
    """

    def __init__(
        self,
        clip_scorer: CLIPScorer | None = None,
        device: torch.device | None = None,
        batch_size: int = 256,
    ) -> None:
        if device is None:
            _, device = get_device()
        self.device = device
        self.batch_size = batch_size

        if clip_scorer is not None and (
            clip_scorer.model_name != "ViT-L-14" or clip_scorer.pretrained != "openai"
        ):
            raise ValueError(
                "AestheticScorer requires CLIP ViT-L-14/openai embeddings "
                "(the MLP was trained on these). "
                f"Got {clip_scorer.model_name}/{clip_scorer.pretrained}."
            )
        self._clip = clip_scorer
        self._mlp: _AestheticMLP | None = None

    def _load(self) -> None:
        """Lazy-load CLIP and MLP weights. No-op if already loaded."""
        if self._mlp is not None:
            return

        if self._clip is None:
            self._clip = CLIPScorer(
                model_name="ViT-L-14",
                pretrained="openai",
                device=self.device,
                batch_size=self.batch_size,
            )

        weights_path = hf_hub_download(
            repo_id=_WEIGHTS_REPO,
            filename=_WEIGHTS_FILE,
        )
        mlp = _AestheticMLP()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        mlp.load_state_dict(state)
        mlp.eval()
        mlp.to(self.device)
        self._mlp = mlp

    @torch.no_grad()
    def score(self, images: list[Image.Image]) -> torch.Tensor:
        """Return aesthetic scores for a list of PIL Images.

        Returns:
            Float tensor of shape [N] on CPU. Values are roughly in 0–10,
            where higher is more aesthetically pleasing.
        """
        self._load()
        assert self._clip is not None and self._mlp is not None

        results: list[torch.Tensor] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            emb = self._clip.embed_images(batch).to(self.device)
            scores = self._mlp(emb).squeeze(-1)
            results.append(scores.cpu())
        return torch.cat(results, dim=0)

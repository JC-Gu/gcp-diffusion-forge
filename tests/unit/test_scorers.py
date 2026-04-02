"""Unit tests for forge.core.scorers — no network or model downloads required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from forge.core.scorers.aesthetic import AestheticScorer, _AestheticMLP
from forge.core.scorers.clip import CLIPScorer


# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dummy_images() -> list[Image.Image]:
    rng = np.random.default_rng(42)
    return [
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        for _ in range(3)
    ]


@pytest.fixture
def dummy_texts() -> list[str]:
    return [
        "a red sunset over the ocean",
        "a portrait of a smiling person",
        "a cat sitting on a windowsill",
    ]


def _make_mock_clip_internals(embed_dim: int = 768):
    """Return (mock_model, mock_preprocess, mock_tokenizer) with correct shapes."""
    model = MagicMock()

    def fake_encode_image(pixels: torch.Tensor) -> torch.Tensor:
        n = pixels.shape[0]
        emb = torch.randn(n, embed_dim)
        return emb / emb.norm(dim=-1, keepdim=True)

    def fake_encode_text(tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[0]
        emb = torch.randn(n, embed_dim)
        return emb / emb.norm(dim=-1, keepdim=True)

    model.encode_image.side_effect = fake_encode_image
    model.encode_text.side_effect = fake_encode_text
    model.visual.output_dim = embed_dim
    model.eval.return_value = model

    preprocess = lambda img: torch.zeros(3, 224, 224)  # noqa: E731
    tokenizer = lambda texts: torch.zeros(len(texts), 77, dtype=torch.long)  # noqa: E731

    return model, preprocess, tokenizer


# ── _AestheticMLP ─────────────────────────────────────────────────────────────


def test_aesthetic_mlp_input_dim() -> None:
    assert _AestheticMLP.INPUT_DIM == 768


def test_aesthetic_mlp_output_shape_batch() -> None:
    mlp = _AestheticMLP()
    mlp.eval()
    with torch.no_grad():
        out = mlp(torch.randn(4, 768))
    assert out.shape == (4, 1)


def test_aesthetic_mlp_output_shape_single() -> None:
    mlp = _AestheticMLP()
    mlp.eval()
    with torch.no_grad():
        out = mlp(torch.randn(1, 768))
    assert out.shape == (1, 1)


def test_aesthetic_mlp_is_deterministic_in_eval() -> None:
    mlp = _AestheticMLP()
    mlp.eval()
    x = torch.randn(2, 768)
    with torch.no_grad():
        out1 = mlp(x)
        out2 = mlp(x)
    assert torch.allclose(out1, out2)


# ── CLIPScorer ────────────────────────────────────────────────────────────────


@pytest.fixture
def clip_scorer_mocked() -> CLIPScorer:
    mock_model, mock_pre, mock_tok = _make_mock_clip_internals(embed_dim=768)
    with (
        patch("open_clip.create_model_and_transforms") as mock_create,
        patch("open_clip.get_tokenizer") as mock_get_tok,
    ):
        mock_create.return_value = (mock_model, MagicMock(), mock_pre)
        mock_get_tok.return_value = mock_tok
        scorer = CLIPScorer(model_name="ViT-L-14", pretrained="openai")
        scorer._load()
        yield scorer


def test_clip_embed_images_shape(
    clip_scorer_mocked: CLIPScorer, dummy_images: list[Image.Image]
) -> None:
    emb = clip_scorer_mocked.embed_images(dummy_images)
    assert emb.shape == (len(dummy_images), 768)


def test_clip_embed_images_unit_normalized(
    clip_scorer_mocked: CLIPScorer, dummy_images: list[Image.Image]
) -> None:
    emb = clip_scorer_mocked.embed_images(dummy_images)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(len(dummy_images)), atol=1e-5)


def test_clip_embed_images_returns_cpu_tensor(
    clip_scorer_mocked: CLIPScorer, dummy_images: list[Image.Image]
) -> None:
    emb = clip_scorer_mocked.embed_images(dummy_images)
    assert emb.device.type == "cpu"


def test_clip_embed_texts_shape(
    clip_scorer_mocked: CLIPScorer, dummy_texts: list[str]
) -> None:
    emb = clip_scorer_mocked.embed_texts(dummy_texts)
    assert emb.shape == (len(dummy_texts), 768)


def test_clip_embed_texts_unit_normalized(
    clip_scorer_mocked: CLIPScorer, dummy_texts: list[str]
) -> None:
    emb = clip_scorer_mocked.embed_texts(dummy_texts)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(len(dummy_texts)), atol=1e-5)


def test_clip_score_shape(
    clip_scorer_mocked: CLIPScorer,
    dummy_images: list[Image.Image],
    dummy_texts: list[str],
) -> None:
    scores = clip_scorer_mocked.score(dummy_images, dummy_texts)
    assert scores.shape == (len(dummy_images),)


def test_clip_score_in_valid_range(
    clip_scorer_mocked: CLIPScorer,
    dummy_images: list[Image.Image],
    dummy_texts: list[str],
) -> None:
    scores = clip_scorer_mocked.score(dummy_images, dummy_texts)
    assert (scores >= -1.0).all() and (scores <= 1.0).all()


def test_clip_score_mismatched_lengths_raises(
    clip_scorer_mocked: CLIPScorer, dummy_images: list[Image.Image]
) -> None:
    with pytest.raises(ValueError, match="same length"):
        clip_scorer_mocked.score(dummy_images, ["only one text"])


def test_clip_embedding_dim_property(clip_scorer_mocked: CLIPScorer) -> None:
    assert clip_scorer_mocked.embedding_dim == 768


# ── AestheticScorer ───────────────────────────────────────────────────────────


@pytest.fixture
def aesthetic_scorer_mocked(
    tmp_path: "pytest.TempPathFactory", dummy_images: list[Image.Image]
) -> AestheticScorer:
    """AestheticScorer with mocked CLIP and random-initialized MLP weights."""
    weights_path = tmp_path / "aesthetic.pth"
    torch.save(_AestheticMLP().state_dict(), weights_path)

    mock_model, mock_pre, mock_tok = _make_mock_clip_internals(embed_dim=768)
    with (
        patch("open_clip.create_model_and_transforms") as mock_create,
        patch("open_clip.get_tokenizer") as mock_get_tok,
        patch("forge.core.scorers.aesthetic.hf_hub_download") as mock_dl,
    ):
        mock_create.return_value = (mock_model, MagicMock(), mock_pre)
        mock_get_tok.return_value = mock_tok
        mock_dl.return_value = str(weights_path)

        scorer = AestheticScorer()
        scorer._load()
        yield scorer


def test_aesthetic_score_shape(
    aesthetic_scorer_mocked: AestheticScorer, dummy_images: list[Image.Image]
) -> None:
    scores = aesthetic_scorer_mocked.score(dummy_images)
    assert scores.shape == (len(dummy_images),)


def test_aesthetic_score_is_float32(
    aesthetic_scorer_mocked: AestheticScorer, dummy_images: list[Image.Image]
) -> None:
    scores = aesthetic_scorer_mocked.score(dummy_images)
    assert scores.dtype == torch.float32


def test_aesthetic_score_returns_cpu_tensor(
    aesthetic_scorer_mocked: AestheticScorer, dummy_images: list[Image.Image]
) -> None:
    scores = aesthetic_scorer_mocked.score(dummy_images)
    assert scores.device.type == "cpu"


def test_aesthetic_scorer_rejects_wrong_clip_model() -> None:
    wrong_clip = CLIPScorer(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
    with pytest.raises(ValueError, match="ViT-L-14"):
        AestheticScorer(clip_scorer=wrong_clip)


def test_aesthetic_scorer_accepts_correct_clip_model() -> None:
    correct_clip = CLIPScorer(model_name="ViT-L-14", pretrained="openai")
    # Should not raise
    AestheticScorer(clip_scorer=correct_clip)


def test_hf_download_called_with_correct_repo(
    tmp_path: "pytest.TempPathFactory",
) -> None:
    weights_path = tmp_path / "aesthetic.pth"
    torch.save(_AestheticMLP().state_dict(), weights_path)

    mock_model, mock_pre, mock_tok = _make_mock_clip_internals()
    with (
        patch("open_clip.create_model_and_transforms") as mock_create,
        patch("open_clip.get_tokenizer") as mock_get_tok,
        patch("forge.core.scorers.aesthetic.hf_hub_download") as mock_dl,
    ):
        mock_create.return_value = (mock_model, MagicMock(), mock_pre)
        mock_get_tok.return_value = mock_tok
        mock_dl.return_value = str(weights_path)

        scorer = AestheticScorer()
        scorer._load()

        mock_dl.assert_called_once_with(
            repo_id="camenduru/improved-aesthetic-predictor",
            filename="sac+logos+ava1-l14-linearMSE.pth",
        )

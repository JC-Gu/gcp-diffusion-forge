"""Shared model wrappers used by both forge-data (filtering) and forge-eval (metrics)."""

from forge.core.scorers.aesthetic import AestheticScorer
from forge.core.scorers.clip import CLIPScorer

__all__ = ["CLIPScorer", "AestheticScorer"]

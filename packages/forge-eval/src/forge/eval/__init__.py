"""forge-eval: Evaluation metrics for diffusion model outputs.

Public API::

    from forge.eval import EvalResult, EvalRunner
    from forge.eval import compute_clip_score, compute_aesthetic_score, compute_fid
"""

from forge.eval.metrics import compute_aesthetic_score, compute_clip_score, compute_fid
from forge.eval.result import EvalResult
from forge.eval.runner import EvalRunner

__all__ = [
    "compute_clip_score",
    "compute_aesthetic_score",
    "compute_fid",
    "EvalResult",
    "EvalRunner",
]

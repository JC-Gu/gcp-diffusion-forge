"""forge-runners: Diffusion model inference runners for local and Kubernetes use.

Runners:
  smoke — fast sanity check (no reference data needed, <30s with sd-turbo)
  eval  — full quality evaluation (CLIP score, aesthetic score, FID)

Public API::

    from forge.runners import RunnerPipeline, RunnerResult, RunnerJobSpec
    from forge.runners import run_smoke, run_eval
    from forge.runners import DiffusionWrapper

CLI::

    forge-run workflows/smoke.yaml
    forge-run workflows/eval.yaml --dry-run
"""

from forge.runners.diffusion import DiffusionWrapper
from forge.runners.eval import run_eval
from forge.runners.pipeline import RunnerPipeline
from forge.runners.smoke import run_smoke
from forge.runners.types import RunnerJobSpec, RunnerResult

__all__ = [
    "DiffusionWrapper",
    "RunnerJobSpec",
    "RunnerPipeline",
    "RunnerResult",
    "run_eval",
    "run_smoke",
]

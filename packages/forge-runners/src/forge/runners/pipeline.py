"""RunnerPipeline: dispatch a RunnerJobSpec to the correct runner function.

Follows the same YAML → normalise params → call function pattern as
DataPipeline in forge-data.  The CLI entrypoint `forge-run` lives here.

k8s usage:
    # Container CMD:
    forge-run /config/spec.yaml
    # Mount spec as a Kubernetes ConfigMap at /config/spec.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import yaml

from forge.runners.eval import run_eval
from forge.runners.smoke import run_smoke
from forge.runners.types import RunnerJobSpec, RunnerResult

# Maps YAML runner names → function names in this module's globals().
# String names (not direct references) so unittest.mock.patch() works.
_RUNNER_REGISTRY: dict[str, str] = {
    "smoke": "run_smoke",
    "eval":  "run_eval",
}

_REQUIRED_PARAMS: dict[str, set[str]] = {
    "smoke": {"model_id", "output_dir"},
    "eval":  {"model_id", "output_dir"},
}


# ── Param normalisation (camelCase YAML → snake_case Python) ──────────────────


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    return {_camel_to_snake(k): v for k, v in params.items()}


# ── RunnerPipeline ─────────────────────────────────────────────────────────────


class RunnerPipeline:
    """Validates a RunnerJobSpec and dispatches to the appropriate runner.

    Args:
        spec:   Validated RunnerJobSpec (from forge.runners.types).
        device: torch.device. Auto-detected by each runner if None.
    """

    def __init__(self, spec: RunnerJobSpec, device=None) -> None:
        self._spec = spec
        self._device = device

    def _validate(self) -> None:
        if self._spec.runner not in _RUNNER_REGISTRY:
            raise ValueError(
                f"Unknown runner {self._spec.runner!r}. "
                f"Known runners: {sorted(_RUNNER_REGISTRY)}"
            )
        params = _normalize_params(self._spec.params)
        missing = _REQUIRED_PARAMS[self._spec.runner] - params.keys()
        if missing:
            raise ValueError(
                f"Runner {self._spec.runner!r} is missing required params: "
                f"{sorted(missing)}"
            )

    def run(self, dry_run: bool = False) -> RunnerResult | None:
        """Execute the runner.

        Args:
            dry_run: If True, validate the spec and return None without
                     loading any model or touching the filesystem.

        Returns:
            RunnerResult on success; None on dry_run.

        Raises:
            ValueError: on unknown runner or missing required params.
        """
        self._validate()
        if dry_run:
            return None

        fn_name = _RUNNER_REGISTRY[self._spec.runner]
        fn = globals()[fn_name]
        kwargs = _normalize_params(self._spec.params)
        if self._device is not None:
            kwargs.setdefault("device", self._device)
        return fn(**kwargs)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: D401 — entry point
    """forge-run CLI: execute a RunnerJob YAML spec."""
    parser = argparse.ArgumentParser(
        prog="forge-run",
        description="Execute a forge RunnerJob YAML spec locally or inside a k8s container.",
    )
    parser.add_argument("spec", help="Path to RunnerJob YAML file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec without generating images",
    )
    args = parser.parse_args()

    with open(args.spec) as fh:
        raw = yaml.safe_load(fh)

    spec = RunnerJobSpec.model_validate(raw)
    result = RunnerPipeline(spec).run(dry_run=args.dry_run)

    if result is None:
        print("dry-run: spec is valid", flush=True)
    else:
        print(json.dumps(result.to_dict(), indent=2), flush=True)

    sys.exit(0)

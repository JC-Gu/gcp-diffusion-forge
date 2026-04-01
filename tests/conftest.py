"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest

from forge.core.device import Backend, get_device


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow integration tests that download models from HuggingFace.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="pass --run-slow to run integration tests")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def current_backend() -> Backend:
    backend, _ = get_device()
    return backend


@pytest.fixture(autouse=True)
def skip_by_backend(request: pytest.FixtureRequest, current_backend: Backend) -> None:
    """Auto-skip tests marked with @pytest.mark.cuda or @pytest.mark.mps
    if the required hardware is not available."""
    if request.node.get_closest_marker("cuda") and current_backend != Backend.CUDA:
        pytest.skip("requires NVIDIA GPU (CUDA)")
    if request.node.get_closest_marker("mps") and current_backend != Backend.MPS:
        pytest.skip("requires Apple Silicon (MPS)")

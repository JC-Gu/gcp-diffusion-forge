"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest

from forge.core.device import Backend, get_device


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

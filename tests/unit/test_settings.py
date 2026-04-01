"""Unit tests for forge.core.settings."""

from __future__ import annotations

from forge.core.settings import ForgeSettings


def test_default_log_level() -> None:
    settings = ForgeSettings()
    assert settings.log_level == "INFO"


def test_default_wandb_project() -> None:
    settings = ForgeSettings()
    assert settings.wandb_project == "gcp-diffusion-forge"


def test_storage_emulator_none_by_default() -> None:
    settings = ForgeSettings()
    assert settings.storage_emulator_host is None


def test_env_override_log_level(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("FORGE_LOG_LEVEL", "DEBUG")
    settings = ForgeSettings()
    assert settings.log_level == "DEBUG"


def test_env_override_storage_emulator(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("FORGE_STORAGE_EMULATOR_HOST", "http://localhost:4443")
    settings = ForgeSettings()
    assert settings.storage_emulator_host == "http://localhost:4443"


def test_env_override_wandb_disabled(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("FORGE_WANDB_ENABLED", "false")
    settings = ForgeSettings()
    assert settings.wandb_enabled is False


def test_env_override_gcp_project(monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("FORGE_GCP_PROJECT_ID", "my-gcp-project")
    settings = ForgeSettings()
    assert settings.gcp_project_id == "my-gcp-project"

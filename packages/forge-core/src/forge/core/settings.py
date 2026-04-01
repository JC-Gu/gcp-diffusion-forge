"""Pydantic Settings for gcp-diffusion-forge.

All configuration is read from environment variables with the FORGE_ prefix,
or from a .env file in the working directory.

Local dev:  copy .env.example to .env and set FORGE_STORAGE_EMULATOR_HOST
Production: set FORGE_* vars via Kubernetes Secrets / GKE Workload Identity
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GCP / Storage
    gcp_project_id: str = "forge-local"
    gcs_bucket_data: str = "forge-data"
    gcs_bucket_checkpoints: str = "forge-checkpoints"
    gcs_bucket_artifacts: str = "forge-artifacts"
    # Set to http://localhost:4443 when using fake-gcs-server locally
    storage_emulator_host: str | None = None

    # Container registry (Artifact Registry or GCR)
    artifact_registry: str = "gcr.io"

    # Observability
    log_level: str = "INFO"
    wandb_project: str = "gcp-diffusion-forge"
    wandb_enabled: bool = True

    # Workflow orchestration
    prefect_api_url: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> ForgeSettings:
    """Return a cached ForgeSettings instance.

    Use this in application code; create ForgeSettings() directly in tests
    so monkeypatching env vars takes effect.
    """
    return ForgeSettings()

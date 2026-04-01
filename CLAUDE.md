# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`gcp-diffusion-forge` is a Python monorepo library providing an end-to-end platform for diffusion model workflows: data curation, training, evaluation, and serving. It targets three compute backends (Apple Silicon MPS, CPU, NVIDIA CUDA) and deploys to Google Kubernetes Engine (GKE) via Terraform.

## Repository Structure

```
gcp-diffusion-forge/
├── pyproject.toml              # Root workspace config (uv workspaces)
├── uv.lock                     # Committed lockfile
├── packages/
│   ├── forge-core/             # Shared: device abstraction, config models, settings
│   │   └── src/forge/core/
│   ├── forge-data/             # Data curation: download, embed, filter, caption
│   │   └── src/forge/data/
│   ├── forge-train/            # Training: accelerate wrappers, LoRA, full finetune
│   │   └── src/forge/train/
│   ├── forge-eval/             # Evaluation: FID, CLIP score, aesthetic scoring
│   │   └── src/forge/eval/
│   └── forge-serve/            # Serving: BentoML-based inference stack
│       └── src/forge/serve/
├── infra/
│   ├── terraform/              # GCP/GKE infrastructure (modules: cluster, network, storage)
│   └── k8s/                    # Helm charts, Kustomize overlays, Kubeflow PyTorchJob manifests
├── workflows/                  # YAML job specs (TrainingJob, DataJob, EvalJob, ServeJob)
├── docker/                     # Dockerfiles per component (train, serve, data)
└── tests/
    ├── unit/                   # No GPU required
    └── integration/            # Smoke tests using MPS or CPU
```

All packages use `src/` layout and the `forge.*` implicit namespace (no `__init__.py` at `src/forge/` level).

## Development Setup

```bash
# Install uv (package manager — do NOT use pip/poetry)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all packages in editable mode
uv sync --all-packages

# Install with CUDA extras (for NVIDIA GPU dev)
uv sync --all-packages --extra cuda

# Run tests
uv run pytest tests/unit/                        # unit tests (no GPU)
uv run pytest tests/integration/ -m mps          # MPS smoke tests
uv run pytest tests/ -k "test_name"              # single test

# Lint and format
uv run ruff check .
uv run ruff format .
uv run mypy packages/

# Local services (GCS emulator + Prefect server)
docker compose up -d
```

## Key Commands

```bash
# Run a workflow job locally
uv run forge run workflows/train_flux_lora.yaml --backend mps

# Run data curation pipeline
uv run forge run workflows/data_pipeline.yaml --dry-run

# Build Docker images
docker build -f docker/Dockerfile.train --build-arg PLATFORM=cuda -t forge-train:latest .

# Terraform: provision GKE cluster
cd infra/terraform && terraform init && terraform plan -var-file=envs/dev.tfvars

# Deploy to GKE via skaffold
skaffold run --profile gke-dev

# Submit training job to GKE
kubectl apply -f infra/k8s/training/pytorch-job.yaml
```

## Architecture: Key Design Decisions

### Multi-Platform Backend (`forge-core`)

All platform-specific logic is centralized in `forge.core.device`. Every other package consumes this abstraction — never call `torch.cuda.is_available()` directly outside of `forge-core`.

```python
from forge.core.device import get_device, get_dtype, get_attn_backend, is_distributed
```

**Platform constraints to respect:**
- **MPS (Apple Silicon)**: No `torch.distributed`/NCCL, no `flash-attn`, no `xformers`, `bfloat16` is unreliable (use `float16`), `torch.compile` disabled by default
- **CUDA (Ampere+)**: Use `bfloat16` + `flash-attn`; enable `torch.compile`
- **CUDA (pre-Ampere, T4)**: Use `xformers` instead of `flash-attn`
- **CPU**: `float32` only, no distributed

### Workflow Definition

Jobs are defined as **YAML specs** (`apiVersion: forge/v1`) and orchestrated by **Prefect v3** flows. The YAML captures declarative job parameters (model, data, resources); Python handles orchestration logic (retry, monitoring, branching).

```yaml
apiVersion: forge/v1
kind: TrainingJob   # or DataJob | EvalJob | ServeJob
metadata:
  name: my-job
spec:
  model: { architecture: flux, base: black-forest-labs/FLUX.1-dev }
  training: { method: lora, lora_rank: 32, steps: 2000 }
  resources: { gpu_type: a100-80gb, gpu_count: 4, spot: true }
```

Pydantic models in `forge.core.config` validate all workflow specs. Prefect flows in each sub-package (`forge.train.flows`, `forge.data.flows`) consume these specs and dispatch Kubeflow `PyTorchJob` CRDs to GKE.

### Training Stack

- **Accelerate + FSDP2** for single/multi-GPU — one codebase handles MPS, single CUDA, and multi-node
- **PEFT (LoRA/DoRA)** for parameter-efficient fine-tuning of FLUX, SDXL, SD3
- Full FLUX fine-tuning requires 8×A100-80GB or 8×H100; LoRA works on single A100-40GB
- Checkpoints always use **safetensors** format
- Optimizer: `adamw8bit` (bitsandbytes) for CUDA, standard `adamw` for MPS/CPU

### Data Pipeline

Stages: `img2dataset` download → OpenCLIP embedding → quality/CLIP/aesthetic filtering → VLM recaptioning → WebDataset `.tar` shards on GCS.

Data always streams via **WebDataset** from GCS (never copied to node disk). GCS FUSE CSI driver mounts buckets directly into training pods.

### GKE Infrastructure

- **Node pools**: system (e2-standard-8), training-a100 (a2-highgpu-8g, spot), training-h100 (a3-highgpu-8g, spot), inference-l4 (g2-standard-8, on-demand)
- Training pools **scale to zero** when idle
- **Workload Identity** for GCS access — no service account key files
- **Kubeflow Training Operator v2** for `PyTorchJob` / multi-node training
- **Kueue** for queue-based resource management across job types
- Private cluster with Cloud NAT for outbound access

### Serving

**BentoML** is the primary serving framework. `forge-serve` wraps model loading and exposes an OpenAI-compatible image generation API. BentoML containers deploy as GKE Deployments on the L4 inference node pool.

### Local Dev vs Production

| Context | Config |
|---|---|
| Mac local | `FORGE_STORAGE_EMULATOR_HOST=http://localhost:4443`, MPS backend |
| GKE dev | Skaffold sync, T4 node pool, dev namespace |
| GKE prod | Terraform-managed, A100/H100/L4 pools, prod namespace |

Set `FORGE_*` env vars (see `forge.core.settings` — Pydantic Settings with `env_prefix="FORGE_"`). Never hardcode bucket names or project IDs.

## Dependencies and Versions

| Area | Key Packages |
|---|---|
| Models | `diffusers>=0.32`, `transformers>=4.47`, `peft>=0.14`, `safetensors>=0.4.5` |
| Training | `torch>=2.5,<2.8`, `accelerate>=1.2`, `bitsandbytes>=0.44` |
| CUDA extras | `xformers>=0.0.28`, `flash-attn>=2.7`, `deepspeed>=0.16` |
| Data | `img2dataset>=1.45`, `open_clip_torch>=2.28`, `webdataset>=0.2.86` |
| Eval | `clean-fid>=0.1.35`, `torchmetrics>=1.6` |
| Serving | `bentoml>=1.3` |
| Orchestration | `prefect>=3.0`, `prefect-kubernetes` |
| Config | `pydantic>=2.10`, `pydantic-settings>=2.7` |
| Infra | Terraform `>=1.9`, `google` provider `>=6.0`, `terraform-google-kubernetes-engine>=35.0` |

Python target: `>=3.11,<3.13`.

## Testing Strategy

- **Unit tests** (`tests/unit/`): no GPU, test config parsing, device abstraction logic, YAML schema validation
- **Integration/smoke tests** (`tests/integration/`): single forward pass with 1-2 steps; run on MPS locally or CPU in CI
- **System tests**: full training run on GKE T4 node (CI gate before merging to main)
- Use `pytest.mark.cuda`, `pytest.mark.mps` markers to skip hardware-specific tests appropriately

## Infrastructure Conventions

- All Terraform state stored in GCS backend (`gs://forge-tf-state/`)
- Module structure: `infra/terraform/modules/{cluster,network,storage,iam}` + `infra/terraform/envs/{dev,staging,prod}`
- GPU node pools always have `nvidia.com/gpu: present` taint; training pods must have the matching toleration
- Spot VM interruption handling: training jobs checkpoint every N steps; Kubeflow restarts from last checkpoint

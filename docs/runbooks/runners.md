# Runners Runbook

Operational guide for `forge-runners` — how to validate a diffusion model checkpoint
locally on a Mac dev machine, and how the same spec runs on Kubernetes.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [First-time setup](#2-first-time-setup)
3. [Smoke test — fast sanity check](#3-smoke-test--fast-sanity-check)
4. [Eval runner — quality metrics](#4-eval-runner--quality-metrics)
5. [Reading the output](#5-reading-the-output)
6. [Mac-specific notes](#6-mac-specific-notes)
7. [Troubleshooting](#7-troubleshooting)
8. [Running on Kubernetes](#8-running-on-kubernetes)

---

## 1. Prerequisites

| Requirement | Version | Check |
|---|---|---|
| macOS | 13+ (Ventura) for stable MPS | `sw_vers -productVersion` |
| Python | 3.11 | `uv run python3 --version` |
| uv | 0.5+ | `uv --version` |
| Disk space | 3 GB free (sd-turbo weights) | `df -h ~` |

`uv` is the only tool you need to install manually. Everything else is managed by the workspace.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. First-time setup

```bash
# Clone the repo
git clone https://github.com/JC-Gu/gcp-diffusion-forge.git
cd gcp-diffusion-forge

# Install all workspace packages and their dependencies
uv sync --all-packages
```

That's it. `uv sync` creates `.venv/` at the repo root, installs all five
`forge-*` packages in editable mode, and resolves every dependency.

Verify the CLI is available:

```bash
uv run forge-run --help
```

Expected output:
```
usage: forge-run [-h] [--dry-run] spec
Execute a forge RunnerJob YAML spec locally or inside a k8s container.
...
```

---

## 3. Smoke test — fast sanity check

The smoke runner generates a small batch of images, checks they are non-degenerate
(not all one colour), and computes CLIP score + aesthetic score.
No reference dataset required.

### 3a. Instant CI smoke (30 MB, random weights)

Use `hf-internal-testing/tiny-stable-diffusion-pipe` when you want to validate
the pipeline wiring without downloading real weights. Output will be noise —
metrics are meaningless but the pipeline will complete in under 10 seconds.

```bash
# Validate the spec first (no model loaded, no files written)
uv run forge-run workflows/smoke_local.yaml --dry-run

# Run with the tiny CI model
cat > /tmp/smoke-ci.yaml << 'EOF'
apiVersion: forge/v1
kind: RunnerJob
metadata:
  name: smoke-ci
runner: smoke
params:
  modelId: hf-internal-testing/tiny-stable-diffusion-pipe
  outputDir: /tmp/forge/smoke-ci
  nImages: 4
  numInferenceSteps: 2
  guidanceScale: 7.5
  metrics:
    - clip_score
    - aesthetic_score
EOF

uv run forge-run /tmp/smoke-ci.yaml
```

**Expected duration**: ~8–15 s on Apple Silicon (MPS), ~30–60 s on CPU.

### 3b. Production smoke (sd-turbo, ~2 GB)

`stabilityai/sd-turbo` is a 1-step distilled model that produces real images.
The weights are downloaded once to `~/.cache/huggingface/` on first run.

```bash
uv run forge-run workflows/smoke_local.yaml
```

**Expected duration**: ~30 s on M-series (MPS) after weights are cached.

Generated images are written to `/tmp/forge/smoke-out/`.

---

## 4. Eval runner — quality metrics

The eval runner generates a larger batch, saves every image, and runs the full
`forge-eval` suite. CLIP score and aesthetic score run on every call; FID is
added automatically when you supply `referenceDir`.

### 4a. Quick local eval (20 images, no FID)

```bash
uv run forge-run workflows/eval_local.yaml
```

This uses the built-in fallback prompts and writes 20 images to `/tmp/forge/eval-out/`.

### 4b. Eval with your own prompts

```bash
# One prompt per line
cat > /tmp/my-prompts.txt << 'EOF'
a red apple on a white table
a golden retriever running in a park
the eiffel tower at sunset
a waterfall in a tropical forest
EOF

cat > /tmp/eval-custom.yaml << 'EOF'
apiVersion: forge/v1
kind: RunnerJob
metadata:
  name: eval-custom
runner: eval
params:
  modelId: stabilityai/sd-turbo
  outputDir: /tmp/forge/eval-custom
  promptsPath: /tmp/my-prompts.txt
  batchSize: 4
  numInferenceSteps: 1
  guidanceScale: 0.0
  metrics:
    - clip_score
    - aesthetic_score
EOF

uv run forge-run /tmp/eval-custom.yaml
```

### 4c. Eval with FID

FID requires a reference directory of real images (PNG or JPEG).
Minimum 2048 images recommended; a warning is printed below that threshold.

```bash
cat > /tmp/eval-fid.yaml << 'EOF'
apiVersion: forge/v1
kind: RunnerJob
metadata:
  name: eval-fid
runner: eval
params:
  modelId: stabilityai/sd-turbo
  outputDir: /tmp/forge/eval-fid
  promptsPath: /tmp/my-prompts.txt
  referenceDir: /path/to/your/reference/images/
  nImages: 100
  batchSize: 4
  numInferenceSteps: 1
  guidanceScale: 0.0
EOF

uv run forge-run /tmp/eval-fid.yaml
```

---

## 5. Reading the output

Both runners print a JSON result to stdout on completion.

```json
{
  "runner_type": "smoke",
  "model_id": "stabilityai/sd-turbo",
  "n_generated": 4,
  "output_dir": "/tmp/forge/smoke-out",
  "elapsed_sec": 28.4,
  "eval": {
    "clip_score": 0.312,
    "aesthetic_score": 5.87,
    "n_generated": 4
  },
  "metadata": {
    "num_inference_steps": 1,
    "guidance_scale": 0.0
  }
}
```

| Field | Interpretation |
|---|---|
| `clip_score` | Mean image–text cosine similarity, range ~0.1–0.4. >0.28 is typical for a well-aligned model. |
| `aesthetic_score` | LAION aesthetic predictor, range 0–10. >5.0 is considered good quality. |
| `fid` | Fréchet Inception Distance (lower = better). <20 is good for photo-realistic models. |
| `elapsed_sec` | Wall-clock time including model load on first run. |

Generated images live in `outputDir` as `0000.png`, `0001.png`, … and can be
previewed directly with macOS Quick Look (`spacebar` in Finder).

To save the result to a file:

```bash
uv run forge-run workflows/smoke_local.yaml | tee /tmp/forge/smoke-result.json
```

---

## 6. Mac-specific notes

### Device selection

`forge-runners` calls `forge.core.device.get_device()` which auto-selects:

| Hardware | Backend | dtype |
|---|---|---|
| Apple Silicon (M1/M2/M3/M4) | MPS | `float16` |
| Intel Mac | CPU | `float32` |
| External GPU (rare) | CUDA | `bfloat16` |

No configuration required — the right backend is detected automatically.

### Memory limits on MPS

MPS uses unified memory shared with the OS. If you see `MPSNDArrayIdentity` or
out-of-memory errors, reduce `batchSize` in your spec:

```yaml
params:
  batchSize: 1   # safest for 8 GB unified memory
  batchSize: 4   # comfortable for 16 GB
  batchSize: 8   # fine for 32 GB+
```

### Model cache location

Downloaded weights live in `~/.cache/huggingface/hub/`. To pre-warm the cache
before a demo or offline session:

```bash
uv run python3 -c "
from diffusers import AutoPipelineForText2Image
import torch
AutoPipelineForText2Image.from_pretrained('stabilityai/sd-turbo', torch_dtype=torch.float16)
print('Cache warm.')
"
```

### First-run latency

On the very first call `DiffusionWrapper._load()` downloads model weights,
compiles MPS kernels, and warms up the pipeline. Expect 60–120 s. Subsequent
runs with cached weights take 5–30 s depending on model size.

---

## 7. Troubleshooting

**`ValueError: No .tar shards found`** (from a data stage embedded in a runner)
→ The `inputDir` does not exist or contains no `*.tar` files. Check the path.

**`ValueError: degenerate image(s) (all one colour)`**
→ The model produced blank output. Common causes:
- Wrong `guidanceScale` for the model family (turbo models use `0.0`; standard SD uses `7.5`)
- `numInferenceSteps: 1` with a non-distilled model — increase to `20`+
- MPS memory pressure — reduce `batchSize` to `1` and retry

**`torch.mps.MPSNDArrayIdentity` / MPS out of memory**
→ Set `batchSize: 1` in your spec. If it still fails, restart the Python process
to release MPS memory, then rerun.

**`ModuleNotFoundError: No module named 'diffusers'`**
→ Run `uv sync --all-packages` to ensure `forge-runners` dependencies are installed.

**`OSError: We couldn't connect to 'https://huggingface.co'`**
→ You are offline. Either connect to the internet for the first run, or set
`HF_HUB_OFFLINE=1` after the weights are cached:
```bash
HF_HUB_OFFLINE=1 uv run forge-run workflows/smoke_local.yaml
```

**Slow metric computation (CLIP score taking >60 s)**
→ The CLIP ViT-L-14 model (~1.7 GB) is being downloaded for the first time.
Subsequent runs use the HuggingFace cache and take a few seconds.

---

## 8. Running on Kubernetes

The runner is intentionally cloud-agnostic. The same `forge-run` binary that
works locally runs inside a container on GKE or EKS — the only difference is
how the spec and data paths are supplied.

### High-level flow

```
1. Build a container image that has forge-runners installed
2. Store your RunnerJob YAML as a Kubernetes ConfigMap
3. Create a Kubernetes Job that runs:
       forge-run /config/spec.yaml
4. Mount data (prompts, reference images) and output via PVCs or GCS Fuse
5. Collect the JSON result from stdout (e.g. via a log aggregator)
```

### Example Job manifest (GKE)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: eval-sd-turbo
spec:
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-nodepool: gpu-inference   # L4 node pool
      containers:
        - name: runner
          image: gcr.io/YOUR_PROJECT/forge-runners:latest
          command: ["forge-run", "/config/spec.yaml"]
          volumeMounts:
            - name: spec
              mountPath: /config
            - name: data
              mountPath: /data
            - name: results
              mountPath: /results
          resources:
            limits:
              nvidia.com/gpu: "1"
      volumes:
        - name: spec
          configMap:
            name: eval-spec
        - name: data
          persistentVolumeClaim:
            claimName: gcs-data-pvc    # GCS Fuse-backed PVC
        - name: results
          persistentVolumeClaim:
            claimName: gcs-results-pvc
      restartPolicy: Never
```

The `workflows/eval_k8s.yaml` spec in this repo shows the corresponding
RunnerJob YAML with `/data/` and `/results/` paths that map to the mounts above.

> Full GKE infrastructure (node pools, GCS PVCs, Workload Identity) will be
> defined in `forge-infra/` (Terraform). This section will be updated once that
> module is available.

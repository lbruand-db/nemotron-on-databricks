# Nemotron on Databricks

[![Databricks AI Runtime](https://img.shields.io/badge/Databricks-AI%20Runtime-FF3621?logo=databricks&logoColor=white)](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
[![Model Serving](https://img.shields.io/badge/Databricks-Model%20Serving-FF3621?logo=databricks&logoColor=white)](https://docs.databricks.com/aws/en/machine-learning/model-serving/)
[![Unity Catalog](https://img.shields.io/badge/Unity%20Catalog-models%20%26%20volumes-FF3621?logo=databricks&logoColor=white)](https://docs.databricks.com/aws/en/data-governance/unity-catalog/)
[![NVIDIA Nemotron](https://img.shields.io/badge/NVIDIA-Nemotron-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/nvidia)
[![Models license](https://img.shields.io/badge/models-NVIDIA%20Open%20Model%20License-76B900)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
<br>
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11%20(cu130)-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![vLLM 0.24](https://img.shields.io/badge/vLLM-0.24-30A2FF)](https://github.com/vllm-project/vllm)
[![MLflow 3](https://img.shields.io/badge/MLflow-3.13-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest%20(CPU)-0A9EDC?logo=pytest&logoColor=white)](tests/)
[![Last commit](https://img.shields.io/github/last-commit/lbruand-db/nemotron-on-databricks)](https://github.com/lbruand-db/nemotron-on-databricks/commits/main)

This cookbook fine-tunes and serves NVIDIA's open **Nemotron document-intelligence models** on Databricks. Those models are the ones behind the NeMo Retriever extraction NIMs, used here through their open Hugging Face weights. Training runs on **AI Runtime (AIR)** serverless GPUs, and serving runs on **Model Serving**, governed by **Unity Catalog**.

Each recipe follows the same three stages:

```
 1. train (AIR GPU workload)          2. register (AIR GPU workload)         3. deploy + query (local)
 ─────────────────────────────        ──────────────────────────────         ─────────────────────────
 prepare data (cached on a Volume)    package pinned NVIDIA code +           create/update a Model Serving
 evaluate the base model              fine-tuned weights as an MLflow        endpoint for one exact UC
 fine-tune                            pyfunc, smoke-test it, register a      model version, then query it
 evaluate the fine-tuned model        new UC model version
 save weights + metrics to Volume
```

The handoff between stages is the Volume path plus the training run's MLflow run ID, so each stage can be retried without repeating the previous one. Every training run logs **base and fine-tuned metrics on the same held-out split**.

## Models

| Model | Task | Architecture | Fine-tuning | Data | Serving | Status |
|---|---|---|---|---|---|---|
| [nemotron-page-elements-v3](https://huggingface.co/nvidia/nemotron-page-elements-v3) | Page layout detection (table, chart, title, infographic, text, header/footer) | YOLOX-L, 54 M | Full fine-tune, SimOTA loss | DocLayNet v1.2 | GPU pyfunc | ✅ [done](recipes/page-elements-v3/): test mAP 0.21 → 0.64 |
| [nemotron-table-structure-v1](https://huggingface.co/nvidia/nemotron-table-structure-v1) | Table cells, rows and columns | YOLOX-L, 54 M | Full fine-tune | PubTables-1M | GPU pyfunc | ✅ [done](recipes/table-structure-v1/): test mAP 0.49 → 0.86 |
| [nemotron-graphic-elements-v1](https://huggingface.co/nvidia/nemotron-graphic-elements-v1) | Chart titles, axes, legends and labels | YOLOX-L, 54 M | Full fine-tune | Synthetic charts | GPU pyfunc | ✅ [done](recipes/graphic-elements-v1/): synthetic test mAP 0.40 → 0.84 |
| [nemotron-ocr-v1](https://huggingface.co/nvidia/nemotron-ocr-v1) | OCR: text detection, recognition and reading order | RegNetY detector + Transformer recognizer + relational model, 52 M | Recognizer only | CORD-v2 | GPU pyfunc + C++/CUDA extension | ✅ [done](recipes/ocr-v1/): CER 4.95 % → 1.63 % |
| [nemotron-nano-12b-v2-vl](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) | Document visual Q&A and extraction | RADIO + hybrid Mamba-Transformer, 12.6 B | LoRA (HF + PEFT) | CORD-v2 → JSON | vLLM (FP8) in a GPU pyfunc on 1× A10G (Custom LLM Serving blocked, SPEC OQ-5) | ✅ [trained and served](recipes/nano-12b-v2-vl/): field F1 0.39 → 0.86 (FP8: 0.85), ~4 s per receipt |

All five models are under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), which allows commercial use. The datasets are CC-BY-4.0, CDLA-Permissive or synthetic.

## Repository layout

```text
SPECS/SPEC.md         design spec: decisions, per-model plans, open questions, M0 findings (§11)
SPECS/dbrx-vs-azure.md  Azure-native vs Azure Databricks (West/North Europe): regions, complexity, cost
src/nemotron_dbx/     shared package, shipped with every workload
  common/             parameters, paths, pinned HF sources, MLflow helpers, rank helpers
  yolox_train/        YOLOX SimOTA loss (ported from Megvii, Apache-2.0), data pipeline, trainer
  detection/          NVIDIA detector loading, shared fine-tune/eval runner, DocLayNet / PubTables-1M prep,
                      synthetic chart generator, COCO mAP, serving pyfunc
  ocr/                OCR recognizer fine-tuning on GT quads, CORD-v2, CER/BoW metrics, serving pyfunc
  vlm/                Nano VL: training deps + remote-code patches, LoRA trainer, shard merge, vLLM eval,
                      serving (classic pyfunc running vLLM, or the Custom LLM Serving placeholder)
  lifecycle/          register (smoke test + UC), deploy (idempotent endpoint), query
recipes/<model>/      thin recipes: workload.yaml, register.yaml, train.py, register.py, README.md
recipes/spikes/       M0 de-risking workloads (vLLM serving, LoRA, OCR build, ...)
tests/                CPU unit tests (pytest)
tools/air_wait.sh     waits for an AIR run, then prints its logs
tools/endpoint_wait.sh  waits for a serving endpoint to be READY (shows crash loops)
```

## Prerequisites

- Databricks CLI ≥ 1.19 with AI Runtime, authenticated with `databricks auth login --profile <PROFILE>`.
- A workspace with AI Runtime GPUs (`GPU_1xA10`, `GPU_1xH100`) and Model Serving GPU workloads.
- An existing writable **Unity Catalog Volume** for outputs and dataset caches, and `CREATE MODEL` on the target schema. The recipes default to `/Volumes/am_rag_fevm_catalog/nemotron/artifacts`; change `output_root` and `registered_model_name` for your workspace.
- Every workload uses the stock `databricks_ai_v6` environment. **No custom Docker images are needed**: the CUDA extensions are built on AIR once and cached as wheels on the Volume (SPEC §11.2).

## Quick start

Run all commands from the repository root:

```bash
# 1. Train + evaluate
databricks air run -f recipes/page-elements-v3/workload.yaml -p <PROFILE>
tools/air_wait.sh <JOB_RUN_ID> <PROFILE>

# 2. Register in Unity Catalog
databricks air run -f recipes/page-elements-v3/register.yaml -p <PROFILE> \
  --override parameters.source_training_run_id=<MLFLOW_RUN_ID>

# 3. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.<MODEL> --model-version <N> \
  --endpoint-name nemotron-page-elements-v3-ft --workload-type GPU_SMALL --profile <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-page-elements-v3-ft --profile <PROFILE> page.png
```

See each recipe's README for its details.

## Development

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install torch torchvision numpy pillow pyarrow \
  pycocotools pyyaml pytest ruff pandas huggingface_hub databricks-sdk
.venv/bin/ruff check src tests recipes
.venv/bin/python -m pytest -q
```

Some tests check the ported code against NVIDIA's real modules. To enable them, set `NEMOTRON_PE3_CODE` to a local copy of the `nemotron_page_elements_v3/` Python files; weights are not needed.

## What M0 established

These results were measured on 2026-11-10; details are in [SPECS/SPEC.md §11](SPECS/SPEC.md).

- **AI Runtime:** `databricks_ai_v6` provides torch 2.11 (cu130), transformers 5.8.1, vLLM 0.24 and peft 0.19. Shared code ships from the repo root through `code_source.snapshot`.
- **Nano 12B VL:**
  - **Serving:** vLLM serves it with image inputs on 1× H100.
  - **LoRA training:** HF + PEFT on 1× H100 peaks at 34.6 GB. It takes 10 documented compatibility patches.
- **OCR:** the C++/CUDA extension builds on AIR in about 4 minutes, and inference takes about 70–90 ms per page on an A10G.

Since M0, M1 has shown that page-elements-v3 trains in 22 min on 1× A10 and serves on a `GPU_SMALL` (T4) endpoint. Use pinned pip requirements, not `env_pack`, for classic pyfuncs (SPEC OQ-17).

M3 then showed that the OCR recognizer fine-tunes in under 9 min, and that its compiled C++/CUDA extension serves on a T4 endpoint, shipped as a wheel with exact torch pins (SPEC OQ-9).

M4 fine-tuned Nano 12B VL with LoRA (CORD field F1 0.39 → 0.86 in 31 min on 1× H100), after fixing a silent transformers 5 image-preprocessing bug (SPEC §11.3 patch 11). It is served on `e2-demo-field-eng` as a classic pyfunc running vLLM with **FP8 weights on one A10G** (`GPU_MEDIUM`): field F1 0.847 vs 0.858 in bf16, ~4 s per receipt. The other two routes failed on both test workspaces. Custom LLM Serving (`env_pack`) deployments never start the entrypoint, even for a cookbook-identical tiny model (SPEC OQ-5). The 48–96 GB serving GPUs (`GPU_LARGE`, `GPU_LARGE_RTX`) never got capacity within the 2 h limit.

M2 added table-structure-v1 (PubTables-1M, test mAP 0.49 → 0.86, served on T4) and graphic-elements-v1 (synthetic charts, 0.40 → 0.84). A real-chart check caught a regression that synthetic mAP hid (captions mislabelled as axis text), fixed by adding captions to the generator.

## Non-goals

- **No NIM containers.** Model Serving doesn't host arbitrary containers, and NIM needs NVIDIA AI Enterprise.
- **No production-scale training.** Recipes are small, cheap integration examples meant to be adapted to your own data.

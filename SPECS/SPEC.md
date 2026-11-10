# SPEC — Nemotron Document-Intelligence Cookbook on Databricks

| | |
|---|---|
| Status | **Draft v0.2** — decisions OQ-1, OQ-8 resolved; target workspace set |
| Date | 2026-11-10 |
| Owner | lbruand-db |
| Scope | Fine-tune **and** serve 5 NVIDIA Nemotron document-intelligence models on Databricks (AI Runtime + Model Serving) |
| Target workspace | FEVM `am-rag-fevm`: `https://fevm-am-rag-fevm.cloud.databricks.com/` (AWS us-west-2, "AWS Stable Serverless" template, **expires 2026-11-28**; CLI profile `am-rag-fevm`) |

### Decision log

| Date | Decision |
|---|---|
| 2026-11-10 | **No upstreaming to `ai-runtime-cookbook` for now.** Recipes keep the cookbook *style*, but they share code through a common package in this repo (resolves OQ-1). |
| 2026-11-10 | **No `€`/`£` support for now.** The OCR charset extension is out of scope (resolves OQ-8). |
| 2026-11-10 | Spikes and end-to-end validation run on the FEVM workspace `fevm-am-rag-fevm`. |

---

## 1. Goal

Provide a runnable cookbook that shows, for each of the five NVIDIA models below, how to:

1. **Fine-tune** the model on Databricks **AI Runtime (AIR)** serverless GPUs, launched with `databricks air run -f workload.yaml`.
2. **Evaluate** base vs. fine-tuned model on a held-out split with a task-appropriate metric.
3. **Register** the result in **Unity Catalog** (MLflow 3) with lineage to the training run.
4. **Serve** the registered version on **Databricks Model Serving** and query it.

The models together form NVIDIA's document-ingestion stack (the NeMo Retriever / "Nemotron" extraction NIMs). The cookbook uses the **open Hugging Face weights** instead of NIM containers, so everything runs natively on Databricks managed infrastructure, governed by Unity Catalog.

### 1.1 In-scope models

| # | NIM name | Hugging Face ID | Task | Architecture | Params |
|---|---|---|---|---|---|
| 1 | nemotron-page-elements-v3 | `nvidia/nemotron-page-elements-v3` | Page layout detection: table, chart, infographic, title, header_footer, text | YOLOX (DarkNet53 + FPN, decoupled head), 1024×1024 input | 54 M |
| 2 | nemotron-table-structure-v1 | `nvidia/nemotron-table-structure-v1` | Table structure detection: cell, row, column | YOLOX, 1024×1024 | 54 M |
| 3 | nemotron-graphic-elements-v1 | `nvidia/nemotron-graphic-elements-v1` | Chart element detection: chart/axis/legend titles, axis/legend/marker/value labels, other | YOLOX, 1024×1024 | 54 M |
| 4 | nemotron-ocr-v1 | `nvidia/nemotron-ocr-v1` | End-to-end OCR: text detection, recognition, reading order | RegNetY-8GF detector (FOTS-style) + Transformer/CTC recognizer + relational model | 52 M |
| 5 | nemotron-nano-12b-v2-vl | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` (also `-FP8`, `-NVFP4-QAD`) | Document VQA, extraction, summarization; multi-image and video | C-RADIOv2-H vision encoder + Nemotron-Nano-12B-v2 hybrid Mamba-Transformer LLM | 12.6 B |

All five are licensed under the **NVIDIA Open Model License** (commercial use allowed). The post-processing scripts in the HF repos are Apache-2.0. None of the HF repos is gated.

### 1.2 Non-goals

- Running **NIM containers** on Databricks. Model Serving does not host arbitrary containers, and NIM needs an NVIDIA AI Enterprise entitlement. We may mention NIM as the alternative for production deployments outside Databricks.
- Production-scale training. Each recipe is a **small, cheap integration example** (minutes, not hours), in line with cookbook norms.
- Re-training from scratch, or changing an architecture beyond what fine-tuning needs (one exception is described in §6.4.3).
- A full RAG / ingestion product. An optional pipeline demo appears in §7, but it is not on the critical path.

---

## 2. Sources and prior art

- Databricks × NVIDIA blog, *Building the agentic era* (2026-06-17). It covers AIR multi-node H100 training, Model Serving on NVIDIA hardware, Triton, NVIDIA Agent Toolkit on Apps, and Nemotron Parse.
- `databricks-solutions/ai-runtime-cookbook`. Our **reference patterns**:
  - `multimodal/lora-nemotron-3.5-super-vl-120b-a12b-automodel-multi-gpu`: custom AIR Docker image pushed to UC, NeMo AutoModel, adapter copied to a UC Volume.
  - `serving/sft-qwen2.5-0.5b-lora-serving`: 3-stage flow. (1) AIR training writes to a Volume. (2) A DAB job registers the model, after a vLLM smoke test. (3) `deploy_endpoint.py` creates or updates a **Custom LLM Serving** endpoint (MLflow pyfunc with a `metadata.entrypoint` vLLM command and `env_pack="databricks_model_serving"`).
  - `CONTRIBUTING.md`: recipe layout, config placement, storage paths, README template.
- HF model cards and source code for the five models. Key facts are captured in §6.
- NVIDIA framework coverage, checked 2026-11-10:
  - **NeMo AutoModel** (`NVIDIA-NeMo/Automodel`) has **no recipe for Nemotron Nano 12B v2 VL**. It covers Nemotron 3.5 Super VL, Nemotron Parse, and others.
  - **Megatron-Bridge** **removed** Nemotron Nano v2 VL support. The last documented version is 0.4.2, with the container `nvcr.io/nvidia/nemo:25.09.nemotron_nano_v2_vl`.
  - The YOLOX and OCR HF repos ship **inference-only code**. The YOLOX cards say "for training refer to the original repo" (Megvii YOLOX, Apache-2.0).

---

## 3. Design principles

1. **Cookbook-style, but with shared code.** Upstreaming is not a goal (decision log). Recipes keep the cookbook conventions, with one deliberate exception: shared code is allowed.
   - A leaf directory with `workload.yaml`, an entry point, and a short `README.md`.
   - Common code (the YOLOX training stack, detector pyfunc, registration and deploy helpers, eval metrics) lives **once** in `src/nemotron_dbx/` and is shipped with each workload (see §4.1).
   - User-configurable values go in `workload.yaml` (`parameters` / `env_variables` / `secrets`).
   - Pinned versions are used for anything added on top of the AIR environment.
   - Durable outputs go under `<volume_root>/<recipe-name>/<MLFLOW_RUN_ID>/…` (volume root is a bundle/workload variable; see OQ-14).
   - `max_retries` is set explicitly. Downloads, logging, and checkpoints are rank-safe.
2. **Same three-stage lifecycle for every model:** `train (AIR)` → `register (DAB job on AIR GPU)` → `deploy (local script)`. The Volume path and source MLflow run ID are the handoff between stages, so each stage can be retried on its own.
3. **Measure, don't assert.** Each training run logs base vs. fine-tuned metrics on the same held-out split to the AIR MLflow run.
4. **License-clean data.** Only datasets whose license allows commercial use and redistribution of derivatives (CC-BY, CDLA-Permissive), or synthetic data.
5. **Pin upstream code.** HF model repos are pinned to an immutable `revision` (commit SHA), because the model code itself is pulled from the Hub.
6. **Uniform serving contract** for the three YOLOX detectors and OCR (§5.3), with a response shape close to the NIM API so migration is easy.

---

## 4. Repository layout

```text
nemotron-on-databricks/
├── README.md                          # index: models, recipes, prerequisites, cost/time table
├── SPECS/SPEC.md                      # this document
├── pyproject.toml                     # shared package + dev tooling (ruff, pytest)
├── databricks.yml                     # ONE DAB bundle: variables (catalog/schema/volume), all registration jobs
├── resources/                         # register_<model>.job.yml (one per model)
├── src/nemotron_dbx/                  # shared code (see §4.1)
│   ├── yolox_train/                   #   SimOTA loss, IOU loss, dataset/augment, train loop, EMA (vendored from Megvii YOLOX, Apache-2.0)
│   ├── detection/                     #   COCO conversion, mAP eval, detector pyfunc (§5.3), NIM-like response formatting
│   ├── ocr/                           #   CTC recognizer fine-tune, CER/WER eval, OCR pyfunc
│   ├── vlm/                           #   CORD prompt/target formatting, LoRA train loop, merge, field-F1/TED eval
│   ├── lifecycle/                     #   register_model (smoke test + log + UC register), deploy_endpoint (idempotent), query client
│   └── common/                        #   paths, MLflow run helpers, rank-safe IO, HF revision pinning
├── tests/                             # unit tests: label mapping, loss shapes, metrics, pyfunc I/O contract (CPU-only)
├── docker/                            # custom AIR images
│   ├── ocr/Dockerfile                 #   nemotron-ocr C++/CUDA extension build
│   └── nano-vl/Dockerfile             #   transformers 4.53.x + mamba-ssm + causal_conv1d + peft
└── recipes/
    ├── page-elements-v3/
    ├── table-structure-v1/
    ├── graphic-elements-v1/
    ├── ocr-v1/
    ├── nano-12b-v2-vl/
    └── spikes/                        # M0 de-risking workloads (§10)
```

Each recipe directory is thin: configuration plus an entry point that calls `nemotron_dbx`.

```text
recipes/<model>/
├── README.md          intro, Before you run, Run (3 stages), Results (observed time + metrics)
├── workload.yaml      AIR training workload (parameters = all hyper-params, data subset, output root)
├── train.py           entry point: prepare data → base eval → fine-tune → fine-tuned eval → save
└── register_job.py    notebook entry for the registration job (widgets → nemotron_dbx.lifecycle)
```

### 4.1 Shipping shared code to workloads

- **AIR training:** `workload.yaml` sets `code_source.root_path` to the **repo root**, with `include_paths: [src/, recipes/<model>/]`. The command runs `pip install --no-deps -e "$CODE_SOURCE_PATH"` (or sets `PYTHONPATH="$CODE_SOURCE_PATH/src"`) and then `python recipes/<model>/train.py`. Workloads are launched from the repo root: `databricks air run -f recipes/<model>/workload.yaml`. *Verify in M0 that `root_path` may point above the YAML's directory. Fallback: put the `workload.yaml` files at repo root as `workloads/<model>.yaml`.*
- **Registration jobs (DAB):** the bundle syncs `src/`. Notebooks add it to `sys.path`, or the bundle builds a wheel artifact (`artifacts:` → `python_wheel`) that the job environment installs.
- **Serving:** each pyfunc is logged with `code_paths=["src/nemotron_dbx"]`, so endpoints never depend on the repo at runtime.

---

## 5. Shared conventions

### 5.1 Storage and naming

| Item | Convention |
|---|---|
| Volume root | `/Volumes/<catalog>/<schema>/<volume>/<recipe-name>`, default `/Volumes/main/default/air_examples/<recipe-name>`; set per workspace (OQ-14) |
| Training outputs | `<root>/<MLFLOW_RUN_ID>/{weights|merged|adapter}/` + `metrics.json` |
| Dataset cache | `<root>/data/<dataset>@<revision>/`, so it can be reused across runs (stable ID, not the run ID) |
| HF cache | `HF_HOME=/local_disk0/hf` (node-local) |
| UC model | `${catalog}.${schema}.nemotron_<model>_ft`, e.g. `main.default.nemotron_page_elements_v3_ft` |
| Endpoint | `nemotron-<model>-ft` |
| MLflow | Log into the AIR-provided `MLFLOW_RUN_ID`. Tag `merged_model_path` / `weights_path` so registration can resolve the artifacts. |

### 5.2 Registration (stage 2)

- Runs as a **DAB job on an AIR GPU task**, so the packaging environment matches the serving environment, particularly for CUDA extensions.
- Steps:
  1. Resolve the artifacts from the training-run tag.
  2. Run a **local smoke test** of the exact packaged artifact (pyfunc `predict` or a local vLLM server).
  3. `mlflow.pyfunc.log_model(...)`.
  4. `mlflow.register_model(..., env_pack="databricks_model_serving")`.
  5. Tag both runs with the registered version.
- Never alias or overwrite. Each run creates a new version, and this is documented.

### 5.3 Serving contract for the detectors and OCR (models 1–4)

The endpoint is a standard (non-LLM) Model Serving pyfunc on GPU.

**Request** (`dataframe_records` or `inputs`):

```json
{"inputs": [{"image": "<base64 PNG/JPEG>", "id": "page-1"}],
 "params": {"conf_threshold": null, "merge_level": "paragraph"}}
```

**Response**: one entry per input image. Coordinates are **normalized [0,1]**, `x_min,y_min,x_max,y_max`. The shape mirrors the NIM object-detection response; exact field names must be checked against NIM docs (OQ-6).

```json
{"predictions": [{"id": "page-1",
  "bounding_boxes": {"table": [{"x_min":0.1,"y_min":0.2,"x_max":0.9,"y_max":0.5,"confidence":0.93}],
                     "title": [...] }}]}
```

OCR adds `text` per box, plus `reading_order` when the relational model is enabled.

- Model weights are packaged as MLflow artifacts. The upstream HF package (pinned revision) goes in as `code_paths` or a vendored wheel. Nothing is downloaded from the Hub at serving time.
- Workload: `GPU_SMALL` by default (verify against regional availability). Scale-to-zero is on.
- Batch handling: the pyfunc accepts N images and runs them as one batch (padded to 1024²).

### 5.4 Serving for Nano VL (model 5)

The model is served through **Custom LLM Serving**, the cookbook pattern: a pyfunc placeholder whose `metadata.entrypoint` starts `vllm.entrypoints.openai.api_server`, with `task: llm/v1/chat`. The endpoint uses the OpenAI-compatible chat format with `image_url` (base64 data URL) content parts. Multimodal support on Custom LLM Serving must be verified (OQ-5).

---

## 6. Per-model recipes

### 6.1 `recipes/page-elements-v3`

| | |
|---|---|
| **Dataset** | **DocLayNet** (`docling-project/DocLayNet-v1.2`, CDLA-Permissive-1.0; 80,863 pages, 11 classes, 6 doc categories incl. *financial reports*). Default subset: `financial_reports` + `manuals`, ~2k train / 300 val / 300 test pages, from the official splits. |
| **Label mapping** (11 → 6) | `Table→table`; `Title, Section-header→title`; `Page-header, Page-footer→header_footer`; `Text, List-item, Caption, Footnote, Formula→text`; `Picture→ambiguous` (chart vs infographic vs photo). **v1 default: drop training pages that contain `Picture`**, which avoids false negatives on chart/infographic. Alternative: ignore-region masking (OQ-2). |
| **Fine-tune method** | Full fine-tune of the 54 M YOLOX model with the same architecture and class count, loading `weights.pth` strictly. The HF `YOLOXHead.forward` has **no training branch**, so we vendor `get_losses` / SimOTA assignment / `IOUloss` from Megvii YOLOX (Apache-2.0), adapted to the HF module names. The HF wrapper's letterbox preprocessing (`resize_pad` to 1024²) is reused for train and inference. Augmentation: scale jitter (0.8–1.2), color/contrast jitter, slight JPEG noise. **No mosaic/flip**, because document layout semantics are orientation-sensitive. |
| **Hyper-params (default)** | AdamW, lr 1e-4 (backbone 1e-5), cosine schedule, 10 epochs, batch 8, bf16 AMP, EMA 0.9998 |
| **Compute** | `GPU_1xA10` (single GPU, about 15–25 min target). DDP optional via `torchrun` on `GPU_8xH100` (documented, not the default). |
| **Eval** | COCO mAP@[.5:.95] and mAP@.5, per class and overall (`pycocotools`), on the test split, **base vs fine-tuned**. Use the HF post-processing (`postprocess_preds_page_element`, per-class thresholds, WBF) so the numbers reflect what serving returns. |
| **Artifacts** | `<root>/<run>/weights/weights.pth` + `config.json` (thresholds) + `metrics.json` |
| **Serving** | pyfunc per §5.3; labels `table, chart, infographic, title, header_footer, text` |

### 6.2 `recipes/table-structure-v1`

| | |
|---|---|
| **Dataset** | **PubTables-1M structure split** (`bsmock/pubtables-1m`, CDLA-Permissive-2.0, PASCAL-VOC XML). Downloads are tar.gz only, so `prepare_data.py` downloads only the shards it needs: *Val* annotations and images, split 80/20 into our train/val, and a subset of *Test* as the held-out test set. The data is cached in the Volume. Default size: ~3k train / 300 val / 300 test table crops. |
| **Label mapping** | `table row→row`, `table column→column`. **`cell`** is derived as row×column intersections, with `table spanning cell` boxes replacing the intersections they cover (merged cells). The mapping must be validated visually against base-model predictions on a sample, to confirm that the NVIDIA "cell" semantics match (OQ-3). Other PubTables classes are dropped. |
| **Fine-tune / compute** | Same YOLOX training stack as §6.1 (vendored loss). Input: table crop letterboxed to 1024². Same compute. |
| **Eval** | mAP per class (cell/row/column) **and** a downstream metric: run `nemotron-ocr-v1` (base) + the HF `table_struct_pp` post-processing to rebuild the table as markdown, then score TEDS / exact-cell accuracy against the PubTables ground-truth cell text on 100 tables. |
| **Serving** | pyfunc per §5.3; labels `cell, row, column`. The optional `return_markdown=true` param requires OCR boxes in the request (see the HF `Demo.ipynb` flow). |

### 6.3 `recipes/graphic-elements-v1`

| | |
|---|---|
| **Dataset (default)** | **Synthetic charts**, generated by `prepare_data.py` with matplotlib: bar, line, scatter, pie, stacked, and horizontal variants, with random fonts, sizes, palettes, and locales (including French number formats such as `1 234,56 €`). Exact ground-truth boxes for all 10 classes come from `Artist.get_window_extent()`. This makes the data fully reproducible and license-clean, with no download. Default: 3k train / 300 val, plus a **real-chart test set** (OQ-4). |
| **Alternative dataset** | ICPR 2022 CHART-Infographics PMC set (used by NVIDIA for v1). Its license and redistribution terms must be checked before use (OQ-4). |
| **Classes** | `chart_title, x_title, y_title, xlabel, ylabel, legend_title, legend_label, mark_label, value_label, other`. The exact label strings are taken from the HF `config.json`. |
| **Fine-tune / compute / serving** | Same as §6.1. Input is the chart crop. |
| **Eval** | mAP per class on (a) the synthetic test set and (b) the real-chart test set. Report both, so improvement on synthetic data that hurts real charts is visible. |

### 6.4 `recipes/ocr-v1`

#### 6.4.1 What fine-tuning means here

The model has three jointly trained components: a detector (45 M), a recognizer (4.9 M), and a relational model (2.3 M). The recognizer is a conv + 3-layer Transformer-encoder **CTC** head (blank = index 0, beam decoding with an optional n-gram LM). It reads **rectified feature maps from the detector backbone**, not raw crops; these come from the FOTS-style RoI rotate using the custom `indirect_grid_sample` op.

**v1 scope: recognizer-only fine-tuning.**

- Freeze the detector and relational models.
- For each training image, run the frozen detector backbone, rectify the **ground-truth quads**, run the recognizer, and compute the CTC loss against the GT text. No gradient flows through the custom CUDA op.
- **Stretch (v2):** detector fine-tuning (FOTS detection loss: score map + geometry) and relational fine-tuning. These need loss code that the repo does not ship, so they are out of v1 (OQ-7).

#### 6.4.2 Data

- **CORD-v2** (`naver-clova-ix/cord-v2`, CC-BY-4.0): 800 / 100 / 100 receipts with **word-level quads and transcriptions** in `ground_truth.valid_line[].words[]`. The aggregation level is word or line, matching the `merge_level` input.
- The same dataset is used by the Nano VL recipe (§6.5), so results can be compared end-to-end on receipts.

#### 6.4.3 Charset constraint

`checkpoints/charset.txt` contains **855 tokens**. It includes French accented letters (é, à, ç, œ…) but **does not include `€` or `£`**.

- **Decision (2026-11-10):** fine-tune within the existing charset only. `€`/`£` support is out of scope for now.
- Ground-truth characters outside the charset are mapped to a configurable fallback (default: drop). The **out-of-charset rate** is reported in the eval, so the gap stays visible.
- *Future (not planned):* grow the classifier's output layer with new tokens, initialized from the mean of existing rows, and save an extended charset.

#### 6.4.4 Build and environment

- Python 3.12, CUDA 12.8, and a **C++/CUDA extension** (`nemotron_ocr_cpp`, built with hatchling and needing GCC with C++17, CUDA headers, and OpenMP).
- A **custom AIR image** is built from a `Dockerfile` (base `nvcr.io/nvidia/pytorch:25.xx-py3`, matching the repo's own Dockerfile). It contains the pinned HF repo revision and a prebuilt wheel, and is pushed with `databricks air images push` (the Super VL recipe pattern).
- For serving, the **prebuilt wheel** is shipped as an MLflow artifact and installed via `extra_pip_requirements` (local wheel path). The alternative is `env_pack`. The wheel must be ABI-compatible with the serving image (OQ-9).

#### 6.4.5 Compute, eval, serving

- **Compute:** `GPU_1xA10`, ~20 min target.
- **Eval:**
  1. Recognition CER/WER on CORD-v2 test, with GT quads (isolates the recognizer).
  2. End-to-end page CER and bag-of-words error with predicted quads.
  3. Out-of-charset rate.
  All three are reported base vs fine-tuned.
- **Serving:** pyfunc per §5.3. It returns `{bbox, text, confidence}` per region and `reading_order`, and accepts the `merge_level ∈ {word, sentence, paragraph}` param.

### 6.5 `recipes/nano-12b-v2-vl`

#### 6.5.1 Fine-tune framework decision

NeMo AutoModel has no Nano v2 VL recipe, and Megatron-Bridge has removed support. **Decision: Hugging Face `transformers` + `peft` LoRA with a minimal custom training loop** (or HF `Trainer`), loading the model with `trust_remote_code=True`.

- Pinned versions, following the model card: `transformers>4.53,<4.54`, `mamba-ssm==2.2.5`, `causal_conv1d`, `timm`, `open_clip_torch`, plus `peft` and `accelerate`.
- These need compiled CUDA kernels (Mamba), so the recipe uses a **custom AIR image** (`Dockerfile`), pushed to UC.
- Fallback: the frozen Megatron-Bridge 0.4.2 container, documented only (OQ-10).

#### 6.5.2 Training setup

| | |
|---|---|
| **Dataset** | **CORD-v2** receipts → JSON key-information extraction (menu items, prices, totals). This is the same task as the Super VL cookbook recipe, so the two models can be compared. The prompt uses the system message `/no_think` and a fixed extraction instruction, with the target being canonical JSON of `gt_parse`. |
| **LoRA targets** | LLM attention `q,k,v,o` and MLP `up,down` projections (plus Mamba `in_proj,out_proj` as an option, OQ-11). Rank 32, α 64, dropout 0.05. The **vision encoder (RADIO) and projector are frozen**. |
| **Precision / memory** | bf16 weights (~25 GB), gradient checkpointing, max seq len 4096, ≤ 12 tiles of 512² per image |
| **Compute** | Single node. Default `GPU_8xH100` with DDP, ~15 min target. A 1×H100 or A100-80GB variant is documented if the accelerator type exists in AIR (OQ-12). |
| **Hyper-params** | lr 1e-4, cosine, 1–2 epochs over 800 train receipts, global batch 16 |
| **Eval** | **Field-level F1 and normalized tree-edit-distance accuracy** (the Donut/CORD standard) on CORD-v2 test (100), plus JSON validity rate. Base vs fine-tuned are both run with **vLLM batch inference**, so the eval measures the serving path. |
| **Artifacts** | `adapter/`, plus a **merged HF checkpoint** in `merged/`. The merged directory must contain all remote-code files (`modeling*.py`, `configuration*.py`, `processing*.py`, `image_processing.py`, chat templates, `evs.py`, `video_io.py`), copied from the pinned base revision. |

#### 6.5.3 Serving

- **Stage 2:** start `vllm serve <merged> --trust-remote-code --dtype bfloat16 --video-pruning-rate 0 --max-model-len 16384` locally on an AIR GPU task, send one image request as a smoke test, then register as a Custom LLM Serving model (§5.4). The vLLM version must support `NemotronH_Nano_VL_V2`; the minimum version is to be pinned.
- **Workload:** 12.6 B bf16 needs ≥ 40 GB GPU memory. Use `GPU_LARGE` (A100-80GB) or an H100 class if offered.
- **Optional:** serve **FP8**, using `--quantization modelopt` from the NVIDIA FP8 checkpoint. That checkpoint has no LoRA, so FP8 + fine-tune would need post-merge ModelOpt PTQ (stretch).
- **Alternative: no merge.** Serve the base model with `--enable-lora` and the adapter, if Custom LLM Serving allows passing adapter artifacts (stretch).
- **Client:** `query_endpoint.py` sends an OpenAI-format chat request with an `image_url` data URL.

---

## 7. Optional: end-to-end pipeline demo (stretch)

A notebook `recipes/pipeline-demo/` that chains the five **served** endpoints on a sample PDF:

1. Page → `page-elements`.
2. Crops → `table-structure` / `graphic-elements` / `ocr`.
3. Assemble markdown.
4. Ask `nano-vl` targeted questions about the page.
5. Write the results to a Delta table.

Compute choices to compare:

- Ray Data batch inference on AIR, for throughput (the cookbook has `ray-data-batch-inference-qwen3-vl-8b-vllm` as a template).
- Endpoints, for latency.

It also shows how the pipeline could be exposed as **tools** for an NVIDIA Agent Toolkit or Databricks agent app. This is out of scope for v1.

---

## 8. Acceptance criteria

For **each** of the 5 recipes:

- [ ] `databricks air run -f recipes/<model>/workload.yaml --watch` (from the repo root) completes on `fevm-am-rag-fevm` with only the documented prerequisites, within the stated time budget (target ≤ 30 min of training).
- [ ] The AIR MLflow run contains params, training curves, **base and fine-tuned metrics on the same test split**, and artifacts. The weights are on the Volume under `<root>/<MLFLOW_RUN_ID>/`.
- [ ] `databricks bundle run register_<model>` passes its smoke test and prints the exact UC model version. Lineage tags are present on both runs.
- [ ] `deploy_endpoint.py` is idempotent: running it twice gives `unchanged`. The endpoint reaches `READY` and serves the requested version.
- [ ] `query_endpoint.py` returns a response that matches the §5.3 or §5.4 contract on the sample image.
- [ ] The README follows the cookbook template. It contains no credentials, personal catalogs, internal workspace names, or job URLs.
- [ ] Observed duration and metrics are recorded in the README, with the model and dataset revisions.

Repo-level:

- [ ] The top-level `README.md` has a model → recipe → compute → time → metric table.
- [ ] Every dependency added on top of the AIR environment is pinned. HF model repos are pinned to commit SHAs.
- [ ] `pytest tests/` passes locally on CPU: label mappings, loss shapes on random tensors, metrics against known values, and the pyfunc request/response contract.

---

## 9. Open questions and risks

| ID | Question / risk | Default until resolved |
|---|---|---|
| ~~OQ-1~~ | ~~Duplicate YOLOX code per recipe vs. a shared package~~ | **Resolved:** shared `src/nemotron_dbx` package, no upstreaming |
| OQ-1b | Can AIR `code_source.root_path` point at the repo root, above the `workload.yaml` directory? | Verify in M0; fallback is root-level `workloads/*.yaml` |
| OQ-2 | DocLayNet `Picture` is ambiguous for chart vs infographic. Drop those pages, add ignore regions, or relabel a subset? | Drop pages |
| OQ-3 | Do PubTables-derived "cells" match NVIDIA's `cell` semantics (empty cells, spanning cells)? | Validate visually on 50 tables before training |
| OQ-4 | Real-chart test set for graphic-elements: which license-clean source? | Small hand-labeled set or PMC subset if license OK |
| OQ-5 | Does Custom LLM Serving accept **multimodal** (`image_url`) chat requests end-to-end, and which GPU workload types does it offer for a 12 B model? | Verify with a product contact early; fallback is a pyfunc wrapping `transformers.generate` |
| OQ-6 | Exact NIM response schemas for page-elements, table-structure, graphic-elements, and OCR, to mirror field names | Mirror best-effort, document differences |
| OQ-7 | OCR detector/relational fine-tuning needs loss code not shipped in the HF repo | v1 = recognizer only |
| ~~OQ-8~~ | ~~OCR charset lacks `€`/`£`~~ | **Resolved:** not needed for now; report the out-of-charset rate only |
| OQ-9 | Will the C++/CUDA OCR extension built on AIR load in the Model Serving GPU image (CUDA, glibc, Python 3.12 ABI)? | Spike in week 1; fallback is CPU-only grid-sample path if available |
| OQ-10 | Nano VL LoRA through HF remote code: are training kernels (Mamba) stable with PEFT and gradient checkpointing under `transformers 4.53.x`? | Spike in week 1; fallback Megatron-Bridge 0.4.2 container |
| OQ-11 | Include Mamba projections in LoRA targets? | Attention + MLP only; Mamba as ablation |
| OQ-12 | Which AIR accelerator types and serving GPU workload types are available on **`fevm-am-rag-fevm`**? Are AIR, the custom Docker image preview, and Custom LLM Serving enabled there? | `GPU_1xA10` for small models, `GPU_8xH100` for Nano VL; check in M0 |
| OQ-14 | FEVM workspace TTL: is it long enough for M0–M5, and which catalog/schema/volume can we write to (`main.default.air_examples` may not exist)? | Check the FEVM deployment; create a project schema and volume if allowed |
| OQ-13 | Customer data: will the cookbook later be pointed at customer documents (in their own UC volume)? If so, each `prepare_data.py` needs a "bring your own COCO/JSONL" path. | Add a `dataset_source: hf|volume` parameter in v1 |

---

## 10. Milestones

| # | Deliverable | Notes |
|---|---|---|
| M0 | This spec reviewed. Spikes on `fevm-am-rag-fevm`, under `recipes/spikes/`: (a) workspace capabilities and storage (OQ-12, OQ-14); (b) `root_path` at repo root (OQ-1b); (c) Nano VL `vllm serve` + image request on Custom LLM Serving (OQ-5); (d) OCR extension build on AIR, then load in serving (OQ-9); (e) Nano VL PEFT LoRA forward/backward, 10 steps (OQ-10). | Highest technical risk first |
| M1 | `page-elements-v3` end-to-end (train → register → serve) | Establishes the YOLOX training stack and the detector serving contract |
| M2 | `table-structure-v1` + `graphic-elements-v1` | Reuse `nemotron_dbx.yolox_train` + `detection` from M1 |
| M3 | `ocr-v1` (recognizer fine-tune, custom image, serving) | |
| M4 | `nano-12b-v2-vl` LoRA → merged → Custom LLM Serving | |
| M5 | Top-level README, metrics table, optional pipeline demo (§7) | |

# SPEC — Nemotron Document-Intelligence Cookbook on Databricks

| | |
|---|---|
| Status | **Draft v0.7**: M0–M3 done (page-elements-v3, table-structure-v1, ocr-v1 served on GPU_SMALL; graphic-elements-v1 endpoint deploying). M4 (Nano VL) trained, serving blocked on the platform. |
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
| 2026-11-10 | **All workloads use the stock `databricks_ai_v6` environment; no custom Docker images.** The OCR extension and `causal-conv1d` are compiled on AIR and cached as wheels on the Volume (§11). `docker/` is kept only as a documented fallback. |
| 2026-11-10 | **Nano VL fine-tuning uses AIR's transformers 5.8.1** (not the model card's 4.53 pin), plus the compatibility patches in §11.3. One environment then serves training, vLLM eval, and serving (vLLM 0.24 requires transformers ≥ 5.5.3). Default compute is **`GPU_1xH100`**. |
| 2026-11-10 | Storage on `fevm-am-rag-fevm`: `am_rag_fevm_catalog.nemotron` schema, `artifacts` managed volume (there is no `main` catalog). |
| 2026-11-10 | **Registration (stage 2) runs as an AIR workload** (`recipes/<model>/register.yaml`), not as a DAB job. It is the same mechanism as training (repo-root snapshot, ai_v6, GPU, `MLFLOW_RUN_ID`), so the packaging environment matches training. The cookbook's requirement ("register from an AI Runtime GPU environment") is met, and no bundle is needed. |
| 2026-11-11 | **Classic pyfuncs are registered without `env_pack`** (pinned pip requirements instead). With `env_pack="databricks_model_serving"`, the endpoint crash-looped on T4 and timed out on A10G (OQ-17). `env_pack` stays the Custom LLM Serving (vLLM) path for Nano VL, as in the cookbook. |
| 2026-11-10 | **page-elements-v3 defaults to `GPU_1xA10`** (batch 4 × grad accum 2). H100 capacity was unavailable for > 90 min (OQ-16); H100 stays an override. |

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
- Re-training from scratch, or changing any architecture (the OCR charset extension was ruled out; see §6.4.3).
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
├── tools/air_wait.sh                   # poll an AIR run (task state incl. PENDING), then print its logs
├── src/nemotron_dbx/                  # shared code (see §4.1)
│   ├── yolox_train/                   #   SimOTA loss, IOU loss, dataset/augment, train loop, EMA (vendored from Megvii YOLOX, Apache-2.0)
│   ├── detection/                     #   COCO conversion, mAP eval, detector pyfunc (§5.3), NIM-like response formatting
│   ├── ocr/                           #   CTC recognizer fine-tune, CER/WER eval, OCR pyfunc
│   ├── vlm/                           #   CORD prompt/target formatting, LoRA train loop, merge, field-F1/TED eval
│   │   └── compat.py                  #   Nano VL remote-code patches for transformers 5 (§11.3)
│   ├── lifecycle/                     #   register (smoke test + log + UC register), deploy (idempotent endpoint), query client
│   └── common/                        #   paths, MLflow run helpers, rank-safe IO, HF revision pinning,
│                                      #   wheel cache (build-once CUDA extensions on the Volume, §11.2)
├── tests/                             # unit tests: label mapping, loss shapes, metrics, pyfunc I/O contract (CPU-only)
├── docker/                            # FALLBACK ONLY (not used by default): custom AIR images if ai_v6 stops working
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
├── register.yaml      AIR registration workload (parameters: source_training_run_id, registered_model_name)
└── register.py        entry point: resolve weights from the training run's tags → nemotron_dbx.lifecycle.register
```

### 4.1 Shipping shared code to workloads

- **AIR training:** `workload.yaml` sets `code_source.snapshot.root_path: ../..` (resolved relative to the YAML, so this is the repo root), with `include_paths: [src, recipes/<model>]`. The command sets `PYTHONPATH="$CODE_SOURCE_PATH/src"` and runs `python recipes/<model>/train.py`. **Verified in M0 (spike a):** the snapshot extracts to `/databricks/code_source/nemotron-on-databricks/` and `import nemotron_dbx` works.
  - Every workload must set `environment.version: databricks_ai_v6`. Without it, the default image has **no torch**.
- **Registration:** also an AIR workload with the same snapshot (see the decision log).
- **Dependencies:** `environment.dependencies` works but adds ~5 min of setup. Small pinned extras (e.g. `pycocotools`) are pip-installed at the start of `command` instead.
- **Serving:** each pyfunc is logged with `code_paths=["src/nemotron_dbx"]`, so endpoints never depend on the repo at runtime.

---

## 5. Shared conventions

### 5.1 Storage and naming

| Item | Convention |
|---|---|
| Volume root | `/Volumes/<catalog>/<schema>/<volume>/<recipe-name>`. On `fevm-am-rag-fevm`: `/Volumes/am_rag_fevm_catalog/nemotron/artifacts/<recipe-name>` |
| Training outputs | `<root>/<MLFLOW_RUN_ID>/{weights|merged|adapter}/` + `metrics.json` |
| Dataset cache | `<root>/data/<dataset>@<revision>/`, so it can be reused across runs (stable ID, not the run ID) |
| Wheel cache | `/Volumes/<catalog>/<schema>/<volume>/wheels/<package>/<torch-version>/*.whl`: CUDA extensions are built once and reused (§11.2) |
| Node-local scratch / HF cache | `/tmp` is the node's local disk (3.5 TB on 1×H100 and 1×A10 nodes). `/local_disk0` does **not** exist by default. Use `HF_HOME=/tmp/hf`. |
| UC model | `${catalog}.${schema}.nemotron_<model>_ft`, e.g. `am_rag_fevm_catalog.nemotron.nemotron_page_elements_v3_ft` |
| Endpoint | `nemotron-<model>-ft` |
| MLflow | Log into the AIR-provided `MLFLOW_RUN_ID`. Tag `merged_model_path` / `weights_path` so registration can resolve the artifacts. |

### 5.2 Registration (stage 2)

- Runs as an **AIR GPU workload** (`register.yaml`), so the packaging environment matches the training environment, particularly for CUDA extensions.
- Steps:
  1. Resolve the artifacts from the training-run tag.
  2. Run a **local smoke test** of the exact packaged artifact (pyfunc `predict` or a local vLLM server).
  3. `mlflow.pyfunc.log_model(...)`.
  4. `mlflow.register_model(...)`. Classic pyfuncs use pinned pip requirements and **no `env_pack`** (OQ-17). Custom LLM Serving models use `env_pack="databricks_model_serving"`.
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

The model is served through **Custom LLM Serving**, the cookbook pattern: a pyfunc placeholder whose `metadata.entrypoint` starts `vllm.entrypoints.openai.api_server` on `0.0.0.0:8080`, with `task: llm/v1/chat`. According to the Custom LLM Serving docs, vision-language models use the same `llm/v1/chat` task. The endpoint uses the OpenAI-compatible chat format with `image_url` (base64 data URL) content parts.

- On AIR, vLLM 0.24 (ai_v6) serves Nano VL with image requests (§11.1, spike c).
- Still to verify: the same request through a deployed endpoint (OQ-5).
- The entrypoint must set `VLLM_USE_FLASHINFER_SAMPLER=0`.

---

## 6. Per-model recipes

### 6.1 `recipes/page-elements-v3`

| | |
|---|---|
| **Dataset** | **DocLayNet** (`docling-project/DocLayNet-v1.2`, CDLA-Permissive-1.0; 80,863 pages, 11 classes, 6 doc categories incl. *financial reports*). Default subset: `financial_reports` + `manuals`, ~2k train / 300 val / 300 test pages, from the official splits. |
| **Label mapping** (11 → 6) | `Table→table`; `Title, Section-header→title`; `Page-header, Page-footer→header_footer`; `Text, List-item, Caption, Footnote, Formula→text`; `Picture→ambiguous` (chart vs infographic vs photo). **v1 default: drop training pages that contain `Picture`**, which avoids false negatives on chart/infographic. Alternative: ignore-region masking (OQ-2). |
| **Fine-tune method** | Full fine-tune of the 54 M YOLOX model with the same architecture and class count, loading `weights.pth` strictly. The HF `YOLOXHead.forward` has **no training branch**, so we vendor `get_losses` / SimOTA assignment / `IOUloss` from Megvii YOLOX (Apache-2.0), adapted to the HF module names. The HF wrapper's letterbox preprocessing (`resize_pad` to 1024²) is reused for train and inference. Augmentation: scale jitter (0.8–1.2), color/contrast jitter, slight JPEG noise. **No mosaic/flip**, because document layout semantics are orientation-sensitive. |
| **Hyper-params (default)** | AdamW, lr 1e-4 (backbone 1e-5), cosine schedule, 10 epochs, batch 8, bf16 AMP, EMA 0.9998 |
| **Compute** | `GPU_1xA10`, batch 4 × grad accum 2, training capped at 45 min (OQ-16). Override to `GPU_1xH100` with batch 8. |
| **Eval** | COCO mAP@[.5:.95] and mAP@.5, per class and overall (`pycocotools`), on the test split, **base vs fine-tuned**. Use the HF post-processing (`postprocess_preds_page_element`, per-class thresholds, WBF) so the numbers reflect what serving returns. |
| **Artifacts** | `<root>/<run>/weights/weights.pth` + `config.json` (thresholds) + `metrics.json` |
| **Serving** | pyfunc per §5.3; labels `table, chart, infographic, title, header_footer, text` |

### 6.2 `recipes/table-structure-v1`

| | |
|---|---|
| **Dataset** | **PubTables-1M structure split** (`bsmock/pubtables-1m`, CDLA-Permissive-2.0, PASCAL-VOC XML). Downloads are tar.gz only, so `prepare_data.py` downloads only the shards it needs: *Val* annotations and images, split 80/20 into our train/val, and a subset of *Test* as the held-out test set. The data is cached in the Volume. Default size: ~3k train / 300 val / 300 test table crops. |
| **Label mapping** | `table row→row`, `table column→column`. **`cell`** is derived as row×column intersections, with `table spanning cell` boxes replacing the intersections they cover (merged cells). The mapping must be validated visually against base-model predictions on a sample, to confirm that the NVIDIA "cell" semantics match (OQ-3). Other PubTables classes are dropped. |
| **Fine-tune / compute** | Same YOLOX training stack as §6.1 (vendored loss). Input: table crop letterboxed to 1024². Same compute. |
| **Eval** | mAP per class (cell/row/column), scoring up to 1,000 detections per table. *Deferred to a follow-up:* a downstream metric: run `nemotron-ocr-v1` (base) + the HF `table_struct_pp` post-processing to rebuild the table as markdown, then score TEDS / exact-cell accuracy against the PubTables ground-truth cell text on 100 tables. |
| **Serving** | pyfunc per §5.3; labels `cell, row, column`. The optional `return_markdown=true` param requires OCR boxes in the request (see the HF `Demo.ipynb` flow). |

### 6.3 `recipes/graphic-elements-v1`

| | |
|---|---|
| **Dataset (default)** | **Synthetic charts**, generated by `prepare_data.py` with matplotlib: bar, line, scatter, pie, stacked, and horizontal variants, with random fonts, sizes, palettes, and locales (including French number formats such as `1 234,56 €`). Exact ground-truth boxes for all 10 classes come from `Artist.get_window_extent()`. This makes the data fully reproducible and license-clean, with no download. Default: 3k train / 300 val, plus a **real-chart test set** (OQ-4). |
| **Alternative dataset** | ICPR 2022 CHART-Infographics PMC set (used by NVIDIA for v1). Its license and redistribution terms must be checked before use (OQ-4). |
| **Classes** | `chart_title, x_title, y_title, xlabel, ylabel, other, legend_label, legend_title, mark_label, value_label` (the `Exp.labels` order). **Pie charts are not generated in v1:** NVIDIA's pie data (DeepRule) was auto-labelled, and the class used for wedge labels isn't documented. |
| **Fine-tune / compute / serving** | Same as §6.1. Input is the chart crop. |
| **Eval** | mAP per class on (a) the synthetic test set and (b) the real-chart test set. Report both, so improvement on synthetic data that hurts real charts is visible. |

### 6.4 `recipes/ocr-v1`

#### 6.4.1 What fine-tuning means here

The model has three jointly trained components: a detector (45 M), a recognizer (4.9 M), and a relational model (2.3 M). The recognizer is a conv + 3-layer Transformer encoder emitting **32 positions × 858 tokens** (0 = blank, 1 = EOS, 2 = rare, 3.. = charset). It reads **rectified feature maps from the detector** (`det_feature_3`), not raw crops: `QuadRectify(8, 32)` builds a sampling grid from each quad (vertex order TL, TR, BR, BL), and the custom CUDA `IndirectGridSample` op samples the features along it.

**Correction (2026-11-11), from reading `RecognitionTargetEncoder`:** the default inference decodes **per position** (argmax, skip blanks, stop at EOS, no duplicate merging). CTC beam decoding is only used with `combine_duplicates` or a language model, and the training targets are `text + EOS` with a mask. So fine-tuning uses **masked per-position cross-entropy** over `[c_1..c_L, EOS]`, not CTC.

**v1 scope: recognizer-only fine-tuning.**

- Freeze the detector and relational models.
- For each training image: preprocess exactly like `NemotronOCR._process_tensor` (pad to a square at the bottom-right with white, resize to 1024), run the frozen detector, rectify the **ground-truth quads** (jittered by ~6 % of the word height to mimic detector noise), run the recognizer, and compute the masked per-position CE against `text + EOS`. Words with out-of-charset characters or ≥ 32 characters are skipped for training. No gradient flows through the custom CUDA ops. BatchNorm statistics are frozen.
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

- Python 3.12 and a **C++/CUDA extension** (`nemotron_ocr_cpp`, built with hatchling and needing GCC with C++17, CUDA headers, and OpenMP). Only core CUDA headers are needed (`cuda_runtime`, thrust, cooperative_groups; `trove` is vendored).
- **No custom image** (spike d). In `databricks_ai_v6` (nvcc 13.0, gcc 13.3), `pip wheel --no-build-isolation` with `TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0+PTX"` builds a 36 MB wheel in **254 s on 1×A10**. `nemotron_dbx.ocr.build.ensure_installed` caches it under `<volume>/wheels/nemotron-ocr/<rev8>-<torch tag>/` and installs it with `--no-deps` (plus `hatchling` and `shapely` if missing).
- For serving, the cached wheel is logged via `code_paths` and installed with the pip requirement `code/<wheel>`. torch and torchvision are pinned **with their CUDA tag** (`torch==2.11.0+cu130` + `--extra-index-url https://download.pytorch.org/whl/cu130`), so the extension sees the same torch ABI and CUDA runtime as at build time. `env_pack` is not used (OQ-17).
- **Output quirk:** in the pipeline output, `upper` > `lower` (the y-axis is flipped). The pyfunc normalizes boxes to `y_min ≤ y_max`.

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

NeMo AutoModel has no Nano v2 VL recipe, and Megatron-Bridge has removed support. **Decision: Hugging Face `transformers` + `peft` LoRA with a minimal custom training loop**, loading the model with `trust_remote_code=True`. **Validated in M0 (spike e).**

- **Environment:** stock `databricks_ai_v6` (torch 2.11+cu130, **transformers 5.8.1**, peft 0.19.1). On top of it:
  - `mamba-ssm==2.3.2.post1`, installed with `MAMBA_SKIP_CUDA_BUILD=TRUE --no-deps` (Triton ops only).
  - `causal-conv1d==1.6.2.post1`, CUDA build, wheel cached on the Volume.
  - `einops`, `timm`, `open_clip_torch`, `ftfy`, `regex`.
  - The compatibility patches in §11.3.
- **No custom image.** The model card's `transformers<4.54` pin is **not** used. The model also loads under 4.53.3, but vLLM 0.24 requires transformers ≥ 5.5.3, so a 4.53 environment could not run eval or serving.
- Fallback (documented only): the frozen Megatron-Bridge 0.4.2 container.

#### 6.5.2 Training setup

| | |
|---|---|
| **Dataset** | **CORD-v2** receipts → JSON key-information extraction (menu items, prices, totals). This is the same task as the Super VL cookbook recipe, so the two models can be compared. The prompt uses the system message `/no_think` and a fixed extraction instruction, with the target being canonical JSON of `gt_parse`. |
| **LoRA targets** | LLM attention `q,k,v,o`, MLP `up,down`, **and Mamba `in_proj,out_proj`**: 90.8 M trainable params at r=32. Rank 32, α 64, dropout 0.05. The **vision encoder (RADIO) and projector (`mlp1`) are frozen** (`exclude_modules`). Attention uses `attn_implementation="eager"`; only 6 of the 62 layers are attention layers. |
| **Precision / memory** | bf16 weights (~25 GB), non-reentrant gradient checkpointing, max seq len 4096, ≤ 12 tiles of 512² per image. **Measured peak: 34.6 GB** at ≤ 3.5k tokens / 13 tiles, batch 1. |
| **Compute** | **`GPU_1xH100`** by default. Measured ~1.1–2 s/step at batch 1; the first step takes ~78 s for Triton JIT warm-up. `GPU_8xH100` + DDP is an optional speed-up. |
| **Hyper-params** | lr 1e-4, cosine, 1–2 epochs over 800 train receipts, global batch 16 |
| **Eval** | **Field-level F1 and normalized tree-edit-distance accuracy** (the Donut/CORD standard) on CORD-v2 test (100), plus JSON validity rate. Base vs fine-tuned are both run with **vLLM batch inference**, so the eval measures the serving path. |
| **Artifacts** | `adapter/`, plus a **merged HF checkpoint** in `merged/`. The merged directory must contain all remote-code files (`modeling*.py`, `configuration*.py`, `processing*.py`, `image_processing.py`, chat templates, `evs.py`, `video_io.py`), copied from the pinned base revision. |

#### 6.5.3 Serving

- **Stage 2:** start `VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <merged> --trust-remote-code --dtype bfloat16 --video-pruning-rate 0 --max-model-len 16384` locally on an AIR GPU task, send one image request as a smoke test, then register as a Custom LLM Serving model (§5.4).
  - vLLM **0.24.0** (ai_v6) resolves `NemotronH_Nano_VL_V2` natively (spike c).
  - Startup takes 230 s including the 25 GB download. A CORD receipt takes 1.1–2.7 s (1.8k–3.4k prompt tokens) on 1×H100.
- **Workload:** 12.6 B bf16 is ~25 GB of weights. On AWS, `GPU_LARGE` is an **L40S 48 GB** (per the Custom LLM Serving docs), which should fit with a reduced `--max-model-len`. `GPU_XLARGE` (H100 80 GB, restricted availability) is the preferred choice if it's enabled. To verify on the workspace (OQ-5).
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
| ~~OQ-1b~~ | ~~Can AIR `code_source.root_path` point at the repo root?~~ | **Resolved (spike a):** yes, `snapshot.root_path: ../..` + `include_paths` |
| OQ-2 | DocLayNet `Picture` is ambiguous for chart vs infographic. Drop those pages, add ignore regions, or relabel a subset? | Drop pages |
| OQ-3 | Do PubTables-derived "cells" match NVIDIA's `cell` semantics (empty cells, spanning cells)? | **Mostly (M2, 2026-11-12).** NVIDIA fine-tuned on Azure Document Intelligence cells (grid cells, merged cells as one box). That's the same model as our mapping (row × column intersections; `table spanning cell` and `table projected row header` replace the cells they cover). Overlays of GT vs base cells agree on columns and ordinary cells. There are three convention differences: (1) section-header rows are full-width merged cells in PubTables, where the base model's cells hug the text; (2) PubTables sometimes splits multi-line cells into sub-rows; (3) PubTables keeps every empty grid cell, where the base model drops or merges empty areas. Part of the fine-tuning gain is therefore adaptation to PubTables conventions (as with DocLayNet in M1). |
| OQ-4 | Real-chart test set for graphic-elements: which license-clean source? | **Open.** v1 reports synthetic-test mAP only. As a sanity check, each run predicts on NVIDIA's own example chart (base vs fine-tuned overlays and per-class counts). The PMC/ICPR chart set's license still needs checking. |
| OQ-5 | Does Custom LLM Serving accept **multimodal** (`image_url`) chat requests end-to-end, and which GPU workload types are enabled on the workspace for a 12 B model? | **Blocked on the platform (M4, 2026-11-12).** Registration works: vLLM smoke test with the exact entrypoint, offline, from the artifacts folder. But every `env_pack` (express) deployment on `fevm-am-rag-fevm` gets a GPU, logs `Downloads completed for both model and environment artifacts`, then prints **no entrypoint output at all** until the update times out. This happened for Nano VL v1 (`GPU_LARGE`, after a 74 min L40S capacity wait), and for a **cookbook-identical Qwen2.5-0.5B** (`recipes/spikes/custom_llm_tiny`, `GPU_MEDIUM`). Same symptom as OQ-17. Not the ARM/x86 mismatch of ES-2316752: the AIR A10 registration node is x86_64 (`recipes/spikes/arch`). Engineering points to Express Deployment V2 (AICMS-658, no date) as the fix. Next: escalate in #model-serving with these endpoints, or try another workspace. |
| OQ-6 | Exact NIM response schemas for page-elements, table-structure, graphic-elements, and OCR, to mirror field names | Mirror best-effort, document differences |
| OQ-7 | OCR detector/relational fine-tuning needs loss code not shipped in the HF repo | v1 = recognizer only |
| ~~OQ-8~~ | ~~OCR charset lacks `€`/`£`~~ | **Resolved:** not needed for now; report the out-of-charset rate only |
| ~~OQ-9~~ | ~~Will the C++/CUDA OCR extension built on AIR load in the Model Serving GPU image?~~ | **Resolved (M3):** the wheel ships via `code_paths` + `code/<wheel>` pip requirement, with `torch==2.11.0+cu130` from the PyTorch cu130 index. Endpoint READY on T4 and serving correct text. Earlier notes: builds and runs on AIR (spike d). The Model Serving GPU host has driver 580 (CUDA 13 capable), and a pip-built image runs torch 2.11 on T4 (M1). Since `env_pack` is out (OQ-17), the wheel must be installed via `pip_requirements`, pointing at the wheel logged as a model artifact. Verify in M3. |
| ~~OQ-10~~ | ~~Nano VL LoRA through HF remote code + PEFT~~ | **Resolved (spike e):** 10 steps OK on 1×H100 with transformers 5.8.1 + patches (§11.3) |
| ~~OQ-11~~ | ~~Include Mamba projections in LoRA targets?~~ | **Default changed:** include `in_proj,out_proj` (validated). Attention+MLP-only stays an ablation. |
| ~~OQ-12~~ | ~~AIR accelerator types / previews on `fevm-am-rag-fevm`~~ | **Resolved:** `GPU_1xA10` and `GPU_1xH100` run (`GPU_8xH100`/`GPU_8xB300` not tried). The Artifact Registry host is configured. The GPU pools API is *not* enabled (not needed). |
| ~~OQ-14~~ | ~~Workspace TTL and storage~~ | **Resolved:** expires **2026-11-28** (extension may be needed for M4/M5); storage is `am_rag_fevm_catalog.nemotron.artifacts` |
| OQ-15 | `databricks air run --watch` log streaming can lag far behind the run (spike e finished at 17:38 UTC, the stream ended at 18:17). | Don't rely on `--watch` for completion; check `databricks air get <run-id>` or the job-run page |
| OQ-17 | **`env_pack="databricks_model_serving"` breaks classic pyfunc endpoints.** On T4 the server restarted every ~3 min right after "Downloads completed", with no traceback; on A10G the update was aborted ("endpoint update timed out"). The same model registered with pip requirements was READY on T4 in < 25 min. | Don't use `env_pack` for classic pyfuncs. If needed, raise it with the Model Serving team. |
| OQ-18 | The Databricks CLI now prints a banner ("Databricks skills are not installed…") on stdout before `-o json` output, which breaks naive JSON parsing. | Tools parse from the first `{`. |
| OQ-16 | **GPU capacity waits are invisible.** A `GPU_1xH100` run stayed task-`PENDING` ("In run") for 90 min, then `TIMEDOUT`, with no logs, while `air get` reported `RUNNING`. Earlier H100 runs started in seconds. | `tools/air_wait.sh` prints the task life-cycle state. Default small models to A10, and cap `timeout_minutes` on smoke runs. |
| OQ-13 | Customer data: will the cookbook later be pointed at customer documents (in their own UC volume)? If so, each `prepare_data.py` needs a "bring your own COCO/JSONL" path. | Add a `dataset_source: hf|volume` parameter in v1 |

---

## 10. Milestones

| # | Deliverable | Notes |
|---|---|---|
| M0 ✅ | Spikes on `fevm-am-rag-fevm`, under `recipes/spikes/`: (a) capabilities and storage ✅; (b) repo-root `root_path` ✅; (c) Nano VL vLLM + image on AIR ✅; (d) OCR extension build + inference on AIR ✅; (e) Nano VL PEFT LoRA, 10 steps ✅. See §11. | Carried over to M1/M3/M4: deploy-side halves of OQ-5 and OQ-9 |
| M1 ✅ | `page-elements-v3` end-to-end (train → register → serve). | Full run on A10 took 22 min: **test mAP 0.211 → 0.635** (300 pages), val 0.183 → 0.599. Registered v2 (pip env). Endpoint `nemotron-page-elements-v3-ft` (GPU_SMALL T4) is READY: cold start ~2 min, warm 3 pages ≈ 5 s. |
| M2 ✅ | `table-structure-v1` + `graphic-elements-v1` | Shared `detection/recipe.py` runner, `pubtables.py` and `charts.py` data. **table-structure:** test mAP 0.488 → 0.861 (A10, 33 min); endpoint `nemotron-table-structure-v1-ft` (T4) READY, 2 tables ≈ 4 s warm. **graphic-elements:** synthetic test mAP 0.399 → 0.837 (A10, 28 min). The first generator (no captions) mislabelled real figure captions as axis text; v2 adds captions. Endpoint deploying. |
| M3 ✅ | `ocr-v1` (recognizer fine-tune, serving with the compiled extension) | A10, 8.6 min: **test GT-quad CER 4.95 % → 1.63 %**, word acc 88.5 → 95.3 %, end-to-end BoW F1 0.819 → 0.848. Endpoint `nemotron-ocr-v1-ft` (GPU_SMALL) READY in ~38 min (container build). No custom image. |
| M4 🟡 | `nano-12b-v2-vl` LoRA → merged → serving | Trained: CORD test field F1 0.39 → 0.86, TED 0.45 → 0.93 (1×H100, 31 min) after patch 11. Serving blocked: Custom LLM Serving (OQ-5); the classic-pyfunc fallback (`VLLMChatModel`, v4) built but timed out waiting for `GPU_LARGE` capacity. |
| M5 | Top-level README, metrics table, optional pipeline demo (§7) | |

---

## 11. M0 findings (2026-11-10, `fevm-am-rag-fevm`)

All spikes live in `recipes/spikes/` and are launched from the repo root with `databricks air run -f recipes/spikes/<spike>/workload.yaml -p am-rag-fevm`.

### 11.1 Results

| Spike | Compute | Result | Key facts |
|---|---|---|---|
| (a) `hello` | 1×A10G, 1×H100 | ✅ | Workloads start in ~15–35 s. Repo-root snapshot works. Volume write works. **The default environment has no torch, so `environment.version: databricks_ai_v6` is required.** ai_v6: Python 3.12.3, torch 2.11.0+cu130, transformers 5.8.1, vLLM 0.24.0, peft 0.19.1, MLflow 3.13.0, flash-attn present. Driver 595.91 / CUDA 13.2. `/tmp` is a 3.5 TB local disk. |
| (c) `nano_vl_vllm` | 1×H100 | ✅ (3rd attempt) | vLLM 0.24 resolves `NemotronH_Nano_VL_V2` natively. **Needs `VLLM_USE_FLASHINFER_SAMPLER=0`**: FlashInfer's sampler JIT fails on the missing `curand.h`. Startup 230 s incl. download. CORD receipts take 1.1–2.7 s each. The base model reads receipts well but **invents its own JSON schema** (`menu_items`/`quantity` instead of CORD's `nm`/`cnt`/`total_price`), which gives clear headroom for fine-tuning. |
| (d) `ocr_build` | 1×A10G | ✅ | Extension wheel built in 254 s (nvcc 13.0, gcc 13.3, archs 7.5/8.6/8.9/9.0+PTX), 36 MB. Saved to `…/artifacts/spikes/ocr_build/<run>/`. Model load 2.1 s. First call 9.8 s (warm-up), then **~70–90 ms/page**. Base errors on CORD: `TIICKET` (conf 0.20), `60,000` vs `60.000`, `00.000`. On the NVIDIA example, `£14,389.43` → `f14,389.43` (`£` is not in the charset). Box `upper`/`lower` are flipped. |
| (e) `nano_vl_lora` | 1×H100 | ✅ (10th attempt) | 10 LoRA steps with **transformers 5.8.1**. Fused Mamba path active. 90.8 M trainable params. **Peak 34.6 GB.** ~1.1–2 s/step (first step 78 s, JIT). Loss 0.98 → 0.69 (noisy, batch 1). The first run includes a **642 s `causal-conv1d` compile**; later runs use the cached wheel. |

### 11.2 Build-once CUDA extensions (wheel cache)

- Both the OCR extension and `causal-conv1d` compile in ai_v6 (nvcc 13.0 is present; `curand.h` is not).
- The shared helper `nemotron_dbx.common.wheels` will:
  1. look for `<volume>/wheels/<pkg>/<torch-version>/*.whl`;
  2. otherwise build with `pip wheel --no-deps --no-build-isolation` and copy the wheel there;
  3. `pip install --no-deps` it.
- `causal-conv1d`'s `setup.py` **ignores `TORCH_CUDA_ARCH_LIST`**. It always compiles ~9 archs (sm_75…sm_121), which takes ~11 min and produces a 205 MB wheel. Acceptable as a one-off. To make it faster, patch `cc_flag` to sm_90 only.
- Upstream prebuilt wheels exist only up to `cu13torch2.10`, and ai_v6 has torch 2.11, so they're not used.

### 11.3 Nano VL remote-code compatibility patches (`nemotron_dbx/vlm/compat.py`, `vlm/env.py`)

| # | Symptom | Patch |
|---|---|---|
| 1 | `ImportError: mamba-ssm is required` (the remote code imports `mamba_ssm.ops.triton.*` unconditionally) | `MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-deps mamba-ssm==2.3.2.post1` |
| 2 | `No module named 'selective_scan_cuda'` (`mamba_ssm/__init__.py` eagerly imports the Mamba-1 CUDA extension) | Replace `mamba_ssm/__init__.py` with a version-only stub (Nano VL only uses the Mamba-2 Triton ops) |
| 3 | ~17 GiB allocation per Mamba layer → OOM in backward (torch fallback path) | Install the `causal-conv1d==1.6.2.post1` CUDA wheel, so `is_fast_path_available` is true |
| 4 | `open_clip` import fails | Also install `ftfy` and `regex` (they were skipped by `--no-deps`) |
| 5 | `does not support Flash Attention 2` (the remote code declares the transformers-4 attribute `_supports_flash_attn_2`) | `attn_implementation="eager"` |
| 6 | `no attribute 'all_tied_weights_keys'` (transformers 5 requires `post_init()`) | Wrap `NemotronH_Nano_VL_V2.__init__` so it calls `post_init()` |
| 7 | CUDA device-side assert in RADIO `_extract_final` (the `summary_idxs` buffer is not in the checkpoint, and transformers 5 re-initializes it with garbage) | After load, recompute `summary_idxs = [i for i, t in enumerate(config.args["teachers"]) if t.get("use_summary", True)]` → `[0, 1, 2]` |
| 8 | `NemotronHCausalLMOutput has no attribute 'past_key_values'` | Set `NemotronHCausalLMOutput.past_key_values = None` (class attribute) |
| 9 | `a view of a leaf Variable that requires grad is being used in an in-place operation` | Remove every `_require_grads_hook` (PEFT/transformers add them for checkpointing). Non-reentrant checkpointing doesn't need them. |
| 10 | `torch.distributed.get_rank()` inside `forward` | Always launch with `torchrun` (process group initialized, even on 1 GPU) |
| 11 | **Silent:** the model learns to ignore the image (train loss plateaus at ~0.5, and every fine-tuned answer collapses to a one-item `"menu": {…}`). The remote image processor's `_process_image` hook (`T.ToTensor()` → [0, 1]) is never called: transformers 5 renamed the hook to `process_image`, which returns uint8 [0, 255]. The tiles are then mean/std-normalized at 255× scale. vLLM preprocesses on its own and is unaffected, so base eval looked fine. Found in M4. | Wrap the instance's `process_image` to divide uint8 output by 255 (`compat.fix_image_rescale`). `data.check_pixel_values` fails fast if \|x\| > 10. The M0 spike (e) also trained on unnormalized pixels. |
| 12 | `FileNotFoundError: …/transformers_modules/<local>/…/configuration_nemotron_h.py` when loading from a **local directory**: for local paths, transformers 5.8.1 copies only the *direct* relative imports of `modeling.py` into its module cache, then follows transitive ones. | Snapshot into the HF cache and load by **repo ID + pinned revision**. vLLM and the merge use the snapshot path. |
| 13 | `config.json` points the LLM and vision remote code at **other repos without a revision** (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base--…`, `nvidia/C-RADIOv2-H--…`), i.e. their `main` branch. | `compat.pin_remote_code` injects `code_revision` for those repos (pins in `common/hf.py`) |

- All patches must be **unit-tested** on CPU where possible (patches 6–9 with a tiny config) and **version-guarded**, so they turn into no-ops if upstream fixes land.
- Pin the HF revision of `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` (`ca9543b126e8bf3176916d3d305ccc415f89fd4d` at the time of the spikes) and of `nvidia/C-RADIOv2-H` (`0d8f4c18c877166eda07ddae1386bcad256b7a6a`).

### 11.4 Not yet checked

- **Model Serving side of OQ-5 and OQ-9:** deploying a real endpoint (Nano VL Custom LLM Serving; OCR pyfunc with the custom wheel).
- **Custom image push:** the Artifact Registry host resolves, but no image was pushed (the local Docker daemon was not running). This is no longer on the critical path.
- **`GPU_8xH100`** availability.
- **Registration jobs:** a DAB job task on AIR GPU compute (`compute.hardware_accelerator`, as in the cookbook Qwen recipe) has not been run yet.

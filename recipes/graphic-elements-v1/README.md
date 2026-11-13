# Fine-tune and serve Nemotron Graphic Elements v1

This recipe fine-tunes [`nvidia/nemotron-graphic-elements-v1`](https://huggingface.co/nvidia/nemotron-graphic-elements-v1) on **synthetic charts**. That model is a 54 M-parameter YOLOX-L detector for the text elements of a chart image. The recipe compares the base and fine-tuned models, registers the fine-tuned model in Unity Catalog, and serves it on a GPU Model Serving endpoint.

The input is a **chart crop**, not a page. In a pipeline, crop charts with [page-elements-v3](../page-elements-v3/) first.

**Classes**, in NVIDIA's order: `chart_title`, `x_title`, `y_title`, `xlabel` / `ylabel` (tick labels), `other`, `legend_label`, `legend_title`, `mark_label` (labels attached to markers), `value_label` (numbers attached to bars or points).

- **Training:** the same stack as page-elements-v3 (`nemotron_dbx.yolox_train`). 1× A10, batch 4 × gradient accumulation 2. Stops after 8 epochs or 40 minutes of training.
- **Data** (`nemotron_dbx.detection.charts`): charts rendered with matplotlib, with exact boxes from `get_window_extent()`. No download, fully reproducible from a seed, license-clean.
  - Kinds: bar, horizontal bar, grouped bar, stacked bar, line, scatter.
  - Variety: 8 matplotlib styles, 3 font families, random sizes, DPI, rotated tick labels, legends with optional titles, value labels, marker labels, source notes.
  - Banking-flavoured vocabulary and **French number formats** (`1 234,5`, `12,5 %`) in about a third of the charts.
  - Defaults: 3,000 train, 300 validation and 300 test charts, generated from disjoint seeds.
  - **Pie charts are left out:** NVIDIA's pie data was auto-labelled, and which class its wedge labels belong to isn't documented.
- **Evaluation:** COCO mAP@[.5:.95] and mAP@.5, overall and per class, on the synthetic test set, through NVIDIA's inference wrapper (class-agnostic NMS).
  - **Caveat:** a synthetic test set measures adaptation to synthetic charts. There's no labelled, license-clean real-chart test set yet (SPEC OQ-4). As a sanity check, the run also predicts on NVIDIA's own example chart with the base and fine-tuned models and saves both overlays.

## Before you run

- Set `parameters.output_root` in `workload.yaml` to an existing writable Volume directory.
- Set `parameters.registered_model_name` in `register.yaml`. You need `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on its schema.

All commands run from the **repository root**.

## Run

```bash
# 1. Train + evaluate
databricks air run -f recipes/graphic-elements-v1/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # last line: TRAINING_RESULT={... "mlflow_run_id": ...}

# 2. Register in Unity Catalog (smoke-tests the packaged pyfunc first)
databricks air run -f recipes/graphic-elements-v1/register.yaml --override parameters.source_training_run_id=<MLFLOW_RUN_ID>
tools/air_wait.sh <JOB_RUN_ID>          # REGISTRATION_RESULT={... "model_version": ...}

# 3. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.nemotron_graphic_elements_v1_ft --model-version <N> \
  --endpoint-name nemotron-graphic-elements-v1-ft --workload-type GPU_SMALL --profile <PROFILE>
tools/endpoint_wait.sh nemotron-graphic-elements-v1-ft <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-graphic-elements-v1-ft --profile <PROFILE> chart.png
```

## Results

Observed on 2026-11-12 (`fevm-am-rag-fevm`, 1× A10; HF revision `4a76546b`; generator `synthetic-charts-v2`, seed 0):

| Synthetic test (300 charts) | Base | Fine-tuned |
|---|---|---|
| mAP@[.5:.95] | 0.399 | **0.837** |
| mAP@.5 | — | 0.993 |
| validation mAP (300 charts) | 0.394 | 0.847 |

Per-class AP on the first run (same data without captions) went up for every class, for example chart_title 0.67 → 0.92, xlabel 0.37 → 0.82, value_label 0.42 → 0.77, other 0.16 → 0.80.

- **Time:** 28 min end to end: 1.5 min to render 3,600 charts (16 processes), 25 min of training (8 epochs), plus the evals.
- **Real-chart check (NVIDIA's example chart):** the fine-tuned model finds the same elements as the base: 1 x title, 1 y title, 9 + 9 tick labels, legend title and 4 legend labels. The figure caption is labelled `other`, one box per line. There's one extra near-empty `other` box between the x title and the caption.
- **Lesson from the first run:** the first version of the generator had no captions. That run scored the same on synthetic data (0.411 → 0.836), but on the real chart it started labelling caption words as `x_title` and `xlabel`, because every text below the axis in the training data was axis text. Adding 1–3-line figure captions (labelled `other`) to ~40 % of the charts fixed it. Synthetic mAP would not have caught this: always check real charts (SPEC OQ-4).
- **Caveat:** these numbers measure adaptation to synthetic charts. A labelled real-chart test set is still missing (SPEC OQ-4).

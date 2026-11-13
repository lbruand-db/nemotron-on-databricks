# Fine-tune and serve Nemotron Table Structure v1

This recipe fine-tunes [`nvidia/nemotron-table-structure-v1`](https://huggingface.co/nvidia/nemotron-table-structure-v1) on [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m) table crops. That model is a 54 M-parameter YOLOX-L detector for the **cells, rows and columns** of a table image. The recipe compares the base and fine-tuned models on a held-out split, registers the fine-tuned model in Unity Catalog, and serves it on a GPU Model Serving endpoint.

The input is a **table crop**, not a page. In a pipeline, crop tables with [page-elements-v3](../page-elements-v3/) first.

- **Training:** the same stack as page-elements-v3 (`nemotron_dbx.yolox_train`): Megvii's SimOTA loss on NVIDIA's head, NVIDIA's letterbox to 1024², EMA weights, backbone at 0.1× the learning rate. 1× A10, batch 4 × gradient accumulation 2. Stops after 8 epochs or 40 minutes of training.
- **Data:** the PubTables-1M **structure** split (CDLA-Permissive-2.0). Only the *Val* and *Test* archives are downloaded (2.7 GB of images each).
  - *Val* tables are ordered by a hash of the file name: the first 3,000 are our train set, the next 300 our validation set. The first 300 hash-ordered *Test* tables are the held-out test set.
  - The subset is cached as parquet under `<output_root>/data/`, so later runs skip the download.
- **Label mapping** (`nemotron_dbx.detection.pubtables`):
  - `table row` → `row`, `table column` → `column`;
  - `cell` = row × column intersections, with merged cells (`table spanning cell`, and `table projected row header`, a full-width merged row) replacing the grid cells they cover;
  - `table` and `table column header` are dropped. NVIDIA's `border` and `header` output slots are marked "not used" in its config.
  - NVIDIA fine-tuned on Azure Document Intelligence cells, which are also grid cells with merged cells as one box. The training run saves overlays (ground-truth cells next to base-model cells) so the match can be checked visually (SPEC OQ-3).
- **Evaluation:** COCO mAP@[.5:.95] and mAP@.5, overall and per class, through NVIDIA's inference wrapper. Tables can have hundreds of cells, so up to 1,000 detections per image are scored.

## Before you run

- Set `parameters.output_root` in `workload.yaml` to an existing writable Volume directory.
- Set `parameters.registered_model_name` in `register.yaml`. You need `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on its schema.

All commands run from the **repository root**.

## Run

```bash
# 1. Train + evaluate
databricks air run -f recipes/table-structure-v1/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # last line: TRAINING_RESULT={... "mlflow_run_id": ...}

# 2. Register in Unity Catalog (smoke-tests the packaged pyfunc first)
databricks air run -f recipes/table-structure-v1/register.yaml --override parameters.source_training_run_id=<MLFLOW_RUN_ID>
tools/air_wait.sh <JOB_RUN_ID>          # REGISTRATION_RESULT={... "model_version": ...}

# 3. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.nemotron_table_structure_v1_ft --model-version <N> \
  --endpoint-name nemotron-table-structure-v1-ft --workload-type GPU_SMALL --profile <PROFILE>
tools/endpoint_wait.sh nemotron-table-structure-v1-ft <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-table-structure-v1-ft --profile <PROFILE> table.png
```

The endpoint returns the same shape as page-elements-v3: normalized boxes grouped by class (`cell`, `row`, `column`, plus the unused `border`/`header`, which stay empty after the class threshold).

## Results

Observed on 2026-11-12 (`fevm-am-rag-fevm`, 1× A10; HF revision `9350162f`, PubTables-1M revision `35b1c097`):

| Split | Metric | Base | Fine-tuned |
|---|---|---|---|
| test (300 tables, 19.8k cells) | mAP@[.5:.95] | 0.488 | **0.861** |
| | mAP@.5 | 0.856 | 0.972 |
| | AP cell | 0.491 | 0.802 |
| | AP row | 0.617 | 0.860 |
| | AP column | 0.356 | 0.922 |
| val (300 tables) | mAP@[.5:.95] | 0.506 | 0.862 |

- **Time:** 33 min end to end: 1.9 min of data preparation (two 2.7 GB archives, then cached), 27.8 min of training (8 epochs over 3,000 tables), plus the evals.
- **Interpreting the gains:** the overlays saved with each run (`<run>/overlays/`) show that the base model and PubTables mostly agree on cells. They differ in three conventions (SPEC OQ-3):
  1. section-header rows are full-width merged cells in PubTables, while the base model's cells hug the text;
  2. PubTables sometimes splits multi-line cells into sub-rows;
  3. PubTables keeps every empty grid cell, while the base model drops or merges empty areas.

  Part of the gain, especially on `column` (0.36 → 0.92) and at the strict IoU thresholds (mAP@.5 was already 0.86), is therefore adaptation to PubTables' box conventions, not only better structure recognition. Measure on your own tables before relying on it.
- **Serving:** `nemotron-table-structure-v1-ft` on `GPU_SMALL` (T4). The container built in ~16 min, then waited ~30 min for a GPU. Two test tables per request: ~4 s warm, ~170 s from scale-to-zero.
  - On the first test table, the endpoint returns exactly the ground truth (20 cells, 4 rows, 5 columns).
  - On the second, it returns **one row per gene** (8 rows), where PubTables splits each gene's two text lines into sub-rows (15 rows). That's the logical-row output a table-to-markdown step needs. mAP counts it as a miss, so the row AP understates usefulness on such tables.
- **Not done yet:** the downstream table-to-markdown metric (OCR + NVIDIA's `table_struct_pp`, scored with TEDS) is deferred (SPEC §6.2).

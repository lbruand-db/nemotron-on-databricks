# Fine-tune and serve Nemotron Page Elements v3

This recipe fine-tunes [`nvidia/nemotron-page-elements-v3`](https://huggingface.co/nvidia/nemotron-page-elements-v3)
on the DocLayNet *financial reports* and *manuals* pages. That model is a 54 M-parameter YOLOX-L page-layout detector with 6 classes: table, chart, title, infographic, text and header/footer.
The recipe then compares the base and fine-tuned models on a held-out split, registers the fine-tuned model in Unity Catalog, and serves it on a GPU Model Serving endpoint.

- **Training:** full fine-tune on 1× A10 (`GPU_1xA10`, batch 4 × gradient accumulation 2; H100 optional), with NVIDIA's own preprocessing (letterbox to 1024² in RGB 0–255).
  - The training head is Megvii's SimOTA / IoU / BCE loss, re-attached to NVIDIA's head modules (`nemotron_dbx.yolox_train`).
  - EMA weights are used. The backbone learns at 0.1× the learning rate.
  - The run stops after 8 epochs or 45 minutes of training, whichever comes first.
- **Data:** [`docling-project/DocLayNet-v1.2`](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2) (CDLA-Permissive-1.0), with the 11 DocLayNet classes mapped to the model's 6.
  - Pages with a `Picture` box are dropped, because a picture can be a chart, an infographic or a photo (SPEC OQ-2).
  - Defaults: 2,000 train, 200 validation and 300 test pages, taken from the official splits.
- **Evaluation:** COCO mAP@[.5:.95] and mAP@.5, overall and per class. It runs through NVIDIA's inference wrapper (NMS with `conf_thresh=0.01`), so the numbers reflect what serving returns.

**Interpreting the gains:** NVIDIA's training labels come from Azure Document Intelligence plus human annotators. DocLayNet uses its own conventions, for example paragraph- and list-item-level `text` boxes. Much of the early mAP gain, especially on `text`, likely comes from adapting to DocLayNet's annotation style rather than from generally better layout detection. Use your own labeled pages to measure what matters for your documents.

## Before you run

- Set `parameters.output_root` in `workload.yaml` to an existing writable Volume directory. The DocLayNet subset is cached under `<output_root>/data/`, and the weights are written under `<output_root>/<MLFLOW_RUN_ID>/`.
- Set `parameters.registered_model_name` in `register.yaml`. You need `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on its schema.
- You need permission to create Model Serving endpoints with GPU workloads.

All commands run from the **repository root**.

## Run

### 1. Train and evaluate (AIR)

```bash
databricks air run -f recipes/page-elements-v3/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # polls status, then prints the log tail
```

The last log line is `TRAINING_RESULT={...}`. It contains the MLflow run ID, the base and fine-tuned test mAP, and the weights path.

### 2. Register in Unity Catalog (AIR)

```bash
databricks air run -f recipes/page-elements-v3/register.yaml \
  --override parameters.source_training_run_id=<MLFLOW_RUN_ID>
tools/air_wait.sh <JOB_RUN_ID>
```

This step does four things:
1. Packages the NVIDIA code (at its pinned revision), the fine-tuned weights and `nemotron_dbx` as an MLflow pyfunc.
2. Runs a smoke test on a test page.
3. Registers a new model version with `env_pack="databricks_model_serving"`.
4. Prints `REGISTRATION_RESULT={...}` with the version.

### 3. Deploy and query (local)

```bash
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.<MODEL> --model-version <VERSION> \
  --endpoint-name nemotron-page-elements-v3-ft --workload-type GPU_SMALL --profile <PROFILE>
databricks serving-endpoints get nemotron-page-elements-v3-ft --profile <PROFILE>   # wait for READY
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-page-elements-v3-ft --profile <PROFILE> page.png
```

The endpoint takes base64 images (`dataframe_records: [{"image", "id"}]`) and returns normalized boxes grouped by class (SPEC §5.3):

```json
{"predictions": [{"id": "page.png", "bounding_boxes": {"table": [{"x_min": 0.08, "y_min": 0.31, "x_max": 0.92, "y_max": 0.55, "confidence": 0.94}], "title": [], "...": []}}]}
```

`params.conf_threshold >= 0` replaces NVIDIA's per-class thresholds with one global threshold.

## Results

The AIR MLflow run contains:
- the parameters;
- the training curves (`train/*`);
- the base and fine-tuned metrics on the same splits (`base/test/map`, `ft/test/map`, per-class `ap/*`, and the same for `val/`);
- `metrics.json`.

The registration run links to the training run through `nemotron.source_training_run_id`.

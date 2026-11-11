# Fine-tune and serve Nemotron OCR v1

This recipe fine-tunes the **recognizer** of [`nvidia/nemotron-ocr-v1`](https://huggingface.co/nvidia/nemotron-ocr-v1) on the [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) receipts (CC-BY-4.0; 800 train / 100 val / 100 test). The model has three parts: a RegNetY text detector, a Transformer recognizer and a relational (reading-order) model, 52 M parameters in total. The recipe compares the base and fine-tuned models, registers the result in Unity Catalog, and serves the full OCR pipeline on a GPU Model Serving endpoint.

**What is trained:** only the recognizer (4.9 M parameters). The detector and relational model stay frozen.
- Each receipt goes through NVIDIA's exact preprocessing: pad to a square, resize to 1024.
- The frozen detector's features are sampled along the **ground-truth word quads**, using NVIDIA's own `QuadRectify` and `IndirectGridSample` CUDA ops. The quads are jittered by ~6 % of the word height to mimic detector noise.
- The recognizer learns with masked per-position cross-entropy on `text + EOS`. This matches NVIDIA's decoder (32 parallel positions, argmax, stop at EOS); it is *not* CTC.
- Words with characters outside the 855-token charset are skipped for training and reported as `oov_char_rate`. Notably, the charset has no `€` or `£`.

**Evaluation**, base vs fine-tuned on the same splits:
- `gtquad/cer` and `gtquad/word_acc`: recognition on ground-truth quads, which isolates the recognizer.
- `e2e/bow_f1`: the full NVIDIA pipeline (detection + NMS + recognition + relational, `merge_level="word"`), scored as page-level bag-of-words F1 against the CORD words.

**Environment:** stock `databricks_ai_v6` on 1× A10, with no custom image.
- The `nemotron-ocr` C++/CUDA extension is compiled once (~4 min) and the wheel is cached at `<wheel_cache>/nemotron-ocr/<revision>-<torch>/`. Later runs reuse it.
- For serving, the same wheel ships inside the model (`code/<wheel>` pip requirement). torch is pinned with its CUDA tag (`2.11.0+cu130`) so the extension's ABI matches.

## Before you run

- Set `parameters.output_root` and `parameters.wheel_cache` in `workload.yaml` (and `wheel_cache` in `register.yaml`) to existing writable Volume directories.
- Set `parameters.registered_model_name` in `register.yaml`. You need `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on its schema.

All commands run from the **repository root**.

## Run

```bash
# 1. Train + evaluate
databricks air run -f recipes/ocr-v1/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # last line: TRAINING_RESULT={... "mlflow_run_id": ...}

# 2. Register in Unity Catalog (smoke-tests the packaged pipeline first)
databricks air run -f recipes/ocr-v1/register.yaml --override parameters.source_training_run_id=<MLFLOW_RUN_ID>
tools/air_wait.sh <JOB_RUN_ID>          # REGISTRATION_RESULT={... "model_version": ...}

# 3. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.nemotron_ocr_v1_ft --model-version <N> \
  --endpoint-name nemotron-ocr-v1-ft --workload-type GPU_SMALL --profile <PROFILE>
tools/endpoint_wait.sh nemotron-ocr-v1-ft <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-ocr-v1-ft --profile <PROFILE> receipt.png
```

The endpoint returns text regions in reading order, with normalized boxes:

```json
{"predictions": [{"id": "receipt.png", "regions": [{"text": "TOTAL 60,000", "confidence": 0.97,
  "x_min": 0.10, "y_min": 0.71, "x_max": 0.81, "y_max": 0.74}]}]}
```

`params.merge_level` is `word`, `sentence` or `paragraph` (the default). NVIDIA's pipeline reports `upper`/`lower` flipped; the serving model normalizes them to `y_min <= y_max`.

## Results

Observed on 2026-11-11 (`fevm-am-rag-fevm`, 1× A10; HF revision `8657d08d`, CORD-v2 revision `7f0115a4`):

| Split | Metric | Base | Fine-tuned |
|---|---|---|---|
| test (100 receipts, 2,356 words) | GT-quad CER | 4.95 % | **1.63 %** |
| | GT-quad word accuracy | 88.5 % | **95.3 %** |
| | end-to-end bag-of-words F1 | 0.819 | **0.848** |
| val (100 receipts) | GT-quad CER | 4.54 % | 0.84 % |

- **Training:** 8.6 min end to end with the cached wheel (6.6 min for 10 epochs over 19,367 words). The first build of the C++/CUDA extension adds ~4 min.
- **Remaining errors:** mostly in detection and word grouping, which are frozen (`CASH IDR` → `CASHIDR`, `( L` → `(L`). Fine-tuning them is out of scope (SPEC OQ-7).
- **Registration:** ~2 min.
- **Serving:** endpoint `GPU_SMALL` (T4) was ready in ~38 min, most of it building the container (PyTorch cu130 index + the extension wheel). Two receipts at `merge_level="word"` took ~6.5 s round trip.


The AIR MLflow run contains:
- the parameters;
- the training curves (`train/loss`, `train/token_acc`);
- `base/*` and `ft/*` metrics;
- `metrics.json`.

The fine-tuned checkpoints are under `<output_root>/<MLFLOW_RUN_ID>/checkpoints/`: the fine-tuned `recognizer.pth` plus the unchanged detector, relational model and charset.

# Fine-tune and serve Nemotron Nano 12B v2 VL

This recipe LoRA fine-tunes [`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) to extract structured JSON from receipt images, using [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC-BY-4.0; 800 train / 100 val / 100 test). It is the same task as the cookbook's Super VL recipe.

- **Model:** a 12.6 B hybrid Mamba-Transformer LLM with a C-RADIOv2-H vision encoder.
- **Evaluation:** base vs fine-tuned, both measured through vLLM.
- **Packaging:** the adapter is merged into the base weights, registered in Unity Catalog, and served on **Custom LLM Serving** (vLLM), queried with OpenAI chat requests.

**Task:** the system message is `/no_think`. The user message is the receipt image plus a fixed instruction (`nemotron_dbx.vlm.data.INSTRUCTION`) that names the CORD keys. The assistant answers with the receipt's `gt_parse` as JSON.

```json
{"menu": [{"nm": "ICED TEA", "cnt": "1", "price": "20,000"}], "total": {"total_price": "20,000", "cashprice": "50,000"}}
```

**Training:** Hugging Face `transformers` + `peft` with a small custom loop. NeMo AutoModel has no Nano v2 VL recipe.
- **LoRA:** r 32, α 64, dropout 0.05, on the LLM attention (`q,k,v,o`), MLP (`up,down`) and Mamba (`in_proj,out_proj`) projections: 90.8 M trainable parameters.
- **Frozen:** the vision encoder and the projector.
- **Settings:** bf16, non-reentrant gradient checkpointing, batch 1 per GPU with gradient accumulation to a global batch of 16, lr 1e-4 with cosine decay, 2 epochs.
- **Images:** up to 12 tiles of 512² plus a thumbnail per receipt, which comes to 2–3.7 k tokens per sample.

**Evaluation:** the same vLLM OpenAI server and chat format as the endpoint, on CORD test (100) and validation (100).
- `field_f1`: Donut's field-level F1 over (key path, value) pairs.
- `ted_acc`: Donut's normalized tree-edit-distance accuracy.
- `json_valid` and `exact_match` are reported alongside.

**Environment:** stock `databricks_ai_v6` (transformers 5.8.1, peft 0.19.1, vLLM 0.24) on 1× H100, with no custom image.
- The model card pins transformers < 4.54, but vLLM 0.24 needs ≥ 5.5.3. The recipe uses AIR's transformers 5.8.1, plus small, version-guarded patches to the model's remote code (`nemotron_dbx.vlm.compat`, SPEC §11.3).
- Training also installs `mamba-ssm` (Triton ops only) and the `causal-conv1d` CUDA extension. The extension is compiled once (~11 min) and cached as a wheel on the Volume.
- vLLM needs neither: it has its own Mamba kernels and Nano VL implementation.

**Merge:** `nemotron_dbx.vlm.merge` adds each LoRA delta directly into the base checkpoint's safetensors shards. Tensor names, dtypes, shards and remote-code files stay identical, so vLLM loads the merged model exactly like the base.

## Before you run

- Set `parameters.output_root` and `parameters.wheel_cache` in `workload.yaml` to existing writable Volume directories. Each run writes ~25 GB (`merged/`).
- Set `parameters.registered_model_name` in `register.yaml`. You need `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on its schema.
- **Custom LLM Serving** is in Public Preview. A workspace admin enables it on the Previews page.
- Serving uses `GPU_LARGE` (1× L40S 48 GB, AWS regions only). `GPU_XLARGE` (H100) needs enrollment through your account team.

All commands run from the **repository root**.

## Run

```bash
# 1. Train + evaluate (base and fine-tuned, both with vLLM) + merge
databricks air run -f recipes/nano-12b-v2-vl/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # last line: TRAINING_RESULT={... "mlflow_run_id": ...}

# 2. Register for Custom LLM Serving (local vLLM smoke test with the exact serving flags first)
databricks air run -f recipes/nano-12b-v2-vl/register.yaml --override parameters.source_training_run_id=<MLFLOW_RUN_ID>
tools/air_wait.sh <JOB_RUN_ID>          # REGISTRATION_RESULT={... "model_version": ...}

# 3. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.nemotron_nano_12b_v2_vl_ft --model-version <N> \
  --endpoint-name nemotron-nano-vl-ft --workload-type GPU_LARGE --profile <PROFILE>
tools/endpoint_wait.sh nemotron-nano-vl-ft <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-nano-vl-ft --chat receipt.png --profile <PROFILE>
```

The endpoint takes OpenAI chat requests. The image goes in as a base64 data URL (see `nemotron_dbx.lifecycle.query.build_chat_request`):

```json
{"messages": [{"role": "system", "content": "/no_think"},
              {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                                           {"type": "text", "text": "<INSTRUCTION>"}]}],
 "max_tokens": 1536, "temperature": 0}
```

The OpenAI client also works, with `base_url="<workspace-url>/serving-endpoints"` and `model="<endpoint-name>"`.

## Results

Observed on 2026-11-12 (`fevm-am-rag-fevm`, 1× H100; HF revision `ca9543b1`, CORD-v2 revision `7f0115a4`). Base and fine-tuned were both evaluated with vLLM, at temperature 0:

| Split | Metric | Base | Fine-tuned |
|---|---|---|---|
| test (100 receipts) | field F1 | 0.391 | **0.858** |
| | TED accuracy | 0.455 | **0.933** |
| | exact match | 0.00 | **0.41** |
| | valid JSON | 1.00 | 1.00 |
| val (100 receipts) | field F1 | 0.399 | 0.879 |
| | TED accuracy | 0.511 | 0.938 |

- **Training:** 31 min end to end. That covers 20.6 min of LoRA training (100 steps, ~10 s per step of 16 receipts, peak 36 GB), two vLLM evals, the merge (58 s) and the 25 GB copy to the Volume (50 s). The cached `causal-conv1d` wheel is reused; a first build adds ~11 min. 2 of the 800 training receipts exceed 4,096 tokens and are skipped.
- **Base model:** reads receipts well but invents its own structure. For example, it always writes `menu` as a list and adds keys that aren't in the CORD schema.
- **Fine-tuned model:** learns the CORD schema, including the dict/list distinction for `menu` and `sub` items. Its median latency in vLLM is 3.4 s per receipt with 16 concurrent requests.
- **Serving (`GPU_LARGE`):** _to be filled once the endpoint has been queried._

The AIR MLflow run contains:
- the parameters;
- the training curves (`train/loss`, `train/lr`, `train/grad_norm`);
- `base/*` and `ft/*` metrics;
- `metrics.json`.

`<output_root>/<MLFLOW_RUN_ID>/` holds `adapter/`, `merged/`, and every prediction (`predictions_base.json`, `predictions_ft.json`).

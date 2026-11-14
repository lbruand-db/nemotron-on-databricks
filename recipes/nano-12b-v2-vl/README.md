# Fine-tune and serve Nemotron Nano 12B v2 VL

This recipe LoRA fine-tunes [`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16) to extract structured JSON from receipt images, using [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC-BY-4.0; 800 train / 100 val / 100 test). It is the same task as the cookbook's Super VL recipe.

- **Model:** a 12.6 B hybrid Mamba-Transformer LLM with a C-RADIOv2-H vision encoder.
- **Evaluation:** base vs fine-tuned, both measured through vLLM.
- **Packaging:** the adapter is merged into the base weights and registered in Unity Catalog. It is served by vLLM inside a **classic pyfunc** (`VLLMChatModel`), in **FP8 on one A10G** (`GPU_MEDIUM`). Custom LLM Serving (OpenAI chat requests) is also supported, but its deployments fail on both test workspaces (SPEC OQ-5).

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
- **GPU for serving:** bf16 weights are ~25 GB, more than an A10G's 24 GB. The commands below therefore serve **FP8 weights** (`--quantization fp8`, quantized by vLLM at load time) on `GPU_MEDIUM` (1× A10G), which costs ~1 point of field F1 (see Results). bf16 needs `GPU_LARGE` (1× L40S 48 GB on AWS, 1× A100 80 GB on Azure): drop the two FP8 overrides. On the test workspaces, `GPU_LARGE` and `GPU_LARGE_RTX` never got capacity within the 2 h deployment limit.
- **Custom LLM Serving** (`serving=custom_llm`, OpenAI chat API) is in Public Preview and must be enabled by a workspace admin. Its deployments never started the vLLM entrypoint on either test workspace (SPEC OQ-5), so the default is the classic pyfunc.

All commands run from the **repository root**.

## Run

```bash
# 1. Train + evaluate (base and fine-tuned, both with vLLM) + merge
databricks air run -f recipes/nano-12b-v2-vl/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # last line: TRAINING_RESULT={... "mlflow_run_id": ...}

# 2. Optional: FP8 accuracy, within an A10G's memory, on 1x H100 (A10G kernels forced; metrics go to the stage-1 run)
databricks air run -f recipes/nano-12b-v2-vl/eval.yaml --override parameters.source_training_run_id=<MLFLOW_RUN_ID>

# 3. Register a classic pyfunc that runs vLLM in FP8 (smoke test: the pyfunc exactly as serving runs it)
databricks air run -f recipes/nano-12b-v2-vl/register.yaml --override parameters.source_training_run_id=<MLFLOW_RUN_ID> \
  --override "parameters.vllm_extra_args=--quantization fp8" --override parameters.vllm_env=VLLM_TEST_FORCE_FP8_MARLIN=1 \
  --override parameters.smoke_gpu_budget_gib=20.2
tools/air_wait.sh <JOB_RUN_ID>          # REGISTRATION_RESULT={... "model_version": ...}

# 4. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.nemotron_nano_12b_v2_vl_ft --model-version <N> \
  --endpoint-name nemotron-nano-vl-ft --workload-type GPU_MEDIUM --profile <PROFILE>
tools/endpoint_wait.sh nemotron-nano-vl-ft <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-nano-vl-ft --vlm receipt.png --profile <PROFILE>
```

`VLLM_TEST_FORCE_FP8_MARLIN=1` makes the H100 smoke test use the same FP8 kernels (Marlin, W8A16) as the A10G; on the A10G they are the default anyway. `smoke_gpu_budget_gib=20.2` caps the smoke test at the A10G's memory, so a config that doesn't fit fails at registration rather than after the deployment. The registered serving config keeps `gpu_memory_utilization` 0.90.

The pyfunc takes one record per receipt, `{"image": "<base64>", "id": "...", "instruction": "<optional>"}`, with params `{"max_tokens": 1536}`. It returns `{id, text, fields, json_valid, usage, latency_s}`, where `fields` is the parsed JSON. `query --vlm` builds that request; `--instruction` and `--max-tokens` set the optional fields.

**Custom LLM Serving variant** (`--override parameters.serving=custom_llm`): the endpoint takes OpenAI chat requests (`query --chat`). The image goes in as a base64 data URL (see `nemotron_dbx.lifecycle.query.build_chat_request`):

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
- **Reproduced on `e2-demo-field-eng`** (2026-11-13): field F1 0.382 → 0.858, TED 0.447 → 0.926.
- **FP8** (`eval.yaml`, A10G kernels, vLLM capped at 20.2 GiB): test field F1 **0.847**, TED 0.910, exact match 0.37, valid JSON 0.99. That is −0.011 F1 / −0.016 TED against bf16 on the same run. Weights take 14.5 GiB, leaving 2.9 GiB for the KV cache and Mamba state.
- **Serving** (`e2-demo-field-eng`, `GPU_MEDIUM` 1× A10G, classic pyfunc v3, FP8): READY ~57 min after deploy (container build, then vLLM start). Warm requests take **~4 s** in the model (4.0 / 4.2 s) and 8–11 s end to end from a laptop. The two held-out receipts tested came back as valid JSON with field F1 0.857 each: one missed field (`num` or `menuqty_cnt`) and one extra (`unitprice` or `cnt`) per receipt. Both receipts in one `--vlm` request take 6.9–7.6 s each, processed concurrently.
- **Scale to zero:** a scaled-to-zero endpoint rejects large requests ("Max request size exceeded for a scale to zero endpoint"). A 7.7 MB PNG was refused, a 338 KB one was accepted. The cold start (new container, then vLLM loading and quantizing the weights) took longer than the SDK's 5 min request timeout, so the first request fails with a timeout while the endpoint keeps scaling up. Wake it with a small image and retry, or disable scale-to-zero (`deploy --no-scale-to-zero`) for interactive use.
- **Instructions:** `--instruction` reaches the model, but the fine-tuned model still answers in the CORD schema it was trained on (asked for the total only, it returned the full receipt). Use the base model, or train on varied instructions, for free-form questions.

The AIR MLflow run contains:
- the parameters;
- the training curves (`train/loss`, `train/lr`, `train/grad_norm`);
- `base/*` and `ft/*` metrics;
- `metrics.json`.

`<output_root>/<MLFLOW_RUN_ID>/` holds `adapter/`, `merged/`, and every prediction (`predictions_base.json`, `predictions_ft.json`).

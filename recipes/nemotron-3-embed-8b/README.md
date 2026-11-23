# Evaluate and serve Nemotron-3-Embed-8B

This recipe serves [`nvidia/Nemotron-3-Embed-8B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16), a text embedding model for indexing the content the other recipes extract (RAG, semantic search). Unlike the other five recipes, it does **not fine-tune**: it evaluates the pinned model on a French retrieval set, then registers it and serves it on Model Serving (SPEC §6.6, M6).

- **Model:** an 8 B bidirectional encoder built on Ministral-3-8B, with mean pooling and L2-normalized 4,096-dim outputs. It covers 34 languages, French included, and accepts up to 32,768 tokens. Released 2026-07-16.
- **License:** OpenMDW-1.1 ("ready for commercial use"). NVIDIA's other 8B embedder, `llama-embed-nemotron-8b`, is non-commercial / research only, so it isn't used here.
- **Prompts:** queries get the prefix `query: `, documents `passage: `. The recipe adds them; callers send raw text plus an `input_type`.
- **Smaller vectors:** keep the first 2,048 or 1,024 dimensions and re-normalize (the model card's method). This matters for vector indexes:

| Index | Dimension limit | Fits |
|---|---|---|
| Databricks AI Search | 4,096 (divisible by 16 on storage-optimized endpoints) | 4,096, 2,048, 1,024 |
| Lakebase, pgvector HNSW / IVFFlat | 2,000 (`vector`), 4,000 (`halfvec`) | 2,048 as `halfvec`, 1,024 as `vector` |
| Lakebase, `lakebase_ann` | none documented | to be tested |

**Evaluation** (`evaluate.py`), on 1× A10, the same GPU class as the `GPU_MEDIUM` endpoint:
1. **Model-card check:** vLLM must reproduce the card's published query/document scores (4 × 4 example, tolerance 0.03). A wrong pooling mode or causal attention would fail here.
2. **Alloprof** ([`antoinelb7/alloprof`](https://huggingface.co/datasets/antoinelb7/alloprof), MIT on the Hub): French questions from a Quebec school help forum, each linked to the reference pages teachers cited. Corpus: all 2,798 French reference pages. Queries: a stable, hash-ordered sample of 1,000 of the 15,666 French questions with a relevant page (1.27 relevant pages per query). Some questions refer to an attached image the model can't see, which lowers the scores a little.
3. **Metrics:** NDCG@10, MRR@10 and recall@1/5/10/100, at 4,096, 2,048 and 1,024 dims, plus documents and queries embedded per second.

Alloprof is not in the model's published training-data list, unlike MIRACL or MLDR, which is why it was chosen (SPEC OQ-19). For a decision on your own documents, build the same kind of set from them: questions, plus the passages that answer them.

**Serving:** `nemotron_dbx.embed.serving.VLLMEmbedModel`, a classic pyfunc that starts vLLM in pooling mode (`--runner pooling`, `--max-model-len 8192`) inside the container, like the Nano VL pyfunc. It doesn't use `env_pack` or Custom LLM Serving, which fail on the test workspaces (SPEC OQ-5). The ~16 GB of BF16 weights fit a 24 GB A10G (`GPU_MEDIUM`) on AWS; Azure Model Serving has no A10G, so use `GPU_LARGE` (A100) there.

## Before you run

- Set `parameters.output_root` in `workload.yaml` to an existing writable Volume directory, and `parameters.registered_model_name` in `register.yaml`. You need `USE CATALOG`, `USE SCHEMA` and `CREATE MODEL` on its schema.
- The jobs download the model (~16 GB) and Alloprof (~390 MB) from Hugging Face.

All commands run from the **repository root**.

## Run

```bash
# 1. Evaluate (model-card check + Alloprof retrieval at 4096 / 2048 / 1024 dims)
databricks air run -f recipes/nemotron-3-embed-8b/workload.yaml
tools/air_wait.sh <JOB_RUN_ID>          # last line: EVAL_RESULT={... "mlflow_run_id": ...}

# 2. Register the pyfunc (smoke test: the pyfunc as serving runs it, against the card's scores)
databricks air run -f recipes/nemotron-3-embed-8b/register.yaml --override parameters.source_eval_run_id=<MLFLOW_RUN_ID>
tools/air_wait.sh <JOB_RUN_ID>          # REGISTRATION_RESULT={... "model_version": ...}

# 3. Deploy + query
python -m nemotron_dbx.lifecycle.deploy --model-name <CATALOG>.<SCHEMA>.nemotron_3_embed_8b --model-version <N> \
  --endpoint-name nemotron-3-embed-8b --workload-type GPU_MEDIUM --profile <PROFILE>
tools/endpoint_wait.sh nemotron-3-embed-8b <PROFILE>
python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-3-embed-8b --embed --input-type query \
  --dimensions 1024 "Comment calculer la constante d'équilibre ?" --profile <PROFILE>
```

The pyfunc takes one record per text and returns one embedding per record:

```json
{"dataframe_records": [{"text": "Les fractions équivalentes...", "id": "d1", "input_type": "document"},
                       {"text": "Comment simplifier une fraction ?", "id": "q1", "input_type": "query"}],
 "params": {"dimensions": 1024}}
```

```json
{"predictions": [{"id": "d1", "input_type": "document", "dimensions": 1024, "embedding": [0.012, ...]}, ...]}
```

`input_type` defaults to `document`. `dimensions` defaults to 4,096.

## Results

Observed on 2026-11-23 (`e2-demo-field-eng`, 1× A10 24 GB; HF revision `d1f2f257`, Alloprof revision `0faa90fe`):

| Alloprof (1,000 queries, 2,798 pages) | 4,096 dims | 2,048 dims | 1,024 dims |
|---|---|---|---|
| NDCG@10 | **0.565** | 0.552 | 0.540 |
| MRR@10 | 0.532 | 0.521 | 0.506 |
| recall@1 | 0.387 | 0.376 | 0.363 |
| recall@10 | 0.717 | 0.697 | 0.690 |
| recall@100 | 0.854 | 0.843 | 0.833 |

- **Model-card check:** vLLM reproduces the card's scores within **0.0007** (diagonal 0.785 / 0.651 / 0.661 / 0.799).
- **Smaller vectors cost little:** 1,024 dims keep 95.5 % of the 4,096-dim NDCG@10 (−0.025) for a quarter of the storage.
- **Speed on the A10:** vLLM ready in 120 s; 33.8 reference pages/s (≈ 490 characters each) and 69.6 queries/s, at 8 concurrent batches of 32. The eval job took 5.7 min end to end, the registration job 6.4 min.
- **Context:** these are scores for the base model on a set it wasn't trained on. There's no fine-tuned comparison: this recipe doesn't fine-tune.
- **Serving** (`e2-demo-field-eng`, `GPU_MEDIUM` 1× A10G, classic pyfunc v1): READY **34 min** after deploy (container build, then vLLM start). A request with 2 texts takes 3–4 s end to end from a laptop, at 4,096 or 1,024 dims. In one mixed request (3 French queries + 3 documents, including a fixed- vs variable-rate loan question), every query ranks its own document first at both sizes (matching scores 0.59–0.69, others ≤ 0.23).

The MLflow run holds every metric (`alloprof/dim<N>/...`), `metrics.json` and the vLLM log. `<output_root>/<MLFLOW_RUN_ID>/predictions.json` lists the top 10 pages for each query, for error analysis.

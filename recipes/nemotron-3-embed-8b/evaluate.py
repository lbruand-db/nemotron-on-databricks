"""Evaluate Nemotron-3-Embed-8B (no fine-tuning) with vLLM on a French retrieval set, at several output sizes.

Stage 1 of 3 (evaluate -> register -> deploy). Run from the repo root:
    databricks air run -f recipes/nemotron-3-embed-8b/workload.yaml

1. Pinned model snapshot and the Alloprof eval set, in parallel.
2. vLLM in pooling mode (the same command the endpoint runs), checked against the model card's published scores.
3. Embed the French corpus and queries once at 4,096 dims, then score each truncated size (e.g. 2,048 and 1,024,
   the sizes that fit Lakebase pgvector indexes): NDCG@10, MRR@10, recall@k.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.hf import NEMOTRON_3_EMBED_8B
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import run_dir, scratch_dir


@dataclass
class Params:
    output_root: str
    n_queries: int = 1000  # 0 = every French question with a relevant page (~15.7k)
    dims: str = "4096,2048,1024"
    batch_size: int = 32
    concurrency: int = 8
    vllm_extra_args: str = ""
    vllm_env: str = ""
    gpu_budget_gib: float = 0.0  # cap vLLM's memory (e.g. 20.2 = an A10G) when evaluating on a bigger GPU


def main() -> None:
    from nemotron_dbx.embed.data import ALLOPROF, load_alloprof
    from nemotron_dbx.embed.metrics import retrieval_metrics, top_k_indices
    from nemotron_dbx.embed.server import (
        EMBED_VLLM,
        card_check,
        embed_texts,
        parse_dims_list,
        snapshot_model,
        truncate_normalize,
        vllm_embed_command,
    )
    from nemotron_dbx.vlm.vllm_server import VLLMServer, gpu_total_gib, with_gpu_budget

    p = build_config(Params, load_parameters())
    print(f"[setup] params: {asdict(p)}", flush=True)
    mlf.log_params(asdict(p))
    try:
        dims = parse_dims_list(p.dims)  # before the 16 GB download and the vLLM start
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    t0 = time.monotonic()
    out = run_dir(p.output_root)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Model (~16 GB) and eval set, in parallel.
    with ThreadPoolExecutor(2) as pool:
        snap = pool.submit(snapshot_model)
        ds = load_alloprof(scratch_dir("alloprof"), p.n_queries or None)
        model_dir = snap.result()
    print(f"[data] Alloprof: {len(ds.docs)} French reference pages, {len(ds.queries)} queries, "
          f"{np.mean([len(r) for r in ds.relevant]):.2f} relevant pages per query; ready at "
          f"{time.monotonic() - t0:.0f}s", flush=True)

    # 2. vLLM pooling server + model-card check.
    cfg = replace(EMBED_VLLM, extra_args=tuple(p.vllm_extra_args.split()), env=tuple(p.vllm_env.split()))
    total_gib = gpu_total_gib()
    cfg = with_gpu_budget(cfg, p.gpu_budget_gib, total_gib)
    print(f"[setup] GPU {total_gib:.1f} GiB; vLLM config {cfg}", flush=True)
    scratch = scratch_dir("nemotron-3-embed-eval")
    served = "nemotron-3-embed"
    metrics: dict[str, float] = {}
    with VLLMServer(str(model_dir), served_model_name=served, port=3081, cfg=cfg, log_path=scratch / "vllm.log",
                    command_builder=vllm_embed_command) as server:
        metrics["vllm_startup_s"] = round(server.startup_s)
        card = card_check(server.port, served)
        print(f"[eval] model-card check: {card}", flush=True)
        metrics["card_max_abs_err"] = card["max_abs_err"]

        # 3. Embed once at full size.
        kw = {"batch_size": p.batch_size, "concurrency": p.concurrency, "max_tokens": cfg.max_model_len}
        t1 = time.monotonic()
        doc_emb = embed_texts(server.port, served, ds.docs, "document", **kw)
        metrics["docs_per_s"] = round(len(ds.docs) / (time.monotonic() - t1), 1)
        t2 = time.monotonic()
        query_emb = embed_texts(server.port, served, ds.queries, "query", **kw)
        metrics["queries_per_s"] = round(len(ds.queries) / (time.monotonic() - t2), 1)
    print(f"[eval] embedded {len(ds.docs)} docs ({metrics['docs_per_s']}/s) and {len(ds.queries)} queries "
          f"({metrics['queries_per_s']}/s)", flush=True)

    results = {}
    top = None  # top-10 at the largest size, reused for predictions.json (no second matmul or full sort)
    for d in sorted(dims, reverse=True):
        q, c = truncate_normalize(query_emb, d), truncate_normalize(doc_emb, d)
        scores = q @ c.T
        m = retrieval_metrics(scores, ds.relevant)
        if top is None:
            top, top_dim = top_k_indices(scores, 10), d
        results[d] = m
        metrics |= {f"alloprof/dim{d}/{k.replace('@', '_at_')}": v for k, v in m.items()}
        print(f"[eval] dims {d}: {m}", flush=True)
    mlf.log_metrics(metrics)

    # Per-query top 10 at the largest evaluated size, for error analysis.
    preds = [{"query_id": qid, "query": qt, "dims": top_dim, "relevant": [ds.doc_ids[i] for i in sorted(rel)],
              "top10": [ds.doc_ids[i] for i in row]}
             for qid, qt, rel, row in zip(ds.query_ids, ds.queries, ds.relevant, top, strict=True)]
    (out / "predictions.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    result = {"model": f"{NEMOTRON_3_EMBED_8B.repo_id}@{NEMOTRON_3_EMBED_8B.revision}",
              "eval_set": f"{ALLOPROF.repo_id}@{ALLOPROF.revision}", "n_docs": len(ds.docs),
              "n_queries": len(ds.queries), "card_check": card, "vllm": asdict(cfg),
              "gpu_total_gib": round(total_gib, 1),
              "retrieval": results, "throughput": {k: metrics[k] for k in ("docs_per_s", "queries_per_s")},
              "minutes": round((time.monotonic() - t0) / 60, 1)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    mlf.log_artifact(str(out / "metrics.json"))
    mlf.log_artifact(str(scratch / "vllm.log"))
    mlf.set_tags({"nemotron.recipe": "nemotron-3-embed-8b", "nemotron.base_source": result["model"],
                  "nemotron.eval_set": result["eval_set"]})
    print("EVAL_RESULT=" + json.dumps({"mlflow_run_id": mlf.run_id(), "minutes": result["minutes"],
                                       **{f"ndcg@10/{d}": results[d]["ndcg@10"] for d in dims}}), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate the merged checkpoint of a stage-1 run with different vLLM serving flags (e.g. FP8 for a 24 GB A10G).

Optional stage between train and register. Run from the repo root:
    databricks air run -f recipes/nano-12b-v2-vl/eval.yaml --override parameters.source_training_run_id=<RUN_ID>

`gpu_budget_gib` caps vLLM at the memory it would get on a smaller serving GPU, so an H100 run also shows the
config fits there. With the default 20.2 GiB budget (0.90 of an A10G's 22.4 GiB), vLLM's
`--gpu-memory-utilization` is set to 20.2 / <this GPU's total>. Metrics go to the stage-1 run, prefixed with `tag`.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import scratch_dir


@dataclass
class Params:
    source_training_run_id: str
    tag: str = "fp8"  # metric prefix, e.g. ft_fp8/test/field_f1
    vllm_extra_args: str = "--quantization fp8"
    vllm_env: str = "VLLM_TEST_FORCE_FP8_MARLIN=1"  # the A10G (Ampere) FP8 kernels, also on H100
    gpu_budget_gib: float = 20.2  # 0 = vLLM's default share of this GPU
    test_items: int = 0  # 0 = all 100
    eval_val: bool = False
    eval_concurrency: int = 16
    max_new_tokens: int = 1536


def main() -> None:
    from nemotron_dbx.vlm.data import load_cord
    from nemotron_dbx.vlm.evaluate import evaluate_with_vllm
    from nemotron_dbx.vlm.vllm_server import VLLMConfig, gpu_total_gib, with_gpu_budget

    p = build_config(Params, load_parameters())
    print(f"[setup] params: {asdict(p)}", flush=True)
    mlf.log_params(asdict(p))
    if not p.source_training_run_id:
        raise SystemExit("Set parameters.source_training_run_id (the stage-1 MLFLOW_RUN_ID)")
    tags = mlflow.MlflowClient().get_run(p.source_training_run_id).data.tags
    merged = tags.get("nemotron.merged_dir")
    if not merged or not (Path(merged) / "config.json").is_file():
        raise SystemExit(f"No merged checkpoint for run {p.source_training_run_id} (nemotron.merged_dir={merged!r})")

    scratch = scratch_dir("nano-vl-eval")
    local = scratch / "merged"
    shutil.rmtree(local, ignore_errors=True)
    t0 = time.monotonic()
    shutil.copytree(merged, local)  # vLLM loads ~3x faster from local disk than from the Volume
    print(f"[setup] copied {merged} -> {local} in {time.monotonic() - t0:.0f}s", flush=True)

    total_gib = gpu_total_gib()  # nvidia-smi: no CUDA context in this process while vLLM runs
    cfg = VLLMConfig(extra_args=tuple(p.vllm_extra_args.split()), env=tuple(p.vllm_env.split()))
    cfg = with_gpu_budget(cfg, p.gpu_budget_gib, total_gib)
    print(f"[setup] GPU {total_gib:.1f} GiB; vLLM config {cfg}", flush=True)

    cache = scratch_dir("cord-v2")
    splits = {"test": load_cord("test", p.test_items or None, cache)}
    if p.eval_val:
        splits["val"] = load_cord("validation", None, cache)
    metrics, preds = evaluate_with_vllm(local, splits, scratch / "vllm.log", cfg=cfg,
                                        concurrency=p.eval_concurrency, max_tokens=p.max_new_tokens)
    log = (scratch / "vllm.log").read_text(errors="replace")
    memory_lines = [ln for ln in log.splitlines() if "GiB" in ln and ("memory" in ln.lower() or "KV" in ln)]
    print("[eval] vLLM memory lines:\n" + "\n".join(memory_lines[-8:]), flush=True)
    print("[eval] metrics:", json.dumps(metrics, indent=1), flush=True)

    mlf.log_metrics(metrics, prefix=f"ft_{p.tag}/")  # this run
    client = mlflow.MlflowClient()  # and the stage-1 run, next to base/ and ft/
    for k, v in metrics.items():
        client.log_metric(p.source_training_run_id, f"ft_{p.tag}/{k}", v)
    out = Path(merged).parent / f"predictions_ft_{p.tag}.json"
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    mlf.log_artifact(str(scratch / "vllm.log"))
    print("EVAL_RESULT=" + json.dumps({"tag": p.tag, "vllm": asdict(cfg), "gpu_total_gib": round(total_gib, 1),
                                       **{k: v for k, v in metrics.items() if k.startswith("test/")}}), flush=True)


if __name__ == "__main__":
    main()

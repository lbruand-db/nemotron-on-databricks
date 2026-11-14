"""LoRA fine-tune Nemotron Nano 12B v2 VL on CORD-v2 receipt extraction; evaluate base vs fine-tuned with vLLM.

Stage 1 of 3 (train -> register -> deploy). Run from the repo root:
    databricks air run -f recipes/nano-12b-v2-vl/workload.yaml

Phases (each GPU phase is its own process, so GPU memory is fully released between them):
  1. Training deps (mamba-ssm Triton ops, cached causal-conv1d wheel) + pinned base snapshot, in parallel.
  2. Base model eval: vLLM OpenAI server, CORD test (+ validation), same chat format as the endpoint.
  3. LoRA training: `torchrun -m nemotron_dbx.vlm.lora` (HF remote code + PEFT, SPEC §11.3 patches).
  4. Merge the adapter into the base shards, then evaluate the merged model with vLLM.
  5. Save adapter/ and merged/ to <output_root>/<MLFLOW_RUN_ID>/ on the Volume.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.hf import CORD_V2, NANO_VL_V2
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import run_dir, scratch_dir
from nemotron_dbx.vlm.merge import MERGED_DIR_TAG

BASE_PATTERNS = ["*.safetensors", "*.json", "*.py", "*.jinja", "*.md", "LICENSE*"]


@dataclass
class Params:
    output_root: str
    wheel_cache: str
    train_items: int = 0  # 0 = all 800 CORD train receipts
    val_items: int = 0  # 0 = all 100
    test_items: int = 0  # 0 = all 100
    eval_val: bool = True
    epochs: int = 2
    global_batch: int = 16
    lr: float = 1e-4
    warmup_steps: int = 5
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    max_len: int = 4096
    max_tiles: int = 12
    max_train_minutes: float = 50.0
    eval_concurrency: int = 16
    max_new_tokens: int = 1536
    nproc_per_node: int = 0  # 0 = every visible GPU
    save_merged: bool = True  # copy merged/ (~25 GB) to the Volume; registration needs it
    seed: int = 0


def gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


def main() -> None:
    p = build_config(Params, load_parameters())
    print(f"[setup] params: {asdict(p)}", flush=True)
    mlf.log_params(asdict(p))
    t0 = time.monotonic()
    scratch = scratch_dir("nano-vl")
    out = run_dir(p.output_root)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Deps and the 25 GB base snapshot, in parallel.
    from nemotron_dbx.vlm.env import ensure_training_deps

    with ThreadPoolExecutor(2) as pool:
        # Into the HF cache (HF_HOME), so the HF training code can load it by repo id + revision.
        snap = pool.submit(NANO_VL_V2.snapshot, None, BASE_PATTERNS)
        deps = ensure_training_deps(p.wheel_cache)
        base_dir = snap.result()
    print(f"[setup] deps + base snapshot ready at {time.monotonic() - t0:.0f}s", flush=True)

    from nemotron_dbx.vlm.data import load_cord
    from nemotron_dbx.vlm.evaluate import evaluate_with_vllm

    cache = scratch_dir("cord-v2")
    test = load_cord("test", p.test_items or None, cache)
    splits = {"test": test}
    if p.eval_val:
        splits["val"] = load_cord("validation", p.val_items or None, cache)
    print(f"[data] eval splits: { {k: len(v) for k, v in splits.items()} }", flush=True)
    ev = {"concurrency": p.eval_concurrency, "max_tokens": p.max_new_tokens}

    # 2. Base model.
    base, base_preds = evaluate_with_vllm(base_dir, splits, scratch / "vllm_base.log", **ev)
    mlf.log_metrics(base, prefix="base/")
    (out / "predictions_base.json").write_text(json.dumps(base_preds, ensure_ascii=False, indent=1))

    # 3. LoRA (subprocess under torchrun; the remote code needs a process group).
    adapter_local = scratch / "adapter"
    shutil.rmtree(adapter_local, ignore_errors=True)
    cfg = {
        "base_dir": str(base_dir), "model_id": NANO_VL_V2.repo_id, "revision": NANO_VL_V2.revision,
        "adapter_dir": str(adapter_local), "data_cache": str(cache),
        "train_items": p.train_items, "epochs": p.epochs, "global_batch": p.global_batch, "lr": p.lr,
        "warmup_steps": p.warmup_steps, "rank": p.lora_rank, "alpha": p.lora_alpha, "dropout": p.lora_dropout,
        "max_len": p.max_len, "max_tiles": p.max_tiles, "max_minutes": p.max_train_minutes, "seed": p.seed,
    }
    cfg_path = scratch / "lora_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    nproc = p.nproc_per_node or gpu_count()
    t1 = time.monotonic()
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}",
         "-m", "nemotron_dbx.vlm.lora", "--config", str(cfg_path)],
        check=True, env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    train_min = (time.monotonic() - t1) / 60
    train_summary = json.loads((adapter_local / "train_summary.json").read_text())

    # 4. Merge + fine-tuned eval (same splits, same server flags).
    from nemotron_dbx.vlm.merge import merge_adapter

    merged_local = scratch / "merged"
    shutil.rmtree(merged_local, ignore_errors=True)
    merge_info = merge_adapter(base_dir, adapter_local, merged_local)
    ft, ft_preds = evaluate_with_vllm(merged_local, splits, scratch / "vllm_ft.log", **ev)
    mlf.log_metrics(ft, prefix="ft/")
    (out / "predictions_ft.json").write_text(json.dumps(ft_preds, ensure_ascii=False, indent=1))

    # 5. Durable outputs.
    shutil.copytree(adapter_local, out / "adapter", dirs_exist_ok=True)
    merged_dir = out / "merged"
    if p.save_merged:
        t2 = time.monotonic()
        shutil.copytree(merged_local, merged_dir, dirs_exist_ok=True)
        print(f"[save] merged -> {merged_dir} in {time.monotonic() - t2:.0f}s", flush=True)
    sample = out / "sample_test_image.png"
    sample.write_bytes(test[0].image)
    (out / "sample_test_gt.json").write_text(json.dumps(test[0].target, ensure_ascii=False))

    result = {
        "adapter_dir": str(out / "adapter"), "merged_dir": str(merged_dir) if p.save_merged else None,
        "base": base, "fine_tuned": ft, "train": train_summary, "merge": merge_info, "deps": deps,
        "revisions": {"model": NANO_VL_V2.revision, "cord": CORD_V2.revision},
        "minutes": {"train": round(train_min, 1), "total": round((time.monotonic() - t0) / 60, 1)},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    mlf.log_artifact(str(out / "metrics.json"))
    mlf.log_artifact(str(adapter_local / "adapter_config.json"), "adapter")
    mlf.set_tags({
        "nemotron.recipe": "nano-12b-v2-vl", "nemotron.adapter_dir": out / "adapter",
        MERGED_DIR_TAG: merged_dir if p.save_merged else "", "nemotron.sample_image": sample,
    })
    keys = ("test/field_f1", "test/ted_acc", "test/json_valid")
    print("TRAINING_RESULT=" + json.dumps({
        "merged_dir": result["merged_dir"], "minutes": result["minutes"], "mlflow_run_id": mlf.run_id(),
        **{f"base_{k}": base.get(k) for k in keys}, **{f"ft_{k}": ft.get(k) for k in keys},
    }), flush=True)


if __name__ == "__main__":
    main()

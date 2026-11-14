"""Merge a PEFT LoRA adapter into the base checkpoint's safetensors shards (SPEC §6.5.2 artifacts).

This edits the base shards directly (W += scale * B @ A for each LoRA target) instead of calling
`merge_and_unload()` + `save_pretrained()`. The output keeps the base's tensor names, dtypes, shard layout
and index, so vLLM loads it exactly like the base model. It also avoids transformers 5 re-serialising
remote-code buffers (e.g. RADIO's `summary_idxs`, which the base checkpoint doesn't have). Non-weight files
(remote code, tokenizer, chat template, configs) are copied unchanged.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

PEFT_PREFIX = "base_model.model."
MERGED_DIR_TAG = "nemotron.merged_dir"  # set by the training recipe on its MLflow run


def check_merged_dir(merged: str | None, training_run_id: str) -> Path:
    """Fail early, with the reason, unless `merged` is a complete merged checkpoint."""
    if not merged:
        raise SystemExit(f"Run {training_run_id} has no {MERGED_DIR_TAG} tag: it isn't a finished training run, or "
                         "it ran with save_merged=false. Re-run training with save_merged=true.")
    path = Path(merged)
    missing = [f for f in ("config.json", "model.safetensors.index.json") if not (path / f).is_file()]
    if missing:
        raise SystemExit(f"Merged checkpoint {path} of run {training_run_id} is incomplete (missing {missing})")
    return path


def merged_checkpoint(training_run_id: str) -> tuple[Path, dict[str, str]]:
    """The merged checkpoint saved by a training run, plus that run's tags (registration also needs
    `nemotron.sample_image`)."""
    if not training_run_id:
        raise SystemExit("Set parameters.source_training_run_id (the stage-1 MLFLOW_RUN_ID)")
    import mlflow

    tags = mlflow.MlflowClient().get_run(training_run_id).data.tags
    return check_merged_dir(tags.get(MERGED_DIR_TAG), training_run_id), tags


def lora_pairs(adapter_tensors: dict, prefix: str = PEFT_PREFIX) -> dict[str, tuple]:
    """{base weight key: (A, B)} from PEFT adapter tensors (`<prefix><module>.lora_A.weight`, ...)."""
    pairs: dict[str, dict] = {}
    for key, t in adapter_tensors.items():
        for part in ("lora_A", "lora_B"):
            marker = f".{part}.weight"
            if key.endswith(marker):
                module = key[: -len(marker)]
                module = module[len(prefix):] if module.startswith(prefix) else module
                pairs.setdefault(f"{module}.weight", {})[part] = t
    incomplete = [k for k, v in pairs.items() if len(v) != 2]
    if incomplete:
        raise ValueError(f"LoRA A/B mismatch for {incomplete[:5]}")
    return {k: (v["lora_A"], v["lora_B"]) for k, v in pairs.items()}


def merge_adapter(base_dir: str | Path, adapter_dir: str | Path, out_dir: str | Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    t0 = time.monotonic()
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if acfg.get("use_rslora") or acfg.get("use_dora"):
        raise NotImplementedError("rsLoRA / DoRA merging is not implemented")
    scale = acfg["lora_alpha"] / acfg["r"]
    pairs = lora_pairs(load_file(str(adapter_dir / "adapter_model.safetensors")))

    index_path = base_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(base_dir / "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    missing = sorted(set(pairs) - set(weight_map))
    if missing:
        raise KeyError(f"{len(missing)} LoRA targets are not in the base checkpoint, e.g. {missing[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():  # remote code, tokenizer, configs, index...
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out_dir / f.name)

    merged = 0
    max_delta = 0.0
    for shard in sorted(set(weight_map.values())):
        tensors, metadata = {}, None
        with safe_open(str(base_dir / shard), "pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                w = f.get_tensor(key)
                if key in pairs:
                    a, b = pairs[key]
                    delta = (b.float() @ a.float()) * scale
                    if delta.shape != w.shape:
                        raise ValueError(f"{key}: delta {tuple(delta.shape)} vs weight {tuple(w.shape)}")
                    w = (w.float() + delta).to(w.dtype)
                    max_delta = max(max_delta, delta.abs().max().item())
                    merged += 1
                tensors[key] = w.contiguous()
        save_file(tensors, str(out_dir / shard), metadata=metadata or {"format": "pt"})
        del tensors
    if merged != len(pairs):
        raise RuntimeError(f"merged {merged} of {len(pairs)} LoRA targets")
    info = {"merged_modules": merged, "scale": scale, "max_abs_delta": max_delta,
            "merge_s": round(time.monotonic() - t0)}
    (out_dir / "nemotron_dbx_merge.json").write_text(json.dumps({**info, "adapter_config": acfg}, indent=2))
    print(f"[merge] {info}", flush=True)
    return info

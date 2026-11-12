"""LoRA fine-tuning of Nano VL with HF remote code + PEFT (SPEC §6.5.2). Run under torchrun:

    torchrun --nproc-per-node=<gpus> -m nemotron_dbx.vlm.lora --config cfg.json

The remote code's forward calls `torch.distributed.get_rank()`, so a process group is required even on one GPU
(patch 10). With several GPUs, each rank takes a disjoint slice of every global batch and the LoRA gradients
(~90 M params) are all-reduced by hand before each optimizer step. This avoids wrapping the remote-code model
in DDP.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "in_proj", "out_proj")
FROZEN_MODULES = r".*(vision_model|mlp1).*"  # RADIO vision tower + projector stay frozen


@dataclass
class LoRATrainConfig:
    base_dir: str  # local snapshot (HF cache) of model_id@revision
    model_id: str
    revision: str
    adapter_dir: str
    data_cache: str
    train_items: int = 0  # 0 = all 800
    epochs: int = 2
    global_batch: int = 16
    lr: float = 1e-4
    warmup_steps: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    targets: list[str] = field(default_factory=lambda: list(LORA_TARGETS))
    max_len: int = 4096
    max_tiles: int = 12
    max_minutes: float = 50.0
    seed: int = 0


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 0 (step is 0-based)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def rank_slice(indices: list[int], step: int, global_batch: int, rank: int, world: int) -> list[int]:
    """This rank's micro-batches for one optimizer step (disjoint across ranks)."""
    batch = indices[step * global_batch:(step + 1) * global_batch]
    return batch[rank::world]


def load_model(cfg: LoRATrainConfig, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from ..common.hf import NANO_VL_V2_REMOTE_CODE
    from . import compat

    # Load by repo id + revision (served from the HF cache snapshot), not by local path: for local directories,
    # transformers 5.8.1 copies only the direct relative imports of modeling.py into its module cache, then
    # fails on the transitive ones (configuration.py -> configuration_nemotron_h.py).
    src = {"pretrained_model_name_or_path": cfg.model_id, "revision": cfg.revision}
    compat.pin_remote_code(NANO_VL_V2_REMOTE_CODE)
    compat.patch_post_init(get_class_from_dynamic_module("modeling.NemotronH_Nano_VL_V2", cfg.model_id,
                                                         revision=cfg.revision))
    model = AutoModelForCausalLM.from_pretrained(
        **src, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation=compat.ATTN_IMPLEMENTATION,
    ).to(device)
    info = {
        "summary_idxs": compat.fix_radio_summary_idxs(model),
        "patched_llm_output": compat.patch_llm_output(model),
        "mamba_fast_path": compat.mamba_fast_path(model),
    }
    if not info["mamba_fast_path"]:
        raise RuntimeError("Fused Mamba kernels unavailable (causal-conv1d CUDA ext missing): the torch path OOMs")
    # Default tokenizer settings (no `fix_mistral_regex`): vLLM loads it the same way, and train/serve
    # tokenization must match.
    tok = AutoTokenizer.from_pretrained(**src, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(**src, trust_remote_code=True)
    proc.image_processor.max_num_tiles = cfg.max_tiles
    info["fixed_image_rescale"] = compat.fix_image_rescale(proc.image_processor)
    return model, tok, proc, info


def add_lora(model, cfg: LoRATrainConfig):
    from peft import LoraConfig, get_peft_model

    from . import compat

    for p in model.parameters():
        p.requires_grad_(False)
    names = {n.split(".")[-1] for n, _ in model.language_model.named_modules()}
    targets = [t for t in cfg.targets if t in names]
    lcfg = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=targets,
                      exclude_modules=FROZEN_MODULES)
    model = get_peft_model(model, lcfg)
    model.base_model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    removed = compat.remove_require_grads_hooks(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {"lora_targets": targets, "trainable_params": trainable, "removed_hooks": removed}


def train(cfg: LoRATrainConfig) -> dict:
    import torch
    import torch.distributed as dist

    from ..common import dist as rk
    from ..common import mlflow_utils as mlf
    from .data import load_cord

    dist.init_process_group("nccl")
    torch.cuda.set_device(rk.local_rank())
    device = torch.device("cuda")
    world, rank = dist.get_world_size(), dist.get_rank()
    t_start = time.monotonic()

    model, tok, proc, info = load_model(cfg, device)
    model, lora_info = add_lora(model, cfg)
    info |= lora_info
    if rk.is_main():
        print(f"[lora] model ready in {time.monotonic() - t_start:.0f}s: {info}", flush=True)

    samples = load_cord("train", cfg.train_items or None, cfg.data_cache)
    from .data import encode_example

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(samples) // cfg.global_batch
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, cfg.warmup_steps))
    eos = tok.eos_token_id
    model.train()
    step, skipped, stop = 0, 0, False
    t_train = time.monotonic()
    for epoch in range(cfg.epochs):
        order = list(range(len(samples)))
        random.Random(cfg.seed + epoch).shuffle(order)  # same permutation on every rank
        for s in range(steps_per_epoch):
            t0 = time.monotonic()
            mine = rank_slice(order, s, cfg.global_batch, rank, world)
            loss_sum = torch.zeros(2, device=device)  # [sum of losses, n samples]
            tokens = 0
            for idx in mine:
                ex = encode_example(proc, tok, samples[idx], cfg.max_len, eos)
                if ex is None:
                    skipped += 1
                    continue
                ex = {k: v.to(device) for k, v in ex.items()}
                pixel_values = ex["pixel_values"].to(torch.bfloat16)
                out = model(
                    pixel_values=pixel_values, input_ids=ex["input_ids"], attention_mask=ex["attention_mask"],
                    image_flags=torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=device),
                    labels=ex["labels"], use_cache=False,
                )
                (out.loss / len(mine)).backward()
                loss_sum += torch.stack([out.loss.detach().float(), torch.ones((), device=device)])
                tokens += ex["input_ids"].shape[1]
            if world > 1:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                dist.all_reduce(loss_sum)
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            elapsed = (time.monotonic() - t_train) / 60
            metrics = {
                "train/loss": (loss_sum[0] / loss_sum[1].clamp(min=1)).item(), "train/lr": sched.get_last_lr()[0],
                "train/grad_norm": float(gnorm), "train/step_s": time.monotonic() - t0,
                "train/tokens_rank0": tokens, "train/epoch": epoch + (s + 1) / steps_per_epoch,
            }
            mlf.log_metrics(metrics, step=step)
            if rk.is_main():
                print(f"[lora] step {step}/{total} " + " ".join(f"{k[6:]}={v:.4g}" for k, v in metrics.items()),
                      flush=True)
            flag = torch.tensor([1.0 if elapsed > cfg.max_minutes else 0.0], device=device)
            if world > 1:
                dist.broadcast(flag, 0)
            if flag.item():
                print(f"[lora] time cap {cfg.max_minutes} min reached at step {step}", flush=True)
                stop = True
                break
        if stop:
            break

    summary = {
        **info, "steps": step, "planned_steps": total, "skipped_too_long": skipped, "world_size": world,
        "train_minutes": round((time.monotonic() - t_train) / 60, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }
    if rk.is_main():
        out = Path(cfg.adapter_dir)
        model.save_pretrained(out)
        (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print("LORA_SUMMARY=" + json.dumps(summary, default=str), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON file with LoRATrainConfig fields")
    args = ap.parse_args(argv)
    raw = json.loads(Path(args.config).read_text())
    names = {f.name for f in dataclasses.fields(LoRATrainConfig)}
    train(LoRATrainConfig(**{k: v for k, v in raw.items() if k in names}))


if __name__ == "__main__":
    main()

"""Fine-tuning loop for Nemotron YOLOX detectors (single GPU)."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import DetectionDataset, collate
from .loss import head_train_forward, yolox_loss


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1  # CSPDarknet backbone learns 10x slower than neck + head
    weight_decay: float = 5e-4
    warmup_steps: int = 100
    min_lr_ratio: float = 0.05
    ema_decay: float = 0.9998
    grad_clip: float = 10.0
    num_workers: int = 8
    log_every: int = 20
    seed: int = 0
    max_minutes: float | None = None  # hard stop for the time budget


class ModelEMA:
    """Exponential moving average of weights (YOLOX `ModelEMA`, decay ramp 1 - exp(-t/2000))."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    """No weight decay on BatchNorm and biases; lower LR for the CSPDarknet backbone."""
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.backbone.")
        no_decay = p.ndim <= 1  # BN weights/biases and conv biases
        groups.setdefault((is_backbone, no_decay), []).append(p)
    out = []
    for (is_backbone, no_decay), params in groups.items():
        out.append({
            "params": params,
            "lr": cfg.lr * (cfg.backbone_lr_mult if is_backbone else 1.0),
            "weight_decay": 0.0 if no_decay else cfg.weight_decay,
            "name": f"{'backbone' if is_backbone else 'neck_head'}{'_nodecay' if no_decay else ''}",
        })
    return out


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train(
    model: nn.Module,
    train_ds: DetectionDataset,
    num_classes: int,
    cfg: TrainConfig,
    device: torch.device,
    log_fn: Callable[[dict[str, float], int], None] | None = None,
) -> nn.Module:
    """Fine-tune `model` (a YOLOX module: `.backbone` + `.head`). Returns the EMA model."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    groups = param_groups(model, cfg)
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)
    ema = ModelEMA(model, cfg.ema_decay)
    steps_per_epoch = max(1, len(loader) // cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    use_amp = device.type == "cuda"
    t0 = time.monotonic()
    step = 0
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        for i, (imgs, labels, _) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                fpn = model.backbone(imgs)
                head_out = head_train_forward(model.head, fpn)
            losses = yolox_loss(head_out, labels, num_classes)
            (losses.loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum:
                continue
            f = lr_factor(step, total_steps, cfg.warmup_steps, cfg.min_lr_ratio)
            for g, base in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = base * f
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            step += 1
            if log_fn and (step % cfg.log_every == 0 or step == 1):
                log_fn({
                    "train/loss": losses.loss.item(),
                    "train/loss_iou": losses.iou.item(),
                    "train/loss_obj": losses.obj.item(),
                    "train/loss_cls": losses.cls.item(),
                    "train/fg_per_gt": losses.num_fg_per_gt,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": opt.param_groups[0]["lr"],
                    "train/epoch": epoch + i / max(1, len(loader)),
                    "train/elapsed_min": (time.monotonic() - t0) / 60,
                }, step)
            if cfg.max_minutes and (time.monotonic() - t0) / 60 > cfg.max_minutes:
                print(f"Stopping at step {step}: time budget {cfg.max_minutes} min reached", flush=True)
                return ema.ema
    return ema.ema

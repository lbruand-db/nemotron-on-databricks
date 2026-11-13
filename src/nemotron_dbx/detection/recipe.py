"""Shared stage-1 flow for the YOLOX detector recipes: base eval -> fine-tune -> fine-tuned eval -> save + tag.

Used by table-structure-v1 and graphic-elements-v1 (page-elements-v3 has the same steps inline). Each recipe
only prepares its records; this module does the rest identically for every detector.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..common import mlflow_utils as mlf
from ..common.paths import run_dir, scratch_dir
from ..yolox_train.data import AugmentConfig, DetectionDataset, DetRecord, decode_image
from ..yolox_train.trainer import TrainConfig, train
from .hf_detector import DetectorSpec, load_yolox, predict, save_weights, snapshot, wrap
from .metrics import coco_map

OVERLAY_COLORS = ["red", "orange", "gold", "green", "cyan", "gray", "blue", "purple", "magenta", "brown"]


@dataclass
class FinetuneParams:
    epochs: int = 8
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 5e-4
    warmup_steps: int = 100
    ema_decay: float = 0.9998
    scale_min: float = 0.75
    num_workers: int = 8
    max_train_minutes: float = 30.0
    eval_batch_size: int = 8
    max_labels: int = 300  # ground-truth boxes per image (tables need ~1000)
    max_dets: int = 300  # detections per image scored by mAP
    seed: int = 0


def evaluate(spec: DetectorSpec, model, exp, repo_dir, records: list[DetRecord], batch_size: int,
             max_dets: int) -> tuple[dict[str, float], list[dict]]:
    wrapper = wrap(spec, repo_dir, model, exp)
    images = [np.asarray(decode_image(r.image)) for r in records]
    preds = predict(wrapper, images, batch_size=batch_size)
    gts = [(r.boxes, r.labels) for r in records]
    return coco_map(gts, preds, [img.shape[:2] for img in images], list(exp.labels), max_dets=max_dets), preds


def overlay(record: DetRecord, pred: dict | None, labels: list[str], threshold: float, path: Path,
            classes: tuple[int, ...] | None = None) -> None:
    """Ground truth (left) | predictions above `threshold` (right), coloured by class, for visual checks."""
    from PIL import Image, ImageDraw

    img = decode_image(record.image)
    w, h = img.size
    panels = []
    for kind in ("gt", "pred"):
        im = img.copy()
        d = ImageDraw.Draw(im)
        if kind == "gt":
            items = zip(record.boxes, record.labels, strict=True)
        elif pred is not None:
            keep = pred["scores"] > threshold
            items = zip(pred["boxes"][keep] * np.array([w, h, w, h]), pred["labels"][keep], strict=True)
        else:
            items = []
        for box, lab in items:
            if classes is None or int(lab) in classes:
                d.rectangle([float(v) for v in box], outline=OVERLAY_COLORS[int(lab) % len(OVERLAY_COLORS)], width=2)
        panels.append(im)
    sheet = Image.new("RGB", (w * 2 + 10, h), "white")
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (w + 10, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def finetune_and_evaluate(
    spec: DetectorSpec,
    recipe: str,
    output_root: str,
    train_recs: list[DetRecord],
    val_recs: list[DetRecord],
    test_recs: list[DetRecord],
    p: FinetuneParams,
    test_data: str | Path,
    extra_result: dict | None = None,
    overlay_classes: tuple[int, ...] | None = None,
    n_overlays: int = 4,
    extra_images: dict[str, bytes] | None = None,
) -> dict:
    """Steps 2-5 of every detector recipe. Returns the result dict (also written to metrics.json)."""
    t0 = time.monotonic()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = run_dir(output_root)
    ev = {"batch_size": p.eval_batch_size, "max_dets": p.max_dets}

    # Base model from the pinned HF snapshot.
    repo_dir = snapshot(spec, scratch_dir("hf", spec.name))
    model, exp = load_yolox(spec, repo_dir, device=str(device))
    labels = list(exp.labels)
    base_test, base_preds = evaluate(spec, model, exp, repo_dir, test_recs, **ev)
    base = {f"test/{k}": v for k, v in base_test.items()}
    base |= {f"val/{k}": v for k, v in evaluate(spec, model, exp, repo_dir, val_recs, **ev)[0].items()}
    mlf.log_metrics(base, prefix="base/")
    print("[eval] base:", json.dumps(base, indent=1), flush=True)
    extra_preds = {}
    if extra_images:
        wrapper = wrap(spec, repo_dir, model, exp)
        extra_preds["base"] = {k: predict(wrapper, [np.asarray(decode_image(v))])[0] for k, v in extra_images.items()}

    # Fine-tune (EMA weights are returned).
    cfg = TrainConfig(
        epochs=p.epochs, batch_size=p.batch_size, grad_accum=p.grad_accum, lr=p.lr,
        backbone_lr_mult=p.backbone_lr_mult, weight_decay=p.weight_decay, warmup_steps=p.warmup_steps,
        ema_decay=p.ema_decay, num_workers=p.num_workers, seed=p.seed, max_minutes=p.max_train_minutes,
    )
    ds = DetectionDataset(train_recs, size=tuple(exp.size), augment=AugmentConfig(scale_min=p.scale_min),
                          max_labels=p.max_labels, seed=p.seed)
    t1 = time.monotonic()
    ft_model = train(model, ds, exp.num_classes, cfg, device,
                     log_fn=lambda m, step: (mlf.log_metrics(m, step=step), print(f"[train] {step} {m}", flush=True)))
    train_min = (time.monotonic() - t1) / 60

    # Fine-tuned (EMA) model on the same splits.
    ft_model.eval()
    ft_test, ft_preds = evaluate(spec, ft_model, exp, repo_dir, test_recs, **ev)
    ft = {f"test/{k}": v for k, v in ft_test.items()}
    ft |= {f"val/{k}": v for k, v in evaluate(spec, ft_model, exp, repo_dir, val_recs, **ev)[0].items()}
    mlf.log_metrics(ft, prefix="ft/")
    print("[eval] fine-tuned:", json.dumps(ft, indent=1), flush=True)

    # Visual checks: GT | prediction for a few test images (base and fine-tuned), plus any extra images.
    thr = float(exp.threshold) if hasattr(exp, "threshold") else 0.1
    for i, rec in enumerate(test_recs[:n_overlays]):
        overlay(rec, base_preds[i], labels, thr, out / "overlays" / f"test{i}_base.png", overlay_classes)
        overlay(rec, ft_preds[i], labels, thr, out / "overlays" / f"test{i}_ft.png", overlay_classes)
    extra_counts = {}
    if extra_images:
        wrapper = wrap(spec, repo_dir, ft_model, exp)
        extra_preds["ft"] = {k: predict(wrapper, [np.asarray(decode_image(v))])[0] for k, v in extra_images.items()}
        for name, data in extra_images.items():
            rec = DetRecord(data, np.zeros((0, 4), np.float32), np.zeros(0, np.int64), name)
            for which in ("base", "ft"):
                pr = extra_preds[which][name]
                overlay(rec, pr, labels, thr, out / "overlays" / f"{name}_{which}.png")
                keep = pr["scores"] > thr
                extra_counts[f"{name}/{which}"] = {labels[c]: int(n) for c, n in
                                                   zip(*np.unique(pr["labels"][keep], return_counts=True), strict=True)}
        print(f"[eval] extra images (detections above {thr}): {json.dumps(extra_counts)}", flush=True)

    weights = out / "weights" / "weights.pth"
    save_weights(ft_model, weights)
    result = {
        "weights_path": str(weights), "test_data": str(test_data), "base": base, "fine_tuned": ft,
        "extra_images": extra_counts,
        "minutes": {"train": round(train_min, 1), "eval_and_train": round((time.monotonic() - t0) / 60, 1)},
        **(extra_result or {}),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    mlf.log_artifact(str(out / "metrics.json"))
    for png in sorted((out / "overlays").glob("*.png")):
        mlf.log_artifact(str(png), "overlays")
    mlf.set_tags({"nemotron.recipe": recipe, "nemotron.weights_path": weights, "nemotron.test_data": test_data,
                  "nemotron.base_source": f"{spec.source.repo_id}@{spec.source.revision}"})
    print("TRAINING_RESULT=" + json.dumps({
        "weights_path": str(weights), "minutes": result["minutes"], "mlflow_run_id": mlf.run_id(),
        "base_test_map": base["test/map"], "ft_test_map": ft["test/map"],
    }), flush=True)
    return result

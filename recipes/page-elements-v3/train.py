"""Fine-tune nvidia/nemotron-page-elements-v3 on a DocLayNet subset; evaluate base vs fine-tuned.

Stage 1 of 3 (train -> register -> deploy). Run from the repo root:
    databricks air run -f recipes/page-elements-v3/workload.yaml --watch
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import dataset_cache_dir, run_dir, scratch_dir
from nemotron_dbx.detection import doclaynet
from nemotron_dbx.detection.hf_detector import (
    PAGE_ELEMENTS,
    load_yolox,
    predict,
    save_weights,
    snapshot,
    wrap,
)
from nemotron_dbx.detection.metrics import coco_map
from nemotron_dbx.yolox_train.data import AugmentConfig, DetectionDataset, DetRecord, decode_image
from nemotron_dbx.yolox_train.trainer import TrainConfig, train


@dataclass
class Params:
    output_root: str
    doc_categories: list[str] = field(default_factory=lambda: ["financial_reports", "manuals"])
    train_pages: int = 2000
    val_pages: int = 200
    test_pages: int = 300
    drop_picture_pages: bool = True
    epochs: int = 8
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 5e-4
    warmup_steps: int = 100
    ema_decay: float = 0.9998
    scale_min: float = 0.75
    num_workers: int = 8
    max_train_minutes: float = 25.0
    eval_batch_size: int = 16
    seed: int = 0


def evaluate(model, exp, repo_dir, records: list[DetRecord], batch_size: int) -> dict[str, float]:
    wrapper = wrap(PAGE_ELEMENTS, repo_dir, model, exp)
    images = [np.asarray(decode_image(r.image)) for r in records]
    preds = predict(wrapper, images, batch_size=batch_size)
    gts = [(r.boxes, r.labels) for r in records]
    return coco_map(gts, preds, [img.shape[:2] for img in images], list(exp.labels))


def main() -> None:
    p = build_config(Params, load_parameters())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlf.log_params({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in asdict(p).items()})
    t0 = time.monotonic()

    # 1. Data: compact DocLayNet subset, cached on the Volume and reused across runs.
    sub = doclaynet.SubsetConfig(
        doc_categories=p.doc_categories,
        max_pages={"train": p.train_pages, "validation": p.val_pages, "test": p.test_pages},
        drop_picture_pages=p.drop_picture_pages,
    )
    cache = dataset_cache_dir(p.output_root, sub.key())
    manifests = {s: doclaynet.prepare_split(s, sub, cache) for s in ("train", "validation", "test")}
    for s, m in manifests.items():
        print(f"[data] {s}: {m['pages']} pages, boxes/class {m['boxes_per_class']}", flush=True)
    train_recs = doclaynet.load_records(cache / "train.parquet")
    val_recs = doclaynet.load_records(cache / "validation.parquet")
    test_recs = doclaynet.load_records(cache / "test.parquet")
    mlf.log_metrics({f"data/{s}_pages": m["pages"] for s, m in manifests.items()})
    data_min = (time.monotonic() - t0) / 60

    # 2. Base model from the pinned HF snapshot.
    repo_dir = snapshot(PAGE_ELEMENTS, scratch_dir("hf", PAGE_ELEMENTS.name))
    model, exp = load_yolox(PAGE_ELEMENTS, repo_dir, device=str(device))
    base = {f"test/{k}": v for k, v in evaluate(model, exp, repo_dir, test_recs, p.eval_batch_size).items()}
    base |= {f"val/{k}": v for k, v in evaluate(model, exp, repo_dir, val_recs, p.eval_batch_size).items()}
    mlf.log_metrics(base, prefix="base/")
    print("[eval] base:", json.dumps(base, indent=1), flush=True)

    # 3. Fine-tune (EMA weights are returned).
    cfg = TrainConfig(
        epochs=p.epochs, batch_size=p.batch_size, grad_accum=p.grad_accum, lr=p.lr,
        backbone_lr_mult=p.backbone_lr_mult, weight_decay=p.weight_decay, warmup_steps=p.warmup_steps,
        ema_decay=p.ema_decay, num_workers=p.num_workers, seed=p.seed, max_minutes=p.max_train_minutes,
    )
    ds = DetectionDataset(train_recs, size=tuple(exp.size), augment=AugmentConfig(scale_min=p.scale_min),
                          seed=p.seed)
    t1 = time.monotonic()
    ft_model = train(model, ds, exp.num_classes, cfg, device,
                     log_fn=lambda m, step: (mlf.log_metrics(m, step=step), print(f"[train] {step} {m}", flush=True)))
    train_min = (time.monotonic() - t1) / 60

    # 4. Evaluate the fine-tuned (EMA) model on the same splits.
    ft_model.eval()
    ft = {f"test/{k}": v for k, v in evaluate(ft_model, exp, repo_dir, test_recs, p.eval_batch_size).items()}
    ft |= {f"val/{k}": v for k, v in evaluate(ft_model, exp, repo_dir, val_recs, p.eval_batch_size).items()}
    mlf.log_metrics(ft, prefix="ft/")
    print("[eval] fine-tuned:", json.dumps(ft, indent=1), flush=True)

    # 5. Persist weights + metrics on the Volume; tag the run for the registration stage.
    out = run_dir(p.output_root)
    weights = out / "weights" / "weights.pth"
    save_weights(ft_model, weights)
    result = {
        "weights_path": str(weights),
        "test_data": str(cache / "test.parquet"),
        "base": base, "fine_tuned": ft,
        "minutes": {"data": round(data_min, 1), "train": round(train_min, 1),
                    "total": round((time.monotonic() - t0) / 60, 1)},
        "manifests": manifests,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    mlf.log_artifact(str(out / "metrics.json"))
    mlf.set_tags({"nemotron.recipe": "page-elements-v3", "nemotron.weights_path": weights,
                  "nemotron.test_data": cache / "test.parquet", "nemotron.base_source":
                  f"{PAGE_ELEMENTS.source.repo_id}@{PAGE_ELEMENTS.source.revision}"})
    print("TRAINING_RESULT=" + json.dumps({k: result[k] for k in ("weights_path", "minutes")}
                                          | {"base_test_map": base["test/map"], "ft_test_map": ft["test/map"],
                                             "mlflow_run_id": mlf.run_id()}), flush=True)


if __name__ == "__main__":
    main()

"""Fine-tune the nemotron-ocr-v1 recognizer on CORD-v2 receipts; evaluate base vs fine-tuned.

Stage 1 of 3 (train -> register -> deploy). Run from the repo root:
    databricks air run -f recipes/ocr-v1/workload.yaml
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import run_dir, scratch_dir
from nemotron_dbx.ocr.build import ensure_installed


@dataclass
class Params:
    output_root: str
    wheel_cache: str
    train_items: int = 0  # 0 = all 800 CORD train receipts
    val_items: int = 0  # 0 = all 100
    test_items: int = 0  # 0 = all 100
    epochs: int = 10
    images_per_step: int = 4
    max_regions_per_step: int = 128
    lr: float = 1e-4
    quad_jitter: float = 0.06
    warmup_steps: int = 50
    max_train_minutes: float = 30.0
    eval_end_to_end: bool = True
    seed: int = 0


def main() -> None:
    p = build_config(Params, load_parameters())
    print(f"[setup] params: {asdict(p)}", flush=True)
    mlf.log_params(asdict(p))
    t0 = time.monotonic()

    # 1. nemotron-ocr package + C++/CUDA extension (wheel built once, cached on the Volume).
    install = ensure_installed(p.wheel_cache, scratch_dir("hf", "nemotron-ocr-v1"))
    print(f"[setup] nemotron-ocr installed from {install.wheel} at {time.monotonic() - t0:.0f}s", flush=True)

    from nemotron_dbx.ocr.cord import load_split
    from nemotron_dbx.ocr.recognizer import (
        OCRStack,
        OCRTrainConfig,
        eval_end_to_end,
        eval_gt_quads,
        make_checkpoint_dir,
        train_recognizer,
    )

    # 2. Data: CORD-v2 word quads + transcriptions.
    cache = scratch_dir("cord-v2")
    train = load_split("train", p.train_items or None, cache)
    val = load_split("validation", p.val_items or None, cache)
    test = load_split("test", p.test_items or None, cache)
    for name, recs in (("train", train), ("validation", val), ("test", test)):
        print(f"[data] {name}: {len(recs)} receipts, {sum(len(r.texts) for r in recs)} words", flush=True)

    # 3. Base model.
    stack = OCRStack(install.checkpoints)
    trainable = sum(stack.charset.trainable(t) for r in train for t in r.texts)
    print(f"[data] trainable train words: {trainable}", flush=True)
    base_test, base_pairs = eval_gt_quads(stack, test)
    base = {f"test/gtquad/{k}": v for k, v in base_test.items()}
    base |= {f"val/gtquad/{k}": v for k, v in eval_gt_quads(stack, val)[0].items()}
    if p.eval_end_to_end:
        base |= {f"test/e2e/{k}": v for k, v in eval_end_to_end(install.checkpoints, test).items()}
    mlf.log_metrics(base, prefix="base/")
    print("[eval] base:", json.dumps(base, indent=1), flush=True)
    print("[eval] base samples (gt -> pred):", [pp for pp in base_pairs if pp[0] != pp[1]][:15], flush=True)

    # 4. Fine-tune the recognizer (detector + relational frozen).
    cfg = OCRTrainConfig(
        epochs=p.epochs, images_per_step=p.images_per_step, max_regions_per_step=p.max_regions_per_step,
        lr=p.lr, quad_jitter=p.quad_jitter, warmup_steps=p.warmup_steps, seed=p.seed,
        max_minutes=p.max_train_minutes,
    )
    t1 = time.monotonic()
    info = train_recognizer(stack, train, cfg, log_fn=lambda m, s: (mlf.log_metrics(m, step=s),
                                                                    print(f"[train] {s} {m}", flush=True)))
    train_min = (time.monotonic() - t1) / 60

    # 5. Save + evaluate the fine-tuned recognizer on the same splits.
    out = run_dir(p.output_root)
    rec_path = out / "recognizer.pth"
    stack.save_recognizer(rec_path)
    ckpt = make_checkpoint_dir(install.checkpoints, rec_path, out / "checkpoints")
    ft_test, ft_pairs = eval_gt_quads(stack, test)
    ft = {f"test/gtquad/{k}": v for k, v in ft_test.items()}
    ft |= {f"val/gtquad/{k}": v for k, v in eval_gt_quads(stack, val)[0].items()}
    if p.eval_end_to_end:
        ft |= {f"test/e2e/{k}": v for k, v in eval_end_to_end(ckpt, test).items()}
    mlf.log_metrics(ft, prefix="ft/")
    print("[eval] fine-tuned:", json.dumps(ft, indent=1), flush=True)
    print("[eval] ft samples (gt -> pred):", [pp for pp in ft_pairs if pp[0] != pp[1]][:15], flush=True)

    sample = out / "sample_test_image.png"
    sample.write_bytes(test[0].image)
    result = {
        "checkpoints_dir": str(ckpt), "wheel": str(install.wheel), "base": base, "fine_tuned": ft,
        "train": info, "minutes": {"train": round(train_min, 1), "total": round((time.monotonic() - t0) / 60, 1)},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    mlf.log_artifact(str(out / "metrics.json"))
    mlf.set_tags({"nemotron.recipe": "ocr-v1", "nemotron.checkpoints_dir": ckpt, "nemotron.wheel": install.wheel,
                  "nemotron.sample_image": sample})
    print("TRAINING_RESULT=" + json.dumps({
        "checkpoints_dir": str(ckpt), "minutes": result["minutes"], "mlflow_run_id": mlf.run_id(),
        "base_test_cer": base["test/gtquad/cer"], "ft_test_cer": ft["test/gtquad/cer"],
        "base_test_bow_f1": base.get("test/e2e/bow_f1"), "ft_test_bow_f1": ft.get("test/e2e/bow_f1"),
    }), flush=True)


if __name__ == "__main__":
    main()

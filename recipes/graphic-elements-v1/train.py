"""Fine-tune nvidia/nemotron-graphic-elements-v1 on synthetic charts; evaluate base vs fine-tuned.

Stage 1 of 3 (train -> register -> deploy). Run from the repo root:
    databricks air run -f recipes/graphic-elements-v1/workload.yaml

The charts are rendered with matplotlib (exact boxes, no download). There is no labelled real-chart test set
yet (SPEC OQ-4), so the recipe also runs base and fine-tuned on NVIDIA's own example chart and saves
overlays, as a sanity check that fine-tuning on synthetic data doesn't break real charts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import dataset_cache_dir, scratch_dir
from nemotron_dbx.detection import charts, doclaynet
from nemotron_dbx.detection.hf_detector import GRAPHIC_ELEMENTS
from nemotron_dbx.detection.recipe import FinetuneParams, finetune_and_evaluate


@dataclass
class Params(FinetuneParams):
    output_root: str = ""
    train_charts: int = 3000
    val_charts: int = 300
    test_charts: int = 300
    data_seed: int = 0


def main() -> None:
    p = build_config(Params, load_parameters())
    if not p.output_root:
        raise SystemExit("Set parameters.output_root")
    print(f"[setup] params: {asdict(p)}", flush=True)
    mlf.log_params(asdict(p))
    t0 = time.monotonic()

    # 1. Data: synthetic charts (disjoint seeds per split), cached on the Volume.
    sub = charts.ChartSubset(train=p.train_charts, val=p.val_charts, test=p.test_charts, seed=p.data_seed)
    cache = dataset_cache_dir(p.output_root, sub.key())
    manifest = charts.prepare(sub, cache, workers=min(16, os.cpu_count() or 4))
    recs = {s: doclaynet.load_records(cache / f"{s}.parquet") for s in ("train", "validation", "test")}
    mlf.log_metrics({f"data/{s}_charts": len(r) for s, r in recs.items()})
    data_min = (time.monotonic() - t0) / 60
    example = GRAPHIC_ELEMENTS.source.file("example.png", local_dir=scratch_dir("hf", "graphic-example")).read_bytes()

    # 2-5. Base eval, fine-tune, fine-tuned eval, save + tag.
    result = finetune_and_evaluate(
        GRAPHIC_ELEMENTS, "graphic-elements-v1", p.output_root, recs["train"], recs["validation"], recs["test"], p,
        test_data=cache / "test.parquet", extra_result={"manifest": manifest, "data_minutes": round(data_min, 1)},
        extra_images={"nvidia_example": example},
    )
    print(f"[done] total {(time.monotonic() - t0) / 60:.1f} min; synthetic test mAP "
          f"{result['base']['test/map']:.3f} -> {result['fine_tuned']['test/map']:.3f}; "
          f"NVIDIA example detections {json.dumps(result['extra_images'])}", flush=True)


if __name__ == "__main__":
    main()

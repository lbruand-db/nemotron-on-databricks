"""Fine-tune nvidia/nemotron-table-structure-v1 on PubTables-1M table crops; evaluate base vs fine-tuned.

Stage 1 of 3 (train -> register -> deploy). Run from the repo root:
    databricks air run -f recipes/table-structure-v1/workload.yaml
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from nemotron_dbx.common import mlflow_utils as mlf
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import dataset_cache_dir
from nemotron_dbx.detection import pubtables
from nemotron_dbx.detection.hf_detector import TABLE_STRUCTURE
from nemotron_dbx.detection.recipe import FinetuneParams, finetune_and_evaluate


@dataclass
class Params(FinetuneParams):
    output_root: str = ""
    train_tables: int = 3000
    val_tables: int = 300
    test_tables: int = 300
    max_boxes: int = 1000
    max_labels: int = 1000
    max_dets: int = 1000


def main() -> None:
    p = build_config(Params, load_parameters())
    if not p.output_root:
        raise SystemExit("Set parameters.output_root")
    print(f"[setup] params: {asdict(p)}", flush=True)
    mlf.log_params(asdict(p))
    t0 = time.monotonic()

    # 1. Data: PubTables-1M structure subset (Val -> train/val, Test -> test), cached on the Volume.
    sub = pubtables.PubTablesSubset(train=p.train_tables, val=p.val_tables, test=p.test_tables,
                                    max_boxes=p.max_boxes)
    cache = dataset_cache_dir(p.output_root, sub.key())
    manifest = pubtables.prepare(sub, cache)
    print(f"[data] {json.dumps(manifest['splits'])} skipped {manifest['skipped']}", flush=True)
    recs = {s: pubtables.load_records(cache / f"{s}.parquet") for s in ("train", "validation", "test")}
    mlf.log_metrics({f"data/{s}_tables": len(r) for s, r in recs.items()})
    data_min = (time.monotonic() - t0) / 60

    # 2-5. Base eval, fine-tune, fine-tuned eval, save + tag. Overlays show cells only (rows/columns clutter).
    result = finetune_and_evaluate(
        TABLE_STRUCTURE, "table-structure-v1", p.output_root, recs["train"], recs["validation"], recs["test"], p,
        test_data=cache / "test.parquet", extra_result={"manifest": manifest, "data_minutes": round(data_min, 1)},
        overlay_classes=(pubtables.CELL,),
    )
    print(f"[done] total {(time.monotonic() - t0) / 60:.1f} min; test mAP "
          f"{result['base']['test/map']:.3f} -> {result['fine_tuned']['test/map']:.3f}", flush=True)


if __name__ == "__main__":
    main()

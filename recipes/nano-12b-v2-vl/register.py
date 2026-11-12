"""Stage 2: register the merged Nano VL checkpoint of a training run for Custom LLM Serving.

    databricks air run -f recipes/nano-12b-v2-vl/register.yaml --override parameters.source_training_run_id=<RUN_ID>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow

from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.lifecycle.register import register_vlm
from nemotron_dbx.vlm.data import VLSample


@dataclass
class Params:
    source_training_run_id: str
    registered_model_name: str


def main() -> None:
    p = build_config(Params, load_parameters())
    if not p.source_training_run_id:
        raise SystemExit("Set parameters.source_training_run_id (the stage-1 MLFLOW_RUN_ID)")
    tags = mlflow.MlflowClient().get_run(p.source_training_run_id).data.tags
    merged = tags.get("nemotron.merged_dir")
    if not merged or not (Path(merged) / "config.json").is_file():
        raise SystemExit(f"No merged checkpoint for run {p.source_training_run_id} (nemotron.merged_dir={merged!r})")
    result = register_vlm(
        merged_dir=merged,
        registered_model_name=p.registered_model_name,
        smoke_sample=VLSample(Path(tags["nemotron.sample_image"]).read_bytes(), {}),
        source_training_run_id=p.source_training_run_id,
        extra_tags={"nemotron.merged_dir": merged},
    )
    print("REGISTRATION_RESULT=" + json.dumps(result.__dict__), flush=True)


if __name__ == "__main__":
    main()

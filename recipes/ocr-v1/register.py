"""Stage 2: register the fine-tuned nemotron-ocr-v1 checkpoints of a training run in UC.

    databricks air run -f recipes/ocr-v1/register.yaml --override parameters.source_training_run_id=<RUN_ID>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow

from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import scratch_dir
from nemotron_dbx.lifecycle.register import register_ocr
from nemotron_dbx.ocr.build import ensure_installed


@dataclass
class Params:
    source_training_run_id: str
    registered_model_name: str
    wheel_cache: str


def main() -> None:
    p = build_config(Params, load_parameters())
    if not p.source_training_run_id:
        raise SystemExit("Set parameters.source_training_run_id (the stage-1 MLFLOW_RUN_ID)")
    tags = mlflow.MlflowClient().get_run(p.source_training_run_id).data.tags
    install = ensure_installed(p.wheel_cache, scratch_dir("hf", "nemotron-ocr-v1"))  # same cached wheel
    result = register_ocr(
        checkpoints=tags["nemotron.checkpoints_dir"],
        wheel=install.wheel,
        registered_model_name=p.registered_model_name,
        sample_images=[Path(tags["nemotron.sample_image"]).read_bytes()],
        source_training_run_id=p.source_training_run_id,
        extra_tags={"nemotron.checkpoints_dir": tags["nemotron.checkpoints_dir"]},
    )
    print("REGISTRATION_RESULT=" + json.dumps(result.__dict__), flush=True)


if __name__ == "__main__":
    main()

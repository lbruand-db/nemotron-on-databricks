"""Stage 2: package the fine-tuned weights of a training run as a pyfunc and register it in UC.

    databricks air run -f recipes/graphic-elements-v1/register.yaml --watch \
        --override parameters.source_training_run_id=<MLFLOW_RUN_ID of stage 1>
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import mlflow

from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.lifecycle.register import register_detector


@dataclass
class Params:
    source_training_run_id: str
    registered_model_name: str
    env_pack: bool = False  # see register.yaml / SPEC OQ-17


def main() -> None:
    p = build_config(Params, load_parameters())
    if not p.source_training_run_id:
        raise SystemExit("Set parameters.source_training_run_id (the stage-1 MLFLOW_RUN_ID)")
    tags = mlflow.MlflowClient().get_run(p.source_training_run_id).data.tags
    weights = tags["nemotron.weights_path"]
    test_data = tags["nemotron.test_data"]

    import pyarrow.parquet as pq

    sample = pq.read_table(test_data, columns=["image"]).slice(0, 2).to_pydict()["image"]
    result = register_detector(
        spec_name="graphic_elements_v1",
        weights=weights,
        registered_model_name=p.registered_model_name,
        sample_images=sample,
        source_training_run_id=p.source_training_run_id,
        env_pack=p.env_pack,
        extra_tags={"nemotron.weights_path": weights},
    )
    print("REGISTRATION_RESULT=" + json.dumps(result.__dict__), flush=True)


if __name__ == "__main__":
    main()

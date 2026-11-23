"""Stage 2: register Nemotron-3-Embed-8B as a classic pyfunc running vLLM in pooling mode (no `env_pack`).

The weights are the pinned Hugging Face snapshot (there is no fine-tuning), downloaded here. The smoke test runs
the pyfunc exactly as serving does and must reproduce the model card's published scores.

    databricks air run -f recipes/nemotron-3-embed-8b/register.yaml \
        --override parameters.source_eval_run_id=<MLFLOW_RUN_ID of stage 1>
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nemotron_dbx.common.hf import NEMOTRON_3_EMBED_8B
from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.common.paths import scratch_dir
from nemotron_dbx.lifecycle.register import register_embed_pyfunc

MODEL_PATTERNS = ["*.safetensors", "*.json", "LICENSE", "NOTICE"]


@dataclass
class Params:
    registered_model_name: str
    source_eval_run_id: str = ""  # optional lineage to the stage-1 eval run
    vllm_extra_args: str = ""
    vllm_env: str = ""
    smoke_gpu_budget_gib: float = 0.0  # cap the smoke test at the serving GPU's memory, e.g. 20.2 for an A10G


def main() -> None:
    p = build_config(Params, load_parameters())
    model_dir = NEMOTRON_3_EMBED_8B.snapshot(scratch_dir("hf", "nemotron-3-embed-8b"), MODEL_PATTERNS)
    result = register_embed_pyfunc(
        model_dir=model_dir,
        registered_model_name=p.registered_model_name,
        source=f"{NEMOTRON_3_EMBED_8B.repo_id}@{NEMOTRON_3_EMBED_8B.revision}",
        source_eval_run_id=p.source_eval_run_id or None,
        vllm_extra_args=tuple(p.vllm_extra_args.split()),
        vllm_env=tuple(p.vllm_env.split()),
        smoke_gpu_budget_gib=p.smoke_gpu_budget_gib,
    )
    print("REGISTRATION_RESULT=" + json.dumps(result.__dict__), flush=True)


if __name__ == "__main__":
    main()

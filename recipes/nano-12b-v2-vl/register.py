"""Stage 2: register the merged Nano VL checkpoint of a training run.

`serving: pyfunc` (default) registers a classic pyfunc that runs vLLM inside the serving container, with pinned
pip requirements. `serving: custom_llm` registers for Custom LLM Serving (vLLM entrypoint + `env_pack`), which is
blocked on the test workspace (SPEC OQ-5).

    databricks air run -f recipes/nano-12b-v2-vl/register.yaml --override parameters.source_training_run_id=<RUN_ID>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nemotron_dbx.common.params import build_config, load_parameters
from nemotron_dbx.lifecycle.register import register_vlm, register_vlm_pyfunc
from nemotron_dbx.vlm.data import VLSample
from nemotron_dbx.vlm.merge import MERGED_DIR_TAG, merged_checkpoint


@dataclass
class Params:
    source_training_run_id: str
    registered_model_name: str
    serving: str = "pyfunc"  # pyfunc | custom_llm
    vllm_extra_args: str = ""  # appended to the serving (and smoke-test) vLLM command, e.g. "--enforce-eager"
    vllm_env: str = ""  # space-separated NAME=VALUE for the vLLM process, e.g. "VLLM_TEST_FORCE_FP8_MARLIN=1"
    smoke_gpu_budget_gib: float = 0.0  # cap the smoke test at the serving GPU's memory, e.g. 20.2 for an A10G


def main() -> None:
    p = build_config(Params, load_parameters())
    merged, tags = merged_checkpoint(p.source_training_run_id)
    if p.serving not in ("pyfunc", "custom_llm"):
        raise SystemExit(f"serving must be pyfunc or custom_llm, got {p.serving!r}")
    register = register_vlm_pyfunc if p.serving == "pyfunc" else register_vlm
    result = register(
        merged_dir=merged,
        registered_model_name=p.registered_model_name,
        smoke_sample=VLSample(Path(tags["nemotron.sample_image"]).read_bytes(), {}),
        source_training_run_id=p.source_training_run_id,
        extra_tags={MERGED_DIR_TAG: str(merged)},
        vllm_extra_args=tuple(p.vllm_extra_args.split()),
        vllm_env=tuple(p.vllm_env.split()),
        smoke_gpu_budget_gib=p.smoke_gpu_budget_gib,
    )
    print("REGISTRATION_RESULT=" + json.dumps(result.__dict__), flush=True)


if __name__ == "__main__":
    main()

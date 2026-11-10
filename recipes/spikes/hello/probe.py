"""M0 spike: verify AIR runs, shared-code shipping (OQ-1b), GPU, and Volume access."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import nemotron_dbx

report = {
    "nemotron_dbx_version": nemotron_dbx.__version__,
    "nemotron_dbx_file": nemotron_dbx.__file__,
    "python": sys.version,
    "platform": platform.platform(),
    "code_source_path": os.environ.get("CODE_SOURCE_PATH"),
    "code_source_listing": sorted(
        str(p.relative_to(os.environ["CODE_SOURCE_PATH"]))
        for p in Path(os.environ["CODE_SOURCE_PATH"]).rglob("*")
        if p.is_file()
    ),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("DATABRICKS_ACCELERATOR", "MLFLOW_RUN_ID", "CUDA", "NVIDIA_VISIBLE"))},
}
try:
    import torch

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as exc:  # noqa: BLE001 - probe should report, not crash
    report["torch_error"] = repr(exc)

for pkg in ("transformers", "mlflow", "databricks.sdk", "vllm", "peft"):
    try:
        mod = __import__(pkg, fromlist=["__version__"])
        report[f"pkg:{pkg}"] = getattr(mod, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        report[f"pkg:{pkg}"] = f"missing ({type(exc).__name__})"

output_root = os.environ.get("OUTPUT_ROOT")
if output_root:
    out = Path(output_root) / os.environ.get("MLFLOW_RUN_ID", "no-run-id")
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(report, indent=2))
    report["volume_write"] = str(out / "probe.json")

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("PROBE_REPORT=" + json.dumps(report, indent=2))

"""Storage layout (SPEC §5.1)."""

from __future__ import annotations

import os
from pathlib import Path


def scratch_dir(*parts: str) -> Path:
    """Node-local scratch. On AIR `/tmp` is the large local disk (`/local_disk0` does not exist)."""
    root = Path(os.environ.get("NEMOTRON_SCRATCH", "/tmp/nemotron"))
    path = root.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(output_root: str | os.PathLike, run_id: str | None = None) -> Path:
    """Attempt-specific durable output directory: `<output_root>/<MLFLOW_RUN_ID>`."""
    run_id = run_id or os.environ.get("MLFLOW_RUN_ID") or "local"
    return Path(output_root) / run_id


def dataset_cache_dir(output_root: str | os.PathLike, name: str) -> Path:
    """Run-independent dataset cache: `<output_root>/data/<name>` (reused across runs)."""
    return Path(output_root) / "data" / name

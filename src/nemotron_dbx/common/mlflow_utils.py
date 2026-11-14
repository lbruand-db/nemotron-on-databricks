"""Log into the MLflow run that AIR creates for each attempt (`MLFLOW_RUN_ID`)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from . import dist


def _client():
    import mlflow

    return mlflow.MlflowClient()


def run_id() -> str | None:
    return os.environ.get("MLFLOW_RUN_ID")


def log_metrics(metrics: Mapping[str, float], step: int | None = None, prefix: str = "",
                target_run_id: str | None = None) -> None:
    """One `log_batch` into this attempt's run, or into `target_run_id` (e.g. the stage-1 run being evaluated)."""
    rid = target_run_id or run_id()
    if not rid or not dist.is_main():
        return
    import time

    from mlflow.entities import Metric

    ts = int(time.time() * 1000)
    batch = [
        Metric(key=_safe_key(f"{prefix}{k}"), value=float(v), timestamp=ts, step=step or 0)
        for k, v in metrics.items()
        if v is not None
    ]
    if batch:
        _client().log_batch(rid, metrics=batch)


def log_params(params: Mapping[str, Any]) -> None:
    rid = run_id()
    if not rid or not dist.is_main():
        return
    from mlflow.entities import Param

    _client().log_batch(rid, params=[Param(_safe_key(k), str(v)[:6000]) for k, v in params.items()])


def set_tags(tags: Mapping[str, Any]) -> None:
    rid = run_id()
    if not rid or not dist.is_main():
        return
    client = _client()
    for k, v in tags.items():
        client.set_tag(rid, k, str(v))


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    rid = run_id()
    if not rid or not dist.is_main():
        return
    _client().log_artifact(rid, local_path, artifact_path)


def _safe_key(key: str) -> str:
    # MLflow keys allow alphanumerics, underscores, dashes, periods, spaces, slashes.
    return "".join(c if c.isalnum() or c in "_-. /" else "_" for c in key)

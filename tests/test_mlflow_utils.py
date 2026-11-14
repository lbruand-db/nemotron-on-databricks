import sys
from types import ModuleType, SimpleNamespace

from nemotron_dbx.common import mlflow_utils as mlf


def test_log_metrics_target_run_and_safe_keys(monkeypatch):
    entities = ModuleType("mlflow.entities")  # mlflow isn't needed for unit tests
    entities.Metric = lambda key, value, timestamp, step: SimpleNamespace(key=key, value=value)
    monkeypatch.setitem(sys.modules, "mlflow", ModuleType("mlflow"))
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    calls = []
    monkeypatch.setattr(mlf, "_client", lambda: SimpleNamespace(log_batch=lambda rid, metrics: calls.append(
        (rid, sorted(m.key for m in metrics)))))
    monkeypatch.setattr(mlf.dist, "is_main", lambda: True)
    monkeypatch.setenv("MLFLOW_RUN_ID", "this-run")
    mlf.log_metrics({"test/field_f1": 0.8, "skip": None}, prefix="ft_fp8+marlin/")
    mlf.log_metrics({"test/field_f1": 0.8}, prefix="ft_fp8+marlin/", target_run_id="stage1")
    assert calls == [("this-run", ["ft_fp8_marlin/test/field_f1"]), ("stage1", ["ft_fp8_marlin/test/field_f1"])]

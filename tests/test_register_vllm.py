"""_register_vllm_pyfunc end to end with a fake MLflow (no Databricks, mlflow or vLLM needed)."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nemotron_dbx.lifecycle import register
from nemotron_dbx.vlm.vllm_server import VLLMConfig


class FakeMlflow(ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.calls: dict = {"tags": {}, "source_tags": {}}
        outer = self

        class Run:
            info = SimpleNamespace(run_id="packaging-run")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def log_model(**kw):
            outer.calls["log_model"] = kw
            return SimpleNamespace(model_uri="models:/m-1")

        self.set_registry_uri = lambda uri: outer.calls.__setitem__("registry", uri)
        self.start_run = lambda run_id=None: Run()
        self.pyfunc = SimpleNamespace(log_model=log_model)
        self.register_model = lambda uri, name: SimpleNamespace(version=7)
        self.set_tags = lambda tags: outer.calls["tags"].update(tags)
        self.MlflowClient = lambda: SimpleNamespace(
            set_tag=lambda run, key, value: outer.calls["source_tags"].__setitem__((run, key), value))


class FakeModel:
    loaded_with: dict = {}

    def load_context(self, context):
        FakeModel.loaded_with = {k: json.loads(Path(v).read_text()) if k == "config" else v
                                 for k, v in context.artifacts.items()}
        self.server = SimpleNamespace(__exit__=lambda *a: FakeModel.loaded_with.update(stopped=True))


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("DATABRICKS_ACCELERATOR", "GPU_1xA10")
    monkeypatch.setattr(register, "_pip_requirements", lambda **kw: ["vllm==0.24.0"])
    return fake


def test_shared_registration_path(fake_mlflow, tmp_path):
    smoke_seen = {}

    def smoke(model):
        smoke_seen["model"] = model
        return {"signature": "SIG", "items": 3, "summary": "ok", "tags": {"nemotron.smoke_max_abs_err": "0.0007"}}

    result = register._register_vllm_pyfunc(
        model_cls=FakeModel, model_dir=tmp_path, vllm_cfg=VLLMConfig(max_model_len=8192), port=3081,
        registered_model_name="c.s.m", smoke=smoke, spec="nemotron_3_embed_8b", source_run_id="eval-run",
        source_run_tag="nemotron.source_eval_run_id", config_extra={"source": "repo@rev"},
        extra_tags={"x": "y"}, smoke_gpu_budget_gib=0,
    )
    assert result.model_version == "7" and result.model_uri == "models:/m-1" and result.smoke_items == 3
    cfg = FakeModel.loaded_with["config"]
    assert cfg["port"] == 3081 and cfg["source"] == "repo@rev" and cfg["vllm"]["max_model_len"] == 8192
    assert FakeModel.loaded_with["stopped"] and isinstance(smoke_seen["model"], FakeModel)
    logged = fake_mlflow.calls["log_model"]
    assert logged["signature"] == "SIG" and logged["pip_requirements"] == ["vllm==0.24.0"]
    assert set(logged["artifacts"]) == {"model_dir", "config"}
    tags = fake_mlflow.calls["tags"]
    assert tags["nemotron.spec"] == "nemotron_3_embed_8b" and tags["nemotron.registered_version"] == "7"
    assert tags["nemotron.source_eval_run_id"] == "eval-run" and tags["nemotron.smoke_max_abs_err"] == "0.0007"
    assert tags["x"] == "y" and fake_mlflow.calls["registry"] == "databricks-uc"
    assert fake_mlflow.calls["source_tags"][("eval-run", "nemotron.packaging_run_id")] == "packaging-run"


def test_smoke_failure_stops_vllm_and_registers_nothing(fake_mlflow, tmp_path):
    def smoke(model):
        raise RuntimeError("wrong answers")

    with pytest.raises(RuntimeError, match="wrong answers"):
        register._register_vllm_pyfunc(model_cls=FakeModel, model_dir=tmp_path, vllm_cfg=VLLMConfig(), port=3080,
                                       registered_model_name="c.s.m", smoke=smoke, spec="s")
    assert FakeModel.loaded_with["stopped"] and "log_model" not in fake_mlflow.calls


def test_requires_ai_runtime(fake_mlflow, monkeypatch, tmp_path):
    monkeypatch.delenv("DATABRICKS_ACCELERATOR")
    with pytest.raises(RuntimeError, match="AI Runtime"):
        register._register_vllm_pyfunc(model_cls=FakeModel, model_dir=tmp_path, vllm_cfg=VLLMConfig(), port=3080,
                                       registered_model_name="c.s.m", smoke=lambda m: {}, spec="s")


def test_register_embed_pyfunc_wrapper(fake_mlflow, monkeypatch, tmp_path):
    import numpy as np

    from nemotron_dbx.embed import server, serving

    u, sv, vt = np.linalg.svd(server.CARD_SCORES.astype(np.float64))
    q, d = (u * np.sqrt(sv)), (vt.T * np.sqrt(sv))  # dot products reproduce the card exactly

    class FakeEmbed(FakeModel):
        def predict(self, context, df, params=None):
            rows = [*q, *d][: len(df)]
            dims = params["dimensions"]
            return [{"embedding": (list(r) + [0.0] * dims)[:dims]} for r in rows]

    monkeypatch.setattr(serving, "VLLMEmbedModel", FakeEmbed)
    monkeypatch.setattr(serving, "model_signature", lambda: "EMBED_SIG")
    result = register.register_embed_pyfunc(tmp_path, "c.s.embed", "nvidia/Nemotron-3-Embed-8B-BF16@rev",
                                            source_eval_run_id="eval-1")
    assert result.smoke_items == 8 and fake_mlflow.calls["log_model"]["signature"] == "EMBED_SIG"
    tags = fake_mlflow.calls["tags"]
    assert tags["nemotron.base_source"].endswith("@rev") and float(tags["nemotron.smoke_max_abs_err"]) < 1e-4
    assert FakeModel.loaded_with["config"]["vllm"]["max_model_len"] == 8192  # EMBED_VLLM, not the chat default

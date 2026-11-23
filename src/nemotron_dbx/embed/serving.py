"""Classic pyfunc serving Nemotron-3-Embed (vLLM pooling server in the container, no `env_pack`, SPEC OQ-5/OQ-17).

    Request:  {"dataframe_records": [{"text": "...", "id": "d1", "input_type": "document"},
                                     {"text": "...", "id": "q1", "input_type": "query"}],
               "params": {"dimensions": 1024}}
    Response: {"predictions": [{"id": "d1", "input_type": "document", "dimensions": 1024,
                                "embedding": [...]}, ...]}

`input_type` is "query" (prefix `query: `) or "document" (prefix `passage: `, the default). `dimensions`
(default 4096) truncates and re-normalizes, for vector indexes with a dimension limit (Lakebase pgvector indexes
take at most 2,000 `vector` / 4,000 `halfvec` dimensions; Databricks AI Search takes 4,096).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - mlflow is only needed at registration/serving time
    PythonModel = object  # type: ignore[misc,assignment]

PYFUNC_PORT = 3081
SERVED_NAME = "nemotron-3-embed"
SERVING_PARAMS = {"dimensions": 4096}


def _value(rec: dict, key: str, default: Any) -> Any:
    v = rec.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):  # a missing DataFrame column comes through as NaN
        return default
    return v


def parse_records(records: list[dict]) -> list[tuple[Any, str, str]]:
    """-> [(id, input_type, text)]; validates before anything is sent to vLLM."""
    out = []
    for i, rec in enumerate(records):
        text = _value(rec, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {i}: 'text' must be a non-empty string")
        input_type = str(_value(rec, "input_type", "document")).lower()
        if input_type not in ("query", "document"):
            raise ValueError(f"record {i}: input_type must be 'query' or 'document', got {input_type!r}")
        out.append((_value(rec, "id", i), input_type, text))
    return out


def parse_dimensions(params: dict[str, Any] | None) -> int:
    """`dimensions` param, validated before any embedding work (a bad value is the caller's error)."""
    from .server import FULL_DIM

    value = {**SERVING_PARAMS, **(params or {})}["dimensions"]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"dimensions must be an integer, got {value!r}")
    try:
        dims = int(value)
    except ValueError:
        raise ValueError(f"dimensions must be an integer, got {value!r}") from None
    if dims != float(value) or not 0 < dims <= FULL_DIM:
        raise ValueError(f"dimensions must be an integer in 1..{FULL_DIM}, got {value!r}")
    return dims


def model_signature():
    """Explicit signature. Inferring it from the smoke DataFrame would make `id` and `input_type` required and type
    `id` as a string, so `{"text": ...}` alone or integer ids would be rejected before `predict` runs. Unity
    Catalog also requires an output schema."""
    from mlflow.models import ModelSignature
    from mlflow.types.schema import AnyType, Array, ColSpec, DataType, ParamSchema, ParamSpec, Schema

    inputs = Schema([ColSpec("string", "text"), ColSpec("string", "input_type", required=False),
                     ColSpec(AnyType(), "id", required=False)])
    outputs = Schema([ColSpec(AnyType(), "id"), ColSpec("string", "input_type"), ColSpec("long", "dimensions"),
                      ColSpec(Array(DataType.float), "embedding")])
    return ModelSignature(inputs=inputs, outputs=outputs,
                          params=ParamSchema([ParamSpec("dimensions", "long", 4096)]))


class VLLMEmbedModel(PythonModel):
    def load_context(self, context) -> None:
        from ..vlm.vllm_server import VLLMServer, vllm_config_from_json
        from .server import vllm_embed_command

        cfg = json.loads(Path(context.artifacts["config"]).read_text())
        self.max_tokens = cfg["vllm"]["max_model_len"]
        self.server = VLLMServer(
            context.artifacts["model_dir"], served_model_name=SERVED_NAME, port=cfg.get("port", PYFUNC_PORT),
            cfg=vllm_config_from_json(cfg["vllm"]), log_path=None, reuse_running=True,
            startup_timeout_s=cfg.get("startup_timeout_s", 1800), command_builder=vllm_embed_command,
        )
        self.server.__enter__()  # runs for the life of the serving process

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        from ..detection.serving import _records
        from .server import embed_texts

        dims = parse_dimensions(params)  # before any GPU work
        recs = parse_records(list(_records(model_input)))
        results: list[dict | None] = [None] * len(recs)
        for input_type in ("query", "document"):
            idx = [i for i, (_, t, _) in enumerate(recs) if t == input_type]
            if not idx:
                continue
            emb = embed_texts(self.server.port, SERVED_NAME, [recs[i][2] for i in idx], input_type, dims=dims,
                              max_tokens=self.max_tokens)
            for i, row in zip(idx, emb, strict=True):
                results[i] = {"id": recs[i][0], "input_type": input_type, "dimensions": dims,
                              "embedding": row.tolist()}
        return results

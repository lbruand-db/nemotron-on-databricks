"""Serving models for Nano VL (SPEC §5.4).

Two ways to serve the merged checkpoint:

- `EntrypointModel`: the **Custom LLM Serving** placeholder. Serving runs `metadata["entrypoint"]` (a vLLM OpenAI
  server on port 8080, started in the model's artifacts folder) and never calls `predict()`. It needs
  `env_pack`, which is blocked on the test workspace (SPEC OQ-5).
- `VLLMChatModel`: a **classic pyfunc** with pinned pip requirements (no `env_pack`). `load_context` starts the
  same vLLM OpenAI server (same `vllm_command`) as a subprocess. `predict` sends each record to it as the
  CORD extraction chat. Contract (same shape as the detector/OCR pyfuncs):

    Request:  {"dataframe_records": [{"image": "<base64>", "id": "r1", "instruction": "<optional>"}],
               "params": {"max_tokens": 1536}}
    Response: {"predictions": [{"id": "r1", "text": "<raw answer>", "fields": {...} | null,
               "json_valid": true, "usage": {...}, "latency_s": 2.9}]}
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

try:
    from mlflow.pyfunc import PythonModel
    from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel
except ImportError:  # pragma: no cover - mlflow is only needed at registration/serving time
    ChatModel = PythonModel = object  # type: ignore[misc,assignment]
    ChatCompletionResponse = None  # type: ignore[assignment]

ARTIFACT_NAME = "nano_vl_ft"  # folder name of the merged weights inside the model's artifacts
PYFUNC_PORT = 3080
SERVING_PARAMS = {"max_tokens": 1536}


class EntrypointModel(ChatModel):
    """Placeholder; Custom LLM Serving starts the metadata entrypoint."""

    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


def _record_image(rec: dict) -> bytes:
    img = rec["image"]
    return base64.b64decode(img) if isinstance(img, str) else bytes(img)


def _instruction(rec: dict) -> str | None:
    value = rec.get("instruction")  # a missing DataFrame column comes through as NaN
    return value if isinstance(value, str) and value.strip() else None


class VLLMChatModel(PythonModel):
    """Classic pyfunc: vLLM OpenAI server in a subprocess, CORD extraction per record."""

    def load_context(self, context) -> None:
        from .vllm_server import VLLMConfig, VLLMServer

        cfg = json.loads(Path(context.artifacts["config"]).read_text())
        vllm_cfg = VLLMConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["vllm"].items()})
        self.server = VLLMServer(
            context.artifacts["model_dir"], served_model_name="nano-vl", port=cfg.get("port", PYFUNC_PORT),
            cfg=vllm_cfg, log_path=None, reuse_running=True,
            startup_timeout_s=cfg.get("startup_timeout_s", 1800),
        )
        self.server.__enter__()  # runs for the life of the serving process

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        from ..detection.serving import _record_id, _records
        from .data import INSTRUCTION, VLSample, openai_messages
        from .metrics import parse_json_output

        params = {**SERVING_PARAMS, **(params or {})}
        recs = list(_records(model_input))
        convs = [
            openai_messages(VLSample(_record_image(r), {}).data_url(), _instruction(r) or INSTRUCTION)
            for r in recs
        ]
        outs = self.server.batch_chat(convs, concurrency=max(1, min(16, len(convs))),
                                      max_tokens=int(params["max_tokens"]))
        results = []
        for i, (rec, out) in enumerate(zip(recs, outs, strict=True)):
            fields = parse_json_output(out["text"])
            results.append({
                "id": _record_id(rec, i), "text": out["text"], "fields": fields, "json_valid": fields is not None,
                "usage": out.get("usage", {}), "latency_s": out["latency_s"],
                **({"error": out["error"]} if "error" in out else {}),
            })
        return results

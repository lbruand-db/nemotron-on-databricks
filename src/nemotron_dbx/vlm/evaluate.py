"""Evaluate a Nano VL checkpoint on CORD with vLLM, through the same OpenAI chat path as the endpoint."""

from __future__ import annotations

import statistics
import threading
import time
from pathlib import Path

from .data import VLSample, openai_messages
from .metrics import extraction_metrics
from .vllm_server import VLLMConfig, VLLMServer


class _Progress:
    """Thread-safe 'n/total' printer (batch_chat calls back from worker threads)."""

    def __init__(self, name: str, total: int, every: int = 25):
        self.name, self.total, self.every, self.done = name, total, every, 0
        self._lock = threading.Lock()

    def __call__(self, _index: int) -> None:
        with self._lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                print(f"[eval] {self.name}: {self.done}/{self.total}", flush=True)


def evaluate_with_vllm(
    model_dir: str | Path,
    splits: dict[str, list[VLSample]],
    log_path: str | Path,
    cfg: VLLMConfig | None = None,
    concurrency: int = 16,
    max_tokens: int = 1536,
) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """One vLLM server, every split. Returns ({"<split>/<metric>": value}, {"<split>": [prediction records]})."""
    metrics: dict[str, float] = {}
    records: dict[str, list[dict]] = {}
    with VLLMServer(str(model_dir), cfg=cfg or VLLMConfig(), log_path=Path(log_path)) as server:
        metrics["vllm_startup_s"] = round(server.startup_s)
        for name, samples in splits.items():
            t0 = time.monotonic()
            outs = server.batch_chat([openai_messages(s.data_url()) for s in samples], concurrency=concurrency,
                                     max_tokens=max_tokens, on_done=_Progress(name, len(samples)))
            wall = time.monotonic() - t0
            m = extraction_metrics([o["text"] for o in outs], [s.target for s in samples])
            m["errors"] = sum("error" in o for o in outs)
            m["truncated"] = sum(o.get("finish_reason") == "length" for o in outs)
            m["latency_p50_s"] = statistics.median(o["latency_s"] for o in outs) if outs else 0.0
            m["receipts_per_s"] = len(samples) / wall if wall else 0.0
            metrics |= {f"{name}/{k}": round(float(v), 5) for k, v in m.items()}
            records[name] = [
                {"id": s.sample_id, "gt": s.target, "pred_text": o["text"], **{k: o[k] for k in o if k != "text"}}
                for s, o in zip(samples, outs, strict=True)
            ]
            print(f"[eval] {name}: {m}", flush=True)
    return metrics, records

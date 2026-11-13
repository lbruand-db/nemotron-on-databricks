"""vLLM OpenAI server for Nano VL: one command builder for eval, the registration smoke test and serving.

The Custom LLM Serving entrypoint must match the command that was tested (docs), so all three call
`vllm_command`, and only the port and model path differ.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VLLMConfig:
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    # Hybrid Mamba models keep per-sequence SSM state; bounding concurrency keeps startup memory predictable
    # on a 48 GB L40S (GPU_LARGE).
    max_num_seqs: int = 16
    extra_args: tuple[str, ...] = ()
    # Extra NAME=VALUE environment for the vLLM process, e.g. ("VLLM_TEST_FORCE_FP8_MARLIN=1",) to run the A10G
    # FP8 kernels (Marlin, W8A16) on an H100 smoke test.
    env: tuple[str, ...] = ()


def vllm_command(model: str, served_model_name: str, port: int, cfg: VLLMConfig | None = None,
                 python: str = "python") -> list[str]:
    cfg = cfg or VLLMConfig()
    return [
        "env", "VLLM_USE_FLASHINFER_SAMPLER=0", "HF_HUB_OFFLINE=1",  # no curand.h on AIR; no Hub at serving time
        *cfg.env,
        python, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", served_model_name,
        "--host", "0.0.0.0", "--port", str(port),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--video-pruning-rate", "0",
        "--max-model-len", str(cfg.max_model_len),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        *cfg.extra_args,
    ]


@dataclass
class VLLMServer:
    """Start a local vLLM server (context manager); `chat()` sends one OpenAI chat request.

    `log_path=None` streams vLLM's output to this process's stdout (Model Serving shows it in the endpoint logs).
    With `reuse_running`, a healthy server already on `port` is reused instead of starting a second one. A file
    lock serialises the check-and-start, in case the serving container runs several workers.
    """

    model: str
    served_model_name: str = "nano-vl"
    port: int = 3080  # AIR allows 3000-3999 for local servers
    cfg: VLLMConfig = field(default_factory=VLLMConfig)
    log_path: Path | None = Path("/tmp/vllm.log")
    startup_timeout_s: int = 1800
    cwd: str | None = None
    reuse_running: bool = False
    startup_s: float = 0.0

    def healthy(self) -> bool:
        import requests

        try:
            return requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2).ok
        except requests.RequestException:
            return False

    def __enter__(self) -> VLLMServer:
        self._proc, self._log = None, None
        if not self.reuse_running:
            return self._start()
        import fcntl

        with open(f"/tmp/nemotron_vllm_{self.port}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.healthy():
                print(f"[vllm] reusing the server already running on port {self.port}", flush=True)
                return self
            return self._start()

    def _start(self) -> VLLMServer:
        cmd = vllm_command(self.model, self.served_model_name, self.port, self.cfg, python=sys.executable)
        print("[vllm] " + " ".join(cmd), flush=True)
        t0 = time.monotonic()
        out = None
        if self.log_path is not None:
            self._log = self.log_path.open("w")
            out = self._log
        self._proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if out else None, cwd=self.cwd,
                                      env={**os.environ})
        deadline = t0 + self.startup_timeout_s
        while not self.healthy():
            if self._proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self._proc.returncode}:\n{self.tail()}")
            if time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise TimeoutError(f"vLLM not healthy after {self.startup_timeout_s}s:\n{self.tail()}")
            time.sleep(5)
        self.startup_s = time.monotonic() - t0
        print(f"[vllm] ready after {self.startup_s:.0f}s", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._log is not None:
            self._log.close()

    def tail(self, n: int = 12000) -> str:
        if self.log_path is None or not self.log_path.exists():
            return "(vLLM output went to stdout)"
        return self.log_path.read_text(errors="replace")[-n:]

    def chat(self, messages: list[dict], max_tokens: int = 1536, temperature: float = 0.0) -> dict:
        import requests

        body = {"model": self.served_model_name, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        r = requests.post(f"http://127.0.0.1:{self.port}/v1/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        return r.json()

    def batch_chat(self, conversations: Sequence[list[dict]], concurrency: int = 16, max_tokens: int = 1536,
                   on_done: Callable[[int], None] | None = None) -> list[dict]:
        """Concurrent requests (vLLM batches them); results keep the input order."""

        def one(i_msgs):
            i, msgs = i_msgs
            t = time.monotonic()
            try:
                out = self.chat(msgs, max_tokens=max_tokens)
                res = {"text": out["choices"][0]["message"]["content"] or "", "usage": out.get("usage", {}),
                       "finish_reason": out["choices"][0].get("finish_reason")}
            except Exception as exc:  # noqa: BLE001 - one failed request must not lose the whole eval
                res = {"text": "", "error": repr(exc)[:500]}
            res["latency_s"] = round(time.monotonic() - t, 2)
            if on_done:
                on_done(i)
            return res

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, enumerate(conversations)))

"""M0 spike (c): can vLLM in databricks_ai_v6 serve Nemotron Nano 12B v2 VL with an image request?"""

import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset

MODEL = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16")
PORT = 8000
SCRATCH = Path(os.environ.get("SCRATCH_DIR", "/tmp"))
SCRATCH.mkdir(parents=True, exist_ok=True)
LOG = SCRATCH / "vllm.log"

cmd = [
    sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--served-model-name", "nano-vl",
    "--trust-remote-code",
    "--dtype", "bfloat16",
    "--video-pruning-rate", "0",
    "--max-model-len", "16384",
    "--gpu-memory-utilization", "0.85",
    "--port", str(PORT),
]
print("CMD:", " ".join(cmd), flush=True)
t0 = time.monotonic()
log = LOG.open("w")
server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def tail(n=15000):
    return LOG.read_text(errors="replace")[-n:]


try:
    deadline = time.monotonic() + 1500
    while True:
        if server.poll() is not None:
            print("VLLM_EXITED\n" + tail(), flush=True)
            sys.exit(1)
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            print("VLLM_TIMEOUT\n" + tail(), flush=True)
            sys.exit(1)
        time.sleep(5)
    startup_s = time.monotonic() - t0
    print(f"VLLM_READY after {startup_s:.0f}s", flush=True)

    ds = load_dataset("naver-clova-ix/cord-v2", split="test")
    results = []
    for i in range(3):
        row = ds[i]
        buf = io.BytesIO()
        row["image"].convert("RGB").save(buf, format="JPEG")
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {
            "model": "nano-vl",
            "messages": [
                {"role": "system", "content": "/no_think"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract the menu items (name, quantity, price) and the total from this receipt as JSON."},
                ]},
            ],
            "max_tokens": 512,
            "temperature": 0,
        }
        t = time.monotonic()
        r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        out = r.json()
        results.append({
            "latency_s": round(time.monotonic() - t, 2),
            "usage": out.get("usage"),
            "answer": out["choices"][0]["message"]["content"],
            "gt": json.loads(row["ground_truth"])["gt_parse"],
        })
    print("SPIKE_RESULT=" + json.dumps({"startup_s": round(startup_s), "results": results}, indent=2, ensure_ascii=False), flush=True)
    print("VLLM_LOG_HEAD\n" + LOG.read_text(errors="replace")[:6000], flush=True)
finally:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()

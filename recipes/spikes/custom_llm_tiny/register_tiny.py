"""M4 isolation spike: does Custom LLM Serving (vLLM entrypoint + env_pack) start on this workspace at all?

Same pattern and flags as the ai-runtime-cookbook `serving/sft-qwen2.5-0.5b-lora-serving/register_model.py`, with
the base Qwen2.5-0.5B-Instruct (no training). If this endpoint also restarts silently after "Downloads completed",
the Nano VL failure is the platform / env_pack, not the model's startup time.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import mlflow
import requests
from huggingface_hub import snapshot_download
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel

REPO, REV = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
NAME = os.environ.get("REGISTERED_MODEL_NAME", "am_rag_fevm_catalog.nemotron.spike_custom_llm_qwen05b")
ART = "qwen05b"


class EntrypointModel(ChatModel):
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


def cmd(port, model, served):
    return ["env", "VLLM_USE_FLASHINFER_SAMPLER=0", "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", served, "--host", "0.0.0.0", "--port", str(port),
            "--dtype", "float16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.7"]


def main():
    import ninja

    os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + os.environ["PATH"]
    os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
    mlflow.set_registry_uri("databricks-uc")
    work = Path("/tmp/spike_tiny")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    snapshot_download(REPO, revision=REV, local_dir=str(work / ART))
    os.chdir(work)
    served = NAME.rsplit(".", 1)[-1]
    log = open(work / "vllm.log", "w")
    server = subprocess.Popen(cmd(3080, ART, served), stdout=log, stderr=subprocess.STDOUT)
    t0 = time.monotonic()
    try:
        while not _healthy():
            if server.poll() is not None or time.monotonic() - t0 > 600:
                raise RuntimeError((work / "vllm.log").read_text()[-5000:])
            time.sleep(3)
        r = requests.post("http://127.0.0.1:3080/v1/chat/completions", timeout=60, json={
            "model": served, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16})
        print(f"local vLLM ok after {time.monotonic() - t0:.0f}s: {r.json()['choices'][0]['message']['content']!r}")
    finally:
        server.terminate()
        server.wait(timeout=30)
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        info = mlflow.pyfunc.log_model(
            name=served, python_model=EntrypointModel(), artifacts={"model_dir": ART},
            metadata={"task": "llm/v1/chat", "entrypoint": " ".join(cmd(8080, ART, served))},
            extra_pip_requirements=[f"mlflow=={mlflow.__version__}", "ninja==1.11.1.4"],
        )
        v = mlflow.register_model(info.model_uri, NAME, env_pack="databricks_model_serving")
    print("REGISTRATION_RESULT=" + json.dumps({"model_name": NAME, "model_version": str(v.version)}), flush=True)


def _healthy():
    try:
        return requests.get("http://127.0.0.1:3080/health", timeout=2).ok
    except requests.RequestException:
        return False


if __name__ == "__main__":
    main()

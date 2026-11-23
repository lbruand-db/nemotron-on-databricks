"""vLLM embedding server for Nemotron-3-Embed, and the client calls the eval and the pyfunc share.

The model is a bidirectional Ministral-3 encoder with mean pooling (sentence-transformers `1_Pooling` config) and
L2-normalized outputs. Retrieval needs the prompts from `config_sentence_transformers.json`: `query: ` for
queries, `passage: ` for documents. `/v1/embeddings` doesn't add them, so `embed_texts` does.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..vlm.vllm_server import VLLMConfig

PREFIXES = {"query": "query: ", "document": "passage: "}
FULL_DIM = 4096
# Retrieval inputs are short (the card evaluates at 4,096 tokens); 8,192 keeps vLLM's memory reservation small
# on a 24 GB A10G. The model itself accepts 32,768.
EMBED_VLLM = VLLMConfig(max_model_len=8192, max_num_seqs=64)

# Sanity check against the model card's "vLLM Offline Python Expected Output" (4 queries x 4 documents; the
# diagonal pairs match). A wrong pooling mode or causal attention would move these far off.
CARD_QUERIES = [
    "Write a Python function that counts the frequency of each element in a list of lists.",
    "Write a function that orders a dictionary with tuple keys by the product of each key's tuple values.",
    "What symptoms and common triggers help distinguish eczema from other inflammatory skin conditions?",
    "How can someone reduce exposure to pollen during allergy season?",
]
CARD_DOCUMENTS = [
    "def frequency_lists(list1):\n    flattened = [item for sublist in list1 for item in sublist]\n    counts = {}\n"
    "    for item in flattened:\n        if item in counts:\n            counts[item] += 1\n        else:\n"
    "            counts[item] = 1\n    return counts",
    "def sort_dict_item(test_dict):\n    return {key: test_dict[key] for key in sorted(test_dict.keys(), "
    "key=lambda ele: ele[0] * ele[1])}",
    "Eczema commonly causes itchy, dry, inflamed patches of skin. The affected areas may look red, scaly, cracked, "
    "or darker than the surrounding skin depending on skin tone. Symptoms can flare after exposure to irritants, "
    "allergens, stress, or changes in weather.",
    "People with pollen allergy can reduce exposure by staying indoors on dry, windy days, avoiding early-morning "
    "outdoor activity, and going outside after rain when pollen levels are lower. They should check pollen "
    "forecasts, close windows and doors when counts are high, and consider starting allergy medication before "
    "symptoms begin if high pollen is expected. After being outside, showering, changing clothes, avoiding outdoor "
    "laundry drying, and wearing a face mask for yard work can help limit pollen contact.",
]
CARD_SCORES = np.array([
    [0.7846, 0.0489, 0.0118, -0.0301],
    [0.0454, 0.6505, -0.0645, 0.0455],
    [-0.0018, -0.0531, 0.6611, 0.1120],
    [-0.0166, -0.0040, 0.1148, 0.7983],
], dtype=np.float32)


def vllm_embed_command(model: str, served_model_name: str, port: int, cfg: VLLMConfig | None = None,
                       python: str = "python") -> list[str]:
    cfg = cfg or EMBED_VLLM
    return [
        "env", "HF_HUB_OFFLINE=1", *cfg.env,
        python, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", served_model_name,
        "--host", "0.0.0.0", "--port", str(port),
        "--runner", "pooling",
        "--dtype", "bfloat16",
        "--max-model-len", str(cfg.max_model_len),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        *cfg.extra_args,
    ]


def truncate_normalize(x: np.ndarray, dims: int | None) -> np.ndarray:
    """Keep the first `dims` coordinates and re-normalize (L2), as the model card prescribes for smaller sizes."""
    x = np.asarray(x, dtype=np.float32)
    if dims is not None:  # None = full size; 0 or negative is an error, not "no truncation"
        if not 0 < dims <= x.shape[-1]:
            raise ValueError(f"dimensions must be in 1..{x.shape[-1]}, got {dims}")
        x = x[..., :dims]
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def retryable(exc: Exception) -> bool:
    """Transient failures only: connection errors, timeouts, 429 and 5xx. A 4xx is the request's fault."""
    import requests

    if isinstance(exc, requests.ConnectionError | requests.Timeout):
        return True
    response = getattr(exc, "response", None)
    return isinstance(exc, requests.HTTPError) and response is not None and (
        response.status_code == 429 or response.status_code >= 500)


def embed_texts(port: int, served_model_name: str, texts: Sequence[str], input_type: str, dims: int | None = None,
                batch_size: int = 32, concurrency: int = 4, max_tokens: int | None = None,
                on_batch: Callable[[int], None] | None = None) -> np.ndarray:
    """Prefix, embed through `/v1/embeddings` (concurrent batches), then truncate/re-normalize. Row order kept."""
    import requests

    if input_type not in PREFIXES:
        raise ValueError(f"input_type must be one of {sorted(PREFIXES)}, got {input_type!r}")
    prefixed = [PREFIXES[input_type] + t for t in texts]
    batches = [prefixed[i:i + batch_size] for i in range(0, len(prefixed), batch_size)]
    url = f"http://127.0.0.1:{port}/v1/embeddings"

    def one(batch: list[str]) -> list[list[float]]:
        body = {"model": served_model_name, "input": batch, "encoding_format": "float"}
        if max_tokens:
            body["truncate_prompt_tokens"] = max_tokens  # long inputs are cut, not rejected
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, timeout=600)
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                if on_batch:
                    on_batch(len(batch))
                return [d["embedding"] for d in data]
            except requests.RequestException as exc:
                if attempt == 2 or not retryable(exc):
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        raise RuntimeError(f"vLLM {exc.response.status_code}: {exc.response.text[:500]}") from exc
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")

    if not batches:
        return np.zeros((0, dims or FULL_DIM), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(batches)))) as pool:
        rows = [row for out in pool.map(one, batches) for row in out]
    return truncate_normalize(np.asarray(rows, dtype=np.float32), dims)


def card_check(port: int, served_model_name: str, tol: float = 0.03) -> dict:
    """Embed the model card's example and compare with its published scores. Raises if they disagree."""
    q = embed_texts(port, served_model_name, CARD_QUERIES, "query")
    d = embed_texts(port, served_model_name, CARD_DOCUMENTS, "document")
    scores = q @ d.T
    max_err = float(np.abs(scores - CARD_SCORES).max())
    result = {"max_abs_err": round(max_err, 4), "dims": int(q.shape[1]),
              "diagonal": [round(float(v), 4) for v in np.diag(scores)]}
    if max_err > tol or not (scores.argmax(axis=1) == np.arange(len(CARD_QUERIES))).all():
        raise RuntimeError(f"Embeddings disagree with the model card (tolerance {tol}): {result}\n{scores.round(4)}")
    return result

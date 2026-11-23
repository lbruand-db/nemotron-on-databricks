"""Query a detector, OCR or Nano VL endpoint with local image files.

    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-page-elements-v3-ft \
        --profile am-rag-fevm page1.png page2.jpg
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-ocr-v1-ft --params '{"merge_level": "word"}' r.png
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-nano-vl-ft --vlm r1.png r2.png   # Nano VL pyfunc
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-nano-vl-ft --chat receipt.png   # Custom LLM
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-3-embed-8b --embed --input-type query \
        --dimensions 1024 "Comment calculer un taux d'intérêt ?"   # embeddings (positional args are texts)
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def build_request(paths: list[str], conf_threshold: float = -1.0, params: dict | None = None) -> dict:
    """Detector params by default; pass `params` to override (e.g. {"merge_level": "word"} for OCR)."""
    return {
        "dataframe_records": [
            {"image": base64.b64encode(Path(p).read_bytes()).decode(), "id": Path(p).name} for p in paths
        ],
        "params": params if params is not None else {"conf_threshold": conf_threshold, "apply_class_thresholds": True},
    }


def build_vlm_request(paths: list[str], instruction: str | None = None, max_tokens: int = 1536) -> dict:
    """Nano VL classic pyfunc (`VLLMChatModel`): one record per image, optional per-record instruction."""
    extra = {"instruction": instruction} if instruction else {}
    return {
        "dataframe_records": [
            {"image": base64.b64encode(Path(p).read_bytes()).decode(), "id": Path(p).name, **extra} for p in paths
        ],
        "params": {"max_tokens": max_tokens},
    }


def build_embed_request(texts: list[str], input_type: str = "document", dimensions: int = 4096) -> dict:
    """Nemotron-3-Embed pyfunc (`VLLMEmbedModel`): one record per text."""
    return {
        "dataframe_records": [{"text": t, "id": i, "input_type": input_type} for i, t in enumerate(texts)],
        "params": {"dimensions": dimensions},
    }


def summarize_embeddings(predictions: list[dict]) -> str:
    """Per text: dims, norm and first values; then the cosine-similarity matrix (vectors are L2-normalized)."""
    import numpy as np

    emb = np.asarray([p["embedding"] for p in predictions], dtype=np.float32)
    lines = [f"{p['id']} {p['input_type']} dims={len(p['embedding'])} norm={np.linalg.norm(e):.3f} "
             f"head={[round(float(v), 4) for v in e[:4]]}" for p, e in zip(predictions, emb, strict=True)]
    if len(emb) > 1:
        lines += ["similarity:"] + ["  " + " ".join(f"{v:7.4f}" for v in row) for row in emb @ emb.T]
    return "\n".join(lines)


def build_chat_request(path: str, instruction: str | None = None, max_tokens: int = 1536) -> dict:
    """OpenAI chat request for the Nano VL endpoint: the CORD extraction conversation used in training."""
    from ..vlm.data import INSTRUCTION, VLSample, openai_messages

    sample = VLSample(Path(path).read_bytes(), {})
    return {"messages": openai_messages(sample.data_url(), instruction or INSTRUCTION), "max_tokens": max_tokens,
            "temperature": 0}


def main(argv: list[str] | None = None) -> None:
    from databricks.sdk import WorkspaceClient

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--conf-threshold", type=float, default=-1.0, help="detectors only")
    ap.add_argument("--params", help='JSON params, overrides the detector defaults, e.g. \'{"merge_level": "word"}\'')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--vlm", action="store_true", help="Nano VL classic pyfunc: one record per image")
    mode.add_argument("--embed", action="store_true", help="Embedding pyfunc: positional arguments are texts")
    mode.add_argument("--chat", action="store_true",
                      help="Nano VL on Custom LLM Serving: OpenAI chat request with the image (one image)")
    ap.add_argument("--instruction", help="--vlm/--chat: replaces the CORD extraction instruction")
    ap.add_argument("--max-tokens", type=int, default=1536, help="--vlm/--chat")
    ap.add_argument("--input-type", choices=("query", "document"), default="document", help="--embed")
    ap.add_argument("--dimensions", type=int, default=4096, help="--embed: truncate + re-normalize, e.g. 1024")
    ap.add_argument("--profile")
    args = ap.parse_args(argv)
    if args.instruction and not (args.vlm or args.chat):
        ap.error("--instruction needs --vlm or --chat")
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    if args.chat:
        if len(args.images) != 1:
            ap.error("--chat takes exactly one image")
        body = build_chat_request(args.images[0], args.instruction, args.max_tokens)
    elif args.vlm:
        body = build_vlm_request(args.images, args.instruction, args.max_tokens)
    elif args.embed:
        body = build_embed_request(args.images, args.input_type, args.dimensions)
    else:
        body = build_request(args.images, args.conf_threshold, json.loads(args.params) if args.params else None)
    resp = client.api_client.do("POST", f"/serving-endpoints/{args.endpoint_name}/invocations", body=body)
    if args.chat and isinstance(resp, dict) and resp.get("choices"):
        print(resp["choices"][0]["message"]["content"])
        print(json.dumps({"usage": resp.get("usage")}), flush=True)
    elif args.embed and isinstance(resp, dict) and resp.get("predictions"):
        print(summarize_embeddings(resp["predictions"]))
    else:
        print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()

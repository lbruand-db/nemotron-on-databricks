"""Query a detector, OCR or Nano VL endpoint with local image files.

    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-page-elements-v3-ft \
        --profile am-rag-fevm page1.png page2.jpg
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-ocr-v1-ft --params '{"merge_level": "word"}' r.png
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-nano-vl-ft --chat receipt.png   # Nano VL
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
    ap.add_argument("--chat", action="store_true", help="Nano VL: OpenAI chat request with the image (one image)")
    ap.add_argument("--instruction", help="--chat only: replaces the CORD extraction instruction")
    ap.add_argument("--profile")
    args = ap.parse_args(argv)
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    if args.chat:
        if len(args.images) != 1:
            ap.error("--chat takes exactly one image")
        body = build_chat_request(args.images[0], args.instruction)
    else:
        body = build_request(args.images, args.conf_threshold, json.loads(args.params) if args.params else None)
    resp = client.api_client.do("POST", f"/serving-endpoints/{args.endpoint_name}/invocations", body=body)
    if args.chat and isinstance(resp, dict) and resp.get("choices"):
        print(resp["choices"][0]["message"]["content"])
        print(json.dumps({"usage": resp.get("usage")}), flush=True)
    else:
        print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()

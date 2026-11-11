"""Query a detector or OCR endpoint with local image files.

    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-page-elements-v3-ft \
        --profile am-rag-fevm page1.png page2.jpg
    python -m nemotron_dbx.lifecycle.query --endpoint-name nemotron-ocr-v1-ft --params '{"merge_level": "word"}' r.png
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


def main(argv: list[str] | None = None) -> None:
    from databricks.sdk import WorkspaceClient

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--conf-threshold", type=float, default=-1.0, help="detectors only")
    ap.add_argument("--params", help='JSON params, overrides the detector defaults, e.g. \'{"merge_level": "word"}\'')
    ap.add_argument("--profile")
    args = ap.parse_args(argv)
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    body = build_request(args.images, args.conf_threshold, json.loads(args.params) if args.params else None)
    resp = client.api_client.do("POST", f"/serving-endpoints/{args.endpoint_name}/invocations", body=body)
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()

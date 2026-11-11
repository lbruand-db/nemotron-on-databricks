"""MLflow pyfunc for nemotron-ocr-v1 (SPEC §5.3 contract, OCR variant).

Request:  {"dataframe_records": [{"image": "<base64>", "id": "r1"}], "params": {"merge_level": "paragraph"}}
Response: {"predictions": [{"id": "r1", "regions": [{"text": "...", "confidence": 0.98,
           "x_min": .., "y_min": .., "x_max": .., "y_max": ..}]}]}   (normalized, reading order)

Artifact `checkpoints`: detector.pth, recognizer.pth (fine-tuned), relational.pth, charset.txt.
The `nemotron_ocr` package (with its C++/CUDA extension) must be installed in the serving environment;
registration ships it as a wheel (`code/<wheel>` pip requirement).
"""

from __future__ import annotations

import math
from typing import Any

from ..detection.serving import _record_id, _records, decode_b64_image

try:
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover
    PythonModel = object  # type: ignore[misc,assignment]

MERGE_LEVELS = ("word", "sentence", "paragraph")


def format_regions(preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """NVIDIA pipeline output -> normalized boxes with y_min <= y_max (the pipeline's upper/lower are flipped)."""
    out = []
    for p in preds:
        text = p.get("text")
        if not isinstance(text, str):
            continue
        vals = [float(p[k]) for k in ("left", "upper", "right", "lower")]
        if any(math.isnan(v) for v in vals):
            continue
        left, upper, right, lower = vals
        out.append({
            "text": text,
            "confidence": round(float(p.get("confidence", 0.0)), 5),
            "x_min": round(min(left, right), 5), "y_min": round(min(upper, lower), 5),
            "x_max": round(max(left, right), 5), "y_max": round(max(upper, lower), 5),
        })
    return out


class OCRModel(PythonModel):
    def load_context(self, context) -> None:
        from nemotron_ocr.inference.pipeline import NemotronOCR

        self.ocr = NemotronOCR(model_dir=context.artifacts["checkpoints"])

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        params = params or {}
        level = params.get("merge_level", "paragraph")
        if level not in MERGE_LEVELS:
            raise ValueError(f"merge_level must be one of {MERGE_LEVELS}")
        out = []
        for i, rec in enumerate(_records(model_input)):
            preds = self.ocr(decode_b64_image(rec["image"]), merge_level=level)
            out.append({"id": _record_id(rec, i), "regions": format_regions(preds)})
        return out

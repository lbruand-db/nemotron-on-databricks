"""MLflow pyfunc for the Nemotron YOLOX detectors (SPEC §5.3 serving contract).

Request:  {"dataframe_records": [{"image": "<base64 PNG/JPEG>", "id": "page-1"}],
           "params": {"conf_threshold": -1.0, "apply_class_thresholds": true}}
          conf_threshold >= 0 applies one global score threshold; < 0 (default) uses the per-class
          thresholds from the config (NVIDIA defaults) unless apply_class_thresholds is false.
Response: {"predictions": [{"id": "page-1", "bounding_boxes": {"table": [{"x_min": .., "y_min": ..,
           "x_max": .., "y_max": .., "confidence": ..}], ...}}]}   (coordinates normalized to [0, 1])

Artifacts: `hf_code` (pinned NVIDIA package source), `weights` (weights.pth), `config` (JSON).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np

try:  # mlflow is only needed when the class is used as a pyfunc
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - unit tests run without mlflow
    PythonModel = object  # type: ignore[misc,assignment]


def decode_b64_image(data: str) -> np.ndarray:
    from PIL import Image

    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def format_prediction(pred: dict[str, np.ndarray], labels: list[str], image_id: Any) -> dict[str, Any]:
    """Normalized xyxy predictions -> NIM-style `bounding_boxes` grouped by class (all classes present)."""
    grouped: dict[str, list[dict[str, float]]] = {name: [] for name in labels}
    order = np.argsort(-pred["scores"]) if len(pred["scores"]) else []
    for i in order:
        x1, y1, x2, y2 = (float(v) for v in pred["boxes"][i])
        grouped[labels[int(pred["labels"][i])]].append({
            "x_min": round(min(x1, x2), 5), "y_min": round(min(y1, y2), 5),
            "x_max": round(max(x1, x2), 5), "y_max": round(max(y1, y2), 5),
            "confidence": round(float(pred["scores"][i]), 5),
        })
    return {"id": image_id, "bounding_boxes": grouped}


def _records(model_input) -> list[dict[str, Any]]:
    if hasattr(model_input, "to_dict"):  # pandas DataFrame
        return model_input.to_dict(orient="records")
    if isinstance(model_input, dict):
        return [model_input]
    return list(model_input)


class DetectorModel(PythonModel):
    """Serves any Nemotron YOLOX detector described by the `config` artifact."""

    def load_context(self, context) -> None:
        import torch

        from .hf_detector import SPECS, class_thresholds, load_yolox, wrap

        with open(context.artifacts["config"], encoding="utf-8") as fh:
            self.config = json.load(fh)
        spec = SPECS[self.config["spec"]]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, exp = load_yolox(spec, context.artifacts["hf_code"], context.artifacts["weights"], device)
        self.wrapper = wrap(spec, context.artifacts["hf_code"], model, exp)
        self.labels = list(exp.labels)
        self.thresholds = dict(self.config.get("thresholds_per_class") or class_thresholds(exp))
        self.batch_size = int(self.config.get("batch_size", 8))

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        from .hf_detector import apply_class_thresholds, predict

        params = params or {}
        conf = params.get("conf_threshold", -1.0)
        conf = None if conf is None or float(conf) < 0 else float(conf)
        use_class_thr = params.get("apply_class_thresholds", True)
        records = _records(model_input)
        images = [decode_b64_image(r["image"]) for r in records]
        preds = predict(self.wrapper, images, batch_size=self.batch_size)
        out = []
        for i, (rec, pred) in enumerate(zip(records, preds, strict=True)):
            if conf is not None:
                keep = pred["scores"] > conf
                pred = {k: v[keep] for k, v in pred.items()}
            elif use_class_thr:
                pred = apply_class_thresholds(pred, self.labels, self.thresholds)
            out.append(format_prediction(pred, self.labels, _record_id(rec, i)))
        return out


def _record_id(rec: dict[str, Any], default: int) -> Any:
    value = rec.get("id")
    if value is None or (isinstance(value, float) and np.isnan(value)):  # missing column in a DataFrame
        return default
    return value

import base64
import io
import math

import numpy as np
import pytest
from PIL import Image

from nemotron_dbx.common.params import build_config
from nemotron_dbx.detection.hf_detector import apply_class_thresholds
from nemotron_dbx.detection.serving import _record_id, decode_b64_image, format_prediction
from nemotron_dbx.lifecycle.query import build_request
from nemotron_dbx.yolox_train.trainer import TrainConfig, lr_factor

LABELS = ["table", "chart", "title"]


def test_coco_map_perfect_and_empty():
    pytest.importorskip("pycocotools")
    from nemotron_dbx.detection.metrics import coco_map

    gt_boxes = np.array([[10, 10, 50, 50], [60, 60, 90, 80]], dtype=np.float32)
    gt_labels = np.array([0, 2])
    h, w = 100, 100
    perfect = {"boxes": gt_boxes / np.array([w, h, w, h]), "labels": gt_labels, "scores": np.array([0.9, 0.8])}
    m = coco_map([(gt_boxes, gt_labels)], [perfect], [(h, w)], LABELS)
    assert m["map"] == pytest.approx(1.0)
    assert m["ap/table"] == pytest.approx(1.0) and math.isnan(m["ap/chart"])

    empty = {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,), int), "scores": np.zeros((0,))}
    assert coco_map([(gt_boxes, gt_labels)], [empty], [(h, w)], LABELS)["map"] == 0.0


def test_format_prediction_groups_sorts_and_normalizes_order():
    pred = {
        "boxes": np.array([[0.1, 0.5, 0.3, 0.2], [0.0, 0.0, 1.0, 1.0]]),  # first has y flipped
        "labels": np.array([2, 2]),
        "scores": np.array([0.4, 0.9]),
    }
    out = format_prediction(pred, LABELS, "p1")
    assert out["id"] == "p1" and set(out["bounding_boxes"]) == set(LABELS)
    titles = out["bounding_boxes"]["title"]
    assert [t["confidence"] for t in titles] == [0.9, 0.4]
    assert titles[1]["y_min"] == 0.2 and titles[1]["y_max"] == 0.5


def test_apply_class_thresholds():
    pred = {"boxes": np.zeros((3, 4)), "labels": np.array([0, 1, 2]), "scores": np.array([0.05, 0.05, 0.5])}
    out = apply_class_thresholds(pred, LABELS, {"table": 0.1, "chart": 0.01, "title": 0.1})
    assert out["labels"].tolist() == [1, 2]


def test_b64_roundtrip_and_request(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (7, 5), (10, 20, 30)).save(buf, format="PNG")
    arr = decode_b64_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    assert arr.shape == (5, 7, 3) and arr[0, 0].tolist() == [10, 20, 30]
    f = tmp_path / "x.png"
    f.write_bytes(buf.getvalue())
    req = build_request([str(f)])
    assert req["dataframe_records"][0]["id"] == "x.png" and req["params"]["conf_threshold"] == -1.0


def test_record_id_defaults():
    assert _record_id({"id": float("nan")}, 3) == 3
    assert _record_id({}, 1) == 1
    assert _record_id({"id": "a"}, 0) == "a"


def test_build_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown parameters"):
        build_config(TrainConfig, {"epochs": 1, "epoch": 2})
    assert build_config(TrainConfig, {"epochs": 3}).epochs == 3


def test_lr_schedule():
    assert lr_factor(0, 100, 10, 0.05) == pytest.approx(0.1)
    assert lr_factor(10, 100, 10, 0.05) == pytest.approx(1.0)
    assert lr_factor(100, 100, 10, 0.05) == pytest.approx(0.05)

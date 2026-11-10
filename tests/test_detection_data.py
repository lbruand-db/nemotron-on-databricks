import io
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from nemotron_dbx.detection.doclaynet import PAGE_ELEMENTS_LABELS, PICTURE_ID, map_page
from nemotron_dbx.yolox_train.data import (
    AugmentConfig,
    DetectionDataset,
    DetRecord,
    collate,
    encode_targets,
    letterbox,
)


def _png(w=200, h=100, color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_map_page_classes_and_clipping():
    # Caption, Page-header, Section-header, Table, Text, Title
    boxes = [[-2, 0, 10, 10], [0, 0, 50, 5], [5, 5, 20, 5], [10, 10, 100, 50], [0, 60, 30, 10], [1, 1, 5, 5]]
    cats = [1, 6, 8, 9, 10, 11]
    out = map_page(boxes, cats, width=100, height=100, drop_picture_pages=True)
    assert out is not None
    xyxy, labels = out
    names = [PAGE_ELEMENTS_LABELS[i] for i in labels]
    assert names == ["text", "header_footer", "title", "table", "text", "title"]
    assert xyxy[0, 0] == 0.0  # clipped at 0
    assert xyxy[3, 2] == 100.0  # x2 = min(width, 10 + 100)


def test_map_page_picture_policy():
    boxes, cats = [[0, 0, 10, 10], [20, 20, 10, 10]], [PICTURE_ID, 10]
    assert map_page(boxes, cats, 100, 100, drop_picture_pages=True) is None
    xyxy, labels = map_page(boxes, cats, 100, 100, drop_picture_pages=False)
    assert len(labels) == 1  # Picture box itself is never mapped


def test_letterbox_matches_nvidia_resize_pad(page_elements_code):
    sys.path.insert(0, str(page_elements_code))
    from nemotron_page_elements_v3.model import resize_pad

    img = torch.rand(3, 700, 1300) * 255
    ours, scale = letterbox(img, (1024, 1024))
    ref = resize_pad(img, (1024, 1024))
    torch.testing.assert_close(ours, ref)
    assert scale == pytest.approx(1024 / 1300)


def test_encode_targets_cxcywh_and_padding():
    boxes = np.array([[10, 20, 30, 60], [0, 0, 0.5, 0.5]], dtype=np.float32)  # 2nd: 1 px after scale
    t = encode_targets(boxes, np.array([2, 1]), scale=2.0, size=(1024, 1024), max_labels=5)
    assert t.shape == (5, 5)
    torch.testing.assert_close(t[0], torch.tensor([2.0, 40.0, 80.0, 40.0, 80.0]))
    assert t[1:].abs().sum() == 0


def test_dataset_and_collate_shapes():
    recs = [DetRecord(_png(), np.array([[10, 10, 60, 40]], np.float32), np.array([0]), "a"),
            DetRecord(_png(120, 300), np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), "b")]
    ds = DetectionDataset(recs, size=(256, 256), augment=AugmentConfig(), max_labels=8)
    imgs, labels, infos = collate([ds[0], ds[1]])
    assert imgs.shape == (2, 3, 256, 256) and imgs.dtype == torch.float32
    assert labels.shape == (2, 8, 5)
    assert 0 <= imgs.min() and imgs.max() <= 255
    assert infos[1]["image_id"] == "b" and labels[1].abs().sum() == 0

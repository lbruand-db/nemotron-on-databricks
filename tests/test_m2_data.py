import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from nemotron_dbx.detection import charts, pubtables
from nemotron_dbx.detection.hf_detector import class_thresholds

VOC = b"""<?xml version="1.0" ?><annotation><size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>table</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>190</xmax><ymax>90</ymax></bndbox></object>
<object><name>table row</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>190</xmax><ymax>50</ymax></bndbox></object>
<object><name>table row</name><bndbox><xmin>10</xmin><ymin>50</ymin><xmax>190</xmax><ymax>90</ymax></bndbox></object>
<object><name>table column</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>100</xmax><ymax>90</ymax></bndbox></object>
<object><name>table column</name><bndbox><xmin>100</xmin><ymin>10</ymin><xmax>190</xmax><ymax>90</ymax></bndbox>
</object>
<object><name>table spanning cell</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>190</xmax><ymax>50</ymax></bndbox>
</object>
<object><name>table projected row header</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>190</xmax><ymax>50</ymax>
</bndbox></object>
<object><name>table column header</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>190</xmax><ymax>50</ymax>
</bndbox></object>
</annotation>"""


def test_derive_cells_grid_and_spans():
    rows = np.array([[0, 0, 10, 5], [0, 5, 10, 10]], dtype=np.float32)
    cols = np.array([[0, 0, 4, 10], [4, 0, 10, 10]], dtype=np.float32)
    grid = pubtables.derive_cells(rows, cols, np.zeros((0, 4)))
    assert len(grid) == 4 and sorted(map(tuple, grid.tolist()))[0] == (0, 0, 4, 5)
    merged = pubtables.derive_cells(rows, cols, np.array([[0, 0, 10, 5]], dtype=np.float32))
    assert len(merged) == 3  # the two top-row cells are replaced by one merged cell
    assert [0, 0, 10, 5] in merged.tolist()


def test_map_voc_table():
    w, h, objs = pubtables.parse_voc(VOC)
    assert (w, h) == (200, 100)
    boxes, labels = pubtables.map_table(objs)
    # spanning cell + projected row header coincide -> one merged top cell; bottom row keeps 2 grid cells
    assert (labels == pubtables.CELL).sum() == 3
    assert (labels == pubtables.ROW).sum() == 2 and (labels == pubtables.COLUMN).sum() == 2
    assert pubtables.map_table([("table", [0, 0, 1, 1])]) is None


def test_synthetic_chart_boxes_are_valid_and_deterministic():
    seen = set()
    for seed in range(6):
        png, boxes, labels, meta = charts.render_chart(seed)
        w, h = Image.open(io.BytesIO(png)).size
        assert [w, h] == meta["size"] and len(boxes) == len(labels) > 0
        assert (boxes[:, 0] >= 0).all() and (boxes[:, 2] <= w).all() and (boxes[:, 1] >= 0).all()
        assert (boxes[:, 3] <= h).all() and (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert labels.min() >= 0 and labels.max() < len(charts.LABELS)
        seen.update(charts.LABELS[i] for i in labels)
    assert {"xlabel", "ylabel"} <= seen
    _, b1, l1, _ = charts.render_chart(3)
    _, b2, l2, _ = charts.render_chart(3)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(l1, l2)


def test_chart_labels_match_nvidia_order():
    # nemotron-graphic-elements-v1 Exp.labels (graphic_element_v1.py at the pinned revision)
    assert charts.LABELS == ["chart_title", "x_title", "y_title", "xlabel", "ylabel", "other", "legend_label",
                             "legend_title", "mark_label", "value_label"]


def test_class_thresholds_fallback():
    assert class_thresholds(SimpleNamespace(thresholds_per_class={"table": 0.3}, labels=["table"])) == {"table": 0.3}
    exp = SimpleNamespace(threshold=0.05, labels=["border", "cell", "row", "column", "header"])
    assert class_thresholds(exp) == {name: 0.05 for name in exp.labels}

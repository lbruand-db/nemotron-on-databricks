import sys

import pytest
import torch
import torch.nn as nn

from nemotron_dbx.yolox_train.loss import bboxes_iou, head_train_forward, iou_loss, yolox_loss


class TinyHead(nn.Module):
    """Same attribute layout as YOLOXHead, 1x1 convs, 3 levels."""

    def __init__(self, num_classes: int = 3, ch: int = 8, strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.stems = nn.ModuleList(nn.Conv2d(ch, ch, 1) for _ in strides)
        self.cls_convs = nn.ModuleList(nn.Conv2d(ch, ch, 1) for _ in strides)
        self.reg_convs = nn.ModuleList(nn.Conv2d(ch, ch, 1) for _ in strides)
        self.cls_preds = nn.ModuleList(nn.Conv2d(ch, num_classes, 1) for _ in strides)
        self.reg_preds = nn.ModuleList(nn.Conv2d(ch, 4, 1) for _ in strides)
        self.obj_preds = nn.ModuleList(nn.Conv2d(ch, 1, 1) for _ in strides)


def _fpn(batch=2, ch=8, size=128, strides=(8, 16, 32)):
    g = torch.Generator().manual_seed(0)
    return [torch.randn(batch, ch, size // s, size // s, generator=g) for s in strides]


def _labels(batch=2, max_labels=10):
    lab = torch.zeros(batch, max_labels, 5)
    lab[0, 0] = torch.tensor([1, 40.0, 40.0, 30.0, 20.0])
    lab[0, 1] = torch.tensor([2, 90.0, 70.0, 40.0, 50.0])
    # image 1 has no objects (padding only)
    return lab


def test_iou_helpers():
    a = torch.tensor([[0.0, 0, 10, 10]])
    assert torch.allclose(bboxes_iou(a, a), torch.ones(1, 1))
    c = torch.tensor([[5.0, 5, 10, 10]])  # cxcywh of the same box
    assert torch.allclose(iou_loss(c, c), torch.zeros(1))


def test_head_forward_shapes():
    head = TinyHead()
    out = head_train_forward(head, _fpn())
    n_anchors = sum((128 // s) ** 2 for s in head.strides)
    assert out.outputs.shape == (2, n_anchors, 5 + 3)
    assert out.x_shifts.shape == out.y_shifts.shape == out.expanded_strides.shape == (1, n_anchors)


def test_loss_finite_backward_and_empty_image():
    head = TinyHead()
    losses = yolox_loss(head_train_forward(head, _fpn()), _labels(), num_classes=3)
    assert torch.isfinite(losses.loss)
    losses.loss.backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert losses.num_fg_per_gt > 0


def test_loss_decreases_when_optimizing():
    torch.manual_seed(0)
    head = TinyHead()
    fpn, labels = _fpn(), _labels()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        loss = yolox_loss(head_train_forward(head, fpn), labels, num_classes=3).loss
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_matches_nvidia_head_decode(page_elements_code):
    """The training branch decodes boxes exactly like NVIDIA's inference head."""
    sys.path.insert(0, str(page_elements_code))
    from nemotron_page_elements_v3.page_element_v3 import Exp

    exp = Exp()
    exp.device = "cpu"
    model = exp.get_model().eval()
    x = torch.rand(1, 3, 256, 256) * 255
    with torch.no_grad():
        fpn = model.backbone(x)
        ref = model.head(fpn)  # [1, A, 5 + C]: decoded boxes, sigmoid(obj), sigmoid(cls)
        ours = head_train_forward(model.head, fpn).outputs
    assert ref.shape == ours.shape
    torch.testing.assert_close(ours[..., :4], ref[..., :4], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(ours[..., 4:].sigmoid(), ref[..., 4:], rtol=1e-4, atol=1e-5)


def test_real_architecture_loss_backward(page_elements_code):
    sys.path.insert(0, str(page_elements_code))
    from nemotron_page_elements_v3.page_element_v3 import Exp

    exp = Exp()
    exp.device = "cpu"
    model = exp.get_model().train()
    x = torch.rand(2, 3, 256, 256) * 255
    labels = torch.zeros(2, 5, 5)
    labels[0, 0] = torch.tensor([0, 128.0, 128.0, 100.0, 60.0])
    labels[1, 0] = torch.tensor([4, 60.0, 50.0, 80.0, 30.0])
    losses = yolox_loss(head_train_forward(model.head, model.backbone(x)), labels, exp.num_classes)
    losses.loss.backward()
    assert torch.isfinite(losses.loss)
    assert model.head.cls_preds[0].weight.grad is not None


@pytest.mark.parametrize("num_gt", [1, 3])
def test_assignment_uses_all_gts(num_gt):
    head = TinyHead()
    lab = torch.zeros(1, 5, 5)
    for i in range(num_gt):
        lab[0, i] = torch.tensor([i % 3, 20.0 + 30 * i, 30.0 + 20 * i, 24.0, 24.0])
    losses = yolox_loss(head_train_forward(head, _fpn(batch=1)), lab, num_classes=3)
    assert losses.num_fg_per_gt >= 1.0

# Copyright (c) Megvii Inc. All rights reserved.  (original YOLOX training code)
# Copyright (c) 2026 nemotron-on-databricks contributors.  (adaptation)
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Megvii-BaseDetection/YOLOX @ 6ddff4824372906469a7fae2dc3206c7aa4bbaee
#   yolox/models/yolo_head.py  (training branch of forward, get_output_and_grid, get_losses,
#                               get_assignments, get_geometry_constraint, simota_matching)
#   yolox/models/losses.py     (IOUloss)
# NVIDIA's Nemotron YOLOX repos (page-elements-v3, table-structure-v1, graphic-elements-v1) ship the
# same YOLOXHead module layout with the training branch removed. This module re-attaches it as free
# functions over an existing head, so pretrained weights load strictly and nothing is monkey-patched.
# Changes vs upstream: no L1 loss branch (only used in the last epochs of from-scratch training), no
# CPU fallback for assignment, no loguru, and the head is passed explicitly.
"""YOLOX SimOTA training loss for NVIDIA Nemotron YOLOX detectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy: bool = True) -> torch.Tensor:
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("boxes must have 4 coordinates")
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """YOLOX `IOUloss(loss_type="iou", reduction="none")` on cxcywh boxes: 1 - IoU^2."""
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
    br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
    area_p = torch.prod(pred[:, 2:], 1)
    area_g = torch.prod(target[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    area_u = area_p + area_g - area_i
    iou = area_i / (area_u + 1e-16)
    return 1 - iou**2


@dataclass
class HeadTrainOutputs:
    outputs: torch.Tensor  # [B, A, 5 + C]: decoded cxcywh (input px), obj logit, cls logits
    x_shifts: torch.Tensor  # [1, A]
    y_shifts: torch.Tensor  # [1, A]
    expanded_strides: torch.Tensor  # [1, A]


def head_train_forward(head: nn.Module, fpn_outs: list[torch.Tensor]) -> HeadTrainOutputs:
    """Training branch of `YOLOXHead.forward`: raw logits + grid-decoded boxes."""
    outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
    for k, (cls_conv, reg_conv, stride, x) in enumerate(
        zip(head.cls_convs, head.reg_convs, head.strides, fpn_outs, strict=True)
    ):
        x = head.stems[k](x)
        cls_output = head.cls_preds[k](cls_conv(x))
        reg_feat = reg_conv(x)
        reg_output = head.reg_preds[k](reg_feat)
        obj_output = head.obj_preds[k](reg_feat)

        output = torch.cat([reg_output, obj_output, cls_output], 1)
        output, grid = _output_and_grid(output, stride, head.num_classes)
        x_shifts.append(grid[:, :, 0])
        y_shifts.append(grid[:, :, 1])
        expanded_strides.append(torch.full((1, grid.shape[1]), float(stride), dtype=output.dtype,
                                           device=output.device))
        outputs.append(output)
    return HeadTrainOutputs(
        outputs=torch.cat(outputs, 1),
        x_shifts=torch.cat(x_shifts, 1),
        y_shifts=torch.cat(y_shifts, 1),
        expanded_strides=torch.cat(expanded_strides, 1),
    )


def _output_and_grid(output: torch.Tensor, stride: int, num_classes: int):
    batch_size = output.shape[0]
    n_ch = 5 + num_classes
    hsize, wsize = output.shape[-2:]
    yv, xv = torch.meshgrid(
        torch.arange(hsize, device=output.device), torch.arange(wsize, device=output.device), indexing="ij"
    )
    grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(output.dtype)
    output = output.view(batch_size, 1, n_ch, hsize, wsize)
    output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, hsize * wsize, -1)
    grid = grid.view(1, -1, 2)
    # Out-of-place (upstream writes in place, which breaks autograd under some torch versions).
    xy = (output[..., :2] + grid) * stride
    wh = torch.exp(output[..., 2:4]) * stride
    output = torch.cat([xy, wh, output[..., 4:]], dim=-1)
    return output, grid


@dataclass
class LossOutputs:
    loss: torch.Tensor
    iou: torch.Tensor
    obj: torch.Tensor
    cls: torch.Tensor
    num_fg_per_gt: float


def yolox_loss(
    head_out: HeadTrainOutputs, labels: torch.Tensor, num_classes: int, reg_weight: float = 5.0
) -> LossOutputs:
    """YOLOX `get_losses` (without L1).

    Args:
        head_out: output of `head_train_forward`.
        labels: [B, max_objects, 5] with rows (class, cx, cy, w, h) in input pixels; all-zero rows are
            padding.
        num_classes: number of classes.
    """
    outputs = head_out.outputs.float()  # loss in fp32 even under bf16 autocast
    labels = labels.float()
    bbox_preds = outputs[:, :, :4]
    obj_preds = outputs[:, :, 4:5]
    cls_preds = outputs[:, :, 5:]
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
    total_num_anchors = outputs.shape[1]

    cls_targets, reg_targets, obj_targets, fg_masks = [], [], [], []
    num_fg, num_gts = 0.0, 0.0
    for b in range(outputs.shape[0]):
        num_gt = int(nlabel[b])
        num_gts += num_gt
        if num_gt == 0:
            cls_target = outputs.new_zeros((0, num_classes))
            reg_target = outputs.new_zeros((0, 4))
            obj_target = outputs.new_zeros((total_num_anchors, 1))
            fg_mask = outputs.new_zeros(total_num_anchors).bool()
        else:
            gt_bboxes = labels[b, :num_gt, 1:5]
            gt_classes = labels[b, :num_gt, 0]
            gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg_img = _get_assignments(
                b, num_gt, gt_bboxes, gt_classes, bbox_preds[b], head_out, cls_preds, obj_preds,
                num_classes,
            )
            num_fg += num_fg_img
            cls_target = F.one_hot(gt_matched_classes.to(torch.int64), num_classes) * pred_ious.unsqueeze(-1)
            obj_target = fg_mask.unsqueeze(-1)
            reg_target = gt_bboxes[matched_gt_inds]
        cls_targets.append(cls_target)
        reg_targets.append(reg_target)
        obj_targets.append(obj_target.to(outputs.dtype))
        fg_masks.append(fg_mask)

    cls_targets = torch.cat(cls_targets, 0)
    reg_targets = torch.cat(reg_targets, 0)
    obj_targets = torch.cat(obj_targets, 0)
    fg_masks = torch.cat(fg_masks, 0)

    num_fg = max(num_fg, 1)
    loss_iou = iou_loss(bbox_preds.reshape(-1, 4)[fg_masks], reg_targets).sum() / num_fg
    loss_obj = F.binary_cross_entropy_with_logits(
        obj_preds.reshape(-1, 1), obj_targets, reduction="none"
    ).sum() / num_fg
    loss_cls = F.binary_cross_entropy_with_logits(
        cls_preds.reshape(-1, num_classes)[fg_masks], cls_targets.to(cls_preds.dtype), reduction="none"
    ).sum() / num_fg
    loss = reg_weight * loss_iou + loss_obj + loss_cls
    return LossOutputs(loss, reg_weight * loss_iou, loss_obj, loss_cls, num_fg / max(num_gts, 1))


@torch.no_grad()
def _get_assignments(
    b, num_gt, gt_bboxes, gt_classes, bboxes_preds, head_out, cls_preds, obj_preds, num_classes
):
    fg_mask, geometry_relation = _geometry_constraint(
        gt_bboxes, head_out.expanded_strides, head_out.x_shifts, head_out.y_shifts
    )
    bboxes_preds = bboxes_preds[fg_mask].float()
    cls_preds_ = cls_preds[b][fg_mask]
    obj_preds_ = obj_preds[b][fg_mask]
    num_in_boxes_anchor = bboxes_preds.shape[0]

    pair_wise_ious = bboxes_iou(gt_bboxes.float(), bboxes_preds, False)
    gt_cls = F.one_hot(gt_classes.to(torch.int64), num_classes).float()
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

    with torch.autocast(device_type=cls_preds_.device.type, enabled=False):
        cls_scores = (cls_preds_.float().sigmoid() * obj_preds_.float().sigmoid()).sqrt()
        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_scores.unsqueeze(0).repeat(num_gt, 1, 1),
            gt_cls.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
            reduction="none",
        ).sum(-1)

    cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 1e6 * (~geometry_relation)
    num_fg, gt_matched_classes, pred_ious, matched_gt_inds = _simota_matching(
        cost, pair_wise_ious, gt_classes, num_gt, fg_mask
    )
    return gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg


def _geometry_constraint(gt_bboxes, expanded_strides, x_shifts, y_shifts, center_radius: float = 1.5):
    strides = expanded_strides[0]
    x_centers = ((x_shifts[0] + 0.5) * strides).unsqueeze(0)
    y_centers = ((y_shifts[0] + 0.5) * strides).unsqueeze(0)
    center_dist = strides.unsqueeze(0) * center_radius
    gt_l = gt_bboxes[:, 0:1] - center_dist
    gt_r = gt_bboxes[:, 0:1] + center_dist
    gt_t = gt_bboxes[:, 1:2] - center_dist
    gt_b = gt_bboxes[:, 1:2] + center_dist
    center_deltas = torch.stack([x_centers - gt_l, y_centers - gt_t, gt_r - x_centers, gt_b - y_centers], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    anchor_filter = is_in_centers.sum(dim=0) > 0
    return anchor_filter, is_in_centers[:, anchor_filter]


def _simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
    n_candidate_k = min(10, pair_wise_ious.size(1))
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[gt_idx], k=int(dynamic_ks[gt_idx]), largest=False)
        matching_matrix[gt_idx][pos_idx] = 1

    anchor_matching_gt = matching_matrix.sum(0)
    if anchor_matching_gt.numel() and anchor_matching_gt.max() > 1:
        multiple = anchor_matching_gt > 1
        _, cost_argmin = torch.min(cost[:, multiple], dim=0)
        matching_matrix[:, multiple] *= 0
        matching_matrix[cost_argmin, multiple] = 1
    fg_mask_inboxes = anchor_matching_gt > 0
    num_fg = int(fg_mask_inboxes.sum().item())
    fg_mask[fg_mask.clone()] = fg_mask_inboxes
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]
    pred_ious = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
    return num_fg, gt_matched_classes, pred_ious, matched_gt_inds

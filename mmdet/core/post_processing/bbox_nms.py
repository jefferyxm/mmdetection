import torch

from mmdet.ops.nms import nms_wrapper
import numpy as np
from shapely.geometry import *


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels



def multiclass_polygon_nms(multi_polygons, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_polygons (Tensor): shape (n, #class*8) or (n, 8)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()

    # nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = getattr(nms_wrapper, nms_type)

    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_polygons.shape[1] == 8:
            _polygons = multi_polygons[cls_inds, :]
        else:
            _polygons = multi_polygons[cls_inds, i * 8:(i + 1) * 8]
        _scores = multi_scores[cls_inds, i]

        # cls_dets = torch.cat([_polygons, _scores[:, None]], dim=1)
        # cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        # apply nms

        nms_th = nms_cfg_.iou_thr
        keep = polygon_nms(_polygons, _scores, nms_th)
        _polygons = _polygons[keep]
        _scores = _scores[keep]
        cls_dets = torch.cat([_polygons, _scores[:, None]], dim=1)

        cls_labels = multi_polygons.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_polygons.new_zeros((0, 9))
        labels = multi_polygons.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def polygon_nms(polygons, scores, thresh):
    assert polygons.shape[0] == scores.shape[0]

    polygons = polygons.cpu().numpy()
    scores = scores.cpu().numpy()

    num_polys = scores.shape[0]
    pts = polygons.reshape((num_polys, -1, 2))
    scores = scores

    areas = np.zeros(scores.shape)
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))

    for il in range(num_polys):
        poly = Polygon(pts[il])
        areas[il] = poly.area
        
        for jl in range(il, num_polys):
            polyj = Polygon(pts[jl])
            
            inS = poly.intersection(polyj)
            inter_areas[il][jl] = inS.area
            inter_areas[jl][il] = inS.area

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return keep
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import wh_delta2bbox
from mmdet.ops import nms
from .adaptive_anchor_head import AdaptiveAnchorHead
from ..registry import HEADS


@HEADS.register_module
class ARPNHead(AdaptiveAnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(ARPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.arpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.arpn_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.arpn_wh = nn.Conv2d(self.feat_channels, 2, 1)
        self.arpn_reg = nn.Conv2d(self.feat_channels, 4, 1)

    def init_weights(self):
        normal_init(self.arpn_conv, std=0.01)
        normal_init(self.arpn_wh, std=0.01)
        normal_init(self.arpn_cls, std=0.01)
        normal_init(self.arpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.arpn_conv(x)
        x = F.relu(x, inplace=True)
        arpn_cls_score = self.arpn_cls(x)
        arpn_shape_wh = self.arpn_wh(x)
        arpn_bbox_pred = self.arpn_reg(x)
        return arpn_cls_score, arpn_shape_wh, arpn_bbox_pred

    def loss(self,
             cls_scores,
             wh_preds,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(ARPNHead, self).loss(
            cls_scores,
            wh_preds,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_reg=losses['loss_reg'])



    def get_bboxes_single(self,
                          cls_scores,
                          shape_whs,
                          bbox_preds,
                          mlvl_anchor_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            
            arpn_cls_score = cls_scores[idx]
            arpn_wh_pred = shape_whs[idx]
            arpn_bbox_pred = bbox_preds[idx]
            
            assert arpn_cls_score.size()[-2:] == arpn_bbox_pred.size()[-2:]

            anchor_points = mlvl_anchor_points[idx]
            arpn_cls_score = arpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                arpn_cls_score = arpn_cls_score.reshape(-1)
                scores = arpn_cls_score.sigmoid()
            else:
                arpn_cls_score = arpn_cls_score.reshape(-1, 2)
                scores = arpn_cls_score.softmax(dim=1)[:, 1]

            arpn_wh_pred = arpn_wh_pred.permute(1, 2, 0).reshape(-1, 2)
            arpn_bbox_pred = arpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)

                arpn_wh_pred = arpn_wh_pred[topk_inds, :]
                arpn_bbox_pred = arpn_bbox_pred[topk_inds, :]
                anchor_points = anchor_points[topk_inds, :]
                scores = scores[topk_inds]

            norm = 2**(idx+2) * 8.0
            proposals = wh_delta2bbox(anchor_points, arpn_wh_pred, arpn_bbox_pred, 
                                      self.target_means, self.target_stds, img_shape, norm)

            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

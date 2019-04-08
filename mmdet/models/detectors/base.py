import logging
from abc import ABCMeta, abstractmethod

import cv2
import os
import mmcv
import numpy as np
import torch.nn as nn
import pycocotools.mask as maskUtils

from mmdet.core import tensor2imgs, get_classes


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset='coco',
                    score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)) or dataset is None:
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))


        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks

            gen_res_file = True
            pt_dir = '../output/pt/'
            im_name = img_meta['imname']
            if not os.path.exists(pt_dir):
                os.makedirs(pt_dir)
            if gen_res_file==True:
                img_index = im_name.split('_')[1]
                img_index = img_index.split('.')[0]
                pt_file = open(pt_dir + 'res_img_' + img_index + '.txt', 'w')

            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                # reshape to ori image
                px = bboxes[i, ::2]
                py = bboxes[i, 1::2]
                o_h, o_w, _ = img_meta['ori_shape']
                px = ((o_w/w)*px).astype(int)
                py = ((o_h/h)*py).astype(int)

                line = str(px[0]) + ',' + str(py[0]) + ',' + str(px[1]) + ',' + str(py[1]) + ',' + \
                        str(px[2]) + ',' + str(py[2]) + ',' + str(px[3]) + ',' + str(py[3]) + '\r\n'
                if gen_res_file==True:  
                    pt_file.write(line)
            if gen_res_file == True:
                pt_file.close()

            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            # mmcv.imshow_det_bboxes(
            #     img_show,
            #     bboxes,
            #     labels,
            #     class_names=class_names,
            #     score_thr=score_thr)
        
        
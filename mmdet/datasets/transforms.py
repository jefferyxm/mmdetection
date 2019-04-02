import mmcv
import numpy as np
import torch

__all__ = ['ImageTransform', 'BboxTransform', 'MaskTransform', 'Numpy2Tensor']


class ImageTransform(object):
    """Preprocess an image.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, polygons, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1

    f_polygons = polygons.copy()
    f_polygons[..., 0::8] = w - polygons[..., 2::8] - 1
    f_polygons[..., 1::8] = polygons[..., 3::8]
    f_polygons[..., 2::8] = w - polygons[..., 0::8] - 1
    f_polygons[..., 3::8] = polygons[..., 1::8]
    f_polygons[..., 4::8] = w - polygons[..., 6::8] - 1
    f_polygons[..., 5::8] = polygons[..., 7::8]
    f_polygons[..., 6::8] = w - polygons[..., 4::8] - 1
    f_polygons[..., 7::8] = polygons[..., 5::8]
    return flipped, f_polygons


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, polygons, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor
        gt_polygons = polygons * scale_factor
        if flip:
            gt_bboxes, gt_polygons = bbox_flip(gt_bboxes, gt_polygons, img_shape)
    
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

        gt_polygons[:, 0::2] = np.clip(gt_polygons[:, 0::2], 0, img_shape[1] - 1)
        gt_polygons[:, 1::2] = np.clip(gt_polygons[:, 1::2], 0, img_shape[0] - 1)

        if self.max_num_gts is None:
            return gt_bboxes, gt_polygons
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes

            padded_polygons = np.zeros((self.max_num_gts, 8), dtype=np.float32)
            padded_polygons[:num_gts, :] = gt_polygons

            return padded_bboxes, padded_polygons


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])

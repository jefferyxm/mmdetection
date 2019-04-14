import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import cv2

cfg = mmcv.Config.fromfile('./work_dirs/ic15-0303-zhou/maskr50.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './work_dirs/ic15-0303-zhou/epoch_60.pth')

# test a single image
img = mmcv.imread('demo/img_260.jpg')
short_e = min(img.shape[0], img.shape[1])
scale = 1120/short_e
img = cv2.resize(img, (0,0), fx=scale, fy=scale)

result = inference_detector(model, img, cfg)
# show_result(img, result)
print(result)

points = result[0]

import matplotlib.pyplot as plt
img_plt = img[:,:,::-1]
plt.imshow(img_plt)
for point in points:
    if point[8]<0.6:
        continue
    point = point[:8]
    px = point[::2]
    py = point[1::2]
    plt.fill(px, py, 'g', alpha=0.5)
plt.show()


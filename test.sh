#!/usr/bin/env bash
python setup.py install
python tools/test.py  configs/maskr50.py work_dirs/mask_rcnn_r50_fpn_1x-arpn/epoch_80.pth --show
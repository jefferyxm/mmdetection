#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python setup.py install
python tools/test.py  work_dirs/ic15-0303/maskr50.py work_dirs/ic15-0303/epoch_60.pth --show --dataset icdar2015

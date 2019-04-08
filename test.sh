#!/usr/bin/env bash
python setup.py install
python tools/test.py  work_dirs/ic15-0303-zhou/maskr50.py work_dirs/ic15-0303-zhou/epoch_60.pth --show --dataset icdar2015

#!/usr/bin/env bash
python setup.py install
python tools/test.py  configs/maskr50.py work_dirs/ic15-0303-zhou/epoch_60.pth --show

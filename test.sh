#!/usr/bin/env bash
python setup.py install
python tools/test.py  configs/maskr50.py work_dirs/debug/epoch_1.pth --show
#!/usr/bin/env bash
python setup.py install
python tools/test.py  configs/maskr50.py work_dirs/reg_4points/epoch_90.pth --show
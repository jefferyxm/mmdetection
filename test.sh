#!/usr/bin/env bash
python setup.py install
python tools/test.py  configs/maskr50.py work_dirs/mix-0203/epoch_40.pth --show --dataset td900
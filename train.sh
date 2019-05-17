#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python setup.py install
tools/dist_train.sh work_dirs/debug/maskr50.py 1

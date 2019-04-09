#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python setup.py install
tools/dist_train.sh configs/maskr50.py 1 --validate
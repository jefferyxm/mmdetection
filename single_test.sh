#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python setup.py install
python simple_test.py
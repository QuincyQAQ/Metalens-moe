#!/bin/bash

RESULT_DIR=${MOCEIR_TEST_RESULT_DIR:-test}
mkdir -p "$RESULT_DIR"
export MOCEIR_TEST_RESULT_DIR="$RESULT_DIR"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test.py "$@"
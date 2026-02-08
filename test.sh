#!/bin/bash

# 优先使用环境变量，其次使用配置中的 test_dir，最后使用默认值
if [ -z "${MOCEIR_TEST_RESULT_DIR:-}" ]; then
    RESULT_DIR=$(python -c 'import config; from pathlib import Path; p = Path(getattr(config, "TEST_DIR", "../test")); print(str(p.resolve()) if not p.is_absolute() else str(p))' 2>/dev/null || echo "../test")
else
    RESULT_DIR="$MOCEIR_TEST_RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"
export MOCEIR_TEST_RESULT_DIR="$RESULT_DIR"

# 如果环境变量未设置，使用硬编码的默认配置（用于独立测试）
# 这些配置可以通过环境变量覆盖，或者直接在这里修改
if [ -z "${MOCEIR_TEST_CKPT_PATH:-}" ]; then
    export MOCEIR_TEST_CKPT_PATH="/media/wsqlab/more/lqj/models/MoCE_IR_S-2026_01_29_23_33_45/checkpoints/best_psnr_ssim-epoch=0-psnr=13.747-ssim=0.1999.ckpt"
fi

if [ -z "${MOCEIR_TEST_DATA_FILE_DIR:-}" ]; then
    export MOCEIR_TEST_DATA_FILE_DIR="/media/wsqlab/more/lqj/data/open_dataset_8_1_1"
fi

if [ -z "${MOCEIR_TEST_TRAINSET:-}" ]; then
    export MOCEIR_TEST_TRAINSET="standard"
fi

if [ -z "${MOCEIR_TEST_BENCHMARKS:-}" ]; then
    export MOCEIR_TEST_BENCHMARKS="gopro"
fi

if [ -z "${MOCEIR_TEST_DE_TYPE:-}" ]; then
    export MOCEIR_TEST_DE_TYPE="deblur"
fi

if [ -z "${MOCEIR_TEST_PATCH_SIZE:-}" ]; then
    export MOCEIR_TEST_PATCH_SIZE="256"
fi

if [ -z "${MOCEIR_TEST_BATCH_SIZE:-}" ]; then
    export MOCEIR_TEST_BATCH_SIZE="1"
fi

if [ -z "${MOCEIR_TEST_SAVE_RESULTS:-}" ]; then
    export MOCEIR_TEST_SAVE_RESULTS="False"
fi

if [ -z "${MOCEIR_TEST_PRECISION:-}" ]; then
    export MOCEIR_TEST_PRECISION="fp16"
fi

if [ -z "${MOCEIR_TEST_FULL_RES_EVAL:-}" ]; then
    export MOCEIR_TEST_FULL_RES_EVAL="True"
fi

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test.py "$@"
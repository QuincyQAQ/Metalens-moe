#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MOCEIRV2_RUN_ID:-}" ]; then
  MODEL_NAME=$(python -c 'import config; print(str(getattr(config, "MODEL", "model")))' 2>/dev/null || echo "model")
  MODEL_NAME=${MODEL_NAME//\//_}
  MODEL_NAME=${MODEL_NAME// /_}

  # 从 DATA_FILE_DIR 或 TRAINSET 推断数据集名称，插入到 experiment 目录名中
  DATASET_NAME=$(python - "$@" << 'PY' 2>/dev/null || echo "data"
import os
import config

dataset_name = None
data_dir = getattr(config, "DATA_FILE_DIR", None)
if data_dir:
    dataset_name = os.path.basename(str(data_dir).rstrip(os.sep))

if not dataset_name:
    dataset_name = str(getattr(config, "TRAINSET", "")).strip() or "data"

dataset_name = dataset_name.replace(os.sep, "_").replace(" ", "_")
print(dataset_name)
PY
)

  # 形如: MoCE_IR_S-CVC_8_1_1_2026_02_13_17_00_43
  RUN_ID="${MODEL_NAME}-${DATASET_NAME}_$(date +%Y_%m_%d_%H_%M_%S)"
  export MOCEIRV2_RUN_ID="$RUN_ID"
else
  RUN_ID="$MOCEIRV2_RUN_ID"
fi

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py

# 使用配置中的 experiment_dir，如果不存在则使用默认值
EXPERIMENT_DIR=$(python -c 'import config; from pathlib import Path; import os; p = Path(getattr(config, "EXPERIMENT_DIR", "../experiment")); print(str(p.resolve()) if not p.is_absolute() else str(p))' 2>/dev/null || echo "../experiment")
CKPT_DIR="${EXPERIMENT_DIR}/${MOCEIRV2_RUN_ID}/checkpoints"

BEST_CKPT=""
BEST_CKPT=$(ls -1t "${CKPT_DIR}"/best_psnr_ssim*.ckpt 2>/dev/null | head -n 1 || true)
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT=$(ls -1t "${CKPT_DIR}"/best_psnr-epoch=*.ckpt 2>/dev/null | head -n 1 || true)
fi
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT="${CKPT_DIR}/last.ckpt"
fi

export MOCEIR_TEST_CKPT_PATH="$(readlink -f "$BEST_CKPT")"

# 传递训练时使用的 data_file_dir 和 trainset，确保测试使用与训练相同的数据集
# 使用 options.py 中的逻辑来获取配置，确保路径解析一致
DATA_FILE_DIR=$(python -c 'from options import train_options; opt = train_options(); print(opt.data_file_dir)' 2>/dev/null || echo "")
TRAINSET=$(python -c 'import config; print(str(getattr(config, "TRAINSET", "standard")))' 2>/dev/null || echo "standard")
DE_TYPE=$(python -c 'import config; de_type = getattr(config, "DE_TYPE", ["deblur"]); print(",".join(de_type))' 2>/dev/null || echo "deblur")

if [ -n "$DATA_FILE_DIR" ]; then
    export MOCEIR_TEST_DATA_FILE_DIR="$DATA_FILE_DIR"
fi
if [ -n "$TRAINSET" ]; then
    export MOCEIR_TEST_TRAINSET="$TRAINSET"
fi
if [ -n "$DE_TYPE" ]; then
    export MOCEIR_TEST_DE_TYPE="$DE_TYPE"
fi

bash test.sh
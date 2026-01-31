#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MOCEIRV2_RUN_ID:-}" ]; then
  MODEL_NAME=$(python -c 'import config; print(str(getattr(config, "MODEL", "model")))' 2>/dev/null || echo "model")
  MODEL_NAME=${MODEL_NAME//\//_}
  MODEL_NAME=${MODEL_NAME// /_}
  RUN_ID="${MODEL_NAME}-$(date +%Y_%m_%d_%H_%M_%S)"
  export MOCEIRV2_RUN_ID="$RUN_ID"
else
  RUN_ID="$MOCEIRV2_RUN_ID"
fi

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py

CKPT_DIR="experiment/${MOCEIRV2_RUN_ID}/checkpoints"

BEST_CKPT=""
BEST_CKPT=$(ls -1t "${CKPT_DIR}"/best_psnr_ssim*.ckpt 2>/dev/null | head -n 1 || true)
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT=$(ls -1t "${CKPT_DIR}"/best_psnr-epoch=*.ckpt 2>/dev/null | head -n 1 || true)
fi
if [ -z "$BEST_CKPT" ]; then
  BEST_CKPT="${CKPT_DIR}/last.ckpt"
fi

export MOCEIR_TEST_CKPT_PATH="$(readlink -f "$BEST_CKPT")"
bash test.sh
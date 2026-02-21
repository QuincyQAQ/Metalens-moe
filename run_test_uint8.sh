#!/bin/bash
set -u  # 不使用 -e，避免某个模型出错时整个脚本直接退出

# Script to run test_uint8.py on three best models and save test images

BASE_DIR="/media/wsqlab/backup/lqj/Metalens-moe"
SCRIPT_DIR="${BASE_DIR}/Metalens-moe"
RESULT_DIR="${SCRIPT_DIR}/result"

# Create result directory if it doesn't exist
mkdir -p "$RESULT_DIR"

cd "$SCRIPT_DIR"

# Model 1: CVC
echo "=========================================="
echo "Testing CVC model..."
echo "=========================================="
CKPT_CVC="${BASE_DIR}/MoCE_IR_S_SV_PhysicsPriorFusion-CVC_8_1_1_2026_02_19_20_40_42/checkpoints/best_psnr_ssim-epoch=379-psnr=39.23065-ssim=0.96908.ckpt"
DATA_CVC="/media/wsqlab/backup/lqj/data/CVC_8_1_1"

if [ ! -f "${CKPT_CVC}" ]; then
  echo "[WARN] Checkpoint not found for CVC: ${CKPT_CVC}"
  echo "[WARN] Skip CVC."
else
  if [ ! -d "${DATA_CVC}" ]; then
    echo "[WARN] Data directory not found for CVC: ${DATA_CVC}"
    echo "[WARN] Skip CVC."
  else
    export MOCEIR_TEST_CKPT_PATH="${CKPT_CVC}"
    export MOCEIR_TEST_DATA_FILE_DIR="${DATA_CVC}"
    export MOCEIR_TEST_TRAINSET="standard"
    export MOCEIR_TEST_BENCHMARKS="gopro"
    export MOCEIR_TEST_DE_TYPE="deblur"
    export MOCEIR_TEST_PATCH_SIZE="128"
    export MOCEIR_TEST_BATCH_SIZE="1"
    export MOCEIR_TEST_SAVE_RESULTS="True"
    export MOCEIR_TEST_RESULTS_DIR="${RESULT_DIR}"
    # NOTE: test_uint8.py uses MOCEIR_TEST_RESULTS_DIR for restored images,
    # but uses MOCEIR_TEST_RESULT_DIR for metrics (metrics.json/test.csv).
    # Point both to the same folder to keep outputs together.
    export MOCEIR_TEST_RESULT_DIR="${RESULT_DIR}"
    export MOCEIR_TEST_PRECISION="fp16"
    export MOCEIR_TEST_FULL_RES_EVAL="True"

    if ! CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test_uint8.py; then
      echo "[ERROR] Test failed for CVC, skip and continue."
    fi
  fi
fi

# Model 2: Endovis17
echo "=========================================="
echo "Testing Endovis17 model..."
echo "=========================================="
CKPT_ENDO="${BASE_DIR}/MoCE_IR_S_SV_PhysicsPriorFusion-Endovis17_8_1_1_2026_02_19_20_40_42/checkpoints/best_psnr_ssim-epoch=279-psnr=31.66963-ssim=0.90565.ckpt"
DATA_ENDO="/media/wsqlab/backup/lqj/data/Endovis17_8_1_1"

if [ ! -f "${CKPT_ENDO}" ]; then
  echo "[WARN] Checkpoint not found for Endovis17: ${CKPT_ENDO}"
  echo "[WARN] Skip Endovis17."
else
  if [ ! -d "${DATA_ENDO}" ]; then
    echo "[WARN] Data directory not found for Endovis17: ${DATA_ENDO}"
    echo "[WARN] Skip Endovis17."
  else
    export MOCEIR_TEST_CKPT_PATH="${CKPT_ENDO}"
    export MOCEIR_TEST_DATA_FILE_DIR="${DATA_ENDO}"
    export MOCEIR_TEST_TRAINSET="standard"
    export MOCEIR_TEST_BENCHMARKS="gopro"
    export MOCEIR_TEST_DE_TYPE="deblur"
    export MOCEIR_TEST_PATCH_SIZE="128"
    export MOCEIR_TEST_BATCH_SIZE="1"
    export MOCEIR_TEST_SAVE_RESULTS="True"
    export MOCEIR_TEST_RESULTS_DIR="${RESULT_DIR}"
    export MOCEIR_TEST_RESULT_DIR="${RESULT_DIR}"
    export MOCEIR_TEST_PRECISION="fp16"
    export MOCEIR_TEST_FULL_RES_EVAL="True"

    if ! CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test_uint8.py; then
      echo "[ERROR] Test failed for Endovis17, skip and continue."
    fi
  fi
fi

# Model 3: Kvasir_SEG
echo "=========================================="
echo "Testing Kvasir_SEG model..."
echo "=========================================="
export MOCEIR_TEST_CKPT_PATH="${BASE_DIR}/MoCE_IR_S_SV_PhysicsPriorFusion-Kvasir_SEG_8_1_1_2026_02_19_20_40_42/checkpoints/best_psnr_ssim-epoch=219-psnr=35.00221-ssim=0.93273.ckpt"
export MOCEIR_TEST_DATA_FILE_DIR="/media/wsqlab/backup/lqj/data/Kvasir_SEG_8_1_1"
export MOCEIR_TEST_TRAINSET="standard"
export MOCEIR_TEST_BENCHMARKS="gopro"
export MOCEIR_TEST_DE_TYPE="deblur"
export MOCEIR_TEST_PATCH_SIZE="128"
export MOCEIR_TEST_BATCH_SIZE="1"
export MOCEIR_TEST_SAVE_RESULTS="True"
export MOCEIR_TEST_RESULTS_DIR="${RESULT_DIR}"
export MOCEIR_TEST_RESULT_DIR="${RESULT_DIR}"
export MOCEIR_TEST_PRECISION="fp16"
export MOCEIR_TEST_FULL_RES_EVAL="True"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test_uint8.py

echo "=========================================="
echo "All tests completed!"
echo "Test images saved to: ${RESULT_DIR}"
echo "=========================================="


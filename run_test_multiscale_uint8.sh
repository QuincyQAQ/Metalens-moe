#!/bin/bash
set -euo pipefail

# Script to run test_uint8.py on all MultiScale experiments (CVC / Endovis17 / Kvasir_SEG)
# using full-resolution evaluation and aggregate metrics into result/test.csv

BASE_DIR="/media/wsqlab/backup/lqj/Metalens-moe"
SCRIPT_DIR="${BASE_DIR}/Metalens-moe"
RESULT_DIR="${SCRIPT_DIR}/result"

# Parent experiment directory for MultiScale runs (note: literal $ in folder name)
MULTISCALE_PARENT_DIR='/media/wsqlab/backup/lqj/Metalens-moe/experiment/MoCE_IR_S_SV_MultiScale-$DATASET_LIST_2026_02_16_15_26_33'

# Create result directory if it doesn't exist
mkdir -p "$RESULT_DIR"

cd "$SCRIPT_DIR"

echo "=========================================="
echo "MultiScale full-resolution testing"
echo "Experiment root: ${MULTISCALE_PARENT_DIR}"
echo "Metrics CSV: ${RESULT_DIR}/test.csv"
echo "=========================================="

# Optional: uncomment if you want to start from a fresh global CSV
# rm -f "${RESULT_DIR}/test.csv"

########################################
# Helper to run one test
########################################
run_test() {
  local CKPT_PATH="$1"
  local DATA_DIR="$2"
  local TAG="$3"

  echo "------------------------------------------"
  echo "Testing ${TAG} ..."
  echo "CKPT: ${CKPT_PATH}"
  echo "DATA: ${DATA_DIR}"
  echo "------------------------------------------"

  export MOCEIR_TEST_CKPT_PATH="${CKPT_PATH}"
  export MOCEIR_TEST_DATA_FILE_DIR="${DATA_DIR}"
  export MOCEIR_TEST_TRAINSET="standard"
  export MOCEIR_TEST_BENCHMARKS="gopro"
  export MOCEIR_TEST_DE_TYPE="deblur"
  export MOCEIR_TEST_PATCH_SIZE="128"
  export MOCEIR_TEST_BATCH_SIZE="1"
  export MOCEIR_TEST_SAVE_RESULTS="True"
  export MOCEIR_TEST_RESULTS_DIR="${RESULT_DIR}"
  # test_uint8.py uses MOCEIR_TEST_RESULT_DIR for metrics (metrics.json/test.csv)
  export MOCEIR_TEST_RESULT_DIR="${RESULT_DIR}"
  export MOCEIR_TEST_PRECISION="fp16"
  export MOCEIR_TEST_FULL_RES_EVAL="True"

  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 test_uint8.py
}

########################################
# CVC
########################################
CVC_EXP_DIR="${MULTISCALE_PARENT_DIR}/MoCE_IR_S_SV_MultiScale-CVC_8_1_1_2026_02_16_15_26_33"
CVC_CKPT="${CVC_EXP_DIR}/checkpoints/last.ckpt"
CVC_DATA="/media/wsqlab/backup/lqj/data/CVC_8_1_1"
run_test "${CVC_CKPT}" "${CVC_DATA}" "MultiScale-CVC"

########################################
# Endovis17
########################################
ENDOVIS_EXP_DIR="${MULTISCALE_PARENT_DIR}/MoCE_IR_S_SV_MultiScale-Endovis17_8_1_1_2026_02_16_15_26_33"
ENDOVIS_CKPT="${ENDOVIS_EXP_DIR}/checkpoints/last.ckpt"
ENDOVIS_DATA="/media/wsqlab/backup/lqj/data/Endovis17_8_1_1"
run_test "${ENDOVIS_CKPT}" "${ENDOVIS_DATA}" "MultiScale-Endovis17"

########################################
# Kvasir_SEG
########################################
KVASIR_EXP_DIR="${MULTISCALE_PARENT_DIR}/MoCE_IR_S_SV_MultiScale-Kvasir_SEG_8_1_1_2026_02_16_15_26_33"
KVASIR_CKPT="${KVASIR_EXP_DIR}/checkpoints/last.ckpt"
KVASIR_DATA="/media/wsqlab/backup/lqj/data/Kvasir_SEG_8_1_1"
run_test "${KVASIR_CKPT}" "${KVASIR_DATA}" "MultiScale-Kvasir_SEG"

echo "=========================================="
echo "All MultiScale tests completed!"
echo "Global CSV (aggregated metrics): ${RESULT_DIR}/test.csv"
echo "Per-experiment CSVs are also written under each experiment's test/ directory."
echo "=========================================="



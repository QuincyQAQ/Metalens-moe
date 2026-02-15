#!/usr/bin/env bash
set -euo pipefail

# 加载 Bark 推送通知功能（训练完成后自动发送通知）
source "$(dirname "$0")/bark_notify.sh"

# 设置环境变量来抑制 PyTorch 分布式警告
export NCCL_DEBUG=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_SHOW_CPP_STACKTRACES=0
export OMP_NUM_THREADS=1  # 设置这个变量以避免警告

# 获取数据集列表
DATASET_LIST=$(python - << 'PY' 2>/dev/null || echo ""
import os
import config
from pathlib import Path

# 检查是否有 DATA_FILE_DIRS
if hasattr(config, "DATA_FILE_DIRS") and isinstance(config.DATA_FILE_DIRS, list):
    data_file_dirs = config.DATA_FILE_DIRS
else:
    # 兼容旧的单个数据集配置
    data_file_dir = getattr(config, "DATA_FILE_DIR", None)
    if data_file_dir:
        data_file_dirs = [data_file_dir]
    else:
        data_file_dirs = []

project_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
for data_dir in data_file_dirs:
    data_dir_path = Path(str(data_dir)).expanduser()
    if not data_dir_path.is_absolute():
        data_dir_path = (project_dir / data_dir_path).resolve()
    else:
        data_dir_path = data_dir_path.resolve()
    print(str(data_dir_path))
PY
)

# 如果没有找到数据集列表，使用默认行为（单个数据集或合并训练）
if [ -z "$DATASET_LIST" ]; then
  # 原有的单个数据集训练逻辑
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

  # 运行训练，使用 sed 过滤掉 OMP_NUM_THREADS 相关的警告行
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py 2>&1 | \
    sed '/Setting OMP_NUM_THREADS/d; /to avoid your system being overloaded/d; /please further tune the variable/d; /site-packages\/torch\/distributed\/run\.py/d; /^\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*$/d'

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

  # 设置环境变量，记录本次训练的数据集名称（用于通知过滤）
  export MOCEIR_TRAINED_DATASETS="$DATASET_NAME"

  # 注释掉：train.py 已经在训练完成后自动测试，不需要再次调用 test.sh
  # bash test.sh
else
  # 循环处理每个数据集
  MODEL_NAME=$(python -c 'import config; print(str(getattr(config, "MODEL", "model")))' 2>/dev/null || echo "model")
  MODEL_NAME=${MODEL_NAME//\//_}
  MODEL_NAME=${MODEL_NAME// /_}

  echo "Found multiple datasets, will train on each separately:"
  
  # 提取所有数据集的共同前缀
  # 例如：open_dataset_8_1_1_mini, open_dataset_8_1_1_mini2, open_dataset_8_1_1_mini3 -> open_dataset_8_1_1
  COMMON_PREFIX=$(python - << 'PY' 2>/dev/null || echo ""
import os
from pathlib import Path

dataset_list_str = """$DATASET_LIST"""
if not dataset_list_str:
    print("")
    exit(0)

# 解析所有数据集名称
dataset_names = []
for data_dir in dataset_list_str.strip().split('\n'):
    if data_dir:
        dataset_name = Path(data_dir).name
        dataset_names.append(dataset_name)

if not dataset_names:
    print("")
    exit(0)

# 找到所有数据集名称的最长公共前缀（逐字符比较）
def find_longest_common_prefix(strings):
    if not strings:
        return ""
    if len(strings) == 1:
        return strings[0]
    
    # 找到最短的字符串长度
    min_len = min(len(s) for s in strings)
    
    # 逐字符比较，找到最长公共前缀
    common_prefix = ""
    for i in range(min_len):
        char = strings[0][i]
        if all(s[i] == char for s in strings):
            common_prefix += char
        else:
            break
    
    # 如果公共前缀不是完整的字符串，找到最后一个下划线的位置
    # 例如：open_dataset_8_1_1_mini -> open_dataset_8_1_1
    if common_prefix and common_prefix != strings[0]:
        last_underscore = common_prefix.rfind('_')
        if last_underscore > 0:
            common_prefix = common_prefix[:last_underscore]
    
    return common_prefix

common_prefix = find_longest_common_prefix(dataset_names)
print(common_prefix)
PY
)
  
  # 生成父文件夹名称（基于共同前缀和时间戳）
  TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)
  if [ -n "$COMMON_PREFIX" ]; then
    PARENT_EXP_DIR="${MODEL_NAME}-${COMMON_PREFIX}_${TIMESTAMP}"
  else
    # 如果没有共同前缀，使用第一个数据集的名称
    FIRST_DATASET=$(echo "$DATASET_LIST" | head -n 1)
    FIRST_DATASET_NAME=$(basename "$FIRST_DATASET")
    FIRST_DATASET_NAME=${FIRST_DATASET_NAME//\//_}
    FIRST_DATASET_NAME=${FIRST_DATASET_NAME// /_}
    PARENT_EXP_DIR="${MODEL_NAME}-${FIRST_DATASET_NAME}_${TIMESTAMP}"
  fi
  
  # 设置父文件夹环境变量
  export MOCEIRV2_PARENT_EXP_DIR="$PARENT_EXP_DIR"
  
  echo "Parent experiment directory: $PARENT_EXP_DIR"
  echo ""
  
  # 存储所有 RUN_ID 用于后续汇总结果
  RUN_IDS=()
  
  # 存储本次训练的数据集名称列表（用于通知过滤）
  TRAINED_DATASET_NAMES=()
  
  # 使用进程替换避免子shell问题
  while IFS= read -r DATA_DIR; do
    if [ -z "$DATA_DIR" ]; then
      continue
    fi
    
    # 获取数据集名称
    DATASET_NAME=$(basename "$DATA_DIR")
    DATASET_NAME=${DATASET_NAME//\//_}
    DATASET_NAME=${DATASET_NAME// /_}
    
    # 生成 RUN_ID（包含完整路径：父文件夹/子文件夹）
    RUN_ID="${PARENT_EXP_DIR}/${MODEL_NAME}-${DATASET_NAME}_${TIMESTAMP}"
    export MOCEIRV2_RUN_ID="$RUN_ID"
    export MOCEIR_TRAIN_DATA_FILE_DIR="$DATA_DIR"
    RUN_IDS+=("$RUN_ID")
    TRAINED_DATASET_NAMES+=("$DATASET_NAME")
    
    echo "=========================================="
    echo "Training on dataset: $DATASET_NAME"
    echo "RUN_ID: $RUN_ID"
    echo "Data directory: $DATA_DIR"
    echo "=========================================="
    
    # 运行训练，使用 sed 过滤掉 OMP_NUM_THREADS 相关的警告行
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py 2>&1 | \
      sed '/Setting OMP_NUM_THREADS/d; /to avoid your system being overloaded/d; /please further tune the variable/d; /site-packages\/torch\/distributed\/run\.py/d; /^\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*$/d'
    
    echo "Training completed for dataset: $DATASET_NAME"
    
    # 训练完成后运行测试
    EXPERIMENT_DIR=$(python -c 'import config; from pathlib import Path; import os; p = Path(getattr(config, "EXPERIMENT_DIR", "../experiment")); print(str(p.resolve()) if not p.is_absolute() else str(p))' 2>/dev/null || echo "../experiment")
    CKPT_DIR="${EXPERIMENT_DIR}/${RUN_ID}/checkpoints"
    
    BEST_CKPT=""
    BEST_CKPT=$(ls -1t "${CKPT_DIR}"/best_psnr_ssim*.ckpt 2>/dev/null | head -n 1 || true)
    if [ -z "$BEST_CKPT" ]; then
      BEST_CKPT=$(ls -1t "${CKPT_DIR}"/best_psnr-epoch=*.ckpt 2>/dev/null | head -n 1 || true)
    fi
    if [ -z "$BEST_CKPT" ]; then
      BEST_CKPT="${CKPT_DIR}/last.ckpt"
    fi
    
    export MOCEIR_TEST_CKPT_PATH="$(readlink -f "$BEST_CKPT" 2>/dev/null || echo "$BEST_CKPT")"
    
    # 传递训练时使用的 data_file_dir 和 trainset
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
    
    # 注释掉：train.py 已经在训练完成后自动测试，不需要再次调用 test.sh
    # echo "Running test for dataset: $DATASET_NAME"
    # bash test.sh 2>&1 | sed '/Setting OMP_NUM_THREADS/d; /to avoid your system being overloaded/d; /please further tune the variable/d; /site-packages\/torch\/distributed\/run\.py/d; /^\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*$/d'
    echo ""
  done <<< "$DATASET_LIST"
  
  echo "All datasets training completed!"
  echo ""
  
  # 设置环境变量，记录本次训练的数据集名称（用于通知过滤）
  export MOCEIR_TRAINED_DATASETS=$(IFS=','; echo "${TRAINED_DATASET_NAMES[*]}")
  
  # 汇总并打印所有数据集的测试结果
  echo "=========================================="
  echo "Summary of Test Results for All Datasets"
  echo "=========================================="
  
  # 获取 test.csv 路径
  RESULT_DIR=$(python -c 'import config; from pathlib import Path; p = Path(getattr(config, "TEST_DIR", "../test")); print(str(p.resolve()) if not p.is_absolute() else str(p))' 2>/dev/null || echo "../test")
  CSV_PATH="${RESULT_DIR}/test.csv"
  
  if [ -f "$CSV_PATH" ]; then
    # 使用 Python 读取并格式化打印 CSV 文件
    python3 << PYEOF
import csv
import sys
from pathlib import Path

# 从环境变量获取 CSV 路径和数据集列表
csv_path = Path("$CSV_PATH")
dataset_list_str = """$DATASET_LIST"""

if not csv_path.exists():
    print(f"Test results file not found: {csv_path}")
    sys.exit(1)

# 解析本次训练的数据集名称列表
trained_datasets = set()
if dataset_list_str:
    for data_dir in dataset_list_str.strip().split('\n'):
        if data_dir:
            # 从路径中提取数据集名称（basename）
            dataset_name = Path(data_dir).name
            trained_datasets.add(dataset_name)

# 读取 CSV 文件（从后往前读取，以便获取最新的结果）
results = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

if not results:
    print("No test results found in CSV file.")
    sys.exit(0)

# 从后往前遍历，为每个数据集找到最后一次测试的结果
# 只处理本次训练的数据集
dataset_last_results = {}
for row in reversed(results):
    dataset = row.get('dataset', 'unknown')
    
    # 跳过 dataset 列是数字的旧数据（这些可能是 gflops 值）
    try:
        float(dataset)
        continue
    except (ValueError, TypeError):
        pass
    
    # 只处理本次训练的数据集
    if trained_datasets and dataset not in trained_datasets:
        continue
    
    # 如果这个数据集还没有记录，则记录它（因为是从后往前遍历，所以这是最新的）
    if dataset not in dataset_last_results:
        dataset_last_results[dataset] = row

# 按数据集名称排序
sorted_datasets = sorted(dataset_last_results.keys())

if not sorted_datasets:
    print("No test results found for trained datasets.")
    sys.exit(0)

# 打印汇总结果
print(f"\nFound {len(sorted_datasets)} dataset(s) with test results:\n")
for dataset_name in sorted_datasets:
    row = dataset_last_results[dataset_name]
    print(f"Dataset: {dataset_name}")
    print("-" * 80)
    print(f"{'Benchmark':<15} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}")
    print("-" * 80)
    benchmark = row.get('benchmark', 'N/A')
    psnr = row.get('psnr', 'N/A')
    ssim = row.get('ssim', 'N/A')
    lpips = row.get('lpips', 'N/A')
    print(f"{benchmark:<15} {psnr:<10} {ssim:<10} {lpips:<10}")
    print("")
PYEOF
    echo "Full results saved to: $CSV_PATH"
  else
    echo "Test results file not found: $CSV_PATH"
  fi
  
  echo "=========================================="
fi
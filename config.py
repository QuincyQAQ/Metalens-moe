import os
import pathlib

# ============================================================================
# 基础训练设置
# ============================================================================
# MODEL 可选: "MoCE_IR", "MoCE_IR_S", "ACFormer", "MoCE_IR_PhysRouting", "MoCE_IR_Spectral"
MODEL = "MoCE_IR_S"  # "MoCE_IR" 或 "MoCE_IR_S" 或 "ACFormer"或者 MoCE_IR_PhysRouting 或 "MoCE_IR_Spectral"
EPOCHS = 1
BATCH_SIZE = 32  # 每个GPU的batch size 20
VAL_EVERY_N_EPOCH = 20  # 每多少个epoch做一次验证
LR = 2e-4
# ============================================================================ open_dataset_8_1_1_mini


# 路径设置 - 支持多个数据集同时训练
DATA_FILE_DIRS = [
    "../../data/open_dataset_8_1_1_mini",
    "../../data/open_dataset_8_1_1_mini2",
    "../../data/open_dataset_8_1_1_mini3"
]

# experiment 根目录：训练产生的所有实验目录（checkpoints、metrics等）  
# CVC_8_1_1      或    Kvasir_SEG_8_1_1
# 相对于项目目录的路径，或使用绝对路径
EXPERIMENT_DIR = "../experiment"
# test 根目录：测试结果保存目录
# 相对于项目目录的路径，或使用绝对路径
TEST_DIR = "../test"
# results 根目录：测试时保存的恢复图像结果
# 相对于项目目录的路径，或使用绝对路径
RESULTS_DIR = "../results"
# ckpt 根目录：包含各个 experiment 子目录（用于测试时指定checkpoint路径）
CKPT_DIR = "../experiment"
# 具体要测的那一次实验的子目录（到 checkpoints 这一层）
CHECKPOINT_ID = "2026_01_22_20_18_57/checkpoints"  # 例子，换成你自己的

# ============================================================================ open_dataset_8_1_1_mini



DE_TYPE = ["deblur"]  # 可选: "denoise_15/25/50", "dehaze", "derain", "deblur", "synllie"
TRAINSET = "standard"  # "standard" 或 "CDD11_*"
LOSS_TYPE = "L1"  # "L1" 或 "fft" focal_l1
PATCH_SIZE = 128
BALANCE_LOSS_WEIGHT = 0.01
FFT_LOSS_WEIGHT = 1.0

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.1
FOCAL_EPSILON = 1e-6

DE_AUX_LOSS_WEIGHT = 0.0
DE_AUX_GAMMA = 2.0
DE_AUX_ALPHA = None
DE_AUX_USE_EXTERNAL_FOCAL = True
ACCUM_GRAD = 1
PRINT_MODEL = False

RESUME_FROM = None  # 从checkpoint恢复训练
FINE_TUNE_FROM = None # 微调checkpoint
CHECKPOINT_ID = None
BENCHMARKS = ["gopro"]
SAVE_RESULTS = True

# ============================================================================
# 性能相关
# ============================================================================
DETERMINISTIC = False
BENCHMARK = True
PRECISION = "16-mixed"  # "16-mixed" 或 "bf16-mixed" (A100/H100)
TF32 = True
LOG_EVERY_N_STEPS = 10
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True


# ============================================================================
# 路径设置
# ============================================================================ open_dataset_8_1_1_mini

OUTPUT_PATH = "output/"
WBLOGGER = False
NUM_GPUS = 2
NUM_WORKERS = 12

# ============================================================================
# 通知设置
# ============================================================================
# 是否启用 Bark 推送通知
ENABLE_BARK_NOTIFICATION = True  # 设置为 False 可禁用通知
# Bark API URL（去掉末尾的"这里改成你自己的推送内容"部分）
BARK_API_URL = "https://api.day.app/ZBPuur5RhDtQHa6KPECoJX"

# ============================================================================ open_dataset_8_1_1_mini

import os
import pathlib

# ============================================================================
# 基础训练设置
# ============================================================================
# MODEL 可选: "MoCE_IR", "MoCE_IR_S", "ACFormer"
MODEL = "MoCE_IR_S"  # "MoCE_IR" 或 "MoCE_IR_S" 或 "ACFormer"
EPOCHS = 1
BATCH_SIZE = 16  # 每个GPU的batch size 20
LR = 2e-4

DE_TYPE = ["deblur"]  # 可选: "denoise_15/25/50", "dehaze", "derain", "deblur", "synllie"
TRAINSET = "standard"  # "standard" 或 "CDD11_*"
LOSS_TYPE = "focal_l1"  # "L1" 或 "fft" focal_l1
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
VAL_EVERY_N_EPOCH = 5  # 每多少个epoch做一次验证

# ============================================================================
# 路径设置
# ============================================================================ open_dataset_8_1_1_mini

OUTPUT_PATH = "output/"
WBLOGGER = False
NUM_GPUS = 2
NUM_WORKERS = 12

# 路径设置
DATA_FILE_DIR = "../../data/open_dataset_8_1_1"
# ckpt 根目录：包含各个 experiment 子目录
CKPT_DIR = "../moceir/experiment"
# 具体要测的那一次实验的子目录（到 checkpoints 这一层）
CHECKPOINT_ID = "2026_01_22_20_18_57/checkpoints"  # 例子，换成你自己的

## 项目说明

### 项目目录结构

本项目采用分离式目录结构，将代码和训练产生的数据分开存储，便于版本控制和代码管理：

```
项目根目录/
├── Metalens-moe/              # 项目代码目录（可上传到 GitHub）
│   ├── config.py              # 配置文件
│   ├── train.py               # 训练脚本
│   ├── test.py                # 测试脚本
│   ├── train.sh               # 训练启动脚本
│   ├── test.sh                # 测试启动脚本
│   ├── net/                   # 网络定义目录
│   ├── data/                  # 数据处理模块
│   ├── utils/                 # 工具函数
│   └── README.md              # 本文件
│
├── experiment/                # 训练输出目录（外部，不提交到 Git）
│   ├── <模型名-数据集-时间戳>/       # 单个数据集训练的实验目录
│   │   ├── checkpoints/       # 模型检查点
│   │   │   ├── last.ckpt      # 最后一个 epoch 的检查点
│   │   │   ├── best_psnr-*.ckpt  # 最佳 PSNR 检查点
│   │   │   └── best_psnr_ssim-*.ckpt  # 最佳综合指标检查点
│   │   ├── net_snapshot/      # 网络文件快照（用于独立测试）
│   │   ├── config_snapshot/   # 配置文件快照
│   │   ├── metrics.csv        # 训练指标记录
│   │   └── opt.json           # 训练配置 JSON
│   │
│   └── <模型名-共同前缀_时间戳>/     # 多数据集训练的父目录
│       ├── <模型名-数据集1_时间戳>/  # 数据集1的训练目录
│       ├── <模型名-数据集2_时间戳>/  # 数据集2的训练目录
│       └── <模型名-数据集3_时间戳>/  # 数据集3的训练目录
│
├── test/                      # 测试结果目录（外部，不提交到 Git）
│   ├── metrics.json           # 测试指标（JSON 格式）
│   ├── metrics.txt            # 测试指标（文本格式）
│   └── test.csv               # 测试结果汇总 CSV（所有模型的测试结果）
│
└── results/                   # 恢复图像结果目录（外部，不提交到 Git）
    └── <checkpoint_id>/       # 按检查点 ID 组织
        └── <benchmark>/       # 按测试集组织
            └── rank<N>/       # 按 GPU rank 组织
                └── *.png      # 恢复的图像文件
```

**目录说明：**

- **`Metalens-moe/`**：项目代码目录，包含所有源代码和配置文件，可以上传到 GitHub。
- **`experiment/`**：训练产生的所有实验数据，包括：
  - `checkpoints/`：模型权重文件
  - `net_snapshot/`：训练时使用的网络文件副本（用于测试时加载）
  - `config_snapshot/`：训练时使用的配置文件副本
  - `metrics.csv`：训练过程中的 PSNR/SSIM/LPIPS 等指标记录
- **`test/`**：测试评估结果，包括指标文件和汇总表
- **`results/`**：测试时保存的恢复图像（当 `SAVE_RESULTS=True` 时）

**路径配置：**

在 `config.py` 中可以配置这些外部目录的路径：
- `EXPERIMENT_DIR`：experiment 目录路径（默认：`"../experiment"`）
- `TEST_DIR`：test 目录路径（默认：`"../test"`）
- `RESULTS_DIR`：results 目录路径（默认：`"../results"`）

可以使用相对路径（相对于项目目录）或绝对路径。

## 训练逻辑详解

### 单数据集训练

当 `config.py` 中只配置了单个数据集（使用 `DATA_FILE_DIR` 或 `DATA_FILE_DIRS` 只有一个元素）时：

1. **训练流程**：
   - 运行 `bash train.sh` 或 `python train.py`
   - 系统会创建一个实验目录：`experiment/<模型名-数据集名_时间戳>/`
   - 训练完成后，**自动在相同数据集上运行测试**（使用训练时的数据集）
   - 测试结果会追加到 `test/test.csv` 文件中

2. **目录结构示例**：
   ```
   experiment/
   └── MoCE_IR_S-open_dataset_8_1_1_2026_02_14_18_43_23/
       ├── checkpoints/
       ├── net_snapshot/
       └── ...
   ```

### 多数据集训练（一次性训练 n 个模型）

当 `config.py` 中配置了多个数据集（`DATA_FILE_DIRS` 是一个包含多个路径的列表）时：

1. **训练流程**：
   - 运行 `bash train.sh`
   - 系统会**自动循环训练每个数据集**，每个数据集训练一个独立的模型
   - 训练顺序按照 `DATA_FILE_DIRS` 列表中的顺序

2. **目录组织**：
   - 系统会创建一个**父实验目录**，名称基于所有数据集的共同前缀和时间戳
   - 每个数据集会在父目录下创建**子实验目录**
   - 目录结构示例：
     ```
     experiment/
     └── MoCE_IR_S-open_dataset_8_1_1_2026_02_14_18_43_23/  # 父目录
         ├── MoCE_IR_S-open_dataset_8_1_1_mini_2026_02_14_18_43_23/  # 数据集1
         ├── MoCE_IR_S-open_dataset_8_1_1_mini2_2026_02_14_18_43_23/  # 数据集2
         └── MoCE_IR_S-open_dataset_8_1_1_mini3_2026_02_14_18_43_23/  # 数据集3
     ```

3. **每个数据集的训练过程**：
   - 对于每个数据集，系统会：
     1. 设置环境变量 `MOCEIR_TRAIN_DATA_FILE_DIR` 指向当前数据集
     2. 创建独立的实验子目录
     3. 在该数据集上训练模型
     4. **训练完成后自动测试**（使用训练时相同的数据集）
     5. 测试结果追加到 `test/test.csv`，并记录对应的数据集名称

4. **结果汇总**：
   - 所有数据集的训练完成后，`train.sh` 会**自动汇总所有数据集的测试结果**
   - 在终端打印每个数据集的最新测试指标（PSNR、SSIM、LPIPS）
   - 完整的测试结果保存在 `test/test.csv` 中，包含：
     - 模型名称
     - 数据集名称（确保与训练时使用的数据集一致）
     - 测试指标（PSNR、SSIM、LPIPS等）
     - 时间戳等信息

5. **配置示例**：
   ```python
   # config.py
   MODEL = "MoCE_IR_S"
   DATA_FILE_DIRS = [
       "../../data/open_dataset_8_1_1_mini",
       "../../data/open_dataset_8_1_1_mini2",
       "../../data/open_dataset_8_1_1_mini3"
   ]
   ```
   运行 `bash train.sh` 后，会依次训练3个模型：
   - 模型1：在 `open_dataset_8_1_1_mini` 上训练 → 测试使用 `open_dataset_8_1_1_mini`
   - 模型2：在 `open_dataset_8_1_1_mini2` 上训练 → 测试使用 `open_dataset_8_1_1_mini2`
   - 模型3：在 `open_dataset_8_1_1_mini3` 上训练 → 测试使用 `open_dataset_8_1_1_mini3`

6. **重要特性**：
   - **数据集一致性保证**：每个模型训练完成后，测试时会**自动使用训练时相同的数据集**，确保测试结果与训练数据集对应
   - **独立模型**：每个数据集训练出的模型是**完全独立的**，互不影响
   - **自动测试**：每个模型训练完成后**自动测试**，无需手动运行测试脚本
   - **结果追踪**：`test/test.csv` 中会记录每个模型对应的数据集名称，便于后续分析和对比

## 数据集准备

### 从魔塔社区（ModelScope）下载数据集

本项目支持从魔塔社区（ModelScope）下载数据集。按照以下步骤操作：

1. **安装 ModelScope**：
   ```bash
   pip install modelscope
   ```

2. **登录 ModelScope**（使用提供的 token）：
   ```bash
   modelscope login --token ms-8f83a86e-1a85-400c-90d8-2d0682d859d9
   ```

3. **下载数据集**：
   ```bash
   modelscope download --dataset quincy123123/ronghe
   ```

下载完成后，数据集会保存在 ModelScope 的默认缓存目录中。你可以在 `config.py` 中配置 `DATA_FILE_DIR` 或 `DATA_FILE_DIRS` 指向下载的数据集路径。

### 推荐使用流程

1. **配置环境**：修改 `config.py`，确定模型、数据路径和训练的超参。
   - 设置 `MODEL` 选择要使用的网络
   - **单数据集训练**：配置 `DATA_FILE_DIR` 指向训练数据目录
   - **多数据集训练**：配置 `DATA_FILE_DIRS` 为包含多个数据集路径的列表
   - 配置 `EXPERIMENT_DIR`、`TEST_DIR`、`RESULTS_DIR` 指向外部目录（如需要）
   
2. **开始训练**：运行 `bash train.sh` 开始训练
   - **单数据集**：训练过程会在 `experiment/<模型名-数据集名_时间戳>/` 下创建实验目录
   - **多数据集**：训练过程会在 `experiment/<模型名-共同前缀_时间戳>/` 下创建父目录，每个数据集有独立的子目录
   - 自动保存检查点、指标记录和网络快照
   - 训练完成后会自动运行测试（使用训练时相同的数据集）
   - 在 `test/test.csv` 中会存有所有模型的详细指标数据
   
3. **评估模型**：运行 `test.sh` 进行单独的模型评估
   - 支持多卡评估（双卡全分辨率 `batch=1`）
   - 单卡评估可直接运行 `python test.py`
   - 测试结果保存在 `test/` 目录
   - 恢复的图像保存在 `results/` 目录（如果 `SAVE_RESULTS=True`）
   
4. **单张图像推理**：使用 `python infer_image.py --ckpt <ckpt> --input <img> --output <out>`
   - 对单张图像进行推理和恢复

## 如何运行新创建的网络

当你创建了一个新的网络文件（例如 `net/YourModel.py`）后，需要按以下步骤操作：

### 步骤 1：确认网络文件结构
确保你的网络文件包含：
- 网络模型类定义（例如 `class YourModel(nn.Module)`）
- `build_model(opt)` 函数，用于根据配置选项构建模型

### 步骤 2：修改 config.py
在 `config.py` 中修改 `MODEL` 变量，将其设置为你的新网络名称（不包含 `.py` 后缀）：
```python
MODEL = "YourModel"  # 你的新网络名称
```

### 步骤 3：（可选）添加模型配置
如果需要自定义模型参数，可以在 `config.py` 中添加模型配置字典：
```python
YourModel_CONFIG = {
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    # ... 其他参数
}
```
如果不添加配置，系统会使用 `build_model` 函数中的默认值。

### 步骤 4：运行训练
直接运行训练脚本：
```bash
bash train.sh
# 或
python train.py
```

系统会自动：
- 从 `net/<MODEL>.py` 动态加载网络
- 使用 `build_model(opt)` 函数构建模型
- 开始训练并保存结果到 `experiment/` 目录

## 核心文件说明

- **train.py**  
  训练脚本。从 `config.py` 读取所有训练配置（模型类型、batch size、学习率、数据路径等），并自动：
  - 动态加载 `net/<MODEL>.py` 中的网络（`MODEL` 在 `config.py` 里设置）。
  - 使用 PyTorch Lightning 和 DDP 进行单机多卡训练。
  - 在 `experiment/<模型名-数据集名_时间戳>/` 下保存（路径由 `config.EXPERIMENT_DIR` 配置）：
    - `checkpoints/`：`last.ckpt` 和最佳指标 ckpt（`best_psnr-*.ckpt`、`best_psnr_ssim-*.ckpt`）。
    - `net_snapshot/`：本次训练使用的单个网络文件快照（独立可用，用于测试时加载）。
    - `config_snapshot/`：本次训练使用的配置文件快照。
    - `metrics.csv`：按 epoch 记录的 PSNR/SSIM/LPIPS 等指标。
    - `opt.json`：训练配置的 JSON 格式备份。
  - **训练完成后自动测试**：在训练时使用的数据集上运行测试，确保测试结果与训练数据集一致。

- **train.sh**  
  训练启动脚本，负责：
  - 检测 `config.py` 中的数据集配置（单个或多个）
  - **单数据集模式**：直接启动训练
  - **多数据集模式**：循环训练每个数据集，每个数据集训练独立的模型
  - 自动设置环境变量，确保训练和测试使用相同的数据集
  - 训练完成后自动汇总所有数据集的测试结果

- **config.py**  
  训练配置文件，只对 `train.py` / `infer_image.py` 有效，用来统一管理：
  - 模型选择：`MODEL`（如 `"MoCE_IR_S"` / `"ACFormer"` 等）。
  - 数据配置：
    - `DATA_FILE_DIR`：单个数据集路径（单数据集训练）
    - `DATA_FILE_DIRS`：多个数据集路径列表（多数据集训练）
  - 训练与数据相关参数：训练超参、workers、精度、验证频率等（按需修改）。
  - 可选模型配置：可以为每个模型添加 `{MODEL}_CONFIG` 字典来自定义模型参数。

- **test.py**  
  测试脚本，**完全独立于 `config.py`**，支持通过环境变量或命令行参数配置：
  - `ckpt_path`：要测试的 ckpt 绝对路径（支持 Lightning checkpoint）。
  - `benchmarks`：如 `["gopro"]`、`["drmi"]` 等测试集列表。
  - `patch_size`：patch 测试时的 patch 大小（默认 128 或 256）。
  - `full_res_eval`：
    - `True`：全图像评估。
    - `False`：使用中心 patch 评估。
  - `save_results`：是否保存恢复的图像（默认 `False`）。
  
  测试结果保存：
  - 测试指标保存在 `test/` 目录（路径由 `config.TEST_DIR` 或环境变量配置）。
  - 恢复的图像保存在 `results/` 目录（路径由 `config.RESULTS_DIR` 配置，当 `save_results=True` 时）。
  
  网络加载顺序：
  1. 优先从 ckpt 同级的 `../net_snapshot/` 加载网络文件（训练时保存的快照）。
  2. 如不存在，则回退到项目内 `net/<MODEL>.py`。

- **test.sh** 
  测试脚本，写好了所有配置，直接运行即可。支持通过环境变量覆盖配置。

- **test_bucketing.py**
  测试脚本（Accelerate 多卡可用），用于全分辨率 `batch_size > 1` 的加速评估：
  - 通过 padding collate 解决不同分辨率无法组成 batch 的问题。
  - 通过按分辨率分桶/排序减少 padding 浪费。
  - 但是一般情况下优先用 `test.py`（更接近原始评测方式）；需要更快全分辨率评测时再用这个。

- **infer_image.py**  
  单张图像推理脚本：
  - 通过命令行参数指定 `--ckpt`、`--input`、`--output`、`--device`。
  - 内部使用 `train_options()` 加载 `config.py`，动态加载 `net/<MODEL>.py` 并读取 ckpt 权重，然后对单张图像做恢复。

- **plot_metrics.py**  
  指标可视化脚本（待完善）。

### 配置文件说明

- **config.py**  
  所有配置都在此文件中，包括：
  - **模型选择**：`MODEL`（如 `"MoCE_IR_S"` / `"ACFormer"` 等）。
  - **训练参数**：epochs、batch_size、learning_rate、loss_type 等。
  - **数据路径**：
    - `DATA_FILE_DIR`：单个数据集路径（单数据集训练）
    - `DATA_FILE_DIRS`：多个数据集路径列表（多数据集训练，推荐使用）
  - **外部目录路径**：
    - `EXPERIMENT_DIR`：训练输出目录（默认 `"../experiment"`）。
    - `TEST_DIR`：测试结果目录（默认 `"../test"`）。
    - `RESULTS_DIR`：恢复图像结果目录（默认 `"../results"`）。
  - **可选模型配置**：可以为每个模型添加 `{MODEL}_CONFIG` 字典来自定义模型参数。

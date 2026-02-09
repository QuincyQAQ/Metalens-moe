## Metalens-moe 多网络版本）

### 项目目录结构

本项目采用分离式目录结构，将代码和训练产生的数据分开存储，便于版本控制和代码管理：

```
/media/wsqlab/more/lqj/Metalens-moe/
├── Metalens-moe/              # 项目代码目录（可上传到 GitHub）
│   ├── config.py              # 配置文件
│   ├── train.py               # 训练脚本
│   ├── test.py                # 测试脚本
│   ├── net/                   # 网络定义目录
│   ├── data/                  # 数据处理模块
│   ├── utils/                 # 工具函数
│   └── README.md              # 本文件
│
├── experiment/                # 训练输出目录（外部，不提交到 Git）
│   └── <模型名-时间戳>/       # 每次训练的实验目录
│       ├── checkpoints/       # 模型检查点
│       │   ├── last.ckpt      # 最后一个 epoch 的检查点
│       │   ├── best_psnr-*.ckpt  # 最佳 PSNR 检查点
│       │   └── best_psnr_ssim-*.ckpt  # 最佳综合指标检查点
│       ├── net_snapshot/      # 网络文件快照（用于独立测试）
│       ├── config_snapshot/   # 配置文件快照
│       ├── metrics.csv        # 训练指标记录
│       └── opt.json           # 训练配置 JSON
│
├── test/                      # 测试结果目录（外部，不提交到 Git）
│   └── <实验标识>/            # 每次测试的结果目录
│       ├── metrics.json       # 测试指标（JSON 格式）
│       ├── metrics.txt        # 测试指标（文本格式）
│       └── test.csv           # 测试结果汇总 CSV
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

### 推荐使用流程：

1. **配置环境**：修改 `config.py`，确定模型、数据路径和训练的超参。
   - 设置 `MODEL` 选择要使用的网络
   - 配置 `DATA_FILE_DIR` 指向训练数据目录
   - 配置 `EXPERIMENT_DIR`、`TEST_DIR`、`RESULTS_DIR` 指向外部目录（如需要）
   
2. **开始训练**：运行 `train.sh` 开始训练
   - 训练过程会在 `experiment/<模型名-时间戳>/` 下创建实验目录
   - 自动保存检查点、指标记录和网络快照
   - 训练完成后会自动运行测试（如果 `train.sh` 中配置了）
   - 在`test/` 目录的`test.csv`中会存有该模型的所有详细指标数据
   
3. **评估模型**：运行 `test.sh` 进行单独的模型评估
   - 支持多卡评估（双卡全分辨率 `batch=1`）
   - 单卡评估可直接运行 `python test.py`
   - 测试结果保存在 `test/` 目录
   - 恢复的图像保存在 `results/` 目录（如果 `SAVE_RESULTS=True`）
   
4. **单张图像推理**：使用 `python infer_image.py --ckpt <ckpt> --input <img> --output <out>`
   - 对单张图像进行推理和恢复

## 如何运行新创建的网络

当你创建了一个新的网络文件（例如 `net/MoCE_IR_Spectral.py`）后，需要按以下步骤操作：

### 步骤 1：确认网络文件结构
确保你的网络文件包含：
- 网络模型类定义（例如 `class MoCEIR(nn.Module)`）
- `build_model(opt)` 函数，用于根据配置选项构建模型

### 步骤 2：修改 config.py
在 `config.py` 中修改 `MODEL` 变量，将其设置为你的新网络名称（不包含 `.py` 后缀）：
```python
MODEL = "MoCE_IR_Spectral"  # 你的新网络名称
```

### 步骤 3：（可选）添加模型配置
如果需要自定义模型参数，可以在 `config.py` 中添加模型配置字典：
```python
MoCE_IR_Spectral_CONFIG = {
    "dim": 48,
    "num_blocks": [4, 6, 6, 8],
    "num_dec_blocks": [2, 4, 4],
    "latent_dim": 2,
    "num_exp_blocks": 4,
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


- **train.py**  
  训练脚本。从 `config.py` 读取所有训练配置（模型类型、batch size、学习率、数据路径等），并自动：
  - 动态加载 `net/<MODEL>.py` 中的网络（`MODEL` 在 `config.py` 里设置，如 `"MoCE_IR_S"` / `"MoCE_IR"` / `"ACFormer"` / `"MoCE_IR_Spectral"` 等）。
  - 使用 Hugging Face `accelerate` 进行单机多卡训练（DDP）。
  - 在 `experiment/<模型名-时间戳>/` 下保存（路径由 `config.EXPERIMENT_DIR` 配置）：
    - `checkpoints/`：`last.ckpt` 和最佳指标 ckpt（`best_psnr-*.ckpt`、`best_psnr_ssim-*.ckpt`）。
    - `net_snapshot/`：本次训练使用的单个网络文件快照（独立可用，用于测试时加载）。
    - `config_snapshot/`：本次训练使用的配置文件快照。
    - `metrics.csv`：按 epoch 记录的 PSNR/SSIM/LPIPS 等指标。
    - `opt.json`：训练配置的 JSON 格式备份。

- **config.py**  
  训练配置文件，只对 `train.py` / `infer_image.py` 有效，用来统一管理：
  - 模型选择：`MODEL`（`"MoCE_IR"` / `"MoCE_IR_S"` / `"ACFormer"` / `"MoCE_IR_PhysRouting"` / `"MoCE_IR_Spectral"` 等）。
  - 训练与数据相关参数：数据根目录、训练超参、workers、精度、验证频率等（按需修改）。
  - 可选模型配置：可以为每个模型添加 `{MODEL}_CONFIG` 字典来自定义模型参数。

- **test.py**  
  测试脚本，**完全独立于 `config.py`**，只需在文件末尾的 `argparse.Namespace` 里填好：
  - `ckpt_path`：要测试的 ckpt 绝对路径（支持 Lightning checkpoint）。
  - `benchmarks`：如 `["gopro"]`、`["drmi"]` 等测试集列表。
  - `patch_size`：patch 测试时的 patch 大小（默认 128 或 256）。
  - `full_res_eval`：
    - `True`：全图像评估。
    - `False`：和原始 MoCE-IR 一样，使用中心 patch 评估。
  - `save_results`：是否保存恢复的图像（默认 `False`）。
  
  测试结果保存：
  - 测试指标保存在 `test/` 目录（路径由 `config.TEST_DIR` 或环境变量 `MOCEIR_TEST_RESULT_DIR` 配置）。
  - 恢复的图像保存在 `results/` 目录（路径由 `config.RESULTS_DIR` 配置，当 `save_results=True` 时）。
  
  网络加载顺序：
  1. 优先从 ckpt 同级的 `../net_snapshot/` 加载网络文件（训练时保存的快照）。
  2. 如不存在，则回退到项目内 `net/<MODEL>.py`。

- **test.sh** 
  测试脚本，写好了所有配置了，直接运行即可。

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
  - **模型选择**：`MODEL`（`"MoCE_IR"` / `"MoCE_IR_S"` / `"ACFormer"` / `"MoCE_IR_PhysRouting"` / `"MoCE_IR_Spectral"` 等）。
  - **训练参数**：epochs、batch_size、learning_rate、loss_type 等。
  - **数据路径**：`DATA_FILE_DIR` 指向训练数据目录。
  - **外部目录路径**：
    - `EXPERIMENT_DIR`：训练输出目录（默认 `"../experiment"`）。
    - `TEST_DIR`：测试结果目录（默认 `"../test"`）。
    - `RESULTS_DIR`：恢复图像结果目录（默认 `"../results"`）。
  - **可选模型配置**：可以为每个模型添加 `{MODEL}_CONFIG` 字典来自定义模型参数。




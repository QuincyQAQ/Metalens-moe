## 项目说明（MoCE-IR 多网络版本）

推荐使用流程：

1. 修改 `config.py`，确定模型、数据路径和训练超参。  
2. 运行 `train.sh` 开始训练，等待 `experiment/` 下生成对应实验目录，训练完所有的epoch会生成测试数据在`train`文件夹 。  
3. 可以自己运行 `test.sh` 做评估（双卡全分辨率 `batch=1`），如果只是python test.py 的话只是单卡效果。  
4. 如需对单张图像做推理，使用 `python infer_image.py --ckpt <ckpt> --input <img> --output <out>`。


- **train.py**  
  训练脚本。从 `config.py` 读取所有训练配置（模型类型、batch size、学习率、数据路径等），并自动：
  - 动态加载 `net/<MODEL>.py` 中的网络（`MODEL` 在 `config.py` 里设置，如 `"MoCE_IR_S"` / `"MoCE_IR"` / `"ACFormer"`）。
  - 使用 Hugging Face `accelerate` 进行单机多卡训练（DDP）。
  - 在 `experiment/<模型名-时间戳>/` 下保存：
    - `checkpoints/`：`last.ckpt` 和最佳指标 ckpt。
    - `net_snapshot/`：本次训练使用的单个网络文件快照（独立可用）。
    - `metrics.csv`：按 epoch 记录的 PSNR/SSIM/LPIPS 等指标。

- **config.py**  
  训练配置文件，只对 `train.py` / `infer_image.py` 有效，用来统一管理：
  - 模型选择：`MODEL`（`"MoCE_IR"` / `"MoCE_IR_S"` / `"ACFormer"`）。
  - 训练与数据相关参数：数据根目录、训练超参、workers、精度、验证频率等（按需修改）。

- **test.py**  
  测试脚本，**完全独立于 `config.py`**，只需在文件末尾的 `argparse.Namespace` 里填好：
  - `ckpt_path`：要测试的 ckpt 绝对路径（支持 Lightning checkpoint）。
  - `benchmarks`：如 `["gopro"]`、`["drmi"]` 等，这里不用动。
  - `patch_size`：patch 测试时的 patch 大小（默认 128 或 256），不用动。
  - `full_res_eval`：
    - `True`：全图像评估。
    - `False`：和原始 MoCE-IR 一样，使用中心 patch 评估。
  测试时会优先从 ckpt 同级的 `../net_snapshot/` 加载网络文件；如不存在，则回退到项目内 `net/<MODEL>.py`。

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
  这个后面再完善
  
- **experiment/ 目录**  
  每次调用 `train.py`，都会在 `experiment/` 下自动创建一个子目录，例如：
  - `experiment/MoCE_IR_S-2026_01_29_23_33_45/`
  - 该目录包含：
    - `checkpoints/`：训练过程中的 ckpt（包括最佳 ckpt）。
    - `net_snapshot/`：本次实验使用的网络入口文件拷贝，供独立测试使用。
    - `metrics.csv`：训练/验证/测试的指标记录。




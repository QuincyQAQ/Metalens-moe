from typing import List

import os
import pathlib
import importlib
import importlib.util
import shutil
import csv
import json
import warnings
import sys
import io
import contextlib
import logging
import numpy as np
from copy import deepcopy

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11, IRBenchmarks
from utils.loss_utils import FFTLoss, FocalL1Loss, FocalLoss

# 尝试导入用于计算模型复杂度的库
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

try:
    from utils.model_summary import get_params_flops
    MODEL_SUMMARY_AVAILABLE = True
except ImportError:
    MODEL_SUMMARY_AVAILABLE = False


# 全局关闭 torch.load(weights_only=False) 的冗长安全提示（Lightning / torchmetrics 内部会触发）。
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
)

# 抑制 PyTorch distributed 的 OMP_NUM_THREADS 警告（通过 logging）
logging.getLogger("torch.distributed.run").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

# 设置环境变量来抑制 PyTorch 分布式警告
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")

# 创建自定义的 stderr 过滤器来抑制 PyTorch 分布式警告
# 这些警告通常直接输出到 stderr，无法通过 warnings 模块捕获
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filter_patterns = [
            "process group has NOT been destroyed",
            "ProcessGroupNCCL",
            "OMP_NUM_THREADS",
            "torch.distributed.run",
            "site-packages/torch/distributed/run.py",
            "Setting OMP_NUM_THREADS",
            "WARNING: process group",
            "*****************************************",
            "to avoid your system being overloaded",
            "please further tune the variable",
        ]
        self.buffer = ""  # 用于处理多行警告
    
    def write(self, text):
        # 将文本添加到缓冲区
        self.buffer += text
        # 检查是否包含换行符，如果有则处理完整的行
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            # 保留最后一行（可能不完整）在缓冲区
            self.buffer = lines[-1]
            # 处理完整的行
            for line in lines[:-1]:
                # 检查是否包含需要过滤的警告
                if not any(pattern in line for pattern in self.filter_patterns):
                    self.original_stderr.write(line + "\n")
        # 如果没有换行符，检查当前缓冲区是否包含警告模式
        elif any(pattern in self.buffer for pattern in self.filter_patterns):
            # 如果包含警告模式，清空缓冲区（不输出）
            if "\n" in text or len(self.buffer) > 200:
                self.buffer = ""
    
    def flush(self):
        # 处理缓冲区中剩余的内容
        if self.buffer and not any(pattern in self.buffer for pattern in self.filter_patterns):
            self.original_stderr.write(self.buffer)
            self.buffer = ""
        self.original_stderr.flush()

# 替换 stderr 以过滤警告
sys.stderr = StderrFilter(sys.stderr)


def _dataloader_worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % 2**32
    try:
        import random
        random.seed(int(seed))
    except Exception:
        pass
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(0)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass


def _is_global_zero() -> bool:
    return str(os.environ.get("RANK", "0")) == "0"


def _mixed_precision_from_opt(opt) -> str:
    precision = str(getattr(opt, "precision", "no")).lower()
    if precision in ("16-mixed", "fp16", "16"):
        return "fp16"
    if precision in ("bf16-mixed", "bf16"):
        return "bf16"
    return "no"


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def _extract_net_state_dict(state_dict: dict) -> dict:
    if any(str(k).startswith("net.") for k in state_dict.keys()):
        return {k[len("net."):]: v for k, v in state_dict.items() if str(k).startswith("net.")}
    return state_dict


def _init_metrics_csv(metrics_csv_path: str) -> None:
    if os.path.exists(metrics_csv_path):
        return
    fieldnames = [
        "phase",
        "epoch",
        "Train_Loss_epoch",
        "Balance_epoch",
        "val_psnr",
        "val_ssim",
        "val_lpips",
        "test_psnr",
        "test_ssim",
        "test_lpips",
        "lr",
    ]
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    with open(metrics_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _write_metrics_row(metrics_csv_path: str, row: dict) -> None:
    if not metrics_csv_path:
        return
    file_exists = os.path.exists(metrics_csv_path)
    fieldnames = list(row.keys())
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    with open(metrics_csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _forward_model(net: nn.Module, x: torch.Tensor, de_id: torch.Tensor) -> torch.Tensor:
    if de_id is None:
        return net(x)
    try:
        return net(x, de_id)
    except TypeError:
        return net(x)


def _extract_balance_loss(net: nn.Module, ref_tensor: torch.Tensor) -> torch.Tensor:
    loss = getattr(net, "total_loss", None)
    if loss is None:
        return ref_tensor.new_zeros(1)
    if torch.is_tensor(loss):
        return loss
    return ref_tensor.new_tensor(loss)


def _snapshot_network_file(opt, project_dir: pathlib.Path, log_dir: pathlib.Path) -> None:
    if not _is_global_zero():
        return

    model_name = getattr(opt, "model", None)
    if not model_name:
        return

    net_snapshot_dir = log_dir / "net_snapshot"
    net_snapshot_dir.mkdir(parents=True, exist_ok=True)
    for p in net_snapshot_dir.glob("*.py"):
        try:
            p.unlink()
        except Exception:
            pass

    net_dir = project_dir / "net"
    snapshot_path = net_snapshot_dir / f"{model_name}.py"

    src_path = net_dir / f"{model_name}.py"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Expected network entry file not found: {src_path}. "
            f"Please ensure net/{model_name}.py exists and is self-contained."
        )

    shutil.copy2(src_path, snapshot_path)


def _snapshot_config_file(project_dir: pathlib.Path, log_dir: pathlib.Path) -> None:
    if not _is_global_zero():
        return

    src_path = project_dir / "config.py"
    if not src_path.exists():
        return

    cfg_snapshot_dir = log_dir / "config_snapshot"
    cfg_snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = cfg_snapshot_dir / "config.py"
    shutil.copy2(src_path, snapshot_path)


def _copy_experiment_to_test_and_merge_csv(log_dir: pathlib.Path, project_dir: pathlib.Path) -> None:
    """
    训练完成后，将实验目录复制到test文件夹，并合并test.csv数据
    
    Args:
        log_dir: 实验目录路径（例如：experiment/MoCE_IR_S-$DATASET_LIST_2026_02_14_19_04_46）
        project_dir: 项目根目录
    """
    # 目标test目录
    test_dir = project_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要复制的实验目录（可能是log_dir本身，也可能是其父目录）
    # 如果log_dir下有test/test.csv，使用log_dir；否则检查父目录（用于多数据集训练）
    experiment_root = log_dir
    experiment_test_csv = log_dir / "test" / "test.csv"
    
    # 检查是否是多数据集训练：如果log_dir的父目录下有test/test.csv，说明是多数据集训练
    parent_dir = log_dir.parent
    parent_test_csv = parent_dir / "test" / "test.csv"
    
    # 判断是否是多数据集训练：
    # 1. 父目录下有test/test.csv
    # 2. 或者父目录下有多个以相同前缀开头的子目录（多个数据集实验）
    is_multi_dataset = False
    if parent_test_csv.exists():
        # 检查父目录下是否有多个子目录（可能是多个数据集实验）
        try:
            subdirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name != "test"]
            # 如果父目录下有多个子目录，且log_dir是其中一个，说明是多数据集训练
            if len(subdirs) > 1 and log_dir in subdirs:
                is_multi_dataset = True
        except Exception:
            pass
    
    # 如果是多数据集训练，使用父目录；否则使用log_dir本身
    is_using_parent = False
    if is_multi_dataset:
        experiment_root = parent_dir
        experiment_test_csv = parent_test_csv
        is_using_parent = True
        print(f"[Post-Train] Multi-dataset training detected, using parent directory: {parent_dir}", flush=True)
    elif not experiment_test_csv.exists() and parent_test_csv.exists():
        # 如果当前目录下没有test/test.csv，但父目录有，使用父目录
        experiment_root = parent_dir
        experiment_test_csv = parent_test_csv
        is_using_parent = True
        print(f"[Post-Train] Found test.csv in parent directory: {parent_dir}", flush=True)
    
    # 复制整个实验目录到test文件夹下
    experiment_name = experiment_root.name
    target_experiment_dir = test_dir / experiment_name
    
    print(f"[Post-Train] Copying experiment directory to test folder...", flush=True)
    print(f"[Post-Train] Source: {experiment_root}", flush=True)
    print(f"[Post-Train] Target: {target_experiment_dir}", flush=True)
    
    # 如果目标目录已存在，先删除（避免旧数据干扰）
    if target_experiment_dir.exists():
        print(f"[Post-Train] Removing existing target directory: {target_experiment_dir}", flush=True)
        shutil.rmtree(target_experiment_dir)
    
    # 复制整个实验目录
    try:
        shutil.copytree(experiment_root, target_experiment_dir)
        print(f"[Post-Train] Experiment directory copied successfully", flush=True)
    except Exception as e:
        print(f"[Post-Train] Error copying experiment directory: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 如果是多数据集训练（使用了父目录），删除test目录下所有对应的子目录
    if is_using_parent and log_dir != experiment_root:
        # log_dir是子目录，需要删除test目录下对应的子目录
        subdir_name = log_dir.name
        target_subdir = test_dir / subdir_name
        if target_subdir.exists() and target_subdir.is_dir():
            print(f"[Post-Train] Removing subdirectory from test folder (multi-dataset training): {target_subdir}", flush=True)
            try:
                shutil.rmtree(target_subdir)
                print(f"[Post-Train] Subdirectory removed successfully: {target_subdir}", flush=True)
            except Exception as e:
                print(f"[Post-Train] Warning: Failed to remove subdirectory {target_subdir}: {e}", flush=True)
        
        # 同时检查并删除父目录下其他子目录在test目录中的对应项
        try:
            parent_subdirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name != "test" and d != log_dir]
            for subdir in parent_subdirs:
                target_subdir = test_dir / subdir.name
                if target_subdir.exists() and target_subdir.is_dir():
                    print(f"[Post-Train] Removing other subdirectory from test folder: {target_subdir}", flush=True)
                    try:
                        shutil.rmtree(target_subdir)
                        print(f"[Post-Train] Subdirectory removed successfully: {target_subdir}", flush=True)
                    except Exception as e:
                        print(f"[Post-Train] Warning: Failed to remove subdirectory {target_subdir}: {e}", flush=True)
        except Exception as e:
            print(f"[Post-Train] Warning: Error checking for other subdirectories: {e}", flush=True)
    
    # 检查test.csv是否存在
    if not experiment_test_csv.exists():
        print(f"[Post-Train] Warning: test/test.csv not found in experiment directory: {experiment_test_csv}", flush=True)
        return
    
    # 目标test.csv文件
    target_test_csv = test_dir / "test.csv"
    
    # 读取实验目录下的test.csv数据
    new_rows = []
    try:
        with open(experiment_test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                new_rows.append(row)
    except Exception as e:
        print(f"[Post-Train] Error reading experiment test.csv: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    if not new_rows:
        print(f"[Post-Train] No data found in experiment test.csv", flush=True)
        return
    
    # 读取现有的test.csv数据（如果存在）
    existing_rows = []
    existing_fieldnames = None
    if target_test_csv.exists():
        try:
            with open(target_test_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames
                for row in reader:
                    existing_rows.append(row)
        except Exception as e:
            print(f"[Post-Train] Warning: Error reading existing test.csv: {e}", flush=True)
            existing_rows = []
    
    # 确保fieldnames一致
    if existing_fieldnames and fieldnames and existing_fieldnames != fieldnames:
        print(f"[Post-Train] Warning: Fieldnames mismatch. Existing: {existing_fieldnames}, New: {fieldnames}", flush=True)
        # 使用并集，确保所有字段都被包含
        all_fieldnames = list(dict.fromkeys(list(existing_fieldnames or []) + list(fieldnames or [])))
        fieldnames = all_fieldnames
    
    # 合并数据：去重（基于net_name, dataset, benchmark, ckpt_path, count）
    # 注意：gflops 不包含在去重键中，因为它是动态计算的，可能有计算误差。
    # count 包含在去重键中，因为不同的count表示不同的测试配置（例如：全量测试 vs 部分测试）
    # 同一个checkpoint、同一个dataset、同一个benchmark、同一个count应该只保留一条记录（保留最新的）
    merged_rows = existing_rows.copy()
    existing_keys = set()
    for row in existing_rows:
        key = (row.get("net_name", ""), row.get("dataset", ""), row.get("benchmark", ""), row.get("ckpt_path", ""), str(row.get("count", "")))
        existing_keys.add(key)
    
    added_count = 0
    for row in new_rows:
        key = (row.get("net_name", ""), row.get("dataset", ""), row.get("benchmark", ""), row.get("ckpt_path", ""), str(row.get("count", "")))
        if key not in existing_keys:
            merged_rows.append(row)
            existing_keys.add(key)
            added_count += 1
        else:
            # 如果已存在，用新记录替换旧记录（保留最新的测试结果）
            for i, existing_row in enumerate(merged_rows):
                existing_key = (existing_row.get("net_name", ""), existing_row.get("dataset", ""), existing_row.get("benchmark", ""), existing_row.get("ckpt_path", ""), str(existing_row.get("count", "")))
                if existing_key == key:
                    merged_rows[i] = row  # 用新记录替换旧记录
                    break
    
    # 写入合并后的test.csv
    try:
        with open(target_test_csv, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in merged_rows:
                    # 确保所有字段都存在
                    complete_row = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(complete_row)
        
        print(f"[Post-Train] Merged {added_count} new rows from experiment test.csv to {target_test_csv}", flush=True)
        print(f"[Post-Train] Total rows in test.csv: {len(merged_rows)} (was {len(existing_rows)})", flush=True)
    except Exception as e:
        print(f"[Post-Train] Error writing merged test.csv: {e}", flush=True)
        import traceback
        traceback.print_exc()


def _unpack_restored(restored):
    if isinstance(restored, (list, tuple)) and len(restored) == 2:
        return restored[0]
    return restored


def _normalize_de_id(de_id, *, device: torch.device, opt=None):
    if de_id is None:
        return None

    if torch.is_tensor(de_id):
        if de_id.device != device:
            de_id = de_id.to(device, non_blocking=True)
        if de_id.dtype != torch.long:
            de_id = de_id.long()
        return de_id

    if isinstance(de_id, (list, tuple)) and len(de_id) > 0:
        first = de_id[0]
        if isinstance(first, (int, np.integer)):
            return torch.tensor(list(de_id), device=device, dtype=torch.long)
        if isinstance(first, str) and opt is not None:
            de_type = getattr(opt, "de_type", None)
            if isinstance(de_type, (list, tuple)):
                idxs = []
                for s in de_id:
                    try:
                        idxs.append(int(list(de_type).index(s)))
                    except Exception:
                        idxs.append(0)
                return torch.tensor(idxs, device=device, dtype=torch.long)

    return None


def _build_model(opt) -> nn.Module:
    model_name = getattr(opt, "model", None)
    if not model_name:
        raise ValueError("opt.model is required")
    module = importlib.import_module(f"net.{model_name}")
    build_fn = getattr(module, "build_model", None)
    if build_fn is None:
        raise AttributeError(f"net.{model_name} must define build_model(opt)")
    return build_fn(opt)


def _load_external_focal_loss(project_dir: pathlib.Path):
    focal_path = project_dir / "net" / "focal-loss-pytorch-main" / "focal_loss.py"
    if not focal_path.exists():
        raise FileNotFoundError(f"External focal loss file not found: {focal_path}")

    spec = importlib.util.spec_from_file_location("external_focal_loss", str(focal_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import external focal loss from: {focal_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_ckpt_path(project_dir: pathlib.Path, opt, ckpt_spec: str) -> str:
    ckpt_spec = str(ckpt_spec)
    if ckpt_spec.endswith(".ckpt"):
        if os.path.isabs(ckpt_spec):
            return ckpt_spec
        return str((project_dir / ckpt_spec).resolve())

    if os.path.isabs(ckpt_spec):
        return os.path.join(ckpt_spec, "last.ckpt")
    return os.path.join(str(getattr(opt, "ckpt_dir", "")), ckpt_spec, "last.ckpt")


def _load_weights(model: nn.Module, ckpt_path: str) -> None:
    ckpt = _safe_torch_load(ckpt_path)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _extract_net_state_dict(state_dict)
    
    # 过滤掉不需要显示的参数名称（dec.2.2.layers下的adapter experts相关参数）
    # 这些参数在加载时会被静默忽略，避免打印警告信息
    excluded_prefixes = [
        "dec.2.2.layers.0.adapter.experts.3.0",
        "dec.2.2.layers.1.adapter.experts.0.0",
        "dec.2.2.layers.1.adapter.experts.1.0",
        "dec.2.2.layers.1.adapter.experts.2.0",
    ]
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in excluded_prefixes)
    }
    
    # 创建一个上下文管理器来过滤掉包含这些参数名称的输出
    @contextlib.contextmanager
    def filter_excluded_params():
        """过滤掉包含excluded_prefixes的输出"""
        import re
        old_stderr = sys.stderr
        old_showwarning = warnings.showwarning
        
        # 创建一个自定义的警告显示函数
        def filtered_showwarning(message, category, filename, lineno, file=None, line=None):
            """自定义警告显示函数，过滤掉包含excluded参数名称的警告"""
            msg_str = str(message)
            # 检查是否包含任何excluded前缀
            if any(prefix in msg_str for prefix in excluded_prefixes):
                return  # 忽略这个警告
            # 检查是否匹配模式 dec.2.2.layers.X.adapter.experts.Y.0
            pattern = r'dec\.2\.2\.layers\.([01])\.adapter\.experts\.([0-3])\.0'
            if re.search(pattern, msg_str):
                return  # 忽略这个警告
            # 显示其他警告
            old_showwarning(message, category, filename, lineno, file, line)
        
        try:
            # 设置自定义警告显示函数
            warnings.showwarning = filtered_showwarning
            # 创建一个StringIO对象来捕获stderr
            filtered_stderr = io.StringIO()
            sys.stderr = filtered_stderr
            yield
        finally:
            # 恢复stderr和警告处理器
            sys.stderr = old_stderr
            warnings.showwarning = old_showwarning
            output = filtered_stderr.getvalue()
            # 过滤掉包含excluded参数名称的行
            filtered_lines = []
            for line in output.split('\n'):
                if line.strip():  # 跳过空行
                    # 检查这一行是否包含任何excluded前缀或匹配的模式
                    should_filter = False
                    # 检查是否包含任何excluded前缀
                    if any(prefix in line for prefix in excluded_prefixes):
                        should_filter = True
                    # 也检查是否匹配模式 dec.2.2.layers.X.adapter.experts.Y.0
                    pattern = r'dec\.2\.2\.layers\.([01])\.adapter\.experts\.([0-3])\.0'
                    if re.search(pattern, line):
                        should_filter = True
                    if not should_filter:
                        filtered_lines.append(line)
            # 只打印过滤后的输出
            if filtered_lines:
                print('\n'.join(filtered_lines), file=old_stderr, end='')
    
    # 使用上下文管理器来过滤输出
    with filter_excluded_params():
        model.load_state_dict(filtered_state_dict, strict=False)


def _extract_net_name(ckpt_path, net_path=None):
    """从 checkpoint 路径或网络路径中提取网络名称"""
    # 首先尝试从 checkpoint 路径中提取（格式：.../MoCE_IR_S-2026_02_01_00_28_26/...）
    try:
        ckpt_path_obj = pathlib.Path(ckpt_path)
        # 向上查找包含模型名的目录
        for parent in ckpt_path_obj.parents:
            parent_name = parent.name
            # 检查是否是实验目录格式（模型名-时间戳）
            if "-" in parent_name and len(parent_name.split("-")) >= 2:
                parts = parent_name.split("-")
                # 尝试找到模型名（通常在第一部分或前几部分）
                if len(parts) >= 2:
                    # 检查是否是日期格式（YYYY_MM_DD）
                    if len(parts[-1].split("_")) >= 3:
                        # 提取模型名（去掉时间戳部分）
                        net_name = "-".join(parts[:-1])
                        return net_name
        
        # 如果没找到，尝试从 net_path 中提取
        if net_path:
            net_path_obj = pathlib.Path(net_path)
            net_file = net_path_obj.name
            if net_file.endswith(".py"):
                net_name = net_file[:-3]  # 去掉 .py 后缀
                return net_name
    except Exception as e:
        print(f"[Warn] Failed to extract net_name: {e}")
    
    return "Unknown"


@contextlib.contextmanager
def _suppress_fvcore_output():
    """抑制 fvcore 的输出（Unsupported operator 和未使用的子模块信息）"""
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    
    # 创建自定义的流来过滤输出
    class FilteredStream:
        def __init__(self, original_stream):
            self.original_stream = original_stream
            self.suppress_mode = False  # 是否处于抑制模式（遇到特定消息后）
        
        def write(self, text):
            if not text:
                return
            
            text_lower = text.lower()
            line = text.strip()
            
            # 过滤 "Unsupported operator" 消息（包括 "encountered X time(s)" 格式）
            if 'unsupported operator' in text_lower:
                return
            if 'encountered' in text_lower and 'time(s)' in text_lower:
                return
            
            # 过滤 "The following submodules" 消息
            if 'the following submodules' in text_lower or 'never called during the trace' in text_lower:
                self.suppress_mode = True
                return
            
            # 如果处于抑制模式
            if self.suppress_mode:
                # 检查是否包含子模块名称（dec. 或 enc. 开头，且包含 adapter.experts）
                # 处理可能在同一行用逗号分隔的多个模块名称
                if line:
                    # 检查是否包含 adapter.experts 的子模块
                    if 'adapter.experts' in line and (line.startswith('dec.') or line.startswith('enc.') or 'dec.' in line or 'enc.' in line):
                        return
                    # 如果是不包含 adapter.experts 的子模块名称，可能是其他重要信息
                    if (line.startswith('dec.') or line.startswith('enc.')) and 'adapter.experts' not in line:
                        # 停止抑制，让这一行通过
                        self.suppress_mode = False
                        self.original_stream.write(text)
                        return
                    # 如果是不以 dec. 或 enc. 开头的非空行，可能是新段落，停止抑制
                    if not (line.startswith('dec.') or line.startswith('enc.') or 'dec.' in line or 'enc.' in line):
                        # 检查是否包含明显的段落分隔符或新主题
                        if len(line) > 0 and not any(keyword in text_lower for keyword in ['adapter', 'experts', 'layers']):
                            self.suppress_mode = False
                            self.original_stream.write(text)
                            return
                # 空行继续抑制（可能是列表中的分隔）
                return
            
            # 其他输出正常显示
            self.original_stream.write(text)
        
        def flush(self):
            self.original_stream.flush()
    
    filtered_stderr = FilteredStream(old_stderr)
    filtered_stdout = FilteredStream(old_stdout)
    
    try:
        sys.stderr = filtered_stderr
        sys.stdout = filtered_stdout
        yield
    finally:
        sys.stderr = old_stderr
        sys.stdout = old_stdout


def _calculate_model_complexity(net, input_size=(1, 3, 256, 256)):
    """计算模型的 GFLOPs 和参数量（百万）"""
    try:
        net.eval()
        try:
            device = next(net.parameters()).device
            dtype = next(net.parameters()).dtype
        except StopIteration:
            return None, None
        
        # 计算参数量（百万）
        try:
            num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            num_params_m = num_params / 1e6
        except Exception:
            num_params_m = None
        
        # 计算 GFLOPs
        gflops = None
        if FVCORE_AVAILABLE:
            try:
                with torch.no_grad(), _suppress_fvcore_output():
                    dummy_input = torch.randn(*input_size, device=device, dtype=dtype)
                    flops = FlopCountAnalysis(net, dummy_input)
                    total_flops = flops.total()
                    gflops = total_flops / 1e9
            except Exception:
                pass
        
        if gflops is None and MODEL_SUMMARY_AVAILABLE:
            try:
                with torch.no_grad(), _suppress_fvcore_output():
                    flops, params = get_params_flops(net, input_dim=input_size[1:])
                    gflops = flops
            except Exception:
                pass
        
        return gflops, num_params_m
    except Exception:
        return None, None


def _evaluate_irbenchmarks(net: nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    net.eval()
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction="mean").to(device)

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    with torch.no_grad():
        # 禁用进度条避免刷屏
        for ([clean_name, de_id], degrad_patch, clean_patch) in data_loader:
            degrad_patch = degrad_patch.to(device, non_blocking=True)
            clean_patch = clean_patch.to(device, non_blocking=True)
            de_id = _normalize_de_id(de_id, device=device)

            restored = _forward_model(net, degrad_patch, de_id)
            restored = _unpack_restored(restored)
            restored = torch.clamp(restored, 0.0, 1.0)

            lpips_val = calc_lpips(clean_patch, restored)
            lpips_vals.append(float(lpips_val.detach().cpu()))

            restored_np = restored.detach().cpu().permute(0, 2, 3, 1).numpy()
            clean_np = clean_patch.detach().cpu().permute(0, 2, 3, 1).numpy()

            for i in range(restored_np.shape[0]):
                psnr_vals.append(float(peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1.0)))
                ssim_vals.append(float(structural_similarity(
                    clean_np[i],
                    restored_np[i],
                    data_range=1.0,
                    channel_axis=2,
                    gaussian_weights=True,
                )))

    return {
        "psnr": float(np.mean(psnr_vals)) if len(psnr_vals) else None,
        "ssim": float(np.mean(ssim_vals)) if len(ssim_vals) else None,
        "lpips": float(np.mean(lpips_vals)) if len(lpips_vals) else None,
    }


def main(opt):
    # 不打印完整Options，只保留关键信息
    run_id = os.environ.get("MOCEIRV2_RUN_ID")
    if run_id is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # 基础模型名
        model_name = getattr(opt, "model", "model")
        model_name = str(model_name).replace(os.sep, "_").replace(" ", "_")

        # 从数据路径或 trainset 推出“数据集名字”，插入到中间：
        # e.g. MoCE_IR_S-CVC_8_1_1-2026_02_13_17_00_43
        dataset_name = None
        data_dir = getattr(opt, "data_file_dir", None)
        if data_dir:
            # 取数据根目录名作为数据集名
            dataset_name = os.path.basename(str(data_dir).rstrip(os.sep))

        if not dataset_name:
            # 回退到 TRAINSET 字段，避免出现空字符串
            dataset_name = str(getattr(opt, "trainset", "")).strip() or "data"

        dataset_name = dataset_name.replace(os.sep, "_").replace(" ", "_")

        run_id = f"{model_name}-{dataset_name}-{timestamp}"
        os.environ["MOCEIRV2_RUN_ID"] = run_id

    time_stamp = run_id

    project_dir = pathlib.Path(__file__).resolve().parent
    # 使用配置中的外部 experiment 目录
    experiment_dir = getattr(opt, "experiment_dir", None)
    if experiment_dir:
        base_exp_dir = pathlib.Path(experiment_dir)
    else:
        base_exp_dir = project_dir / "experiment"
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否有父文件夹环境变量（用于多数据集训练）
    # 如果 run_id 包含路径分隔符，说明已经包含了父文件夹路径
    if "/" in time_stamp or "\\" in time_stamp:
        # run_id 已经包含了父文件夹路径，直接使用
        log_dir = base_exp_dir / time_stamp
    else:
        # 检查是否有父文件夹环境变量
        parent_exp_dir = os.environ.get("MOCEIRV2_PARENT_EXP_DIR")
        if parent_exp_dir:
            # 在父文件夹下创建子文件夹
            log_dir = base_exp_dir / parent_exp_dir / time_stamp
        else:
            # 原有逻辑：直接在 base_exp_dir 下创建
            log_dir = base_exp_dir / time_stamp
    
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = str(log_dir / "metrics.csv")
    setattr(opt, "metrics_csv", metrics_csv_path)
    if _is_global_zero():
        _init_metrics_csv(metrics_csv_path)

    _snapshot_network_file(opt, project_dir=project_dir, log_dir=log_dir)
    _snapshot_config_file(project_dir=project_dir, log_dir=log_dir)

    if torch.cuda.is_available() and getattr(opt, "tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if torch.cuda.is_available() and bool(getattr(opt, "benchmark", True)) and not bool(getattr(opt, "deterministic", False)):
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    mixed_precision = _mixed_precision_from_opt(opt)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(getattr(opt, "accum_grad", 1)),
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(0)

    writer = None
    use_wandb = bool(getattr(opt, "wblogger", False))
    wandb_run = None
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=str(log_dir))
        try:
            with open(str(log_dir / "opt.json"), "w") as f:
                json.dump(getattr(opt, "__dict__", {}), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if use_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    name=str(getattr(opt, "model", "model")) + "_" + str(time_stamp),
                    dir=str(log_dir),
                    config=getattr(opt, "__dict__", {}),
                )
            except Exception:
                wandb_run = None

    accelerator.wait_for_everyone()

    model = _build_model(opt)
    if getattr(opt, "fine_tune_from", None):
        ckpt_spec = getattr(opt, "fine_tune_from")
        ckpt_path = _resolve_ckpt_path(project_dir, opt, ckpt_spec)
        if accelerator.is_main_process:
            print(f"[Fine-tune] Resolved fine_tune_from '{ckpt_spec}' to: {ckpt_path}")
        _load_weights(model, ckpt_path)

    # 计算并显示模型复杂度（只在主进程）
    if accelerator.is_main_process:
        model_name = getattr(opt, "model", "model")
        # 将模型移到设备上以便计算复杂度
        temp_device = accelerator.device
        temp_model = accelerator.unwrap_model(model)
        gflops, params_m = _calculate_model_complexity(temp_model, input_size=(1, 3, 256, 256))
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        if gflops is not None:
            print(f"GFLOPs: {gflops:.5f}")
        else:
            print(f"GFLOPs: N/A")
        if params_m is not None:
            print(f"Parameters: {params_m:.5f}M")
        else:
            print(f"Parameters: N/A")
        print(f"Experiment Dir: {log_dir}")
        print(f"{'='*60}\n")

    if getattr(opt, "print_model", False) and accelerator.is_main_process:
        print(model)

    de_aux_loss_weight = float(getattr(opt, "de_aux_loss_weight", 0.0))
    de_aux_enabled = bool(de_aux_loss_weight > 0) and len(getattr(opt, "de_type", [])) >= 2
    de_cls_head = None
    de_aux_loss_fn = None
    if de_aux_enabled:
        freq_dim = None
        if hasattr(model, "freq_embed") and hasattr(getattr(model, "freq_embed"), "mlp"):
            try:
                freq_dim = int(model.freq_embed.mlp[-1].out_features)
            except Exception:
                freq_dim = None
        if freq_dim is None:
            de_aux_enabled = False
        else:
            de_cls_head = nn.Linear(freq_dim, int(len(opt.de_type)))
            use_external = bool(getattr(opt, "de_aux_use_external_focal", False))
            if use_external:
                try:
                    ext = _load_external_focal_loss(project_dir)
                    ext_FocalLoss = getattr(ext, "FocalLoss")
                    de_aux_loss_fn = ext_FocalLoss(
                        gamma=float(getattr(opt, "de_aux_gamma", 2.0)),
                        alpha=getattr(opt, "de_aux_alpha", None),
                        reduction="mean",
                        task_type="multi-class",
                        num_classes=int(len(opt.de_type)),
                    )
                except Exception as e:
                    if accelerator.is_main_process:
                        print(f"[Warn] Failed to load external focal loss, fallback to internal FocalLoss. err={e}")
                    de_aux_loss_fn = FocalLoss(
                        gamma=float(getattr(opt, "de_aux_gamma", 2.0)),
                        alpha=getattr(opt, "de_aux_alpha", None),
                        reduction="mean",
                        num_classes=int(len(opt.de_type)),
                    )
            else:
                de_aux_loss_fn = FocalLoss(
                    gamma=float(getattr(opt, "de_aux_gamma", 2.0)),
                    alpha=getattr(opt, "de_aux_alpha", None),
                    reduction="mean",
                    num_classes=int(len(opt.de_type)),
                )

    params = list(model.parameters())
    if de_aux_enabled and de_cls_head is not None:
        params = params + list(de_cls_head.parameters())
    optimizer = optim.AdamW(params, lr=float(getattr(opt, "lr", 2e-4)))
    warmup_epochs = 1 if getattr(opt, "fine_tune_from", None) else 15
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=int(warmup_epochs),
        max_epochs=int(getattr(opt, "epochs", 150)),
    )

    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_", maxsplit=1)
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)

    trainloader_kwargs = dict(
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        worker_init_fn=_dataloader_worker_init_fn,
    )
    if opt.num_workers and opt.num_workers > 0:
        trainloader_kwargs.update(
            persistent_workers=bool(getattr(opt, "persistent_workers", True)),
            prefetch_factor=int(getattr(opt, "prefetch_factor", 2)),
        )
    trainloader = DataLoader(trainset, **trainloader_kwargs)

    val_loader = None
    if "CDD11" not in opt.trainset:
        val_opt = deepcopy(opt)
        val_opt.benchmarks = ["gopro"]
        # 验证集显式使用 val split，避免使用 test 数据做验证
        setattr(val_opt, "split", "val")
        # 设置全分辨率评估
        setattr(val_opt, "full_res_eval", True)
        valset = IRBenchmarks(val_opt)
        val_loader = DataLoader(
            valset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    if de_aux_enabled and de_cls_head is not None:
        model, de_cls_head, optimizer, trainloader = accelerator.prepare(model, de_cls_head, optimizer, trainloader)
    else:
        model, optimizer, trainloader = accelerator.prepare(model, optimizer, trainloader)

    checkpoint_path = log_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    if getattr(opt, "resume_from", None):
        resume_spec = str(getattr(opt, "resume_from"))
        resume_state_dir = None
        if resume_spec.endswith(".ckpt"):
            resume_ckpt = resume_spec
            if not os.path.isabs(resume_ckpt):
                resume_ckpt = str((project_dir / resume_ckpt).resolve())
            if accelerator.is_main_process:
                print(f"[Resume] Loading weights from: {resume_ckpt}")
            _load_weights(accelerator.unwrap_model(model), resume_ckpt)
        else:
            cand = base_exp_dir / resume_spec / "checkpoints" / "accelerate_last"
            if cand.exists() and cand.is_dir():
                resume_state_dir = cand
            if resume_state_dir is not None:
                if accelerator.is_main_process:
                    print(f"[Resume] Loading accelerator state from: {resume_state_dir}")
                accelerator.load_state(str(resume_state_dir))
                state_file = resume_state_dir / "train_state.json"
                if state_file.exists():
                    try:
                        with open(str(state_file), "r") as f:
                            st = json.load(f)
                        start_epoch = int(st.get("epoch", 0)) + 1
                        global_step = int(st.get("global_step", 0))
                    except Exception:
                        pass

    loss_fn = nn.L1Loss()
    aux_fn = None
    loss_type = str(getattr(opt, "loss_type", "L1")).lower()
    if loss_type == "fft":
        aux_fn = FFTLoss(loss_weight=float(getattr(opt, "fft_loss_weight", 1.0)))
    elif loss_type in ("focal_l1", "focall1", "focal"):
        loss_fn = FocalL1Loss(
            gamma=float(getattr(opt, "focal_gamma", getattr(opt, "FOCAL_GAMMA", 2.0))),
            epsilon=float(getattr(opt, "focal_epsilon", getattr(opt, "FOCAL_EPSILON", 1e-6))),
            alpha=float(getattr(opt, "focal_alpha", getattr(opt, "FOCAL_ALPHA", 0.1))),
        )

    best_val_psnr = None
    best_joint_psnr = None
    best_joint_ssim = None

    max_epochs = int(getattr(opt, "epochs", 1))
    save_every_n_epochs = 5
    val_every_n_epoch = int(getattr(opt, "check_val_every_n_epoch", 15))
    log_every_n_steps = int(getattr(opt, "log_every_n_steps", 50))

    for epoch in range(start_epoch, max_epochs):
        model.train()
        raw_model = accelerator.unwrap_model(model)
        if hasattr(trainloader, "sampler") and hasattr(trainloader.sampler, "set_epoch"):
            try:
                trainloader.sampler.set_epoch(epoch)
            except Exception:
                pass

        loss_sum = torch.zeros((), device=accelerator.device)
        balance_sum = torch.zeros((), device=accelerator.device)
        de_aux_sum = torch.zeros((), device=accelerator.device) if de_aux_enabled else None
        step_count = torch.zeros((), device=accelerator.device)

        # 获取数据集名称用于显示在进度条中
        dataset_name = None
        data_dir = getattr(opt, "data_file_dir", None)
        if data_dir:
            dataset_name = os.path.basename(str(data_dir).rstrip(os.sep))
        if not dataset_name:
            dataset_name = str(getattr(opt, "trainset", "")).strip() or "data"
        dataset_name = dataset_name.replace(os.sep, "_").replace(" ", "_")
        
        # 启用训练进度条，只在主进程显示，完成后自动清除
        pbar = tqdm(
            trainloader,
            disable=not accelerator.is_main_process,
            desc=f"[Train] Epoch {int(epoch) + 1}/{int(max_epochs)} [{dataset_name}]",
            leave=False,  # epoch结束后自动清除进度条
        )
        for batch in pbar:
            ([clean_name, de_id], degrad_patch, clean_patch) = batch

            degrad_patch = degrad_patch.to(accelerator.device, non_blocking=True)
            clean_patch = clean_patch.to(accelerator.device, non_blocking=True)
            de_id = _normalize_de_id(de_id, device=accelerator.device, opt=opt)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    restored = _forward_model(model, degrad_patch, de_id)
                    restored = _unpack_restored(restored)

                    balance_loss = _extract_balance_loss(raw_model, restored)
                    loss = loss_fn(restored, clean_patch)
                    if aux_fn is not None:
                        loss = loss + aux_fn(restored, clean_patch)

                    de_aux_loss = None
                    if de_aux_enabled and de_cls_head is not None and de_aux_loss_fn is not None and de_id is not None:
                        freq_emb = getattr(raw_model, "last_freq_emb", None)
                        if freq_emb is not None:
                            de_targets = de_id.view(-1)
                            de_logits = de_cls_head(freq_emb)
                            de_aux_loss = de_aux_loss_fn(de_logits, de_targets)
                            loss = loss + de_aux_loss_weight * de_aux_loss

                    if hasattr(raw_model, "total_loss"):
                        loss = loss + float(getattr(opt, "balance_loss_weight", 0.0)) * balance_loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_det = loss.detach()
            balance_det = balance_loss.detach()
            loss_sum = loss_sum + loss_det
            balance_sum = balance_sum + balance_det
            if de_aux_enabled and de_aux_sum is not None:
                if de_aux_loss is None:
                    de_aux_sum = de_aux_sum + loss_det.new_zeros(())
                else:
                    de_aux_sum = de_aux_sum + de_aux_loss.detach()
            step_count = step_count + 1.0
            global_step += 1

            if accelerator.is_main_process:
                # 更新进度条显示当前loss
                current_loss = float(loss_det.cpu().item())
                current_balance = float(balance_det.cpu().item())
                pbar.set_postfix({
                    'loss': f'{current_loss:.5f}',
                    'balance': f'{current_balance:.5f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # 只记录到tensorboard/wandb，不打印到控制台
                if writer is not None and (global_step % log_every_n_steps == 0):
                    writer.add_scalar("Train_Loss", float(loss_det), global_step)
                    writer.add_scalar("Balance", float(balance_det), global_step)
                    if de_aux_enabled and de_aux_loss is not None:
                        writer.add_scalar("DeAux", float(de_aux_loss.detach()), global_step)
                    writer.add_scalar("LR", float(optimizer.param_groups[0]["lr"]), global_step)
                if wandb_run is not None and (global_step % log_every_n_steps == 0):
                    try:
                        wandb_run.log({
                            "Train_Loss": float(loss_det),
                            "Balance": float(balance_det),
                            "DeAux": float(de_aux_loss.detach()) if (de_aux_enabled and de_aux_loss is not None) else None,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "step": int(global_step),
                        })
                    except Exception:
                        pass

        scheduler.step()

        loss_epoch = (accelerator.reduce(loss_sum, reduction="sum") / accelerator.reduce(step_count, reduction="sum")).detach().float().cpu().item()
        balance_epoch = (accelerator.reduce(balance_sum, reduction="sum") / accelerator.reduce(step_count, reduction="sum")).detach().float().cpu().item()
        de_aux_epoch = None
        if de_aux_enabled and de_aux_sum is not None:
            de_aux_epoch = (accelerator.reduce(de_aux_sum, reduction="sum") / accelerator.reduce(step_count, reduction="sum")).detach().float().cpu().item()

        # 每个epoch只显示一条训练日志
        if accelerator.is_main_process:
            print(f"[Train] Epoch {int(epoch) + 1}/{int(max_epochs)} | Loss: {loss_epoch:.5f} | Balance: {balance_epoch:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if accelerator.is_main_process:
            if writer is not None:
                writer.add_scalar("Train_Loss_epoch", float(loss_epoch), epoch)
                writer.add_scalar("Balance_epoch", float(balance_epoch), epoch)
                if de_aux_epoch is not None:
                    writer.add_scalar("DeAux_epoch", float(de_aux_epoch), epoch)
                writer.add_scalar("LR_epoch", float(optimizer.param_groups[0]["lr"]), epoch)
            _write_metrics_row(metrics_csv_path, {
                "phase": "train",
                "epoch": int(epoch),
                "Train_Loss_epoch": f"{float(loss_epoch):.5f}",
                "Balance_epoch": f"{float(balance_epoch):.5f}",
                "val_psnr": None,
                "val_ssim": None,
                "val_lpips": None,
                "test_psnr": None,
                "test_ssim": None,
                "test_lpips": None,
                "lr": f"{float(optimizer.param_groups[0]['lr']):.5f}",
            })

        do_val = (val_loader is not None) and ((epoch + 1) % val_every_n_epoch == 0 or (epoch + 1) == max_epochs)
        if do_val:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                eval_net = accelerator.unwrap_model(model)
                eval_net.eval()
                val_metrics = _evaluate_irbenchmarks(
                    eval_net,
                    val_loader,
                    device=accelerator.device,
                )
                if writer is not None:
                    if val_metrics.get("psnr") is not None:
                        writer.add_scalar("val_psnr", float(val_metrics["psnr"]), epoch)
                    if val_metrics.get("ssim") is not None:
                        writer.add_scalar("val_ssim", float(val_metrics["ssim"]), epoch)
                    if val_metrics.get("lpips") is not None:
                        writer.add_scalar("val_lpips", float(val_metrics["lpips"]), epoch)
                if wandb_run is not None:
                    try:
                        wandb_run.log({
                            "val_psnr": val_metrics.get("psnr"),
                            "val_ssim": val_metrics.get("ssim"),
                            "val_lpips": val_metrics.get("lpips"),
                            "epoch": int(epoch),
                        })
                    except Exception:
                        pass
                _write_metrics_row(metrics_csv_path, {
                    "phase": "val",
                    "epoch": int(epoch),
                    "Train_Loss_epoch": f"{float(loss_epoch):.5f}",
                    "Balance_epoch": f"{float(balance_epoch):.5f}",
                    "val_psnr": f"{val_metrics.get('psnr'):.5f}" if val_metrics.get("psnr") is not None else None,
                    "val_ssim": f"{val_metrics.get('ssim'):.5f}" if val_metrics.get("ssim") is not None else None,
                    "val_lpips": f"{val_metrics.get('lpips'):.5f}" if val_metrics.get("lpips") is not None else None,
                    "test_psnr": None,
                    "test_ssim": None,
                    "test_lpips": None,
                    "lr": f"{float(optimizer.param_groups[0]['lr']):.5f}",
                })

                # 打印val结果，不刷新（使用end=''和flush=True）
                psnr_val = val_metrics.get("psnr")
                ssim_val = val_metrics.get("ssim")
                lpips_val = val_metrics.get("lpips")
                print(f"[Val] Epoch {int(epoch) + 1}/{int(max_epochs)} | PSNR: {psnr_val:.5f} | SSIM: {ssim_val:.5f} | LPIPS: {lpips_val:.5f}", flush=True)

                if psnr_val is not None:
                    if best_val_psnr is None or float(psnr_val) > float(best_val_psnr):
                        best_val_psnr = float(psnr_val)
                        ckpt_name = f"best_psnr-epoch={int(epoch)}-psnr={float(psnr_val):.5f}-ssim={float(ssim_val) if ssim_val is not None else 0.0:.5f}.ckpt"
                        torch.save({"state_dict": eval_net.state_dict()}, str(checkpoint_path / ckpt_name))

                if psnr_val is not None and ssim_val is not None:
                    if best_joint_psnr is None or best_joint_ssim is None:
                        improved_joint = True
                    else:
                        improved_joint = (float(psnr_val) >= float(best_joint_psnr)) and (float(ssim_val) >= float(best_joint_ssim))
                    if improved_joint:
                        best_joint_psnr = float(psnr_val)
                        best_joint_ssim = float(ssim_val)
                        ckpt_name = f"best_psnr_ssim-epoch={int(epoch)}-psnr={float(psnr_val):.5f}-ssim={float(ssim_val):.5f}.ckpt"
                        torch.save({"state_dict": eval_net.state_dict()}, str(checkpoint_path / ckpt_name))

            accelerator.wait_for_everyone()

        save_last = ((epoch + 1) % save_every_n_epochs == 0) or ((epoch + 1) == max_epochs)
        if save_last:
            accelerator.wait_for_everyone()
            accelerator.save_state(str(checkpoint_path / "accelerate_last"))
            if accelerator.is_main_process:
                eval_net = accelerator.unwrap_model(model)
                torch.save({"state_dict": eval_net.state_dict()}, str(checkpoint_path / "last.ckpt"))
                try:
                    with open(str(checkpoint_path / "accelerate_last" / "train_state.json"), "w") as f:
                        json.dump({"epoch": int(epoch), "global_step": int(global_step)}, f)
                except Exception:
                    pass
            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()

    if "CDD11" not in opt.trainset:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            test_opt = deepcopy(opt)
            test_opt.benchmarks = ["gopro"]
            # 测试集显式使用 test split
            setattr(test_opt, "split", "test")
            # 设置全分辨率评估
            setattr(test_opt, "full_res_eval", True)
            
            # 关键修复：直接使用 opt.data_file_dir，因为它已经指向当前训练的数据集
            # 当使用 MOCEIR_TRAIN_DATA_FILE_DIR 时，opt.data_file_dir 已经设置为当前数据集
            # 这是最可靠的方法，确保测试使用与训练相同的数据集
            current_dataset_name = None
            data_dir = getattr(opt, "data_file_dir", None)
            # 也检查环境变量，确保使用正确的数据集
            env_data_dir = os.environ.get("MOCEIR_TRAIN_DATA_FILE_DIR", None)
            if env_data_dir:
                # 优先使用环境变量指定的数据集（这是训练时实际使用的数据集）
                data_dir = env_data_dir
                print(f"[Test] Using MOCEIR_TRAIN_DATA_FILE_DIR: {data_dir}", flush=True)
            if data_dir:
                # 直接使用训练时使用的 data_file_dir
                test_opt.data_file_dir = str(data_dir)
                current_dataset_name = os.path.basename(str(data_dir).rstrip(os.sep))
                print(f"[Test] Using dataset: {current_dataset_name} (data_file_dir: {test_opt.data_file_dir})", flush=True)
            else:
                # 回退逻辑：尝试从 log_dir 名称中推断
                log_dir_name = log_dir.name
                if "-" in log_dir_name:
                    parts = log_dir_name.split("-")
                    # 查找包含数据集名称的部分
                    for i, part in enumerate(parts):
                        if "mini" in part.lower() or any(ds in part for ds in ["open_dataset", "CVC", "Kvasir"]):
                            if i > 0:
                                potential_name = "-".join(parts[1:i+1])
                                data_file_dirs = getattr(opt, "data_file_dirs", [])
                                if data_file_dirs:
                                    for data_dir_candidate in data_file_dirs:
                                        dir_basename = os.path.basename(str(data_dir_candidate).rstrip(os.sep))
                                        if potential_name == dir_basename or dir_basename in potential_name:
                                            current_dataset_name = dir_basename
                                            test_opt.data_file_dir = str(data_dir_candidate)
                                            break
                                if current_dataset_name:
                                    break
                
                # 如果还是没找到，使用 opt.data_file_dirs 的第一个
                if not current_dataset_name:
                    data_file_dirs = getattr(opt, "data_file_dirs", [])
                    if data_file_dirs:
                        test_opt.data_file_dir = str(data_file_dirs[0])
                        current_dataset_name = os.path.basename(str(data_file_dirs[0]).rstrip(os.sep))
                    else:
                        # 最后回退：使用 trainset
                        dataset_name = str(getattr(opt, "trainset", "")).strip() or "data"
                        current_dataset_name = dataset_name.replace(os.sep, "_").replace(" ", "_")
                        print(f"[Test] Warning: Could not determine dataset, using trainset: {current_dataset_name}", flush=True)
                
                if current_dataset_name:
                    print(f"[Test] Using dataset: {current_dataset_name} (data_file_dir: {test_opt.data_file_dir})", flush=True)
            
            testset = IRBenchmarks(test_opt)
            test_loader = DataLoader(
                testset,
                batch_size=1,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

            eval_net = accelerator.unwrap_model(model)
            eval_net.eval()
            test_metrics = _evaluate_irbenchmarks(
                eval_net,
                test_loader,
                device=accelerator.device,
            )
            if writer is not None:
                if test_metrics.get("psnr") is not None:
                    writer.add_scalar("test_psnr", float(test_metrics["psnr"]), max_epochs - 1)
                if test_metrics.get("ssim") is not None:
                    writer.add_scalar("test_ssim", float(test_metrics["ssim"]), max_epochs - 1)
                if test_metrics.get("lpips") is not None:
                    writer.add_scalar("test_lpips", float(test_metrics["lpips"]), max_epochs - 1)
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        "test_psnr": test_metrics.get("psnr"),
                        "test_ssim": test_metrics.get("ssim"),
                        "test_lpips": test_metrics.get("lpips"),
                    })
                except Exception:
                    pass
            _write_metrics_row(metrics_csv_path, {
                "phase": "test",
                "epoch": int(max_epochs - 1),
                "Train_Loss_epoch": None,
                "Balance_epoch": None,
                "val_psnr": None,
                "val_ssim": None,
                "val_lpips": None,
                "test_psnr": f"{test_metrics.get('psnr'):.5f}" if test_metrics.get("psnr") is not None else None,
                "test_ssim": f"{test_metrics.get('ssim'):.5f}" if test_metrics.get("ssim") is not None else None,
                "test_lpips": f"{test_metrics.get('lpips'):.5f}" if test_metrics.get("lpips") is not None else None,
                "lr": f"{float(optimizer.param_groups[0]['lr']):.5f}",
            })
            
            # 打印test结果，不刷新
            psnr_test = test_metrics.get("psnr")
            ssim_test = test_metrics.get("ssim")
            lpips_test = test_metrics.get("lpips")
            print(f"[Test] Final | PSNR: {psnr_test:.5f} | SSIM: {ssim_test:.5f} | LPIPS: {lpips_test:.5f}", flush=True)
            
            # 保存结果到 test 目录的 test.csv
            try:
                # 找到最佳 checkpoint
                best_ckpt = None
                best_ckpt_path = None
                if checkpoint_path.exists():
                    # 优先选择 best_psnr_ssim
                    best_psnr_ssim_ckpts = list(checkpoint_path.glob("best_psnr_ssim*.ckpt"))
                    if best_psnr_ssim_ckpts:
                        best_ckpt = sorted(best_psnr_ssim_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    else:
                        # 如果没有 best_psnr_ssim，选择 best_psnr
                        best_psnr_ckpts = list(checkpoint_path.glob("best_psnr*.ckpt"))
                        if best_psnr_ckpts:
                            best_ckpt = sorted(best_psnr_ckpts, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    if best_ckpt:
                        best_ckpt_path = best_ckpt
                
                if best_ckpt_path is None:
                    # 如果没有找到最佳 checkpoint，使用 last.ckpt
                    last_ckpt = checkpoint_path / "last.ckpt"
                    if last_ckpt.exists():
                        best_ckpt_path = last_ckpt
                
                if best_ckpt_path and best_ckpt_path.exists():
                    # 获取 test_dir 配置
                    result_root = os.environ.get("MOCEIR_TEST_RESULT_DIR", None)
                    if result_root is None:
                        result_root = getattr(opt, "test_dir", "test")
                    result_root_path = pathlib.Path(str(result_root)).expanduser()
                    if not result_root_path.is_absolute():
                        project_dir = pathlib.Path(__file__).resolve().parent
                        result_root_path = (project_dir / result_root_path).resolve()
                    result_root_path.mkdir(parents=True, exist_ok=True)
                    
                    # 获取网络名称和数据集名称
                    net_name = getattr(opt, "model", "model")
                    net_name = str(net_name).replace(os.sep, "_").replace(" ", "_")
                    
                    # 关键修复：使用测试时实际使用的数据集名称（current_dataset_name）
                    # 确保CSV中记录的数据集名称与测试时使用的数据集一致
                    # 优先使用 current_dataset_name（这是从训练时使用的 data_file_dir 提取的）
                    dataset_name = current_dataset_name if current_dataset_name else None
                    if not dataset_name:
                        # 回退逻辑1：从 test_opt.data_file_dir 获取（测试时实际使用的数据集）
                        test_data_dir = getattr(test_opt, "data_file_dir", None)
                        if test_data_dir:
                            dataset_name = os.path.basename(str(test_data_dir).rstrip(os.sep))
                            print(f"[Test] Warning: Using dataset name from test_opt.data_file_dir: {dataset_name}", flush=True)
                    if not dataset_name:
                        # 回退逻辑2：从环境变量获取（训练时实际使用的数据集）
                        env_data_dir = os.environ.get("MOCEIR_TRAIN_DATA_FILE_DIR", None)
                        if env_data_dir:
                            dataset_name = os.path.basename(str(env_data_dir).rstrip(os.sep))
                            print(f"[Test] Warning: Using dataset name from MOCEIR_TRAIN_DATA_FILE_DIR: {dataset_name}", flush=True)
                    if not dataset_name:
                        # 回退逻辑3：从 opt.data_file_dir 获取
                        data_dir = getattr(opt, "data_file_dir", None)
                        if data_dir:
                            dataset_name = os.path.basename(str(data_dir).rstrip(os.sep))
                            print(f"[Test] Warning: Using dataset name from opt.data_file_dir: {dataset_name}", flush=True)
                    if not dataset_name:
                        # 最后回退：使用 trainset
                        dataset_name = str(getattr(opt, "trainset", "")).strip() or "data"
                        print(f"[Test] Warning: Using dataset name from trainset: {dataset_name}", flush=True)
                    dataset_name = dataset_name.replace(os.sep, "_").replace(" ", "_")
                    print(f"[Test] Final dataset name for CSV: {dataset_name}", flush=True)
                    
                    # 获取网络文件路径
                    net_path = None
                    try:
                        net_snapshot_dir = log_dir / "net_snapshot"
                        if net_snapshot_dir.exists():
                            net_files = list(net_snapshot_dir.glob("*.py"))
                            if net_files:
                                net_path = str(net_files[0].resolve())
                    except Exception:
                        pass
                    
                    # 计算模型复杂度（如果还没有计算）
                    gflops = None
                    parameters = None
                    try:
                        gflops, parameters = _calculate_model_complexity(eval_net, input_size=(1, 3, 256, 256))
                    except Exception:
                        pass
                    
                    # 保存到 test.csv（同时保存到项目目录和实验目录）
                    csv_path = result_root_path / "test.csv"
                    csv_exists = csv_path.exists()
                    fieldnames = [
                        "net_name",
                        "dataset",
                        "gflops",
                        "parameters",
                        "benchmark",
                        "psnr",
                        "ssim",
                        "lpips",
                        "count",
                        "ckpt_path",
                        "net_path",
                    ]
                    
                    # 准备要写入的数据
                    row_data = {
                        "net_name": net_name,
                        "dataset": dataset_name,
                        "gflops": f"{gflops:.5f}" if gflops is not None else "",
                        "parameters": f"{parameters:.5f}" if parameters is not None else "",
                        "benchmark": "gopro",
                        "psnr": f"{psnr_test:.5f}" if psnr_test is not None else "",
                        "ssim": f"{ssim_test:.5f}" if ssim_test is not None else "",
                        "lpips": f"{lpips_test:.5f}" if lpips_test is not None else "",
                        "count": test_metrics.get("count", ""),
                        "ckpt_path": str(best_ckpt_path.resolve()),
                        "net_path": net_path if net_path else "",
                    }
                    
                    # 保存到项目目录下的test.csv（默认位置）
                    with open(str(csv_path), "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not csv_exists:
                            writer.writeheader()
                        writer.writerow(row_data)
                    print(f"[Test] Results saved to: {csv_path}", flush=True)
                    
                    # 同时保存到实验目录下的test/test.csv（用于后续自动合并）
                    try:
                        # 确定实验目录的test文件夹位置
                        # 如果是多数据集训练，test.csv应该在父目录的test文件夹下
                        exp_test_dir = log_dir / "test"
                        parent_exp_dir = os.environ.get("MOCEIRV2_PARENT_EXP_DIR")
                        if parent_exp_dir:
                            # 多数据集训练：在父目录下创建test目录
                            # 从log_dir推断父目录路径
                            # log_dir格式可能是: experiment/parent_exp_dir/sub_exp_dir
                            parent_exp_path = log_dir.parent
                            # 检查父目录是否存在且包含多个子实验（多数据集训练的特征）
                            if parent_exp_path.exists():
                                # 检查父目录下是否有多个子目录（可能是多个数据集实验）
                                # 如果父目录下有多个以net_name开头的子目录，说明是多数据集训练
                                subdirs = [d for d in parent_exp_path.iterdir() if d.is_dir() and d.name.startswith(net_name)]
                                if len(subdirs) > 1:
                                    exp_test_dir = parent_exp_path / "test"
                        exp_test_dir.mkdir(parents=True, exist_ok=True)
                        exp_csv_path = exp_test_dir / "test.csv"
                        exp_csv_exists = exp_csv_path.exists()
                        
                        with open(str(exp_csv_path), "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            if not exp_csv_exists:
                                writer.writeheader()
                            writer.writerow(row_data)
                        print(f"[Test] Results also saved to experiment test.csv: {exp_csv_path}", flush=True)
                    except Exception as e:
                        print(f"[Test] Warning: Failed to save to experiment test.csv: {e}", flush=True)
            except Exception as e:
                print(f"[Test] Warning: Failed to save results to test.csv: {e}", flush=True)
                import traceback
                traceback.print_exc()
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    
    # 训练完成后，自动复制实验目录到test文件夹并合并test.csv
    if accelerator.is_main_process:
        try:
            # 使用父目录作为项目根目录，以便将test文件夹放在 /home/cxhlab/lqj/Metalens-moe/test
            # 而不是 /home/cxhlab/lqj/Metalens-moe/Metalens-moe/test
            test_project_dir = project_dir.parent
            _copy_experiment_to_test_and_merge_csv(log_dir, test_project_dir)
        except Exception as e:
            print(f"[Post-Train] Warning: Failed to copy experiment and merge test.csv: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)




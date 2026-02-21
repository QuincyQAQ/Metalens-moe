import os
import pathlib
import argparse
import importlib
import importlib.util
import glob
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import csv
import sys
import traceback
import types
import logging
import contextlib

# 抑制常见警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# 抑制 PyTorch 分布式训练相关的警告
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.run").setLevel(logging.ERROR)
# 抑制 NCCL 相关的警告
os.environ["NCCL_DEBUG"] = "WARN"  # 只显示 WARN 及以上级别，不显示 INFO
# 抑制 torch.distributed.run 的 OMP_NUM_THREADS 警告
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# 抑制 skimage 的警告
warnings.filterwarnings("ignore", message=".*skimage.*")
warnings.filterwarnings("ignore", message=".*img_as_ubyte.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
from PIL import Image

# 进一步抑制 PyTorch 分布式警告（在导入后设置）
# 抑制 ProcessGroupNCCL 的警告
if hasattr(torch.distributed, "ProcessGroupNCCL"):
    # 通过设置环境变量抑制 NCCL 警告
    os.environ["NCCL_DEBUG"] = "ERROR"  # 只显示 ERROR 级别
# 抑制 torch.distributed.run 的警告
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

from utils.test_utils import save_img, load_img
from utils.image_utils import crop_img
from data.dataset_utils import IRBenchmarks, CDD11
import tempfile
import shutil

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
        # 确保模型在 eval 模式
        net.eval()
        
        # 获取设备和数据类型
        try:
            device = next(net.parameters()).device
            dtype = next(net.parameters()).dtype
        except StopIteration:
            # 如果没有参数，返回 None
            return None, None
        
        # 计算参数量（百万）
        try:
            num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            num_params_m = num_params / 1e6
        except Exception as e:
            print(f"[Warn] Failed to calculate parameters: {e}")
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
            except Exception as e:
                print(f"[Warn] Failed to calculate FLOPs with fvcore: {e}")
        
        if gflops is None and MODEL_SUMMARY_AVAILABLE:
            try:
                with torch.no_grad():
                    flops, params = get_params_flops(net, input_dim=input_size[1:])
                    gflops = flops
            except Exception as e:
                print(f"[Warn] Failed to calculate FLOPs with model_summary: {e}")
        
        return gflops, num_params_m
    except Exception as e:
        print(f"[Warn] Failed to calculate model complexity: {e}")
        traceback.print_exc()
        return None, None


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



def _resolve_test_mixed_precision(opt) -> str:
    val = getattr(opt, "precision", None)
    if val is None or str(val).strip() == "":
        val = os.environ.get("MOCEIR_TEST_PRECISION", "no")
    val = str(val).strip().lower()
    if val in ("fp16", "float16", "16", "half"):
        return "fp16"
    if val in ("bf16", "bfloat16"):
        return "bf16"
    return "no"


def _get_test_amp_autocast_kwargs(opt, device: torch.device) -> dict:
    mp = _resolve_test_mixed_precision(opt)
    if device.type != "cuda" or mp == "no":
        return {"enabled": False}
    if mp == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    if mp == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    return {"enabled": False}


def _resolve_ffn_chunk(opt) -> int:
    val = getattr(opt, "ffn_chunk", None)
    if val is None or str(val).strip() == "":
        val = os.environ.get("MOCEIR_FFN_CHUNK", "0")
    try:
        return int(val)
    except Exception:
        return 0


def _apply_ffn_chunking(net: nn.Module, *, chunk_size: int) -> int:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        return 0

    patched = 0
    for m in net.modules():
        if m.__class__.__name__ != "FeedForward":
            continue
        if not hasattr(m, "project_in") or not hasattr(m, "dwconv") or not hasattr(m, "project_out"):
            continue

        def _chunked_forward(self, x):
            b, c, h, w = x.shape
            hidden2 = int(self.project_in.out_channels)
            if hidden2 % 2 != 0:
                y = self.project_in(x)
                y1, y2 = self.dwconv(y).chunk(2, dim=1)
                y = F.gelu(y1) * y2
                return self.project_out(y)

            hidden = hidden2 // 2
            cs = int(chunk_size)
            if cs <= 0 or cs >= hidden:
                y = self.project_in(x)
                y1, y2 = self.dwconv(y).chunk(2, dim=1)
                y = F.gelu(y1) * y2
                return self.project_out(y)

            out_ch = int(self.project_out.out_channels)
            out = torch.zeros((b, out_ch, h, w), device=x.device, dtype=x.dtype)
            w_in = self.project_in.weight
            b_in = self.project_in.bias
            w_dw = self.dwconv.weight
            b_dw = self.dwconv.bias
            w_out = self.project_out.weight
            b_out = self.project_out.bias

            for s in range(0, hidden, cs):
                e = min(s + cs, hidden)
                in_w = torch.cat([w_in[s:e], w_in[hidden + s:hidden + e]], dim=0)
                in_b = None
                if b_in is not None:
                    in_b = torch.cat([b_in[s:e], b_in[hidden + s:hidden + e]], dim=0)

                y = F.conv2d(x, in_w, bias=in_b, stride=1, padding=0)

                dw_w = w_dw[s:e]
                dw2_w = w_dw[hidden + s:hidden + e]
                dw_w = torch.cat([dw_w, dw2_w], dim=0)
                if b_dw is not None:
                    dw_b1 = b_dw[s:e]
                    dw_b2 = b_dw[hidden + s:hidden + e]
                    dw_b = torch.cat([dw_b1, dw_b2], dim=0)
                else:
                    dw_b = None

                y = F.conv2d(y, dw_w, bias=dw_b, stride=1, padding=1, groups=2 * (e - s))
                y1, y2 = y.chunk(2, dim=1)
                y = F.gelu(y1) * y2

                out_w = w_out[:, s:e]
                out = out + F.conv2d(y, out_w, bias=None, stride=1, padding=0)

            if b_out is not None:
                out = out + b_out.view(1, -1, 1, 1).to(dtype=out.dtype, device=out.device)

            return out

        m.forward = types.MethodType(_chunked_forward, m)
        patched += 1

    return patched



####################################################################################################
## HELPERS
def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)

def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))

def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=False)



####################################################################################################
## HELPERS
def _forward_model(net: nn.Module, x: torch.Tensor, de_id: torch.Tensor) -> torch.Tensor:
    if de_id is None:
        return net(x)
    try:
        return net(x, de_id)
    except TypeError:
        return net(x)


def _normalize_de_id(de_id, *, device: torch.device, opt=None):
    if de_id is None:
        return None

    if torch.is_tensor(de_id):
        if de_id.device != device:
            de_id = de_id.to(device, non_blocking=True)
        if de_id.dtype != torch.long:
            de_id = de_id.long()
        return de_id

    if isinstance(de_id, (int, np.integer)):
        return torch.tensor([int(de_id)], device=device, dtype=torch.long)

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

    if isinstance(de_id, str) and opt is not None:
        de_type = getattr(opt, "de_type", None)
        if isinstance(de_type, (list, tuple)):
            try:
                return torch.tensor([int(list(de_type).index(de_id))], device=device, dtype=torch.long)
            except Exception:
                return torch.tensor([0], device=device, dtype=torch.long)

    return None


def _load_module_from_file(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_torch_load(path: str):
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`.*",
    )

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


def _resolve_ckpt_path(opt) -> pathlib.Path:
    env_ckpt = os.environ.get("MOCEIR_TEST_CKPT_PATH", "")
    if str(env_ckpt).strip() != "":
        return pathlib.Path(str(env_ckpt)).expanduser()

    ckpt_path = getattr(opt, "ckpt_path", None)
    if ckpt_path:
        return pathlib.Path(str(ckpt_path)).expanduser()

    checkpoint_id = getattr(opt, "checkpoint_id", None)
    if not checkpoint_id:
        raise ValueError("Either opt.ckpt_path or opt.checkpoint_id must be set")

    # 优先使用 experiment_dir 而不是 ckpt_dir
    experiment_dir = getattr(opt, "experiment_dir", None)
    if not experiment_dir:
        # 如果没有 experiment_dir，尝试使用 ckpt_dir 作为后备
        ckpt_dir = getattr(opt, "ckpt_dir", None)
        if not ckpt_dir:
            raise ValueError("Either opt.experiment_dir or opt.ckpt_dir must be set")
        experiment_dir = ckpt_dir
    
    experiment_dir = pathlib.Path(str(experiment_dir)).expanduser().resolve()
    checkpoint_id = str(checkpoint_id)
    
    # 如果 checkpoint_id 是完整的 checkpoint 文件名（以 .ckpt 结尾）
    if checkpoint_id.lower().endswith(".ckpt"):
        # 在 experiment_dir 下搜索所有实验目录，找到包含该 checkpoint 文件的目录
        ckpt_name = pathlib.Path(checkpoint_id).name
        for exp_dir in experiment_dir.iterdir():
            if exp_dir.is_dir():
                ckpt_file = exp_dir / "checkpoints" / ckpt_name
                if ckpt_file.exists():
                    return ckpt_file
        # 如果找不到，尝试直接路径
        direct_path = experiment_dir / checkpoint_id
        if direct_path.exists():
            return direct_path
        raise FileNotFoundError(
            f"Checkpoint file not found: {ckpt_name}\n"
            f"Searched in: {experiment_dir}/*/checkpoints/{ckpt_name}"
        )
    
    # 如果 checkpoint_id 已经包含 /checkpoints，直接使用
    if "/checkpoints" in checkpoint_id:
        full_path = experiment_dir / checkpoint_id
        if full_path.exists():
            return full_path
        raise FileNotFoundError(f"Checkpoint path not found: {full_path}")
    
    # 如果 checkpoint_id 是实验目录名，在 checkpoints 子目录下查找
    # 统一使用 last.ckpt（最后一次epoch训练结束时的模型）
    ckpt_path = experiment_dir / checkpoint_id / "checkpoints" / "last.ckpt"
    if ckpt_path.exists():
        return ckpt_path
    
    # 如果 last.ckpt 不存在，列出可用的checkpoint文件以便调试
    checkpoints_dir = experiment_dir / checkpoint_id / "checkpoints"
    available_ckpts = []
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        available_ckpts = [f.name for f in checkpoints_dir.glob("*.ckpt")]
    
    # 抛出错误，明确要求使用 last.ckpt
    error_msg = (
        f"Checkpoint file not found: {experiment_dir / checkpoint_id / 'checkpoints' / 'last.ckpt'}\n"
        f"Experiment directory: {experiment_dir}\n"
        f"Checkpoint ID: {checkpoint_id}\n"
        f"Available checkpoints: {available_ckpts if available_ckpts else 'None'}\n"
        f"Note: Test script is configured to use 'last.ckpt' (the model from the last training epoch).\n"
        f"Available experiments: {[d.name for d in experiment_dir.iterdir() if d.is_dir()][:10]}"
    )
    raise FileNotFoundError(error_msg)


def _load_net_module(opt, ckpt_path: pathlib.Path):
    ckpt_path = ckpt_path.resolve()
    if ckpt_path.parent.name == "checkpoints":
        run_dir = ckpt_path.parent.parent
    else:
        run_dir = ckpt_path.parent
    net_snapshot_dir = run_dir / "net_snapshot"

    if net_snapshot_dir.exists() and net_snapshot_dir.is_dir():
        py_files = sorted(net_snapshot_dir.glob("*.py"))
        if len(py_files) == 0:
            pass
        elif len(py_files) == 1:
            net_file = py_files[0]
            module = _load_module_from_file(net_file, f"net_snapshot_{net_file.stem}")
            return module
        else:
            model_name = getattr(opt, "model", None)
            if model_name:
                cand = net_snapshot_dir / f"{model_name}.py"
                if cand.exists():
                    module = _load_module_from_file(cand, f"net_snapshot_{model_name}")
                    return module
            raise RuntimeError(f"Multiple .py files found in net_snapshot: {net_snapshot_dir}")

    project_dir = pathlib.Path(__file__).resolve().parent
    model_name = getattr(opt, "model", None)
    if not model_name:
        raise RuntimeError("opt.model is required when net_snapshot is not available")
    return importlib.import_module(f"net.{model_name}")



####################################################################################################
## DRMI Test Dataset (test/meta + test/ground_truth)
class DRMITestDataset(Dataset):
    """Simple test dataset for DRMI_dataset.

    Expected structure under data_file_dir:
        test/meta/*.png          (degraded)
        test/ground_truth/*.png  (ground truth)
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        # use 'deblur' id if present, otherwise 0
        self.de_id = self.de_dict.get("deblur", 0)
        self.patch_size = getattr(self.args, "patch_size", 128)
        self.full_res_eval = bool(getattr(self.args, "full_res_eval", False))

        data_dir = self.args.data_file_dir
        lr_dir = os.path.join(data_dir, "test", "meta")
        hr_dir = os.path.join(data_dir, "test", "ground_truth")

        self.lr = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
        self.hr = sorted(glob.glob(os.path.join(hr_dir, "*.png")))

        if len(self.lr) == 0 or len(self.hr) == 0:
            raise ValueError(f"No DRMI test images found under {lr_dir} and {hr_dir}")
        if len(self.lr) != len(self.hr):
            raise ValueError(f"LR/HR count mismatch in DRMI test set: {len(self.lr)} vs {len(self.hr)}")

    def __len__(self):
        return len(self.lr)

    def _center_crop_patch(self, img: np.ndarray) -> np.ndarray:
        """Center crop to patch_size x patch_size (if larger)."""
        h, w = img.shape[:2]
        p = int(self.patch_size)
        if h <= p or w <= p:
            return img

        top = (h - p) // 2
        left = (w - p) // 2
        return img[top:top + p, left:left + p, :]

    def __getitem__(self, idx):
        lr_path = self.lr[idx]
        hr_path = self.hr[idx]

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_np = np.array(lr_img)
        hr_np = np.array(hr_img)

        # Ensure size is multiple of 16 (same as training/validation) then take center patch
        lr_np = crop_img(lr_np, base=16)
        hr_np = crop_img(hr_np, base=16)

        if not self.full_res_eval:
            lr_np = self._center_crop_patch(lr_np)
            hr_np = self._center_crop_patch(hr_np)

        lr = self.toTensor(lr_np)
        hr = self.toTensor(hr_np)

        return [lr_path, self.de_id], lr, hr


####################################################################################################
def run_test(opts, accelerator: Accelerator, net, dataset, factor=8):
    batch_size = getattr(opts, "batch_size", 1)
    if bool(getattr(opts, "full_res_eval", False)) and int(batch_size) > 1:
        batch_size = 1

    if accelerator.num_processes > 1:
        indices = list(range(int(accelerator.process_index), len(dataset), int(accelerator.num_processes)))
        dataset = Subset(dataset, indices)
 
    testloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    if accelerator.num_processes == 1:
        testloader = accelerator.prepare(testloader)
    
    if opts.save_results:
        # 使用配置中的 results_dir，如果不存在则使用默认值
        results_base = getattr(opts, "results_dir", "results")
        results_base_path = pathlib.Path(str(results_base)).expanduser()
        if not results_base_path.is_absolute():
            project_dir = pathlib.Path(__file__).resolve().parent
            results_base_path = (project_dir / results_base_path).resolve()
        # 移除 rank 路径，所有图像保存到同一目录（只让 rank0 保存以避免重复）
        out_dir = results_base_path / str(opts.checkpoint_id) / str(opts.benchmarks[0])
        if accelerator.is_main_process:  # 只在主进程（rank0）创建目录
            out_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()  # 等待 rank0 创建目录

    calc_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg',
        normalize=True,
        reduction="sum",
    ).to(accelerator.device)

    psnr_sum_local = 0.0
    ssim_sum_local = 0.0
    lpips_sum_local = torch.zeros((), device=accelerator.device)
    count_local = 0
    amp_kwargs = _get_test_amp_autocast_kwargs(opts, accelerator.device)
    # 可选：使用 patch 拼接测试（例如 256x256 patch），通过环境变量控制
    tile_patch_size_env = os.environ.get("MOCEIR_TEST_TILE_PATCH_SIZE", "").strip()
    tile_patch_size = int(tile_patch_size_env) if tile_patch_size_env.isdigit() else None
    use_tiled_patches = tile_patch_size is not None and tile_patch_size > 0

    with torch.inference_mode(), torch.cuda.amp.autocast(**amp_kwargs):

        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader, disable=not accelerator.is_main_process):
            if torch.is_tensor(degrad_patch) and degrad_patch.device != accelerator.device:
                degrad_patch = degrad_patch.to(accelerator.device, non_blocking=True)
            if torch.is_tensor(clean_patch) and clean_patch.device != accelerator.device:
                clean_patch = clean_patch.to(accelerator.device, non_blocking=True)

            de_id = _normalize_de_id(de_id, device=accelerator.device, opt=opts)

            if use_tiled_patches:
                # 目前仅支持 batch_size=1 的 patch 拼接模式
                if degrad_patch.dim() != 4 or degrad_patch.size(0) != 1:
                    raise RuntimeError("Tiled patch testing currently only supports batch_size=1.")

                # [1, C, H, W]
                b, c, h, w = degrad_patch.shape
                ps = int(tile_patch_size)

                # 反射 padding 到 patch_size 的整数倍，便于无重叠拼接
                pad_h = (ps - (h % ps)) % ps
                pad_w = (ps - (w % ps)) % ps
                if pad_h > 0 or pad_w > 0:
                    degrad_padded = F.pad(degrad_patch, (0, pad_w, 0, pad_h), mode="reflect")
                    clean_padded = F.pad(clean_patch, (0, pad_w, 0, pad_h), mode="reflect")
                else:
                    degrad_padded = degrad_patch
                    clean_padded = clean_patch

                _, _, H_pad, W_pad = degrad_padded.shape
                n_h = H_pad // ps
                n_w = W_pad // ps

                # 切成 [Npatch, C, ps, ps]
                degrad_patches = degrad_padded.view(1, c, n_h, ps, n_w, ps)
                degrad_patches = degrad_patches.permute(0, 2, 4, 1, 3, 5).reshape(n_h * n_w, c, ps, ps)

                clean_patches = clean_padded.view(1, c, n_h, ps, n_w, ps)
                clean_patches = clean_patches.permute(0, 2, 4, 1, 3, 5).reshape(n_h * n_w, c, ps, ps)

                # 为每个 patch 复制 de_id
                if de_id is not None:
                    if de_id.dim() == 1:
                        de_id_patches = de_id.expand(degrad_patches.size(0))
                    else:
                        de_id_patches = de_id.expand(degrad_patches.size(0), *de_id.shape[1:])
                else:
                    de_id_patches = None

                restored_patches = _forward_model(net, degrad_patches, de_id_patches)
                if isinstance(restored_patches, (list, tuple)) and len(restored_patches) == 2:
                    restored_patches, _ = restored_patches

                # 拼接回整张图 [1, C, H_pad, W_pad]
                restored_patches = restored_patches.view(1, n_h, n_w, c, ps, ps)
                restored_patches = restored_patches.permute(0, 3, 1, 4, 2, 5).reshape(1, c, H_pad, W_pad)

                # 裁回原始尺寸
                restored = restored_patches[:, :, :h, :w]
                clean_patch = clean_padded[:, :, :h, :w]
            else:
                # 原始：整图一次性前向，不做 patch 拼接
                restored = _forward_model(net, degrad_patch, de_id)
                if isinstance(restored, (list, tuple)) and len(restored) == 2:
                    restored, _ = restored
            
            # Unpad images to original dimensions / consistency check
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."

            # save output images
            restored = torch.clamp(restored,0,1)
            lpips_vals = calc_lpips(clean_patch, restored).detach().float()
            if lpips_vals.numel() > 1:
                lpips_sum_local = lpips_sum_local + lpips_vals.sum()
            else:
                lpips_sum_local = lpips_sum_local + lpips_vals.reshape(()).sum()
              
            restored_np = restored.detach().cpu().permute(0, 2, 3, 1).numpy()
            clean_np = clean_patch.detach().cpu().permute(0, 2, 3, 1).numpy()
            bs = int(restored_np.shape[0])
            
            # Create temporary directory for saving images
            temp_dir = tempfile.mkdtemp(prefix="test_uint8_")
            try:
                for i in range(bs):
                    # Convert to uint8 and save restored image
                    restored_uint8 = img_as_ubyte(restored_np[i])
                    temp_restored_path = os.path.join(temp_dir, f"restored_{i}.png")
                    save_img(temp_restored_path, restored_uint8)
                    
                    # Load back the saved image (now in uint8 format)
                    restored_loaded = load_img(temp_restored_path)
                    
                    # Convert clean image to uint8 for comparison
                    clean_uint8 = img_as_ubyte(clean_np[i])
                    
                    # Calculate metrics on uint8 images (data_range=255)
                    ssim_sum_local += float(structural_similarity(clean_uint8, restored_loaded, channel_axis=2, gaussian_weights=True, data_range=255))
                    psnr_sum_local += float(peak_signal_noise_ratio(clean_uint8, restored_loaded, data_range=255))
                count_local += bs
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            if opts.save_results and accelerator.is_main_process:  # 只在主进程（rank0）保存图像
                for i in range(int(restored_np.shape[0])):
                    # 使用原文件名（不添加 PSNR 后缀）
                    restored_uint8 = img_as_ubyte(restored_np[i])
                    # 从 clean_name（低分辨率图像路径）提取文件名，保持原文件名
                    original_filename = os.path.split(clean_name[i])[-1]
                    # 如果原文件名没有扩展名，添加 .png
                    if not os.path.splitext(original_filename)[1]:
                        save_name = original_filename + '.png'
                    else:
                        # 保持原扩展名，但如果是非图片格式，改为 .png
                        ext = os.path.splitext(original_filename)[1].lower()
                        if ext in ['.png', '.jpg', '.jpeg']:
                            save_name = original_filename
                        else:
                            save_name = os.path.splitext(original_filename)[0] + '.png'
                    save_img(str(out_dir / save_name), restored_uint8)

    psnr_sum = accelerator.reduce(torch.tensor(psnr_sum_local, device=accelerator.device), reduction="sum")
    ssim_sum = accelerator.reduce(torch.tensor(ssim_sum_local, device=accelerator.device), reduction="sum")
    lpips_sum = accelerator.reduce(lpips_sum_local, reduction="sum")
    total_count = accelerator.reduce(torch.tensor(count_local, device=accelerator.device, dtype=torch.long), reduction="sum")
    accelerator.wait_for_everyone()

    metrics = None
    if accelerator.is_main_process:
        denom = float(total_count.detach().cpu()) if int(total_count.detach().cpu()) > 0 else 1.0
        psnr = float(psnr_sum.detach().cpu()) / denom
        ssim = float(ssim_sum.detach().cpu()) / denom
        lpips = float(lpips_sum.detach().cpu()) / denom
        count = int(total_count.detach().cpu())
        print('PSNR: {:.5f} SSIM: {:.5f} LPIPS: {:.5f} COUNT: {}\n'.format(psnr, ssim, lpips, count))
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "count": count,
        }

    return metrics

## test LolV1
def run_lolv1(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)
     
## test GoPro
def run_gopro(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)
         
## test Derain
def run_derain(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)
         
## test Dehaze
def run_dehaze(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)
     
## test synthetic denoising
def run_denoise_15(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)
     
def run_denoise_25(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)
     
def run_denoise_50(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)

# test CDD11
def run_cdd11(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)

# test DRMI_dataset
def run_drmi(opts, accelerator: Accelerator, net, dataset, factor=8):
    return run_test(opts, accelerator, net, dataset, factor)


####################################################################################################
## main
def main(opt):
    try:
        # 抑制所有警告输出
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        
        # 在运行时进一步抑制 PyTorch 分布式警告
        warnings.filterwarnings("ignore", message=".*process group.*")
        warnings.filterwarnings("ignore", message=".*ProcessGroupNCCL.*")
        warnings.filterwarnings("ignore", message=".*OMP_NUM_THREADS.*")
        warnings.filterwarnings("ignore", message=".*torch.distributed.run.*")
        
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
                    "Setting OMP_NUM_THREADS",
                    "WARNING: process group",
                    "*****************************************",
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
        
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        mp = _resolve_test_mixed_precision(opt)
        if mp == "no":
            accelerator = Accelerator()
        else:
            accelerator = Accelerator(mixed_precision=mp)

        if accelerator.is_main_process:
            print(f"[Test] precision={mp}")
            # 清空 metrics，避免重复累积
            if hasattr(opt, "_metrics"):
                opt._metrics = []

        # Load model
        ckpt_path = _resolve_ckpt_path(opt)
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        
        # 检查是否存在父文件夹结构（用于多数据集训练）
        # 如果 run_dir 的父目录存在且包含多个子文件夹，说明是父文件夹结构
        parent_exp_dir = None
        sub_exp_dir = run_dir
        if run_dir.parent.exists() and run_dir.parent.is_dir():
            # 检查父目录是否包含多个子实验文件夹（通过检查是否有多个包含相同时间戳的文件夹）
            # 例如：MoCE_IR_S-open_dataset_8_1_1_mini, MoCE_IR_S-open_dataset_8_1_1_mini2, MoCE_IR_S-open_dataset_8_1_1_mini3
            sub_dir_name = run_dir.name
            # 匹配模式：MoCE_IR_S-open_dataset_8_1_1_mini3_2026_02_14_18_43_23
            # 提取：MoCE_IR_S-open_dataset_8_1_1 和 2026_02_14_18_43_23
            pattern = r'^(.*?)(_mini\d*)?(_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$'
            match = re.match(pattern, sub_dir_name)
            if match:
                base_name = match.group(1)
                timestamp = match.group(3)
                potential_parent_name = f"{base_name}{timestamp}"
                # 检查父目录是否就是这个名称（说明是父文件夹结构）
                if run_dir.parent.name == potential_parent_name:
                    parent_exp_dir = run_dir.parent
                    sub_exp_dir = run_dir
                else:
                    # 方法1：检查父目录名是否包含 $DATASET_LIST 或类似模式（多数据集训练的标识）
                    parent_name = run_dir.parent.name
                    if '$DATASET_LIST' in parent_name or 'DATASET_LIST' in parent_name:
                        # 检查父目录名和子目录名是否有相同的时间戳
                        parent_timestamp_pattern = r'_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$'
                        parent_timestamp_match = re.search(parent_timestamp_pattern, parent_name)
                        if parent_timestamp_match and parent_timestamp_match.group(1) == timestamp.lstrip('_'):
                            parent_exp_dir = run_dir.parent
                            sub_exp_dir = run_dir
                    
                    # 方法2：检查父目录下是否有多个包含相同时间戳的子文件夹
                    # 查找所有匹配模式的子文件夹
                    if parent_exp_dir is None:
                        parent_candidates = []
                        for candidate in run_dir.parent.iterdir():
                            if candidate.is_dir() and re.match(pattern, candidate.name):
                                parent_candidates.append(candidate)
                        # 如果有多个子文件夹，说明父目录就是父文件夹
                        if len(parent_candidates) > 1:
                            parent_exp_dir = run_dir.parent
                            sub_exp_dir = run_dir
                        # 即使只有一个子文件夹，如果父目录名包含 $DATASET_LIST，也认为是父文件夹结构
                        elif len(parent_candidates) == 1 and '$DATASET_LIST' in parent_name:
                            parent_exp_dir = run_dir.parent
                            sub_exp_dir = run_dir
        
        if getattr(opt, "checkpoint_id", None) is None:
            # 简化 checkpoint_id：只使用 checkpoint 文件名（去掉路径和扩展名）
            # 例如：last.ckpt -> last, best_psnr_ssim-epoch=379-psnr=39.23065-ssim=0.96908.ckpt -> best_psnr_ssim-epoch=379-psnr=39.23065-ssim=0.96908
            opt.checkpoint_id = ckpt_path.stem
        print(f"[Test] Loading checkpoint from: {ckpt_path}")

        module = _load_net_module(opt, ckpt_path)
        build_fn = getattr(module, "build_model", None)
        if build_fn is None:
            raise AttributeError("Network module must define build_model(opt)")

        net = build_fn(opt)
        ckpt = _safe_torch_load(str(ckpt_path))
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = _extract_net_state_dict(state_dict)
        # 抑制 load_state_dict 的警告输出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.load_state_dict(state_dict, strict=False)
        net.eval()

        ffn_chunk = _resolve_ffn_chunk(opt)
        patched = _apply_ffn_chunking(net, chunk_size=ffn_chunk)
        if accelerator.is_main_process and ffn_chunk > 0:
            print(f"[Test] ffn_chunk={ffn_chunk} patched_ffn={patched}")

        net = accelerator.prepare(net)
        net.eval()
        
        # 计算模型复杂度（只在主进程计算一次）
        gflops = None
        parameters = None
        if accelerator.is_main_process:
            # print("[Test] Calculating model complexity...")
            try:
                unwrapped_net = accelerator.unwrap_model(net)
                gflops, parameters = _calculate_model_complexity(unwrapped_net)
                if gflops is not None and parameters is not None:
                    print(f"[Test] {gflops:.2f}G / {parameters:.2f}M")
                elif parameters is not None:
                    print(f"[Test] Params: {parameters:.2f}M")
            except Exception as e:
                # print(f"[Test] Error calculating model complexity: {e}")
                # traceback.print_exc()
                gflops = None
                parameters = None
        
        # 确保所有进程同步（等待主进程完成计算）
        accelerator.wait_for_everyone()
        
        # 获取数据集列表，去重确保每个 benchmark 只测试一次
        benchmarks_list = list(dict.fromkeys(opt.benchmarks))  # 保持顺序并去重
        num_benchmarks = len(benchmarks_list)
        
        for idx, de in enumerate(benchmarks_list):
            # 创建独立的 opt 副本，避免修改原始 opt
            ind_opt = copy.deepcopy(opt)
            ind_opt.benchmarks = [de]
            # 独立测试脚本统一使用 test split
            setattr(ind_opt, "split", "test")
            
            if de == "drmi":
                dataset = DRMITestDataset(ind_opt)
            elif "CDD11" in opt.trainset:
                _, subset = opt.trainset.split("_", maxsplit=1)
                dataset = CDD11(ind_opt, split="test", subset=subset)
            else:
                dataset = IRBenchmarks(ind_opt)
            
            # print("--------> Testing on", de, "testset.")
            # print("\n")
            # 使用 ind_opt 而不是 opt，确保只测试当前 benchmark
            m = globals()[f"run_{de}"](ind_opt, accelerator, net, dataset, factor=8)
            if accelerator.is_main_process:
                # 确保 _metrics 已初始化（在测试开始时已清空）
                if not hasattr(opt, "_metrics"):
                    opt._metrics = []
                # 检查是否已存在该 benchmark 的 metrics，避免重复追加
                existing_benchmark = any(row.get("benchmark") == str(de) for row in opt._metrics)
                if not existing_benchmark:
                    opt._metrics.append({
                        "benchmark": str(de),
                        **(m or {}),
                    })
                
                # 在数据集之间添加分割线（最后一个数据集后不添加）
                if idx < num_benchmarks - 1:
                    print("=" * 80)

        if accelerator.is_main_process:
            # 优先使用环境变量，其次使用配置中的 test_dir，最后使用默认值
            result_root = os.environ.get("MOCEIR_TEST_RESULT_DIR", None)
            if result_root is None:
                result_root = getattr(opt, "test_dir", "test")
            result_root_path = pathlib.Path(str(result_root)).expanduser()
            if not result_root_path.is_absolute():
                project_dir = pathlib.Path(__file__).resolve().parent
                result_root_path = (project_dir / result_root_path).resolve()
            result_root_path.mkdir(parents=True, exist_ok=True)

            ckpt_abs = str(ckpt_path.resolve())
            net_abs = ""
            try:
                net_abs = str(pathlib.Path(str(getattr(module, "__file__", ""))).resolve())
            except Exception:
                net_abs = str(getattr(module, "__file__", ""))
            
            # 提取网络名称
            net_name = _extract_net_name(ckpt_abs, net_abs)

            # 提取数据集名称：优先从 data_file_dir 取最后一级目录名，否则用 trainset
            # 关键修复：确保使用测试时实际使用的数据集名称
            dataset_name = None
            # 优先从环境变量获取（如果是从train.sh调用的，环境变量会包含训练时使用的数据集）
            env_data_dir = os.environ.get("MOCEIR_TEST_DATA_FILE_DIR", None)
            if env_data_dir:
                dataset_name = os.path.basename(str(env_data_dir).rstrip(os.sep))
                print(f"[Test] Using dataset name from MOCEIR_TEST_DATA_FILE_DIR: {dataset_name}", flush=True)
            # 如果没有环境变量，从 opt.data_file_dir 获取
            if not dataset_name:
                data_dir = getattr(opt, "data_file_dir", None)
                if data_dir:
                    dataset_name = os.path.basename(str(data_dir).rstrip(os.sep))
                    print(f"[Test] Using dataset name from opt.data_file_dir: {dataset_name}", flush=True)
            # 最后回退：使用 trainset
            if not dataset_name:
                dataset_name = str(getattr(opt, "trainset", "")).strip()
                print(f"[Test] Warning: Using dataset name from trainset: {dataset_name}", flush=True)
            dataset_name = dataset_name or ""
            print(f"[Test] Final dataset name for CSV: {dataset_name}", flush=True)

            ckpt_str = str(ckpt_path)
            ckpt_name = ckpt_path.stem
            ckpt_parent = ckpt_path.parent.name
            run_tag = str(getattr(opt, "checkpoint_id", "test")).replace("/", "_")
            out_dir = result_root_path / run_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "net_name": net_name,
                "dataset": dataset_name,
                "gflops": gflops,
                "parameters": parameters,
                "ckpt_path": ckpt_str,
                "ckpt_name": ckpt_name,
                "ckpt_parent": ckpt_parent,
                "benchmarks": list(getattr(opt, "benchmarks", [])),
                "full_res_eval": bool(getattr(opt, "full_res_eval", False)),
                "batch_size": int(getattr(opt, "batch_size", 1)),
                "metrics": getattr(opt, "_metrics", []),
            }

            metrics_json = out_dir / "metrics.json"
            with open(str(metrics_json), "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            metrics_txt = out_dir / "metrics.txt"
            with open(str(metrics_txt), "w") as f:
                for row in payload["metrics"]:
                    f.write(
                        f"benchmark={row.get('benchmark')} psnr={row.get('psnr')} ssim={row.get('ssim')} lpips={row.get('lpips')} count={row.get('count')}\n"
                    )

            csv_path = result_root_path / "test.csv"
            csv_exists = csv_path.exists()
            # 新的列顺序：net_name, dataset, gflops, parameters, benchmark, psnr, ssim, lpips, count, ckpt_path, net_path
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
            with open(str(csv_path), "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not csv_exists:
                    writer.writeheader()
                # 确保每个 benchmark 只写入一次，使用 set 去重
                seen_benchmarks = set()
                for row in payload.get("metrics", []) or []:
                    benchmark_key = row.get("benchmark", "")
                    # 如果该 benchmark 已经写入过，跳过（避免重复记录）
                    if benchmark_key and benchmark_key in seen_benchmarks:
                        continue
                    seen_benchmarks.add(benchmark_key)
                    writer.writerow(
                        {
                            "net_name": net_name,
                            "dataset": dataset_name,
                            "gflops": f"{gflops:.5f}" if gflops is not None else "",
                            "parameters": f"{parameters:.5f}" if parameters is not None else "",
                            "benchmark": benchmark_key,
                            "psnr": f"{row.get('psnr', ''):.5f}" if isinstance(row.get("psnr"), (int, float)) else row.get("psnr", ""),
                            "ssim": f"{row.get('ssim', ''):.5f}" if isinstance(row.get("ssim"), (int, float)) else row.get("ssim", ""),
                            "lpips": f"{row.get('lpips', ''):.5f}" if isinstance(row.get("lpips"), (int, float)) else row.get("lpips", ""),
                            "count": str(row.get("count", "")) if row.get("count") is not None and row.get("count") != "" else "",
                            "ckpt_path": ckpt_abs,
                            "net_path": net_abs,
                        }
                    )

            exp_test_root = None
            try:
                # 如果存在父文件夹结构，在父文件夹下创建 test 目录
                # 否则在子文件夹下创建 test 目录
                if parent_exp_dir is not None:
                    # 在父文件夹下创建 test 目录
                    exp_test_root = parent_exp_dir.resolve() / "test"
                    exp_test_root.mkdir(parents=True, exist_ok=True)
                    # 在 test 目录下为每个子实验创建对应的子目录（使用子文件夹名称）
                    exp_out_dir = exp_test_root / sub_exp_dir.name / str(ckpt_path.stem)
                else:
                    # 原有逻辑：在子文件夹下创建 test 目录
                    exp_test_root = pathlib.Path(str(run_dir)).resolve() / "test"
                    exp_test_root.mkdir(parents=True, exist_ok=True)
                    exp_out_dir = exp_test_root / str(ckpt_path.stem)
                exp_out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                exp_test_root = None
                exp_out_dir = None

            if exp_test_root is not None and exp_out_dir is not None:

                exp_metrics_json = exp_out_dir / "metrics.json"
                with open(str(exp_metrics_json), "w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                exp_metrics_txt = exp_out_dir / "metrics.txt"
                with open(str(exp_metrics_txt), "w") as f:
                    for row in payload["metrics"]:
                        f.write(
                            f"benchmark={row.get('benchmark')} psnr={row.get('psnr')} ssim={row.get('ssim')} lpips={row.get('lpips')} count={row.get('count')}\n"
                        )

                exp_csv_path = exp_test_root / "test.csv"
                exp_csv_exists = exp_csv_path.exists()
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
                with open(str(exp_csv_path), "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not exp_csv_exists:
                        writer.writeheader()
                    for row in payload.get("metrics", []) or []:
                        writer.writerow(
                            {
                                "net_name": net_name,
                                "dataset": dataset_name,
                                "gflops": f"{gflops:.5f}" if gflops is not None else "",
                                "parameters": f"{parameters:.5f}" if parameters is not None else "",
                                "benchmark": row.get("benchmark", ""),
                                "psnr": f"{row.get('psnr', ''):.5f}" if isinstance(row.get("psnr"), (int, float)) else row.get("psnr", ""),
                                "ssim": f"{row.get('ssim', ''):.5f}" if isinstance(row.get("ssim"), (int, float)) else row.get("ssim", ""),
                                "lpips": f"{row.get('lpips', ''):.5f}" if isinstance(row.get("lpips"), (int, float)) else row.get("lpips", ""),
                                "count": row.get("count", ""),
                                "ckpt_path": ckpt_abs,
                                "net_path": net_abs,
                            }
                        )
    except Exception as e:
        rank = os.environ.get("RANK")
        local_rank = os.environ.get("LOCAL_RANK")
        print(f"[Test][Error] rank={rank} local_rank={local_rank} err={e}", file=sys.stderr)
        traceback.print_exc()
        raise
    

def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
if __name__ == '__main__':
    # test.py 作为函数库，__main__ 部分只从环境变量读取配置（由 train.sh 或 test.sh 设置）
    # 这样 train.sh 可以通过环境变量调用 test.py 的测试模块
    
    # 从环境变量读取配置（优先级：环境变量 > None，让 _resolve_ckpt_path 处理）
    ckpt_path = os.environ.get("MOCEIR_TEST_CKPT_PATH", None)
    data_file_dir = os.environ.get("MOCEIR_TEST_DATA_FILE_DIR", None)
    trainset = os.environ.get("MOCEIR_TEST_TRAINSET", None)
    benchmarks_str = os.environ.get("MOCEIR_TEST_BENCHMARKS", None)
    de_type_str = os.environ.get("MOCEIR_TEST_DE_TYPE", None)
    patch_size_str = os.environ.get("MOCEIR_TEST_PATCH_SIZE", None)
    batch_size_str = os.environ.get("MOCEIR_TEST_BATCH_SIZE", None)
    save_results_str = os.environ.get("MOCEIR_TEST_SAVE_RESULTS", None)
    precision = os.environ.get("MOCEIR_TEST_PRECISION", None)
    full_res_eval_str = os.environ.get("MOCEIR_TEST_FULL_RES_EVAL", None)
    results_dir = os.environ.get("MOCEIR_TEST_RESULTS_DIR", None)
    
    # 解析 benchmarks
    benchmarks = ["gopro"]  # 默认值
    if benchmarks_str:
        benchmarks = [x.strip() for x in benchmarks_str.split(",")]
    
    # 解析 de_type
    de_type = ["deblur"]  # 默认值
    if de_type_str:
        de_type = [x.strip() for x in de_type_str.split(",")]
    
    # 解析其他参数
    patch_size = int(patch_size_str) if patch_size_str else 256
    batch_size = int(batch_size_str) if batch_size_str else 1
    save_results = str2bool(save_results_str) if save_results_str else False
    if precision is None:
        precision = "fp16"  # 默认值
    full_res_eval = str2bool(full_res_eval_str) if full_res_eval_str else True
    
    train_opt = argparse.Namespace(
        ckpt_path=ckpt_path,
        model=None,
        data_file_dir=data_file_dir,
        trainset=trainset,
        benchmarks=benchmarks,
        de_type=de_type,
        patch_size=patch_size,
        batch_size=batch_size,
        save_results=save_results,
        precision=precision,
        full_res_eval=full_res_eval,
        results_dir=results_dir,
    )
    main(train_opt)
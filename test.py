import os
import pathlib
import argparse
import importlib
import importlib.util
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import csv
import sys
import traceback
import types

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
from PIL import Image

from utils.test_utils import save_img
from utils.image_utils import crop_img
from data.dataset_utils import IRBenchmarks, CDD11

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
                with torch.no_grad():
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

    ckpt_dir = getattr(opt, "ckpt_dir", None)
    checkpoint_id = getattr(opt, "checkpoint_id", None)
    if not ckpt_dir or not checkpoint_id:
        raise ValueError("Either opt.ckpt_path or (opt.ckpt_dir and opt.checkpoint_id) must be set")

    if str(checkpoint_id).lower().endswith(".ckpt"):
        return (pathlib.Path(str(ckpt_dir)) / str(checkpoint_id)).expanduser()
    return (pathlib.Path(str(ckpt_dir)) / str(checkpoint_id) / "last.ckpt").expanduser()


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
        out_dir = results_base_path / str(opts.checkpoint_id) / str(opts.benchmarks[0]) / f"rank{accelerator.process_index}"
        out_dir.mkdir(parents=True, exist_ok=True)

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
    with torch.inference_mode(), torch.cuda.amp.autocast(**amp_kwargs):

        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader, disable=not accelerator.is_main_process):
            if torch.is_tensor(degrad_patch) and degrad_patch.device != accelerator.device:
                degrad_patch = degrad_patch.to(accelerator.device, non_blocking=True)
            if torch.is_tensor(clean_patch) and clean_patch.device != accelerator.device:
                clean_patch = clean_patch.to(accelerator.device, non_blocking=True)

            de_id = _normalize_de_id(de_id, device=accelerator.device, opt=opts)

            # Forward pass (full image, no tiling)
            restored = _forward_model(net, degrad_patch, de_id)
            if isinstance(restored, (list, tuple)) and len(restored) == 2:
                restored, _ = restored
            
            # Unpad images to original dimensions
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
            for i in range(bs):
                ssim_sum_local += float(calc_ssim(clean_np[i], restored_np[i]))
                psnr_sum_local += float(peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1))
            count_local += bs
            
            if opts.save_results:
                for i in range(int(restored_np.shape[0])):
                    psnr_temp = float(peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1))
                    save_name = os.path.splitext(os.path.split(clean_name[i])[-1])[0] + '_' + str(round(psnr_temp, 2)) +'.png'
                    save_img(str(out_dir / save_name), img_as_ubyte(restored_np[i]))

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
        print('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(psnr, ssim, lpips))
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "count": int(total_count.detach().cpu()),
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

        # Load model
        ckpt_path = _resolve_ckpt_path(opt)
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        if getattr(opt, "checkpoint_id", None) is None:
            opt.checkpoint_id = f"{run_dir.name}-{ckpt_path.stem}"
        print(f"[Test] Loading checkpoint from: {ckpt_path}")

        module = _load_net_module(opt, ckpt_path)
        build_fn = getattr(module, "build_model", None)
        if build_fn is None:
            raise AttributeError("Network module must define build_model(opt)")

        net = build_fn(opt)
        ckpt = _safe_torch_load(str(ckpt_path))
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = _extract_net_state_dict(state_dict)
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
            print("[Test] Calculating model complexity...")
            try:
                unwrapped_net = accelerator.unwrap_model(net)
                gflops, parameters = _calculate_model_complexity(unwrapped_net)
                if gflops is not None:
                    print(f"[Test] Model complexity: {gflops:.2f} GFLOPs, {parameters:.2f}M parameters")
                elif parameters is not None:
                    print(f"[Test] Model parameters: {parameters:.2f}M (GFLOPs calculation failed)")
                else:
                    print(f"[Test] Failed to calculate model complexity")
            except Exception as e:
                print(f"[Test] Error calculating model complexity: {e}")
                traceback.print_exc()
                gflops = None
                parameters = None
        
        # 确保所有进程同步（等待主进程完成计算）
        accelerator.wait_for_everyone()
        
        for de in opt.benchmarks:
            ind_opt = opt
            ind_opt.benchmarks = [de]
            
            if de == "drmi":
                dataset = DRMITestDataset(ind_opt)
            elif "CDD11" in opt.trainset:
                _, subset = opt.trainset.split("_", maxsplit=1)
                dataset = CDD11(opt, split="test", subset=subset)
            else:
                dataset = IRBenchmarks(ind_opt)
            
            print("--------> Testing on", de, "testset.")
            print("\n")
            m = globals()[f"run_{de}"](opt, accelerator, net, dataset, factor=8)
            if accelerator.is_main_process:
                if not hasattr(opt, "_metrics"):
                    opt._metrics = []
                opt._metrics.append({
                    "benchmark": str(de),
                    **(m or {}),
                })

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

            ckpt_str = str(ckpt_path)
            ckpt_name = ckpt_path.stem
            ckpt_parent = ckpt_path.parent.name
            run_tag = str(getattr(opt, "checkpoint_id", "test")).replace("/", "_")
            out_dir = result_root_path / run_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "net_name": net_name,
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
            # 新的列顺序：net_name, gflops, parameters, benchmark, psnr, ssim, lpips, count, ckpt_path, net_path
            fieldnames = [
                "net_name",
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
                for row in payload.get("metrics", []) or []:
                    writer.writerow(
                        {
                            "net_name": net_name,
                            "gflops": f"{gflops:.4f}" if gflops is not None else "",
                            "parameters": f"{parameters:.4f}" if parameters is not None else "",
                            "benchmark": row.get("benchmark", ""),
                            "psnr": row.get("psnr", ""),
                            "ssim": row.get("ssim", ""),
                            "lpips": row.get("lpips", ""),
                            "count": row.get("count", ""),
                            "ckpt_path": ckpt_abs,
                            "net_path": net_abs,
                        }
                    )

            exp_test_root = None
            try:
                exp_test_root = pathlib.Path(str(run_dir)).resolve() / "test"
                exp_test_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                exp_test_root = None

            if exp_test_root is not None:
                exp_out_dir = exp_test_root / str(ckpt_path.stem)
                exp_out_dir.mkdir(parents=True, exist_ok=True)

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
                                "gflops": f"{gflops:.4f}" if gflops is not None else "",
                                "parameters": f"{parameters:.4f}" if parameters is not None else "",
                                "benchmark": row.get("benchmark", ""),
                                "psnr": row.get("psnr", ""),
                                "ssim": row.get("ssim", ""),
                                "lpips": row.get("lpips", ""),
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
    )
    main(train_opt)
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
import sys
import traceback

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import types
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor
from PIL import Image

from utils.test_utils import save_img
from utils.image_utils import crop_img
from data.dataset_utils import IRBenchmarks, CDD11


def compute_psnr(image_true, image_test, image_mask, data_range=None):
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(
        tar_img,
        prd_img,
        channel_axis=2,
        gaussian_weights=True,
        data_range=1.0,
        full=True,
    )
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]
    ssim = ssim.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim


def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))


def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range=1.0, full=False)


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


class DRMITestDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
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

        lr_np = crop_img(lr_np, base=16)
        hr_np = crop_img(hr_np, base=16)

        if not self.full_res_eval:
            lr_np = self._center_crop_patch(lr_np)
            hr_np = self._center_crop_patch(hr_np)

        lr = self.toTensor(lr_np)
        hr = self.toTensor(hr_np)

        return [lr_path, self.de_id], lr, hr


def _init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size, True

    return 0, 0, 1, False


def _ddp_reduce_sum(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def _resolve_test_precision(opt) -> str:
    val = getattr(opt, "precision", None)
    if val is None or str(val).strip() == "":
        val = os.environ.get("MOCEIR_TEST_AMP", "")
    val = str(val).strip().lower()
    if val in ("1", "true", "on", "fp16", "float16", "16", "half"):
        return "fp16"
    if val in ("bf16", "bfloat16"):
        return "bf16"
    return "no"


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

                dw_w = torch.cat([w_dw[s:e], w_dw[hidden + s:hidden + e]], dim=0)
                if b_dw is not None:
                    dw_b = torch.cat([b_dw[s:e], b_dw[hidden + s:hidden + e]], dim=0)
                else:
                    dw_b = None

                y = F.conv2d(y, dw_w, bias=dw_b, stride=1, padding=1, groups=2 * (e - s))
                y1, y2 = y.chunk(2, dim=1)
                y = F.gelu(y1) * y2

                out = out + F.conv2d(y, w_out[:, s:e], bias=None, stride=1, padding=0)

            if b_out is not None:
                out = out + b_out.view(1, -1, 1, 1).to(dtype=out.dtype, device=out.device)
            return out

        m.forward = types.MethodType(_chunked_forward, m)
        patched += 1

    return patched


def _get_amp_autocast_kwargs(opt, device: torch.device) -> dict:
    mp = _resolve_test_precision(opt)
    if device.type != "cuda":
        return {"enabled": False}
    if mp == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    if mp == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    return {"enabled": False}


def run_test(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    batch_size = getattr(opts, "batch_size", 1)
    if bool(getattr(opts, "full_res_eval", False)) and int(batch_size) > 1:
        batch_size = 1

    if world_size > 1:
        indices = list(range(int(rank), len(dataset), int(world_size)))
        dataset = Subset(dataset, indices)

    testloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    if opts.save_results:
        # 使用配置中的 results_dir，如果不存在则使用默认值
        results_base = getattr(opts, "results_dir", "results")
        results_base_path = pathlib.Path(str(results_base)).expanduser()
        if not results_base_path.is_absolute():
            project_dir = pathlib.Path(__file__).resolve().parent
            results_base_path = (project_dir / results_base_path).resolve()
        out_dir = results_base_path / str(opts.checkpoint_id) / str(opts.benchmarks[0]) / f"rank{rank}"
        out_dir.mkdir(parents=True, exist_ok=True)

    calc_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg",
        normalize=True,
        reduction="none",
    ).to(device)

    psnr_sum_local = 0.0
    ssim_sum_local = 0.0
    lpips_sum_local = torch.zeros((), device=device)
    count_local = 0

    amp_kwargs = _get_amp_autocast_kwargs(opts, device)
    with torch.inference_mode(), torch.cuda.amp.autocast(**amp_kwargs):
        it = tqdm(testloader, disable=(rank != 0))
        for ([clean_name, de_id], degrad_patch, clean_patch) in it:
            if torch.is_tensor(degrad_patch) and degrad_patch.device != device:
                degrad_patch = degrad_patch.to(device, non_blocking=True)
            if torch.is_tensor(clean_patch) and clean_patch.device != device:
                clean_patch = clean_patch.to(device, non_blocking=True)

            de_id = _normalize_de_id(de_id, device=device, opt=opts)

            restored = _forward_model(net, degrad_patch, de_id)
            if isinstance(restored, (list, tuple)) and len(restored) == 2:
                restored, _ = restored

            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."

            restored = torch.clamp(restored, 0, 1)
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
                    save_name = os.path.splitext(os.path.split(clean_name[i])[-1])[0] + "_" + str(round(psnr_temp, 2)) + ".png"
                    save_img(str(out_dir / save_name), img_as_ubyte(restored_np[i]))

    psnr_sum = _ddp_reduce_sum(torch.tensor(psnr_sum_local, device=device), ddp_enabled)
    ssim_sum = _ddp_reduce_sum(torch.tensor(ssim_sum_local, device=device), ddp_enabled)
    lpips_sum = _ddp_reduce_sum(lpips_sum_local, ddp_enabled)
    total_count = _ddp_reduce_sum(torch.tensor(count_local, device=device, dtype=torch.long), ddp_enabled)

    if ddp_enabled:
        dist.barrier()

    metrics = None
    if rank == 0:
        denom = float(total_count.detach().cpu()) if int(total_count.detach().cpu()) > 0 else 1.0
        psnr = float(psnr_sum.detach().cpu()) / denom
        ssim = float(ssim_sum.detach().cpu()) / denom
        lpips = float(lpips_sum.detach().cpu()) / denom
        print("PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n".format(psnr, ssim, lpips))
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "count": int(total_count.detach().cpu()),
        }

    return metrics


def run_lolv1(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_gopro(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_derain(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_dehaze(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_denoise_15(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_denoise_25(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_denoise_50(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_cdd11(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


def run_drmi(opts, device: torch.device, net, dataset, ddp_enabled: bool, rank: int, world_size: int, factor=8):
    return run_test(opts, device, net, dataset, ddp_enabled, rank, world_size, factor)


@record
def main(opt):
    rank = None
    local_rank = None
    ddp_enabled = False
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        rank, local_rank, world_size, ddp_enabled = _init_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        if rank == 0:
            mp = _resolve_test_precision(opt)
            ra = bool(getattr(opt, "recompute_activations", False))
            print(f"[TestDDP] precision={mp} recompute_activations={ra}")
            if ra:
                print("[TestDDP] Note: recompute-activations is a training/backward memory technique; in inference_mode it has no effect.")

        ckpt_path = _resolve_ckpt_path(opt)
        if getattr(opt, "checkpoint_id", None) is None:
            run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
            opt.checkpoint_id = f"{run_dir.name}-{ckpt_path.stem}"
        if rank == 0:
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
        patched_ffn = _apply_ffn_chunking(net, chunk_size=ffn_chunk)
        if rank == 0 and ffn_chunk > 0:
            print(f"[TestDDP] ffn_chunk={ffn_chunk} patched_ffn={patched_ffn}")

        net = net.to(device)
        wrap_ddp = bool(int(os.environ.get("MOCEIR_TEST_WRAP_DDP", "0")))
        if ddp_enabled and wrap_ddp:
            net = DDP(net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        net.eval()

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

            if rank == 0:
                print("--------> Testing on", de, "testset.")
                print("\n")

            m = globals()[f"run_{de}"](opt, device, net, dataset, ddp_enabled, rank, world_size, factor=8)
            if rank == 0:
                if not hasattr(opt, "_metrics"):
                    opt._metrics = []
                opt._metrics.append({
                    "benchmark": str(de),
                    **(m or {}),
                })

        if rank == 0:
            # 优先使用环境变量，其次使用配置中的 test_dir，最后使用默认值
            result_root = os.environ.get("MOCEIR_TEST_RESULT_DIR", None)
            if result_root is None:
                result_root = getattr(opt, "test_dir", "test")
            result_root_path = pathlib.Path(str(result_root)).expanduser()
            if not result_root_path.is_absolute():
                project_dir = pathlib.Path(__file__).resolve().parent
                result_root_path = (project_dir / result_root_path).resolve()
            result_root_path.mkdir(parents=True, exist_ok=True)

            ckpt_str = str(ckpt_path)
            ckpt_name = ckpt_path.stem
            ckpt_parent = ckpt_path.parent.name
            run_tag = str(getattr(opt, "checkpoint_id", "test")).replace("/", "_")
            out_dir = result_root_path / run_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            payload = {
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

    except Exception as e:
        r = os.environ.get("RANK")
        lr = os.environ.get("LOCAL_RANK")
        print(f"[Test][Error] rank={r} local_rank={lr} err={e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if ddp_enabled and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def depth_type(value):
    try:
        return int(value)
    except ValueError:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--recompute-activations", action="store_true")
    parser.add_argument("--ffn-chunk", type=int, default=0)
    args, _unknown = parser.parse_known_args()

    precision = "no"
    if bool(getattr(args, "bf16", False)):
        precision = "bf16"
    if bool(getattr(args, "fp16", False)):
        precision = "fp16"

    train_opt = argparse.Namespace(
        ckpt_path="/media/wsqlab/more/lqj/models/MoCE_IR_S-2026_01_29_23_33_45/checkpoints/best_psnr_ssim-epoch=0-psnr=13.747-ssim=0.1999.ckpt",
        model=None,
        data_file_dir="/media/wsqlab/more/lqj/data/open_dataset_8_1_1",
        trainset="standard",
        benchmarks=["gopro"],
        de_type=["deblur"],
        patch_size=256,
        batch_size=1,
        save_results=False,
        full_res_eval=True,
        precision=precision,
        recompute_activations=bool(getattr(args, "recompute_activations", False)),
        ffn_chunk=int(getattr(args, "ffn_chunk", 0)),
    )

    main(train_opt)

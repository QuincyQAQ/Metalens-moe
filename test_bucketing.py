import os
import pathlib
import argparse
import importlib
import importlib.util
import glob
import warnings

import numpy as np
from tqdm import tqdm
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import ToTensor

from accelerate import Accelerator
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from PIL import Image

from utils.test_utils import save_img
from utils.image_utils import crop_img
from data.dataset_utils import IRBenchmarks, CDD11


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


def _fullres_pad_collate(batch):
    metas, lrs, hrs = zip(*batch)
    clean_names = [m[0] for m in metas]
    de_ids = [m[1] for m in metas]

    shapes = [(int(x.shape[-2]), int(x.shape[-1])) for x in lrs]
    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)

    lrs_pad = []
    hrs_pad = []
    for lr, hr, (h, w) in zip(lrs, hrs, shapes):
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h or pad_w:
            lr = F.pad(lr.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
            hr = F.pad(hr.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate").squeeze(0)
        lrs_pad.append(lr)
        hrs_pad.append(hr)

    return [clean_names, de_ids, shapes], torch.stack(lrs_pad, 0), torch.stack(hrs_pad, 0)


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
            return _load_module_from_file(net_file, f"net_snapshot_{net_file.stem}")
        else:
            model_name = getattr(opt, "model", None)
            if model_name:
                cand = net_snapshot_dir / f"{model_name}.py"
                if cand.exists():
                    return _load_module_from_file(cand, f"net_snapshot_{model_name}")
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


def _infer_path_for_index(dataset, idx: int) -> str:
    lr = getattr(dataset, "lr", None)
    if isinstance(lr, list) and len(lr) > 0:
        item = lr[idx]
        if isinstance(item, dict) and "img" in item:
            return str(item["img"])
        if isinstance(item, str):
            return str(item)

    degraded_dict = getattr(dataset, "degraded_dict", None)
    subset = getattr(dataset, "subset", None)
    dataset_split = getattr(dataset, "dataset_split", None)
    if isinstance(degraded_dict, dict) and dataset_split != "train":
        key = subset
        if key in degraded_dict:
            return str(degraded_dict[key][idx])

    raise RuntimeError("Unable to infer file path for dataset index")


def _get_hw_fast(path: str) -> Tuple[int, int]:
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return int(h), int(w)
    except Exception:
        pass

    img = Image.open(path)
    w, h = img.size
    return int(h), int(w)


class SortedSizeBatchSampler(Sampler[List[int]]):
    def __init__(self, indices: List[int], batch_size: int, drop_last: bool = False):
        self.indices = indices
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        n = len(self.indices)
        bs = self.batch_size
        for start in range(0, n, bs):
            batch = self.indices[start:start + bs]
            if len(batch) < bs and self.drop_last:
                break
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def _build_sorted_indices_by_hw(dataset) -> List[int]:
    pairs = []
    for idx in range(len(dataset)):
        path = _infer_path_for_index(dataset, idx)
        h, w = _get_hw_fast(path)
        # approximate crop_img(base=16): shrink to nearest multiple of 16
        h = (h // 16) * 16
        w = (w // 16) * 16
        pairs.append((h * w, h, w, idx))

    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    return [p[-1] for p in pairs]


def run_test(opts, accelerator: Accelerator, net, dataset):
    batch_size = int(getattr(opts, "batch_size", 1))
    full_res = bool(getattr(opts, "full_res_eval", False))

    if not full_res:
        testloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    else:
        if batch_size <= 1:
            testloader = DataLoader(
                dataset,
                batch_size=1,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
        else:
            sorted_indices = _build_sorted_indices_by_hw(dataset)
            batch_sampler = SortedSizeBatchSampler(sorted_indices, batch_size=batch_size, drop_last=False)
            testloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                pin_memory=True,
                num_workers=0,
                collate_fn=_fullres_pad_collate,
            )

    testloader = accelerator.prepare(testloader)

    if getattr(opts, "save_results", False):
        out_dir = pathlib.Path(os.path.join(
            os.getcwd(),
            f"results/{opts.checkpoint_id}/{opts.benchmarks[0]}/rank{accelerator.process_index}",
        ))
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    calc_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg",
        normalize=True,
        reduction="none",
    ).to(accelerator.device)

    psnr_sum_local = 0.0
    ssim_sum_local = 0.0
    lpips_sum_local = torch.zeros((), device=accelerator.device)
    count_local = 0

    with torch.no_grad():
        for meta, degrad, clean in tqdm(testloader, disable=not accelerator.is_main_process):
            if full_res and batch_size > 1 and isinstance(meta, (list, tuple)) and len(meta) == 3:
                clean_name, de_id, shapes = meta
            else:
                clean_name, de_id = meta
                shapes = None

            if torch.is_tensor(degrad) and degrad.device != accelerator.device:
                degrad = degrad.to(accelerator.device, non_blocking=True)
            if torch.is_tensor(clean) and clean.device != accelerator.device:
                clean = clean.to(accelerator.device, non_blocking=True)

            de_id = _normalize_de_id(de_id, device=accelerator.device, opt=opts)

            restored = _forward_model(net, degrad, de_id)
            if isinstance(restored, (list, tuple)) and len(restored) == 2:
                restored, _ = restored

            restored = torch.clamp(restored, 0, 1)

            restored_np = restored.detach().cpu().permute(0, 2, 3, 1).numpy()
            clean_np = clean.detach().cpu().permute(0, 2, 3, 1).numpy()
            bs = int(restored_np.shape[0])

            if shapes is None:
                lpips_vals = calc_lpips(clean, restored).detach().float()
                lpips_sum_local = lpips_sum_local + lpips_vals.sum() if lpips_vals.numel() > 1 else lpips_vals.reshape(()).sum()
                for i in range(bs):
                    ssim_sum_local += float(structural_similarity(clean_np[i], restored_np[i], channel_axis=2, gaussian_weights=True, data_range=1.0, full=False))
                    psnr_sum_local += float(peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1))
                    if out_dir is not None:
                        psnr_temp = float(peak_signal_noise_ratio(clean_np[i], restored_np[i], data_range=1))
                        save_name = os.path.splitext(os.path.split(clean_name[i])[-1])[0] + '_' + str(round(psnr_temp, 2)) + '.png'
                        save_img(str(out_dir / save_name), (restored_np[i] * 255.0).round().clip(0, 255).astype(np.uint8))
            else:
                for i, (h, w) in enumerate(shapes):
                    clean_i = clean_np[i][:h, :w, :]
                    restored_i = restored_np[i][:h, :w, :]
                    ssim_sum_local += float(structural_similarity(clean_i, restored_i, channel_axis=2, gaussian_weights=True, data_range=1.0, full=False))
                    psnr_sum_local += float(peak_signal_noise_ratio(clean_i, restored_i, data_range=1))

                    c_i = clean[i:i + 1, :, :h, :w]
                    r_i = restored[i:i + 1, :, :h, :w]
                    lpips_sum_local = lpips_sum_local + calc_lpips(c_i, r_i).detach().float().sum()

                    if out_dir is not None:
                        psnr_temp = float(peak_signal_noise_ratio(clean_i, restored_i, data_range=1))
                        save_name = os.path.splitext(os.path.split(clean_name[i])[-1])[0] + '_' + str(round(psnr_temp, 2)) + '.png'
                        save_img(str(out_dir / save_name), (restored_i * 255.0).round().clip(0, 255).astype(np.uint8))

            count_local += bs

    psnr_sum = accelerator.reduce(torch.tensor(psnr_sum_local, device=accelerator.device), reduction="sum")
    ssim_sum = accelerator.reduce(torch.tensor(ssim_sum_local, device=accelerator.device), reduction="sum")
    lpips_sum = accelerator.reduce(lpips_sum_local, reduction="sum")
    total_count = accelerator.reduce(torch.tensor(count_local, device=accelerator.device, dtype=torch.long), reduction="sum")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        denom = float(total_count.detach().cpu()) if int(total_count.detach().cpu()) > 0 else 1.0
        print("PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n".format(
            float(psnr_sum.detach().cpu()) / denom,
            float(ssim_sum.detach().cpu()) / denom,
            float(lpips_sum.detach().cpu()) / denom,
        ))


def run_gopro(opts, accelerator: Accelerator, net, dataset):
    run_test(opts, accelerator, net, dataset)


def run_drmi(opts, accelerator: Accelerator, net, dataset):
    run_test(opts, accelerator, net, dataset)


def run_cdd11(opts, accelerator: Accelerator, net, dataset):
    run_test(opts, accelerator, net, dataset)


def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    accelerator = Accelerator()

    ckpt_path = _resolve_ckpt_path(opt)
    if getattr(opt, "checkpoint_id", None) is None:
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
        opt.checkpoint_id = f"{run_dir.name}-{ckpt_path.stem}"

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

    net = accelerator.prepare(net)
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

        fn = globals().get(f"run_{de}")
        if fn is None:
            fn = run_test
        fn(opt, accelerator, net, dataset)


if __name__ == "__main__":
    train_opt = argparse.Namespace(
        ckpt_path="",
        model=None,
        data_file_dir="",
        trainset="standard",
        benchmarks=["gopro"],
        de_type=["deblur"],
        patch_size=256,
        batch_size=4,
        save_results=True,
        full_res_eval=True,
    )
    main(train_opt)

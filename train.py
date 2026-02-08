from typing import List

import os
import pathlib
import importlib
import importlib.util
import shutil
import csv
import json
import warnings
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


# 全局关闭 torch.load(weights_only=False) 的冗长安全提示（Lightning / torchmetrics 内部会触发）。
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
)


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
    model.load_state_dict(state_dict, strict=False)


def _evaluate_irbenchmarks(net: nn.Module, data_loader: DataLoader, device: torch.device) -> dict:
    net.eval()
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction="mean").to(device)

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    with torch.no_grad():
        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(data_loader, leave=False):
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
    print("Options")
    print(opt)
    run_id = os.environ.get("MOCEIRV2_RUN_ID")
    if run_id is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        model_name = getattr(opt, "model", "model")
        model_name = str(model_name).replace(os.sep, "_").replace(" ", "_")
        run_id = f"{model_name}-{timestamp}"
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
        if accelerator.is_main_process:
            print(f"[Train] Epoch {int(epoch) + 1}/{int(max_epochs)}")
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

        pbar = tqdm(
            trainloader,
            disable=not accelerator.is_main_process,
            desc=f"Epoch {int(epoch) + 1}/{int(max_epochs)}",
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
                pbar.set_postfix(loss=float(loss_det))
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
                "Train_Loss_epoch": float(loss_epoch),
                "Balance_epoch": float(balance_epoch),
                "val_psnr": None,
                "val_ssim": None,
                "val_lpips": None,
                "test_psnr": None,
                "test_ssim": None,
                "test_lpips": None,
                "lr": float(optimizer.param_groups[0]["lr"]),
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
                    "Train_Loss_epoch": float(loss_epoch),
                    "Balance_epoch": float(balance_epoch),
                    "val_psnr": val_metrics.get("psnr"),
                    "val_ssim": val_metrics.get("ssim"),
                    "val_lpips": val_metrics.get("lpips"),
                    "test_psnr": None,
                    "test_ssim": None,
                    "test_lpips": None,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                })

                psnr_val = val_metrics.get("psnr")
                ssim_val = val_metrics.get("ssim")
                if psnr_val is not None:
                    if best_val_psnr is None or float(psnr_val) > float(best_val_psnr):
                        best_val_psnr = float(psnr_val)
                        ckpt_name = f"best_psnr-epoch={int(epoch)}-psnr={float(psnr_val):.3f}-ssim={float(ssim_val) if ssim_val is not None else 0.0:.4f}.ckpt"
                        torch.save({"state_dict": eval_net.state_dict()}, str(checkpoint_path / ckpt_name))

                if psnr_val is not None and ssim_val is not None:
                    if best_joint_psnr is None or best_joint_ssim is None:
                        improved_joint = True
                    else:
                        improved_joint = (float(psnr_val) >= float(best_joint_psnr)) and (float(ssim_val) >= float(best_joint_ssim))
                    if improved_joint:
                        best_joint_psnr = float(psnr_val)
                        best_joint_ssim = float(ssim_val)
                        ckpt_name = f"best_psnr_ssim-epoch={int(epoch)}-psnr={float(psnr_val):.3f}-ssim={float(ssim_val):.4f}.ckpt"
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
                "test_psnr": test_metrics.get("psnr"),
                "test_ssim": test_metrics.get("ssim"),
                "test_lpips": test_metrics.get("lpips"),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
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




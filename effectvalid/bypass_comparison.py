"""
对照实验：验证 Freq_LKA 模块是否引入方向性伪影（十字条纹）和边缘效应

对比：
1. with Freq_LKA（正常前向）
2. bypass Freq_LKA（直接返回输入 = identity）

如果 bypass 后十字/边界条纹显著减弱，则证明是模块引入了伪影。
"""

import os
import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 导入模型
from net.MoCE_IR_S_Freq_LKA import MoCEIR_Freq_LKA, Freq_LKA_Expert, Freq_LKA_AdapterLayer


# ==================== FFT 工具 ====================

def fft2_mag(x: torch.Tensor) -> torch.Tensor:
    """返回幅度谱"""
    if x.dim() == 3:
        X = torch.fft.fft2(x, norm="ortho")
        X = torch.fft.fftshift(X, dim=(-2, -1))
        mag = X.abs()
    elif x.dim() == 2:
        X = torch.fft.fft2(x, norm="ortho")
        X = torch.fft.fftshift(X)
        mag = X.abs()
    return mag


# ==================== 模型加载 ====================

def safe_torch_load(ckpt_path: str, map_location="cpu"):
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        warnings.filterwarnings("ignore", category=FutureWarning)
        return torch.load(ckpt_path, map_location=map_location)


def load_model(checkpoint_path: str, device: str = "cuda:0") -> nn.Module:
    model = MoCEIR_Freq_LKA()
    state = safe_torch_load(checkpoint_path, map_location="cpu")
    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_image(img_path: str, img_size: int = 256, device: str = "cuda:0") -> torch.Tensor:
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return x


# ==================== 找目标模块 ====================

def find_first_module(model: nn.Module, target_class_name: str):
    for m in model.modules():
        if type(m).__name__ == target_class_name:
            return m
    return None


def capture_freq_io(model: nn.Module, img: torch.Tensor, target_class: str = "Freq_LKA_AdapterLayer"):
    """抓取 Freq_LKA 模块的输入输出"""
    target = find_first_module(model, target_class)
    target_type = target_class
    
    if target is None:
        # 退回找 Expert
        for m in model.modules():
            if isinstance(m, Freq_LKA_Expert):
                target = m
                break
        target_type = "Freq_LKA_Expert"
    
    if target is None:
        raise RuntimeError(f"Cannot find {target_class} or Freq_LKA_Expert")
    
    pack = {"x_in": None, "phys": None, "x_out": None}
    
    def hook(module, inp, out):
        if inp and len(inp) >= 1 and torch.is_tensor(inp[0]):
            pack["x_in"] = inp[0].detach()
        if inp and len(inp) >= 2 and torch.is_tensor(inp[1]):
            pack["phys"] = inp[1].detach()
        
        y = out
        if isinstance(y, (tuple, list)):
            y = y[0]
        if torch.is_tensor(y):
            pack["x_out"] = y.detach()
    
    h = target.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(img)
    h.remove()
    
    if pack["x_in"] is None or pack["x_out"] is None:
        raise RuntimeError(f"Hook failed to capture x_in/x_out from {target_type}")
    
    return target, target_type, pack["x_in"], pack["phys"], pack["x_out"]


# ==================== Bypass 模式设置 ====================

class BypassWrapper:
    """包装模块使其 bypass（直接返回输入）"""
    def __init__(self, module):
        self.module = module
        self.original_forward = module.forward
    
    def bypass_forward(self, x: torch.Tensor, phys_emb: torch.Tensor = None) -> torch.Tensor:
        """直接返回输入，相当于 identity"""
        return x
    
    def enable_bypass(self):
        self.module.forward = self.bypass_forward
    
    def disable_bypass(self):
        self.module.forward = self.original_forward


def setup_bypass_mode(model: nn.Module, target_class: str = "Freq_LKA_AdapterLayer"):
    """找到目标模块并包装，使其可以切换 bypass 模式"""
    target = find_first_module(model, target_class)
    if target is None:
        for m in model.modules():
            if isinstance(m, Freq_LKA_Expert):
                target = m
                break
    
    if target is None:
        raise RuntimeError("Cannot find Freq_LKA module")
    
    wrapper = BypassWrapper(target)
    return wrapper, target


# ==================== 可视化函数 ====================

def plot_spatial_delta_comparison(x_in, x_out_with, x_out_bypass, save_path="spatial_delta_comparison.png"):
    """对比 spatial delta：with vs bypass"""
    x_in = x_in.detach().cpu()
    x_out_with = x_out_with.detach().cpu()
    x_out_bypass = x_out_bypass.detach().cpu()
    
    def normalize(t):
        t = t - t.min()
        t = t / (t.max() + 1e-8)
        return t
    
    # 取第一个通道
    ch = 0
    inp = normalize(x_in[0, ch])
    out_with = normalize(x_out_with[0, ch])
    out_bypass = normalize(x_out_bypass[0, ch])
    
    delta_with = (out_with - inp).abs()
    delta_bypass = (out_bypass - inp).abs()
    
    # 归一化 delta 到相同范围便于对比
    vmax = max(delta_with.max(), delta_bypass.max())
    delta_with = delta_with / (vmax + 1e-8)
    delta_bypass = delta_bypass / (vmax + 1e-8)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 第一行：with Freq_LKA
    axes[0, 0].imshow(inp, cmap="viridis")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(out_with, cmap="viridis")
    axes[0, 1].set_title("Output (with Freq_LKA)")
    axes[0, 1].axis("off")
    
    im = axes[0, 2].imshow(delta_with, cmap="hot")
    axes[0, 2].set_title("|Out-In| (with Freq_LKA)")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # 第二行：bypass
    axes[1, 0].imshow(inp, cmap="viridis")
    axes[1, 0].set_title("Input (same)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(out_bypass, cmap="viridis")
    axes[1, 1].set_title("Output (bypass)")
    axes[1, 1].axis("off")
    
    im = axes[1, 2].imshow(delta_bypass, cmap="hot")
    axes[1, 2].set_title("|Out-In| (bypass)")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Compare] Saved spatial delta comparison to {save_path}")
    
    # 打印统计
    print(f"[Compare] Delta mean: with={delta_with.mean():.6f}, bypass={delta_bypass.mean():.6f}")
    print(f"[Compare] Delta max:  with={delta_with.max():.6f}, bypass={delta_bypass.max():.6f}")


def plot_spectrum_comparison(x_in, x_out_with, x_out_bypass, save_path="spectrum_comparison.png"):
    """对比频谱：with vs bypass"""
    x_in = x_in.detach().cpu()
    x_out_with = x_out_with.detach().cpu()
    x_out_bypass = x_out_bypass.detach().cpu()
    
    ch = 0
    mag_in = fft2_mag(x_in[0, ch]).numpy()
    mag_with = fft2_mag(x_out_with[0, ch]).numpy()
    mag_bypass = fft2_mag(x_out_bypass[0, ch]).numpy()
    
    def norm(mag):
        ref_max = mag_in.max()
        mag = mag / (ref_max + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag
    
    mag_in_n = norm(mag_in)
    mag_with_n = norm(mag_with)
    mag_bypass_n = norm(mag_bypass)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # 第一行：with Freq_LKA
    axes[0, 0].imshow(mag_in_n, cmap="viridis")
    axes[0, 0].set_title("Input Spectrum")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(mag_with_n, cmap="viridis")
    axes[0, 1].set_title("Output Spectrum (with Freq_LKA)")
    axes[0, 1].axis("off")
    
    diff_with = mag_with_n - mag_in_n
    im = axes[0, 2].imshow(diff_with, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title("Delta Spectrum (with - input)")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # 第二行：bypass
    axes[1, 0].imshow(mag_in_n, cmap="viridis")
    axes[1, 0].set_title("Input Spectrum (same)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(mag_bypass_n, cmap="viridis")
    axes[1, 1].set_title("Output Spectrum (bypass)")
    axes[1, 1].axis("off")
    
    diff_bypass = mag_bypass_n - mag_in_n
    im = axes[1, 2].imshow(diff_bypass, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title("Delta Spectrum (bypass - input)")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Compare] Saved spectrum comparison to {save_path}")
    
    # 打印统计
    print(f"[Compare] Spectrum delta mean: with={diff_with.mean():.6f}, bypass={diff_bypass.mean():.6f}")


def plot_side_by_side_spectrum(x_out_with, x_out_bypass, save_path="spectrum_side_by_side.png"):
    """并排对比频谱，更清晰显示十字伪影"""
    x_out_with = x_out_with.detach().cpu()
    x_out_bypass = x_out_bypass.detach().cpu()
    
    ch = 0
    mag_with = fft2_mag(x_out_with[0, ch]).numpy()
    mag_bypass = fft2_mag(x_out_bypass[0, ch]).numpy()
    
    # 归一化到相同范围
    vmax = max(mag_with.max(), mag_bypass.max())
    
    def norm(mag):
        mag = mag / (vmax + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag
    
    mag_with_n = norm(mag_with)
    mag_bypass_n = norm(mag_bypass)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(mag_with_n, cmap="viridis")
    axes[0].set_title("With Freq_LKA\n(Check for cross/artifact)", fontsize=12)
    axes[0].axis("off")
    
    axes[1].imshow(mag_bypass_n, cmap="viridis")
    axes[1].set_title("Bypass Freq_LKA\n(Should be cleaner)", fontsize=12)
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Compare] Saved side-by-side spectrum to {save_path}")


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Bypass 对照实验：验证 Freq_LKA 引入的伪影")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--img", type=str, required=True, help="Test image path")
    parser.add_argument("--out_dir", type=str, default="effectvalid_out", help="Output directory")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. 加载模型和图像
    print("Loading model and image...")
    model = load_model(args.ckpt, device=device)
    img = load_image(args.img, img_size=args.img_size, device=device)
    
    # 2. 设置 bypass 包装器
    print("Setting up bypass wrapper...")
    wrapper, target = setup_bypass_mode(model, target_class="Freq_LKA_AdapterLayer")
    target_name = type(target).__name__
    print(f"Target module: {target_name}")
    
    # 3. 正常前向（with Freq_LKA）
    print("\n=== Running with Freq_LKA ===")
    wrapper.disable_bypass()
    _, _, x_in, phys_real, x_out_with = capture_freq_io(model, img, target_class="Freq_LKA_AdapterLayer")
    print(f"Captured: x_in shape={x_in.shape}, x_out shape={x_out_with.shape}")
    if phys_real is not None:
        print(f"phys_emb: shape={phys_real.shape}, mean={phys_real.mean():.4f}")
    
    # 4. Bypass 前向
    print("\n=== Running bypass (identity) ===")
    wrapper.enable_bypass()
    # 需要重新 capture，因为 forward 行为变了
    pack_bypass = {"x_in": None, "x_out": None}
    
    def hook_bypass(module, inp, out):
        if inp and len(inp) >= 1 and torch.is_tensor(inp[0]):
            pack_bypass["x_in"] = inp[0].detach()
        y = out
        if isinstance(y, (tuple, list)):
            y = y[0]
        if torch.is_tensor(y):
            pack_bypass["x_out"] = y.detach()
    
    h = target.register_forward_hook(hook_bypass)
    with torch.no_grad():
        _ = model(img)
    h.remove()
    
    x_out_bypass = pack_bypass["x_out"]
    print(f"Bypass output shape: {x_out_bypass.shape}")
    
    # 恢复模型
    wrapper.disable_bypass()
    
    # 5. 对齐形状
    x_in_vis = x_in.detach()
    x_out_with_vis = x_out_with.detach()
    x_out_bypass_vis = x_out_bypass.detach()
    
    if x_out_with_vis.shape[2:] != x_in_vis.shape[2:]:
        x_out_with_vis = F.interpolate(x_out_with_vis, size=x_in_vis.shape[2:], mode="bilinear", align_corners=False)
        x_out_bypass_vis = F.interpolate(x_out_bypass_vis, size=x_in_vis.shape[2:], mode="bilinear", align_corners=False)
    
    if x_out_with_vis.shape[1] != x_in_vis.shape[1]:
        Cmin = min(x_out_with_vis.shape[1], x_in_vis.shape[1])
        x_out_with_vis = x_out_with_vis[:, :Cmin]
        x_out_bypass_vis = x_out_bypass_vis[:, :Cmin]
        x_in_vis = x_in_vis[:, :Cmin]
    
    # 6. 生成对比图
    print("\n=== Generating comparison plots ===")
    
    # Spatial delta 对比
    spatial_path = os.path.join(args.out_dir, "bypass_spatial_delta_comparison.png")
    plot_spatial_delta_comparison(x_in_vis, x_out_with_vis, x_out_bypass_vis, save_path=spatial_path)
    
    # Spectrum 对比（详细版）
    spectrum_path = os.path.join(args.out_dir, "bypass_spectrum_comparison.png")
    plot_spectrum_comparison(x_in_vis, x_out_with_vis, x_out_bypass_vis, save_path=spectrum_path)
    
    # Spectrum 并排对比
    side_by_side_path = os.path.join(args.out_dir, "bypass_spectrum_side_by_side.png")
    plot_side_by_side_spectrum(x_out_with_vis, x_out_bypass_vis, save_path=side_by_side_path)
    
    print("\n=== DONE ===")
    print(f"Output files:")
    print(f"  1. {spatial_path}")
    print(f"  2. {spectrum_path}")
    print(f"  3. {side_by_side_path}")


if __name__ == "__main__":
    main()


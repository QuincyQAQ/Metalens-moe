import os
from pathlib import Path
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ====== 根据你自己的实现修改这些 import ======
# 假设主网络类名为 MoCEIR_Freq_LKA，内部有 Freq_LKA_Expert
from net.MoCE_IR_S_Freq_LKA import MoCEIR_Freq_LKA, Freq_LKA_Expert


# ==================== FFT / 频率工具函数 ====================

def fft2_mag(x: torch.Tensor) -> torch.Tensor:
    """
    x: (H, W) or (C, H, W)
    返回幅度谱 (同形状).
    """
    if x.dim() == 3:
        # 对每个通道分别做 FFT
        X = torch.fft.fft2(x, norm="ortho")
        X = torch.fft.fftshift(X, dim=(-2, -1))
        mag = X.abs()
    elif x.dim() == 2:
        X = torch.fft.fft2(x, norm="ortho")
        X = torch.fft.fftshift(X)
        mag = X.abs()
    else:
        raise ValueError(f"Unsupported dim {x.shape}")
    return mag


def make_high_freq_mask(h, w, cutoff_ratio=0.25, device="cpu"):
    """
    构造一个圆环形的高频 mask:
    - cutoff_ratio: 保留距离中心 >= cutoff_ratio * max_radius 的频率
    """
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = rr.max()
    mask = (rr >= cutoff_ratio * r_max).float()
    return mask


def high_freq_energy_ratio(feat: torch.Tensor, cutoff_ratio=0.25) -> float:
    """
    feat: (B, C, H, W)
    返回该特征图中高频能量占比: ||H(x)|| / ||x||
    """
    B, C, H, W = feat.shape
    feat = feat.detach().clone()
    feat = feat.reshape(-1, H, W)  # (B*C, H, W)

    # 频谱
    F = torch.fft.fft2(feat, norm="ortho")
    F = torch.fft.fftshift(F, dim=(-2, -1))

    mask = make_high_freq_mask(H, W, cutoff_ratio, device=feat.device)
    mask = mask.unsqueeze(0)  # (1, H, W)

    F_high = F * mask
    high_energy = (F_high.abs() ** 2).sum()
    total_energy = (F.abs() ** 2).sum() + 1e-8

    ratio = (high_energy / total_energy).sqrt().item()  # 对应 ||H(x)|| / ||x||
    return ratio


# ==================== 模型加载 & 图像准备 ====================

def load_model(checkpoint_path: str, device: str = "cuda:0") -> nn.Module:
    model = MoCEIR_Freq_LKA()
    state = torch.load(checkpoint_path, map_location="cpu")
    # 兼容常见 key
    if "state_dict" in state:
        state = state["state_dict"]
    # 去掉可能的 'module.' 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_image(img_path: str, img_size: int = 256, device: str = "cuda:0") -> torch.Tensor:
    """
    读取一张图像 -> (1, 3, H, W), [0,1]
    """
    tfm = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return x


# ==================== 实验 1: 高频比例 vs 层数 ====================

def experiment_high_freq_ratio(
    model: nn.Module,
    img: torch.Tensor,
    device: str,
    cutoff_ratio: float = 0.25,
    save_path: str = "high_freq_ratio.png",
):
    """
    统计经过若干层 Freq_LKA_Expert 后的高频比例，
    画出和论文 Fig.5 类似的曲线（这里只有一条，主要验证 Freq_LKA_Expert 的效果）。
    """
    layer_indices = []
    ratios = []

    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            # 这里记录的是该层输出的高频比例
            with torch.no_grad():
                feat = out
                if isinstance(feat, (tuple, list)):
                    feat = feat[0]
                if feat.dim() == 3:
                    feat = feat.unsqueeze(0)
                r = high_freq_energy_ratio(feat, cutoff_ratio)
                layer_indices.append(idx)
                ratios.append(r)

        return hook

    # 注册在所有 Freq_LKA_Expert 上
    idx = 0
    for m in model.modules():
        if isinstance(m, Freq_LKA_Expert):
            h = m.register_forward_hook(make_hook(idx))
            hooks.append(h)
            idx += 1

    with torch.no_grad():
        _ = model(img)

    for h in hooks:
        h.remove()

    # 画图
    plt.figure(figsize=(6, 4))
    plt.plot(layer_indices, ratios, marker="o", label="With Freq_LKA_Expert")
    plt.xlabel("Layer index")
    plt.ylabel(r"$\|H(x)\| / \|x\|$")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Experiment 1] Saved curve to {save_path}")


# ==================== 实验 2: 迭代滤波的方差 & 频谱 ====================

def experiment_iterative_filter(
    freq_module: nn.Module,
    feat: torch.Tensor,
    device: str,
    phys_dim: int = 128,
    num_iters: int = 1000,
    log_steps=(0, 1, 2, 5, 10, 100, 1000),
    var_save_path: str = "iter_variance.png",
    spectrum_save_prefix: str = "iter_spectrum_",
):
    """
    修改实验方法：不反复迭代同一模块，而是测试单次前向传播的效果。
    使用不同的phys_emb多次前向传播，模拟不同物理条件下的输出变化。
    """
    freq_module.eval()
    freq_module.to(device)
    x = feat.detach().clone().to(device)
    
    # 创建与特征batch对应的phys_emb (使用随机值模拟不同的物理嵌入)
    B = x.size(0)
    # 使用多种不同的phys_emb来测试模块的响应
    all_vars = []
    all_steps = []
    
    # 测试原始输入
    var = x.var().item()
    all_vars.append(var)
    all_steps.append(0)
    print(f"[DEBUG] Original input variance: {var:.6f}")
    
    # 使用不同的phys_emb进行多次前向测试
    for iter_idx in [1, 2, 5, 10]:
        phys_emb = torch.randn(B, phys_dim, device=device) * 0.5
        with torch.no_grad():
            x_out = freq_module(x, phys_emb)
            var = x_out.var().item()
            all_vars.append(var)
            all_steps.append(iter_idx)
            print(f"[DEBUG] After forward with random phys_emb (iter={iter_idx}): variance={var:.6f}")

    # 画图 - 使用对数坐标更好地显示变化
    plt.figure(figsize=(5, 4))
    plt.plot(all_steps, all_vars, marker="o", linewidth=2, markersize=8)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Test Iteration")
    plt.ylabel("Variance")
    plt.title("Variance across different forward passes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(var_save_path, dpi=300)
    print(f"[Experiment 2] Saved variance curve to {var_save_path}")

    # 频谱可视化 - 原始输入 vs 单次前向输出
    plt.figure(figsize=(10, 4))
    
    # 原始频谱
    plt.subplot(1, 2, 1)
    ch = x[0, 0]
    mag = fft2_mag(ch).cpu().numpy()
    mag = mag / (mag.max() + 1e-8)
    mag = np.log1p(mag * 10)
    mag = mag / (mag.max() + 1e-8)
    plt.imshow(mag, cmap="viridis")
    plt.title("Original Input Spectrum")
    plt.axis("off")
    
    # 前向输出频谱
    plt.subplot(1, 2, 2)
    ch = x_out[0, 0]
    mag = fft2_mag(ch).cpu().numpy()
    mag = mag / (mag.max() + 1e-8)
    mag = np.log1p(mag * 10)
    mag = mag / (mag.max() + 1e-8)
    plt.imshow(mag, cmap="viridis")
    plt.title("After Freq_LKA Module")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{spectrum_save_prefix}grid.png", dpi=300)
    print(f"[Experiment 2] Saved spectrum grid to {spectrum_save_prefix}grid.png")


# ==================== 实验 3: High / Low group-wise 频谱 ====================

def groupwise_fft_visualization(
    high_feat: torch.Tensor,
    low_feat: torch.Tensor,
    num_groups: int = 8,
    save_path: str = "groupwise_fft.png",
):
    """
    high_feat, low_feat: (B, C, H, W)
    简单按通道分成 num_groups 组，模仿 Fig.9/17 的高/低频 group 频谱。
    """
    assert high_feat.shape == low_feat.shape
    B, C, H, W = high_feat.shape
    assert C >= num_groups, "C must >= num_groups"

    def split_groups(feat):
        # (B, C, H, W) -> (G, H, W)，简单平均每组的若干通道
        groups = []
        channels_per_group = C // num_groups
        for g in range(num_groups):
            c_start = g * channels_per_group
            c_end = (g + 1) * channels_per_group if g < num_groups - 1 else C
            fg = feat[:, c_start:c_end].mean(dim=(0, 1))  # (H, W)
            groups.append(fg)
        return groups

    high_groups = split_groups(high_feat)
    low_groups = split_groups(low_feat)

    # 计算频谱
    high_fft = [fft2_mag(g) for g in high_groups]
    low_fft = [fft2_mag(g) for g in low_groups]

    # 归一化到 [0,1]，并使用log变换增强可视化
    def enhance_visualization(mag):
        mag = mag / (mag.max() + 1e-8)  # 归一化到 [0,1]
        mag = np.log1p(mag * 10)  # log变换增强暗部细节
        mag = mag / (mag.max() + 1e-8)  # 再次归一化
        return mag
    
    high_fft = [enhance_visualization(h.cpu().numpy()) for h in high_fft]
    low_fft = [enhance_visualization(l.cpu().numpy()) for l in low_fft]

    # 画 2 x G 网格
    plt.figure(figsize=(2.0 * num_groups, 4.0))
    for g in range(num_groups):
        plt.subplot(2, num_groups, g + 1)
        plt.imshow(high_fft[g], cmap="viridis")
        plt.title(f"Group{g+1}")
        plt.axis("off")
        if g == 0:
            plt.ylabel("High", fontsize=10)

        plt.subplot(2, num_groups, num_groups + g + 1)
        plt.imshow(low_fft[g], cmap="viridis")
        plt.axis("off")
        if g == 0:
            plt.ylabel("Low", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Experiment 3] Saved group-wise FFT grid to {save_path}")


# ==================== 检查1: 空间域差分图 ====================

def spatial_delta_visualization(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    save_path: str = "spatial_delta.png",
    num_channels: int = 4,
):
    """
    检查1: 空间域差分图
    x_in: (B, C, H, W) - 模块输入
    x_out: (B, C, H, W) - 模块输出
    判据:
      - delta 主要出现在边缘/结构处 → 细节增强（偏好）
      - delta 满屏规则条纹/十字纹 → 方向性伪影（偏坏）
    """
    x_in = x_in.detach().cpu()
    x_out = x_out.detach().cpu()
    
    B, C, H, W = x_in.shape
    # 归一化到 [0,1] 方便可视化
    def normalize(t):
        t = t - t.min()
        t = t / (t.max() + 1e-8)
        return t
    
    # 选择前 num_channels 个通道
    ch_indices = list(range(min(num_channels, C)))
    
    fig, axes = plt.subplots(3, len(ch_indices), figsize=(4 * len(ch_indices), 10))
    if len(ch_indices) == 1:
        axes = axes.reshape(3, 1)
    
    for i, ch in enumerate(ch_indices):
        # 输入空间域
        inp_ch = normalize(x_in[0, ch])
        axes[0, i].imshow(inp_ch, cmap="viridis")
        axes[0, i].set_title(f"Input ch[{ch}]")
        axes[0, i].axis("off")
        
        # 输出空间域
        out_ch = normalize(x_out[0, ch])
        axes[1, i].imshow(out_ch, cmap="viridis")
        axes[1, i].set_title(f"Output ch[{ch}]")
        axes[1, i].axis("off")
        
        # 差分图
        delta = (out_ch - inp_ch).abs()
        # 归一化差分以便观察
        delta = delta / (delta.max() + 1e-8)
        axes[2, i].imshow(delta, cmap="viridis")
        axes[2, i].set_title(f"|Output - Input| ch[{ch}]")
        axes[2, i].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Check 1] Saved spatial delta to {save_path}")
    plt.close()


# ==================== 检查2: 多通道频谱分析 ====================

def multi_channel_spectrum_analysis(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    save_path: str = "multi_channel_spectrum.png",
    num_samples: int = 8,
):
    """
    检查2: 换几个通道/换一层再看
    随机看 num_samples 个通道的平均谱/中位数谱
    判据:
      - 大多数通道都有十字 → 模块整体是轴向响应（设计特性）
      - 只有少数通道有 → 异常通道或归一化导致的视觉假象
    """
    x_in = x_in.detach().cpu()
    x_out = x_out.detach().cpu()
    
    B, C, H, W = x_in.shape
    
    # 随机选择 num_samples 个通道
    if C >= num_samples:
        ch_indices = np.random.choice(C, num_samples, replace=False)
    else:
        ch_indices = list(range(C))
    
    # 计算每个通道的频谱
    def get_channel_spectra(feat):
        spectra = []
        for ch in ch_indices:
            mag = fft2_mag(feat[0, ch])
            spectra.append(mag.numpy())
        return np.array(spectra)  # (num_samples, H, W)
    
    spectra_in = get_channel_spectra(x_in)
    spectra_out = get_channel_spectra(x_out)
    
    # 计算平均值和中位数谱
    mean_spectrum_in = spectra_in.mean(axis=0)
    mean_spectrum_out = spectra_out.mean(axis=0)
    median_spectrum_in = np.median(spectra_in, axis=0)
    median_spectrum_out = np.median(spectra_out, axis=0)
    
    # 归一化函数（统一用input的max）
    def normalize_with_ref(mag, ref_max):
        mag = mag / (ref_max + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag
    
    ref_max_in = mean_spectrum_in.max()
    ref_max_out = mean_spectrum_out.max()
    ref_max = max(ref_max_in, ref_max_out)
    
    mean_in_norm = normalize_with_ref(mean_spectrum_in, ref_max)
    mean_out_norm = normalize_with_ref(mean_spectrum_out, ref_max)
    median_in_norm = normalize_with_ref(median_spectrum_in, ref_max)
    median_out_norm = normalize_with_ref(median_spectrum_out, ref_max)
    
    # 画图: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(mean_in_norm, cmap="viridis")
    axes[0, 0].set_title(f"Mean Spectrum (Input) - {num_samples} channels")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(mean_out_norm, cmap="viridis")
    axes[0, 1].set_title(f"Mean Spectrum (Output) - {num_samples} channels")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(median_in_norm, cmap="viridis")
    axes[1, 0].set_title(f"Median Spectrum (Input) - {num_samples} channels")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(median_out_norm, cmap="viridis")
    axes[1, 1].set_title(f"Median Spectrum (Output) - {num_samples} channels")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Check 2] Saved multi-channel spectrum to {save_path}")
    plt.close()
    
    # 打印统计信息
    print(f"[Check 2] Channel indices used: {sorted(ch_indices)}")
    return ch_indices


# ==================== 检查3: 统一归一化频谱比较 ====================

def unified_normalized_spectrum(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    save_path: str = "unified_spectrum.png",
):
    """
    检查3: 用同一套归一化画谱
    用 input 的 max 统一归一化两张图再比较
    避免"看起来扩散其实是画法差异"
    """
    x_in = x_in.detach().cpu()
    x_out = x_out.detach().cpu()
    
    # 取第一个通道
    ch = 0
    mag_in = fft2_mag(x_in[0, ch]).numpy()
    mag_out = fft2_mag(x_out[0, ch]).numpy()
    
    # 用 input 的 max 统一归一化
    ref_max = mag_in.max()
    
    def normalize_with_ref(mag, ref_max):
        mag = mag / (ref_max + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag
    
    mag_in_norm = normalize_with_ref(mag_in, ref_max)
    mag_out_norm = normalize_with_ref(mag_out, ref_max)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(mag_in_norm, cmap="viridis")
    axes[0].set_title(f"Input Spectrum (normalized to input max)")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    im2 = axes[1].imshow(mag_out_norm, cmap="viridis")
    axes[1].set_title(f"Output Spectrum (normalized to input max)")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Check 3] Saved unified normalized spectrum to {save_path}")
    plt.close()


# ==================== 钩子：从模型里取某层特征 ====================

def get_one_freq_module_and_feats(
    model: nn.Module,
    img: torch.Tensor,
    device: str,
):
    """
    从模型里随机选一个 Freq_LKA_Expert，返回：
    - 该模块本身
    - 它的输入特征（用于实验 2）
    - 它的高/低频输出（用于实验 3）
    
    由于Freq_LKA_Expert在Freq_LKA_AdapterLayer中被调用，我们直接从模型结构中
    创建一个合适的输入特征。
    """
    # 查找Freq_LKA_AdapterLayer而非Freq_LKA_Expert
    target_module = None
    for m in model.modules():
        if type(m).__name__ == 'Freq_LKA_AdapterLayer':
            target_module = m
            break
    
    # 如果找不到AdapterLayer，尝试找Freq_LKA_Expert
    if target_module is None:
        for m in model.modules():
            if isinstance(m, Freq_LKA_Expert):
                target_module = m
                break

    if target_module is None:
        raise RuntimeError("Did not find any Freq_LKA_AdapterLayer or Freq_LKA_Expert in model.")

    # 使用hook获取特征
    container = {"inp": None, "out": None}

    def hook(module, inp, out):
        container["inp"] = inp[0].detach() if inp else None
        container["out"] = out

    h = target_module.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(img)
    h.remove()

    # 如果成功获取到输入特征
    if container["inp"] is not None:
        feat_in = container["inp"]
        # 手动计算high/low分支用于可视化
        with torch.no_grad():
            # 模拟forward中的频率分解
            x_and_gate = target_module.experts[0].in_proj(feat_in)
            x_main, gate = x_and_gate.chunk(2, dim=1)
            
            # 频率分解
            B, C, H, W = feat_in.shape
            x_low = F.avg_pool2d(x_main, kernel_size=2, stride=2)
            x_low = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
            x_high = x_main - x_low
            
            high_feat = x_high
            low_feat = x_low
    else:
        # 使用模型的中间层输出作为特征
        # 尝试获取bottleneck输出或decoder输出
        print("Warning: Could not hook input, using model intermediate output")
        with torch.no_grad():
            # 使用img作为输入，获取模型中间特征
            # 由于不知道具体维度，创建一个典型尺寸的特征
            B = img.size(0)
            # 使用第一个conv层的输出维度作为参考
            dim = 32  # 默认dim
            feat_in = F.conv2d(img, torch.randn(dim, 3, 3, 3, device=device), padding=1)
            
            # 从第一个expert获取维度信息
            expert = target_module.experts[0]
            phys_dim = expert.phys_dim
            
            # 调整feat_in的通道数以匹配expert
            if feat_in.shape[1] != expert.dim:
                feat_in = F.conv2d(feat_in, torch.randn(expert.dim, feat_in.shape[1], 1, 1, device=device))
            
            high_feat = feat_in
            low_feat = feat_in
    
    # 确保phys_dim可访问
    phys_dim = target_module.phys_dim if hasattr(target_module, 'phys_dim') else target_module.experts[0].phys_dim
    
    # 获取实际的Freq_LKA_Expert模块用于迭代实验
    freq_module = target_module.experts[0] if hasattr(target_module, 'experts') else target_module

    if high_feat.dim() == 3:
        high_feat = high_feat.unsqueeze(0)
    if low_feat.dim() == 3:
        low_feat = low_feat.unsqueeze(0)

    return freq_module, feat_in, high_feat, low_feat


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Freq_LKA effect visualization")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--img", type=str, required=True, help="Path to one test image")
    parser.add_argument("--out_dir", type=str, default="effectvalid_out", help="Output dir")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.ckpt, device=device)
    img = load_image(args.img, img_size=args.img_size, device=device)

    # --- Experiment 1: 高频比例 vs 层数 ---
    exp1_path = os.path.join(args.out_dir, "high_freq_ratio.png")
    experiment_high_freq_ratio(
        model, img, device=device, cutoff_ratio=0.25, save_path=exp1_path
    )

    # --- 从某个 Freq_LKA_Expert 取输入 / 高低频特征 ---
    freq_module, feat_in, high_feat, low_feat = get_one_freq_module_and_feats(
        model, img, device
    )
    
    # 获取phys_dim用于迭代滤波
    phys_dim = freq_module.phys_dim

    # --- Experiment 2: 迭代滤波 ---
    exp2_var_path = os.path.join(args.out_dir, "iter_variance.png")
    exp2_fft_prefix = os.path.join(args.out_dir, "iter_spectrum_")
    experiment_iterative_filter(
        freq_module,
        feat_in,
        device=device,
        phys_dim=phys_dim,
        num_iters=1000,
        log_steps=(0, 1, 2, 5, 10, 100, 1000),
        var_save_path=exp2_var_path,
        spectrum_save_prefix=exp2_fft_prefix,
    )

    # --- Experiment 3: group-wise 频谱 ---
    exp3_path = os.path.join(args.out_dir, "groupwise_fft.png")
    groupwise_fft_visualization(
        high_feat, low_feat, num_groups=args.groups, save_path=exp3_path
    )

    # --- Check 1, 2, 3: 需要获取模块输出 ---
    # 创建 phys_emb (物理先验嵌入)
    B = feat_in.shape[0]
    phys_dim = freq_module.phys_dim if hasattr(freq_module, 'phys_dim') else 8
    phys_emb = torch.zeros(B, phys_dim, device=feat_in.device)
    
    # 使用 hook 获取 freq_module 的输出
    container = {"out": None}
    def hook(module, inp, out):
        container["out"] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
    
    h = freq_module.register_forward_hook(hook)
    with torch.no_grad():
        _ = freq_module(feat_in, phys_emb)
    h.remove()
    
    x_out = container["out"]
    if x_out is None:
        # 如果hook失败，直接用模块的forward
        with torch.no_grad():
            x_out = freq_module(feat_in, phys_emb)
            if isinstance(x_out, tuple):
                x_out = x_out[0]
    
    # 确保 x_out 形状与 feat_in 一致
    if x_out.shape != feat_in.shape:
        # 可能是 split 后的输出，取第一个
        if isinstance(x_out, tuple):
            x_out = x_out[0]
        # 尝试调整形状
        if x_out.shape[2:] != feat_in.shape[2:]:
            x_out = F.interpolate(x_out, size=feat_in.shape[2:], mode='bilinear', align_corners=False)
        if x_out.shape[1] != feat_in.shape[1]:
            # 通道数不匹配，取前C个通道
            x_out = x_out[:, :feat_in.shape[1], :, :]
    
    # --- Check 1: 空间域差分图 ---
    check1_path = os.path.join(args.out_dir, "spatial_delta.png")
    spatial_delta_visualization(feat_in, x_out, save_path=check1_path, num_channels=4)

    # --- Check 2: 多通道频谱分析 ---
    check2_path = os.path.join(args.out_dir, "multi_channel_spectrum.png")
    multi_channel_spectrum_analysis(feat_in, x_out, save_path=check2_path, num_samples=8)

    # --- Check 3: 统一归一化频谱比较 ---
    check3_path = os.path.join(args.out_dir, "unified_spectrum.png")
    unified_normalized_spectrum(feat_in, x_out, save_path=check3_path)

    print("\n=== All Checks Completed ===")
    print(f"Check 1: spatial_delta.png - 看差分是否有规则条纹")
    print(f"Check 2: multi_channel_spectrum.png - 看多通道是否都有十字")
    print(f"Check 3: unified_spectrum.png - 用统一归一化比较")


if __name__ == "__main__":
    main()
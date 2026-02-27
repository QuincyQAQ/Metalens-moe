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

# ====== 根据你自己的实现修改 import ======
# 你原本就有：
from net.MoCE_IR_S_Freq_LKA import MoCEIR_Freq_LKA, Freq_LKA_Expert, Freq_LKA_AdapterLayer


# ==================== FFT / 频率工具函数 ====================

def fft2_mag(x: torch.Tensor) -> torch.Tensor:
    """
    x: (H, W) or (C, H, W)
    返回幅度谱 (同形状).
    """
    if x.dim() == 3:
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
    构造圆环形高频 mask:
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
    返回高频能量占比: ||H(x)|| / ||x||
    """
    B, C, H, W = feat.shape
    feat = feat.detach().clone()
    feat = feat.reshape(-1, H, W)  # (B*C, H, W)

    Freq = torch.fft.fft2(feat, norm="ortho")
    Freq = torch.fft.fftshift(Freq, dim=(-2, -1))

    mask = make_high_freq_mask(H, W, cutoff_ratio, device=feat.device).unsqueeze(0)
    F_high = Freq * mask

    high_energy = (F_high.abs() ** 2).sum()
    total_energy = (Freq.abs() ** 2).sum() + 1e-8

    ratio = (high_energy / total_energy).sqrt().item()
    return ratio


# ==================== 模型加载 & 图像准备 ====================

def safe_torch_load(ckpt_path: str, map_location="cpu"):
    """
    优先使用 weights_only=True，PyTorch 版本不支持时自动回退。
    """
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)
    except TypeError:
        # 老版本 pytorch 不支持 weights_only
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
    x = tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)
    return x


# ==================== 实验 1: 高频比例 vs 层数 ====================

def experiment_high_freq_ratio(
    model: nn.Module,
    img: torch.Tensor,
    cutoff_ratio: float = 0.25,
    save_path: str = "high_freq_ratio.png",
):
    """
    统计经过若干层 Freq_LKA_Expert / Freq_LKA_AdapterLayer 后的高频比例，
    画出和论文 Fig.5 类似的曲线。
    """
    layer_indices, ratios = [], []
    layer_names = []
    hooks = []

    def make_hook(idx, name):
        def hook(module, inp, out):
            feat = out
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat is None:
                return
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)
            if feat.dim() != 4:
                return
            r = high_freq_energy_ratio(feat, cutoff_ratio)
            layer_indices.append(idx)
            ratios.append(r)
            layer_names.append(name)
        return hook

    idx = 0
    # 优先注册 Freq_LKA_AdapterLayer（每个Adapter包含多个Expert）
    for m in model.modules():
        if type(m).__name__ == "Freq_LKA_AdapterLayer":
            hooks.append(m.register_forward_hook(make_hook(idx, f"Adapter{idx}")))
            idx += 1

    # 如果没有AdapterLayer，再尝试找 Expert
    if idx == 0:
        for m in model.modules():
            if isinstance(m, Freq_LKA_Expert):
                hooks.append(m.register_forward_hook(make_hook(idx, f"Expert{idx}")))
                idx += 1
            idx += 1

    with torch.no_grad():
        _ = model(img)

    for h in hooks:
        h.remove()

    if len(ratios) == 0:
        print("[Experiment 1] WARNING: No Freq_LKA module hook fired.")
        return

    print(f"[Experiment 1] Found {len(ratios)} layers: {layer_names}")

    plt.figure(figsize=(8, 4))
    plt.plot(layer_indices, ratios, marker="o", linewidth=2, markersize=8)
    
    # 添加层名称标签
    for i, name in enumerate(layer_names):
        plt.annotate(name, (layer_indices[i], ratios[i]), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    plt.xlabel("Layer index")
    plt.ylabel(r"$\|H(x)\| / \|x\|$")
    plt.ylim(0.0, 1.05)
    plt.xticks(layer_indices)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Experiment 1] Saved curve to {save_path}")


# ==================== 关键：真实抓取 freq 输入/输出/phys_emb ====================

def find_first_module(model: nn.Module, target_class_name: str):
    """
    按类名找第一个模块（避免你工程里类不在同一个 import 导致 isinstance 失败）。
    """
    for m in model.modules():
        if type(m).__name__ == target_class_name:
            return m
    return None


def capture_one_freq_io_and_phys(model: nn.Module, img: torch.Tensor):
    """
    从模型里找一个模块来抓：
    - x_in: 传给 freq 模块的特征
    - phys_emb: 同时传进去的物理嵌入（如果存在）
    - x_out: freq 模块输出

    优先抓 Freq_LKA_AdapterLayer（通常它会把 phys_emb 传给 expert）
    抓不到再抓 Freq_LKA_Expert（可能你代码直接调用 expert）
    """
    # 1) 优先 AdapterLayer（更可能拿到 phys_emb）
    target = find_first_module(model, "Freq_LKA_AdapterLayer")
    target_type = "Freq_LKA_AdapterLayer"

    # 2) 再退回 Expert
    if target is None:
        for m in model.modules():
            if isinstance(m, Freq_LKA_Expert):
                target = m
                break
        target_type = "Freq_LKA_Expert"

    if target is None:
        raise RuntimeError("Did not find Freq_LKA_AdapterLayer or Freq_LKA_Expert in model.")

    pack = {"x_in": None, "phys": None, "x_out": None}

    def hook(module, inp, out):
        # inp 可能是 (x,) 或 (x, phys) 或更多参数
        if inp and len(inp) >= 1 and torch.is_tensor(inp[0]):
            pack["x_in"] = inp[0].detach()
        if inp and len(inp) >= 2 and torch.is_tensor(inp[1]):
            pack["phys"] = inp[1].detach()

        # out 可能是 tensor 或 tuple/list
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
        raise RuntimeError(
            f"Hook on {target_type} did not capture x_in/x_out. "
            f"Please check that this module is actually executed in forward."
        )

    return target, target_type, pack["x_in"], pack["phys"], pack["x_out"]


# ==================== 实验 2: 用真实 phys_emb 做多次前向（对比不同条件） ====================

def experiment_phys_sensitivity(
    freq_module: nn.Module,
    x_in: torch.Tensor,
    phys_emb_real: torch.Tensor | None,
    device: str,
    var_save_path: str = "phys_sensitivity_variance.png",
    spectrum_save_path: str = "phys_sensitivity_spectrum.png",
):
    """
    目标：避免你之前 randn phys_emb 造成“门控压死”。
    做法：
      - Case A: phys = 0
      - Case B: phys = real (如果能抓到)
      - Case C: phys = real * 0.5 (尺度敏感性)
      - Case D: phys = real * 2.0
    并输出：in_var/out_var/delta_var/rms_ratio，帮助判断是否“异常压缩”。
    """
    freq_module.eval()
    freq_module.to(device)

    x = x_in.detach().to(device)

    # 获取 phys_dim：优先从 module 属性拿
    phys_dim = None
    if hasattr(freq_module, "phys_dim"):
        phys_dim = int(freq_module.phys_dim)

    B = x.size(0)
    if phys_dim is None:
        # 如果拿不到 phys_dim，但抓到真实 phys，就用真实 phys 的维度
        if phys_emb_real is not None:
            phys_dim = phys_emb_real.size(1)
        else:
            # 实在没有，就默认 128（你原来就是 128）
            phys_dim = 128

    phys0 = torch.zeros(B, phys_dim, device=device)

    cases = [("phys=0", phys0)]

    if phys_emb_real is not None:
        physr = phys_emb_real.detach().to(device)
        # 若维度不一致，报错（避免 silently 截断导致莫名其妙）
        if physr.size(1) != phys_dim:
            raise RuntimeError(f"Captured phys_dim={physr.size(1)} but module phys_dim={phys_dim}.")
        cases += [
            ("phys=real", physr),
            ("phys=real*0.5", physr * 0.5),
            ("phys=real*2.0", physr * 2.0),
        ]
    else:
        print("[Experiment 2] WARNING: Could not capture real phys_emb. Only testing phys=0.")

    stats = []
    outputs = {}

    def print_stats(tag, y):
        in_var = x.var().item()
        out_var = y.var().item()
        delta_var = (y - x).var().item()
        rms_ratio = (y.pow(2).mean().sqrt() / (x.pow(2).mean().sqrt() + 1e-8)).item()
        print(f"[Experiment 2] {tag:>14s} | in_var={in_var:.6f} out_var={out_var:.6f} "
              f"delta_var={delta_var:.6f} rms_ratio={rms_ratio:.6f}")
        return (tag, in_var, out_var, delta_var, rms_ratio)

    with torch.no_grad():
        for tag, phys in cases:
            y = freq_module(x, phys)
            if isinstance(y, (tuple, list)):
                y = y[0]
            if y.shape != x.shape:
                # 形状不一致就插值对齐（空间维）
                if y.dim() == 4 and x.dim() == 4 and y.shape[2:] != x.shape[2:]:
                    y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
                # 通道不一致就先截断到最小通道数（为了能算 delta）
                if y.dim() == 4 and y.shape[1] != x.shape[1]:
                    Cmin = min(y.shape[1], x.shape[1])
                    y = y[:, :Cmin]
                    x_cmp = x[:, :Cmin]
                else:
                    x_cmp = x
            else:
                x_cmp = x
            # 统计用对齐后的
            stats.append(print_stats(tag, y))
            outputs[tag] = (x_cmp.detach().cpu(), y.detach().cpu())

    # 画方差曲线
    tags = [s[0] for s in stats]
    out_vars = [s[2] for s in stats]

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(tags)), out_vars, marker="o", linewidth=2)
    plt.xticks(range(len(tags)), tags, rotation=20, ha="right")
    plt.ylabel("Output Variance")
    plt.title("Variance under different phys_emb")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(var_save_path, dpi=300)
    plt.close()
    print(f"[Experiment 2] Saved variance plot to {var_save_path}")

    # 画频谱：输入 + 最后一个case的输出（如果只有一个case就用它）
    last_tag = tags[-1]
    x_show, y_show = outputs[last_tag]

    # 取第一个通道展示频谱
    ch = 0
    xin = x_show[0, ch]
    yout = y_show[0, ch]

    mag_in = fft2_mag(xin).numpy()
    mag_out = fft2_mag(yout).numpy()

    def vis_norm(mag):
        mag = mag / (mag.max() + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag

    mag_in = vis_norm(mag_in)
    mag_out = vis_norm(mag_out)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mag_in, cmap="viridis")
    plt.title("Input Spectrum")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mag_out, cmap="viridis")
    plt.title(f"Output Spectrum ({last_tag})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(spectrum_save_path, dpi=300)
    plt.close()
    print(f"[Experiment 2] Saved spectrum compare to {spectrum_save_path}")


# ==================== 实验 3: High/Low group-wise FFT（用真实 x_in 做分解） ====================

def groupwise_fft_visualization(
    high_feat: torch.Tensor,
    low_feat: torch.Tensor,
    num_groups: int = 8,
    save_path: str = "groupwise_fft.png",
):
    assert high_feat.shape == low_feat.shape
    B, C, H, W = high_feat.shape
    assert C >= num_groups, "C must >= num_groups"

    def split_groups(feat):
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

    high_fft = [fft2_mag(g) for g in high_groups]
    low_fft = [fft2_mag(g) for g in low_groups]

    def enhance(mag):
        mag = mag / (mag.max() + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag

    high_fft = [enhance(h.cpu().numpy()) for h in high_fft]
    low_fft = [enhance(l.cpu().numpy()) for l in low_fft]

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
    plt.close()
    print(f"[Experiment 3] Saved group-wise FFT grid to {save_path}")


def simple_high_low_split(x: torch.Tensor):
    """
    用一个简单方式从 x 得到 low/high（和你之前一致）：
      low = avgpool -> upsample
      high = x - low
    """
    B, C, H, W = x.shape
    low = F.avg_pool2d(x, kernel_size=2, stride=2)
    low = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
    high = x - low
    return high, low


# ==================== Check: 空间差分 + 多通道谱 + 统一归一化谱 ====================

def spatial_delta_visualization(x_in, x_out, save_path="spatial_delta.png", num_channels=4):
    x_in = x_in.detach().cpu()
    x_out = x_out.detach().cpu()

    B, C, H, W = x_in.shape

    def normalize(t):
        t = t - t.min()
        t = t / (t.max() + 1e-8)
        return t

    ch_indices = list(range(min(num_channels, C)))
    fig, axes = plt.subplots(3, len(ch_indices), figsize=(4 * len(ch_indices), 10))
    if len(ch_indices) == 1:
        axes = axes.reshape(3, 1)

    for i, ch in enumerate(ch_indices):
        inp_ch = normalize(x_in[0, ch])
        out_ch = normalize(x_out[0, ch])

        axes[0, i].imshow(inp_ch, cmap="viridis")
        axes[0, i].set_title(f"Input ch[{ch}]")
        axes[0, i].axis("off")

        axes[1, i].imshow(out_ch, cmap="viridis")
        axes[1, i].set_title(f"Output ch[{ch}]")
        axes[1, i].axis("off")

        delta = (out_ch - inp_ch).abs()
        delta = delta / (delta.max() + 1e-8)
        axes[2, i].imshow(delta, cmap="viridis")
        axes[2, i].set_title(f"|Out-In| ch[{ch}]")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Check] Saved spatial delta to {save_path}")


def multi_channel_spectrum_analysis(x_in, x_out, save_path="multi_channel_spectrum.png", num_samples=8):
    x_in = x_in.detach().cpu()
    x_out = x_out.detach().cpu()

    B, C, H, W = x_in.shape
    if C >= num_samples:
        ch_indices = np.random.choice(C, num_samples, replace=False)
    else:
        ch_indices = np.arange(C)

    def get_spectra(feat):
        arr = []
        for ch in ch_indices:
            arr.append(fft2_mag(feat[0, ch]).numpy())
        return np.array(arr)

    spec_in = get_spectra(x_in)
    spec_out = get_spectra(x_out)

    mean_in = spec_in.mean(axis=0)
    mean_out = spec_out.mean(axis=0)
    med_in = np.median(spec_in, axis=0)
    med_out = np.median(spec_out, axis=0)

    ref_max = max(mean_in.max(), mean_out.max())

    def norm(mag):
        mag = mag / (ref_max + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(norm(mean_in), cmap="viridis")
    axes[0, 0].set_title(f"Mean Spectrum (Input) - {len(ch_indices)} ch")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(norm(mean_out), cmap="viridis")
    axes[0, 1].set_title(f"Mean Spectrum (Output) - {len(ch_indices)} ch")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(norm(med_in), cmap="viridis")
    axes[1, 0].set_title(f"Median Spectrum (Input) - {len(ch_indices)} ch")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(norm(med_out), cmap="viridis")
    axes[1, 1].set_title(f"Median Spectrum (Output) - {len(ch_indices)} ch")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Check] Saved multi-channel spectrum to {save_path}")
    print(f"[Check] Channels used: {sorted(ch_indices.tolist())}")


def unified_normalized_spectrum(x_in, x_out, save_path="unified_spectrum.png"):
    x_in = x_in.detach().cpu()
    x_out = x_out.detach().cpu()

    ch = 0
    mag_in = fft2_mag(x_in[0, ch]).numpy()
    mag_out = fft2_mag(x_out[0, ch]).numpy()

    ref_max = mag_in.max()

    def norm(mag):
        mag = mag / (ref_max + 1e-8)
        mag = np.log1p(mag * 10)
        mag = mag / (mag.max() + 1e-8)
        return mag

    a = norm(mag_in)
    b = norm(mag_out)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(a, cmap="viridis")
    axes[0].set_title("Input Spectrum (norm to input max)")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(b, cmap="viridis")
    axes[1].set_title("Output Spectrum (norm to input max)")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Check] Saved unified normalized spectrum to {save_path}")


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Freq_LKA effect visualization (fixed)")
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

    # --- Experiment 1 ---
    exp1_path = os.path.join(args.out_dir, "high_freq_ratio.png")
    experiment_high_freq_ratio(model, img, cutoff_ratio=0.25, save_path=exp1_path)

    # --- 关键：抓真实 x_in / phys / x_out ---
    target, target_type, x_in, phys_real, x_out = capture_one_freq_io_and_phys(model, img)
    print(f"[Hook] Captured from: {target_type} ({type(target).__name__})")
    if phys_real is None:
        print("[Hook] phys_emb: NOT captured (module may not take phys as input)")
    else:
        print(f"[Hook] phys_emb captured: shape={tuple(phys_real.shape)} "
              f"mean={phys_real.mean().item():.4f} std={phys_real.std().item():.4f}")

    # --- 找一个真正可单独调用的 freq_module ---
    # 如果抓到的是 AdapterLayer，通常里面有 experts[0] 或类似结构；否则就是 Expert 本身
    freq_module = None
    if hasattr(target, "experts") and isinstance(target.experts, (list, nn.ModuleList)) and len(target.experts) > 0:
        freq_module = target.experts[0]
        print(f"[Select] Using target.experts[0] as freq_module: {type(freq_module).__name__}")
    elif isinstance(target, Freq_LKA_Expert):
        freq_module = target
        print(f"[Select] Using target itself as freq_module: {type(freq_module).__name__}")
    else:
        # 最后兜底：直接用 target（可能 forward(x, phys)）
        freq_module = target
        print(f"[Select] Fallback use target as freq_module: {type(freq_module).__name__}")

    # --- Experiment 2: phys 敏感性（不再 randn） ---
    exp2_var_path = os.path.join(args.out_dir, "phys_sensitivity_variance.png")
    exp2_spec_path = os.path.join(args.out_dir, "phys_sensitivity_spectrum.png")
    experiment_phys_sensitivity(
        freq_module=freq_module,
        x_in=x_in,
        phys_emb_real=phys_real,
        device=device,
        var_save_path=exp2_var_path,
        spectrum_save_path=exp2_spec_path,
    )

    # --- Experiment 3: group-wise FFT（用真实 x_in 做高低频分解） ---
    high_feat, low_feat = simple_high_low_split(x_in.detach())
    exp3_path = os.path.join(args.out_dir, "groupwise_fft.png")
    groupwise_fft_visualization(high_feat, low_feat, num_groups=args.groups, save_path=exp3_path)

    # --- Checks: 用 hook 的 x_in 和 x_out 做可视化 ---
    # 对齐形状（防止通道/空间不一致导致图画不了）
    x_in_vis = x_in.detach()
    x_out_vis = x_out.detach()
    if x_out_vis.dim() == 4 and x_in_vis.dim() == 4:
        if x_out_vis.shape[2:] != x_in_vis.shape[2:]:
            x_out_vis = F.interpolate(x_out_vis, size=x_in_vis.shape[2:], mode="bilinear", align_corners=False)
        if x_out_vis.shape[1] != x_in_vis.shape[1]:
            Cmin = min(x_out_vis.shape[1], x_in_vis.shape[1])
            x_out_vis = x_out_vis[:, :Cmin]
            x_in_vis = x_in_vis[:, :Cmin]

    check1_path = os.path.join(args.out_dir, "spatial_delta.png")
    spatial_delta_visualization(x_in_vis, x_out_vis, save_path=check1_path, num_channels=4)

    check2_path = os.path.join(args.out_dir, "multi_channel_spectrum.png")
    multi_channel_spectrum_analysis(x_in_vis, x_out_vis, save_path=check2_path, num_samples=8)

    check3_path = os.path.join(args.out_dir, "unified_spectrum.png")
    unified_normalized_spectrum(x_in_vis, x_out_vis, save_path=check3_path)

    print("\n=== All Done ===")
    print(f"1) {exp1_path}")
    print(f"2) {exp2_var_path}")
    print(f"3) {exp2_spec_path}")
    print(f"4) {exp3_path}")
    print(f"5) {check1_path}")
    print(f"6) {check2_path}")
    print(f"7) {check3_path}")


if __name__ == "__main__":
    main()
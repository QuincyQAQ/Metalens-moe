"""
计算 Freq_LKA 模块中高频和低频的能量占比
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

# 导入模型
from net.MoCE_IR_S_Freq_LKA import MoCEIR_Freq_LKA, Freq_LKA_Expert, Freq_LKA_AdapterLayer


def safe_torch_load(ckpt_path: str, map_location="cpu"):
    import warnings
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


def find_first_module(model: nn.Module, target_class_name: str):
    for m in model.modules():
        if type(m).__name__ == target_class_name:
            return m
    return None


def compute_frequency_energy_ratio(args):
    """计算高频/低频能量占比"""
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = load_model(args.ckpt, device=device)
    img = load_image(args.img, img_size=256, device=device)
    
    # 找到 Freq_LKA_Expert
    expert = None
    for m in model.modules():
        if isinstance(m, Freq_LKA_Expert):
            expert = m
            break
    
    if expert is None:
        print("ERROR: Cannot find Freq_LKA_Expert")
        return
    
    print(f"Found expert: {type(expert).__name__}")
    print(f"Expert hidden_dim: {expert.hidden_dim}")
    
    # Hook 来抓取 in_proj 的输出，然后手动计算频率分解
    pack = {"x_main": None, "x_low": None, "x_high": None}
    
    def hook_inproj(module, inp, out):
        """抓取 in_proj 输出后的 x_main (chunk后的主特征)"""
        # out 是 in_proj 的输出，形状是 [B, hidden_dim*2, H, W]
        x_and_gate = out
        if x_and_gate is not None:
            # chunk 成两个
            x_main, _ = x_and_gate.chunk(2, dim=1)
            b, c, h, w = x_main.shape
            
            # 频率分解（和原代码一样）
            x_low = F.avg_pool2d(x_main, kernel_size=2, stride=2)
            x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
            x_high = x_main - x_low
            
            pack["x_main"] = x_main.detach()
            pack["x_low"] = x_low.detach()
            pack["x_high"] = x_high.detach()
    
    h1 = expert.in_proj.register_forward_hook(hook_inproj)
    
    # 触发 forward
    with torch.no_grad():
        _ = model(img)
    
    h1.remove()
    
    if pack["x_main"] is None:
        print("ERROR: Failed to capture intermediate features")
        return
    
    x_main = pack["x_main"]
    x_low = pack["x_low"]
    x_high = pack["x_high"]
    
    print(f"Captured shapes: x_main={x_main.shape}, x_low={x_low.shape}, x_high={x_high.shape}")
    
    # 计算能量 (L2 norm 的平方 = sum of squares)
    energy_main = (x_main ** 2).sum().item()
    energy_low = (x_low ** 2).sum().item()
    energy_high = (x_high ** 2).sum().item()
    
    print("\n" + "="*60)
    print("频率分解能量分析 (使用 avg_pool + interpolate 方法)")
    print("="*60)
    print(f"x_main 能量: {energy_main:.4f} (总特征)")
    print(f"x_low (低频)     能量: {energy_low:.4f}")
    print(f"x_high (高频)   能量: {energy_high:.4f}")
    print("-"*60)
    print(f"低频能量占比: {energy_low/energy_main*100:.2f}%")
    print(f"高频能量占比: {energy_high/energy_main*100:.2f}%")
    print(f"低频+高频 (验证): {(energy_low+energy_high)/energy_main*100:.2f}%")
    print("="*60)
    
    # 额外分析：查看 x_low 的实际值范围
    print(f"\nx_low 统计: min={x_low.min():.4f}, max={x_low.max():.4f}, mean={x_low.mean():.4f}, std={x_low.std():.4f}")
    print(f"x_high 统计: min={x_high.min():.4f}, max={x_high.max():.4f}, mean={x_high.mean():.6f}, std={x_high.std():.4f}")
    
    # 可视化
    import matplotlib.pyplot as plt
    
    ch = 0  # 第一个通道
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    def norm(t):
        t = t - t.min()
        t = t / (t.max() + 1e-8)
        return t
    
    axes[0].imshow(norm(x_main[0, ch]).cpu().numpy(), cmap="viridis")
    axes[0].set_title(f"x_main (Total)\nE={energy_main:.2f}")
    axes[0].axis("off")
    
    axes[1].imshow(norm(x_low[0, ch]).cpu().numpy(), cmap="viridis")
    axes[1].set_title(f"x_low (Low-freq)\n{energy_low/energy_main*100:.1f}%")
    axes[1].axis("off")
    
    axes[2].imshow(norm(x_high[0, ch]).cpu().numpy(), cmap="viridis")
    axes[2].set_title(f"x_high (High-freq)\n{energy_high/energy_main*100:.1f}%")
    axes[2].axis("off")
    
    # 验证：x_main ≈ x_low + x_high
    recon = x_low + x_high
    diff = (x_main - recon).abs()
    axes[3].imshow(norm(diff[0, ch]).cpu().numpy(), cmap="viridis")
    axes[3].set_title(f"Reconstruction Error\n{diff.mean().item():.2e}")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig("frequency_energy_analysis.png", dpi=150)
    print(f"\n已保存可视化到 frequency_energy_analysis.png")
    
    return {
        "energy_main": energy_main,
        "energy_low": energy_low,
        "energy_high": energy_high,
        "low_ratio": energy_low/energy_main*100,
        "high_ratio": energy_high/energy_main*100
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    compute_frequency_energy_ratio(args)

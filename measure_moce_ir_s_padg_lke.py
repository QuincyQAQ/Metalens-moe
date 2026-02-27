#!/usr/bin/env python3
"""
单独计算 MoCE_IR_S_PADG_LKE 模型的 GFLOPs 和参数量。

使用 train._calculate_model_complexity 计算，输入尺寸为 1x3x256x256。
"""

import sys
import torch

# 添加项目路径
CUR_DIR = __file__.replace("\\", "/").rsplit("/", 1)[0]
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import config
from options import train_options
from train import _calculate_model_complexity


def build_model(model_name: str):
    """构建模型"""
    # 设置 config.MODEL
    setattr(config, "MODEL", model_name)
    
    # 构建 opt
    opt = train_options()
    
    # 动态导入模型
    module_name = f"net.{model_name}"
    module = __import__(module_name, fromlist=['build_model'])
    build_model_fn = module.build_model
    
    # 构建模型
    model = build_model_fn(opt)
    return model


def main():
    model_name = "MoCE_IR_S_PADG_LKE"
    input_size = (1, 3, 256, 256)
    
    print("=" * 60)
    print(f"计算模型复杂度: {model_name}")
    print(f"输入尺寸: {input_size}")
    print("=" * 60)
    
    # 构建模型
    print("\n[1] 构建模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    使用设备: {device}")
    
    try:
        model = build_model(model_name)
        model = model.to(device)
        model.eval()
        print("    ✓ 模型构建成功")
    except Exception as e:
        print(f"    ✗ 模型构建失败: {e}")
        return
    
    # 计算参数量
    print("\n[2] 计算参数量...")
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_m = num_params / 1e6
        print(f"    参数量: {num_params:,} ({params_m:.6f} M)")
    except Exception as e:
        print(f"    ✗ 参数量计算失败: {e}")
        params_m = None
    
    # 计算 GFLOPs
    print("\n[3] 计算 GFLOPs...")
    try:
        gflops, _ = _calculate_model_complexity(model, input_size=input_size)
        if gflops is not None:
            print(f"    GFLOPs: {gflops:.6f}")
        else:
            print("    ✗ GFLOPs 计算失败")
    except Exception as e:
        print(f"    ✗ GFLOPs 计算异常: {e}")
        gflops = None
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("最终结果:")
    print("=" * 60)
    if gflops is not None and params_m is not None:
        print(f"  模型名称: {model_name}")
        print(f"  输入尺寸: {input_size[1]}x{input_size[2]}x{input_size[3]}")
        print(f"  GFLOPs:   {gflops:.6f}")
        print(f"  参数量:   {params_m:.6f} M")
        print(f"\n  按 1 MAC ≈ 2 FLOPs 计算:")
        print(f"  MACs:    {gflops * 1e9 / 2:.0f} M")
    else:
        print("  计算未成功完成")
    print("=" * 60)


if __name__ == "__main__":
    main()


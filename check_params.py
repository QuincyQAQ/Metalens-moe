#!/usr/bin/env python3
"""
计算模型参数量的脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import MODEL
from options import train_options

# 动态导入模型模块
model_module_name = f"net.{MODEL}"
model_module = __import__(model_module_name, fromlist=['build_model'])
build_model = getattr(model_module, 'build_model')

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_params(num_params):
    """格式化参数量显示"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

def main():
    print(f"正在计算模型 {MODEL} 的参数量...")
    print("-" * 60)
    
    # 使用 train_options 创建配置对象
    opt = train_options()
    
    try:
        # 构建模型
        model = build_model(opt)
        
        # 计算参数量
        total_params, trainable_params = count_parameters(model)
        
        print(f"模型名称: {MODEL}")
        print(f"总参数量: {total_params:,} ({format_params(total_params)})")
        print(f"可训练参数: {trainable_params:,} ({format_params(trainable_params)})")
        print(f"不可训练参数: {total_params - trainable_params:,} ({format_params(total_params - trainable_params)})")
        print("-" * 60)
        
        # 按模块统计参数量
        print("\n各模块参数量统计:")
        print("-" * 60)
        module_params = {}
        for name, param in model.named_parameters():
            module_name = name.split('.')[0]
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param.numel()
        
        # 按参数量排序
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
        for module_name, params in sorted_modules[:20]:  # 显示前20个最大的模块
            print(f"  {module_name:30s}: {params:>12,} ({format_params(params)})")
        
        return total_params
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    total_params = main()
    if total_params:
        print(f"\n✓ 参数量计算完成: {total_params:,} ({format_params(total_params)})")


#!/usr/bin/env python3
"""
全面的模型验证脚本
在训练开始前验证模型代码，确保无懈可击
"""

import sys
import os
import traceback
import warnings
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入配置和模型
import config
import importlib
from options import train_options

# 抑制警告以便更清晰地看到错误
warnings.filterwarnings('error', category=UserWarning)
warnings.filterwarnings('error', category=RuntimeWarning)

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.errors = []
        self.warnings = []
        self.passed_tests = []
        self.failed_tests = []
        
    def log_error(self, test_name: str, error: Exception):
        """记录错误"""
        error_msg = f"[FAIL] {test_name}: {str(error)}"
        self.errors.append(error_msg)
        self.failed_tests.append(test_name)
        print(f"\n❌ {error_msg}")
        print(f"   详细错误: {traceback.format_exc()}")
        
    def log_warning(self, test_name: str, warning: str):
        """记录警告"""
        warning_msg = f"[WARN] {test_name}: {warning}"
        self.warnings.append(warning_msg)
        print(f"\n⚠️  {warning_msg}")
        
    def log_success(self, test_name: str):
        """记录成功"""
        self.passed_tests.append(test_name)
        print(f"✓ {test_name}")
        
    def test_model_instantiation(self) -> Optional[nn.Module]:
        """测试1: 模型实例化"""
        test_name = "模型实例化"
        try:
            print(f"\n{'='*60}")
            print(f"测试1: {test_name}")
            print(f"{'='*60}")
            
            # 创建配置对象
            opt = train_options()
            
            # 动态导入并构建模型
            model_name = getattr(opt, "model", None)
            if not model_name:
                raise ValueError("opt.model 未设置，请在 config.py 中设置 MODEL")
            
            module = importlib.import_module(f"net.{model_name}")
            build_fn = getattr(module, "build_model", None)
            if build_fn is None:
                raise AttributeError(f"net.{model_name} 缺少 build_model 函数")
            
            model = build_fn(opt)
            model = model.to(self.device)
            model.train()  # 设置为训练模式
            
            # 检查模型是否有参数
            num_params = sum(p.numel() for p in model.parameters())
            if num_params == 0:
                raise ValueError("模型没有参数！")
            
            print(f"    模型参数量: {num_params:,}")
            print(f"    可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            self.log_success(test_name)
            return model
            
        except Exception as e:
            self.log_error(test_name, e)
            return None
    
    def test_forward_pass_basic(self, model: nn.Module):
        """测试2: 基础前向传播"""
        test_name = "基础前向传播 (batch=1, size=128x128)"
        try:
            print(f"\n{'='*60}")
            print(f"测试2: {test_name}")
            print(f"{'='*60}")
            
            # 标准输入尺寸
            x = torch.randn(1, 3, 128, 128, device=self.device)
            
            with torch.no_grad():
                output = model(x)
            
            # 处理模型可能返回元组的情况（output, loss）
            if isinstance(output, (tuple, list)):
                if len(output) == 0:
                    raise ValueError("模型返回空元组/列表")
                output = output[0]  # 取第一个元素作为输出
            
            # 检查输出是否为tensor
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"模型输出不是tensor，而是 {type(output)}")
            
            # 检查输出形状
            if output.shape != x.shape:
                raise ValueError(f"输出形状不匹配: 期望 {x.shape}, 得到 {output.shape}")
            
            # 检查输出值是否有效
            if torch.isnan(output).any():
                raise ValueError("输出包含NaN值！")
            if torch.isinf(output).any():
                raise ValueError("输出包含Inf值！")
            
            print(f"    输入形状: {x.shape}")
            print(f"    输出形状: {output.shape}")
            print(f"    输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_forward_pass_various_sizes(self, model: nn.Module):
        """测试3: 不同输入尺寸的前向传播"""
        test_name = "不同输入尺寸的前向传播"
        try:
            print(f"\n{'='*60}")
            print(f"测试3: {test_name}")
            print(f"{'='*60}")
            
            # 测试多种输入尺寸（包括训练时可能遇到的尺寸）
            test_sizes = [
                (1, 3, 64, 64),      # 小尺寸
                (1, 3, 128, 128),    # 标准尺寸
                (1, 3, 256, 256),    # 大尺寸
                (2, 3, 128, 128),    # batch=2
                (4, 3, 128, 128),    # batch=4
                (8, 3, 128, 128),    # batch=8
                (1, 3, 96, 96),      # 非2的幂次
                (1, 3, 160, 160),    # 非2的幂次
                (1, 3, 192, 192),    # 非2的幂次
            ]
            
            for size in test_sizes:
                try:
                    x = torch.randn(*size, device=self.device)
                    
                    with torch.no_grad():
                        output = model(x)
                    
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                    
                    if not isinstance(output, torch.Tensor):
                        raise ValueError(f"尺寸 {size}: 输出不是tensor，而是 {type(output)}")
                    
                    if output.shape[:2] != x.shape[:2] or output.shape[2:] != x.shape[2:]:
                        raise ValueError(f"尺寸 {size}: 输出形状 {output.shape} 不匹配输入 {x.shape}")
                    
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        raise ValueError(f"尺寸 {size}: 输出包含NaN或Inf值")
                    
                    print(f"    ✓ {size} -> {output.shape}")
                    
                except Exception as e:
                    raise ValueError(f"尺寸 {size} 失败: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_backward_pass(self, model: nn.Module):
        """测试4: 反向传播"""
        test_name = "反向传播和梯度计算"
        try:
            print(f"\n{'='*60}")
            print(f"测试4: {test_name}")
            print(f"{'='*60}")
            
            # 测试不同batch size
            for batch_size in [1, 2, 4]:
                try:
                    x = torch.randn(batch_size, 3, 128, 128, device=self.device, requires_grad=True)
                    target = torch.randn_like(x)
                    
                    # 前向传播
                    output = model(x)
                    
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                    
                    if not isinstance(output, torch.Tensor):
                        raise ValueError(f"batch_size={batch_size}: 输出不是tensor")
                    
                    # 计算损失
                    loss = F.mse_loss(output, target)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 检查梯度
                    has_grad = False
                    nan_grads = 0
                    inf_grads = 0
                    zero_grads = 0
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            has_grad = True
                            if torch.isnan(param.grad).any():
                                nan_grads += 1
                            if torch.isinf(param.grad).any():
                                inf_grads += 1
                            if (param.grad == 0).all():
                                zero_grads += 1
                    
                    if not has_grad:
                        raise ValueError(f"batch_size={batch_size}: 没有参数有梯度！")
                    
                    if nan_grads > 0:
                        raise ValueError(f"batch_size={batch_size}: {nan_grads} 个参数有NaN梯度！")
                    
                    if inf_grads > 0:
                        raise ValueError(f"batch_size={batch_size}: {inf_grads} 个参数有Inf梯度！")
                    
                    print(f"    ✓ batch_size={batch_size}: loss={loss.item():.6f}, 有梯度参数: {has_grad}")
                    
                    # 清零梯度以便下次测试
                    model.zero_grad()
                    
                except Exception as e:
                    raise ValueError(f"batch_size={batch_size} 失败: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_mixed_precision_fp16(self, model: nn.Module):
        """测试5: 混合精度训练 (FP16)"""
        test_name = "混合精度训练 (FP16)"
        try:
            print(f"\n{'='*60}")
            print(f"测试5: {test_name}")
            print(f"{'='*60}")
            
            if not torch.cuda.is_available():
                self.log_warning(test_name, "CUDA不可用，跳过FP16测试")
                return
            
            # 测试不同batch size和输入尺寸
            test_cases = [
                (1, 128, 128),
                (2, 128, 128),
                (4, 128, 128),
                (1, 256, 256),
                (2, 256, 256),
            ]
            
            scaler = GradScaler()
            
            for batch_size, h, w in test_cases:
                try:
                    x = torch.randn(batch_size, 3, h, w, device=self.device)
                    target = torch.randn_like(x)
                    
                    model.zero_grad()
                    
                    # 使用autocast进行混合精度前向传播
                    with autocast():
                        output = model(x)
                        # 处理模型可能返回元组的情况
                        if isinstance(output, (tuple, list)) and len(output) > 0:
                            output = output[0]
                        if not isinstance(output, torch.Tensor):
                            raise ValueError(f"batch={batch_size}, size={h}x{w}: 输出不是tensor")
                        loss = F.mse_loss(output, target)
                    
                    # 检查输出是否有效
                    if torch.isnan(output).any():
                        raise ValueError(f"batch={batch_size}, size={h}x{w}: FP16前向传播产生NaN")
                    if torch.isinf(output).any():
                        raise ValueError(f"batch={batch_size}, size={h}x{w}: FP16前向传播产生Inf")
                    
                    # 混合精度反向传播
                    scaler.scale(loss).backward()
                    
                    # 检查梯度
                    has_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            has_grad = True
                            if torch.isnan(param.grad).any():
                                raise ValueError(f"batch={batch_size}, size={h}x{w}: FP16梯度包含NaN")
                            if torch.isinf(param.grad).any():
                                raise ValueError(f"batch={batch_size}, size={h}x{w}: FP16梯度包含Inf")
                            break
                    
                    if not has_grad:
                        raise ValueError(f"batch={batch_size}, size={h}x{w}: FP16没有梯度")
                    
                    scaler.step(torch.optim.Adam(model.parameters(), lr=1e-4))
                    scaler.update()
                    
                    print(f"    ✓ batch={batch_size}, size={h}x{w}: loss={loss.item():.6f}")
                    
                except Exception as e:
                    raise ValueError(f"batch={batch_size}, size={h}x{w} 失败: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_fft_operations(self, model: nn.Module):
        """测试6: FFT操作（这是之前出错的地方）"""
        test_name = "FFT操作验证"
        try:
            print(f"\n{'='*60}")
            print(f"测试6: {test_name}")
            print(f"{'='*60}")
            
            # 测试各种可能导致FFT错误的尺寸
            # cuFFT对某些尺寸有限制，特别是在fp16下
            test_sizes = [
                (1, 3, 64, 64),
                (1, 3, 96, 96),
                (1, 3, 128, 128),
                (1, 3, 160, 160),
                (1, 3, 192, 192),
                (1, 3, 256, 256),
                (2, 3, 128, 128),
                (4, 3, 128, 128),
                (8, 3, 128, 128),
                (16, 3, 128, 128),
            ]
            
            for size in test_sizes:
                try:
                    x = torch.randn(*size, device=self.device)
                    
                    # 测试fp32
                    with torch.no_grad():
                        output_fp32 = model(x.float())
                        # 处理模型可能返回元组的情况
                        if isinstance(output_fp32, (tuple, list)) and len(output_fp32) > 0:
                            output_fp32 = output_fp32[0]
                    
                    # 测试fp16（如果支持）
                    if torch.cuda.is_available():
                        with torch.no_grad(), autocast():
                            output_fp16 = model(x.half())
                            # 处理模型可能返回元组的情况
                            if isinstance(output_fp16, (tuple, list)) and len(output_fp16) > 0:
                                output_fp16 = output_fp16[0]
                        
                        # 检查fp16输出是否有效
                        if not isinstance(output_fp16, torch.Tensor):
                            raise ValueError(f"尺寸 {size}: FP16输出不是tensor")
                        if torch.isnan(output_fp16).any():
                            raise ValueError(f"尺寸 {size}: FP16 FFT产生NaN")
                        if torch.isinf(output_fp16).any():
                            raise ValueError(f"尺寸 {size}: FP16 FFT产生Inf")
                    
                    print(f"    ✓ {size}: FFT操作正常")
                    
                except RuntimeError as e:
                    if "cuFFT" in str(e) or "CUFFT" in str(e):
                        raise ValueError(f"尺寸 {size}: cuFFT错误 - {str(e)}")
                    else:
                        raise
                except Exception as e:
                    raise ValueError(f"尺寸 {size}: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_memory_usage(self, model: nn.Module):
        """测试7: 内存使用"""
        test_name = "内存使用检查"
        try:
            print(f"\n{'='*60}")
            print(f"测试7: {test_name}")
            print(f"{'='*60}")
            
            if not torch.cuda.is_available():
                self.log_warning(test_name, "CUDA不可用，跳过内存测试")
                return
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # 测试不同batch size的内存使用
            batch_sizes = [1, 2, 4, 8, 16, 32]
            
            for batch_size in batch_sizes:
                try:
                    x = torch.randn(batch_size, 3, 128, 128, device=self.device)
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                    with torch.no_grad():
                        _ = model(x)
                    
                    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                    
                    print(f"    batch_size={batch_size:2d}: {memory_mb:8.2f} MB")
                    
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError:
                    self.log_warning(test_name, f"batch_size={batch_size} 时内存不足")
                    break
                except Exception as e:
                    raise ValueError(f"batch_size={batch_size}: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_edge_cases(self, model: nn.Module):
        """测试8: 边界情况"""
        test_name = "边界情况测试"
        try:
            print(f"\n{'='*60}")
            print(f"测试8: {test_name}")
            print(f"{'='*60}")
            
            # 测试1: 全零输入
            try:
                x = torch.zeros(1, 3, 128, 128, device=self.device)
                with torch.no_grad():
                    output = model(x)
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                if not isinstance(output, torch.Tensor):
                    raise ValueError("全零输入输出不是tensor")
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError("全零输入产生NaN或Inf")
                print(f"    ✓ 全零输入: 正常")
            except Exception as e:
                raise ValueError(f"全零输入失败: {str(e)}")
            
            # 测试2: 全一输入
            try:
                x = torch.ones(1, 3, 128, 128, device=self.device)
                with torch.no_grad():
                    output = model(x)
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                if not isinstance(output, torch.Tensor):
                    raise ValueError("全一输入输出不是tensor")
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError("全一输入产生NaN或Inf")
                print(f"    ✓ 全一输入: 正常")
            except Exception as e:
                raise ValueError(f"全一输入失败: {str(e)}")
            
            # 测试3: 极大值输入
            try:
                x = torch.ones(1, 3, 128, 128, device=self.device) * 100.0
                with torch.no_grad():
                    output = model(x)
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                if isinstance(output, torch.Tensor):
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        self.log_warning(test_name, "极大值输入产生NaN或Inf（可能正常）")
                    else:
                        print(f"    ✓ 极大值输入: 正常")
                else:
                    self.log_warning(test_name, f"极大值输入输出不是tensor: {type(output)}")
            except Exception as e:
                self.log_warning(test_name, f"极大值输入异常: {str(e)}（可能正常）")
            
            # 测试4: 极小值输入
            try:
                x = torch.ones(1, 3, 128, 128, device=self.device) * 0.001
                with torch.no_grad():
                    output = model(x)
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                if not isinstance(output, torch.Tensor):
                    raise ValueError("极小值输入输出不是tensor")
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError("极小值输入产生NaN或Inf")
                print(f"    ✓ 极小值输入: 正常")
            except Exception as e:
                raise ValueError(f"极小值输入失败: {str(e)}")
            
            # 测试5: 非方形输入（如果支持）
            try:
                x = torch.randn(1, 3, 128, 160, device=self.device)
                with torch.no_grad():
                    output = model(x)
                    # 处理模型可能返回元组的情况
                    if isinstance(output, (tuple, list)) and len(output) > 0:
                        output = output[0]
                print(f"    ✓ 非方形输入 (128x160): 正常")
            except Exception as e:
                self.log_warning(test_name, f"非方形输入可能不支持: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_training_step_simulation(self, model: nn.Module):
        """测试9: 完整训练步骤模拟"""
        test_name = "完整训练步骤模拟"
        try:
            print(f"\n{'='*60}")
            print(f"测试9: {test_name}")
            print(f"{'='*60}")
            
            # 模拟真实的训练步骤
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
            scaler = GradScaler() if torch.cuda.is_available() else None
            
            # 模拟多个训练步骤
            num_steps = 5
            batch_size = 4
            h, w = 128, 128
            
            for step in range(num_steps):
                try:
                    # 准备数据
                    x = torch.randn(batch_size, 3, h, w, device=self.device)
                    target = torch.randn_like(x)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播（混合精度）
                    if scaler is not None:
                        with autocast():
                            output = model(x)
                            # 处理模型可能返回元组的情况
                            if isinstance(output, (tuple, list)) and len(output) > 0:
                                output = output[0]
                            if not isinstance(output, torch.Tensor):
                                raise ValueError(f"步骤 {step+1}: 输出不是tensor")
                            loss = F.mse_loss(output, target)
                        
                        # 反向传播
                        scaler.scale(loss).backward()
                        
                        # 检查梯度
                        has_nan_grad = False
                        for param in model.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any():
                                    has_nan_grad = True
                                    break
                        
                        if has_nan_grad:
                            raise ValueError(f"步骤 {step+1}: 梯度包含NaN")
                        
                        # 更新参数
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(x)
                        # 处理模型可能返回元组的情况
                        if isinstance(output, (tuple, list)) and len(output) > 0:
                            output = output[0]
                        if not isinstance(output, torch.Tensor):
                            raise ValueError(f"步骤 {step+1}: 输出不是tensor")
                        loss = F.mse_loss(output, target)
                        loss.backward()
                        optimizer.step()
                    
                    if torch.isnan(loss):
                        raise ValueError(f"步骤 {step+1}: 损失为NaN")
                    
                    print(f"    步骤 {step+1}/{num_steps}: loss={loss.item():.6f}")
                    
                except Exception as e:
                    raise ValueError(f"步骤 {step+1} 失败: {str(e)}")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def test_model_eval_mode(self, model: nn.Module):
        """测试10: 评估模式"""
        test_name = "评估模式测试"
        try:
            print(f"\n{'='*60}")
            print(f"测试10: {test_name}")
            print(f"{'='*60}")
            
            model.eval()
            
            x = torch.randn(1, 3, 128, 128, device=self.device)
            
            with torch.no_grad():
                output = model(x)
            
            # 处理模型可能返回元组的情况
            if isinstance(output, (tuple, list)) and len(output) > 0:
                output = output[0]
            
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"评估模式输出不是tensor，而是 {type(output)}")
            
            if output.shape != x.shape:
                raise ValueError(f"评估模式输出形状不匹配: {output.shape} vs {x.shape}")
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("评估模式输出包含NaN或Inf")
            
            print(f"    ✓ 评估模式正常工作")
            
            self.log_success(test_name)
            
        except Exception as e:
            self.log_error(test_name, e)
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始全面模型验证")
        print("="*60)
        print(f"设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)
        
        # 测试1: 模型实例化
        model = self.test_model_instantiation()
        if model is None:
            print("\n❌ 模型实例化失败，无法继续测试")
            return False
        
        # 运行所有测试
        self.test_forward_pass_basic(model)
        self.test_forward_pass_various_sizes(model)
        self.test_backward_pass(model)
        self.test_mixed_precision_fp16(model)
        self.test_fft_operations(model)
        self.test_memory_usage(model)
        self.test_edge_cases(model)
        self.test_training_step_simulation(model)
        self.test_model_eval_mode(model)
        
        # 打印总结
        self.print_summary()
        
        # 返回是否全部通过
        return len(self.failed_tests) == 0
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"通过测试: {len(self.passed_tests)}/{len(self.passed_tests) + len(self.failed_tests)}")
        print(f"失败测试: {len(self.failed_tests)}")
        print(f"警告: {len(self.warnings)}")
        
        if self.passed_tests:
            print("\n✓ 通过的测试:")
            for test in self.passed_tests:
                print(f"  - {test}")
        
        if self.failed_tests:
            print("\n❌ 失败的测试:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        if self.warnings:
            print("\n⚠️  警告:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print("\n详细错误信息:")
            for error in self.errors:
                print(f"  {error}")
        
        print("="*60)
        
        if len(self.failed_tests) == 0:
            print("\n🎉 所有测试通过！模型代码验证无懈可击，可以开始训练。")
        else:
            print(f"\n❌ 有 {len(self.failed_tests)} 个测试失败，请修复后再开始训练。")
            sys.exit(1)


def main():
    """主函数"""
    validator = ModelValidator()
    success = validator.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()


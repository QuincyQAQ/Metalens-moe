#!/usr/bin/env python3
"""
测试模型是否可以正常训练
实际运行几个训练步骤，而不是仅仅验证模型结构
"""
import sys
import os
import traceback
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 抑制警告
warnings.filterwarnings('ignore')
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

def test_model_training(model_name, num_batches=3):
    """测试模型的实际训练过程"""
    print(f"\n{'='*60}")
    print(f"测试模型训练: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        # 导入配置
        import config
        from options import train_options
        
        # 动态导入模型
        model_module_name = f"net.{model_name}"
        model_module = __import__(model_module_name, fromlist=['build_model'])
        build_model = model_module.build_model
        
        # 创建配置对象
        opt = train_options()
        
        # 构建模型
        print("1. 构建模型...")
        model = build_model(opt)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.train()
        print(f"   ✓ 模型已构建并移动到 {device}")
        
        # 创建优化器
        print("2. 创建优化器...")
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        scaler = GradScaler() if device == 'cuda' else None
        print("   ✓ 优化器已创建")
        
        # 创建模拟数据集
        print("3. 创建模拟训练数据...")
        batch_size = 2
        patch_size = 128
        num_samples = batch_size * num_batches
        
        # 创建模拟的degraded和clean图像
        degraded_images = torch.randn(num_samples, 3, patch_size, patch_size)
        clean_images = torch.randn(num_samples, 3, patch_size, patch_size)
        
        # 创建DataLoader
        dataset = TensorDataset(degraded_images, clean_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"   ✓ 创建了 {num_batches} 个batch的训练数据")
        
        # 实际训练几个batch
        print(f"\n4. 开始训练 {num_batches} 个batch...")
        total_loss = 0.0
        
        for batch_idx, (degraded, clean) in enumerate(dataloader):
            degraded = degraded.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            
            # 检查模型forward签名
            import inspect
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.keys())
            
            # 准备物理先验（如果模型需要）
            raw_physics_priors = None
            if 'raw_physics_priors' in params:
                # 创建模拟的物理先验
                raw_physics_priors = {
                    "depth_map": torch.randn(batch_size, 1, patch_size, patch_size).to(device),
                    "spectral_data": torch.randn(batch_size, 3, patch_size, patch_size).to(device),
                    "optical_params": torch.randn(batch_size, 16).to(device)
                }
            
            # 前向传播
            try:
                if raw_physics_priors is not None:
                    output = model(degraded, labels=clean, raw_physics_priors=raw_physics_priors)
                elif 'labels' in params:
                    output = model(degraded, labels=clean)
                else:
                    output = model(degraded)
            except Exception as e:
                print(f"\n   ❌ Batch {batch_idx + 1} 前向传播失败:")
                print(f"   错误: {str(e)}")
                raise
            
            # 检查输出
            if output.shape != degraded.shape:
                raise ValueError(f"输出形状不匹配: {output.shape} vs {degraded.shape}")
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("输出包含NaN或Inf值")
            
            # 计算损失
            loss = F.l1_loss(output, clean)
            
            # 如果有辅助损失（从模型属性获取）
            if hasattr(model, 'total_loss') and model.total_loss is not None:
                loss = loss + model.total_loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"损失值为NaN或Inf: {loss.item()}")
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # 检查梯度
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"   警告: {name} 有NaN梯度")
                            has_nan_grad = True
                
                if has_nan_grad:
                    print(f"   ⚠ Batch {batch_idx + 1} 检测到NaN梯度，跳过此batch")
                    scaler.update()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # 检查梯度
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"   警告: {name} 有NaN梯度")
                            has_nan_grad = True
                
                if has_nan_grad:
                    print(f"   ⚠ Batch {batch_idx + 1} 检测到NaN梯度，跳过此batch")
                    optimizer.step()
                    continue
                
                optimizer.step()
            
            total_loss += loss.item()
            print(f"   Batch {batch_idx + 1}/{num_batches}: loss = {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"\n   ✓ 训练完成！平均损失: {avg_loss:.6f}")
        
        # 测试推理模式
        print("\n5. 测试推理模式...")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, patch_size, patch_size).to(device)
            if raw_physics_priors is not None:
                test_physics_priors = {
                    "depth_map": torch.randn(1, 1, patch_size, patch_size).to(device),
                    "spectral_data": torch.randn(1, 3, patch_size, patch_size).to(device),
                    "optical_params": torch.randn(1, 16).to(device)
                }
                test_output = model(test_input, raw_physics_priors=test_physics_priors)
            elif 'labels' in params:
                test_output = model(test_input, labels=None)
            else:
                test_output = model(test_input)
            
            if test_output.shape[:2] != test_input.shape[:2]:
                raise ValueError(f"推理输出形状不匹配: {test_output.shape} vs {test_input.shape}")
            if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                raise ValueError("推理输出包含NaN或Inf值")
        
        print("   ✓ 推理模式正常")
        
        print(f"\n{'='*60}")
        print(f"✓ 模型 {model_name} 训练测试通过！")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 模型 {model_name} 训练测试失败")
        print(f"{'='*60}")
        print(f"错误: {str(e)}")
        print(f"\n详细错误信息:")
        traceback.print_exc()
        print()
        return False


if __name__ == "__main__":
    # 要测试的模型列表
    models_to_test = [
        "MoCE_IR_S_SV_PhysicsPriorFusion_CGM_CR_EI",
        "MoCE_IR_S_SV_PhysicsPriorFusion_DEPKD_CR",
        "MoCE_IR_S_SV_PhysicsPriorFusion_FASE_DRR",
        "MoCE_IR_S_SV_PhysicsPriorFusion_HEC_AG"
    ]
    
    # 修改config.py中的MODEL参数
    import config
    
    results = {}
    for model_name in models_to_test:
        # 临时修改config中的MODEL
        original_model = config.MODEL
        config.MODEL = model_name
        
        try:
            success = test_model_training(model_name, num_batches=3)
            results[model_name] = success
        finally:
            # 恢复原始MODEL
            config.MODEL = original_model
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print()
    
    passed = [m for m, s in results.items() if s]
    failed = [m for m, s in results.items() if not s]
    
    print(f"通过的模型 ({len(passed)} 个):")
    for model in passed:
        print(f"  ✓ {model}")
    
    print()
    if failed:
        print(f"失败的模型 ({len(failed)} 个):")
        for model in failed:
            print(f"  ❌ {model}")
    
    print()
    if len(passed) == len(models_to_test):
        print("✓ 所有模型训练测试通过！")
        sys.exit(0)
    else:
        print(f"❌ {len(failed)} 个模型训练测试失败")
        sys.exit(1)


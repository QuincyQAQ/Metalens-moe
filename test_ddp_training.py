#!/usr/bin/env python3
"""
测试模型在DDP环境下的训练
使用accelerate库模拟多GPU训练
"""
import sys
import os
import traceback
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# 抑制警告
warnings.filterwarnings('ignore')
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

def test_ddp_training(model_name, num_batches=3):
    """测试模型在DDP环境下的实际训练过程"""
    print(f"\n{'='*60}")
    print(f"测试DDP训练: {model_name}")
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
        print(f"   ✓ 模型已构建")
        
        # 创建DDP加速器（模拟多GPU）
        print("2. 初始化Accelerator (DDP)...")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=None,
            kwargs_handlers=[ddp_kwargs],
        )
        print(f"   ✓ Accelerator已初始化")
        
        # 准备数据
        print("3. 准备数据...")
        batch_size = 2
        patch_size = 128
        num_channels = 3
        
        # 创建虚拟数据
        degraded_images = torch.randn(batch_size, num_channels, patch_size, patch_size)
        clean_images = torch.randn(batch_size, num_channels, patch_size, patch_size)
        
        dataset = TensorDataset(degraded_images, clean_images)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 准备模型、优化器和数据加载器
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        print(f"   ✓ 模型、优化器和数据加载器已准备")
        
        # 训练循环
        print("4. 开始训练循环...")
        model.train()
        loss_fn = nn.L1Loss()
        
        for batch_idx, (degraded, clean) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            print(f"   Batch {batch_idx + 1}/{num_batches}...")
            
            # 前向传播
            try:
                # 准备输入
                # 模型通常需要特征图，我们使用degraded作为输入
                x = degraded  # (B, 3, H, W)
                
                # 创建频率嵌入（模型需要）
                freq_dim = getattr(opt, 'freq_dim', 128)
                freq_emb = torch.randn(batch_size, freq_dim, device=x.device)
                
                # 调用模型forward (labels参数用于传递clean_image给router)
                output = model(degraded, labels=clean)
                
                # 计算损失
                if isinstance(output, tuple):
                    output = output[0]
                
                # 确保输出和clean的尺寸匹配
                if output.shape != clean.shape:
                    output = F.interpolate(output, size=clean.shape[2:], mode='bilinear', align_corners=False)
                
                loss = loss_fn(output, clean)
                
                print(f"      ✓ 前向传播完成，loss: {loss.item():.6f}")
                
                # 反向传播
                accelerator.backward(loss)
                print(f"      ✓ 反向传播完成")
                
                # 更新参数
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                print(f"      ✓ 参数更新完成")
                
            except Exception as e:
                print(f"      ✗ 训练步骤失败:")
                print(f"         {str(e)}")
                traceback.print_exc()
                raise
        
        print(f"\n✓ DDP训练测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ DDP训练测试失败:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试CGM_CR_EI模型（这是有问题的模型）
    model_name = "MoCE_IR_S_SV_PhysicsPriorFusion_CGM_CR_EI"
    success = test_ddp_training(model_name, num_batches=3)
    
    if success:
        print("\n" + "="*60)
        print("所有测试通过！")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("测试失败！")
        print("="*60)
        sys.exit(1)


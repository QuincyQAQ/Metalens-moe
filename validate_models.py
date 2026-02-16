#!/usr/bin/env python3
"""
验证所有网络模型是否可以正常训练
在正式训练前进行快速验证，避免训练到一半时崩溃
"""
import sys
import os
import importlib
import traceback
from typing import Tuple, Dict
import torch
import torch.nn as nn
from pathlib import Path

# 添加当前目录到路径
project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir))

# 设置环境变量来抑制警告
os.environ.setdefault("NCCL_DEBUG", "ERROR")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "0")

# 抑制警告
import warnings
warnings.filterwarnings("ignore")


def validate_model(model_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    验证单个模型是否可以正常训练
    
    Args:
        model_name: 模型名称（不包含.py扩展名）
        verbose: 是否显示详细输出
    
    Returns:
        (success: bool, error_message: str)
    """
    try:
        # 1. 检查模型文件是否存在
        model_file = project_dir / "net" / f"{model_name}.py"
        if not model_file.exists():
            return False, f"模型文件不存在: {model_file}"
        
        if verbose:
            print(f"  ✓ 模型文件存在: {model_file}")
        
        # 2. 尝试导入模型模块
        try:
            module = importlib.import_module(f"net.{model_name}")
        except Exception as e:
            return False, f"无法导入模型模块: {str(e)}"
        
        if verbose:
            print(f"  ✓ 模型模块导入成功")
        
        # 3. 检查 build_model 函数是否存在
        if not hasattr(module, "build_model"):
            return False, "模型模块缺少 build_model 函数"
        
        build_fn = getattr(module, "build_model")
        if not callable(build_fn):
            return False, "build_model 不是可调用函数"
        
        if verbose:
            print(f"  ✓ build_model 函数存在")
        
        # 4. 尝试获取训练配置（需要先设置 config.MODEL）
        original_model = None
        try:
            import config
            # 临时设置 MODEL
            original_model = getattr(config, "MODEL", None)
            config.MODEL = model_name
            
            from options import train_options
            opt = train_options()
            
            # 恢复原始 MODEL（如果存在）
            if original_model is not None:
                config.MODEL = original_model
        except Exception as e:
            # 恢复原始 MODEL
            if original_model is not None:
                import config
                config.MODEL = original_model
            return False, f"无法获取训练配置: {str(e)}"
        
        if verbose:
            print(f"  ✓ 训练配置获取成功")
        
        # 5. 尝试构建模型
        try:
            model = build_fn(opt)
        except Exception as e:
            error_msg = f"无法构建模型: {str(e)}"
            if verbose:
                error_msg += f"\n{traceback.format_exc()}"
            return False, error_msg
        
        if not isinstance(model, nn.Module):
            return False, f"build_model 返回的不是 nn.Module 实例，而是 {type(model)}"
        
        if verbose:
            print(f"  ✓ 模型构建成功: {type(model).__name__}")
        
        # 6. 尝试将模型移到设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = model.to(device)
            model.eval()  # 设置为评估模式
        except Exception as e:
            return False, f"无法将模型移到设备 {device}: {str(e)}"
        
        if verbose:
            print(f"  ✓ 模型已移到设备: {device}")
        
        # 7. 尝试进行一次前向传播（使用虚拟输入）
        try:
            # 根据配置获取输入尺寸（通常从 opt 中获取）
            # 默认使用常见的输入尺寸
            input_height = getattr(opt, "input_height", 256)
            input_width = getattr(opt, "input_width", 256)
            
            # 创建虚拟输入（通常是 3 通道图像）
            dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
            
            if verbose:
                print(f"  → 输入形状: {dummy_input.shape}")
            
            with torch.no_grad():
                output = model(dummy_input)
            
            # 检查输出是否有效
            if output is None:
                return False, "模型前向传播返回 None"
            
            # 处理不同的输出类型（Tensor、字典、元组等）
            if isinstance(output, torch.Tensor):
                output_shape = output.shape
            elif isinstance(output, dict):
                # 如果是字典，检查是否有有效的输出键
                if len(output) == 0:
                    return False, "模型输出是空字典"
                # 获取第一个值作为示例
                first_value = next(iter(output.values()))
                if isinstance(first_value, torch.Tensor):
                    output_shape = first_value.shape
                else:
                    output_shape = f"dict with {len(output)} keys"
            elif isinstance(output, (tuple, list)):
                # 如果是元组或列表，检查第一个元素
                if len(output) == 0:
                    return False, "模型输出是空元组/列表"
                first_item = output[0]
                if isinstance(first_item, torch.Tensor):
                    output_shape = first_item.shape
                else:
                    output_shape = f"{type(output).__name__} with {len(output)} items"
            else:
                # 其他类型也接受，但给出警告
                output_shape = str(type(output))
            
            if verbose:
                print(f"  ✓ 前向传播成功，输出类型: {type(output)}, 形状: {output_shape}")
        
        except RuntimeError as e:
            # 对于 RuntimeError（通常是维度不匹配），提供更详细的错误信息
            error_msg = f"前向传播失败: {str(e)}"
            
            # 尝试提取张量形状信息
            error_str = str(e)
            if "size" in error_str.lower() and "tensor" in error_str.lower():
                error_msg += "\n提示: 这通常是张量维度不匹配导致的。"
                error_msg += "\n建议: 检查网络中的卷积、拼接或矩阵乘法操作的输入维度。"
            
            # 总是包含完整的堆栈跟踪
            error_msg += f"\n\n完整堆栈跟踪:\n{traceback.format_exc()}"
            
            return False, error_msg
        
        except Exception as e:
            error_msg = f"前向传播失败: {str(e)}"
            error_msg += f"\n\n完整堆栈跟踪:\n{traceback.format_exc()}"
            return False, error_msg
        
        # 8. 清理
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True, "验证通过"
    
    except Exception as e:
        error_msg = f"验证过程中发生未预期的错误: {str(e)}"
        if verbose:
            error_msg += f"\n{traceback.format_exc()}"
        return False, error_msg


def validate_all_models(model_list: list, verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """
    验证所有模型
    
    Args:
        model_list: 模型名称列表
        verbose: 是否显示详细输出
    
    Returns:
        dict[model_name, (success, error_message)]
    """
    results = {}
    
    for model_name in model_list:
        if verbose:
            print(f"\n验证模型: {model_name}")
            print("-" * 60)
        
        success, message = validate_model(model_name, verbose=verbose)
        results[model_name] = (success, message)
        
        if verbose:
            status = "✓ 通过" if success else "✗ 失败"
            print(f"{status}: {message}")
        else:
            status = "✓" if success else "✗"
            print(f"{status} {model_name}: {message}")
    
    return results


def main():
    """主函数：从命令行参数读取模型列表"""
    if len(sys.argv) < 2:
        print("用法: python validate_models.py <model1> [model2] [model3] ...")
        print("或者: python validate_models.py --list <model1,model2,model3>")
        sys.exit(1)
    
    # 解析参数
    if sys.argv[1] == "--list" and len(sys.argv) > 2:
        # 从逗号分隔的字符串中解析
        model_list = [m.strip() for m in sys.argv[2].split(",") if m.strip()]
    else:
        # 从多个参数中解析
        model_list = [m.strip() for m in sys.argv[1:] if m.strip()]
    
    if not model_list:
        print("错误: 没有指定要验证的模型")
        sys.exit(1)
    
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print("=" * 60)
    print("开始验证模型...")
    print("=" * 60)
    
    results = validate_all_models(model_list, verbose=verbose)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    passed = []
    failed = []
    
    for model_name, (success, message) in results.items():
        if success:
            passed.append(model_name)
        else:
            failed.append((model_name, message))
    
    print(f"\n通过: {len(passed)}/{len(model_list)}")
    if passed:
        for model in passed:
            print(f"  ✓ {model}")
    
    if failed:
        print(f"\n失败: {len(failed)}/{len(model_list)}")
        for model, error in failed:
            print(f"  ✗ {model}: {error}")
    
    # 返回适当的退出码
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()


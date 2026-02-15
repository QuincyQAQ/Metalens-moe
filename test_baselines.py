#!/usr/bin/env python3
"""
测试脚本：测试18个Baseline网络的参数量、FPS和内存占用
"""

import os
import sys
import time
import traceback
import importlib.util
import glob
import types
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import psutil
import gc

# 抑制torch.meshgrid的警告
warnings.filterwarnings('ignore', message='.*torch.meshgrid.*', category=UserWarning)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    try:
        from utils.model_summary import get_params_flops
        MODEL_SUMMARY_AVAILABLE = True
    except ImportError:
        MODEL_SUMMARY_AVAILABLE = False


class SimpleImageDataset:
    """简单的图像数据集，用于测试"""
    def __init__(self, data_dir: str, max_samples: int = 10):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.toTensor = ToTensor()
        
        # 查找图像文件
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            # 查找val/gt目录
            val_gt = self.data_dir / 'val' / 'gt'
            if val_gt.exists():
                self.image_paths.extend(sorted(glob.glob(str(val_gt / ext))))
            
            # 也查找test目录
            test_dir = self.data_dir / 'test'
            if test_dir.exists():
                for subdir in ['meta', 'gt', 'ground_truth']:
                    test_subdir = test_dir / subdir
                    if test_subdir.exists():
                        self.image_paths.extend(sorted(glob.glob(str(test_subdir / ext))))
        
        if len(self.image_paths) == 0:
            # 如果没找到，尝试直接在根目录查找
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(sorted(glob.glob(str(self.data_dir / '**' / ext), recursive=True)))
        
        self.image_paths = self.image_paths[:max_samples]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        print(f"[Dataset] Found {len(self.image_paths)} images for testing")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.toTensor(img)
        return img_tensor, img_path


def count_parameters(model: nn.Module) -> float:
    """计算模型参数量（百万）"""
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_params / 1e6
    except Exception as e:
        print(f"[Warn] Failed to count parameters: {e}")
        return None


def measure_memory_usage(model: nn.Module, device: torch.device, input_tensor: torch.Tensor) -> Dict[str, float]:
    """测量模型推理时的内存占用（MB）"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(device) / 1e6
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e6
        memory_used = peak_memory - initial_memory
        return {
            'initial_mb': initial_memory,
            'peak_mb': peak_memory,
            'used_mb': memory_used
        }
    else:
        # CPU内存测量
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1e6
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = process.memory_info().rss / 1e6
        memory_used = peak_memory - initial_memory
        return {
            'initial_mb': initial_memory,
            'peak_mb': peak_memory,
            'used_mb': memory_used
        }


def measure_fps(model: nn.Module, device: torch.device, input_tensor: torch.Tensor, 
                warmup: int = 10, num_runs: int = 100) -> float:
    """测量模型推理的FPS"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 实际测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    fps = num_runs / elapsed
    
    return fps


def load_model_from_file(model_file: Path, device: torch.device) -> Optional[nn.Module]:
    """从文件加载模型"""
    try:
        # 动态导入模块
        module_name = f"model_module_{model_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            print(f"[Error] Failed to load spec from {model_file}")
            return None
        
        # 添加模型文件所在目录到sys.path，以便导入相对模块
        model_dir = model_file.parent
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        
        # 添加compare目录的父目录，以便导入models等模块
        compare_parent = model_dir.parent
        if str(compare_parent) not in sys.path:
            sys.path.insert(0, str(compare_parent))
        
        # 预先处理可能缺失的依赖（针对pfan等模型）
        _prehandle_missing_imports()
        
        module = importlib.util.module_from_spec(spec)
        # 避免重复导入
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except (ModuleNotFoundError, ImportError) as e:
                # 如果导入失败，尝试添加更多路径
                print(f"[Warn] Import error: {e}, trying to handle missing dependencies...")
                # 创建一个mock模块来处理缺失的依赖
                _handle_missing_imports(str(e))
                # 再次尝试执行
                try:
                    spec.loader.exec_module(module)
                except Exception as e2:
                    print(f"[Error] Failed to load module after handling imports: {e2}")
                    return None
        
        # 尝试找到模型类
        model_class = None
        model_name = model_file.stem
        
        # 常见的模型类名映射
        name_mapping = {
            'FFA': 'FFA',
            'AECRNet': 'AECRNet',
            'GridDehazeNet': 'GridDehazeNet',
            'dehazeformer': 'DehazeFormer',
            'dehamer_model': 'dehamer',  # dehamer是一个对象类，不是nn.Module，需要特殊处理
            'DIACMPN': 'DIACMPN',
            'SFSNiD': 'SFSNiD',
            'mitnet': 'MITNet',
            'pfan': 'PFAN',
            'MSBDN-DFF-v1-1': 'Net',  # 使用Net类而不是MSBDN
        }
        
        # 首先尝试映射的名称
        if model_name in name_mapping:
            mapped_name = name_mapping[model_name]
            if hasattr(module, mapped_name):
                attr = getattr(module, mapped_name)
                # 对于dehamer，它可以是object类，不是nn.Module
                if model_name == 'dehamer_model' or isinstance(attr, type):
                    model_class = attr
        
        # 如果没找到，尝试原始名称
        if model_class is None:
            possible_names = [
                model_name,
                model_name.replace('_', '').replace('-', ''),
                'AECRNet', 'FFA', 'GridDehazeNet', 'DehazeNet', 'AODNet',
                'DehazeFormer', 'Dehamer', 'DIACMPN', 'SFSNiD', 'MITNet',
                'PFAN', 'DesmokeNet', 'MSBDN', 'DCP', 'MSBDN_DFF'
            ]
            
            for name in possible_names:
                if hasattr(module, name):
                    attr = getattr(module, name)
                    if isinstance(attr, type) and issubclass(attr, nn.Module):
                        model_class = attr
                        break
        
        # 如果还没找到，查找所有nn.Module子类，优先选择主模型类
        if model_class is None:
            # 优先选择的类名（按优先级排序）
            preferred_names = ['Net', 'Model', 'Network', model_name, model_name.replace('_', '').replace('-', '')]
            
            # 首先尝试优先名称
            for preferred_name in preferred_names:
                if hasattr(module, preferred_name):
                    attr = getattr(module, preferred_name)
                    if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                        model_class = attr
                        print(f"[Info] Found model class: {preferred_name}")
                        break
            
            # 如果还没找到，查找所有nn.Module子类（排除辅助类）
            if model_class is None:
                # 排除的辅助类名
                excluded_names = ['ConvLayer', 'UpsampleConvLayer', 'ResidualBlock', 'make_dense', 
                                'DownSample', 'UpSample', 'Block', 'Group', 'PALayer', 'CALayer']
                
                for attr_name in dir(module):
                    if attr_name.startswith('_') or attr_name in excluded_names:
                        continue
                    try:
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, nn.Module) and 
                            attr != nn.Module and
                            attr_name[0].isupper()):  # 类名通常以大写字母开头
                            model_class = attr
                            print(f"[Info] Found model class: {attr_name}")
                            break
                    except:
                        continue
        
        if model_class is None:
            print(f"[Error] Could not find model class in {model_file}")
            return None
        
        # 尝试实例化模型
        model = None
        
        # 特殊处理：dehamer是一个object类，不是nn.Module
        if model_name == 'dehamer_model' and hasattr(module, 'dehamer'):
            try:
                # dehamer需要params参数，创建一个简单的mock params
                class MockParams:
                    def __init__(self):
                        self.learning_rate = 0.001
                        self.adam = [0.9, 0.999]
                params = MockParams()
                dehamer_obj = model_class(params, trainable=False)
                # dehamer对象有一个model属性
                if hasattr(dehamer_obj, 'model'):
                    model = dehamer_obj.model
                else:
                    print(f"[Error] dehamer object has no 'model' attribute")
                    return None
            except Exception as e:
                print(f"[Error] Failed to instantiate dehamer: {e}")
                return None
        else:
            # 普通nn.Module类的实例化
            instantiation_attempts = [
                # 尝试1: 无参数
                lambda: model_class(),
                # 尝试2: FFA特定参数
                lambda: model_class(gps=3, blocks=19) if 'FFA' in str(model_class) else None,
                # 尝试3: DehazeFormer特定参数
                lambda: model_class(dim=32, num_blocks=[2,2,2,2]) if 'DehazeFormer' in str(model_class) else None,
                # 尝试4: 使用inspect获取参数
                lambda: _instantiate_with_inspect(model_class),
            ]
            
            for attempt in instantiation_attempts:
                try:
                    result = attempt()
                    if result is not None:
                        model = result
                        break
                except Exception as e:
                    continue
        
        if model is None:
            print(f"[Error] Failed to instantiate {model_name}")
            return None
        
        # 确保model是nn.Module
        if not isinstance(model, nn.Module):
            print(f"[Error] Model is not an instance of nn.Module")
            return None
        
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"[Error] Failed to load model from {model_file}: {e}")
        traceback.print_exc()
        return None


def _instantiate_with_inspect(model_class):
    """使用inspect尝试实例化模型"""
    import inspect
    try:
        sig = inspect.signature(model_class.__init__)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param.default != inspect.Parameter.empty:
                params[param_name] = param.default
            else:
                # 使用常见默认值
                param_lower = param_name.lower()
                if 'channel' in param_lower or 'dim' in param_lower:
                    params[param_name] = 64
                elif 'kernel' in param_lower:
                    params[param_name] = 3
                elif 'block' in param_lower or 'num_block' in param_lower:
                    params[param_name] = 4
                elif 'gps' in param_lower or 'group' in param_lower:
                    params[param_name] = 3
                else:
                    params[param_name] = 1
        return model_class(**params)
    except:
        return None


def _prehandle_missing_imports():
    """预先处理可能缺失的导入，创建mock对象"""
    # 创建mock的models模块（用于pfan等）
    if 'models' not in sys.modules:
        models_mock = types.ModuleType('models')
        sys.modules['models'] = models_mock
    
    # 创建mock的models.seaformer模块
    if 'models.seaformer' not in sys.modules:
        seaformer_mock = types.ModuleType('models.seaformer')
        # 创建一个简单的mock类
        class MockSeaAttention(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x
        seaformer_mock.Sea_Attention = MockSeaAttention
        sys.modules['models.seaformer'] = seaformer_mock
    
    # 创建mock的residual_dense_block模块（用于GridDehazeNet）
    if 'residual_dense_block' not in sys.modules:
        rdb_mock = types.ModuleType('residual_dense_block')
        # RDB类：接受 (in_channels, num_dense_layer, growth_rate) 或类似参数
        class MockRDB(nn.Module):
            def __init__(self, in_channels, num_dense_layer=4, growth_rate=32, *args, **kwargs):
                super().__init__()
                self.in_channels = in_channels
                self.num_dense_layer = num_dense_layer
                self.growth_rate = growth_rate
                # 创建一个简单的卷积层作为占位符
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            def forward(self, x):
                return self.conv(x) + x  # 残差连接
        rdb_mock.RDB = MockRDB
        sys.modules['residual_dense_block'] = rdb_mock
    
    # 创建mock的networks模块（用于MSBDN-DFF-v1-1）
    if 'networks' not in sys.modules:
        networks_mock = types.ModuleType('networks')
        sys.modules['networks'] = networks_mock
    
    # 创建mock的networks.base_networks模块
    if 'networks.base_networks' not in sys.modules:
        base_networks_mock = types.ModuleType('networks.base_networks')
        # Encoder_MDCBlock1: 接受 (channels, num, mode='iter2')
        class MockEncoder_MDCBlock1(nn.Module):
            def __init__(self, channels, num, mode='iter2', *args, **kwargs):
                super().__init__()
                self.channels = channels
                self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            def forward(self, x):
                return self.conv(x) + x
        
        # Decoder_MDCBlock1: 接受 (channels, num, mode='iter2')
        class MockDecoder_MDCBlock1(nn.Module):
            def __init__(self, channels, num, mode='iter2', *args, **kwargs):
                super().__init__()
                self.channels = channels
                self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            def forward(self, x):
                return self.conv(x) + x
        
        base_networks_mock.Encoder_MDCBlock1 = MockEncoder_MDCBlock1
        base_networks_mock.Decoder_MDCBlock1 = MockDecoder_MDCBlock1
        sys.modules['networks.base_networks'] = base_networks_mock


def _handle_missing_imports(error_msg: str):
    """处理缺失的导入，创建mock对象"""
    # 检查错误信息中提到的缺失模块
    if 'models' in error_msg:
        # 创建mock的models模块
        if 'models' not in sys.modules:
            models_mock = types.ModuleType('models')
            sys.modules['models'] = models_mock
        
        # 如果缺少models.seaformer，创建mock
        if ('seaformer' in error_msg or 'Sea_Attention' in error_msg) and 'models.seaformer' not in sys.modules:
            seaformer_mock = types.ModuleType('models.seaformer')
            # 创建一个简单的mock类
            class MockSeaAttention(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                def forward(self, x):
                    return x
            seaformer_mock.Sea_Attention = MockSeaAttention
            sys.modules['models.seaformer'] = seaformer_mock
    
    # 处理residual_dense_block模块
    if 'residual_dense_block' in error_msg and 'residual_dense_block' not in sys.modules:
        rdb_mock = types.ModuleType('residual_dense_block')
        class MockRDB(nn.Module):
            def __init__(self, in_channels, num_dense_layer=4, growth_rate=32, *args, **kwargs):
                super().__init__()
                self.in_channels = in_channels
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            def forward(self, x):
                return self.conv(x) + x
        rdb_mock.RDB = MockRDB
        sys.modules['residual_dense_block'] = rdb_mock
    
    # 处理networks模块
    if 'networks' in error_msg:
        if 'networks' not in sys.modules:
            networks_mock = types.ModuleType('networks')
            sys.modules['networks'] = networks_mock
        
        if 'base_networks' in error_msg and 'networks.base_networks' not in sys.modules:
            base_networks_mock = types.ModuleType('networks.base_networks')
            class MockEncoder_MDCBlock1(nn.Module):
                def __init__(self, channels, num, mode='iter2', *args, **kwargs):
                    super().__init__()
                    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                def forward(self, x):
                    return self.conv(x) + x
            class MockDecoder_MDCBlock1(nn.Module):
                def __init__(self, channels, num, mode='iter2', *args, **kwargs):
                    super().__init__()
                    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
                def forward(self, x):
                    return self.conv(x) + x
            base_networks_mock.Encoder_MDCBlock1 = MockEncoder_MDCBlock1
            base_networks_mock.Decoder_MDCBlock1 = MockDecoder_MDCBlock1
            sys.modules['networks.base_networks'] = base_networks_mock
    
    # 处理swin_unet模块
    if 'swin_unet' in error_msg and 'swin_unet' not in sys.modules:
        swin_unet_mock = types.ModuleType('swin_unet')
        class MockUNet_emb(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, kernel_size=3, padding=1),
                )
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        swin_unet_mock.UNet_emb = MockUNet_emb
        sys.modules['swin_unet'] = swin_unet_mock
    
    # 处理utils模块
    if 'utils' in error_msg and 'utils' not in sys.modules:
        utils_mock = types.ModuleType('utils')
        sys.modules['utils'] = utils_mock
    
    # 创建mock的swin_unet模块（用于dehamer_model）
    if 'swin_unet' not in sys.modules:
        swin_unet_mock = types.ModuleType('swin_unet')
        # UNet_emb类：用于dehamer
        class MockUNet_emb(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                # 创建一个简单的编码-解码结构
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, kernel_size=3, padding=1),
                )
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        swin_unet_mock.UNet_emb = MockUNet_emb
        sys.modules['swin_unet'] = swin_unet_mock
    
    # 创建mock的utils模块（用于dehamer_model）
    if 'utils' not in sys.modules:
        utils_mock = types.ModuleType('utils')
        # 添加一些常用的工具函数占位符
        sys.modules['utils'] = utils_mock


def test_single_model(model_file: Path, device: torch.device, 
                     dataset: SimpleImageDataset, input_size: Tuple[int, int] = (256, 256)) -> Dict:
    """测试单个模型"""
    model_name = model_file.stem
    results = {
        'model_name': model_name,
        'file': str(model_file),
        'parameters_m': None,
        'fps': None,
        'memory_mb': None,
        'status': 'failed',
        'error': None
    }
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # 加载模型
        print(f"[1/4] Loading model...")
        model = load_model_from_file(model_file, device)
        if model is None:
            results['error'] = "Failed to load model"
            return results
        
        # 准备输入
        print(f"[2/4] Preparing input...")
        # 使用数据集中的第一张图像，或创建随机输入
        try:
            sample_img, _ = dataset[0]
            # 调整大小
            from torchvision.transforms import Resize
            resize = Resize(input_size)
            if len(sample_img.shape) == 3:
                sample_img = sample_img.unsqueeze(0)
            sample_img = resize(sample_img)
        except Exception as e:
            # 如果数据集加载失败，使用随机输入
            print(f"[Warn] Using random input: {e}")
            sample_img = torch.randn(1, 3, input_size[0], input_size[1])
        
        sample_img = sample_img.to(device)
        
        # 测试前向传播是否正常
        try:
            with torch.no_grad():
                _ = model(sample_img)
        except Exception as e:
            print(f"[Error] Model forward pass failed: {e}")
            results['error'] = f"Forward pass failed: {e}"
            return results
        
        # 计算参数量
        print(f"[3/4] Counting parameters...")
        num_params = count_parameters(model)
        results['parameters_m'] = num_params
        print(f"  Parameters: {num_params:.2f}M" if num_params else "  Parameters: N/A")
        
        # 测量内存
        print(f"[4/4] Measuring memory usage...")
        memory_info = measure_memory_usage(model, device, sample_img)
        results['memory_mb'] = memory_info['used_mb']
        print(f"  Memory used: {memory_info['used_mb']:.2f} MB")
        
        # 测量FPS
        print(f"[5/5] Measuring FPS...")
        try:
            fps = measure_fps(model, device, sample_img, warmup=5, num_runs=50)
            results['fps'] = fps
            print(f"  FPS: {fps:.2f}")
        except Exception as e:
            print(f"[Warn] FPS measurement failed: {e}")
            results['fps'] = None
        
        results['status'] = 'success'
        
        # 清理
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        results['error'] = str(e)
        print(f"[Error] {e}")
        traceback.print_exc()
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test baseline models: parameters, FPS, and memory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认设置
  python test_baselines.py
  
  # 指定数据集和设备
  python test_baselines.py --data_dir /path/to/data --device cuda
  
  # 指定输入尺寸
  python test_baselines.py --input_size 512 512
  
  # 指定输出文件
  python test_baselines.py --output my_results.csv
        """
    )
    parser.add_argument('--data_dir', type=str, 
                       default='/home/cxhlab/lqj/data/open_dataset_8_1_1_mini',
                       help='Path to dataset directory')
    parser.add_argument('--compare_dir', type=str,
                       default='/home/cxhlab/lqj/Metalens-moe/Metalens-moe/net/compare',
                       help='Path to compare directory with model files')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256],
                       metavar=('H', 'W'),
                       help='Input image size (H W)')
    parser.add_argument('--output', type=str, default='baseline_test_results.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载数据集
    print(f"\nLoading dataset from: {args.data_dir}")
    try:
        dataset = SimpleImageDataset(args.data_dir, max_samples=10)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return
    
    # 查找所有模型文件
    compare_dir = Path(args.compare_dir)
    model_files = sorted(compare_dir.glob('*.py'))
    
    if len(model_files) == 0:
        print(f"[Error] No model files found in {compare_dir}")
        return
    
    print(f"\nFound {len(model_files)} model files:")
    for f in model_files:
        print(f"  - {f.name}")
    
    # 测试所有模型
    all_results = []
    for model_file in model_files:
        results = test_single_model(
            model_file, 
            device, 
            dataset, 
            tuple(args.input_size)
        )
        all_results.append(results)
    
    # 保存结果
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    import csv
    output_path = Path(args.output)
    fieldnames = ['model_name', 'parameters_m', 'fps', 'memory_mb', 'status', 'error']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            # 只写入fieldnames中定义的字段
            row = {k: v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
            # 打印摘要
            if r['status'] == 'success':
                print(f"{r['model_name']:30s} | "
                      f"Params: {r['parameters_m']:8.2f}M | "
                      f"FPS: {r['fps']:8.2f} | "
                      f"Memory: {r['memory_mb']:8.2f}MB")
            else:
                print(f"{r['model_name']:30s} | FAILED: {r.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()


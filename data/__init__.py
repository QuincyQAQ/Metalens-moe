import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils.image_utils import crop_img


class SimpleTestDataset(Dataset):
    """简单的测试数据集，返回 [inputs, targets, filename] 格式的数据"""
    
    def __init__(self, val_dir, target='target', img_options=None):
        super().__init__()
        self.val_dir = val_dir
        self.target = target
        self.img_options = img_options or {}
        self.toTensor = ToTensor()
        
        # 尝试不同的输入/目标目录名称
        input_dirs = ['meta', 'lr', 'input']
        target_dirs = [target, 'ground_truth', 'gt', 'target']
        
        # 查找输入图像目录
        self.input_dir = None
        for input_dir_name in input_dirs:
            candidate = os.path.join(val_dir, input_dir_name)
            if os.path.exists(candidate):
                self.input_dir = candidate
                break
        
        # 查找目标图像目录
        self.target_dir = None
        for target_dir_name in target_dirs:
            candidate = os.path.join(val_dir, target_dir_name)
            if os.path.exists(candidate):
                self.target_dir = candidate
                break
        
        if self.input_dir is None or self.target_dir is None:
            raise ValueError(f"Cannot find input/target directories in {val_dir}. "
                           f"Tried input dirs: {input_dirs}, target dirs: {target_dirs}")
        
        # 获取图像文件列表
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, "*.png")))
        self.target_files = sorted(glob.glob(os.path.join(self.target_dir, "*.png")))
        
        if len(self.input_files) == 0:
            raise ValueError(f"No input images found in {self.input_dir}")
        if len(self.target_files) == 0:
            raise ValueError(f"No target images found in {self.target_dir}")
        if len(self.input_files) != len(self.target_files):
            raise ValueError(f"Input/Target count mismatch: {len(self.input_files)} vs {len(self.target_files)}")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        target_path = self.target_files[idx]
        
        # 读取图像
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        # 转换为 numpy 数组
        input_np = np.array(input_img)
        target_np = np.array(target_img)
        
        # 裁剪到 base 的倍数（通常是 16）
        input_np = crop_img(input_np, base=16)
        target_np = crop_img(target_np, base=16)
        
        # 转换为 tensor
        input_tensor = self.toTensor(input_np)
        target_tensor = self.toTensor(target_np)
        
        # 获取文件名（只使用文件名，不包含路径）
        filename = os.path.basename(input_path)
        
        # 返回 [inputs, targets, filename] 格式
        return input_tensor, target_tensor, filename


def get_test_data(val_dir, target='target', img_options=None):
    """创建测试数据集
    
    Args:
        val_dir: 测试数据目录（应包含 meta/ 或 lr/ 和 ground_truth/ 或 gt/ 子目录）
        target: 目标目录名称（默认 'target'）
        img_options: 图像选项字典（目前未使用，保留以兼容现有代码）
    
    Returns:
        SimpleTestDataset 实例
    """
    return SimpleTestDataset(val_dir, target=target, img_options=img_options)


#!/usr/bin/env python3
"""
创建 mini 版本的数据集用于快速测试
从 open_dataset_8_1_1 中随机选择少量样本创建 open_dataset_8_1_1_mini
"""
import os
import shutil
import random
from pathlib import Path
import glob

def create_mini_dataset(source_dir, target_dir, num_samples=50):
    """
    创建 mini 数据集
    确保 lr 和 gt 中的文件一一对应
    
    Args:
        source_dir: 源数据集目录
        target_dir: 目标 mini 数据集目录
        num_samples: 每个子集选择的样本数量
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise ValueError(f"源数据集不存在: {source_dir}")
    
    # 如果目标目录已存在，先删除
    if target_path.exists():
        print(f"删除已存在的目标目录: {target_dir}")
        shutil.rmtree(target_path)
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在从 {source_dir} 创建 mini 数据集到 {target_dir}")
    print(f"每个子集选择 {num_samples} 个样本")
    
    # 处理训练集和测试集
    for split in ['train', 'test']:
        split_source = source_path / split
        split_target = target_path / split
        
        if not split_source.exists():
            print(f"警告: {split_source} 不存在，跳过")
            continue
        
        print(f"\n处理 {split} 集...")
        
        # 获取 lr 和 gt 目录
        lr_source = split_source / 'lr'
        gt_source = split_source / 'gt'
        lr_target = split_target / 'lr'
        gt_target = split_target / 'gt'
        
        if not lr_source.exists() or not gt_source.exists():
            print(f"警告: {split_source} 中缺少 lr 或 gt 目录，跳过")
            continue
        
        # 获取 lr 目录中的所有图片文件（作为基准）
        lr_files = sorted(glob.glob(str(lr_source / "*.png"))) + \
                   sorted(glob.glob(str(lr_source / "*.jpg"))) + \
                   sorted(glob.glob(str(lr_source / "*.PNG"))) + \
                   sorted(glob.glob(str(lr_source / "*.JPG")))
        
        if len(lr_files) == 0:
            print(f"警告: {lr_source} 中没有图片文件，跳过")
            continue
        
        # 只选择在 lr 和 gt 中都存在的文件对
        valid_pairs = []
        for lr_file in lr_files:
            lr_name = Path(lr_file).name
            gt_file = gt_source / lr_name
            if gt_file.exists():
                valid_pairs.append((lr_file, str(gt_file)))
        
        if len(valid_pairs) == 0:
            print(f"警告: 没有找到匹配的 lr/gt 文件对，跳过")
            continue
        
        print(f"  找到 {len(valid_pairs)} 个有效的文件对")
        
        # 随机选择样本
        num_to_select = min(num_samples, len(valid_pairs))
        selected_pairs = random.sample(valid_pairs, num_to_select)
        
        # 创建目标目录
        lr_target.mkdir(parents=True, exist_ok=True)
        gt_target.mkdir(parents=True, exist_ok=True)
        
        # 复制文件对
        copied_count = 0
        for lr_file, gt_file in selected_pairs:
            try:
                lr_src = Path(lr_file)
                gt_src = Path(gt_file)
                lr_dst = lr_target / lr_src.name
                gt_dst = gt_target / gt_src.name
                
                # 使用 copyfile 而不是 copy2，避免复制元数据可能的问题
                shutil.copyfile(lr_src, lr_dst)
                shutil.copyfile(gt_src, gt_dst)
                
                # 验证文件大小
                if lr_src.stat().st_size != lr_dst.stat().st_size:
                    raise ValueError(f"文件大小不匹配: {lr_file}")
                if gt_src.stat().st_size != gt_dst.stat().st_size:
                    raise ValueError(f"文件大小不匹配: {gt_file}")
                
                copied_count += 1
            except Exception as e:
                print(f"  警告: 复制文件对失败 {Path(lr_file).name}: {e}")
        
        print(f"  {split}: 成功复制 {copied_count}/{num_to_select} 个文件对")
    
    # 验证最终结果
    train_lr_count = len(list((target_path / 'train' / 'lr').glob('*.*'))) if (target_path / 'train' / 'lr').exists() else 0
    train_gt_count = len(list((target_path / 'train' / 'gt').glob('*.*'))) if (target_path / 'train' / 'gt').exists() else 0
    test_lr_count = len(list((target_path / 'test' / 'lr').glob('*.*'))) if (target_path / 'test' / 'lr').exists() else 0
    test_gt_count = len(list((target_path / 'test' / 'gt').glob('*.*'))) if (target_path / 'test' / 'gt').exists() else 0
    
    print(f"\n✅ Mini 数据集创建完成: {target_dir}")
    print(f"   训练集: lr={train_lr_count}, gt={train_gt_count}")
    print(f"   测试集: lr={test_lr_count}, gt={test_gt_count}")
    
    if train_lr_count != train_gt_count or test_lr_count != test_gt_count:
        print(f"⚠️  警告: lr 和 gt 文件数量不匹配！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="创建 mini 数据集")
    parser.add_argument(
        "--source",
        type=str,
        default="/media/wsqlab/more/lqj/data/open_dataset_8_1_1",
        help="源数据集目录"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/media/wsqlab/more/lqj/data/open_dataset_8_1_1_mini",
        help="目标 mini 数据集目录"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="每个子集选择的样本数量（默认: 50）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建 mini 数据集
    create_mini_dataset(args.source, args.target, args.num_samples)


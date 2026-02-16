#!/usr/bin/env python3
"""测试脚本：检查三个数据集的数量是否正确"""

import os
import glob
from pathlib import Path

def check_dataset(data_dir, dataset_name):
    """检查单个数据集的数量"""
    print(f"\n{'='*60}")
    print(f"检查数据集: {dataset_name}")
    print(f"路径: {data_dir}")
    print(f"{'='*60}")
    
    # 检查是否存在嵌套结构
    nested_train = os.path.join(data_dir, dataset_name, "train")
    generic_train = os.path.join(data_dir, "train")
    
    if os.path.exists(nested_train):
        lr_dir = os.path.join(nested_train, "lr")
        gt_dir = os.path.join(nested_train, "gt")
        print(f"使用嵌套结构: {nested_train}")
    elif os.path.exists(generic_train):
        lr_dir = os.path.join(generic_train, "lr")
        gt_dir = os.path.join(generic_train, "gt")
        print(f"使用通用结构: {generic_train}")
    else:
        print(f"❌ 错误: 找不到train目录")
        return False
    
    if not os.path.exists(lr_dir):
        print(f"❌ 错误: 找不到lr目录: {lr_dir}")
        return False
    if not os.path.exists(gt_dir):
        print(f"❌ 错误: 找不到gt目录: {gt_dir}")
        return False
    
    # 获取所有文件
    lr_files = {os.path.basename(x): x for x in glob.glob(os.path.join(lr_dir, "*.png"))}
    gt_files = {os.path.basename(x): x for x in glob.glob(os.path.join(gt_dir, "*.png"))}
    
    lr_count = len(lr_files)
    gt_count = len(gt_files)
    common_names = set(lr_files.keys()) & set(gt_files.keys())
    matched_count = len(common_names)
    
    print(f"LR文件数量: {lr_count}")
    print(f"GT文件数量: {gt_count}")
    print(f"匹配的对数: {matched_count}")
    
    if lr_count == gt_count == matched_count:
        print(f"✅ 数量正确: 所有文件都匹配")
        return True
    else:
        print(f"⚠️  数量不匹配:")
        if lr_count != matched_count:
            missing_in_gt = set(lr_files.keys()) - set(gt_files.keys())
            print(f"  - LR中有{lr_count - matched_count}个文件在GT中不存在:")
            for name in sorted(list(missing_in_gt))[:5]:
                print(f"    {name}")
            if len(missing_in_gt) > 5:
                print(f"    ... 还有{len(missing_in_gt) - 5}个文件")
        
        if gt_count != matched_count:
            missing_in_lr = set(gt_files.keys()) - set(lr_files.keys())
            print(f"  - GT中有{gt_count - matched_count}个文件在LR中不存在:")
            for name in sorted(list(missing_in_lr))[:5]:
                print(f"    {name}")
            if len(missing_in_lr) > 5:
                print(f"    ... 还有{len(missing_in_lr) - 5}个文件")
        
        return False

def main():
    base_dir = "/media/wsqlab/backup/lqj/data"
    
    datasets = [
        ("Endovis17_8_1_1", "Endovis17_8_1_1"),
        ("CVC_8_1_1", "CVC_8_1_1"),
        ("Kvasir_SEG_8_1_1", "Kvasir_SEG_8_1_1"),
    ]
    
    results = []
    for data_dir_name, dataset_name in datasets:
        data_dir = os.path.join(base_dir, data_dir_name)
        if os.path.exists(data_dir):
            result = check_dataset(data_dir, dataset_name)
            results.append((dataset_name, result))
        else:
            print(f"\n❌ 数据集目录不存在: {data_dir}")
            results.append((dataset_name, False))
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    for dataset_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{dataset_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\n总体结果: {'✅ 所有数据集数量正确' if all_passed else '❌ 部分数据集数量不匹配'}")

if __name__ == "__main__":
    main()



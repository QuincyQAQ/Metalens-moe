#!/usr/bin/env python3
"""
分析 Endovis17 和 Kvasir_SEG 数据集图片的频域、色差等特点
"""
import os
import numpy as np
import cv2
from pathlib import Path
from scipy import fftpack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 数据集路径
DATA_DIR = Path("/media/wsqlab/backup/lqj/data")

def analyze_image(img, name="image"):
    """分析单张图片的频域和空域特征"""
    results = {}
    
    # 转换为灰度图进行频域分析
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # 1. 频域分析
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    magnitude = np.abs(fshift)
    
    # 计算频域能量分布
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    
    # 低频、中频、高频能量比例
    radius_low = min(h, w) // 8
    radius_mid = min(h, w) // 4
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    low_mask = dist <= radius_low
    mid_mask = (dist > radius_low) & (dist <= radius_mid)
    high_mask = dist > radius_mid
    
    total_energy = np.sum(magnitude**2) + 1e-10
    results['low_freq_ratio'] = np.sum(magnitude[low_mask]**2) / total_energy
    results['mid_freq_ratio'] = np.sum(magnitude[mid_mask]**2) / total_energy
    results['high_freq_ratio'] = np.sum(magnitude[high_mask]**2) / total_energy
    
    # 2. 颜色通道分析 (如果是彩色图)
    if len(img.shape) == 3:
        # 计算各通道的标准差（反映对比度）
        results['r_std'] = np.std(img[:, :, 0])
        results['g_std'] = np.std(img[:, :, 1])
        results['b_std'] = np.std(img[:, :, 2])
        
        # 色差分析：计算各通道与灰度的差异
        gray_float = gray.astype(np.float32)
        r_diff = np.mean(np.abs(img[:, :, 0].astype(np.float32) - gray_float))
        g_diff = np.mean(np.abs(img[:, :, 1].astype(np.float32) - gray_float))
        b_diff = np.mean(np.abs(img[:, :, 2].astype(np.float32) - gray_float))
        results['r_chromatic'] = r_diff / (gray.mean() + 1e-10)
        results['g_chromatic'] = g_diff / (gray.mean() + 1e-10)
        results['b_chromatic'] = b_diff / (gray.mean() + 1e-10)
        
        # 色差强度 (R-G, R-B, G-B 差异)
        rg_diff = np.mean(np.abs(img[:, :, 0].astype(np.float32) - img[:, :, 1].astype(np.float32)))
        rb_diff = np.mean(np.abs(img[:, :, 0].astype(np.float32) - img[:, :, 2].astype(np.float32)))
        gb_diff = np.mean(np.abs(img[:, :, 1].astype(np.float32) - img[:, :, 2].astype(np.float32)))
        results['chromatic_aberration'] = (rg_diff + rb_diff + gb_diff) / 3 / 255.0
        
    # 3. 边缘强度分析
    edges = cv2.Canny(gray, 50, 150)
    results['edge_ratio'] = np.sum(edges > 0) / (h * w)
    
    # 4. 局部对比度（使用 Laplacian 方差）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    results['laplacian_var'] = laplacian.var()
    
    # 5. 图像亮度
    results['mean_intensity'] = gray.mean() / 255.0
    
    return results

def analyze_dataset(dataset_name, split="val", num_samples=10):
    """分析整个数据集的特征"""
    dataset_path = DATA_DIR / dataset_name
    
    gt_dir = dataset_path / split / "gt"
    lr_dir = dataset_path / split / "lr"
    
    if not gt_dir.exists():
        print(f"  Warning: {gt_dir} not found")
        return None
    
    gt_files = sorted(list(gt_dir.glob("*.png")))[:num_samples]
    lr_files = sorted(list(lr_dir.glob("*.png")))[:num_samples]
    
    gt_results = []
    lr_results = []
    
    print(f"  Analyzing {len(gt_files)} images from {split}/gt...")
    for gt_file in gt_files:
        img = cv2.imread(str(gt_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = analyze_image(img, gt_file.name)
        gt_results.append(results)
    
    print(f"  Analyzing {len(lr_files)} images from {split}/lr...")
    for lr_file in lr_files:
        img = cv2.imread(str(lr_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = analyze_image(img, lr_file.name)
        lr_results.append(results)
    
    return gt_results, lr_results

def print_analysis(dataset_name, split="val"):
    """打印分析结果"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    results = analyze_dataset(dataset_name, split)
    if results is None:
        return
    
    gt_results, lr_results = results
    
    print(f"\n--- Ground Truth (GT) ---")
    avg_results = {}
    for key in gt_results[0].keys():
        avg_results[key] = np.mean([r[key] for r in gt_results])
        print(f"  {key}: {avg_results[key]:.4f}")
    
    print(f"\n--- Low Resolution (LR/Degraded) ---")
    lr_avg = {}
    for key in lr_results[0].keys():
        lr_avg[key] = np.mean([r[key] for r in lr_results])
        print(f"  {key}: {lr_avg[key]:.4f}")
    
    print(f"\n--- GT vs LR 差异 ---")
    for key in gt_results[0].keys():
        diff = lr_avg[key] - avg_results[key]
        pct = diff / (avg_results[key] + 1e-10) * 100
        print(f"  {key}: {diff:+.4f} ({pct:+.2f}%)")

# 分析各数据集
print("开始分析数据集...")
print_analysis("Endovis17_8_1_1", "val")
print_analysis("Kvasir_SEG_8_1_1", "val")
print_analysis("CVC_8_1_1", "val")

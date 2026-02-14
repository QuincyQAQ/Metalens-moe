#!/usr/bin/env python3
"""
训练后自动测试脚本
从 config.py 读取配置，找到最新的 checkpoint，然后调用 test.py 的 main 函数
"""
import os
import sys
import pathlib
import argparse
from options import train_options
from test import main as test_main

def find_latest_checkpoint(ckpt_dir):
    """找到最新的 checkpoint"""
    ckpt_path = pathlib.Path(ckpt_dir)
    if not ckpt_path.exists():
        return None
    
    # 优先级：best_psnr_ssim > best_psnr > last.ckpt
    best_psnr_ssim = sorted(ckpt_path.glob("best_psnr_ssim*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if best_psnr_ssim:
        return str(best_psnr_ssim[0].resolve())
    
    best_psnr = sorted(ckpt_path.glob("best_psnr-epoch=*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if best_psnr:
        return str(best_psnr[0].resolve())
    
    last_ckpt = ckpt_path / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt.resolve())
    
    return None

def main():
    # 从 config.py 读取配置
    opt = train_options()
    
    # 找到最新的 checkpoint
    run_id = os.environ.get("MOCEIRV2_RUN_ID", None)
    if run_id is None:
        print("[Error] MOCEIRV2_RUN_ID environment variable is not set")
        sys.exit(1)
    
    experiment_dir = getattr(opt, "experiment_dir", "../experiment")
    ckpt_dir = pathlib.Path(experiment_dir) / run_id / "checkpoints"
    
    ckpt_path = find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        print(f"[Error] No checkpoint found in {ckpt_dir}")
        sys.exit(1)
    
    print(f"[Test] Using checkpoint: {ckpt_path}")
    
    # 准备测试配置
    test_opt = argparse.Namespace(
        ckpt_path=ckpt_path,
        model=None,
        data_file_dir=opt.data_file_dir,  # 使用训练时的 data_file_dir
        trainset=opt.trainset,  # 使用训练时的 trainset
        benchmarks=opt.benchmarks,  # 使用训练时的 benchmarks
        de_type=opt.de_type,  # 使用训练时的 de_type
        patch_size=getattr(opt, "patch_size", 256),
        batch_size=1,
        save_results=getattr(opt, "save_results", False),
        precision=getattr(opt, "precision", "fp16"),
        full_res_eval=getattr(opt, "full_res_eval", True),
    )
    
    # 调用 test.py 的 main 函数
    test_main(test_opt)

if __name__ == '__main__':
    main()


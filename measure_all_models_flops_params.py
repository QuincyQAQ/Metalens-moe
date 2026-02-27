#!/usr/bin/env python3
"""
在 MoCE-IR (Metalens-moe) 环境下，统一统计 net/ 目录中所有模型的 FLOPs 和参数量。

不修改任何模型代码，只是：
1. 遍历 net/*.py 中带有 build_model(opt) 的模型文件
2. 设置 config.MODEL 为对应名字
3. 通过 options.train_options() 构建 opt
4. 动态导入 net.<MODEL>.build_model，构建模型
5. 复用 train._calculate_model_complexity 计算 GFLOPs 和参数量（百万）
6. 将结果打印到终端，并写入 CSV 文件 all_models_flops_params.csv
"""

import os
import sys
import csv
import importlib
from pathlib import Path

import torch

# 保证可以直接以 python measure_all_models_flops_params.py 运行
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

import config  # noqa: E402
from options import train_options  # noqa: E402
from train import _calculate_model_complexity  # noqa: E402


def _iter_model_names_from_net_dir(net_dir: Path):
    """遍历 net 目录中所有包含 build_model 的模型文件，返回模型名列表（不带 .py）"""
    model_names = []
    for py_path in sorted(net_dir.glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        # 只选择包含 build_model 的文件，避免比较杂的工具文件
        try:
            text = py_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "def build_model" not in text:
            continue
        model_names.append(py_path.stem)
    return model_names


def _build_model_for_name(model_name: str):
    """给定模型名，设置 config.MODEL 并构建模型实例"""
    # 动态修改 config.MODEL（不修改其它配置）
    setattr(config, "MODEL", model_name)

    # 构建 opt
    opt = train_options()

    # 动态导入 net.<MODEL>.build_model
    module_name = f"net.{model_name}"
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"[Skip] 无法导入模块 {module_name}: {e}")
        return None

    if not hasattr(module, "build_model"):
        print(f"[Skip] 模块 {module_name} 中未找到 build_model(opt)")
        return None

    build_model = getattr(module, "build_model")

    try:
        model = build_model(opt)
    except Exception as e:
        print(f"[Skip] 构建模型 {model_name} 失败: {e}")
        return None

    return model


def main():
    project_dir = CUR_DIR
    net_dir = project_dir / "net"

    if not net_dir.exists():
        print(f"[Error] net 目录不存在: {net_dir}")
        return

    model_names = _iter_model_names_from_net_dir(net_dir)
    if not model_names:
        print("[Error] 在 net/ 中没有找到包含 build_model 的模型文件")
        return

    print("=" * 80)
    print("将在 MoCE-IR 环境下为以下模型重新计算 FLOPs / 参数量：")
    for name in model_names:
        print(f"  - {name}")
    print("=" * 80)

    results = []

    for model_name in model_names:
        print(f"\n{'-' * 80}")
        print(f"[Model] {model_name}")
        print("-" * 80)

        model = _build_model_for_name(model_name)
        if model is None:
            results.append(
                dict(
                    model=model_name,
                    gflops="",
                    params_m="",
                    note="build_failed",
                )
            )
            continue

        # 将模型移动到可用设备（只用于计算复杂度，不训练）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        try:
            gflops, params_m = _calculate_model_complexity(
                model, input_size=(1, 3, 256, 256)
            )
        except Exception as e:
            print(f"[Warn] 计算复杂度失败: {e}")
            gflops, params_m = None, None

        if gflops is not None:
            print(f"  GFLOPs (1x3x256x256): {gflops:.6f}")
        else:
            print("  GFLOPs: 计算失败")

        if params_m is not None:
            print(f"  Params (M): {params_m:.6f}")
        else:
            print("  Params: 计算失败")

        results.append(
            dict(
                model=model_name,
                gflops=f"{gflops:.6f}" if gflops is not None else "",
                params_m=f"{params_m:.6f}" if params_m is not None else "",
                note="ok" if (gflops is not None and params_m is not None) else "partial_or_fail",
            )
        )

        # 及时释放显存
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    # 写入 CSV
    csv_path = project_dir / "all_models_flops_params.csv"
    fieldnames = ["model", "gflops", "params_m", "note"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("\n" + "=" * 80)
    print(f"所有模型的 FLOPs / 参数量结果已写入: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()



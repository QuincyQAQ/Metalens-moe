#!/usr/bin/env bash
# Bark 推送通知函数
# 使用方法：在 train.sh 开头 source 这个文件，然后在脚本末尾调用 bark_notify
# 配置从 config.py 读取：ENABLE_BARK_NOTIFICATION 和 BARK_API_URL

# 记录脚本开始时间
SCRIPT_START_TIME=$(date +%s)

# 从 config.py 读取配置的函数
load_bark_config() {
    # 尝试找到 config.py 的路径
    local config_path=""
    local script_dir=""
    
    # 尝试获取 train.sh 所在目录（通过 BASH_SOURCE 或 $0）
    if [ -n "${BASH_SOURCE[0]:-}" ]; then
        script_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")
    elif [ -n "$0" ] && [ "$0" != "bash" ]; then
        script_dir=$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")
    fi
    
    # 尝试多个可能的 config.py 路径（优先当前工作目录，因为 train.sh 通常在 Metalens-moe/Metalens-moe/ 目录下运行）
    local possible_config_paths=(
        "$(pwd)/config.py"  # 最优先：当前工作目录（train.sh 通常在这里运行）
        "$(pwd)/Metalens-moe/Metalens-moe/config.py"
        "$(pwd)/../Metalens-moe/Metalens-moe/config.py"
        "$HOME/lqj/Metalens-moe/Metalens-moe/config.py"
        "${script_dir}/../Metalens-moe/Metalens-moe/config.py"
        "${script_dir}/../../Metalens-moe/Metalens-moe/config.py"
    )
    
    for path in "${possible_config_paths[@]}"; do
        if [ -f "$path" ]; then
            config_path="$path"
            break
        fi
    done
    
    # 如果找到了 config.py，读取配置
    if [ -n "$config_path" ]; then
        # 读取 ENABLE_BARK_NOTIFICATION（默认为 True）
        ENABLE_BARK_NOTIFICATION=$(cd "$(dirname "$config_path")" && python3 -c "import config; print(getattr(config, 'ENABLE_BARK_NOTIFICATION', True))" 2>/dev/null || echo "True")
        
        # 读取 BARK_API_URL（默认为空）
        BARK_API_URL=$(cd "$(dirname "$config_path")" && python3 -c "import config; print(getattr(config, 'BARK_API_URL', ''))" 2>/dev/null || echo "")
        
        # 如果配置为 False 或空，禁用通知
        if [ "$ENABLE_BARK_NOTIFICATION" != "True" ] || [ -z "$BARK_API_URL" ]; then
            ENABLE_BARK_NOTIFICATION="False"
        fi
    else
        # 如果找不到 config.py，默认禁用通知
        echo "警告: 未找到 config.py，禁用 Bark 通知"
        ENABLE_BARK_NOTIFICATION="False"
        BARK_API_URL=""
    fi
    
    # 导出 BARK_URL 供其他函数使用
    export BARK_URL="$BARK_API_URL"
}

# 初始化时加载配置
load_bark_config

# 获取 GPU 型号的函数
get_gpu_info() {
    # 尝试使用 nvidia-smi 获取 GPU 信息
    if command -v nvidia-smi &> /dev/null; then
        # 获取所有 GPU 的型号，去重并格式化
        local gpu_models=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//' | sed 's/,/, /g')
        if [ -n "$gpu_models" ]; then
            echo "$gpu_models"
            return 0
        fi
    fi
    
    # 如果 nvidia-smi 不可用，尝试其他方法
    if [ -f /proc/driver/nvidia/version ]; then
        echo "NVIDIA GPU (型号未知)"
        return 0
    fi
    
    # 如果都没有，返回未知
    echo "GPU 型号未知"
    return 1
}

# 发送 Bark 通知的函数
bark_notify() {
    # 如果通知被禁用，直接返回
    if [ "$ENABLE_BARK_NOTIFICATION" != "True" ] || [ -z "$BARK_URL" ]; then
        return 0
    fi
    
    local status="$1"  # "成功" 或 "失败"
    local message="$2"  # 自定义消息
    
    # 计算运行时间
    local end_time=$(date +%s)
    local duration=$((end_time - SCRIPT_START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    # 格式化运行时间
    local time_str=""
    if [ $hours -gt 0 ]; then
        time_str="${hours}小时${minutes}分${seconds}秒"
    elif [ $minutes -gt 0 ]; then
        time_str="${minutes}分${seconds}秒"
    else
        time_str="${seconds}秒"
    fi
    
    # 构建通知内容
    local title="${status} - 训练完成"
    local content="${message}
运行时间: ${time_str}"
    
    # URL 编码函数（使用 Python，更可靠）
    url_encode() {
        local string="$1"
        # 使用 Python 进行 URL 编码（通过 stdin 传递，避免引号问题）
        echo "$string" | python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read().strip()))" 2>/dev/null || \
        echo "$string" | python -c "import sys, urllib; print(urllib.quote(sys.stdin.read().strip()))" 2>/dev/null || \
        # 如果 Python 不可用，使用 sed 进行基本编码
        echo "$string" | sed 's/ /%20/g; s/:/%3A/g; s/\//%2F/g; s/?/%3F/g; s/#/%23/g; s/\[/%5B/g; s/\]/%5D/g; s/@/%40/g; s/!/%21/g; s/\$/%24/g; s/&/%26/g; s/'\''/%27/g; s/(/%28/g; s/)/%29/g; s/*/%2A/g; s/+/%2B/g; s/,/%2C/g; s/;/%3B/g; s/=/%3D/g; s/%/%25/g'
    }
    
    # 对标题和内容进行 URL 编码
    local encoded_title=$(url_encode "$title")
    local encoded_content=$(url_encode "$content")
    
    # 发送通知（使用 curl，如果没有则尝试 wget）
    if command -v curl &> /dev/null; then
        curl -s "${BARK_URL}/${encoded_title}/${encoded_content}" > /dev/null 2>&1
    elif command -v wget &> /dev/null; then
        wget -q -O /dev/null "${BARK_URL}/${encoded_title}/${encoded_content}" 2>&1
    else
        echo "错误: 未找到 curl 或 wget，无法发送通知"
        return 1
    fi
}

# 设置退出时自动发送通知
# 训练成功时自动查找并发送测试结果，如果找不到则发送简单通知
bark_notify_on_exit() {
    # 如果通知被禁用，直接返回
    if [ "$ENABLE_BARK_NOTIFICATION" != "True" ] || [ -z "$BARK_URL" ]; then
        return 0
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        # 尝试查找 test.csv 文件
        # 首先尝试从 config.py 获取 TEST_DIR（从 train.sh 所在目录或当前目录查找 config.py）
        local csv_path=""
        local script_dir=""
        
        # 尝试获取 train.sh 所在目录（通过 BASH_SOURCE 或 $0）
        if [ -n "${BASH_SOURCE[0]:-}" ]; then
            script_dir=$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")
        elif [ -n "$0" ] && [ "$0" != "bash" ]; then
            script_dir=$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")
        fi
        
        # 尝试从 config.py 获取 TEST_DIR
        local test_dir=""
        local config_path=""
        
        # 尝试找到 config.py（优先当前工作目录）
        local possible_config_paths=(
            "$(pwd)/config.py"  # 最优先：当前工作目录（train.sh 通常在这里运行）
            "$(pwd)/Metalens-moe/Metalens-moe/config.py"
            "$(pwd)/../Metalens-moe/Metalens-moe/config.py"
            "$HOME/lqj/Metalens-moe/Metalens-moe/config.py"
            "${script_dir}/../Metalens-moe/Metalens-moe/config.py"
            "${script_dir}/../../Metalens-moe/Metalens-moe/config.py"
        )
        
        for path in "${possible_config_paths[@]}"; do
            if [ -f "$path" ]; then
                config_path="$path"
                break
            fi
        done
        
        if [ -n "$config_path" ]; then
            test_dir=$(cd "$(dirname "$config_path")" && python3 -c 'import config; from pathlib import Path; p = Path(getattr(config, "TEST_DIR", "../test")); print(str(p.resolve()) if not p.is_absolute() else str(p))' 2>/dev/null || echo "")
        fi
        
        # 如果没找到，尝试从当前目录查找
        if [ -z "$test_dir" ]; then
            test_dir=$(python3 -c 'import sys; sys.path.insert(0, "."); import config; from pathlib import Path; p = Path(getattr(config, "TEST_DIR", "../test")); print(str(p.resolve()) if not p.is_absolute() else str(p))' 2>/dev/null || echo "../test")
        fi
        
        # 构建可能的 CSV 路径并检查
        local possible_paths=(
            "${test_dir}/test.csv"
            "$(pwd)/${test_dir}/test.csv"
            "$(pwd)/../test/test.csv"
            "${script_dir}/../test/test.csv"
            "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)/test/test.csv"
        )
        
        for path in "${possible_paths[@]}"; do
            if [ -f "$path" ]; then
                csv_path="$path"
                break
            fi
        done
        
        # 如果找到了 test.csv，发送包含测试结果的通知
        if [ -f "$csv_path" ]; then
            echo "找到测试结果文件: $csv_path，发送详细通知..."
            local gpu_info=$(get_gpu_info)
            if ! bark_notify_test_results "$csv_path" "" "" "" "$gpu_info"; then
                # 如果解析 CSV 失败，发送简单的错误通知
                echo "解析 CSV 文件失败，发送错误通知..."
                local end_time=$(date +%s)
                local duration=$((end_time - SCRIPT_START_TIME))
                local hours=$((duration / 3600))
                local minutes=$(((duration % 3600) / 60))
                local seconds=$((duration % 60))
                local time_str=""
                if [ $hours -gt 0 ]; then
                    time_str="${hours}小时${minutes}分${seconds}秒"
                elif [ $minutes -gt 0 ]; then
                    time_str="${minutes}分${seconds}秒"
                else
                    time_str="${seconds}秒"
                fi
                bark_notify "❌出错" "训练完成，但解析测试结果文件失败
文件路径: $csv_path
运行时间: ${time_str}"
            fi
        else
            # 如果没找到，发送简单的成功通知
            echo "未找到测试结果文件，发送简单通知..."
            local gpu_info=$(get_gpu_info)
            
            # 计算运行时间
            local end_time=$(date +%s)
            local duration=$((end_time - SCRIPT_START_TIME))
            local hours=$((duration / 3600))
            local minutes=$(((duration % 3600) / 60))
            local seconds=$((duration % 60))
            
            # 格式化运行时间
            local time_str=""
            if [ $hours -gt 0 ]; then
                time_str="${hours}小时${minutes}分${seconds}秒"
            elif [ $minutes -gt 0 ]; then
                time_str="${minutes}分${seconds}秒"
            else
                time_str="${seconds}秒"
            fi
            
            # 构建通知内容
            local message="训练完成
GPU: ${gpu_info}
运行时间: ${time_str}"
            
            # 发送通知
            bark_notify "✅成功" "$message"
        fi
    else
        bark_notify "❌失败" "训练脚本执行失败，退出码: ${exit_code}"
    fi
}

# 发送测试结果通知的函数
# 用法: bark_notify_test_results <csv_file> [model_name] [gflops] [parameters] [gpu_info]
# 如果未提供 model_name/gflops/parameters，会从 CSV 文件中自动提取
bark_notify_test_results() {
    # 如果通知被禁用，直接返回
    if [ "$ENABLE_BARK_NOTIFICATION" != "True" ] || [ -z "$BARK_URL" ]; then
        return 0
    fi
    
    local csv_file="$1"
    local model_name="$2"
    local gflops="$3"
    local parameters="$4"
    local gpu_info="$5"
    
    if [ ! -f "$csv_file" ]; then
        echo "错误: CSV 文件不存在: $csv_file"
        return 1
    fi
    
    # 获取本次训练的数据集列表（从环境变量）
    local trained_datasets_env="${MOCEIR_TRAINED_DATASETS:-}"
    
    # 使用 Python 解析 CSV 并格式化输出
    local result=$(python3 << EOF
import csv
import sys
import os
from collections import defaultdict

csv_file = "$csv_file"
model_name = "$model_name" if "$model_name" else None
gflops = "$gflops" if "$gflops" else None
parameters = "$parameters" if "$parameters" else None
gpu_info = "$gpu_info" if "$gpu_info" else None
trained_datasets_str = "$trained_datasets_env"

try:
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("CSV 文件为空")
        sys.exit(1)
    
    # 解析本次训练的数据集列表
    trained_datasets = set()
    if trained_datasets_str:
        for ds in trained_datasets_str.split(','):
            ds = ds.strip()
            if ds:
                trained_datasets.add(ds)
    
    # 按数据集分组，只统计本次训练的数据集
    datasets = defaultdict(list)
    for row in rows:
        dataset = row.get('dataset', '')
        if dataset:
            # 如果指定了本次训练的数据集列表，只统计这些数据集
            if trained_datasets:
                if dataset in trained_datasets:
                    datasets[dataset].append(row)
            else:
                # 如果没有指定，统计所有数据集（向后兼容）
                datasets[dataset].append(row)
    
    # 如果没有找到本次训练的数据集，尝试从后往前查找最新的结果
    if trained_datasets and not datasets:
        # 从后往前遍历，找到每个训练数据集的最新结果
        for row in reversed(rows):
            dataset = row.get('dataset', '')
            if dataset in trained_datasets:
                if dataset not in datasets:
                    datasets[dataset] = [row]
    
    # 如果没有提供模型信息，从本次训练的数据集结果中提取（优先使用最新的）
    filtered_rows = []
    for dataset_name in sorted(datasets.keys()):
        # 对于每个数据集，只取最后一次测试结果（列表中的最后一个）
        if datasets[dataset_name]:
            filtered_rows.append(datasets[dataset_name][-1])
    
    # 如果过滤后没有结果，使用所有行（向后兼容）
    if not filtered_rows:
        filtered_rows = rows
    
    # 如果没有提供模型信息，从过滤后的第一行提取
    if not model_name and filtered_rows:
        model_name = filtered_rows[0].get('net_name', 'Unknown')
    if not gflops and filtered_rows:
        gflops = filtered_rows[0].get('gflops', '')
    if not parameters and filtered_rows:
        parameters = filtered_rows[0].get('parameters', '')
    
    # 构建消息
    msg_parts = []
    
    # 模型信息
    if model_name:
        msg_parts.append(f"模型: {model_name}")
    if gflops:
        try:
            gflops_float = float(gflops)
            msg_parts.append(f"GFLOPs: {gflops_float:.2f}")
        except:
            msg_parts.append(f"GFLOPs: {gflops}")
    if parameters:
        try:
            params_float = float(parameters)
            msg_parts.append(f"参数量: {params_float:.2f}M")
        except:
            msg_parts.append(f"参数量: {parameters}M")
    if gpu_info:
        msg_parts.append(f"GPU: {gpu_info}")
    
    msg_parts.append("")
    msg_parts.append(f"Found {len(datasets)} dataset(s) with test results:")
    msg_parts.append("")
    
    # 格式化每个数据集的结果（只显示每个数据集的最新结果）
    for dataset_name in sorted(datasets.keys()):
        msg_parts.append(f"Dataset: {dataset_name}")
        msg_parts.append("-" * 80)
        msg_parts.append(f"{'Benchmark':<15} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12}")
        msg_parts.append("-" * 80)
        
        # 只显示每个数据集的最新结果（列表中的最后一个）
        dataset_rows = datasets[dataset_name]
        if dataset_rows:
            # 取最后一次测试结果
            row = dataset_rows[-1]
            benchmark = row.get('benchmark', '')
            psnr = row.get('psnr', '')
            ssim = row.get('ssim', '')
            lpips = row.get('lpips', '')
            
            # 格式化数值
            try:
                psnr_float = float(psnr)
                psnr_str = f"{psnr_float:.5f}"
            except:
                psnr_str = psnr
            
            try:
                ssim_float = float(ssim)
                ssim_str = f"{ssim_float:.5f}"
            except:
                ssim_str = ssim
            
            try:
                lpips_float = float(lpips)
                lpips_str = f"{lpips_float:.5f}"
            except:
                lpips_str = lpips
            
            msg_parts.append(f"{benchmark:<15} {psnr_str:<12} {ssim_str:<12} {lpips_str:<12}")
        
        msg_parts.append("")
    
    print("\n".join(msg_parts))
    
except Exception as e:
    print(f"解析 CSV 文件时出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
)
    
    local parse_exit_code=$?
    if [ $parse_exit_code -ne 0 ] || [ -z "$result" ]; then
        echo "错误: 解析 CSV 文件失败"
        # 尝试发送一个简单的错误通知
        local end_time=$(date +%s)
        local duration=$((end_time - SCRIPT_START_TIME))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        local time_str=""
        if [ $hours -gt 0 ]; then
            time_str="${hours}小时${minutes}分${seconds}秒"
        elif [ $minutes -gt 0 ]; then
            time_str="${minutes}分${seconds}秒"
        else
            time_str="${seconds}秒"
        fi
        bark_notify "❌出错" "训练完成，但解析测试结果文件失败
文件路径: $csv_file
运行时间: ${time_str}"
        return 1
    fi
    
    # 计算运行时间
    local end_time=$(date +%s)
    local duration=$((end_time - SCRIPT_START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    local time_str=""
    if [ $hours -gt 0 ]; then
        time_str="${hours}小时${minutes}分${seconds}秒"
    elif [ $minutes -gt 0 ]; then
        time_str="${minutes}分${seconds}秒"
    else
        time_str="${seconds}秒"
    fi
    
    # 构建完整的通知内容
    local title="✅训练完成 - 测试结果"
    local content="${result}
运行时间: ${time_str}"
    
    # 发送通知
    bark_notify_raw "$title" "$content"
}

# 原始发送通知函数（不添加额外信息）
bark_notify_raw() {
    # 如果通知被禁用，直接返回
    if [ "$ENABLE_BARK_NOTIFICATION" != "True" ] || [ -z "$BARK_URL" ]; then
        return 0
    fi
    
    local title="$1"
    local content="$2"
    
    # URL 编码函数
    url_encode() {
        local string="$1"
        echo "$string" | python3 -c "import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read().strip()))" 2>/dev/null || \
        echo "$string" | python -c "import sys, urllib; print(urllib.quote(sys.stdin.read().strip()))" 2>/dev/null || \
        echo "$string" | sed 's/ /%20/g; s/:/%3A/g; s/\//%2F/g; s/?/%3F/g; s/#/%23/g; s/\[/%5B/g; s/\]/%5D/g; s/@/%40/g; s/!/%21/g; s/\$/%24/g; s/&/%26/g; s/'\''/%27/g; s/(/%28/g; s/)/%29/g; s/*/%2A/g; s/+/%2B/g; s/,/%2C/g; s/;/%3B/g; s/=/%3D/g; s/%/%25/g'
    }
    
    local encoded_title=$(url_encode "$title")
    local encoded_content=$(url_encode "$content")
    
    if command -v curl &> /dev/null; then
        curl -s "${BARK_URL}/${encoded_title}/${encoded_content}" > /dev/null 2>&1
    elif command -v wget &> /dev/null; then
        wget -q -O /dev/null "${BARK_URL}/${encoded_title}/${encoded_content}" 2>&1
    else
        echo "错误: 未找到 curl 或 wget，无法发送通知"
        return 1
    fi
}

# 注册退出时的处理函数
trap bark_notify_on_exit EXIT


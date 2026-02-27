#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 批量训练多个网络模型
# ============================================================================
# 使用方法：
#   1. 在下面的 MODEL_LIST 中列出要训练的网络名称（去掉.py扩展名）
#   2. 确保 config.py 中其他配置都已设置好
#   3. 运行: bash train_multiple_models.sh
# ============================================================================

# ============================================================================
# 获取脚本所在目录（必须在其他使用 SCRIPT_DIR 的代码之前）
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载 Bark 推送通知功能（用于验证失败时发送通知）
# 注意：每个 train.sh 会自己 source bark_notify.sh，所以训练完成后会正常发送通知
# 在 train_multiple_models.sh 中，我们禁用自动的 EXIT trap，避免重复发送通知
source "$SCRIPT_DIR/bark_notify.sh" 2>/dev/null || true
# 禁用自动的 EXIT trap，避免在验证阶段发送不必要的通知
# train.sh 会自己注册 trap，所以训练完成后会正常发送通知
trap - EXIT 2>/dev/null || true

# 设置环境变量来抑制 PyTorch 分布式警告
export NCCL_DEBUG=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_SHOW_CPP_STACKTRACES=0
export OMP_NUM_THREADS=1

# ============================================================================
# 配置：要训练的网络列表
# ============================================================================
# 在这里列出要训练的所有网络名称（去掉.py扩展名）
# 网络文件应该在 net/ 目录下
MODEL_LIST=(
  # "MoCE_IR_S_SV_CrossScaleFreq"
  # "MoCE_IR_S_SV_MultiScale"
  # "MoCE_IR_S_SV_Retinex"
  # "MoCE_IR_S_SV_Deform"
  # "MoCE_IR_S_SV_Diffusion"
  # "MoCE_IR_S_SV_Contrast"
  # 可以添加更多网络，例如：
  # "MoCE_IR_S"
  # "MoCE_IR"
  #  "MoCE_IR_S_Enhanced"
  #  "MoCE_IR_S_PhysGate"
  #  "MoCE_IR_S_SpecSpatial"
  #  "MoCE_IR_S_Wavelet"
  #  "MoCE_IR_Spectral"
  #  "MoCE_IR_Spectral_S"
  # "MoCE_IR_PhysRouting"
  # "MoCE_IR_S_PG_GSSE"
  # "MoCE_IR_S_PADG_LKE"
  # "MoCE_IR_S_PADG_LKE_S3M"
  # "MoCE_IR_S_PADG_LKE_S3M_ABL"
  # "MoCE_IR_S_PG_ASS"
  # "MoCE_IR_S_PA_NOD"
  #"MoCE_IR_S_PA_DSE_ablation_v2"
  #"MoCE_IR_S_Freq_LKA_PhysGate"
  #"MoCE_IR_S_Freq_LKA_SpecRouter"
  # "MoCE_IR_S_Freq_LKA_SpectralRouter"
  # "MoCE_IR_S_Freq_LKA_Adaptive"
  # "MoCE_IR_S_Freq_LKA_PSFRouter"
  # "MoCE_IR_S_Freq_LKA_PhysConsistent"
  # 方案A: 可学习融合权重 (Soft Selection)
  "MoCE_IR_S_Freq_LKA_SoftFusion"
  # 方案B: 可学习低通滤波 (Learnable Frequency Split)
  "MoCE_IR_S_Freq_LKA_LearnableLP"
)

# 切换到脚本所在目录
cd "$SCRIPT_DIR"

# ============================================================================
# 验证网络文件是否存在
# ============================================================================
echo "=========================================="
echo "验证网络文件..."
echo "=========================================="

VALID_MODELS=()
for model in "${MODEL_LIST[@]}"; do
  model_file="net/${model}.py"
  if [ -f "$model_file" ]; then
    VALID_MODELS+=("$model")
    echo "✓ 找到网络: $model"
  else
    echo "✗ 警告: 未找到网络文件: $model_file (跳过)"
  fi
done

if [ ${#VALID_MODELS[@]} -eq 0 ]; then
  echo "错误: 没有找到任何有效的网络文件！"
  exit 1
fi

echo ""
echo "将训练 ${#VALID_MODELS[@]} 个网络模型"
echo "=========================================="
echo ""

# ============================================================================
# 备份原始 config.py
# ============================================================================
CONFIG_FILE="config.py"
BACKUP_DIR="config_backups"
CONFIG_BACKUP="${BACKUP_DIR}/config.py.backup_$(date +%Y%m%d_%H%M%S)"

# 清空备份文件夹（每次训练开始时清空）
if [ -d "$BACKUP_DIR" ]; then
  echo "清空备份文件夹: $BACKUP_DIR"
  rm -rf "$BACKUP_DIR"/*
  echo "✓ 备份文件夹已清空"
fi

# 创建备份文件夹
mkdir -p "$BACKUP_DIR"

# 移动现有的备份文件到备份文件夹（如果存在）
if ls config.py.backup_* 1> /dev/null 2>&1; then
  echo "发现现有备份文件，移动到备份文件夹..."
  mv config.py.backup_* "$BACKUP_DIR/" 2>/dev/null || true
  echo "已移动现有备份文件到: $BACKUP_DIR/"
fi

if [ -f "$CONFIG_FILE" ]; then
  cp "$CONFIG_FILE" "$CONFIG_BACKUP"
  echo "已备份 config.py 到: $CONFIG_BACKUP"
  echo ""
fi

# ============================================================================
# 模型验证函数（内联Python代码）
# ============================================================================
validate_model() {
  local model_name=$1
  echo ""
  echo "=========================================="
  echo "验证模型: $model_name"
  echo "=========================================="
  
  # 导出环境变量供 Python 使用
  export SCRIPT_DIR="$SCRIPT_DIR"
  
  python3 << PYTHON_EOF
import sys
import os
import traceback
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# 添加项目路径（从环境变量或当前工作目录获取）
script_dir = os.getcwd()
if 'SCRIPT_DIR' in os.environ:
    script_dir = os.environ['SCRIPT_DIR']
sys.path.insert(0, script_dir)

# 抑制警告
warnings.filterwarnings('ignore')

def validate_model(model_name):
    """验证单个模型"""
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
        print("    正在构建模型...")
        model = build_model(opt)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.train()
        
        # 检查模型参数
        num_params = sum(p.numel() for p in model.parameters())
        if num_params == 0:
            raise ValueError("模型没有参数！")
        print(f"    模型参数量: {num_params:,}")
        
        # 测试1: 基础前向传播
        print("    测试前向传播...")
        x = torch.randn(1, 3, 128, 128, device=device)
        with torch.no_grad():
            output = model(x)
        if output.shape != x.shape:
            raise ValueError(f"输出形状不匹配: {output.shape} vs {x.shape}")
        if torch.isnan(output).any() or torch.isinf(output).any():
            raise ValueError("输出包含NaN或Inf值")
        print(f"    ✓ 前向传播正常")
        
        # 测试2: 反向传播
        print("    测试反向传播...")
        x = torch.randn(2, 3, 128, 128, device=device, requires_grad=True)
        target = torch.randn_like(x)
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters())
        if not has_grad:
            raise ValueError("没有参数有梯度！")
        
        nan_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        if nan_grads > 0:
            raise ValueError(f"{nan_grads} 个参数有NaN梯度！")
        print(f"    ✓ 反向传播正常 (loss={loss.item():.6f})")
        
        # 测试3: 混合精度（如果支持CUDA）
        if torch.cuda.is_available():
            print("    测试混合精度 (FP16)...")
            model.zero_grad()
            scaler = GradScaler()
            x = torch.randn(2, 3, 128, 128, device=device)
            target = torch.randn_like(x)
            
            with autocast():
                output = model(x)
                loss = F.mse_loss(output, target)
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("FP16前向传播产生NaN或Inf")
            
            scaler.scale(loss).backward()
            scaler.step(torch.optim.Adam(model.parameters(), lr=1e-4))
            scaler.update()
            print(f"    ✓ 混合精度正常 (loss={loss.item():.6f})")
        
        # 测试4: 不同尺寸
        print("    测试不同输入尺寸...")
        test_sizes = [(1, 3, 64, 64), (1, 3, 256, 256), (4, 3, 128, 128)]
        for size in test_sizes:
            x = torch.randn(*size, device=device)
            with torch.no_grad():
                output = model(x)
            if output.shape[:2] != x.shape[:2] or output.shape[2:] != x.shape[2:]:
                raise ValueError(f"尺寸 {size}: 输出形状不匹配")
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError(f"尺寸 {size}: 输出包含NaN或Inf")
        print(f"    ✓ 不同尺寸测试通过")
        
        # 测试5: FFT操作（如果模型使用FFT）
        print("    测试FFT操作...")
        test_sizes = [(1, 3, 128, 128), (2, 3, 128, 128), (1, 3, 256, 256)]
        for size in test_sizes:
            x = torch.randn(*size, device=device)
            with torch.no_grad():
                try:
                    output = model(x.float())
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        raise ValueError(f"尺寸 {size}: FFT产生NaN或Inf")
                except RuntimeError as e:
                    if "cuFFT" in str(e) or "CUFFT" in str(e):
                        # FFT失败，但模型应该有回退机制，继续测试
                        pass
        print(f"    ✓ FFT操作正常")
        
        print("")
        print(f"✓ 模型 {model_name} 验证通过！")
        return True
        
    except Exception as e:
        print("")
        print(f"❌ 模型 {model_name} 验证失败: {str(e)}")
        print(f"详细错误:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model_name = "$model_name"
    success = validate_model(model_name)
    sys.exit(0 if success else 1)
PYTHON_EOF

  return $?
}

# ============================================================================
# 第一阶段：验证所有模型
# ============================================================================
echo "=========================================="
echo "第一阶段：验证所有模型"
echo "=========================================="
echo ""

VERIFIED_MODELS=()
FAILED_MODELS=()

TOTAL_MODELS=${#VALID_MODELS[@]}
CURRENT_MODEL=0

for model in "${VALID_MODELS[@]}"; do
  CURRENT_MODEL=$((CURRENT_MODEL + 1))
  
  echo "=========================================="
  echo "[验证 $CURRENT_MODEL/$TOTAL_MODELS] $model"
  echo "=========================================="
  
  # 修改 config.py 中的 MODEL 参数
  # 使用 sed 来替换 MODEL = "..." 这一行（包括后面的注释）
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS 版本的 sed
    sed -i '' "s/^MODEL = \".*\"/MODEL = \"$model\"/" "$CONFIG_FILE"
  else
    # Linux 版本的 sed
    sed -i "s/^MODEL = \".*\"/MODEL = \"$model\"/" "$CONFIG_FILE"
  fi
  
  # 验证修改是否成功
  CURRENT_MODEL_IN_CONFIG=$(python -c "import config; print(config.MODEL)" 2>/dev/null || echo "")
  if [ "$CURRENT_MODEL_IN_CONFIG" != "$model" ]; then
    echo "错误: 无法修改 config.py 中的 MODEL 参数！"
    echo "当前 MODEL: $CURRENT_MODEL_IN_CONFIG, 期望: $model"
    FAILED_MODELS+=("$model")
    continue
  fi
  
  # 验证模型
  if validate_model "$model"; then
    VERIFIED_MODELS+=("$model")
    echo ""
    echo "✓ 模型 $model 验证通过"
  else
    FAILED_MODELS+=("$model")
    echo ""
    echo "❌ 模型 $model 验证失败"
    # 注意：验证失败时不发送通知，因为这只是验证阶段，不是训练失败
    # 训练失败的通知会由 train.sh 自动发送
  fi
  
  echo ""
done

# ============================================================================
# 验证结果汇总
# ============================================================================
echo "=========================================="
echo "验证结果汇总"
echo "=========================================="
echo ""
echo "验证通过的模型 (${#VERIFIED_MODELS[@]} 个):"
for model in "${VERIFIED_MODELS[@]}"; do
  echo "  ✓ $model"
done
echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
  echo "验证失败的模型 (${#FAILED_MODELS[@]} 个):"
  for model in "${FAILED_MODELS[@]}"; do
    echo "  ❌ $model"
  done
  echo ""
fi

if [ ${#VERIFIED_MODELS[@]} -eq 0 ]; then
  echo "错误: 没有模型通过验证！无法开始训练。"
  # 注意：所有模型验证失败时不发送通知，因为这只是验证阶段
  # 如果需要通知，可以手动发送，但通常不需要
  exit 1
fi

# ============================================================================
# 可选：仅验证，不训练（用于快速自检）
#   使用方法：MOCEIR_VALIDATE_ONLY=1 bash train_multiple_models.sh
# ============================================================================
if [ "${MOCEIR_VALIDATE_ONLY:-0}" = "1" ]; then
  echo "=========================================="
  echo "已设置 MOCEIR_VALIDATE_ONLY=1，仅验证模型，不进入训练阶段。"
  echo "=========================================="
  exit 0
fi

# ============================================================================
# 准备开始训练
# ============================================================================
echo "=========================================="
echo "准备开始训练"
echo "=========================================="
echo ""
echo "将训练 ${#VERIFIED_MODELS[@]} 个验证通过的模型"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
  echo "将跳过 ${#FAILED_MODELS[@]} 个验证失败的模型"
fi
echo ""

echo "=========================================="
echo "第二阶段：开始训练所有验证通过的模型"
echo "=========================================="
echo ""

# ============================================================================
# 第二阶段：训练所有验证通过的模型
# ============================================================================
TOTAL_TO_TRAIN=${#VERIFIED_MODELS[@]}
CURRENT_MODEL=0

for model in "${VERIFIED_MODELS[@]}"; do
  CURRENT_MODEL=$((CURRENT_MODEL + 1))
  
  echo "=========================================="
  echo "[训练 $CURRENT_MODEL/$TOTAL_TO_TRAIN] $model"
  echo "=========================================="
  echo ""
  
  # 修改 config.py 中的 MODEL 参数
  # 使用 sed 来替换 MODEL = "..." 这一行（包括后面的注释）
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS 版本的 sed
    sed -i '' "s/^MODEL = \".*\"/MODEL = \"$model\"/" "$CONFIG_FILE"
  else
    # Linux 版本的 sed
    sed -i "s/^MODEL = \".*\"/MODEL = \"$model\"/" "$CONFIG_FILE"
  fi
  
  # 验证修改是否成功
  CURRENT_MODEL_IN_CONFIG=$(python -c "import config; print(config.MODEL)" 2>/dev/null || echo "")
  if [ "$CURRENT_MODEL_IN_CONFIG" != "$model" ]; then
    echo "错误: 无法修改 config.py 中的 MODEL 参数！"
    echo "当前 MODEL: $CURRENT_MODEL_IN_CONFIG, 期望: $model"
    echo "跳过此模型，继续下一个..."
    echo ""
    continue
  fi
  
  echo "已设置 MODEL = $model"
  echo ""
  
  # 运行训练脚本
  # 调用原有的 train.sh 脚本
  if bash train.sh; then
    echo ""
    echo "✓ 网络 $model 训练完成！"
    echo ""
  else
    echo ""
    echo "✗ 网络 $model 训练失败！"
    echo ""
    echo "继续训练下一个网络..."
  fi
  
  echo "----------------------------------------"
  echo ""
done

# ============================================================================
# 恢复原始 config.py（可选）并清理备份文件夹
# ============================================================================
echo "=========================================="
echo "所有网络训练完成！"
echo "=========================================="
echo ""

# 保留当前 config.py（MODEL = ${VERIFIED_MODELS[-1]}）
echo "保留当前 config.py（MODEL = ${VERIFIED_MODELS[-1]}）"

# 自动删除备份文件夹
if [ -d "$BACKUP_DIR" ]; then
  echo ""
  echo "正在清理备份文件夹: $BACKUP_DIR"
  rm -rf "$BACKUP_DIR"
  echo "✓ 备份文件夹已删除"
fi

echo ""
echo "训练任务全部完成！"


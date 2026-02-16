# 方案一：Retinex理论驱动的光照感知专家（推荐度：⭐⭐⭐⭐⭐）

**网络结构文件**：[`net/MoCE_IR_S_SV_Retinex.py`](../net/MoCE_IR_S_SV_Retinex.py)

**基础网络**：[`net/MoCE_IR_S_SV.py`](../net/MoCE_IR_S_SV.py) (第一个创新点：SpatiallyVariantFreqEmbedding)

## 1. 学术叙事

### 1.1 核心故事

超透镜内窥镜图像的非均匀退化不仅来自像差，更来自**光照不均**。内窥镜的短工作距离和点光源特性导致图像中心过亮、边缘过暗，这种**Retinex退化**（反射率×光照）严重影响病灶识别。

### 1.2 创新点

- 引入Retinex理论，将图像分解为**反射率（Reflectance）**和**光照（Illumination）**两个物理分量
- 设计**光照感知专家**，专门修复非均匀光照，同时保持反射率（病灶纹理）不变
- 结合SVFE的频率感知，实现**"频率-光照"双维度退化诊断**

### 1.3 顶会叙事

> "We propose a Retinex-aware expert routing mechanism that decomposes metalens endoscopy images into reflectance and illumination components. By leveraging spatially-variant frequency embedding, our model diagnoses both optical aberrations and non-uniform illumination, enabling specialized experts to restore illumination while preserving critical reflectance details for clinical diagnosis."

## 2. 技术实现

### 2.1 核心模块代码

#### RetinexDecomposition类

**插入位置**：在`SpatiallyVariantFreqEmbedding`类之后（约第768行`return out`之后）

```python
##########################################################################
## Retinex Decomposition Module
class RetinexDecomposition(nn.Module):
    """
    基于Retinex理论的图像分解：I = R × L
    I: 输入图像, R: 反射率（病灶纹理）, L: 光照（退化源）
    """
    def __init__(self, dim):
        super().__init__()
        # 反射率分支：提取结构纹理（高频）
        self.reflectance_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        # 光照分支：提取平滑光照（低频）
        self.illumination_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 15, padding=7),  # 大核提取平滑光照
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        # 反射率：高频细节（病灶纹理）
        R = self.reflectance_branch(x)
        R = torch.sigmoid(R)  # 归一化到[0,1]
        
        # 光照：低频平滑（非均匀光照）
        L = self.illumination_branch(x)
        L = torch.sigmoid(L) + 0.1  # 避免除零，最小光照0.1
        
        # 重建：I = R × L
        I_recon = R * L
        
        return R, L, I_recon
```

#### IlluminationAwareExpert类

**插入位置**：紧接在`RetinexDecomposition`类之后

```python
##########################################################################
## Illumination-Aware Expert
class IlluminationAwareExpert(nn.Module):
    """
    光照修复专家：专门修复非均匀光照，保持反射率不变
    """
    def __init__(self, dim):
        super().__init__()
        self.retinex = RetinexDecomposition(dim)
        
        # 光照修复网络
        self.illumination_restorer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.Sigmoid()  # 输出归一化光照
        )
        
        # 反射率增强（轻微）
        self.reflectance_enhancer = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
        
    def forward(self, x):
        # 分解
        R, L, _ = self.retinex(x)
        
        # 修复光照：将非均匀光照L修复为均匀光照L_restored
        L_restored = self.illumination_restorer(L)
        # 确保光照在合理范围
        L_restored = L_restored * 0.8 + 0.2  # [0.2, 1.0]
        
        # 轻微增强反射率（病灶细节）
        R_enhanced = R + 0.1 * self.reflectance_enhancer(R)
        R_enhanced = torch.clamp(R_enhanced, 0, 1)
        
        # 重建：I_restored = R_enhanced × L_restored
        I_restored = R_enhanced * L_restored
        
        return I_restored
```

## 3. 代码修改指令

### 步骤1：添加Retinex相关类

**精确位置**：打开`MoCE_IR_S_SV.py`，找到第768行（`SpatiallyVariantFreqEmbedding.forward`方法的`return out`之后），添加上述两个类。

### 步骤2：修改RoutingFunction类

**找到`RoutingFunction.__init__`方法**（约第480行），在`self.freq_gate`之后添加：

```python
# 原代码位置：self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
# 在这之后添加：
self.illumination_gate = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    Rearrange('b c 1 1 -> b c'),
    nn.Linear(dim, num_experts, bias=False)
)
```

**修改`RoutingFunction.forward`方法**（约第502行）：

```python
def forward(self, x, freq_emb):
    # 原代码：logits = self.gate(x) + self.freq_gate(freq_emb)
    # 修改为：
    spatial_logits = self.gate(x)
    freq_logits = self.freq_gate(freq_emb)
    illum_logits = self.illumination_gate(x)  # 新增
    logits = spatial_logits + freq_logits + illum_logits  # 三路融合
    
    # 后续代码保持不变
    if self.training:
        loss_imp = self.importance_loss(logits.softmax(dim=-1))
    # ... 其余代码不变
```

### 步骤3：修改ModExpert类

**找到`ModExpert.__init__`方法**（约第359行），修改为：

```python
def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size:int, expert_type="standard"):
    super(ModExpert, self).__init__()
    
    self.expert_type = expert_type
    self.depth = depth
    
    if expert_type == "illumination_aware":
        # 光照感知专家：使用IlluminationAwareExpert
        self.body = IlluminationAwareExpert(dim)
        self.proj = None  # 不需要投影
    else:
        # 标准专家：原有逻辑
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
```

**修改`ModExpert.forward`方法**（约第384行）：

```python
def forward(self, x, shared):
    b, c, h, w = x.shape
    
    if b == 0:
        return x
    else:
        if self.expert_type == "illumination_aware":
            # 光照感知专家：直接处理
            return self.body(x)
        else:
            # 标准专家：原有逻辑
            return self.feat_extract(x, shared)
```

### 步骤4：修改AdapterLayer类，支持光照感知专家

**找到`AdapterLayer.__init__`方法**（约第398行），在创建experts时添加专家类型参数：

```python
# 原代码：self.experts = nn.ModuleList([...])
# 修改为支持expert_type参数（需要在AdapterLayer的__init__中添加expert_type参数）
# 或者在创建ModExpert时，根据某些条件选择expert_type
```

**具体修改**：在`AdapterLayer.__init__`中添加`expert_type_list`参数：

```python
def __init__(self, 
             dim: int, rank: int, num_experts: int = 4, top_k: int=2, expert_layer: nn.Module=FFTAttention, stage_depth: int=1,
             depth_type: str="lin", rank_type: str="constant", freq_dim: int=128, 
             with_complexity: bool=False, complexity_scale: str="min",
             expert_type_list=None):  # 新增参数
    # ... 原有代码 ...
    
    # 修改experts创建部分
    if expert_type_list is None:
        expert_type_list = ["standard"] * num_experts
    
    self.experts = nn.ModuleList([
        MySequential(*[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel, expert_type=expert_type_list[idx])])
        for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
    ])
```

## 4. GitHub参考

- **RetinexNet**: https://github.com/weichen582/RetinexNet
- **KinD**: https://github.com/zhangyhuaee/KinD (Kindling the Darkness)
- **论文**: "Deep Retinex Decomposition for Low-Light Enhancement" (BMVC 2018)

## 5. 预期效果

- **PSNR提升**：0.3-0.6 dB（针对光照不均图像）
- **视觉效果**：显著改善中心过亮、边缘过暗问题
- **临床价值**：保持病灶反射率细节，提升诊断准确性

## 6. 验证步骤

1. **语法检查**：`python -m py_compile MoCE_IR_S_SV.py`
2. **导入测试**：`python -c "from net.MoCE_IR_S_SV import MoCEIR; print('OK')"`
3. **前向传播测试**：
```python
import torch
from net.MoCE_IR_S_SV import MoCEIR

model = MoCEIR(dim=32, num_blocks=[1,1,1,3], num_dec_blocks=[1,1,1])
x = torch.randn(2, 3, 128, 128)
out = model(x)
print(f"Input: {x.shape}, Output: {out.shape}")
```


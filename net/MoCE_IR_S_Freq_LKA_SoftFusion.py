from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 方案A: 可学习融合权重 (Soft Selection Fusion)
## 
## 核心创新: 让融合权重可学习，根据内容和物理先验动态决定
## 更依赖低频全局建模还是高频细节修复
##########################################################################


class LKA(nn.Module):
    """
    [学术定义]: Local Kernel Extractor (LKA) / 局部核提取器
    
    物理背景: 超透镜的退化在空间域表现为随视场角剧烈变化的模糊。
    数学逻辑: 通过分解大核卷积（5x5 DW + 7x7 DW-dilation=3）实现对高维非等晕核的低秩近似 (Low-rank Approximation)，
    在保持线性复杂度的同时，获取足以覆盖大尺寸 PSF 的全局感受野。
    """

    def __init__(self, dim: int):
        super().__init__()
        # 局部特征提取 (Local Feature Extraction)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 空间长距离依赖建模 (Long-range Spatial Dependency Modeling)
        # 模拟非等晕场中的远距离像素干涉
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        # 通道间信息融合 (Inter-channel Communication)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        # 空间自适应权重调制 (Spatially-Adaptive Weight Modulation)
        return u * attn


class Freq_LKA_Expert_SoftFusion(nn.Module):
    """
    [学术定义]: Frequency-Aware Co-Gating Expert with Soft Fusion (FACE-SF)
    
    核心创新: 
    1. 显式分离高低频特征（LKA处理低频，轻量级卷积处理高频）
    2. 新增可学习的融合门控，根据内容和物理先验动态决定融合权重
    3. 实现了"自由使用工具"的软路由机制
    
    能讲述的故事:
    专家不是固定相加，而是根据内容和物理先验动态决定更依赖低频全局建模还是高频细节修复。
    这就叫"自由使用工具"，代码上确实支持。
    """

    def __init__(self, dim: int, phys_dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 特征投影与隐式物理对齐 (Implicit Physics Alignment)
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 2. 低频分支算子：大核注意力 (LKA)
        self.lka = LKA(self.hidden_dim)

        # 3. 高频分支算子：轻量级卷积 (例如 3x3 DW Conv)
        self.high_freq_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)

        # ========== 方案A核心修改: 可学习融合门控 ==========
        # 新增融合权重生成器: 根据低频+高频特征生成 0~1 的融合权重
        # alpha = 1 表示完全使用低频, alpha = 0 表示完全使用高频
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),  # 输出单个通道的权重
        )
        
        # 可选：将物理先验注入融合权重
        self.phys_gate_fuse = nn.Sequential(
            nn.Linear(phys_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        # =====================================================

        # 4. [核心创新]: 物理信息神经算子调制器 (Neural Operator Modulator, NOM)
        # 将物理嵌入映射为非线性的门控张量，实现对特征流的物理一致性校准。
        self.physics_gate_gen = nn.Sequential(
            nn.Linear(phys_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 5. 投影回原始维度
        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 图像内容特征
            phys_emb: [B, phys_dim] - 物理先验嵌入 (Physics Prior Embedding)
        """
        b, c, h, w = x.shape

        # 1. 输入投影并拆分 (Feature Splitting)
        x_and_gate = self.in_proj(x)
        x_main, gate = x_and_gate.chunk(2, dim=1)

        # 2. 频率分解 (Frequency Decomposition) - 保持不变
        x_low = F.avg_pool2d(x_main, kernel_size=2, stride=2)
        x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
        x_high = x_main - x_low

        # 3. 低频处理：利用 LKA 捕捉全局退化
        x_low_processed = self.lka(x_low)

        # 4. 高频处理：利用轻量级卷积修复细节
        x_high_processed = self.high_freq_conv(x_high)

        # ========== 方案A核心修改: 可学习融合 ==========
        # 5. 融合频率分量（可学习权重）
        # 生成融合权重 alpha: [B, 1, H, W], 范围 0~1
        fused_feat = torch.cat([x_low_processed, x_high_processed], dim=1)
        
        # 内容驱动的融合权重
        alpha = self.fuse_gate(fused_feat)  # [B, 1, H, W]
        
        # 物理先验调制（可选：将物理嵌入注入融合决策）
        phys_w = self.phys_gate_fuse(phys_emb).view(b, -1, 1, 1)  # [B, 1, 1, 1]
        alpha = alpha * phys_w  # 内容权重受物理先验调节
        
        alpha = torch.sigmoid(alpha)  # 确保 0~1
        
        # 软融合: alpha * low + (1-alpha) * high
        x_fused = alpha * x_low_processed + (1 - alpha) * x_high_processed
        # =====================================================

        # 6. 物理域调制：生成物理引导的动态门控 (Physics-Guided Dynamic Gating)
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        # 7. 协同门控融合 (Co-Gating Fusion)
        #x_out = x_fused * F.gelu(gate)
        x_out = x_fused * torch.sigmoid(gate)

        return self.out_proj(x_out)


##########################################################################
## 创新点2: 物理引导的轻量化路由 (Physics-Guided Router)
## 
## 核心逻辑: 解决了传统 MoE 路由器的"空间盲目性 (Routing Blindness)"。
## 通过结合图像内容分支与物理先验分支，实现了基于物理因果链的专家调度。
## 创新点: 物理先验直接参与专家选择，确保专家分工的物理确定性。
##########################################################################


class Physics_Router(nn.Module):
    """
    [学术定义]: Physics-Guided Router (PGR) / 物理引导路由器
    
    核心创新: 物理先验直接参与专家选择分数计算
    - 内容分支: 提取图像内容特征用于专家选择
    - 物理分支: 将物理先验嵌入映射为专家选择权重
    - 协同决策: 两者相加后 softmax 得到最终专家选择概率
    """
    def __init__(self, dim: int, phys_dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # 内容感知分支 (Content-Aware Branch)
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )

        # 物理因果分支 (Physics-Causal Branch)
        # 直接利用物理先验指导专家选择，确保专家分工的物理确定性。
        self.physics_branch = nn.Linear(phys_dim, num_experts)

    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 协同决策逻辑 (Collaborative Decision Logic)
        logits = self.content_branch(x) + self.physics_branch(phys_emb)
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class Freq_LKA_AdapterLayer_SoftFusion(nn.Module):
    """
    [学术定义]: Physics-Aware Frequency-decomposed Mixture-of-Experts with Soft Fusion (PA-FD-MoE-SF)
    
    功能: 封装了物理感知频率分解专家与路由机制，作为 MoCE-IR 架构的核心计算单元。
    """

    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 - 使用 SoftFusion 专家
        self.experts = nn.ModuleList(
            [Freq_LKA_Expert_SoftFusion(dim, phys_dim) for _ in range(num_experts)]
        )
        # 物理引导路由器
        self.router = Physics_Router(dim, phys_dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scores, indices, top_k_scores = self.router(x, phys_emb)

        final_out = torch.zeros_like(x)

        # 动态专家分发与聚合 (Dynamic Dispatch & Aggregation)
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        return self.out_proj(final_out)


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + PA-FD-MoE 适配层
##########################################################################


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class MoCEIR_Freq_LKA_SoftFusion(nn.Module):
    """
    [学术定义]: Physics-Informed Frequency-decomposed Mixture-of-Conditional-Experts with Soft Fusion
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干。
    2. 在 Decoder 阶段引入 PA-FD-MoE-SF 层，实现对多尺度物理退化的自适应修复。
    3. 物理嵌入由特征自适应提取，实现了端到端的物理对齐学习。
    4. 核心创新: 频率融合权重可学习，实现软路由。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        phys_dim: int = 128,
        num_experts: int = 4,
        topk: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.total_loss: Optional[torch.Tensor] = None

        # Encoder (多尺度特征提取)
        self.enc1 = ConvBlock(inp_channels, dim)
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = Downsample(dim * 4, dim * 8)

        # Bottleneck (深层语义表征)
        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        # Decoder (多尺度物理自适应修复)
        self.up2 = Upsample(dim * 8, dim * 4)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        # 在深层特征空间引入 PA-FD-MoE-SF
        self.freq_lka_adapter2 = Freq_LKA_AdapterLayer_SoftFusion(dim * 4, phys_dim, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 在浅层特征空间引入 PA-FD-MoE-SF
        self.freq_lka_adapter1 = Freq_LKA_AdapterLayer_SoftFusion(dim * 2, phys_dim, num_experts=num_experts, k=topk)

        # 输出头
        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

        # Global physical embedding extraction
        self.global_phys_embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim * 8, phys_dim, kernel_size=1),
            nn.Flatten()
        )

    def _global_phys_embedding(self, feat: torch.Tensor) -> torch.Tensor:
        return self.global_phys_embedder(feat)

    def forward(
        self, x: torch.Tensor, de_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        inp = x

        # Encoder
        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        # Bottleneck
        b = self.bottleneck(d2)

        # Decoder stage 2
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d_dec2 = self.dec2(u2)

        global_phys_emb = self._global_phys_embedding(b)
        d_dec2 = self.freq_lka_adapter2(d_dec2, global_phys_emb)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1 = self.freq_lka_adapter1(d_dec1, global_phys_emb)

        out = self.out_conv(d_dec1) + inp # Residual connection

        self.total_loss = torch.tensor(
            0.0, device=out.device, dtype=out.dtype, requires_grad=False
        )

        return out


def build_model(opt) -> nn.Module:
    dim = getattr(opt, "dim", 32)
    phys_dim = getattr(opt, "phys_dim", 128)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR_Freq_LKA_SoftFusion(
        dim=dim,
        phys_dim=phys_dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 1
    in_channels = 3
    image_size = 64
    feature_dim = 32
    physical_embedding_dim = 128
    num_experts = 4

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    model = MoCEIR_Freq_LKA_SoftFusion(inp_channels=in_channels, out_channels=in_channels, dim=feature_dim, phys_dim=physical_embedding_dim, num_experts=num_experts)

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


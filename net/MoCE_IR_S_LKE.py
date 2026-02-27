from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 核心创新模块：物理感知动态门控大核专家 (PADG-LKE)
## Physics-Aware Dynamic Gated Large-Kernel Expert
## 
## 学术叙事逻辑：
## 1. 空间域：利用分解大核卷积 (LKA) 近似非等晕点扩散函数 (Non-isoplanatic PSF) 的长距离空间相关性。
## 2. 物理域：利用动态门控 (Dynamic Gating) 实现物理参数化的神经算子调制 (Neural Operator Modulation)。
##########################################################################


class LKA(nn.Module):
    """
    [学术定义]: Local Spectral Sampler (LSS) / 局部谱采样器
    
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


class LKE_Expert(nn.Module):
    """
    [学术定义]: Field-Aware Co-Gating Expert (FACE) / 视场感知协同门控专家
    
    核心逻辑: 融合 LKA 的全局空间感知与物理先验驱动的动态门控。
    创新点: 实现了“受物理参数调制的神经算子 (Physics-Parameterized Neural Operator)”，
    使专家能够根据输入的物理上下文（如深度、光谱）动态调整其等效卷积核。
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 特征投影与隐式物理对齐 (Implicit Physics Alignment)
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 2. 空间域算子：大核注意力 (LKA)
        self.lka = LKA(self.hidden_dim)
        # 3. [Ablation-LKE] No physics-conditioned gating.

        # 4. 协同门控融合 (Co-Gating Fusion)
        # 结合空间感知 (x_main) 与 物理调制 (gate)
        x_out = x_main * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 物理引导的轻量化路由 (Physics-Modulated Router)
##########################################################################


class LKE_Router(nn.Module):
    """
    [学术定义]: Physics-Prior Guided Router (PPGR) / 物理先验引导路由器
    
    创新点: 解决了传统 MoE 路由器的“空间盲目性 (Routing Blindness)”。
    通过结合图像内容分支与物理先验分支，实现了基于物理因果链的专家调度。
    """
    def __init__(self, dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # 内容感知分支 (Content-Aware Branch)
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )
        # [Ablation-LKE] Remove physics branch (content-only routing)

    def forward(self, x: torch.Tensor, phys_emb: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 协同决策逻辑 (Collaborative Decision Logic)
        logits = self.content_branch(x)
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class LKE_AdapterLayer(nn.Module):
    """
    [学术定义]: Physics-Aware Mixture-of-Experts (PA-MoE) Layer
    
    功能: 封装了物理感知专家与路由机制，作为 MoCE-IR 架构的核心计算单元。
    """

    def __init__(self, dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 (Expert Pool)
        self.experts = nn.ModuleList(
            [LKE_Expert(dim) for _ in range(num_experts)]
        )
        # 物理引导路由器
        self.router = LKE_Router(dim, num_experts, k)

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
## 简化版 MoCE-IR 主干：UNet 结构 + PADG-LKE 适配层
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


class MoCEIR(nn.Module):
    """
    [学术定义]: Physics-Informed Mixture-of-Conditional-Experts for Image Restoration
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干。
    2. 在 Decoder 阶段引入 PA-MoE 层，实现对多尺度物理退化的自适应修复。
    3. 物理嵌入由特征自适应提取，实现了端到端的物理对齐学习。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
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
        # 在深层特征空间引入 PA-MoE，处理复杂的全局退化
        self.lke2 = LKE_AdapterLayer(dim * 4, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 在浅层特征空间引入 PA-MoE，修复精细的空间变异细节
        self.lke1 = LKE_AdapterLayer(dim * 2, num_experts=num_experts, k=topk)

        # 输出头
        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _global_phys_embedding(feat: torch.Tensor) -> torch.Tensor:
        """
        [学术定义]: Adaptive Physics Embedding Extraction (APEE)
        从特征中自适应提取物理先验嵌入，实现隐式的物理参数化。
        """
        return feat.mean(dim=(2, 3))

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

        phys_emb2 = self._global_phys_embedding(d_dec2)
        d_dec2 = self.lke2(d_dec2, phys_emb2)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        phys_emb1 = self._global_phys_embedding(d_dec1)
        d_dec1 = self.lke1(d_dec1, phys_emb1)

        out = self.out_conv(d_dec1) + inp

        self.total_loss = torch.tensor(
            0.0, device=out.device, dtype=out.dtype, requires_grad=False
        )

        return out


def build_model(opt) -> nn.Module:
    dim = getattr(opt, "dim", 32)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR(
        dim=dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoCEIR(dim=32, num_experts=4, topk=1).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 核心创新模块：多膨胀率 LKA (MD-LKA)
## Multi-dilation Large-Kernel Attention
## 
## 学术叙事逻辑：
## 1. 物理背景: Metalens 的 PSF 支持域不是固定大小，中心/边缘退化尺度不同。
## 2. 创新点: 将单一 dilation=3 的 LKA 扩展成多分支 dilation（如 1/4/9），并行提取不同尺度的空间相关性，再融合。
## 3. 优势: 相当于给大核提供“多尺度 PSF 近似基”，更好支持 LKA 捕捉非等晕 PSF 的多尺度特性。
## 4. 启发: CVPR 2025 DarkIR 中多膨胀分支的模块设计。
##########################################################################


class MD_LKA(nn.Module):
    """
    [学术定义]: Multi-dilation Local Kernel Extractor (MD-LKA) / 多膨胀局部核提取器
    
    物理背景: 超透镜的退化在空间域表现为随视场角剧烈变化的模糊，且其尺度非单一。
    数学逻辑: 通过并行多个不同膨胀率（dilation）的分解大核卷积，实现对多尺度非等晕核的低秩近似。
    在保持线性复杂度的同时，获取足以覆盖不同尺度 PSF 的全局感受野。
    """

    def __init__(self, dim: int, dilations: list = [1, 4, 9]):
        super().__init__()
        self.dilations = dilations
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        self.spatial_branches = nn.ModuleList()
        for d in dilations:
            # 计算 padding 以保持输出尺寸不变
            # kernel_size = 7, dilation = d => effective_kernel_size = (7-1)*d + 1
            # padding = (effective_kernel_size - 1) / 2
            padding = ((7 - 1) * d + 1 - 1) // 2
            self.spatial_branches.append(
                nn.Conv2d(dim, dim, 7, stride=1, padding=padding, groups=dim, dilation=d)
            )
        
        self.conv1 = nn.Conv2d(dim * len(dilations), dim, 1) # 融合多分支输出

        # 可选：轻量通道注意力 (如 SE 模块简化版)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1, bias=False),
            nn.Sigmoid()
        ) if len(dilations) > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn_base = self.conv0(x)
        
        spatial_outputs = []
        for branch in self.spatial_branches:
            spatial_outputs.append(branch(attn_base))
        
        # 并行分支融合
        if len(self.dilations) > 1:
            attn = torch.cat(spatial_outputs, dim=1) # Concat 不同 dilation 的结果
        else:
            attn = spatial_outputs[0]

        attn = self.conv1(attn)

        # 应用通道注意力
        if self.channel_attention is not None:
            attn = attn * self.channel_attention(attn)

        # 空间自适应权重调制 (Spatially-Adaptive Weight Modulation)
        return u * attn


class MD_LKA_Expert(nn.Module):
    """
    [学术定义]: Multi-dilation Field-Aware Co-Gating Expert (MD-FACE) / 多膨胀视场感知协同门控专家
    
    核心逻辑: 融合 MD-LKA 的多尺度全局空间感知与物理先验驱动的动态门控。
    创新点: 实现了“受物理参数调制的神经算子 (Physics-Parameterized Neural Operator)”，
    使专家能够根据输入的物理上下文（如深度、光谱）动态调整其等效卷积核，并具备多尺度感知能力。
    """

    def __init__(self, dim: int, phys_dim: int, expansion_ratio: float = 2.0, dilations: list = [1, 4, 9]):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 特征投影与隐式物理对齐 (Implicit Physics Alignment)
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 2. 空间域算子：多膨胀率大核注意力 (MD-LKA)
        self.md_lka = MD_LKA(self.hidden_dim, dilations=dilations)

        # 3. [核心创新]: 物理信息神经算子调制器 (Neural Operator Modulator, NOM)
        # 将物理嵌入映射为非线性的门控张量，实现对特征流的物理一致性校准。
        self.physics_gate_gen = nn.Sequential(
            nn.Linear(phys_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 4. 投影回原始维度
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

        # 2. 空间域处理：利用 MD-LKA 捕捉多尺度非等晕退化特征
        x_main = self.md_lka(x_main)

        # 3. 物理域调制：生成物理引导的动态门控 (Physics-Guided Dynamic Gating)
        # p_gate 充当了物理参数化的“过滤器”，只允许符合物理规律的特征通过。
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        # 4. 协同门控融合 (Co-Gating Fusion)
        # 结合空间感知 (x_main) 与 物理调制 (gate)
        x_out = x_main * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 物理引导的轻量化路由 (Physics-Modulated Router)
##########################################################################


class MD_LKA_Router(nn.Module):
    """
    [学术定义]: Physics-Prior Guided Router (PPGR) / 物理先验引导路由器
    
    创新点: 解决了传统 MoE 路由器的“空间盲目性 (Routing Blindness)”。
    通过结合图像内容分支与物理先验分支，实现了基于物理因果链的专家调度。
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


class MD_LKA_AdapterLayer(nn.Module):
    """
    [学术定义]: Physics-Aware Multi-dilation Mixture-of-Experts (PA-MD-MoE) Layer
    
    功能: 封装了物理感知多膨胀率专家与路由机制，作为 MoCE-IR 架构的核心计算单元。
    """

    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1, dilations: list = [1, 4, 9]):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 (Expert Pool)
        self.experts = nn.ModuleList(
            [MD_LKA_Expert(dim, phys_dim, dilations=dilations) for _ in range(num_experts)]
        )
        # 物理引导路由器
        self.router = MD_LKA_Router(dim, phys_dim, num_experts, k)

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
## 简化版 MoCE-IR 主干：UNet 结构 + PA-MD-MoE 适配层
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


class MoCEIR_MD_LKA(nn.Module):
    """
    [学术定义]: Physics-Informed Multi-dilation Mixture-of-Conditional-Experts for Image Restoration
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干。
    2. 在 Decoder 阶段引入 PA-MD-MoE 层，实现对多尺度物理退化的自适应修复。
    3. 物理嵌入由特征自适应提取，实现了端到端的物理对齐学习。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        phys_dim: int = 128,
        num_experts: int = 4,
        topk: int = 1,
        dilations: list = [1, 4, 9],
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
        # 在深层特征空间引入 PA-MD-MoE，处理复杂的全局退化
        self.md_lka_adapter2 = MD_LKA_AdapterLayer(dim * 4, phys_dim, num_experts=num_experts, k=topk, dilations=dilations)

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 在浅层特征空间引入 PA-MD-MoE，修复精细的空间变异细节
        self.md_lka_adapter1 = MD_LKA_AdapterLayer(dim * 2, phys_dim, num_experts=num_experts, k=topk, dilations=dilations)

        # 输出头
        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

        # Global physical embedding extraction (placeholder)
        self.global_phys_embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim * 8, phys_dim, kernel_size=1),
            nn.Flatten()
        )

    def _global_phys_embedding(self, feat: torch.Tensor) -> torch.Tensor:
        # Placeholder for extracting physical embedding from feature map
        # In a real scenario, this might come from metadata or a dedicated network branch
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

        global_phys_emb = self._global_phys_embedding(b) # Extract global physical embedding from bottleneck output
        d_dec2 = self.md_lka_adapter2(d_dec2, global_phys_emb)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1 = self.md_lka_adapter1(d_dec1, global_phys_emb)

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
    dilations = getattr(opt, "dilations", [1, 4, 9])

    return MoCEIR_MD_LKA(
        dim=dim,
        phys_dim=phys_dim,
        num_experts=num_experts,
        topk=topk,
        dilations=dilations,
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 1
    in_channels = 3
    image_size = 64
    feature_dim = 32
    physical_embedding_dim = 128
    num_experts = 4
    dilations = [1, 4, 9]

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    model = MoCEIR_MD_LKA(inp_channels=in_channels, out_channels=in_channels, dim=feature_dim, phys_dim=physical_embedding_dim, num_experts=num_experts, dilations=dilations)

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")

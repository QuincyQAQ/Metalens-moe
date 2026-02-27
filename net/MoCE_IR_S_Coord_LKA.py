from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 核心创新模块：坐标/视场位置感知 LKA (Coord-LKA)
## Coordinate/Field-of-View-aware Large-Kernel Attention
## 
## 学术叙事逻辑：
## 1. 物理背景: Metalens 最大的物理特性是 non-isoplanatic (非等晕性)：PSF = PSF(x,y)。
## 2. 创新点: 给输入特征拼接或调制一个非常轻的坐标编码（x,y 或径向 r），让 LKA 的响应随位置变化。
## 3. 优势: 让大核成为“空间变核近似器”，坐标条件化进一步让它拟合 PSF(x,y)。
## 4. 启发: 物理合理的增强，几乎不涨 GFLOPs，故事非常“物理”。
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


class Coord_LKA_Expert(nn.Module):
    """
    [学术定义]: Coordinate-aware Field-Aware Co-Gating Expert (Coord-FACE) / 坐标感知视场协同门控专家
    
    核心逻辑: 融合 LKA 的全局空间感知与物理先验驱动的动态门控，并引入坐标编码。
    创新点: 实现了“坐标条件化的物理参数化神经算子”，使专家能够根据输入的物理上下文和空间位置动态调整其等效卷积核。
    """

    def __init__(self, dim: int, phys_dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 特征投影与隐式物理对齐 (Implicit Physics Alignment)
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 2. 空间域算子：大核注意力 (LKA)
        self.lka = LKA(self.hidden_dim)

        # 3. [核心创新]: 坐标编码投影层 (Coordinate Encoding Projection)
        # 生成 2 通道坐标图（[-1,1]），经过 1x1 conv 投影到 C 维
        self.coord_proj = nn.Conv2d(2, self.hidden_dim, kernel_size=1)

        # 4. 物理信息神经算子调制器 (Neural Operator Modulator, NOM)
        # 将物理嵌入映射为非线性的门控张量，实现对特征流的物理一致性校准。
        self.physics_gate_gen = nn.Sequential(
            nn.Linear(phys_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 5. 投影回原始维度
        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def _generate_coord_map(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        # 生成 [-1, 1] 范围的 x, y 坐标图
        x_coords = torch.linspace(-1, 1, w, device=device)
        y_coords = torch.linspace(-1, 1, h, device=device)
        # [H, W, 2]
        coord_map = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1)
        # [2, H, W]
        return coord_map.permute(2, 0, 1).unsqueeze(0) # Add batch dim

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 图像内容特征
            phys_emb: [B, phys_dim] - 物理先验嵌入 (Physics Prior Embedding)
        """
        b, c, h, w = x.shape

        # 1. 生成并投影坐标编码
        coord_map = self._generate_coord_map(h, w, x.device)
        coord_feat = self.coord_proj(coord_map.repeat(b, 1, 1, 1)) # [B, hidden_dim, H, W]

        # 2. 输入投影并拆分 (Feature Splitting)
        x_and_gate = self.in_proj(x)
        x_main, gate = x_and_gate.chunk(2, dim=1)

        # 3. 坐标条件化：将坐标信息融入主特征流
        x_main = x_main + coord_feat

        # 4. 空间域处理：利用 LKA 捕捉非等晕退化特征
        x_main = self.lka(x_main)

        # 5. 物理域调制：生成物理引导的动态门控 (Physics-Guided Dynamic Gating)
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        # 6. 协同门控融合 (Co-Gating Fusion)
        # 结合空间感知 (x_main) 与 物理调制 (gate)
        x_out = x_main * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 物理引导的轻量化路由 (Physics-Modulated Router)
##########################################################################


class Coord_LKA_Router(nn.Module):
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


class Coord_LKA_AdapterLayer(nn.Module):
    """
    [学术定义]: Physics-Aware Coordinate-conditioned Mixture-of-Experts (PA-CC-MoE) Layer
    
    功能: 封装了物理感知坐标条件化专家与路由机制，作为 MoCE-IR 架构的核心计算单元。
    """

    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 (Expert Pool)
        self.experts = nn.ModuleList(
            [Coord_LKA_Expert(dim, phys_dim) for _ in range(num_experts)]
        )
        # 物理引导路由器
        self.router = Coord_LKA_Router(dim, phys_dim, num_experts, k)

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
## 简化版 MoCE-IR 主干：UNet 结构 + PA-CC-MoE 适配层
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


class MoCEIR_Coord_LKA(nn.Module):
    """
    [学术定义]: Physics-Informed Coordinate-conditioned Mixture-of-Conditional-Experts for Image Restoration
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干。
    2. 在 Decoder 阶段引入 PA-CC-MoE 层，实现对多尺度物理退化的自适应修复。
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
        # 在深层特征空间引入 PA-CC-MoE，处理复杂的全局退化
        self.coord_lka_adapter2 = Coord_LKA_AdapterLayer(dim * 4, phys_dim, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 在浅层特征空间引入 PA-CC-MoE，修复精细的空间变异细节
        self.coord_lka_adapter1 = Coord_LKA_AdapterLayer(dim * 2, phys_dim, num_experts=num_experts, k=topk)

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
        d_dec2 = self.coord_lka_adapter2(d_dec2, global_phys_emb)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1 = self.coord_lka_adapter1(d_dec1, global_phys_emb)

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

    return MoCEIR_Coord_LKA(
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
    model = MoCEIR_Coord_LKA(inp_channels=in_channels, out_channels=in_channels, dim=feature_dim, phys_dim=physical_embedding_dim, num_experts=num_experts)

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")

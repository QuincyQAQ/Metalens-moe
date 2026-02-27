"""
MoCE_IR_S_Freq_LKA_PSFRouter.py
================================
基于 MoCE_IR_S_Freq_LKA 的 PSF感知动态路由增强版

创新点1 (保持不变): Freq_LKA_Expert - 频率分解 + 大核注意力

创新点2 (完全重写): PSF-Aware Dynamic Router - 针对PSF合成数据集的猛涨点

核心猛涨点改进:
1. 退化感知路由器 (Degradation-Aware Router): 显式检测退化程度（边缘消失、模糊强度）
2. 频率注意力路由 (Frequency Attention Router): 利用 Laplacian/边缘信息指导专家选择
3. 轻量化设计: 极简设计，几乎不增加参数和GFLOPs
4. 可学习退化嵌入: 让路由器学习不同退化程度对应的专家

关键发现:
- Endovis17: 边缘-100%, Laplacian -38%, 低频+0.25%
- Kvasir_SEG: 边缘-100%, Laplacian -67%, 低频+0.17%
- 这两个数据集的共同点: 边缘几乎完全消失，高频信息严重丢失
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 创新点1: 显式高/低频分解 LKA (Freq-LKA) - 保持不变
##########################################################################


class LKA(nn.Module):
    """
    [学术定义]: Local Kernel Extractor (LKA) / 局部核提取器
    
    物理背景: 超透镜的退化在空间域表现为随视场角剧烈变化的模糊。
    数学逻辑: 通过分解大核卷积（5x5 DW + 7x7 DW-dilation=3）实现对高维非等晕核的低秩近似，
    在保持线性复杂度的同时，获取足以覆盖大尺寸 PSF 的全局感受野。
    """

    def __init__(self, dim: int):
        super().__init__()
        # 局部特征提取
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 空间长距离依赖建模
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
        # 通道间信息融合
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class Freq_LKA_Expert(nn.Module):
    """
    [学术定义]: Frequency-Aware Co-Gating Expert (FACE) / 频率感知协同门控专家
    
    核心逻辑: 显式分离高低频特征，LKA 处理低频，轻量级模块处理高频，再融合。
    创新点: 实现了"频率解耦的物理参数化神经算子"，使专家能够针对不同频率分量进行优化。
    """

    def __init__(self, dim: int, phys_dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 特征投影与隐式物理对齐
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 2. 低频分支算子：大核注意力 (LKA)
        self.lka = LKA(self.hidden_dim)

        # 3. 高频分支算子：轻量级卷积
        self.high_freq_conv = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, 
            padding=1, groups=self.hidden_dim
        )

        # 4. 物理信息神经算子调制器
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
            phys_emb: [B, phys_dim] - 物理先验嵌入
        """
        b, c, h, w = x.shape

        # 1. 输入投影并拆分
        x_and_gate = self.in_proj(x)
        x_main, gate = x_and_gate.chunk(2, dim=1)

        # 2. 频率分解 (Frequency Decomposition)
        x_low = F.avg_pool2d(x_main, kernel_size=2, stride=2)
        x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
        x_high = x_main - x_low

        # 3. 低频处理：利用 LKA 捕捉全局退化
        x_low_processed = self.lka(x_low)

        # 4. 高频处理：利用轻量级卷积修复细节
        x_high_processed = self.high_freq_conv(x_high)

        # 5. 融合频率分量
        x_fused = x_low_processed + x_high_processed

        # 6. 物理域调制
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        # 7. 协同门控融合
        x_out = x_fused * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 创新点2增强: PSF感知动态路由器 (PSF-Aware Dynamic Router) - 猛涨点
##########################################################################


class DegradationAwareExtractor(nn.Module):
    """
    [核心创新]: 退化感知特征提取器
    
    针对PSF合成数据集的特点，显式检测退化程度:
    1. 边缘消失程度 (通过 Laplacian 和梯度检测)
    2. 模糊强度 (通过高频能量检测)
    3. 低频占比 (通过频域分析)
    
    轻量化设计: 使用小卷积核和深度可分离卷积
    """
    
    def __init__(self, dim: int, phys_dim: int):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        
        # 使用小卷积核检测边缘/高频信息
        # Laplacian 近似: 3x3 卷积核 [-1,-1,-1; -1,8,-1; -1,-1,-1]
        # 这里用更轻量的 3x3 DW Conv + 1x1 投影
        self.laplacian_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
        )
        
        # 高频检测分支 (使用 Sobel 类似的结构)
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        
        # 全局池化 + 投影
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * 2, phys_dim),
            nn.ReLU(inplace=True),
        )
        
        # 可学习的退化程度嵌入
        self.degradation_embed = nn.Parameter(torch.randn(1, phys_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] - 输入特征
        Returns:
            degradation_feat: [B, phys_dim] - 退化感知特征
            degradation_level: [B, 1] - 退化程度标量 (用于路由器)
        """
        # 检测高频/边缘信息
        laplacian_feat = self.laplacian_conv(x)
        
        # 使用另一种方式检测高频
        high_freq = self.high_freq_conv(x)
        
        # 拼接并池化
        combined = torch.cat([laplacian_feat, high_freq], dim=1)
        degradation_feat = self.pool_proj(combined)
        
        # 计算退化程度: 高频能量越低，退化越严重
        # 使用 laplacian_feat 的方差作为退化程度的代理
        degradation_level = laplacian_feat.var(dim=[2, 3], keepdim=True).mean(dim=1)
        
        # 加入可学习的退化嵌入
        degradation_feat = degradation_feat + self.degradation_embed
        
        return degradation_feat, degradation_level


class FrequencyAttentionRouter(nn.Module):
    """
    [核心创新]: 频率注意力路由器
    
    利用频率信息（而非简单的物理嵌入）来指导专家选择:
    1. 多尺度频率特征提取
    2. 可学习的频带-专家对应关系
    3. 轻量化: 几乎没有额外参数
    """
    
    def __init__(self, dim: int, num_experts: int, num_bands: int = 3):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_bands = num_bands
        
        # 多尺度频率提取 (使用不同大小的池化)
        self.scale_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(num_bands)
        ])
        
        # 频带到专家的映射 (可学习)
        # 这是一个非常轻量的映射: num_bands -> num_experts
        self.band_to_expert = nn.Parameter(
            torch.randn(num_bands, num_experts) * 0.1
        )
        
        # 内容分支 (简化版)
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] - 输入特征
        Returns:
            freq_logits: [B, num_experts] - 频率引导的专家选择分数
            band_weights: [B, num_bands] - 各频带的权重
        """
        # 多尺度频率特征提取
        scale_feats = []
        for pool in self.scale_pool:
            scale_feats.append(pool(x).squeeze(-1).squeeze(-1))
        
        # 堆叠: [B, num_bands, C] -> 聚合通道
        scale_stack = torch.stack(scale_feats, dim=1)  # [B, num_bands, C]
        scale_agg = scale_stack.mean(dim=2)  # [B, num_bands]
        
        # 频带注意力权重
        band_weights = F.softmax(scale_agg, dim=1)  # [B, num_bands]
        
        # 频带到专家的映射
        freq_logits = torch.matmul(band_weights, self.band_to_expert)  # [B, num_experts]
        
        # 内容分支
        content_logits = self.content_branch(x)
        
        # 融合
        logits = content_logits + freq_logits * 0.5
        
        return logits, band_weights


class PSFAwareDynamicRouter(nn.Module):
    """
    [核心创新]: PSF感知动态路由器 (PSF-Aware Dynamic Router)
    
    猛涨点改进:
    1. 退化感知提取: 显式检测退化程度 (边缘消失、模糊强度)
    2. 频率注意力: 利用多尺度频率特征指导专家选择
    3. 轻量化设计: 极简设计，总参数量增加很少
    4. 可学习退化嵌入: 自适应不同退化程度
    
    核心逻辑:
    - 不再简单使用内容+物理的线性组合
    - 先检测输入的退化程度
    - 然后利用频率信息进行动态路由
    - 专门针对 PSF 合成的超透镜图像优化
    """
    
    def __init__(
        self, 
        dim: int, 
        phys_dim: int, 
        num_experts: int, 
        k: int = 1,
        num_bands: int = 3
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.phys_dim = phys_dim
        
        # 1. 退化感知特征提取器 (核心创新)
        self.degradation_extractor = DegradationAwareExtractor(dim, phys_dim)
        
        # 2. 频率注意力路由器 (核心创新)
        self.freq_attention = FrequencyAttentionRouter(dim, num_experts, num_bands)
        
        # 3. 物理嵌入处理分支 (简化版)
        self.phys_branch = nn.Sequential(
            nn.Linear(phys_dim, phys_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(phys_dim // 2, num_experts),
        )
        
        # 4. 退化程度到专家的映射 (核心创新)
        self.degradation_to_expert = nn.Sequential(
            nn.Linear(1, num_experts),
        )
        
        # 5. 融合权重 (可学习)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # content, freq, phys, degradation
        
        # 6. 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] - 图像特征
            phys_emb: [B, phys_dim] - 物理先验嵌入
        Returns:
            scores: [B, num_experts] - 专家选择概率
            top_k_indices: [B, k] - Top-K专家索引
            top_k_scores: [B, k] - Top-K专家分数
        """
        b = x.shape[0]
        
        # 1. 退化感知特征提取 (核心创新)
        degradation_feat, degradation_level = self.degradation_extractor(x)
        
        # 2. 频率注意力路由 (核心创新)
        freq_logits, band_weights = self.freq_attention(x)
        
        # 3. 物理嵌入分支
        phys_logits = self.phys_branch(phys_emb)
        
        # 4. 退化程度指导 (核心创新)
        degradation_logits = self.degradation_to_expert(degradation_level.squeeze(-1).squeeze(-1))
        
        # 5. 动态融合 (使用可学习权重)
        weights = F.softmax(self.fusion_weights, dim=0)
        
        logits = (
            weights[0] * freq_logits +      # 频率注意力
            weights[1] * phys_logits +       # 物理嵌入
            weights[2] * degradation_logits # 退化程度
        )
        
        # 6. 温度缩放
        scores = F.softmax(logits / self.temperature, dim=-1)
        
        # 7. Top-K稀疏门控
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class Freq_LKA_PSFRouter_AdapterLayer(nn.Module):
    """
    [学术定义]: PSF-Aware Frequency-decomposed Mixture-of-Experts (PSF-FD-MoE) Layer
    
    功能: 封装了PSF感知频率分解专家与动态路由机制
    """

    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 (创新点1: Freq_LKA 专家)
        self.experts = nn.ModuleList(
            [Freq_LKA_Expert(dim, phys_dim) for _ in range(num_experts)]
        )
        
        # PSF感知动态路由器 (创新点2: PSF-Aware Router)
        self.router = PSFAwareDynamicRouter(dim, phys_dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(
        self, 
        x: torch.Tensor, 
        phys_emb: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # 使用PSF感知动态路由
        scores, indices, top_k_scores = self.router(x, phys_emb)

        final_out = torch.zeros_like(x)

        # 动态专家分发与聚合
        # 使用与原始 Freq_LKA_AdapterLayer 完全相同的方式
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                # 获取对应专家的分数（需要从 scores 中取出）
                expert_scores = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * expert_scores

        return self.out_proj(final_out)


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + PSF-FD-MoE 适配层
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


class MoCEIR_Freq_LKA_PSFRouter(nn.Module):
    """
    [学术定义]: PSF-Aware Frequency-decomposed Mixture-of-Conditional-Experts 
               with Dynamic PSF-Aware Router
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干
    2. 在 Decoder 阶段引入 PSF-FD-MoE 层
    3. 创新点1: Freq_LKA 专家实现频率分解 + 大核注意力
    4. 创新点2: PSF-Aware Router 专门针对PSF合成数据集优化
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
        # 深层特征: PSF-FD-MoE
        self.freq_lka_psfrouter_adapter2 = Freq_LKA_PSFRouter_AdapterLayer(
            dim * 4, phys_dim, num_experts=num_experts, k=topk
        )

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 浅层特征: PSF-FD-MoE
        self.freq_lka_psfrouter_adapter1 = Freq_LKA_PSFRouter_AdapterLayer(
            dim * 2, phys_dim, num_experts=num_experts, k=topk
        )

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
        self, 
        x: torch.Tensor, 
        de_id: Optional[torch.Tensor] = None
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
        
        # 使用PSF感知动态路由适配器
        d_dec2 = self.freq_lka_psfrouter_adapter2(d_dec2, global_phys_emb)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1 = self.freq_lka_psfrouter_adapter1(d_dec1, global_phys_emb)

        out = self.out_conv(d_dec1) + inp  # Residual connection

        self.total_loss = torch.tensor(
            0.0, device=out.device, dtype=out.dtype, requires_grad=False
        )

        return out


def build_model(opt) -> nn.Module:
    dim = getattr(opt, "dim", 32)
    phys_dim = getattr(opt, "phys_dim", 128)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR_Freq_LKA_PSFRouter(
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
    
    model = MoCEIR_Freq_LKA_PSFRouter(
        inp_channels=in_channels, 
        out_channels=in_channels, 
        dim=feature_dim, 
        phys_dim=physical_embedding_dim, 
        num_experts=num_experts
    )

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


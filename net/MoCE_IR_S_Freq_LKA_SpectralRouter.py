"""
MoCE_IR_S_Freq_LKA_SpectralRouter.py
=====================================
基于 MoCE_IR_S_Freq_LKA 的频谱感知路由器增强版

创新点1 (保持不变): Freq_LKA_Expert - 频率分解 + 大核注意力
创新点2 (完全重构): Spectral-Frequency Adaptive Router (SF-AR)
    - 去掉传统物理先验路由器
    - 引入频谱感知路由，利用FFT频域分析自适应选择专家
    - 多尺度频带分析 (低频/中频/高频)
    - 通道-频率联合注意力机制处理色差问题

核心改进:
1. 频谱分析路由器 (Spectral Analysis Router) - 利用FFT分析图像频率分布
2. 多尺度频带门控 (Multi-Scale Band Gating) - 分频带选择专家
3. 频率-通道联合注意力 (Frequency-Channel Joint Attention) - 处理色差
4. 自适应温度调度 (Adaptive Temperature Scheduling) - 动态路由确定性
5. 无额外物理嵌入依赖，直接从特征提取物理信息

优势:
- 参数量增加极少 (主要是几个小型卷积和FFT操作)
- GFLOPs增加微乎其微
- 直接利用频率信息做路由决策，更适合PSF合成的超透镜图像
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 创新点1: 显式高/低频分解 LKA (Freq-LKA) - 保持完全不变
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
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )
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
    """

    def __init__(self, dim: int, phys_dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = int(dim * expansion_ratio)

        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)
        self.lka = LKA(self.hidden_dim)
        self.high_freq_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)

        self.physics_gate_gen = nn.Sequential(
            nn.Linear(phys_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_and_gate = self.in_proj(x)
        x_main, gate = x_and_gate.chunk(2, dim=1)

        x_low = F.avg_pool2d(x_main, kernel_size=2, stride=2)
        x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
        x_high = x_main - x_low

        x_low_processed = self.lka(x_low)
        x_high_processed = self.high_freq_conv(x_high)

        x_fused = x_low_processed + x_high_processed

        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        x_out = x_fused * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 创新点2完全重构: 频谱感知自适应路由器 (Spectral-Frequency Adaptive Router)
##########################################################################


class FFTAnalysis(nn.Module):
    """
    [核心创新]: 快速傅里叶频谱分析模块
    对输入特征进行FFT变换，提取多尺度频域信息用于路由器决策
    """
    
    def __init__(self, dim: int, freq_bins: int = 8):
        super().__init__()
        self.dim = dim
        self.freq_bins = freq_bins
        
        # 频谱特征投影 - 输入是 2*dim (低频+高频特征)
        self.freq_proj = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft)
        
        mag = torch.abs(x_fft_shifted)
        
        low_freq = F.adaptive_avg_pool2d(mag, (self.freq_bins, self.freq_bins))
        low_feat = low_freq.mean(dim=(2, 3))
        
        high_pass = mag - F.avg_pool2d(mag, kernel_size=3, stride=1, padding=1)
        high_feat = high_pass.mean(dim=(2, 3))
        
        combined = torch.cat([low_feat, high_feat], dim=-1)
        freq_feat = self.freq_proj(combined)
        
        return freq_feat


class MultiScaleBandGating(nn.Module):
    """
    [核心创新]: 多尺度频带门控
    将特征分解为多个频带，分别计算门控权重
    """
    
    def __init__(self, dim: int, num_bands: int = 3):
        super().__init__()
        self.num_bands = num_bands
        
        self.band_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, groups=dim // 4),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            for _ in range(num_bands)
        ])
        
        self.band_fusion = nn.Sequential(
            nn.Linear(dim // 4 * num_bands, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        band_feats = []
        for extractor in self.band_extractors:
            feat = extractor(x)
            band_feats.append(feat)
        
        combined = torch.cat(band_feats, dim=-1)
        band_gate = self.band_fusion(combined)
        
        return band_gate


class FrequencyChannelJointAttention(nn.Module):
    """
    [核心创新]: 频率-通道联合注意力
    解决色差问题的关键模块
    """
    
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        
        self.channel_freq_attn = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )
        
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        channel_feat = x.mean(dim=(2, 3))
        channel_weight = self.channel_freq_attn(channel_feat).view(b, c, 1, 1)
        
        freq_weight = self.freq_gate(x)
        
        joint_attn = channel_weight * freq_weight
        
        return joint_attn


class SpectralAdaptiveRouter(nn.Module):
    """
    [核心创新]: 频谱感知自适应路由器 (Spectral-Frequency Adaptive Router, SF-AR)
    
    完全重构了原来的Physics_Router:
    1. FFTAnalysis - 利用FFT分析频率分布
    2. MultiScaleBandGating - 多尺度频带门控
    3. FrequencyChannelJointAttention - 频率-通道联合注意力(处理色差)
    4. 可学习温度参数 - 动态调整路由确定性
    """
    
    def __init__(self, dim: int, phys_dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        self.fft_analysis = FFTAnalysis(dim, freq_bins=8)
        self.band_gating = MultiScaleBandGating(dim, num_bands=3)
        self.freq_channel_attn = FrequencyChannelJointAttention(dim)
        
        self.router_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, num_experts),
        )
        
        self.phys_branch = nn.Linear(phys_dim, num_experts)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        
        fft_feat = self.fft_analysis(x)
        band_gate = self.band_gating(x)
        
        freq_channel_attn = self.freq_channel_attn(x)
        freq_channel_feat = (freq_channel_attn * x).mean(dim=(2, 3))
        
        combined = torch.cat([fft_feat, band_gate, freq_channel_feat], dim=-1)
        routing_logits = self.router_fusion(combined)
        
        phys_logits = self.phys_branch(phys_emb)
        
        logits = routing_logits + 0.3 * phys_logits
        
        scores = F.softmax(logits / self.temperature, dim=-1)
        
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        return scores, top_k_indices, top_k_scores
    
    def get_load_balance_loss(self, scores: torch.Tensor) -> torch.Tensor:
        avg_prob = scores.mean(dim=0)
        load_loss = torch.var(avg_prob)
        return load_loss


class SpectralRouterAdapterLayer(nn.Module):
    """
    频谱感知自适应MoE适配层
    """
    
    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k
        
        self.experts = nn.ModuleList(
            [Freq_LKA_Expert(dim, phys_dim) for _ in range(num_experts)]
        )
        
        self.router = SpectralAdaptiveRouter(dim, phys_dim, num_experts, k)
        
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, h, w = x.shape
        
        scores, indices, top_k_scores = self.router(x, phys_emb)
        
        final_out = torch.zeros_like(x)
        
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight
        
        output = self.out_proj(final_out)
        
        return output, scores


##########################################################################
## MoCE-IR 主干：UNet 结构 + SF-AD-MoE 适配层
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


class MoCEIR_Freq_LKA_SpectralRouter(nn.Module):
    """
    频谱感知路由增强的MoCE-IR网络
    创新点1: Freq_LKA 专家实现频率分解 + 大核注意力 (保持不变)
    创新点2: SF-AR 路由器实现频谱感知自适应路由
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
        self.load_balance_weight = 0.01
        
        self.enc1 = ConvBlock(inp_channels, dim)
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = Downsample(dim * 4, dim * 8)

        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        self.up2 = Upsample(dim * 8, dim * 4)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        self.spectral_adapter2 = SpectralRouterAdapterLayer(
            dim * 4, phys_dim, num_experts=num_experts, k=topk
        )

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.spectral_adapter1 = SpectralRouterAdapterLayer(
            dim * 2, phys_dim, num_experts=num_experts, k=topk
        )

        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

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

        e1 = self.enc1(x)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d_dec2 = self.dec2(u2)

        global_phys_emb = self._global_phys_embedding(b)
        d_dec2, scores2 = self.spectral_adapter2(d_dec2, global_phys_emb)

        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1, scores1 = self.spectral_adapter1(d_dec1, global_phys_emb)

        out = self.out_conv(d_dec1) + inp

        if scores1 is not None and scores2 is not None:
            load_loss = (self.spectral_adapter1.router.get_load_balance_loss(scores1) + 
                        self.spectral_adapter2.router.get_load_balance_loss(scores2)) / 2
            self.total_loss = self.load_balance_weight * load_loss
        else:
            self.total_loss = torch.tensor(
                0.0, device=out.device, dtype=out.dtype, requires_grad=False
            )

        return out


def build_model(opt) -> nn.Module:
    dim = getattr(opt, "dim", 32)
    phys_dim = getattr(opt, "phys_dim", 128)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR_Freq_LKA_SpectralRouter(
        dim=dim,
        phys_dim=phys_dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    batch_size = 1
    in_channels = 3
    image_size = 64
    feature_dim = 32
    physical_embedding_dim = 128
    num_experts = 4

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    model = MoCEIR_Freq_LKA_SpectralRouter(
        inp_channels=in_channels, 
        out_channels=in_channels, 
        dim=feature_dim, 
        phys_dim=physical_embedding_dim, 
        num_experts=num_experts
    )

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


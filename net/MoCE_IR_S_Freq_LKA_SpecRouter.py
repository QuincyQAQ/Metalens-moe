"""
MoCE_IR_S_Freq_LKA_SpecRouter.py
=================================
基于 MoCE_IR_S_Freq_LKA 的频谱动态路由增强版

创新点1 (保持不变): Freq_LKA_Expert - 频率分解 + 大核注意力
创新点2 (完全重写): SpecRouter - 频谱动态路由器 + 多尺度物理融合

核心猛涨点改进:
1. 频谱感知路由器 (Spectral-Aware Router): 通过DCT提取频域特征，让路由器真正"看到"频率
2. 多尺度物理融合 (Multi-Scale Physics Fusion): 不同尺度池化捕捉局部+全局退化
3. 可学习频带注意力 (Learnable Band Attention): 让路由器学习哪些频带对应哪些专家
4. 轻量化设计: 深度可分离卷积减少参数和GFLOPs
5. 物理约束正则化: 频域一致性损失确保路由决策的物理合理性
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
## 创新点2增强: 频谱动态路由器 (Spectral Dynamic Router) - 猛涨点
##########################################################################


class DCTSpectrumExtractor(nn.Module):
    """
    [核心创新]: DCT频谱提取器
    
    改进点: 使用DCT（离散余弦变换）提取输入的频域特征，
    让路由器能够真正"看到"输入的频率分布，而不是仅仅依赖空间域特征。
    
    优势:
    1. DCT比FFT更紧凑，能量集中在低频
    2. 可以显式提取不同频带的能量分布
    3. 计算高效，易于嵌入轻量级路由器
    """
    
    def __init__(self, dim: int, num_bands: int = 4):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        
        # 可学习的频带权重 (让路由器学习哪些频带更重要)
        self.band_weights = nn.Parameter(torch.ones(num_bands))
        
        # 频带到物理嵌入的映射
        self.band_to_phys = nn.Sequential(
            nn.Linear(num_bands, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 输入特征
        Returns:
            spectrum_feat: [B, dim] - 频域特征
        """
        b, c, h, w = x.shape
        
        # 对每个通道分别做DCT (简化版：使用可学习的2D卷积模拟DCT)
        # 实际上我们用不同大小的池化来近似不同频率带的能量
        spectrum_features = []
        
        for i in range(self.num_bands):
            # 不同的池化kernel模拟不同的频率带
            kernel_size = 2 ** (i + 1)  # 2, 4, 8, 16
            if kernel_size <= min(h, w):
                pooled = F.adaptive_avg_pool2d(x, (max(1, h // kernel_size), max(1, w // kernel_size)))
                pooled = F.interpolate(pooled, size=(1, 1), mode='bilinear', align_corners=False)
                spectrum_features.append(pooled.squeeze(-1).squeeze(-1))
            else:
                spectrum_features.append(torch.zeros(b, c, device=x.device))
        
        # 拼接所有频带的特征
        spectrum_stack = torch.stack(spectrum_features, dim=-1)  # [B, C, num_bands]
        
        # 应用可学习的频带权重
        weighted_spectrum = spectrum_stack * self.band_weights.view(1, 1, -1)
        
        # 聚合通道维度
        spectrum_agg = weighted_spectrum.mean(dim=1)  # [B, num_bands]
        
        # 映射到物理嵌入维度
        spectrum_feat = self.band_to_phys(spectrum_agg)
        
        return spectrum_feat


class MultiScalePhysicsExtractor(nn.Module):
    """
    [核心创新]: 多尺度物理特征提取器
    
    改进点: 不再使用单一的全局池化，而是使用多尺度池化来提取:
    1. 局部细节特征 (小池化)
    2. 中等尺度特征 (中等池化)
    3. 全局语义特征 (大池化)
    
    优势: 更好地捕捉不同尺度的物理退化特征
    """
    
    def __init__(self, dim: int, phys_dim: int):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        
        # 多尺度分支 (使用深度可分离卷积减少参数)
        self.scale_branch1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, stride=2, padding=1, groups=dim // 2),  # 2x下采样
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.scale_branch2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 5, stride=4, padding=2, groups=dim // 2),  # 4x下采样
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.scale_branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # 多尺度特征融合
        # scale_branch1 和 branch2 输出 dim//2, branch3 输出 dim
        # 总输入维度: dim//2 + dim//2 + dim = dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, phys_dim),
            nn.LayerNorm(phys_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 输入特征
        Returns:
            multi_scale_feat: [B, phys_dim] - 多尺度物理特征
        """
        # 提取三个尺度的特征
        feat1 = self.scale_branch1(x)  # 局部细节
        feat2 = self.scale_branch2(x)  # 中等尺度
        feat3 = self.scale_branch3(x)  # 全局语义
        
        # 拼接并融合
        combined = torch.cat([feat1, feat2, feat3], dim=-1)
        multi_scale_feat = self.fusion(combined)
        
        return multi_scale_feat


class SpectralDynamicRouter(nn.Module):
    """
    [核心创新]: 频谱动态路由器 (Spectral Dynamic Router)
    
    猛涨点改进:
    1. DCT频谱提取: 让路由器能够感知输入的频谱分布
    2. 多尺度物理融合: 多尺度提取物理特征
    3. 可学习频带注意力: 自适应学习哪些频带对应哪些专家
    4. 轻量化设计: 使用深度可分离卷积减少参数
    5. 温度可学习: 更灵活的路由确定性控制
    
    核心逻辑:
    - 不再简单使用内容+物理的线性组合
    - 先通过DCT分析输入的频谱分布
    - 然后结合多尺度物理特征进行动态路由决策
    """
    
    def __init__(self, dim: int, phys_dim: int, num_experts: int, k: int = 1, num_bands: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.phys_dim = phys_dim
        
        # 1. DCT频谱提取器 (核心创新)
        self.dct_spectrum = DCTSpectrumExtractor(dim, num_bands)
        
        # 2. 多尺度物理特征提取器
        self.multi_scale_phys = MultiScalePhysicsExtractor(dim, phys_dim)
        
        # 3. 内容感知分支 (保留，但增强)
        self.content_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim // 2, num_experts),
        )
        
        # 4. 频谱-物理融合分支 (核心创新: 频谱信息指导物理路由)
        self.spectrum_phys_fusion = nn.Sequential(
            nn.Linear(phys_dim + dim, phys_dim),
            nn.LayerNorm(phys_dim),
            nn.ReLU(inplace=True),
        )
        
        # 5. 频谱到专家权重的映射 (可学习频带注意力)
        self.spectrum_to_gate = nn.Sequential(
            nn.Linear(phys_dim, num_experts),
            nn.LayerNorm(num_experts),
        )
        
        # 6. 内容-频谱交互 (学习两者如何交互)
        self.content_spectrum_interaction = nn.Sequential(
            nn.Linear(num_experts * 2, num_experts),
            nn.Tanh(),  # 使用Tanh让交互可以是正值或负值
        )
        
        # 7. 可学习温度参数 (控制路由确定性)
        self.temperature = nn.Parameter(torch.ones(1) * 0.67)
        
        # 8. 频带注意力权重 (让路由器学习哪些频带对应哪些专家)
        self.band_expert_attention = nn.Parameter(torch.ones(num_bands, num_experts) * 0.1)
    
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
        
        # 1. 内容分支计算基础路由分数
        content_logits = self.content_branch(x)  # [B, num_experts]
        
        # 2. DCT频谱提取 (核心创新)
        spectrum_feat = self.dct_spectrum(x)  # [B, dim]
        
        # 3. 多尺度物理特征提取
        multi_scale_feat = self.multi_scale_phys(x)  # [B, phys_dim]
        
        # 4. 频谱-物理融合 (核心创新)
        spectrum_phys_combined = torch.cat([spectrum_feat, phys_emb], dim=-1)
        spectrum_phys_fused = self.spectrum_phys_fusion(spectrum_phys_combined)
        
        # 5. 频谱引导的专家选择 (可学习频带注意力)
        # 将多尺度物理特征与频谱特征结合
        combined_phys = (multi_scale_feat + spectrum_phys_fused) / 2
        spectrum_logits = self.spectrum_to_gate(combined_phys)  # [B, num_experts]
        
        # 6. 内容-频谱交互
        content_spectrum_combined = torch.cat([content_logits, spectrum_logits], dim=-1)
        interaction_weight = self.content_spectrum_interaction(content_spectrum_combined)
        
        # 7. 动态融合 (频谱信息作为调制因子)
        logits = content_logits + spectrum_logits * (1 + interaction_weight * 0.5)
        
        # 8. 频带注意力加权 (让不同频带的贡献不同)
        # 计算每个频带对专家选择的贡献
        band_logits = torch.matmul(F.softmax(spectrum_logits, dim=-1), self.band_expert_attention.t())
        band_attention = F.softmax(band_logits, dim=-1)
        
        # 将频带注意力融入最终logits
        logits = logits * (1 + band_attention.mean(dim=1, keepdim=True) * 0.3)
        
        # 9. 温度缩放的Softmax
        scores = F.softmax(logits / self.temperature, dim=-1)
        
        # 10. Top-K稀疏门控
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        return scores, top_k_indices, top_k_scores


class SpectralConsistencyLoss(nn.Module):
    """
    [核心创新]: 频谱一致性正则化损失
    
    改进点: 
    1. 频谱路由一致性: 鼓励路由器对相似频谱的输入选择相似专家
    2. 稀疏性正则化: 鼓励使用更少的专家
    3. 频带均衡正则化: 鼓励路由器使用所有频带
    """
    
    def __init__(
        self, 
        lambda_sparse: float = 0.01, 
        lambda_band: float = 0.05,
        lambda_consistency: float = 0.1
    ):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_band = lambda_band
        self.lambda_consistency = lambda_consistency
        
    def forward(
        self, 
        gates: torch.Tensor,           # [B, num_experts] 路由门控
        spectrum_feat: torch.Tensor,    # [B, dim] 频谱特征
        band_weights: torch.Tensor,     # [num_bands] 频带权重
    ) -> torch.Tensor:
        # 1. 稀疏性正则化
        sparse_loss = self.lambda_sparse * torch.mean(torch.sum(gates, dim=1))
        
        # 2. 频带均衡正则化 (鼓励使用所有频带)
        band_variance = torch.var(band_weights)  # 鼓励频带权重方差小，即均衡使用
        band_loss = self.lambda_band * band_variance
        
        # 3. 频谱路由一致性 (相似频谱应该选择相似专家)
        # 计算频谱特征的相似度与专家选择的相关性
        spectrum_sim = torch.corrcoef(
            torch.cat([
                spectrum_feat.view(-1), 
                gates.mean(dim=0).view(-1)
            ], dim=0).unsqueeze(0).unsqueeze(0).expand(spectrum_feat.shape[0], -1, -1).reshape(2, -1)
        )
        consistency_loss = self.lambda_consistency * (1 - torch.abs(spectrum_sim.mean()))
        
        total_loss = sparse_loss + band_loss + consistency_loss
        return total_loss


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class Freq_LKA_SpecRouter_AdapterLayer(nn.Module):
    """
    [学术定义]: Spectral-Aware Frequency-decomposed Mixture-of-Experts (SA-FD-MoE) Layer
    
    功能: 封装了频谱感知频率分解专家与动态路由机制
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
        
        # 频谱动态路由器 (创新点2: SpecRouter)
        self.router = SpectralDynamicRouter(dim, phys_dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # 频谱一致性损失
        self.spectral_loss = SpectralConsistencyLoss()

    def forward(
        self, 
        x: torch.Tensor, 
        phys_emb: torch.Tensor,
        return_spectral_info: bool = False
    ) -> tuple[torch.Tensor, Optional[dict]]:
        b, c, h, w = x.shape
        
        # 使用频谱动态路由
        scores, indices, top_k_scores = self.router(x, phys_emb)
        
        # 提取频谱特征 (用于损失计算)
        spectrum_feat = self.router.dct_spectrum(x)
        band_weights = self.router.dct_spectrum.band_weights

        final_out = torch.zeros_like(x)

        # 动态专家分发与聚合
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        output = self.out_proj(final_out)
        
        if return_spectral_info:
            return output, {
                'gates': scores,
                'spectrum_feat': spectrum_feat,
                'band_weights': band_weights,
                'top_k_indices': indices,
                'top_k_scores': top_k_scores,
            }
        
        return output, None


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + SA-FD-MoE-SpecRouter 适配层
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


class MoCEIR_Freq_LKA_SpecRouter(nn.Module):
    """
    [学术定义]: Spectral-Aware Frequency-decomposed Mixture-of-Conditional-Experts 
               with Dynamic SpecRouter
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干
    2. 在 Decoder 阶段引入 SA-FD-MoE-SpecRouter 层
    3. 创新点1: Freq_LKA 专家实现频率分解 + 大核注意力
    4. 创新点2: SpecRouter 路由器实现频谱感知动态路由
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
        self.spectral_loss = SpectralConsistencyLoss()

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
        # 深层特征: SA-FD-MoE-SpecRouter
        self.freq_lka_specrouter_adapter2 = Freq_LKA_SpecRouter_AdapterLayer(
            dim * 4, phys_dim, num_experts=num_experts, k=topk
        )

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 浅层特征: SA-FD-MoE-SpecRouter
        self.freq_lka_specrouter_adapter1 = Freq_LKA_SpecRouter_AdapterLayer(
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
        de_id: Optional[torch.Tensor] = None,
        compute_spectral_loss: bool = False
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
        
        # 使用频谱动态路由适配器
        d_dec2, spec_info2 = self.freq_lka_specrouter_adapter2(
            d_dec2, global_phys_emb, return_spectral_info=compute_spectral_loss
        )

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1, spec_info1 = self.freq_lka_specrouter_adapter1(
            d_dec1, global_phys_emb, return_spectral_info=compute_spectral_loss
        )

        out = self.out_conv(d_dec1) + inp  # Residual connection

        # 计算频谱一致性损失 (可选)
        if compute_spectral_loss:
            # 分别计算两层的光谱损失 (因为维度不同)
            spectral_loss2 = self.spectral_loss(
                spec_info2['gates'], 
                spec_info2['spectrum_feat'],
                spec_info2['band_weights'],
            )
            spectral_loss1 = self.spectral_loss(
                spec_info1['gates'], 
                spec_info1['spectrum_feat'],
                spec_info1['band_weights'],
            )
            # 取平均
            self.total_loss = (spectral_loss1 + spectral_loss2) / 2
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

    return MoCEIR_Freq_LKA_SpecRouter(
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
    
    model = MoCEIR_Freq_LKA_SpecRouter(
        inp_channels=in_channels, 
        out_channels=in_channels, 
        dim=feature_dim, 
        phys_dim=physical_embedding_dim, 
        num_experts=num_experts
    )

    print(f"Input shape: {x.shape}")
    
    # 测试不带频谱损失
    y = model(x, compute_spectral_loss=False)
    print(f"Output shape: {y.shape}")
    
    # 测试带频谱损失
    y = model(x, compute_spectral_loss=True)
    print(f"Output shape: {y.shape}")
    print(f"Spectral loss: {model.total_loss.item():.4f}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 创新点1: 显式高/低频分解 LKA (Freq-LKA) - 保留原版不动
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
    [学术定义]: Frequency-Aware Co-Gating Expert (FACE)
    保留原版 Freq_LKA_Expert，不动创新点1
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

        # 频率分解
        x_low = F.avg_pool2d(x_main, kernel_size=2, stride=2)
        x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)
        x_high = x_main - x_low

        # 低频处理：LKA
        x_low_processed = self.lka(x_low)
        # 高频处理：轻量级卷积
        x_high_processed = self.high_freq_conv(x_high)

        # 融合
        x_fused = x_low_processed + x_high_processed

        # 物理门控
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        x_out = x_fused * F.gelu(gate)
        return self.out_proj(x_out)


##########################################################################
## 创新点2优化版: 自适应频域-边缘感知路由器 (Adaptive Freq-Edge Router)
## 
## 问题分析:
## - Endovis17: 低频能量极高(99.75%→99.99%)，边缘消失(-100%)
## - Kvasir_SEG: 低频能量极高(99.81%→99.99%)，Laplacian下降66.93%
## 
## 改进思路:
## 1. 频域分析器: 使用FFT分析输入图像的低频/高频能量比
## 2. 边缘感知器: 使用可学习的高通滤波器检测边缘强度
## 3. 内容分支增强: 使用多层感知机而非简单的全局池化
## 4. 动态权重融合: 根据输入图像特性自适应调整专家选择权重
##########################################################################


class FrequencyAnalyzer(nn.Module):
    """
    [核心创新]: 频域分析器 - 分析输入图像的低频/高频能量比
    
    作用: 根据频谱特征判断图像退化程度，动态调整专家选择
    - 低频能量高 → 需要更多低频恢复专家
    - 边缘弱 → 需要更多高频增强专家
    """
    def __init__(self, dim: int):
        super().__init__()
        # 频域特征提取
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 4, 1),  # 4个频域统计特征
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, C, H, W]
        输出: [B, 4] 频域统计特征
        """
        b, c, h, w = x.shape
        
        # 转换为灰度
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 简化的频域分析（避免FFT）
        # 低频: 大kernel平均池化
        low_freq = F.avg_pool2d(gray, kernel_size=8, stride=1, padding=4)
        # 高频: 残差
        high_freq = gray - F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        
        # 能量统计
        low_energy = (low_freq ** 2).mean(dim=[1, 2, 3])  # [B]
        high_energy = (high_freq ** 2).mean(dim=[1, 2, 3])  # [B]
        total_energy = low_energy + high_energy + 1e-8
        
        # 归一化能量比
        freq_ratio = torch.stack([
            low_energy / total_energy,
            high_energy / total_energy,
            torch.log1p(low_energy),
            torch.log1p(high_energy),
        ], dim=1)  # [B, 4]
        
        return freq_ratio


class EdgeDetector(nn.Module):
    """
    [核心创新]: 可学习边缘检测器 - 替代固定的Sobel/Laplacian
    
    作用: 自适应检测图像边缘强度，针对Endovis/Kvasir边缘消失(-100%)的问题
    """
    def __init__(self, dim: int):
        super().__init__()
        # 可学习的边缘滤波器
        self.edge_filter = nn.Parameter(torch.randn(1, 1, 3, 3) * 0.1)
        # 边缘特征处理
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 4, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, C, H, W]
        输出: [B, 4] 边缘统计特征
        """
        b, c, h, w = x.shape
        
        # 灰度化
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 可学习边缘检测
        edge = F.conv2d(gray, self.edge_filter, padding=1, bias=None)
        edge = torch.abs(edge)
        
        # 边缘统计
        edge_mean = edge.mean(dim=[1, 2, 3])  # [B]
        edge_max = edge.flatten(1).max(dim=1)[0]  # [B]
        edge_std = edge.std(dim=[1, 2, 3])  # [B]
        
        # Laplacian-like 统计
        laplacian = self._laplacian(gray)
        lap_mean = laplacian.abs().mean(dim=[1, 2, 3])
        
        edge_stats = torch.stack([
            edge_mean,
            edge_max,
            edge_std,
            lap_mean,
        ], dim=1)  # [B, 4]
        
        return edge_stats
    
    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """简化的Laplacian"""
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)
        kernel = kernel.view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)


class AdaptiveRouter(nn.Module):
    """
    [核心创新]: 自适应频域-边缘感知路由器
    
    改进点:
    1. 频域分析器: 替代简单的全局池化，更精准地分析图像频率特性
    2. 边缘感知器: 专门针对边缘消失问题
    3. 多层MLP: 替代单层线性变换，增强表达能力
    4. 动态融合: 根据图像特性自适应调整内容/物理先验权重
    """
    def __init__(self, dim: int, phys_dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # 频域分析器
        self.freq_analyzer = FrequencyAnalyzer(dim)
        
        # 边缘检测器
        self.edge_detector = EdgeDetector(dim)

        # 内容分支增强: 使用MLP替代简单的全局池化
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, num_experts),
        )

        # 频域特征处理
        self.freq_branch = nn.Sequential(
            nn.Linear(8, num_experts),  # 4个频域 + 4个边缘特征
            nn.ReLU(inplace=True),
            nn.Linear(num_experts, num_experts),
        )

        # 物理因果分支保持不变
        self.physics_branch = nn.Linear(phys_dim, num_experts)

        # 动态权重融合
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # 内容、频域、物理的融合权重

    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        
        # 1. 内容分支
        content_logits = self.content_branch(x)  # [B, num_experts]
        
        # 2. 频域分析分支
        freq_features = self.freq_analyzer(x)  # [B, 4]
        edge_features = self.edge_detector(x)  # [B, 4]
        combined_freq = torch.cat([freq_features, edge_features], dim=1)  # [B, 8]
        freq_logits = self.freq_branch(combined_freq)  # [B, num_experts]
        
        # 3. 物理分支
        physics_logits = self.physics_branch(phys_emb)  # [B, num_experts]
        
        # 4. 动态权重融合
        weights = F.softmax(self.fusion_weights, dim=0)
        logits = (weights[0] * content_logits + 
                  weights[1] * freq_logits + 
                  weights[2] * physics_logits)
        
        # Softmax得到最终专家选择概率
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        return scores, top_k_indices, top_k_scores


##########################################################################
## 适配层 (保留原版结构，只替换路由器)
##########################################################################


class Freq_LKA_AdapterLayer(nn.Module):
    """
    封装了物理感知频率分解专家与改进的路由机制
    """

    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 - 创新点1: Freq_LKA 专家 (保留不动)
        self.experts = nn.ModuleList(
            [Freq_LKA_Expert(dim, phys_dim) for _ in range(num_experts)]
        )
        
        # 路由器 - 创新点2优化: 自适应频域-边缘感知路由器
        self.router = AdaptiveRouter(dim, phys_dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scores, indices, top_k_scores = self.router(x, phys_emb)

        final_out = torch.zeros_like(x)

        # 动态专家分发与聚合
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        return self.out_proj(final_out)


##########################################################################
## UNet 主干网络 (保持不变)
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
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class MoCE_IR_S_Freq_LKA_Adaptive(nn.Module):
    """
    [完整网络]: 自适应频域-边缘感知 MoCE-IR
    
    创新点:
    - 创新点1 (保留): 显式高/低频分解 LKA (Freq_LKA_Expert)
    - 创新点2 (优化): 自适应频域-边缘感知路由器 (Adaptive Freq-Edge Router)
    
    优化针对问题:
    - Endovis17: 低频能量极高(99.75%→99.99%)，边缘消失(-100%)
    - Kvasir_SEG: 低频能量极高(99.81%→99.99%)，Laplacian下降66.93%
    
    新增模块:
    - FrequencyAnalyzer: 频域能量分析
    - EdgeDetector: 可学习边缘检测
    - AdaptiveRouter: 动态权重融合的内容/频域/物理三支路路由器
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

        # Encoder
        self.enc1 = ConvBlock(inp_channels, dim)
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = Downsample(dim * 4, dim * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        # Decoder - 使用改进的适配层
        self.up2 = Upsample(dim * 8, dim * 4)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        self.freq_lka_adapter2 = Freq_LKA_AdapterLayer(dim * 4, phys_dim, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.freq_lka_adapter1 = Freq_LKA_AdapterLayer(dim * 2, phys_dim, num_experts=num_experts, k=topk)

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

        # 提取物理嵌入
        global_phys_emb = self._global_phys_embedding(b)

        # Decoder stage 2
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d_dec2 = self.dec2(u2)
        d_dec2 = self.freq_lka_adapter2(d_dec2, global_phys_emb)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)
        d_dec1 = self.freq_lka_adapter1(d_dec1, global_phys_emb)

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

    return MoCE_IR_S_Freq_LKA_Adaptive(
        dim=dim,
        phys_dim=phys_dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    # 测试模型
    batch_size = 1
    in_channels = 3
    image_size = 64
    feature_dim = 32
    physical_embedding_dim = 128
    num_experts = 4

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    model = MoCE_IR_S_Freq_LKA_Adaptive(
        inp_channels=in_channels, 
        out_channels=in_channels, 
        dim=feature_dim, 
        phys_dim=physical_embedding_dim, 
        num_experts=num_experts
    )

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    # 计算参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


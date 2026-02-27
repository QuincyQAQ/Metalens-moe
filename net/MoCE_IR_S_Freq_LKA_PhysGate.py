"""
MoCE_IR_S_Freq_LKA_PhysGate.py
================================
基于 MoCE_IR_S_Freq_LKA 的物理引导路由增强版

创新点1 (保持不变): Freq_LKA_Expert - 频率分解 + 大核注意力
创新点2 (大幅增强): PhysGate_Router - 多粒度物理约束自适应门控路由

核心改进:
1. 多粒度物理约束建模 (光学MTF + 空间相关性 + 亮度响应)
2. 物理感知路由 (Physical-Aware Routing) 而非简单线性映射
3. 稀疏Top-K门控 (降低计算，提升路由决策质量)
4. 物理一致性正则化 (损失函数级约束)
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
## 创新点2增强: 多粒度物理约束自适应门控路由 (PhysGate Router)
##########################################################################


class MultiGrainedPhysConstraint(nn.Module):
    """
    [核心创新]: 多粒度物理约束建模
    
    改进点: 不再简单使用物理嵌入的线性映射，而是从特征中提取多粒度物理约束:
    1. MTF约束 (Modulation Transfer Function) - 提取中高频响应特性
    2. PSF约束 (Point Spread Function) - 提取空间模糊特性  
    3. Luminance约束 - 提取亮度分布特性
    
    优势: 更细粒度地建模物理退化，提升路由决策质量
    """
    
    def __init__(self, dim: int, phys_dim: int):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        
        # 1. MTF分支 - 提取中高频响应 (使用不同池化尺度模拟不同频率响应)
        self.mtf_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, stride=2, padding=1),  # 下采样提取高频
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim // 2, phys_dim),
        )
        
        # 2. PSF分支 - 提取空间模糊特性 (大核卷积模拟PSF)
        self.psf_branch = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=7, padding=3, groups=dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim // 2, phys_dim),
        )
        
        # 3. Luminance分支 - 提取全局亮度分布
        self.lum_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, phys_dim),
        )
        
        # 4. 约束融合层
        self.constraint_fusion = nn.Sequential(
            nn.Linear(phys_dim * 3, phys_dim),
            nn.LayerNorm(phys_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 输入特征
        Returns:
            phys_constraint: [B, phys_dim] - 融合后的物理约束向量
        """
        # 提取三种粒度的物理约束
        mtf_feat = self.mtf_branch(x)      # [B, phys_dim]
        psf_feat = self.psf_branch(x)     # [B, phys_dim]
        lum_feat = self.lum_branch(x)     # [B, phys_dim]
        
        # 拼接并融合
        combined = torch.cat([mtf_feat, psf_feat, lum_feat], dim=-1)
        phys_constraint = self.constraint_fusion(combined)
        
        return phys_constraint


class PhysGateRouter(nn.Module):
    """
    [核心创新]: 物理感知自适应门控路由器
    
    改进点:
    1. 不再是简单的 Content + Physics 线性相加
    2. 使用多粒度物理约束动态调制路由决策
    3. 引入稀疏Top-K门控 (只激活部分expert，降低计算同时提升决策质量)
    4. 可学习的温度参数 (控制路由的"确定性")
    """
    
    def __init__(self, dim: int, phys_dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        # 1. 多粒度物理约束建模 (新增)
        self.phys_constraint = MultiGrainedPhysConstraint(dim, phys_dim)
        
        # 2. 内容感知分支 (保留)
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )
        
        # 3. 物理约束到专家选择权重的映射 (替代简单的 Linear)
        self.phys_to_gate = nn.Sequential(
            nn.Linear(phys_dim, num_experts),
            nn.LayerNorm(num_experts),
        )
        
        # 4. 内容-物理交互层 (新增 - 学习两者的交互权重)
        self.gate_interaction = nn.Sequential(
            nn.Linear(num_experts * 2, num_experts),
            nn.Sigmoid(),
        )
        
        # 5. 可学习温度参数 (控制路由确定性)
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
        # 1. 内容分支计算基础路由分数
        content_logits = self.content_branch(x)  # [B, num_experts]
        
        # 2. 多粒度物理约束 (从特征中提取，而非直接用 phys_emb)
        phys_constraint = self.phys_constraint(x)  # [B, phys_dim]
        
        # 3. 物理约束映射为路由权重 (与 phys_emb 结合)
        # 使用学习到的物理约束 + 原始物理先验
        phys_logits_raw = self.phys_to_gate(phys_constraint + phys_emb)  # [B, num_experts]
        
        # 4. 内容-物理交互 (学习更好的融合方式)
        combined = torch.cat([content_logits, phys_logits_raw], dim=-1)
        interaction_weight = self.gate_interaction(combined)  # [B, num_experts]
        
        # 5. 交互调制后的路由分数
        logits = content_logits * (1 + interaction_weight)  # 物理作为调制因子
        
        # 6. 温度缩放的Softmax (增加路由确定性)
        scores = F.softmax(logits / self.temperature, dim=-1)
        
        # 7. Top-K 稀疏门控
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        return scores, top_k_indices, top_k_scores


class PhysConsistencyLoss(nn.Module):
    """
    [核心创新]: 物理一致性正则化损失
    
    在损失函数级别加入物理一致性约束:
    - 鼓励高物理约束区域的expert激活权重更高
    - 稀疏性正则化 (降低冗余计算)
    """
    
    def __init__(self, lambda_sparse: float = 0.01, lambda_phys: float = 0.1):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_phys = lambda_phys
        
    def forward(
        self, 
        gates: torch.Tensor,           # [B, num_experts] 路由门控
        phys_constraint: torch.Tensor, # [B, phys_dim] 物理约束
        pred: torch.Tensor,            # [B, C, H, W] 预测
        target: torch.Tensor           # [B, C, H, W] 目标
    ) -> torch.Tensor:
        # 1. 基础重建损失
        recon_loss = F.l1_loss(pred, target)
        
        # 2. 稀疏性正则化 (鼓励使用更少的expert)
        # 只对Top-1路由生效，Top-K可以适当放松
        sparse_loss = self.lambda_sparse * torch.mean(torch.sum(gates, dim=1))
        
        # 3. 物理一致性正则化
        # 物理约束高的区域应该激活更多expert (即专家应该关注物理复杂区域)
        # 这里用物理约束的方差与门控的相关性来衡量
        phys_variance = torch.var(phys_constraint, dim=1).mean()  # 物理约束变化程度
        gate_variance = torch.var(gates, dim=1).mean()            # 门控变化程度
        
        # 鼓励两者变化趋势一致
        phys_loss = self.lambda_phys * (1 - torch.abs(phys_variance - gate_variance))
        
        total_loss = recon_loss + sparse_loss + phys_loss
        return total_loss


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class Freq_LKA_PhysGate_AdapterLayer(nn.Module):
    """
    [学术定义]: Physics-Aware Frequency-decomposed Mixture-of-Experts (PA-FD-MoE) Layer
    
    功能: 封装了物理感知频率分解专家与增强的物理门控路由机制
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
        
        # 物理门控路由器 (创新点2: 增强版物理路由)
        self.router = PhysGateRouter(dim, phys_dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # 物理一致性损失
        self.phys_loss = PhysConsistencyLoss()

    def forward(
        self, 
        x: torch.Tensor, 
        phys_emb: torch.Tensor,
        return_phys_info: bool = False
    ) -> tuple[torch.Tensor, Optional[dict]]:
        b, c, h, w = x.shape
        
        # 使用增强的物理门控路由
        scores, indices, top_k_scores = self.router(x, phys_emb)
        
        # 提取物理约束信息 (用于损失计算)
        phys_constraint = self.router.phys_constraint(x)

        final_out = torch.zeros_like(x)

        # 动态专家分发与聚合
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        output = self.out_proj(final_out)
        
        if return_phys_info:
            return output, {
                'gates': scores,
                'phys_constraint': phys_constraint,
                'top_k_indices': indices,
                'top_k_scores': top_k_scores,
            }
        
        return output, None


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + PA-FD-MoE-PhysGate 适配层
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


class MoCEIR_Freq_LKA_PhysGate(nn.Module):
    """
    [学术定义]: Physics-Informed Frequency-decomposed Mixture-of-Conditional-Experts 
               with Enhanced PhysGate Routing
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干
    2. 在 Decoder 阶段引入 PA-FD-MoE-PhysGate 层
    3. 创新点1: Freq_LKA 专家实现频率分解 + 大核注意力
    4. 创新点2: PhysGate 路由器实现多粒度物理约束自适应门控
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
        self.phys_loss = PhysConsistencyLoss()

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
        # 深层特征: PA-FD-MoE-PhysGate
        self.freq_lka_physgate_adapter2 = Freq_LKA_PhysGate_AdapterLayer(
            dim * 4, phys_dim, num_experts=num_experts, k=topk
        )

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 浅层特征: PA-FD-MoE-PhysGate
        self.freq_lka_physgate_adapter1 = Freq_LKA_PhysGate_AdapterLayer(
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
        target: Optional[torch.Tensor] = None,
        compute_phys_loss: bool = False
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
        
        # 使用增强的物理门控适配器
        d_dec2, phys_info2 = self.freq_lka_physgate_adapter2(
            d_dec2, global_phys_emb, return_phys_info=compute_phys_loss
        )

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1, phys_info1 = self.freq_lka_physgate_adapter1(
            d_dec1, global_phys_emb, return_phys_info=compute_phys_loss
        )

        out = self.out_conv(d_dec1) + inp  # Residual connection

        # 计算物理一致性损失 (可选)
        if compute_phys_loss and target is not None:
            # 综合两层的信息
            combined_gates = (phys_info2['gates'] + phys_info1['gates']) / 2
            combined_phys = (phys_info2['phys_constraint'] + phys_info1['phys_constraint']) / 2
            
            phys_loss = self.phys_loss(
                combined_gates, 
                combined_phys,
                out, 
                target
            )
            self.total_loss = phys_loss
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

    return MoCEIR_Freq_LKA_PhysGate(
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
    target = torch.randn(batch_size, in_channels, image_size, image_size)
    
    model = MoCEIR_Freq_LKA_PhysGate(
        inp_channels=in_channels, 
        out_channels=in_channels, 
        dim=feature_dim, 
        phys_dim=physical_embedding_dim, 
        num_experts=num_experts
    )

    print(f"Input shape: {x.shape}")
    
    # 测试不带物理损失
    y = model(x, compute_phys_loss=False)
    print(f"Output shape: {y.shape}")
    
    # 测试带物理损失
    y = model(x, target=target, compute_phys_loss=True)
    print(f"Output shape: {y.shape}")
    print(f"Physical loss: {model.total_loss.item():.4f}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


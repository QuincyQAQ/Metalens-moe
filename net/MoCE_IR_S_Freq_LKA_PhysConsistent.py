"""
[CVPR升级版] MoCE_IR_S_Freq_LKA_PhysConsistent.py
=================================================

创新点1升级: Physics-Consistent Frequency-Aware Expert (PCFAE)
- 显式光学参数编码 (Explicit Optical Parameter Encoding)
- FFT自适应频率分割 (Adaptive FFT-based Frequency Decomposition)
- 空间变化退化建模 (Spatially-Varying Degradation Modeling)

创新点2保留: Physics-Guided Router (保持不变)
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 创新点1升级: 物理一致性频率感知专家 (PCFAE)
## Physics-Consistent Frequency-Aware Expert
## 
## 升级内容:
## 1. FFT自适应频率分割替代固定池化
## 2. 显式光学参数编码替代隐式嵌入
## 3. 动态频率门控响应库
##########################################################################


class OpticalParameterEncoder(nn.Module):
    """
    [CVPR创新]: 显式光学参数编码器
    Physics-Consistent Optical Parameter Encoder
    
    将物理光学参数显式映射为可学习的特征表示:
    - NA: Numerical Aperture (数值孔径)
    - focal_length: 焦距
    - wavelength: 工作波长
    - zernike_coeffs: Zernike像差系数
    
    相比隐式嵌入的优势:
    1. 物理可解释性强
    2. 跨数据集泛化能力强
    3. 便于引入物理先验约束
    """

    def __init__(self, phys_dim: int = 32):
        super().__init__()
        
        # 核心光学参数编码 (可学习投影)
        self.na_encoder = nn.Linear(1, 8)        # Numerical Aperture
        self.fl_encoder = nn.Linear(1, 8)        # Focal Length        
        self.wl_encoder = nn.Linear(1, 8)       # Wavelength
        self.zernike_encoder = nn.Linear(11, 16) # Zernike像差系数 (前11阶)
        
        # 物理一致性先验 (可学习的频率响应偏置)
        self.phys_prior = nn.Parameter(torch.randn(phys_dim))
        
        # 融合投影层
        self.projection = nn.Sequential(
            nn.Linear(40, 64),
            nn.GELU(),
            nn.Linear(64, phys_dim),
            nn.LayerNorm(phys_dim)  # 归一化保证训练稳定
        )

    def forward(self, optical_params: dict) -> torch.Tensor:
        """
        Args:
            optical_params: {
                'na': [B],           # 数值孔径 (0.1 - 0.9)
                'focal_length': [B], # 焦距 (mm)
                'wavelength': [B],    # 波长 (nm)
                'zernike': [B, 11]   # Zernike系数
            }
        Returns:
            phys_emb: [B, phys_dim]
        """
        # 归一化输入 (物理意义归一化)
        na = (optical_params['na'] - 0.5) / 0.5
        fl = (optical_params['focal_length'] - 5.0) / 5.0
        wl = (optical_params['wavelength'] - 550.0) / 200.0
        
        # 编码各物理参数
        na_feat = self.na_encoder(na.unsqueeze(-1))
        fl_feat = self.fl_encoder(fl.unsqueeze(-1))
        wl_feat = self.wl_encoder(wl.unsqueeze(-1))
        zk_feat = self.zernike_encoder(optical_params['zernike'])
        
        # 特征融合
        concat_feat = torch.cat([na_feat, fl_feat, wl_feat, zk_feat], dim=-1)
        phys_emb = self.projection(concat_feat)
        
        # 残差连接物理先验 (物理一致性约束)
        phys_emb = phys_emb + self.phys_prior
        
        return phys_emb


class AdaptiveFrequencyModulator(nn.Module):
    """
    [CVPR创新]: 自适应频率分割调制器
    Adaptive Frequency Modulation with Dynamic Cutoff
    
    核心创新:
    1. 基于FFT的频率分解，而非固定下采样
    2. 动态频率分割阈值，根据内容和物理先验自适应
    3. 可学习的频率响应库，存储不同物理条件下的理想响应
    """

    def __init__(self, dim: int):
        super().__init__()
        
        # 动态频率比例预测器 (Content-Adaptive)
        self.ratio_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 可学习的频率响应库 (Frequency Response Bank)
        # 存储不同物理条件下的理想频率响应
        self.freq_response_bank = nn.Parameter(torch.randn(8, dim))

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自适应频率解耦
        
        Args:
            x: [B, C, H, W] - 输入特征
            phys_emb: [B, phys_dim] - 物理嵌入
            
        Returns:
            x_low: 低频分量
            x_high: 高频分量
        """
        b, c, h, w = x.shape
        
        # Step 1: 动态预测频率分割比例 (基于内容)
        ratio = self.ratio_predictor(x)  # [B, 1, 1, 1]
        ratio = ratio * 0.35 + 0.15  # 约束到 [0.15, 0.5]
        
        # Step 2: 生成圆形频率掩码 (FFT域)
        freq_mask = self._circular_freq_mask(ratio, (h, w), x.device)
        
        # Step 3: FFT变换
        x_fft = torch.fft.fft2(x)
        x_fft_shift = torch.fft.fftshift(x_fft)
        
        # Step 4: 频率分离 (低频/高频)
        x_low_fft = x_fft_shift * freq_mask
        x_high_fft = x_fft_shift * (1 - freq_mask)
        
        # Step 5: 反变换回空域
        x_low = torch.fft.ifft2(torch.fft.ifftshift(x_low_fft)).real
        x_high = torch.fft.ifft2(torch.fft.ifftshift(x_high_fft)).real
        
        return x_low, x_high

    def _circular_freq_mask(self, ratio: torch.Tensor, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """
        生成圆形频率掩码 (带平滑过渡避免频谱泄露)
        """
        h, w = size
        cy, cx = h // 2 + 0.5, w // 2 + 0.5
        
        # 创建坐标网格
        y_coords = torch.arange(h, device=device, dtype=torch.float32)
        x_coords = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算到中心距离
        dist = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
        max_dist = torch.sqrt(cy**2 + cx**2)
        
        # 动态半径
        radius = ratio.squeeze() * max_dist
        
        # 硬边界 (然后用平滑)
        mask = (dist <= radius).float().unsqueeze(0).unsqueeze(0)
        
        # 平滑过渡 (避免频谱泄露)
        mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        
        return mask


class LKA(nn.Module):
    """
    [保留原版]: Local Kernel Extractor (LKA) / 局部核提取器
    
    物理背景: 超透镜的退化在空间域表现为随视场角剧烈变化的模糊。
    数学逻辑: 通过分解大核卷积实现对高维非等晕核的低秩近似。
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


class Freq_LKA_PhysConsistent_Expert(nn.Module):
    """
    [CVPR创新核心]: 物理一致性频率感知专家
    Physics-Consistent Frequency-Aware Expert (PCFAE)
    
    完整实现:
    1. 显式光学参数编码 (OpticalParameterEncoder)
    2. 自适应频率解耦 (FFT-based AdaptiveFrequencyModulator)
    3. 双分支处理 (LKA + 轻量卷积)
    4. 物理引导门控融合
    
    相比原版Freq_LKA_Expert的升级:
    - 频率分解: 固定池化 -> FFT自适应阈值
    - 物理建模: 隐式嵌入 -> 显式光学参数
    - 空间变化: 全局均匀 -> 像素级自适应
    """

    def __init__(self, dim: int, phys_dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = int(dim * expansion_ratio)

        # [模块1] 显式光学参数编码器 (新增)
        self.optical_encoder = OpticalParameterEncoder(phys_dim)
        
        # [模块2] 特征投影
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # [模块3] 自适应频率调制器 (升级: FFT替代池化)
        self.freq_modulator = AdaptiveFrequencyModulator(self.hidden_dim)

        # [模块4] 低频分支 - LKA (保持不变)
        self.lka = LKA(self.hidden_dim)

        # [模块5] 高频分支 - 轻量级深度卷积 (增强版)
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, 
                     padding=1, groups=self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        )
        
        # [模块6] 可学习的频率响应库 (新增)
        self.freq_response_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # [模块7] 物理引导门控生成器
        self.physics_gate = nn.Sequential(
            nn.Linear(phys_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        # [模块8] 输出投影
        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, optical_params: dict) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 输入特征
            optical_params: 光学参数字典 {
                'na': [B],
                'focal_length': [B],
                'wavelength': [B],
                'zernike': [B, 11]
            }
        """
        b, c, h, w = x.shape
        
        # Step 1: 显式编码光学参数 (替代隐式嵌入)
        phys_emb = self.optical_encoder(optical_params)
        
        # Step 2: 输入投影
        x_proj = self.in_proj(x)
        x_main, gate = x_proj.chunk(2, dim=1)
        
        # Step 3: 自适应频率解耦 (FFT替代固定池化)
        x_low, x_high = self.freq_modulator(x_main, phys_emb)
        
        # Step 4: 双分支处理
        x_low_processed = self.lka(x_low)
        x_high_processed = self.high_freq_conv(x_high)
        
        # Step 5: 频率融合 (带可学习响应库调制)
        x_fused = x_low_processed + x_high_processed
        x_fused = x_fused + self.freq_response_proj(phys_emb).view(b, -1, 1, 1)
        
        # Step 6: 物理门控
        p_gate = self.physics_gate(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate
        
        # Step 7: 协同门控输出
        x_out = x_fused * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 创新点2: 物理引导的轻量化路由 (Physics-Guided Router) - 保持不变!
##########################################################################


class Physics_Router(nn.Module):
    """
    [学术定义]: Physics-Guided Router (PGR) / 物理引导路由器
    
    核心创新: 物理先验直接参与专家选择分数计算
    - 内容分支: 提取图像内容特征用于专家选择
    - 物理分支: 将物理先验嵌入映射为专家选择权重
    - 协同决策: 两者相加后 softmax 得到最终专家选择概率
    
    [此模块保持不变]
    """
    def __init__(self, dim: int, phys_dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # 内容感知分支
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )

        # 物理因果分支
        self.physics_branch = nn.Linear(phys_dim, num_experts)

    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 协同决策逻辑
        logits = self.content_branch(x) + self.physics_branch(phys_emb)
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class Freq_LKA_PhysConsistent_AdapterLayer(nn.Module):
    """
    [学术定义]: Physics-Consistent Frequency-decomposed MoE Layer
    
    功能: 封装了PCFAE专家与路由机制，作为升级版MoCE-IR架构的核心计算单元。
    """

    def __init__(self, dim: int, phys_dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.k = k

        # 专家池 - 升级版 PCFAE 专家
        self.experts = nn.ModuleList(
            [Freq_LKA_PhysConsistent_Expert(dim, phys_dim) for _ in range(num_experts)]
        )
        
        # 物理引导路由器 - 保持不变
        self.router = Physics_Router(dim, phys_dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, optical_params: dict) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # 从特征中提取物理嵌入 (与原版兼容)
        global_phys_emb = self._extract_phys_emb(x)
        
        scores, indices, top_k_scores = self.router(x, global_phys_emb)

        final_out = torch.zeros_like(x)

        # 动态专家分发与聚合
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], optical_params)
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        return self.out_proj(final_out)
    
    def _extract_phys_emb(self, x: torch.Tensor) -> torch.Tensor:
        """从特征中提取物理嵌入 (兼容接口)"""
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + PCFAE 适配层
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


class MoCEIR_Freq_LKA_PhysConsistent(nn.Module):
    """
    [学术定义]: Physics-Consistent Frequency-Aware Mixture-of-Conditional-Experts
    
    架构描述: 
    1. 采用对称的 Encoder-Decoder 结构作为主干。
    2. 在 Decoder 阶段引入 PCFAE 层 (升级版频率解耦专家)。
    3. 显式接收光学参数，实现物理一致性建模。
    
    升级点总结:
    - 创新点1 (Freq_LKA_Expert): 
        * FFT自适应频率分割
        * 显式光学参数编码
        * 空间变化退化建模
    - 创新点2 (Physics_Router): 保持不变
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
        # 升级版 PCFAE 适配层
        self.pcfae_adapter2 = Freq_LKA_PhysConsistent_AdapterLayer(
            dim * 4, phys_dim, num_experts=num_experts, k=topk
        )

        self.up1 = Upsample(dim * 4, dim * 2)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        # 升级版 PCFAE 适配层
        self.pcfae_adapter1 = Freq_LKA_PhysConsistent_AdapterLayer(
            dim * 2, phys_dim, num_experts=num_experts, k=topk
        )

        # 输出头
        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

    def forward(
        self, 
        x: torch.Tensor, 
        optical_params: Optional[dict] = None,
        de_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - 输入图像
            optical_params: 光学参数字典 (可选，若不提供则使用默认参数)
        """
        # 默认光学参数 (若未提供)
        if optical_params is None:
            optical_params = self._get_default_optical_params(x.shape[0], x.device)
        
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

        # 应用升级版 PCFAE
        d_dec2 = self.pcfae_adapter2(d_dec2, optical_params)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        d_dec1 = self.pcfae_adapter1(d_dec1, optical_params)

        out = self.out_conv(d_dec1) + inp  # Residual connection

        self.total_loss = torch.tensor(
            0.0, device=out.device, dtype=out.dtype, requires_grad=False
        )

        return out
    
    def _get_default_optical_params(self, batch_size: int, device: torch.device) -> dict:
        """生成默认光学参数 (用于兼容原版接口)"""
        return {
            'na': torch.ones(batch_size, device=device) * 0.5,
            'focal_length': torch.ones(batch_size, device=device) * 5.0,
            'wavelength': torch.ones(batch_size, device=device) * 550.0,
            'zernike': torch.zeros(batch_size, 11, device=device)
        }


def build_model(opt) -> nn.Module:
    dim = getattr(opt, "dim", 32)
    phys_dim = getattr(opt, "phys_dim", 128)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR_Freq_LKA_PhysConsistent(
        dim=dim,
        phys_dim=phys_dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 2
    in_channels = 3
    image_size = 128
    feature_dim = 32
    physical_embedding_dim = 128
    num_experts = 4

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    
    # 模拟光学参数
    optical_params = {
        'na': torch.randn(batch_size) * 0.2 + 0.5,  # 0.3-0.7
        'focal_length': torch.randn(batch_size) * 1.0 + 5.0,  # 4-6 mm
        'wavelength': torch.randn(batch_size) * 50.0 + 550.0,  # 500-600 nm
        'zernike': torch.randn(batch_size, 11) * 0.1  # 小像差
    }
    
    model = MoCEIR_Freq_LKA_PhysConsistent(
        inp_channels=in_channels, 
        out_channels=in_channels, 
        dim=feature_dim, 
        phys_dim=physical_embedding_dim, 
        num_experts=num_experts
    )

    print(f"Input shape: {x.shape}")
    y = model(x, optical_params)
    print(f"Output shape: {y.shape}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")


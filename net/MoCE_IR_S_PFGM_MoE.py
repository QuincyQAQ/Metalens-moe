from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 核心创新模块：物理引导傅里叶门控曼巴专家 (PFGM-Expert)
## Physics-Guided Fourier-Gated Mamba MoE (PFGM-MoE)
##########################################################################


class FourierGating(nn.Module):
    """
    傅里叶门控模块 (Fourier Gating Unit)
    在频域进行全局特征交互，以 O(N log N) 复杂度实现真正的全局感受野。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 1x1 频域通道投影
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=1)
        # 预留：可以扩展为不同方向的频域门控
        self.conv_w = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        orig_dtype = x.dtype

        # cuFFT 在 half 精度下对非 2 的幂尺寸支持有限，这里强制使用 float32 做 FFT
        x_fp32 = x.float()

        # 1. 快速傅里叶变换到频域：复杂度 O(N log N)
        x_fft = torch.fft.rfft2(x_fp32, norm="ortho")

        # 2. 频域全局交互 (模拟全局卷积)
        #    先对频谱取幅值，作为 real 特征，通过 1x1 conv 生成频域门控权重
        amp = torch.abs(x_fft)
        gate = torch.sigmoid(self.conv_h(amp))
        x_fft = x_fft * gate

        # 3. 逆傅里叶变换回到空间域，并还原到原始 dtype（如 fp16）
        x_out = torch.fft.irfft2(x_fft, s=(h, w), norm="ortho")
        return x_out.to(orig_dtype)


class SimpleMamba(nn.Module):
    """
    简化版曼巴算子 (Simplified Mamba/SSM)
    实现空间自适应的选择性扫描。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1)
        # 深度可分离 1D 卷积，沿空间序列进行扫描
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # 选择性扫描参数（简化版）
        self.x_proj = nn.Linear(dim, dim)
        self.dt_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_in = self.in_proj(x)
        x_main, res = x_in.chunk(2, dim=1)

        # 展平为序列进行扫描: [B, C, H, W] -> [B, C, L]
        x_seq = rearrange(x_main, "b c h w -> b c (h w)")
        x_seq = self.conv1d(x_seq)

        # 简化版选择性扫描逻辑 (Selective Scan)
        # 这里模拟 Mamba 的核心：输入相关的状态转移
        x_seq_t = rearrange(x_seq, "b c l -> b l c")
        gate = torch.sigmoid(self.x_proj(x_seq_t))
        gate = rearrange(gate, "b l c -> b c l")
        x_seq = x_seq * gate

        x_out = rearrange(x_seq, "b c (h w) -> b c h w", h=h, w=w)
        return self.out_proj(x_out * F.silu(res))


class PFGM_Expert(nn.Module):
    """
    物理引导傅里叶门控曼巴专家 (PFGM Expert)
    结合了傅里叶频域全局感知与曼巴空间自适应选择性扫描。
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 输入投影
        self.in_proj = nn.Conv2d(dim, self.hidden_dim, kernel_size=1)

        # 2. 频域分支：傅里叶门控 (全局感知)
        self.fourier_branch = FourierGating(self.hidden_dim)

        # 3. 空域分支：曼巴扫描 (空间自适应)
        self.mamba_branch = SimpleMamba(self.hidden_dim)

        # 4. 物理引导的特征融合
        self.physics_fusion = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 5. 输出投影
        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C] (物理先验嵌入)
        """
        x = self.in_proj(x)

        # 频域与空域并行处理
        f_feat = self.fourier_branch(x)
        m_feat = self.mamba_branch(x)

        # 物理引导的动态融合: phys_emb 决定频域和空域特征的权重
        p_weight = self.physics_fusion(phys_emb).view(x.shape[0], -1, 1, 1)
        combined = f_feat * p_weight + m_feat * (1.0 - p_weight)

        return self.out_proj(combined)


##########################################################################
## 物理引导的熵约束路由 (Physics-Entropy Router)
##########################################################################


class PFGM_Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.router_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )
        self.phys_proj = nn.Linear(dim, num_experts)

    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 结合内容与物理先验
        logits = self.router_net(x) + self.phys_proj(phys_emb)
        scores = F.softmax(logits, dim=-1)

        # 计算路由熵 (用于后续 Loss 约束，确保专家特异化) —— 此处仅保留 scores 接口
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class PFGM_AdapterLayer(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([PFGM_Expert(dim) for _ in range(num_experts)])
        self.router = PFGM_Router(dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C]
        """
        b, c, h, w = x.shape
        scores, indices, top_k_scores = self.router(x, phys_emb)

        final_out = torch.zeros_like(x)
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)  # [B]
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                # 权重调制
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        return self.out_proj(final_out)


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + PFGM-MoE 适配层
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
    简化版 MoCE-IR：
      - 3 层 UNet 结构
      - 在 decoder 阶段引入 PFGM_AdapterLayer 作为 PFGM-MoE 消融实验
      - 不依赖外部物理标签，物理嵌入由特征自适应提取
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

        # Encoder
        self.enc1 = ConvBlock(inp_channels, dim)
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = Downsample(dim * 4, dim * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        # Decoder
        self.up2 = Upsample(dim * 8, dim * 4)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        self.pfgm2 = PFGM_AdapterLayer(dim * 4, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        # 上采样后与 enc1 拼接，通道数为 dim*2 + dim = 3*dim
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.pfgm1 = PFGM_AdapterLayer(dim * 2, num_experts=num_experts, k=topk)

        # 输出头
        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _global_phys_embedding(feat: torch.Tensor) -> torch.Tensor:
        """
        从特征中自适应提取物理先验嵌入：
          - 使用全局平均池化获得 [B, C]
        """
        return feat.mean(dim=(2, 3))

    def forward(
        self, x: torch.Tensor, de_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
            de_id: 兼容 train.py 的第二个参数（未使用）
        """
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

        # 物理嵌入来自该尺度特征
        phys_emb2 = self._global_phys_embedding(d_dec2)
        d_dec2 = self.pfgm2(d_dec2, phys_emb2)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        phys_emb1 = self._global_phys_embedding(d_dec1)
        d_dec1 = self.pfgm1(d_dec1, phys_emb1)

        out = self.out_conv(d_dec1) + inp

        # 本模型不引入额外平衡损失，但保持 total_loss 接口
        self.total_loss = torch.tensor(
            0.0, device=out.device, dtype=out.dtype, requires_grad=False
        )

        return out


##########################################################################
## 构建函数：与现有框架保持一致
##########################################################################


def build_model(opt) -> nn.Module:
    """
    从全局配置构建 PFGM-MoE 消融模型。

    仅使用 opt 中通用字段：dim, num_exp_blocks, topk 等，
    保证在现有脚本下即可插拔使用。
    """
    dim = getattr(opt, "dim", 32)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR(
        dim=dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    # 简单自检：形状与稳定性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoCEIR(dim=32, num_experts=4, topk=1).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")



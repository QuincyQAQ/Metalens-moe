from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 核心创新模块：物理感知动态门控大核专家 (PADG-LKE)
## Physics-Aware Dynamic Gated Large-Kernel Expert
##########################################################################


class LKA(nn.Module):
    """
    Large Kernel Attention (LKA) 模块
    通过分解大核卷积（5x5 DW + 7x7 DW-dilation=3）实现大感受野的注意力，
    计算复杂度保持线性。
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


class PADG_LKE_Expert(nn.Module):
    """
    物理感知动态门控大核专家 (PADG-LKE Expert)
    结合 LKA 全局感受野 与 物理先验调制的动态门控。
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 1. 输入投影，显式分出门控分支
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 2. 大核注意力算子
        self.lka = LKA(self.hidden_dim)

        # 3. 物理先验驱动的动态门控生成
        self.physics_gate_gen = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 4. 输出投影
        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C] (物理先验嵌入，与通道数 C 对齐)
        """
        b, c, h, w = x.shape

        # 1. 输入投影并拆分
        x_and_gate = self.in_proj(x)  # [B, 2*hidden_dim, H, W]
        x_main, gate = x_and_gate.chunk(2, dim=1)

        # 2. LKA 处理主分支
        x_main = self.lka(x_main)

        # 3. 物理引导门控：生成 [B, hidden_dim, 1, 1] 的门控权重
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        # 4. 动态门控融合
        x_out = x_main * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 物理引导的轻量化路由 (Physics-Modulated Router)
##########################################################################


class PADG_Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # 内容特征路由分支
        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )

        # 物理先验路由分支
        self.physics_branch = nn.Linear(dim, num_experts)

    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C]
        Returns:
            scores: [B, num_experts]
            top_k_indices: [B, k]
            top_k_scores: [B, k]
        """
        logits = self.content_branch(x) + self.physics_branch(phys_emb)
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class PADG_AdapterLayer(nn.Module):
    """
    简化版 PADG-LKE Mixture-of-Experts 适配层：
      - 使用 PADG_LKE_Expert 作为大核专家
      - 使用 PADG_Router 进行内容 + 物理先验联合路由
      - 接口与标准卷积块兼容，仅在特征通道维度上工作
    """

    def __init__(self, dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList(
            [PADG_LKE_Expert(dim) for _ in range(num_experts)]
        )
        self.router = PADG_Router(dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C]
        """
        b, c, h, w = x.shape
        scores, indices, top_k_scores = self.router(x, phys_emb)

        final_out = torch.zeros_like(x)

        # 简化版 dispatch/combine：按样本子集路由到各个专家
        for i in range(self.router.num_experts):
            mask = (indices == i).any(dim=-1)  # [B]
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
    简化版 MoCE-IR：
      - 3 层 UNet 结构
      - 在 decoder 阶段引入 PADG_AdapterLayer 作为 PADG-LKE 消融实验
      - 物理嵌入由特征自适应提取（不依赖外部物理标签）
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
        self.padg2 = PADG_AdapterLayer(dim * 4, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        # 上采样后与 enc1 拼接，通道数为 dim*2 + dim = 3*dim
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.padg1 = PADG_AdapterLayer(dim * 2, num_experts=num_experts, k=topk)

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

        phys_emb2 = self._global_phys_embedding(d_dec2)
        d_dec2 = self.padg2(d_dec2, phys_emb2)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        phys_emb1 = self._global_phys_embedding(d_dec1)
        d_dec1 = self.padg1(d_dec1, phys_emb1)

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
    从全局配置构建 PADG-LKE 消融模型。

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



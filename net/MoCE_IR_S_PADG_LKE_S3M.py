"""
MoCE-IR-S + PADG-LKE + S3M (Selective Scan State Space Modulation) - self-contained.

核心创新点 1: 物理引导动态门控 + 大核注意力 (PADG-LKE)
核心创新点 2: 物理调制的轻量化选择性扫描 (S3M / Mamba-lite)

注意：
- 本文件不引用其他网络/模块，保证 train.py 的 net_snapshot 能完整拷贝可复现网络定义。
- forward(x, de_id=None) 接口与工程现有训练逻辑兼容（de_id 会被忽略）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
## 核心创新模块 1：大核注意力 (LKA)
##########################################################################


class LKA(nn.Module):
    """Large Kernel Attention (LKA) 模块：5x5 DW + 7x7 DW(dilation=3) + 1x1."""

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


##########################################################################
## 核心创新模块 2：轻量化选择性扫描 (Selective Scan / Mamba-lite)
##########################################################################


class SelectiveScanLite(nn.Module):
    """
    轻量化选择性扫描模块 (Inspired by Mamba/SSM).

    说明：
    - 为了保持 self-contained & 易训练，这里实现一个“扫描风格”的线性复杂度模块。
    - 物理先验 phys_emb 通过线性映射生成 delta 门控，对 token 序列进行选择性更新。
    """

    def __init__(self, dim: int, phys_dim: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.phys_dim = int(phys_dim) if phys_dim is not None else int(dim)

        # 将 phys_emb 投影到与序列特征相同的维度（修复 dim 不一致问题）
        if self.phys_dim != self.dim:
            self.phys_proj = nn.Linear(self.phys_dim, self.dim)
        else:
            self.phys_proj = nn.Identity()

        self.phys_to_delta = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, phys_dim]
        Returns:
            x_out: [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")  # [B, L, C]

        phys = self.phys_proj(phys_emb)  # [B, C]
        delta = torch.sigmoid(self.phys_to_delta(phys)).unsqueeze(1)  # [B, 1, C]

        # Simplified selective update (scan-like gating)
        x_scanned = x_flat * delta + (1.0 - delta) * torch.tanh(x_flat)
        x_out = self.out_proj(x_scanned)
        x_out = rearrange(x_out, "b (h w) c -> b c h w", h=h, w=w)
        return x_out


##########################################################################
## 物理引导门控大核曼巴专家 (PG-GLKM Expert)
##########################################################################


class PG_GLKM_Expert(nn.Module):
    """
    PG-GLKM Expert: LKA(局部-中程) + S3M(全局) + 物理引导门控融合。
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0, *, enable_s3m: bool = True):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(self.dim * float(expansion_ratio))
        self.enable_s3m = bool(enable_s3m)

        # 输入投影，显式分出门控分支
        self.in_proj = nn.Conv2d(self.dim, self.hidden_dim * 2, kernel_size=1)

        # 局部-中程：大核注意力
        self.lka = LKA(self.hidden_dim)

        # 全局：选择性扫描（phys_emb 维度为 dim，需要投影到 hidden_dim）
        self.s3m = SelectiveScanLite(self.hidden_dim, phys_dim=self.dim)

        # 物理先验驱动动态门控（生成 hidden_dim 门控向量）
        self.physics_gate_gen = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(self.hidden_dim, self.dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape

        x_and_gate = self.in_proj(x)
        x_main, gate = x_and_gate.chunk(2, dim=1)  # [B, hidden_dim, H, W] each

        x_lka = self.lka(x_main)
        if self.enable_s3m:
            x_s3m = self.s3m(x_main, phys_emb)
            x_fused = x_lka + x_s3m
        else:
            x_fused = x_lka

        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        x_out = x_fused * F.gelu(gate)
        return self.out_proj(x_out)


##########################################################################
## 物理引导路由与适配层 (MoE)
##########################################################################


class PG_GLKM_Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = int(num_experts)
        self.k = int(k)

        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, self.num_experts),
        )
        self.physics_branch = nn.Linear(dim, self.num_experts)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor):
        logits = self.content_branch(x) + self.physics_branch(phys_emb)
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


class PG_GLKM_AdapterLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        k: int = 1,
        *,
        enable_s3m: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.k = int(k)

        self.experts = nn.ModuleList(
            [PG_GLKM_Expert(self.dim, enable_s3m=enable_s3m) for _ in range(self.num_experts)]
        )
        self.router = PG_GLKM_Router(self.dim, self.num_experts, self.k)
        self.out_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        scores, indices, _ = self.router(x, phys_emb)
        final_out = torch.zeros_like(x)

        for i in range(self.num_experts):
            mask = (indices == i).any(dim=-1)  # [B]
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        return self.out_proj(final_out)


##########################################################################
## 主干网络：MoCE-IR-S + PG-GLKM-MoE
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


class MoCEIR_PG_GLKM_S3M(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        num_experts: int = 4,
        topk: int = 1,
        *,
        enable_s3m: bool = True,
    ):
        super().__init__()
        self.total_loss = None
        self.enable_s3m = bool(enable_s3m)

        # Encoder
        self.enc1 = ConvBlock(inp_channels, dim)
        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = nn.Conv2d(dim * 4, dim * 8, 4, 2, 1)

        # Bottleneck
        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        # Decoder
        self.up2 = nn.ConvTranspose2d(dim * 8, dim * 4, 4, 2, 1)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        self.pg_glkm2 = PG_GLKM_AdapterLayer(dim * 4, num_experts, topk, enable_s3m=self.enable_s3m)

        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.pg_glkm1 = PG_GLKM_AdapterLayer(dim * 2, num_experts, topk, enable_s3m=self.enable_s3m)

        self.out_conv = nn.Conv2d(dim * 2, out_channels, 3, 1, 1)

    @staticmethod
    def _get_phys_emb(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor, de_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = x

        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d_dec2 = self.dec2(u2)
        d_dec2 = self.pg_glkm2(d_dec2, self._get_phys_emb(d_dec2))

        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)
        d_dec1 = self.pg_glkm1(d_dec1, self._get_phys_emb(d_dec1))

        out = self.out_conv(d_dec1) + inp
        self.total_loss = torch.tensor(0.0, device=out.device, dtype=out.dtype, requires_grad=False)
        return out


def build_model(opt) -> nn.Module:
    dim = getattr(opt, "dim", 32)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)
    enable_s3m = getattr(opt, "enable_s3m", True)
    return MoCEIR_PG_GLKM_S3M(
        dim=dim,
        num_experts=num_experts,
        topk=topk,
        enable_s3m=enable_s3m,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoCEIR_PG_GLKM_S3M(dim=32, num_experts=4, topk=1, enable_s3m=True).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")



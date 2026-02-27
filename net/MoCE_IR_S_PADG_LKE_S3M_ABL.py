"""
MoCE-IR-S + PADG-LKE + S3M(Ablation OFF) - self-contained.

与 `MoCE_IR_S_PADG_LKE_S3M.py` 结构一致，但将核心创新点 2（S3M / SelectiveScanLite）
完全关闭，用于消融实验。为了确保 net_snapshot 可复现，这里不 import 任何其他网络文件。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LKA(nn.Module):
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


class SelectiveScanLite(nn.Module):
    """
    保留模块壳（以便结构一致），但在本消融版本中不会被调用。
    """

    def __init__(self, dim: int, phys_dim: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.phys_dim = int(phys_dim) if phys_dim is not None else int(dim)
        if self.phys_dim != self.dim:
            self.phys_proj = nn.Linear(self.phys_dim, self.dim)
        else:
            self.phys_proj = nn.Identity()
        self.phys_to_delta = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        phys = self.phys_proj(phys_emb)
        delta = torch.sigmoid(self.phys_to_delta(phys)).unsqueeze(1)
        x_scanned = x_flat * delta + (1.0 - delta) * torch.tanh(x_flat)
        x_out = self.out_proj(x_scanned)
        x_out = rearrange(x_out, "b (h w) c -> b c h w", h=h, w=w)
        return x_out


class PG_GLKM_Expert(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 2.0, *, enable_s3m: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(self.dim * float(expansion_ratio))
        self.enable_s3m = bool(enable_s3m)  # 强制为 False（消融）

        self.in_proj = nn.Conv2d(self.dim, self.hidden_dim * 2, kernel_size=1)
        self.lka = LKA(self.hidden_dim)
        self.s3m = SelectiveScanLite(self.hidden_dim, phys_dim=self.dim)
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
        x_main, gate = x_and_gate.chunk(2, dim=1)

        x_lka = self.lka(x_main)
        # 消融：不使用 S3M
        x_fused = x_lka

        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate
        x_out = x_fused * F.gelu(gate)
        return self.out_proj(x_out)


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
    def __init__(self, dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.experts = nn.ModuleList([PG_GLKM_Expert(self.dim, enable_s3m=False) for _ in range(self.num_experts)])
        self.router = PG_GLKM_Router(self.dim, self.num_experts, self.k)
        self.out_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        scores, indices, _ = self.router(x, phys_emb)
        final_out = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight
        return self.out_proj(final_out)


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


class MoCEIR_PG_GLKM_S3M_ABL(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        num_experts: int = 4,
        topk: int = 1,
    ):
        super().__init__()
        self.total_loss = None

        self.enc1 = ConvBlock(inp_channels, dim)
        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = nn.Conv2d(dim * 4, dim * 8, 4, 2, 1)

        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        self.up2 = nn.ConvTranspose2d(dim * 8, dim * 4, 4, 2, 1)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        self.pg_glkm2 = PG_GLKM_AdapterLayer(dim * 4, num_experts, topk)

        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.pg_glkm1 = PG_GLKM_AdapterLayer(dim * 2, num_experts, topk)

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
    return MoCEIR_PG_GLKM_S3M_ABL(dim=dim, num_experts=num_experts, topk=topk)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoCEIR_PG_GLKM_S3M_ABL(dim=32, num_experts=4, topk=1).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")



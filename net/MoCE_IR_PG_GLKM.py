import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

##########################################################################
## 核心创新模块 1：大核注意力 (LKA)
##########################################################################


class LKA(nn.Module):
    """
    Large Kernel Attention (LKA) 模块
    通过分解大核卷积（5x5 DW + 7x7 DW-dilation=3）实现大感受野的注意力。
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


##########################################################################
## 核心创新模块 2：轻量化选择性扫描 (Selective Scan / Mamba-lite)
##########################################################################


class SelectiveScanLite(nn.Module):
    """
    轻量化选择性扫描模块 (Inspired by Mamba/SSM)
    利用物理先验调制扫描参数，实现线性复杂度的全局建模。
    """

    def __init__(self, dim: int, d_state: int = 16, phys_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.phys_dim = phys_dim if phys_dim is not None else dim

        # 物理感知的参数生成器（从物理嵌入 -> 扫描参数）
        self.phys_to_delta = nn.Linear(self.phys_dim, dim)
        self.phys_to_ABC = nn.Linear(self.phys_dim, d_state * 3)  # A, B, C matrices（当前未显式使用，预留扩展）

        self.x_proj = nn.Linear(dim, d_state)
        self.dt_proj = nn.Linear(self.phys_dim, dim)

        # 简化版扫描：使用门控机制模拟状态转移
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]，其中 C = dim
            phys_emb: [B, P]，P = phys_dim（通常为原始特征通道数）
        """
        b, c, h, w = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")

        # 1. 物理调制参数
        delta = torch.sigmoid(self.phys_to_delta(phys_emb)).unsqueeze(1)  # [B, 1, C]
        _ = self.phys_to_ABC(phys_emb).unsqueeze(1)  # 预留：未来可扩展为显式 SSM 状态矩阵

        # 2. 选择性门控扫描 (Simplified)
        # 使用物理感知的 delta 来调制特征通过率
        x_scanned = x_flat * delta + (1.0 - delta) * torch.tanh(x_flat)

        x_out = self.out_proj(x_scanned)
        x_out = rearrange(x_out, "b (h w) c -> b c h w", h=h, w=w)
        return x_out


##########################################################################
## 核心专家：物理引导门控大核曼巴专家 (PG-GLKM Expert)
##########################################################################


class PG_GLKM_Expert(nn.Module):
    """
    物理引导门控大核曼巴专家 (PG-GLKM Expert)
    融合 LKA (局部-中程) 与 Selective Scan (全局) 的物理感知专家。
    """

    def __init__(self, dim: int, expansion_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expansion_ratio)

        # 输入投影
        self.in_proj = nn.Conv2d(dim, self.hidden_dim * 2, kernel_size=1)

        # 1. 局部-中程：大核注意力（工作在 hidden_dim 通道上）
        self.lka = LKA(self.hidden_dim)

        # 2. 全局：选择性扫描（工作在 hidden_dim 通道上，物理嵌入维度 = dim）
        self.ssm = SelectiveScanLite(self.hidden_dim, d_state=16, phys_dim=dim)

        # 3. 物理先验驱动的动态门控（生成 hidden_dim 维度的门控张量）
        self.physics_gate_gen = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )

        # 输出投影
        self.out_proj = nn.Conv2d(self.hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]，C = dim
            phys_emb: [B, C]，C = dim
        """
        b, c, h, w = x.shape

        # 1. 输入投影并拆分
        x_and_gate = self.in_proj(x)  # [B, 2*hidden_dim, H, W]
        x_main, gate = x_and_gate.chunk(2, dim=1)  # [B, hidden_dim, H, W] * 2

        # 2. 协同处理：LKA (空间) + SSM (序列/全局)
        x_lka = self.lka(x_main)
        x_ssm = self.ssm(x_main, phys_emb)
        x_fused = x_lka + x_ssm

        # 3. 物理引导门控，生成 [B, hidden_dim, 1, 1] 的门控权重
        p_gate = self.physics_gate_gen(phys_emb).view(b, -1, 1, 1)
        gate = gate * p_gate

        # 4. 动态融合
        x_out = x_fused * F.gelu(gate)

        return self.out_proj(x_out)


##########################################################################
## 物理引导路由与适配层
##########################################################################


class PG_GLKM_Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        self.content_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_experts),
        )
        self.physics_branch = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor):
        logits = self.content_branch(x) + self.physics_branch(phys_emb)
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)
        return scores, top_k_indices, top_k_scores


class PG_GLKM_AdapterLayer(nn.Module):
    """
    PG-GLKM 适配层：
      - 使用 PG_GLKM_Expert 作为专家
      - 使用 PG_GLKM_Router 进行内容 + 物理先验联合路由
      - 与标准卷积块接口兼容，仅在特征通道维度上工作
    """

    def __init__(self, dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([PG_GLKM_Expert(dim) for _ in range(num_experts)])
        self.router = PG_GLKM_Router(dim, num_experts, k)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
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
## 主干网络：MoCE-IR-PG-GLKM（简化版 UNet + PG-GLKM-MoE）
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


class MoCEIR_PG_GLKM(nn.Module):
    """
    简化版 MoCE-IR：
      - 3 层 UNet 结构
      - 在 decoder 阶段引入 PG_GLKM_AdapterLayer 作为 PG-GLKM 消融实验
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
        self.down1 = nn.Conv2d(dim, dim * 2, 4, 2, 1)
        self.enc2 = ConvBlock(dim * 2, dim * 4)
        self.down2 = nn.Conv2d(dim * 4, dim * 8, 4, 2, 1)

        # Bottleneck
        self.bottleneck = ConvBlock(dim * 8, dim * 8)

        # Decoder
        self.up2 = nn.ConvTranspose2d(dim * 8, dim * 4, 4, 2, 1)
        self.dec2 = ConvBlock(dim * 8, dim * 4)
        self.pg_glkm2 = PG_GLKM_AdapterLayer(dim * 4, num_experts, topk)

        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1)
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.pg_glkm1 = PG_GLKM_AdapterLayer(dim * 2, num_experts, topk)

        self.out_conv = nn.Conv2d(dim * 2, out_channels, 3, 1, 1)

    @staticmethod
    def _get_phys_emb(x: torch.Tensor) -> torch.Tensor:
        # 使用全局平均池化获得 [B, C] 的物理感知嵌入
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

        # 保持 total_loss 接口（此消融模型不引入额外损失）
        self.total_loss = torch.tensor(
            0.0, device=out.device, dtype=out.dtype, requires_grad=False
        )
        return out


##########################################################################
## 构建函数：与现有框架保持一致
##########################################################################


def build_model(opt) -> nn.Module:
    """
    从全局配置构建 PG-GLKM 消融模型。
    仅使用 opt 中通用字段：dim, num_exp_blocks, topk 等，
    保证在现有脚本下即可插拔使用。
    """
    dim = getattr(opt, "dim", 32)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    topk = getattr(opt, "topk", 1)

    return MoCEIR_PG_GLKM(
        dim=dim,
        num_experts=num_experts,
        topk=topk,
    )


if __name__ == "__main__":
    # 简单自检：形状与复杂度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoCEIR_PG_GLKM(dim=32, num_experts=4, topk=1).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")

    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, x)
        gflops = flops.total() / 1e9
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"GFLOPs (128x128): {gflops:.4f}")
        print(f"Params:           {params_m:.4f}M")
    except Exception as e:
        print(f"Skip FLOPs/Params analysis due to: {e}")



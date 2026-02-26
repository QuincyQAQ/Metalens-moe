from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


##########################################################################
## 核心创新模块：物理引导门控选择性扫描专家 (PG-GSSE)
##########################################################################


class PG_GSSE_Expert(nn.Module):
    """
    物理引导门控选择性扫描专家 (PG-GSSE Expert)
    利用类 Mamba 线性复杂度实现全局依赖建模，并通过物理先验动态调制扫描算子。
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = int(self.expand * dim)

        # 输入投影：将输入映射到更高维空间，并显式保留残差分支
        self.in_proj = nn.Conv2d(dim, self.inner_dim * 2, kernel_size=1, bias=False)

        # 局部特征提取：深度卷积
        self.conv2d = nn.Conv2d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.inner_dim,
            bias=True,
        )

        # 选择性扫描参数生成（简化版 SSM）
        self.x_proj = nn.Linear(self.inner_dim, d_state * 2 + 1, bias=False)

        # 物理调制门控 (Physics-Modulated Gating)
        self.physics_gate = nn.Sequential(
            nn.Linear(dim, self.inner_dim),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C] (物理先验嵌入，与通道数 C 对齐)
        """
        b, c, h, w = x.shape

        # 1. 输入投影与分支
        x_and_res = self.in_proj(x)  # [B, 2*inner_dim, H, W]
        x_main, res = x_and_res.chunk(2, dim=1)

        # 2. 局部卷积与激活
        x_main = self.conv2d(x_main)[:, :, :h, :w]
        x_main = F.silu(x_main)

        # 3. 物理调制门控：利用物理先验动态调整残差分支的权重
        # phys_emb: [B, C] -> [B, inner_dim, 1, 1]
        p_gate = self.physics_gate(phys_emb).view(b, -1, 1, 1)
        res = res * p_gate

        # 4. 序列化 + 选择性扫描 (简化版 SSM)
        # 将 2D 特征展平为序列 [B, L, inner_dim]
        x_flat = rearrange(x_main, "b c h w -> b (h w) c")

        # 生成选择性参数
        x_proj = self.x_proj(x_flat)  # [B, L, 2*d_state + 1]
        dt, B_param, C_param = (
            x_proj[:, :, :1],
            x_proj[:, :, 1 : 1 + self.d_state],
            x_proj[:, :, 1 + self.d_state :],
        )

        # 使用门控线性单元 (GLU) 近似选择性扫描
        # 注意：这里只保留最关键的 dt 门控，B/C 仅作为占位符，方便后续扩展
        x_ssm = x_flat * torch.sigmoid(dt)

        # 5. 还原并融合
        x_out = rearrange(x_ssm, "b (h w) c -> b c h w", h=h, w=w)
        x_out = x_out * F.silu(res)

        return self.out_proj(x_out)


##########################################################################
## 物理引导的选择性路由 (Physics-Guided Selective Routing)
##########################################################################


class PGSSRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int, k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # 内容特征路由
        self.content_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_experts),
        )

        # 物理先验路由分支
        self.phys_router = nn.Linear(dim, num_experts)

    def forward(
        self, x: torch.Tensor, phys_emb: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C]
        Returns:
            scores: [B, num_experts] 软路由分数
            top_k_indices: [B, k] Top-k 专家索引
            top_k_scores: [B, k] Top-k 分数
        """
        content_logits = self.content_router(x)
        phys_logits = self.phys_router(phys_emb)

        logits = content_logits + phys_logits
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=-1)

        return scores, top_k_indices, top_k_scores


##########################################################################
## 集成到 MoCE-IR 框架的适配层
##########################################################################


class PGSSAdapterLayer(nn.Module):
    """
    简化版 PG-GSSE Mixture-of-Experts 适配层：
      - 使用 PG_GSSE_Expert 作为专家
      - 使用 PGSSRouter 进行基于内容 + 物理先验的路由
      - 仅在特征通道维度上工作，接口与传统卷积块兼容
    """

    def __init__(self, dim: int, num_experts: int = 4, k: int = 1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList(
            [PG_GSSE_Expert(dim) for _ in range(num_experts)]
        )
        self.router = PGSSRouter(dim, num_experts, k)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phys_emb: [B, C]
        """
        b, c, h, w = x.shape
        scores, indices, top_k_scores = self.router(x, phys_emb)

        # 初始化输出为 0
        final_out = torch.zeros_like(x)

        # 专家并行处理 (简化版 Dispatch/Combine)
        for i in range(self.router.num_experts):
            # 找到选择了该专家的样本索引
            mask = (indices == i).any(dim=-1)  # [B]
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                # 加权融合
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight

        return self.out_proj(final_out)


##########################################################################
## 简化版 MoCE-IR 主干：UNet 结构 + PG-GSSE 适配层
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
      - 在 decoder 阶段引入 PGSSAdapterLayer 作为 PG-GSSE 消融实验
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
        self.pgss2 = PGSSAdapterLayer(dim * 4, num_experts=num_experts, k=topk)

        self.up1 = Upsample(dim * 4, dim * 2)
        # 上采样后与 enc1 拼接，通道数为 dim*2 + dim = 3*dim
        self.dec1 = ConvBlock(dim * 3, dim * 2)
        self.pgss1 = PGSSAdapterLayer(dim * 2, num_experts=num_experts, k=topk)

        # 输出头
        self.out_conv = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _global_phys_embedding(feat: torch.Tensor) -> torch.Tensor:
        """
        从特征中自适应提取物理先验嵌入：
          - 使用全局平均池化获得 [B, C]
        """
        # feat: [B, C, H, W]
        return feat.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor, de_id: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        d_dec2 = self.pgss2(d_dec2, phys_emb2)

        # Decoder stage 1
        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        phys_emb1 = self._global_phys_embedding(d_dec1)
        d_dec1 = self.pgss1(d_dec1, phys_emb1)

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
    从全局配置构建 PG-GSSE 消融模型。

    仅使用 opt 中通用字段：dim, num_blocks, num_dec_blocks, heads 等不会严格依赖，
    以保证在现有脚本下即插即用。
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
    # 简单自检
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoCEIR(dim=32, num_experts=4, topk=1).to(device)
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")



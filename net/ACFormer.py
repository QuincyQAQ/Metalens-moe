import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops import rearrange


# Optional FlashAttention dependency.
# In Metalens-Transformer, MAFG_CA uses flash_attn for efficiency.
# Here we make it optional and fall back to standard scaled dot-product
# attention when flash_attn is not available.
try:  # pragma: no cover - optional dependency
    from flash_attn import flash_attn_func as _flash_attn_func
    _HAVE_FLASH_ATTN = True
except Exception:  # pragma: no cover - optional dependency
    _flash_attn_func = None
    _HAVE_FLASH_ATTN = False


def _flash_attn_or_fallback(Q: torch.Tensor,
                            K: torch.Tensor,
                            V: torch.Tensor) -> torch.Tensor:
    """Wrapper around flash_attn with a safe PyTorch fallback.

    Parameters
    ----------
    Q, K, V : (B, L_q/L_k, H, D)
        Batch of query / key / value sequences with num_heads H and head_dim D.

    Returns
    -------
    out : (B, L_q, H, D)
        Attended sequence.
    """
    if _HAVE_FLASH_ATTN and _flash_attn_func is not None:
        # flash_attn expects (batch, seqlen, nheads, headdim)
        return _flash_attn_func(Q, K, V, causal=False).to(torch.float32)

    # Fallback: regular scaled dot-product attention in PyTorch
    # Shapes: B, Lq, H, D  and  B, Lk, H, D
    B, Lq, H, D = Q.shape
    _, Lk, _, _ = K.shape

    # Merge heads for batched matmul: (B*H, Lq, D)
    q = Q.permute(0, 2, 1, 3).reshape(B * H, Lq, D)
    k = K.permute(0, 2, 1, 3).reshape(B * H, Lk, D)
    v = V.permute(0, 2, 1, 3).reshape(B * H, Lk, D)

    scale = 1.0 / math.sqrt(D)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn_logits, dim=-1)
    out = torch.matmul(attn, v)  # (B*H, Lq, D)

    # Restore to (B, Lq, H, D)
    out = out.reshape(B, H, Lq, D).permute(0, 2, 1, 3).contiguous()
    return out


def to_3d(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layernorm_type: str):
        super().__init__()
        if layernorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""

    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool, ksize: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.ksize = ksize
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if ksize:
            self.avg = nn.AvgPool2d(kernel_size=ksize, stride=1, padding=(ksize - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if self.ksize:
            q = q - self.avg(q)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv."""

    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _to(x: torch.Tensor) -> dict:
    return {"device": x.device, "dtype": x.dtype}


def expand_dim(t: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x: torch.Tensor) -> torch.Tensor:
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **_to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **_to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q: torch.Tensor, rel_k: torch.Tensor) -> torch.Tensor:
    # q is expected to be 4D: (b, h, w, d), matching the original
    # Metalens-Transformer implementation. We keep the same logic as
    # basicsr/models/archs/restormer_arch.py to ensure identical
    # behavior and tensor shapes.

    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size: int, rel_size: int, dim_head: int):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        block = self.block_size
        q = rearrange(q, "b (x y) c -> b x y c", x=block)

        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


class OCAB(nn.Module):
    """Overlapping Cross-Attention (OCA)."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        dim_head: int,
        bias: bool,
        ksize: int = 0,
    ) -> None:
        super().__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head ** -0.5
        self.ksize = ksize

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=window_size + (self.overlap_win_size - window_size),
            dim_head=self.dim_head,
        )
        if ksize:
            self.avg = nn.AvgPool2d(kernel_size=ksize, stride=1, padding=(ksize - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        if self.ksize:
            qs = qs - self.avg(qs)

        qs = rearrange(
            qs,
            "b c (h p1) (w p2) -> (b h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(
            lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.inner_dim),
            (ks, vs),
        )

        qs, ks, vs = map(
            lambda t: rearrange(t, "b n (head c) -> (b head) n c", head=self.num_spatial_heads),
            (qs, ks, vs),
        )

        qs = qs * self.scale
        spatial_attn = torch.matmul(qs, ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = torch.matmul(spatial_attn, vs)
        out = rearrange(
            out,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=h // self.window_size,
            w=w // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )
        out = self.project_out(out)
        return out


class AttentionFusion(nn.Module):
    def __init__(self, dim: int, bias: bool, channel_fusion: bool):
        super().__init__()
        self.channel_fusion = channel_fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=bias),
        )
        self.dim = dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fusion_map = self.fusion(x)
        if self.channel_fusion:
            weight = torch.sigmoid(torch.mean(fusion_map, 1, keepdim=True))
        else:
            weight = torch.sigmoid(torch.mean(fusion_map, (2, 3), keepdim=True))
        fused_feature = x[:, : self.dim] * weight + x[:, self.dim :] * (1 - weight)
        return fused_feature


class Transformer_STAF(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio: float,
        num_channel_heads: int,
        num_spatial_heads: int,
        spatial_dim_head: int,
        ffn_expansion_factor: float,
        bias: bool,
        layernorm_type: str,
        channel_fusion: bool,
        query_ksize: int = 0,
    ) -> None:
        super().__init__()
        self.spatial_attn = OCAB(
            dim,
            window_size,
            overlap_ratio,
            num_spatial_heads,
            spatial_dim_head,
            bias,
            ksize=query_ksize,
        )
        self.channel_attn = Attention(dim, num_channel_heads, bias, ksize=query_ksize)

        self.norm1 = LayerNorm(dim, layernorm_type)
        self.norm2 = LayerNorm(dim, layernorm_type)
        self.norm3 = LayerNorm(dim, layernorm_type)
        self.norm4 = LayerNorm(dim, layernorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.fusion = AttentionFusion(dim * 2, bias, channel_fusion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = x + self.spatial_attn(self.norm1(x))
        sa = sa + self.spatial_ffn(self.norm2(sa))
        ca = x + self.channel_attn(self.norm3(x))
        ca = ca + self.channel_ffn(self.norm4(ca))
        fused = self.fusion(torch.cat([sa, ca], 1))
        return fused


class MAFG_CA(nn.Module):
    """Metalens Angular-Frequency Guided Cross-Attention.

    This is a lightly cleaned version of the implementation used in
    Metalens-Transformer. It keeps the architecture and math the same, while
    making FlashAttention optional via a PyTorch fallback.
    """

    def __init__(self, embed_dim: int, num_heads: int, M: int, window_size: int = 0, eps: float = 1e-6):
        super().__init__()
        self.M = M
        self.Q_idx = M // 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.wsize = window_size

        self.proj_high = nn.Conv2d(3, embed_dim, kernel_size=1)
        self.proj_rgb = nn.Conv2d(embed_dim, 3, kernel_size=1)

        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj_out = nn.Linear(embed_dim, embed_dim, bias=False)

        self.max_seq = 2**16 - 1  # kept for API compatibility, not used in fallback

        # window based sliding similar to OCAB
        self.overlap_wsize = int(self.wsize * 0.5) + self.wsize
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_wsize, self.overlap_wsize),
            stride=window_size,
            padding=(self.overlap_wsize - self.wsize) // 2,
        )
        self.scale = self.embed_dim ** -0.5
        self.pos_emb_q = nn.Parameter(torch.zeros(self.wsize**2, embed_dim))
        self.pos_emb_k = nn.Parameter(torch.zeros(self.overlap_wsize**2, embed_dim))
        nn.init.trunc_normal_(self.pos_emb_q, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_k, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*M, 3, H, W)
        x = self.proj_high(x)
        BM, E, H, W = x.shape

        x_seq = x.view(BM, E, -1).permute(0, 2, 1)  # (BM, HW, E)
        x_seq = self.norm(x_seq)
        B = BM // self.M

        QKV = self.qkv(x_seq)
        QKV = QKV.view(BM, H, W, 3, -1).permute(3, 0, 4, 1, 2).contiguous()
        Q, K, V = QKV[0], QKV[1], QKV[2]  # each (BM, E, H, W)

        Q_bm = Q.view(B, self.M, E, H, W)
        _Q = Q_bm[:, self.Q_idx : self.Q_idx + 1]
        Q = torch.stack([__Q.repeat(self.M, 1, 1, 1) for __Q in _Q]).view(BM, E, H, W)

        Q = rearrange(
            Q,
            "b c (h p1) (w p2) -> (b h w) (p1 p2) c",
            p1=self.wsize,
            p2=self.wsize,
        )
        K, V = map(lambda t: self.unfold(t), (K, V))

        # After unfold: (BM, E * patch, num_patches)
        # We reshape to (BM * num_patches, patch, E)
        K, V = map(
            lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.embed_dim),
            (K, V),
        )

        # Absolute positional embedding
        Q = Q + self.pos_emb_q
        K = K + self.pos_emb_k

        s, eq, _ = Q.shape
        _, ek, _ = K.shape

        Q = Q.view(s, eq, self.num_heads, self.head_dim)
        K = K.view(s, ek, self.num_heads, self.head_dim)
        V = V.view(s, ek, self.num_heads, self.head_dim)

        # Use FlashAttention if available; otherwise standard attention
        out = _flash_attn_or_fallback(Q, K, V)

        out = rearrange(
            out,
            "(b nh nw) (ph pw) h d -> b (nh ph nw pw) (h d)",
            nh=H // self.wsize,
            nw=W // self.wsize,
            ph=self.wsize,
            pw=self.wsize,
        )
        out = self.proj_out(out)

        mixed_feature = out.view(BM, H, W, E).permute(0, 3, 1, 2).contiguous() + x
        return self.proj_rgb(mixed_feature).reshape(B, -1, H, W)


class ACFormer(nn.Module):
    """Aberration Correction Transformer for Metalens (clean port).

    Ported from Metalens-Transformer's `restormer_arch.ACFormer`, with:
    - No dependency on BasicSR internals.
    - Optional FlashAttention (falls back to regular attention otherwise).

    The forward interface is kept the same:
      * If input is 5D (B, M, C, H, W): multi-angle input, residual on center view.
      * If input is 4D (B, C, H, W): treated as a single image.
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        channel_heads = (1, 2, 4, 8),
        spatial_heads = (2, 2, 3, 4),
        overlap_ratio = (0.5, 0.5, 0.5, 0.5),
        window_size: int = 8,
        spatial_dim_head: int = 16,
        bias: bool = False,
        ffn_expansion_factor: float = 2.66,
        layernorm_type: str = "WithBias",
        M: int = 13,
        ca_heads: int = 2,
        ca_dim: int = 32,
        window_size_ca: int = 0,
        query_ksize=None,
    ) -> None:
        super().__init__()

        if query_ksize is None:
            # Default from Metalens-Transformer: [15, 11, 7, 3, 3]
            query_ksize = [15, 11, 7, 3, 3]

        self.center_idx = M // 2
        self.ca = MAFG_CA(embed_dim=ca_dim, num_heads=ca_heads, M=M, window_size=window_size_ca)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=dim,
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[0],
                    num_channel_heads=channel_heads[0],
                    num_spatial_heads=spatial_heads[0],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=False,
                    query_ksize=0,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[1],
                    num_channel_heads=channel_heads[1],
                    num_spatial_heads=spatial_heads[1],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=False,
                    query_ksize=0,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**2),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[2],
                    num_channel_heads=channel_heads[2],
                    num_spatial_heads=spatial_heads[2],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=False,
                    query_ksize=0,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**3),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[3],
                    num_channel_heads=channel_heads[3],
                    num_spatial_heads=spatial_heads[3],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=False,
                    query_ksize=query_ksize[0] if i % 2 == 1 else 0,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**2),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[2],
                    num_channel_heads=channel_heads[2],
                    num_spatial_heads=spatial_heads[2],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=True,
                    query_ksize=query_ksize[1] if i % 2 == 1 else 0,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[1],
                    num_channel_heads=channel_heads[1],
                    num_spatial_heads=spatial_heads[1],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=True,
                    query_ksize=query_ksize[2] if i % 2 == 1 else 0,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[0],
                    num_channel_heads=channel_heads[0],
                    num_spatial_heads=spatial_heads[0],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=True,
                    query_ksize=query_ksize[3] if i % 2 == 1 else 0,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                Transformer_STAF(
                    dim=int(dim * 2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[0],
                    num_channel_heads=channel_heads[0],
                    num_spatial_heads=spatial_heads[0],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layernorm_type=layernorm_type,
                    channel_fusion=True,
                    query_ksize=query_ksize[4] if i % 2 == 1 else 0,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inp_img : torch.Tensor
            Either (B, C, H, W) or (B, M, C, H, W). In the multi-view case,
            the residual connection uses the center view (index M//2).
        """
        if inp_img.ndim == 5:
            B, M, C, H, W = inp_img.shape
            center_img = inp_img[:, self.center_idx]
            inp_img = inp_img.view(B * M, C, H, W).contiguous()
        else:
            center_img = inp_img
            B, C, H, W = inp_img.shape
            M = 1

        if self.ca is None:
            inp_enc_level1 = inp_img.view(B, M * C, H, W)
        else:
            inp_enc_level1 = self.ca(inp_img)

        inp_enc_level1 = self.patch_embed(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + center_img
        return out_dec_level1



def build_model(opt):
    dim = getattr(opt, 'ac_dim', 48)
    num_blocks = getattr(opt, 'ac_num_blocks', [2, 4, 4, 4])
    num_refinement_blocks = getattr(opt, 'ac_num_refinement_blocks', 4)
    channel_heads = getattr(opt, 'ac_channel_heads', [1, 2, 4, 8])
    spatial_heads = getattr(opt, 'ac_spatial_heads', [1, 2, 4, 8])
    overlap_ratio = getattr(opt, 'ac_overlap_ratio', [0.5, 0.5, 0.5, 0.5])
    window_size = getattr(opt, 'ac_window_size', 8)
    spatial_dim_head = getattr(opt, 'ac_spatial_dim_head', 16)
    bias = getattr(opt, 'ac_bias', False)
    ffn_expansion_factor = getattr(opt, 'ac_ffn_expansion_factor', 2.66)
    layernorm_type = getattr(opt, 'ac_layernorm_type', 'WithBias')
    M = getattr(opt, 'ac_M', 13)
    ca_heads = getattr(opt, 'ac_ca_heads', 2)
    ca_dim = getattr(opt, 'ac_ca_dim', 32)
    window_size_ca = getattr(opt, 'ac_window_size_ca', 8)
    query_ksize = getattr(opt, 'ac_query_ksize', [15, 11, 7, 3, 3])

    net = ACFormer(
        inp_channels=3,
        out_channels=3,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        channel_heads=channel_heads,
        spatial_heads=spatial_heads,
        overlap_ratio=overlap_ratio,
        window_size=window_size,
        spatial_dim_head=spatial_dim_head,
        bias=bias,
        ffn_expansion_factor=ffn_expansion_factor,
        layernorm_type=layernorm_type,
        M=M,
        ca_heads=ca_heads,
        ca_dim=ca_dim,
        window_size_ca=window_size_ca,
        query_ksize=query_ksize,
    )

    if not getattr(opt, 'ac_use_ca', False):
        net.ca = None

    return net

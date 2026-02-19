from collections import OrderedDict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numbers
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
from fvcore.nn import FlopCountAnalysis, flop_count_table


##########################################################################
## Helper functions
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class MySequential(nn.Sequential):
    def forward(self, x1, x2):
        # Iterate through all layers in sequential order
        for layer in self:
            # Check if the layer takes two inputs (i.e., custom layers)
            if isinstance(layer, nn.Module):
                # Pass both inputs to the layer
                x1 = layer(x1, x2)
            else:
                # For non-module layers, pass the two inputs directly
                x1 = layer(x1, x2)
        return x1

def softmax_with_temperature(logits, temperature=1.0):
    """
    Apply softmax with temperature to the logits.
    
    Args:
    - logits (torch.Tensor): The input logits.
    - temperature (float): The temperature factor.
    
    Returns:
    - torch.Tensor: The softmax output with temperature.
    """
    # Scale the logits by the temperature
    scaled_logits = logits / temperature
    
    # Apply softmax
    return F.softmax(scaled_logits, dim=-1)

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # NOTE:
        # - 这里每步分配一个 [B,C,H,W] 的大零张量是热点；不需要 requires_grad=True（梯度会从 stitched 反传）
        # - 避免无条件 .float()，以便 AMP 下保持更快的 dtype（必要时会自动提升精度）
        zeros = torch.zeros(
            self._gates.size(0),
            stitched.size(1),
            stitched.size(2),
            stitched.size(3),
            device=stitched.device,
            dtype=stitched.dtype,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined
    
    def to_spatial(self, x, x_shape):
        h, w = x_shape
        amp, phase = x.chunk(2, dim=1)
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        x = real + 1j * imag
        x = torch.fft.ifft2(x, s=(h, w), norm="backward").real
        return x

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class HighPassConv2d(nn.Module):
    def __init__(self, c, freeze):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=c, 
            out_channels=c, 
            kernel_size=3, 
            padding=1, 
            bias=False, 
            groups=c
        )
        
        kernel = torch.tensor([[[[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]]]], dtype=torch.float32)
        self.conv.weight.data = kernel.repeat(c, 1, 1, 1)
        
        if freeze:
            self.conv.requires_grad_ = False
        
    def forward(self, x):
        return self.conv(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x 
    
##########################################################################
## Multi-DConv Head Transposed Self-Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, y):
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
##########################################################################
## Self-Attention in Fourier Domain
class FFTAttention(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super(FFTAttention, self).__init__()

        self.patch_size = kwargs["patch_size"]
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7, stride=1, padding=7//2, groups=dim*2)
        self.norm = LayerNorm(dim, "WithBias")
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)        
        
    def pad_and_rearrange(self, x):
        b, c, h, w = x.shape
        
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        x = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        return x
    
    def rearrange_to_original(self, x, x_shape):
        h, w = x_shape
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        x = x[:, :, :h, :w]  # Slice out the original height and width
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
            
        q = self.pad_and_rearrange(q)
        k = self.pad_and_rearrange(k)
            
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        
        out = self.rearrange_to_original(out, (h, w))
        
        out = self.norm(out)
        out = out * v
        
        out = self.proj_out(out)
        return out

    
     
##########################################################################
## Adapter Block    
class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size:int):
        super(ModExpert, self).__init__()
        
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
            
    def process(self, x, shared):
        shortcut = x
        x = self.proj[0](x)
        x = self.body(x) * F.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared):
        for _ in range(self.depth):
            feat = self.process(feats, shared)
        return feat
    
    def forward(self, x, shared):
        b, c, h, w = x.shape
        
        if b == 0:
            return x
        else:
            x = self.feat_extract(x, shared)
            return x


##########################################################################
## 创新点2: 多模态物理先验融合专家
## 物理先验编码器

class DepthEncoder(nn.Module):
    """
    深度图编码器：将深度图编码为与图像特征维度匹配的特征
    用于深度感知模糊校正专家
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
    
    def forward(self, depth_map): 
        # depth_map: (B, 1, H, W)
        return self.encoder(depth_map)


class SpectralEncoder(nn.Module):
    """
    光谱信息编码器：将光谱数据编码为特征
    用于光谱色差校正专家
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        # 假设 spectral_data 是 (B, C_spectral, H, W) 的特征图
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, spectral_data, H=None, W=None): 
        # spectral_data: (B, C_spectral, H, W) or (B, C_spectral)
        if spectral_data.dim() == 2:  # (B, C_spectral) -> (B, C_spectral, H, W)
            if H is None or W is None:
                raise ValueError("SpectralEncoder needs H, W for global spectral data.")
            spectral_data = spectral_data.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return self.encoder(spectral_data)


class OpticalParamEncoder(nn.Module):
    """
    光学参数编码器：将光学参数（如PSF参数）编码为特征
    用于PSF引导细节恢复专家
    """
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels * 2),
            nn.GELU(),
            nn.Linear(out_channels * 2, out_channels)
        )

    def forward(self, optical_params, H, W): 
        # optical_params: (B, C_params)
        encoded_params = self.encoder(optical_params)  # (B, out_channels)
        # 扩展为特征图 (B, out_channels, H, W)
        if H is None or W is None:
            raise ValueError("OpticalParamEncoder needs H, W for global optical parameters.")
        return encoded_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)


##########################################################################
## 融合机制

class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合：使用交叉注意力机制融合图像特征和物理先验特征
    参考：Attention Is All You Need (Vaswani et al., 2017)
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query_feat, key_value_feat): 
        # query_feat: image_feat, key_value_feat: physics_feat
        # 将特征图展平为序列 (B, H*W, C)
        B, C, H, W = query_feat.shape
        query_feat_flat = query_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        key_value_feat_flat = key_value_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Cross-attention
        attn_output, _ = self.attn(query_feat_flat, key_value_feat_flat, key_value_feat_flat)
        attn_output = self.proj(attn_output) + query_feat_flat  # Residual connection
        
        # 恢复为特征图 (B, C, H, W)
        fused_feat = attn_output.transpose(1, 2).reshape(B, C, H, W)
        return fused_feat


class GatedFusion(nn.Module):
    """
    门控融合：使用门控机制自适应融合图像特征和物理先验特征
    参考：Squeeze-and-Excitation Networks (Hu et al., 2018)
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.main_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

    def forward(self, image_feat, physics_feat):
        combined_feat = torch.cat([image_feat, physics_feat], dim=1)
        gate = self.sigmoid(self.gate_conv(combined_feat))
        fused_feat = gate * self.main_conv(combined_feat) + (1 - gate) * image_feat  # Gated fusion with residual
        return fused_feat


##########################################################################
## 物理先验融合专家

class HeteroExpert(nn.Module):
    """
    异构专家：支持物理先验融合的专家
    可以处理不同类型的物理先验（深度、光谱、光学参数）
    """
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int, 
                 expert_type: str = "standard", fusion_type: str = "gated"):
        super(HeteroExpert, self).__init__()
        
        self.depth = depth
        self.expert_type = expert_type
        self.fusion_type = fusion_type
        
        # 投影层
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        # 主体处理（使用传入的func，如FFTAttention）
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
        
        # 物理先验融合模块（仅当expert_type需要时创建）
        if expert_type == "depth_aware_deblurring":
            if fusion_type == "cross_attn":
                self.fusion_module = CrossAttentionFusion(dim)
            else:
                self.fusion_module = GatedFusion(dim)
        elif expert_type == "spectral_chromatic_aberration_correction":
            if fusion_type == "cross_attn":
                self.fusion_module = CrossAttentionFusion(dim)
            else:
                self.fusion_module = GatedFusion(dim)
        elif expert_type == "psf_guided_detail_restoration":
            if fusion_type == "cross_attn":
                self.fusion_module = CrossAttentionFusion(dim)
            else:
                self.fusion_module = GatedFusion(dim)
        else:
            self.fusion_module = None
            
    def process(self, x, shared, physics_priors=None):
        shortcut = x
        
        # 如果使用物理先验融合
        if self.fusion_module is not None and physics_priors is not None:
            if self.expert_type == "depth_aware_deblurring" and "depth_feat" in physics_priors:
                x = self.fusion_module(x, physics_priors["depth_feat"])
            elif self.expert_type == "spectral_chromatic_aberration_correction" and "spectral_feat" in physics_priors:
                x = self.fusion_module(x, physics_priors["spectral_feat"])
            elif self.expert_type == "psf_guided_detail_restoration" and "optical_param_feat" in physics_priors:
                x = self.fusion_module(x, physics_priors["optical_param_feat"])
        
        x = self.proj[0](x)
        x = self.body(x) * F.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared, physics_priors=None):
        for _ in range(self.depth):
            feat = self.process(feats, shared, physics_priors)
        return feat
    
    def forward(self, x, shared, physics_priors=None):
        b, c, h, w = x.shape
        
        if b == 0:
            return x
        else:
            x = self.feat_extract(x, shared, physics_priors)
            return x


##########################################################################
## 物理先验置信度预测器

class PhysicsConfidencePredictor(nn.Module):
    """
    物理先验置信度预测器：评估每个物理先验的可靠性
    用于自适应物理约束路由
    """
    def __init__(self, dim_physics_feat, num_physics_priors):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim_physics_feat * num_physics_priors, dim_physics_feat),
            nn.GELU(),
            nn.Linear(dim_physics_feat, num_physics_priors),
            nn.Sigmoid()  # 输出每个物理先验的置信度 [0, 1]
        )

    def forward(self, encoded_physics_priors_pooled): 
        # 拼接并池化后的物理先验特征 (B, dim_physics_feat * num_physics_priors)
        return self.predictor(encoded_physics_priors_pooled)


########################################################################### 
## Adapter Layer (支持物理先验融合)
class AdapterLayer(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int=2, expert_layer: nn.Module=FFTAttention, stage_depth: int=1,
                 depth_type: str="lin", rank_type: str="constant", freq_dim: int=128, 
                 with_complexity: bool=False, complexity_scale: str="min",
                 use_physics_prior: bool=False, expert_types: List[str]=None):
        super().__init__()            
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts
        self.use_physics_prior = use_physics_prior
        
        if expert_types is None:
            expert_types = ["standard"] * num_experts
        if len(expert_types) != num_experts:
            expert_types = expert_types[:num_experts] + ["standard"] * (num_experts - len(expert_types))

        patch_sizes = [2**(i+2) for i in range(num_experts)]
        kernel_sizes = [3+(2*i) for i in range(num_experts)]
        
        if depth_type == "lin":
            depths = [stage_depth+i for i in range(num_experts)]
        elif depth_type == "double":
            depths = [stage_depth+(2*i) for i in range(num_experts)]
        elif depth_type == "exp":
            depths = [2**(i) for i in range(num_experts)]
        elif depth_type == "fact":
            depths = [math.factorial(i+1) for i in range(num_experts)]
        elif isinstance(depth_type, int):
            depths = [depth_type for _ in range(num_experts)]
        elif depth_type == "constant":
            depths = [stage_depth for i in range(num_experts)]
        else:
            raise(NotImplementedError)
        
        if rank_type == "constant":
            ranks = [rank for _ in range(num_experts)]
        elif rank_type == "lin":
            ranks = [rank+i for i in range(num_experts)]
        elif rank_type == "double":
            ranks = [rank+(2*i) for i in range(num_experts)]
        elif rank_type == "exp":
            ranks = [rank**(i+1) for i in range(num_experts)]
        elif rank_type == "fact":
            ranks = [math.factorial(rank+i) for i in range(num_experts)]
        elif rank_type == "spread":
            ranks = [dim//(2**i) for i in range(num_experts)][::-1]
        else:
            raise(NotImplementedError)
        
        # 使用HeteroExpert（支持物理先验融合）或ModExpert
        if use_physics_prior:
            self.experts = nn.ModuleList([
                MySequential(*[HeteroExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, 
                                          kernel_size=kernel, expert_type=expert_type)])
                for idx, (depth, rank, patch, kernel, expert_type) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes, expert_types))
            ])
        else:
            self.experts = nn.ModuleList([
                MySequential(*[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
                for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
            ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale,
            use_physics_prior=use_physics_prior,
            min_k=1, max_k=top_k  # DEPKD-CR: 动态K值范围
        )
        
    def forward(self, x, freq_emb, shared, raw_physics_priors=None):
        gates, top_k_indices, top_k_values, aux_loss, encoded_physics_priors = self.routing(x, freq_emb, raw_physics_priors)
        self.loss = aux_loss
                
        # routing
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
            
            # 如果使用物理先验，需要分发物理先验特征
            if self.use_physics_prior and encoded_physics_priors is not None:
                expert_outputs = []
                for exp in range(len(self.experts)):
                    # 为每个专家准备对应的物理先验
                    physics_priors_exp = {}
                    if encoded_physics_priors is not None:
                        for key, feat in encoded_physics_priors.items():
                            if feat is not None:
                                feat_exp = dispatcher.dispatch(feat)[exp]
                                physics_priors_exp[key] = feat_exp
                    expert_outputs.append(self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp], physics_priors_exp))
            else:
                expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp]) for exp in range(len(self.experts))]
            
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            # 推理阶段：处理动态K值
            B = x.size(0)
            # 计算每个样本实际激活的专家数量
            k_per_sample = (gates > 0).sum(dim=1)  # (B,)
            max_k = k_per_sample.max().item()
            
            # 为每个样本收集专家输出
            all_expert_outputs = []
            for i in range(B):
                k_i = k_per_sample[i].item()
                if k_i == 0:
                    # 如果没有激活的专家，使用第一个专家
                    k_i = 1
                    selected_indices = [0]
                else:
                    # 获取当前样本激活的专家索引
                    selected_indices = top_k_indices[i, :k_i].cpu().tolist()
                
                # 为当前样本运行选中的专家
                sample_outputs = []
                for idx in selected_indices:
                    if self.use_physics_prior and encoded_physics_priors is not None:
                        sample_outputs.append(self.experts[idx](x[i:i+1], shared[i:i+1], encoded_physics_priors))
                    else:
                        sample_outputs.append(self.experts[idx](x[i:i+1], shared[i:i+1]))
                
                # 加权聚合（使用gates中的权重）
                sample_gates = gates[i, selected_indices]  # (k_i,)
                sample_gates = sample_gates / (sample_gates.sum() + 1e-9)  # 归一化
                sample_output = sum(w * out for w, out in zip(sample_gates, sample_outputs))
                all_expert_outputs.append(sample_output)
            
            out = torch.cat(all_expert_outputs, dim=0)
            
        out = self.proj_out(out)
        return out

    

##########################################################################
## 创新点3: 基于动态专家剪枝与知识蒸馏的轻量化协同路由 (DEPKD-CR)

class TeacherRouter(nn.Module):
    """
    教师路由器：功能强大、参数量较大的路由器，用于训练阶段指导学生路由器
    """
    def __init__(self, dim, freq_dim, num_experts, use_physics_prior=False, H=None, W=None, hidden_dim=256):
        super(TeacherRouter, self).__init__()
        self.use_physics_prior = use_physics_prior
        self.H = H
        self.W = W
        
        # 特征提取：从SVFE特征和频率嵌入中提取特征
        if use_physics_prior:
            self.depth_encoder = DepthEncoder(in_channels=1, out_channels=dim // 4)
            self.spectral_encoder = SpectralEncoder(in_channels=3, out_channels=dim // 4)
            self.optical_param_encoder = OpticalParamEncoder(in_channels=16, out_channels=dim // 4)
            self.num_physics_priors = 3
            self.physics_confidence_predictor = PhysicsConfidencePredictor(dim_physics_feat=dim // 4, num_physics_priors=self.num_physics_priors)
            # 输入维度：SVFE (dim) + 频率嵌入 (freq_dim) + 物理先验特征 (dim//4 * 3) + 置信度 (3)
            router_input_dim = dim + freq_dim + (dim // 4 * self.num_physics_priors) + self.num_physics_priors
        else:
            self.depth_encoder = None
            self.spectral_encoder = None
            self.optical_param_encoder = None
            self.num_physics_priors = 0
            self.physics_confidence_predictor = None
            router_input_dim = dim + freq_dim
        
        # 教师路由器：更深的MLP
        self.mlp = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x, freq_emb, raw_physics_priors=None):
        B, C, H, W = x.shape
        
        # 提取SVFE特征
        svfe_vector = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, dim)
        
        # 构建路由器输入
        router_input_list = [svfe_vector, freq_emb]
        
        if self.use_physics_prior and raw_physics_priors is not None:
            encoded_physics_priors_pooled = []
            if "depth_map" in raw_physics_priors and raw_physics_priors["depth_map"] is not None:
                depth_feat = self.depth_encoder(raw_physics_priors["depth_map"])
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(depth_feat, (1, 1)).squeeze(-1).squeeze(-1))
            if "spectral_data" in raw_physics_priors and raw_physics_priors["spectral_data"] is not None:
                spectral_feat = self.spectral_encoder(raw_physics_priors["spectral_data"], H=H, W=W)
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(spectral_feat, (1, 1)).squeeze(-1).squeeze(-1))
            if "optical_params" in raw_physics_priors and raw_physics_priors["optical_params"] is not None:
                optical_param_feat = self.optical_param_encoder(raw_physics_priors["optical_params"], H=H, W=W)
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(optical_param_feat, (1, 1)).squeeze(-1).squeeze(-1))
            
            if len(encoded_physics_priors_pooled) > 0:
                encoded_physics_priors_pooled_cat = torch.cat(encoded_physics_priors_pooled, dim=-1)
                physics_confidence = self.physics_confidence_predictor(encoded_physics_priors_pooled_cat)
                router_input_list.append(encoded_physics_priors_pooled_cat)
                router_input_list.append(physics_confidence)
            else:
                # 如果没有物理先验，添加零向量
                router_input_list.append(torch.zeros(B, 0, device=x.device))
                router_input_list.append(torch.ones(B, self.num_physics_priors, device=x.device))
        else:
            if self.use_physics_prior:
                router_input_list.append(torch.zeros(B, 0, device=x.device))
                router_input_list.append(torch.ones(B, self.num_physics_priors, device=x.device))
        
        combined_input = torch.cat(router_input_list, dim=1)
        teacher_logits = self.mlp(combined_input)
        return teacher_logits


class StudentRouter(nn.Module):
    """
    学生路由器：轻量级、参数量较小的路由器，用于推理阶段
    """
    def __init__(self, dim, freq_dim, num_experts, use_physics_prior=False, H=None, W=None, hidden_dim=64):
        super(StudentRouter, self).__init__()
        self.use_physics_prior = use_physics_prior
        self.H = H
        self.W = W
        
        # 特征提取：从SVFE特征和频率嵌入中提取特征
        if use_physics_prior:
            self.depth_encoder = DepthEncoder(in_channels=1, out_channels=dim // 4)
            self.spectral_encoder = SpectralEncoder(in_channels=3, out_channels=dim // 4)
            self.optical_param_encoder = OpticalParamEncoder(in_channels=16, out_channels=dim // 4)
            self.num_physics_priors = 3
            self.physics_confidence_predictor = PhysicsConfidencePredictor(dim_physics_feat=dim // 4, num_physics_priors=self.num_physics_priors)
            router_input_dim = dim + freq_dim + (dim // 4 * self.num_physics_priors) + self.num_physics_priors
        else:
            self.depth_encoder = None
            self.spectral_encoder = None
            self.optical_param_encoder = None
            self.num_physics_priors = 0
            self.physics_confidence_predictor = None
            router_input_dim = dim + freq_dim
        
        # 学生路由器：更浅的MLP
        self.mlp = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x, freq_emb, raw_physics_priors=None):
        B, C, H, W = x.shape
        
        # 提取SVFE特征
        svfe_vector = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, dim)
        
        # 构建路由器输入
        router_input_list = [svfe_vector, freq_emb]
        
        if self.use_physics_prior and raw_physics_priors is not None:
            encoded_physics_priors_pooled = []
            if "depth_map" in raw_physics_priors and raw_physics_priors["depth_map"] is not None:
                depth_feat = self.depth_encoder(raw_physics_priors["depth_map"])
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(depth_feat, (1, 1)).squeeze(-1).squeeze(-1))
            if "spectral_data" in raw_physics_priors and raw_physics_priors["spectral_data"] is not None:
                spectral_feat = self.spectral_encoder(raw_physics_priors["spectral_data"], H=H, W=W)
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(spectral_feat, (1, 1)).squeeze(-1).squeeze(-1))
            if "optical_params" in raw_physics_priors and raw_physics_priors["optical_params"] is not None:
                optical_param_feat = self.optical_param_encoder(raw_physics_priors["optical_params"], H=H, W=W)
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(optical_param_feat, (1, 1)).squeeze(-1).squeeze(-1))
            
            if len(encoded_physics_priors_pooled) > 0:
                encoded_physics_priors_pooled_cat = torch.cat(encoded_physics_priors_pooled, dim=-1)
                physics_confidence = self.physics_confidence_predictor(encoded_physics_priors_pooled_cat)
                router_input_list.append(encoded_physics_priors_pooled_cat)
                router_input_list.append(physics_confidence)
            else:
                router_input_list.append(torch.zeros(B, 0, device=x.device))
                router_input_list.append(torch.ones(B, self.num_physics_priors, device=x.device))
        else:
            if self.use_physics_prior:
                router_input_list.append(torch.zeros(B, 0, device=x.device))
                router_input_list.append(torch.ones(B, self.num_physics_priors, device=x.device))
        
        combined_input = torch.cat(router_input_list, dim=1)
        student_logits = self.mlp(combined_input)
        return student_logits


class DynamicPruningController(nn.Module):
    """
    动态剪枝控制器：根据输入特征动态预测需要激活的专家数量K
    """
    def __init__(self, dim, freq_dim, num_experts, use_physics_prior=False, H=None, W=None, 
                 min_k=1, max_k=None, hidden_dim=32):
        super(DynamicPruningController, self).__init__()
        self.num_experts = num_experts
        self.min_k = min_k
        self.max_k = max_k if max_k is not None else num_experts
        self.use_physics_prior = use_physics_prior
        self.H = H
        self.W = W
        
        # 特征提取
        if use_physics_prior:
            self.depth_encoder = DepthEncoder(in_channels=1, out_channels=dim // 4)
            self.spectral_encoder = SpectralEncoder(in_channels=3, out_channels=dim // 4)
            self.optical_param_encoder = OpticalParamEncoder(in_channels=16, out_channels=dim // 4)
            self.num_physics_priors = 3
            self.physics_confidence_predictor = PhysicsConfidencePredictor(dim_physics_feat=dim // 4, num_physics_priors=self.num_physics_priors)
            k_input_dim = dim + freq_dim + (dim // 4 * self.num_physics_priors) + self.num_physics_priors
        else:
            self.depth_encoder = None
            self.spectral_encoder = None
            self.optical_param_encoder = None
            self.num_physics_priors = 0
            self.physics_confidence_predictor = None
            k_input_dim = dim + freq_dim
        
        # K值预测器
        self.k_predictor = nn.Sequential(
            nn.Linear(k_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, x, freq_emb, raw_physics_priors=None):
        B, C, H, W = x.shape
        
        # 提取SVFE特征
        svfe_vector = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B, dim)
        
        # 构建输入
        k_input_list = [svfe_vector, freq_emb]
        
        if self.use_physics_prior and raw_physics_priors is not None:
            encoded_physics_priors_pooled = []
            if "depth_map" in raw_physics_priors and raw_physics_priors["depth_map"] is not None:
                depth_feat = self.depth_encoder(raw_physics_priors["depth_map"])
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(depth_feat, (1, 1)).squeeze(-1).squeeze(-1))
            if "spectral_data" in raw_physics_priors and raw_physics_priors["spectral_data"] is not None:
                spectral_feat = self.spectral_encoder(raw_physics_priors["spectral_data"], H=H, W=W)
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(spectral_feat, (1, 1)).squeeze(-1).squeeze(-1))
            if "optical_params" in raw_physics_priors and raw_physics_priors["optical_params"] is not None:
                optical_param_feat = self.optical_param_encoder(raw_physics_priors["optical_params"], H=H, W=W)
                encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(optical_param_feat, (1, 1)).squeeze(-1).squeeze(-1))
            
            if len(encoded_physics_priors_pooled) > 0:
                encoded_physics_priors_pooled_cat = torch.cat(encoded_physics_priors_pooled, dim=-1)
                physics_confidence = self.physics_confidence_predictor(encoded_physics_priors_pooled_cat)
                k_input_list.append(encoded_physics_priors_pooled_cat)
                k_input_list.append(physics_confidence)
            else:
                k_input_list.append(torch.zeros(B, 0, device=x.device))
                k_input_list.append(torch.ones(B, self.num_physics_priors, device=x.device))
        else:
            if self.use_physics_prior:
                k_input_list.append(torch.zeros(B, 0, device=x.device))
                k_input_list.append(torch.ones(B, self.num_physics_priors, device=x.device))
        
        combined_input = torch.cat(k_input_list, dim=1)
        
        # 预测K值：输出 [0, 1]，然后缩放到 [min_k, max_k]
        predicted_k_norm = self.k_predictor(combined_input).squeeze(1)  # (B,)
        k_dynamic = torch.round(self.min_k + predicted_k_norm * (self.max_k - self.min_k)).long()
        k_dynamic = torch.clamp(k_dynamic, self.min_k, self.max_k)
        
        return k_dynamic


class RoutingFunction(nn.Module):
    """
    DEPKD-CR路由器：集成教师路由器、学生路由器和动态剪枝控制器
    """
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max",
                 use_physics_prior: bool=False, H=None, W=None, min_k=1, max_k=None, 
                 distillation_temp=2.0, lambda_distillation=0.5):
        super(RoutingFunction, self).__init__()
        
        self.use_physics_prior = use_physics_prior
        self.H = H
        self.W = W
        self.num_experts = num_experts
        self.k = k  # 保留作为默认值
        self.distillation_temp = distillation_temp
        self.lambda_distillation = lambda_distillation
        
        # DEPKD-CR核心组件
        self.teacher_router = TeacherRouter(dim, freq_dim, num_experts, use_physics_prior, H, W)
        self.student_router = StudentRouter(dim, freq_dim, num_experts, use_physics_prior, H, W)
        self.pruning_controller = DynamicPruningController(dim, freq_dim, num_experts, use_physics_prior, H, W, 
                                                           min_k=min_k, max_k=max_k if max_k is not None else num_experts)
        
        # 用于编码物理先验（与TeacherRouter和StudentRouter共享）
        if use_physics_prior:
            self.depth_encoder = DepthEncoder(in_channels=1, out_channels=dim // 4)
            self.spectral_encoder = SpectralEncoder(in_channels=3, out_channels=dim // 4)
            self.optical_param_encoder = OpticalParamEncoder(in_channels=16, out_channels=dim // 4)
            self.num_physics_priors = 3
            self.physics_confidence_predictor = PhysicsConfidencePredictor(dim_physics_feat=dim // 4, num_physics_priors=self.num_physics_priors)
        else:
            self.depth_encoder = None
            self.spectral_encoder = None
            self.optical_param_encoder = None
            self.num_physics_priors = 0
            self.physics_confidence_predictor = None
        
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, freq_emb, raw_physics_priors=None):
        B, C, H, W = x.shape
        
        # 编码物理先验（如果需要）
        encoded_physics_priors = None
        if self.use_physics_prior and raw_physics_priors is not None:
            encoded_physics_priors = {}
            if "depth_map" in raw_physics_priors and raw_physics_priors["depth_map"] is not None:
                encoded_physics_priors["depth_feat"] = self.depth_encoder(raw_physics_priors["depth_map"])
            else:
                encoded_physics_priors["depth_feat"] = None
            if "spectral_data" in raw_physics_priors and raw_physics_priors["spectral_data"] is not None:
                encoded_physics_priors["spectral_feat"] = self.spectral_encoder(raw_physics_priors["spectral_data"], H=H, W=W)
            else:
                encoded_physics_priors["spectral_feat"] = None
            if "optical_params" in raw_physics_priors and raw_physics_priors["optical_params"] is not None:
                encoded_physics_priors["optical_param_feat"] = self.optical_param_encoder(raw_physics_priors["optical_params"], H=H, W=W)
            else:
                encoded_physics_priors["optical_param_feat"] = None
        
        # 1. 学生路由器（用于推理和训练）
        student_logits = self.student_router(x, freq_emb, raw_physics_priors)
        
        # 2. 动态剪枝控制器：预测动态K值
        k_dynamic = self.pruning_controller(x, freq_emb, raw_physics_priors)  # (B,)
        
        # 3. 为每个样本选择Top-K_dynamic专家
        gates = torch.zeros(B, self.num_experts, device=x.device, dtype=student_logits.dtype)
        top_k_indices_list = []
        top_k_values_list = []
        
        for i in range(B):
            current_k = k_dynamic[i].item()
            # 获取当前样本的Top-K专家
            top_k_values, top_k_indices = torch.topk(student_logits[i], current_k, dim=-1)
            # 计算协作权重（仅对选中的专家进行softmax）
            selected_weights = F.softmax(top_k_values, dim=-1)
            # 确保dtype匹配（处理FP16/FP32混合精度）
            selected_weights = selected_weights.to(dtype=gates.dtype)
            gates[i, top_k_indices] = selected_weights
            top_k_indices_list.append(top_k_indices)
            top_k_values_list.append(top_k_values)
        
        # 转换为张量（处理不同K值的情况）
        max_k = k_dynamic.max().item()
        top_k_indices_padded = torch.zeros(B, max_k, dtype=torch.long, device=x.device)
        top_k_values_padded = torch.zeros(B, max_k, dtype=student_logits.dtype, device=x.device)
        for i in range(B):
            k_i = k_dynamic[i].item()
            top_k_indices_padded[i, :k_i] = top_k_indices_list[i]
            top_k_values_padded[i, :k_i] = top_k_values_list[i]
        
        # 4. 计算辅助损失（包括知识蒸馏损失）
        aux_loss = 0
        distillation_loss = 0
        
        if self.training:
            # 教师路由器（仅在训练时使用）
            teacher_logits = self.teacher_router(x, freq_emb, raw_physics_priors)
            
            # 知识蒸馏损失
            soft_teacher_probs = F.softmax(teacher_logits / self.distillation_temp, dim=-1)
            soft_student_log_probs = F.log_softmax(student_logits / self.distillation_temp, dim=-1)
            distillation_loss = F.kl_div(soft_student_log_probs, soft_teacher_probs, reduction='batchmean') * (self.distillation_temp ** 2)
            
            # 负载均衡损失（基于学生路由器的输出）
            mean_expert_prob = F.softmax(student_logits, dim=-1).mean(dim=0)
            load_balance_loss = -torch.sum(mean_expert_prob * torch.log(mean_expert_prob + 1e-9))
            
            # 重要性损失
            importance = gates.sum(dim=0)
            if self.use_complexity_bias:
                importance = importance * self.complexity
            imp_mean = importance.mean()
            imp_std = importance.std()
            importance_loss = (imp_std / (imp_mean + 1e-8)) ** 2
            
            aux_loss = 0.3 * importance_loss + 0.2 * load_balance_loss + self.lambda_distillation * distillation_loss
        
        # 返回格式与原始路由器兼容
        # 注意：top_k_indices和top_k_values的形状可能不一致（因为动态K），但AdapterLayer会处理
        return gates, top_k_indices_padded, top_k_values_padded, aux_loss, encoded_physics_priors




##########################################################################
## Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        
        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type)
        ])
        
        self.mixer = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.mixer(self.norms[0](x))
        x = x + self.ffn(self.norms[1](x))
        return x
        


##########################################################################
## Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None, freq_dim:int=128, 
                 with_complexity: bool=False, use_physics_prior: bool=False, expert_types: List[str]=None):
        super().__init__()

        self.norms = nn.ModuleList([
          LayerNorm(dim, LayerNorm_type),
          LayerNorm(dim, LayerNorm_type),
        ])
        
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        ])
        
        self.shared = Attention(dim, num_heads, bias)
        self.mixer = CrossAttention(dim, num_heads=num_heads, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        
        self.adapter = AdapterLayer(
            dim, rank, 
            top_k=top_k, num_experts=num_experts, expert_layer=expert_layer, freq_dim=freq_dim,
            depth_type=depth_type, rank_type=rank_type, stage_depth=stage_depth, 
            with_complexity=with_complexity, complexity_scale=complexity_scale,
            use_physics_prior=use_physics_prior, expert_types=expert_types
        )
        
    def forward(self, x, freq_emb=None, raw_physics_priors=None):    
        shortcut = x
        x = self.norms[0](x)
        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter(x_a, freq_emb, x_s, raw_physics_priors)
        x = self.mixer(x_a, x_s) + shortcut

        x = x + self.ffn(self.norms[1](x))
        return x, self.adapter.loss

    

######################################################################
## Encoder Residual Group
class EncoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type)
            )

    def forward(self, x):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x)
            i += 1
        return x    
    
    
    
######################################################################
## Decoder Residual Group
class DecoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool, complexity_scale=None,
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None, rank_type=None, freq_dim:int=128, 
                 with_complexity: bool=False, use_physics_prior: bool=False, expert_types: List[str]=None):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    dim, num_heads, ffn_expansion, bias, LayerNorm_type, 
                    expert_layer=expert_layer, rank=rank, num_experts=num_experts, top_k=top_k, 
                    stage_depth=stage_depth, freq_dim=freq_dim, complexity_scale=complexity_scale,
                    depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity,
                    use_physics_prior=use_physics_prior, expert_types=expert_types
                )
            )

    def forward(self, x, freq_emb=None, raw_physics_priors=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x , loss = self.layers[i](x, freq_emb, raw_physics_priors)
            self.loss += loss
            i += 1
        return x  
    
    
     
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
    
    
##########################################################################
## Spatially-Variant Frequency Embedding
class SpatiallyVariantFreqEmbedding(nn.Module):
    """
    增强版频率嵌入：通过多尺度局部池化感知空间变异的频率特征
    """
    def __init__(self, dim):
        super(SpatiallyVariantFreqEmbedding, self).__init__()
        # 1. 物理感知分支：高通滤波提取高频细节（像差最敏感的部分）
        self.high_conv = nn.Sequential(
            HighPassConv2d(dim, freeze=True),
            nn.GELU()
        )
        
        # 2. 空间变异感知：使用不同尺度的池化来捕捉局部频率分布
        # 相比 GAP，这能保留"中心 vs 边缘"的特征差异
        self.local_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),  # 全局信息
            nn.AdaptiveAvgPool2d(2),  # 2x2 区域信息（区分四个象限）
            nn.AdaptiveAvgPool2d(4)   # 4x4 细粒度区域信息
        ])
        
        # 3. 特征融合 MLP
        # 1*1 + 2*2 + 4*4 = 21 个空间位置
        self.fusion = nn.Sequential(
            nn.Linear(dim * 21, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x: 来自 bottleneck 的特征 (B, C, H, W)
        x = self.high_conv(x)
        
        # 提取多尺度局部特征
        features = []
        for pool in self.local_pools:
            f = pool(x) # (B, C, h_p, w_p)
            f = f.flatten(1) # (B, C * h_p * w_p)
            features.append(f)
        
        # 拼接并融合
        combined = torch.cat(features, dim=1)
        out = self.fusion(combined)
        return out # 输出 (B, dim) 的频率嵌入
    
    
##########################################################################
##
class MoCEIR(nn.Module):
    def __init__(self,
                inp_channels=3, 
                out_channels=3, 
                dim = 32,
                levels: int = 4,
                heads = [1,1,1,1],
                num_blocks = [1,1,1,3],
                num_dec_blocks = [1, 1, 1],
                ffn_expansion_factor = 2,
                num_refinement_blocks = 1,
                LayerNorm_type = 'WithBias', ## Other option 'BiasFree'
                bias = False,
                rank=2,
                num_experts=4,
                depth_type="lin",
                stage_depth=[3,2,1],
                rank_type="constant",
                topk=1,
                expert_layer=FFTAttention,
                with_complexity=False,
                complexity_scale="max",
                use_physics_prior=False,
                expert_types=None,
                ):
        super(MoCEIR, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        self.use_physics_prior = use_physics_prior
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.freq_embed = SpatiallyVariantFreqEmbedding(dims[-1])
                
        # -- Encoder --        
        self.enc = nn.ModuleList([])
        for i in range(levels-1):
            self.enc.append(nn.ModuleList([
                EncoderResidualGroup(
                    dim=dims[i], 
                    num_blocks=num_blocks[i], 
                    num_heads=heads[i],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=True,),
                Downsample(dim*2**i)
                ])
            )
        
        # -- Latent --
        self.latent = EncoderResidualGroup(
            dim=dims[-1],
            num_blocks=num_blocks[-1], 
            num_heads=heads[-1], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
                  
        # -- Decoder --
        dims = dims[::-1]
        ranks = ranks[::-1]
        heads = heads[::-1]
        num_dec_blocks = num_dec_blocks[::-1]
        
        # 为每个decoder stage设置expert_types
        if expert_types is None:
            if use_physics_prior:
                # 为不同stage设置不同的专家类型
                expert_types_list = [
                    ["depth_aware_deblurring", "spectral_chromatic_aberration_correction", "psf_guided_detail_restoration", "standard"],
                ] * (levels - 1)
            else:
                expert_types_list = [None] * (levels - 1)
        else:
            expert_types_list = expert_types if isinstance(expert_types[0], list) else [expert_types] * (levels - 1)
        
        self.dec = nn.ModuleList([])
        for i in range(levels-1):
            self.dec.append(nn.ModuleList([
                Upsample(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=1, bias=bias),
                DecoderResidualGroup(
                    dim=dims[i+1],
                    num_blocks=num_dec_blocks[i], 
                    num_heads=heads[i+1],
                    ffn_expansion=ffn_expansion_factor, 
                    LayerNorm_type=LayerNorm_type, bias=bias, expert_layer=expert_layer, freq_dim=dims[0], with_complexity=with_complexity,
                    rank=ranks[i], num_experts=num_experts, stage_depth=stage_depth[i], depth_type=depth_type, rank_type=rank_type, top_k=topk, complexity_scale=complexity_scale,
                    use_physics_prior=use_physics_prior, expert_types=expert_types_list[i] if use_physics_prior else None),
                ])
            )

        # -- Refinement --
        heads = heads[::-1]
        self.refinement = EncoderResidualGroup(
            dim=dim,
            num_blocks=num_refinement_blocks, 
            num_heads=heads[0], 
            ffn_expansion=ffn_expansion_factor,
            LayerNorm_type=LayerNorm_type, bias=True,)
        
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.total_loss = None
        self.last_freq_emb = None
    
    def forward(self, x, labels=None, raw_physics_priors=None):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            labels: 标签（可选）
            raw_physics_priors: 物理先验字典，包含：
                - "depth_map": (B, 1, H, W) 深度图
                - "spectral_data": (B, C_spectral, H, W) 或 (B, C_spectral) 光谱数据
                - "optical_params": (B, C_params) 光学参数
        """
        B, C, H, W = x.shape
                
        feats = self.patch_embed(x)
        
        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)
        
        feats = self.latent(feats)
        freq_emb = self.freq_embed(feats)
        self.last_freq_emb = freq_emb
        
        # 计算每个decoder stage的空间尺寸
        current_H, current_W = H // (2 ** (self.levels - 1)), W // (2 ** (self.levels - 1))
        
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            
            # 更新当前stage的空间尺寸
            current_H, current_W = current_H * 2, current_W * 2
            
            # 如果使用物理先验，需要调整物理先验的尺寸以匹配当前stage
            adjusted_physics_priors = None
            if self.use_physics_prior and raw_physics_priors is not None:
                adjusted_physics_priors = {}
                if "depth_map" in raw_physics_priors and raw_physics_priors["depth_map"] is not None:
                    adjusted_physics_priors["depth_map"] = F.interpolate(
                        raw_physics_priors["depth_map"], 
                        size=(current_H, current_W), 
                        mode='bilinear', 
                        align_corners=False
                    )
                if "spectral_data" in raw_physics_priors and raw_physics_priors["spectral_data"] is not None:
                    if raw_physics_priors["spectral_data"].dim() == 4:  # (B, C, H, W)
                        adjusted_physics_priors["spectral_data"] = F.interpolate(
                            raw_physics_priors["spectral_data"], 
                            size=(current_H, current_W), 
                            mode='bilinear', 
                            align_corners=False
                        )
                    else:  # (B, C) - 全局向量，不需要调整
                        adjusted_physics_priors["spectral_data"] = raw_physics_priors["spectral_data"]
                if "optical_params" in raw_physics_priors and raw_physics_priors["optical_params"] is not None:
                    # 光学参数是全局向量，不需要调整尺寸
                    adjusted_physics_priors["optical_params"] = raw_physics_priors["optical_params"]
            
            feats = block(feats, freq_emb, adjusted_physics_priors)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x
    
                    
    

if __name__ == "__main__":
    # test
    model = MoCEIR(rank=2, num_blocks=[4,6,6,8], num_dec_blocks=[2,4,4], levels=4, dim=48, num_refinement_blocks=4, 
                   with_complexity=True, complexity_scale="max", stage_depth=[1,1,1], depth_type="constant", rank_type="spread", 
                   num_experts=4, topk=1, expert_layer=FFTAttention, use_physics_prior=True).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    
    # 测试物理先验输入
    depth_map = torch.randn(1, 1, 224, 224).cuda()
    spectral_data = torch.randn(1, 3, 224, 224).cuda()
    optical_params = torch.randn(1, 16).cuda()
    raw_physics_priors = {
        "depth_map": depth_map,
        "spectral_data": spectral_data,
        "optical_params": optical_params
    }
    
    _ = model(x, raw_physics_priors=raw_physics_priors)
    print(model.total_loss)
    # Memory usage  
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
  
    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x,))
    print(flop_count_table(flops))


def build_model(opt):
    dim = getattr(opt, 'dim', 32)
    num_blocks = getattr(opt, 'num_blocks', [4, 6, 6, 8])
    num_dec_blocks = getattr(opt, 'num_dec_blocks', [2, 4, 4])
    heads = getattr(opt, 'heads', [1, 2, 4, 8])
    num_refinement_blocks = getattr(opt, 'num_refinement_blocks', 4)
    topk = getattr(opt, 'topk', 1)
    num_experts = getattr(opt, 'num_exp_blocks', 4)
    rank = getattr(opt, 'latent_dim', 2)
    with_complexity = getattr(opt, 'with_complexity', False)
    depth_type = getattr(opt, 'depth_type', 'constant')
    stage_depth = getattr(opt, 'stage_depth', [1, 1, 1])
    rank_type = getattr(opt, 'rank_type', 'spread')
    complexity_scale = getattr(opt, 'complexity_scale', 'max')
    use_physics_prior = getattr(opt, 'use_physics_prior', False)
    expert_types = getattr(opt, 'expert_types', None)

    return MoCEIR(
        dim=dim,
        num_blocks=num_blocks,
        num_dec_blocks=num_dec_blocks,
        levels=len(num_blocks),
        heads=heads,
        num_refinement_blocks=num_refinement_blocks,
        topk=topk,
        num_experts=num_experts,
        rank=rank,
        with_complexity=with_complexity,
        depth_type=depth_type,
        stage_depth=stage_depth,
        rank_type=rank_type,
        complexity_scale=complexity_scale,
        use_physics_prior=use_physics_prior,
        expert_types=expert_types,
    )
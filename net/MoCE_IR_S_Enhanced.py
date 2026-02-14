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
## 新增模块1: FrequencyPrior (FFT+Sobel) - 频域先验提取
class FrequencyPrior(nn.Module):
    """提取频域先验特征 (FFT + Sobel边缘检测)"""
    def __init__(self, in_channels=3, out_channels=32):
        super(FrequencyPrior, self).__init__()
        # FFT特征提取
        self.fft_conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        
        # Sobel边缘检测
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(in_channels, 1, 1, 1))
        
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # FFT特征 - 提取频域信息
        fft_feat = torch.fft.rfft2(x, norm='backward')  # [B, C, H, W//2+1]
        fft_mag = torch.abs(fft_feat)  # [B, C, H, W//2+1]
        fft_phase = torch.angle(fft_feat)  # [B, C, H, W//2+1]
        
        # 将频域特征池化/插值回原始空间尺寸，以便与边缘特征融合
        fft_mag_spatial = F.interpolate(fft_mag, size=(h, w), mode='bilinear', align_corners=False)  # [B, C, H, W]
        fft_phase_spatial = F.interpolate(fft_phase, size=(h, w), mode='bilinear', align_corners=False)  # [B, C, H, W]
        fft_combined = torch.cat([fft_mag_spatial, fft_phase_spatial], dim=1)  # [B, 2*C, H, W]
        fft_out = self.fft_conv(fft_combined)  # [B, out_channels, H, W]
        
        # Sobel边缘特征
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=c)  # [B, C, H, W]
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=c)  # [B, C, H, W]
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)  # [B, C, H, W]
        edge_out = self.edge_conv(edge)  # [B, out_channels, H, W]
        
        # 融合 - 现在两个特征图尺寸匹配 [B, out_channels, H, W]
        out = self.fusion(torch.cat([fft_out, edge_out], dim=1))
        return out

     
##########################################################################
## 新增模块2: Wavelet分解 - 小波分解送入不同专家
class WaveletDecomposition(nn.Module):
    """Haar小波分解，将特征分解为LL, LH, HL, HH四个频带"""
    def __init__(self):
        super(WaveletDecomposition, self).__init__()
        # Haar小波滤波器
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        
        self.register_buffer('ll_kernel', ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_kernel', lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_kernel', hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_kernel', hh.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: [LL, LH, HL, HH] 每个都是 [B, C, H//2, W//2]
        """
        b, c, h, w = x.shape
        
        # 确保尺寸是偶数
        if h % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
        if w % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
        
        # 对每个通道分别做小波分解
        ll_list, lh_list, hl_list, hh_list = [], [], [], []
        for i in range(c):
            x_ch = x[:, i:i+1, :, :]
            ll = F.conv2d(x_ch, self.ll_kernel, stride=2, padding=0)
            lh = F.conv2d(x_ch, self.lh_kernel, stride=2, padding=0)
            hl = F.conv2d(x_ch, self.hl_kernel, stride=2, padding=0)
            hh = F.conv2d(x_ch, self.hh_kernel, stride=2, padding=0)
            ll_list.append(ll)
            lh_list.append(lh)
            hl_list.append(hl)
            hh_list.append(hh)
        
        ll = torch.cat(ll_list, dim=1)
        lh = torch.cat(lh_list, dim=1)
        hl = torch.cat(hl_list, dim=1)
        hh = torch.cat(hh_list, dim=1)
        
        return [ll, lh, hl, hh]
    
    def inverse(self, coeffs):
        """小波逆变换，重建图像"""
        ll, lh, hl, hh = coeffs
        b, c, h, w = ll.shape
        
        # 对每个通道分别重建
        recon_list = []
        for i in range(c):
            ll_ch = ll[:, i:i+1, :, :]
            lh_ch = lh[:, i:i+1, :, :]
            hl_ch = hl[:, i:i+1, :, :]
            hh_ch = hh[:, i:i+1, :, :]
            
            # 上采样并应用逆滤波器
            ll_up = F.conv_transpose2d(ll_ch, self.ll_kernel, stride=2, padding=0)
            lh_up = F.conv_transpose2d(lh_ch, self.lh_kernel, stride=2, padding=0)
            hl_up = F.conv_transpose2d(hl_ch, self.hl_kernel, stride=2, padding=0)
            hh_up = F.conv_transpose2d(hh_ch, self.hh_kernel, stride=2, padding=0)
            
            recon = ll_up + lh_up + hl_up + hh_up
            recon_list.append(recon)
        
        recon = torch.cat(recon_list, dim=1)
        return recon


##########################################################################
## 新增模块3: PSF-Aware Prompt调制
class PSFAwarePrompt(nn.Module):
    """PSF感知的提示调制，用于调制专家特征"""
    def __init__(self, dim, prompt_dim=64):
        super(PSFAwarePrompt, self).__init__()
        # 从频域嵌入生成PSF提示
        self.prompt_gen = nn.Sequential(
            nn.Linear(dim, prompt_dim * 2),
            nn.GELU(),
            nn.Linear(prompt_dim * 2, prompt_dim)
        )
        
        # 调制网络
        self.modulation = nn.Sequential(
            nn.Conv2d(dim + prompt_dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, freq_emb):
        """
        x: [B, C, H, W] 专家特征
        freq_emb: [B, freq_dim] 频域嵌入
        """
        b, c, h, w = x.shape
        
        # 生成PSF提示
        prompt = self.prompt_gen(freq_emb)  # [B, prompt_dim]
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)  # [B, prompt_dim, H, W]
        
        # 拼接并调制
        x_cat = torch.cat([x, prompt], dim=1)
        mod_weight = self.modulation(x_cat)
        
        return x * mod_weight


##########################################################################
## 新增模块4: Mamba融合 - 简化的状态空间模型融合专家输出
class MambaFusion(nn.Module):
    """使用简化的状态空间模型融合多个专家输出"""
    def __init__(self, dim, num_experts=4, d_state=16):
        super(MambaFusion, self).__init__()
        self.num_experts = num_experts
        self.dim = dim
        
        # 确保 d_state 能被 dim 整除（用于深度可分离卷积）
        # 如果 d_state < dim，则使用 d_state = dim
        # 否则调整为 dim 的倍数
        if d_state < dim:
            d_state = dim
        elif d_state % dim != 0:
            # 向上取整到最近的 dim 的倍数
            d_state = ((d_state + dim - 1) // dim) * dim
        
        self.d_state = d_state
        
        # 简化的状态空间：使用深度可分离卷积模拟长距离依赖
        # 使用 groups=1 避免整除问题，或者使用 depthwise + pointwise 结构
        self.state_conv = nn.ModuleList([
            nn.Sequential(
                # 深度可分离卷积：先depthwise，再pointwise扩展
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # depthwise
                nn.GELU(),
                nn.Conv2d(dim, d_state, kernel_size=1),  # pointwise扩展
                nn.GELU(),
                nn.Conv2d(d_state, dim, kernel_size=1)  # pointwise压缩回dim
            ) for _ in range(num_experts)
        ])
        
        # 跨专家交互（模拟状态空间的长距离依赖）
        self.cross_expert = nn.Sequential(
            nn.Conv2d(dim * num_experts, dim * num_experts, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * num_experts, dim, kernel_size=1)
        )
        
    def forward(self, expert_outputs):
        """
        expert_outputs: List of [B, C, H, W]
        """
        if isinstance(expert_outputs, list):
            # 对每个专家应用状态空间处理
            processed = []
            for i, expert_out in enumerate(expert_outputs):
                if i < len(self.state_conv):
                    processed.append(self.state_conv[i](expert_out))
                else:
                    processed.append(expert_out)
            
            # 拼接所有专家输出
            x = torch.cat(processed, dim=1)  # [B, C*num_experts, H, W]
        else:
            # 如果已经是tensor
            b, num_exp, c, h, w = expert_outputs.shape
            processed = []
            for i in range(num_exp):
                if i < len(self.state_conv):
                    processed.append(self.state_conv[i](expert_outputs[:, i, :, :, :]))
                else:
                    processed.append(expert_outputs[:, i, :, :, :])
            x = torch.cat(processed, dim=1)
        
        # 跨专家融合
        out = self.cross_expert(x)
        
        return out


##########################################################################
## 新增模块5: Dual Attention (Channel + Spatial)
class DualAttention(nn.Module):
    """双重注意力：通道注意力 + 空间注意力"""
    def __init__(self, dim):
        super(DualAttention, self).__init__()
        
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.GELU(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel Attention
        ca_weight = self.ca(x)
        x = x * ca_weight
        
        # Spatial Attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa_weight = self.sa(sa_input)
        x = x * sa_weight
        
        return x

    
     
##########################################################################
## Adapter Block (增强版：支持PSF Prompt和Dual Attention)    
class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size:int, 
                 use_psf_prompt=False, freq_dim=128, use_dual_attn=True):
        super(ModExpert, self).__init__()
        
        self.depth = depth
        self.use_psf_prompt = use_psf_prompt
        self.use_dual_attn = use_dual_attn
        
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)
            
        # PSF-Aware Prompt调制
        if use_psf_prompt:
            self.psf_prompt = PSFAwarePrompt(rank, prompt_dim=rank//2)
        
        # Dual Attention精修
        if use_dual_attn:
            self.dual_attn = DualAttention(rank)
            
    def process(self, x, shared, freq_emb=None):
        shortcut = x
        x = self.proj[0](x)
        
        # PSF-Aware Prompt调制
        if self.use_psf_prompt and freq_emb is not None:
            x = self.psf_prompt(x, freq_emb)
        
        x = self.body(x) * F.silu(self.proj[1](shared))
        
        # Dual Attention精修
        if self.use_dual_attn:
            x = self.dual_attn(x)
        
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared, freq_emb=None):
        for _ in range(self.depth):
            feat = self.process(feats, shared, freq_emb)
        return feat
    
    def forward(self, x, shared, freq_emb=None):
        b, c, h, w = x.shape
        
        if b == 0:
            return x
        else:
            x = self.feat_extract(x, shared, freq_emb)
            return x
        



########################################################################### 
## Adapter Layer (增强版：支持Wavelet分解和Mamba融合)
class AdapterLayer(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int=2, expert_layer: nn.Module=FFTAttention, stage_depth: int=1,
                 depth_type: str="lin", rank_type: str="constant", freq_dim: int=128, 
                 with_complexity: bool=False, complexity_scale: str="min",
                 use_wavelet: bool=True, use_mamba: bool=True, use_psf_prompt: bool=True, use_dual_attn: bool=True):
        super().__init__()            
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts
        self.use_wavelet = use_wavelet
        self.use_mamba = use_mamba

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
        
        # Wavelet分解（如果使用）
        if use_wavelet:
            self.wavelet = WaveletDecomposition()
        
        # 创建专家 - 直接使用ModExpert，不需要MySequential包装
        self.experts = nn.ModuleList([
            ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel,
                     use_psf_prompt=use_psf_prompt, freq_dim=freq_dim, use_dual_attn=use_dual_attn)
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])
                
        # Mamba融合
        if use_mamba:
            self.mamba_fusion = MambaFusion(dim, num_experts=num_experts, d_state=16)
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )
        
    def forward(self, x, freq_emb, shared):
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, freq_emb)
        self.loss = aux_loss
                
        # Wavelet分解（如果使用）
        if self.use_wavelet and self.num_experts == 4:
            wavelet_coeffs = self.wavelet(x)  # [LL, LH, HL, HH]
            # 每个专家处理对应的小波频带
            expert_outputs = []
            for i, coeff in enumerate(wavelet_coeffs):
                if i < len(self.experts):
                    # 上采样到原始尺寸
                    coeff_up = F.interpolate(coeff, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                    expert_out = self.experts[i](coeff_up, shared, freq_emb)
                    expert_outputs.append(expert_out)
            
            # 使用Mamba融合或直接加权融合
            if self.use_mamba:
                out = self.mamba_fusion(expert_outputs)
            else:
                # 加权融合
                expert_stack = torch.stack(expert_outputs, dim=1)  # [B, num_exp, C, H, W]
                gates_expanded = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, num_exp, 1, 1, 1]
                out = (expert_stack * gates_expanded).sum(dim=1)
        else:
            # 原始路由逻辑
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
                expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp], freq_emb) 
                                 for exp in range(len(self.experts))]
                
                if self.use_mamba:
                    out = self.mamba_fusion(expert_outputs)
                else:
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
                # 非训练模式：选择top-k专家并加权融合
                # top_k_indices: [B, k], 每行是当前样本的top-k专家索引
                # 简化处理：对每个样本，选择其top-k专家，然后加权融合
                b = x.shape[0]
                k = top_k_indices.shape[1]
                
                # 收集所有需要的专家输出
                all_expert_outputs = []
                for exp_idx in range(self.num_experts):
                    # 检查哪些样本使用了这个专家
                    mask = (top_k_indices == exp_idx).any(dim=1)  # [B]
                    if mask.any():
                        # 只对使用该专家的样本计算
                        expert_out = self.experts[exp_idx](x, shared, freq_emb)  # [B, C, H, W]
                        all_expert_outputs.append(expert_out)
                    else:
                        # 创建零输出占位
                        all_expert_outputs.append(torch.zeros_like(x))
                
                # 加权融合
                if self.use_mamba:
                    # Mamba融合期望专家输出列表
                    out = self.mamba_fusion(all_expert_outputs)
                else:
                    # 直接加权融合
                    expert_stack = torch.stack(all_expert_outputs, dim=1)  # [B, num_experts, C, H, W]
                    gates_expanded = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, num_experts, 1, 1, 1]
                    out = (expert_stack * gates_expanded).sum(dim=1)  # [B, C, H, W]
            
        out = self.proj_out(out)
        return out

    

class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max"):
        super(RoutingFunction, self).__init__()
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        ) 
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        
        self.k = k
        self.tau = 1
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, freq_emb):
        logits = self.gate(x) + self.freq_gate(freq_emb)
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
        
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        # Final auxiliary loss
        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0
        
        # AMP 下 softmax/topk 的输出 dtype 可能与 logits 不一致，scatter_ 要求两者 dtype 相同
        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values.to(dtype=logits.dtype))
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        importance = importance * (self.complexity * self.tau) if self.use_complexity_bias else importance
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        # Compute the noise threshold
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        
        # Compute the load for each expert
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )
        
        # Calculate noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        
        # Probability of being above the threshold
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        
        # Compute mean probability for each expert over examples
        p_mean = p.mean(dim=0)
        
        # Compute p_mean's coefficient of variation squared
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        
        return loss_load




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
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None, freq_dim:int=128, with_complexity: bool=False):
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
            use_wavelet=True, use_mamba=True, use_psf_prompt=True, use_dual_attn=True
        )
        
    def forward(self, x, freq_emb=None):    
        shortcut = x
        x = self.norms[0](x)
        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.adapter(x_a, freq_emb, x_s)
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
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None, rank_type=None, freq_dim:int=128, with_complexity: bool=False):
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
                    depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity
                )
            )

    def forward(self, x, freq_emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x , loss = self.layers[i](x, freq_emb)
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
## Frequency Embedding
class FrequencyEmbedding(nn.Module):
    """
    Embeds magnitude and phase features extracted from the bottleneck of the U-Net.
    """
    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        self.high_conv = nn.Sequential(
            HighPassConv2d(dim, freeze=True),
            nn.GELU())
        
        self.mlp= nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
            )

    def forward(self, x):
        x = self.high_conv(x)
        x = x.mean(dim=(-2, -1))
        x = self.mlp(x)
        return x
    
    
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
                ):
        super(MoCEIR, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]

        # -- Frequency Prior (新增)
        self.freq_prior = FrequencyPrior(in_channels=inp_channels, out_channels=dim)

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        self.freq_embed = FrequencyEmbedding(dims[-1])
                
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
                    rank=ranks[i], num_experts=num_experts, stage_depth=stage_depth[i], depth_type=depth_type, rank_type=rank_type, top_k=topk, complexity_scale=complexity_scale),
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
    
    def forward(self, x, labels=None):
        # Frequency Prior提取
        freq_prior_feat = self.freq_prior(x)
                
        # Patch Embedding + Frequency Prior融合
        feats = self.patch_embed(x) + freq_prior_feat
        
        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)
        
        feats = self.latent(feats)
        freq_emb = self.freq_embed(feats)
        self.last_freq_emb = freq_emb
                
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, freq_emb)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x
    
                    
    
    
if __name__ == "__main__":
    # test
    model = MoCEIR(rank=2, num_blocks=[4,6,6,8], num_dec_blocks=[2,4,4], levels=4, dim=48, num_refinement_blocks=4, 
                   with_complexity=True, complexity_scale="max", stage_depth=[1,1,1], depth_type="constant", rank_type="spread", 
                   num_experts=4, topk=1, expert_layer=FFTAttention).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    _ = model(x)
    print(model.total_loss)
    # Memory usage  
    print('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
  
    # FLOPS and PARAMS
    flops = FlopCountAnalysis(model, (x))
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
    )

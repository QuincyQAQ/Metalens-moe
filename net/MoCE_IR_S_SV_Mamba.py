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

    def dispatch(self, inp, prompts=None):
        """
        Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        
        Args:
            inp: Input tensor [B, C, H, W]
            prompts: Optional prompts tensor [B, num_experts, d_state] for visual prompt routing
        """
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        inp_split = torch.split(inp_exp, self._part_sizes, dim=0)
        
        if prompts is not None:
            # Dispatch prompts similarly
            prompts_exp = prompts[self._batch_index]  # [num_samples, num_experts, d_state]
            # For each expert, extract the corresponding prompt
            prompts_split = []
            for i in range(self._num_experts):
                expert_mask = (self._expert_index.squeeze(1) == i)
                if expert_mask.any():
                    expert_prompts = prompts_exp[expert_mask, i, :]  # [num_samples_for_expert_i, d_state]
                    prompts_split.append(expert_prompts)
                else:
                    prompts_split.append(None)
            return inp_split, prompts_split
        else:
            return inp_split

    def combine(self, expert_out, multiply_by_gates=True, soft_gates=None):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        
        Args:
            expert_out: List of expert outputs
            multiply_by_gates: Whether to multiply by gates
            soft_gates: Optional soft gates for weighting (if provided, used instead of sparse gates)
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if soft_gates is not None:
                # Use soft_gates for weighting if provided
                batch_gates = soft_gates[self._batch_index]  # [num_samples, num_experts]
                expert_gates = batch_gates.gather(1, self._expert_index)  # [num_samples, 1]
                stitched = stitched.mul(expert_gates.unsqueeze(-1).unsqueeze(-1))
            else:
                # Fallback to sparse gates if soft_gates not provided
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
## Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


##########################################################################
## Mamba Block (物理感知 Mamba 专家核心)
class MambaBlock(nn.Module):
    """
    简化版 Mamba Block 结构，用于物理感知的全局建模
    参考：https://github.com/hustvl/Vim/blob/main/models/vim.py
    或 https://github.com/Mamba-Image/Mamba-Image-Recognition/blob/main/vmamba/vmamba.py
    
    注意：这是一个概念验证版本，实际的 Mamba 实现会更复杂，可能需要引入 selective_scan_cuda 等自定义操作
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.conv_branch = nn.Conv1d(dim, dim, kernel_size=d_conv, padding=d_conv // 2, groups=dim)
        # x_proj 用于生成 Mamba 的选择性参数 B 和 C
        self.x_proj = nn.Linear(dim, d_state * 2) 
        # dt_proj 用于生成 Mamba 的时间步长 delta
        self.dt_proj = nn.Linear(dim, d_state)
        # A_proj 用于生成 Mamba 的状态矩阵 A (如果 A 是可学习的)
        self.a_proj = nn.Linear(dim, d_state)
        self.out_proj = nn.Linear(dim, dim)

        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # State matrix A (固定为简单示例，实际可学习)
        # 这是一个非常简化的 A 矩阵，实际 Mamba 的 A 矩阵通常是固定的或通过参数化生成
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).reshape(1, d_state, 1))

    def forward(self, x, prompt=None):
        """
        Args:
            x: (B, L, C) - 输入序列
            prompt: (B, d_state) - 视觉提示向量，用于动态调整 Mamba 的 B 矩阵
        Returns:
            y: (B, L, C) - 输出序列
        """
        B, L, C = x.shape
        
        x_norm = self.norm(x)

        # 卷积分支，用于提取局部特征
        x_conv = self.conv_branch(x_norm.transpose(1, 2)).transpose(1, 2) # (B, L, C)
        x_conv = F.silu(x_conv)

        # 线性投影，生成 Mamba 参数
        x_proj = self.x_proj(x_conv) # (B, L, d_state * 2)
        B_mat, C_mat = x_proj.chunk(2, dim=-1) # (B, L, d_state)

        # 视觉提示注入：动态调整 Mamba 的 B 矩阵
        if prompt is not None: # prompt: (B, d_state)
            # 将 prompt 扩展到序列长度 L，并加到 B_mat 上
            B_mat = B_mat + prompt.unsqueeze(1).expand(-1, L, -1) 

        delta = F.softplus(self.dt_proj(x_conv)) # (B, L, d_state)
        A_mat = self.a_proj(x_conv) # (B, L, d_state)

        # 简化版 SSM 操作 (概念性实现，非真实 Mamba)
        # 真实的 Mamba 涉及更复杂的选择性扫描和状态更新
        # 这里仅为占位符，实际需要替换为 Vision Mamba 的核心 SSM 逻辑
        # 例如，可以参考 VMamba 的 selective_scan_2d 函数
        
        # 模拟 SSM 的输入
        x_ssm = F.silu(x_norm * self.expand)

        # 简化处理：直接通过线性层作为 SSM 的输出
        # 实际应替换为 Mamba 的核心选择性扫描机制
        y = self.out_proj(x_ssm) # (B, L, C)

        return y + x # 残差连接


##########################################################################
## Heterogeneous Expert (支持 Mamba 专家类型)
class HeteroExpert(nn.Module):
    """
    异构专家：不同类型的专家处理不同尺度的信息或不同类型的退化
    
    支持的专家类型：
    - global_field_curvature_mamba: 全局场曲校正专家，使用 Mamba Block
    - long_range_contrast_mamba: 长程对比度增强专家，结合 Mamba 与通道注意力
    - high_frequency_detail: 高频细节恢复专家（传统 CNN）
    - low_frequency_structure: 低频结构修复专家（传统 CNN）
    - fft_based_enhancement: FFT 域增强专家
    """
    def __init__(self, dim, expert_type="global_field_curvature_mamba", H=None, W=None):
        super().__init__()
        self.expert_type = expert_type
        self.H = H # 用于 Rearrange 恢复 2D 形状
        self.W = W # 用于 Rearrange 恢复 2D 形状

        if expert_type == "global_field_curvature_mamba":
            # 全局场曲校正专家：使用 Mamba Block
            # 利用全局感受野修复由场曲引起的边缘失焦和几何畸变
            self.body = nn.ModuleList([
                nn.Conv2d(dim, dim, 1), # 2D -> 1D 准备
                MambaBlock(dim), # Mamba Block
                nn.Conv2d(dim, dim, 1), # 后处理
                ChannelAttention(dim) # 局部注意力
            ])
        elif expert_type == "long_range_contrast_mamba":
            # 长程对比度增强专家：结合 Mamba 与通道注意力
            # 修复内窥镜成像中常见的中心过亮、边缘过暗以及全局对比度下降问题
            self.body = nn.ModuleList([
                nn.Conv2d(dim, dim, 1), # 2D -> 1D 准备
                MambaBlock(dim), # Mamba Block
                nn.Conv2d(dim, dim, 1), # 后处理
                nn.AdaptiveAvgPool2d(1), # 全局上下文
                nn.Conv2d(dim, dim, 1)
            ])
        elif expert_type == "high_frequency_detail":
            # 高频细节恢复专家：3x3 深度可分离卷积，专注于高频细节
            self.body = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False)
            )
        elif expert_type == "low_frequency_structure":
            # 低频结构修复专家：带空洞率的 3x3 卷积，扩大感受野
            self.body = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 2, dilation=2, groups=dim, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False)
            )
        elif expert_type == "fft_based_enhancement":
            # FFT 域增强专家：利用 FFT 变换在频域进行处理
            self.body = nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False)
            )
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")

    def forward(self, x, prompt=None):
        """
        Args:
            x: (B, C, H, W) - 输入特征
            prompt: (B, d_state) - 视觉提示向量（仅用于 Mamba 专家）
        Returns:
            out: (B, C, H, W) - 输出特征
        """
        B, C, H, W = x.shape

        # 处理 Mamba 专家
        if self.expert_type == "global_field_curvature_mamba":
            # 假设 body 结构为 [Conv2d, MambaBlock, Conv2d, ChannelAttention]
            x_processed = self.body[0](x) # Conv2d
            x_processed = rearrange(x_processed, 'b c h w -> b (h w) c') # 2D -> 1D
            x_processed = self.body[1](x_processed, prompt=prompt) # MambaBlock with prompt
            x_processed = rearrange(x_processed, 'b (h w) c -> b c h w', h=H, w=W) # 1D -> 2D
            # 经过后续层
            for i in range(2, len(self.body)):
                x_processed = self.body[i](x_processed)
            return x_processed
        elif self.expert_type == "long_range_contrast_mamba":
            # 假设 body 结构为 [Conv2d, MambaBlock, Conv2d, AdaptiveAvgPool2d(1), Conv2d]
            # 需要特殊处理 AdaptiveAvgPool2d，因为它会将空间维度降为 1x1
            x_processed = self.body[0](x) # Conv2d
            x_processed = rearrange(x_processed, 'b c h w -> b (h w) c') # 2D -> 1D
            x_processed = self.body[1](x_processed, prompt=prompt) # MambaBlock with prompt
            x_processed = rearrange(x_processed, 'b (h w) c -> b c h w', h=H, w=W) # 1D -> 2D
            x_processed = self.body[2](x_processed) # Conv2d
            x_processed = self.body[3](x_processed) # AdaptiveAvgPool2d(1) -> (B, C, 1, 1)
            x_processed = self.body[4](x_processed) # Conv2d -> (B, C, 1, 1)
            # 上采样回原始空间尺寸
            x_processed = F.interpolate(x_processed, size=(H, W), mode='bilinear', align_corners=False)
            return x_processed

        # 处理传统专家类型
        elif self.expert_type == "high_frequency_detail" or self.expert_type == "low_frequency_structure":
            return self.body(x)
        elif self.expert_type == "fft_based_enhancement":
            # FFT 专家直接处理 x
            x_fft = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
            magnitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)
            processed_magnitude = magnitude * 1.1
            processed_fft = torch.view_as_complex(torch.stack([processed_magnitude * torch.cos(phase), processed_magnitude * torch.sin(phase)], dim=-1))
            y_h = torch.fft.ifft2(processed_fft, norm="ortho").real
            return self.body(y_h)
        else:
            raise ValueError(f"Unknown expert_type: {self.expert_type}")


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
## Spatially-Variant Frequency Embedding (第一个创新点)
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
## Dynamic Routing with Visual Prompt (视觉提示路由)
class DynamicRouting(nn.Module):
    """
    视觉提示路由：不仅分配专家，还生成动态的"视觉提示词"来实时调节专家的恢复行为
    
    支持的路由类型：
    - visual_prompt_mamba: 视觉提示路由（推荐，用于 Mamba 专家）
    - frequency_spatial: 结合频率和空间信息进行路由（传统路由）
    - frequency_only: 仅使用频率信息
    - spatial_only: 仅使用空间信息
    """
    def __init__(self, dim, num_experts, k=1, temperature=1.0, router_type="visual_prompt_mamba"):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        self.router_type = router_type

        if router_type == "visual_prompt_mamba":
            # 视觉提示路由：生成提示向量和专家选择权重
            self.freq_embed = SpatiallyVariantFreqEmbedding(dim) # 空间变异频率嵌入 (SVFE)
            self.spatial_context_embed = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=1, bias=False)
            ) 
            
            # 提示编码器：将融合特征映射为每个专家的提示向量
            # 假设每个专家的提示向量维度与 MambaBlock 的 d_state 相同 (默认为 16)
            self.prompt_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim + dim // 4, dim), # 融合 SVFE (dim), 空间上下文 (dim//4)
                    nn.GELU(),
                    nn.Linear(dim, 16) # 输出提示向量维度，MambaBlock 的 d_state
                ) for _ in range(num_experts)
            ])
            
            # 专家选择器：用于生成传统的专家权重，以决定哪些专家被激活
            self.expert_selector = nn.Sequential(
                nn.Linear(dim + dim // 4, dim * 2), # 融合 SVFE (dim), 空间上下文 (dim//4)
                nn.GELU(),
                nn.Linear(dim * 2, num_experts)
            )

        elif router_type == "frequency_spatial":
            # 结合频率和空间信息进行路由（传统路由，不生成提示）
            self.freq_embed = SpatiallyVariantFreqEmbedding(dim)
            self.spatial_embed = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=1, bias=False)
            )
            self.fusion = nn.Sequential(
                nn.Linear(dim + dim // 4, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, num_experts)
            )
        elif router_type == "frequency_only":
            self.freq_embed = SpatiallyVariantFreqEmbedding(dim)
            self.router = nn.Linear(dim, num_experts)
        elif router_type == "spatial_only":
            self.spatial_embed = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=1, bias=False)
            )
            self.router = nn.Linear(dim // 4, num_experts)
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        # Load balancing loss components
        self.register_buffer("mean_expert_load", torch.zeros(num_experts))
        self.register_buffer("mean_expert_importance", torch.zeros(num_experts))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - 输入特征
        
        Returns:
            sparse_gates: [B, num_experts] - 稀疏门控（用于 SparseDispatcher）
            soft_gates: [B, num_experts] - 软门控（用于加权专家输出）
            prompts: [B, num_experts, d_state] - 视觉提示向量（仅当 router_type="visual_prompt_mamba" 时返回）
        """
        B, C, H, W = x.shape

        if self.router_type == "visual_prompt_mamba":
            freq_diag_feat = self.freq_embed(x) # SVFE 诊断图
            spatial_ctx_feat = self.spatial_context_embed(x)

            # 对 SVFE 和空间上下文特征进行全局池化
            freq_diag_pool = F.adaptive_avg_pool2d(freq_diag_feat.unsqueeze(-1).unsqueeze(-1), (1, 1)).squeeze(-1).squeeze(-1) if freq_diag_feat.dim() == 2 else freq_diag_feat
            spatial_ctx_pool = F.adaptive_avg_pool2d(spatial_ctx_feat, (1, 1)).squeeze(-1).squeeze(-1)

            # 融合特征用于提示生成和专家选择
            fused_feat_for_prompt = torch.cat([freq_diag_pool, spatial_ctx_pool], dim=-1)

            # 生成每个专家的提示向量
            prompts = []
            for i in range(self.num_experts):
                prompts.append(self.prompt_encoders[i](fused_feat_for_prompt).unsqueeze(1)) # (B, 1, d_state)
            prompts = torch.cat(prompts, dim=1) # (B, num_experts, d_state)

            # 生成专家选择权重
            logits = self.expert_selector(fused_feat_for_prompt)

        elif self.router_type == "frequency_spatial":
            freq_feat = self.freq_embed(x)  # [B, dim]
            spatial_feat = self.spatial_embed(x)  # [B, dim//4, H, W]
            spatial_feat_pool = F.adaptive_avg_pool2d(spatial_feat, (1, 1)).squeeze(-1).squeeze(-1)  # [B, dim//4]
            fused_feat = torch.cat([freq_feat, spatial_feat_pool], dim=-1)  # [B, dim + dim//4]
            logits = self.fusion(fused_feat)  # [B, num_experts]
            prompts = None
        elif self.router_type == "frequency_only":
            freq_feat = self.freq_embed(x)  # [B, dim]
            logits = self.router(freq_feat)  # [B, num_experts]
            prompts = None
        elif self.router_type == "spatial_only":
            spatial_feat = self.spatial_embed(x)  # [B, dim//4, H, W]
            spatial_feat_pool = F.adaptive_avg_pool2d(spatial_feat, (1, 1)).squeeze(-1).squeeze(-1)  # [B, dim//4]
            logits = self.router(spatial_feat_pool)  # [B, num_experts]
            prompts = None
        else:
            raise ValueError(f"Unknown router_type: {self.router_type}")

        # Softmax with temperature
        soft_gates = softmax_with_temperature(logits, self.temperature)  # [B, num_experts]

        # Select top-k experts (for sparse dispatching)
        top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)  # [B, k]
        # Create sparse gates (binary for SparseDispatcher)
        sparse_gates = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)  # [B, num_experts]

        # Update load balancing statistics (for auxiliary loss)
        with torch.no_grad():
            expert_load = sparse_gates.sum(0)  # [num_experts] - Number of samples routed to each expert
            expert_importance = soft_gates.sum(0)  # [num_experts] - Sum of soft gate values for each expert
            self.mean_expert_load = self.mean_expert_load * 0.99 + expert_load.detach() * 0.01
            self.mean_expert_importance = self.mean_expert_importance * 0.99 + expert_importance.detach() * 0.01

        return sparse_gates, soft_gates, prompts


########################################################################### 
## MoE Layer with Mamba Experts and Visual Prompt Routing
class MoELayer(nn.Module):
    """
    MoE 层：使用异构专家（包括 Mamba 专家）和视觉提示路由
    """
    def __init__(self, 
                 dim: int, 
                 num_experts: int = 4, 
                 top_k: int = 1,
                 expert_types: Optional[List[str]] = None,
                 router_type: str = "visual_prompt_mamba",
                 temperature: float = 1.0):
        super().__init__()            
        
        self.loss = None
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 设置异构专家类型
        if expert_types is None:
            # 默认异构专家类型：包含 Mamba 专家
            expert_types = ["global_field_curvature_mamba", "long_range_contrast_mamba", 
                          "high_frequency_detail", "low_frequency_structure"][:num_experts]
        assert len(expert_types) == num_experts, "Number of expert types must match num_experts"
        
        # 创建异构专家
        self.experts = nn.ModuleList([
            HeteroExpert(dim, expert_type=expert_types[i]) 
            for i in range(num_experts)
        ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        
        # 动态路由（支持视觉提示）
        self.routing = DynamicRouting(
            dim=dim, 
            num_experts=num_experts, 
            k=top_k, 
            temperature=temperature, 
            router_type=router_type
        )
        
    def forward(self, x, shared=None):
        """
        Args:
            x: [B, C, H, W] - 输入特征
            shared: [B, C, H, W] - 共享特征（用于兼容原有接口，但当前不使用）
        
        Returns:
            out: [B, C, H, W] - 输出特征
        """
        # 获取路由门控和提示向量
        sparse_gates, soft_gates, prompts = self.routing(x)
        
        # 路由到专家
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, sparse_gates)
            if prompts is not None:
                # 如果有提示向量，需要分发提示
                expert_inputs, expert_prompts = dispatcher.dispatch(x, prompts)
                expert_outputs = []
                for i in range(len(self.experts)):
                    if expert_prompts[i] is not None:
                        # Mamba 专家使用提示向量
                        expert_outputs.append(self.experts[i](expert_inputs[i], prompt=expert_prompts[i]))
                    else:
                        # 传统专家不使用提示向量
                        expert_outputs.append(self.experts[i](expert_inputs[i]))
                out = dispatcher.combine(expert_outputs, multiply_by_gates=True, soft_gates=soft_gates)
            else:
                # 没有提示向量，传统路由
                expert_inputs = dispatcher.dispatch(x)
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(len(self.experts))]
                out = dispatcher.combine(expert_outputs, multiply_by_gates=True, soft_gates=soft_gates)
        else:
            # 推理时：选择 top-k 专家
            top_k_values, top_k_indices = sparse_gates.topk(self.top_k, dim=-1)  # [B, k]
            
            # 为每个batch选择对应的专家并处理
            B, C, H, W = x.shape
            expert_outputs_list = []
            for b in range(B):
                batch_expert_indices = top_k_indices[b]  # [k]
                batch_expert_outputs = []
                for k_idx in range(self.top_k):
                    expert_idx = batch_expert_indices[k_idx].item()
                    if prompts is not None:
                        # 使用对应的提示向量
                        expert_prompt = prompts[b, expert_idx, :]  # [d_state]
                        expert_output = self.experts[expert_idx](x[b:b+1], prompt=expert_prompt)  # [1, C, H, W]
                    else:
                        expert_output = self.experts[expert_idx](x[b:b+1])  # [1, C, H, W]
                    batch_expert_outputs.append(expert_output)
                expert_outputs_list.append(torch.cat(batch_expert_outputs, dim=0))  # [k, C, H, W]
            
            expert_outputs = torch.stack(expert_outputs_list, dim=0)  # [B, k, C, H, W]
            
            # 使用软门控加权
            top_k_soft_gates = soft_gates.gather(1, top_k_indices)  # [B, k]
            weighted_outputs = expert_outputs * top_k_soft_gates.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, k, C, H, W]
            out = weighted_outputs.sum(dim=1)  # [B, C, H, W]
            
        out = self.proj_out(out)
        return out


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
## Decoder Block with Mamba MoE
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 num_experts=None, top_k=None, expert_types=None, router_type="visual_prompt_mamba", temperature=1.0):
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
        
        # 使用 Mamba MoE 层
        self.moe = MoELayer(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_types=expert_types,
            router_type=router_type,
            temperature=temperature
        )
        
    def forward(self, x, freq_emb=None):    
        shortcut = x
        x = self.norms[0](x)
        
        x_s = self.proj[0](x)
        x_a = self.proj[1](x)
        x_s = self.shared(x_s)
        x_a = self.moe(x_a, x_s)  # Mamba MoE 层
        x = self.mixer(x_a, x_s) + shortcut

        x = x + self.ffn(self.norms[1](x))
        return x

    
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
## Decoder Residual Group with Mamba MoE
class DecoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool,
                 num_experts=None, top_k=None, expert_types=None, router_type="visual_prompt_mamba", temperature=1.0):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    dim, num_heads, ffn_expansion, bias, LayerNorm_type,
                    num_experts=num_experts, top_k=top_k, expert_types=expert_types,
                    router_type=router_type, temperature=temperature
                )
            )

    def forward(self, x, freq_emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x = self.layers[i](x, freq_emb)
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
## Main Model: MoCEIR with Mamba Experts and Visual Prompt Routing
class MoCEIR(nn.Module):
    """
    融合物理感知 Mamba 专家与视觉提示路由的超透镜图像重建模型：
    1. 空间变异频率嵌入 (Spatially-Variant Frequency Embedding)
    2. 物理感知 Mamba 专家 (Physics-Aware Mamba Experts)
    3. 视觉提示路由 (Visual Prompt Routing)
    """
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
                LayerNorm_type = 'WithBias',
                bias = False,
                num_experts=4,
                topk=1,
                expert_types: Optional[List[str]] = None,
                router_type: str = "visual_prompt_mamba",
                temperature: float = 1.0,
                ):
        super(MoCEIR, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]

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
                    LayerNorm_type=LayerNorm_type, bias=bias,
                    num_experts=num_experts, top_k=topk, expert_types=expert_types,
                    router_type=router_type, temperature=temperature),
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
        """
        Forward pass of the MoCEIR model with Mamba experts and visual prompt routing.
        
        Args:
            x: Input degraded image [B, C, H, W]
            labels: Optional labels (not used in this implementation)
        
        Returns:
            x: Restored image [B, C, H, W]
        """
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
                
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            feats = block(feats, freq_emb)
            # Note: MoELayer doesn't have loss attribute, so we skip loss accumulation
            # If you want to add load balancing loss, you can add it to MoELayer

        feats = self.refinement(feats)
        x = self.output(feats) + x

        return x


##########################################################################
## Build Model Function
def build_model(opt):
    """
    Build the MoCEIR model with Mamba experts from configuration.
    
    Args:
        opt: Configuration object with model parameters
    
    Returns:
        model: MoCEIR model instance
    """
    dim = getattr(opt, 'dim', 32)
    num_blocks = getattr(opt, 'num_blocks', [4, 6, 6, 8])
    num_dec_blocks = getattr(opt, 'num_dec_blocks', [2, 4, 4])
    heads = getattr(opt, 'heads', [1, 2, 4, 8])
    num_refinement_blocks = getattr(opt, 'num_refinement_blocks', 4)
    topk = getattr(opt, 'topk', 1)
    num_experts = getattr(opt, 'num_exp_blocks', 4)
    expert_types = getattr(opt, 'expert_types', None)
    router_type = getattr(opt, 'router_type', 'visual_prompt_mamba')
    temperature = getattr(opt, 'temperature', 1.0)

    return MoCEIR(
        dim=dim,
        num_blocks=num_blocks,
        num_dec_blocks=num_dec_blocks,
        levels=len(num_blocks),
        heads=heads,
        num_refinement_blocks=num_refinement_blocks,
        topk=topk,
        num_experts=num_experts,
        expert_types=expert_types,
        router_type=router_type,
        temperature=temperature,
    )


##########################################################################
## Test Code
if __name__ == "__main__":
    # Test the model
    print("Testing MoCEIR with Mamba Experts and Visual Prompt Routing...")
    
    model = MoCEIR(
        num_blocks=[4,6,6,8], 
        num_dec_blocks=[2,4,4], 
        levels=4, 
        dim=48, 
        num_refinement_blocks=4, 
        num_experts=4, 
        topk=1, 
        expert_types=["global_field_curvature_mamba", "long_range_contrast_mamba", 
                     "high_frequency_detail", "low_frequency_structure"],
        router_type="visual_prompt_mamba",
        temperature=1.0
    ).cuda()

    x = torch.randn(1, 3, 224, 224).cuda()
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Memory usage  
    print('{:>16s} : {:<.3f} [M]'.format('Max Memory', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))
  
    # FLOPS and PARAMS
    try:
        flops = FlopCountAnalysis(model, (x,))
        print(flop_count_table(flops))
    except Exception as e:
        print(f"FLOPS calculation failed: {e}")
    
    print("Model test completed successfully!")

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
## 第二个创新点：频率域解耦专家（Frequency-Decoupled Experts）
## 核心思想：通过FFT将特征分解为低频和高频成分，分别设计专家处理
## 参考：CVPR 2024/2025 频率域图像恢复方法

class FrequencyDecoupledExpert(nn.Module):
    """
    频率域解耦专家：将输入特征分解为低频和高频成分，分别处理后再融合
    低频专家：处理结构信息、全局对比度
    高频专家：处理细节纹理、边缘信息
    """
    def __init__(self, dim: int, rank: int, expert_type: str = "low_frequency", 
                 depth: int = 1, patch_size: int = 4, kernel_size: int = 3):
        super(FrequencyDecoupledExpert, self).__init__()
        
        self.expert_type = expert_type
        self.dim = dim
        self.rank = rank
        
        # 频率分解参数：定义低频和高频的阈值
        # 使用可学习的频率掩码，而不是硬阈值
        self.freq_mask_low = nn.Parameter(torch.ones(1, 1, patch_size, patch_size // 2 + 1))
        self.freq_mask_high = nn.Parameter(torch.ones(1, 1, patch_size, patch_size // 2 + 1))
        
        # 投影层
        self.proj_in = nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False)
        self.proj_shared = nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False)
        self.proj_out = nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        
        # 根据专家类型设计不同的处理模块
        if expert_type == "low_frequency":
            # 低频专家：大感受野，关注结构
            self.body = nn.Sequential(
                nn.Conv2d(rank, rank, kernel_size=kernel_size, padding=kernel_size//2, groups=rank),
                nn.GELU(),
                nn.Conv2d(rank, rank, kernel_size=kernel_size*2+1, padding=kernel_size, groups=rank),
                nn.GELU(),
            )
        elif expert_type == "high_frequency":
            # 高频专家：小感受野，关注细节
            self.body = nn.Sequential(
                nn.Conv2d(rank, rank, kernel_size=kernel_size, padding=kernel_size//2, groups=rank),
                nn.GELU(),
                nn.Conv2d(rank, rank, kernel_size=1, padding=0),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown expert_type: {expert_type}")
    
    def frequency_decompose(self, x):
        """
        将输入特征分解为低频和高频成分
        x: (B, C, H, W)
        返回: x_low, x_high
        
        注意：cuFFT对某些尺寸有限制，特别是：
        - 尺寸必须 > 0
        - 某些小尺寸（如 < 4）可能不支持
        - fp16下限制更严格
        因此我们强制使用float32进行FFT操作
        如果FFT失败，使用空间域低通/高通滤波器作为回退
        """
        B, C, H, W = x.shape
        
        # 检查输入尺寸有效性
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid input size for FFT: H={H}, W={W}")
        
        # 保存原始数据类型和设备
        original_dtype = x.dtype
        original_device = x.device
        
        # 强制转换为 float32 进行 FFT（cuFFT在fp16下对某些尺寸有限制）
        # 即使输入已经是float32，也确保是contiguous的
        x = x.float().contiguous()
        
        # 尝试FFT分解，如果失败则使用空间域回退
        try:
            # FFT变换
            x_fft = torch.fft.rfft2(x, norm="ortho")  # (B, C, H, W//2+1) 复数
            
            # 获取频率掩码（可学习）
            # 对频率域进行软掩码，而不是硬分割
            mask_low = torch.sigmoid(self.freq_mask_low)  # 低频掩码 (1, 1, patch_size, patch_size//2+1)
            mask_high = 1.0 - mask_low  # 高频掩码
            
            # 将掩码插值到与 x_fft 相同的尺寸
            # x_fft 的形状是 (B, C, H, W//2+1)
            # 需要将掩码从 (1, 1, patch_size, patch_size//2+1) 插值到 (1, 1, H, W//2+1)
            _, _, mask_h, mask_w = mask_low.shape
            if mask_h != H or mask_w != (W // 2 + 1):
                # 使用双线性插值将掩码调整到输入尺寸
                mask_low_resized = F.interpolate(
                    mask_low,
                    size=(H, W // 2 + 1),
                    mode='bilinear',
                    align_corners=False
                )
                mask_high_resized = F.interpolate(
                    mask_high,
                    size=(H, W // 2 + 1),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                mask_low_resized = mask_low
                mask_high_resized = mask_high
            
            # 应用掩码（广播机制会自动处理维度）
            x_low_fft = x_fft * mask_low_resized
            x_high_fft = x_fft * mask_high_resized
            
            # 逆FFT变换回空间域
            x_low = torch.fft.irfft2(x_low_fft, s=(H, W), norm="ortho")
            x_high = torch.fft.irfft2(x_high_fft, s=(H, W), norm="ortho")
            
        except RuntimeError as e:
            # FFT失败（cuFFT或MKL错误），使用空间域回退方案
            # 使用简单的低通和高通滤波器进行分解
            # 低通：使用平均池化 + 上采样
            # 高通：原始 - 低通
            
            # 计算合适的kernel size（基于patch_size）
            _, _, patch_h, patch_w = self.freq_mask_low.shape
            kernel_size = min(patch_h, patch_w, H, W)
            if kernel_size % 2 == 0:
                kernel_size += 1  # 确保是奇数
            
            # 低通滤波：使用高斯模糊（通过多次平均池化近似）
            x_low = x
            for _ in range(2):  # 两次平滑
                x_low = F.avg_pool2d(x_low, kernel_size=3, stride=1, padding=1)
            
            # 上采样回原始尺寸
            if x_low.shape[2:] != (H, W):
                x_low = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
            
            # 高通 = 原始 - 低通
            x_high = x - x_low
        
        # 转换回原始数据类型
        x_low = x_low.to(original_dtype)
        x_high = x_high.to(original_dtype)
        
        return x_low, x_high
    
    def forward(self, x, shared):
        """
        x: 输入特征 (B, C, H, W)
        shared: 共享特征 (B, C, H, W)
        """
        shortcut = x
        
        # 投影
        x = self.proj_in(x)
        shared_proj = self.proj_shared(shared)
        
        # 频率分解
        x_low, x_high = self.frequency_decompose(x)
        
        # 根据专家类型选择处理低频或高频
        if self.expert_type == "low_frequency":
            x_processed = self.body(x_low)
        else:  # high_frequency
            x_processed = self.body(x_high)
        
        # 门控机制：使用共享特征调制
        x_processed = x_processed * F.silu(shared_proj)
        
        # 输出投影
        x_out = self.proj_out(x_processed)
        
        return x_out + shortcut


##########################################################################
## 第二个创新点：跨尺度频率感知路由（Cross-Scale Frequency-Aware Routing）
## 核心思想：在多个尺度上提取频率特征，融合后用于路由决策
## 参考：ECCV 2024 多尺度频率感知方法

class CrossScaleFreqRouting(nn.Module):
    """
    跨尺度频率感知路由：在多个尺度上提取频率特征，融合后用于专家路由
    相比单一尺度的SVFE，能够捕获不同尺度的退化模式
    """
    def __init__(self, dim, freq_dim, num_experts, k, complexity, 
                 use_complexity_bias: bool = True, complexity_scale: str = "max",
                 num_scales: int = 3, freq_emb_global_dim: Optional[int] = None):
        super(CrossScaleFreqRouting, self).__init__()
        
        self.dim = dim
        self.freq_dim = freq_dim
        self.num_experts = num_experts
        self.k = k
        self.num_scales = num_scales
        
        # 如果 freq_emb_global 的维度与 freq_dim 不匹配，需要投影
        # freq_emb_global 来自 SpatiallyVariantFreqEmbedding(dims[-1])，维度是 dims[-1]
        # 但 freq_dim 是 dims[0]，所以需要投影
        if freq_emb_global_dim is not None and freq_emb_global_dim != freq_dim:
            self.freq_emb_proj = nn.Linear(freq_emb_global_dim, freq_dim)
        else:
            self.freq_emb_proj = None
        
        # 多尺度频率特征提取
        # 在不同分辨率下提取频率特征，捕获不同尺度的退化模式
        self.scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((2**i, 2**i)) for i in range(num_scales)
        ])
        
        # 每个尺度的频率嵌入提取器
        # 确保所有尺度的输出拼接后总维度等于 freq_dim
        scale_dims = []
        base_dim = freq_dim // num_scales
        remainder = freq_dim % num_scales
        for i in range(num_scales):
            # 将余数分配给前 remainder 个尺度
            scale_dim = base_dim + (1 if i < remainder else 0)
            scale_dims.append(scale_dim)
        
        self.scale_freq_extractors = nn.ModuleList([
            nn.Sequential(
                HighPassConv2d(dim, freeze=True),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                Rearrange('b c 1 1 -> b c'),
                nn.Linear(dim, scale_dims[i])
            ) for i in range(num_scales)
        ])
        
        # 融合多尺度频率特征
        # 注意：输入是 freq_emb_global (投影到freq_dim) + multi_scale_freq (freq_dim) = freq_dim * 2
        self.freq_fusion = nn.Sequential(
            nn.Linear(freq_dim * 2, freq_dim * 2),
            nn.GELU(),
            nn.Linear(freq_dim * 2, freq_dim)
        )
        
        # 空间特征提取（保持原有逻辑）
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        
        # 频率门控
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        
        # 复杂度偏置
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)
        
        self.tau = 1
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias
    
    def forward(self, x, freq_emb_global):
        """
        x: 输入特征 (B, C, H, W)
        freq_emb_global: 全局频率嵌入 (B, freq_emb_global_dim) - 来自SVFE，维度可能是 dims[-1]
        """
        B, C, H, W = x.shape
        
        # 如果 freq_emb_global 的维度与 freq_dim 不匹配，进行投影
        if self.freq_emb_proj is not None:
            freq_emb_global = self.freq_emb_proj(freq_emb_global)  # (B, freq_dim)
        
        # 多尺度频率特征提取
        scale_freq_features = []
        for i, (pool, extractor) in enumerate(zip(self.scale_pools, self.scale_freq_extractors)):
            # 下采样到不同尺度
            x_scale = pool(x)  # (B, C, 2^i, 2^i)
            # 提取该尺度的频率特征（每个尺度的输出维度可能不同，但总和为 freq_dim）
            freq_feat_scale = extractor(x_scale)
            scale_freq_features.append(freq_feat_scale)
        
        # 拼接多尺度频率特征（确保总维度为 freq_dim）
        multi_scale_freq = torch.cat(scale_freq_features, dim=-1)  # (B, freq_dim)
        
        # 融合全局频率嵌入和多尺度频率特征
        # 这里可以设计不同的融合策略：相加、拼接、注意力等
        # 当前使用拼接后融合
        combined_freq = torch.cat([freq_emb_global, multi_scale_freq], dim=-1)  # (B, freq_dim * 2)
        fused_freq = self.freq_fusion(combined_freq)  # (B, freq_dim)
        
        # 生成路由logits
        spatial_logits = self.spatial_gate(x)  # (B, num_experts)
        freq_logits = self.freq_gate(fused_freq)  # (B, num_experts)
        
        logits = spatial_logits + freq_logits
        
        # 计算辅助损失
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
        
        # 添加噪声（用于训练时的负载均衡）
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)
        
        # 最终辅助损失
        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0
        
        # 生成稀疏门控
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
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2
        return loss_load


##########################################################################
## Adapter Block with Frequency-Decoupled Experts
class ModExpertFreqDecoupled(nn.Module):
    def __init__(self, dim: int, rank: int, expert_type: str, depth: int, 
                 patch_size: int, kernel_size: int):
        super(ModExpertFreqDecoupled, self).__init__()
        
        self.depth = depth
        self.expert = FrequencyDecoupledExpert(
            dim=dim, rank=rank, expert_type=expert_type,
            depth=depth, patch_size=patch_size, kernel_size=kernel_size
        )
    
    def forward(self, x, shared):
        for _ in range(self.depth):
            x = self.expert(x, shared)
        return x


########################################################################### 
## Adapter Layer with Cross-Scale Frequency-Aware Routing
class AdapterLayerCrossScaleFreq(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int = 2, 
                 stage_depth: int = 1, depth_type: str = "lin", 
                 rank_type: str = "constant", freq_dim: int = 128, 
                 with_complexity: bool = False, complexity_scale: str = "min",
                 num_scales: int = 3, freq_emb_global_dim: Optional[int] = None):
        super().__init__()
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.num_experts = num_experts
        
        patch_sizes = [2**(i+2) for i in range(num_experts)]
        kernel_sizes = [3+(2*i) for i in range(num_experts)]
        
        # 深度配置
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
        
        # Rank配置
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
        
        # 创建频率域解耦专家
        # 交替分配低频和高频专家
        expert_types = []
        for i in range(num_experts):
            if i % 2 == 0:
                expert_types.append("low_frequency")
            else:
                expert_types.append("high_frequency")
        
        self.experts = nn.ModuleList([
            ModExpertFreqDecoupled(
                dim=dim, rank=ranks[i], expert_type=expert_types[i],
                depth=depths[i], patch_size=patch_sizes[i], kernel_size=kernel_sizes[i]
            ) for i in range(num_experts)
        ])
        
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        
        # 计算专家复杂度
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        
        # 跨尺度频率感知路由
        self.routing = CrossScaleFreqRouting(
            dim=dim, freq_dim=freq_dim, num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity,
            complexity_scale=complexity_scale, num_scales=num_scales,
            freq_emb_global_dim=freq_emb_global_dim
        )
    
    def forward(self, x, freq_emb, shared):
        """
        x: 输入特征 (B, C, H, W)
        freq_emb: 全局频率嵌入 (B, freq_dim) - 来自SVFE
        shared: 共享特征 (B, C, H, W)
        """
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, freq_emb)
        self.loss = aux_loss
        
        # 路由分发
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_inputs = dispatcher.dispatch(shared)
            expert_outputs = [
                self.experts[exp](expert_inputs[exp], expert_shared_inputs[exp]) 
                for exp in range(len(self.experts))
            ]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]
            expert_outputs = torch.stack([expert(x, shared) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs
            out = weighted_outputs.sum(dim=1)
        
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
## Decoder Block with Cross-Scale Frequency-Aware Adapter
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 rank=None, num_experts=None, top_k=None, depth_type=None, 
                 rank_type=None, stage_depth=None, freq_dim: int = 128, 
                 with_complexity: bool = False, complexity_scale=None,
                 num_scales: int = 3, freq_emb_global_dim: Optional[int] = None):
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
        
        # 使用跨尺度频率感知适配器
        self.adapter = AdapterLayerCrossScaleFreq(
            dim=dim, rank=rank, top_k=top_k, num_experts=num_experts, 
            freq_dim=freq_dim, depth_type=depth_type, rank_type=rank_type, 
            stage_depth=stage_depth, with_complexity=with_complexity, 
            complexity_scale=complexity_scale, num_scales=num_scales,
            freq_emb_global_dim=freq_emb_global_dim
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
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, 
                 LayerNorm_type: str, bias: bool):
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
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, 
                 LayerNorm_type: str, bias: bool, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, 
                 stage_depth=None, rank_type=None, freq_dim: int = 128, 
                 with_complexity: bool = False, num_scales: int = 3,
                 freq_emb_global_dim: Optional[int] = None):
        super().__init__()

        self.loss = None   
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(
                DecoderBlock(
                    dim, num_heads, ffn_expansion, bias, LayerNorm_type, 
                    rank=rank, num_experts=num_experts, top_k=top_k, 
                    stage_depth=stage_depth, freq_dim=freq_dim, 
                    complexity_scale=complexity_scale,
                    depth_type=depth_type, rank_type=rank_type, 
                    with_complexity=with_complexity, num_scales=num_scales,
                    freq_emb_global_dim=freq_emb_global_dim
                )
            )

    def forward(self, x, freq_emb=None):
        i = 0
        self.loss = 0
        while i < len(self.layers):
            x, loss = self.layers[i](x, freq_emb)
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
## Spatially-Variant Frequency Embedding (第一个创新点)
class SpatiallyVariantFreqEmbedding(nn.Module):
    """
    增强版频率嵌入：通过多尺度局部池化感知空间变异的频率特征
    这是第一个创新点，保持不变
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
## Main Model: MoCEIR with Cross-Scale Frequency-Aware Routing
class MoCEIR(nn.Module):
    """
    超透镜内窥镜图像重建模型
    第一个创新点：空间变异频率嵌入（SVFE）
    第二个创新点：跨尺度频率感知路由 + 频率域解耦专家
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
                rank=2,
                num_experts=4,
                depth_type="lin",
                stage_depth=[3,2,1],
                rank_type="constant",
                topk=1,
                with_complexity=False,
                complexity_scale="max",
                num_scales=3,  # 第二个创新点：多尺度数量
                ):
        super(MoCEIR, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]
        
        # 保存 freq_emb_global 的维度（dims[-1]，即最大维度）
        freq_emb_global_dim = dims[-1]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        # 第一个创新点：空间变异频率嵌入
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
                    freq_dim=dims[0], with_complexity=with_complexity,
                    rank=ranks[i], num_experts=num_experts, 
                    stage_depth=stage_depth[i], depth_type=depth_type, 
                    rank_type=rank_type, top_k=topk, 
                    complexity_scale=complexity_scale, num_scales=num_scales,
                    freq_emb_global_dim=freq_emb_global_dim
                ),
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
                
        feats = self.patch_embed(x)
        
        self.total_loss = 0
        enc_feats = []
        for i, (block, downsample) in enumerate(self.enc):
            feats = block(feats)
            enc_feats.append(feats)
            feats = downsample(feats)
        
        feats = self.latent(feats)
        # 第一个创新点：空间变异频率嵌入
        freq_emb = self.freq_embed(feats)
        self.last_freq_emb = freq_emb
                
        for i, (upsample, fusion, block) in enumerate(self.dec):
            feats = upsample(feats)
            feats = fusion(torch.cat([feats, enc_feats.pop()], dim=1))
            # 第二个创新点：跨尺度频率感知路由在DecoderBlock内部使用
            feats = block(feats, freq_emb)
            self.total_loss += block.loss

        feats = self.refinement(feats)
        x = self.output(feats) + x

        self.total_loss /= sum(self.num_dec_blocks)
        return x


if __name__ == "__main__":
    # test
    model = MoCEIR(
        rank=2, num_blocks=[4,6,6,8], num_dec_blocks=[2,4,4], levels=4, dim=48, 
        num_refinement_blocks=4, with_complexity=True, complexity_scale="max", 
        stage_depth=[1,1,1], depth_type="constant", rank_type="spread", 
        num_experts=4, topk=1, num_scales=3
    ).cuda()

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
    num_scales = getattr(opt, 'num_scales', 3)  # 第二个创新点参数

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
        num_scales=num_scales,
    )


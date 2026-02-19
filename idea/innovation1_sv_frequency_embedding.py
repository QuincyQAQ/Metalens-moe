"""
创新点1: 空间变异频率诊断 (Spatially-Variant Frequency Diagnosis, SVFD)

核心思想:
- 超透镜的像差具有空间变异特性: 中心区域（近轴）退化较轻，边缘区域（远轴）退化严重
- 通过多尺度局部池化显式建模空间变异，使用高通滤波器提取像差敏感的高频区域
- 生成全局频率诊断嵌入，作为路由器的"诊断信号"

论文表述:
"Unlike conventional image restoration methods that assume spatially-uniform degradation, 
we recognize that metalens aberrations exhibit spatially-variant characteristics due to 
field curvature. We propose a Spatially-Variant Frequency Embedding (SVFE) module that 
explicitly models the frequency distribution differences between central (paraxial) and 
peripheral (off-axis) regions."
"""

from collections import OrderedDict
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal


##########################################################################
## Helper functions
class MySequential(nn.Sequential):
    def forward(self, x1, x2):
        for layer in self:
            if isinstance(layer, nn.Module):
                x1 = layer(x1, x2)
            else:
                x1 = layer(x1, x2)
        return x1

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(
            self._gates.size(0),
            stitched.size(1),
            stitched.size(2),
            stitched.size(3),
            device=stitched.device,
            dtype=stitched.dtype,
        )
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class HighPassConv2d(nn.Module):
    """高通滤波器: 提取高频细节（像差最敏感的部分）"""
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
## 创新点1: 空间变异频率嵌入 (Spatially-Variant Frequency Embedding)
class SpatiallyVariantFreqEmbedding(nn.Module):
    """
    空间变异频率嵌入模块
    
    核心机制:
    1. 物理感知分支: 使用冻结的高通滤波器提取高频细节，这些区域对像差最敏感
    2. 空间变异感知: 通过多尺度局部池化 (1×1, 2×2, 4×4) 捕捉"中心 vs 边缘"的频率分布差异
    3. 特征融合: 将21个空间位置的特征拼接后通过MLP融合，生成全局频率诊断嵌入
    
    输出: (B, dim) 的频率嵌入，作为路由器的"诊断信号"
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
            nn.AdaptiveAvgPool2d(1),  # 全局信息 (1×1 = 1个位置)
            nn.AdaptiveAvgPool2d(2),  # 2×2 区域信息（区分四个象限）(2×2 = 4个位置)
            nn.AdaptiveAvgPool2d(4)   # 4×4 细粒度区域信息 (4×4 = 16个位置)
        ])
        
        # 3. 特征融合 MLP
        # 1*1 + 2*2 + 4*4 = 1 + 4 + 16 = 21 个空间位置
        self.fusion = nn.Sequential(
            nn.Linear(dim * 21, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        """
        Args:
            x: 来自 bottleneck 的特征 (B, C, H, W)
        Returns:
            out: 频率嵌入 (B, dim)
        """
        # 高通滤波提取高频细节
        x = self.high_conv(x)
        
        # 提取多尺度局部特征
        features = []
        for pool in self.local_pools:
            f = pool(x)  # (B, C, h_p, w_p)
            f = f.flatten(1)  # (B, C * h_p * w_p)
            features.append(f)
        
        # 拼接并融合
        combined = torch.cat(features, dim=1)  # (B, C * 21)
        out = self.fusion(combined)  # (B, dim)
        return out


##########################################################################
## 基础模块 (用于构建完整网络)
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
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
        b, c, h, w = x.shape
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
        x = x[:, :, :h, :w]
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

class ModExpert(nn.Module):
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int):
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
## Routing Function (使用SV频率嵌入)
class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max"):
        super(RoutingFunction, self).__init__()
        
        # 基础路由：图像特征和频率嵌入
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
        """
        Args:
            x: 图像特征 (B, C, H, W)
            freq_emb: SV频率嵌入 (B, freq_dim) - 来自 SpatiallyVariantFreqEmbedding
        """
        # 标准路由：融合图像特征和频率嵌入
        logits = self.gate(x) + self.freq_gate(freq_emb)
        
        if self.training:
            loss_imp = self.importance_loss(logits.softmax(dim=-1))
        
        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * loss_imp + 0.5 * loss_load
        else:
            aux_loss = 0
        
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
## Adapter Layer
class AdapterLayer(nn.Module):
    def __init__(self, 
                 dim: int, rank: int, num_experts: int = 4, top_k: int=2, expert_layer: nn.Module=FFTAttention, stage_depth: int=1,
                 depth_type: str="lin", rank_type: str="constant", freq_dim: int=128, 
                 with_complexity: bool=False, complexity_scale: str="min"):
        super().__init__()            
        
        self.tau = 1
        self.loss = None
        self.top_k = top_k
        self.noise_eps = 1e-2
        self.num_experts = num_experts
        
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
        
        self.experts = nn.ModuleList([
            MySequential(*[ModExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, kernel_size=kernel)])
            for idx, (depth, rank, patch, kernel) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes))
        ])
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale
        )
        
    def forward(self, x, freq_emb, shared):
        """
        Args:
            x: 图像特征 (B, C, H, W)
            freq_emb: SV频率嵌入 (B, freq_dim) - 来自 SpatiallyVariantFreqEmbedding
            shared: 共享特征 (B, C, H, W)
        """
        gates, top_k_indices, top_k_values, aux_loss = self.routing(x, freq_emb)
        self.loss = aux_loss
                
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
            expert_outputs = [self.experts[exp](expert_inputs[exp], expert_shared_intputs[exp]) for exp in range(len(self.experts))]
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
## Encoder/Decoder Blocks
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

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, expert_layer, complexity_scale=None,
                 rank=None, num_experts=None, top_k=None, depth_type=None, rank_type=None, stage_depth=None, freq_dim:int=128, 
                 with_complexity: bool=False):
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
            with_complexity=with_complexity, complexity_scale=complexity_scale
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

class EncoderResidualGroup(nn.Module):
    def __init__(self, dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool):
        super().__init__()
        self.loss = None   
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(EncoderBlock(dim, num_heads, ffn_expansion, bias, LayerNorm_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x    

class DecoderResidualGroup(nn.Module):
    def __init__(self, 
                 dim: int, num_heads: List[int], num_blocks: int, ffn_expansion: int, LayerNorm_type: str, bias: bool, complexity_scale=None,
                 rank=None, num_experts=None, expert_layer=None, top_k=None, depth_type=None, stage_depth=None, rank_type=None, freq_dim:int=128, 
                 with_complexity: bool=False):
        super().__init__()
        self.loss = None   
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList([])
        for i in range(num_blocks):
            self.layers.append(DecoderBlock(
                dim, num_heads, ffn_expansion, bias, LayerNorm_type, 
                expert_layer=expert_layer, rank=rank, num_experts=num_experts, top_k=top_k, 
                stage_depth=stage_depth, freq_dim=freq_dim, complexity_scale=complexity_scale,
                depth_type=depth_type, rank_type=rank_type, with_complexity=with_complexity
            ))

    def forward(self, x, freq_emb=None):
        self.loss = 0
        for layer in self.layers:
            x, loss = layer(x, freq_emb)
            self.loss += loss
        return x  


##########################################################################
## Patch Embedding & Resizing
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

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
## 完整网络: MoCE-IR with SV Frequency Embedding
class MoCEIR_SV(nn.Module):
    """
    使用空间变异频率嵌入的MoCE-IR网络
    
    创新点: Spatially-Variant Frequency Embedding (SVFE)
    - 从latent特征提取空间变异的频率嵌入
    - 作为路由器的"诊断信号"，帮助路由器理解不同空间位置的退化模式
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
                expert_layer=FFTAttention,
                with_complexity=False,
                complexity_scale="max",
                ):
        super(MoCEIR_SV, self).__init__()
        
        self.levels = levels
        self.num_blocks = num_blocks
        self.num_dec_blocks = num_dec_blocks
        self.num_refinement_blocks = num_refinement_blocks
        
        dims = [dim*2**i for i in range(levels)]
        ranks = [rank for i in range(levels-1)]

        # -- Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=False)
        
        # -- 创新点1: 空间变异频率嵌入
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
        """
        Args:
            x: 输入图像 (B, C, H, W)
            labels: 标签（可选）
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
        
        # -- 创新点1: 从latent特征提取空间变异频率嵌入 --
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
    # 测试代码
    model = MoCEIR_SV(
        rank=2, 
        num_blocks=[4,6,6,8], 
        num_dec_blocks=[2,4,4], 
        levels=4, 
        dim=48, 
        num_refinement_blocks=4, 
        with_complexity=True, 
        complexity_scale="max", 
        stage_depth=[1,1,1], 
        depth_type="constant", 
        rank_type="spread", 
        num_experts=4, 
        topk=1, 
        expert_layer=FFTAttention
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(1, 3, 224, 224).cuda()
    else:
        x = torch.randn(1, 3, 224, 224)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Frequency embedding shape: {model.last_freq_emb.shape}")
    print(f"Total loss: {model.total_loss}")
    print("创新点1: SV频率嵌入网络测试成功!")


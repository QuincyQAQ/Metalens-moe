"""
创新点2: 多模态物理先验融合专家 (Multi-Modal Physics-Prior Fusion Experts, PGHC)

核心思想:
- 超透镜内窥镜图像的退化受多种物理因素影响: 深度依赖的模糊、组织光谱特性引起的色差、PSF导致的细节损失
- 仅依赖图像特征难以准确理解这些复杂的物理退化机制
- 引入深度图、光谱数据、光学参数等多模态物理先验，设计异构专家，每个专家融合特定的物理先验

论文表述:
"We recognize that metalens endoscope image degradation is governed by multiple coupled 
physical mechanisms: depth-dependent blur, tissue spectral properties causing chromatic 
aberration, and PSF-induced detail loss. Inspired by multi-modal learning and Physics-Informed 
Neural Networks (PINNs), we propose Multi-Modal Physics-Prior Fusion Experts that explicitly 
incorporate external physical priors (depth maps, spectral data, optical parameters) into the 
restoration process."
"""

from collections import OrderedDict
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal


##########################################################################
## Helper functions (与创新点1相同的基础模块)
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
## Layer Norm (与创新点1相同)
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


##########################################################################
## 创新点2: 物理先验编码器

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
## 创新点2: 融合机制

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
## 创新点2: 物理先验融合专家

class HeteroExpert(nn.Module):
    """
    异构专家：支持物理先验融合的专家
    可以处理不同类型的物理先验（深度、光谱、光学参数）
    
    专家类型:
    - "depth_aware_deblurring": 深度感知模糊校正
    - "spectral_chromatic_aberration_correction": 光谱色差校正
    - "psf_guided_detail_restoration": PSF引导细节恢复
    - "standard": 标准专家（不使用物理先验）
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
            if self.expert_type == "depth_aware_deblurring" and "depth_feat" in physics_priors and physics_priors["depth_feat"] is not None:
                x = self.fusion_module(x, physics_priors["depth_feat"])
            elif self.expert_type == "spectral_chromatic_aberration_correction" and "spectral_feat" in physics_priors and physics_priors["spectral_feat"] is not None:
                x = self.fusion_module(x, physics_priors["spectral_feat"])
            elif self.expert_type == "psf_guided_detail_restoration" and "optical_param_feat" in physics_priors and physics_priors["optical_param_feat"] is not None:
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
## 创新点2: 物理先验置信度预测器

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


##########################################################################
## Routing Function (支持物理先验)
class RoutingFunction(nn.Module):
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias: bool = True, complexity_scale: str="max",
                 use_physics_prior: bool=False, H=None, W=None):
        super(RoutingFunction, self).__init__()
        
        self.use_physics_prior = use_physics_prior
        self.H = H
        self.W = W
        
        # 基础路由：图像特征和频率嵌入
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        ) 
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        
        # 物理先验相关模块
        if use_physics_prior:
            # 物理先验编码器（假设有3种物理先验：深度、光谱、光学参数）
            self.depth_encoder = DepthEncoder(in_channels=1, out_channels=dim // 4)
            self.spectral_encoder = SpectralEncoder(in_channels=3, out_channels=dim // 4)
            self.optical_param_encoder = OpticalParamEncoder(in_channels=16, out_channels=dim // 4)
            self.num_physics_priors = 3  # 深度、光谱、光学参数

            # 物理先验置信度预测器
            self.physics_confidence_predictor = PhysicsConfidencePredictor(dim_physics_feat=dim // 4, num_physics_priors=self.num_physics_priors)

            # 专家选择器：融合 SVFE、空间上下文、物理先验特征和置信度，生成专家权重
            self.expert_selector = nn.Sequential(
                nn.Linear(dim + (dim // 4 * self.num_physics_priors) + self.num_physics_priors, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, num_experts)
            )
        else:
            self.depth_encoder = None
            self.spectral_encoder = None
            self.optical_param_encoder = None
            self.num_physics_priors = 0
            self.physics_confidence_predictor = None
            self.expert_selector = None
        
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

    def forward(self, x, freq_emb, raw_physics_priors=None):
        B, C, H, W = x.shape
        
        if self.use_physics_prior:
            # 编码物理先验
            encoded_physics_priors = {}
            encoded_physics_priors_pooled = []
            physics_confidence = None

            if raw_physics_priors is not None:
                if "depth_map" in raw_physics_priors and raw_physics_priors["depth_map"] is not None:
                    depth_feat = self.depth_encoder(raw_physics_priors["depth_map"])
                    encoded_physics_priors["depth_feat"] = depth_feat
                    encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(depth_feat, (1, 1)).squeeze(-1).squeeze(-1))
                else:
                    encoded_physics_priors["depth_feat"] = None
                    
                if "spectral_data" in raw_physics_priors and raw_physics_priors["spectral_data"] is not None:
                    spectral_feat = self.spectral_encoder(raw_physics_priors["spectral_data"], H=H, W=W)
                    encoded_physics_priors["spectral_feat"] = spectral_feat
                    encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(spectral_feat, (1, 1)).squeeze(-1).squeeze(-1))
                else:
                    encoded_physics_priors["spectral_feat"] = None
                    
                if "optical_params" in raw_physics_priors and raw_physics_priors["optical_params"] is not None:
                    optical_param_feat = self.optical_param_encoder(raw_physics_priors["optical_params"], H=H, W=W)
                    encoded_physics_priors["optical_param_feat"] = optical_param_feat
                    encoded_physics_priors_pooled.append(F.adaptive_avg_pool2d(optical_param_feat, (1, 1)).squeeze(-1).squeeze(-1))
                else:
                    encoded_physics_priors["optical_param_feat"] = None
                
                if len(encoded_physics_priors_pooled) > 0:
                    encoded_physics_priors_pooled_cat = torch.cat(encoded_physics_priors_pooled, dim=-1)
                    physics_confidence = self.physics_confidence_predictor(encoded_physics_priors_pooled_cat)
                else:
                    physics_confidence = torch.ones(B, self.num_physics_priors, device=x.device)
            else:
                physics_confidence = torch.ones(B, self.num_physics_priors, device=x.device)

            # 对所有特征进行全局池化，用于专家选择
            freq_diag_pool = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
            
            # 融合所有特征用于专家选择
            fused_feat_for_selector_list = [freq_diag_pool]
            if len(encoded_physics_priors_pooled) > 0:
                fused_feat_for_selector_list.append(torch.cat(encoded_physics_priors_pooled, dim=-1))
            else:
                fused_feat_for_selector_list.append(torch.zeros(B, 0, device=x.device))
            fused_feat_for_selector_list.append(physics_confidence)

            fused_feat_for_selector = torch.cat(fused_feat_for_selector_list, dim=-1)

            # 生成专家选择权重
            logits = self.expert_selector(fused_feat_for_selector) + self.freq_gate(freq_emb)
        else:
            # 标准路由（不使用物理先验）
            logits = self.gate(x) + self.freq_gate(freq_emb)
            encoded_physics_priors = None
        
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
        return gates, top_k_indices, top_k_values, aux_loss, encoded_physics_priors

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
        
        # 使用HeteroExpert（支持物理先验融合）
        if use_physics_prior:
            self.experts = nn.ModuleList([
                MySequential(*[HeteroExpert(dim, rank=rank, func=expert_layer, depth=depth, patch_size=patch, 
                                          kernel_size=kernel, expert_type=expert_type)])
                for idx, (depth, rank, patch, kernel, expert_type) in enumerate(zip(depths, ranks, patch_sizes, kernel_sizes, expert_types))
            ])
        else:
            # 如果不使用物理先验，使用标准专家（这里简化处理，实际可以复用ModExpert）
            raise ValueError("This module requires use_physics_prior=True for innovation 2")
                
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(
            dim, freq_dim, 
            num_experts=num_experts, k=top_k,
            complexity=expert_complexity, use_complexity_bias=with_complexity, complexity_scale=complexity_scale,
            use_physics_prior=use_physics_prior
        )
        
    def forward(self, x, freq_emb, shared, raw_physics_priors=None):
        gates, top_k_indices, top_k_values, aux_loss, encoded_physics_priors = self.routing(x, freq_emb, raw_physics_priors)
        self.loss = aux_loss
                
        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x)
            expert_shared_intputs = dispatcher.dispatch(shared)
            
            # 如果使用物理先验，需要分发物理先验特征
            if self.use_physics_prior and encoded_physics_priors is not None:
                expert_outputs = []
                for exp in range(len(self.experts)):
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
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]
            if self.use_physics_prior and encoded_physics_priors is not None:
                expert_outputs = torch.stack([
                    expert(x, shared, encoded_physics_priors) for expert in selected_experts
                ], dim=1)
            else:
                expert_outputs = torch.stack([expert(x, shared) for expert in selected_experts], dim=1)
            gates = gates.gather(1, top_k_indices)  
            weighted_outputs = gates.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs 
            out = weighted_outputs.sum(dim=1)
            
        out = self.proj_out(out)
        return out


##########################################################################
## 完整网络架构 (简化版，展示核心创新点)
## 注意: 完整网络需要包含Encoder/Decoder等，这里仅展示核心创新点
## 实际使用时，可以参考 innovation1 的完整网络结构

if __name__ == "__main__":
    # 测试物理先验编码器
    print("=" * 60)
    print("测试创新点2: 物理先验融合")
    print("=" * 60)
    
    B, H, W = 2, 64, 64
    dim = 128
    
    # 测试深度编码器
    depth_encoder = DepthEncoder(in_channels=1, out_channels=dim // 4)
    depth_map = torch.randn(B, 1, H, W)
    depth_feat = depth_encoder(depth_map)
    print(f"深度编码器: {depth_map.shape} -> {depth_feat.shape}")
    
    # 测试光谱编码器
    spectral_encoder = SpectralEncoder(in_channels=3, out_channels=dim // 4)
    spectral_data = torch.randn(B, 3, H, W)
    spectral_feat = spectral_encoder(spectral_data, H=H, W=W)
    print(f"光谱编码器: {spectral_data.shape} -> {spectral_feat.shape}")
    
    # 测试光学参数编码器
    optical_param_encoder = OpticalParamEncoder(in_channels=16, out_channels=dim // 4)
    optical_params = torch.randn(B, 16)
    optical_param_feat = optical_param_encoder(optical_params, H=H, W=W)
    print(f"光学参数编码器: {optical_params.shape} -> {optical_param_feat.shape}")
    
    # 测试融合机制
    image_feat = torch.randn(B, dim, H, W)
    gated_fusion = GatedFusion(dim)
    fused_feat = gated_fusion(image_feat, depth_feat)
    print(f"门控融合: {image_feat.shape} + {depth_feat.shape} -> {fused_feat.shape}")
    
    # 测试异构专家
    hetero_expert = HeteroExpert(
        dim=dim, rank=32, func=FFTAttention, depth=2, 
        patch_size=4, kernel_size=3, 
        expert_type="depth_aware_deblurring", fusion_type="gated"
    )
    shared = torch.randn(B, dim, H, W)
    physics_priors = {"depth_feat": depth_feat}
    expert_output = hetero_expert(image_feat, shared, physics_priors)
    print(f"异构专家: {image_feat.shape} -> {expert_output.shape}")
    
    print("\n创新点2: 物理先验融合网络测试成功!")


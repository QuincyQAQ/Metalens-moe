import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 1. Physics-Aware FNO (PA-FNO)
class PhysicsAwareFNO(nn.Module):
    def __init__(self, dim: int, phys_dim: int, num_fourier_modes: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_fourier_modes = num_fourier_modes
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        # Physics-guided spectral weight decomposition
        # This layer will generate modulation parameters for the spectral filters based on phys_emb
        # We generate complex weights for num_fourier_modes * num_fourier_modes
        # For simplicity, let's assume a single spectral layer for now, and a simple modulation
        self.phys_to_spectral_weights = nn.Linear(phys_dim, hidden_dim * num_fourier_modes * 2) # *2 for real and imag parts

        # Learnable spectral filters (base filters)
        # This is for 2D FNO, where num_fourier_modes is for H and num_fourier_modes // 2 + 1 for W
        self.spectral_filter_weights_complex = nn.Parameter(torch.randn(self.hidden_dim, self.num_fourier_modes, self.num_fourier_modes // 2 + 1, dtype=torch.cfloat))
        
        # Re-initialize phys_to_spectral_weights to modulate these complex weights
        self.phys_to_spectral_weights = nn.Linear(phys_dim, self.hidden_dim * self.num_fourier_modes * (self.num_fourier_modes // 2 + 1) * 2)
        # *2 for real and imag parts

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # phys_emb: [B, phys_dim]

        b, c, h, w = x.shape
        x = self.in_proj(x) # [B, hidden_dim, H, W]

        # 1. FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho") # [B, hidden_dim, H, W/2+1]

        # 2. Physics-guided spectral modulation
        # Generate modulation factors from physical embedding
        modulation_factors = self.phys_to_spectral_weights(phys_emb)
        modulation_factors = modulation_factors.view(b, self.hidden_dim, self.num_fourier_modes, self.num_fourier_modes // 2 + 1, 2)
        modulation_factors_complex = torch.complex(modulation_factors[..., 0], modulation_factors[..., 1])

        # Modulate the base spectral filter weights
        modulated_filters = self.spectral_filter_weights_complex.unsqueeze(0) * modulation_factors_complex
        
        # Apply filters to low-frequency modes
        out_freq = torch.zeros_like(x_freq, dtype=torch.cfloat)
        out_freq[:, :, :self.num_fourier_modes, :self.num_fourier_modes // 2 + 1] = \
            x_freq[:, :, :self.num_fourier_modes, :self.num_fourier_modes // 2 + 1] * modulated_filters

        # 3. IFFT
        x = torch.fft.irfft2(out_freq, s=(h, w), dim=(-2, -1), norm="ortho")

        x = self.out_proj(x) # [B, dim, H, W]
        return x

# 2. PA-NOD Expert (Physics-Aware Neural Operator Decomposition Expert)
class PA_NOD_Expert(nn.Module):
    def __init__(self, dim: int, phys_dim: int, num_fourier_modes: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.pa_fno = PhysicsAwareFNO(hidden_dim, phys_dim, num_fourier_modes, hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # phys_emb: [B, phys_dim]
        x = self.in_proj(x)
        x = self.pa_fno(x, phys_emb)
        x = self.out_proj(x)
        return x

# 3. LKE Router (Local Kernel Expert Router) - Reusing from previous task
class LKE_Router(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor):
        # x: [B, C, H, W]
        # phys_emb: [B, phys_dim] - for global routing

        # Global average pooling to get a global feature for routing
        global_feature = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1) # [B, C]
        
        # Concatenate global feature with physical embedding for routing decision
        routing_input = torch.cat([global_feature, phys_emb], dim=-1)

        logits = self.gate(routing_input) # [B, num_experts]
        scores = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

        return scores, top_k_indices, top_k_scores

# 4. PA-MoE Layer (Physics-Aware Mixture-of-Experts Layer)
class PA_MoE_Layer(nn.Module):
    def __init__(self, dim: int, phys_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.dim = dim
        self.phys_dim = phys_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = LKE_Router(dim + phys_dim, num_experts, top_k) # Router input dimension adjusted
        self.experts = nn.ModuleList([PA_NOD_Expert(dim, phys_dim) for _ in range(num_experts)])
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, phys_emb: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scores, indices, top_k_scores = self.router(x, phys_emb)
        final_out = torch.zeros_like(x)

        for i in range(self.num_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x[mask], phys_emb[mask])
                weight = scores[mask, i].view(-1, 1, 1, 1)
                final_out[mask] += expert_out * weight
        return self.out_proj(final_out)

# 5. MoCE-IR Model (Metalens Computational Imaging Reconstruction) with PA-NOD
class MoCE_IR_PA_NOD(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, dim: int = 64, phys_dim: int = 128, num_experts: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.phys_dim = phys_dim

        # Initial convolution
        self.conv_in = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
        self.enc2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)

        # Bottleneck with PA-MoE Layer
        self.bottleneck = PA_MoE_Layer(dim, phys_dim, num_experts)

        # Decoder
        self.up2 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

        # Output convolution
        self.out_conv = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1)

        # Global physical embedding extraction (placeholder)
        self.global_phys_embedder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, phys_dim, kernel_size=1),
            nn.Flatten()
        )

    def _global_phys_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder for extracting physical embedding from feature map
        # In a real scenario, this might come from metadata or a dedicated network branch
        return self.global_phys_embedder(x)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp: [B, C_in, H, W]
        x = self.conv_in(inp)

        # Encoder path
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        # Bottleneck with PA-NOD
        phys_emb = self._global_phys_embedding(d2) # Extract physical embedding at bottleneck
        b = self.bottleneck(d2, phys_emb)

        # Decoder path
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d_dec2 = self.dec2(u2)

        u1 = self.up1(d_dec2)
        u1 = torch.cat([u1, e1], dim=1)
        d_dec1 = self.dec1(u1)

        out = self.out_conv(d_dec1) + inp # Residual connection
        return out

if __name__ == "__main__":
    # Test the model
    batch_size = 1
    in_channels = 3
    image_size = 64
    feature_dim = 64
    physical_embedding_dim = 128
    num_experts = 4

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    model = MoCE_IR_PA_NOD(in_ch=in_channels, out_ch=in_channels, dim=feature_dim, phys_dim=physical_embedding_dim, num_experts=num_experts)

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Test with different physical embeddings (conceptual)
    # phys_emb_test = torch.randn(batch_size, physical_embedding_dim)
    # y_modulated = model(x, phys_emb_test) # In this setup, phys_emb is extracted internally

    # Test PA_NOD_Expert directly
    # pa_nod_expert = PA_NOD_Expert(feature_dim, physical_embedding_dim)
    # x_expert = torch.randn(batch_size, feature_dim, image_size, image_size)
    # phys_emb_expert = torch.randn(batch_size, physical_embedding_dim)
    # y_expert = pa_nod_expert(x_expert, phys_emb_expert)
    # print(f"PA_NOD_Expert output shape: {y_expert.shape}")

    # Test PhysicsAwareFNO directly
    # pa_fno = PhysicsAwareFNO(feature_dim, physical_embedding_dim, num_fourier_modes=16, hidden_dim=256)
    # x_fno = torch.randn(batch_size, feature_dim, image_size, image_size)
    # phys_emb_fno = torch.randn(batch_size, physical_embedding_dim)
    # y_fno = pa_fno(x_fno, phys_emb_fno)
    # print(f"PhysicsAwareFNO output shape: {y_fno.shape}")


def build_model(opt) -> nn.Module:
    """
    从全局配置构建 PA-NOD 模型。
    """
    dim = getattr(opt, "dim", 64)
    num_experts = getattr(opt, "num_exp_blocks", 4)
    phys_dim = getattr(opt, "phys_dim", 128)
    
    return MoCE_IR_PA_NOD(
        in_ch=3,
        out_ch=3,
        dim=dim,
        phys_dim=phys_dim,
        num_experts=num_experts
    )
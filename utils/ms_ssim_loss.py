# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
PyTorch implementation of Multi-Scale SSIM (MSSSIM) loss functions.
Converted from Caffe implementation to PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_kernel_2d(sigma, channels=1, truncate=3.0):
    """
    Create a 2D Gaussian kernel.
    
    Args:
        sigma: Standard deviation of the Gaussian
        channels: Number of channels
        truncate: Truncate the kernel at this many standard deviations
    """
    # Calculate kernel size (should be odd)
    kernel_size = int(2 * truncate * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create coordinate grids
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    
    # Create 1D Gaussian
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    
    # Create 2D Gaussian kernel
    kernel = g[:, None] * g[None, :]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


class MSSSIM(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MSSSIM) loss.
    Computes (1 - MSSSIM) as the loss.
    """
    
    def __init__(self, C1=0.01**2, C2=0.03**2, sigma=(0.5, 1.0, 2.0, 4.0, 8.0), data_range=1.0):
        """
        Args:
            C1: Constant for luminance comparison
            C2: Constant for contrast comparison
            sigma: Tuple of sigma values for different scales
            data_range: Range of input data (default 1.0 for [0, 1] normalized images)
        """
        super(MSSSIM, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.sigma = sigma
        self.data_range = data_range
        self.num_scales = len(sigma)
        
    def _create_gaussian_kernels(self, channels, device):
        """Create Gaussian kernels for all scales."""
        kernels = []
        for s in self.sigma:
            kernel = gaussian_kernel_2d(s, channels)
            kernels.append(kernel.to(device))
        return kernels
    
    def forward(self, pred, target):
        """
        Compute MSSSIM loss: 1 - MSSSIM
        
        Args:
            pred: Predicted image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        B, C, H, W = pred.shape
        
        # Ensure odd dimensions
        if H % 2 == 0 or W % 2 == 0:
            # Pad to make odd
            pred = F.pad(pred, (0, 1, 0, 1), mode='reflect')
            target = F.pad(target, (0, 1, 0, 1), mode='reflect')
            H, W = H + 1, W + 1
        
        # Create Gaussian kernels
        kernels = self._create_gaussian_kernels(C, pred.device)
        
        # Compute SSIM at each scale
        l_values = []
        cs_values = []
        
        for i, kernel in enumerate(kernels):
            # Get kernel size for padding
            kernel_size = kernel.shape[-1]
            padding = kernel_size // 2
            
            # Convolve with Gaussian kernel
            mu_x = F.conv2d(pred, kernel, padding=padding, groups=C)
            mu_y = F.conv2d(target, kernel, padding=padding, groups=C)
            
            mu_x_sq = mu_x ** 2
            mu_y_sq = mu_y ** 2
            mu_xy = mu_x * mu_y
            
            sigma_x_sq = F.conv2d(pred ** 2, kernel, padding=padding, groups=C) - mu_x_sq
            sigma_y_sq = F.conv2d(target ** 2, kernel, padding=padding, groups=C) - mu_y_sq
            sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=C) - mu_xy
            
            # Luminance term
            l = (2 * mu_xy + self.C1) / (mu_x_sq + mu_y_sq + self.C1)
            
            # Contrast-structure term
            cs = (2 * sigma_xy + self.C2) / (sigma_x_sq + sigma_y_sq + self.C2)
            
            l_values.append(l)
            cs_values.append(cs)
        
        # Compute MSSSIM: l at finest scale * product of cs at all scales
        l_finest = l_values[-1]
        cs_product = cs_values[0]
        for cs in cs_values[1:]:
            cs_product = cs_product * cs
        
        msssim = (l_finest * cs_product).mean()
        loss = 1.0 - msssim
        
        return loss


class MSSSIML1(nn.Module):
    """
    Combined Multi-Scale SSIM and L1 loss.
    Computes: alpha * (1 - MSSSIM) + (1 - alpha) * L1
    """
    
    def __init__(self, C1=0.01**2, C2=0.03**2, sigma=(0.5, 1.0, 2.0, 4.0, 8.0), 
                 alpha=0.025, data_range=1.0):
        """
        Args:
            C1: Constant for luminance comparison
            C2: Constant for contrast comparison
            sigma: Tuple of sigma values for different scales
            alpha: Weight for MSSSIM term (default 0.025)
            data_range: Range of input data (default 1.0 for [0, 1] normalized images)
        """
        super(MSSSIML1, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.sigma = sigma
        self.alpha = alpha
        self.data_range = data_range
        self.num_scales = len(sigma)
        self.msssim = MSSSIM(C1=C1, C2=C2, sigma=sigma, data_range=data_range)
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        """
        Compute combined MSSSIM + L1 loss.
        
        Args:
            pred: Predicted image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        msssim_loss = self.msssim(pred, target)
        l1_loss = self.l1_loss(pred, target)
        
        total_loss = self.alpha * msssim_loss + (1 - self.alpha) * l1_loss
        
        return total_loss


class MSSSIML2(nn.Module):
    """
    Combined Multi-Scale SSIM and L2 (MSE) loss.
    Computes: alpha * (1 - MSSSIM) + (1 - alpha) * L2
    """
    
    def __init__(self, C1=0.01**2, C2=0.03**2, sigma=(0.5, 1.0, 2.0, 4.0, 8.0), 
                 alpha=0.1, data_range=1.0):
        """
        Args:
            C1: Constant for luminance comparison
            C2: Constant for contrast comparison
            sigma: Tuple of sigma values for different scales
            alpha: Weight for MSSSIM term (default 0.1)
            data_range: Range of input data (default 1.0 for [0, 1] normalized images)
        """
        super(MSSSIML2, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.sigma = sigma
        self.alpha = alpha
        self.data_range = data_range
        self.num_scales = len(sigma)
        self.msssim = MSSSIM(C1=C1, C2=C2, sigma=sigma, data_range=data_range)
        self.l2_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        """
        Compute combined MSSSIM + L2 loss.
        
        Args:
            pred: Predicted image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            
        Returns:
            Scalar loss value
        """
        msssim_loss = self.msssim(pred, target)
        l2_loss = self.l2_loss(pred, target)
        
        total_loss = self.alpha * msssim_loss + (1 - self.alpha) * l2_loss
        
        return total_loss

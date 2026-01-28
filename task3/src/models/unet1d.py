"""
Conditional 1D U-Net for temporal action sequence denoising.

Adapted from: https://github.com/jannerm/diffuser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) timestep tensor
        Returns:
            emb: (B, dim) positional embedding
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class Downsample1d(nn.Module):
    """1D downsampling with convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling with transposed convolution."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    1D convolutional block with GroupNorm and Mish activation.
    Supports FiLM conditioning.
    """
    
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size,
                padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    Residual block with conditional information via FiLM.
    
    FiLM (Feature-wise Linear Modulation) applies scale and bias
    based on conditioning information.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        
        # FiLM conditioning: predicts scale and bias for each block
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),  # scale and bias
        )
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input tensor
            cond: (B, cond_dim) conditioning tensor
        Returns:
            out: (B, C', T) output tensor
        """
        out = self.blocks[0](x)
        
        # Apply FiLM conditioning
        cond_emb = self.cond_encoder(cond)  # (B, out_channels * 2)
        scale, bias = torch.chunk(cond_emb, 2, dim=-1)  # Each (B, out_channels)
        
        # Apply to feature maps: (B, C, T)
        out = out * scale[:, :, None] + bias[:, :, None]
        
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D U-Net for diffusion model with global conditioning.
    
    Takes noisy action sequences and predicts the noise,
    conditioned on observations via FiLM.
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        """
        Args:
            input_dim: Dimension of input actions
            global_cond_dim: Dimension of global conditioning (flattened obs features)
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: Channel dimensions for downsampling path
            kernel_size: Kernel size for conv layers
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        
        # Timestep embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        
        # Combine timestep and global conditioning
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        
        # Input projection
        self.input_proj = Conv1dBlock(input_dim, down_dims[0], kernel_size, n_groups)
        
        # Downsampling path
        self.down_modules = nn.ModuleList()
        in_channels = down_dims[0]
        
        for idx, dim in enumerate(down_dims[1:]):
            is_last = (idx == len(down_dims) - 2)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    in_channels, dim, cond_dim, kernel_size, n_groups
                ),
                ConditionalResidualBlock1D(
                    dim, dim, cond_dim, kernel_size, n_groups
                ),
                Downsample1d(dim) if not is_last else nn.Identity(),
            ]))
            in_channels = dim
        
        # Middle
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim, kernel_size, n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim, kernel_size, n_groups
            ),
        ])
        
        # Upsampling path (reversed down_dims, skip first which is current resolution)
        self.up_modules = nn.ModuleList()
        
        for idx in range(len(down_dims) - 1):
            dim_out = down_dims[-(idx+1)]  # Current level (from bottom up)
            dim_in = down_dims[-(idx+2)]   # Next level (going up)
            is_last = (idx == len(down_dims) - 2)
            
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim, kernel_size, n_groups  # *2 for skip connection
                ),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim, kernel_size, n_groups
                ),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))
        
        in_channels = down_dims[0]
        
        # Output projection
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) noisy action sequence
            timestep: (B,) diffusion timestep
            global_cond: (B, global_cond_dim) global conditioning
        Returns:
            noise_pred: (B, T, input_dim) predicted noise
        """
        # Permute to (B, C, T) for conv1d
        x = x.permute(0, 2, 1)  # (B, input_dim, T)
        
        # Encode timestep
        timestep_emb = self.diffusion_step_encoder(timestep)  # (B, diffusion_step_embed_dim)
        
        # Combine conditioning
        if global_cond is not None:
            cond = torch.cat([timestep_emb, global_cond], dim=-1)  # (B, cond_dim)
        else:
            cond = timestep_emb
        
        # Input projection
        h0 = self.input_proj(x)  # (B, down_dims[0], T)
        x = h0
        
        # Downsampling with skip connections
        h_list = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            h_list.append(x)  # Save before downsampling
            x = downsample(x)  # Downsample (or identity for last)
        
        # Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, cond)
        
        # Upsampling with skip connections
        for resnet1, resnet2, upsample in self.up_modules:
            # Concatenate with skip connection
            h_skip = h_list.pop()
            x = torch.cat([x, h_skip], dim=1)
            x = resnet1(x, cond)  # Handles channel reduction from 2*dim to dim
            x = resnet2(x, cond)
            x = upsample(x)  # Upsample (or identity for last)
        
        # Output - no concatenation needed, x is already at down_dims[0]
        x = self.final_conv(x)  # (B, input_dim, T)
        
        # Permute back to (B, T, input_dim)
        x = x.permute(0, 2, 1)
        
        return x

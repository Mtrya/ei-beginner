"""
Diffusion Policy for visuomotor control.

Combines vision encoder and conditional UNet for action generation
via denoising diffusion.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .vision_encoder import MultiImageObsEncoder
from .unet1d import ConditionalUnet1D


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for robot control.
    
    Generates action sequences by iteratively denoising random noise,
    conditioned on visual observations.
    """
    
    def __init__(
        self,
        # Observation space
        image_shape: tuple = (3, 96, 96),
        lowdim_dim: int = 2,
        # Action space
        action_dim: int = 2,
        # Horizon parameters
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        # Architecture
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        # Diffusion
        num_train_timesteps: int = 100,
        num_inference_timesteps: int = 10,
        # Normalization
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        # Vision encoder
        self.obs_encoder = MultiImageObsEncoder(
            image_shape=image_shape,
            lowdim_dim=lowdim_dim,
            pretrained=False,
            use_group_norm=True,
        )
        
        obs_feat_dim = self.obs_encoder.feature_dim
        # Flatten observation features across time
        global_cond_dim = obs_feat_dim * n_obs_steps
        
        # Denoising U-Net
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        
        # Noise scheduler for training (DDPM)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",  # Predict noise
        )
        
        # Noise scheduler for inference (DDIM - faster)
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.num_inference_timesteps = num_inference_timesteps
        
        # Action normalization statistics
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))
        if action_stats is not None:
            self.set_action_stats(action_stats)
    
    def set_action_stats(self, stats: Dict[str, torch.Tensor]):
        """Set action normalization statistics."""
        self.action_mean.copy_(stats['mean'])
        self.action_std.copy_(stats['std'])
    
    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action to zero mean and unit variance."""
        return (action - self.action_mean) / self.action_std
    
    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Unnormalize action back to original scale."""
        return action * self.action_std + self.action_mean
    
    def compute_loss(
        self,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diffusion loss for training.
        
        Args:
            obs: Dictionary with 'image' (B, T_obs, C, H, W) and 'agent_pos' (B, T_obs, 2)
            action: (B, T_pred, action_dim) ground truth actions (already normalized)
        
        Returns:
            loss: Scalar MSE loss between predicted and actual noise
        """
        batch_size = action.shape[0]
        
        # Encode observations
        obs_features = self.obs_encoder(obs)  # (B, T_obs, obs_feat_dim)
        # Flatten observation features across time
        global_cond = obs_features.flatten(start_dim=1)  # (B, T_obs * obs_feat_dim)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=action.device,
            dtype=torch.long,
        )
        
        # Sample noise
        noise = torch.randn_like(action)
        
        # Add noise to actions
        noisy_action = self.noise_scheduler.add_noise(
            action, noise, timesteps
        )
        
        # Predict noise
        noise_pred = self.model(
            noisy_action,
            timesteps,
            global_cond=global_cond,
        )
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Predict action sequence using DDIM sampling.
        
        Args:
            obs: Dictionary with 'image' and 'agent_pos'
        
        Returns:
            action: (B, T_pred, action_dim) predicted actions (unnormalized)
        """
        batch_size = obs['image'].shape[0]
        device = obs['image'].device
        
        # Encode observations
        obs_features = self.obs_encoder(obs)  # (B, T_obs, obs_feat_dim)
        global_cond = obs_features.flatten(start_dim=1)  # (B, T_obs * obs_feat_dim)
        
        # Initialize with random noise
        action = torch.randn(
            (batch_size, self.horizon, self.action_dim),
            device=device,
            dtype=torch.float32,
        )
        
        # Set inference timesteps
        self.inference_scheduler.set_timesteps(self.num_inference_timesteps)
        
        # Iterative denoising
        for t in self.inference_scheduler.timesteps:
            # Predict noise
            timesteps = t.unsqueeze(0).repeat(batch_size).to(device)
            noise_pred = self.model(
                action,
                timesteps,
                global_cond=global_cond,
            )
            
            # Denoise
            action = self.inference_scheduler.step(
                noise_pred,
                t,
                action,
            ).prev_sample
        
        # Unnormalize actions
        action = self.unnormalize_action(action)
        
        return action
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            obs: Observation dictionary
            action: Ground truth actions for training (optional)
        
        Returns:
            pred_action: Predicted actions
            loss: Training loss (if action is provided)
        """
        if action is not None:
            # Training mode
            loss = self.compute_loss(obs, action)
            return None, loss
        else:
            # Inference mode
            pred_action = self.predict_action(obs)
            return pred_action, None


def create_diffusion_policy(
    obs_horizon: int = 2,
    action_horizon: int = 8,
    prediction_horizon: int = 16,
    **kwargs
) -> DiffusionPolicy:
    """
    Factory function to create diffusion policy with standard configs.
    
    Args:
        obs_horizon: Number of observation frames
        action_horizon: Number of actions to execute
        prediction_horizon: Total prediction horizon
        **kwargs: Additional arguments for DiffusionPolicy
    
    Returns:
        policy: DiffusionPolicy instance
    """
    return DiffusionPolicy(
        horizon=prediction_horizon,
        n_obs_steps=obs_horizon,
        n_action_steps=action_horizon,
        **kwargs
    )

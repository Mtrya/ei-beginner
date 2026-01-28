"""Vision encoder for image observations using ResNet."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict
from einops import rearrange


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax for extracting spatial features from convolutional layers.
    Maps feature maps to keypoint coordinates.
    """
    
    def __init__(self, height: int, width: int, channel: int):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        
        # Create coordinate grid
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing='ij'
        )
        pos_x = pos_x.reshape(height * width)
        pos_y = pos_y.reshape(height * width)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W)
        Returns:
            keypoints: (B, C*2) - x,y coordinates for each channel
        """
        batch_size = features.shape[0]
        
        # Flatten spatial dimensions
        features = rearrange(features, 'b c h w -> b c (h w)')
        
        # Compute softmax over spatial locations
        attention = torch.softmax(features, dim=-1)  # (B, C, H*W)
        
        # Compute expected positions
        expected_x = torch.sum(self.pos_x * attention, dim=-1)  # (B, C)
        expected_y = torch.sum(self.pos_y * attention, dim=-1)  # (B, C)
        
        # Interleave x and y: (B, C*2)
        output = torch.cat([expected_x, expected_y], dim=-1)
        
        return output


class VisionEncoder(nn.Module):
    """
    Vision encoder using ResNet18 backbone with spatial softmax pooling.
    
    Takes image observations and produces a fixed-size embedding.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        pretrained: bool = False,
        use_group_norm: bool = True,
        spatial_softmax: bool = True,
    ):
        super().__init__()
        
        # Load ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv if input channels != 3
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace BatchNorm with GroupNorm for stability
        if use_group_norm:
            resnet = self._replace_batchnorm_with_groupnorm(resnet)
        
        # Remove global average pooling and fc layer
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # Output from ResNet18 layer4: 512 channels, 3x3 spatial for 96x96 input
        self.out_channels = 512
        self.spatial_size = 3  # For 96x96 input after ResNet18
        
        # Spatial softmax or flatten
        if spatial_softmax:
            self.pool = SpatialSoftmax(
                height=self.spatial_size,
                width=self.spatial_size,
                channel=self.out_channels
            )
            self.feature_dim = self.out_channels * 2  # x,y per channel
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = self.out_channels
    
    def _replace_batchnorm_with_groupnorm(self, module: nn.Module) -> nn.Module:
        """Replace all BatchNorm layers with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                # Replace with GroupNorm (8 groups)
                num_channels = child.num_features
                num_groups = min(32, num_channels)  # Ensure divisibility
                setattr(
                    module, name,
                    nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                )
            else:
                self._replace_batchnorm_with_groupnorm(child)
        return module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B*T, C, H, W) image tensor
        Returns:
            features: (B, feature_dim) or (B*T, feature_dim)
        """
        features = self.backbone(x)  # (B, 512, 3, 3)
        features = self.pool(features)  # (B, 512*2) or (B, 512)
        
        if not isinstance(self.pool, SpatialSoftmax):
            features = features.flatten(1)
        
        return features
    
    def output_shape(self) -> tuple:
        """Return output feature dimension."""
        return (self.feature_dim,)


class MultiImageObsEncoder(nn.Module):
    """
    Encoder for multiple image observations and low-dim state.
    
    Encodes image observations from multiple timesteps and concatenates
    with low-dim observations (e.g., agent position).
    """
    
    def __init__(
        self,
        image_shape: tuple = (3, 96, 96),
        lowdim_dim: int = 2,
        pretrained: bool = False,
        use_group_norm: bool = True,
    ):
        super().__init__()
        
        self.image_shape = image_shape
        self.lowdim_dim = lowdim_dim
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            input_channels=image_shape[0],
            pretrained=pretrained,
            use_group_norm=use_group_norm,
            spatial_softmax=True,
        )
        
        # Output dimension
        self.feature_dim = self.vision_encoder.feature_dim + lowdim_dim
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            obs_dict: Dictionary with:
                - 'image': (B, T_obs, C, H, W)
                - 'agent_pos': (B, T_obs, lowdim_dim)
        Returns:
            features: (B, T_obs, feature_dim)
        """
        images = obs_dict['image']  # (B, T_obs, C, H, W)
        agent_pos = obs_dict['agent_pos']  # (B, T_obs, lowdim_dim)
        
        batch_size, T_obs = images.shape[:2]
        
        # Flatten batch and time for vision encoder
        images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
        
        # Encode images
        img_features = self.vision_encoder(images_flat)  # (B*T_obs, img_feat_dim)
        
        # Reshape back
        img_features = rearrange(
            img_features, '(b t) d -> b t d', b=batch_size, t=T_obs
        )
        
        # Concatenate with low-dim observations
        features = torch.cat([img_features, agent_pos], dim=-1)  # (B, T_obs, feature_dim)
        
        return features
    
    def output_shape(self) -> tuple:
        """Return output feature dimension."""
        return (self.feature_dim,)

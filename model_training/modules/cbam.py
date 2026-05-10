"""
CBAM: Convolutional Block Attention Module
for night-time pedestrian detection.

Channel attention identifies illuminated regions.
Spatial attention focuses on pedestrian shapes.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention: highlights which feature channels are most relevant."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        mx = self.max_pool(x).view(b, c)
        attn = (self.mlp(avg) + self.mlp(mx)).sigmoid().view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention: highlights WHERE in the feature map to focus."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        attn = torch.cat([avg, mx], dim=1)
        attn = self.conv(attn).sigmoid()
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Applies channel attention then spatial attention sequentially.
    Input:  [B, C, H, W]
    Output: [B, C, H, W]  (same shape, attended features)

    Inserted after backbone stages P3, P4, P5 in YOLOv8.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

"""
BiFPN: Weighted Bidirectional Feature Pyramid Network.

Replaces YOLOv8 PANet neck with weighted bidirectional cross-scale
connections for better small-object multi-scale fusion under low light.
"""

import torch
import torch.nn as nn


class BiFPNConv(nn.Module):
    """Depthwise-separable conv block used in BiFPN for efficiency."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BiFPNFusion(nn.Module):
    """
    Single BiFPN fusion round with learned fast-normalised weights.

    Top-down pathway:  P5_td → P4_td → P3_td
    Bottom-up pathway: P3_out → P4_out → P5_out
    """

    def __init__(self, channels: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = 1e-4

        # Top-down weights (per level, from higher level + skip)
        # w1 for current level skip, w2 for upsampled higher level
        self.td_weights = nn.Parameter(torch.ones(num_levels - 1, 2))

        # Bottom-up weights
        # w1 for original, w2 for td, w3 for downsampled lower level
        self.bu_weights = nn.Parameter(torch.ones(num_levels - 1, 3))

        # Post-fusion convs
        self.td_convs = nn.ModuleList([BiFPNConv(channels) for _ in range(num_levels - 1)])
        self.bu_convs = nn.ModuleList([BiFPNConv(channels) for _ in range(num_levels - 1)])
        self.p_out_convs = nn.ModuleList([BiFPNConv(channels) for _ in range(num_levels)])

    def _fast_norm(self, w: torch.Tensor) -> torch.Tensor:
        """Fast normalised fusion: w / (sum(w) + epsilon) via ReLU for stability."""
        return w.relu() / (w.relu().sum(dim=0, keepdim=True) + self.epsilon)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # features ordered: [P3, P4, P5] — low to high resolution
        p3, p4, p5 = features

        # --- Top-down ---
        # P5_td = P5 (no fusion needed for top level)
        p5_td = p5

        # P4_td = Conv( w1*P4 + w2*Upsample(P5_td) / (w1+w2+eps) )
        td_w = self._fast_norm(self.td_weights[1])  # for P4
        p5_up = nn.functional.interpolate(p5_td, size=p4.shape[2:], mode='nearest')
        p4_td = self.td_convs[1](td_w[0] * p4 + td_w[1] * p5_up)

        # P3_td = Conv( w1*P3 + w2*Upsample(P4_td) / (w1+w2+eps) )
        td_w = self._fast_norm(self.td_weights[0])  # for P3
        p4_up = nn.functional.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_td = self.td_convs[0](td_w[0] * p3 + td_w[1] * p4_up)

        # --- Bottom-up ---
        # P3_out = P3_td
        p3_out = self.p_out_convs[0](p3_td)

        # P4_out = Conv( w1*P4 + w2*P4_td + w3*Down(P3_out) / (w1+w2+w3+eps) )
        bu_w = self._fast_norm(self.bu_weights[0])
        p3_down = nn.functional.max_pool2d(p3_out, 2)
        p4_out = self.bu_convs[0](
            bu_w[0] * p4 + bu_w[1] * p4_td + bu_w[2] * p3_down
        )

        # P5_out = Conv( w1*P5 + w2*P5_td + w3*Down(P4_out) / (w1+w2+w3+eps) )
        bu_w = self._fast_norm(self.bu_weights[1])
        p4_down = nn.functional.max_pool2d(p4_out, 2)
        p5_out = self.bu_convs[1](
            bu_w[0] * p5 + bu_w[1] * p5_td + bu_w[2] * p4_down
        )

        return [p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    """
    Complete BiFPN neck — weighted bidirectional FPN repeated 3 times.

    Input:  [P3, P4, P5] backbone features at strides 8, 16, 32
    Output: [P3, P4, P5] enhanced features, same shapes
    """

    def __init__(self, channels: int = 256, backbone_channels: list[int] | None = None, num_repeats: int = 3):
        super().__init__()
        self.num_repeats = num_repeats

        if backbone_channels is None:
            backbone_channels = [128, 256, 512]

        self.channel_align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            for c in backbone_channels
        ])

        self.fpn_layers = nn.ModuleList([
            BiFPNFusion(channels) for _ in range(num_repeats)
        ])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = features
        p3 = self.channel_align[0](p3)
        p4 = self.channel_align[1](p4)
        p5 = self.channel_align[2](p5)

        feats = [p3, p4, p5]
        for fpn_layer in self.fpn_layers:
            feats = fpn_layer(feats)

        return feats

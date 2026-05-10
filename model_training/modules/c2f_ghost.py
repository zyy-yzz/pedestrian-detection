"""
C2f-Ghost: Replace C2f blocks with Ghost convolutions for YOLOv8 backbone.

Ghost module generates half the features via cheap linear operations,
reducing parameters by ~30% while maintaining receptive field.
"""

import torch
import torch.nn as nn


class GhostConv(nn.Module):
    """
    Ghost Convolution: generates intrinsic features via regular conv,
    then produces ghost features via cheap depthwise conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        ratio: int = 2,
    ):
        super().__init__()
        self.intrinsic_channels = out_channels // ratio
        self.ghost_channels = out_channels - self.intrinsic_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.intrinsic_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.intrinsic_channels),
            nn.SiLU(inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.intrinsic_channels, self.ghost_channels, 3, 1, 1,
                      groups=self.intrinsic_channels, bias=False),
            nn.BatchNorm2d(self.ghost_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intrinsic = self.primary_conv(x)
        ghost = self.cheap_operation(intrinsic)
        return torch.cat([intrinsic, ghost], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost bottleneck block for C2f-Ghost."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels * 2
        self.ghost1 = GhostConv(channels, hidden, kernel_size=3)
        self.ghost2 = GhostConv(hidden, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ghost2(self.ghost1(x))


class C2fGhost(nn.Module):
    """
    C2f-Ghost block: CSP bottleneck with Ghost convolutions.

    Drop-in replacement for ultralytics C2f module.
    """

    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(out_channels * (2 + n), out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.ghost_bottlenecks = nn.ModuleList(
            [GhostBottleneck(out_channels) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for bottleneck in self.ghost_bottlenecks:
            y.append(bottleneck(y[-1]))
        return self.cv2(torch.cat(y, dim=1))

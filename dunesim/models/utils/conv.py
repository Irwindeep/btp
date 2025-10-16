import torch
import torch.nn as nn


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Conv3dBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class UpSample(nn.Module):
    def __init__(self, channels: int, scale: int = 2) -> None:
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(input))

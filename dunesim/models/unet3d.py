import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_channels: int,
    ) -> None:
        """
        Parameters:
            in_channels: number of input time channels i.e., memory
            out_channels: number of output time channels i.e., future
        """
        super(UNet3D, self).__init__()

        self.initial = nn.Conv3d(in_channels, down_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.bottleneck = nn.Identity()
        self.up_blocks = nn.Identity()
        self.final = nn.Identity()

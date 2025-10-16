import torch
import torch.nn as nn

from typing import List, Tuple
from dunesim.models.utils import Conv3dBlock


_EncoderOutput = Tuple[torch.Tensor, List[torch.Tensor]]


class Encoder3D(nn.Module):
    def __init__(self, channels: List[int]) -> None:
        super(Encoder3D, self).__init__()

        in_channels, out_channels = channels[:-1], channels[1:]

        # convolutional blocks
        self.conv_blocks = nn.ModuleList()

        for in_chann, out_chann in zip(in_channels, out_channels):
            self.conv_blocks.append(
                nn.Sequential(
                    Conv3dBlock(in_chann, out_chann),
                    Conv3dBlock(out_chann, out_chann),
                )
            )

        # downsampling blocks (downsample channel, height and wodth)
        self.down_blocks = nn.ModuleList()

        for out_chann in out_channels:
            self.down_blocks.append(
                nn.Conv3d(out_chann, out_chann, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )

    def forward(self, input: torch.Tensor) -> _EncoderOutput:
        output, outputs = input, []
        for conv, down in zip(self.conv_blocks, self.down_blocks):
            output = conv(output)
            outputs.append(output)
            output = down(output)

        return output, outputs


class Decoder3D(nn.Module):
    def __init__(self, encoder_channels: List[int]) -> None:
        super(Decoder3D, self).__init__()
        channels = [2 * ch for ch in encoder_channels[::-1]]
        in_channels, out_channels = channels[:-1], channels[1:]

        # upsampling blocks
        self.up_blocks = nn.ModuleList()

        for in_chann in in_channels:
            ch = in_chann // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=(1, 2, 2)),
                    nn.Conv3d(ch, ch, kernel_size=3, padding=1),
                )
            )

        # convolutional blocks
        self.conv_blocks = nn.ModuleList()

        for in_chann, out_chann in zip(in_channels, out_channels):
            self.conv_blocks.append(
                nn.Sequential(
                    Conv3dBlock(in_chann, out_chann // 2),
                    Conv3dBlock(out_chann // 2, out_chann // 2),
                )
            )

    def forward(self, decoder_input: _EncoderOutput) -> torch.Tensor:
        output, outputs = decoder_input

        for up_block, conv in zip(self.up_blocks, self.conv_blocks):
            output = up_block(output)
            output = conv(torch.cat([output, outputs.pop()], dim=1))

        return output


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down_channels: int,
        channel_multipliers: List[int] = [1, 2, 2, 4],
    ) -> None:
        """
        Parameters:
            in_channels: number of input time channels i.e., memory
            out_channels: number of output time channels i.e., future
        """
        super(UNet3D, self).__init__()

        self.initial = nn.Conv3d(in_channels, down_channels, kernel_size=3, padding=1)

        encoder_channels = [down_channels * i for i in channel_multipliers]
        self.encoder = Encoder3D(channels=encoder_channels)
        self.bottleneck = nn.Sequential(
            Conv3dBlock(encoder_channels[-1], 2 * encoder_channels[-1]),
            Conv3dBlock(2 * encoder_channels[-1], encoder_channels[-1]),
        )
        self.decoder = Decoder3D(encoder_channels=encoder_channels)

        # use batchnorm and relu for now as heightmap must always be positive
        self.final = Conv3dBlock(down_channels, out_channels)

    def forward(self, input: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        input = torch.cat([input, aux], dim=2)

        output = self.initial(input)
        output, outputs = self.encoder(output)
        output = self.bottleneck(output)
        output = self.decoder((output, outputs))
        output = self.final(output)

        return output

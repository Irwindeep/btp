from typing import List
import torch
import torch.nn as nn

from functools import partial

from dunesim.models.utils import AttentionBlock, ResidualBlock, timestep_embd, UpSample


class UNet(nn.Module):
    def __init__(
        self,
        num_channels: int,
        down_channs: int,
        dropout: float,
        memory: int,
        current: int,
    ) -> None:
        super(UNet, self).__init__()

        ResBlock = partial(
            ResidualBlock, dropout=dropout, time_embd_dim=down_channs * 4
        )

        in_channels = num_channels * (memory + current)
        in_channels += 4  # auxilary info - vegetation, bedrock, wind

        down_blocks: List[nn.Module] = [
            nn.Conv2d(in_channels, down_channs, kernel_size=3, padding=1),
        ]

        channels = [down_channs * n for n in [1, 2, 4, 4]]
        in_channs, out_channs = channels[:-1], channels[1:]

        channel_size = [down_channs]
        for i, (in_chann, out_chann) in enumerate(zip(in_channs, out_channs)):
            down_blocks.append(ResBlock(in_chann, out_chann))
            if i == 1:
                down_blocks.append(AttentionBlock(channels=out_chann))

            down_blocks.append(ResBlock(out_chann, out_chann))
            if i == 1:
                down_blocks.append(AttentionBlock(channels=out_chann))

            if i != len(out_channs) - 1:
                down_blocks.append(
                    nn.Conv2d(out_chann, out_chann, kernel_size=3, stride=2, padding=1)
                )
                channel_size += 3 * [out_chann]
            else:
                channel_size += 2 * [out_chann]

        self.down = nn.ModuleList(down_blocks)

        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResBlock(out_channs[-1], out_channs[-1]))
        self.bottleneck.append(AttentionBlock(out_channs[-1]))
        self.bottleneck.append(ResBlock(out_channs[-1], out_channs[-1]))

        self.up_blocks = nn.ModuleList()
        for i, (in_chann, out_chann) in enumerate(
            zip(out_channs[::-1], in_channs[::-1])
        ):
            for j in range(3):
                res_chann = in_chann if j == 0 else out_chann
                self.up_blocks.append(
                    ResBlock(res_chann + channel_size.pop(), out_chann)
                )
                if i == len(out_channs) - 2:
                    self.up_blocks.append(AttentionBlock(out_chann))

            if i != len(out_channs) - 1:
                self.up_blocks.append(UpSample(out_chann))

        self.norm = nn.GroupNorm(num_channels=down_channs, num_groups=32)
        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.GroupNorm(num_channels=down_channs, num_groups=32),
            nn.ReLU(),
            nn.Conv2d(down_channs, num_channels * current, kernel_size=3, padding=1),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(down_channs, down_channs * 4),
            nn.ReLU(),
            nn.Linear(down_channs * 4, down_channs * 4),
            nn.ReLU(),
        )
        self.down_channs = down_channs

    def forward(
        self,
        input: torch.Tensor,
        timesteps: torch.Tensor,
        past: torch.Tensor,
        auxilliary: torch.Tensor,
    ) -> torch.Tensor:
        time_embd = timestep_embd(timesteps, self.down_channs)
        time_embd = self.time_embedding(time_embd)

        output = torch.cat([input, past, auxilliary], dim=1)

        outputs = []
        for module in self.down:
            if isinstance(module, ResidualBlock):
                output = module(output, time_embd)
            else:
                output = module(output)

            outputs.append(output)

            if isinstance(module, AttentionBlock):
                outputs.pop()

        for module in self.bottleneck:
            if isinstance(module, ResidualBlock):
                output = module(output, time_embd)
            else:
                output = module(output)

        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                output = module(torch.cat([output, outputs.pop()], dim=1), time_embd)
            else:
                output = module(output)

        output = self.out(output)
        return output

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        time_embd_dim: int | None = None,
        num_groups: int = 32,
    ) -> None:
        super(ResidualBlock, self).__init__()

        self.dropout = nn.Dropout2d(p=dropout)
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_proj = nn.Identity()
        if time_embd_dim is not None:
            self.time_proj = nn.Linear(time_embd_dim, out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        input: torch.Tensor,
        time_embd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.conv1(input)

        if time_embd is not None:
            time_embd = self.time_proj(time_embd).unsqueeze(-1).unsqueeze(-1)
            output = output + time_embd

        output = self.dropout(output)
        output = self.conv2(output) + self.shortcut(input)

        return output

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 32) -> None:
        super(AttentionBlock, self).__init__()

        self.Q = nn.Conv2d(channels, channels, kernel_size=1)
        self.K = nn.Conv2d(channels, channels, kernel_size=1)
        self.V = nn.Conv2d(channels, channels, kernel_size=1)

        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        self.d = channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm(input)
        q, k, v = self.Q(x), self.K(x), self.V(x)

        energy = torch.einsum("abcd,abef->acdef", q, k) / (self.d**0.5)

        N, W, *_ = energy.size()
        energy = nn.functional.softmax(energy.view(N, W, W, W * W), dim=-1)
        energy = energy.view(N, *[W] * 4)  # [N, W, W, W, W]

        output = torch.einsum("abcde,afde->afbc", energy, v)
        return output

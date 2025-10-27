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


class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(LinearAttentionBlock, self).__init__()

        self.W_l = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.W_g = nn.Conv2d(in_features, in_features, kernel_size=1)

        self.gate_c = nn.Sequential(
            nn.Conv1d(2 * in_features, in_features, kernel_size=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
        )

        self.gate_s = nn.Sequential(
            nn.Conv2d(2 * in_features, in_features // 4, kernel_size=1),
            nn.InstanceNorm2d(in_features // 4),
            nn.ReLU(),
            nn.Conv2d(
                in_features // 4, in_features // 4, kernel_size=3, padding=4, dilation=4
            ),
            nn.InstanceNorm2d(in_features // 4),
            nn.ReLU(),
            nn.Conv2d(in_features // 4, 1, kernel_size=1),
        )

    def forward(self, input: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = input.size()
        c = torch.zeros((N, C, H, W), device=input.device)

        for t in range(T):
            layer_output = input[:, t]
            l_ = self.W_l(layer_output)
            g_ = self.W_g(g)
            cat_f = nn.functional.relu(torch.cat([l_, g_], dim=1))

            avg_f = nn.functional.adaptive_avg_pool2d(cat_f, 1).view(N, 2 * C, 1)
            avg_f = self.gate_c(avg_f).view(N, C, 1, 1)

            s_gate = torch.sigmoid(self.gate_s(cat_f))
            c = c + layer_output * torch.sigmoid(avg_f * s_gate)

        return c

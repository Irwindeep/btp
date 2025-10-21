import math
import torch

from dunesim.models.utils.attn import AttentionBlock
from dunesim.models.utils.residual import ResidualBlock
from dunesim.models.utils.conv import Conv3dBlock, UpSample
from dunesim.models.utils.convlstm import ConvLSTMCell, ConvLSTM


def timestep_embd(timesteps: torch.Tensor, embd_dim: int) -> torch.Tensor:
    half_dim = embd_dim // 2
    embd = math.log(10000.0) / (half_dim - 1)
    embd = torch.exp(
        -embd * torch.arange(half_dim, dtype=torch.float, device=timesteps.device)
    )
    embd = timesteps.float().unsqueeze(1) * embd.unsqueeze(0)

    embd = torch.cat([torch.sin(embd), torch.cos(embd)], dim=1)
    if embd_dim % 2 == 1:
        embd = torch.nn.ZeroPad2d((0, 1, 0, 0))(embd)

    return embd


__all__ = [
    "AttentionBlock",
    "Conv3dBlock",
    "ConvLSTMCell",
    "ConvLSTM",
    "ResidualBlock",
    "timestep_embd",
    "UpSample",
]

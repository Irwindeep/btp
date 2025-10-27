import torch
import torch.nn as nn

from dunesim.models import UNet3D
from dunesim.models.utils import ConvLSTM, Conv3dBlock


class ATPPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        memory: int = 5,
        future: int = 5,
    ) -> None:
        super(ATPPNet, self).__init__()

        self.encoder = ConvLSTM(input_dim + 4, hidden_dim, 3, num_layers=num_layers)
        self.decoder = ConvLSTM(input_dim + 4, hidden_dim, 3, num_layers=num_layers)

        self.unet3d = UNet3D(in_channels=memory, out_channels=future, down_channels=32)
        self.out_conv = nn.Conv2d(hidden_dim, output_dim + 4, kernel_size=1)
        self.aux_embedding_s = Conv3dBlock(1, memory)
        self.aux_embedding_t = Conv3dBlock(1, future)

        self.future = future

    def forward(
        self,
        src: torch.Tensor,
        aux: torch.Tensor,
        trg: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        pred_len = self.future
        ch = src.size(2)
        if len(aux.size()) == 4:
            aux = aux.unsqueeze(1)

        aux_embd = self.aux_embedding_s(aux)
        input = torch.cat([src, aux_embd], dim=2)
        _, (h, c) = self.encoder(input)

        if trg is not None:
            aux_embd = self.aux_embedding_t(aux)
            trg = torch.cat([trg, aux_embd], dim=2)

        input_t = input[:, -1].unsqueeze(1)
        outputs = []
        for t in range(pred_len):
            output, (h, c) = self.decoder(input_t, (h, c))

            output = self.out_conv(output.squeeze(1))
            outputs.append(output)

            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_t = trg[:, t].unsqueeze(1)
            else:
                input_t = output.detach().unsqueeze(1)

        output = torch.stack(outputs, dim=1)[:, :, :ch, :, :]
        output = output + self.unet3d(src, aux)
        return output

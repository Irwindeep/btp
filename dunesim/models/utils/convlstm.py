import torch
import torch.nn as nn

from typing import List, Tuple, cast


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
        peephole: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )
        self.bias = bias
        self.peephole = peephole

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        if self.peephole:
            self.w_ci = nn.Parameter(torch.zeros(self.hidden_dim))
            self.w_cf = nn.Parameter(torch.zeros(self.hidden_dim))
            self.w_co = nn.Parameter(torch.zeros(self.hidden_dim))

        self._initialize_parameters()

    def _initialize_parameters(self):
        if self.peephole:
            nn.init.zeros_(self.w_ci)
            nn.init.zeros_(self.w_cf)
            nn.init.zeros_(self.w_co)

    def forward(
        self,
        input: torch.Tensor,
        h_prev: torch.Tensor | None,
        c_prev: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, _, H, W = input.shape

        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, H, W, device=input.device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.hidden_dim, H, W, device=input.device)

        combined = torch.cat([input, h_prev], dim=1)
        conv_out = self.conv(combined)

        i_conv, f_conv, g_conv, o_conv = torch.split(conv_out, self.hidden_dim, dim=1)

        if self.peephole:
            w_ci = self.w_ci.view(1, -1, 1, 1)
            w_cf = self.w_cf.view(1, -1, 1, 1)
            w_co = self.w_co.view(1, -1, 1, 1)

            i = torch.sigmoid(i_conv + w_ci * c_prev)
            f = torch.sigmoid(f_conv + w_cf * c_prev)
        else:
            i = torch.sigmoid(i_conv)
            f = torch.sigmoid(f_conv)

        g = torch.tanh(g_conv)

        c_next = f * c_prev + i * g

        if self.peephole:
            o = torch.sigmoid(o_conv + w_co * c_next)  # type: ignore
        else:
            o = torch.sigmoid(o_conv)

        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else torch.float32

        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        bias: bool = True,
        peephole: bool = True,
    ) -> None:
        super().__init__()

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_channs = input_dim if i == 0 else hidden_dim

            cell = ConvLSTMCell(
                input_dim=in_channs,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias,
                peephole=peephole,
            )
            self.cells.append(cell)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        h_list, c_list = [], []

        for cell in self.cells:
            cell = cast(ConvLSTMCell, cell)
            h, c = cell.init_hidden(batch_size, height, width, device, dtype)
            h_list.append(h)
            c_list.append(c)

        return h_list, c_list

    def forward(
        self,
        input: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        N, T, _, H, W = input.size()
        seq = input.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            h_list, c_list = self._init_hidden(N, H, W, device=input.device)
        else:
            h_in, c_in = hidden_state
            h_list = [h_in[i] for i in range(self.num_layers)]
            c_list = [c_in[i] for i in range(self.num_layers)]

        output_inner = []
        for t in range(T):
            x_t = seq[t]

            for i, cell in enumerate(self.cells):
                h_next, c_next = cell(x_t, h_list[i], c_list[i])

                h_list[i] = h_next
                c_list[i] = c_next

                x_t = h_next

            output_inner.append(x_t)

        output_seq = torch.stack(output_inner, dim=0)
        output_seq = output_seq.permute(1, 0, 2, 3, 4)

        h_stack = torch.stack(h_list, dim=1)
        c_stack = torch.stack(c_list, dim=1)

        return output_seq, (h_stack, c_stack)

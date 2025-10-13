import torch
import torch.nn as nn

from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_STEPS = 1000
BETAS = torch.linspace(1e-4, 0.02, NUM_STEPS).to(DEVICE)
ALPHAS = torch.cumprod(1 - BETAS.flip(0), 0).flip(0).to(DEVICE)


@torch.no_grad()
def ddpm_sampler(
    model: nn.Module,
    T: int,
    past: torch.Tensor,
    aux: torch.Tensor,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    N, *_ = shape
    x = torch.randn(shape, device=DEVICE)

    for t in reversed(range(T)):
        timesteps = torch.full((N,), t, dtype=torch.long, device=DEVICE)
        epsilon = model(x, timesteps, past, aux)

        alpha = ALPHAS[t]
        x0_pred = (x - ((1 - alpha) ** 0.5) * epsilon) / (alpha**0.5)

        if t > 0:
            alpha_prev = ALPHAS[t - 1]
        else:
            alpha_prev = torch.tensor(1.0, device=DEVICE)

        coef1 = ((alpha_prev**0.5) * BETAS[t]) / (1 - alpha)
        coef2 = ((alpha**0.5) * (1 - alpha_prev)) / (1 - alpha)

        mu = coef1 * x0_pred + coef2 * x
        var = BETAS[t] * (1 - alpha_prev) / (1 - alpha)

        if t == 0:
            x = mu
        else:
            noise = torch.randn_like(x)
            x = mu + (var**0.5) * noise

    return x


@torch.no_grad()
def ddim_sampler(
    model: nn.Module,
    T: int,
    steps: int,
    past: torch.Tensor,
    aux: torch.Tensor,
    shape: Tuple[int, ...],
    eta: float = 0.0,
) -> torch.Tensor:
    N, *_ = shape

    timesteps = torch.linspace(0, T - 1, steps, dtype=torch.long, device=DEVICE)
    x = torch.randn(shape, device=DEVICE)

    for i, t in enumerate(timesteps):
        t_ = int(t.item())
        epsilon = model(
            x,
            torch.full((N,), t_, dtype=torch.long, device=DEVICE),
            past,
            aux,
        )

        alpha = ALPHAS[t]
        x0_pred = (x - ((1 - alpha) ** 0.5) * epsilon) / (alpha**0.5)

        if i == len(timesteps) - 1:
            s_idx = 0
        else:
            s_idx = int(timesteps[i + 1].item())

        alpha_s = ALPHAS[s_idx]
        if t_ == s_idx:
            sigma = 0.0
        else:
            sigma = (
                eta
                * ((1 - alpha_s) / (1 - alpha) * (1 - alpha / (alpha_s + 1e-8))) ** 0.5
            )

        pred_eps_dir = (x - (alpha**0.5) * x0_pred) / ((1 - alpha) ** 0.5)
        x_mean = (alpha_s**0.5) * x0_pred + ((1 - alpha_s) ** 0.5) * pred_eps_dir

        if sigma == 0:
            x = x_mean
        else:
            noise = torch.randn_like(x)
            x = x_mean + sigma * noise

    return x

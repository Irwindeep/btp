import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_STEPS = 1000
BETAS = torch.linspace(1e-4, 0.02, NUM_STEPS).to(DEVICE)
ALPHAS = torch.cumprod(1 - BETAS.flip(0), 0).flip(0).to(DEVICE)


def val_epoch(
    model: nn.Module, val_loader: DataLoader[torch.Tensor], loss_fn: nn.Module
) -> float:
    model.eval()

    val_loss, num_batches = 0.0, len(val_loader)
    with torch.no_grad():
        for curr, past, aux in val_loader:
            N, T, C, H, W = curr.size()
            curr = curr.reshape(N, T * C, H, W).to(DEVICE)
            past = past.reshape(N, T * C, H, W).to(DEVICE)
            aux = aux.to(DEVICE)

            timesteps = torch.randint(low=0, high=NUM_STEPS, size=(N,)).to(DEVICE)
            alphas = ALPHAS[timesteps].reshape(N, 1, 1, 1).to(DEVICE)

            z = torch.randn_like(curr, device=DEVICE)
            curr = alphas.sqrt() * curr + (1 - alphas).sqrt() * z

            z_pred = model(curr, timesteps, past, aux)

            loss = loss_fn(z_pred, z)
            val_loss += loss.item()

    return val_loss / num_batches


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    desc: str,
) -> float:
    model.train()

    train_loss, num_batches = 0.0, len(train_loader)
    pbar = tqdm(train_loader, desc=desc)
    for batch, (curr, past, aux) in enumerate(pbar, start=1):
        N, T, C, H, W = curr.size()
        curr = curr.reshape(N, T * C, H, W).to(DEVICE)
        past = past.reshape(N, T * C, H, W).to(DEVICE)
        aux = aux.to(DEVICE)

        timesteps = torch.randint(low=0, high=NUM_STEPS, size=(N,)).to(DEVICE)
        alphas = ALPHAS[timesteps].reshape(N, 1, 1, 1).to(DEVICE)

        z = torch.randn_like(curr, device=DEVICE)
        curr = alphas.sqrt() * curr + (1 - alphas).sqrt() * z

        z_pred = model(curr, timesteps, past, aux)

        optim.zero_grad()
        loss = loss_fn(z_pred, z)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        pbar.set_postfix(
            {
                "Batch Loss": f"{loss.item():.4f}",
                "Train Loss": f"{train_loss / batch:.4f}",
            }
        )

    pbar.close()
    return train_loss / num_batches

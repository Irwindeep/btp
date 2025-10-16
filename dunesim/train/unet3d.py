import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dunesim import DEVICE


def val_epoch(
    model: nn.Module, val_loader: DataLoader[torch.Tensor], loss_fn: nn.Module
) -> float:
    model.eval()

    val_loss, num_batches = 0.0, len(val_loader)
    with torch.no_grad():
        for past, future, aux in val_loader:
            T = past.size(1)
            past = past.to(DEVICE)
            future = future.to(DEVICE)
            aux = aux.to(DEVICE)

            aux = aux.unsqueeze(1)
            aux = aux.expand(-1, T, -1, -1, -1)

            pred = model(past, aux)
            pred = pred[:, :, :2, :, :]

            loss = loss_fn(pred, future)
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
    for batch, (past, future, aux) in enumerate(pbar, start=1):
        past = past.to(DEVICE)
        future = future.to(DEVICE)
        aux = aux.to(DEVICE)

        pred = model(past, aux)
        pred = pred[:, :, :2, :, :]

        optim.zero_grad()
        loss = loss_fn(pred, future)
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

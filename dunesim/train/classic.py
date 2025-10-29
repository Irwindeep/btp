from functools import partial
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from typing import Tuple
from dunesim import DEVICE

tqdm = partial(tqdm, ascii=" =")
_DataLoader = DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def val_epoch(
    model: nn.Module,
    val_loader: _DataLoader,
    loss_fn: nn.Module,
) -> float:
    model.eval()

    val_loss, num_batches = 0.0, len(val_loader)
    with torch.no_grad():
        for past, future, aux in val_loader:
            past = past.to(DEVICE)
            future = future.to(DEVICE)
            aux = aux.to(DEVICE)

            with torch.autocast(DEVICE):
                pred = model(past, aux)
                loss = loss_fn(pred, future)

            val_loss += loss.item()

    return val_loss / num_batches


def train_epoch(
    model: nn.Module,
    train_loader: _DataLoader,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    desc: str,
) -> float:
    model.train()

    train_loss, num_batches = 0.0, len(train_loader)
    pbar = tqdm(train_loader, desc=desc)
    for batch, (past, future, aux) in enumerate(pbar, start=1):
        past = past.to(DEVICE)
        future = future.to(DEVICE)
        aux = aux.to(DEVICE)

        with torch.autocast(DEVICE):
            pred = model(past, aux)
            loss = loss_fn(pred, future)

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix(
            {
                "Batch Loss": f"{loss.item():.4f}",
                "Train Loss": f"{train_loss / batch:.4f}",
            }
        )

    pbar.close()
    return train_loss / num_batches

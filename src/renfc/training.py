"""Generic PyTorch training loop with early stopping on validation loss."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import log


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(X: np.ndarray, Y: np.ndarray, F: np.ndarray | None, batch_size: int, shuffle: bool,
            num_workers: int = 0) -> DataLoader:
    tensors = [torch.from_numpy(X), torch.from_numpy(Y)]
    tensors.append(torch.from_numpy(F) if F is not None and F.size else torch.zeros(len(X), 0))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False)


def _forward(model: nn.Module, xb: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    return model(xb, fb if fb.shape[1] else None)


class StepMSE(nn.Module):
    """Mean-squared error restricted to selected output steps (all steps when ``steps`` is None).

    With a multi-step head covering 24 h, an unrestricted loss is dominated by
    the many far-horizon outputs; restricting it to the lead times that are
    evaluated focuses the model on those targets.
    """

    def __init__(self, steps: list[int] | None = None):
        super().__init__()
        self.register_buffer("idx", torch.tensor(sorted(set(steps)), dtype=torch.long) if steps else None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.idx is not None:
            pred, target = pred[:, self.idx], target[:, self.idx]
        return torch.mean((pred - target) ** 2)


def train_model(model: nn.Module, X_train, Y_train, X_val, Y_val, *, F_train=None, F_val=None, epochs: int,
                batch_size: int, lr: float, patience: int, weight_decay: float = 0.0, num_workers: int = 0,
                device: torch.device | None = None, verbose: bool = False, loss_steps: list[int] | None = None) -> dict:
    """Train in place; returns a dict with timing and the loss history.

    ``loss_steps`` are 0-based output indices on which the loss is computed
    (default: all output steps).
    """
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = StepMSE(loss_steps).to(device)
    train_loader = _loader(X_train, Y_train, F_train, batch_size, True, num_workers)
    val_loader = _loader(X_val, Y_val, F_val, batch_size, False, num_workers)

    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist_train, hist_val = [], []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb, fb in train_loader:
            xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(_forward(model, xb, fb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        train_loss = tot / max(n, 1)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        hist_train.append(train_loss)
        hist_val.append(val_loss)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            log.info("  epoch %3d  train %.5f  val %.5f", epoch, train_loss, val_loss)
        if val_loss < best_val - 1e-7:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_time_sec": time.time() - t0,
        "epochs_run": len(hist_train),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_loss_history": hist_train,
        "val_loss_history": hist_val,
    }


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for xb, yb, fb in loader:
        xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
        tot += loss_fn(_forward(model, xb, fb), yb).item() * len(xb)
        n += len(xb)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, F: np.ndarray | None = None, batch_size: int = 512,
            device: torch.device | None = None) -> np.ndarray:
    device = device or get_device()
    model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        fb = torch.from_numpy(F[i : i + batch_size]).to(device) if F is not None and F.size else None
        out.append(model(xb, fb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype="float32")

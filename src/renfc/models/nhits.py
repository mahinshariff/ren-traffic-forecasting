"""N-HiTS: Neural Hierarchical Interpolation for Time Series forecasting.

A compact PyTorch implementation of the architecture of Challu et al.
(AAAI 2023), with the three ingredients that distinguish it from a plain
MLP or from N-BEATS:

1. **Multi-rate input sampling** – each stack max-pools its input with a
   stack-specific kernel, so early stacks see a coarse version of the lookback
   window and later stacks see finer detail.
2. **Hierarchical interpolation** – each block predicts a small number of
   forecast "knots" (``horizon / freq_downsample``) and linearly interpolates
   them to the full horizon, so early stacks model low-frequency components.
3. **Doubly residual stacking** – every block also emits a backcast that is
   subtracted from the running input, and the partial forecasts are summed.

The model operates on windows ``(batch, lookback, n_features)`` whose first
feature is the scaled target; the backcast/residual path uses the target
only, while the calendar covariates of the lookback (flattened) and of the
first forecast step are appended to every block's MLP input as static
exogenous information. It returns ``(batch, horizon)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, lookback: int, horizon: int, pool: int, freq_downsample: int,
                 mlp_units: int, mlp_layers: int, dropout: float, n_exog: int):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.pool = nn.MaxPool1d(kernel_size=pool, stride=pool, ceil_mode=True)
        in_dim = math.ceil(lookback / pool) + n_exog
        self.n_knots = max(math.ceil(horizon / max(freq_downsample, 1)), 1)
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(mlp_layers):
            layers += [nn.Linear(d, mlp_units), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = mlp_units
        self.mlp = nn.Sequential(*layers)
        self.theta = nn.Linear(d, lookback + self.n_knots)

    def forward(self, x: torch.Tensor, exog: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:  # x: (B, L)
        h = self.pool(x.unsqueeze(1)).squeeze(1)
        if exog is not None:
            h = torch.cat([h, exog], dim=1)
        theta = self.theta(self.mlp(h))
        backcast = theta[:, : self.lookback]
        knots = theta[:, self.lookback :]
        if self.n_knots == self.horizon:
            forecast = knots
        else:
            forecast = F.interpolate(knots.unsqueeze(1), size=self.horizon, mode="linear", align_corners=True).squeeze(1)
        return backcast, forecast


class NHiTS(nn.Module):
    def __init__(self, lookback: int, horizon: int, n_features: int = 1, n_future: int = 0, n_stacks: int = 3,
                 n_blocks: int = 1, mlp_units: int = 512, mlp_layers: int = 2, pooling_sizes=(8, 4, 1),
                 freq_downsample=(24, 12, 1), dropout: float = 0.0, exog_lookback_steps: int = 24):
        super().__init__()
        pooling_sizes = list(pooling_sizes)[:n_stacks]
        freq_downsample = list(freq_downsample)[:n_stacks]
        if len(pooling_sizes) < n_stacks or len(freq_downsample) < n_stacks:
            raise ValueError("pooling_sizes and freq_downsample must have one entry per stack")
        self.n_cov = n_features - 1
        self.n_future = n_future
        # calendar covariates of the most recent `exog_lookback_steps` steps + first forecast step
        self.exog_steps = min(exog_lookback_steps, lookback)
        n_exog = self.n_cov * self.exog_steps + n_future
        blocks = []
        for s in range(n_stacks):
            pool = max(1, min(pooling_sizes[s], lookback))
            for _ in range(n_blocks):
                blocks.append(_Block(lookback, horizon, pool, freq_downsample[s], mlp_units, mlp_layers, dropout, n_exog))
        self.blocks = nn.ModuleList(blocks)
        self.horizon = horizon

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:  # x: (B, L, C)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        target = x[:, :, 0]
        exog = None
        if self.n_cov or self.n_future:
            parts = []
            if self.n_cov:
                parts.append(x[:, -self.exog_steps :, 1:].reshape(x.size(0), -1))
            if self.n_future:
                parts.append(f)
            exog = torch.cat(parts, dim=1)
        residual = target
        forecast = target.new_zeros(target.size(0), self.horizon)
        for block in self.blocks:
            backcast, partial = block(residual, exog)
            residual = residual - backcast
            forecast = forecast + partial
        return forecast

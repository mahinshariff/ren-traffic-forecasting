"""Recurrent forecasters: the hybrid GRU-LSTM model and its two ablations.

The hybrid architecture is the one from the conference paper: a GRU layer
feeding an LSTM layer, each followed by dropout, then a dense layer and a
linear multi-step output head. ``GRUOnly`` and ``LSTMOnly`` replace the
second recurrent layer with a second layer of the same type, keeping the
unit counts and dropout rates identical so that the comparison isolates the
effect of mixing the two cell types.

Inputs are ``(batch, lookback, n_features)`` where the first feature is the
scaled target and the remaining ones are optional calendar covariates. When
``n_future`` > 0, the calendar covariates of the first forecast step are
concatenated to the last hidden state before the dense head, so the model
knows the phase of the daily/weekly cycle it is forecasting into.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TwoLayerRecurrent(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module, second_units: int, dense_units: int,
                 dropout1: float, dropout2: float, horizon: int, n_future: int):
        super().__init__()
        self.rnn1 = first
        self.drop1 = nn.Dropout(dropout1)
        self.rnn2 = second
        self.drop2 = nn.Dropout(dropout2)
        self.n_future = n_future
        self.fc1 = nn.Linear(second_units + n_future, dense_units)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, horizon)

    def forward(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:  # x: (B, L, C)
        out, _ = self.rnn1(x)
        out = self.drop1(out)
        out, _ = self.rnn2(out)
        out = self.drop2(out[:, -1, :])
        if self.n_future:
            out = torch.cat([out, f], dim=1)
        return self.fc2(self.act(self.fc1(out)))


class HybridGRULSTM(_TwoLayerRecurrent):
    def __init__(self, horizon: int, n_features: int = 1, n_future: int = 0, gru_units: int = 64,
                 lstm_units: int = 256, dense_units: int = 64, dropout1: float = 0.25, dropout2: float = 0.20):
        super().__init__(
            nn.GRU(n_features, gru_units, batch_first=True),
            nn.LSTM(gru_units, lstm_units, batch_first=True),
            lstm_units, dense_units, dropout1, dropout2, horizon, n_future,
        )


class GRUOnly(_TwoLayerRecurrent):
    def __init__(self, horizon: int, n_features: int = 1, n_future: int = 0, units1: int = 64,
                 units2: int = 256, dense_units: int = 64, dropout1: float = 0.25, dropout2: float = 0.20):
        super().__init__(
            nn.GRU(n_features, units1, batch_first=True),
            nn.GRU(units1, units2, batch_first=True),
            units2, dense_units, dropout1, dropout2, horizon, n_future,
        )


class LSTMOnly(_TwoLayerRecurrent):
    def __init__(self, horizon: int, n_features: int = 1, n_future: int = 0, units1: int = 64,
                 units2: int = 256, dense_units: int = 64, dropout1: float = 0.25, dropout2: float = 0.20):
        super().__init__(
            nn.LSTM(n_features, units1, batch_first=True),
            nn.LSTM(units1, units2, batch_first=True),
            units2, dense_units, dropout1, dropout2, horizon, n_future,
        )

"""Forecasting models. :func:`build_model` is the single construction entry point."""

from __future__ import annotations

import torch.nn as nn

from .nhits import NHiTS
from .rnn import GRUOnly, HybridGRULSTM, LSTMOnly

REGISTRY = {
    "hybrid_gru_lstm": HybridGRULSTM,
    "gru_only": GRUOnly,
    "lstm_only": LSTMOnly,
    "nhits": NHiTS,
}


def build_model(name: str, lookback: int, horizon: int, params: dict | None = None,
                n_features: int = 1, n_future: int = 0) -> nn.Module:
    """Construct a model by name.

    ``n_features`` is the number of channels in the lookback window (target
    plus calendar covariates) and ``n_future`` the number of known-future
    covariates of the first forecast step (0 when calendar features are off).
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    params = dict(params or {})
    cls = REGISTRY[name]
    if name == "nhits":
        return cls(lookback=lookback, horizon=horizon, n_features=n_features, n_future=n_future, **params)
    return cls(horizon=horizon, n_features=n_features, n_future=n_future, **params)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["build_model", "count_parameters", "REGISTRY", "HybridGRULSTM", "GRUOnly", "LSTMOnly", "NHiTS"]

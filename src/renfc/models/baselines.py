"""Training-free reference forecasters.

``naive_last``  – the last observed value persists over the whole horizon.
``snaive_24h``  – the value observed one day earlier at the same time of day
                  (seasonal naive with a 24-hour period).

Both are evaluated on exactly the same test origins and horizons as the
learned models, so every learned result can be read against "what does
yesterday's value at the same hour give you".
"""

from __future__ import annotations

import numpy as np

from ..series import Dataset
from ..utils import steps_in


def naive_last(ds: Dataset, split: str = "test") -> np.ndarray:
    y = ds.y_raw.to_numpy(dtype="float64")
    origins = ds.origin_idx[split]
    last = y[origins]
    return np.repeat(last[:, None], ds.horizon, axis=1)


def snaive_24h(ds: Dataset, split: str = "test") -> np.ndarray:
    y = ds.y_raw.to_numpy(dtype="float64")
    season = steps_in(24, ds.window)
    if season is None or season < 1:
        raise ValueError(f"Window {ds.window} does not divide 24 h")
    origins = ds.origin_idx[split]
    steps = np.arange(1, ds.horizon + 1)
    idx = origins[:, None] + steps[None, :] - season       # t + h - S
    idx = np.where(idx < 0, origins[:, None], idx)          # fall back to last value
    return y[idx]


BASELINES = {"naive_last": naive_last, "snaive_24h": snaive_24h}

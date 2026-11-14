"""Time-series preparation for the forecasting sweep.

Given the base-bin packet series of a router, this module resamples it to an
aggregation window, builds (lookback, horizon) supervised windows over the
*whole* series and assigns each window to train / validation / test by the
timestamp of its first forecast step. Building windows before splitting means
the test block keeps every one of its targets even at coarse windows (a
test block of 7 days at a 1-day window still yields 7 targets), while the
scaler and the model only ever see training-period values.

Optionally, calendar covariates (sine/cosine encodings of the hour of day and
day of week) are attached to every lookback step, and the covariates of the
first forecast step are provided as *known future* inputs, so that a model can
anchor its forecast to the phase of the daily and weekly cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, out_dir
from .utils import steps_in, to_timedelta


def load_base_series(cfg: Config, router: str) -> pd.Series:
    path = out_dir(cfg, "series", create=False) / f"{router}_{cfg.data.base_bin}.parquet"
    df = pd.read_parquet(path)
    s = pd.Series(df["packets"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(df["timestamp"]), name="packets")
    return s


def resample_series(base: pd.Series, window: str) -> pd.Series:
    """Sum the base-bin series into ``window`` bins (left-closed, left-labelled)."""
    return base.resample(window, closed="left", label="left").sum()


@dataclass
class Scaler:
    """Standardisation (optionally after log1p) fitted on training values only."""

    transform: str = "none"
    mean: float = 0.0
    std: float = 1.0

    def _fwd(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(x, 0.0)) if self.transform == "log1p" else x

    def _inv(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(x) if self.transform == "log1p" else x

    def fit(self, x: np.ndarray) -> "Scaler":
        t = self._fwd(np.asarray(x, dtype="float64"))
        self.mean = float(np.mean(t))
        self.std = float(np.std(t)) or 1.0
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (self._fwd(np.asarray(x, dtype="float64")) - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self._inv(np.asarray(z, dtype="float64") * self.std + self.mean)


@dataclass
class Dataset:
    router: str
    window: str
    lag_hours: float
    lookback: int                       # steps
    horizon: int                        # max forecast steps (24 h)
    horizon_steps: dict[int, int]       # lead time (h) -> step index (1-based)
    scaler: Scaler
    y_raw: pd.Series                    # resampled series (original units)
    X: dict[str, np.ndarray] = field(default_factory=dict)   # split -> (n, L, C) scaled target (+ calendar)
    Y: dict[str, np.ndarray] = field(default_factory=dict)   # split -> (n, H) scaled
    F: dict[str, np.ndarray] = field(default_factory=dict)   # split -> (n, 4) calendar of first forecast step
    origin_idx: dict[str, np.ndarray] = field(default_factory=dict)  # split -> index of origin t
    target_time: dict[str, pd.DatetimeIndex] = field(default_factory=dict)  # split -> first-step time

    def n(self, split: str) -> int:
        return len(self.origin_idx.get(split, []))

    @property
    def n_features(self) -> int:
        return int(self.X["train"].shape[-1])

    @property
    def n_future(self) -> int:
        return int(self.F["train"].shape[-1]) if "train" in self.F and self.F["train"].size else 0


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Sine/cosine encodings of hour-of-day and day-of-week, shape (n, 4)."""
    hod = (index.hour + index.minute / 60.0).to_numpy(dtype="float64")
    dow = index.dayofweek.to_numpy(dtype="float64") + hod / 24.0
    return np.stack([
        np.sin(2 * np.pi * hod / 24.0), np.cos(2 * np.pi * hod / 24.0),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
    ], axis=1)


def horizon_table(window: str, horizons_hours: list[int]) -> dict[int, int]:
    """Lead time (h) -> step index for the horizons that are whole steps >= 1."""
    out = {}
    for h in horizons_hours:
        s = steps_in(h, window)
        if s is not None and s >= 1:
            out[h] = s
    return out


def build_dataset(
    base: pd.Series,
    router: str,
    window: str,
    lag_hours: float,
    horizons_hours: list[int],
    split: dict[str, float],
    transform: str = "none",
    min_lookback_steps: int = 4,
    max_lookback_steps: int | None = None,
    calendar: bool = False,
) -> Dataset | None:
    """Return a :class:`Dataset` or ``None`` when the (window, lag) pair is invalid."""
    y = resample_series(base, window)
    L = steps_in(lag_hours, window)
    if L is None or L < min_lookback_steps:
        return None
    if max_lookback_steps is not None and L > max_lookback_steps:
        return None
    hsteps = horizon_table(window, horizons_hours)
    if not hsteps:
        return None
    H = max(hsteps.values())
    n = len(y)
    if n < L + H + 10:
        return None

    t0, t_end = y.index[0], y.index[-1] + to_timedelta(window)
    span = t_end - t0
    t_train_end = t0 + span * split["train"]
    t_val_end = t0 + span * (split["train"] + split["val"])

    scaler = Scaler(transform).fit(y[y.index < t_train_end].to_numpy())
    z = scaler.encode(y.to_numpy())

    origins = np.arange(L - 1, n - H)                      # index of last observed step
    first_target_time = y.index[origins + 1]
    X = np.stack([z[o - L + 1 : o + 1] for o in origins])[:, :, None]
    Y = np.stack([z[o + 1 : o + 1 + H] for o in origins])
    if calendar:
        cal = calendar_features(y.index)
        X = np.concatenate([X, np.stack([cal[o - L + 1 : o + 1] for o in origins])], axis=2)
        F = cal[origins + 1]
    else:
        F = np.zeros((len(origins), 0))

    ds = Dataset(router, window, lag_hours, L, H, hsteps, scaler, y)
    masks = {
        "train": first_target_time < t_train_end,
        "val": (first_target_time >= t_train_end) & (first_target_time < t_val_end),
        "test": first_target_time >= t_val_end,
    }
    for name, m in masks.items():
        ds.X[name] = X[m].astype("float32")
        ds.Y[name] = Y[m].astype("float32")
        ds.F[name] = F[m].astype("float32")
        ds.origin_idx[name] = origins[m]
        ds.target_time[name] = first_target_time[m]
    if ds.n("train") < 10 or ds.n("val") < 2 or ds.n("test") < 2:
        return None
    return ds

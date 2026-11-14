"""Forecast accuracy metrics and per-run sanity checks.

All metrics are computed in the original units (packets per window) after
inverse scaling. Percentage errors are computed only where the actual value
is positive; the fraction of such points is reported as ``pct_support`` so a
reader can see when zeros made a percentage metric unreliable.
"""

from __future__ import annotations

import numpy as np


def _flat(a) -> np.ndarray:
    return np.asarray(a, dtype="float64").ravel()


def smape(y_true, y_pred) -> float:
    t, p = _flat(y_true), _flat(y_pred)
    denom = (np.abs(t) + np.abs(p)) / 2.0
    ok = denom > 0
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(p[ok] - t[ok]) / denom[ok]) * 100.0)


def ape(y_true, y_pred) -> np.ndarray:
    """Absolute percentage error on points with positive actuals."""
    t, p = _flat(y_true), _flat(y_pred)
    ok = t > 0
    return np.abs(p[ok] - t[ok]) / t[ok] * 100.0


def r2(y_true, y_pred) -> float:
    t, p = _flat(y_true), _flat(y_pred)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def msle(y_true, y_pred) -> float:
    t, p = np.maximum(_flat(y_true), 0), np.maximum(_flat(y_pred), 0)
    return float(np.mean((np.log1p(t) - np.log1p(p)) ** 2))


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    t, p = _flat(y_true), _flat(y_pred)
    a = ape(t, p)
    err = p - t
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "sMAPE": smape(t, p),
        "MAPE": float(np.mean(a)) if a.size else float("nan"),
        "P99_APE": float(np.percentile(a, 99)) if a.size else float("nan"),
        "pct_support": float((t > 0).mean()) if t.size else float("nan"),
        "R2": r2(t, p),
        "MSLE": msle(t, p),
        "bias_ratio": float(p.mean() / t.mean()) if t.mean() != 0 else float("nan"),
        "corr": float(np.corrcoef(t, p)[0, 1]) if t.size > 2 and t.std() > 0 and p.std() > 0 else float("nan"),
        "neg_frac": float((p < 0).mean()),
        "n_test": int(t.size),
    }
    return out


def sanity_flags(m: dict[str, float]) -> dict[str, bool]:
    """Cheap checks that catch a broken run (wrong window, bad scaling, ...)."""
    scale_ok = np.isfinite(m["bias_ratio"]) and 0.5 <= m["bias_ratio"] <= 2.0
    corr_ok = np.isfinite(m["corr"]) and m["corr"] > 0.0
    nonneg_ok = m["neg_frac"] < 0.05
    return {"scale_ok": bool(scale_ok), "corr_ok": bool(corr_ok), "nonneg_ok": bool(nonneg_ok),
            "sanity_ok": bool(scale_ok and corr_ok and nonneg_ok)}

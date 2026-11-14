"""Stage 4 – traffic-distribution statistics and a statistics-driven window rule.

Three complementary views of each router's traffic, computed from the ETL
artefacts (no raw flows are touched):

1. **Flow-size distribution** (packets per sampled flow): complementary CDFs
   per week and for the whole window, with maximum-likelihood power-law fits
   (Clauset, Shalizi & Newman 2009) and likelihood-ratio comparisons against
   log-normal and exponential alternatives.
2. **Self-similarity / burstiness of the volume series**: variance–time plots
   on the base-bin packet series and the Hurst exponent estimated from their
   slope (aggregated-variance method).
3. **Aggregation-window statistics**: coefficient of variation, lag-1
   autocorrelation and STL seasonal strength of the volume series at every
   aggregation window used in the sweep.

From (3) a training-free rule proposes an aggregation window per router and
lead time: among the windows that are no longer than the lead time, choose
the one at which the magnitude of the series' autocorrelation at that lead
is largest. The squared autocorrelation at the lead is the variance a linear
predictor can explain from that lag alone; it rises with aggregation while
noise is averaged out but falls once the window becomes too coarse to resolve
the cycle. When sweep results
exist the rule is compared with the empirically best window per router and
lead time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Config, out_dir
from .coverage import load_selection
from .report import Report
from .series import load_base_series, resample_series
from .utils import log, steps_in, window_sort_key


# ----------------------------------------------------------------------------
# 1. Flow sizes
# ----------------------------------------------------------------------------

def powerlaw_fit(sample: np.ndarray, xmin="auto", max_points: int = 200_000, seed: int = 0) -> dict:
    import powerlaw  # imported lazily; only this stage needs it

    x = np.asarray(sample, dtype="float64")
    x = x[x >= 1]
    if len(x) > max_points:
        x = np.random.default_rng(seed).choice(x, max_points, replace=False)
    if len(x) < 100:
        return {"n": int(len(x)), "alpha": np.nan, "xmin": np.nan, "sigma": np.nan,
                "R_pl_vs_lognormal": np.nan, "p_pl_vs_lognormal": np.nan,
                "R_pl_vs_exponential": np.nan, "p_pl_vs_exponential": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {} if xmin == "auto" else {"xmin": float(xmin)}
        fit = powerlaw.Fit(x, discrete=True, estimate_discrete=True, verbose=False, **kw)
        R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        R_ex, p_ex = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    return {"n": int(len(x)), "alpha": float(fit.power_law.alpha), "xmin": float(fit.power_law.xmin),
            "sigma": float(fit.power_law.sigma), "R_pl_vs_lognormal": float(R_ln), "p_pl_vs_lognormal": float(p_ln),
            "R_pl_vs_exponential": float(R_ex), "p_pl_vs_exponential": float(p_ex),
            "tail_fraction": float((x >= fit.power_law.xmin).mean())}


def flow_size_stats(cfg: Config, router: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    npz = np.load(out_dir(cfg, "flows", create=False) / f"{router}_flowsize.npz")
    keys = [k for k in npz.files if k.startswith("week_")] + ["all"]
    rows = []
    for k in keys:
        s = npz[k]
        if len(s) == 0:
            continue
        fit = powerlaw_fit(s, cfg.distribution.powerlaw_xmin)
        rows.append({"router": router, "slice": k, "mean": float(s.mean()), "median": float(np.median(s)),
                     "p99": float(np.percentile(s, 99)), "max": float(s.max()), **fit})
    hist = pd.read_csv(out_dir(cfg, "flows", create=False) / f"{router}_flowsize_hist.csv")
    ccdf = hist[["bin_left"]].copy()
    for k in keys:
        counts = hist[k].to_numpy(dtype="float64")
        tail = counts[::-1].cumsum()[::-1]
        ccdf[k] = tail / max(counts.sum(), 1.0)
    ccdf.insert(0, "router", router)
    return pd.DataFrame(rows), ccdf


# ----------------------------------------------------------------------------
# 2. Variance–time / Hurst
# ----------------------------------------------------------------------------

def variance_time(base: pd.Series, scales_min: list[int]) -> pd.DataFrame:
    x = base.to_numpy(dtype="float64")
    var1 = np.var(x)
    rows = []
    for m in scales_min:
        n = len(x) // m
        if n < 10:
            continue
        blocks = x[: n * m].reshape(n, m).mean(axis=1)
        rows.append({"scale_min": m, "n_blocks": n, "var_ratio": float(np.var(blocks) / var1) if var1 > 0 else np.nan})
    return pd.DataFrame(rows)


def hurst_from_variance_time(vt: pd.DataFrame) -> dict:
    vt = vt.dropna()
    vt = vt[vt["var_ratio"] > 0]
    if len(vt) < 3:
        return {"beta": np.nan, "hurst": np.nan, "r2_fit": np.nan}
    lx, ly = np.log10(vt["scale_min"].to_numpy()), np.log10(vt["var_ratio"].to_numpy())
    beta, intercept = np.polyfit(lx, ly, 1)
    pred = beta * lx + intercept
    r2 = 1 - np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12)
    return {"beta": float(beta), "hurst": float(1 + beta / 2), "r2_fit": float(r2)}


# ----------------------------------------------------------------------------
# 3. Statistics at each aggregation window
# ----------------------------------------------------------------------------

def seasonal_strength(y: np.ndarray, period: int) -> float:
    """STL-based strength of seasonality (Wang, Smith & Hyndman 2006)."""
    if period < 4 or len(y) < 3 * period:
        return np.nan
    from statsmodels.tsa.seasonal import STL

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = STL(y, period=period, robust=True).fit()
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    return float(max(0.0, 1.0 - var_r / var_sr)) if var_sr > 0 else np.nan


def acf_at(y: np.ndarray, lag_steps: int) -> float:
    """Autocorrelation of ``y`` at ``lag_steps`` (NaN when undefined)."""
    if lag_steps < 1 or len(y) <= lag_steps + 2:
        return np.nan
    a, b = y[:-lag_steps], y[lag_steps:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def window_stats(base: pd.Series, windows: list[str], horizons_hours: list[int] | None = None) -> pd.DataFrame:
    rows = []
    horizons_hours = horizons_hours or []
    for w in sorted(windows, key=window_sort_key):
        y = resample_series(base, w).to_numpy(dtype="float64")
        if len(y) < 8:
            continue
        period = steps_in(24, w) or 0
        mean = y.mean()
        cov = float(y.std() / mean) if mean > 0 else np.nan
        row = {"window": w, "n_points": len(y), "steps_per_day": period, "mean": mean, "cov": cov,
               "acf1": acf_at(y, 1), "seasonal_strength_daily": seasonal_strength(y, period),
               "zero_fraction": float((y == 0).mean())}
        for h in horizons_hours:
            s = steps_in(h, w)
            row[f"acf_lead_{h}h"] = acf_at(y, s) if s is not None and s >= 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def recommend_windows(stats: pd.DataFrame, horizons_hours: list[int]) -> pd.DataFrame:
    """Training-free rule: per lead time, the window (≤ lead) whose series has the
    largest |autocorrelation| at that lead (the sign does not matter for a linear
    predictor; at a 12 h lead the daily cycle makes the correlation negative)."""
    rows = []
    for h in horizons_hours:
        col = f"acf_lead_{h}h"
        if col not in stats:
            continue
        cand = stats.dropna(subset=[col])
        if cand.empty:
            continue
        best = cand.loc[cand[col].abs().idxmax()]
        rows.append({"lead_hours": h, "recommended_window": best["window"], "acf_at_lead": float(best[col]),
                     "abs_acf_at_lead": float(abs(best[col]))})
    return pd.DataFrame(rows)


def ml_optimal_windows(cfg: Config) -> pd.DataFrame | None:
    """Best window per router and lead time in the sweep (learned models only, mean over seeds)."""
    from .experiment import merge_results

    try:
        df = merge_results(cfg)
    except FileNotFoundError:
        return None
    metric = cfg.sweep.headline_metric
    learned = df[df["model"].isin(cfg.sweep.models)]
    if learned.empty:
        return None
    per = learned.groupby(["router", "horizon_hours", "window", "model", "lag_hours"])[metric].mean().reset_index()
    idx = per.groupby(["router", "horizon_hours"])[metric].idxmin()
    best_any = per.loc[idx][["router", "horizon_hours", "window", "model", "lag_hours", metric]]
    best_any = best_any.rename(columns={"horizon_hours": "lead_hours", "window": "ml_best_window", "model": "ml_best_model",
                                        "lag_hours": "ml_best_lag_hours", metric: f"ml_best_{metric}"})
    by_window = per.groupby(["router", "horizon_hours", "window"])[metric].mean().reset_index()
    idx2 = by_window.groupby(["router", "horizon_hours"])[metric].idxmin()
    best_avg = by_window.loc[idx2][["router", "horizon_hours", "window"]].rename(columns={"horizon_hours": "lead_hours", "window": "ml_best_window_model_avg"})
    return best_any.merge(best_avg, on=["router", "lead_hours"])


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run(cfg: Config, routers: list[str] | None = None) -> None:
    routers = routers or load_selection(cfg)["routers"]
    ddir = out_dir(cfg, "distribution")
    fits, ccdfs, vts, hursts, stats, recs = [], [], [], [], [], []
    for router in routers:
        log.info("Distribution statistics: %s", router)
        f, c = flow_size_stats(cfg, router)
        fits.append(f)
        ccdfs.append(c)
        base = load_base_series(cfg, router)
        vt = variance_time(base, list(cfg.distribution.variance_time_scales_min))
        vt.insert(0, "router", router)
        vts.append(vt)
        hursts.append({"router": router, **hurst_from_variance_time(vt)})
        st = window_stats(base, list(cfg.sweep.windows), list(cfg.sweep.horizons_hours))
        st.insert(0, "router", router)
        stats.append(st)
        rec = recommend_windows(st, list(cfg.sweep.horizons_hours))
        rec.insert(0, "router", router)
        recs.append(rec)

    fits_df, ccdf_df = pd.concat(fits, ignore_index=True), pd.concat(ccdfs, ignore_index=True)
    vt_df, hurst_df = pd.concat(vts, ignore_index=True), pd.DataFrame(hursts)
    stats_df, rec_df = pd.concat(stats, ignore_index=True), pd.concat(recs, ignore_index=True)
    ml = ml_optimal_windows(cfg)
    if ml is not None:
        rec_df = rec_df.merge(ml, on=["router", "lead_hours"], how="left")
        rec_df["match_best_any"] = rec_df["recommended_window"] == rec_df["ml_best_window"]
        rec_df["match_model_avg"] = rec_df["recommended_window"] == rec_df["ml_best_window_model_avg"]

    fits_df.to_csv(ddir / "powerlaw_fits.csv", index=False)
    ccdf_df.to_csv(ddir / "flowsize_ccdf.csv", index=False)
    vt_df.to_csv(ddir / "variance_time.csv", index=False)
    hurst_df.to_csv(ddir / "hurst.csv", index=False)
    stats_df.to_csv(ddir / "stats_vs_window.csv", index=False)
    rec_df.to_csv(ddir / "window_recommendation.csv", index=False)

    rep = Report(cfg, "04_distribution", "Stage 4 – Traffic distribution statistics")
    rep.section("Power-law fits of packets-per-flow (whole window)")
    rep.table(fits_df[fits_df["slice"] == "all"].drop(columns=["slice"]))
    rep.section("Power-law exponent per week")
    rep.table(fits_df.pivot_table(index="slice", columns="router", values="alpha").reset_index(), floatfmt=".3f")
    rep.section("Hurst exponent (variance–time method)")
    rep.table(hurst_df, floatfmt=".3f")
    rep.section("Statistics vs aggregation window")
    rep.table(stats_df, max_rows=80, floatfmt=".3g")
    rep.section("Autocorrelation-at-lead window rule vs sweep optimum (per router and lead time)")
    rep.table(rec_df, max_rows=40)
    rep.write()

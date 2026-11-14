"""Stage 6 – paper figures and LaTeX tables.

Reads the CSV outputs of the previous stages and writes vector figures to
``<out_dir>/figures`` and ``\\input``-ready tables to ``<out_dir>/tables``.
Figure conventions: large bold labels, capitalised router names, consistent
router colours, "aggregation window" / "lead time" / "lookback" wording.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config, out_dir
from .coverage import commonality_routers, load_selection
from .report import Report
from .utils import log, pretty_window, window_sort_key

MODEL_LABELS = {
    "hybrid_gru_lstm": "Hybrid GRU-LSTM", "gru_only": "GRU", "lstm_only": "LSTM", "nhits": "N-HiTS",
    "naive_last": "Last value", "snaive_24h": "Seasonal naive (24 h)",
}
MODEL_STYLE = {
    "hybrid_gru_lstm": dict(color="#d62728", marker="o"), "gru_only": dict(color="#1f77b4", marker="s"),
    "lstm_only": dict(color="#2ca02c", marker="^"), "nhits": dict(color="#9467bd", marker="D"),
    "snaive_24h": dict(color="#7f7f7f", marker="x", linestyle="--"), "naive_last": dict(color="#bcbd22", marker="+", linestyle=":"),
}


def _style(cfg: Config) -> None:
    fs = cfg.figures.font_scale
    plt.rcParams.update({
        "font.size": 11 * fs, "axes.labelsize": 12 * fs, "axes.titlesize": 12 * fs,
        "xtick.labelsize": 10 * fs, "ytick.labelsize": 10 * fs, "legend.fontsize": 9 * fs,
        "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.dpi": 100,
        "savefig.dpi": cfg.figures.dpi, "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.0,
        "lines.markersize": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _panels(cfg: Config, n: int = 4, sharex: bool = True, sharey: bool = False):
    """2x2 grid (default) or a single row of panels (``figures.panel_layout: row``)."""
    if cfg.figures.get("panel_layout", "grid") == "row":
        fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), sharex=sharex, sharey=sharey)
        return fig, np.atleast_1d(axes), True
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=sharex, sharey=sharey)
    return fig, axes.ravel(), False


def _short_window(w: str) -> str:
    return pretty_window(w).replace(" min", "m").replace(" h", "h").replace(" days", "d").replace(" day", "d")


def _label(cfg: Config, router: str) -> str:
    return cfg.figures.router_labels.get(router, router.capitalize())


_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#9467bd"]


def _color(cfg: Config, router: str) -> str:
    """Configured colour, or a stable fallback for routers not listed in the config."""
    colors = cfg.figures.router_colors
    if router in colors:
        return colors[router]
    extra = [r for r in sorted(commonality_routers(cfg)) if r not in colors]
    return _FALLBACK_COLORS[extra.index(router) % len(_FALLBACK_COLORS)]


def _save(fig, fdir: Path, name: str, rect=None) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(fdir / f"{name}.pdf")
    fig.savefig(fdir / f"{name}.png")
    plt.close(fig)
    log.info("figure %s", name)


def _bold_ticks(ax) -> None:
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def _legend(fig, axes, row: bool, loc: str = "best"):
    """Legend inside the first panel (grid layout) or one shared legend below the
    panels (row layout), where the panels are too small to host it.  Returns the
    ``rect`` to pass to :func:`_save` so that ``tight_layout`` leaves room for it."""
    if not row:
        axes[0].legend(loc=loc)
        return None
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.5, columnspacing=1.8)
    return (0.0, 0.10, 1.0, 1.0)


def _latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, floatfmt: str = "%.2f",
                 col_format: str | None = None) -> None:
    """Minimal IEEEtran-friendly tabular (no extra packages required)."""
    def esc(s: str) -> str:
        return str(s).replace("_", "\\_").replace("%", "\\%")

    cols = list(df.columns)
    col_format = col_format or ("l" + "c" * (len(cols) - 1))
    lines = [f"\\begin{{table}}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering",
             f"\\begin{{tabular}}{{{col_format}}}", "\\hline",
             " & ".join(f"\\textbf{{{esc(c)}}}" for c in cols) + " \\\\", "\\hline"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("--" if np.isnan(v) else floatfmt % v)
            else:
                cells.append(esc(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("table %s", path.name)


# ----------------------------------------------------------------------------
# Sweep aggregation helpers
# ----------------------------------------------------------------------------

def _sweep(cfg: Config) -> pd.DataFrame | None:
    from .experiment import merge_results

    try:
        df = merge_results(cfg)
    except FileNotFoundError:
        return None
    df["is_baseline"] = df["model"].isin(cfg.sweep.baselines)
    return df


def _agg_seeds(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["router", "window", "lag_hours", "model", "horizon_hours"])[metric]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def _best_lag(agg: pd.DataFrame) -> pd.DataFrame:
    """For each (router, window, model, horizon) keep the lag with the lowest mean."""
    idx = agg.groupby(["router", "window", "model", "horizon_hours"])["mean"].idxmin()
    return agg.loc[idx]


def _best_config(agg_h: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Best (window, lag, model) per router among learned models at one horizon."""
    learned = agg_h[agg_h["model"].isin(cfg.sweep.models)]
    idx = learned.groupby("router")["mean"].idxmin()
    return learned.loc[idx].set_index("router")


# ----------------------------------------------------------------------------
# Figures from the sweep
# ----------------------------------------------------------------------------

def fig_sweep_window(cfg, df, routers, fdir):
    metric, hh = cfg.sweep.headline_metric, cfg.sweep.headline_horizon_hours
    agg = _best_lag(_agg_seeds(df[df["horizon_hours"] == hh], metric))
    windows = sorted(df["window"].unique(), key=window_sort_key)
    fig, axes, row = _panels(cfg)
    for ax, r in zip(axes, routers):
        sub = agg[agg["router"] == r]
        for m in list(cfg.sweep.models) + ["snaive_24h"]:
            s = sub[sub["model"] == m].set_index("window").reindex(windows)
            if s["mean"].notna().sum() == 0:
                continue
            x = np.arange(len(windows))
            ax.plot(x, s["mean"], label=MODEL_LABELS.get(m, m), **MODEL_STYLE.get(m, {}))
            ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], alpha=0.15, color=MODEL_STYLE.get(m, {}).get("color"))
        ax.set_title(_label(cfg, r))
        ax.set_xticks(np.arange(len(windows)))
        ax.set_xticklabels([(_short_window(w) if row else pretty_window(w)) for w in windows], rotation=45, ha="right")
        ax.set_ylabel(f"{metric} (%)" if metric in ("sMAPE", "MAPE", "P99_APE") else metric)
        ax.set_xlabel("Aggregation window")
        _bold_ticks(ax)
    if not row:
        for ax in axes[:2]:
            ax.set_xlabel("")
    rect = _legend(fig, axes, row)
    if cfg.figures.get("suptitles", True):
        fig.suptitle(f"Forecast error vs aggregation window ({hh} h lead time, best lookback per model)", fontweight="bold")
    _save(fig, fdir, "fig_sweep_window", rect=rect)


def fig_sweep_lag(cfg, df, routers, fdir):
    metric, hh = cfg.sweep.headline_metric, cfg.sweep.headline_horizon_hours
    agg = _agg_seeds(df[df["horizon_hours"] == hh], metric)
    best = _best_config(_best_lag(agg), cfg)
    fig, axes, row = _panels(cfg)
    for ax, r in zip(axes, routers):
        if r not in best.index:
            continue
        w = best.loc[r, "window"]
        sub = agg[(agg["router"] == r) & (agg["window"] == w)]
        lags = sorted(agg["lag_hours"].unique())          # same positions in every panel
        for m in cfg.sweep.models:
            s = sub[sub["model"] == m].set_index("lag_hours").reindex(lags)
            ax.errorbar(range(len(lags)), s["mean"], yerr=s["std"], label=MODEL_LABELS.get(m, m), capsize=3, **MODEL_STYLE.get(m, {}))
        ax.set_title(f"{_label(cfg, r)} (window {pretty_window(w)})")
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([f"{int(l)} h" if l < 168 else f"{int(l)//24} d" for l in lags])
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("Lookback window")
        _bold_ticks(ax)
    if not row:
        for ax in axes[:2]:
            ax.set_xlabel("")
    rect = _legend(fig, axes, row)
    if cfg.figures.get("suptitles", True):
        fig.suptitle(f"Effect of the lookback window at each router's best aggregation window ({hh} h lead time)", fontweight="bold")
    _save(fig, fdir, "fig_sweep_lag", rect=rect)


def fig_sweep_horizon(cfg, df, routers, fdir):
    metric, hh = cfg.sweep.headline_metric, cfg.sweep.headline_horizon_hours
    agg_all = _agg_seeds(df, metric)
    best = _best_config(_best_lag(agg_all[agg_all["horizon_hours"] == hh]), cfg)
    fig, axes, row = _panels(cfg)
    for ax, r in zip(axes, routers):
        if r not in best.index:
            continue
        w, lag = best.loc[r, "window"], best.loc[r, "lag_hours"]
        sub = agg_all[(agg_all["router"] == r) & (agg_all["window"] == w)]
        horizons = sorted(agg_all["horizon_hours"].unique())   # same positions in every panel
        for m in list(cfg.sweep.models) + ["snaive_24h"]:
            s = sub[(sub["model"] == m)]
            s = s[s["lag_hours"] == lag] if m in cfg.sweep.models else s
            s = s.set_index("horizon_hours").reindex(horizons)
            if s["mean"].notna().sum() == 0:
                continue
            ax.errorbar(range(len(horizons)), s["mean"], yerr=s["std"], label=MODEL_LABELS.get(m, m), capsize=3, **MODEL_STYLE.get(m, {}))
        ax.set_title(f"{_label(cfg, r)} ({pretty_window(w)}, lookback {int(lag)} h)" if row else f"{_label(cfg, r)} (window {pretty_window(w)}, lookback {int(lag)} h)")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"{h} h" for h in horizons])
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("Lead time")
        _bold_ticks(ax)
    if not row:
        for ax in axes[:2]:
            ax.set_xlabel("")
    rect = _legend(fig, axes, row)
    if cfg.figures.get("suptitles", True):
        fig.suptitle("Error growth with lead time at each router's best configuration", fontweight="bold")
    _save(fig, fdir, "fig_sweep_horizon", rect=rect)


def fig_model_ablation(cfg, df, routers, fdir):
    metric, hh = cfg.sweep.headline_metric, cfg.sweep.headline_horizon_hours
    agg = _best_lag(_agg_seeds(df[df["horizon_hours"] == hh], metric))
    models = list(cfg.sweep.models) + list(cfg.sweep.baselines)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = [agg[agg["model"] == m]["mean"].dropna().to_numpy() for m in models]
    bp = ax.boxplot(data, tick_labels=[MODEL_LABELS.get(m, m) for m in models], patch_artist=True, showmeans=True)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_STYLE.get(m, {}).get("color", "#cccccc"))
        patch.set_alpha(0.5)
    ax.set_ylabel(f"{metric} (%) across routers × windows")
    ax.set_title(f"Model comparison over all routers and aggregation windows ({hh} h lead time)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _bold_ticks(ax)
    _save(fig, fdir, "fig_model_ablation")


def fig_qualitative(cfg, df, routers, fdir):
    metric, hh = cfg.sweep.headline_metric, cfg.sweep.headline_horizon_hours
    agg = _agg_seeds(df[df["horizon_hours"] == hh], metric)
    best = _best_config(_best_lag(agg), cfg)
    pdir = out_dir(cfg, "predictions", create=False)
    fig, axes, row = _panels(cfg, sharex=False)
    for ax, r in zip(axes, routers):
        if r not in best.index:
            continue
        b = best.loc[r]
        seed = sorted(cfg.sweep.seeds)[0]
        run_id = f"{r}__{b['window']}__lag{int(b['lag_hours'])}__{b['model']}__seed{seed}"
        path = pdir / f"{run_id}.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path)
        t = pd.to_datetime(p["target_time"])
        ax.plot(t, p[f"y_true_{hh}h"], color="black", label="Actual", lw=1.8)
        ax.plot(t, p[f"y_pred_{hh}h"], color=MODEL_STYLE[b["model"]]["color"], label=MODEL_LABELS[b["model"]], lw=1.8, ls="--")
        base_lag = df[(df["router"] == r) & (df["window"] == b["window"]) & (df["model"] == "snaive_24h")]["lag_hours"]
        if len(base_lag):
            bpath = pdir / f"{r}__{b['window']}__lag{int(base_lag.iloc[0])}__snaive_24h__seed0.parquet"
            if bpath.exists():
                q = pd.read_parquet(bpath)
                ax.plot(pd.to_datetime(q["target_time"]), q[f"y_pred_{hh}h"], color="#7f7f7f", lw=1.2, ls=":", label="Seasonal naive")
        ax.set_title(f"{_label(cfg, r)}: {MODEL_LABELS[b['model']]} ({pretty_window(b['window'])}, L = {int(b['lag_hours'])} h)" if row
                     else f"{_label(cfg, r)}: {MODEL_LABELS[b['model']]}, window {pretty_window(b['window'])}, lookback {int(b['lag_hours'])} h")
        ax.set_ylabel("Packets per window")
        if row:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)
        _bold_ticks(ax)
    axes[0].legend(loc="upper left" if row else "best")
    if cfg.figures.get("suptitles", True):
        fig.suptitle(f"Test-period forecasts at the {hh} h lead time (best configuration per router)", fontweight="bold")
    _save(fig, fdir, "fig_qualitative")


# ----------------------------------------------------------------------------
# Dataset / coverage figures
# ----------------------------------------------------------------------------

def fig_coverage(cfg, routers, sel, fdir):
    path = out_dir(cfg, "coverage", create=False) / "daily_grid.csv"
    if not path.exists():
        return
    grid = pd.read_csv(path, parse_dates=["day"])
    fig, ax = plt.subplots(figsize=(13, 5))
    for r in routers:
        g = grid[grid["router"] == r].sort_values("day")
        ax.plot(g["day"], g["flows"] / max(g["median_flows"].iloc[0], 1), label=_label(cfg, r), color=_color(cfg, r))
    ax.axvspan(pd.Timestamp(sel["window_start"]), pd.Timestamp(sel["window_end"]) + pd.Timedelta(days=1), color="green", alpha=0.08, label="Study window")
    ax.axhline(cfg.data.good_day_fraction, color="red", ls="--", lw=1, label="Good-day threshold")
    ax.set_ylabel("Daily flows / router median")
    ax.set_xlabel("Day")
    ax.set_yscale("log")
    ax.legend(loc="lower left", ncol=3)
    ax.tick_params(axis="x", rotation=30)
    _bold_ticks(ax)
    _save(fig, fdir, "fig_coverage")


def fig_hourly_volumes(cfg, routers, fdir):
    from .series import load_base_series

    fig, ax = plt.subplots(figsize=(16, 6))
    for r in routers:
        s = load_base_series(cfg, r).resample("1h").sum()
        ax.plot(s.index, s.values, label=_label(cfg, r), color=_color(cfg, r), lw=1.3)
    ax.set_ylabel("Packets per hour")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", title="Router")
    ax.tick_params(axis="x", rotation=30)
    _bold_ticks(ax)
    _save(fig, fdir, "fig_hourly_volumes")


# ----------------------------------------------------------------------------
# Distribution figures
# ----------------------------------------------------------------------------

def fig_flowsize_ccdf(cfg, routers, fdir):
    path = out_dir(cfg, "distribution", create=False) / "flowsize_ccdf.csv"
    if not path.exists():
        return
    ccdf = pd.read_csv(path)
    weeks = [c for c in ccdf.columns if c.startswith("week_")][: cfg.distribution.weeks]
    fig, axes, row = _panels(cfg, sharex=True, sharey=True)
    for ax, r in zip(axes, routers):
        c = ccdf[ccdf["router"] == r]
        for i, w in enumerate(weeks):
            ax.plot(c["bin_left"], c[w], lw=1.4, alpha=0.8, label=f"Week {i + 1}")
        ax.plot(c["bin_left"], c["all"], color="black", lw=2.2, label="All weeks")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1.1)
        ax.set_title(_label(cfg, r))
        ax.set_xlabel("Packets per flow")
        _bold_ticks(ax)
    if row:
        axes[0].set_ylabel("CCDF  P(X ≥ x)")
    else:
        for ax in axes[:2]:
            ax.set_xlabel("")
        axes[0].set_ylabel("CCDF  P(X ≥ x)")
        axes[2].set_ylabel("CCDF  P(X ≥ x)")
    axes[0].legend(loc="lower left")
    if cfg.figures.get("suptitles", True):
        fig.suptitle("Flow-size distributions per week and over the whole window", fontweight="bold")
    _save(fig, fdir, "fig_flowsize_ccdf")


def fig_variance_time(cfg, routers, fdir):
    ddir = out_dir(cfg, "distribution", create=False)
    if not (ddir / "variance_time.csv").exists():
        return
    vt, hu = pd.read_csv(ddir / "variance_time.csv"), pd.read_csv(ddir / "hurst.csv").set_index("router")
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in routers:
        v = vt[vt["router"] == r]
        h = hu.loc[r, "hurst"] if r in hu.index else np.nan
        ax.plot(v["scale_min"], v["var_ratio"], marker="o", label=f"{_label(cfg, r)} (H = {h:.2f})", color=_color(cfg, r))
    m = np.array([vt["scale_min"].min(), vt["scale_min"].max()], dtype=float)
    ax.plot(m, m ** -1.0, color="black", ls=":", lw=1.5, label="H = 0.5 (no long-range dependence)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aggregation scale m (minutes)")
    ax.set_ylabel("Var(X(m)) / Var(X(1))")
    ax.set_title("Variance–time plot of packet volumes")
    ax.legend(loc="lower left")
    _bold_ticks(ax)
    _save(fig, fdir, "fig_variance_time")


def fig_stats_vs_window(cfg, routers, fdir):
    ddir = out_dir(cfg, "distribution", create=False)
    if not (ddir / "stats_vs_window.csv").exists():
        return
    st = pd.read_csv(ddir / "stats_vs_window.csv")
    rec = pd.read_csv(ddir / "window_recommendation.csv")
    hh = cfg.sweep.headline_horizon_hours
    rec = rec[rec["lead_hours"] == hh].set_index("router")
    windows = sorted(st["window"].unique(), key=window_sort_key)
    x = np.arange(len(windows))
    acf_col = f"acf_lead_{hh}h"
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for r in routers:
        s = st[st["router"] == r].set_index("window").reindex(windows)
        axes[0].plot(x, s["seasonal_strength_daily"], marker="o", label=_label(cfg, r), color=_color(cfg, r))
        axes[1].plot(x, s["cov"], marker="o", label=_label(cfg, r), color=_color(cfg, r))
        if acf_col in s:
            axes[2].plot(x, s[acf_col].abs(), marker="o", label=_label(cfg, r), color=_color(cfg, r))
            if r in rec.index:
                rw = rec.loc[r, "recommended_window"]
                if rw in windows:
                    axes[2].scatter([windows.index(rw)], [abs(s.loc[rw, acf_col])], s=200, facecolors="none", edgecolors=_color(cfg, r), linewidths=2.5, zorder=5)
                mw = rec.loc[r].get("ml_best_window", np.nan)
                if isinstance(mw, str) and mw in windows:
                    axes[2].scatter([windows.index(mw)], [abs(s.loc[mw, acf_col])], s=200, marker="s", facecolors="none", edgecolors=_color(cfg, r), linewidths=2.5, zorder=5)
    axes[0].set_ylabel("Daily seasonal strength (STL)")
    axes[1].set_ylabel("Coefficient of variation")
    axes[2].set_ylabel(f"|Autocorrelation| at {hh} h lead")
    axes[0].set_title("Cycle resolution")
    axes[1].set_title("Burstiness")
    axes[2].set_title(f"Circle: rule, square: sweep optimum ({hh} h lead)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_window(w) for w in windows], rotation=45, ha="right")
        ax.set_xlabel("Aggregation window")
        _bold_ticks(ax)
    axes[0].legend(loc="best")
    _save(fig, fdir, "fig_stats_vs_window")


# ----------------------------------------------------------------------------
# Commonality figures
# ----------------------------------------------------------------------------

def _pair_matrix(pairs: pd.DataFrame, routers: list[str], col: str, diag: float = 1.0) -> np.ndarray:
    M = np.full((len(routers), len(routers)), np.nan)
    for i in range(len(routers)):
        M[i, i] = diag
    for row in pairs.itertuples(index=False):
        a, b = row.pair.split("-")
        if a in routers and b in routers:
            i, j = routers.index(a), routers.index(b)
            M[i, j] = M[j, i] = getattr(row, col)
    return M


def fig_commonality_heatmaps(cfg, routers, fdir):
    cdir = out_dir(cfg, "commonality", create=False)
    if not (cdir / "temporal_similarity_pairs.csv").exists():
        return
    pairs = pd.read_csv(cdir / "temporal_similarity_pairs.csv")
    panels = [("xcorr_max", "Max. cross-correlation of hourly volumes"), ("diurnal_cosine", "Cosine similarity of diurnal profiles")]
    if "prefix_cosine_src" in pairs.columns and pairs["prefix_cosine_src"].notna().any():
        panels.append(("prefix_cosine_src", "Cosine similarity of source-prefix shares"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
    axes = np.atleast_1d(axes)
    labels = [_label(cfg, r) for r in routers]
    for ax, (col, title) in zip(axes, panels):
        M = _pair_matrix(pairs, routers, col)
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(routers)))
        ax.set_yticks(range(len(routers)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(routers)):
            for j in range(len(routers)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] < 0.6 else "black", fontweight="bold")
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"] * 0.85)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, fdir, "fig_commonality_heatmaps")


def fig_commonality_daily(cfg, routers, fdir):
    cdir = out_dir(cfg, "commonality", create=False)
    if not (cdir / "temporal_similarity_daily.csv").exists():
        return
    daily = pd.read_csv(cdir / "temporal_similarity_daily.csv", parse_dates=["day"])
    fig, ax = plt.subplots(figsize=(13, 5))
    for pair, g in daily.groupby("pair"):
        a, b = pair.split("-")
        ax.plot(g["day"], g["pearson_daily"], marker="o", ms=4, lw=1.5, label=f"{_label(cfg, a)} – {_label(cfg, b)}")
    ax.set_ylabel("Pearson correlation of hourly volumes (per day)")
    ax.set_xlabel("Day")
    ax.set_ylim(-1, 1)
    ax.legend(loc="lower left", ncol=3)
    ax.tick_params(axis="x", rotation=30)
    _bold_ticks(ax)
    _save(fig, fdir, "fig_commonality_daily")


def fig_diurnal_profiles(cfg, routers, fdir):
    cdir = out_dir(cfg, "commonality", create=False)
    if not (cdir / "hourly_volumes.csv").exists():
        return
    hourly = pd.read_csv(cdir / "hourly_volumes.csv", index_col=0, parse_dates=True)
    prof = hourly.groupby(hourly.index.hour).mean()
    prof = prof / prof.mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in routers:
        if r in prof.columns:
            ax.plot(prof.index, prof[r], marker="o", label=_label(cfg, r), color=_color(cfg, r))
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Volume / daily mean")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Mean diurnal profiles")
    ax.legend(loc="best")
    _bold_ticks(ax)
    _save(fig, fdir, "fig_diurnal_profiles")


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def tables(cfg, df, routers, sel, tdir):
    metric, hh = cfg.sweep.headline_metric, cfg.sweep.headline_horizon_hours
    # dataset table
    cov = out_dir(cfg, "coverage", create=False) / "daily_grid.csv"
    rows = []
    if cov.exists():
        grid = pd.read_csv(cov, parse_dates=["day"])
        g = grid[(grid["day"] >= pd.Timestamp(sel["window_start"])) & (grid["day"] <= pd.Timestamp(sel["window_end"]))]
        for r in routers:
            s = g[g["router"] == r]
            rows.append({"Router": _label(cfg, r), "Days": len(s), "Flows (M)": s["flows"].sum() / 1e6,
                         "Packets (B)": s["packets"].sum() / 1e9, "Median daily flows (K)": s["flows"].median() / 1e3})
        _latex_table(pd.DataFrame(rows), tdir / "tab_dataset.tex", "Study window traffic per router.", "tab:dataset", "%.1f")

    if df is None:
        return
    agg = _agg_seeds(df, metric)
    agg_h = _best_lag(agg[agg["horizon_hours"] == hh])

    # overall per model (median over router × window at headline horizon, best lag)
    per_model = agg_h.groupby("model")["mean"].median().rename(f"Median {metric} (%)").reset_index()
    extra = df[df["horizon_hours"] == hh].groupby("model")[["MAPE", "R2", "MSLE", "P99_APE", "train_time_sec"]].median().reset_index()
    per_model = per_model.merge(extra, on="model")
    per_model["model"] = per_model["model"].map(lambda m: MODEL_LABELS.get(m, m))
    per_model = per_model.rename(columns={"model": "Model", "MAPE": "MAPE (%)", "R2": "$R^2$", "P99_APE": "P99 APE (%)", "train_time_sec": "Train time (s)"})
    _latex_table(per_model, tdir / "tab_models.tex", f"Median performance per model over all routers and aggregation windows ({hh} h lead time).", "tab:models", "%.2f")

    # best window per router
    learned = agg_h[agg_h["model"].isin(cfg.sweep.models)]
    best = learned.loc[learned.groupby("router")["mean"].idxmin()].set_index("router")
    by_w = learned.groupby(["router", "window"])["mean"].min().reset_index()
    rec_path = out_dir(cfg, "distribution", create=False) / "window_recommendation.csv"
    rec_all = pd.read_csv(rec_path) if rec_path.exists() else None
    rec = rec_all[rec_all["lead_hours"] == hh].set_index("router") if rec_all is not None else None
    rows = []
    for r in routers:
        if r not in best.index:
            continue
        b = best.loc[r]
        worst = by_w[by_w["router"] == r]["mean"].max()
        sn = agg_h[(agg_h["router"] == r) & (agg_h["window"] == b["window"]) & (agg_h["model"] == "snaive_24h")]["mean"]
        row = {"Router": _label(cfg, r), "Best window": pretty_window(b["window"]), "Model": MODEL_LABELS[b["model"]],
               "Lookback": f"{int(b['lag_hours'])} h", f"Best {metric} (%)": f"{b['mean']:.1f} $\\pm$ {b['std']:.1f}",
               f"Worst-window {metric} (%)": f"{worst:.1f}", "Gain (%)": f"{100 * (worst - b['mean']) / worst:.0f}",
               "Seasonal naive (%)": f"{sn.iloc[0]:.1f}" if len(sn) else "--"}
        if rec is not None and r in rec.index:
            row["Statistics rule"] = pretty_window(rec.loc[r, "recommended_window"])
        rows.append(row)
    _latex_table(pd.DataFrame(rows), tdir / "tab_best_window.tex", f"Best aggregation window per router ({hh} h lead time; mean $\\pm$ std over seeds).", "tab:best_window")

    # window rule vs sweep optimum, per router and lead time
    if rec_all is not None and "ml_best_window" in rec_all.columns:
        rows = []
        for r in routers:
            sub = rec_all[rec_all["router"] == r].sort_values("lead_hours")
            for row in sub.itertuples(index=False):
                rows.append({"Router": _label(cfg, r), "Lead time": f"{int(row.lead_hours)} h",
                             "Rule window": pretty_window(row.recommended_window),
                             "|ACF| at lead": row.abs_acf_at_lead,
                             "Sweep-best window": pretty_window(row.ml_best_window) if isinstance(row.ml_best_window, str) else "--",
                             "Model": MODEL_LABELS.get(row.ml_best_model, row.ml_best_model) if isinstance(row.ml_best_model, str) else "--",
                             "Match": "yes" if row.match_best_any else "no"})
        _latex_table(pd.DataFrame(rows), tdir / "tab_window_rule.tex",
                     "Training-free window rule (largest autocorrelation magnitude at the lead time) versus the best window in the forecasting sweep.",
                     "tab:window_rule", "%.2f")

    # lag table: best sMAPE per (router, lag) at best window, model-averaged
    rows = []
    for r in routers:
        if r not in best.index:
            continue
        w = best.loc[r, "window"]
        sub = agg[(agg["router"] == r) & (agg["window"] == w) & (agg["horizon_hours"] == hh) & (agg["model"].isin(cfg.sweep.models))]
        row = {"Router": _label(cfg, r), "Window": pretty_window(w)}
        for lag, g in sub.groupby("lag_hours"):
            row[f"Lookback {int(lag)} h"] = g["mean"].mean()
        rows.append(row)
    if rows:
        _latex_table(pd.DataFrame(rows), tdir / "tab_lag.tex", f"Model-averaged {metric} (%) versus lookback window at each router's best aggregation window ({hh} h lead time).", "tab:lag", "%.1f")

    # power-law / Hurst table
    ddir = out_dir(cfg, "distribution", create=False)
    if (ddir / "powerlaw_fits.csv").exists():
        fits = pd.read_csv(ddir / "powerlaw_fits.csv")
        hu = pd.read_csv(ddir / "hurst.csv").set_index("router")
        rows = []
        for r in routers:
            f = fits[fits["router"] == r]
            a = f[f["slice"] == "all"]
            wk = f[f["slice"] != "all"]["alpha"]
            if a.empty:
                continue
            rows.append({"Router": _label(cfg, r), "$\\alpha$ (all)": a["alpha"].iloc[0],
                         "$\\alpha$ weekly range": f"{wk.min():.2f}--{wk.max():.2f}" if len(wk) else "--",
                         "$x_{\\min}$": a["xmin"].iloc[0], "Tail fraction": a["tail_fraction"].iloc[0],
                         "LR vs log-normal": a["R_pl_vs_lognormal"].iloc[0], "Hurst $H$": hu.loc[r, "hurst"] if r in hu.index else np.nan})
        _latex_table(pd.DataFrame(rows), tdir / "tab_distribution.tex", "Heavy-tail and self-similarity statistics per router.", "tab:distribution", "%.2f")

    # commonality table
    cdir = out_dir(cfg, "commonality", create=False)
    if (cdir / "temporal_similarity_pairs.csv").exists():
        pairs = pd.read_csv(cdir / "temporal_similarity_pairs.csv")
        pairs["Pair"] = pairs["pair"].map(lambda p: " -- ".join(_label(cfg, x) for x in p.split("-")))
        cols = {"Pair": "Pair", "hops": "Hops", "xcorr_max": "Max. xcorr", "xcorr_lag_hours": "Lag (h)", "diurnal_cosine": "Diurnal cosine", "diurnal_dtw": "Diurnal DTW"}
        if "prefix_cosine_src" in pairs.columns:
            cols["prefix_cosine_src"] = "Prefix cosine"
        t = pairs[list(cols)].rename(columns=cols)
        _latex_table(t, tdir / "tab_commonality.tex", "Cross-router commonality and backbone hop count.", "tab:commonality", "%.2f")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run(cfg: Config, routers: list[str] | None = None) -> None:
    _style(cfg)
    sel = load_selection(cfg)
    routers = routers or sel["routers"]
    fdir, tdir = out_dir(cfg, "figures"), out_dir(cfg, "tables")
    df = _sweep(cfg)
    fig_coverage(cfg, routers, sel, fdir)
    fig_hourly_volumes(cfg, routers, fdir)
    if df is not None:
        fig_sweep_window(cfg, df, routers, fdir)
        fig_sweep_lag(cfg, df, routers, fdir)
        fig_sweep_horizon(cfg, df, routers, fdir)
        fig_model_ablation(cfg, df, routers, fdir)
        fig_qualitative(cfg, df, routers, fdir)
    fig_flowsize_ccdf(cfg, routers, fdir)
    fig_variance_time(cfg, routers, fdir)
    fig_stats_vs_window(cfg, routers, fdir)
    all_routers = commonality_routers(cfg) if routers == sel["routers"] else routers
    fig_commonality_heatmaps(cfg, all_routers, fdir)
    fig_commonality_daily(cfg, all_routers, fdir)
    fig_diurnal_profiles(cfg, all_routers, fdir)
    tables(cfg, df, routers, sel, tdir)
    rep = Report(cfg, "06_figures", "Stage 6 – Figures and tables")
    rep.kv("Figures", ", ".join(sorted(p.name for p in fdir.glob("*.pdf"))))
    rep.kv("Tables", ", ".join(sorted(p.name for p in tdir.glob("*.tex"))))
    rep.write()

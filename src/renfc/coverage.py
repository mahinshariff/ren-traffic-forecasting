"""Stage 1 – data coverage per router and study-window selection.

Why this stage exists
---------------------
The collection is not uniform: some routers stop exporting flows weeks before
others, some have outages where the export rate collapses, and the first and
last days are partial. A forecasting split that lands its test block in such a
period measures the outage, not the model. This stage therefore measures, for
every router and every calendar day, how many flows were recorded, flags
"good" days relative to each router's own median, and derives:

* the fourth router (if ``data.fourth_router: auto``): the lowest-volume router
  that still has (almost) complete coverage and is not too sparse, and
* the study window: the longest run of consecutive days on which all selected
  routers are good.

Outputs (under ``<out_dir>/coverage``): ``daily_coverage.csv``,
``router_summary.csv``, ``selection.json`` and ``coverage_heatmap.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, out_dir
from .rawio import iter_batches, list_routers, parse_timestamps
from .report import Report
from .utils import log, timed


# ----------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------

def scan_router_daily(cfg: Config, router: str) -> pd.DataFrame:
    """Flows and packets per calendar day for one router (streaming)."""
    tcol, pcol = cfg.data.timestamp_col, cfg.data.packets_col
    acc: dict[pd.Timestamp, list[float]] = {}
    for chunk in iter_batches(cfg.paths.raw_dir, router, [tcol, pcol]):
        ts = parse_timestamps(chunk[tcol])
        ok = ts.notna()
        if not ok.all():
            chunk, ts = chunk[ok], ts[ok]
        days = ts.dt.floor("D")
        pk = chunk[pcol].astype("float64") if pcol in chunk else pd.Series(0.0, index=chunk.index)
        grouped = pd.DataFrame({"day": days.values, "pk": pk.values}).groupby("day")["pk"].agg(["size", "sum"])
        for day, row in grouped.iterrows():
            cur = acc.setdefault(day, [0, 0.0])
            cur[0] += int(row["size"])
            cur[1] += float(row["sum"])
    out = pd.DataFrame(
        [(router, day, v[0], v[1]) for day, v in sorted(acc.items())],
        columns=["router", "day", "flows", "packets"],
    )
    return out


def scan_all(cfg: Config, routers: list[str] | None = None) -> pd.DataFrame:
    routers = routers or list_routers(cfg.paths.raw_dir)
    frames = []
    for router in routers:
        with timed(f"coverage scan: {router}"):
            frames.append(scan_router_daily(cfg, router))
    daily = pd.concat(frames, ignore_index=True)
    return daily


# ----------------------------------------------------------------------------
# Good-day logic and selection
# ----------------------------------------------------------------------------

def complete_daily_grid(daily: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the full (router × day) grid; missing days become zeros."""
    days = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
    routers = sorted(daily["router"].unique())
    idx = pd.MultiIndex.from_product([routers, days], names=["router", "day"])
    grid = daily.set_index(["router", "day"]).reindex(idx, fill_value=0).reset_index()
    return grid


def flag_good_days(grid: pd.DataFrame, good_fraction: float) -> pd.DataFrame:
    """A day is good when BOTH its flow count and its packet count reach
    ``good_fraction`` × that router's median over days with data.

    Using both counts guards against partial outages in which the exporter
    keeps emitting flow records but the sampled packet volume collapses.
    """
    grid = grid.copy()
    with_data = grid[grid["flows"] > 0]
    med = with_data.groupby("router")[["flows", "packets"]].median()
    med.columns = ["median_flows", "median_packets"]
    grid = grid.merge(med, on="router", how="left")
    grid[["median_flows", "median_packets"]] = grid[["median_flows", "median_packets"]].fillna(0)
    flows_ok = grid["flows"] >= good_fraction * grid["median_flows"]
    packets_ok = grid["packets"] >= good_fraction * grid["median_packets"]
    grid["good"] = flows_ok & packets_ok
    grid.loc[(grid["median_flows"] <= 0) | (grid["median_packets"] <= 0), "good"] = False
    return grid


def longest_common_run(grid: pd.DataFrame, routers: list[str]) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Longest run of consecutive days on which every router in ``routers`` is good."""
    sub = grid[grid["router"].isin(routers)]
    good_all = sub.groupby("day")["good"].all().sort_index()
    best_len, best_start, best_end = 0, None, None
    run_start, run_len = None, 0
    for day, ok in good_all.items():
        if ok:
            if run_start is None:
                run_start, run_len = day, 0
            run_len += 1
            if run_len > best_len:
                best_len, best_start, best_end = run_len, run_start, day
        else:
            run_start, run_len = None, 0
    return best_start, best_end, best_len


def router_summary(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid.groupby("router")
    summary = pd.DataFrame(
        {
            "days_with_data": g.apply(lambda d: int((d["flows"] > 0).sum())),
            "good_days": g["good"].sum().astype(int),
            "first_day": g.apply(lambda d: d.loc[d["flows"] > 0, "day"].min()),
            "last_day": g.apply(lambda d: d.loc[d["flows"] > 0, "day"].max()),
            "median_daily_flows": g["median_flows"].first(),
            "median_daily_packets": g.apply(lambda d: d.loc[d["flows"] > 0, "packets"].median()),
            "total_flows": g["flows"].sum(),
            "total_packets": g["packets"].sum(),
        }
    ).reset_index()
    return summary.sort_values("median_daily_packets", ascending=False)


def select_routers_and_window(grid: pd.DataFrame, cfg: Config) -> dict:
    """Apply the selection rules from the config. Returns a JSON-able dict."""
    fixed = list(cfg.data.fixed_routers)
    missing = [r for r in fixed if r not in set(grid["router"])]
    if missing:
        raise ValueError(f"Fixed routers missing from raw data: {missing}")

    fourth = cfg.data.fourth_router
    candidates = pd.DataFrame()
    if fourth == "auto":
        start0, end0, n0 = longest_common_run(grid, fixed)
        if n0 == 0:
            raise ValueError("Fixed routers share no good day; inspect daily_coverage.csv")
        win = grid[(grid["day"] >= start0) & (grid["day"] <= end0)]
        stats = win.groupby("router").agg(
            good_fraction=("good", "mean"),
            median_daily_flows=("flows", "median"),
            median_daily_packets=("packets", "median"),
        ).reset_index()
        rules = cfg.data.auto_select
        candidates = stats[
            (~stats["router"].isin(fixed))
            & (stats["good_fraction"] >= rules.min_day_fraction)
            & (stats["median_daily_flows"] >= rules.min_median_daily_flows)
        ].sort_values("median_daily_packets")
        if candidates.empty:
            raise ValueError(
                "No candidate fourth router satisfies the auto-selection rules; "
                "relax data.auto_select or set data.fourth_router explicitly."
            )
        fourth = str(candidates.iloc[0]["router"])
    routers = fixed + [fourth]

    wcfg = cfg.data.window
    if wcfg.mode == "auto":
        start, end, n_days = longest_common_run(grid, routers)
    else:
        start, end = pd.Timestamp(wcfg.start), pd.Timestamp(wcfg.end)
        n_days = int((end - start).days + 1)
    if n_days < wcfg.min_days:
        raise ValueError(f"Study window has only {n_days} days (< data.window.min_days={wcfg.min_days})")

    return {
        "routers": routers,
        "fourth_router": fourth,
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "n_days": int(n_days),
        "candidates": candidates.to_dict(orient="records"),
    }


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def coverage_heatmap(grid: pd.DataFrame, selection: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = grid.pivot(index="router", columns="day", values="flows").fillna(0)
    med = grid.groupby("router")["median_flows"].first().reindex(pivot.index).replace(0, np.nan)
    rel = pivot.div(med, axis=0).clip(upper=2.0)
    order = list(selection["routers"]) + [r for r in pivot.index if r not in selection["routers"]]
    rel = rel.reindex(order)
    fig, ax = plt.subplots(figsize=(14, max(4, 0.3 * len(rel))))
    im = ax.imshow(rel.values, aspect="auto", cmap="viridis", vmin=0, vmax=2)
    ax.set_yticks(range(len(rel)))
    ax.set_yticklabels(rel.index)
    xt = np.arange(0, rel.shape[1], max(1, rel.shape[1] // 12))
    ax.set_xticks(xt)
    ax.set_xticklabels([d.strftime("%m-%d") for d in rel.columns[xt]], rotation=45)
    ax.set_xlabel("Day")
    ax.set_title("Daily flow count relative to each router's median (study routers on top)")
    days = list(rel.columns)
    s = days.index(pd.Timestamp(selection["window_start"]))
    e = days.index(pd.Timestamp(selection["window_end"]))
    ax.axvline(s - 0.5, color="red", lw=2)
    ax.axvline(e + 0.5, color="red", lw=2)
    fig.colorbar(im, ax=ax, label="flows / median flows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def run(cfg: Config, routers: list[str] | None = None) -> dict:
    cov_dir = out_dir(cfg, "coverage")
    daily_path = cov_dir / "daily_coverage.csv"
    if daily_path.exists() and routers is None:
        log.info("Using existing %s (delete it to rescan)", daily_path)
        daily = pd.read_csv(daily_path, parse_dates=["day"])
    else:
        daily = scan_all(cfg, routers)
        daily.to_csv(daily_path, index=False)

    grid = flag_good_days(complete_daily_grid(daily), cfg.data.good_day_fraction)
    grid.to_csv(cov_dir / "daily_grid.csv", index=False)
    summary = router_summary(grid)
    summary.to_csv(cov_dir / "router_summary.csv", index=False)
    selection = select_routers_and_window(grid, cfg)
    (cov_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    coverage_heatmap(grid, selection, cov_dir / "coverage_heatmap.png")

    rep = Report(cfg, "01_coverage", "Stage 1 – Data coverage and study window")
    rep.section("Selection")
    rep.kv("Selected routers", ", ".join(selection["routers"]))
    rep.kv("Fourth router", selection["fourth_router"])
    rep.kv("Study window", f"{selection['window_start']} to {selection['window_end']} ({selection['n_days']} days)")
    rep.kv("Good-day rule", f"flows and packets both >= {cfg.data.good_day_fraction} × the router's median daily value")
    if selection["candidates"]:
        rep.section("Fourth-router candidates (lowest volume first)")
        rep.table(pd.DataFrame(selection["candidates"]))
    rep.section("Per-router summary (all routers in raw_dir)")
    rep.table(summary, max_rows=60)
    rep.section("Selected routers – day-by-day (flows, good flag)")
    sel = grid[grid["router"].isin(selection["routers"])].pivot(index="day", columns="router", values="flows")
    good = grid[grid["router"].isin(selection["routers"])].pivot(index="day", columns="router", values="good")
    sel.columns = [f"{c}_flows" for c in sel.columns]
    good.columns = [f"{c}_good" for c in good.columns]
    rep.table(pd.concat([sel, good], axis=1).reset_index(), max_rows=120, floatfmt=".4g")
    path = rep.write()
    log.info("Report written to %s", path)
    return selection


def load_selection(cfg: Config) -> dict:
    path = out_dir(cfg, "coverage", create=False) / "selection.json"
    if not path.exists():
        raise FileNotFoundError("Run the coverage stage first (scripts/01_coverage.py)")
    return json.loads(path.read_text())


def extra_routers(cfg: Config) -> list[str]:
    """Routers processed by the ETL and used in the commonality analysis only."""
    sel = load_selection(cfg)["routers"]
    return [r for r in (cfg.data.get("extra_routers") or []) if r not in sel]


def commonality_routers(cfg: Config) -> list[str]:
    return load_selection(cfg)["routers"] + extra_routers(cfg)

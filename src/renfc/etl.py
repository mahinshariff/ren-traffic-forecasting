"""Stage 2 – one streaming pass per selected router over the study window.

For each router the pass produces three compact artefacts that every later
stage reads instead of the raw flows:

``series/<router>_<base_bin>.parquet``
    Packet (and flow) counts per base bin (default 1 min) over the study
    window, with explicit zeros for empty bins. The forecasting sweep
    resamples this to each aggregation window.

``flows/<router>_flowsize.npz`` and ``flows/<router>_flowsize_hist.csv``
    Reservoir samples of packets-per-flow per calendar week of the window
    (plus the whole window) for power-law fitting, and exact log-binned
    histograms for CCDF plots.

``prefixes/<router>_prefix_daily.parquet``
    For every day and direction (src/dst), the flow and packet counts of the
    most popular /N address prefixes, plus daily totals, for the cross-router
    commonality analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, out_dir
from .coverage import load_selection
from .rawio import available_columns, ip_prefix, iter_batches, parse_timestamps, write_parquet
from .report import Report
from .utils import log, timed

HIST_EDGES = np.logspace(0, 10, 201)  # packets per flow, 20 bins per decade


class Reservoir:
    """Vectorised reservoir sample of fixed size ``k`` over a stream of values."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k = int(k)
        self.n = 0
        self.buf = np.empty(self.k, dtype=np.float64)
        self.rng = rng

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        m = len(values)
        if m == 0:
            return
        fill = max(0, min(self.k - self.n, m))
        if fill:
            self.buf[self.n : self.n + fill] = values[:fill]
            self.n += fill
            values = values[fill:]
            m -= fill
        if m == 0:
            return
        positions = self.n + np.arange(1, m + 1)
        r = (self.rng.random(m) * positions).astype(np.int64)
        mask = r < self.k
        self.buf[r[mask]] = values[mask]
        self.n += m

    def sample(self) -> np.ndarray:
        return self.buf[: min(self.n, self.k)].copy()


def _week_index(days: pd.Series, start: pd.Timestamp) -> np.ndarray:
    return ((days - start).dt.days // 7).to_numpy()


def process_router(cfg: Config, router: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Stream one router and write its series / flow-size / prefix artefacts."""
    d = cfg.data
    cols = [d.timestamp_col, d.packets_col, d.src_col, d.dst_col]
    if d.get("bytes_col"):
        cols.append(d.bytes_col)
    present = set(available_columns(cfg.paths.raw_dir, router))
    has_bytes = bool(d.get("bytes_col")) and d.bytes_col in present
    has_addr = d.src_col in present and d.dst_col in present
    end_excl = end + pd.Timedelta(days=1)

    base = d.base_bin
    bins_index = pd.date_range(start, end_excl, freq=base, inclusive="left")
    packets_bins = pd.Series(0.0, index=bins_index)
    flows_bins = pd.Series(0, index=bins_index, dtype="int64")
    bytes_bins = pd.Series(0.0, index=bins_index) if has_bytes else None

    rng = np.random.default_rng(12345)
    n_weeks = int(np.ceil(((end_excl - start).days) / 7))
    reservoirs = {w: Reservoir(cfg.distribution.flow_sample_size, rng) for w in range(n_weeks)}
    reservoirs["all"] = Reservoir(cfg.distribution.flow_sample_size, rng)
    hist = {w: np.zeros(len(HIST_EDGES) - 1, dtype=np.int64) for w in list(range(n_weeks)) + ["all"]}

    prefix_len = cfg.commonality.prefix_len
    prefix_acc: dict[tuple[pd.Timestamp, str], pd.DataFrame] = {}
    daily_totals: dict[pd.Timestamp, list[float]] = {}

    n_rows = n_kept = 0
    for chunk in iter_batches(cfg.paths.raw_dir, router, cols):
        n_rows += len(chunk)
        ts = parse_timestamps(chunk[d.timestamp_col])
        keep = ts.notna() & (ts >= start) & (ts < end_excl)
        if not keep.any():
            continue
        chunk = chunk.loc[keep]
        ts = ts.loc[keep]
        n_kept += len(chunk)
        pk = chunk[d.packets_col].astype("float64").to_numpy()

        # 1) base-bin series
        binned = ts.dt.floor(base)
        g = pd.DataFrame({"b": binned.to_numpy(), "pk": pk}).groupby("b")["pk"].agg(["sum", "size"])
        packets_bins = packets_bins.add(g["sum"], fill_value=0.0)
        flows_bins = flows_bins.add(g["size"].astype("int64"), fill_value=0).astype("int64")
        if has_bytes:
            gb = pd.DataFrame({"b": binned.to_numpy(), "by": chunk[d.bytes_col].astype("float64").to_numpy()}).groupby("b")["by"].sum()
            bytes_bins = bytes_bins.add(gb, fill_value=0.0)

        # 2) flow sizes per week
        days = ts.dt.floor("D")
        weeks = _week_index(days, start)
        pk_pos = np.where(pk >= 1, pk, 1.0)
        for w in np.unique(weeks):
            sel = weeks == w
            reservoirs[int(w)].add(pk[sel])
            hist[int(w)] += np.histogram(pk_pos[sel], bins=HIST_EDGES)[0]
        reservoirs["all"].add(pk)
        hist["all"] += np.histogram(pk_pos, bins=HIST_EDGES)[0]

        # 3) per-day prefix tables
        for day, sub_idx in days.groupby(days).groups.items():
            sub = chunk.loc[sub_idx]
            tot = daily_totals.setdefault(day, [0, 0.0])
            tot[0] += len(sub)
            tot[1] += float(sub[d.packets_col].sum())
            if not has_addr:
                continue
            for direction, col in (("src", d.src_col), ("dst", d.dst_col)):
                pref = ip_prefix(sub[col], prefix_len)
                agg = pd.DataFrame({"prefix": pref.to_numpy(), "pk": sub[d.packets_col].astype("float64").to_numpy()})
                agg = agg.groupby("prefix")["pk"].agg(flows="size", packets="sum")
                key = (day, direction)
                prev = prefix_acc.get(key)
                prefix_acc[key] = agg if prev is None else prev.add(agg, fill_value=0)

    # ---- write artefacts -------------------------------------------------
    series_dir = out_dir(cfg, "series")
    series = pd.DataFrame({"timestamp": bins_index, "packets": packets_bins.to_numpy(), "flows": flows_bins.to_numpy()})
    if has_bytes:
        series["bytes"] = bytes_bins.to_numpy()
    write_parquet(series, series_dir / f"{router}_{base}.parquet")

    flows_dir = out_dir(cfg, "flows")
    np.savez_compressed(
        flows_dir / f"{router}_flowsize.npz",
        **{f"week_{w}": reservoirs[w].sample() for w in range(n_weeks)},
        all=reservoirs["all"].sample(),
        counts=np.array([reservoirs[w].n for w in range(n_weeks)] + [reservoirs["all"].n]),
    )
    hist_df = pd.DataFrame({"bin_left": HIST_EDGES[:-1], "bin_right": HIST_EDGES[1:]})
    for w in range(n_weeks):
        hist_df[f"week_{w}"] = hist[w]
    hist_df["all"] = hist["all"]
    hist_df.to_csv(flows_dir / f"{router}_flowsize_hist.csv", index=False)

    pref_dir = out_dir(cfg, "prefixes")
    keep_top = cfg.commonality.keep_top_prefixes_per_day
    frames = []
    for (day, direction), agg in prefix_acc.items():
        top = agg.sort_values("flows", ascending=False).head(keep_top).reset_index()
        top.insert(0, "direction", direction)
        top.insert(0, "day", day)
        top["n_prefixes_total"] = len(agg)
        frames.append(top)
    if frames:
        write_parquet(pd.concat(frames, ignore_index=True), pref_dir / f"{router}_prefix_daily.parquet")
    totals = pd.DataFrame([(day, v[0], v[1]) for day, v in sorted(daily_totals.items())], columns=["day", "flows", "packets"])
    totals.to_csv(pref_dir / f"{router}_daily_totals.csv", index=False)

    return {
        "router": router,
        "rows_scanned": n_rows,
        "rows_in_window": n_kept,
        "base_bins": len(series),
        "empty_bins": int((series["flows"] == 0).sum()),
        "total_packets": float(series["packets"].sum()),
        "has_bytes": has_bytes,
        "has_addresses": has_addr,
        "weeks": n_weeks,
    }


def run(cfg: Config, routers: list[str] | None = None) -> list[dict]:
    selection = load_selection(cfg)
    routers = routers or (selection["routers"] + [r for r in (cfg.data.get("extra_routers") or []) if r not in selection["routers"]])
    start, end = pd.Timestamp(selection["window_start"]), pd.Timestamp(selection["window_end"])
    log.info("ETL for %s over %s..%s", routers, start.date(), end.date())
    infos = []
    for router in routers:
        with timed(f"ETL: {router}"):
            infos.append(process_router(cfg, router, start, end))
    (out_dir(cfg, "series") / "etl_info.json").write_text(json.dumps(infos, indent=2))

    rep = Report(cfg, "02_etl", "Stage 2 – ETL (series, flow sizes, prefixes)")
    rep.kv("Routers", ", ".join(routers))
    rep.kv("Window", f"{start.date()} to {end.date()}")
    rep.section("Per-router summary")
    rep.table(pd.DataFrame(infos))
    rep.section("Hourly packet volume – first/last 24 h per router (sanity)")
    for router in routers:
        s = pd.read_parquet(out_dir(cfg, "series") / f"{router}_{cfg.data.base_bin}.parquet").set_index("timestamp")["packets"]
        hourly = s.resample("1h").sum()
        rep.text(f"**{router}**: {len(hourly)} hourly bins, mean {hourly.mean():.3g}, "
                 f"min {hourly.min():.3g}, max {hourly.max():.3g}, zero hours {(hourly == 0).sum()}")
    rep.write()
    return infos

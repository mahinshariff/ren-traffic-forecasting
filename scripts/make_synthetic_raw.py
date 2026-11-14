#!/usr/bin/env python
"""Generate synthetic raw NetFlow-like Parquet files for testing the pipeline.

The files mimic the schema of the Internet2 collection (one file per router,
columns ``t_first``, ``src4_addr``, ``dst4_addr``, ``in_packets``, ...) and
reproduce the properties that matter for the pipeline: diurnal and weekly
cycles, heavy-tailed flow sizes, a router whose collection stops early, a
router with an outage, a very sparse router and partial first/last days.

Usage::

    python scripts/make_synthetic_raw.py --out /tmp/renfc_raw --days 21 [--scale 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROUTERS = {
    # name: (flows per day, diurnal amplitude, special behaviour)
    "dallas": (60_000, 0.6, "outage_last_4_days"),
    "atlanta": (30_000, 0.5, None),
    "elpaso": (20_000, 0.5, None),
    "boston": (400, 0.4, "stops_after_12_days"),
    "denver": (12_000, 0.5, None),
    "albany": (6_000, 0.5, None),
}


def make_router(name: str, flows_per_day: int, amp: float, special: str | None, days: int,
                start: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        n = flows_per_day
        if special == "stops_after_12_days" and d >= 12:
            break
        if special == "outage_last_4_days" and d >= days - 4:
            n = int(flows_per_day * 0.05)
        if d == 0:
            n = int(n * 0.4)  # partial first day
        weekday_factor = 0.7 if day.weekday() >= 5 else 1.0
        n = int(n * weekday_factor)
        # diurnal intensity: peak around 15:00 UTC
        hours = rng.random(n * 3) * 24
        intensity = 1 + amp * np.cos((hours - 15) / 24 * 2 * np.pi)
        keep = rng.random(len(hours)) < intensity / (1 + amp)
        hours = hours[keep][:n]
        t = day + pd.to_timedelta(hours * 3600, unit="s")
        sizes = np.minimum((rng.pareto(1.6, len(t)) + 1) * 5, 5e6).astype("int64")
        # addresses: a popularity-skewed set of /16 prefixes, same pseudonyms every day
        prefixes = (rng.zipf(1.3, len(t)) % 4000).astype("int64")
        src = (prefixes << 16) | rng.integers(0, 65536, len(t))
        dst_prefixes = (rng.zipf(1.3, len(t)) % 4000).astype("int64")
        dst = (dst_prefixes << 16) | rng.integers(0, 65536, len(t))
        frames.append(pd.DataFrame({
            "export_sysid": 1, "t_first": t.sort_values().values, "t_last": t.sort_values().values + pd.Timedelta(seconds=30),
            "proto": 6, "src4_addr": src, "dst4_addr": dst, "src_port": 443, "dst_port": rng.integers(1024, 65535, len(t)),
            "in_packets": sizes, "in_bytes": sizes * 900, "src_as": 0, "dst_as": 0, "router": name,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--start", default="2021-10-07")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the flows per day of every router")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    start = pd.Timestamp(args.start) + pd.Timedelta(hours=14)
    for name, (fpd, amp, special) in ROUTERS.items():
        df = make_router(name, int(fpd * args.scale), amp, special, args.days, start.floor("D") if name != "dallas" else start, rng)
        if name == "dallas":
            df = df[df["t_first"] >= start]  # partial first day
        # store addresses as dotted strings for one router to exercise both code paths
        if name == "atlanta":
            for col in ("src4_addr", "dst4_addr"):
                v = df[col].to_numpy()
                df[col] = [f"{(x >> 24) & 255}.{(x >> 16) & 255}.{(x >> 8) & 255}.{x & 255}" for x in v]
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} flows, {df['t_first'].min()} -> {df['t_first'].max()}")


if __name__ == "__main__":
    main()

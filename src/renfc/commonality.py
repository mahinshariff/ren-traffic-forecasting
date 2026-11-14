"""Stage 5 – cross-router traffic commonality.

Two independent ways of asking "how similar is the traffic seen at router A
to the traffic seen at router B?", both evaluated day by day because the
address anonymisation in the collection is applied per day:

* **Prefix-share similarity.** For every day, direction (source/destination)
  and router, the ETL stored the share of flows and packets carried by each
  /N address prefix. Router pairs are compared with cosine similarity and
  Jensen–Shannon divergence of these share vectors and with the Jaccard
  overlap of their top-N prefix sets. Because the comparison is only
  meaningful if the anonymisation maps the same real prefix to the same
  pseudonym at every router on a given day, the stage first measures the
  overlap *within* a router across consecutive days and *across* routers on
  the same day against the overlap expected from random prefix sets, and
  reports whether the cross-router comparison is informative.
* **Temporal-pattern similarity.** Hourly packet volumes are compared with
  per-day Pearson correlation, the maximum cross-correlation within a lag
  window, cosine similarity of mean diurnal profiles and a dynamic-time-
  warping distance between those profiles. This view does not depend on the
  anonymisation at all.

Hop counts between router pairs are taken from the configuration and joined
to the similarity table so that commonality can be read against topology.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import Config, out_dir
from .coverage import commonality_routers
from .report import Report
from .series import load_base_series
from .utils import log


# ----------------------------------------------------------------------------
# Prefix shares
# ----------------------------------------------------------------------------

def load_prefix_tables(cfg: Config, routers: list[str]) -> pd.DataFrame:
    frames = []
    for r in routers:
        path = out_dir(cfg, "prefixes", create=False) / f"{r}_prefix_daily.parquet"
        if not path.exists():
            log.warning("No prefix table for %s (addresses missing in raw data?)", r)
            continue
        df = pd.read_parquet(path)
        df.insert(0, "router", r)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _top_set(df: pd.DataFrame, n: int) -> set:
    return set(df.nlargest(n, "flows")["prefix"].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else np.nan


def expected_random_jaccard(n: int, universe: int) -> float:
    """Jaccard of two random n-subsets of a universe of ``universe`` prefixes."""
    if universe <= 0 or n <= 0:
        return np.nan
    inter = n * n / universe
    return inter / (2 * n - inter)


def _share_vectors(a: pd.DataFrame, b: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    union = pd.Index(sorted(set(a["prefix"]) | set(b["prefix"])))
    va = a.set_index("prefix")[col].reindex(union).fillna(0).to_numpy(dtype="float64")
    vb = b.set_index("prefix")[col].reindex(union).fillna(0).to_numpy(dtype="float64")
    return va, vb


def cosine(va: np.ndarray, vb: np.ndarray) -> float:
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else np.nan


def jensen_shannon(va: np.ndarray, vb: np.ndarray) -> float:
    pa, pb = va / va.sum(), vb / vb.sum()
    m = 0.5 * (pa + pb)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def prefix_similarity(cfg: Config, tables: pd.DataFrame, routers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pairwise cross-router table, within-router day-over-day table)."""
    top_n = cfg.commonality.top_n
    cross, within = [], []
    for direction, tdir in tables.groupby("direction"):
        by = {(r, d): g for (r, d), g in tdir.groupby(["router", "day"])}
        days = sorted(tdir["day"].unique())
        # within-router consistency
        for r in routers:
            for d0, d1 in zip(days[:-1], days[1:]):
                if (r, d0) in by and (r, d1) in by:
                    a, b = by[(r, d0)], by[(r, d1)]
                    universe = int(max(a["n_prefixes_total"].iloc[0], b["n_prefixes_total"].iloc[0]))
                    within.append({"direction": direction, "router": r, "day": d1,
                                   "jaccard_topN": jaccard(_top_set(a, top_n), _top_set(b, top_n)),
                                   "expected_random": expected_random_jaccard(top_n, universe)})
        # cross-router same day
        for d in days:
            for r1, r2 in combinations(routers, 2):
                if (r1, d) not in by or (r2, d) not in by:
                    continue
                a, b = by[(r1, d)], by[(r2, d)]
                universe = int(max(a["n_prefixes_total"].iloc[0], b["n_prefixes_total"].iloc[0]))
                vf_a, vf_b = _share_vectors(a, b, "flows")
                vp_a, vp_b = _share_vectors(a, b, "packets")
                cross.append({"direction": direction, "day": d, "pair": f"{r1}-{r2}",
                              "jaccard_topN": jaccard(_top_set(a, top_n), _top_set(b, top_n)),
                              "expected_random": expected_random_jaccard(top_n, universe),
                              "cosine_flows": cosine(vf_a, vf_b), "cosine_packets": cosine(vp_a, vp_b),
                              "jsd_flows": jensen_shannon(vf_a, vf_b), "jsd_packets": jensen_shannon(vp_a, vp_b)})
    return pd.DataFrame(cross), pd.DataFrame(within)


# ----------------------------------------------------------------------------
# Temporal patterns
# ----------------------------------------------------------------------------

def hourly_frame(cfg: Config, routers: list[str]) -> pd.DataFrame:
    cols = {}
    for r in routers:
        s = load_base_series(cfg, r).resample(cfg.commonality.hourly_resample).sum()
        cols[r] = s
    return pd.DataFrame(cols)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def temporal_similarity(cfg: Config, hourly: pd.DataFrame, routers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_lag = int(cfg.commonality.max_xcorr_lag_hours)
    z = (hourly - hourly.mean()) / hourly.std().replace(0, np.nan)
    hour = hourly.index.hour
    diurnal = hourly.groupby(hour).mean()
    diurnal = diurnal / diurnal.mean()
    daily_rows, pair_rows = [], []
    for r1, r2 in combinations(routers, 2):
        pair = f"{r1}-{r2}"
        # per-day Pearson on 24 hourly values
        for d, g in hourly.groupby(hourly.index.floor("D")):
            if len(g) < 12 or g[r1].std() == 0 or g[r2].std() == 0:
                continue
            daily_rows.append({"pair": pair, "day": d, "pearson_daily": float(np.corrcoef(g[r1], g[r2])[0, 1])})
        # max cross-correlation within ±max_lag hours (whole window)
        best, best_lag = -np.inf, 0
        a, b = z[r1].to_numpy(), z[r2].to_numpy()
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                x, y = a[lag:], b[: len(b) - lag]
            else:
                x, y = a[: len(a) + lag], b[-lag:]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 48:
                continue
            c = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if c > best:
                best, best_lag = c, lag
        pair_rows.append({"pair": pair, "xcorr_max": best, "xcorr_lag_hours": best_lag,
                          "pearson_lag0": float(np.corrcoef(z[r1].fillna(0), z[r2].fillna(0))[0, 1]),
                          "diurnal_cosine": cosine(diurnal[r1].to_numpy(), diurnal[r2].to_numpy()),
                          "diurnal_dtw": dtw_distance(diurnal[r1].to_numpy(), diurnal[r2].to_numpy())})
    return pd.DataFrame(pair_rows), pd.DataFrame(daily_rows)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def _hops_table(cfg: Config, routers: list[str]) -> pd.DataFrame:
    hops = dict(cfg.get("topology", {}).get("hops", {}) or {})
    rows = []
    for r1, r2 in combinations(routers, 2):
        v = hops.get(f"{r1}-{r2}", hops.get(f"{r2}-{r1}"))
        rows.append({"pair": f"{r1}-{r2}", "hops": v})
    return pd.DataFrame(rows)


def run(cfg: Config, routers: list[str] | None = None) -> None:
    routers = routers or commonality_routers(cfg)
    cdir = out_dir(cfg, "commonality")
    rep = Report(cfg, "05_commonality", "Stage 5 – Cross-router commonality")

    tables = load_prefix_tables(cfg, routers)
    prefix_ok = False
    if not tables.empty:
        cross, within = prefix_similarity(cfg, tables, routers)
        cross.to_csv(cdir / "prefix_similarity_daily.csv", index=False)
        within.to_csv(cdir / "prefix_within_router_consistency.csv", index=False)
        within_ratio = (within["jaccard_topN"] / within["expected_random"]).median() if len(within) else np.nan
        cross_ratio = (cross["jaccard_topN"] / cross["expected_random"]).median() if len(cross) else np.nan
        pair_max = cross.groupby("pair")["jaccard_topN"].mean()
        prefix_ok = bool(np.isfinite(cross_ratio) and cross_ratio > 3)
        rep.section("Anonymisation consistency check")
        rep.kv("Median within-router day-over-day top-N Jaccard / random expectation", f"{within_ratio:.1f}")
        rep.kv("Median cross-router same-day top-N Jaccard / random expectation", f"{cross_ratio:.1f}")
        rep.kv("Highest mean same-day top-N Jaccard between two routers",
               f"{pair_max.max():.2f} ({pair_max.idxmax()})" if len(pair_max) else "n/a")
        rep.kv("Prefix-based cross-router comparison informative", prefix_ok)
        rep.text("Interpretation: a within-router ratio near 1 means pseudonyms change every day; a cross-router "
                 "ratio near 1 means pseudonyms differ between routers, in which case only the temporal-pattern "
                 "similarity below should be used in the paper. A pair of adjacent routers with a high same-day "
                 "overlap is direct evidence that pseudonyms are shared across routers within a day.")
        summary = cross.groupby(["direction", "pair"])[["jaccard_topN", "cosine_flows", "cosine_packets", "jsd_flows", "jsd_packets"]].mean().reset_index()
        summary.to_csv(cdir / "prefix_similarity_summary.csv", index=False)
        rep.section("Prefix-share similarity (mean over days)")
        rep.table(summary, floatfmt=".3f")
    else:
        rep.section("Prefix-share similarity")
        rep.text("No prefix tables found (address columns absent) – skipped.")

    hourly = hourly_frame(cfg, routers)
    hourly.to_csv(cdir / "hourly_volumes.csv")
    pairs, daily = temporal_similarity(cfg, hourly, routers)
    pairs = pairs.merge(_hops_table(cfg, routers), on="pair", how="left")
    if not tables.empty:
        ps = cross[cross["direction"] == "src"].groupby("pair")[["cosine_flows", "jaccard_topN"]].mean().reset_index()
        pairs = pairs.merge(ps.rename(columns={"cosine_flows": "prefix_cosine_src", "jaccard_topN": "prefix_jaccard_src"}), on="pair", how="left")
    pairs.to_csv(cdir / "temporal_similarity_pairs.csv", index=False)
    daily.to_csv(cdir / "temporal_similarity_daily.csv", index=False)
    rep.section("Temporal-pattern similarity per router pair")
    rep.table(pairs, floatfmt=".3f")
    rep.section("Per-day Pearson correlation of hourly volumes (mean ± std)")
    rep.table(daily.groupby("pair")["pearson_daily"].agg(["mean", "std", "count"]).reset_index(), floatfmt=".3f")
    rep.write()

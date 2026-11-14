"""Stage 3 – the forecasting sweep.

For every selected router the runner iterates over aggregation window ×
lookback (lag) × model × seed, trains the model once with a multi-step head
covering the longest lead time (24 h), and evaluates each configured lead
time on the test block. Training-free baselines are evaluated on the same
test origins. Results are appended to ``results/sweep_results.csv`` one run
at a time and a marker file is written per finished run, so an interrupted
job resumes where it stopped.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, out_dir
from .coverage import load_selection
from .metrics import compute_metrics, sanity_flags
from .models import build_model, count_parameters
from .models.baselines import BASELINES
from .report import Report
from .series import build_dataset, load_base_series
from .training import get_device, predict, train_model
from .utils import log, set_seed, window_sort_key

RESULT_COLUMNS = [
    "run_id", "router", "window", "lag_hours", "model", "seed", "horizon_hours", "horizon_step",
    "lookback_steps", "n_train", "n_val", "n_test", "n_params", "epochs_run", "best_epoch",
    "best_val_loss", "train_time_sec",
    "MAE", "RMSE", "sMAPE", "MAPE", "P99_APE", "pct_support", "R2", "MSLE", "bias_ratio", "corr",
    "neg_frac", "scale_ok", "corr_ok", "nonneg_ok", "sanity_ok",
]


def results_path(cfg: Config, router: str | None = None) -> Path:
    """Per-router CSV (safe for parallel array jobs); ``None`` gives the merged file."""
    name = "sweep_results.csv" if router is None else f"sweep_results_{router}.csv"
    return out_dir(cfg, "results") / name


def _append_rows(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_COLUMNS})


def _marker(cfg: Config, run_id: str) -> Path:
    return out_dir(cfg, "results", "done") / f"{run_id}.done"


def _save_predictions(cfg: Config, run_id: str, ds, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if not cfg.sweep.get("save_predictions", True):
        return
    df = pd.DataFrame({"target_time": ds.target_time["test"]})
    for h, step in ds.horizon_steps.items():
        df[f"y_true_{h}h"] = y_true[:, step - 1]
        df[f"y_pred_{h}h"] = y_pred[:, step - 1]
    df.to_parquet(out_dir(cfg, "predictions") / f"{run_id}.parquet", index=False)


def _rows_for(ds, base: dict, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for h, step in ds.horizon_steps.items():
        m = compute_metrics(y_true[:, step - 1], y_pred[:, step - 1])
        m.update(sanity_flags(m))
        rows.append({**base, "horizon_hours": h, "horizon_step": step, **m})
    return rows


def run_router(cfg: Config, router: str, base_series: pd.Series | None = None) -> int:
    """Run every configuration for one router. Returns number of runs executed."""
    sw = cfg.sweep
    device = get_device()
    log.info("Router %s on device %s", router, device)
    base = base_series if base_series is not None else load_base_series(cfg, router)
    rpath = results_path(cfg, router)
    executed = 0
    windows = sorted(sw.windows, key=window_sort_key)

    for window in windows:
        # ---- baselines (independent of lag/seed; use the smallest valid lag) --
        ds_b = None
        calendar = bool(sw.get("calendar_features", False))
        for lag in sorted(sw.lags_hours):
            ds_b = build_dataset(base, router, window, lag, sw.horizons_hours, sw.split, sw.transform,
                                 sw.min_lookback_steps, sw.max_lookback_steps, calendar)
            if ds_b is not None:
                break
        if ds_b is None:
            log.warning("No valid (window=%s, lag) pair for %s; skipping window", window, router)
            continue
        y_true_b = ds_b.scaler.decode(ds_b.Y["test"])
        for bname in sw.baselines:
            run_id = f"{router}__{window}__lag{int(ds_b.lag_hours)}__{bname}__seed0"
            if _marker(cfg, run_id).exists():
                continue
            y_pred = BASELINES[bname](ds_b, "test")
            info = {"run_id": run_id, "router": router, "window": window, "lag_hours": ds_b.lag_hours,
                    "model": bname, "seed": 0, "lookback_steps": ds_b.lookback,
                    "n_train": ds_b.n("train"), "n_val": ds_b.n("val"), "n_test": ds_b.n("test"),
                    "n_params": 0, "epochs_run": 0, "best_epoch": 0, "best_val_loss": float("nan"),
                    "train_time_sec": 0.0}
            _append_rows(rpath, _rows_for(ds_b, info, y_true_b, y_pred))
            _save_predictions(cfg, run_id, ds_b, y_true_b, y_pred)
            _marker(cfg, run_id).touch()
            executed += 1

        # ---- learned models --------------------------------------------------
        for lag in sorted(sw.lags_hours):
            ds = build_dataset(base, router, window, lag, sw.horizons_hours, sw.split, sw.transform,
                               sw.min_lookback_steps, sw.max_lookback_steps, calendar)
            if ds is None:
                log.info("skip %s window=%s lag=%sh (invalid lookback or too little data)", router, window, lag)
                continue
            y_true = ds.scaler.decode(ds.Y["test"])
            for model_name in sw.models:
                for seed in sw.seeds:
                    run_id = f"{router}__{window}__lag{int(lag)}__{model_name}__seed{seed}"
                    if _marker(cfg, run_id).exists():
                        continue
                    set_seed(int(seed))
                    t0 = time.time()
                    model = build_model(model_name, ds.lookback, ds.horizon, sw.model_params.get(model_name, {}),
                                        n_features=ds.n_features, n_future=ds.n_future)
                    tr = sw.train
                    loss_steps = [s - 1 for s in ds.horizon_steps.values()] if sw.get("loss_on_evaluated_leads", True) else None
                    fit = train_model(model, ds.X["train"], ds.Y["train"], ds.X["val"], ds.Y["val"],
                                      F_train=ds.F["train"], F_val=ds.F["val"],
                                      epochs=tr.epochs, batch_size=tr.batch_size, lr=tr.lr, patience=tr.patience,
                                      weight_decay=tr.get("weight_decay", 0.0), num_workers=tr.get("num_workers", 0),
                                      device=device, loss_steps=loss_steps)
                    y_pred = ds.scaler.decode(predict(model, ds.X["test"], ds.F["test"], device=device))
                    info = {"run_id": run_id, "router": router, "window": window, "lag_hours": lag,
                            "model": model_name, "seed": seed, "lookback_steps": ds.lookback,
                            "n_train": ds.n("train"), "n_val": ds.n("val"), "n_test": ds.n("test"),
                            "n_params": count_parameters(model), "epochs_run": fit["epochs_run"],
                            "best_epoch": fit["best_epoch"], "best_val_loss": fit["best_val_loss"],
                            "train_time_sec": fit["train_time_sec"]}
                    rows = _rows_for(ds, info, y_true, y_pred)
                    _append_rows(rpath, rows)
                    _save_predictions(cfg, run_id, ds, y_true, y_pred)
                    hist_path = out_dir(cfg, "results", "histories") / f"{run_id}.json"
                    hist_path.write_text(json.dumps({"train": fit["train_loss_history"], "val": fit["val_loss_history"]}))
                    _marker(cfg, run_id).touch()
                    executed += 1
                    head = rows[-1]
                    log.info("%-60s L=%4d  ep=%3d  %.0fs  sMAPE@%dh=%6.2f  R2=%6.3f  ok=%s",
                             run_id, ds.lookback, fit["epochs_run"], time.time() - t0,
                             head["horizon_hours"], head["sMAPE"], head["R2"], head["sanity_ok"])
    return executed


def merge_results(cfg: Config) -> pd.DataFrame:
    """Concatenate the per-router CSVs into ``sweep_results.csv`` and return it."""
    rdir = out_dir(cfg, "results")
    parts = sorted(rdir.glob("sweep_results_*.csv"))
    if not parts:
        raise FileNotFoundError("No sweep results found; run the sweep stage first")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["run_id", "horizon_hours"], keep="last")
    df.to_csv(results_path(cfg), index=False)
    return df


def load_results(cfg: Config) -> pd.DataFrame:
    return merge_results(cfg)


def write_report(cfg: Config, routers: list[str]) -> Path:
    df = load_results(cfg)
    hh = cfg.sweep.headline_horizon_hours
    rep = Report(cfg, "03_sweep", "Stage 3 – Forecasting sweep")
    rep.kv("Rows in results CSV", len(df))
    rep.kv("Runs", df["run_id"].nunique())
    rep.kv("Routers", ", ".join(sorted(df["router"].unique())))
    rep.kv("Sanity failures (rows)", int((~df["sanity_ok"].astype(bool)).sum()))
    rep.section(f"Mean sMAPE at {hh} h lead time (over seeds) – router × window, best lag per model")
    sub = df[df["horizon_hours"] == hh]
    agg = sub.groupby(["router", "window", "lag_hours", "model"])["sMAPE"].mean().reset_index()
    best_lag = agg.loc[agg.groupby(["router", "window", "model"])["sMAPE"].idxmin()]
    piv = best_lag.pivot_table(index=["router", "window"], columns="model", values="sMAPE").reset_index()
    piv["_k"] = piv["window"].map(window_sort_key)
    piv = piv.sort_values(["router", "_k"]).drop(columns="_k")
    rep.table(piv, max_rows=80, floatfmt=".1f")
    rep.section("Best learned configuration per router (min mean sMAPE at headline lead time) vs seasonal naive")
    rows = []
    learned = agg[agg["model"].isin(cfg.sweep.models)]
    for r, g in learned.groupby("router"):
        b = g.loc[g["sMAPE"].idxmin()]
        sn = agg[(agg["router"] == r) & (agg["window"] == b["window"]) & (agg["model"] == "snaive_24h")]["sMAPE"]
        rows.append({"router": r, "window": b["window"], "lag_hours": b["lag_hours"], "model": b["model"],
                     "sMAPE": b["sMAPE"], "snaive_24h_same_window": sn.iloc[0] if len(sn) else float("nan")})
    rep.table(pd.DataFrame(rows))
    rep.section("Median metrics by model (all rows)")
    rep.table(df.groupby("model")[["sMAPE", "MAPE", "R2", "MSLE", "P99_APE", "train_time_sec"]].median().reset_index())
    rep.section("Rows failing sanity checks")
    bad = df[~df["sanity_ok"].astype(bool)][["run_id", "horizon_hours", "bias_ratio", "corr", "neg_frac", "n_test"]]
    rep.table(bad, max_rows=40)
    return rep.write()


def run(cfg: Config, routers: list[str] | None = None, report: bool = True) -> None:
    routers = routers or load_selection(cfg)["routers"]
    total = 0
    for router in routers:
        total += run_router(cfg, router)
    log.info("Executed %d new runs", total)
    if report:
        write_report(cfg, routers)

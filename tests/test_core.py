"""Unit tests for the pieces of the pipeline that are easy to get subtly wrong."""

import numpy as np
import pandas as pd
import pytest
import torch

from renfc.metrics import compute_metrics, sanity_flags, smape
from renfc.models import build_model
from renfc.models.baselines import naive_last, snaive_24h
from renfc.rawio import ip_prefix
from renfc.series import Scaler, build_dataset, calendar_features, horizon_table
from renfc.utils import steps_in, window_sort_key


def _synthetic_base(days: int = 20, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2021-10-08", periods=days * 1440, freq="1min")
    rng = np.random.default_rng(seed)
    hours = idx.hour + idx.minute / 60
    y = 1000 * (1.5 + np.cos((hours - 15) / 24 * 2 * np.pi)) + rng.gamma(2.0, 100, len(idx))
    return pd.Series(y, index=idx)


def test_steps_in_rejects_non_integer_multiples():
    assert steps_in(24, "5min") == 288
    assert steps_in(1, "45min") is None
    assert steps_in(6, "45min") == 8
    assert steps_in(24, "1D") == 1


def test_window_sort_key_orders_by_duration():
    ws = ["1D", "5min", "1h", "45min", "12h"]
    assert sorted(ws, key=window_sort_key) == ["5min", "45min", "1h", "12h", "1D"]


def test_horizon_table_drops_sub_window_horizons():
    assert horizon_table("12h", [1, 6, 12, 24]) == {12: 1, 24: 2}
    assert horizon_table("1D", [1, 6, 12, 24]) == {24: 1}
    assert horizon_table("5min", [1, 6, 12, 24]) == {1: 12, 6: 72, 12: 144, 24: 288}


def test_scaler_roundtrip_both_transforms():
    x = np.array([0.0, 10.0, 1e6, 3e6])
    for t in ("none", "log1p"):
        s = Scaler(t).fit(x)
        np.testing.assert_allclose(s.decode(s.encode(x)), x, rtol=1e-9, atol=1e-6)


def test_build_dataset_split_and_shapes():
    base = _synthetic_base(days=20)
    ds = build_dataset(base, "r", "1h", 24, [1, 6, 12, 24], {"train": 0.7, "val": 0.15, "test": 0.15})
    assert ds is not None
    assert ds.lookback == 24 and ds.horizon == 24
    assert ds.X["train"].shape[1:] == (24, 1) and ds.Y["train"].shape[1] == 24
    # chronological: every train target precedes every val target precedes every test target
    assert ds.target_time["train"].max() < ds.target_time["val"].min() < ds.target_time["test"].min()
    # test block keeps all its targets (15% of 480 h = 72 h) minus the horizon that must fit inside the series
    assert ds.n("test") == 72 - 24 + 1
    # scaled targets decode to the original series values
    o = ds.origin_idx["test"][0]
    np.testing.assert_allclose(ds.scaler.decode(ds.Y["test"][0]), base.resample("1h").sum().to_numpy()[o + 1 : o + 25], rtol=1e-5)


def test_build_dataset_rejects_short_lookback():
    base = _synthetic_base(days=20)
    assert build_dataset(base, "r", "1D", 24, [24], {"train": 0.7, "val": 0.15, "test": 0.15}, min_lookback_steps=4) is None
    assert build_dataset(base, "r", "5min", 168, [24], {"train": 0.7, "val": 0.15, "test": 0.15}, max_lookback_steps=576) is None


def test_seasonal_naive_uses_value_one_day_earlier():
    base = _synthetic_base(days=20)
    ds = build_dataset(base, "r", "1h", 24, [1, 24], {"train": 0.7, "val": 0.15, "test": 0.15})
    y = ds.y_raw.to_numpy()
    pred = snaive_24h(ds, "test")
    o = ds.origin_idx["test"][5]
    assert pred[5, 0] == y[o + 1 - 24]
    assert pred[5, 23] == y[o + 24 - 24]
    assert np.all(naive_last(ds, "test")[5] == y[o])


@pytest.mark.parametrize("name", ["hybrid_gru_lstm", "gru_only", "lstm_only", "nhits"])
def test_models_forward_shape(name):
    params = {"nhits": {"mlp_units": 32, "pooling_sizes": [8, 4, 1], "freq_downsample": [24, 12, 1]}}.get(name, {})
    m = build_model(name, lookback=48, horizon=24, params=params)
    assert m(torch.randn(5, 48, 1)).shape == (5, 24)
    # with calendar covariates in the lookback and known-future covariates of the first step
    m = build_model(name, lookback=48, horizon=24, params=params, n_features=5, n_future=4)
    assert m(torch.randn(5, 48, 5), torch.randn(5, 4)).shape == (5, 24)


def test_calendar_features_and_dataset_channels():
    idx = pd.date_range("2021-10-08", periods=48, freq="1h")
    cal = calendar_features(idx)
    assert cal.shape == (48, 4)
    np.testing.assert_allclose(cal[0], [0.0, 1.0, np.sin(2 * np.pi * 4 / 7), np.cos(2 * np.pi * 4 / 7)], atol=1e-9)  # Friday 00:00
    np.testing.assert_allclose(cal[6, :2], [1.0, 0.0], atol=1e-9)  # 06:00 -> sin=1
    base = _synthetic_base(days=20)
    ds = build_dataset(base, "r", "1h", 24, [1, 24], {"train": 0.7, "val": 0.15, "test": 0.15}, calendar=True)
    assert ds.n_features == 5 and ds.n_future == 4
    assert ds.X["train"].shape[1:] == (24, 5) and ds.F["train"].shape == (ds.n("train"), 4)
    # the future covariate equals the calendar encoding of the first target time
    np.testing.assert_allclose(ds.F["test"][0], calendar_features(ds.target_time["test"][:1])[0], atol=1e-6)


def test_nhits_handles_tiny_windows():
    m = build_model("nhits", lookback=4, horizon=1, params={"mlp_units": 16, "pooling_sizes": [8, 4, 1], "freq_downsample": [24, 12, 1]})
    assert m(torch.randn(3, 4, 1)).shape == (3, 1)


def test_metrics_and_sanity():
    t = np.array([100.0, 200.0, 300.0, 400.0])
    m = compute_metrics(t, t)
    assert m["sMAPE"] == 0 and m["MAPE"] == 0 and m["R2"] == 1 and sanity_flags(m)["sanity_ok"]
    bad = compute_metrics(t, t[::-1] * 3)
    assert not sanity_flags(bad)["scale_ok"] and not sanity_flags(bad)["corr_ok"]
    assert smape(np.array([0.0]), np.array([0.0])) != smape(np.array([1.0]), np.array([3.0]))


def test_ip_prefix_int_and_string_agree():
    ints = pd.Series([(10 << 24) | (20 << 16) | (30 << 8) | 40, (192 << 24) | (168 << 16) | 5])
    strs = pd.Series(["10.20.30.40", "192.168.0.5"])
    assert ip_prefix(ints, 16).tolist() == ip_prefix(strs, 16).tolist() == [(10 << 8) | 20, (192 << 8) | 168]

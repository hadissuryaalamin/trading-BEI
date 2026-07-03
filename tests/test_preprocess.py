"""Correctness tests for the feature pipeline: corporate actions, gaps,
target alignment, universe screen, and (most importantly) no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import compute_features, normalize, TARGET_COLUMN, ALL_FEATURES
from src.market import apply_universe, build_market
from tests.conftest import make_panel


def _row(feats, ticker, i_date, dates):
    r = feats[(feats["ticker"] == ticker) & (feats["date"] == dates[i_date])]
    assert len(r) == 1
    return r.iloc[0]


# --------------------------------------------------------------------------- #
# Corporate actions
# --------------------------------------------------------------------------- #
def test_split_is_not_a_return():
    """A 2:1 split must produce ~0 return, not -50%."""
    panel = make_panel({"SPLT": {"split_day": 20}, "FLAT": {}})
    dates = sorted(panel["date"].unique())
    feats = compute_features(panel)

    r = _row(feats, "SPLT", 20, dates)
    assert abs(r["log_return"]) < 1e-9          # close/prev_close = 50/50
    assert abs(r["mom_5"]) < 1e-9               # adjusted log-price path is flat
    # the day BEFORE the split: forward return crosses the split, still ~0
    r_before = _row(feats, "SPLT", 19, dates)
    assert abs(r_before[TARGET_COLUMN]) < 1e-9


def test_split_adjusted_market_returns():
    panel = make_panel({"SPLT": {"split_day": 20, "ratio": 10}})
    market = build_market(panel)
    dates = sorted(panel["date"].unique())
    r = market[(market["ticker"] == "SPLT") & (market["date"] == dates[20])]["ret"].iloc[0]
    assert abs(r) < 1e-9                        # NOT -90%


# --------------------------------------------------------------------------- #
# Gaps / suspensions / stale prices
# --------------------------------------------------------------------------- #
def test_target_nan_across_row_gap():
    """If the ticker has no row on t+1, the 'next-day' label must not exist."""
    panel = make_panel({"GAPP": {"skip_days": {21, 22, 23}}, "FLAT": {}})
    dates = sorted(panel["date"].unique())
    feats = compute_features(panel)
    assert np.isnan(_row(feats, "GAPP", 20, dates)[TARGET_COLUMN])   # spans the gap
    assert np.isfinite(_row(feats, "GAPP", 19, dates)[TARGET_COLUMN])  # t+1 exists
    assert _row(feats, "GAPP", 20, dates)["fwd_gap"] == 4


def test_target_nan_when_next_day_not_traded():
    """A stale (volume=0) next-day close is not a real, realizable return."""
    panel = make_panel({"STAL": {"no_trade": {21}}, "FLAT": {}})
    dates = sorted(panel["date"].unique())
    feats = compute_features(panel)
    assert np.isnan(_row(feats, "STAL", 20, dates)[TARGET_COLUMN])
    r = _row(feats, "STAL", 20, dates)
    assert r["valid_day"]                        # day t itself traded fine


def test_valid_day_false_when_not_traded():
    panel = make_panel({"STAL": {"no_trade": {21}}, "FLAT": {}})
    dates = sorted(panel["date"].unique())
    feats = compute_features(panel)
    assert not _row(feats, "STAL", 21, dates)["valid_day"]


# --------------------------------------------------------------------------- #
# Universe screen
# --------------------------------------------------------------------------- #
def test_universe_filters_illiquid():
    panel = make_panel({
        "LIQD": {"value": 5e9},
        "TINY": {"value": 1e7},
    })
    feats = compute_features(panel)
    feats = apply_universe(feats, panel, {"min_value_idr": 1e9, "window": 20})
    assert feats.loc[feats["ticker"] == "LIQD", "valid_day"].iloc[10:].all()
    assert not feats.loc[feats["ticker"] == "TINY", "valid_day"].any()


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def test_no_lookahead_features(simple_panel):
    """Mutating all data strictly after day t must not change any feature at <= t
    (the target at t is ALLOWED to change -- it is a label, not an input)."""
    dates = sorted(simple_panel["date"].unique())
    t = dates[25]

    base = compute_features(simple_panel)
    base = normalize(base)

    mutated = simple_panel.copy()
    fut = mutated["date"] > t
    mutated.loc[fut, ["close", "high", "low", "prev_close"]] *= 3.7
    mutated.loc[fut, "volume"] *= 11.0
    mutated.loc[fut, "value"] *= 11.0
    mut = compute_features(mutated)
    mut = normalize(mut)

    past_b = base[base["date"] <= t].reset_index(drop=True)
    past_m = mut[mut["date"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        past_b[["date", "ticker", *ALL_FEATURES]],
        past_m[["date", "ticker", *ALL_FEATURES]],
    )


def test_target_is_the_next_day_return(simple_panel):
    """fwd_return at t equals the adjusted log return realized on t+1."""
    feats = compute_features(simple_panel)
    a = feats[feats["ticker"] == "AAAA"].reset_index(drop=True)
    got = a[TARGET_COLUMN].to_numpy()[:-1]
    nxt = a["log_return"].to_numpy()[1:]
    ok = np.isfinite(got)
    assert ok.sum() > 20
    np.testing.assert_allclose(got[ok], nxt[ok], atol=1e-10)


# --------------------------------------------------------------------------- #
# Dataset windowing
# --------------------------------------------------------------------------- #
def test_dataset_window_content_and_validity(simple_panel):
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.dataset import IDXWindowDataset

    feats = normalize(compute_features(simple_panel))
    lb = 10
    ds = IDXWindowDataset(feats, lookback=lb)
    assert len(ds) > 0
    x, y, meta = ds[0]
    assert x.shape == (lb, len(ALL_FEATURES))
    # the window is exactly the lb rows ending at meta.date for meta.ticker
    block = feats[feats["ticker"] == meta["ticker"]].sort_values("date")
    end_i = block.index[block["date"] == meta["date"]][0]
    end_pos = block.index.get_loc(end_i)
    expect = block.iloc[end_pos - lb + 1 : end_pos + 1][ALL_FEATURES].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(x.numpy(), expect)


def test_dataset_skips_invalid_days(simple_panel):
    pytest.importorskip("torch")
    from src.dataset import IDXWindowDataset

    feats = normalize(compute_features(simple_panel))
    feats2 = feats.copy()
    feats2.loc[feats2["ticker"] == "AAAA", "valid_day"] = False
    ds = IDXWindowDataset(feats2, lookback=10)
    tickers = {ds[i][2]["ticker"] for i in range(len(ds))}
    assert "AAAA" not in tickers

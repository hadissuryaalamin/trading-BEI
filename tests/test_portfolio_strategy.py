"""Tests for the positive-score pro-rata portfolio strategy.

Covers the pure helper `target_weights` (weighting logic in isolation) and the
quick validation backtest `backtest_long_only` (weighted gross + weight-change
costs). The stateful simulator is exercised in test_backtest.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    target_weights,
    STRATEGY_EQUAL_TOPN,
    STRATEGY_POSITIVE_TOPN_PRORATA,
)
from src.train_test import backtest_long_only


PRORATA = STRATEGY_POSITIVE_TOPN_PRORATA


def s(d):
    return pd.Series(d, dtype=float)


# --------------------------------------------------------------------------- #
# target_weights: pure helper
# --------------------------------------------------------------------------- #
def test_prorata_ignores_negative_and_zero_scores():
    w = target_weights(s({"A": 3.0, "B": -1.0, "C": 0.0, "D": 1.0}), top_n=10,
                       strategy=PRORATA)
    assert set(w) == {"A", "D"}                      # B (<0) and C (==0) dropped


def test_prorata_selects_only_top_n_positive():
    w = target_weights(s({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}),
                       top_n=3, strategy=PRORATA)
    assert set(w) == {"A", "B", "C"}                 # the 3 highest positives


def test_prorata_weights_sum_to_one():
    w = target_weights(s({"A": 3.0, "B": 1.0, "C": 6.0}), top_n=10, strategy=PRORATA)
    assert sum(w.values()) == pytest.approx(1.0)


def test_prorata_weights_proportional_to_score():
    w = target_weights(s({"A": 3.0, "B": 1.0}), top_n=10, strategy=PRORATA)
    assert w["A"] == pytest.approx(0.75)             # 3 / (3+1)
    assert w["B"] == pytest.approx(0.25)             # 1 / (3+1)
    assert w["A"] / w["B"] == pytest.approx(3.0)


def test_prorata_all_nonpositive_is_cash():
    assert target_weights(s({"A": -1.0, "B": 0.0}), top_n=10, strategy=PRORATA) == {}
    assert target_weights(s({}), top_n=10, strategy=PRORATA) == {}


def test_prorata_ignores_nan_scores():
    w = target_weights(s({"A": 2.0, "B": np.nan, "C": 1.0}), top_n=10, strategy=PRORATA)
    assert set(w) == {"A", "C"}


def test_equal_topn_weights():
    w = target_weights(s({"A": 3.0, "B": 2.0, "C": 1.0, "D": -9.0}), top_n=2,
                       strategy=STRATEGY_EQUAL_TOPN)
    assert set(w) == {"A", "B"}                      # equal weight ignores sign
    assert w["A"] == pytest.approx(0.5) and w["B"] == pytest.approx(0.5)


def test_legacy_alias_maps_to_equal():
    w = target_weights(s({"A": 1.0, "B": 2.0}), top_n=2, strategy="long_only")
    assert w == pytest.approx({"A": 0.5, "B": 0.5})


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        target_weights(s({"A": 1.0}), top_n=1, strategy="nonsense")


# --------------------------------------------------------------------------- #
# backtest_long_only: quick validation backtest, pro-rata path
# --------------------------------------------------------------------------- #
def mk_scores(rows):
    df = pd.DataFrame(rows, columns=["date", "ticker", "score", "fwd_return"])
    df["date"] = pd.to_datetime(df["date"])
    return df


D1, D2 = "2025-01-06", "2025-01-07"


def test_quick_prorata_gross_uses_prorata_weights():
    # one day: A(score 3) and B(score 1) -> weights 0.75 / 0.25; C(<0) ignored.
    # log returns chosen so simple returns are clean: A +10%, B -20%.
    rA, rB = np.log1p(0.10), np.log1p(-0.20)
    scores = mk_scores([(D1, "A", 3.0, rA), (D1, "B", 1.0, rB), (D1, "C", -5.0, 0.0)])
    m, daily = backtest_long_only(scores, top_n=10, buy_cost_bps=0, sell_cost_bps=0,
                                  strategy=STRATEGY_POSITIVE_TOPN_PRORATA)
    expect = 0.75 * 0.10 + 0.25 * -0.20              # weighted simple return
    assert daily.iloc[0]["gross"] == pytest.approx(expect)


def test_quick_prorata_all_negative_is_cash():
    scores = mk_scores([(D1, "A", -1.0, np.log1p(0.5)), (D1, "B", -2.0, np.log1p(0.9))])
    m, daily = backtest_long_only(scores, top_n=10, buy_cost_bps=10, sell_cost_bps=10,
                                  strategy=STRATEGY_POSITIVE_TOPN_PRORATA)
    assert daily.iloc[0]["gross"] == pytest.approx(0.0)   # no positive -> flat
    assert daily.iloc[0]["cost"] == pytest.approx(0.0)    # nothing bought


def test_quick_prorata_cost_from_weight_changes():
    # Day 1: enter A=0.75, B=0.25 from cash -> all buys, cost = 1.0 * buy_c.
    # Day 2: flip to A=0.25, B=0.75 -> sell 0.5 of A, buy 0.5 of B.
    buy_bps, sell_bps = 20.0, 30.0
    buy_c, sell_c = buy_bps / 1e4, sell_bps / 1e4
    scores = mk_scores([
        (D1, "A", 3.0, 0.0), (D1, "B", 1.0, 0.0),
        (D2, "A", 1.0, 0.0), (D2, "B", 3.0, 0.0),
    ])
    m, daily = backtest_long_only(scores, top_n=10, buy_cost_bps=buy_bps,
                                  sell_cost_bps=sell_bps,
                                  strategy=STRATEGY_POSITIVE_TOPN_PRORATA)
    d = daily.set_index("date")
    assert d.loc[pd.Timestamp(D1), "cost"] == pytest.approx(1.0 * buy_c)   # buys 0.75+0.25
    # day 2: A 0.75->0.25 (sell 0.5), B 0.25->0.75 (buy 0.5)
    assert d.loc[pd.Timestamp(D2), "cost"] == pytest.approx(0.5 * buy_c + 0.5 * sell_c)


def test_quick_equal_topn_unchanged_by_new_arg():
    # default strategy must reproduce the legacy equal-weight accounting exactly.
    rA, rB = np.log1p(0.10), np.log1p(-0.20)
    scores = mk_scores([(D1, "A", 3.0, rA), (D1, "B", 1.0, rB)])
    m_def, d_def = backtest_long_only(scores, top_n=2, buy_cost_bps=15, sell_cost_bps=25)
    m_eq, d_eq = backtest_long_only(scores, top_n=2, buy_cost_bps=15, sell_cost_bps=25,
                                    strategy=STRATEGY_EQUAL_TOPN)
    assert d_def.iloc[0]["gross"] == pytest.approx(0.5 * 0.10 + 0.5 * -0.20)
    assert d_def.iloc[0]["gross"] == pytest.approx(d_eq.iloc[0]["gross"])
    assert d_def.iloc[0]["cost"] == pytest.approx(d_eq.iloc[0]["cost"])

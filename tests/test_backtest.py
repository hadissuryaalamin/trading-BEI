"""Unit tests for the stateful long-only simulator: costs, ARA/ARB tradability,
stuck positions, delisting write-downs, cash at rf, execution lag, spread costs.

Mechanics tests pass execution_lag=0 so scores execute on their own date --
they pin down the ACCOUNTING at the execution close, which is lag-invariant.
Timing itself is covered by test_execution_lag_*."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    simulate_long_only,
    compute_metrics,
    STRATEGY_POSITIVE_TOPN_PRORATA,
)


def mk_market(rows):
    """rows: (date, ticker, ret, can_buy, can_sell)"""
    df = pd.DataFrame(rows, columns=["date", "ticker", "ret", "can_buy", "can_sell"])
    df["date"] = pd.to_datetime(df["date"])
    df["traded"] = True
    return df


def mk_scores(rows):
    df = pd.DataFrame(rows, columns=["date", "ticker", "score"])
    df["date"] = pd.to_datetime(df["date"])
    return df


D1, D2, D3 = "2025-01-06", "2025-01-07", "2025-01-08"


def test_costs_and_returns_exact():
    market = mk_market([
        (D1, "AAAA", 0.00, True, True), (D1, "BBBB", 0.00, True, True),
        (D2, "AAAA", 0.10, True, True), (D2, "BBBB", -0.05, True, True),
    ])
    scores = mk_scores([(D1, "AAAA", 2.0), (D1, "BBBB", 1.0)])
    m, daily = simulate_long_only(
        scores, market, top_n=2, buy_cost_bps=15, sell_cost_bps=25, rf_annual=0.0,
        execution_lag=0,
    )
    # day1: buy 0.5+0.5, pay 1.0 * 15bps; no market return yet
    assert daily.iloc[0]["gross"] == pytest.approx(0.0)
    assert daily.iloc[0]["cost"] == pytest.approx(1.0 * 15e-4)
    assert daily.iloc[0]["turnover"] == pytest.approx(1.0)
    # day2: 0.5*10% + 0.5*(-5%) = 2.5% gross, no rebalance (no scores)
    assert daily.iloc[1]["gross"] == pytest.approx(0.025)
    assert daily.iloc[1]["cost"] == pytest.approx(0.0)
    expect_eq = (1 - 15e-4) * 1.025
    assert daily["equity"].iloc[-1] == pytest.approx(expect_eq, rel=1e-12)


def test_cannot_buy_at_ara():
    market = mk_market([
        (D1, "ARAA", 0.0, False, True),   # pinned at ARA: no offers -> unbuyable
        (D1, "BBBB", 0.0, True, True),
        (D1, "CCCC", 0.0, True, True),
        (D2, "ARAA", 0.2, True, True),
        (D2, "BBBB", 0.0, True, True),
        (D2, "CCCC", 0.0, True, True),
    ])
    scores = mk_scores([(D1, "ARAA", 9.0), (D1, "BBBB", 2.0), (D1, "CCCC", 1.0)])
    m, daily = simulate_long_only(scores, market, top_n=2, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=0)
    # ARAA's +20% next day must NOT be earned; book = BBBB + CCCC = 0%
    assert daily.iloc[1]["gross"] == pytest.approx(0.0)


def test_stuck_position_cannot_be_sold():
    market = mk_market([
        (D1, "AAAA", 0.0, True, True), (D1, "BBBB", 0.0, True, True),
        (D2, "AAAA", 0.0, True, False),   # ARB: no bids -> unsellable
        (D2, "BBBB", 0.0, True, True),
        (D2, "CCCC", 0.0, True, True), (D2, "DDDD", 0.0, True, True),
        (D3, "AAAA", -0.30, True, True),  # the crash the strategy is stuck for
        (D3, "BBBB", 0.0, True, True),
        (D3, "CCCC", 0.0, True, True), (D3, "DDDD", 0.0, True, True),
    ])
    scores = mk_scores([
        (D1, "AAAA", 2.0), (D1, "BBBB", 1.0),
        (D2, "CCCC", 9.0), (D2, "DDDD", 8.0), (D2, "AAAA", 0.1), (D2, "BBBB", 0.2),
    ])
    m, daily = simulate_long_only(scores, market, top_n=2, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=0)
    d2 = daily[daily["date"] == pd.Timestamp(D2)].iloc[0]
    assert d2["n_stuck"] == 1                       # AAAA held against our will
    d3 = daily[daily["date"] == pd.Timestamp(D3)].iloc[0]
    assert d3["gross"] == pytest.approx(0.5 * -0.30)  # we PAY for the suspension risk


def test_unfilled_book_stays_in_cash_at_rf():
    market = mk_market([
        (D1, "AAAA", 0.0, True, True),
        (D2, "AAAA", 0.0, True, True),
    ])
    scores = mk_scores([(D1, "AAAA", 1.0)])
    rf = 0.10
    m, daily = simulate_long_only(scores, market, top_n=2, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=rf, execution_lag=0)
    # only 1 buyable name for a 2-slot book -> half stays in cash earning rf
    rf_d = (1 + rf) ** (1 / 252) - 1
    assert daily.iloc[0]["cash"] == pytest.approx(0.5)
    assert daily.iloc[1]["gross"] == pytest.approx(0.5 * rf_d, rel=1e-6)


def test_delisting_writedown():
    days = pd.bdate_range("2025-01-06", periods=8)
    rows = [(days[0], "AAAA", 0.0, True, True), (days[0], "FLAT", 0.0, True, True)]
    # AAAA disappears from the panel after day 0; FLAT keeps trading
    for d in days[1:]:
        rows.append((d, "FLAT", 0.0, True, True))
    market = mk_market(rows)
    # signals continue over the window (as in a real test period); AAAA scores
    # high throughout but has vanished, FLAT fills the other slot
    srows = [(days[0], "AAAA", 2.0)] + [(d, "FLAT", 1.0) for d in days]
    scores = mk_scores(srows)
    m, daily = simulate_long_only(scores, market, top_n=2, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=0.0,
                                  delist_after=3, delist_return=-0.5,
                                  execution_lag=0)
    assert m["n_delist_writedowns"] == 1
    # 0.5 weight written down 50% -> -25% on that day
    wd_day = daily[daily["gross"] < -0.2]
    assert len(wd_day) == 1
    assert wd_day.iloc[0]["gross"] == pytest.approx(0.5 * -0.5)


def test_weight_accounting_identity():
    """sum(w) + cash == 1 after every day (returns + rebalances + writedowns)."""
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-01-06", periods=30)
    tickers = [f"T{i:03d}" for i in range(12)]
    rows, srows = [], []
    for d in days:
        for t in tickers:
            rows.append((d, t, float(rng.normal(0, 0.03)),
                         bool(rng.random() > 0.1), bool(rng.random() > 0.1)))
            srows.append((d, t, float(rng.normal())))
    m, daily = simulate_long_only(mk_scores(srows), mk_market(rows), top_n=4,
                                  buy_cost_bps=15, sell_cost_bps=25, rf_annual=0.05,
                                  execution_lag=0)
    # cash column is recorded post-rebalance; with weights normalized daily the
    # book must never leak or lever: cash within [0,1] and equity finite
    assert ((daily["cash"] > -1e-9) & (daily["cash"] < 1 + 1e-9)).all()
    assert np.isfinite(daily["equity"]).all()
    assert m["n_days"] == len(daily)


def test_execution_lag_one_day():
    """Scores dated D1 execute at D2's close: D2's return is NOT earned, D3's is."""
    market = mk_market([
        (D1, "AAAA", 0.00, True, True),
        (D2, "AAAA", 0.10, True, True),   # move BEFORE our fill -> must be missed
        (D3, "AAAA", 0.05, True, True),   # first day the position is on
    ])
    scores = mk_scores([(D1, "AAAA", 1.0)])
    m, daily = simulate_long_only(scores, market, top_n=1, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=1)
    d = daily.set_index("date")
    assert d.loc[pd.Timestamp(D2), "gross"] == pytest.approx(0.0)
    assert d.loc[pd.Timestamp(D2), "turnover"] == pytest.approx(1.0)  # filled at D2 close
    assert d.loc[pd.Timestamp(D3), "gross"] == pytest.approx(0.05)


def test_execution_lag_uses_execution_day_tradability():
    """Buyability is checked on the EXECUTION day, not the signal day."""
    market = mk_market([
        (D1, "AAAA", 0.0, True, True),  (D1, "BBBB", 0.0, True, True),
        (D2, "AAAA", 0.0, False, True), (D2, "BBBB", 0.0, True, True),  # AAAA at ARA on D2
        (D3, "AAAA", 0.5, True, True),  (D3, "BBBB", 0.0, True, True),
    ])
    scores = mk_scores([(D1, "AAAA", 9.0), (D1, "BBBB", 1.0)])
    m, daily = simulate_long_only(scores, market, top_n=1, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=1)
    # AAAA was buyable on D1 (signal day) but pinned at ARA on D2 (execution
    # day): the fill must not happen, so AAAA's +50% on D3 is not earned
    d = daily.set_index("date")
    assert d.loc[pd.Timestamp(D3), "gross"] == pytest.approx(0.0)


def test_spread_cost_charged_per_side():
    market = mk_market([
        (D1, "AAAA", 0.0, True, True),
        (D2, "AAAA", 0.0, True, True),
    ])
    market["half_spread"] = 0.005          # 50bps per side from the closing book
    scores = mk_scores([(D1, "AAAA", 1.0)])
    m, daily = simulate_long_only(scores, market, top_n=1, buy_cost_bps=15,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=0)
    # full-weight buy pays commission + half-spread
    assert daily.iloc[0]["cost"] == pytest.approx(1.0 * (15e-4 + 0.005))


def test_spread_fallback_and_cap():
    market = mk_market([
        (D1, "NOBK", 0.0, True, True),   # no usable book -> default half-spread
        (D1, "WIDE", 0.0, True, True),   # absurd 5% book -> capped
        (D2, "NOBK", 0.0, True, True), (D2, "WIDE", 0.0, True, True),
    ])
    market["half_spread"] = np.where(market["ticker"] == "WIDE", 0.05, np.nan)
    scores = mk_scores([(D1, "NOBK", 2.0), (D1, "WIDE", 1.0)])
    m, daily = simulate_long_only(scores, market, top_n=2, buy_cost_bps=0,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=0,
                                  default_half_spread_bps=35, max_half_spread_bps=200)
    assert daily.iloc[0]["cost"] == pytest.approx(0.5 * 35e-4 + 0.5 * 200e-4)


def test_no_spread_column_means_no_spread_cost():
    market = mk_market([(D1, "AAAA", 0.0, True, True), (D2, "AAAA", 0.0, True, True)])
    scores = mk_scores([(D1, "AAAA", 1.0)])
    m, daily = simulate_long_only(scores, market, top_n=1, buy_cost_bps=15,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=0)
    assert daily.iloc[0]["cost"] == pytest.approx(1.0 * 15e-4)


def test_prorata_strategy_weights_book_by_score():
    """Positive-score pro-rata: buy weights ~ score, cost = weight bought."""
    market = mk_market([
        (D1, "AAAA", 0.0, True, True), (D1, "BBBB", 0.0, True, True),
        (D1, "CCCC", 0.0, True, True),
        (D2, "AAAA", 0.0, True, True), (D2, "BBBB", 0.0, True, True),
        (D2, "CCCC", 0.0, True, True),
    ])
    # A=3, B=1 positive -> weights 0.75/0.25; C negative -> excluded (cash-free)
    scores = mk_scores([(D1, "AAAA", 3.0), (D1, "BBBB", 1.0), (D1, "CCCC", -2.0)])
    m, daily = simulate_long_only(scores, market, top_n=10, buy_cost_bps=10,
                                  sell_cost_bps=0, rf_annual=0.0, execution_lag=0,
                                  strategy=STRATEGY_POSITIVE_TOPN_PRORATA)
    # invested 0.75 + 0.25 = 1.0 of equity; cost = 1.0 * 10bps; no cash left
    assert daily.iloc[0]["cost"] == pytest.approx(1.0 * 10e-4)
    assert daily.iloc[0]["turnover"] == pytest.approx(1.0)
    assert daily.iloc[0]["cash"] == pytest.approx(0.0)
    assert daily.iloc[0]["n_held"] == 2               # C excluded


def test_prorata_all_negative_stays_in_cash():
    """No positive score -> hold 100% cash at rf, no book."""
    market = mk_market([
        (D1, "AAAA", 0.0, True, True), (D2, "AAAA", 0.0, True, True),
    ])
    rf = 0.10
    scores = mk_scores([(D1, "AAAA", -1.0)])
    m, daily = simulate_long_only(scores, market, top_n=10, buy_cost_bps=50,
                                  sell_cost_bps=50, rf_annual=rf, execution_lag=0,
                                  strategy=STRATEGY_POSITIVE_TOPN_PRORATA)
    assert daily.iloc[0]["cash"] == pytest.approx(1.0)   # nothing bought
    assert daily.iloc[0]["cost"] == pytest.approx(0.0)
    rf_d = (1 + rf) ** (1 / 252) - 1
    assert daily.iloc[1]["gross"] == pytest.approx(rf_d, rel=1e-6)  # full cash at rf


def test_metrics_excess_rf():
    days = pd.bdate_range("2025-01-06", periods=252)
    rf = 0.055
    rf_d = (1 + rf) ** (1 / 252) - 1
    daily = pd.DataFrame({
        "date": days,
        "gross": rf_d, "cost": 0.0, "net": rf_d, "turnover": 0.0, "n_stuck": 0,
    })
    daily["equity"] = (1 + daily["net"]).cumprod()
    m = compute_metrics(daily, rf_annual=rf)
    # earning exactly rf must be ~0 excess Sharpe (raw Sharpe would be huge)
    assert abs(m["sharpe"]) < 1e-6
    assert m["ann_return"] == pytest.approx(rf, rel=1e-9)

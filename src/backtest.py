"""Stateful long-only daily backtest with IDX execution realism.

Simulation model (all pure pandas/numpy -- no torch):

- Signals dated day t are executed at the close of t + `execution_lag`
  trading days (default 1). Features are computed FROM day-t closing data, so
  executing at that same close is not implementable; `execution_lag=0` keeps
  the old same-close (MOC) convention for signal-decay diagnostics only.
- Each execution day the strategy targets the top-N scored names among those
  it can actually BUY (traded, offers present -> not pinned at ARA), equal
  weight; tradability is evaluated on the EXECUTION day.
- Positions it wants to exit but CANNOT sell (suspended, or pinned at ARB with
  no bids) stay in the book ("stuck") and keep earning their subsequent
  returns -- the backtest pays for suspension risk instead of deleting it.
- A held name whose panel rows disappear (delisting) earns 0 while missing and
  is written down by `delist_return` after `delist_after` consecutive missing
  days, then removed.
- Costs: `buy_cost_bps` on weight bought, `sell_cost_bps` on weight sold
  (IDX: ~15bps commission per side + 10bps sales tax on sells -> 15/25 default),
  PLUS the name's own half-spread from the closing book on every fill (capped
  at `max_half_spread_bps`; `default_half_spread_bps` when the book is empty).
  Commission alone flatters IDX badly: the liquid-universe median spread is
  ~70bps, i.e. ~35bps per side on top of commission.
- Idle cash earns the risk-free rate (`rf_annual`, BI-rate-ish).

Weights are fractions of current equity; each day applies returns first (book
drifts), then rebalances on that day's scores. Costs are charged against the
day's net return (standard approximation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Portfolio strategies understood by target_weights / the backtests.
STRATEGY_EQUAL_TOPN = "long_only_equal_topn"
STRATEGY_POSITIVE_TOPN_PRORATA = "long_only_positive_topn_prorata"
# Pre-existing configs used the bare name; keep it as an equal-weight alias.
STRATEGY_LEGACY_ALIAS = "long_only"


def _canon_strategy(strategy: str) -> str:
    return STRATEGY_EQUAL_TOPN if strategy == STRATEGY_LEGACY_ALIAS else strategy


def target_weights(
    scores: pd.Series, top_n: int, strategy: str = STRATEGY_EQUAL_TOPN
) -> dict[str, float]:
    """Map a day's ticker->score into target portfolio weights.

    Pure and stateless: no costs, no tradability, no drift -- just the target
    book the strategy WANTS today. Weights are fractions of the invested book
    and sum to 1.0 when non-empty; an empty dict means "hold 100% cash".

    Strategies
    ----------
    long_only_equal_topn (default, legacy):
        the top-`top_n` names by score, equal weight (1/n). Kept for backward
        compatibility; the production equal-weight paths in the backtests do
        NOT route through here (they preserve their own conventions), so this
        branch exists mainly for dispatch and tests.

    long_only_positive_topn_prorata:
        keep only names with score > 0, take the highest `top_n` of those, and
        weight them PRO-RATA by score (w_i = score_i / sum score_j). Returns {}
        when no score is positive -> the day is spent in cash. Here "score > 0"
        means "above the cash anchor" (train_dlsa's allow_cash logit is fixed at
        0), NOT a predicted positive return -- see configs/README.md.
    """
    if scores is None or len(scores) == 0:
        return {}
    strategy = _canon_strategy(strategy)
    s = scores[np.isfinite(scores)].sort_values(ascending=False)

    if strategy == STRATEGY_EQUAL_TOPN:
        picks = s.head(top_n)
        if len(picks) == 0:
            return {}
        w = 1.0 / len(picks)
        return {str(t): w for t in picks.index}

    if strategy == STRATEGY_POSITIVE_TOPN_PRORATA:
        pos = s[s > 0].head(top_n)
        total = float(pos.sum())
        if len(pos) == 0 or total <= 0:
            return {}
        return {str(t): float(v) / total for t, v in pos.items()}

    raise ValueError(f"unknown portfolio strategy: {strategy!r}")


def simulate_long_only(
    scores: pd.DataFrame,
    market: pd.DataFrame,
    top_n: int = 10,
    buy_cost_bps: float = 15.0,
    sell_cost_bps: float = 25.0,
    rf_annual: float = 0.055,
    delist_after: int = 20,
    delist_return: float = -0.5,
    execution_lag: int = 1,
    default_half_spread_bps: float = 35.0,
    max_half_spread_bps: float = 200.0,
    strategy: str = STRATEGY_EQUAL_TOPN,
) -> tuple[dict, pd.DataFrame]:
    """Run the simulator. scores: [date, ticker, score]; market: see build_market.

    `strategy` selects the target book each rebalance (see target_weights):
    the legacy equal-weight top-N (default) or positive-score top-N pro-rata.
    Both share the same tradability / stuck / delisting / cost machinery below.

    Scores dated t are executed at the close `execution_lag` trading days
    later (on the market calendar). Spread costs come from the market's
    `half_spread` column when present; rows without a usable book fall back to
    `default_half_spread_bps`, and pathological books are capped at
    `max_half_spread_bps`. A market with no `half_spread` column charges no
    spread (commission-only), which keeps synthetic-market tests exact.

    Returns (metrics-ready daily DataFrame is the second element):
        daily columns: date, gross, cost, net, turnover, n_held, n_stuck, cash,
                       equity
    """
    buy_c, sell_c = buy_cost_bps / 1e4, sell_cost_bps / 1e4
    rf_d = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0

    scores = scores[["date", "ticker", "score"]].copy()
    if execution_lag:
        cal = np.sort(market["date"].unique())
        pos = np.searchsorted(cal, scores["date"].to_numpy())
        tgt = pos + execution_lag
        ok = tgt < len(cal)
        scores = scores.loc[ok].assign(date=cal[tgt[ok]])
    if scores.empty:
        raise ValueError("no executable scores (all past the end of the market calendar)")

    has_spread = "half_spread" in market.columns
    default_hs = default_half_spread_bps / 1e4 if has_spread else 0.0
    max_hs = max_half_spread_bps / 1e4

    score_by_day = {pd.Timestamp(d): g for d, g in scores.groupby("date")}
    first_day = scores["date"].min()
    last_day = pd.Timestamp(scores["date"].max())

    m = market[market["date"] >= first_day].sort_values("date")
    all_days = [pd.Timestamp(d) for d in m["date"].unique()]  # sorted calendar
    # simulate one day past the last signal so its position return is realized
    idx_last = int(np.searchsorted(np.array(all_days), last_day))
    sim_days = all_days[: min(idx_last + 2, len(all_days))]

    # fast per-day lookups
    by_day: dict = {
        pd.Timestamp(d): {
            "ret": dict(zip(g["ticker"], g["ret"])),
            "can_buy": set(g.loc[g["can_buy"], "ticker"]),
            "can_sell": set(g.loc[g["can_sell"], "ticker"]),
            "hs": dict(zip(g["ticker"], g["half_spread"])) if has_spread else {},
        }
        for d, g in m[m["date"].isin(sim_days)].groupby("date")
    }

    w: dict[str, float] = {}          # ticker -> weight (fraction of equity)
    cash = 1.0
    target_set: set[str] = set()      # latest wanted book (for stuck counting)
    missing: dict[str, int] = {}      # consecutive days without a panel row
    n_writedowns = 0
    rows = []

    for d in sim_days:
        day = by_day.get(d, {"ret": {}, "can_buy": set(), "can_sell": set(), "hs": {}})
        rets, can_buy, can_sell = day["ret"], day["can_buy"], day["can_sell"]
        hs_day = day["hs"]

        def hs(t):
            """Per-side spread cost for ticker t at today's close."""
            v = hs_day.get(t)
            return min(v, max_hs) if v is not None and np.isfinite(v) else default_hs

        # ---- 1. mark the book: apply day d returns to yesterday's holdings ----
        gross = cash * rf_d
        cash *= 1.0 + rf_d
        for t in list(w):
            r = rets.get(t)
            if r is None:  # no row today: stale, possibly delisting
                missing[t] = missing.get(t, 0) + 1
                if missing[t] >= delist_after:
                    gross += w[t] * delist_return
                    cash += w.pop(t) * (1.0 + delist_return)  # forced realization
                    n_writedowns += 1
                continue
            missing.pop(t, None)
            gross += w[t] * r
            w[t] *= 1.0 + r
        total = 1.0 + gross
        for t in w:
            w[t] /= total
        cash /= total

        # ---- 2. rebalance at the close on today's scores ----
        cost = turnover = 0.0
        if d in score_by_day:
            g = score_by_day[d]
            # Only names we can actually BUY today are eligible for the target.
            buyable = g[g["ticker"].isin(can_buy)]
            if strategy == STRATEGY_POSITIVE_TOPN_PRORATA:
                # pro-rata weights among positive scores (empty -> all cash)
                tw = target_weights(
                    buyable.set_index("ticker")["score"], top_n, strategy
                )
                target = list(tw)                       # score-desc order
            else:
                # legacy equal-weight top-N: split the investable book 1/top_n,
                # so an underfilled book leaves the remainder in cash (unchanged).
                ranked = buyable.sort_values("score", ascending=False)["ticker"]
                target = list(ranked.head(top_n))
                tw = None
            target_set = set(target)

            # exits: sell what we can; what we can't sell stays (stuck)
            for t in list(w):
                if t in target_set:
                    continue
                if t in can_sell:
                    sold = w.pop(t)
                    cash += sold
                    cost += sold * (sell_c + hs(t))
                    turnover += sold
            stuck_w = sum(v for t, v in w.items() if t not in target_set)
            avail = max(0.0, 1.0 - stuck_w)             # capital free to deploy

            # move each target name to its weight (pro-rata * avail, or 1/top_n)
            for t in target:
                w_target = avail * tw[t] if tw is not None else avail / top_n
                cur = w.get(t, 0.0)
                delta = w_target - cur
                if delta > 1e-12:
                    buy = min(delta, max(cash, 0.0))
                    if buy > 0:
                        w[t] = cur + buy
                        cash -= buy
                        cost += buy * (buy_c + hs(t))
                        turnover += buy
                elif delta < -1e-12 and t in can_sell:
                    w[t] = w_target
                    cash += -delta
                    cost += -delta * (sell_c + hs(t))
                    turnover += -delta

        net = gross - cost
        n_stuck = sum(1 for t in w if t not in target_set)
        rows.append((d, gross, cost, net, turnover, len(w), n_stuck, cash))

    daily = pd.DataFrame(
        rows, columns=["date", "gross", "cost", "net", "turnover", "n_held", "n_stuck", "cash"]
    )
    daily["equity"] = (1.0 + daily["net"]).cumprod()
    metrics = compute_metrics(daily, rf_annual=rf_annual)
    metrics["n_delist_writedowns"] = n_writedowns
    return metrics, daily


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(daily: pd.DataFrame, rf_annual: float = 0.055, ann: int = TRADING_DAYS) -> dict:
    """Performance metrics from the daily net return series.

    `sharpe` is computed on returns in EXCESS of the risk-free rate -- with
    Indonesian rf around 5-6% a raw-return Sharpe flatters any long-only book.
    """
    r = daily["net"].to_numpy()
    if len(r) == 0:
        return {}
    rf_d = (1.0 + rf_annual) ** (1.0 / ann) - 1.0
    eq = daily["equity"].to_numpy()
    total = float(eq[-1] - 1.0)
    ann_ret = float(eq[-1] ** (ann / len(r)) - 1.0)  # geometric
    vol = float(np.std(r, ddof=0))
    sharpe = float((np.mean(r) - rf_d) / (vol + 1e-12) * np.sqrt(ann))
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    return {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": float(vol * np.sqrt(ann)),
        "sharpe": sharpe,                      # excess of rf
        "sharpe_raw": float(np.mean(r) / (vol + 1e-12) * np.sqrt(ann)),
        "max_drawdown": max_dd,
        "win_rate": float((r > 0).mean()),
        "avg_daily_net": float(np.mean(r)),
        "avg_daily_cost": float(daily["cost"].mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "avg_n_stuck": float(daily["n_stuck"].mean()),
        "n_days": int(len(r)),
        "rf_annual": rf_annual,
    }


def benchmark_relative_metrics(
    daily: pd.DataFrame, ihsg_daily: pd.Series, rf_annual: float = 0.055, ann: int = TRADING_DAYS
) -> dict:
    """Alpha/beta/IR of the strategy vs the IHSG proxy over the same dates.

    Long-only books are beta-dominated; a raw Sharpe can be pure market
    exposure. Alpha (CAPM, vs the cap-weighted proxy) and the information
    ratio are the numbers that say whether the model adds anything.
    """
    s = ihsg_daily.reindex(pd.DatetimeIndex(pd.to_datetime(daily["date"])))
    r = daily["net"].to_numpy()
    b = s.to_numpy()
    ok = np.isfinite(b)
    r, b = r[ok], b[ok]
    if len(r) < 20:
        return {"beta": float("nan"), "alpha_ann": float("nan"), "info_ratio": float("nan")}
    rf_d = (1.0 + rf_annual) ** (1.0 / ann) - 1.0
    var = np.var(b, ddof=0)
    beta = float(np.cov(r, b, ddof=0)[0, 1] / (var + 1e-18))
    alpha_d = np.mean(r) - rf_d - beta * (np.mean(b) - rf_d)
    active = r - b
    ir = float(np.mean(active) / (np.std(active, ddof=0) + 1e-12) * np.sqrt(ann))
    return {
        "beta": beta,
        "alpha_ann": float(alpha_d * ann),
        "info_ratio": ir,
    }

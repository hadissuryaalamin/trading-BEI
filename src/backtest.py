"""Stateful long-only daily backtest with IDX execution realism.

Simulation model (all pure pandas/numpy -- no torch):

- Signals at the close of day t are executed AT that close (MOC assumption --
  the standard close-to-close research convention; documented, not hidden).
- Each day the strategy targets the top-N scored names among those it can
  actually BUY (traded, offers present -> not pinned at ARA), equal weight.
- Positions it wants to exit but CANNOT sell (suspended, or pinned at ARB with
  no bids) stay in the book ("stuck") and keep earning their subsequent
  returns -- the backtest pays for suspension risk instead of deleting it.
- A held name whose panel rows disappear (delisting) earns 0 while missing and
  is written down by `delist_return` after `delist_after` consecutive missing
  days, then removed.
- Costs: `buy_cost_bps` on weight bought, `sell_cost_bps` on weight sold
  (IDX: ~15bps commission per side + 10bps sales tax on sells -> 15/25 default).
- Idle cash earns the risk-free rate (`rf_annual`, BI-rate-ish).

Weights are fractions of current equity; each day applies returns first (book
drifts), then rebalances on that day's scores. Costs are charged against the
day's net return (standard approximation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate_long_only(
    scores: pd.DataFrame,
    market: pd.DataFrame,
    top_n: int = 10,
    buy_cost_bps: float = 15.0,
    sell_cost_bps: float = 25.0,
    rf_annual: float = 0.055,
    delist_after: int = 20,
    delist_return: float = -0.5,
) -> tuple[dict, pd.DataFrame]:
    """Run the simulator. scores: [date, ticker, score]; market: see build_market.

    Returns (metrics-ready daily DataFrame is the second element):
        daily columns: date, gross, cost, net, turnover, n_held, n_stuck, cash,
                       equity
    """
    buy_c, sell_c = buy_cost_bps / 1e4, sell_cost_bps / 1e4
    rf_d = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0

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
        day = by_day.get(d, {"ret": {}, "can_buy": set(), "can_sell": set()})
        rets, can_buy, can_sell = day["ret"], day["can_buy"], day["can_sell"]

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
            ranked = g.sort_values("score", ascending=False)["ticker"]
            target = []
            for t in ranked:
                if t in can_buy:
                    target.append(t)
                if len(target) == top_n:
                    break
            target_set = set(target)

            # exits: sell what we can; what we can't sell stays (stuck)
            for t in list(w):
                if t in target_set:
                    continue
                if t in can_sell:
                    sold = w.pop(t)
                    cash += sold
                    cost += sold * sell_c
                    turnover += sold
            stuck_w = sum(v for t, v in w.items() if t not in target_set)

            # equal-weight the target book with whatever isn't stuck
            w_star = max(0.0, 1.0 - stuck_w) / top_n
            for t in target:
                cur = w.get(t, 0.0)
                delta = w_star - cur
                if delta > 1e-12:
                    buy = min(delta, max(cash, 0.0))
                    if buy > 0:
                        w[t] = cur + buy
                        cash -= buy
                        cost += buy * buy_c
                        turnover += buy
                elif delta < -1e-12 and t in can_sell:
                    w[t] = w_star
                    cash += -delta
                    cost += -delta * sell_c
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

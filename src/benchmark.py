"""IHSG (composite index) proxy benchmark, reconstructed from the panel itself.

The proxy is a market-cap-weighted daily return of the whole cross-section using
PREVIOUS-day weights (so it is causal / has no look-ahead). It prefers IDX's own
`weight_for_index` column when present, else falls back to close * listed_shares.

Purpose: a long-only strategy is exposed to market beta, so a high Sharpe alone
can still lose to simply buying and holding the market. This benchmark is the
minimum bar -- the strategy must beat IHSG buy-and-hold (ABLATION_PLAN sec 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import adjusted_log_return

_KEYS = ("total_return", "ann_return", "ann_vol", "sharpe", "n_days")


def ihsg_proxy_returns(panel: pd.DataFrame) -> pd.Series:
    """Cap-weighted daily market return (IHSG proxy), indexed by date.

    Weight for day t is the stock's market cap as of day t-1, so the return earned
    on day t uses only information available before t.
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)
    # split-adjusted simple daily return (close / reported Previous); a raw
    # pct_change would book every stock split as a huge fake index move
    ret = np.expm1(adjusted_log_return(df))

    if "weight_for_index" in df.columns and df["weight_for_index"].gt(0).any():
        cap = df["weight_for_index"].astype(float)
    elif "listed_shares" in df.columns:
        cap = df["close"] * df["listed_shares"]
    else:
        cap = pd.Series(1.0, index=df.index)
    w_prev = cap.groupby(df["ticker"], sort=False).shift(1)  # yesterday's weight (causal)

    valid = ret.notna() & w_prev.notna() & (w_prev > 0)
    tmp = pd.DataFrame({"date": df["date"], "ret": ret, "w": w_prev})[valid]
    num = (tmp["ret"] * tmp["w"]).groupby(tmp["date"]).sum()
    den = tmp["w"].groupby(tmp["date"]).sum()
    daily = (num / den).sort_index()
    daily.name = "ihsg_return"
    return daily


def benchmark_metrics(ihsg_daily: pd.Series, dates=None, ann: int = 252, prefix: str = "ihsg_") -> dict:
    """Backtest-style metrics for the IHSG proxy, optionally restricted to `dates`.

    Pass the portfolio's test dates so the benchmark is measured over exactly the
    same window (same annualization convention as train_test._metrics).
    """
    s = ihsg_daily
    if dates is not None:
        s = s[s.index.isin(pd.DatetimeIndex(pd.to_datetime(list(dates))))]
    r = s.to_numpy()
    if len(r) == 0:
        return {f"{prefix}{k}": float("nan") for k in _KEYS}
    eq = np.cumprod(1 + r)
    return {
        f"{prefix}total_return": float(eq[-1] - 1),
        f"{prefix}ann_return": float((1 + np.mean(r)) ** ann - 1),
        f"{prefix}ann_vol": float(np.std(r, ddof=0) * np.sqrt(ann)),
        f"{prefix}sharpe": float(np.mean(r) / (np.std(r, ddof=0) + 1e-12) * np.sqrt(ann)),
        f"{prefix}n_days": int(len(r)),
    }


def ihsg_equity(ihsg_daily: pd.Series, dates=None) -> pd.DataFrame:
    """IHSG proxy equity curve (buy-and-hold), as DataFrame[date, ihsg_equity]."""
    s = ihsg_daily
    if dates is not None:
        s = s[s.index.isin(pd.DatetimeIndex(pd.to_datetime(list(dates))))]
    s = s.sort_index()
    return pd.DataFrame({"date": s.index, "ihsg_equity": np.cumprod(1 + s.to_numpy())})


def plot_equity_vs_ihsg(daily: pd.DataFrame, ihsg_daily: pd.Series, path, title: str = "") -> None:
    """Save a portfolio-equity vs IHSG-buy-and-hold plot over the portfolio dates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench = ihsg_equity(ihsg_daily, dates=daily["date"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pd.to_datetime(daily["date"]), daily["equity"], label="strategy (long-only top-N)")
    ax.plot(pd.to_datetime(bench["date"]), bench["ihsg_equity"], label="IHSG proxy (buy & hold)", ls="--")
    ax.set_title(title or "Strategy vs IHSG (test period)")
    ax.set_ylabel("equity (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

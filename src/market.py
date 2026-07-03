"""Market-side data derived from the panel: liquid universe + tradability matrix.

Two concerns live here, both about what you can ACTUALLY trade on IDX:

1. Universe eligibility (`apply_universe`) -- a causal liquidity screen. A
   (ticker, day) is eligible when its trailing-20-day median traded value is at
   least `min_value_idr`. Without this, a top-N picker on ~900 names loads up
   on micro-caps whose prints can't absorb a position ("saham gorengan").

2. Tradability (`build_market`) -- IDX auto-rejection (ARA/ARB) and suspensions
   read directly off the end-of-day book: a stock pinned at ARA has no offers
   (offer == 0 -> you cannot buy at the close), one at ARB has no bids
   (bid == 0 -> you cannot sell), and a non-traded stock (volume == 0) has a
   stale close. The backtest consumes these flags plus split-adjusted daily
   simple returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess import adjusted_log_return


def eligibility(
    panel: pd.DataFrame,
    min_value_idr: float = 1_000_000_000,
    window: int = 20,
    min_periods: int = 5,
) -> pd.Series:
    """Boolean per panel row: is this (ticker, day) in the liquid universe?

    Causal: uses the trailing `window`-day median of daily traded value up to
    and including the current day (known at the close, when we trade).
    """
    df = panel.sort_values(["ticker", "date"])
    med = (
        df.groupby("ticker", sort=False)["value"]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).median())
    )
    elig = med >= float(min_value_idr)
    return elig.reindex(panel.index).fillna(False)


def apply_universe(features: pd.DataFrame, panel: pd.DataFrame, ucfg: dict | None) -> pd.DataFrame:
    """AND the liquidity screen into features.valid_day (returns a copy).

    `ucfg` is the config's `universe` block, e.g. {min_value_idr: 1e9, window: 20}.
    min_value_idr <= 0 disables the screen (universe = everything that traded).
    """
    ucfg = ucfg or {}
    min_value = float(ucfg.get("min_value_idr", 0) or 0)
    out = features.copy()
    if min_value <= 0:
        return out
    elig = eligibility(panel, min_value, window=int(ucfg.get("window", 20)))
    key = pd.MultiIndex.from_frame(panel[["date", "ticker"]])
    elig_map = pd.Series(elig.to_numpy(), index=key)
    fkey = pd.MultiIndex.from_frame(out[["date", "ticker"]])
    out["valid_day"] = out["valid_day"].to_numpy() & elig_map.reindex(fkey).fillna(False).to_numpy()
    n = out["valid_day"].sum()
    print(f"universe: min_value_idr={min_value:,.0f} -> {n:,}/{len(out):,} tradable stock-days")
    return out


def build_market(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (date, ticker) execution-relevant facts for the backtest simulator.

    Returns columns:
        ret      : split-adjusted simple daily return (stale/no-trade days ~ 0)
        traded   : volume > 0 that day
        can_buy  : traded and offers exist at the close (not pinned at ARA)
        can_sell : traded and bids exist at the close (not pinned at ARB)
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    ret = np.expm1(adjusted_log_return(df))
    traded = (df["volume"].fillna(0) > 0) if "volume" in df.columns else pd.Series(True, index=df.index)
    offer = df["offer"].fillna(0) if "offer" in df.columns else pd.Series(1.0, index=df.index)
    bid = df["bid"].fillna(0) if "bid" in df.columns else pd.Series(1.0, index=df.index)
    out = pd.DataFrame(
        {
            "date": df["date"],
            "ticker": df["ticker"],
            "ret": ret.fillna(0.0),
            "traded": traded.to_numpy(),
            "can_buy": (traded & (offer > 0)).to_numpy(),
            "can_sell": (traded & (bid > 0)).to_numpy(),
        }
    )
    return out.reset_index(drop=True)

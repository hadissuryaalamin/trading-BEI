"""Shared synthetic-panel builder for the test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_panel(tickers: dict[str, dict], n_days: int = 40, start="2024-01-01") -> pd.DataFrame:
    """Build a synthetic IDX-style panel.

    tickers: name -> spec dict with optional keys:
        prices     : list/array of closes (len n_days); default flat 100
        volume     : scalar or array; default 1e6
        value      : scalar or array; default close * volume
        skip_days  : set of day indices with NO row at all (delisted/suspended-off-board)
        no_trade   : set of day indices with volume 0 (stale close)
        split_day  : day index of a `ratio`:1 split (close divides; prev_close adjusted)
        ratio      : split ratio (default 2)
    """
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for tk, spec in tickers.items():
        prices = np.asarray(spec.get("prices", np.full(n_days, 100.0)), dtype=float)
        volume = np.broadcast_to(np.asarray(spec.get("volume", 1e6), dtype=float), (n_days,)).copy()
        split_day = spec.get("split_day")
        ratio = spec.get("ratio", 2.0)
        if split_day is not None:
            prices = prices.copy()
            prices[split_day:] = prices[split_day:] / ratio
        no_trade = set(spec.get("no_trade", ()))
        skip = set(spec.get("skip_days", ()))
        value = spec.get("value")
        prev = None
        for i, d in enumerate(dates):
            if i in skip:
                continue
            close = prices[i]
            if i in no_trade:
                vol = 0.0
                close = prev if prev is not None else close
            else:
                vol = volume[i]
            # IDX reports Previous ADJUSTED for corporate actions on the ex-date
            if prev is None:
                prev_close = close
            elif split_day is not None and i == split_day:
                prev_close = prev / ratio
            else:
                prev_close = prev
            val = (close * vol) if value is None else np.broadcast_to(np.asarray(value, dtype=float), (n_days,))[i]
            rows.append({
                "date": d, "ticker": tk, "prev_close": prev_close,
                "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                "volume": vol, "value": val, "frequency": 100.0,
                "foreign_buy": vol * 0.1, "foreign_sell": vol * 0.05,
                "listed_shares": 1e9, "tradeable_shares": 1e9,
                "bid": close * 0.995 if vol > 0 else 0.0,
                "offer": close * 1.005 if vol > 0 else 0.0,
                "bid_volume": 1e4 if vol > 0 else 0.0,
                "offer_volume": 1e4 if vol > 0 else 0.0,
            })
            prev = close
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture
def simple_panel():
    rng = np.random.default_rng(0)
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 40)))
    return make_panel({
        "AAAA": {"prices": walk},
        "BBBB": {},
        "CCCC": {"prices": 100 + np.arange(40.0)},
    })

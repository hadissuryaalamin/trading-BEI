"""Tests for the IHSG proxy: cap weighting (close * shares), causality.

Regression test for a real bug: `weight_for_index` is IDX's float-adjusted
share COUNT, not a market cap. Using it directly as the weight share-weights
the index, letting trillion-share penny stocks (GOTO et al.) dominate -- the
proxy showed +53% over a window where the real IHSG fell ~20%.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.benchmark import ihsg_proxy_returns


def _mk_panel(rows):
    df = pd.DataFrame(
        rows,
        columns=["date", "ticker", "close", "prev_close", "volume", "value",
                 "weight_for_index"],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_proxy_is_cap_weighted_not_share_weighted():
    # BIGC: price 100, 1e6 index shares -> cap 1e8 (dominates by VALUE)
    # PENN: price 1,  1e7 index shares -> cap 1e7 (dominates by SHARE COUNT)
    d1, d2 = "2025-01-06", "2025-01-07"
    panel = _mk_panel([
        (d1, "BIGC", 100.0, 100.0, 1e6, 1e8, 1e6),
        (d1, "PENN", 1.00, 1.00, 1e7, 1e7, 1e7),
        (d2, "BIGC", 110.0, 100.0, 1e6, 1e8, 1e6),   # +10%
        (d2, "PENN", 0.90, 1.00, 1e7, 1e7, 1e7),     # -10%
    ])
    r = ihsg_proxy_returns(panel)
    # cap-weighted: (1e8 * 0.10 + 1e7 * -0.10) / 1.1e8 = +8.18%
    # share-weighted (the bug) would give (1e6*0.10 + 1e7*-0.10)/1.1e7 = -8.18%
    assert r.loc[pd.Timestamp(d2)] == pytest.approx((1e8 * 0.10 - 1e7 * 0.10) / 1.1e8)


def test_proxy_uses_previous_day_weights():
    # A stock that triples on day 2 must not get its day-2 cap as its own weight
    d1, d2 = "2025-01-06", "2025-01-07"
    panel = _mk_panel([
        (d1, "AAAA", 100.0, 100.0, 1e6, 1e8, 1e6),
        (d1, "BBBB", 100.0, 100.0, 1e6, 1e8, 1e6),
        (d2, "AAAA", 300.0, 100.0, 1e6, 1e8, 1e6),   # +200%
        (d2, "BBBB", 100.0, 100.0, 1e6, 1e8, 1e6),   # 0%
    ])
    r = ihsg_proxy_returns(panel)
    # equal PREVIOUS-day caps -> simple average of returns, +100%
    assert r.loc[pd.Timestamp(d2)] == pytest.approx(1.0)

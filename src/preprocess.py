"""Turn the cleaned panel into model-ready features.

No factor model / residuals (unlike upstream DLSA): features are computed
directly from raw prices/volumes, all causal (no look-ahead), then normalized
cross-sectionally per day. Windowing into sequences happens in dataset.py.

Input : long panel DataFrame (date, ticker, open/high/low/close, volume, value,
        foreign_buy/sell, listed_shares, tradeable_shares, ...)
Output: long feature DataFrame (date, ticker, <features...>, target)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_return",
    "overnight_return",
    "hl_range",
    "log_volume",
    "log_value",
    "turnover",
    "foreign_flow_ratio",
]
TARGET_COLUMN = "fwd_return"
EPS = 1e-9


def compute_features(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Compute causal features + forward-return target from the raw panel.

    All per-asset series use groupby(ticker) so shifts never cross tickers.
    The target is the forward log return over `horizon` days (shift(-horizon)),
    used only as the label -- never as an input feature.
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)

    prev_close = g["close"].shift(1)
    df["log_return"] = np.log(df["close"] / prev_close.replace(0, np.nan))
    df["overnight_return"] = np.log(df["open"] / prev_close.replace(0, np.nan))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    df["log_value"] = np.log1p(df["value"].clip(lower=0))

    shares = df.get("tradeable_shares", df.get("listed_shares"))
    df["turnover"] = df["volume"] / (shares + EPS) if shares is not None else 0.0

    if {"foreign_buy", "foreign_sell"}.issubset(df.columns):
        df["foreign_flow_ratio"] = (df["foreign_buy"] - df["foreign_sell"]) / (
            df["volume"].abs() + 1.0
        )
    else:
        df["foreign_flow_ratio"] = 0.0

    df[TARGET_COLUMN] = g["close"].shift(-horizon)
    df[TARGET_COLUMN] = np.log(df[TARGET_COLUMN] / df["close"].replace(0, np.nan))

    keep = ["date", "ticker", *FEATURE_COLUMNS, TARGET_COLUMN]
    out = df[keep].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def normalize(
    features: pd.DataFrame,
    method: str = "cross_sectional_zscore",
    clip: float = 5.0,
) -> pd.DataFrame:
    """Standardize features cross-sectionally within each date.

    Uses only that day's cross-section, so it is causal and has no train-fit
    stats to leak -- matches the DLSA convention.
    """
    if method != "cross_sectional_zscore":
        raise ValueError(f"unknown normalize method: {method}")

    df = features.copy()
    grp = df.groupby("date", sort=False)[FEATURE_COLUMNS]
    mean = grp.transform("mean")
    std = grp.transform("std", ddof=0).replace(0, np.nan)
    df[FEATURE_COLUMNS] = ((df[FEATURE_COLUMNS] - mean) / std).fillna(0.0).clip(-clip, clip)
    return df

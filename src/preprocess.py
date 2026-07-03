"""Turn the cleaned panel into model-ready features, organized by economic group.

No factor model / residuals (unlike upstream DLSA): features are computed
directly from raw prices/volumes, all causal (no look-ahead), then normalized
cross-sectionally per day. Windowing into sequences happens in dataset.py.

Corporate actions & gaps (correctness-critical)
-----------------------------------------------
IDX's raw `close` is NOT adjusted for splits/reverse-splits, but the reported
`Previous` (prev_close) IS adjusted on the ex-date. Daily returns therefore use
`close / prev_close` (falling back to the lagged close when prev_close is
missing), which makes a 1:10 split a ~0% return instead of a fake -90%. An
adjusted log-price path (cumsum of these returns) drives all multi-day
quantities (momentum, forward target), so they are split-consistent too.
Cash dividends are NOT adjusted for (no dividend data in the summary); returns
are price returns.

The forward target is only defined when the NEXT calendar trading day (global
exchange calendar) has a row for the ticker AND the ticker actually trades on
it. A "next-day" return that silently spans a 3-week suspension, or marks a
stale price, would otherwise leak into training. `fwd_gap` records the actual
day distance for diagnostics.

Sample validity: `traded` (volume > 0 today) and `valid_day` (traded AND in the
liquid universe, see market.apply_universe) are carried through; the datasets
only emit samples where `valid_day` is True -- you can't buy a stock that
didn't trade, and scoring it anyway just adds stale-price noise.

Feature ablation (see ABLATION_PLAN.md)
--------------------------------------
`compute_features()` always computes the FULL superset of features (all groups).
Which groups/columns a given experiment actually feeds to the model is resolved
from config via `resolve_features()` and threaded through the datasets. Rows
are dropped only when the CORE feature (`log_return`) is missing, so the sample
set is invariant to which optional groups are active.

Input : long panel DataFrame (date, ticker, prev_close, open/high/low/close,
        volume, value, foreign_buy/sell, listed/tradeable_shares, bid/offer...)
Output: long feature DataFrame (date, ticker, <all features...>, target, meta)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- feature groups (economic dimensions) ----------------------------------
# Ablation is done per group, not per single feature. See ABLATION_PLAN.md sec 2.
FEATURE_GROUPS: dict[str, list[str]] = {
    "G1": ["log_return", "mom_5", "mom_10", "mom_20"],      # return & momentum
    "G2": ["hl_range", "roll_vol_20", "range_ratio"],        # volatility / range
    "G3": ["log_volume", "log_value", "turnover", "amihud"], # volume & liquidity
    "G4": ["foreign_flow_ratio", "foreign_roll_5"],          # foreign flow
    "G5": ["bid_ask_spread", "book_imbalance"],              # microstructure
}
# Full superset, in a stable order.
ALL_FEATURES: list[str] = [f for cols in FEATURE_GROUPS.values() for f in cols]

# Always-required feature: drives which (ticker, day) samples exist. Kept fixed
# across experiments so the sample set never depends on the active feature set.
CORE_FEATURES: list[str] = ["log_return"]

# Carried through to the datasets/backtest but never fed to the model.
META_COLUMNS: list[str] = ["traded", "valid_day", "fwd_gap"]

# Back-compat: modules that still import FEATURE_COLUMNS get the full superset.
FEATURE_COLUMNS = ALL_FEATURES
TARGET_COLUMN = "fwd_return"
EPS = 1e-9


def resolve_features(features_cfg: dict | None) -> list[str]:
    """Resolve the active feature list from a config's `features` block.

    Accepts either:
      - `feature_groups: [G1, G2, ...]`  -> expand groups (preferred for ablation)
      - `columns: [log_return, ...]`     -> explicit list
    Falls back to the full superset when neither is given. Order is preserved and
    duplicates are removed.
    """
    if not features_cfg:
        return list(ALL_FEATURES)

    if features_cfg.get("feature_groups"):
        cols: list[str] = []
        for grp in features_cfg["feature_groups"]:
            if grp not in FEATURE_GROUPS:
                raise ValueError(
                    f"unknown feature group: {grp!r} (have {list(FEATURE_GROUPS)})"
                )
            cols.extend(FEATURE_GROUPS[grp])
    elif features_cfg.get("columns"):
        cols = list(features_cfg["columns"])
        unknown = [c for c in cols if c not in ALL_FEATURES]
        if unknown:
            raise ValueError(f"unknown feature columns: {unknown}")
    else:
        return list(ALL_FEATURES)

    seen: dict[str, None] = {}
    for c in cols:
        seen.setdefault(c, None)
    return list(seen)


def _safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """log(num/den) with non-positive inputs mapped to NaN (no divide-by-zero).

    IDX rows for non-trading days carry zeros (open/high/low, sometimes close),
    which would otherwise feed 0 or a negative ratio into np.log and raise
    'divide by zero encountered in log'. Masking to NaN first avoids the warning
    and the -inf, and the row is later neutralized/dropped as appropriate.
    """
    num = num.where(num > 0)
    den = den.where(den > 0)
    return np.log(num / den)


def adjusted_log_return(df: pd.DataFrame) -> pd.Series:
    """Split-adjusted daily log return: log(close / Previous).

    IDX adjusts the reported `Previous` for corporate actions on the ex-date,
    so close/prev_close is continuous across splits while close/close.shift(1)
    is not. Falls back to the lagged close where prev_close is missing/zero.
    `df` must be sorted by (ticker, date).
    """
    lag_close = df.groupby("ticker", sort=False)["close"].shift(1)
    if "prev_close" in df.columns:
        base = df["prev_close"].where(df["prev_close"] > 0)
        base = base.fillna(lag_close)
    else:
        base = lag_close
    return _safe_log_ratio(df["close"], base)


def calendar_positions(dates: pd.Series, all_dates: np.ndarray | None = None) -> pd.Series:
    """Map each date to its position on the global exchange calendar (0..n-1)."""
    if all_dates is None:
        all_dates = np.sort(dates.unique())
    cal = pd.Series(np.arange(len(all_dates), dtype=np.int64), index=all_dates)
    return dates.map(cal).astype(np.int64)


def compute_features(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Compute the full causal feature superset + forward-return target.

    All per-asset series use groupby(ticker) so shifts/rollings never cross
    tickers. Return-based quantities come from the split-adjusted log-price
    path (see module docstring). The target is the forward log return over
    `horizon` STRICTLY CONSECUTIVE calendar trading days ending on a day the
    ticker actually trades; anything else (suspension gap, stale price,
    delisting) is NaN and never becomes a training label.
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    df["_pos"] = calendar_positions(df["date"])
    df["traded"] = (df["volume"].fillna(0) > 0) if "volume" in df.columns else True
    close = df["close"]

    # --- G1: return & momentum (from the adjusted log-price path) ---
    df["log_return"] = adjusted_log_return(df)
    # Adjusted log price: cumsum of adjusted daily log returns per ticker.
    # NaN returns (close<=0 rows) contribute 0 so the path stays defined.
    df["_alp"] = df["log_return"].fillna(0.0).groupby(df["ticker"], sort=False).cumsum()
    gt = df.groupby("ticker", sort=False)
    for k in (5, 10, 20):
        df[f"mom_{k}"] = df["_alp"] - gt["_alp"].shift(k)

    # --- G2: volatility / range ---
    hl_range = (df["high"] - df["low"]) / close.where(close > 0)
    df["hl_range"] = hl_range.where(hl_range >= 0)  # invalid if prices are 0
    gt = df.groupby("ticker", sort=False)  # regroup: now sees the new columns
    df["roll_vol_20"] = gt["log_return"].transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    hl_mean20 = gt["hl_range"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["range_ratio"] = df["hl_range"] / hl_mean20.replace(0, np.nan)  # range expansion

    # --- G3: volume & liquidity ---
    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    df["log_value"] = np.log1p(df["value"].clip(lower=0))
    shares = df["tradeable_shares"] if "tradeable_shares" in df else df.get("listed_shares")
    df["turnover"] = df["volume"] / (shares + EPS) if shares is not None else 0.0
    df["amihud"] = df["log_return"].abs() / df["value"].where(df["value"] > 0)  # illiquidity

    # --- G4: foreign flow ---
    if {"foreign_buy", "foreign_sell"}.issubset(df.columns):
        df["foreign_flow_ratio"] = (df["foreign_buy"] - df["foreign_sell"]) / (
            df["volume"].abs() + 1.0
        )
    else:
        df["foreign_flow_ratio"] = 0.0
    df["foreign_roll_5"] = df.groupby("ticker", sort=False)["foreign_flow_ratio"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # --- G5: microstructure (best bid/offer + their sizes) ---
    if {"bid", "offer"}.issubset(df.columns):
        mid = (df["bid"] + df["offer"]) / 2.0
        valid = (df["bid"] > 0) & (df["offer"] > 0) & (df["offer"] >= df["bid"])
        df["bid_ask_spread"] = ((df["offer"] - df["bid"]) / mid).where(valid)
    else:
        df["bid_ask_spread"] = np.nan
    if {"bid_volume", "offer_volume"}.issubset(df.columns):
        tot = df["bid_volume"] + df["offer_volume"]
        df["book_imbalance"] = (df["bid_volume"] - df["offer_volume"]) / tot.where(tot > 0)
    else:
        df["book_imbalance"] = np.nan

    # --- target: forward log return over `horizon` consecutive trading days ---
    gt = df.groupby("ticker", sort=False)
    fwd_alp = gt["_alp"].shift(-horizon)
    fwd_pos = gt["_pos"].shift(-horizon)
    fwd_traded = gt["traded"].shift(-horizon).fillna(False).astype(bool)
    df["fwd_gap"] = fwd_pos - df["_pos"]
    contiguous = df["fwd_gap"] == horizon
    df[TARGET_COLUMN] = (fwd_alp - df["_alp"]).where(contiguous & fwd_traded)

    # Sample validity today: must have traded (stale prices are unbuyable noise).
    # market.apply_universe() further ANDs a liquidity screen into valid_day.
    df["valid_day"] = df["traded"]

    keep = ["date", "ticker", *ALL_FEATURES, TARGET_COLUMN, *META_COLUMNS]
    out = df[keep].replace([np.inf, -np.inf], np.nan)
    # Drop only on the core feature so the sample set is invariant to which
    # optional groups are active (target-NaN rows on the last `horizon` days are
    # kept for inference; the datasets filter them via `require_target`).
    return out.dropna(subset=CORE_FEATURES).reset_index(drop=True)


def normalize(
    features: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "cross_sectional_zscore",
    clip: float = 5.0,
    stats_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Standardize features cross-sectionally within each date.

    Uses only that day's cross-section, so it is causal and has no train-fit
    stats to leak -- matches the DLSA convention. Missing values (sparse optional
    features, rolling warm-ups) become 0 after standardization, i.e. neutral.

    `stats_mask` selects which rows the per-day mean/std are ESTIMATED from
    (default: rows with valid_day=True, i.e. the tradable universe, so penny
    stocks don't distort the z-scores); the transform is applied to all rows.

    `columns` selects which feature columns to normalize; defaults to whichever
    of the full superset are present in `features`.
    """
    if method != "cross_sectional_zscore":
        raise ValueError(f"unknown normalize method: {method}")

    cols = list(columns) if columns is not None else [c for c in ALL_FEATURES if c in features.columns]
    df = features.copy()
    if stats_mask is None and "valid_day" in df.columns:
        stats_mask = df["valid_day"].astype(bool)
    src = df[cols].where(stats_mask, np.nan) if stats_mask is not None else df[cols]
    grp = src.groupby(df["date"], sort=False)
    mean = grp.transform("mean")
    std = grp.transform("std", ddof=0).replace(0, np.nan)
    df[cols] = ((df[cols] - mean) / std).fillna(0.0).clip(-clip, clip)
    return df

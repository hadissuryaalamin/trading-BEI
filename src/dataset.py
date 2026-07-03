"""Windowing of the normalized feature panel into sequences for the Transformer.

Produces samples x of shape (lookback, n_features) per (ticker, day t), with the
label being the forward return at t (already computed in preprocess as target).

Design
------
- Per ticker, build a contiguous time-indexed matrix and slide a window of
  length `lookback` ending at each day t. A sample is valid only if the full
  lookback window has no gaps and the target at t is present.
- Splits are by TIME (walk-forward): a sample belongs to a split by the date of
  its last window step t. This prevents any train/test leakage.
- No cross-asset batching tricks needed: each (ticker, t) is an independent
  sample; portfolio construction happens later in train_test.py using meta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import FEATURE_COLUMNS, TARGET_COLUMN


class IDXWindowDataset(Dataset):
    """Sliding-window dataset over a normalized feature panel.

    Parameters
    ----------
    features : long df with columns [date, ticker, *FEATURE_COLUMNS, target]
    lookback : window length in trading days
    start, end : optional date bounds (inclusive) on the window's LAST day t,
        used to carve walk-forward train/val/test splits
    require_target : if True, drop samples whose forward return is NaN
        (e.g. the final day). Set False for pure inference.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        lookback: int = 60,
        start=None,
        end=None,
        require_target: bool = True,
        feature_cols: list[str] | None = None,
    ):
        self.lookback = lookback
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
        self.n_features = len(self.feature_cols)

        df = features.sort_values(["ticker", "date"]).reset_index(drop=True)
        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None

        self._X: list[np.ndarray] = []   # per-ticker feature matrices
        self._y: list[np.ndarray] = []
        self._dates: list[np.ndarray] = []
        self._tickers: list[str] = []
        self.index: list[tuple[int, int]] = []  # (ticker_idx, end_row) samples

        for ticker, block in df.groupby("ticker", sort=False):
            X = block[self.feature_cols].to_numpy(dtype=np.float32)
            y = block[TARGET_COLUMN].to_numpy(dtype=np.float32)
            dates = block["date"].to_numpy()
            if len(block) < lookback:
                continue
            ti = len(self._tickers)
            self._tickers.append(ticker)
            self._X.append(X)
            self._y.append(y)
            self._dates.append(dates)
            for t in range(lookback - 1, len(block)):
                d = pd.Timestamp(dates[t])
                if start is not None and d < start:
                    continue
                if end is not None and d > end:
                    continue
                if require_target and not np.isfinite(y[t]):
                    continue
                self.index.append((ti, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ti, t = self.index[idx]
        x = self._X[ti][t - self.lookback + 1 : t + 1]      # (lookback, F)
        y = self._y[ti][t]
        meta = {"ticker": self._tickers[ti], "date": pd.Timestamp(self._dates[ti][t])}
        return (
            torch.from_numpy(x),
            torch.tensor(0.0 if not np.isfinite(y) else y, dtype=torch.float32),
            meta,
        )


def make_splits(features, lookback, train_end, val_end, data_end=None, feature_cols=None):
    """Convenience: build (train, val, test) datasets with walk-forward dates."""
    train = IDXWindowDataset(features, lookback, end=train_end, feature_cols=feature_cols)
    val = IDXWindowDataset(features, lookback, start=_next_day(train_end), end=val_end, feature_cols=feature_cols)
    test = IDXWindowDataset(features, lookback, start=_next_day(val_end), end=data_end, feature_cols=feature_cols)
    return train, val, test


def _next_day(d):
    return pd.Timestamp(d) + pd.Timedelta(days=1)

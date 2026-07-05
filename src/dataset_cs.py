"""Cross-sectional dataset: one sample = one trading day (all stocks that day).

Unlike IDXWindowDataset (one sample = one stock-day), here __getitem__(i) returns
EVERY stock that has a full lookback window ending on day i, so the model can
attend across stocks. Stock count varies by day, so we process one day per
forward pass (no padding needed).

Returns per day:
    X       : FloatTensor (N_stocks, lookback, n_features)
    y       : FloatTensor (N_stocks,)   forward returns (labels)
    tickers : list[str]  length N_stocks
    date    : pd.Timestamp
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import FEATURE_COLUMNS, TARGET_COLUMN


class IDXCrossSectionalDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        lookback: int = 60,
        start=None,
        end=None,
        require_target: bool = True,
        min_stocks: int = 20,
        feature_cols: list[str] | None = None,
        day_stride: int = 1,
    ):
        self.lookback = lookback
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
        self.n_features = len(self.feature_cols)

        df = features.sort_values(["ticker", "date"]).reset_index(drop=True)
        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None

        self._X, self._y, self._dates, self._tickers = [], [], [], []
        by_day: dict[pd.Timestamp, list[tuple[int, int]]] = {}

        for ticker, block in df.groupby("ticker", sort=False):
            if len(block) < lookback:
                continue
            X = block[self.feature_cols].to_numpy(dtype=np.float32)
            y = block[TARGET_COLUMN].to_numpy(dtype=np.float32)
            dates = block["date"].to_numpy()
            valid = (
                block["valid_day"].to_numpy(dtype=bool)
                if "valid_day" in block.columns
                else np.ones(len(block), dtype=bool)
            )
            ti = len(self._tickers)
            self._tickers.append(ticker); self._X.append(X); self._y.append(y); self._dates.append(dates)
            for t in range(lookback - 1, len(block)):
                if not valid[t]:
                    continue
                d = pd.Timestamp(dates[t])
                if start is not None and d < start:
                    continue
                if end is not None and d > end:
                    continue
                if require_target and not np.isfinite(y[t]):
                    continue
                by_day.setdefault(d, []).append((ti, t))

        # keep only days with enough names to form a cross-section
        self.days = sorted(d for d, items in by_day.items() if len(items) >= min_stocks)
        # day_stride > 1 = a slower trading cadence (e.g. 5 = weekly): keep every
        # k-th eligible day, so consecutive dataset entries are one rebalance
        # PERIOD apart -- train_dlsa's day-over-day costs then model weekly
        # turnover, and the simulator only sees (and trades) these dates.
        if day_stride > 1:
            self.days = self.days[::day_stride]
        self.by_day = by_day

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, i: int):
        date = self.days[i]
        items = self.by_day[date]
        lb = self.lookback
        X = np.stack([self._X[ti][t - lb + 1 : t + 1] for ti, t in items])  # (N, lb, F)
        y = np.array([self._y[ti][t] for ti, t in items], dtype=np.float32)  # (N,)
        tickers = [self._tickers[ti] for ti, _ in items]
        return torch.from_numpy(X), torch.from_numpy(y), tickers, date


def make_cs_splits(features, lookback, train_end, val_end, data_end=None, min_stocks=20, feature_cols=None):
    n = lambda d: pd.Timestamp(d) + pd.Timedelta(days=1)
    tr = IDXCrossSectionalDataset(features, lookback, end=train_end, min_stocks=min_stocks, feature_cols=feature_cols)
    va = IDXCrossSectionalDataset(features, lookback, start=n(train_end), end=val_end, min_stocks=min_stocks, feature_cols=feature_cols)
    te = IDXCrossSectionalDataset(features, lookback, start=n(val_end), end=data_end, min_stocks=min_stocks, feature_cols=feature_cols)
    return tr, va, te

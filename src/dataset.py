"""Windowing of the feature panel into sequences for the Transformer.

Produces samples of shape (T_lookback, F_features) per (asset, day), with the
target being the next-period return used by the trading-policy loss.

Key design choices
-------------------
- Rolling lookback window T (e.g. 60 trading days) ending at day t.
- One sample per (ticker, t) where the full window is available.
- Universe masking: on any given day some tickers are missing/suspended; emit a
  mask so batching over the cross-section stays rectangular.
- Train/val/test split is by TIME (walk-forward), never random, to prevent
  look-ahead. See configs for split dates.
"""
from __future__ import annotations

from torch.utils.data import Dataset


class IDXWindowDataset(Dataset):
    """Sliding-window dataset over the normalized feature panel."""

    def __init__(self, features, lookback: int = 60, horizon: int = 1):
        self.features = features
        self.lookback = lookback
        self.horizon = horizon
        raise NotImplementedError("Build index of valid (ticker, t) samples.")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        # returns (x[T, F], y[horizon], meta) — meta carries ticker/date for backtest
        raise NotImplementedError

"""Turn the cleaned panel into model-ready features.

Unlike upstream DLSA there is NO factor model / residual step here. We compute
features directly from raw prices/volumes and normalize them, then hand off to
`dataset.py` for windowing.

Feature ideas (all causal, no look-ahead):
- log return:            r_t = log(close_t / close_{t-1})
- overnight/intraday split, high-low range
- log volume / log value, turnover
- foreign net flow ratio: (foreign_buy - foreign_sell) / value
- rolling z-score of returns (cross-sectional or per-asset)

Normalization:
- Cross-sectional standardization per day (rank or z-score across assets), the
  DLSA-style choice, keeps the model focused on relative moves.
- Fit normalization stats on TRAIN window only to avoid leakage.
"""
from __future__ import annotations


def compute_features(panel):
    """panel (long df) -> feature df with same (date, ticker) index."""
    raise NotImplementedError


def normalize(features, method: str = "cross_sectional_zscore", stats=None):
    """Standardize features. Returns (normed, stats) so stats can be reused on test."""
    raise NotImplementedError

"""Training loop + walk-forward backtest for a trading-policy model.

Mirrors the role of DLSA's train_test.py, minus the residual machinery.

The model outputs a position/weight per asset per day. The loss is the negative
(mean/Sharpe of) portfolio return, optionally with:
- turnover / transaction-cost penalty
- position-size (L1/L2) regularization
- dollar-neutral or leverage constraint applied to raw outputs

Backtest: walk-forward over test windows, accumulate daily PnL, report
Sharpe, mean return, volatility, max drawdown, turnover.
"""
from __future__ import annotations


def train(model, train_loader, val_loader, cfg):
    """Optimize the policy; early-stop on validation Sharpe. Returns best model."""
    raise NotImplementedError


def backtest(model, test_loader, cfg) -> dict:
    """Run the trained policy over the test period; return metrics dict + curves."""
    raise NotImplementedError

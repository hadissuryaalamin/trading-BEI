"""Training loop + long-only top-N backtest.

Strategy: each trading day, score every stock, buy the top-N by score
(equal weight), hold one day, sell, repeat. Model is trained to predict the
next-day return (MSE); ranking those predictions drives the trades.

torch is imported lazily inside the functions that need it, so backtest() and
its metrics can be used (and tested) without torch installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model, train_loader, val_loader, cfg, device="cpu"):
    """Train with MSE on next-day return; early-stop on validation loss.

    Returns the model loaded with the best (lowest val-loss) weights.
    """
    import torch
    import torch.nn as nn

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    loss_fn = nn.MSELoss()
    patience = cfg.get("early_stop_patience", 8)
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(cfg.get("epochs", 50)):
        model.train()
        tr_loss = n = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y); n += len(y)
        tr_loss /= max(n, 1)

        val_loss = _eval_loss(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val - 1e-7:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best val {best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_loss(model, loader, loss_fn, device):
    import torch
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            tot += loss_fn(model(x), y).item() * len(y); n += len(y)
    return tot / max(n, 1)


def predict_scores(model, dataset, device="cpu", batch_size=1024) -> pd.DataFrame:
    """Run the model over a dataset -> DataFrame(date, ticker, score, fwd_return)."""
    import torch
    from torch.utils.data import DataLoader

    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return xs, ys, metas

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    rows = []
    model.eval()
    with torch.no_grad():
        for x, y, metas in loader:
            score = model(x.to(device)).cpu().numpy()
            for s, yi, m in zip(score, y.numpy(), metas):
                rows.append((m["date"], m["ticker"], float(s), float(yi)))
    return pd.DataFrame(rows, columns=["date", "ticker", "score", "fwd_return"])


# --------------------------------------------------------------------------- #
# Backtest (pure pandas/numpy -- no torch)
# --------------------------------------------------------------------------- #
def backtest_long_only(scores: pd.DataFrame, top_n=10, cost_bps=20.0):
    """Daily long-only top-N. scores has columns [date, ticker, score, fwd_return].

    fwd_return is the realized next-day LOG return of each stock. Each day we buy
    the N highest-scored stocks equally, earn their mean simple return, and pay a
    turnover-based cost when the held set changes day to day.

    Returns (metrics: dict, daily: DataFrame[date, gross, net, equity]).
    """
    cost = cost_bps / 1e4
    daily = []
    prev_holdings: set[str] = set()

    for date, day in scores.groupby("date"):
        picks = day.sort_values("score", ascending=False).head(top_n)
        held = set(picks["ticker"])
        simple = np.expm1(picks["fwd_return"].to_numpy())   # log -> simple
        gross = float(np.mean(simple)) if len(simple) else 0.0

        # turnover: fraction of the book replaced vs yesterday (buy side); selling
        # yesterday's names is the other leg -> approximate round-trip with 2x.
        new_frac = 1.0 if not prev_holdings else len(held - prev_holdings) / max(len(held), 1)
        turnover = new_frac
        net = gross - 2 * cost * turnover
        daily.append((date, gross, net, turnover))
        prev_holdings = held

    d = pd.DataFrame(daily, columns=["date", "gross", "net", "turnover"]).sort_values("date")
    d["equity"] = (1 + d["net"]).cumprod()
    return _metrics(d), d


def _metrics(d: pd.DataFrame, ann: int = 252) -> dict:
    r = d["net"].to_numpy()
    if len(r) == 0:
        return {}
    eq = d["equity"].to_numpy()
    total = float(eq[-1] - 1)
    ann_ret = float((1 + np.mean(r)) ** ann - 1)
    ann_vol = float(np.std(r, ddof=0) * np.sqrt(ann))
    sharpe = float(np.mean(r) / (np.std(r, ddof=0) + 1e-12) * np.sqrt(ann))
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    win = float((r > 0).mean())
    return {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win,
        "avg_daily_net": float(np.mean(r)),
        "avg_turnover": float(d["turnover"].mean()),
        "n_days": int(len(r)),
    }

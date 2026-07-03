"""Training loops + quick score-based backtest.

Three objectives:
  train()      per-stock regression (MSE on next-day return), windowed batches
  train_cs()   same MSE but one day (full cross-section) per step
  train_dlsa() DLSA-style economic objective: maximize the Sharpe of a
               long-only softmax portfolio, NET of transaction costs, over
               blocks of CONSECUTIVE trading days (so turnover is real).

The realistic simulator (suspensions, ARA/ARB, delistings) lives in
src/backtest.py; the `backtest_long_only` here is the quick label-based
variant used for validation-time model selection and smoke tests.

torch is imported lazily inside the functions that need it, so backtest() and
its metrics can be used (and tested) without torch installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import compute_metrics, TRADING_DAYS


# --------------------------------------------------------------------------- #
# Regression training (per-stock windows)
# --------------------------------------------------------------------------- #
def train(model, train_loader, val_loader, cfg, device="cpu"):
    """Train with MSE on next-day return; early-stop on validation loss.

    Returns the model loaded with the best (lowest val-loss) weights.
    """
    import time
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
    t_run = time.perf_counter()

    for epoch in range(cfg.get("epochs", 50)):
        t_ep = time.perf_counter()
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
        print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {val_loss:.6f} | {time.perf_counter() - t_ep:.1f}s")

        if val_loss < best_val - 1e-7:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best val {best_val:.6f})")
                break

    print(f"trained {epoch + 1} epochs in {time.perf_counter() - t_run:.1f}s")
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
# Quick label-based backtest (pure pandas/numpy -- no torch, no market matrix)
# --------------------------------------------------------------------------- #
def backtest_long_only(
    scores: pd.DataFrame,
    top_n: int = 10,
    buy_cost_bps: float = 15.0,
    sell_cost_bps: float = 25.0,
    rf_annual: float = 0.055,
):
    """Daily long-only top-N on the labels in `scores` [date,ticker,score,fwd_return].

    Frictionless-execution approximation (no ARA/ARB or suspension modelling --
    use src.backtest.simulate_long_only for the real evaluation). Used for
    validation-time model selection, where speed matters and the relative
    ordering of models is what counts.

    Returns (metrics: dict, daily: DataFrame).
    """
    buy_c, sell_c = buy_cost_bps / 1e4, sell_cost_bps / 1e4
    daily = []
    prev_holdings: set[str] = set()

    for date, day in scores.groupby("date"):
        picks = day.sort_values("score", ascending=False).head(top_n)
        held = set(picks["ticker"])
        simple = np.expm1(picks["fwd_return"].to_numpy())   # log -> simple
        gross = float(np.mean(simple)) if len(simple) else 0.0

        # equal-weight book: fraction replaced = one sell leg + one buy leg
        new_frac = 1.0 if not prev_holdings else len(held - prev_holdings) / max(len(held), 1)
        cost = new_frac * (buy_c + sell_c)
        daily.append((date, gross, cost, gross - cost, new_frac))
        prev_holdings = held

    d = pd.DataFrame(daily, columns=["date", "gross", "cost", "net", "turnover"]).sort_values("date")
    d["equity"] = (1 + d["net"]).cumprod()
    d["n_stuck"] = 0
    return compute_metrics(d, rf_annual=rf_annual), d


# --------------------------------------------------------------------------- #
# Cross-sectional MSE variant (one day per step; loss over that day's stocks)
# --------------------------------------------------------------------------- #
def train_cs(model, train_ds, val_ds, cfg, device="cpu"):
    """Train the cross-sectional model with MSE. Each step = one trading day."""
    import random
    import time
    import torch
    import torch.nn as nn

    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 3e-4),
                           weight_decay=cfg.get("weight_decay", 1e-5))
    loss_fn = nn.MSELoss()
    patience = cfg.get("early_stop_patience", 8)
    best_val, best_state, bad = float("inf"), None, 0
    order = list(range(len(train_ds)))
    t_run = time.perf_counter()

    for epoch in range(cfg.get("epochs", 50)):
        t_ep = time.perf_counter()
        model.train()
        random.shuffle(order)
        tr = n = 0
        for i in order:
            X, y, _, _ = train_ds[i]
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            tr += loss.item(); n += 1
        val = _eval_loss_cs(model, val_ds, loss_fn, device)
        print(f"epoch {epoch:03d} | train {tr/max(n,1):.6f} | val {val:.6f} | {time.perf_counter() - t_ep:.1f}s")
        if val < best_val - 1e-7:
            best_val, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best val {best_val:.6f})")
                break
    print(f"trained {epoch + 1} epochs in {time.perf_counter() - t_run:.1f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_loss_cs(model, ds, loss_fn, device):
    import torch
    model.eval()
    tot = n = 0
    with torch.no_grad():
        for i in range(len(ds)):
            X, y, _, _ = ds[i]
            tot += loss_fn(model(X.to(device)), y.to(device)).item(); n += 1
    return tot / max(n, 1)


def predict_scores_cs(model, ds, device="cpu") -> pd.DataFrame:
    """Run the cross-sectional model day by day -> DataFrame(date,ticker,score,fwd_return)."""
    import torch
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            X, y, tickers, date = ds[i]
            s = model(X.to(device)).cpu().numpy()
            for tk, sc, yi in zip(tickers, s, y.numpy()):
                rows.append((date, tk, float(sc), float(yi)))
    return pd.DataFrame(rows, columns=["date", "ticker", "score", "fwd_return"])


# --------------------------------------------------------------------------- #
# DLSA-style training: optimize NET portfolio Sharpe end-to-end (long-only)
# --------------------------------------------------------------------------- #
def train_dlsa(model, train_ds, val_ds, cfg, device="cpu", top_n: int = 10):
    """End-to-end economic objective (DLSA-style), long-only, cost-aware.

    Each gradient step takes a block of `days_per_step` CONSECUTIVE trading
    days. Per day the model scores every stock; softmax over the cross-section
    (temperature `softmax_temp`; with `allow_cash` an extra always-zero logit
    lets the portfolio retreat to cash at the risk-free rate) gives long-only
    weights. Day-over-day weight changes within the block are charged
    buy/sell costs (aligned by ticker), and the loss is the negative Sharpe of
    the block's NET returns in excess of rf. Trained this way the model pays
    for churning, unlike a gross-Sharpe objective.

    Early stopping / model selection uses the net Sharpe of the ACTUAL traded
    rule -- top-N equal weight after costs -- on the validation days, not the
    softmax portfolio, so we select the model we in fact trade.
    """
    import random
    import time
    import torch

    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 3e-4),
                           weight_decay=cfg.get("weight_decay", 1e-5))
    K = cfg.get("days_per_step", 32)
    temp = cfg.get("softmax_temp", 0.1)
    allow_cash = cfg.get("allow_cash", True)
    buy_c = cfg.get("buy_cost_bps", 15.0) / 1e4
    sell_c = cfg.get("sell_cost_bps", 25.0) / 1e4
    rf_annual = cfg.get("rf_annual", 0.055)
    rf_d = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    ann = TRADING_DAYS
    patience = cfg.get("early_stop_patience", 8)
    best, best_state, bad = -1e18, None, 0
    starts = list(range(0, max(len(train_ds) - 1, 1), K))  # block starts (consecutive days inside)
    t_run = time.perf_counter()

    def day_portfolio(i):
        """Weights + simple returns for day i (optionally with a cash slot)."""
        X, y, tickers, _ = train_ds[i]
        s = model(X.to(device)) / temp                     # (N,)
        simple = torch.expm1(y.to(device))                 # log -> simple return
        if allow_cash:
            s = torch.cat([s, s.new_zeros(1)])             # cash logit = 0
            simple = torch.cat([simple, simple.new_full((1,), rf_d)])
        w = torch.softmax(s, dim=0)
        return w, simple, tickers                          # tickers excl. cash slot

    def block_cost(w, tickers, prev_w, prev_tickers):
        """Cost of moving the STOCK book from prev day's weights (cash is free)."""
        prev_map = {t: j for j, t in enumerate(prev_tickers)}
        cur_map = {t: j for j, t in enumerate(tickers)}
        common = [t for t in tickers if t in prev_map]
        buys = w.new_zeros(())
        sells = w.new_zeros(())
        if common:
            ci = torch.tensor([cur_map[t] for t in common], device=w.device)
            pi = torch.tensor([prev_map[t] for t in common], device=w.device)
            delta = w[ci] - prev_w[pi]
            buys = buys + delta.clamp(min=0).sum()
            sells = sells + (-delta).clamp(min=0).sum()
        new_idx = [cur_map[t] for t in tickers if t not in prev_map]
        gone_idx = [prev_map[t] for t in prev_tickers if t not in cur_map]
        if new_idx:
            buys = buys + w[torch.tensor(new_idx, device=w.device)].sum()
        if gone_idx:
            sells = sells + prev_w[torch.tensor(gone_idx, device=w.device)].sum()
        return buys * buy_c + sells * sell_c

    for epoch in range(cfg.get("epochs", 50)):
        t_ep = time.perf_counter()
        model.train()
        random.shuffle(starts)
        losses = []
        for s0 in starts:
            block = range(s0, min(s0 + K, len(train_ds)))
            nets = []
            prev_w = prev_tickers = None
            for i in block:
                w, simple, tickers = day_portfolio(i)
                gross = (w * simple).sum()
                if prev_w is None:
                    cost = w.new_zeros(())                 # entering the block is free
                else:
                    cost = block_cost(w, tickers, prev_w, prev_tickers)
                nets.append(gross - cost)
                prev_w, prev_tickers = w, tickers
            if len(nets) < 2:
                continue
            r = torch.stack(nets)
            sharpe = (r.mean() - rf_d) / (r.std() + 1e-6) * (ann ** 0.5)
            loss = -sharpe
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_sharpe = _eval_topn_net_sharpe(
            model, val_ds, device, top_n=top_n,
            buy_cost_bps=buy_c * 1e4, sell_cost_bps=sell_c * 1e4, rf_annual=rf_annual,
        )
        print(f"epoch {epoch:03d} | train_loss {np.mean(losses):.4f} | "
              f"val_topN_net_sharpe {val_sharpe:.4f} | {time.perf_counter() - t_ep:.1f}s")
        if val_sharpe > best + 1e-6:
            best, bad = val_sharpe, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {epoch} (best val_topN_net_sharpe {best:.4f})")
                break
    print(f"trained {epoch + 1} epochs in {time.perf_counter() - t_run:.1f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _eval_topn_net_sharpe(model, ds, device, top_n, buy_cost_bps, sell_cost_bps, rf_annual):
    """Validation metric: net Sharpe of the traded rule (top-N after costs)."""
    if len(ds) == 0:
        return 0.0
    scores = predict_scores_cs(model, ds, device=device)
    m, _ = backtest_long_only(
        scores, top_n=top_n,
        buy_cost_bps=buy_cost_bps, sell_cost_bps=sell_cost_bps, rf_annual=rf_annual,
    )
    return float(m.get("sharpe", 0.0))

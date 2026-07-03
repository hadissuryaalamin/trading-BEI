"""End-to-end smoke test: panel -> features -> dataset -> model -> backtest.

Run after building the panel:
    python smoke_test.py
Trains for a couple of epochs on a slice and runs the long-only top-N backtest
so you can confirm the whole pipeline is wired before a full run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.preprocess import compute_features, normalize, FEATURE_COLUMNS
from src.dataset import IDXWindowDataset
from src import train_test as tt
from models.transformer import TransformerPolicy


def collate(b):
    return torch.stack([r[0] for r in b]), torch.stack([r[1] for r in b]), [r[2] for r in b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/panel.parquet")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    if not Path(args.panel).exists():
        raise SystemExit(f"Panel not found: {args.panel}. Build it with scraper.build_panel first.")

    panel = pd.read_parquet(args.panel)
    print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    feats = normalize(compute_features(panel, horizon=1))
    d = feats["date"].iloc[len(feats)//2]
    day = feats[feats["date"] == d][FEATURE_COLUMNS]
    print(f"features: {len(feats):,} rows | per-date |mean|max={abs(day.mean()).max():.2e} "
          f"std~({day.std(ddof=0).min():.2f},{day.std(ddof=0).max():.2f})")

    tr = IDXWindowDataset(feats, args.lookback, end="2024-12-31")
    va = IDXWindowDataset(feats, args.lookback, start="2025-01-01", end="2025-06-30")
    te = IDXWindowDataset(feats, args.lookback, start="2025-07-01")
    print(f"samples: train={len(tr):,} val={len(va):,} test={len(te):,}")

    model = TransformerPolicy(n_features=len(FEATURE_COLUMNS), lookback=args.lookback, output="linear")
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    # tiny training just to confirm the loop runs (not a real fit)
    cfg = {"epochs": 2, "lr": 3e-4, "early_stop_patience": 5}
    tt.train(model,
             DataLoader(tr, batch_size=512, shuffle=True, collate_fn=collate),
             DataLoader(va, batch_size=512, shuffle=False, collate_fn=collate),
             cfg)

    scores = tt.predict_scores(model, te, batch_size=512)
    metrics, _ = tt.backtest_long_only(scores, top_n=args.top_n)
    print("\n=== BACKTEST (test, long-only top-N; model barely trained) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()

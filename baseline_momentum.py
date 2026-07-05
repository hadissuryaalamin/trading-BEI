"""No-training baseline: periodic long-only top-N by a single ranking feature,
evaluated by the SAME honest simulator as the models (T+1 fills, ARA/ARB,
commission + per-name half-spread).

This is the bar any trained model must beat. If a 5M-parameter transformer
cannot outperform "rank by 20-day momentum, buy the top 10, rebalance weekly",
the honest conclusion is that the model adds nothing.

    python baseline_momentum.py                          # weekly, mom_20
    python baseline_momentum.py --signal foreign_roll_5  # rank by foreign flow
    python baseline_momentum.py --every 21               # monthly cadence
    python baseline_momentum.py --invert                 # short-term reversal (buy losers)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.preprocess import compute_features
from src.market import apply_universe, build_market
from src.backtest import simulate_long_only, benchmark_relative_metrics
from src.benchmark import ihsg_proxy_returns, benchmark_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Honest no-training baseline (top-N by one feature).")
    ap.add_argument("--panel", default="data/processed/panel.parquet")
    ap.add_argument("--signal", default="mom_20", help="feature column to rank by (see preprocess.ALL_FEATURES)")
    ap.add_argument("--invert", action="store_true", help="rank ascending (e.g. buy short-term losers)")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--every", type=int, default=5, help="rebalance every k trading days (5=weekly, 21=monthly)")
    ap.add_argument("--start", default="2024-07-01", help="test start (match the model runs)")
    ap.add_argument("--min_value_idr", type=float, default=1e9)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    feats = compute_features(panel)                     # features only; labels unused here
    feats = apply_universe(feats, panel, {"min_value_idr": args.min_value_idr, "window": 20})
    feats = feats[feats["valid_day"] & feats[args.signal].notna()]

    days = np.sort(feats.loc[feats["date"] >= pd.Timestamp(args.start), "date"].unique())
    pick_days = days[:: args.every]
    scores = feats.loc[feats["date"].isin(pick_days), ["date", "ticker", args.signal]]
    scores = scores.rename(columns={args.signal: "score"})
    if args.invert:
        scores["score"] = -scores["score"]
    print(f"baseline: top-{args.top_n} by {'-' if args.invert else ''}{args.signal}, "
          f"every {args.every} trading days | {len(pick_days)} rebalances from {pick_days[0]}")

    market = build_market(panel)
    metrics, daily = simulate_long_only(scores, market, top_n=args.top_n)
    ihsg = ihsg_proxy_returns(panel)
    bm = benchmark_metrics(ihsg, dates=daily["date"])
    metrics.update(bm)
    metrics.update(benchmark_relative_metrics(daily, ihsg))
    metrics["excess_ann_return"] = metrics["ann_return"] - bm["ihsg_ann_return"]

    print(f"\n=== BASELINE [{args.signal}] (honest simulator, T+1 fills, spread costs) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    verdict = "BEATS" if metrics["ann_return"] > bm["ihsg_ann_return"] else "LOSES TO"
    print(f"  -> {verdict} IHSG (baseline ann {metrics['ann_return']:.2%} "
          f"vs IHSG {bm['ihsg_ann_return']:.2%})")


if __name__ == "__main__":
    main()

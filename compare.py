"""Run both models and print a side-by-side comparison of test-period metrics.

    python compare.py
Runs configs/transformer_base.yaml and configs/cross_sectional.yaml, then
tabulates the key backtest metrics so you can see which wins.
"""
from __future__ import annotations

from src.run_train_test import run

CONFIGS = {
    "baseline (per-stock)": "configs/transformer_base.yaml",
    "cross-sectional": "configs/cross_sectional.yaml",
}
KEYS = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "win_rate", "avg_turnover", "n_days"]


def main() -> None:
    results = {}
    for label, cfg in CONFIGS.items():
        print(f"\n{'='*60}\nRUN: {label}  ({cfg})\n{'='*60}")
        results[label] = run(cfg)

    print(f"\n{'='*60}\nCOMPARISON (test period)\n{'='*60}")
    w = max(len(k) for k in KEYS) + 2
    header = "metric".ljust(w) + "".join(l.rjust(22) for l in results)
    print(header)
    for k in KEYS:
        row = k.ljust(w)
        for label in results:
            v = results[label].get(k, float("nan"))
            row += (f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(22)
        print(row)


if __name__ == "__main__":
    main()

"""Multi-seed sweep for ONE config: run it across seeds, print mean +/- std.

One seed at monthly cadence is ~23 rebalance decisions -- pure noise. This
answers the only question that matters: is the excess return consistent?

    python sweep_seeds.py -c configs/cross_sectional_monthly.yaml
    python sweep_seeds.py -c configs/cross_sectional_weekly.yaml --seeds 4
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from src.run_train_test import run

KEYS = ["ann_return", "excess_ann_return", "sharpe", "alpha_ann", "max_drawdown"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one config across N seeds.")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start-seed", type=int, default=0)
    args = ap.parse_args()

    name = Path(args.config).stem
    rows = []
    for s in range(args.start_seed, args.start_seed + args.seeds):
        m = run(args.config, overrides={"seed": s, "save": False,
                                        "experiment_name": f"{name}_seed{s}"})
        rows.append({"seed": s, **{k: float(m.get(k, float("nan"))) for k in KEYS},
                     "beats_ihsg": bool(m.get("beats_ihsg", False))})
        print(f"\n>>> SEED {s}: ann {m['ann_return']:+.2%} | "
              f"excess vs IHSG {m['excess_ann_return']:+.2%} | sharpe {m['sharpe']:.2f}\n")

    print(f"{'='*72}\nSWEEP SUMMARY: {name} x {len(rows)} seeds\n{'='*72}")
    for k in KEYS:
        vals = [r[k] for r in rows if r[k] == r[k]]
        print(f"  {k:>20}: {st.fmean(vals):+.4f} +/- {st.pstdev(vals):.4f}")
    print(f"  {'beats_ihsg':>20}: {sum(r['beats_ihsg'] for r in rows)}/{len(rows)}")

    out = Path("results") / f"{name}_seeds.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

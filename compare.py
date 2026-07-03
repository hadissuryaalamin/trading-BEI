"""Feature-ablation sweep: run configs A-E across multiple seeds and tabulate
out-of-sample backtest metrics as mean +/- std (see ABLATION_PLAN.md sec 5-6).

    python compare.py                      # all configs/ablation/*.yaml, 8 seeds
    python compare.py --seeds 3            # quicker partial sweep
    python compare.py --config-dir configs/ablation --seeds 8 --start-seed 0

Each (experiment, seed) is one walk-forward train+backtest via run(). Per-seed
runs don't write to results/ (save disabled); this script writes the aggregate.

NOTE: the long-only benchmark vs IHSG buy-and-hold (ABLATION_PLAN sec 5) and
equity-curve plots (sec 6.5) are a separate follow-up -- not produced here.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from src.run_train_test import run
from src.utils import load_config
from src.preprocess import resolve_features

# Ranking metric first; the rest are context (ABLATION_PLAN sec 5).
# excess_ann_return = strategy ann_return - IHSG proxy ann_return (long-only alpha).
METRICS = ["sharpe", "ann_return", "excess_ann_return", "max_drawdown", "avg_turnover"]


def _agg(vals: list[float]) -> tuple[float, float]:
    clean = [v for v in vals if v == v]  # drop NaN
    if not clean:
        return float("nan"), float("nan")
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the feature-ablation seed sweep.")
    ap.add_argument("--config-dir", default="configs/ablation")
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds per experiment")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()

    cfg_paths = sorted(p for p in Path(args.config_dir).glob("*.yaml") if not p.stem.startswith("_"))
    if not cfg_paths:
        raise SystemExit(f"No experiment configs (non-'_') in {args.config_dir}")
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    summary: list[dict] = []   # one aggregated row per experiment
    per_seed: list[dict] = []  # one row per (experiment, seed)

    for cp in cfg_paths:
        cfg = load_config(str(cp))
        label = cfg.get("experiment_name", cp.stem)
        groups = ",".join(cfg.get("features", {}).get("feature_groups", []) or [])
        active = resolve_features(cfg.get("features"))
        collected = {m: [] for m in METRICS}
        beats, ihsg_ann, ihsg_sh = [], [], []

        for s in seeds:
            print(f"\n{'='*64}\n{label}  [{groups} | {len(active)} feats]  seed={s}\n{'='*64}")
            metrics = run(str(cp), overrides={
                "seed": s,
                "experiment_name": f"{label}_seed{s}",
                "save": False,
            })
            rec = {"experiment": label, "groups": groups, "n_features": len(active), "seed": s}
            for m in METRICS:
                v = float(metrics.get(m, float("nan")))
                collected[m].append(v)
                rec[m] = v
            rec["beats_ihsg"] = bool(metrics.get("beats_ihsg", False))
            beats.append(rec["beats_ihsg"])
            ihsg_ann.append(float(metrics.get("ihsg_ann_return", float("nan"))))
            ihsg_sh.append(float(metrics.get("ihsg_sharpe", float("nan"))))
            per_seed.append(rec)

        row = {"experiment": label, "groups": groups, "n_features": len(active), "n_seeds": len(seeds)}
        for m in METRICS:
            mean, std = _agg(collected[m])
            row[f"{m}_mean"], row[f"{m}_std"] = mean, std
        row["beats_ihsg"] = f"{sum(beats)}/{len(beats)}"
        row["ihsg_ann_return"] = _agg(ihsg_ann)[0]
        row["ihsg_sharpe"] = _agg(ihsg_sh)[0]
        summary.append(row)

    # --- print table (mean +/- std) ---
    print(f"\n{'='*124}\nFEATURE ABLATION -- {len(seeds)} seeds, mean +/- std (ranking metric: sharpe)\n{'='*124}")
    header = f"{'experiment':<14}{'groups':<15}{'nf':>3}  " + "".join(f"{m:>19}" for m in METRICS) + f"{'beats_ihsg':>12}"
    print(header)
    for r in summary:
        line = f"{r['experiment']:<14}{r['groups']:<15}{r['n_features']:>3}  "
        line += "".join(f"{r[f'{m}_mean']:7.3f}+/-{r[f'{m}_std']:<5.3f}".rjust(19) for m in METRICS)
        line += f"{r['beats_ihsg']:>12}"
        print(line)
    ihsg_ann = _agg([r["ihsg_ann_return"] for r in summary])[0]
    ihsg_sh = _agg([r["ihsg_sharpe"] for r in summary])[0]
    ref = {"sharpe": f"{ihsg_sh:.3f}", "ann_return": f"{ihsg_ann:.3f}"}
    line = f"{'IHSG b&h':<14}{'-':<15}{'-':>3}  "
    line += "".join(f"{ref.get(m, '-'):>19}" for m in METRICS) + f"{'-':>12}"
    print(line)

    best = max(summary, key=lambda r: r["sharpe_mean"] if r["sharpe_mean"] == r["sharpe_mean"] else float("-inf"))
    print(f"\nbest mean Sharpe: {best['experiment']} ({best['groups']}) = {best['sharpe_mean']:.3f} "
          f"| excess vs IHSG {best['excess_ann_return_mean']:+.3f} | beats {best['beats_ihsg']}")

    # --- save ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader(); w.writerows(per_seed)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nsaved -> {out}/per_seed.csv, {out}/summary.csv")


if __name__ == "__main__":
    main()

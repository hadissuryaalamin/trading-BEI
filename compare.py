"""Feature-ablation sweep: run configs A-E across seeds, optionally in parallel,
and tabulate out-of-sample metrics as mean +/- std vs an IHSG proxy benchmark
(see ABLATION_PLAN.md sec 5-6).

    python compare.py                      # all configs/ablation/*.yaml, 8 seeds
    python compare.py --seeds 4            # quicker partial sweep
    python compare.py --parallel 3         # 3 concurrent runs (needs ~3x VRAM)

Parallelism is VRAM-bound: each run holds ~7-8 GB (K = train.days_per_step
autograd graphs). Set --parallel to floor(GPU_VRAM_GB / 8):
    8 GB laptop  -> 1 (parallel won't fit; keep sequential)
    24 GB card   -> 3
    32 GB card   -> 4
In parallel mode each run's stdout goes to results/ablation/logs/<label>_seed<n>.log
(so the console stays readable); sequential mode prints live to the console.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.utils import load_config
from src.preprocess import resolve_features

# Ranking metric first; the rest are context (ABLATION_PLAN sec 5).
# excess_ann_return = strategy ann_return - IHSG proxy ann_return (long-only alpha).
METRICS = ["sharpe", "ann_return", "excess_ann_return", "max_drawdown", "avg_turnover"]


def _run_job(args):
    """Worker: one (config, seed) walk-forward run. Top-level so it is picklable
    for the spawn-based process pool. Returns (label, seed, metrics)."""
    cfg_path, seed, label, log_path = args
    if log_path:  # parallel mode -> isolate this run's noisy logs to a file
        f = open(log_path, "w", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = f
    from src.run_train_test import run  # imported here so torch/CUDA init happens per worker
    metrics = run(cfg_path, overrides={
        "seed": seed,
        "experiment_name": f"{label}_seed{seed}",
        "save": False,
    })
    return label, seed, metrics


def _agg(vals: list[float]) -> tuple[float, float]:
    clean = [v for v in vals if v == v]  # drop NaN
    if not clean:
        return float("nan"), float("nan")
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the feature-ablation seed sweep (parallel-capable).")
    ap.add_argument("--config-dir", default="configs/ablation")
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds per experiment")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1,
                    help="concurrent runs; VRAM-bound (~8 GB each). 1 on an 8 GB laptop, ~3 on 24 GB")
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()

    cfg_paths = sorted(p for p in Path(args.config_dir).glob("*.yaml") if not p.stem.startswith("_"))
    if not cfg_paths:
        raise SystemExit(f"No experiment configs (non-'_') in {args.config_dir}")
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))

    # per-experiment metadata (in config order) + the flat job list
    meta: dict[str, tuple[str, int]] = {}   # label -> (groups, n_features)
    order: list[str] = []
    jobs: list[tuple] = []
    logdir = Path(args.out) / "logs"
    for cp in cfg_paths:
        cfg = load_config(str(cp))
        label = cfg.get("experiment_name", cp.stem)
        groups = ",".join(cfg.get("features", {}).get("feature_groups", []) or [])
        meta[label] = (groups, len(resolve_features(cfg.get("features"))))
        order.append(label)
        for s in seeds:
            log_path = None
            if args.parallel > 1:
                logdir.mkdir(parents=True, exist_ok=True)
                log_path = str(logdir / f"{label}_seed{s}.log")
            jobs.append((str(cp), s, label, log_path))

    print(f"{len(jobs)} runs = {len(cfg_paths)} configs x {len(seeds)} seeds | parallel={args.parallel}")
    if args.parallel > 1:
        print(f"per-run logs -> {logdir}/")

    # --- execute ---
    results: dict[tuple[str, int], dict] = {}
    if args.parallel <= 1:
        for cfg_path, seed, label, _ in jobs:
            g, nf = meta[label]
            print(f"\n{'='*64}\n{label}  [{g} | {nf} feats]  seed={seed}\n{'='*64}")
            _, _, m = _run_job((cfg_path, seed, label, None))  # console output
            results[(label, seed)] = m
    else:
        ctx = multiprocessing.get_context("spawn")  # required for CUDA in workers
        with ProcessPoolExecutor(max_workers=args.parallel, mp_context=ctx) as ex:
            futures = [ex.submit(_run_job, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                label, seed, m = fut.result()
                results[(label, seed)] = m
                print(f"[{i}/{len(jobs)}] done: {label} seed={seed} | "
                      f"sharpe={m.get('sharpe', float('nan')):.3f} | beats_ihsg={m.get('beats_ihsg')}")

    # --- aggregate per experiment (preserve config order) ---
    summary: list[dict] = []
    per_seed: list[dict] = []
    for label in order:
        groups, nf = meta[label]
        collected = {k: [] for k in METRICS}
        beats, ihsg_ann, ihsg_sh = [], [], []
        for s in seeds:
            m = results.get((label, s), {})
            rec = {"experiment": label, "groups": groups, "n_features": nf, "seed": s}
            for k in METRICS:
                v = float(m.get(k, float("nan")))
                collected[k].append(v)
                rec[k] = v
            rec["beats_ihsg"] = bool(m.get("beats_ihsg", False))
            beats.append(rec["beats_ihsg"])
            ihsg_ann.append(float(m.get("ihsg_ann_return", float("nan"))))
            ihsg_sh.append(float(m.get("ihsg_sharpe", float("nan"))))
            per_seed.append(rec)

        row = {"experiment": label, "groups": groups, "n_features": nf, "n_seeds": len(seeds)}
        for k in METRICS:
            row[f"{k}_mean"], row[f"{k}_std"] = _agg(collected[k])
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

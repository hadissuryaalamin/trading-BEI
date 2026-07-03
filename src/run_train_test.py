"""CLI: config -> data -> features -> splits -> model -> train -> backtest -> save.

Two independent switches:
  cfg["model"]["name"]     : transformer (per-stock)  |  cross_sectional (attends across stocks)
  cfg["train"]["objective"]: regression (MSE on next-day return)
                             sharpe     (DLSA-style: optimize long-only portfolio Sharpe)

The 'sharpe' objective always batches by day (needs the full cross-section to
form a portfolio), so it uses the per-day dataset regardless of model type.

Usage
-----
    python -m src.run_train_test -c configs/transformer_base.yaml
    python -m src.run_train_test -c configs/cross_sectional.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .preprocess import compute_features, normalize, resolve_features
from .benchmark import ihsg_proxy_returns, benchmark_metrics, plot_equity_vs_ihsg
from .utils import load_config, set_seed, deep_merge
from . import train_test as tt


def _collate_windows(b):
    import torch
    return (torch.stack([r[0] for r in b]),
            torch.stack([r[1] for r in b]),
            [r[2] for r in b])


def _load_features(cfg, active):
    dcfg, wcfg = cfg["data"], cfg["window"]
    panel = pd.read_parquet(dcfg["panel"])
    if dcfg.get("start"):
        panel = panel[panel["date"] >= pd.Timestamp(dcfg["start"])]
    if dcfg.get("end"):
        panel = panel[panel["date"] <= pd.Timestamp(dcfg["end"])]
    feats = normalize(
        compute_features(panel, horizon=wcfg.get("horizon", 1)),
        columns=active,
        method=cfg["features"].get("normalize", "cross_sectional_zscore"),
    )
    ihsg = ihsg_proxy_returns(panel)  # cap-weighted market return (IHSG proxy)
    print(f"features: {len(feats):,} rows, {len(active)} cols -> {active}")
    return feats, ihsg


def _build_model(model_name, mcfg, lookback, n_features):
    import torch  # noqa: F401
    if model_name == "cross_sectional":
        from models.cross_sectional import CrossSectionalModel
        return CrossSectionalModel(
            n_features=n_features, d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
            temporal_layers=mcfg.get("temporal_layers", 2), cross_layers=mcfg.get("cross_layers", 2),
            dim_ff=mcfg["dim_ff"], dropout=mcfg["dropout"], lookback=lookback,
            pooling=mcfg.get("pooling", "last"), output="linear",
        )
    from models.transformer import TransformerPolicy
    return TransformerPolicy(
        n_features=n_features, d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        n_layers=mcfg.get("n_layers", 3), dim_ff=mcfg["dim_ff"], dropout=mcfg["dropout"],
        lookback=lookback, pooling=mcfg.get("pooling", "last"), output="linear",
    )


def run(config_path: str, overrides: dict | None = None) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    set_seed(cfg.get("seed", 42))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    name = cfg.get("experiment_name", "run")
    model_name = cfg["model"].get("name", "transformer")
    objective = cfg["train"].get("objective", "regression")
    scfg, wcfg, mcfg = cfg["split"], cfg["window"], cfg["model"]
    lookback = wcfg["lookback"]
    active = resolve_features(cfg.get("features"))
    feats, ihsg = _load_features(cfg, active)
    nxt = lambda d: pd.Timestamp(d) + pd.Timedelta(days=1)

    # 'sharpe' needs the daily cross-section; cross_sectional model also does.
    per_day = objective == "sharpe" or model_name == "cross_sectional"
    print(f"model={model_name} | objective={objective} | per_day_batching={per_day}")

    model = _build_model(model_name, mcfg, lookback, len(active)).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    if per_day:
        from .dataset_cs import IDXCrossSectionalDataset as DS
        tr = DS(feats, lookback, end=scfg["train_end"], feature_cols=active)
        va = DS(feats, lookback, start=nxt(scfg["train_end"]), end=scfg["val_end"], feature_cols=active)
        te = DS(feats, lookback, start=nxt(scfg["val_end"]), end=cfg["data"].get("end"), feature_cols=active)
        print(f"days: train={len(tr)} val={len(va)} test={len(te)}")
        if objective == "sharpe":
            tt.train_dlsa(model, tr, va, cfg["train"], device=device)
        else:
            tt.train_cs(model, tr, va, cfg["train"], device=device)
        scores = tt.predict_scores_cs(model, te, device=device)
    else:
        from torch.utils.data import DataLoader
        from .dataset import IDXWindowDataset
        tr = IDXWindowDataset(feats, lookback, end=scfg["train_end"], feature_cols=active)
        va = IDXWindowDataset(feats, lookback, start=nxt(scfg["train_end"]), end=scfg["val_end"], feature_cols=active)
        te = IDXWindowDataset(feats, lookback, start=nxt(scfg["val_end"]), end=cfg["data"].get("end"), feature_cols=active)
        print(f"samples: train={len(tr):,} val={len(va):,} test={len(te):,}")
        bs = cfg["train"]["batch_size"]
        pin = device == "cuda"
        nw = cfg["train"].get("num_workers", 0)
        tt.train(model,
                 DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=_collate_windows,
                            num_workers=nw, pin_memory=pin),
                 DataLoader(va, batch_size=bs, shuffle=False, collate_fn=_collate_windows,
                            num_workers=nw, pin_memory=pin),
                 cfg["train"], device=device)
        scores = tt.predict_scores(model, te, device=device, batch_size=bs)

    # --- backtest (shared): rank by score, buy top-N (which are 'going up') ---
    pcfg = cfg.get("portfolio", {})
    metrics, daily = tt.backtest_long_only(
        scores, top_n=pcfg.get("top_n", 10),
        cost_bps=cfg["train"].get("transaction_cost_bps", 20),
    )
    # --- benchmark vs IHSG buy-and-hold over the SAME test dates (long-only beta) ---
    bm = benchmark_metrics(ihsg, dates=daily["date"])
    metrics.update(bm)
    metrics["excess_ann_return"] = metrics.get("ann_return", float("nan")) - bm["ihsg_ann_return"]
    metrics["beats_ihsg"] = bool(metrics.get("ann_return", float("-inf")) > bm["ihsg_ann_return"])

    print(f"\n=== BACKTEST [{model_name}/{objective}] (test, long-only top-{pcfg.get('top_n',10)}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  -> {'BEATS' if metrics['beats_ihsg'] else 'LOSES TO'} IHSG "
          f"(strat ann {metrics.get('ann_return', float('nan')):.2%} vs IHSG {bm['ihsg_ann_return']:.2%})")

    # --- save (skipped during sweeps via overrides={"save": False}) ---
    if cfg.get("save", True):
        out = Path("results") / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({**metrics, "model": model_name, "objective": objective}, indent=2))
        daily.to_csv(out / "daily_returns.csv", index=False)
        scores.to_csv(out / "test_scores.csv", index=False)
        plot_equity_vs_ihsg(daily, ihsg, out / "equity_vs_ihsg.png",
                            title=f"{name}: strategy vs IHSG (test)")
        Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path("models/checkpoints") / f"{name}.pt")
        print(f"\nsaved -> {out}/ and models/checkpoints/{name}.pt")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train + backtest a trading policy.")
    p.add_argument("-c", "--config", required=True, help="path to YAML config")
    run(p.parse_args().config)


if __name__ == "__main__":
    main()

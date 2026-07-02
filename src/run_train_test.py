"""CLI: config -> data -> features -> splits -> model -> train -> backtest -> save.

Usage
-----
    python -m src.run_train_test -c configs/transformer_base.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .preprocess import compute_features, normalize, FEATURE_COLUMNS
from .dataset import IDXWindowDataset
from .utils import load_config, set_seed
from . import train_test as tt


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    import torch
    from torch.utils.data import DataLoader
    from models.transformer import TransformerPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg.get("experiment_name", "run")
    dcfg, scfg, wcfg = cfg["data"], cfg["split"], cfg["window"]
    lookback = wcfg["lookback"]

    # --- data ---
    panel = pd.read_parquet(dcfg["panel"])
    if dcfg.get("start"):
        panel = panel[panel["date"] >= pd.Timestamp(dcfg["start"])]
    if dcfg.get("end"):
        panel = panel[panel["date"] <= pd.Timestamp(dcfg["end"])]
    feats = normalize(
        compute_features(panel, horizon=wcfg.get("horizon", 1)),
        method=cfg["features"].get("normalize", "cross_sectional_zscore"),
    )
    print(f"features: {len(feats):,} rows, {len(FEATURE_COLUMNS)} cols")

    # --- walk-forward splits ---
    def _next(d): return pd.Timestamp(d) + pd.Timedelta(days=1)
    train_ds = IDXWindowDataset(feats, lookback, end=scfg["train_end"])
    val_ds = IDXWindowDataset(feats, lookback, start=_next(scfg["train_end"]), end=scfg["val_end"])
    test_ds = IDXWindowDataset(feats, lookback, start=_next(scfg["val_end"]), end=dcfg.get("end"))
    print(f"samples: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    def collate(b):
        return (torch.stack([r[0] for r in b]),
                torch.stack([r[1] for r in b]),
                [r[2] for r in b])
    bs = cfg["train"]["batch_size"]
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)

    # --- model ---
    mcfg = cfg["model"]
    model = TransformerPolicy(
        n_features=len(FEATURE_COLUMNS), d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"], dim_ff=mcfg["dim_ff"], dropout=mcfg["dropout"],
        lookback=lookback, pooling=mcfg.get("pooling", "last"), output="linear",
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    # --- train ---
    tt.train(model, train_ld, val_ld, cfg["train"], device=device)

    # --- backtest on test ---
    scores = tt.predict_scores(model, test_ds, device=device, batch_size=bs)
    pcfg = cfg.get("portfolio", {})
    metrics, daily = tt.backtest_long_only(
        scores, top_n=pcfg.get("top_n", 10),
        cost_bps=cfg["train"].get("transaction_cost_bps", 20),
    )
    print("\n=== BACKTEST (test period, long-only top-N) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- save ---
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    daily.to_csv(out / "daily_returns.csv", index=False)
    scores.to_csv(out / "test_scores.csv", index=False)
    torch.save(model.state_dict(), Path("models/checkpoints") / f"{name}.pt")
    print(f"\nsaved -> {out}/ and models/checkpoints/{name}.pt")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train + backtest a trading policy.")
    p.add_argument("-c", "--config", required=True, help="path to YAML config")
    run(p.parse_args().config)


if __name__ == "__main__":
    main()

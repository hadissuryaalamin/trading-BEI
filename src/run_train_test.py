"""CLI: config -> data -> features -> splits -> model -> train -> backtest -> save.

Supports two model types via cfg["model"]["name"]:
    transformer     -> per-stock baseline (one sample = one stock-day)
    cross_sectional -> attention across stocks (one sample = one day)

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

from .preprocess import compute_features, normalize, FEATURE_COLUMNS
from .utils import load_config, set_seed
from . import train_test as tt


def _load_features(cfg):
    dcfg, wcfg = cfg["data"], cfg["window"]
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
    return feats


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    name = cfg.get("experiment_name", "run")
    model_name = cfg["model"].get("name", "transformer")
    scfg, wcfg, mcfg = cfg["split"], cfg["window"], cfg["model"]
    lookback = wcfg["lookback"]
    feats = _load_features(cfg)
    n = lambda d: pd.Timestamp(d) + pd.Timedelta(days=1)

    if model_name == "cross_sectional":
        from .dataset_cs import IDXCrossSectionalDataset as DS
        from models.cross_sectional import CrossSectionalModel
        tr = DS(feats, lookback, end=scfg["train_end"])
        va = DS(feats, lookback, start=n(scfg["train_end"]), end=scfg["val_end"])
        te = DS(feats, lookback, start=n(scfg["val_end"]), end=cfg["data"].get("end"))
        print(f"days: train={len(tr)} val={len(va)} test={len(te)}")
        model = CrossSectionalModel(
            n_features=len(FEATURE_COLUMNS), d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
            temporal_layers=mcfg.get("temporal_layers", 2), cross_layers=mcfg.get("cross_layers", 2),
            dim_ff=mcfg["dim_ff"], dropout=mcfg["dropout"], lookback=lookback,
            pooling=mcfg.get("pooling", "last"), output="linear",
        ).to(device)
        print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
        tt.train_cs(model, tr, va, cfg["train"], device=device)
        scores = tt.predict_scores_cs(model, te, device=device)
    else:
        from torch.utils.data import DataLoader
        from .dataset import IDXWindowDataset
        from models.transformer import TransformerPolicy
        tr = IDXWindowDataset(feats, lookback, end=scfg["train_end"])
        va = IDXWindowDataset(feats, lookback, start=n(scfg["train_end"]), end=scfg["val_end"])
        te = IDXWindowDataset(feats, lookback, start=n(scfg["val_end"]), end=cfg["data"].get("end"))
        print(f"samples: train={len(tr):,} val={len(va):,} test={len(te):,}")
        def collate(b):
            return torch.stack([r[0] for r in b]), torch.stack([r[1] for r in b]), [r[2] for r in b]
        bs = cfg["train"]["batch_size"]
        model = TransformerPolicy(
            n_features=len(FEATURE_COLUMNS), d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
            n_layers=mcfg.get("n_layers", 3), dim_ff=mcfg["dim_ff"], dropout=mcfg["dropout"],
            lookback=lookback, pooling=mcfg.get("pooling", "last"), output="linear",
        ).to(device)
        print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
        pin = device == "cuda"
        nw = cfg["train"].get("num_workers", 4)
        tt.train(model,
                 DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate,
                            num_workers=nw, pin_memory=pin),
                 DataLoader(va, batch_size=bs, shuffle=False, collate_fn=collate,
                            num_workers=nw, pin_memory=pin),
                 cfg["train"], device=device)
        scores = tt.predict_scores(model, te, device=device, batch_size=bs)

    # --- backtest (shared) ---
    pcfg = cfg.get("portfolio", {})
    metrics, daily = tt.backtest_long_only(
        scores, top_n=pcfg.get("top_n", 10),
        cost_bps=cfg["train"].get("transaction_cost_bps", 20),
    )
    print(f"\n=== BACKTEST [{model_name}] (test, long-only top-{pcfg.get('top_n',10)}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- save ---
    out = Path("results") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({**metrics, "model": model_name}, indent=2))
    daily.to_csv(out / "daily_returns.csv", index=False)
    scores.to_csv(out / "test_scores.csv", index=False)
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

"""CLI: config -> data -> features -> walk-forward folds -> train -> backtest -> save.

Two independent switches:
  cfg["model"]["name"]     : transformer (per-stock)  |  cross_sectional (attends across stocks)
  cfg["train"]["objective"]: regression (MSE on next-day return)
                             sharpe     (DLSA-style: optimize long-only NET portfolio Sharpe)

The 'sharpe' objective always batches by day (needs the full cross-section to
form a portfolio), so it uses the per-day dataset regardless of model type.

Splits (cfg["split"]):
  mode: single       -> one train/val/test split (train_end / val_end)
  mode: walk_forward -> rolling retrains; per fold: train up to the val window,
        validate on `val_months`, trade the next `step_months` out-of-sample,
        then roll forward. Test scores from all folds are stitched into one
        out-of-sample series before the backtest.

Evaluation is the stateful simulator (src/backtest.py): ARA/ARB tradability,
suspensions held not deleted, buy/sell costs incl. sell tax PLUS per-name
half-spread from the closing book, cash at rf, metrics in excess of rf plus
alpha/beta/IR vs the IHSG proxy.

Execution convention: signals computed from day-t closing data are executed at
the close of t + window.execution_lag (default 1) -- you cannot trade the very
close your features are computed from. Training labels and the simulator share
the same lag, so the model is trained on the return it can actually capture.
Set execution_lag: 0 only for signal-decay diagnostics (same-close MOC).

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
from .market import apply_universe, build_market
from .backtest import simulate_long_only, benchmark_relative_metrics
from .benchmark import ihsg_proxy_returns, benchmark_metrics, plot_equity_vs_ihsg
from .utils import load_config, set_seed, deep_merge
from . import train_test as tt


def _collate_windows(b):
    import torch
    return (torch.stack([r[0] for r in b]),
            torch.stack([r[1] for r in b]),
            [r[2] for r in b])


def _load_data(cfg, active):
    """Panel -> (normalized features, market matrix, IHSG proxy returns)."""
    dcfg, wcfg = cfg["data"], cfg["window"]
    panel = pd.read_parquet(dcfg["panel"])
    if dcfg.get("start"):
        panel = panel[panel["date"] >= pd.Timestamp(dcfg["start"])]
    if dcfg.get("end"):
        panel = panel[panel["date"] <= pd.Timestamp(dcfg["end"])]
    feats = compute_features(panel, horizon=wcfg.get("horizon", 1),
                             execution_lag=wcfg.get("execution_lag", 1))
    feats = apply_universe(feats, panel, cfg.get("universe"))
    feats = normalize(
        feats,
        columns=active,
        method=cfg["features"].get("normalize", "cross_sectional_zscore"),
    )
    market = build_market(panel)           # adjusted returns + tradability flags
    ihsg = ihsg_proxy_returns(panel)       # cap-weighted market return (IHSG proxy)
    print(f"features: {len(feats):,} rows, {len(active)} cols -> {active}")
    return feats, market, ihsg


def make_folds(scfg: dict, data_end) -> list[dict]:
    """Resolve cfg['split'] into a list of {train_end, val_end, test_start, test_end}."""
    day = pd.Timedelta(days=1)
    if scfg.get("mode", "single") == "single":
        return [{
            "train_end": pd.Timestamp(scfg["train_end"]),
            "val_end": pd.Timestamp(scfg["val_end"]),
            "test_start": pd.Timestamp(scfg["val_end"]) + day,
            "test_end": pd.Timestamp(data_end) if data_end else None,
        }]
    wf = scfg["walk_forward"]
    step = pd.DateOffset(months=int(wf.get("step_months", 6)))
    val_len = pd.DateOffset(months=int(wf.get("val_months", 6)))
    f0 = pd.Timestamp(wf["test_start"])
    end = pd.Timestamp(wf.get("test_end") or data_end)
    folds = []
    while f0 <= end:
        f1 = min(f0 + step - day, end)
        folds.append({
            "train_end": f0 - val_len - day,
            "val_end": f0 - day,
            "test_start": f0,
            "test_end": f1,
        })
        f0 = f0 + step
    return folds


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


def _train_one_fold(cfg, fold, feats, active, device):
    """Train a fresh model on one fold, return its test-period scores."""
    model_name = cfg["model"].get("name", "transformer")
    objective = cfg["train"].get("objective", "regression")
    lookback = cfg["window"]["lookback"]
    ccfg = cfg.get("costs", {})
    # rebalance_every > 1 = slower cadence (5 = weekly). Applies to the per-day
    # (cross-sectional / sharpe) path: datasets are strided so consecutive
    # entries are one period apart, and the training objective charges turnover
    # per period. Pair it with window.horizon = rebalance_every so the label
    # covers the actual holding period.
    rebalance_every = int(cfg.get("portfolio", {}).get("rebalance_every", 1))
    tcfg = {
        **cfg["train"],
        # training loss uses EFFECTIVE costs (commission + typical spread) via
        # train_*_bps; the simulator charges commission + each name's real
        # spread itself, so giving it the effective number would double-count
        "buy_cost_bps": ccfg.get("train_buy_bps", ccfg.get("buy_bps", 15.0)),
        "sell_cost_bps": ccfg.get("train_sell_bps", ccfg.get("sell_bps", 25.0)),
        "rf_annual": ccfg.get("rf_annual", 0.055),
        "period_days": rebalance_every,
        # validation model-selection uses the SAME portfolio rule as the backtest
        "strategy": cfg.get("portfolio", {}).get("strategy", "long_only_equal_topn"),
    }
    per_day = objective == "sharpe" or model_name == "cross_sectional"
    top_n = cfg.get("portfolio", {}).get("top_n", 10)

    import torch  # noqa: F401
    model = _build_model(model_name, cfg["model"], lookback, len(active)).to(device)

    if per_day:
        from .dataset_cs import IDXCrossSectionalDataset as DS
        tr = DS(feats, lookback, end=fold["train_end"], feature_cols=active,
                day_stride=rebalance_every)
        va = DS(feats, lookback, start=fold["train_end"] + pd.Timedelta(days=1),
                end=fold["val_end"], feature_cols=active, day_stride=rebalance_every)
        te = DS(feats, lookback, start=fold["test_start"], end=fold["test_end"],
                feature_cols=active, day_stride=rebalance_every)
        print(f"days: train={len(tr)} val={len(va)} test={len(te)}")
        if objective == "sharpe":
            tt.train_dlsa(model, tr, va, tcfg, device=device, top_n=top_n)
        else:
            tt.train_cs(model, tr, va, tcfg, device=device)
        return tt.predict_scores_cs(model, te, device=device)

    from torch.utils.data import DataLoader
    from .dataset import IDXWindowDataset
    tr = IDXWindowDataset(feats, lookback, end=fold["train_end"], feature_cols=active)
    va = IDXWindowDataset(feats, lookback, start=fold["train_end"] + pd.Timedelta(days=1),
                          end=fold["val_end"], feature_cols=active)
    te = IDXWindowDataset(feats, lookback, start=fold["test_start"], end=fold["test_end"],
                          feature_cols=active)
    print(f"samples: train={len(tr):,} val={len(va):,} test={len(te):,}")
    bs = tcfg["batch_size"]
    pin = device == "cuda"
    nw = tcfg.get("num_workers", 0)
    tt.train(model,
             DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=_collate_windows,
                        num_workers=nw, pin_memory=pin),
             DataLoader(va, batch_size=bs, shuffle=False, collate_fn=_collate_windows,
                        num_workers=nw, pin_memory=pin),
             tcfg, device=device)
    return tt.predict_scores(model, te, device=device, batch_size=bs)


def run(config_path: str, overrides: dict | None = None) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    seed = cfg.get("seed", 42)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    name = cfg.get("experiment_name", "run")
    model_name = cfg["model"].get("name", "transformer")
    objective = cfg["train"].get("objective", "regression")
    active = resolve_features(cfg.get("features"))
    feats, market, ihsg = _load_data(cfg, active)

    folds = make_folds(cfg["split"], cfg["data"].get("end") or feats["date"].max())
    print(f"model={model_name} | objective={objective} | folds={len(folds)}")

    # Portfolio rule shared by validation model-selection and the final backtest.
    strategy = cfg.get("portfolio", {}).get("strategy", "long_only_equal_topn")
    if (strategy == "long_only_positive_topn_prorata"
            and not cfg.get("train", {}).get("allow_cash", False)):
        # "score > 0" is only meaningful vs the cash anchor, which exists only
        # when train_dlsa adds the fixed-0 cash logit (allow_cash: true).
        print("WARNING: strategy 'long_only_positive_topn_prorata' assumes "
              "train.allow_cash: true (score>0 == above the cash anchor); "
              "allow_cash is false, so the score>0 threshold is not anchored.")

    all_scores = []
    for k, fold in enumerate(folds):
        print(f"\n--- fold {k + 1}/{len(folds)}: train<= {fold['train_end'].date()} | "
              f"val<= {fold['val_end'].date()} | test {fold['test_start'].date()}"
              f"..{fold['test_end'].date() if fold['test_end'] is not None else 'end'} ---")
        set_seed(seed)  # identical init per fold -> differences come from data
        scores_k = _train_one_fold(cfg, fold, feats, active, device)
        print(f"fold {k + 1}: {scores_k['date'].nunique()} test days, {len(scores_k):,} scores")
        all_scores.append(scores_k)
    scores = pd.concat(all_scores, ignore_index=True).sort_values(["date", "ticker"])

    # --- realistic backtest over the stitched out-of-sample scores ---
    pcfg = cfg.get("portfolio", {})
    ccfg = cfg.get("costs", {})
    rf_annual = ccfg.get("rf_annual", 0.055)
    execution_lag = cfg["window"].get("execution_lag", 1)
    metrics, daily = simulate_long_only(
        scores, market,
        top_n=pcfg.get("top_n", 10),
        buy_cost_bps=ccfg.get("buy_bps", 15.0),
        sell_cost_bps=ccfg.get("sell_bps", 25.0),
        rf_annual=rf_annual,
        delist_after=pcfg.get("delist_after", 20),
        delist_return=pcfg.get("delist_return", -0.5),
        execution_lag=execution_lag,
        default_half_spread_bps=ccfg.get("default_half_spread_bps", 35.0),
        max_half_spread_bps=ccfg.get("max_half_spread_bps", 200.0),
        strategy=strategy,
    )
    metrics["execution_lag"] = execution_lag
    # --- benchmark vs IHSG buy-and-hold over the SAME dates (long-only beta) ---
    bm = benchmark_metrics(ihsg, dates=daily["date"])
    metrics.update(bm)
    metrics.update(benchmark_relative_metrics(daily, ihsg, rf_annual=rf_annual))
    metrics["excess_ann_return"] = metrics.get("ann_return", float("nan")) - bm["ihsg_ann_return"]
    metrics["beats_ihsg"] = bool(metrics.get("ann_return", float("-inf")) > bm["ihsg_ann_return"])

    print(f"\n=== BACKTEST [{model_name}/{objective}] "
          f"(stitched test, long-only top-{pcfg.get('top_n', 10)}, {len(folds)} folds) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  -> {'BEATS' if metrics['beats_ihsg'] else 'LOSES TO'} IHSG "
          f"(strat ann {metrics.get('ann_return', float('nan')):.2%} vs IHSG {bm['ihsg_ann_return']:.2%})")

    # --- save (skipped during sweeps via overrides={"save": False}) ---
    if cfg.get("save", True):
        out = Path("results") / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(
            {**metrics, "model": model_name, "objective": objective, "n_folds": len(folds)}, indent=2))
        daily.to_csv(out / "daily_returns.csv", index=False)
        scores.to_csv(out / "test_scores.csv", index=False)
        plot_equity_vs_ihsg(daily, ihsg, out / "equity_vs_ihsg.png",
                            title=f"{name}: strategy vs IHSG (test)")
        print(f"\nsaved -> {out}/")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Train + backtest a trading policy.")
    p.add_argument("-c", "--config", required=True, help="path to YAML config")
    run(p.parse_args().config)


if __name__ == "__main__":
    main()

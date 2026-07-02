"""CLI entrypoint: config -> data -> train -> backtest -> save results.

Usage
-----
    python -m src.run_train_test -c configs/transformer_base.yaml
"""
from __future__ import annotations

import argparse


def run(config_path: str) -> dict:
    """Full pipeline for one experiment config. Returns metrics."""
    # 1. load config + set seed
    # 2. load data/processed/panel.parquet
    # 3. preprocess.compute_features -> normalize (fit on train)
    # 4. build IDXWindowDataset train/val/test (walk-forward)
    # 5. build model (models.transformer.TransformerPolicy)
    # 6. train_test.train -> train_test.backtest
    # 7. save metrics + plots to results/
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Train + backtest a trading policy.")
    p.add_argument("-c", "--config", required=True, help="path to YAML config")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

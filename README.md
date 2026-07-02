# trading-BEI

Deep-learning statistical arbitrage on **Bursa Efek Indonesia (IDX)** equities.

Adapted from [DLSA — *Deep Learning Statistical Arbitrage*](https://github.com/gregzanotti/dlsa-public)
(Guijarro-Ordonez, Pelger & Zanotti, 2021), with one key difference:

> **No factor model.** Instead of building residuals from a factor model and
> feeding those to the trading policy, we feed **raw IDX daily stock-summary
> data directly into a Transformer** which learns the representation and the
> trading signal end-to-end.

## Pipeline at a glance

```
IDX daily "Ringkasan Saham"  ──scrape──▶  data/raw/
        (per trading day, ~5y)                │
                                              ▼ build_panel
                            data/processed/ panel.parquet
                                              │
                                              ▼ dataset (windowing + normalize)
                            (N assets × T lookback × F features)
                                              │
                                              ▼ Transformer policy
                            per-asset trading position  ─▶ PnL / Sharpe
```

Compared to upstream DLSA the `factor_models/` and `residuals/` stages are
**removed**; raw normalized features are the model input.

## Structure

- `scraper/`   — download & assemble IDX daily stock summary into a panel
- `data/`      — `raw/` (scraped, gitignored), `processed/` (panel, gitignored)
- `src/`       — preprocessing, dataset, training loop, CLI entrypoints, utils
- `models/`    — Transformer trading-policy model(s)
- `configs/`   — YAML experiment configs (hyperparameters, universe, dates)
- `results/`   — metrics & plots (gitignored)
- `logs/`      — run logs (gitignored)
- `PLAN.md`    — full build plan and architecture rationale

## Quickstart (target workflow)

```bash
pip install -r requirements.txt

# 1. Scrape ~5 years of daily stock summary from IDX
python -m scraper.idx_scraper --start 2020-07-01 --end 2025-06-30 --out data/raw

# 2. Build a clean panel (parquet)
python -m scraper.build_panel --raw data/raw --out data/processed/panel.parquet

# 3. Train + backtest a transformer policy
python -m src.run_train_test -c configs/transformer_base.yaml
```

## Status

Scaffold + plan only. See [PLAN.md](PLAN.md) for the roadmap and open questions.

## Notes & disclaimer

- IDX data is scraped from the official [idx.co.id](https://www.idx.co.id)
  stock-summary endpoint for research use; respect their terms and rate limits.
- Research code, **not investment advice**. No warranty of profitability.
- Upstream DLSA is MIT-licensed; see `LICENSE`.

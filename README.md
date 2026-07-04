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
- `src/`       — preprocessing, universe/tradability (`market.py`), datasets,
                 training loops, stateful backtest (`backtest.py`), CLI, utils
- `models/`    — Transformer trading-policy model(s)
- `configs/`   — YAML experiment configs (hyperparameters, universe, dates)
- `tests/`     — anti-look-ahead, corporate-action, gap & backtest unit tests
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

## Long-only realism (the constraint that shapes everything)

The strategy is **long-only top-N**, so the backtest and objective model what
you can actually do on IDX:

- **Execution timing**: features use day-t *closing* data, so orders execute
  at the close of **t+1** (`window.execution_lag: 1`) — training labels and
  the simulator share the lag, so the model learns the return it can actually
  capture. Same-close (lag 0) exists only as a signal-decay diagnostic.
- **Corporate actions**: daily returns use `close / Previous` (IDX adjusts
  `Previous` on split ex-dates) — splits are not returns.
- **Tradability**: no buying names pinned at ARA (no offers at the close), no
  selling at ARB (no bids) — checked on the *execution* day; suspended
  holdings stay in the book and take their resume-day gap. Delisted holdings
  get written down.
- **Universe**: causal liquidity screen (trailing 20d median traded value).
- **Costs**: buy/sell bps split (commission + 0.1% sell tax) **plus each
  name's half-spread from its own closing book** on every fill (the liquid
  IDX universe's median spread is ~70bps — commission alone flatters);
  idle cash at rf.
- **Objective**: negative *net* Sharpe (excess rf) over consecutive-day blocks
  with turnover charged inside the loss; model selection by net top-N Sharpe.
- **Evaluation**: walk-forward retrains, stitched out-of-sample scores, Sharpe
  in excess of rf, plus alpha/beta/IR vs a cap-weighted IHSG proxy
  (`close × weight_for_index`, validated against the real index) — the bar a
  long-only book must beat is buy-and-hold, not zero.

Run `python -m pytest tests/` for the anti-look-ahead and mechanics tests.

## Status

Pipeline implemented end-to-end (scrape → panel → features → walk-forward
train → realistic backtest). See [PLAN.md](PLAN.md) for architecture and the
2026-07-04 realism-layer changelog; [ABLATION_PLAN.md](ABLATION_PLAN.md) for
the feature-ablation study design.

## Notes & disclaimer

- IDX data is scraped from the official [idx.co.id](https://www.idx.co.id)
  stock-summary endpoint for research use; respect their terms and rate limits.
- Research code, **not investment advice**. No warranty of profitability.
- Upstream DLSA is MIT-licensed; see `LICENSE`.

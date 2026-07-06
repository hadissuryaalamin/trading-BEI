# Configs

Each YAML defines one experiment: data range, universe, features, window,
model hyperparameters, training objective, and portfolio constraints.

Run with:

```bash
python -m src.run_train_test -c configs/transformer_base.yaml
```

Copy `transformer_base.yaml` to try variants (e.g. different `lookback`,
`d_model`, or `objective`).

## Portfolio strategy

`portfolio.strategy` selects how daily scores become a target book (used
identically by validation model-selection and the final backtest):

- `long_only_equal_topn` (default) — buy the top-`top_n` names by score,
  equal weight. Backward-compatible with the old `long_only`.
- `long_only_positive_topn_prorata` — keep only names with `score > 0`, take
  the top-`top_n` of those, and weight them **pro-rata by score**
  (`w_i = score_i / Σ score_j`); if no score is positive, hold 100% cash.

**Important:** `score > 0` is meaningful for the pro-rata strategy only when the
model is trained with the DLSA objective and `train.allow_cash: true`, because
the cash asset then has a fixed score/logit of 0 and `score > 0` means "ranked
above cash". It is **not** a predicted positive return unless the model is
trained with a regression objective.

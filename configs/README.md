# Configs

Each YAML defines one experiment: data range, universe, features, window,
model hyperparameters, training objective, and portfolio constraints.

Run with:

```bash
python -m src.run_train_test -c configs/transformer_base.yaml
```

Copy `transformer_base.yaml` to try variants (e.g. different `lookback`,
`d_model`, or `objective`).

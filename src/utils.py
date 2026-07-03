"""Shared helpers: config loading, seeding, logging, metrics."""
from __future__ import annotations

import os
import random


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` onto `base` (nested dicts merged, else replaced)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    """Load a YAML experiment config, resolving an optional `base:` include.

    A config may set `base: some.yaml` (path relative to that config's own
    directory); its own keys are deep-merged over the base. Lets the ablation
    experiments (A-E) share one base and differ only in `feature_groups`.
    """
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        cfg = deep_merge(load_config(base_path), cfg)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sharpe(returns, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio of a return series."""
    raise NotImplementedError

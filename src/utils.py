"""Shared helpers: config loading, seeding, logging, metrics."""
from __future__ import annotations

import random


def load_config(path: str) -> dict:
    """Load a YAML experiment config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


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

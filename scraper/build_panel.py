"""Assemble raw per-day IDX files into a single tidy panel.

Reads every file in data/raw, concatenates, cleans, and writes a long-format
parquet indexed by (date, ticker) with the numeric feature columns used by the
model.

Target schema (long format)
----------------------------
    date : datetime64        trading day
    ticker : str             IDX code, e.g. 'BBCA'
    open, high, low, close : float
    prev_close : float
    volume, value, frequency : float
    foreign_buy, foreign_sell : float
    shares_out : float
    ... (derived later in preprocess)

Cleaning steps
--------------
- Drop non-equity rows / rights / warrants if desired (configurable).
- Coerce numeric types, handle thousands separators.
- Drop days with all-zero volume for a ticker (non-trading / suspended).
- Optionally forward-fill short gaps (configurable in preprocess, not here).

Usage
-----
    python -m scraper.build_panel --raw data/raw --out data/processed/panel.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_raw(raw_dir: Path):
    """Read all per-day files into one DataFrame with a 'date' column."""
    raise NotImplementedError


def clean(df):
    """Type-coerce, filter instruments, drop bad rows."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Build tidy panel from raw IDX files.")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/processed/panel.parquet")
    p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()

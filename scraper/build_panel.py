"""Assemble raw per-day IDX stock-summary JSON into one tidy panel.

Reads idx_YYYYMMDD.json files in data/raw (optionally filtered by --start/--end),
concatenates, cleans, and writes a long-format parquet indexed by (date, ticker).

Usage
-----
    python -m scraper.build_panel --raw data/raw --out data/processed/panel.parquet
    python -m scraper.build_panel --start 2022-01-01                # from 2022 onward
    python -m scraper.build_panel --start 2022-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# Map IDX GetStockSummary fields -> tidy column names.
COLUMN_MAP = {
    "Date": "date",
    "StockCode": "ticker",
    "StockName": "name",
    "Previous": "prev_close",
    "OpenPrice": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Change": "change",
    "Volume": "volume",
    "Value": "value",
    "Frequency": "frequency",
    "ForeignBuy": "foreign_buy",
    "ForeignSell": "foreign_sell",
    "ListedShares": "listed_shares",
    "TradebleShares": "tradeable_shares",
    "Bid": "bid",
    "Offer": "offer",
    "Remarks": "remarks",
}

NUMERIC = [
    "prev_close", "open", "high", "low", "close", "change",
    "volume", "value", "frequency", "foreign_buy", "foreign_sell",
    "listed_shares", "tradeable_shares", "bid", "offer",
]

_DATE_RE = re.compile(r"idx_(\d{8})\.json$")


def _file_date(path: Path):
    m = _DATE_RE.search(path.name)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def load_raw(raw_dir: Path, start=None, end=None) -> pd.DataFrame:
    frames = []
    n_files = 0
    for fp in sorted(raw_dir.glob("idx_*.json")):
        fd = _file_date(fp)
        if fd is None:
            continue
        if start and fd < start:
            continue
        if end and fd > end:
            continue
        rows = json.loads(fp.read_text(encoding="utf-8"))
        n_files += 1
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise SystemExit(f"No non-empty raw files in {raw_dir} for the given range")
    print(f"Loaded {n_files} daily files ({len(frames)} with trading data)")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    present = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=present)[list(present.values())]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in [c for c in NUMERIC if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ticker", "date", "close"])
    df = df[df["ticker"].str.len() == 4]          # standard IDX equity codes
    df = df.drop_duplicates(subset=["date", "ticker"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def _parse(d):
    return datetime.strptime(d, "%Y-%m-%d").date() if d else None


def main() -> None:
    p = argparse.ArgumentParser(description="Build tidy panel from raw IDX files.")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/processed/panel.parquet")
    p.add_argument("--start", help="YYYY-MM-DD, earliest date to include")
    p.add_argument("--end", help="YYYY-MM-DD, latest date to include")
    args = p.parse_args()

    df = clean(load_raw(Path(args.raw), _parse(args.start), _parse(args.end)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"Wrote {out}: {len(df):,} rows, "
        f"{df['ticker'].nunique()} tickers, "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )


if __name__ == "__main__":
    main()

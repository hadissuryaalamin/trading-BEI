"""Download IDX daily 'Ringkasan Saham' (stock summary) files.

IDX publishes a per-trading-day stock summary via its trading-summary endpoint,
e.g.:
    https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date=YYYYMMDD

Each response is a JSON list with one row per listed stock containing fields like
open/high/low/close, previous close, volume, value, frequency, foreign buy/sell,
number of shares outstanding, bid/offer, etc.

This module walks a date range, skips weekends/holidays (empty responses), and
saves one raw file per trading day to ``--out`` (default: data/raw). It is
intentionally polite: retry with backoff + a delay between requests.

Usage
-----
    python -m scraper.idx_scraper --start 2020-07-01 --end 2025-06-30 --out data/raw

TODO
----
- Confirm the current endpoint/field names against idx.co.id (may change).
- Handle the site's headers/cookies if the endpoint requires them.
- Add --resume to skip dates already downloaded.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

IDX_STOCK_SUMMARY_URL = (
    "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"
    "?length=9999&start=0&date={yyyymmdd}"
)


def daterange(start: date, end: date):
    """Yield each calendar date from start to end inclusive (weekends filtered later)."""
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri; holidays handled by empty responses
            yield d
        d += timedelta(days=1)


def fetch_day(d: date):
    """Fetch one day's stock summary. Returns list-of-dicts or None if no trading."""
    raise NotImplementedError("Implement HTTP GET + retry/backoff (requests + tenacity).")


def save_day(rows, out_dir: Path, d: date) -> Path:
    """Persist one day's rows to out_dir/idx_YYYYMMDD.json (or .parquet)."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape IDX daily stock summary.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--out", default="data/raw", help="output directory")
    p.add_argument("--sleep", type=float, default=1.0, help="delay between requests (s)")
    args = p.parse_args()
    raise NotImplementedError("Wire daterange -> fetch_day -> save_day here.")


if __name__ == "__main__":
    main()

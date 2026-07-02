"""Download IDX daily Stock Summary ("Ringkasan Saham") over a date range.

An IDXClient owns the HTTP session; this script walks dates, caches, and skips
work already done. One JSON file per trading day -> data/raw/idx_YYYYMMDD.json.
Weekends are skipped; holidays return empty and are recorded so --resume won't
refetch them.

Usage
-----
    python -m scraper.idx_scraper --start 2020-07-01 --end 2025-06-30 --out data/raw --resume
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from .idx_client import IDXClient, IDXHTTPError


def daterange(start: date, end: date):
    """Yield weekdays from start to end inclusive; holidays filtered by empty response."""
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            yield d
        d += timedelta(days=1)


def _parse(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape IDX daily stock summary.")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--out", default="data/raw", help="output directory")
    p.add_argument("--sleep", type=float, default=1.0, help="delay between requests (s)")
    p.add_argument("--resume", action="store_true", help="skip dates already saved")
    p.add_argument("--debug", action="store_true", help="print HTTP status/body on errors")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = IDXClient(debug=args.debug)

    n_ok = n_empty = n_skip = n_err = 0
    for d in daterange(_parse(args.start), _parse(args.end)):
        stamp = d.strftime("%Y%m%d")
        dest = out_dir / f"idx_{stamp}.json"
        if args.resume and dest.exists():
            n_skip += 1
            continue
        try:
            rows = client.get_stock_summary(stamp)
        except IDXHTTPError as e:
            print(f"[{stamp}] fetch failed: HTTP {e.status_code}")
            n_err += 1
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[{stamp}] fetch failed: {type(e).__name__}: {e}")
            n_err += 1
            continue

        # Save even empty days (holiday marker) so --resume won't refetch them.
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        if rows:
            n_ok += 1
            print(f"[{stamp}] {len(rows)} stocks")
        else:
            n_empty += 1
            print(f"[{stamp}] no trading (empty)")
        time.sleep(args.sleep)

    print(f"Done. trading_days={n_ok} empty={n_empty} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()

"""Thin HTTP client for idx.co.id endpoints.

IDX sits behind Cloudflare: plain `requests` gets the "Just a moment..." JS
challenge (HTTP 403) because Cloudflare fingerprints the TLS handshake (JA3),
not just headers. We use curl_cffi with Chrome impersonation, which presents a
real Chrome TLS fingerprint and passes the bot check without solving a challenge.

Primary endpoint - Stock Summary ("Ringkasan Saham"):
    GET /primary/TradingSummary/GetStockSummary?length=9999&start=0&date=YYYYMMDD
Response: {"draw":0,"recordsTotal":N,"recordsFiltered":N,"data":[ {row}, ... ]}
Verified live: 2025-06-30 -> 960 rows, 2020-07-01 -> 696 rows, Sunday -> 0 rows.
"""
from __future__ import annotations

from curl_cffi import requests as crequests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

BASE = "https://www.idx.co.id"
STOCK_SUMMARY_PATH = "/primary/TradingSummary/GetStockSummary"
IMPERSONATE = "chrome"  # curl_cffi TLS/JA3 profile

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    "Referer": "https://www.idx.co.id/en/market-data/trading-summary/stock-summary/",
    "X-Requested-With": "XMLHttpRequest",
}


class IDXHTTPError(Exception):
    """Non-200 response from IDX; carries the status code and a body snippet."""

    def __init__(self, status_code: int, snippet: str = ""):
        self.status_code = status_code
        self.snippet = snippet
        super().__init__(f"HTTP {status_code}")


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 5xx and any network-level error; never on 4xx (won't self-heal)."""
    if isinstance(exc, IDXHTTPError):
        return exc.status_code >= 500
    return True  # curl_cffi connection/timeout errors -> retry


class IDXClient:
    def __init__(self, timeout: int = 30, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        self.session = crequests.Session(impersonate=IMPERSONATE)
        self.session.headers.update(DEFAULT_HEADERS)
        self._warm_up()

    def _warm_up(self) -> None:
        """Hit homepage then stock-summary page so Cloudflare sets its cookies."""
        for url in (BASE + "/", DEFAULT_HEADERS["Referer"]):
            try:
                self.session.get(url, timeout=self.timeout)
            except Exception:  # noqa: BLE001
                pass
        if self.debug:
            print("[warmup] cookies:", list(self.session.cookies.keys()))

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def get_stock_summary(self, yyyymmdd: str) -> list[dict]:
        """Return per-stock rows for one date ('YYYYMMDD'). [] if no trading."""
        params = {"length": 9999, "start": 0, "date": yyyymmdd}
        resp = self.session.get(
            BASE + STOCK_SUMMARY_PATH, params=params, timeout=self.timeout
        )
        if resp.status_code != 200:
            if self.debug:
                print(f"[{yyyymmdd}] HTTP {resp.status_code}; body[:200]={resp.text[:200]!r}")
            raise IDXHTTPError(resp.status_code, resp.text[:200])
        return resp.json().get("data", []) or []

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def download(self, file_path: str, dest: str) -> None:
        """Download a file referenced by an IDX-relative path to `dest`."""
        url = file_path if file_path.startswith("http") else BASE + file_path
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise IDXHTTPError(resp.status_code, resp.text[:200])
        with open(dest, "wb") as fh:
            fh.write(resp.content)

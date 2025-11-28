"""
HTTP client wrapper for the deployed pyPSX API with SQLite caching.

BACKEND-ONLY: This module is for backend internal use only.
SDK client code should NOT use this directly - use PyTrader client instead.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import httpx

from .cache.sqlite_cache import SQLiteCache
from ..config import settings
from ..utils.exceptions import DataProviderError, SymbolNotFoundError

DEFAULT_TIMEOUT = float(os.getenv("PYTRADER_DATA_TIMEOUT", "120"))
DEFAULT_BASE_URL = (
    os.getenv("PYPSX_API_BASE")
    or os.getenv("PYPSX_BASE_URL")
    or settings.psx_base_url
).rstrip("/")

INTRADAY_TTL_SECONDS = int(os.getenv("PYTRADER_INTRADAY_TTL", "600"))  # 10 minutes
HISTORICAL_TTL_SECONDS = int(os.getenv("PYTRADER_HISTORICAL_TTL", "21600"))  # 6 hours
METADATA_TTL_SECONDS = int(os.getenv("PYTRADER_METADATA_TTL", "86400"))  # 1 day
PRICES_TTL_SECONDS = int(os.getenv("PYTRADER_PRICES_TTL", "60"))  # 1 minute snapshot
MARKET_WATCH_TTL_SECONDS = int(os.getenv("PYTRADER_MARKET_WATCH_TTL", "30"))  # 30 seconds


def _build_cache_key(kind: str, *parts: object) -> str:
    normalized = [kind]
    for part in parts:
        if part is None:
            normalized.append("none")
        else:
            normalized.append(str(part).lower())
    return "::".join(normalized)


class _BaseService:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache: Optional[SQLiteCache] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        resolved_base = (base_url or DEFAULT_BASE_URL).rstrip("/")
        if not resolved_base:
            raise DataProviderError(
                "PYPSX API base URL is not configured. "
                "Set PYPSX_API_BASE or PYPSX_BASE_URL to your deployed pypsx-library endpoint."
            )
        self.base_url = resolved_base
        self.timeout = timeout
        self.cache = cache or SQLiteCache()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _raise_for_response(self, response: httpx.Response, url: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DataProviderError(f"HTTP error for {url}: {exc}") from exc


class PyPSXService(_BaseService):
    """
    Synchronous facade around the Render-hosted pyPSX API.
    
    BACKEND-ONLY: This class is for backend internal use only.
    SDK client code should use PyTrader client instead.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache: Optional[SQLiteCache] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        super().__init__(base_url, timeout, cache, max_retries, backoff_factor)
        self._client = httpx.Client(timeout=self.timeout)

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None):
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.request(method, url, params=params)
                self._raise_for_response(response, url)
                return response.json()
            except (httpx.HTTPError, DataProviderError) as exc:
                last_exc = exc
                sleep_for = self.backoff_factor * attempt
                time.sleep(sleep_for)
        raise DataProviderError(f"Request failed for {url}: {last_exc}") from last_exc

    def _cached_get(
        self,
        cache_key: str,
        ttl_seconds: int,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ):
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached and not cached.expired:
                return cached.value
        payload = self._request("GET", path, params=params)
        if use_cache:
            self.cache.set(cache_key, payload, ttl_seconds)
        return payload

    def get_symbols(self) -> List[Dict[str, Any]]:
        """Get list of available symbols."""
        cache_key = _build_cache_key("symbols")
        return self._cached_get(cache_key, METADATA_TTL_SECONDS, "/symbols")

    def get_intraday(
        self, symbol: str, lookback_days: int = 2, *, use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get intraday data for a symbol."""
        cache_key = _build_cache_key("intraday", symbol, lookback_days)
        return self._cached_get(
            cache_key,
            INTRADAY_TTL_SECONDS,
            f"/intraday/{symbol.upper()}",
            params={"days": lookback_days},
            use_cache=use_cache,
        )

    def get_historical(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Uses the /historical/{symbol} endpoint which expects:
        - start_date: YYYY-MM-DD format
        - end_date: YYYY-MM-DD format
        - interval: "1d" (daily) or "1w" (weekly)
        
        Returns list of OHLCV records: [{"date": "2024-01-01", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 1000000}, ...]
        """
        # Convert datetime to YYYY-MM-DD format as expected by the API
        start_date_str = start.strftime("%Y-%m-%d") if start else None
        end_date_str = end.strftime("%Y-%m-%d") if end else None
        cache_key = _build_cache_key("historical", symbol, start_date_str, end_date_str, interval)
        params: Dict[str, Any] = {"interval": interval}
        if start_date_str:
            params["start_date"] = start_date_str  # API expects start_date, not start
        if end_date_str:
            params["end_date"] = end_date_str  # API expects end_date, not end
        return self._cached_get(
            cache_key,
            HISTORICAL_TTL_SECONDS,
            f"/historical/{symbol.upper()}",
            params=params,
        )

    def get_prices_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Get current prices snapshot."""
        cache_key = _build_cache_key("prices")
        return self._cached_get(cache_key, PRICES_TTL_SECONDS, "/prices")

    def get_market_watch(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get market watch snapshot.
        
        Args:
            symbol: Optional symbol filter. When provided, only the matching symbol is returned.
        
        Returns:
            List of market watch entries.
        """
        cache_key = _build_cache_key("market_watch", symbol or "all")
        params = {"symbol": symbol.upper()} if symbol else None
        payload = self._cached_get(
            cache_key,
            MARKET_WATCH_TTL_SECONDS,
            "/market_watch" if not symbol else "/market_watch",
            params=params,
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        return []

    def get_market_watch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a single symbol quote from market watch endpoint.
        """
        if not symbol:
            return None
        cache_key = _build_cache_key("market_watch_symbol", symbol.upper())
        payload = self._cached_get(
            cache_key,
            MARKET_WATCH_TTL_SECONDS,
            f"/market_watch/{symbol.upper()}",
        )
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return payload[0] if payload else None
        return None

    def get_market_watch_bulk(self, symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market watch quotes for a collection of symbols.
        """
        unique_symbols = {str(symbol).upper() for symbol in symbols if symbol}
        if not unique_symbols:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        # If there are many symbols, fetch the entire board once to reduce API calls
        if len(unique_symbols) > 8:
            try:
                entries = self.get_market_watch()
            except DataProviderError:
                entries = []
            for entry in entries:
                sym = str(entry.get("symbol", "")).upper()
                if sym in unique_symbols:
                    results[sym] = entry
            return results

        for symbol in unique_symbols:
            quote = self.get_market_watch_quote(symbol)
            if quote:
                key = str(quote.get("symbol", symbol)).upper()
                results[key] = quote
        return results


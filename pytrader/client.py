"""
Lightweight HTTP client used by local bots to push telemetry to the backend and read dashboards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .telemetry import TelemetryClient


class PyTrader(TelemetryClient):
    """
    Convenience wrapper that extends the telemetry client with read helpers.

    Example:
        client = PyTrader(bot_id="ahmer_bot_1", api_token="dev-token", backend_url="https://api.pytrader")
        client.update_portfolio(equity=105_000, cash=90_000, positions={"OGDC": 100})
        snapshot = client.get_portfolio()
    """

    def __init__(
        self,
        *,
        bot_id: str,
        api_token: str,
        backend_url: Optional[str] = None,
        bot_label: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        super().__init__(bot_id=bot_id, api_token=api_token, backend_url=backend_url, bot_label=bot_label, timeout=timeout)
        self._reader = httpx.Client(timeout=timeout)

    def get_portfolio(self, bot_id: Optional[str] = None) -> Dict[str, Any]:
        target = bot_id or self.bot_id
        return self._get(f"/portfolio/{target}")

    def get_performance(self, bot_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        target = bot_id or self.bot_id
        return self._get(f"/performance/{target}", params={"limit": limit})

    def get_trades(self, bot_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        target = bot_id or self.bot_id
        return self._get(f"/trades/{target}", params={"limit": limit})

    def get_logs(self, bot_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        target = bot_id or self.bot_id
        return self._get(f"/logs/{target}", params={"limit": limit})

    def get_symbols(self) -> List[Dict[str, Any]]:
        response = self._get("/symbols")
        return response.get("symbols", [])

    def get_intraday(self, symbol: str, days: int = 2) -> List[Dict[str, Any]]:
        response = self._get(f"/intraday/{symbol.upper()}", params={"days": days})
        return response.get("data", [])

    def get_historical(
        self,
        symbol: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"interval": interval}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        response = self._get(f"/historical/{symbol.upper()}", params=params)
        return response.get("data", [])

    def close(self) -> None:
        super().close()
        self._reader.close()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"X-PyTrader-Token": self.api_token}
        response = self._reader.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


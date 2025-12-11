from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import httpx

from .auth import require_token, DEFAULT_BACKEND_URL


def _ensure_iso_timestamp(value: datetime | None) -> str:
    ts = value or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


class TelemetryClient:
    """
    Thin HTTP client that pushes portfolio, performance, trades, and log events to the backend.

    Example:
        client = TelemetryClient(
            bot_id="ahmer_bot_1",
            api_token="dev-token",
            backend_url="https://backend.example.com",
        )
        client.update_portfolio(equity=100_000, cash=90_000, positions={"OGDC": 100})
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
        self.bot_id = bot_id
        self.bot_label = bot_label
        self.base_url = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")
        self.api_token = require_token(api_token=api_token, backend_url=self.base_url)
        self._client = httpx.Client(timeout=timeout)

    def update_portfolio(
        self,
        *,
        equity: float,
        cash: float,
        positions: Dict[str, Any] | List[Dict[str, Any]],
        positions_value: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        status: Optional[str] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        payload = {
            "bot_id": self.bot_id,
            "bot_label": self.bot_label,
            "timestamp": _ensure_iso_timestamp(timestamp),
            "equity": equity,
            "cash": cash,
            "positions_value": positions_value,
            "positions": positions,
            "status": status,
            "recent_trades": recent_trades or [],
        }
        self._post("/portfolio_update", payload)

    def update_performance(
        self,
        *,
        equity: float,
        cash: float,
        positions_value: Optional[float],
        metrics: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> None:
        payload = {
            "bot_id": self.bot_id,
            "bot_label": self.bot_label,
            "timestamp": _ensure_iso_timestamp(timestamp),
            "equity": equity,
            "cash": cash,
            "positions_value": positions_value,
            "metrics": metrics,
            "status": status,
        }
        self._post("/performance_update", payload)

    def log_trades(self, trades: Iterable[Dict[str, Any]]) -> None:
        trade_list = []
        for trade in trades:
            payload = dict(trade)
            ts = payload.get("timestamp")
            if not isinstance(ts, datetime):
                payload["timestamp"] = _ensure_iso_timestamp(None)
            else:
                payload["timestamp"] = _ensure_iso_timestamp(ts)
            if "symbol" in payload and isinstance(payload["symbol"], str):
                payload["symbol"] = payload["symbol"].upper()
            trade_list.append(payload)
        if not trade_list:
            return
        self._post(
            "/trade_log",
            {
                "bot_id": self.bot_id,
                "bot_label": self.bot_label,
                "trades": trade_list,
            },
        )

    def log_event(
        self,
        message: str,
        *,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        stream: Optional[str] = None,
    ) -> None:
        payload = {
            "bot_id": self.bot_id,
            "bot_label": self.bot_label,
            "message": message,
            "level": level,
            "context": context or {},
            "timestamp": _ensure_iso_timestamp(timestamp),
        }
        if stream:
            payload["stream"] = stream
        self._post("/log_event", payload)

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, body: Dict[str, Any]) -> None:
        def _sanitize(value: Any) -> Any:
            from datetime import datetime, date
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            if isinstance(value, list):
                return [_sanitize(v) for v in value]
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            return value

        safe_body = _sanitize(body)
        url = f"{self.base_url}{path}"
        headers = {"X-PyTrader-Token": self.api_token}
        try:
            response = self._client.post(url, json=safe_body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Add more context to HTTP errors
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.json()
                if "detail" in error_body:
                    error_detail = f"{error_detail}: {error_body['detail']}"
            except Exception:
                error_detail = f"{error_detail}: {e.response.text[:200]}"
            raise httpx.HTTPStatusError(
                f"{error_detail} for {path}",
                request=e.request,
                response=e.response
            ) from e
        except httpx.TimeoutException as e:
            raise httpx.TimeoutException(f"Timeout calling {path}: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Failed to call {path}: {str(e)}") from e


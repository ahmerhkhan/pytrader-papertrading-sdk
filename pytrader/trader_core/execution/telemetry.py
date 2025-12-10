from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

import httpx

from ..portfolio.metrics import TradeMetrics
from ...utils.logger import log_line
from ...telemetry import TelemetryClient


@dataclass
class CycleReport:
    bot_id: str
    timestamp: datetime
    status: str
    equity: float
    cash: float
    positions_value: float
    metrics: TradeMetrics
    positions: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    prices: Dict[str, float]
    batches: List[Dict[str, Any]]
    total_fees: float = 0.0
    avg_slippage_bps: float = 0.0
    recent_trades: List[Dict[str, Any]] | None = None


class TelemetrySink(Protocol):
    def publish(self, report: CycleReport) -> None:
        ...

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:
        ...

    def close(self) -> None:
        ...


class NullTelemetry:
    def publish(self, report: CycleReport) -> None:  # pragma: no cover - noop
        return

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:  # pragma: no cover - noop
        return

    def close(self) -> None:  # pragma: no cover - noop
        return


class CompositeTelemetry:
    def __init__(self, sinks: Iterable[TelemetrySink]) -> None:
        self._sinks = list(sinks)

    def publish(self, report: CycleReport) -> None:
        for sink in self._sinks:
            try:
                sink.publish(report)
            except Exception as exc:  # pragma: no cover - defensive logging
                log_line(f"[{report.bot_id}] Telemetry publish failed: {exc}")

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:
        for sink in self._sinks:
            try:
                sink.publish_intraday(symbol, rows)
            except Exception as exc:  # pragma: no cover - defensive logging
                log_line(f"[telemetry] Failed to persist intraday for {symbol}: {exc}")

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:  # pragma: no cover - noop
                pass


class FileTelemetry:
    def __init__(self, root: Path, bot_id: str) -> None:
        self.root = Path(root).expanduser().resolve() / bot_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._metrics_json = self.root / "metrics.jsonl"
        self._metrics_csv = self.root / "metrics.csv"
        self._trades_json = self.root / "trades.jsonl"
        self._trades_csv = self.root / "trades.csv"
        self._intraday_dir = self.root / "intraday"
        self._intraday_dir.mkdir(parents=True, exist_ok=True)

    def publish(self, report: CycleReport) -> None:
        metrics_dict = asdict(report.metrics)
        payload = {
            "timestamp": report.timestamp.isoformat(),
            "bot_id": report.bot_id,
            "status": report.status,
            "equity": report.equity,
            "cash": report.cash,
            "positions_value": report.positions_value,
            "metrics": metrics_dict,
            "positions": report.positions,
            "prices": report.prices,
            "trades": report.trades,
            "batches": report.batches,
            "total_fees": report.total_fees,
            "avg_slippage_bps": report.avg_slippage_bps,
            "recent_trades": report.recent_trades or [],
        }
        self._append_json(self._metrics_json, payload)
        csv_row = {
            "timestamp": report.timestamp.isoformat(),
            "status": report.status,
            "equity": f"{report.equity:.2f}",
            "cash": f"{report.cash:.2f}",
            "positions_value": f"{report.positions_value:.2f}",
            "total_fees": f"{report.total_fees:.4f}",
            "avg_slippage_bps": f"{report.avg_slippage_bps:.2f}",
        }
        csv_row.update({k: metrics_dict.get(k) for k in sorted(metrics_dict)})
        self._append_csv(self._metrics_csv, csv_row)

        if report.trades:
            for trade in report.trades:
                trade_payload = {
                    **trade,
                    "bot_id": report.bot_id,
                    "timestamp": trade.get("timestamp"),
                }
                self._append_json(self._trades_json, trade_payload)
            self._append_csv_bulk(
                self._trades_csv,
                [
                    {
                        "timestamp": trade.get("timestamp"),
                        "symbol": trade.get("symbol"),
                        "side": trade.get("side"),
                        "quantity": trade.get("quantity"),
                        "price": trade.get("price"),
                        "cost": trade.get("cost"),
                        "pnl_realized": trade.get("pnl_realized"),
                    "commission": trade.get("commission"),
                    "slippage_bps": trade.get("slippage_bps"),
                    }
                    for trade in report.trades
                ],
            )

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:
        path = self._intraday_dir / f"{symbol.lower()}.jsonl"
        wrote_any = False
        for row in rows:
            record = dict(row)
            ts = record.get("ts")
            if isinstance(ts, datetime):
                record["ts"] = ts.isoformat()
            wrote_any = True
            self._append_json(path, record)
        if wrote_any:
            path.touch(exist_ok=True)

    def close(self) -> None:
        return

    def _append_json(self, path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=self._json_default))
            fh.write("\n")

    def _append_csv(self, path: Path, row: Dict[str, Any]) -> None:
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _append_csv_bulk(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        file_exists = path.exists()
        fieldnames = list(rows[0].keys())
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj


class HttpTelemetry:
    def __init__(
        self,
        base_url: str,
        token: str,
        endpoint: str = "/live/metrics",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.token = token
        self._client = httpx.Client(timeout=timeout)

    def publish(self, report: CycleReport) -> None:
        payload = _report_to_payload(report)
        headers = {"X-PyTrader-Token": self.token}
        try:
            self._client.post(f"{self.base_url}{self.endpoint}", json=payload, headers=headers)
        except Exception as exc:  # pragma: no cover - network
            log_line(f"[{report.bot_id}] Failed to push metrics to backend: {exc}")

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:
        return

    def close(self) -> None:
        self._client.close()


class BackendRelayTelemetry:
    """
    Telemetry sink that relays cycle reports to the backend ingestion API.
    """

    def __init__(self, client: TelemetryClient) -> None:
        self._client = client

    def publish(self, report: CycleReport) -> None:
        try:
            # 1. Update portfolio
            self._client.update_portfolio(
                equity=report.equity,
                cash=report.cash,
                positions=report.positions,
                positions_value=report.positions_value,
                timestamp=report.timestamp,
                status=report.status,
                recent_trades=report.recent_trades or [],
            )
            
            # 2. Update performance
            metrics_dict = asdict(report.metrics)
            self._client.update_performance(
                equity=report.equity,
                cash=report.cash,
                positions_value=report.positions_value,
                metrics=metrics_dict,
                timestamp=report.timestamp,
                status=report.status,
            )
            
            # 3. Log trades if any
            if report.trades:
                trades_payload = []
                for trade in report.trades:
                    payload = dict(trade)
                    if "timestamp" not in payload or not payload["timestamp"]:
                        payload["timestamp"] = report.timestamp
                    trades_payload.append(payload)
                self._client.log_trades(trades_payload)
            
            # Success! (only log occasionally to avoid spam)
            import random
            if random.random() < 0.1:  # Log 10% of successful pushes
                log_line(f"[{report.bot_id}] ✓ Telemetry pushed successfully")
                
        except Exception as exc:  # pragma: no cover - network failures logged
            # Print detailed error to console
            import traceback
            error_msg = f"[{report.bot_id}] ✗ BackendRelayTelemetry failed: {exc}"
            log_line(error_msg)
            # Also print stack trace for debugging
            print(f"\n{'='*80}")
            print(f"TELEMETRY ERROR: {exc}")
            print(f"{'='*80}")
            traceback.print_exc()
            print(f"{'='*80}\n")

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:  # pragma: no cover - not used
        return

    def close(self) -> None:
        self._client.close()


class CallbackTelemetry:
    def __init__(self, callback, intraday_callback=None) -> None:
        self._callback = callback
        self._intraday_callback = intraday_callback

    def publish(self, report: CycleReport) -> None:
        if self._callback:
            self._callback(_report_to_payload(report))

    def publish_intraday(self, symbol: str, rows: Iterable[Dict[str, Any]]) -> None:
        if self._intraday_callback:
            self._intraday_callback(symbol, list(rows))

    def close(self) -> None:
        return


def _report_to_payload(report: CycleReport) -> Dict[str, Any]:
    metrics_dict = asdict(report.metrics)
    return {
        "bot_id": report.bot_id,
        "timestamp": report.timestamp.isoformat(),
        "status": report.status,
        "equity": report.equity,
        "cash": report.cash,
        "positions_value": report.positions_value,
        "metrics": metrics_dict,
        "positions": report.positions,
        "prices": report.prices,
        "trades": report.trades,
        "batches": report.batches,
        "total_fees": report.total_fees,
        "avg_slippage_bps": report.avg_slippage_bps,
        "recent_trades": report.recent_trades or [],
    }



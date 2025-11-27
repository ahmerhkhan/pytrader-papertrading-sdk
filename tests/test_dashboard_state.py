from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytrader.dashboard.state import DashboardState


def test_dashboard_state_initial_snapshot():
    state = DashboardState(bot_id="demo", symbols=["OGDC", "HBL"])
    snapshot = state.snapshot()

    assert snapshot["bot"]["id"] == "demo"
    assert snapshot["positions"] == []
    assert snapshot["recent_trades"] == []
    assert snapshot["equity_history"] == []


def test_dashboard_state_ingest_updates_history():
    state = DashboardState(bot_id="demo", symbols=["OGDC"])

    payload = {
        "bot_id": "demo",
        "timestamp": "2024-01-01T10:00:00Z",
        "status": "ok",
        "equity": 1_200_000.0,
        "cash": 600_000.0,
        "positions_value": 600_000.0,
        "metrics": {"session_return_pct": 1.25},
        "positions": [
            {
                "symbol": "OGDC",
                "qty": 1000,
                "avg_cost": 100.0,
                "current_price": 105.0,
            }
        ],
        "trades": [
            {
                "timestamp": "2024-01-01T09:45:00Z",
                "symbol": "OGDC",
                "side": "BUY",
                "quantity": 1000,
                "price": 100.0,
                "pnl_realized": 0.0,
            }
        ],
    }

    snapshot = state.ingest(payload)

    assert snapshot["equity"] == 1_200_000.0
    assert snapshot["metrics"]["session_return_pct"] == 1.25
    assert snapshot["positions"][0]["symbol"] == "OGDC"
    assert snapshot["recent_trades"][0]["symbol"] == "OGDC"
    assert len(snapshot["equity_history"]) == 1


"""
Demo dashboard publisher that runs entirely on dummy data.

Usage:
    python examples/demo_dashboard.py --port 8787

Then open http://localhost:8787 to preview the live dashboard UI.
"""

from __future__ import annotations

import argparse
import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from pytrader.dashboard.server import DashboardServer


SYMBOLS = ["OGDC", "HBL", "UBL", "PSO"]


@dataclass
class Position:
    symbol: str
    qty: int
    avg_cost: float


def _initial_positions() -> List[Position]:
    random.seed(42)
    return [
        Position(symbol=symbol, qty=random.randint(300, 1200), avg_cost=random.uniform(80, 160))
        for symbol in SYMBOLS
    ]


def _generate_prices(base_prices: Dict[str, float]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for symbol, base in base_prices.items():
        move = random.uniform(-1.25, 1.50)
        prices[symbol] = max(10.0, base * (1 + move / 100))
        base_prices[symbol] = prices[symbol]
    return prices


def _build_payload(
    *,
    timestamp: datetime,
    positions: List[Position],
    prices: Dict[str, float],
    cash: float,
    equity: float,
    total_pnl: float,
    trade_id: int,
) -> Dict[str, Any]:
    positions_payload = []
    trades_payload = []
    positions_value = 0.0
    for pos in positions:
        price = prices[pos.symbol]
        market_value = price * pos.qty
        positions_value += market_value
        positions_payload.append(
            {
                "symbol": pos.symbol,
                "qty": pos.qty,
                "avg_cost": pos.avg_cost,
                "current_price": price,
                "market_value": market_value,
            }
        )

    recent_symbol = random.choice(SYMBOLS)
    trade_price = prices[recent_symbol]
    side = random.choice(["BUY", "SELL"])
    qty = random.randint(100, 400)
    pnl_realized = random.uniform(-1_000, 2_000)
    trades_payload.append(
        {
            "timestamp": timestamp.isoformat(),
            "symbol": recent_symbol,
            "side": side,
            "quantity": qty,
            "price": trade_price,
            "pnl_realized": pnl_realized,
            "note": f"demo trade #{trade_id}",
        }
    )

    payload = {
        "bot_id": "demo-bot",
        "timestamp": timestamp.isoformat(),
        "status": "ok",
        "equity": equity,
        "cash": cash,
        "positions_value": positions_value,
        "positions": positions_payload,
        "trades": trades_payload,
        "prices": prices,
        "metrics": {
            "session_return_pct": total_pnl / max(1.0, equity - total_pnl) * 100,
            "daily_pnl": total_pnl,
        },
    }
    return payload


async def run_demo(host: str, port: int) -> None:
    print(f"Starting dashboard demo on http://{host}:{port} (Ctrl+C to stop)")
    server = DashboardServer(bot_id="demo-bot", symbols=SYMBOLS, host=host, port=port, log_level="info")
    server.start()

    positions = _initial_positions()
    base_prices = {sym: random.uniform(80, 160) for sym in SYMBOLS}
    cash = 500_000.0
    equity = 1_000_000.0
    total_pnl = 0.0
    trade_id = 1

    try:
        while True:
            prices = _generate_prices(base_prices)
            drift = random.uniform(-2_000, 4_000)
            total_pnl += drift
            equity = cash + sum(prices[pos.symbol] * pos.qty for pos in positions)
            payload = _build_payload(
                timestamp=datetime.now(timezone.utc),
                positions=positions,
                prices=prices,
                cash=cash,
                equity=equity,
                total_pnl=total_pnl,
                trade_id=trade_id,
            )
            trade_id += 1
            server.publish(payload)
            await asyncio.sleep(1.5)
    except KeyboardInterrupt:
        print("\nStopping dashboard demo...")
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PyTrader dashboard with dummy data.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the dashboard server.")
    parser.add_argument("--port", type=int, default=8787, help="Port for the dashboard server.")
    args = parser.parse_args()
    asyncio.run(run_demo(args.host, args.port))


if __name__ == "__main__":
    main()


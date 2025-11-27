from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running without installing the package globally
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pytrader.dashboard.server import DashboardServer


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standalone PyTrader dashboard with synthetic data.",
    )
    parser.add_argument("--bot-id", default="demo-bot", help="Bot identifier to display.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["OGDC", "HBL"],
        help="Symbols to include in the mock portfolio.",
    )
    parser.add_argument("--port", type=int, default=8877, help="Dashboard port.")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between synthetic updates.",
    )
    parser.add_argument(
        "--start-equity",
        type=float,
        default=1_000_000.0,
        help="Initial equity value.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    server = DashboardServer(
        bot_id=args.bot_id,
        symbols=args.symbols,
        port=args.port,
    )
    server.start()

    print(
        f"\nPyTrader dashboard demo is live at http://127.0.0.1:{args.port} "
        "(Ctrl+C to stop)\n"
    )

    equity = args.start_equity
    cash = equity * 0.4
    positions_value = equity - cash
    cycle = 0

    try:
        while True:
            drift = random.uniform(-0.8, 1.2)
            equity = max(100_000.0, equity * (1 + drift / 100))
            cash = max(0.0, cash * (1 + random.uniform(-0.2, 0.2) / 100))
            positions_value = max(0.0, equity - cash)

            positions = []
            for symbol in args.symbols:
                qty = random.randint(100, 1500)
                avg_cost = random.uniform(50, 120)
                current_price = avg_cost * random.uniform(0.95, 1.08)
                positions.append(
                    {
                        "symbol": symbol,
                        "qty": qty,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "unrealized_pnl": (current_price - avg_cost) * qty,
                    }
                )

            trades = [
                {
                    "timestamp": _timestamp(),
                    "symbol": random.choice(args.symbols),
                    "side": random.choice(["BUY", "SELL"]),
                    "quantity": random.randint(100, 1500),
                    "price": random.uniform(60, 120),
                    "pnl_realized": random.uniform(-1_000, 2_000),
                }
            ]

            metrics = {
                "session_return_pct": random.uniform(-1, 1),
                "cumulative_return_pct": random.uniform(-10, 15),
                "sharpe_ratio": random.uniform(0.5, 2.5),
                "sortino_ratio": random.uniform(0.8, 3.0),
                "max_drawdown_pct": random.uniform(-12, -2),
            }

            server.publish(
                {
                    "bot_id": args.bot_id,
                    "timestamp": _timestamp(),
                    "status": "ok",
                    "equity": equity,
                    "cash": cash,
                    "positions_value": positions_value,
                    "positions": positions,
                    "trades": trades,
                    "metrics": metrics,
                    "batches": [],
                    "total_fees": cycle * 10.0,
                    "avg_slippage_bps": random.uniform(2, 12),
                }
            )
            cycle += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping dashboard demo...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()


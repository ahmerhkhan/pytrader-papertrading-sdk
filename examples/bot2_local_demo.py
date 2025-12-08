"""
Local paper-trading demo that streams telemetry to the PyTrader backend
under bot_id="bot2", so it can be visualized on the web frontend.

Run from repo root:

    python -m pytrader_sdk.examples.bot2_local_demo
"""

from __future__ import annotations

import os
from typing import Optional

from pytrader import Trader, Strategy


class Bot2Strategy(Strategy):
    def on_data(self, data):
        # Simple demo: keep buying a small OGDC position
        self.buy("OGDC", 25)


def run_bot2(max_cycles: Optional[int] = None) -> None:
    api_token = os.getenv("PYTRADER_API_TOKEN", "ahmer-token")
    backend_url = os.getenv("PYTRADER_BACKEND_URL")  # optional, defaults to production

    trader = Trader(
        strategy=Bot2Strategy,
        symbols=["OGDC"],
        cycle_minutes=15,
        bot_id="bot2",
    )

    run_kwargs = {
        "api_token": api_token,
        "warm_start": True,
        "max_cycles": max_cycles,
    }
    if backend_url:
        run_kwargs["backend_url"] = backend_url

    trader.run_paper_trading(**run_kwargs)


if __name__ == "__main__":
    run_bot2()



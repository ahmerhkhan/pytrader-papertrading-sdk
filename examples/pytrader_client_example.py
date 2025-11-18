"""
Minimal examples for the telemetry-first PyTrader workflow.
"""

from datetime import datetime, timezone

from pytrader import PyTrader, Trader, Strategy
from pytrader.sdk import run_backtest

# ---------------------------------------------------------------------------
# 1) Local backtest (no code uploaded to backend)
# ---------------------------------------------------------------------------
backtest = run_backtest(
    strategy_name="sma_momentum",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-03-31",
    api_token="USER_TOKEN",
    backend_url="http://localhost:8000",
)
print("Backtest final equity:", backtest["summary"]["final_portfolio_value"])


# ---------------------------------------------------------------------------
# 2) Local paper trading loop
# ---------------------------------------------------------------------------
class DemoStrategy(Strategy):
    def on_data(self, data):
        self.buy("OGDC", 25)


trader = Trader(
    strategy=DemoStrategy,
    symbols=["OGDC"],
    cycle_minutes=15,
    bot_id="example-bot",
)
trader.run_paper_trading(
    api_token="USER_TOKEN",
    backend_url="http://localhost:8000",
    warm_start=False,
    max_cycles=2,
)


# ---------------------------------------------------------------------------
# 3) Push additional telemetry + read dashboards
# ---------------------------------------------------------------------------
client = PyTrader(
    bot_id="example-bot",
    api_token="USER_TOKEN",
    backend_url="http://localhost:8000",
)

client.log_event("Manual adjustment made", level="info")
client.update_portfolio(
    equity=101_250,
    cash=90_000,
    positions={"OGDC": 150},
    timestamp=datetime.now(timezone.utc),
    status="running",
)

snapshot = client.get_portfolio()
performance = client.get_performance()
print("Latest snapshot:", snapshot)
print("Recent Sharpe:", performance["snapshots"][-1]["metrics"].get("sharpe_ratio"))

# PyTrader SDK Usage Guide

This guide shows you how to use the PyTrader SDK for backtesting and paper trading.

---

## 1. Installation

```bash
pip install "git+https://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@github.com/ahmerhkhan/pytrader-papertrading-sdk.git"
```

**Note:** The SDK is installed directly from the private GitHub repository. Do not use `pip install pytrader` as it will not work.

**Requirements:**
- Python 3.10+
- Backend server URL and API token

---

## 2. Setup

Set your backend URL and API token:

```bash
export PYTRADER_BACKEND_URL="http://localhost:8000"
export PYTRADER_API_TOKEN="your-token-here"
```

Or pass them directly when creating the client:

```python
from pytrader import PyTrader

client = PyTrader(
    api_token="your-token-here",
    backend_url="http://localhost:8000"
)
```

---

## 3. Quick Start: Backtesting (Local)

```python
from pytrader.sdk import run_backtest

result = run_backtest(
    strategy_name="sma_momentum",
    symbols=["OGDC", "HBL"],
    start="2024-01-01",
    end="2024-06-30",
    api_token="your-token",
    backend_url="http://localhost:8000",
    initial_cash=1_000_000.0,
)

print("Final equity:", result["summary"]["final_portfolio_value"])
print("Trades executed:", len(result["trades"]))
```

---

## 4. Quick Start: Paper Trading (Local Loop)

```python
from pytrader import Trader, Strategy

class MeanReversion(Strategy):
    def on_data(self, data):
        self.buy("OGDC", 100)

trader = Trader(strategy=MeanReversion, symbols=["OGDC"], cycle_minutes=15, bot_id="founder-bot-1")
trader.run_paper_trading(
    api_token="your-token",
    backend_url="http://localhost:8000",
    warm_start=True,
)
```

---

## 5. Streaming Telemetry

```python
from pytrader import PyTrader
from datetime import datetime, timezone

client = PyTrader(
    bot_id="founder-bot-1",
    api_token="your-token",
    backend_url="http://localhost:8000",
)

client.update_portfolio(
    equity=105_000,
    cash=90_000,
    positions={"OGDC": 100},
    timestamp=datetime.now(timezone.utc),
    status="running",
)

client.update_performance(
    equity=105_000,
    cash=90_000,
    positions_value=15_000,
    metrics={"total_return_pct": 5.0},
)

client.log_trades([
    {"symbol": "OGDC", "side": "BUY", "quantity": 100, "price": 95.0},
])
```

---

## 6. Reading Snapshots

```python
portfolio = client.get_portfolio()
performance = client.get_performance(limit=50)
trades = client.get_trades(limit=20)
logs = client.get_logs(limit=20)

print("Latest equity:", portfolio["equity"])
print("Recent Sharpe:", performance["snapshots"][-1]["metrics"].get("sharpe_ratio"))
```

---

## 5. Getting Market Data

### Get Available Symbols

```python
symbols = client.get_symbols()
for sym in symbols:
    print(f"{sym['symbol']}: {sym.get('name', 'N/A')}")
```

### Get Intraday Data

```python
# Get last 2 days of intraday data
data = client.get_intraday("OGDC", days=2)
print(f"Retrieved {len(data)} data points")
```

### Get Historical Data

```python
# Get historical daily data
historical = client.get_historical(
    symbol="OGDC",
    start="2024-01-01",
    end="2024-12-31",
    interval="1d"
)
```

---

## 7. Complete Examples

See `examples/pytrader_client_example.py` for complete working examples.

---

## Next Steps

- Read the [API Reference](sdk_reference.md) for complete method documentation
- Check out [Custom Strategies](custom_strategy.md) to create your own algorithms
- Explore [Advanced Patterns](advanced_strategy_guide.md) for complex strategies

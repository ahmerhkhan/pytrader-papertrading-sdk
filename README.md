# PyTrader SDK

**A Python SDK for algorithmic trading on the Pakistan Stock Exchange (PSX).**

PyTrader provides a simple, powerful interface for backtesting strategies and running paper trading bots.

---

## Features

- 📊 **Backtesting** - Test strategies on historical data
- 💹 **Paper Trading** - Run live paper trading bots with real-time data
- 📈 **Technical Indicators** - SMA, EMA, VWAP, RSI, MACD, Bollinger Bands
- 🎯 **Built-in Strategies** - Pre-configured strategies ready to use
- 🔧 **Custom Strategies** - Create your own trading algorithms
- 📝 **Performance Metrics** - Comprehensive analytics and reporting

---

## Installation

```bash
pip install pytrader
```

**Requirements:**
- Python 3.10+
- Backend server URL and API token

---

## Quick Start

### 1. Setup

Set your API token (backend URL is automatically configured):

```bash
export PYTRADER_API_TOKEN="your-token-here"
```

**Note:** The backend URL is automatically set to the production deployment. You only need to provide your API token.

### 2. Basic Usage - Reading Data from Backend

```python
from pytrader import PyTrader

# Initialize client (for reading data/telemetry from backend)
client = PyTrader(
    bot_id="my-bot",
    api_token="your-token-here"
)

# Get available symbols
symbols = client.get_symbols()
print(f"Available symbols: {len(symbols)}")

# Get market data
data = client.get_intraday("OGDC", days=2)
print(f"Data points: {len(data)}")

# Get portfolio snapshot
portfolio = client.get_portfolio()
print(f"Equity: {portfolio['equity']:,.2f}")

# Get performance metrics
performance = client.get_performance()
print(f"Metrics: {performance['metrics']}")
```

### 3. Running Paper Trading with Custom Strategy

```python
import asyncio
from pytrader import Trader, Strategy
from pytrader.indicators import SMA

# Define your custom strategy
class MyStrategy(Strategy):
    def on_start(self):
        """Called once before trading begins."""
        self.sma_period = 20
        self.position_size = 100
    
    def on_data(self, data):
        """
        Called every cycle with fresh market data.
        
        Args:
            data: Dict mapping symbol to DataFrame with OHLCV data
        """
        for symbol, df in data.items():
            if len(df) < self.sma_period:
                continue
            
            # Calculate indicators
            sma = SMA(df['close'], self.sma_period)
            current_price = df['close'].iloc[-1]
            
            # Generate trading signals
            if current_price > sma.iloc[-1]:
                self.buy(symbol, self.position_size)
            else:
                self.sell(symbol, self.position_size)

# Create trader and run paper trading
trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC", "HBL"],
    cycle_minutes=15,
    position_notional=100_000.0,
    bot_id="my-custom-bot"
)

# Run paper trading (synchronous wrapper)
trader.run_paper_trading(
    api_token="your-token-here",
    warm_start=True,  # Replay today's data first
    max_cycles=None   # Run indefinitely until interrupted
)
```

### 4. Running Paper Trading with Built-in Strategy

```python
import asyncio
from pytrader import start_paper_trading

# Start paper trading with built-in strategy
start_paper_trading(
    strategy_name="sma_momentum",
    symbols=["OGDC", "HBL"],
    api_token="your-token-here",
    cycle_minutes=15,
    capital=1_000_000.0,
    position_notional=100_000.0,
    warm_start=True,
    max_cycles=None  # Run indefinitely
)
```

### 5. Running Backtests

```python
from pytrader import Trader, Strategy

# With custom strategy
class MyStrategy(Strategy):
    def on_data(self, data):
        # Your trading logic
        pass

trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC"],
    initial_cash=1_000_000.0
)

result = trader.run_backtest(
    start="2024-01-01",
    end="2024-12-31",
    api_token="your-token-here"
)

print(f"Total Return: {result['summary']['final_portfolio_value']:,.2f}")
print(f"Total Trades: {len(result['trades'])}")
```

Or with built-in strategies:

```python
from pytrader import run_backtest

result = run_backtest(
    strategy_name="sma_momentum",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31",
    api_token="your-token-here",
    initial_cash=1_000_000.0
)
```

---

## API Reference

### PyTrader Client (Telemetry/Data Reading)

The `PyTrader` client is used to read data and telemetry from the backend. **It does NOT run strategies** - strategies run locally using `Trader` or `start_paper_trading()`.

```python
client = PyTrader(
    bot_id="my-bot",
    api_token="your-token"
)
```

#### Data Methods

- `get_symbols()` - Get list of available symbols
- `get_intraday(symbol, days=2)` - Get intraday market data
- `get_historical(symbol, start, end, interval="1d")` - Get historical data
- `get_portfolio(bot_id=None)` - Get portfolio snapshot
- `get_performance(bot_id=None, limit=100)` - Get performance metrics
- `get_trades(bot_id=None, limit=100)` - Get trade history
- `get_logs(bot_id=None, limit=200)` - Get bot logs

### Trader Class (Run Strategies Locally)

The `Trader` class is used to run custom strategies locally for backtesting or paper trading.

```python
from pytrader import Trader, Strategy

trader = Trader(
    strategy=MyStrategy,  # Your Strategy class or instance
    symbols=["OGDC", "HBL"],
    cycle_minutes=15,
    initial_cash=1_000_000.0,
    position_notional=100_000.0,
    bot_id="my-bot"
)

# Run backtest
result = trader.run_backtest(start="2024-01-01", end="2024-12-31", api_token="...")

# Run paper trading (synchronous)
trader.run_paper_trading(api_token="...", ...)

# Or async
await trader.start_paper_trading(api_token="...", ...)
```

### Built-in Functions

For built-in strategies, use these convenience functions:

```python
from pytrader import run_backtest, start_paper_trading

# Backtest
result = run_backtest(
    strategy_name="sma_momentum",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31",
    api_token="..."
)

# Paper trading
start_paper_trading(
    strategy_name="sma_momentum",
    symbols=["OGDC"],
    api_token="..."
)
```

---

## Built-in Strategies

PyTrader includes several pre-built strategies:

- `sma` - Simple moving average momentum
- `dual_sma_momentum` - Dual SMA crossover
- `rsi_momentum` - RSI-based momentum
- `macd_crossover` - MACD crossover signals
- `bollinger_mean_reversion` - Bollinger Bands mean reversion
- `vwap_reversion` - VWAP reversion strategy

### Example: Using Built-in Strategies

```python
from pytrader import run_backtest, start_paper_trading

# Run backtest with built-in strategy
result = run_backtest(
    strategy_name="sma_momentum",
    symbols=["OGDC", "HBL"],
    start="2024-01-01",
    end="2024-12-31",
    api_token="your-token",
    initial_cash=1_000_000.0,
    position_notional=100_000.0
)

# Start paper trading with built-in strategy
start_paper_trading(
    strategy_name="dual_sma_momentum",
    symbols=["OGDC"],
    api_token="your-token",
    cycle_minutes=15,
    capital=1_000_000.0
)
```

---

## Creating Custom Strategies

Create your own trading strategies by subclassing `Strategy`:

```python
from pytrader import Strategy
from pytrader.indicators import SMA

class MyStrategy(Strategy):
    def on_start(self):
        """Called once before trading begins."""
        self.sma_period = 20
        self.position_size = 100
    
    def on_data(self, data):
        """
        Called every cycle with fresh market data.
        
        Args:
            data: Dict mapping symbol to DataFrame with OHLCV data
        """
        for symbol, df in data.items():
            if len(df) < self.sma_period:
                continue
            
            # Calculate indicators
            sma = SMA(df['close'], self.sma_period)
            current_price = df['close'].iloc[-1]
            
            # Generate trading signals
            if current_price > sma.iloc[-1]:
                self.buy(symbol, self.position_size)
            else:
                self.sell(symbol, self.position_size)
    
    def on_end(self):
        """Called once after trading ends."""
        print("Strategy execution completed")
```

**Note:** Custom strategies run **locally** on your machine. The backend only receives telemetry (portfolio snapshots, trades, logs) for monitoring via the web dashboard.

---

## Technical Indicators

PyTrader provides a comprehensive set of technical indicators:

```python
from pytrader.indicators import (
    SMA, EMA, VWAP, RSI, MACD, BollingerBands
)

# Simple Moving Average
sma = SMA(df['close'], period=20)

# Exponential Moving Average
ema = EMA(df['close'], period=12)

# Volume Weighted Average Price
vwap = VWAP(df, price_col='close', volume_col='volume')

# Relative Strength Index
rsi = RSI(df['close'], period=14)

# MACD (returns macd_line, signal_line, histogram)
macd, signal, histogram = MACD(df['close'])

# Bollinger Bands (returns upper, middle, lower)
upper, middle, lower = BollingerBands(df['close'], period=20, std_dev=2.0)
```

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PYTRADER_API_TOKEN` | Your API token | Yes |
| `PYTRADER_BACKEND_URL` | Backend server URL (optional, defaults to production) | No |

### Backtest Parameters

```python
from pytrader import Trader, Strategy

trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC"],
    initial_cash=1_000_000.0,      # Starting capital
    position_notional=100_000.0,    # Position size per trade
)

result = trader.run_backtest(
    start="2024-01-01",
    end="2024-12-31",
    api_token="your-token",
    min_lot=1,                      # Minimum lot size
    slippage_bps=5.0,               # Slippage in basis points
    commission_per_share=0.02,     # Commission per share
    commission_pct_notional=0.0005, # Commission as % of notional
    allow_short=False,              # Allow short selling (default: False)
    interval="1d"                   # Data interval
)
```

### Paper Trading Parameters

```python
from pytrader import Trader, Strategy

trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC"],
    cycle_minutes=15,               # Trading cycle duration
    position_notional=100_000.0,    # Position size per trade
    bot_id="my-bot",                # Bot identifier
    initial_cash=1_000_000.0,       # Starting capital
)

trader.run_paper_trading(
    api_token="your-token",
    warm_start=True,                # Replay today's data first
    max_cycles=None,                # Max cycles (None = unlimited)
    log_dir="logs",                 # Directory for logs
)
```

---

## Examples

See `examples/pytrader_client_example.py` for a complete working example.

### Complete Backtest Example

```python
from pytrader import Trader, Strategy

class MyStrategy(Strategy):
    def on_data(self, data):
        for symbol, df in data.items():
            if len(df) >= 20:
                sma = df['close'].rolling(20).mean()
                if df['close'].iloc[-1] > sma.iloc[-1]:
                    self.buy(symbol, 100)

trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC", "HBL"],
    initial_cash=1_000_000.0
)

# Run backtest
result = trader.run_backtest(
    start="2024-01-01",
    end="2024-12-31",
    api_token="your-token"
)

# Analyze results
summary = result['summary']
print(f"Initial Value: {summary['initial_portfolio_value']:,.2f}")
print(f"Final Value: {summary['final_portfolio_value']:,.2f}")
print(f"Total Return: {(summary['final_portfolio_value'] / summary['initial_portfolio_value'] - 1) * 100:.2f}%")
print(f"Total Trades: {len(result['trades'])}")
```

### Complete Paper Trading Example

```python
import asyncio
from pytrader import Trader, Strategy
from pytrader import PyTrader

class MyStrategy(Strategy):
    def on_data(self, data):
        for symbol, df in data.items():
            if len(df) >= 20:
                sma = df['close'].rolling(20).mean()
                if df['close'].iloc[-1] > sma.iloc[-1]:
                    self.buy(symbol, 100)

# Create trader
trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC"],
    cycle_minutes=15,
    bot_id="my-paper-bot"
)

# Start paper trading in background
async def run_bot():
    await trader.start_paper_trading(
        api_token="your-token",
        warm_start=True
    )

# Monitor bot from another process/script
client = PyTrader(
    bot_id="my-paper-bot",
    api_token="your-token"
)

# In a separate terminal/script, monitor the bot:
for _ in range(10):
    import time
    time.sleep(60)  # Wait 1 minute
    
    portfolio = client.get_portfolio()
    performance = client.get_performance()
    
    print(f"Equity: {portfolio['equity']:,.2f} | "
          f"Cash: {portfolio['cash']:,.2f} | "
          f"Positions: {len(portfolio['positions'])}")

# Run the bot (this blocks until interrupted)
asyncio.run(run_bot())
```

---

## Error Handling

The SDK raises `AuthenticationError` for authentication failures:

```python
from pytrader import AuthenticationError
from pytrader import Trader, Strategy

try:
    trader = Trader(strategy=MyStrategy, symbols=["OGDC"])
    trader.run_paper_trading(api_token="invalid-token", backend_url="...")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
```

---

## Documentation

- [`docs/usage.md`](docs/usage.md) - Detailed usage guide
- [`docs/custom_strategy.md`](docs/custom_strategy.md) - Creating custom strategies
- [`docs/advanced_strategy_guide.md`](docs/advanced_strategy_guide.md) - Advanced patterns
- [`docs/config.md`](docs/config.md) - Configuration options
- [`docs/sdk_reference.md`](docs/sdk_reference.md) - Complete API reference

---

## Support

For questions, issues, or contributions, please open an issue on GitHub.

---

**PyTrader** - Algorithmic trading made simple. 📈

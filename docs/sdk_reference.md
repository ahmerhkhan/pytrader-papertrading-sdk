# PyTrader SDK Reference

Complete API reference for the PyTrader SDK client.

---

## PyTrader Client

The main client class for interacting with the PyTrader backend.

### Constructor

```python
PyTrader(api_token: str, backend_url: Optional[str] = None, timeout: float = 30.0)
```

**Parameters:**
- `api_token` (str, required): Your API token for authentication
- `backend_url` (str, optional): Backend server URL (defaults to `PYTRADER_BACKEND_URL` env var)
- `timeout` (float, optional): Request timeout in seconds (default: 30.0)

**Example:**
```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")
# or
client = PyTrader(api_token="your-token", backend_url="http://localhost:8000")
```

---

## Data Methods

### get_symbols()

Get list of available symbols.

```python
symbols = client.get_symbols()
```

**Returns:** `List[Dict[str, Any]]` - List of symbol dictionaries with `symbol`, `name`, `sector` fields

**Example:**
```python
symbols = client.get_symbols()
for sym in symbols:
    print(f"{sym['symbol']}: {sym.get('name', 'N/A')}")
```

---

### get_intraday()

Get intraday market data for a symbol.

```python
data = client.get_intraday(symbol: str, days: int = 2)
```

**Parameters:**
- `symbol` (str, required): Stock symbol (e.g., "OGDC")
- `days` (int, optional): Number of days of data to fetch (default: 2, max: 10)

**Returns:** `List[Dict[str, Any]]` - List of intraday data points

**Example:**
```python
data = client.get_intraday("OGDC", days=2)
print(f"Retrieved {len(data)} data points")
```

---

### get_historical()

Get historical market data for a symbol.

```python
data = client.get_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d"
)
```

**Parameters:**
- `symbol` (str, required): Stock symbol (e.g., "OGDC")
- `start` (str, optional): Start date in YYYY-MM-DD format
- `end` (str, optional): End date in YYYY-MM-DD format
- `interval` (str, optional): Data interval (default: "1d")

**Returns:** `List[Dict[str, Any]]` - List of historical data points

**Example:**
```python
data = client.get_historical(
    symbol="OGDC",
    start="2024-01-01",
    end="2024-12-31",
    interval="1d"
)
```

---

## Trading Methods

### backtest()

Run a backtest on historical data.

```python
result = client.backtest(
    strategy: str,
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    strategy_config: Optional[Dict[str, Any]] = None,
    initial_cash: Optional[float] = None,
    position_notional: Optional[float] = None,
    min_lot: Optional[int] = None,
    interval: str = "1d",
    slippage_bps: Optional[float] = None,
    commission_per_share: Optional[float] = None,
    commission_pct_notional: Optional[float] = None,
    allow_short: Optional[bool] = None,
    bot_id: Optional[str] = None
)
```

**Parameters:**
- `strategy` (str, required): Strategy name (e.g., "sma", "sma_momentum")
- `symbols` (List[str], required): List of symbols to trade
- `start` (str, required): Start date in YYYY-MM-DD format
- `end` (str, optional): End date in YYYY-MM-DD format (defaults to start)
- `strategy_config` (Dict, optional): Strategy configuration parameters
- `initial_cash` (float, optional): Starting cash
- `position_notional` (float, optional): Position size per trade
- `min_lot` (int, optional): Minimum lot size
- `interval` (str, optional): Data interval (default: "1d")
- `slippage_bps` (float, optional): Slippage in basis points
- `commission_per_share` (float, optional): Commission per share
- `commission_pct_notional` (float, optional): Commission as percentage of notional
- `allow_short` (bool, optional): Allow short selling (default: False)
- `bot_id` (str, optional): Bot ID for this backtest

**Returns:** `Dict[str, Any]` - Backtest results with:
- `bot_id`: Bot identifier
- `start`, `end`: Date range
- `metrics`: Performance metrics dictionary
- `equity_curve`: Equity curve data
- `trades`: List of trades
- `hourly_summary`: Hourly summary (if available)
- `summary`: Summary statistics

**Example:**
```python
result = client.backtest(
    strategy="sma",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31",
    initial_cash=1_000_000.0
)
```

---

### start_bot()

Start a paper trading bot.

```python
bot = client.start_bot(
    strategy: str,
    symbols: List[str],
    bot_id: Optional[str] = None,
    cycle_minutes: int = 15,
    position_notional: float = 100_000.0,
    min_lot: Optional[int] = None,
    slippage_bps: Optional[float] = None,
    commission_per_share: Optional[float] = None,
    commission_pct_notional: Optional[float] = None,
    allow_short: Optional[bool] = None,
    warm_start: bool = True,
    max_cycles: Optional[int] = None,
    user_id: Optional[str] = None,
    notes: Optional[str] = None
)
```

**Parameters:**
- `strategy` (str, required): Strategy name
- `symbols` (List[str], required): List of symbols to trade
- `bot_id` (str, optional): Bot identifier (auto-generated if not provided)
- `cycle_minutes` (int, optional): Trading cycle duration in minutes (default: 15)
- `position_notional` (float, optional): Position size per trade (default: 100000.0)
- `min_lot` (int, optional): Minimum lot size
- `slippage_bps` (float, optional): Slippage in basis points
- `commission_per_share` (float, optional): Commission per share
- `commission_pct_notional` (float, optional): Commission as percentage of notional
- `allow_short` (bool, optional): Allow short selling (default: False)
- `warm_start` (bool, optional): Enable warm-start replay (default: True)
- `max_cycles` (int, optional): Maximum cycles to run (None = run indefinitely)
- `user_id` (str, optional): User identifier
- `notes` (str, optional): Notes for this bot

**Returns:** `Dict[str, Any]` - Bot information with:
- `bot_id`: Bot identifier
- `status`: Bot status ("running")
- `message`: Status message

**Example:**
```python
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    cycle_minutes=15
)
```

---

### stop_bot()

Stop a running bot.

```python
result = client.stop_bot(bot_id: str)
```

**Parameters:**
- `bot_id` (str, required): Bot identifier

**Returns:** `Dict[str, Any]` - Bot status with `bot_id`, `status` ("stopped"), `message`

---

### list_bots()

List all bots for the authenticated user.

```python
bots = client.list_bots()
```

**Returns:** `List[Dict[str, Any]]` - List of bot summaries

---

### get_bot_status()

Get status of a specific bot.

```python
status = client.get_bot_status(bot_id: str)
```

**Parameters:**
- `bot_id` (str, required): Bot identifier

**Returns:** `Dict[str, Any]` - Bot status information

---

### get_portfolio()

Get portfolio information for a bot.

```python
portfolio = client.get_portfolio(bot_id: str)
```

**Parameters:**
- `bot_id` (str, required): Bot identifier

**Returns:** `Dict[str, Any]` - Portfolio information with:
- `bot_id`: Bot identifier
- `equity`: Total equity
- `cash`: Available cash
- `positions_value`: Total value of positions
- `positions`: List of position entries

---

### get_performance()

Get performance metrics for a bot.

```python
performance = client.get_performance(bot_id: str)
```

**Parameters:**
- `bot_id` (str, required): Bot identifier

**Returns:** `Dict[str, Any]` - Performance information with:
- `bot_id`: Bot identifier
- `equity`: Total equity
- `cash`: Available cash
- `positions_value`: Total value of positions
- `metrics`: Performance metrics dictionary

---

### get_live_metrics()

Get live metrics for a running bot.

```python
metrics = client.get_live_metrics(bot_id: str, history_limit: int = 50)
```

**Parameters:**
- `bot_id` (str, required): Bot identifier
- `history_limit` (int, optional): Number of historical metrics to include (default: 50, max: 500)

**Returns:** `Dict[str, Any]` - Live metrics and history

---

### get_trade_logs()

Get trade logs for a bot.

```python
trades = client.get_trade_logs(bot_id: str, limit: int = 100)
```

**Parameters:**
- `bot_id` (str, required): Bot identifier
- `limit` (int, optional): Maximum number of trades to return (default: 100, max: 500)

**Returns:** `List[Dict[str, Any]]` - List of trade log entries

---

## Exceptions

### AuthenticationError

Raised when authentication fails.

```python
from pytrader import PyTrader, AuthenticationError

try:
    client = PyTrader(api_token="invalid-token")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
```

---

## Context Manager

The client supports context manager usage:

```python
with PyTrader(api_token="your-token") as client:
    data = client.get_intraday("OGDC")
    # Client automatically closed when exiting context
```

---

## Built-in Strategies

Available strategy names for `backtest()` and `start_bot()`:

- `sma` / `sma_momentum` - Simple moving average momentum
- `dual_sma_momentum` - Dual SMA crossover
- `rsi_momentum` - RSI-based momentum
- `macd_crossover` - MACD crossover signals
- `bollinger_mean_reversion` - Bollinger Bands mean reversion
- `vwap_reversion` - VWAP reversion strategy
- `ml_layer` - Machine learning layer strategy
- `buying_on_up` - Buy on upward momentum
- `buying_on_down` - Buy on downward momentum

---

## Technical Indicators

Available indicators (for use in custom strategies):

- `SMA(series, period)` - Simple Moving Average
- `EMA(series, period)` - Exponential Moving Average
- `VWAP(dataframe, price_col, volume_col)` - Volume Weighted Average Price
- `RSI(series, period)` - Relative Strength Index
- `MACD(series)` - Moving Average Convergence Divergence
- `BollingerBands(series, period, std_dev)` - Bollinger Bands

See [`indicators.py`](../pytrader/indicators.py) for complete indicator documentation.

# Creating Custom Trading Strategies

PyTrader SDK allows you to create custom trading strategies. Custom strategies are executed on the backend server. This guide shows you how to write your own strategies.

**Note:** Custom strategies must be registered with the backend. Contact your administrator for details on uploading custom strategies.

## Quick Start

```python
from pytrader import Strategy
from pytrader.indicators import SMA

class MyStrategy(Strategy):
    def on_start(self):
        self.sma_period = 20
    
    def on_data(self, data):
        df = data['OGDC']
        sma = SMA(df['close'], self.sma_period)
        
        if df['close'].iloc[-1] > sma.iloc[-1]:
            self.buy('OGDC', 100)
        else:
            self.sell('OGDC', 100)
```

Once your strategy is registered with the backend, you can use it with the PyTrader client:

```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")

# Use your custom strategy
result = client.backtest(
    strategy="my_strategy",  # Strategy name as registered on backend
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31"
)
```

## Strategy Interface

### Base Class: `Strategy`

All custom strategies must inherit from `pytrader.Strategy`:

```python
from pytrader import Strategy

class MyStrategy(Strategy):
    def on_start(self):
        """Called once before trading begins."""
        pass
    
    def on_data(self, data):
        """Called every cycle with fresh market data."""
        pass
    
    def on_end(self):
        """Called once after trading ends."""
        pass
```

### Methods

#### `on_start()`

Called once before trading begins. Use this to:
- Initialize indicators
- Set strategy parameters
- Prepare any data structures

```python
def on_start(self):
    self.sma_period = 20
    self.min_bars = 25
    self.position_size = 100
```

#### `on_data(data: Dict[str, pd.DataFrame])`

Called every cycle (15 minutes by default) with fresh OHLCV data.

**Parameters:**
- `data`: Dictionary mapping symbol to DataFrame with columns:
  - `ts`: Timestamp
  - `close` or `price`: Closing price
  - `volume`: Trading volume
  - Optional: `open`, `high`, `low` (if available)

**Example:**
```python
def on_data(self, data):
    ogdc_df = data['OGDC']
    
    # Ensure we have enough data
    if len(ogdc_df) < self.min_bars:
        return
    
    # Calculate indicator
    sma = SMA(ogdc_df['close'], self.sma_period)
    current_price = ogdc_df['close'].iloc[-1]
    current_sma = sma.iloc[-1]
    
    # Generate signals
    if current_price > current_sma:
        self.buy('OGDC', self.position_size)
    elif current_price < current_sma:
        self.sell('OGDC', self.position_size)
```

#### `on_end()`

Called once after trading ends. Use this for:
- Final calculations
- Cleanup
- Summary reporting

```python
def on_end(self):
    print("Strategy execution completed")
```

### Order Methods

#### `buy(symbol: str, quantity: int)`

Place a buy order.

```python
self.buy('OGDC', 100)  # Buy 100 shares of OGDC
```

#### `sell(symbol: str, quantity: int)`

Place a sell order.

```python
self.sell('OGDC', 100)  # Sell 100 shares of OGDC
```

**Note:** Orders are executed at the end of each cycle. You can place multiple orders per cycle, but only one order per symbol will be processed (the last one).

## Available Indicators

PyTrader provides common technical indicators:

```python
from pytrader.indicators import SMA, EMA, VWAP, RSI, MACD, BollingerBands

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

## Running Strategies

### Backtesting

```python
from pytrader import Trader, Strategy

class MyStrategy(Strategy):
    def on_data(self, data):
        # Your logic here
        pass

trader = Trader(
    strategy=MyStrategy,
    symbols=['OGDC', 'HBL'],
    initial_cash=1_000_000.0,
    cycle_minutes=15,
)

result = trader.run_backtest(
    start='2024-01-01',
    end='2024-12-31',
)

print(f"Total Return: {result['metrics'].total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result['metrics'].sharpe_ratio}")
print(f"Trades: {len(result['trades'])}")
```

### Live Paper Trading

```python
import asyncio

async def main():
    trader = Trader(
        strategy=MyStrategy,
        symbols=['OGDC', 'HBL'],
        cycle_minutes=15,
    )
    
    await trader.start_paper_trading(
        # Optional: token for backend integration (not required for standalone SDK usage)
        warm_start=True,  # Replay today's data first
    )

asyncio.run(main())
```

## Strategy Loading

PyTrader supports multiple ways to specify strategies:

### 1. Class Instance

```python
trader = Trader(strategy=MyStrategy, symbols=['OGDC'])
```

### 2. Class Type

```python
trader = Trader(strategy=MyStrategy, symbols=['OGDC'])
```

### 3. Built-in Strategy Name

```python
trader = Trader(strategy='sma_momentum', symbols=['OGDC'])
```

Available built-in strategies:
- `sma_momentum`
- `dual_sma_momentum`
- `bollinger_mean_reversion`
- `vwap_reversion`

### 4. File Path

```python
# my_strategy.py
from pytrader import Strategy

class MyStrategy(Strategy):
    def on_data(self, data):
        # Your logic
        pass

# Load from file
trader = Trader(strategy='my_strategy.py', symbols=['OGDC'])
```

### 5. Module Path

```python
trader = Trader(
    strategy='user_strategies.my_strategy.MyStrategy',
    symbols=['OGDC'],
)
```

## Complete Example

```python
from pytrader import Trader, Strategy
from pytrader.indicators import SMA, EMA

class DualMovingAverageStrategy(Strategy):
    """Dual EMA crossover strategy."""
    
    def on_start(self):
        self.fast_period = 12
        self.slow_period = 26
        self.min_bars = self.slow_period + 5
    
    def on_data(self, data):
        for symbol, df in data.items():
            if len(df) < self.min_bars:
                continue
            
            # Ensure 'close' column exists
            if 'close' not in df.columns and 'price' in df.columns:
                df = df.copy()
                df['close'] = df['price']
            
            # Calculate EMAs
            fast_ema = EMA(df['close'], self.fast_period)
            slow_ema = EMA(df['close'], self.slow_period)
            
            if len(fast_ema) < 2:
                continue
            
            # Check for crossover
            current_fast = fast_ema.iloc[-1]
            current_slow = slow_ema.iloc[-1]
            prev_fast = fast_ema.iloc[-2]
            prev_slow = slow_ema.iloc[-2]
            
            # Buy signal: fast crosses above slow
            if prev_fast <= prev_slow and current_fast > current_slow:
                self.buy(symbol, 100)
            
            # Sell signal: fast crosses below slow
            elif prev_fast >= prev_slow and current_fast < current_slow:
                self.sell(symbol, 100)
    
    def on_end(self):
        print("Strategy execution completed")

# Run backtest
if __name__ == "__main__":
    trader = Trader(
        strategy=DualMovingAverageStrategy,
        symbols=['OGDC', 'HBL'],
        initial_cash=1_000_000.0,
    )
    
    result = trader.run_backtest(start='2024-01-01', end='2024-01-31')
    
    if result.get('metrics'):
        metrics = result['metrics']
        print(f"\nResults:")
        print(f"  Total Return: {metrics.total_return_pct:.2f}%")
        sharpe_ratio = getattr(metrics, "sharpe_ratio", None)
        sharpe_available = getattr(metrics, "sharpe_ratio_available", sharpe_ratio is not None)
        sharpe_str = f"{sharpe_ratio:.2f}" if sharpe_available and sharpe_ratio is not None else "-"
        print(f"  Sharpe Ratio: {sharpe_str}")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"  Trades: {len(result.get('trades', []))}")
```

## Best Practices

1. **Check Data Availability**: Always check if you have enough data before calculating indicators:
   ```python
   if len(df) < self.min_bars:
       return
   ```

2. **Handle Missing Columns**: Normalize column names:
   ```python
   if 'close' not in df.columns and 'price' in df.columns:
       df['close'] = df['price']
   ```

3. **Use Indicators Correctly**: Indicators return Series - access values with `.iloc[-1]`:
   ```python
   sma = SMA(df['close'], 20)
   current_sma = sma.iloc[-1]  # Latest value
   ```

4. **One Order Per Symbol**: Only the last order per symbol in each cycle is executed.

5. **Test First**: Always backtest your strategy before running live paper trading.

## Troubleshooting

### Strategy not generating signals

- Check that `on_data` is being called (add print statements)
- Verify data format (check DataFrame columns)
- Ensure indicators have enough data points

### Orders not executing

- Verify order format: `self.buy('SYMBOL', quantity)`
- Check that symbol matches exactly (case-insensitive)
- Ensure you have sufficient cash/positions

### Import errors

- Make sure you're using `from pytrader import Strategy` (not internal imports)
- Custom strategies must be registered with the backend before use
- Check that all dependencies are installed: `pip install pytrader`

## Next Steps

- See `examples/custom_strategy_example.py` for more examples
- See [`advanced_strategy_guide.md`](advanced_strategy_guide.md) for advanced patterns
- Check `docs/paper_trading_modes.md` for warm-start vs cold-start modes


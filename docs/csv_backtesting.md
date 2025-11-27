# CSV Backtesting Support

PyTrader SDK now supports backtesting with CSV files, allowing you to use your own historical data in various formats.

## Features

- **Flexible Column Mapping**: Automatically detects common column names or specify custom mappings
- **Multiple Formats**: Supports OHLCV, simple (symbol, timestamp, price, volume), and custom formats
- **Multiple Symbols**: Handle multiple symbols in a single CSV file
- **No API Required**: Run backtests without API tokens when using CSV data

## Quick Start

### Simple Format (symbol, timestamp, price, volume)

```python
from pytrader import Trader, Strategy

class MyStrategy(Strategy):
    def on_data(self, data):
        for symbol, df in data.items():
            # Your trading logic
            if df['close'].iloc[-1] > df['close'].iloc[-2]:
                self.buy(symbol, 100)

trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC"],
    initial_cash=1_000_000.0,
)

# CSV format: symbol,timestamp,price,volume
result = trader.run_backtest(
    csv_path="data.csv",
    # No API token needed!
)
```

### Custom Column Mapping

If your CSV has different column names:

```python
result = trader.run_backtest(
    csv_path="data.csv",
    csv_column_mapping={
        "timestamp": "ts",        # Your timestamp column
        "price": "close_price",   # Your price column
        "volume": "vol",          # Your volume column
        "symbol": "ticker",        # Your symbol column (if present)
    },
)
```

## CSV Format Requirements

### Minimum Required Columns

Your CSV must have at least:
- **Timestamp column**: Any name (auto-detected or specified)
- **Price column**: Any name (auto-detected or specified)

### Supported Formats

#### 1. Simple Format (Recommended for beginners)

```csv
symbol,timestamp,price,volume
OGDC,2024-01-01 09:00:00,95.50,1000000
OGDC,2024-01-01 09:15:00,95.75,1200000
```

**Auto-detection**: Works automatically, no mapping needed!

#### 2. OHLCV Format

```csv
symbol,datetime,open,high,low,close,volume
OGDC,2024-01-01 09:00:00,95.00,96.00,94.50,95.50,1000000
OGDC,2024-01-01 09:15:00,95.50,96.25,95.00,95.75,1200000
```

**Auto-detection**: Automatically detects OHLCV columns.

#### 3. Custom Format

```csv
ticker,ts,close_price,vol
OGDC,2024-01-01 09:00:00,95.50,1000000
```

**Mapping required**: Specify column mapping.

## Column Mapping

### Auto-Detection

The SDK automatically detects common column names:

**Timestamp**: `timestamp`, `ts`, `date`, `datetime`, `time`
**Price**: `price`, `p`, `last_price`, `last`, `close`, `c`, `closing_price`
**OHLC**: `open`, `o`, `high`, `h`, `low`, `l`, `close`, `c`
**Volume**: `volume`, `vol`, `v`, `qty`, `quantity`
**Symbol**: `symbol`, `sym`, `ticker`, `stock`

### Manual Mapping

If auto-detection fails or you have custom column names:

```python
csv_column_mapping = {
    "timestamp": "your_timestamp_col",
    "price": "your_price_col",      # Required if no OHLCV
    "open": "your_open_col",         # Optional
    "high": "your_high_col",         # Optional
    "low": "your_low_col",           # Optional
    "close": "your_close_col",       # Optional
    "volume": "your_volume_col",     # Optional (defaults to 0)
    "symbol": "your_symbol_col",     # Optional (for multi-symbol CSVs)
}
```

## Multiple Symbols

### Single CSV with Multiple Symbols

```csv
symbol,timestamp,price,volume
OGDC,2024-01-01 09:00:00,95.50,1000000
HBL,2024-01-01 09:00:00,150.00,2000000
OGDC,2024-01-01 09:15:00,95.75,1200000
HBL,2024-01-01 09:15:00,151.00,1800000
```

```python
trader = Trader(
    strategy=MyStrategy,
    symbols=["OGDC", "HBL"],  # Multiple symbols
)

result = trader.run_backtest(
    csv_path="multi_symbol_data.csv",
    csv_column_mapping={
        "timestamp": "timestamp",
        "price": "price",
        "volume": "volume",
        "symbol": "symbol",  # Important: specify symbol column
    },
)
```

### Separate CSV Files per Symbol

```python
# Use different CSV files for each symbol
# (Not directly supported - use single CSV with symbol column instead)
```

## Advanced Options

### Custom Delimiter

```python
result = trader.run_backtest(
    csv_path="data.tsv",
    csv_delimiter="\t",  # Tab-separated
)
```

### Custom Encoding

```python
result = trader.run_backtest(
    csv_path="data.csv",
    csv_encoding="latin-1",  # For non-UTF-8 files
)
```

### Date Filtering

```python
result = trader.run_backtest(
    csv_path="data.csv",
    start="2024-01-01",  # Optional: filter by start date
    end="2024-12-31",    # Optional: filter by end date
)
```

## Examples

See `examples/csv_backtest_example.py` for complete working examples.

## Limitations

1. **CSV files must be sorted by timestamp** (or will be sorted automatically)
2. **Symbol filtering**: If CSV contains symbol column, it's used for filtering. Otherwise, CSV is assumed to be for a single symbol.
3. **Missing OHLC**: If only price is provided, open/high/low are set equal to price (close).
4. **Missing volume**: Defaults to 0.0 if not provided.

## Error Handling

Common errors and solutions:

### "CSV file not found"
- Check the file path is correct
- Use absolute path if relative path doesn't work

### "Could not auto-detect timestamp column"
- Specify `csv_column_mapping` manually
- Ensure your CSV has a timestamp/date column

### "No price column available"
- Ensure your CSV has a price column (or OHLC columns)
- Specify column mapping if using custom names

### "No data found for symbol X in CSV"
- Check that symbol column exists and contains the requested symbol
- Ensure symbol names match (case-insensitive)

## Best Practices

1. **Use consistent timestamp format**: ISO format (YYYY-MM-DD HH:MM:SS) works best
2. **Include volume data**: Even if 0, it helps with liquidity checks
3. **Validate data**: Check for missing values before backtesting
4. **Use OHLCV when available**: More accurate than single price point
5. **Test with small datasets first**: Verify column mapping before running full backtest

## API vs CSV

| Feature | API (Default) | CSV |
|---------|--------------|-----|
| Data Source | PyPSX API | Local CSV file |
| API Token | Required | Not required |
| Column Mapping | Fixed format | Flexible |
| Multiple Symbols | Separate calls | Single file |
| Date Filtering | API handles | Optional in CSV |
| Real-time Updates | Yes | No (static data) |

Choose CSV when:
- You have your own historical data
- You want to test without API access
- You need custom data formats
- You're working offline

Choose API when:
- You need latest market data
- You want automatic data updates
- You prefer standardized format


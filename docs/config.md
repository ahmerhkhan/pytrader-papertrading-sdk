# Configuration Reference

PyTrader provides configuration options for backtests and paper trading bots to control execution realism and behavior.

---

## Backtest Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_cash` | `float` | `1_000_000.0` | Starting capital for the backtest |
| `position_notional` | `float` | `100_000.0` | Target notional size per trade |
| `min_lot` | `int` | `1` | Minimum lot size when sizing positions |
| `interval` | `str` | `"1d"` | Data interval (e.g., "1d", "1h") |
| `slippage_bps` | `float` | `0.0` | Per-trade slippage in basis points (0.01% per bps) |
| `commission_per_share` | `float` | `0.0` | Flat commission (PKR) per executed share |
| `commission_pct_notional` | `float` | `0.0` | Commission as fraction of trade notional (e.g., 0.0005 = 0.05%) |
| `allow_short` | `bool` | `False` | Allow short selling (selling without existing position) |

### Example

```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")

result = client.backtest(
    strategy="sma",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31",
    initial_cash=1_000_000.0,
    position_notional=100_000.0,
    min_lot=1,
    slippage_bps=5.0,              # 0.05% price impact
    commission_per_share=0.02,     # 2 paisa per share
    commission_pct_notional=0.0005, # 0.05% of notional
    allow_short=False              # Disable short selling (default)
)
```

---

## Paper Trading Bot Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cycle_minutes` | `int` | `15` | Interval between trading cycles (minutes). Supports 15 minutes to 30 days (43200 minutes). Common values: 15 (intraday), 1440 (daily), 10080 (weekly), 43200 (monthly) |
| `position_notional` | `float` | `100_000.0` | Target notional size per trade |
| `min_lot` | `int` | `1` | Minimum lot size when sizing positions |
| `warm_start` | `bool` | `True` | Replay today's data before live trading |
| `max_cycles` | `int` | `None` | Maximum cycles to run (None = unlimited) |
| `slippage_bps` | `float` | `0.0` | Per-trade slippage in basis points |
| `commission_per_share` | `float` | `0.0` | Flat commission (PKR) per executed share |
| `commission_pct_notional` | `float` | `0.0` | Commission as fraction of trade notional |
| `allow_short` | `bool` | `False` | Allow short selling (selling without existing position) |
| `user_id` | `str` | `None` | User identifier for account persistence |
| `notes` | `str` | `None` | Notes for this bot |

### Examples

#### Intraday Trading (15-minute cycles)

```python
from pytrader import PyTrader

client = PyTrader(api_token="your-token")

bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    cycle_minutes=15,              # Trade every 15 minutes during market hours
    position_notional=100_000.0,
    min_lot=1,
    slippage_bps=5.0,
    commission_per_share=0.02,
    commission_pct_notional=0.0005,
    allow_short=False,
    warm_start=True,
    max_cycles=None
)
```

#### Daily Trading (once per day)

```python
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    cycle_minutes=1440,            # Trade once per day (1440 minutes = 24 hours)
    position_notional=100_000.0,
    warm_start=True,
)
```

#### Weekly Trading (once per week)

```python
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    cycle_minutes=10080,            # Trade once per week (10080 minutes = 7 days)
    position_notional=100_000.0,
    warm_start=True,
)
```

#### Monthly Trading (once per month)

```python
bot = client.start_bot(
    strategy="sma",
    symbols=["OGDC"],
    cycle_minutes=43200,            # Trade once per month (~30 days)
    position_notional=100_000.0,
    warm_start=True,
)
```

**Note:** For cycles >= 1 day, the bot automatically:
- Aligns to market open on the target day
- Skips weekends (Saturday/Sunday)
- Waits until the next trading session if the target day falls on a weekend

---

## Slippage and Commissions

### Slippage

Slippage simulates real-world price impact when executing trades. It's expressed in basis points (bps), where 1 bps = 0.01%.

- **Positive slippage** worsens BUY prices and improves SELL prices
- Example: `slippage_bps=5.0` means 0.05% price impact
- For a BUY at 100 PKR with 5 bps slippage: execution price = 100.05 PKR
- For a SELL at 100 PKR with 5 bps slippage: execution price = 99.95 PKR

### Commissions

Commissions can be specified in two ways:

1. **Per Share**: Fixed amount per share/lot
   - Example: `commission_per_share=0.02` = 2 paisa per share

2. **Percentage of Notional**: Percentage of total trade value
   - Example: `commission_pct_notional=0.0005` = 0.05% of trade value

Both can be used together - they are additive.

### Example Calculation

For a trade of 1000 shares at 100 PKR:
- Notional: 100,000 PKR
- Slippage (5 bps): +0.05 PKR per share = +50 PKR total
- Commission per share (0.02): 1000 × 0.02 = 20 PKR
- Commission % (0.05%): 100,000 × 0.0005 = 50 PKR
- **Total cost**: 100,000 + 50 + 20 + 50 = 100,120 PKR

---

## Short Selling

Short selling allows you to sell securities you don't own, profiting when prices decline.

### How It Works

When `allow_short=True`:
- **SELL without position**: Opens a short position (negative quantity)
- **BUY when short**: Covers the short position (reduces negative quantity)
- **PnL calculation**: Profit when price decreases (inverse of long positions)
- **Average cost**: Tracked for short positions separately

### Example

```python
# Enable short selling
result = client.backtest(
    strategy="sma_momentum",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31",
    allow_short=True  # Enable short selling
)

# Paper trading with shorting
bot = client.start_bot(
    strategy="sma_momentum",
    symbols=["OGDC"],
    allow_short=True  # Enable short selling
)
```

### Important Notes

- **Default**: `allow_short=False` (short selling disabled)
- **Position tracking**: Short positions show as negative quantities
- **Covering shorts**: BUY orders when short will cover the position
- **PnL**: Short positions profit when price goes down

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PYTRADER_BACKEND_URL` | Backend server URL | Yes |
| `PYTRADER_API_TOKEN` | Your API token | Yes |

---

## Strategy Configuration

Some built-in strategies accept configuration parameters:

```python
# Strategy with custom parameters
result = client.backtest(
    strategy="dual_sma_momentum",
    symbols=["OGDC"],
    start="2024-01-01",
    end="2024-12-31",
    strategy_config={
        "fast_period": 12,
        "slow_period": 26
    }
)
```

Available strategy configurations depend on the strategy. See the strategy documentation for details.

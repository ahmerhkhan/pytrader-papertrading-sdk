# Advanced Strategy Guide

PyTrader supports creating sophisticated, Alpaca-style strategies with full portfolio access, position tracking, and multi-symbol scanning.

## Overview

The enhanced `Strategy` interface provides access to:
- **Portfolio State**: Cash, equity, unrealized/realized P&L
- **Positions**: Current holdings with cost basis and market values
- **Trade History**: Recent trades for analysis
- **Position Metadata**: Custom tracking (buy dates, hold days, etc.)
- **Current Prices**: Real-time prices for all symbols

## Key Properties

### Portfolio Access

```python
class MyStrategy(Strategy):
    def on_data(self, data):
        portfolio = self.portfolio
        
        cash = portfolio['cash']           # Available cash
        equity = portfolio['equity']      # Total portfolio value
        unrealized_pnl = portfolio['unrealized_pnl']
        realized_pnl = portfolio['realized_pnl']
```

### Positions Access

```python
def on_data(self, data):
    positions = self.positions
    
    for symbol, pos in positions.items():
        qty = pos['qty']                  # Quantity held
        avg_cost = pos['avg_cost']        # Average cost basis
        market_value = pos['market_value'] # Current market value
        current_price = pos['current_price'] # Current price
```

### Position Metadata

```python
def on_data(self, data):
    # Get hold days for a position
    hold_days = self.get_position_hold_days('OGDC')
    
    # Get P&L percentage
    pnl_pct = self.get_position_pnl_pct('OGDC')
    
    # Update custom metadata
    self.update_position_metadata('OGDC', {
        'stop_loss': 245.0,
        'profit_target': 260.0,
    })
```

## Advanced Strategy Example

Here's a complete example similar to the Alpaca strategies:

```python
from pytrader import Strategy
from pytrader.indicators import SMA, RSI, MACD, BollingerBands
import pandas as pd
import json
import os

class AdvancedMomentumStrategy(Strategy):
    def on_start(self):
        # Load learned weights
        self.weights_file = "learned_weights.json"
        self.default_weights = {
            'SMA_cross': 1.0,
            'RSI_bull': 0.8,
            'MACD_bull': 0.8,
            'Volume_spike': 0.7,
            'BB_breakout': 0.6,
        }
        
        if os.path.exists(self.weights_file):
            with open(self.weights_file, 'r') as f:
                self.weights = json.load(f)
        else:
            self.weights = self.default_weights.copy()
        
        # Strategy parameters
        self.stop_loss_pct = 4.0
        self.profit_target_pct = 20.0
        self.min_hold_days = 1
        self.max_hold_days = 7
        self.max_budget = 500_000.0
        self.position_size_pct = 0.1  # 10% of cash per position
    
    def on_data(self, data):
        portfolio = self.portfolio
        positions = self.positions
        current_prices = self.current_prices
        
        # 1. Check stop loss / profit targets for existing positions
        for symbol, pos in positions.items():
            buy_price = pos['avg_cost']
            current_price = current_prices.get(symbol, buy_price)
            pnl_pct = ((current_price - buy_price) / buy_price) * 100
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                print(f"🛑 Stop loss triggered for {symbol}: {pnl_pct:.2f}%")
                self.sell(symbol, pos['qty'])
                continue
            
            # Profit target
            if pnl_pct >= self.profit_target_pct:
                print(f"🎯 Profit target hit for {symbol}: {pnl_pct:.2f}%")
                self.sell(symbol, pos['qty'])
                continue
            
            # Max hold days
            hold_days = self.get_position_hold_days(symbol)
            if hold_days and hold_days >= self.max_hold_days:
                print(f"⏰ Max hold days reached for {symbol}: {hold_days} days")
                self.sell(symbol, pos['qty'])
                continue
        
        # 2. Calculate available budget
        used_budget = sum(pos['market_value'] for pos in positions.values())
        available_budget = min(portfolio['cash'], self.max_budget - used_budget)
        
        if available_budget < 10_000:
            return  # Not enough cash
        
        # 3. Scan and evaluate new opportunities
        candidates = []
        for symbol, df in data.items():
            # Skip if already have position
            if symbol in positions:
                continue
            
            # Need minimum data
            if len(df) < 50:
                continue
            
            # Calculate score
            score = self._calculate_score(df)
            if score > 0.65:
                candidates.append((symbol, score, df))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Buy top candidates
        for symbol, score, df in candidates[:5]:  # Top 5
            current_price = current_prices.get(symbol, df['close'].iloc[-1])
            position_value = available_budget * self.position_size_pct
            qty = int(position_value / current_price)
            
            if qty > 0:
                self.buy(symbol, qty)
                print(f"✅ Buying {symbol}: {qty} shares @ {current_price:.2f} (score: {score:.3f})")
                available_budget -= position_value
    
    def _calculate_score(self, df: pd.DataFrame) -> float:
        """Calculate weighted score for a symbol."""
        conditions = {}
        
        # Calculate indicators
        sma_20 = SMA(df['close'], 20)
        sma_50 = SMA(df['close'], 50)
        rsi = RSI(df['close'], 14)
        macd, signal, _ = MACD(df['close'])
        
        if len(df) < 50:
            return 0.0
        
        # Condition checks
        current_price = df['close'].iloc[-1]
        conditions['SMA_cross'] = int(current_price > sma_20.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1])
        conditions['RSI_bull'] = int(55 < rsi.iloc[-1] < 70)
        conditions['MACD_bull'] = int(macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2])
        
        # Volume spike
        avg_vol = df['volume'].rolling(5).mean().iloc[-1]
        current_vol = df['volume'].iloc[-1]
        conditions['Volume_spike'] = int(current_vol > avg_vol * 1.5)
        
        # Bollinger breakout
        upper, middle, lower = BollingerBands(df['close'], 20, 2.0)
        conditions['BB_breakout'] = int(current_price > upper.iloc[-1])
        
        # Weighted score
        weighted_score = sum(conditions[k] * self.weights.get(k, 0.0) for k in conditions)
        max_score = sum(self.weights.values())
        
        return weighted_score / max_score if max_score > 0 else 0.0
```

## Multi-Symbol Scanning

You can scan multiple symbols dynamically:

```python
class MultiSymbolStrategy(Strategy):
    def on_start(self):
        # Load symbol list from CSV or API
        import pandas as pd
        self.symbol_list = pd.read_csv('stocks.csv')['Symbol'].tolist()
    
    def on_data(self, data):
        # Process all symbols in data dict
        for symbol, df in data.items():
            if self._should_buy(df):
                self.buy(symbol, 100)
```

## Position Tracking

Track custom metadata for positions:

```python
def on_data(self, data):
    for symbol, pos in self.positions.items():
        # Get hold days
        hold_days = self.get_position_hold_days(symbol)
        
        # Get P&L
        pnl_pct = self.get_position_pnl_pct(symbol)
        
        # Update metadata
        self.update_position_metadata(symbol, {
            'last_check': datetime.now().isoformat(),
            'max_pnl': max(self.position_metadata[symbol].get('max_pnl', 0), pnl_pct),
        })
```

## Learning & Adaptation

Implement weight learning similar to Alpaca strategies:

```python
def on_end(self):
    # Analyze performance and update weights
    trades = self.trades
    
    # Calculate win rate per condition
    condition_performance = {}
    for trade in trades:
        # Analyze which conditions were met
        # Update weights based on performance
        pass
    
    # Save updated weights
    with open('learned_weights.json', 'w') as f:
        json.dump(self.weights, f, indent=2)
```

## Best Practices

1. **Always check portfolio state** before placing orders
2. **Use position metadata** to track custom information
3. **Implement stop loss/profit targets** in on_data
4. **Check hold days** to enforce position management rules
5. **Calculate position sizes** based on available cash
6. **Use weighted scoring** for multi-factor strategies
7. **Save/load weights** for adaptive strategies

## Comparison with Alpaca

| Feature | Alpaca | PyTrader |
|---------|--------|----------|
| Portfolio access | `api.get_account()` | `self.portfolio` |
| Positions | `api.list_positions()` | `self.positions` |
| Current prices | `api.get_latest_trade()` | `self.current_prices` |
| Trade history | `api.list_orders()` | `self.trades` |
| Position metadata | Custom CSV | `self.position_metadata` |
| Multi-symbol scan | Manual loop | `on_data(data)` with all symbols |

PyTrader provides a cleaner, more Pythonic interface while maintaining the same power and flexibility as Alpaca strategies.


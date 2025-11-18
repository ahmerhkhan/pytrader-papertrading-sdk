"""
Example: Creating and running a custom trading strategy.

This example demonstrates how to create a custom strategy using the PyTrader SDK
without needing to modify the internal codebase.
"""

from pytrader import Trader, Strategy
from pytrader.indicators import SMA, EMA


class SimpleMovingAverageStrategy(Strategy):
    """
    Simple moving average crossover strategy.
    
    Buys when price crosses above SMA, sells when it crosses below.
    """
    
    def on_start(self):
        """Initialize strategy parameters."""
        self.sma_period = 20
        self.min_bars = self.sma_period + 5
    
    def on_data(self, data):
        """
        Called every cycle with fresh market data.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
        """
        for symbol, df in data.items():
            if len(df) < self.min_bars:
                continue
            
            # Ensure we have a 'close' column
            if 'close' not in df.columns and 'price' in df.columns:
                df = df.copy()
                df['close'] = df['price']
            
            # Calculate SMA
            sma = SMA(df['close'], self.sma_period)
            
            if len(sma) < 2:
                continue
            
            current_price = df['close'].iloc[-1]
            current_sma = sma.iloc[-1]
            prev_price = df['close'].iloc[-2]
            prev_sma = sma.iloc[-2]
            
            # Buy signal: price crosses above SMA
            if prev_price <= prev_sma and current_price > current_sma:
                self.buy(symbol, 100)
            
            # Sell signal: price crosses below SMA
            elif prev_price >= prev_sma and current_price < current_sma:
                self.sell(symbol, 100)


class DualMovingAverageStrategy(Strategy):
    """
    Dual moving average crossover strategy.
    
    Uses fast and slow EMAs. Buys when fast EMA crosses above slow EMA.
    """
    
    def on_start(self):
        """Initialize strategy parameters."""
        self.fast_period = 12
        self.slow_period = 26
        self.min_bars = self.slow_period + 5
    
    def on_data(self, data):
        """Generate trading signals based on EMA crossover."""
        for symbol, df in data.items():
            if len(df) < self.min_bars:
                continue
            
            # Ensure we have a 'close' column
            if 'close' not in df.columns and 'price' in df.columns:
                df = df.copy()
                df['close'] = df['price']
            
            # Calculate EMAs
            fast_ema = EMA(df['close'], self.fast_period)
            slow_ema = EMA(df['close'], self.slow_period)
            
            if len(fast_ema) < 2 or len(slow_ema) < 2:
                continue
            
            # Check for crossover
            current_fast = fast_ema.iloc[-1]
            current_slow = slow_ema.iloc[-1]
            prev_fast = fast_ema.iloc[-2]
            prev_slow = slow_ema.iloc[-2]
            
            # Buy signal: fast EMA crosses above slow EMA
            if prev_fast <= prev_slow and current_fast > current_slow:
                self.buy(symbol, 100)
            
            # Sell signal: fast EMA crosses below slow EMA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                self.sell(symbol, 100)


def main():
    """Run example strategies."""
    print("=" * 80)
    print("PyTrader Custom Strategy Example")
    print("=" * 80)
    
    # Example 1: Simple SMA Strategy Backtest
    print("\n1. Running Simple SMA Strategy Backtest...")
    trader1 = Trader(
        strategy=SimpleMovingAverageStrategy,
        symbols=['OGDC', 'HBL'],
        initial_cash=1_000_000.0,
    )
    
    result = trader1.run_backtest(start='2024-01-01', end='2024-01-31')
    
    if result.get('metrics'):
        metrics = result['metrics']
        print(f"   Total Return: {metrics.total_return_pct:.2f}%")
        sharpe_ratio = getattr(metrics, 'sharpe_ratio', None)
        sharpe_available = getattr(metrics, 'sharpe_ratio_available', sharpe_ratio is not None)
        sharpe_str = f"{sharpe_ratio:.2f}" if sharpe_available and sharpe_ratio is not None else "-"
        print(f"   Sharpe Ratio: {sharpe_str}")
        print(f"   Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"   Trades: {len(result.get('trades', []))}")
    
    # Example 2: Dual EMA Strategy (would run paper trading with token)
    print("\n2. Dual EMA Strategy ready for paper trading")
    print("   (Requires API token for live paper trading)")
    print("   trader2 = Trader(strategy=DualMovingAverageStrategy, symbols=['OGDC'])")
    print("   await trader2.start_paper_trading(token='your-token')")


if __name__ == "__main__":
    main()


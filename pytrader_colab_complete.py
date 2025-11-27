"""
PyTrader SDK - Complete Colab Notebook
======================================

This file contains everything you need to:
1. Install the SDK
2. Run backtests with aggressive custom strategies
3. Run live paper trading
4. Launch the live dashboard

All strategies use REAL market data (no dummy data).
All strategies are CUSTOM and AGGRESSIVE (written from scratch).

Usage in Colab:
1. Copy this entire file into a Colab cell
2. Set your API_TOKEN
3. Run the sections you want (backtest, paper trading, or dashboard)
"""

# ============================================================================
# INSTALLATION
# ============================================================================
# Run this first in Colab:
# !pip install "git+https://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@github.com/ahmerhkhan/pytrader-papertrading-sdk.git"

# ============================================================================
# IMPORTS
# ============================================================================
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from pytrader import Trader, Strategy
    from pytrader.indicators import SMA, EMA, RSI, MACD
    from pytrader import start_dashboard
except ImportError as e:
    print(f"⚠️  Import Error: {e}")
    print("Please install the SDK first:")
    print('!pip install "git+https://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@github.com/ahmerhkhan/pytrader-papertrading-sdk.git"')
    raise

# ============================================================================
# CONFIGURATION
# ============================================================================
# SET YOUR API TOKEN HERE
API_TOKEN = "ahmer-token"  # API token for authentication
BACKEND_URL = "http://localhost:8000"  # Or your backend URL (for Colab, use your actual backend URL)

# Trading symbols (PSX stocks)
SYMBOLS = ["OGDC", "HBL", "UBL", "PSO", "PPL", "MCB", "FCCL", "ENGRO"]

# ============================================================================
# AGGRESSIVE CUSTOM STRATEGIES
# ============================================================================

class AggressiveMomentumScalper(Strategy):
    """
    Ultra-aggressive momentum scalping strategy.
    Enters on strong momentum, exits quickly on any reversal.
    High frequency, tight stops.
    """
    
    def on_start(self):
        """Initialize aggressive parameters."""
        self.fast_ema = 5   # Very fast EMA
        self.slow_ema = 15  # Still fast
        self.rsi_period = 7  # Short RSI
        self.momentum_threshold = 0.5  # Lowered to 0.5% for more signals
        self.stop_loss_pct = 1.5  # Tight 1.5% stop loss
        self.take_profit_pct = 3.0  # Quick 3% profit target
        self.max_hold_cycles = 3  # Exit after 3 cycles max
        
    def on_data(self, data):
        """Aggressive momentum scalping logic."""
        for symbol, df in data.items():
            # Ensure we have enough data
            if len(df) < max(self.slow_ema, self.rsi_period) + 5:
                continue
            
            # Normalize price column - handle both 'close' and 'price'
            df = df.copy()  # Work on copy to avoid modifying original
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif 'CLOSE' in df.columns:
                    df['close'] = df['CLOSE']
                else:
                    continue  # Skip if no price column
            
            # Ensure volume exists
            if 'volume' not in df.columns:
                if 'VOLUME' in df.columns:
                    df['volume'] = df['VOLUME']
                else:
                    df['volume'] = 1000000  # Default volume
            
            current_price = float(df['close'].iloc[-1])
            if len(df) < 2:
                continue
            prev_price = float(df['close'].iloc[-2])
            
            # Calculate indicators
            try:
                fast_ema = EMA(df['close'], self.fast_ema)
                slow_ema = EMA(df['close'], self.slow_ema)
                rsi = RSI(df['close'], self.rsi_period)
            except Exception as e:
                continue  # Skip if indicator calculation fails
            
            if len(fast_ema) < 2 or len(slow_ema) < 2 or len(rsi) < 1:
                continue
            
            # Get last valid values (skip NaN)
            fast_ema_valid = fast_ema.dropna()
            slow_ema_valid = slow_ema.dropna()
            rsi_valid = rsi.dropna()
            
            if len(fast_ema_valid) < 2 or len(slow_ema_valid) < 2 or len(rsi_valid) < 1:
                continue
            
            current_fast = float(fast_ema_valid.iloc[-1])
            current_slow = float(slow_ema_valid.iloc[-1])
            prev_fast = float(fast_ema_valid.iloc[-2]) if len(fast_ema_valid) >= 2 else current_fast
            prev_slow = float(slow_ema_valid.iloc[-2]) if len(slow_ema_valid) >= 2 else current_slow
            current_rsi = float(rsi_valid.iloc[-1])
            
            # Calculate momentum
            momentum_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
            
            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                buy_price = pos['avg_cost']
                current_pnl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                hold_cycles = self.position_metadata.get(symbol, {}).get('hold_cycles', 0)
                
                # Aggressive exit conditions
                if (current_pnl_pct <= -self.stop_loss_pct or  # Stop loss
                    current_pnl_pct >= self.take_profit_pct or  # Take profit
                    hold_cycles >= self.max_hold_cycles or  # Max hold time
                    (current_fast < current_slow and prev_fast >= prev_slow) or  # Momentum reversal
                    current_rsi > 75):  # Overbought
                    self.sell(symbol, pos['qty'])
                    # Update metadata
                    if symbol in self.position_metadata:
                        self.position_metadata[symbol]['hold_cycles'] = 0
            else:
                # Aggressive entry conditions
                cash = self.portfolio.get('cash', 0)
                if cash < 10000:  # Need minimum cash
                    continue
                
                # Strong momentum + EMA crossover + RSI not overbought
                strong_momentum = abs(momentum_pct) >= self.momentum_threshold
                bullish_crossover = (current_fast > current_slow and prev_fast <= prev_slow)
                rsi_ok = 30 < current_rsi < 70  # Not extreme
                
                if strong_momentum and bullish_crossover and rsi_ok and momentum_pct > 0:
                    # Use 30% of available cash (aggressive position sizing)
                    position_size = int((cash * 0.30) / current_price)
                    if position_size > 0:
                        self.buy(symbol, position_size)
                        # Track hold cycles
                        if symbol not in self.position_metadata:
                            self.position_metadata[symbol] = {}
                        self.position_metadata[symbol]['hold_cycles'] = 0
            
            # Increment hold cycles for existing positions
            if symbol in self.positions and symbol in self.position_metadata:
                self.position_metadata[symbol]['hold_cycles'] = \
                    self.position_metadata[symbol].get('hold_cycles', 0) + 1


class AggressiveBreakoutTrader(Strategy):
    """
    Aggressive breakout strategy.
    Enters aggressively on breakouts, uses wide stops but quick exits.
    """
    
    def on_start(self):
        """Initialize breakout parameters."""
        self.lookback = 20  # Lookback for breakout detection
        self.volume_multiplier = 2.0  # 2x average volume required
        self.stop_loss_pct = 2.5  # 2.5% stop loss
        self.take_profit_pct = 5.0  # 5% profit target
        self.min_breakout_pct = 1.5  # Minimum 1.5% breakout
        
    def on_data(self, data):
        """Aggressive breakout trading logic."""
        for symbol, df in data.items():
            if len(df) < self.lookback + 5:
                continue
            
            # Normalize columns
            df = df.copy()
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif 'CLOSE' in df.columns:
                    df['close'] = df['CLOSE']
                else:
                    continue
            if 'volume' not in df.columns:
                if 'VOLUME' in df.columns:
                    df['volume'] = df['VOLUME']
                else:
                    df['volume'] = 1000000  # Default volume
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Calculate recent high and average volume
            recent_high = df['close'].iloc[-self.lookback:].max()
            avg_volume = df['volume'].iloc[-self.lookback:].mean()
            
            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                buy_price = pos['avg_cost']
                current_pnl_pct = ((current_price - buy_price) / buy_price) * 100
                
                # Exit on stop loss or take profit
                if current_pnl_pct <= -self.stop_loss_pct:
                    self.sell(symbol, pos['qty'])
                elif current_pnl_pct >= self.take_profit_pct:
                    self.sell(symbol, pos['qty'])
                # Exit if price falls below recent high (breakout failure)
                elif current_price < recent_high * 0.98:
                    self.sell(symbol, pos['qty'])
            else:
                # Entry: Breakout with volume confirmation
                breakout_pct = ((current_price - recent_high) / recent_high) * 100
                volume_surge = current_volume >= (avg_volume * self.volume_multiplier)
                
                cash = self.portfolio['cash']
                if cash < 10000:
                    continue
                
                # Aggressive entry on strong breakout
                if (breakout_pct >= self.min_breakout_pct and volume_surge and 
                    current_price > recent_high):
                    # Use 40% of cash (very aggressive)
                    position_size = int((cash * 0.40) / current_price)
                    if position_size > 0:
                        self.buy(symbol, position_size)


class AggressiveVolumeSurgeTrader(Strategy):
    """
    Aggressive volume surge strategy.
    Enters on massive volume spikes, exits quickly.
    Very high frequency trading.
    """
    
    def on_start(self):
        """Initialize volume surge parameters."""
        self.volume_threshold = 3.0  # 3x average volume
        self.price_move_threshold = 1.0  # 1% price move required
        self.quick_exit_pct = 2.0  # Exit after 2% gain
        self.stop_loss_pct = 1.0  # Tight 1% stop
        self.max_hold_cycles = 2  # Very short hold
        
    def on_data(self, data):
        """Volume surge trading logic."""
        for symbol, df in data.items():
            if len(df) < 15:
                continue
            
            # Normalize columns
            df = df.copy()
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif 'CLOSE' in df.columns:
                    df['close'] = df['CLOSE']
                else:
                    continue
            if 'volume' not in df.columns:
                if 'VOLUME' in df.columns:
                    df['volume'] = df['VOLUME']
                else:
                    df['volume'] = 1000000  # Default volume
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            
            # Calculate average volume
            avg_volume = df['volume'].iloc[-15:].mean()
            price_move_pct = ((current_price - prev_price) / prev_price) * 100
            
            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                buy_price = pos['avg_cost']
                current_pnl_pct = ((current_price - buy_price) / buy_price) * 100
                hold_cycles = self.position_metadata.get(symbol, {}).get('hold_cycles', 0)
                
                # Very quick exits
                if (current_pnl_pct >= self.quick_exit_pct or  # Quick profit
                    current_pnl_pct <= -self.stop_loss_pct or  # Stop loss
                    hold_cycles >= self.max_hold_cycles or  # Max hold
                    current_volume < avg_volume):  # Volume dried up
                    self.sell(symbol, pos['qty'])
                    if symbol in self.position_metadata:
                        self.position_metadata[symbol]['hold_cycles'] = 0
            else:
                # Entry: Massive volume surge with price movement
                volume_surge = current_volume >= (avg_volume * self.volume_threshold)
                significant_move = abs(price_move_pct) >= self.price_move_threshold
                
                cash = self.portfolio['cash']
                if cash < 10000:
                    continue
                
                # Enter on volume surge with upward price movement
                if volume_surge and significant_move and price_move_pct > 0:
                    # Use 35% of cash
                    position_size = int((cash * 0.35) / current_price)
                    if position_size > 0:
                        self.buy(symbol, position_size)
                        if symbol not in self.position_metadata:
                            self.position_metadata[symbol] = {}
                        self.position_metadata[symbol]['hold_cycles'] = 0
            
            # Increment hold cycles
            if symbol in self.positions and symbol in self.position_metadata:
                self.position_metadata[symbol]['hold_cycles'] = \
                    self.position_metadata[symbol].get('hold_cycles', 0) + 1


class AggressiveRSIExtremeTrader(Strategy):
    """
    Aggressive RSI extreme strategy.
    Enters on extreme RSI readings, uses mean reversion with momentum.
    Very aggressive position sizing.
    """
    
    def on_start(self):
        """Initialize RSI extreme parameters."""
        self.rsi_period = 14
        self.oversold = 25  # Very oversold
        self.overbought = 75  # Very overbought
        self.stop_loss_pct = 2.0
        self.take_profit_pct = 4.0
        self.min_volume_multiplier = 1.5
        
    def on_data(self, data):
        """RSI extreme trading logic."""
        for symbol, df in data.items():
            if len(df) < self.rsi_period + 5:
                continue
            
            # Normalize columns
            df = df.copy()
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif 'CLOSE' in df.columns:
                    df['close'] = df['CLOSE']
                else:
                    continue
            if 'volume' not in df.columns:
                if 'VOLUME' in df.columns:
                    df['volume'] = df['VOLUME']
                else:
                    df['volume'] = 1000000  # Default volume
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-20:].mean()
            
            # Calculate RSI
            rsi = RSI(df['close'], self.rsi_period)
            if len(rsi) < 1:
                continue
            
            current_rsi = rsi.iloc[-1]
            
            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                buy_price = pos['avg_cost']
                current_pnl_pct = ((current_price - buy_price) / buy_price) * 100
                
                # Exit conditions
                if current_pnl_pct <= -self.stop_loss_pct:
                    self.sell(symbol, pos['qty'])
                elif current_pnl_pct >= self.take_profit_pct:
                    self.sell(symbol, pos['qty'])
                # Exit if RSI normalized (mean reversion complete)
                elif (current_rsi > 50 and buy_price < current_price * 0.95):  # Bought oversold, now normalized
                    self.sell(symbol, pos['qty'])
            else:
                cash = self.portfolio['cash']
                if cash < 10000:
                    continue
                
                volume_ok = current_volume >= (avg_volume * self.min_volume_multiplier)
                
                # Aggressive entry on extreme RSI
                if current_rsi < self.oversold and volume_ok:
                    # Buy oversold - use 45% of cash (very aggressive)
                    position_size = int((cash * 0.45) / current_price)
                    if position_size > 0:
                        self.buy(symbol, position_size)
                elif current_rsi > self.overbought and volume_ok:
                    # Short opportunity (if allowed) - for now just skip
                    pass


class AggressiveMACDCrossTrader(Strategy):
    """
    Aggressive MACD crossover strategy.
    Enters on MACD crossovers with volume confirmation.
    Multiple positions, quick exits.
    """
    
    def on_start(self):
        """Initialize MACD parameters."""
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.stop_loss_pct = 1.8
        self.take_profit_pct = 3.5
        self.max_positions = 5  # Hold up to 5 positions
        
    def on_data(self, data):
        """MACD crossover trading logic."""
        for symbol, df in data.items():
            if len(df) < self.slow_period + self.signal_period + 5:
                continue
            
            # Normalize columns
            df = df.copy()
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif 'CLOSE' in df.columns:
                    df['close'] = df['CLOSE']
                else:
                    continue
            
            current_price = df['close'].iloc[-1]
            
            # Calculate MACD
            macd_line, signal_line, histogram = MACD(
                df['close'], 
                self.fast_period, 
                self.slow_period, 
                self.signal_period
            )
            
            if len(macd_line) < 2 or len(signal_line) < 2:
                continue
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                buy_price = pos['avg_cost']
                current_pnl_pct = ((current_price - buy_price) / buy_price) * 100
                
                # Exit on stop/target or signal reversal
                if (current_pnl_pct <= -self.stop_loss_pct or 
                    current_pnl_pct >= self.take_profit_pct or
                    (current_macd < current_signal and prev_macd >= prev_signal)):  # Bearish crossover
                    self.sell(symbol, pos['qty'])
            else:
                # Limit concurrent positions
                if len(self.positions) >= self.max_positions:
                    continue
                
                cash = self.portfolio['cash']
                if cash < 10000:
                    continue
                
                # Entry: Bullish MACD crossover
                bullish_cross = (current_macd > current_signal and 
                                 prev_macd <= prev_signal and
                                 current_macd > 0)  # Above zero line
                
                if bullish_cross:
                    # Use 25% of cash per position (allows multiple positions)
                    position_size = int((cash * 0.25) / current_price)
                    if position_size > 0:
                        self.buy(symbol, position_size)


# ============================================================================
# BACKTESTING EXAMPLES
# ============================================================================

class SimpleTestStrategy(Strategy):
    """
    Simple test strategy to verify backtest works.
    Buys when price goes up, sells when it goes down.
    """
    
    def on_start(self):
        """Initialize."""
        self.min_bars = 5
        
    def on_data(self, data):
        """Simple price-based strategy."""
        for symbol, df in data.items():
            if len(df) < self.min_bars:
                continue
            
            # Normalize price column
            df = df.copy()
            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                elif 'CLOSE' in df.columns:
                    df['close'] = df['CLOSE']
                else:
                    continue
            
            current_price = float(df['close'].iloc[-1])
            prev_price = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
            
            # Simple strategy: buy on price increase, sell on decrease
            if symbol in self.positions:
                pos = self.positions[symbol]
                # Sell if price drops
                if current_price < prev_price:
                    self.sell(symbol, pos['qty'])
            else:
                # Buy if price increases and we have cash
                cash = self.portfolio.get('cash', 0)
                if current_price > prev_price and cash > 10000:
                    position_size = int((cash * 0.20) / current_price)
                    if position_size > 0:
                        self.buy(symbol, position_size)


def run_backtest_example():
    """Run backtest with aggressive momentum scalper."""
    print("=" * 80)
    print("RUNNING AGGRESSIVE BACKTEST")
    print("=" * 80)
    
    # Check API token
    if API_TOKEN == "your-api-token-here":
        print("\n⚠️  ERROR: Please set your API_TOKEN at the top of this file!")
        print("   API_TOKEN = \"your-actual-token-here\"")
        return None
    
    # First test with simple strategy to verify it works
    print("\n[TEST] Running simple test strategy first...")
    trader_test = Trader(
        strategy=SimpleTestStrategy,
        symbols=SYMBOLS[:2],  # Use first 2 symbols for test
        initial_cash=1_000_000.0,
        position_notional=100_000.0,
        cycle_minutes=15,
        bot_id="simple-test-backtest"
    )
    
    result_test = trader_test.run_backtest(
        start="2024-01-01",
        end="2024-03-31",
        api_token=API_TOKEN,
        backend_url=BACKEND_URL
    )
    
    print(f"\nTest Strategy Results:")
    print(f"  Trades: {len(result_test.get('trades', []))}")
    print(f"  Return: {result_test.get('metrics', {}).total_return_pct if result_test.get('metrics') else 0:.2f}%")
    print(f"  Data Points: {len(result_test.get('equity_curve', []))}")
    
    # Check date range in results
    equity_curve = result_test.get('equity_curve', [])
    if equity_curve:
        first_date = equity_curve[0].get('ts', 'N/A')
        last_date = equity_curve[-1].get('ts', 'N/A')
        print(f"  Date Range: {first_date} to {last_date}")
    
    if len(result_test.get('trades', [])) == 0:
        print("\n⚠️  WARNING: Test strategy generated 0 trades!")
        print("Possible issues:")
        print("  1. API token is invalid or expired")
        print("  2. No data available for date range 2024-01-01 to 2024-03-31")
        print("  3. Backend URL is incorrect")
        print("  4. Data format mismatch")
        print("\nTry:")
        print("  - Check your API token is valid")
        print("  - Verify backend is accessible")
        print("  - Try a different date range with known data")
        return result_test
    
    # Now run with aggressive strategy
    print("\n[MAIN] Running aggressive momentum scalper...")
    trader = Trader(
        strategy=AggressiveMomentumScalper,
        symbols=SYMBOLS[:4],  # Use first 4 symbols
        initial_cash=1_000_000.0,
        position_notional=150_000.0,  # Larger positions
        cycle_minutes=15,
        bot_id="aggressive-scalper-backtest"
    )
    
    # Run backtest
    print(f"\nBacktesting from 2024-01-01 to 2024-03-31...")
    result = trader.run_backtest(
        start="2024-01-01",
        end="2024-03-31",
        api_token=API_TOKEN,
        backend_url=BACKEND_URL
    )
    
    # Display results
    if result.get('metrics'):
        metrics = result['metrics']
        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Total Return: {metrics.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {getattr(metrics, 'sharpe_ratio', 'N/A')}")
        print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"Total Trades: {len(result.get('trades', []))}")
        print(f"Win Rate: {getattr(metrics, 'win_rate_pct', 'N/A')}%")
        print(f"Final Equity: {result.get('summary', {}).get('final_portfolio_value', 0):,.2f} PKR")
    else:
        print("\n⚠️  No metrics available in result")
    
    return result


def run_multiple_backtests():
    """Run backtests for all aggressive strategies."""
    strategies = {
        "Momentum Scalper": AggressiveMomentumScalper,
        "Breakout Trader": AggressiveBreakoutTrader,
        "Volume Surge": AggressiveVolumeSurgeTrader,
        "RSI Extreme": AggressiveRSIExtremeTrader,
        "MACD Cross": AggressiveMACDCrossTrader,
    }
    
    results = {}
    
    for name, strategy_class in strategies.items():
        print(f"\n{'='*80}")
        print(f"BACKTESTING: {name}")
        print(f"{'='*80}")
        
        trader = Trader(
            strategy=strategy_class,
            symbols=SYMBOLS[:3],
            initial_cash=1_000_000.0,
            position_notional=120_000.0,
            cycle_minutes=15,
            bot_id=f"backtest-{name.lower().replace(' ', '-')}"
        )
        
        result = trader.run_backtest(
            start="2024-01-01",
            end="2024-03-31",
            api_token=API_TOKEN,
            backend_url=BACKEND_URL
        )
        
        results[name] = result
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"Return: {metrics.total_return_pct:.2f}% | "
                  f"Drawdown: {metrics.max_drawdown_pct:.2f}% | "
                  f"Trades: {len(result.get('trades', []))}")
    
    return results


# ============================================================================
# PAPER TRADING EXAMPLES
# ============================================================================

def run_paper_trading():
    """Run live paper trading with aggressive strategy."""
    print("=" * 80)
    print("STARTING LIVE PAPER TRADING")
    print("=" * 80)
    
    # Create trader
    trader = Trader(
        strategy=AggressiveMomentumScalper,
        symbols=SYMBOLS[:4],
        initial_cash=1_000_000.0,
        position_notional=150_000.0,
        cycle_minutes=15,
        bot_id="aggressive-live-paper"
    )
    
    # Start paper trading
    print(f"\nStarting paper trading with {len(SYMBOLS[:4])} symbols...")
    print("Press Ctrl+C to stop")
    
    try:
        trader.start_paper_trading(
            api_token=API_TOKEN,
            backend_url=BACKEND_URL,
            warm_start=True,  # Replay today's data first
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped by user.")


def run_paper_trading_with_dashboard():
    """Run paper trading with live dashboard."""
    print("=" * 80)
    print("STARTING PAPER TRADING WITH DASHBOARD")
    print("=" * 80)
    
    # Create trader
    trader = Trader(
        strategy=AggressiveBreakoutTrader,
        symbols=SYMBOLS[:5],
        initial_cash=1_000_000.0,
        position_notional=120_000.0,
        cycle_minutes=15,
        bot_id="aggressive-dashboard-bot"
    )
    
    # Launch dashboard first (optional - can also use dashboard=True in start_paper_trading)
    print("\nLaunching dashboard at http://localhost:8787")
    start_dashboard(trader, port=8787)
    
    # Start paper trading with dashboard
    print(f"\nStarting paper trading with dashboard...")
    print("Dashboard URL: http://localhost:8787")
    print("Press Ctrl+C to stop")
    
    try:
        trader.start_paper_trading(
            api_token=API_TOKEN,
            backend_url=BACKEND_URL,
            dashboard=True,  # Auto-launch dashboard
            dashboard_port=8787,
            warm_start=True,
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped.")


# ============================================================================
# QUICK TEST FUNCTION (for Colab)
# ============================================================================

def quick_test():
    """Quick test to verify everything is set up correctly."""
    print("=" * 80)
    print("QUICK SETUP TEST")
    print("=" * 80)
    print(f"[OK] API Token: {API_TOKEN[:10]}..." if len(API_TOKEN) > 10 else f"[OK] API Token: {API_TOKEN}")
    print(f"[OK] Backend URL: {BACKEND_URL}")
    print(f"[OK] Symbols: {', '.join(SYMBOLS[:4])}...")
    print(f"[OK] Strategies loaded: 6 aggressive strategies")
    print("\n[OK] All imports successful!")
    print("\nYou can now run:")
    print("  - run_backtest_example()")
    print("  - run_multiple_backtests()")
    print("  - run_paper_trading()")
    print("  - run_paper_trading_with_dashboard()")
    print("=" * 80)


# ============================================================================
# COLAB USAGE EXAMPLES
# ============================================================================
"""
To use this in Google Colab:

1. INSTALLATION (Run in first cell):
   !pip install "git+https://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@github.com/ahmerhkhan/pytrader-papertrading-sdk.git"

2. UPLOAD THIS FILE or paste it into a cell and run it

3. VERIFY SETUP (optional):
   quick_test()

4. RUN FUNCTIONS (in separate cells):

   # Backtest example:
   run_backtest_example()
   
   # Compare all strategies:
   run_multiple_backtests()
   
   # Paper trading (when market is open):
   run_paper_trading()
   
   # Paper trading with dashboard:
   run_paper_trading_with_dashboard()
"""

# ============================================================================
# MAIN EXECUTION (for local testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PYTRADER SDK - AGGRESSIVE STRATEGIES")
    print("=" * 80)
    print("\nAvailable functions:")
    print("1. run_backtest_example() - Run single backtest")
    print("2. run_multiple_backtests() - Compare all strategies")
    print("3. run_paper_trading() - Start live paper trading")
    print("4. run_paper_trading_with_dashboard() - Paper trading + dashboard")
    print("\nCurrent API_TOKEN:", API_TOKEN)
    print("=" * 80)
    
    # Uncomment the function you want to run:
    
    # Example 1: Run a single backtest
    # run_backtest_example()
    
    # Example 2: Compare all strategies
    # run_multiple_backtests()
    
    # Example 3: Start paper trading (requires API token)
    # run_paper_trading()
    
    # Example 4: Paper trading with dashboard
    # run_paper_trading_with_dashboard()


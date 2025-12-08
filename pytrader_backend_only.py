"""
PyTrader SDK - Backend-Only Paper Trading Bot
==============================================

This bot runs paper trading WITHOUT a local dashboard.
All data is automatically sent to the backend and monitored via your Vercel web app.

✅ No local dashboard
✅ Backend URL automatically configured
✅ Only need API token + bot ID
✅ Monitor via web app: https://your-vercel-app.com

Usage:
    1. Set your API_TOKEN below
    2. Set your BOT_ID (unique name for your bot)
    3. Run: python pytrader_backend_only.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
from datetime import datetime
from pytrader import Trader, Strategy
from pytrader.indicators import SMA, EMA, RSI, MACD

# ============================================================================
# CONFIGURATION - Only 2 things needed!
# ============================================================================
# ⚠️ REQUIRED: Set your API token here
API_TOKEN = "dev-token"  # ✅ Change this to your API token

# ⚠️ REQUIRED: Set your bot name/ID
BOT_ID = "my-trading-bot"  # ✅ Change this to your unique bot name

# Trading Configuration (Optional - adjust as needed)
SYMBOLS = ["OGDC", "HBL", "UBL", "PSO", "PPL", "MCB", "FCCL", "ENGRO"]
INITIAL_CASH = 1_000_000.0
POSITION_NOTIONAL = 150_000.0
CYCLE_MINUTES = 15

# ============================================================================
# AGGRESSIVE CUSTOM STRATEGY
# ============================================================================

class AggressiveMultiSignalStrategy(Strategy):
    """
    Advanced multi-signal aggressive strategy combining:
    - Momentum (EMA crossovers)
    - RSI extremes
    - Volume surges
    - Breakout detection
    
    This strategy is designed for high-frequency paper trading with:
    - Multiple entry conditions
    - Tight risk management
    - Quick profit taking
    - Adaptive position sizing
    """

    def on_start(self):
        """Initialize strategy parameters."""
        # EMA Parameters
        self.fast_ema = 8
        self.slow_ema = 21
        
        # RSI Parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Volume Parameters
        self.volume_lookback = 20
        self.volume_surge_multiplier = 2.0
        
        # Risk Management
        self.stop_loss_pct = 2.0  # 2% stop loss
        self.take_profit_pct = 4.0  # 4% take profit
        self.trailing_stop_pct = 1.5  # 1.5% trailing stop
        
        # Position Management
        self.max_positions = 4
        self.position_size_pct = 0.25  # 25% per position
        self.max_hold_cycles = 5
        
        # Momentum thresholds
        self.min_momentum_pct = 0.8
        
        print(f"[STRATEGY] AggressiveMultiSignalStrategy initialized")
        print(f"  - Fast EMA: {self.fast_ema}, Slow EMA: {self.slow_ema}")
        print(f"  - RSI: {self.rsi_period} | Oversold: {self.rsi_oversold} | Overbought: {self.rsi_overbought}")
        print(f"  - Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
        print(f"  - Max Positions: {self.max_positions} | Position Size: {self.position_size_pct*100}%")

    def on_data(self, data):
        """Execute trading logic on new data."""
        for symbol, df in data.items():
            # Validate data
            if len(df) < max(self.slow_ema, self.rsi_period, self.volume_lookback) + 5:
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

            # Current data
            current_price = float(df['close'].iloc[-1])
            current_volume = float(df['volume'].iloc[-1])
            prev_price = float(df['close'].iloc[-2]) if len(df) > 1 else current_price

            # Calculate indicators
            try:
                fast_ema = EMA(df['close'], self.fast_ema)
                slow_ema = EMA(df['close'], self.slow_ema)
                rsi = RSI(df['close'], self.rsi_period)
            except Exception as e:
                continue

            # Validate indicators
            if len(fast_ema) < 2 or len(slow_ema) < 2 or len(rsi) < 1:
                continue

            fast_ema_valid = fast_ema.dropna()
            slow_ema_valid = slow_ema.dropna()
            rsi_valid = rsi.dropna()

            if len(fast_ema_valid) < 2 or len(slow_ema_valid) < 2 or len(rsi_valid) < 1:
                continue

            # Get indicator values
            current_fast = float(fast_ema_valid.iloc[-1])
            current_slow = float(slow_ema_valid.iloc[-1])
            prev_fast = float(fast_ema_valid.iloc[-2])
            prev_slow = float(slow_ema_valid.iloc[-2])
            current_rsi = float(rsi_valid.iloc[-1])

            # Calculate momentum and volume metrics
            momentum_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
            avg_volume = float(df['volume'].iloc[-self.volume_lookback:].mean())
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Calculate breakout level
            recent_high = float(df['close'].iloc[-20:].max())
            breakout_pct = ((current_price - recent_high) / recent_high) * 100 if recent_high > 0 else 0

            # ====================================================================
            # POSITION MANAGEMENT (Exit Logic)
            # ====================================================================
            if symbol in self.positions:
                pos = self.positions[symbol]
                buy_price = pos['avg_cost']
                current_pnl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                hold_cycles = self.position_metadata.get(symbol, {}).get('hold_cycles', 0)
                peak_price = self.position_metadata.get(symbol, {}).get('peak_price', buy_price)

                # Update peak price for trailing stop
                if current_price > peak_price:
                    if symbol not in self.position_metadata:
                        self.position_metadata[symbol] = {}
                    self.position_metadata[symbol]['peak_price'] = current_price
                    peak_price = current_price

                # Calculate trailing stop
                trailing_stop_price = peak_price * (1 - self.trailing_stop_pct / 100)

                # EXIT CONDITIONS
                exit_signal = False
                exit_reason = ""

                # 1. Stop Loss
                if current_pnl_pct <= -self.stop_loss_pct:
                    exit_signal = True
                    exit_reason = f"STOP LOSS ({current_pnl_pct:.2f}%)"

                # 2. Take Profit
                elif current_pnl_pct >= self.take_profit_pct:
                    exit_signal = True
                    exit_reason = f"TAKE PROFIT ({current_pnl_pct:.2f}%)"

                # 3. Trailing Stop
                elif current_price < trailing_stop_price:
                    exit_signal = True
                    exit_reason = f"TRAILING STOP (price {current_price:.2f} < stop {trailing_stop_price:.2f})"

                # 4. Max Hold Time
                elif hold_cycles >= self.max_hold_cycles:
                    exit_signal = True
                    exit_reason = f"MAX HOLD TIME ({hold_cycles} cycles)"

                # 5. Momentum Reversal
                elif current_fast < current_slow and prev_fast >= prev_slow:
                    exit_signal = True
                    exit_reason = "MOMENTUM REVERSAL"

                # 6. RSI Overbought
                elif current_rsi > self.rsi_overbought + 5:  # Extra buffer
                    exit_signal = True
                    exit_reason = f"RSI OVERBOUGHT ({current_rsi:.1f})"

                # Execute exit
                if exit_signal:
                    self.sell(symbol, pos['qty'])
                    print(f"[EXIT] {symbol}: {exit_reason} | PnL: {current_pnl_pct:.2f}%")
                    if symbol in self.position_metadata:
                        self.position_metadata[symbol]['hold_cycles'] = 0
                        self.position_metadata[symbol]['peak_price'] = 0

            # ====================================================================
            # ENTRY LOGIC (Multiple Signal Confirmation)
            # ====================================================================
            else:
                # Check if we can open new positions
                if len(self.positions) >= self.max_positions:
                    continue

                cash = self.portfolio.get('cash', 0)
                if cash < 10000:
                    continue

                # SIGNAL CALCULATION
                signals = []
                signal_strength = 0

                # Signal 1: EMA Bullish Crossover
                if current_fast > current_slow and prev_fast <= prev_slow:
                    signals.append("EMA_CROSS")
                    signal_strength += 2

                # Signal 2: Strong Momentum
                if momentum_pct >= self.min_momentum_pct:
                    signals.append(f"MOMENTUM_{momentum_pct:.1f}%")
                    signal_strength += 1

                # Signal 3: RSI Oversold
                if current_rsi < self.rsi_oversold:
                    signals.append(f"RSI_OVERSOLD_{current_rsi:.1f}")
                    signal_strength += 2

                # Signal 4: Volume Surge
                if volume_ratio >= self.volume_surge_multiplier:
                    signals.append(f"VOLUME_SURGE_{volume_ratio:.1f}x")
                    signal_strength += 1

                # Signal 5: Breakout
                if breakout_pct >= 1.0:  # 1% above recent high
                    signals.append(f"BREAKOUT_{breakout_pct:.1f}%")
                    signal_strength += 1

                # Signal 6: EMA Alignment (both above)
                if current_fast > current_slow and current_price > current_fast:
                    signals.append("EMA_ALIGNED")
                    signal_strength += 1

                # ENTRY DECISION (need at least 2 signals with strength >= 3)
                if len(signals) >= 2 and signal_strength >= 3:
                    # Calculate position size
                    position_value = cash * self.position_size_pct
                    position_size = int(position_value / current_price)

                    if position_size > 0:
                        self.buy(symbol, position_size)
                        print(f"[ENTRY] {symbol} @ {current_price:.2f} | Qty: {position_size}")
                        print(f"  Signals: {', '.join(signals)} | Strength: {signal_strength}")
                        
                        # Initialize position metadata
                        if symbol not in self.position_metadata:
                            self.position_metadata[symbol] = {}
                        self.position_metadata[symbol]['hold_cycles'] = 0
                        self.position_metadata[symbol]['peak_price'] = current_price

            # Increment hold cycles for existing positions
            if symbol in self.positions and symbol in self.position_metadata:
                self.position_metadata[symbol]['hold_cycles'] = \
                    self.position_metadata[symbol].get('hold_cycles', 0) + 1


# ============================================================================
# MAIN PAPER TRADING FUNCTION (BACKEND ONLY)
# ============================================================================

def run_backend_only_trading():
    """
    Run paper trading with backend telemetry only (no local dashboard).
    All data is automatically sent to backend and monitored via Vercel web app.
    """
    print("=" * 80)
    print("PyTrader - Backend-Only Paper Trading Bot")
    print("=" * 80)
    print(f"✅ API Token: {API_TOKEN[:10]}..." if len(API_TOKEN) > 10 else f"✅ API Token: {API_TOKEN}")
    print(f"✅ Bot ID: {BOT_ID}")
    print(f"✅ Backend: Auto-configured (https://pytrader-backend.onrender.com)")
    print(f"📊 Symbols: {', '.join(SYMBOLS[:5])}")
    print(f"💰 Initial Cash: {INITIAL_CASH:,.0f} PKR")
    print(f"⏰ Cycle Interval: {CYCLE_MINUTES} minutes")
    print(f"🎯 Strategy: AggressiveMultiSignalStrategy")
    print("=" * 80)
    print()
    print("🚀 Starting paper trading bot...")
    print("📊 Monitor via your Vercel web app")
    print("⚠️  Press Ctrl+C to stop")
    print()

    # Create trader instance
    trader = Trader(
        strategy=AggressiveMultiSignalStrategy,
        symbols=SYMBOLS[:5],  # Trade first 5 symbols
        initial_cash=INITIAL_CASH,
        position_notional=POSITION_NOTIONAL,
        cycle_minutes=CYCLE_MINUTES,
        bot_id=BOT_ID
    )

    # Start paper trading WITHOUT dashboard
    # Backend URL is automatically configured - no need to specify!
    try:
        trader.run_paper_trading(
            api_token=API_TOKEN,
            # backend_url is NOT needed - auto-configured! ✅
            dashboard=False,  # ✅ NO LOCAL DASHBOARD
            warm_start=True,  # Start with today's data
            detailed_logs=True  # Detailed console logs
        )
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Paper trading stopped by user.")
        print("=" * 80)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    run_backend_only_trading()


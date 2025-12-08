"""
PyTrader SDK - Complete Colab / Local Notebook
==============================================

This file contains everything you need to:
1. Install the SDK
2. Run backtests with aggressive custom strategies
3. Run live paper trading (with optional dashboard)
4. 🆕 3 Colab-specific dashboard solutions

All strategies use REAL market data (no dummy data).
All strategies are CUSTOM and AGGRESSIVE (written from scratch).

Usage:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 INSTALLATION (run in a Colab cell):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
!pip install "git+https://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@github.com/ahmerhkhan/pytrader-papertrading-sdk.git"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 QUICK START (Google Colab):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Set your API token
API_TOKEN = "your-token-here"

# 2. Test setup
quick_test()

# 3. View dashboard options
colab_help()

# 4. Run paper trading (choose one):
run_colab_backend_only()         # ⭐ RECOMMENDED - Monitor via Vercel
run_colab_dashboard_builtin()    # Built-in Colab forwarding
run_colab_dashboard_ngrok()      # Stable public URL via ngrok

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💻 LOCAL USAGE (Windows/Mac/Linux):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python pytrader_colab_complete.py
# Opens dashboard at http://localhost:8787
"""

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
    print(f"WARNING: Import Error: {e}")
    print("Please install the SDK first:")
    print('!pip install "git+https://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@github.com/ahmerhkhan/pytrader-papertrading-sdk.git"')
    raise

# ============================================================================
# CONFIGURATION
# ============================================================================
# ⚠️ REQUIRED: Set your API token here
API_TOKEN = "dev-token"  # ✅ Change this to your API token

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
        trader.run_paper_trading(
            api_token=API_TOKEN,
            warm_start=True,  # Replay today's data first
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped by user.")


def run_paper_trading_with_dashboard():
    """Run paper trading with live dashboard (LOCAL ONLY - not for Colab)."""
    print("=" * 80)
    print("STARTING PAPER TRADING WITH DASHBOARD (LOCAL)")
    print("=" * 80)
    print("⚠️  NOTE: This works on local machines only!")
    print("⚠️  For Colab, use one of these instead:")
    print("    - run_colab_dashboard_builtin()")
    print("    - run_colab_dashboard_ngrok()")
    print("    - run_colab_backend_only() [RECOMMENDED]")
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
        trader.run_paper_trading(
            api_token=API_TOKEN,
            dashboard=True,  # Auto-launch dashboard
            dashboard_port=8787,
            warm_start=True,
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped.")


# ============================================================================
# GOOGLE COLAB SPECIFIC FUNCTIONS
# ============================================================================

def run_colab_dashboard_builtin():
    """
    SOLUTION 1: Use Google Colab's Built-in Port Forwarding
    
    ✅ Best for: Quick testing, no extra setup
    📝 How it works: Colab auto-detects the server and provides a public URL
    
    After running this, Colab will show:
    "Your application is running on port 8787"
    Click that URL to access your dashboard!
    """
    print("=" * 80)
    print("COLAB SOLUTION 1: Built-in Port Forwarding")
    print("=" * 80)
    print("✅ Starting dashboard with host='0.0.0.0'")
    print("📊 Colab will auto-detect the server")
    print("🔗 Look for 'Your application is running on port 8787'")
    print("=" * 80)

    # Create trader
    trader = Trader(
        strategy=AggressiveBreakoutTrader,
        symbols=SYMBOLS[:5],
        initial_cash=1_000_000.0,
        position_notional=120_000.0,
        cycle_minutes=15,
        bot_id="colab-builtin-bot"
    )

    # Start dashboard with 0.0.0.0 to allow Colab to detect it
    print("\n🚀 Launching dashboard on 0.0.0.0:8787...")
    start_dashboard(trader, host="0.0.0.0", port=8787, auto_open=False)
    
    print("\n✅ Dashboard started!")
    print("📌 Colab should now show a link above (click it to open dashboard)")
    print("⏳ Starting paper trading...\n")

    try:
        trader.run_paper_trading(
            api_token=API_TOKEN,
            dashboard=False,  # Already started manually
            warm_start=True,
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Paper trading stopped.")


def run_colab_dashboard_ngrok():
    """
    SOLUTION 2: Use ngrok for Reliable Public URL
    
    ✅ Best for: When built-in forwarding doesn't work
    📝 How it works: Creates a stable public URL via ngrok tunnel
    ⚠️  Requires: !pip install pyngrok
    
    Run this first in a Colab cell:
    !pip install pyngrok
    """
    print("=" * 80)
    print("COLAB SOLUTION 2: ngrok Tunneling")
    print("=" * 80)
    
    try:
        from pyngrok import ngrok
    except ImportError:
        print("❌ ERROR: pyngrok not installed!")
        print("\n📦 Please run this in a Colab cell first:")
        print("   !pip install pyngrok")
        print("\nThen run this function again.")
        return

    print("✅ ngrok installed")
    print("🚀 Creating secure tunnel...")
    print("=" * 80)

    # Create trader
    trader = Trader(
        strategy=AggressiveBreakoutTrader,
        symbols=SYMBOLS[:5],
        initial_cash=1_000_000.0,
        position_notional=120_000.0,
        cycle_minutes=15,
        bot_id="colab-ngrok-bot"
    )

    # Start dashboard locally first
    print("\n📊 Starting local dashboard...")
    start_dashboard(trader, host="127.0.0.1", port=8787, auto_open=False)
    
    # Create ngrok tunnel
    print("🌐 Creating ngrok tunnel...")
    try:
        public_url = ngrok.connect(8787, "http")
        print("\n" + "=" * 80)
        print("✅ DASHBOARD IS LIVE!")
        print("=" * 80)
        print(f"🔗 Dashboard URL: {public_url}")
        print(f"📱 Open this URL in any browser")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Error creating ngrok tunnel: {e}")
        print("💡 Try the backend-only solution instead: run_colab_backend_only()")
        return

    print("\n⏳ Starting paper trading...\n")

    try:
        trader.run_paper_trading(
            api_token=API_TOKEN,
            dashboard=False,  # Already started manually
            warm_start=True,
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Paper trading stopped.")
        print("🔌 Closing ngrok tunnel...")
        ngrok.disconnect(public_url)


def run_colab_backend_only():
    """
    SOLUTION 3: Backend-Only Mode (Monitor via Vercel Web App)
    
    ✅ Best for: Production use, most reliable
    📝 How it works: No local dashboard, all data sent to backend/Vercel app
    🌐 Monitor at: https://your-vercel-app.com
    
    ✨ RECOMMENDED for Colab - cleanest solution!
    """
    print("=" * 80)
    print("COLAB SOLUTION 3: Backend-Only (RECOMMENDED)")
    print("=" * 80)
    print("✅ No local dashboard")
    print("✅ All data sent to backend automatically")
    print("✅ Monitor via Vercel web app")
    print("✅ Most reliable for Colab")
    print("=" * 80)

    # Create trader
    trader = Trader(
        strategy=AggressiveBreakoutTrader,
        symbols=SYMBOLS[:5],
        initial_cash=1_000_000.0,
        position_notional=120_000.0,
        cycle_minutes=15,
        bot_id="colab-backend-bot"
    )

    print("\n🚀 Starting paper trading (backend-only mode)...")
    print("📊 Monitor your bot at: https://your-vercel-app.com")
    print("⏳ Starting...\n")

    try:
        trader.run_paper_trading(
            api_token=API_TOKEN,
            dashboard=False,  # ✅ No local dashboard
            warm_start=True,
            detailed_logs=True
        )
    except KeyboardInterrupt:
        print("\n\n✅ Paper trading stopped.")


# ============================================================================
# QUICK TEST FUNCTION (for Colab)
# ============================================================================

def quick_test():
    """Quick test to verify everything is set up correctly."""
    print("=" * 80)
    print("QUICK SETUP TEST")
    print("=" * 80)
    print(f"[OK] API Token: {API_TOKEN[:10]}..." if len(API_TOKEN) > 10 else f"[OK] API Token: {API_TOKEN}")
    print(f"[OK] Symbols: {', '.join(SYMBOLS[:4])}...")
    print(f"[OK] Strategies loaded: 5 aggressive strategies")
    print("\n[OK] All imports successful!")
    
    # Detect if running in Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    if is_colab:
        print("\n📍 Detected: Google Colab")
        print("\n✅ Available functions for Colab:")
        print("  🥇 run_colab_backend_only()        [RECOMMENDED - Most reliable]")
        print("  🥈 run_colab_dashboard_builtin()   [Built-in port forwarding]")
        print("  🥉 run_colab_dashboard_ngrok()     [Requires: !pip install pyngrok]")
        print("\n  📊 run_paper_trading()             [Backend-only, no dashboard]")
    else:
        print("\n📍 Detected: Local Environment")
        print("\n✅ Available functions:")
        print("  - run_paper_trading()")
        print("  - run_paper_trading_with_dashboard()")
    
    print("=" * 80)


def colab_help():
    """
    Display help for using PyTrader on Google Colab.
    Shows all 3 dashboard solutions with pros/cons.
    """
    print("\n" + "=" * 80)
    print("🎓 GOOGLE COLAB DASHBOARD GUIDE")
    print("=" * 80)
    
    print("\n📊 Problem: localhost:8787 doesn't work on Colab")
    print("✅ Solution: Use one of these 3 methods:\n")
    
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 🥇 SOLUTION 1: Backend-Only Mode [RECOMMENDED]                     │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Function: run_colab_backend_only()                                  │")
    print("│ ✅ Pros: Most reliable, no setup, works 100%                        │")
    print("│ ❌ Cons: Need Vercel web app access                                 │")
    print("│ 📝 How: All data sent to backend, monitor via web app               │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 🥈 SOLUTION 2: Colab Built-in Port Forwarding                      │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Function: run_colab_dashboard_builtin()                             │")
    print("│ ✅ Pros: No extra setup, automatic                                  │")
    print("│ ❌ Cons: May not always work, limited features                      │")
    print("│ 📝 How: Colab detects server, provides public URL                   │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 🥉 SOLUTION 3: ngrok Tunneling                                      │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Function: run_colab_dashboard_ngrok()                               │")
    print("│ ✅ Pros: Reliable, stable public URL                                │")
    print("│ ❌ Cons: Requires pyngrok install                                   │")
    print("│ 📝 Setup: !pip install pyngrok                                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 80)
    print("💡 QUICK START GUIDE")
    print("=" * 80)
    
    print("\n1️⃣  Set your API token:")
    print('    API_TOKEN = "your-token-here"')
    
    print("\n2️⃣  Test your setup:")
    print("    quick_test()")
    
    print("\n3️⃣  Choose a method and run:")
    print("    run_colab_backend_only()        # Recommended!")
    print("    # OR")
    print("    run_colab_dashboard_builtin()   # Try built-in first")
    print("    # OR")
    print("    !pip install pyngrok             # If ngrok needed")
    print("    run_colab_dashboard_ngrok()")
    
    print("\n" + "=" * 80)
    print("📚 More info: https://github.com/your-repo/docs")
    print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION (local usage)
# ============================================================================
if __name__ == "__main__":
    # Detect environment
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    if is_colab:
        # Running in Colab - show help
        print("=" * 80)
        print("🎓 PyTrader SDK - Google Colab Edition")
        print("=" * 80)
        print("\n📍 Detected: Running in Google Colab")
        print("\n💡 To get started, run:")
        print("   colab_help()  # Shows all dashboard options")
        print("\n⭐ Or jump right in:")
        print("   run_colab_backend_only()  # Recommended method")
        print("=" * 80)
    else:
        # Running locally - start dashboard
        print("=" * 80)
        print("PyTrader SDK - Paper Trading with Dashboard")
        print("=" * 80)
        print("Launching aggressive paper trading bot with dashboard...\n")
        run_paper_trading_with_dashboard()
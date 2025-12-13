"""
Live paper trading engine with 15-minute cycles and integrated metrics.

BACKEND-ONLY: This engine is for backend internal use only.
SDK client code should use PyTrader client instead.
"""

from __future__ import annotations

import warnings
import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import pandas as pd

from ...data.pypsx_service import PyPSXService
from ...telemetry import TelemetryClient
from .telemetry import (
    CallbackTelemetry,
    CompositeTelemetry,
    CycleReport,
    FileTelemetry,
    HttpTelemetry,
    NullTelemetry,
    BackendRelayTelemetry,
)
from ...utils.logger import PaperTradingLogger
from ..utils import is_market_open, log_line, now_tz
import sys
import os
from ..utils.metrics_csv import MetricsCSVWriter
from ...config import settings
from zoneinfo import ZoneInfo
from ..portfolio.metrics import (
    TradeMetrics,
    compute_portfolio_metrics,
)
from ..portfolio.service import PortfolioService, PortfolioSummary
from ...utils.exceptions import DataProviderError
from ...utils.time_utils import next_market_open, today_session_window
from ...utils.market_hours import PSXMarketHours
from .paper_account import PaperAccountManager
from .signal_queue import SignalQueue


def _empty_intraday_frame() -> pd.DataFrame:
    """Return a typed empty DataFrame for intraday data flows."""
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="datetime64[ns]"),
            "price": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def _as_series(value: Any) -> pd.Series:
    """Helper to appease static type checkers for pandas column access."""
    return cast(pd.Series, value)


@dataclass
class EngineConfig:
    cycle_minutes: int = 15
    lookback_days: int = 2
    position_notional: float = 100_000.0
    capital_allocation: Optional[float] = None  # Percentage of equity per trade (e.g., 0.2 = 20%)
    min_lot: int = 1
    initial_cash: Optional[float] = None
    log_dir: Path = Path("logs")
    backend_url: Optional[str] = None
    backend_endpoint: str = "/live/metrics"
    api_token: Optional[str] = None
    require_token: bool = False
    bias_threshold_pct: float = 0.1
    use_cache: bool = False
    verbose_warm_start: bool = False
    record_warm_start_trades: bool = True
    fast_sleep_seconds: Optional[float] = None  # Internal: for development/testing only (not for production use)
    slippage_bps: float = 0.0
    commission_per_share: float = 0.0
    commission_pct_notional: float = 0.0015  # Default: 0.15% per share (percentage-based)
    warm_start: bool = True  # Default: warm-start ON (replay from open)
    reset_account: bool = False  # Reset account to initial cash
    user_id: str = "default"  # User ID for account persistence
    allow_short: bool = False  # Allow short selling (selling without existing position)
    detailed_logs: bool = False  # Print verbose stats every cycle (default: concise)
    verbose: bool = False  # Same as detailed_logs for backward compatibility
    metrics_path: Optional[Path] = None  # Path to metrics CSV (default: pytrader_metrics.csv)
    trades_path: Optional[Path] = None  # Path to trades CSV (default: pytrader_trades.csv)
    price_log_threshold_pct: float = 0.25  # Minimum % move before price logs emit
    unrealized_log_threshold: float = 250.0  # Minimum PKR change in unrealized P/L before logging
    reset_session_metrics_on_warm_start: bool = True  # Forward-only metrics when joining mid-session


@dataclass
class SignalSnapshot:
    symbol: str
    side: Optional[str]
    strategy_signal: str
    bias: str
    generated_at: datetime
    signal_price: float
    vwap: float
    target_qty: int
    note: str = ""
    delta_pct: float = 0.0
    executed_at: Optional[datetime] = None
    batch_label: Optional[str] = None


class TradingEngine:
    """
    Paper trading loop that simulates live trading using 15-minute data batches.
    
    BACKEND-ONLY: This engine is for backend internal use only.
    SDK client code should use PyTrader client instead.
    """

    def __init__(
        self,
        symbols: List[str],
        strategy: Any,
        *,
        portfolio: Optional[PortfolioService] = None,
        data_service: Optional[PyPSXService] = None,
        config: Optional[EngineConfig] = None,
        bot_id: str = "default",
        telemetry: Optional[CompositeTelemetry] = None,
        in_process_push: Optional[Any] = None,
    ) -> None:
        # Warn if used from SDK client code (backend should pass data_service)
        if data_service is None:
            warnings.warn(
                "TradingEngine is backend-only. SDK client code should use PyTrader client instead: "
                "client = PyTrader(api_token='...'); client.start_bot(...)",
                DeprecationWarning,
                stacklevel=2,
            )
        self.symbols = [s.upper().strip() for s in symbols]
        self.strategy = strategy
        self._strategy_name = self._determine_strategy_name(strategy)
        self.config = config or EngineConfig()
        self.service = data_service or PyPSXService()
        self.bot_id = bot_id

        # Initialize account manager for persistent state
        self.account_manager = PaperAccountManager(
            user_id=self.config.user_id,
            default_cash=self.config.initial_cash or 1_000_000.0,
        )
        
        # Handle account reset if requested
        if self.config.reset_account:
            self.account_manager.reset_account(initial_cash=self.config.initial_cash)
        
        # Initialize portfolio with account state or defaults
        initial_cash = self.config.initial_cash
        account_state = None
        if not self.config.reset_account:
            account_state = self.account_manager.load_account()
            if account_state:
                initial_cash = account_state.cash
                log_line(f"Loaded saved account state: cash {initial_cash:,.0f}, equity {account_state.equity:,.0f}.")
        
        self.portfolio = portfolio or PortfolioService(initial_cash=initial_cash, allow_short=self.config.allow_short)
        
        if portfolio is None and initial_cash is not None:
            summary_snapshot = self.portfolio.get_summary()
            delta = float(initial_cash) - float(summary_snapshot.cash)
            if abs(delta) > 1e-6:
                self.portfolio.apply_cash_adjustment(delta)
        
        # Restore positions from account state if available
        if account_state and account_state.positions:
            self._restore_positions_from_account(account_state.positions)
        
        base_summary = self.portfolio.get_summary()
        self._initial_equity: Optional[float] = float(base_summary.equity)
        self._session_start_equity: Optional[float] = self._initial_equity
        self._trade_count_baseline: int = 0
        self._joined_mid_session: bool = False
        self._metrics_baseline = {
            "equity": float(base_summary.equity),
            "cash": float(base_summary.cash),
            "unrealized_pnl": float(base_summary.unrealized_pnl),
            "realized_pnl": float(base_summary.realized_pnl),
        }

        self.is_running: bool = False
        self._cycle_delta = timedelta(minutes=self.config.cycle_minutes)
        self._sleep_seconds = self.config.fast_sleep_seconds
        self._intraday_cache: Dict[str, pd.DataFrame] = {}
        self._session_data: Dict[str, pd.DataFrame] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self._last_seen_ts: Dict[str, pd.Timestamp] = {}
        self._last_vwap: Dict[str, float] = {}
        self._session_start: Optional[datetime] = None
        self._warm_start_complete: bool = False
        self._warm_start_cycles: int = 0
        self._supports_color: bool = self._check_color_support()
        self._session_end: Optional[datetime] = None
        self._last_cycle_close: Optional[datetime] = None
        self._local_tz: Optional[ZoneInfo] = None

        # Market day filtering: Track today's first trade and first price per symbol
        self._symbol_first_trade_time: Dict[str, datetime] = {}  # symbol -> first trade timestamp today
        self._symbol_open_price_today: Dict[str, float] = {}  # symbol -> first price today
        self._today_date: Optional[date] = None  # Current market day date (PKT)
        self._off_hours_scan_done: bool = False  # Track if off-hours scan has run

        if self.config.require_token and not self.config.api_token:
            raise RuntimeError(
                "PyTrader Live Trading requires an API token. "
                "Set PYTRADER_TOKEN or provide EngineConfig.api_token."
            )

        self._backend_client: Optional[TelemetryClient] = None
        self._telemetry, self._backend_client = self._build_telemetry(telemetry, in_process_push)
        self._backend_log_hook = self._build_backend_log_hook()
        log_root = self.config.log_dir
        if not isinstance(log_root, Path):
            log_root = Path(log_root)
        self._paper_logger = PaperTradingLogger(
            bot_id=self.bot_id,
            log_root=log_root / "streams",
            backend_hook=self._backend_log_hook,
        )
        self._price_log_threshold_pct = max(0.0, float(self.config.price_log_threshold_pct))
        self._unrealized_log_threshold = max(0.0, float(self.config.unrealized_log_threshold))
        self._last_seen_price: Dict[str, float] = {}
        self._last_logged_price: Dict[str, float] = {}
        self._last_logged_unrealized: Dict[str, float] = {}
        self._last_logged_qty: Dict[str, int] = {}
        self._active_signals: Dict[str, SignalSnapshot] = {}
        # Queue for orders generated after market close
        self._queued_signals: List[SignalSnapshot] = []
        
        # Initialize persistent signal queue
        self.signal_queue = SignalQueue(
            bot_id=self.bot_id,
            user_id=self.config.user_id,
        )
        
        # Load queued signals from persistent storage on startup
        self._load_queued_signals_from_storage()
        
        # Initialize CSV writer for metrics storage
        scoped_metrics_path = self.config.metrics_path
        scoped_trades_path = self.config.trades_path
        if scoped_metrics_path is None:
            scoped_metrics_path = log_root / f"{self.account_manager.user_id}-{self.bot_id}-metrics.csv"
        if scoped_trades_path is None:
            scoped_trades_path = log_root / f"{self.account_manager.user_id}-{self.bot_id}-trades.csv"

        self.csv_writer = MetricsCSVWriter(
            metrics_path=Path(scoped_metrics_path),
            trades_path=Path(scoped_trades_path),
        )
        
        # Handle verbose flag (backward compatibility with detailed_logs)
        if self.config.verbose and not self.config.detailed_logs:
            self.config.detailed_logs = self.config.verbose
    
    def _check_color_support(self) -> bool:
        """Check if terminal supports ANSI color codes."""
        if os.getenv("NO_COLOR") or os.getenv("TERM") == "dumb":
            return False
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _build_positions_snapshot(
        self,
        summary_positions: List[Dict[str, Any]],
        prices: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """
        Build a per-position snapshot enriched with live prices, market value, and unrealized PnL.
        """
        price_map = prices or {}
        snapshot: List[Dict[str, Any]] = []
        total_value = 0.0
        total_unrealized = 0.0
        for pos in summary_positions:
            qty = pos.get("qty", 0)
            if qty <= 0:
                continue
            symbol = pos["symbol"]
            avg_cost = float(pos.get("avg_cost", 0.0))
            current_price = float(price_map.get(symbol, avg_cost))
            market_value = current_price * qty
            unrealized = (current_price - avg_cost) * qty
            snapshot.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized,
                }
            )
            total_value += market_value
            total_unrealized += unrealized
        return snapshot, total_value, total_unrealized

    def _refresh_session_baseline(self) -> None:
        summary = self.portfolio.get_summary()
        self._session_start_equity = float(summary.equity)
        if self.config.reset_session_metrics_on_warm_start:
            self._reset_metrics_snapshot(summary)
            self._log_system_event(
                event="session_metrics_reset",
                message="Session metrics reset at warm start",
                level="info",
            )

    def _reset_metrics_snapshot(self, summary: PortfolioSummary) -> None:
        self._metrics_baseline = {
            "equity": float(summary.equity),
            "cash": float(summary.cash),
            "unrealized_pnl": float(summary.unrealized_pnl),
            "realized_pnl": float(summary.realized_pnl),
        }
    
    def _colorize(self, text: str, color: str) -> str:
        """Add ANSI color codes if terminal supports it."""
        if not self._supports_color:
            return text
        colors = {
            "green": "\033[32m",
            "red": "\033[31m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "cyan": "\033[36m",
            "reset": "\033[0m",
            "bold": "\033[1m",
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def start(self) -> None:
        if self.is_running:
            return
        now = now_tz()
        self._session_start, session_market_close = today_session_window(now)
        # Extend session end by 15 minutes to allow processing final batch
        self._session_end = session_market_close + timedelta(minutes=15)
        # Local timezone for display/alignment
        self._local_tz = ZoneInfo(getattr(settings, "timezone", "Asia/Karachi"))
        
        # Set today's date for strict filtering
        self._today_date = now.date()
        
        # Reset first trade tracking for new day
        self._symbol_first_trade_time.clear()
        self._symbol_open_price_today.clear()
        # Detect first available timestamp from today's data to avoid hard-coded start
        # Also detect if we're starting BEFORE first trade of the day
        earliest_ts: Optional[pd.Timestamp] = None
        has_today_trades = False
        probe_symbols = list(self.symbols)
        
        for sym in probe_symbols:
            try:
                # Load today's data with strict filtering
                df_today = self._load_symbol_history(
                    sym,
                    up_to=now,
                    use_cache=self.config.use_cache,
                    publish=False,
                )
                if not df_today.empty:
                    has_today_trades = True
                    cur_earliest_raw = df_today["ts"].min()
                    if pd.notna(cur_earliest_raw):
                        cur_earliest = pd.Timestamp(cur_earliest_raw)
                    if earliest_ts is None or cur_earliest < earliest_ts:
                        earliest_ts = cur_earliest
            except Exception:
                continue
        
        # Handle case: Starting BEFORE first trade of the day
        if not has_today_trades or earliest_ts is None:
            # Check if market is closed - if so, allow off-hours scan to run
            can_paper_trade = PSXMarketHours.can_paper_trade(now)
            if not can_paper_trade:
                # Market is closed - allow bot to run so off-hours scan can execute
                log_line("Market is closed. Bot will run off-hours scan to generate queued signals.")
                self._last_cycle_close = now
                self._last_seen_ts.clear()
                self._last_vwap.clear()
                self.is_running = True
                return  # Allow cycle_once to run off-hours scan
            else:
                # Market should be open but no trades yet - wait for first trade
                print("\n" + "-" * 70)
                print("⏳ WAITING FOR FIRST MARKET TRADE")
                print("-" * 70)
                print("No trades available for today yet.")
                print("Portfolio loaded. Waiting for first price update...")
                print("-" * 70 + "\n")
                # Set session start to current time, will update when first trade arrives
                self._last_cycle_close = now
                self._last_seen_ts.clear()
                self._last_vwap.clear()
                # Mark as running so cycle_once can check for first trade
                self.is_running = True
                return  # Don't proceed with full initialization until first trade
        
        # We have today's trades - align to cycle grid
        if earliest_ts is not None:
            # Align to our cycle grid and clamp not before declared session start
            aligned = earliest_ts.floor(f"{self.config.cycle_minutes}min")
            if aligned > self._session_start:
                self._session_start = aligned
        self._last_cycle_close = self._session_start
        self._last_seen_ts.clear()
        self._last_vwap.clear()
        
        # Log initialization (clean format for users)
        symbols_text = ", ".join(self.symbols)
        print("\n" + "-" * 70)
        if self.config.warm_start:
            print(f"MODE: LIVE-WARM | Symbols: {symbols_text} | Interval: {self.config.cycle_minutes}m")
            print("-" * 70)
            current_str = self._format_local_time(now)
            print(f"\nWarm starting from current time: {current_str}")
            print("Restoring account state and revaluing with current market prices...\n")
            self._initialize_history(now)
            summary = self.portfolio.get_summary()
            print("\n" + "-" * 70)
            print("✅ Warm Start Completed")
            print(f"Cash: {summary.cash:,.0f} PKR | Equity: {summary.equity:,.0f} PKR")
            if summary.positions:
                print(f"Positions: {len(summary.positions)} open")
            print("Now entering LIVE PAPER TRADING...")
            print("-" * 70 + "\n")
            self._warm_start_complete = True
            self._joined_mid_session = has_today_trades
        else:
            print(f"MODE: LIVE | Symbols: {symbols_text} | Interval: {self.config.cycle_minutes}m")
            print("-" * 70)
            summary = self.portfolio.get_summary()
            print("\nStarting fresh from current time")
            print(f"Cash: {summary.cash:,.0f} PKR | Equity: {summary.equity:,.0f} PKR\n")
            self._warm_start_complete = True  # No warm-start, so we're immediately in live mode
        
        self._trade_count_baseline = self._current_trade_count()
        
        self.is_running = True
        self._log_system_event(
            event="bot_started",
            message="Paper trading engine started",
            symbols=self.symbols,
            cycle_minutes=self.config.cycle_minutes,
            warm_start=self.config.warm_start,
                joined_mid_session=self._joined_mid_session,
        )

    def stop(self) -> None:
        """Stop the trading engine, persist state, and close telemetry."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Print session summary before stopping
        if self.metrics_history:
            self._print_session_summary()
        
        log_line("Trading session stopped")
        self._log_system_event(event="bot_stopped", message="Paper trading engine stopped")
        
        # Save final account state before stopping
        try:
            summary = self.portfolio.get_summary()
            positions_dict = {pos["symbol"]: pos["qty"] for pos in summary.positions}
            self.account_manager.save_account(
                cash=float(summary.cash),
                positions=positions_dict,
                equity=float(summary.equity),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log_line(f"Failed to save account state on stop: {exc}")
        
        if hasattr(self, "_telemetry") and self._telemetry is not None:
            try:
                self._telemetry.close()
            except Exception:
                pass

    async def run_forever(self, max_cycles: Optional[int] = None) -> None:
        """
        Continuous loop that runs until `stop()` is called or max_cycles is reached.
        Handles market hours, errors, and continues running across sessions.
        """
        self.start()
        
        # Check for queued signals on startup and execute if market is open
        now = now_tz()
        market_is_open = is_market_open(now)
        can_trade = market_is_open and PSXMarketHours.can_start_trading(now)
        if can_trade and self._queued_signals:
            log_line("Market is open on startup. Processing queued signals...")
            self._process_queued_signals(now)
        
        cycles = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_hourly_summary: Optional[datetime] = None
        try:
            while self.is_running:
                try:
                    await self.cycle_once()
                    consecutive_errors = 0  # Reset error counter on success
                    cycles += 1
                    
                    # Print hourly summary only if at least 1 hour has passed
                    now = now_tz()
                    if last_hourly_summary is None:
                        # First cycle - set initial time but don't print summary yet
                        last_hourly_summary = now
                    elif (now - last_hourly_summary).total_seconds() >= 3600:
                        # At least 1 hour has passed - print summary
                        self._print_hourly_summary(now)
                        last_hourly_summary = now
                    
                    if max_cycles is not None and cycles >= max_cycles:
                        log_line(f"Reached max_cycles ({max_cycles}). Stopping.")
                        break
                except asyncio.CancelledError:
                    log_line("Gracefully stopping live engine...")
                    break
                except KeyboardInterrupt:
                    log_line("Manual stop detected.")
                    break
                except Exception as exc:
                    consecutive_errors += 1
                    message = f"Cycle {cycles + 1} failed: {exc}"
                    log_line(message)
                    self._log_system_event(
                        event="cycle_error",
                        message=message,
                        level="error",
                        cycle=cycles + 1,
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        warn_msg = f"Too many consecutive errors ({consecutive_errors}). Stopping."
                        log_line(warn_msg)
                        self._log_system_event(
                            event="cycle_error_limit",
                            message=warn_msg,
                            level="error",
                            consecutive_errors=consecutive_errors,
                        )
                        break
                    await asyncio.sleep(5.0)  # Brief pause before retrying
                    continue
                
                await self._sleep_until_next_cycle()
        except (KeyboardInterrupt, asyncio.CancelledError):
            log_line("Shutdown signal received. Stopping gracefully...")
        finally:
            self.stop()
            log_line(f"Session ended. Total cycles: {cycles}")

    def _initialize_history(self, now: datetime) -> None:
        """
        Warm start: Restore account state and revalue with current prices.
        Historical replay is handled separately to ensure we process every
        completed cycle before switching to live mode.
        """
        # Load current market prices for all symbols (today's data only)
        current_prices: Dict[str, float] = {}
        
        # Get latest prices from market feed (today's trades only)
        for symbol in self.symbols:
            try:
                # Load intraday data with strict today filtering
                df_full = self._load_symbol_history(
                    symbol,
                    up_to=now,
                    use_cache=self.config.use_cache,
                    publish=False,
                )
                self._intraday_cache[symbol] = df_full
                self._session_data[symbol] = df_full
                if not df_full.empty:
                    # Get the most recent price from today
                    latest_price = float(df_full["price"].iloc[-1])
                    current_prices[symbol] = latest_price
                    ts_series = cast(pd.Series, df_full["ts"])
                    last_ts = pd.Timestamp(ts_series.max())
                    if pd.notna(last_ts):
                        self._last_seen_ts[symbol] = last_ts
                elif symbol in self._symbol_open_price_today:
                    # No trades yet today, but we have first price cached
                    current_prices[symbol] = self._symbol_open_price_today[symbol]
                    log_line(f"[{symbol}] Using today's first price: {current_prices[symbol]:.2f}")
            except Exception as exc:
                log_line(f"Could not load price for {symbol} during warm start: {exc}")
                # Try to get price from portfolio if position exists
                summary = self.portfolio.get_summary()
                for pos in summary.positions:
                    if pos["symbol"] == symbol and pos.get("avg_cost", 0.0) > 0:
                        current_prices[symbol] = pos["avg_cost"]
                        break
        
        # Revalue portfolio with current prices (this updates unrealized PnL and equity)
        if current_prices:
            self.portfolio.revalue_and_snapshot(now, current_prices)
            log_line(f"Revalued portfolio with current market prices for {len(current_prices)} symbols")
        else:
            log_line("Warning: No prices available for warm start. Portfolio not revalued.")
        
        self._log_daily_first_trades()
        
        # Get final summary after revaluation
        summary = self.portfolio.get_summary()
        
        # Log warm start completion
        if summary.positions:
            pos_count = len(summary.positions)
            log_line(f"Warm start: Restored {pos_count} positions, cash: {summary.cash:,.0f} PKR, equity: {summary.equity:,.0f} PKR")
        else:
            log_line(f"Warm start: No previous positions, cash: {summary.cash:,.0f} PKR, equity: {summary.equity:,.0f} PKR")
        self._session_start_equity = float(summary.equity)

        # Prime signal snapshots without executing trades
        for symbol, df_full in self._session_data.items():
            if df_full is None or df_full.empty:
                continue
            snapshot, _ = self._build_signal_snapshot(symbol, df_full, now)
            if snapshot:
                self._active_signals[symbol] = snapshot
        log_line("Warm start: Signals primed from cumulative intraday data (no trades executed).")

    async def cycle_once(self) -> None:
        if not self.is_running:
            return
        now = now_tz()
        
        # Check if market day has changed (reset tracking)
        if self._today_date and now.date() != self._today_date:
            log_line("Market day changed. Resetting first trade tracking.")
            self._today_date = now.date()
            self._symbol_first_trade_time.clear()
            self._symbol_open_price_today.clear()
            self._last_seen_ts.clear()
            self._off_hours_scan_done = False  # Reset to allow new off-hours scan on new market day
            self._refresh_session_baseline()
        
        # Check if we're in paper trading window (includes post-market data processing)
        can_paper_trade = PSXMarketHours.can_paper_trade(now)
        market_is_open = is_market_open(now)
        can_trade = market_is_open and PSXMarketHours.can_start_trading(now)
        
        if not can_paper_trade:
            # Outside paper trading window - run off-hours scan if not done yet
            if not self._off_hours_scan_done:
                self._run_off_hours_scan()
                return
            
            # Off-hours scan already done - just log queued signals and sleep
            if self._queued_signals:
                log_line(f"Paper trading closed. {len(self._queued_signals)} signal(s) queued for next market open.")
            
            # Market fully closed - skip first trade check and go to sleep
            # The sleep will happen in _sleep_until_next_cycle()
            cycle_end = self._determine_cycle_end(now)
            if cycle_end is None:
                # No cycle to process, just sleep until next market open
                await self._sleep_until_next_cycle()
                return
            # If cycle_end is set, continue to process cycle (shouldn't happen when market closed, but handle gracefully)
            
        elif market_is_open and not can_trade:
            # Market is open but data not yet available (waiting for first 15-min batch)
            local_time = now.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else now
            message = f"Waiting for first data batch (15-min delay). Market opened at 9:32 AM, data available ~9:45 AM..."
            log_line(message)
            self._log_system_event(
                event="waiting_for_data",
                message=message,
                level="info",
                local_time=local_time.isoformat(),
            )
            await asyncio.sleep(30.0)  # Retry after 30 seconds
            return
        
        # Process queued signals if market just opened
        if market_is_open and can_trade and self._queued_signals:
            self._process_queued_signals(now)
        
        # Check if we're still waiting for first trade per symbol (only if market is open)
        # Skip this check if market is closed - off-hours scan uses fallback data
        if can_paper_trade:
            missing_first_trades = [sym for sym in self.symbols if sym not in self._symbol_first_trade_time]
            if missing_first_trades:
                for sym in missing_first_trades:
                    try:
                        self._load_symbol_history(sym, up_to=now, use_cache=False, publish=False)
                    except Exception:
                        continue
                if not self._symbol_first_trade_time:
                    log_line("Waiting for today's first market trade...")
                    await asyncio.sleep(30.0)  # Retry after 30 seconds
                    return
        cycle_end = self._determine_cycle_end(now)
        if cycle_end is None:
            # If we have a last cycle, calculate when the next one should be
            if self._last_cycle_close is not None:
                next_expected = self._last_cycle_close + self._cycle_delta
                session_end = self._session_end
                if session_end is not None and next_expected > session_end:
                    last_cycle_str = self._format_local_time(self._last_cycle_close)
                    print(f"\n⏸️  Market session ended. Last cycle was at {last_cycle_str}\n")
                else:
                    local_time = now.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else now
                    next_local = next_expected.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else next_expected
                    next_label = self._format_local_time(next_local)
                    current_label = self._format_local_time(local_time)
                    print(f"\n⏸️  Waiting for next cycle at {next_label} (current: {current_label})\n")
            return
        try:
            self._process_cycle(cycle_end, telemetry=True, warm_start=False, log_warm=False)
        except Exception as exc:
            log_line(f"Cycle processing failed: {exc}. Continuing to next cycle...")
            import traceback
            log_line(f"Traceback details: {traceback.format_exc()}")

    def _run_off_hours_scan(self) -> None:
        """Run a single strategy scan using the most recent available data."""
        self._off_hours_scan_done = True
        log_line("\n" + "=" * 70)
        log_line("🌙 MARKET CLOSED: Running off-hours strategy scan...")
        log_line("=" * 70)
        
        # Use a fake cycle end time (now) for the scan
        scan_time = now_tz()
        
        try:
            # Force process cycle with fallback enabled
            self._process_cycle(
                scan_time, 
                telemetry=False, # Don't publish off-hours data to stream
                warm_start=False, 
                log_warm=False,
                fallback_to_recent=True # ENABLE FALLBACK
            )
            
            # Report queued signals
            queued_count = len(self._queued_signals)
            if queued_count > 0:
                log_line("=" * 70)
                log_line(f"✅ OFF-HOURS SCAN COMPLETE: {queued_count} signal(s) queued for next market open.")
                for sig in self._queued_signals:
                    log_line(f"   -> {sig.symbol}: {sig.side} {sig.target_qty} @ ~{sig.signal_price:.2f}")
                log_line("=" * 70 + "\n")
                
                # All signals are now automatically persisted to SQLite storage
            else:
                log_line("✅ OFF-HOURS SCAN COMPLETE: No signals generated.\n")
                
        except Exception as exc:
            log_line(f"Off-hours scan failed: {exc}")
            import traceback
            log_line(traceback.format_exc())

    def _determine_cycle_end(self, now: datetime) -> Optional[datetime]:
        if self._session_start is None or self._session_end is None:
            self._session_start, session_market_close = today_session_window(now)
            # Extend session end by 15 minutes to allow processing final batch
            self._session_end = session_market_close + timedelta(minutes=15)
        if now < self._session_start:
            return None
        if self._sleep_seconds is not None:
            # Dev mode: run cycles immediately
            base = self._last_cycle_close or self._session_start
            return base + self._cycle_delta
        # Normal mode: align to cycle boundaries
        cycle_end = self._floor_to_cycle(now)
        # If we've already processed this exact cycle, wait for the next one
        if self._last_cycle_close is not None and cycle_end <= self._last_cycle_close:
            return None
        # If cycle_end is before session start, use session start as first cycle
        if cycle_end < self._session_start:
            cycle_end = self._session_start
        if cycle_end > self._session_end:
            return None
        return cycle_end

    def _process_cycle(
        self,
        cycle_end: datetime,
        *,
        telemetry: bool,
        warm_start: bool,
        log_warm: bool,
        fallback_to_recent: bool = False,
    ) -> None:
        if self._session_start is None:
            return
        if self._last_cycle_close is None:
            self._last_cycle_close = self._session_start
        cycle_start = self._last_cycle_close
        portfolio_summary = self.portfolio.get_summary()
        summary_positions = {pos["symbol"]: pos["qty"] for pos in portfolio_summary.positions}
        cash_bucket = {"value": float(portfolio_summary.cash)}
        prices: Dict[str, float] = {}
        cycle_trades: List[Dict[str, Any]] = []
        batch_reports: List[Dict[str, Any]] = []
        cycle_commission_total = 0.0
        slippage_weighted_sum = 0.0
        slippage_qty_total = 0

        for symbol in self.symbols:
            try:
                df_full = self._load_symbol_history(
                    symbol,
                    up_to=cycle_end,
                    use_cache=self.config.use_cache,
                    publish=telemetry,
                    fallback_to_recent=fallback_to_recent,
                )
                self._intraday_cache[symbol] = df_full
                self._session_data[symbol] = df_full
                if df_full.empty:
                    if symbol not in self._symbol_first_trade_time:
                        if not warm_start:
                            log_line(f"[{symbol}] Waiting for today's first trade...")
                    elif not warm_start:
                        cycle_label = self._format_local_time(cycle_end)
                        message = f"No data available for {symbol} in cycle {cycle_label}."
                        log_line(message)
                        self._log_system_event(
                            event="missing_market_data",
                            message=message,
                            level="warning",
                            symbol=symbol,
                            cycle_end=cycle_end.isoformat(),
                        )
                    batch_reports.append(
                        self._summarize_empty_batch(symbol, cycle_start, cycle_end)
                    )
                    continue

                last_price = float(df_full["price"].iloc[-1])
                prices[symbol] = last_price
                self._last_seen_price[symbol] = last_price
                prev_snapshot = self._active_signals.get(symbol)
                snapshot, summary = self._build_signal_snapshot(symbol, df_full, cycle_end)
                signal_status = "no_signal"
                snapshot_to_execute: Optional[SignalSnapshot] = None

                if prev_snapshot and prev_snapshot.executed_at is None:
                    snapshot_to_execute = prev_snapshot
                    signal_status = "execute_pending"
                    if snapshot and not self._signals_equal(prev_snapshot, snapshot):
                        self._active_signals[symbol] = snapshot
                    elif snapshot:
                        self._active_signals[symbol] = prev_snapshot
                    else:
                        self._active_signals[symbol] = prev_snapshot
                elif snapshot:
                    if prev_snapshot and self._signals_equal(prev_snapshot, snapshot):
                        signal_status = "unchanged"
                        self._active_signals[symbol] = prev_snapshot
                    else:
                        self._active_signals[symbol] = snapshot
                        snapshot_to_execute = snapshot
                        signal_status = "refreshed"
                else:
                    self._active_signals.pop(symbol, None)

                summary["signal_status"] = signal_status
                batch_reports.append(summary)

                self._log_system_event(
                    event="signal_status",
                    message=f"{symbol} signal {signal_status}",
                    level="info",
                    symbol=symbol,
                    status=signal_status,
                    side=(snapshot_to_execute.side if snapshot_to_execute else (snapshot.side if snapshot else None)),
                    qty=(snapshot_to_execute.target_qty if snapshot_to_execute else (snapshot.target_qty if snapshot else 0)),
                )

                if warm_start or snapshot_to_execute is None:
                    summary["note"] = summary.get("note", "signal_pending")
                    continue

                # Check if market is open and can trade (accounting for 15-min delay)
                market_is_open = is_market_open(cycle_end)
                can_trade_now = market_is_open and PSXMarketHours.can_start_trading(cycle_end)
                
                if not can_trade_now:
                    # Market is closed or data not available - queue the signal
                    if snapshot_to_execute.side in {"BUY", "SELL"}:
                        self._queued_signals.append(snapshot_to_execute)
                        # Save to persistent storage
                        self._save_signal_to_queue(snapshot_to_execute)
                        summary["note"] = "queued_for_next_market_open"
                        summary["execution_side"] = snapshot_to_execute.side or "HOLD"
                        summary["executed_qty"] = 0
                        summary["execution_price"] = None
                        log_line(f"[{symbol}] Signal {snapshot_to_execute.side} queued (market closed). Will execute at next market open.")
                        self._log_system_event(
                            event="signal_queued",
                            message=f"{symbol} {snapshot_to_execute.side} signal queued for next market open",
                            level="info",
                            symbol=symbol,
                            side=snapshot_to_execute.side,
                            qty=snapshot_to_execute.target_qty,
                        )
                    continue

                executed = self._execute_signal(
                    snapshot_to_execute,
                    summary,
                    cycle_end,
                    summary_positions,
                    cash_bucket,
                )
                batch_reports[-1] = executed["summary"]
                trade_payload = executed["trade"]
                commission_value = executed.get("commission", 0.0)
                applied_slippage = executed.get("slippage_bps", 0.0)
                executed_qty = executed["summary"].get("executed_qty", 0)
                if commission_value:
                    cycle_commission_total += commission_value
                if executed_qty:
                    slippage_weighted_sum += applied_slippage * executed_qty
                    slippage_qty_total += executed_qty
                if trade_payload and telemetry:
                    cycle_trades.append(trade_payload)
            except Exception as exc:
                log_line(f"Error processing {symbol} in cycle {cycle_end}: {exc}. Continuing...")
                batch_reports.append(
                    {
                        "symbol": symbol,
                        "window_start": cycle_start.isoformat(),
                        "window_end": cycle_end.isoformat(),
                        "status": "error",
                        "note": str(exc),
                        "bias": "NEUTRAL",
                        "delta_pct": 0.0,
                        "total_volume": 0.0,
                        "trades": 0,
                        "vwap": None,
                        "strategy_signal": "HOLD",
                        "execution_side": "HOLD",
                        "executed_qty": 0,
                        "execution_price": None,
                        "last_price": None,
                    }
                )

        if prices:
            # Always revalue portfolio and compute metrics after each cycle (warm or live)
            self.portfolio.revalue_and_snapshot(cycle_end, prices)
            metrics = compute_portfolio_metrics(
                self.portfolio,
                timestamp=cycle_end,
                latest_prices=prices,
            )
            avg_slippage_bps = (
                slippage_weighted_sum / slippage_qty_total if slippage_qty_total else 0.0
            )
            metrics["total_fees"] = cycle_commission_total
            metrics["avg_slippage_bps"] = avg_slippage_bps
            self.metrics_history.append(metrics)
            
            # Get portfolio summary for telemetry and account saving
            summary = self.portfolio.get_summary()
            positions_for_cycle, positions_value, unrealized_total = self._build_positions_snapshot(
                summary.positions,
                prices,
            )
            total_equity = float(summary.cash) + positions_value
            if self.config.reset_session_metrics_on_warm_start and self._metrics_baseline:
                session_unrealized = unrealized_total - self._metrics_baseline["unrealized_pnl"]
                session_realized = float(summary.realized_pnl) - self._metrics_baseline["realized_pnl"]
                summary_display = PortfolioSummary(
                    cash=float(summary.cash),
                    positions=summary.positions,
                    unrealized_pnl=session_unrealized,
                    realized_pnl=session_realized,
                    equity=float(self._metrics_baseline["equity"] + session_realized + session_unrealized),
                    last_updated=summary.last_updated,
                )
            else:
                summary_display = summary
            summary.unrealized_pnl = unrealized_total
            summary.equity = total_equity
            if self._initial_equity is None:
                self._initial_equity = total_equity
            if self._session_start_equity is None:
                self._session_start_equity = total_equity
            session_baseline = self._session_start_equity or total_equity
            cumulative_baseline = self._initial_equity or total_equity
            session_return_pct = (
                ((total_equity - session_baseline) / session_baseline) * 100 if session_baseline else 0.0
            )
            cumulative_return_pct = (
                ((total_equity - cumulative_baseline) / cumulative_baseline) * 100 if cumulative_baseline else 0.0
            )
            metrics["session_return_pct"] = round(session_return_pct, 2)
            metrics["cumulative_return_pct"] = round(cumulative_return_pct, 2)
            if self.config.reset_session_metrics_on_warm_start and self._metrics_baseline:
                metrics["realized_pnl"] = float(summary.realized_pnl) - self._metrics_baseline["realized_pnl"]
                metrics["unrealized_pnl"] = float(unrealized_total) - self._metrics_baseline["unrealized_pnl"]
            else:
                metrics["realized_pnl"] = float(summary.realized_pnl)
                metrics["unrealized_pnl"] = float(unrealized_total)
            
            # Save account state after each cycle (only in live mode, not warm-start)
            if telemetry and not warm_start:
                positions_dict = {pos["symbol"]: pos["qty"] for pos in summary.positions}
                self.account_manager.save_account(
                    cash=float(summary.cash),
                    positions=positions_dict,
                    equity=total_equity,
                )
            
            # Log metrics and publish telemetry if enabled
            # Note: record_warm_start_trades controls whether warm-start cycles use telemetry

            if self._paper_logger:
                positions_map = {pos["symbol"]: pos["qty"] for pos in summary.positions}
                avg_cost_map = {pos["symbol"]: pos.get("avg_cost", 0.0) for pos in summary.positions}
                for sym, updated_price in prices.items():
                    old_price = self._last_seen_price.get(sym)
                    qty = positions_map.get(sym, 0)
                    avg_cost = avg_cost_map.get(sym, updated_price)
                    unrealized = (updated_price - avg_cost) * qty if qty else 0.0
                    last_logged_price = self._last_logged_price.get(sym)
                    if last_logged_price and last_logged_price != 0:
                        price_change_pct = abs((float(updated_price) - last_logged_price) / last_logged_price) * 100
                    else:
                        price_change_pct = float("inf") if last_logged_price == 0 else 100.0
                    last_logged_unreal = self._last_logged_unrealized.get(sym)
                    unrealized_delta = abs(unrealized - (last_logged_unreal or 0.0))
                    qty_changed = self._last_logged_qty.get(sym) != qty
                    should_log = (
                        last_logged_price is None
                        or qty_changed
                        or price_change_pct >= self._price_log_threshold_pct
                        or unrealized_delta >= self._unrealized_log_threshold
                    )
                    if should_log:
                        self._paper_logger.log_price_update(
                            timestamp=cycle_end,
                            symbol=sym,
                            old_price=old_price,
                            new_price=float(updated_price),
                            position_qty=qty,
                            unrealized_pnl=float(unrealized),
                            equity=total_equity,
                        )
                        self._last_logged_price[sym] = float(updated_price)
                        self._last_logged_unrealized[sym] = float(unrealized)
                        self._last_logged_qty[sym] = qty
                    self._last_seen_price[sym] = float(updated_price)
                self._paper_logger.log_portfolio_snapshot(
                    timestamp=cycle_end,
                    cash=float(summary_display.cash),
                    positions_value=float(positions_value),
                    equity=summary_display.equity,
                    realized_pnl=float(summary_display.realized_pnl),
                    unrealized_pnl=float(summary_display.unrealized_pnl),
                )

            if telemetry:
                
                
                # Write metrics to CSV (always, for analysis)
                trade_metrics = metrics.get("metrics")
                if not isinstance(trade_metrics, TradeMetrics):
                    trade_metrics = TradeMetrics(
                        total_return_pct=float(metrics.get("total_return_pct", 0.0)),
                        daily_return_pct=float(metrics.get("daily_return_pct", 0.0)),
                        session_return_pct=float(metrics.get("session_return_pct", metrics.get("daily_return_pct", 0.0))),
                        cumulative_return_pct=float(metrics.get("cumulative_return_pct", metrics.get("total_return_pct", 0.0))),
                        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
                        session_sharpe_ratio=float(metrics.get("session_sharpe_ratio", metrics.get("sharpe_ratio", 0.0))),
                        sortino_ratio=float(metrics.get("sortino_ratio", 0.0)),
                        max_drawdown_pct=float(metrics.get("max_drawdown_pct", 0.0)),
                        volatility_pct=float(metrics.get("volatility_pct", 0.0)),
                        win_loss_ratio=float(metrics.get("win_loss_ratio", 0.0)),
                        exposure_pct=float(metrics.get("exposure_pct", 0.0)),
                        turnover_pct=float(metrics.get("turnover_pct", 0.0)),
                        cumulative_pnl=float(metrics.get("cumulative_pnl", 0.0)),
                        sharpe_ratio_available=bool(metrics.get("sharpe_ratio")),
                        session_sharpe_ratio_available=bool(metrics.get("session_sharpe_ratio")),
                        sortino_ratio_available=bool(metrics.get("sortino_ratio")),
                        volatility_available=bool(metrics.get("volatility_pct")),
                        win_loss_ratio_available=bool(metrics.get("win_loss_ratio")),
                    )
                metrics["equity"] = total_equity
                metrics["cash"] = float(summary.cash)
                metrics["positions_value"] = positions_value
                metrics["realized_pnl"] = float(summary_display.realized_pnl)
                metrics["unrealized_pnl"] = float(summary_display.unrealized_pnl)
                metrics["session_return_pct"] = round(session_return_pct, 2)
                metrics["cumulative_return_pct"] = round(cumulative_return_pct, 2)
                metrics["session_sharpe_ratio"] = (
                    trade_metrics.session_sharpe_ratio if trade_metrics.session_sharpe_ratio_available else None
                )
                csv_metrics_payload: Dict[str, Any] = {
                    "total_return_pct": trade_metrics.total_return_pct,
                    "daily_return_pct": trade_metrics.daily_return_pct,
                    "session_return_pct": trade_metrics.session_return_pct,
                    "cumulative_return_pct": trade_metrics.cumulative_return_pct,
                    "unrealized_pnl": float(unrealized_total),
                    "realized_pnl": float(summary.realized_pnl),
                    "sharpe_ratio": trade_metrics.sharpe_ratio if trade_metrics.sharpe_ratio_available else None,
                    "session_sharpe_ratio": (
                        trade_metrics.session_sharpe_ratio if trade_metrics.session_sharpe_ratio_available else None
                    ),
                    "sortino_ratio": trade_metrics.sortino_ratio if trade_metrics.sortino_ratio_available else None,
                    "max_drawdown_pct": trade_metrics.max_drawdown_pct,
                    "volatility_pct": trade_metrics.volatility_pct if trade_metrics.volatility_available else None,
                    "win_loss_ratio": trade_metrics.win_loss_ratio if trade_metrics.win_loss_ratio_available else None,
                    "exposure_pct": trade_metrics.exposure_pct,
                    "turnover_pct": trade_metrics.turnover_pct,
                }
                self.csv_writer.write_metrics(
                    timestamp=cycle_end,
                    equity=total_equity,
                    cash=float(summary.cash),
                    positions_value=positions_value,
                    metrics=csv_metrics_payload,
                    total_fees=cycle_commission_total,
                    avg_slippage_bps=avg_slippage_bps,
                    positions_snapshot=positions_for_cycle,  # Optional JSON snapshot (spec requirement)
                )
                
                # Write trades to CSV
                # Get portfolio state after all trades (cash_after and equity_after)
                # Note: For trades in the same cycle, we use the final state after all trades
                for trade in cycle_trades:
                    price_executed = trade.get("price_executed") or trade.get("price", 0.0)
                    quantity = trade.get("quantity", 0)
                    realized_pnl = trade.get("pnl_realized", 0.0)
                    
                    self.csv_writer.write_trade(
                        timestamp=cycle_end,
                        symbol=trade.get("symbol", ""),
                        side=trade.get("side", ""),
                        quantity=quantity,
                        price=price_executed,
                        realized_pnl=realized_pnl,
                        cash_after=float(summary.cash),
                        equity_after=total_equity,
                        slippage_value=trade.get("slippage_value"),
                    )
                
                # Write positions snapshot to CSV
                self.csv_writer.write_positions(cycle_end, positions_for_cycle, prices)
                
                # Include per-position snapshot in metrics payload for downstream logging
                metrics["positions"] = positions_for_cycle
                
                log_payload = {
                    "metrics": metrics,  # This is the dict from compute_portfolio_metrics
                    "positions": positions_for_cycle,
                    "prices": prices,  # Include prices for fallback in _log_metrics
                    "total_fees": cycle_commission_total,
                    "avg_slippage_bps": avg_slippage_bps,
                    "realized_pnl": float(summary_display.realized_pnl),
                    "unrealized_pnl": float(summary_display.unrealized_pnl),
                    "equity": total_equity,
                    "cash": float(summary.cash),
                    "positions_value": positions_value,
                    "joined_mid_session": self._joined_mid_session,
                }
                recent_trades = self._get_recent_trades()
                
                # Log metrics (concise or detailed based on config)
                self._log_metrics(log_payload, cycle_end, warm_start, recent_trades=recent_trades)
                
                self._publish_cycle(
                    cycle_end,
                    metrics,
                    prices,
                    cycle_trades,
                    batch_reports,
                    total_fees=cycle_commission_total,
                    avg_slippage_bps=avg_slippage_bps,
                    recent_trades=recent_trades,
                )
                # Batch completion is already logged in _log_metrics, no need to duplicate
        else:
            summary = self.portfolio.get_summary()
            positions_for_cycle, positions_value, unrealized_total = self._build_positions_snapshot(summary.positions, prices)
            total_equity = float(summary.cash) + positions_value
            summary.unrealized_pnl = unrealized_total
            summary.equity = total_equity
            if self._paper_logger:
                self._paper_logger.log_portfolio_snapshot(
                    timestamp=cycle_end,
                    cash=float(summary_display.cash),
                    positions_value=float(positions_value),
                    equity=summary_display.equity,
                    realized_pnl=float(summary_display.realized_pnl),
                    unrealized_pnl=float(summary_display.unrealized_pnl),
                )
            self._log_system_event(
                event="price_missing",
                message="No prices available for cycle",
                level="warning",
                cycle_end=cycle_end.isoformat(),
            )
            if telemetry:
                self._publish_snapshot(cycle_end, status="no_prices")

        self._last_cycle_close = cycle_end

    def _floor_to_cycle(self, when: datetime) -> datetime:
        """
        Floor a timestamp to the nearest cycle boundary.
        For intraday cycles: aligns within the trading session.
        For long cycles (>= 1 day): aligns to market open on the target day.
        """
        cycle_minutes = self.config.cycle_minutes
        is_long_cycle = cycle_minutes >= 1440  # >= 1 day
        
        if is_long_cycle:
            # For long cycles, align to market open on the day
            session_start, _ = today_session_window(when)
            # Skip weekends
            while session_start.weekday() >= 5:
                session_start += timedelta(days=1)
                session_start, _ = today_session_window(session_start)
            return session_start
        
        # For intraday cycles, use session-based alignment
        if self._session_start is None:
            self._session_start, _ = today_session_window(when)
        if when <= self._session_start:
            return self._session_start
        elapsed = when - self._session_start
        steps = int(elapsed.total_seconds() // self._cycle_delta.total_seconds())
        return self._session_start + steps * self._cycle_delta

    def _format_local_time(self, when: datetime, fmt: str = "%H:%M %Z") -> str:
        local_tz = getattr(self, "_local_tz", None)
        if local_tz is None:
            try:
                local_tz = ZoneInfo(getattr(settings, "timezone", "Asia/Karachi"))
                self._local_tz = local_tz
            except Exception:
                local_tz = when.tzinfo
        if when.tzinfo is None and local_tz is not None:
            when = when.replace(tzinfo=local_tz)
        if local_tz is None:
            return when.strftime(fmt)
        return when.astimezone(local_tz).strftime(fmt)

    def _get_symbol_session_start(self, symbol: str) -> Optional[datetime]:
        return self._symbol_first_trade_time.get(symbol)

    def _record_symbol_first_trade(self, symbol: str, ts_utc: pd.Timestamp, price: float) -> None:
        if symbol in self._symbol_first_trade_time:
            return
        local_tz = getattr(self, "_local_tz", None)
        if local_tz is None:
            try:
                local_tz = ZoneInfo(getattr(settings, "timezone", "Asia/Karachi"))
                self._local_tz = local_tz
            except Exception:
                local_tz = None
        first_trade_local = ts_utc
        if local_tz is not None:
            first_trade_local = ts_utc.tz_convert(local_tz)
        first_trade_time = first_trade_local.to_pydatetime()
        self._symbol_first_trade_time[symbol] = first_trade_time
        self._symbol_open_price_today[symbol] = float(price)
        log_line(
            f"[{symbol}] First trade today: {first_trade_local.strftime('%H:%M:%S %Z')} @ {float(price):.2f}"
        )

    def _to_utc_timestamp(self, value: datetime | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError("Cannot convert NaT to UTC timestamp")
        if ts.tzinfo is None:
            local_tz = getattr(self, "_local_tz", None)
            if local_tz is None:
                try:
                    local_tz = ZoneInfo(getattr(settings, "timezone", "Asia/Karachi"))
                    self._local_tz = local_tz
                except Exception:
                    local_tz = None
            if local_tz is not None:
                ts = ts.replace(tzinfo=local_tz)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC")
        return ts

    def _log_daily_first_trades(self) -> None:
        if not self._symbol_first_trade_time:
            log_line("No first trades recorded yet today.")
            return
        log_line("Today's first trades (PKT):")
        for symbol in self.symbols:
            ts = self._symbol_first_trade_time.get(symbol)
            if not ts:
                log_line(f"[{symbol}] No trades yet today.")
                continue
            price = self._symbol_open_price_today.get(symbol)
            ts_label = self._format_local_time(ts)
            price_label = f"{price:.2f}" if price is not None else "n/a"
            log_line(f"[{symbol}] {ts_label} @ {price_label}")

    def _load_symbol_history(
        self,
        symbol: str,
        *,
        up_to: datetime,
        use_cache: bool,
        publish: bool,
        fallback_to_recent: bool = False,
    ) -> pd.DataFrame:
        """
        Load symbol history.
        
        Args:
            symbol: Symbol to load.
            up_to: Max timestamp to include.
            use_cache: whether to use cached data.
            publish: whether to publish to telemetry.
            fallback_to_recent: If True, allows returning data from the most recent 
                                trading day if "today" has no data (for off-hours analysis).
        
        CRITICAL: By default (fallback_to_recent=False), only returns trades from 
        today's market day (PKT timezone). Rejects all trades from yesterday or other days.
        """
        try:
            records = self.service.get_intraday(
                symbol,
                lookback_days=self.config.lookback_days,
                use_cache=use_cache,
            )
        except DataProviderError as exc:
            message = f"Intraday fetch failed for {symbol}: {exc}"
            log_line(message)
            self._log_system_event(
                event="data_provider_error",
                message=message,
                level="error",
                symbol=symbol,
            )
            return self._intraday_cache.get(symbol, _empty_intraday_frame())

        if not records:
            return _empty_intraday_frame()

        df = pd.DataFrame.from_records(records)
        if "timestamp" in df.columns and "ts" not in df.columns:
            df = df.rename(columns={"timestamp": "ts"})
        if "price" not in df.columns and "PRICE" in df.columns:
            df = df.rename(columns={"PRICE": "price"})
        if "volume" not in df.columns:
            if "VOLUME" in df.columns:
                df = df.rename(columns={"VOLUME": "volume"})
            else:
                df["volume"] = 0

        # Robust timestamp parsing:
        # - If ts is already ISO8601, pd.to_datetime will parse it
        # - If ts is numeric, prefer seconds; if too large, treat as ms
        ts_series = _as_series(df["ts"])
        if pd.api.types.is_numeric_dtype(ts_series):
            try:
                # Heuristic: unix ms are typically > 1e12; seconds < 1e11
                unit = "ms" if float(ts_series.iloc[0]) > 1e12 else "s"
                df["ts"] = pd.to_datetime(ts_series, unit=unit, utc=True, errors="coerce")
            except Exception:
                df["ts"] = pd.to_datetime(ts_series, errors="coerce", utc=True)
        else:
            df["ts"] = pd.to_datetime(ts_series, errors="coerce", utc=True)
        df = df.dropna(subset=["ts", "price"])
        volume_series = pd.to_numeric(df["volume"], errors="coerce")
        if not isinstance(volume_series, pd.Series):
            volume_series = pd.Series(volume_series)
        df["volume"] = volume_series.fillna(0)
        df = df.sort_values("ts")

        # CRITICAL: Strict today's date filtering (PKT timezone)
        # Convert timestamps to local market timezone and filter by today's date
        if self._today_date is None:
            self._today_date = now_tz().date()
        
        # Convert to local timezone for date comparison
        ts_series = _as_series(df["ts"])
        df["ts_local"] = ts_series.dt.tz_convert(self._local_tz)
        df["ts_date"] = _as_series(df["ts_local"]).dt.date
        
        # Determine target date: either today, or most recent if fallback enabled
        target_date = self._today_date
        
        # Check if we have data for today
        has_today_data = (_as_series(df["ts_date"]) == self._today_date).any()
        
        if not has_today_data and fallback_to_recent:
            # Fallback: Find the most recent date in the dataframe
            if not df.empty:
                unique_dates = df["ts_date"].unique()
                if len(unique_dates) > 0:
                    target_date = max(unique_dates)
                    log_line(f"[{symbol}] No data for today. Falling back to most recent data: {target_date}")
        
        # Filter to target date
        mask_target = _as_series(df["ts_date"]) == target_date
        df = cast(pd.DataFrame, df.loc[mask_target])
        
        # Clamp to current time (converted to UTC)
        # Note: If fallback used, we ignore 'up_to' time filtering to get full day
        if target_date == self._today_date:
            up_to_ts = self._to_utc_timestamp(up_to)
            mask_up_to = _as_series(df["ts"]) <= up_to_ts
            df = cast(pd.DataFrame, df.loc[mask_up_to])
        
        # Track today's first trade and first price per symbol
        if not df.empty and target_date == self._today_date:
            first_trade_row = df.iloc[0]
            first_trade_ts = first_trade_row["ts"]
            if pd.notna(first_trade_ts):
                self._record_symbol_first_trade(symbol, pd.Timestamp(first_trade_ts), float(first_trade_row["price"]))
        
        symbol_start = self._get_symbol_session_start(symbol)
        if symbol_start is not None:
            cutoff = self._to_utc_timestamp(symbol_start)
            mask_cutoff = _as_series(df["ts"]) >= cutoff
            df = cast(pd.DataFrame, df.loc[mask_cutoff])

        if not df.empty:
            last_seen = self._last_seen_ts.get(symbol)
            new_rows_df: pd.DataFrame
            if last_seen is None:
                new_rows_df = df
            else:
                mask_new = _as_series(df["ts"]) > last_seen
                new_rows_df = cast(pd.DataFrame, df[mask_new])
            
            # Real-time stream filtering: reject any trades not from today or before first trade
            if publish and not new_rows_df.empty:
                filtered_new_rows = []
                for _, row in new_rows_df.iterrows():
                    row_ts_value = row["ts"]
                    if pd.isna(row_ts_value):
                        continue
                    row_ts = pd.Timestamp(row_ts_value).to_pydatetime()
                    row_date = row_ts.astimezone(self._local_tz).date()
                    first_trade_time = self._symbol_first_trade_time.get(symbol)
                    
                    # Reject if not today
                    if row_date != self._today_date:
                        continue
                    
                    # Reject if before first trade of the day
                    if first_trade_time and row_ts < first_trade_time:
                        continue
                    
                    filtered_new_rows.append({
                        "ts": row_ts.isoformat(),
                        "price": float(row["price"]),
                        "volume": float(row.get("volume", 0)),
                    })
                
                if filtered_new_rows:
                    self._telemetry.publish_intraday(symbol, filtered_new_rows)
            
            ts_series = _as_series(df["ts"])
            last_ts = pd.Timestamp(ts_series.max())
            if pd.notna(last_ts):
                self._last_seen_ts[symbol] = last_ts
        
        # Drop temporary columns before returning
        df = df.drop(columns=["ts_local", "ts_date"], errors="ignore")
        return df.reset_index(drop=True)

    def _summarize_empty_batch(
        self,
        symbol: str,
        cycle_start: datetime,
        cycle_end: datetime,
    ) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "window_start": cycle_start.isoformat(),
            "window_end": cycle_end.isoformat(),
            "status": "no_data",
            "bias": "NEUTRAL",
            "delta_pct": 0.0,
            "total_volume": 0.0,
            "trades": 0,
            "vwap": None,
            "strategy_signal": "HOLD",
            "execution_side": "HOLD",
            "executed_qty": 0,
            "execution_price": None,
            "note": "no trades in window",
            "last_price": None,
        }

    def _summarize_batch(
        self,
        symbol: str,
        batch_df: pd.DataFrame,
        cycle_start: datetime,
        cycle_end: datetime,
    ) -> Dict[str, Any]:
        summary = {
            "symbol": symbol,
            "window_start": cycle_start.isoformat(),
            "window_end": cycle_end.isoformat(),
            "status": "ok",
            "bias": "NEUTRAL",
            "delta_pct": 0.0,
            "total_volume": 0.0,
            "trades": 0,
            "vwap": None,
            "strategy_signal": "HOLD",
            "execution_side": "HOLD",
            "executed_qty": 0,
            "execution_price": None,
            "note": "",
            "last_price": None,
        }

        if batch_df.empty:
            summary["status"] = "no_trades"
            return summary

        volume = float(batch_df["volume"].sum())
        notional = float((batch_df["price"] * batch_df["volume"]).sum())
        first_price = float(batch_df["price"].iloc[0])
        last_price = float(batch_df["price"].iloc[-1])
        vwap = notional / volume if volume > 0 else last_price
        prev_vwap = self._last_vwap.get(symbol)
        delta_pct = 0.0
        if prev_vwap:
            delta_pct = ((vwap - prev_vwap) / prev_vwap) * 100 if prev_vwap else 0.0
        else:
            if first_price:
                delta_pct = ((last_price - first_price) / first_price) * 100

        bias = "NEUTRAL"
        threshold = self.config.bias_threshold_pct
        if delta_pct > threshold:
            bias = "BUY"
        elif delta_pct < -threshold:
            bias = "SELL"

        self._last_vwap[symbol] = vwap

        summary.update(
            {
                "bias": bias,
                "delta_pct": delta_pct,
                "total_volume": volume,
                "trades": int(len(batch_df)),
                "vwap": vwap,
                "first_price": first_price,
                "last_price": last_price,
            }
        )
        return summary

    def _build_signal_snapshot(
        self,
        symbol: str,
        df_full: pd.DataFrame,
        cycle_end: datetime,
    ) -> Tuple[Optional[SignalSnapshot], Dict[str, Any]]:
        if df_full.empty:
            return None, self._summarize_empty_batch(symbol, cycle_end - self._cycle_delta, cycle_end)

        session_start = self._session_start or (cycle_end - self._cycle_delta)
        summary = self._summarize_batch(symbol, df_full, session_start, cycle_end)
        summary["window_start"] = session_start.isoformat()
        summary["window_end"] = cycle_end.isoformat()

        try:
            signal = self.strategy.generate_signal(symbol, df_full)
        except Exception as exc:  # pragma: no cover - strategy failure
            message = f"Strategy error for {symbol}: {exc}"
            log_line(message)
            self._log_system_event(
                event="strategy_error",
                message=message,
                level="error",
                symbol=symbol,
            )
            summary["note"] = "strategy_error"
            summary["execution_side"] = "HOLD"
            summary["strategy_signal"] = "HOLD"
            return None, summary

        summary["strategy_signal"] = signal
        execution_side = self._decide_execution(summary, signal)
        summary["execution_side"] = execution_side or "HOLD"
        if not execution_side:
            return None, summary

        last_price = float(df_full["price"].iloc[-1])
        vwap = float(summary.get("vwap") or last_price)
        target_qty = self._position_size_for(last_price)

        snapshot = SignalSnapshot(
            symbol=symbol,
            side=execution_side,
            strategy_signal=signal,
            bias=summary.get("bias", "NEUTRAL"),
            generated_at=cycle_end,
            signal_price=last_price,
            vwap=vwap,
            target_qty=target_qty,
            note=summary.get("note", ""),
            delta_pct=float(summary.get("delta_pct", 0.0)),
            batch_label=self._format_local_time(cycle_end),
        )
        summary["target_qty"] = target_qty
        summary["signal_price"] = last_price
        return snapshot, summary

    @staticmethod
    def _signals_equal(a: SignalSnapshot, b: SignalSnapshot) -> bool:
        return (
            a.side == b.side
            and a.strategy_signal == b.strategy_signal
            and a.target_qty == b.target_qty
        )

    def _decide_execution(self, summary: Dict[str, Any], signal: str) -> Optional[str]:
        bias = summary.get("bias", "NEUTRAL")
        if summary.get("total_volume", 0) <= 0:
            summary["note"] = "zero volume"
            return None
        if bias == "NEUTRAL":
            if signal in {"BUY", "SELL"}:
                summary["note"] = "strategy-driven trade"
                return signal
            summary["note"] = "neutral bias"
            return None
        if signal == bias:
            summary["note"] = "bias aligned with strategy"
            return bias
        if signal == "HOLD":
            summary["note"] = "bias-driven trade"
            return bias
        summary["note"] = f"bias overrode strategy ({signal})"
        return bias

    def _process_queued_signals(self, now: datetime) -> None:
        """
        Process queued signals when market opens.
        
        Args:
            now: Current datetime
        """
        if not self._queued_signals:
            return
        
        log_line(f"Processing {len(self._queued_signals)} queued signal(s) from previous session...")
        
        # Get current portfolio state
        portfolio_summary = self.portfolio.get_summary()
        summary_positions = {pos["symbol"]: pos["qty"] for pos in portfolio_summary.positions}
        cash_bucket = {"value": float(portfolio_summary.cash)}
        
        # Process each queued signal
        processed_signals = []
        for snapshot in self._queued_signals:
            # Log signal age for visibility
            signal_age = (now - snapshot.generated_at).total_seconds() / 3600  # hours
            if signal_age > 24:
                log_line(f"[{snapshot.symbol}] Warning: Signal is {signal_age:.1f} hours old (generated at {snapshot.generated_at.strftime('%Y-%m-%d %H:%M:%S')})")
            try:
                # Get current price for the symbol
                symbol = snapshot.symbol
                if symbol not in self._last_seen_price:
                    # Try to load current data
                    try:
                        df_full = self._load_symbol_history(symbol, up_to=now, use_cache=False, publish=False)
                        if not df_full.empty:
                            self._last_seen_price[symbol] = float(df_full["price"].iloc[-1])
                        else:
                            log_line(f"[{symbol}] No data available for queued signal. Skipping.")
                            continue
                    except Exception as exc:
                        log_line(f"[{symbol}] Error loading data for queued signal: {exc}. Skipping.")
                        continue
                
                # Update snapshot with current price if needed
                current_price = self._last_seen_price.get(symbol, snapshot.signal_price)
                snapshot.signal_price = current_price
                snapshot.generated_at = now
                
                # Create a summary dict for the queued signal
                summary = {
                    "symbol": symbol,
                    "window_start": now.isoformat(),
                    "window_end": now.isoformat(),
                    "status": "executing_queued",
                    "bias": snapshot.bias,
                    "delta_pct": 0.0,
                    "total_volume": 0.0,
                    "trades": 0,
                    "vwap": snapshot.vwap,
                    "strategy_signal": snapshot.strategy_signal,
                    "execution_side": snapshot.side or "HOLD",
                }
                
                # Execute the queued signal
                executed = self._execute_signal(
                    snapshot,
                    summary,
                    now,
                    summary_positions,
                    cash_bucket,
                )
                
                if executed.get("trade"):
                    log_line(f"[{symbol}] Queued {snapshot.side} signal executed successfully.")
                    self._log_system_event(
                        event="queued_signal_executed",
                        message=f"{symbol} queued {snapshot.side} signal executed",
                        level="info",
                        symbol=symbol,
                        side=snapshot.side,
                        qty=executed["summary"].get("executed_qty", 0),
                    )
                
                processed_signals.append(snapshot)
            except Exception as exc:
                log_line(f"Error processing queued signal for {snapshot.symbol}: {exc}. Continuing...")
                # Keep the signal in queue if there was an error
                continue
        
        # Remove processed signals from queue and mark as executed in storage
        for signal in processed_signals:
            if signal in self._queued_signals:
                self._queued_signals.remove(signal)
                # Mark as executed in persistent storage (if signal has signal_id attribute)
                signal_id = getattr(signal, 'signal_id', None)
                if signal_id:
                    self.signal_queue.mark_executed(signal_id, now)
        
        if processed_signals:
            log_line(f"Processed {len(processed_signals)} queued signal(s).")

    def _execute_signal(
        self,
        snapshot: SignalSnapshot,
        summary: Dict[str, Any],
        cycle_end: datetime,
        positions_snapshot: Dict[str, int],
        cash_bucket: Dict[str, float],
    ) -> Dict[str, Any]:
        symbol = snapshot.symbol
        execution_side = snapshot.side
        summary["execution_side"] = execution_side or "HOLD"
        if execution_side not in {"BUY", "SELL"}:
            summary["note"] = "no_signal"
            return {"summary": summary, "trade": None, "commission": 0.0, "slippage_bps": 0.0}

        signal_price = snapshot.signal_price
        vwap = snapshot.vwap or signal_price

        config_slippage = float(self.config.slippage_bps or 0.0)
        side_sign = 1 if execution_side == "BUY" else -1
        applied_slippage_bps = config_slippage * side_sign
        slippage_multiplier = 1 + applied_slippage_bps / 10_000
        executed_price = signal_price * slippage_multiplier
        slippage_value = abs(executed_price - signal_price)
        
        fees_per_share = float(self.config.commission_per_share or 0.0)
        fees_pct_notional = float(self.config.commission_pct_notional or 0.0)
        
        quantity = max(1, snapshot.target_qty)
        summary["requested_qty"] = quantity

        if execution_side == "BUY":
            available_cash = float(cash_bucket.get("value", 0.0))
            max_affordable = self.portfolio.calculate_affordable_quantity(
                price=signal_price,
                fees_per_share=fees_per_share,
                fees_pct_notional=fees_pct_notional,
                slippage_bps=applied_slippage_bps,
                available_cash=available_cash,
            )
            if max_affordable <= 0:
                summary["note"] = "insufficient_cash"
                return {"summary": summary, "trade": None, "commission": 0.0, "slippage_bps": 0.0}
            quantity = min(quantity, max_affordable)
            lot = max(1, self.config.min_lot)
            quantity = max(lot, (quantity // lot) * lot)
            if quantity <= 0:
                summary["note"] = "insufficient_cash"
                return {"summary": summary, "trade": None, "commission": 0.0, "slippage_bps": 0.0}
        else:
            available = positions_snapshot.get(symbol, 0)
            if available <= 0 and not self.config.allow_short:
                summary["note"] = "no_position_to_sell"
                return {"summary": summary, "trade": None, "commission": 0.0, "slippage_bps": 0.0}
            if available > 0:
                quantity = min(quantity, available)
                if quantity <= 0:
                    summary["note"] = "no_position_to_sell"
                    return {"summary": summary, "trade": None, "commission": 0.0, "slippage_bps": 0.0}

        notional = executed_price * quantity
        commission = quantity * fees_per_share + abs(notional) * fees_pct_notional

        try:
            trade = self.portfolio.record_trade(
                ts=cycle_end,
                symbol=symbol,
                side=execution_side,
                quantity=quantity,
                price=executed_price,
                fees=commission,
                slippage_bps=applied_slippage_bps,
            )
        except ValueError as exc:
            error_msg = str(exc)
            summary["note"] = error_msg.lower().replace(" ", "_")
            self._log_system_event(
                event="order_rejected",
                message=error_msg,
                level="warning",
                symbol=symbol,
                side=execution_side,
            )
            return {"summary": summary, "trade": None, "commission": 0.0, "slippage_bps": 0.0}

        trade_value = quantity * executed_price
        position_before = positions_snapshot.get(symbol, 0)
        cash_before = float(cash_bucket.get("value", 0.0))

        if execution_side == "BUY":
            positions_snapshot[symbol] = position_before + quantity
            cash_bucket["value"] = max(0.0, cash_before - trade_value - commission)
        else:
            positions_snapshot[symbol] = position_before - quantity
            cash_bucket["value"] = cash_before + trade_value - commission

        summary["executed_qty"] = quantity
        summary["execution_price"] = executed_price
        summary["execution_price_raw"] = signal_price
        summary["slippage_bps"] = applied_slippage_bps
        summary["slippage_value"] = slippage_value
        summary["commission"] = commission
        summary["realized_pnl"] = float(trade.pnl_realized)

        snapshot.executed_at = cycle_end

        trade_payload = self._serialize_trade(trade)
        trade_payload.update(
            {
                "bias": snapshot.bias,
                "strategy_signal": snapshot.strategy_signal,
                "vwap": snapshot.vwap,
                "volume": summary.get("total_volume", 0.0),
                "window_end": summary.get("window_end"),
                "commission": commission,
                "slippage_bps": applied_slippage_bps,
                "slippage_value": slippage_value,
                "price_raw": signal_price,
                "price_executed": executed_price,
            }
        )

        if self._paper_logger:
            summary_after = self.portfolio.get_summary()
            price_hints = dict(self._last_seen_price)
            price_hints[symbol] = float(executed_price)
            _, positions_value_after, unrealized_after = self._build_positions_snapshot(
                summary_after.positions,
                price_hints,
            )
            equity_after = float(summary_after.cash) + positions_value_after
            self._paper_logger.log_trade(
                timestamp=cycle_end,
                symbol=symbol,
                side=execution_side,
                quantity=quantity,
                price=executed_price,
                cost=trade_value,
                commission=commission,
                cash_before=cash_before,
                cash_after=float(cash_bucket.get("value", 0.0)),
                position_before=position_before,
                position_after=positions_snapshot.get(symbol, 0),
                realized_pnl=float(trade.pnl_realized),
                equity_after=float(equity_after),
                unrealized_after=float(unrealized_after),
                slippage_value=slippage_value,
            )

        return {
            "summary": summary,
            "trade": trade_payload,
            "commission": commission,
            "slippage_bps": applied_slippage_bps,
        }

    def _position_size_for(self, price: float) -> int:
        # Use capital_allocation if set, otherwise use position_notional
        if self.config.capital_allocation is not None:
            # Get current equity from portfolio
            summary = self.portfolio.get_summary()
            equity = summary.equity
            notional = equity * self.config.capital_allocation
        else:
            notional = self.config.position_notional
        
        lot = max(1, self.config.min_lot)
        qty = int(notional // max(price, 1e-6))
        return max(lot, (qty // lot) * lot)

    def _get_recent_trades(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent trades from portfolio service.
        
        Args:
            limit: Maximum number of recent trades to return
            
        Returns:
            List of trade dicts with timestamp, symbol, side, quantity, price, pnl_realized
        """
        try:
            # Get recent trades from portfolio service (already returns dicts)
            fetch_limit = limit or 10_000
            trades = self.portfolio.get_trades(limit=fetch_limit)
            if not trades:
                return []
            
            today = self._today_date or now_tz().date()
            recent_trades = []
            for trade in trades:
                ts_value = trade.get("ts", "")
                ts_dt = None
                if isinstance(ts_value, str):
                    try:
                        ts_dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                    except Exception:
                        ts_dt = None
                elif isinstance(ts_value, datetime):
                    ts_dt = ts_value
                
                if ts_dt is not None:
                    ts_local = ts_dt.astimezone(self._local_tz) if getattr(self, "_local_tz", None) else ts_dt
                    if ts_local.date() != today:
                        continue
                trade_dict_ts = ts_value
                if isinstance(ts_value, datetime):
                    trade_dict_ts = ts_value.isoformat()
                
                # Convert timestamp field from 'ts' to 'timestamp' for consistency
                trade_dict = {
                    "timestamp": trade_dict_ts,
                    "symbol": trade.get("symbol", ""),
                    "side": trade.get("side", ""),
                    "quantity": trade.get("quantity", 0),
                    "price": float(trade.get("price", 0.0)),
                    "pnl_realized": float(trade.get("pnl_realized", 0.0)),
                }
                if ts_dt is not None:
                    trade_dict["_sort_key"] = ts_dt
                recent_trades.append(trade_dict)
            
            # Sort trades chronologically so logs show cumulative order from session start
            recent_trades.sort(key=lambda t: t.get("_sort_key") or t.get("timestamp", ""))
            for trade in recent_trades:
                trade.pop("_sort_key", None)
            if limit:
                recent_trades = recent_trades[-limit:]
            return recent_trades
        except Exception as exc:
            # Gracefully handle any issues with getting trades
            log_line(f"Could not retrieve recent trades: {exc}")
            return []

    def _log_metrics(
        self,
        payload: Dict[str, Any],
        cycle_end: datetime,
        warm_start: bool = False,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Log cycle metrics - clean, professional format.
        """
        # Extract TradeMetrics from payload
        # payload["metrics"] is a dict from compute_portfolio_metrics with "metrics" key
        metrics_payload = payload.get("metrics", {})
        trade_metrics: TradeMetrics
        if isinstance(metrics_payload, dict) and "metrics" in metrics_payload:
            trade_metrics = metrics_payload["metrics"]
        elif isinstance(metrics_payload, TradeMetrics):
            trade_metrics = metrics_payload
        else:
            # Fallback: build TradeMetrics from dict-like input
            sharpe_raw = metrics_payload.get("sharpe_ratio")
            session_sharpe_raw = metrics_payload.get("session_sharpe_ratio")
            sortino_raw = metrics_payload.get("sortino_ratio")
            volatility_raw = metrics_payload.get("volatility_pct")
            win_loss_raw = metrics_payload.get("win_loss_ratio")
            sharpe_ratio_available = metrics_payload.get("sharpe_ratio_available")
            if sharpe_ratio_available is None:
                sharpe_ratio_available = sharpe_raw is not None
            session_sharpe_ratio_available = metrics_payload.get("session_sharpe_ratio_available")
            if session_sharpe_ratio_available is None:
                session_sharpe_ratio_available = session_sharpe_raw is not None
            sortino_ratio_available = metrics_payload.get("sortino_ratio_available")
            if sortino_ratio_available is None:
                sortino_ratio_available = sortino_raw is not None
            volatility_available = metrics_payload.get("volatility_available")
            if volatility_available is None:
                volatility_available = volatility_raw is not None
            win_loss_ratio_available = metrics_payload.get("win_loss_ratio_available")
            if win_loss_ratio_available is None:
                win_loss_ratio_available = win_loss_raw is not None
            trade_metrics = TradeMetrics(
                total_return_pct=float(metrics_payload.get("total_return_pct", 0.0)),
                daily_return_pct=float(metrics_payload.get("daily_return_pct", 0.0)),
                session_return_pct=float(metrics_payload.get("session_return_pct", metrics_payload.get("daily_return_pct", 0.0))),
                cumulative_return_pct=float(metrics_payload.get("cumulative_return_pct", metrics_payload.get("total_return_pct", 0.0))),
                sharpe_ratio=float(sharpe_raw or 0.0),
                session_sharpe_ratio=float(session_sharpe_raw or 0.0),
                sortino_ratio=float(sortino_raw or 0.0),
                max_drawdown_pct=float(metrics_payload.get("max_drawdown_pct", 0.0)),
                volatility_pct=float(volatility_raw or 0.0),
                win_loss_ratio=float(win_loss_raw or 0.0),
                exposure_pct=float(metrics_payload.get("exposure_pct", 0.0)),
                turnover_pct=float(metrics_payload.get("turnover_pct", 0.0)),
                cumulative_pnl=float(metrics_payload.get("cumulative_pnl", 0.0)),
                sharpe_ratio_available=bool(sharpe_ratio_available),
                session_sharpe_ratio_available=bool(session_sharpe_ratio_available),
                sortino_ratio_available=bool(sortino_ratio_available),
                volatility_available=bool(volatility_available),
                win_loss_ratio_available=bool(win_loss_ratio_available),
            )

        # Always get fresh portfolio summary to ensure accurate positions
        portfolio_summary = self.portfolio.get_summary()
        equity = float(payload.get("equity", portfolio_summary.equity))
        cash = float(payload.get("cash", portfolio_summary.cash))

        # Use positions from payload (with current price info) when available,
        # otherwise build them from the portfolio summary.
        positions = payload.get("positions", []) or []
        if not positions and portfolio_summary.positions:
            prices = payload.get("prices", {})
            for pos in portfolio_summary.positions:
                qty = pos.get("qty", 0)
                if qty <= 0:
                    continue
                symbol = pos["symbol"]
                avg_cost = pos.get("avg_cost", 0.0)
                current_price = prices.get(symbol, avg_cost)
                market_value = current_price * qty
                positions.append(
                    {
                        "symbol": symbol,
                        "qty": qty,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "market_value": market_value,
                    }
                )

        # Total PnL = realized + unrealized from portfolio summary (fallback to cumulative if available)
        total_pnl = float(trade_metrics.cumulative_pnl)
        if total_pnl == 0.0 and (portfolio_summary.realized_pnl or portfolio_summary.unrealized_pnl):
            total_pnl = float(portfolio_summary.realized_pnl) + float(portfolio_summary.unrealized_pnl)

        # Clean, professional logging format matching spec
        cycle_time_str = self._format_local_time(cycle_end)
        
        # Skip logging during warm-start (we'll show summary at end)
        if warm_start and not self.config.verbose_warm_start:
            return
        
        # Calculate total PnL (realized + unrealized)
        total_pnl = float(portfolio_summary.realized_pnl) + float(portfolio_summary.unrealized_pnl)
        
        # Determine bot status
        if not self.is_running:
            bot_status = "stop"
        elif warm_start:
            bot_status = "warm"
        else:
            bot_status = "run"

        if not self.config.detailed_logs:
            log_line(
                f"{bot_status.upper()} {cycle_time_str} | Cash {cash:,.0f} | Equity {equity:,.0f} | "
                f"PnL {total_pnl:+,.0f} | Session {trade_metrics.session_return_pct:.2f}% | "
                f"Total {trade_metrics.total_return_pct:.2f}% | Positions {len(positions)} | "
                f"Exposure {trade_metrics.exposure_pct:.2f}%"
            )
            return

        sharpe_text = f"{trade_metrics.sharpe_ratio:.2f}" if trade_metrics.sharpe_ratio_available else "-"
        sortino_text = (
            f"{trade_metrics.sortino_ratio:.2f}" if trade_metrics.sortino_ratio_available else "-"
        )
        volatility_text = (
            f"{trade_metrics.volatility_pct:.2f}%" if trade_metrics.volatility_available else "-"
        )
        win_loss_text = (
            f"{trade_metrics.win_loss_ratio:.2f}" if trade_metrics.win_loss_ratio_available else "-"
        )
        session_sharpe_text = (
            f"{trade_metrics.session_sharpe_ratio:.2f}"
            if trade_metrics.session_sharpe_ratio_available
            else "-"
        )

        log_line(
            f"Portfolio {trade_metrics.session_return_pct:.2f}% today | "
            f"Total {trade_metrics.total_return_pct:.2f}% | Sharpe {sharpe_text} | "
            f"Session Sharpe {session_sharpe_text} | "
            f"Sortino {sortino_text} | MaxDD {trade_metrics.max_drawdown_pct:.2f}% | "
            f"Vol {volatility_text} | Win/Loss {win_loss_text} | "
            f"Exposure {trade_metrics.exposure_pct:.2f}% | Turnover {trade_metrics.turnover_pct:.2f}%"
        )

        top_positions = positions[:3]
        if top_positions:
            pos_parts = [
                f"{p['symbol']} {p['qty']} @ {p['current_price']:.2f} ({p['market_value']:,.0f})"
                for p in top_positions
            ]
            log_line("Top positions: " + " | ".join(pos_parts))

        total_fees = float(payload.get("total_fees", 0.0))
        avg_slippage = float(payload.get("avg_slippage_bps", 0.0))
        if total_fees or avg_slippage:
            log_line(f"Cycle fees {total_fees:.2f} | Avg slippage {avg_slippage:.2f} bps")

        print("")

        # TOP SECTION: cash, equity, total PnL, open positions summary, bot status
        print("\n" + "-" * 70)
        print("PORTFOLIO SUMMARY")
        print("-" * 70)
        print(f"Cash:        {cash:,.0f} PKR")
        print(f"Equity:      {equity:,.0f} PKR")
        
        # Format total PnL with color
        pnl_str = f"{total_pnl:+,.0f} PKR"
        if total_pnl >= 0:
            pnl_str = self._colorize(pnl_str, "green")
        else:
            pnl_str = self._colorize(pnl_str, "red")
        print(f"Total PnL:   {pnl_str}")
        
        # Open positions summary
        if positions:
            pos_count = len(positions)
            total_value = sum(p.get("market_value", 0.0) for p in positions)
            print(f"Positions:   {pos_count} open | Total Value: {total_value:,.0f} PKR")
        else:
            print("Positions:   0 open")
        
        print(f"Status:      {bot_status.upper()}")
        print("-" * 70)
        
        # PER-SYMBOL DETAILS: quantity, avg cost, current price, unrealized PnL, value
        if positions:
            print("\nPOSITIONS")
            print("-" * 70)
            print(f"{'Symbol':<10} {'Qty':>8} {'Avg Cost':>12} {'Price':>12} {'Unreal PnL':>15} {'Value':>15}")
            print("-" * 70)
            
            for pos in positions:
                symbol = pos.get("symbol", "")
                qty = int(pos.get("qty", 0))
                if not symbol or qty <= 0:
                    continue
                avg_cost = float(pos.get("avg_cost", 0.0))
                current_price = float(pos.get("current_price", avg_cost))
                unrealized_pnl = (current_price - avg_cost) * qty
                market_value = float(pos.get("market_value", current_price * qty))
                
                # Format unrealized PnL with color
                pnl_display = f"{unrealized_pnl:+,.0f}"
                if unrealized_pnl >= 0:
                    pnl_display = self._colorize(pnl_display, "green")
                else:
                    pnl_display = self._colorize(pnl_display, "red")
                
                print(f"{symbol:<10} {qty:>8} {avg_cost:>12.2f} {current_price:>12.2f} {pnl_display:>15} {market_value:>15,.0f}")
            
            print("-" * 70)
        else:
            print("\nPOSITIONS: None")
        
        # BOTTOM ROLLING FEED: last few trades, strategy messages
        recent_trades = recent_trades if recent_trades is not None else self._get_recent_trades()
        if recent_trades:
            print("\nRECENT TRADES (since open)")
            print("-" * 70)
            for trade in recent_trades:
                ts_value = trade.get("timestamp") or trade.get("ts")
                ts_str = str(ts_value) if ts_value is not None else "-"
                parsed_ts: Optional[datetime] = None
                if isinstance(ts_value, str):
                    try:
                        parsed_ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                    except Exception:
                        parsed_ts = None
                elif isinstance(ts_value, datetime):
                    parsed_ts = ts_value
                
                if parsed_ts is not None:
                    ts_str = self._format_local_time(parsed_ts)
                
                symbol = trade.get("symbol", "")
                side = trade.get("side", "")
                quantity = trade.get("quantity", 0)
                price = trade.get("price", 0.0)
                pnl = trade.get("pnl_realized", 0.0)
                
                # Format side with color
                side_display = side
                if side == "BUY":
                    side_display = self._colorize(side, "green")
                elif side == "SELL":
                    side_display = self._colorize(side, "red")
                
                # Format PnL with color
                pnl_display = f"{pnl:+,.0f}"
                if pnl >= 0:
                    pnl_display = self._colorize(pnl_display, "green")
                else:
                    pnl_display = self._colorize(pnl_display, "red")
                
                print(f"{ts_str} | {side_display:<4} {quantity:>4} {symbol:<8} @ {price:>8.2f} | PnL: {pnl_display}")
            print("-" * 70)
        
        print()  # Blank line for readability

    def _print_hourly_summary(self, now: datetime) -> None:
        """Print hourly summary of portfolio performance."""
        if not self.metrics_history:
            return
        last_metrics = self.metrics_history[-1]
        metrics: TradeMetrics = last_metrics["metrics"]
        summary = self.portfolio.get_summary()
        local_time = now.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else now
        
        # Calculate best/worst performing symbol
        positions = last_metrics.get("positions", [])
        best_symbol = None
        worst_symbol = None
        best_pnl = float('-inf')
        worst_pnl = float('inf')
        
        for pos in positions:
            unrealized_pnl = pos.get("unrealized_pnl", 0.0)
            symbol = pos.get("symbol", "")
            if unrealized_pnl > best_pnl:
                best_pnl = unrealized_pnl
                best_symbol = symbol
            if unrealized_pnl < worst_pnl:
                worst_pnl = unrealized_pnl
                worst_symbol = symbol
        
        # Get trades executed this hour
        hour_start = now - timedelta(hours=1)
        all_trades = self.portfolio.get_trades(limit=10000)
        trades_this_hour = []
        for trade in all_trades:
            try:
                trade_ts_str = trade.get("ts", "")
                if isinstance(trade_ts_str, str):
                    trade_ts = datetime.fromisoformat(trade_ts_str.replace("Z", "+00:00"))
                    if trade_ts >= hour_start:
                        trades_this_hour.append(trade)
            except:
                continue
        
        # Calculate win rate from trades this hour
        winning_trades = 0
        losing_trades = 0
        for trade in trades_this_hour:
            pnl = float(trade.get("pnl_realized", 0.0))
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1
        
        total_trades_with_pnl = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades_with_pnl * 100) if total_trades_with_pnl > 0 else 0.0
        
        # Format summary
        sharpe_text = f"{metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio_available else "-"
        sortino_text = f"{metrics.sortino_ratio:.2f}" if metrics.sortino_ratio_available else "-"
        
        time_str = self._format_local_time(now)
        heartbeat_str = self._format_local_time(now, "%Y-%m-%d %H:%M:%S %Z")
        
        print("")
        print("-" * 70)
        log_line(f"HOURLY SUMMARY at {time_str}")
        print("-" * 70)
        log_line(f"Equity:              {summary.equity:,.0f} PKR")
        log_line(f"Total PnL:           {metrics.cumulative_pnl:+,.0f} PKR")
        
        # Best/worst performing symbol
        if best_symbol:
            best_pnl_str = f"{best_symbol} ({best_pnl:+,.0f} PKR)"
            if best_pnl >= 0:
                best_pnl_str = self._colorize(best_pnl_str, "green")
            log_line(f"Best Performer:      {best_pnl_str}")
        else:
            log_line("Best Performer:      None")
        
        if worst_symbol:
            worst_pnl_str = f"{worst_symbol} ({worst_pnl:+,.0f} PKR)"
            if worst_pnl < 0:
                worst_pnl_str = self._colorize(worst_pnl_str, "red")
            log_line(f"Worst Performer:     {worst_pnl_str}")
        else:
            log_line("Worst Performer:     None")
        
        log_line(f"Trades This Hour:    {len(trades_this_hour)}")
        log_line(f"Win Rate:            {win_rate:.1f}% ({winning_trades}W/{losing_trades}L)")
        log_line(f"Drawdown:            {metrics.max_drawdown_pct:.2f}%")
        log_line(f"Last Heartbeat:       {heartbeat_str}")
        print("-" * 70)
        print("")
    
    def _print_session_summary(self) -> None:
        """Print final session summary with all key metrics."""
        if not self.metrics_history:
            return
        
        last_metrics = self.metrics_history[-1]
        metrics: TradeMetrics = last_metrics["metrics"]
        summary = self.portfolio.get_summary()
        
        sharpe_text = f"{metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio_available else "-"
        sortino_text = f"{metrics.sortino_ratio:.2f}" if metrics.sortino_ratio_available else "-"
        
        # Count total trades (scoped to live session)
        total_trades = max(self._current_trade_count() - self._trade_count_baseline, 0)
        total_cycles = len(self.metrics_history)
        
        # Calculate final return correctly
        initial_equity = self._session_start_equity or self.portfolio.initial_cash
        final_return_pct = ((summary.equity - initial_equity) / initial_equity * 100) if initial_equity > 0 else 0.0
        
        return_str = f"{final_return_pct:+.2f}%"
        if final_return_pct >= 0:
            return_str = self._colorize(return_str, "green")
        else:
            return_str = self._colorize(return_str, "red")
        
        print("\n" + "-" * 70)
        print("SESSION SUMMARY")
        print("-" * 70)
        print(f"Total Cycles: {total_cycles} ({self._warm_start_cycles} warm-start + {total_cycles - self._warm_start_cycles} live)")
        print(f"Joined Mid-Session: {'Yes' if self._joined_mid_session else 'No'}")
        print(f"Total Trades: {total_trades}")
        print(f"Final Equity: {summary.equity:,.0f} PKR")
        print(f"Final Cash: {summary.cash:,.0f} PKR")
        print(f"Total Return: {return_str}")
        print(f"Sharpe Ratio: {sharpe_text}")
        print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        
        # Show all positions
        positions = last_metrics.get("positions", [])
        if positions:
            pos_parts = [
                f"{p['symbol']} {p['qty']} @ {p['current_price']:.2f}"
                for p in positions
            ]
            log_line(f"Positions: {' | '.join(pos_parts)}")
        else:
            log_line("Positions: None")
        
        # Show CSV file locations
        log_line(f"Metrics CSV: {self.csv_writer.metrics_path}")
        log_line(f"Trades CSV: {self.csv_writer.trades_path}")
        positions_csv = self.csv_writer.metrics_path.parent / "pytrader_positions.csv"
        if positions_csv.exists():
            log_line(f"Positions CSV: {positions_csv}")
        print("")

    async def _sleep_until_next_cycle(self) -> None:
        if self._sleep_seconds is not None:
            await asyncio.sleep(self._sleep_seconds)
            return
        now = now_tz()
        cycle_minutes = self.config.cycle_minutes
        is_long_cycle = cycle_minutes >= 1440  # >= 1 day
        
        if not is_market_open(now):
            # Market is closed, wait until next market open
            next_open = next_market_open(now)
            delay = max(60.0, (next_open - now).total_seconds())
            local_time = now.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else now
            next_open_local = next_open.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else next_open
            current_label = self._format_local_time(local_time)
            next_label = self._format_local_time(next_open_local)
            log_line(f"Market closed at {current_label}. Waiting until next session at {next_label}...")
            await asyncio.sleep(min(delay, 3600.0))  # Sleep max 1 hour at a time
            return
        
        next_cycle = self._next_cycle_ts(now)
        delay = max(5.0, (next_cycle - now).total_seconds())
        
        if delay > 60:
            local_time = now.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else now
            next_cycle_local = next_cycle.astimezone(self._local_tz) if hasattr(self, "_local_tz") and self._local_tz else next_cycle
            next_label = self._format_local_time(next_cycle_local)
            
            # Format cycle description for long cycles
            if is_long_cycle:
                days = cycle_minutes // 1440
                if days == 1:
                    cycle_desc = "daily"
                elif days == 7:
                    cycle_desc = "weekly"
                elif days >= 28:
                    cycle_desc = f"monthly (~{days} days)"
                else:
                    cycle_desc = f"{days}-day"
                log_line(f"Waiting for next {cycle_desc} cycle at {next_label}...")
            else:
                log_line(f"Waiting for next {cycle_minutes}-minute batch at {next_label}...")
        
        await asyncio.sleep(delay)

    def _next_cycle_ts(self, now: datetime) -> datetime:
        """
        Calculate the next cycle timestamp, aligned to cycle boundaries.
        Supports cycles from 15 minutes to 30 days (weekly/monthly trading).
        
        For intraday cycles (< 1 day): Aligns to 15-minute boundaries within trading hours.
        For daily/weekly/monthly cycles: Aligns to market open on the target day, skipping weekends.
        """
        cycle_minutes = self.config.cycle_minutes
        is_long_cycle = cycle_minutes >= 1440  # >= 1 day (1440 minutes)
        
        if self._session_start is None or self._session_end is None:
            self._session_start, session_market_close = today_session_window(now)
            # Extend session end by 15 minutes to allow processing final batch
            # Market closes at 3:30 PM, but paper trading continues until 3:45 PM
            self._session_end = session_market_close + timedelta(minutes=15)
        
        if self._last_cycle_close is None:
            # First cycle - calculate next cycle from now
            if is_long_cycle:
                # For long cycles, align to market open on the target day
                days_ahead = cycle_minutes // 1440
                target_date = now.date() + timedelta(days=days_ahead)
                # Skip weekends
                while target_date.weekday() >= 5:
                    target_date += timedelta(days=1)
                # Create datetime for target date at market open
                open_hour = settings.market_hours.open_hour
                open_minute = settings.market_hours.open_minute
                target_dt = datetime.combine(target_date, dt_time(open_hour, open_minute))
                if now.tzinfo:
                    target_dt = target_dt.replace(tzinfo=now.tzinfo)
                session_start, _ = today_session_window(target_dt)
                return session_start
            else:
                # For intraday cycles, align to next cycle boundary
                return self._floor_to_cycle(now) + self._cycle_delta
        
        # Next cycle is last cycle + cycle_delta
        target = self._last_cycle_close + self._cycle_delta
        
        if is_long_cycle:
            # For long cycles, ensure target is on a trading day (skip weekends)
            while target.weekday() >= 5:
                target += timedelta(days=1)
            # Align to market open on that day
            session_start, _ = today_session_window(target)
            if target < session_start:
                target = session_start
            elif target > session_start.replace(hour=15, minute=30):
                # If past market close, move to next trading day
                target = next_market_open(target)
            return target
        
        # For intraday cycles: handle session boundaries
        if self._session_end and target >= self._session_end:
            # Cycle extends past session end - wait until next market open
            return next_market_open(now)
        
        # If target is in the past (shouldn't happen, but handle gracefully)
        if target <= now:
            # Align to next cycle boundary from now
            next_boundary = self._floor_to_cycle(now) + self._cycle_delta
            # If next boundary is past session end, move to next market open
            if self._session_end and next_boundary >= self._session_end:
                return next_market_open(now)
            return max(next_boundary, now + timedelta(seconds=5))
        
        return target

    def _serialize_trade(self, trade: Any) -> Dict[str, Any]:
        return {
            "id": trade.id,
            "timestamp": trade.ts.isoformat() if isinstance(trade.ts, datetime) else None,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "price": float(trade.price),
            "cost": float(trade.cost),
            "pnl_realized": float(trade.pnl_realized),
            "commission": float(getattr(trade, "fees", 0.0)),
            "slippage_bps": float(getattr(trade, "slippage_bps", 0.0)),
            "source": self.bot_id,
            "strategy_name": self._strategy_name,
            "execution_type": "average_cost",
        }

    def _current_trade_count(self) -> int:
        try:
            return len(self.portfolio.get_trades(limit=10_000))
        except Exception:
            return 0

    def _publish_cycle(
        self,
        ts: datetime,
        payload: Dict[str, Any],
        prices: Dict[str, float],
        trades: List[Dict[str, Any]],
        batches: List[Dict[str, Any]],
        *,
        total_fees: float,
        avg_slippage_bps: float,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        summary_equity = float(payload["equity"])
        summary_cash = float(payload["cash"])
        positions_value = float(payload["positions_value"])
        metrics: TradeMetrics = payload["metrics"]
        report = CycleReport(
            bot_id=self.bot_id,
            timestamp=ts,
            status="ok",
            equity=summary_equity,
            cash=summary_cash,
            positions_value=positions_value,
            metrics=metrics,
            positions=payload["positions"],
            trades=trades,
            prices={k: float(v) for k, v in prices.items()},
            batches=batches,
            total_fees=total_fees,
            avg_slippage_bps=avg_slippage_bps,
            recent_trades=recent_trades or [],
        )
        log_line(f"[{self.bot_id}] Publishing cycle report: equity={summary_equity:.2f}, positions={len(payload['positions'])}")
        self._telemetry.publish(report)

    def _determine_strategy_name(self, strategy: Any) -> str:
        if hasattr(strategy, "name") and isinstance(getattr(strategy, "name"), str):
            return getattr(strategy, "name")
        strategy_cls = getattr(strategy, "__class__", None)
        if strategy_cls and hasattr(strategy_cls, "__name__"):
            return strategy_cls.__name__
        if isinstance(strategy, str):
            return strategy
        return "custom_strategy"

    def _publish_snapshot(self, ts: datetime, status: str) -> None:
        summary = self.portfolio.get_summary()
        recent_trades = self._get_recent_trades()
        report = CycleReport(
            bot_id=self.bot_id,
            timestamp=ts,
            status=status,
            equity=float(summary.equity),
            cash=float(summary.cash),
            positions_value=max(float(summary.equity) - float(summary.cash), 0.0),
            metrics=TradeMetrics(
                total_return_pct=0.0,
                daily_return_pct=0.0,
                session_return_pct=0.0,
                cumulative_return_pct=0.0,
                sharpe_ratio=0.0,
                session_sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown_pct=0.0,
                volatility_pct=0.0,
                win_loss_ratio=0.0,
                exposure_pct=0.0,
                turnover_pct=0.0,
                cumulative_pnl=float(summary.realized_pnl + summary.unrealized_pnl),
                sharpe_ratio_available=False,
                session_sharpe_ratio_available=False,
                sortino_ratio_available=False,
                volatility_available=False,
                win_loss_ratio_available=False,
            ),
            positions=summary.positions,
            trades=[],
            prices={},
            batches=[],
            total_fees=0.0,
            avg_slippage_bps=0.0,
            recent_trades=recent_trades or [],
        )
        self._telemetry.publish(report)

    def _build_telemetry(
        self,
        telemetry: Optional[CompositeTelemetry],
        in_process_push: Optional[Any],
    ) -> tuple[CompositeTelemetry, Optional[TelemetryClient]]:
        if telemetry:
            return telemetry, None

        sinks: List[Any] = []
        sinks.append(FileTelemetry(self.config.log_dir, self.bot_id))
        backend_client: Optional[TelemetryClient] = None

        if in_process_push is not None:
            log_line(f"[{self.bot_id}] Using CallbackTelemetry (dashboard mode)")
            sinks.append(CallbackTelemetry(in_process_push))
        elif self.config.backend_url and self.config.api_token:
            log_line(f"[{self.bot_id}] ✓ Initializing BackendRelayTelemetry")
            log_line(f"[{self.bot_id}]   Backend URL: {self.config.backend_url}")
            log_line(f"[{self.bot_id}]   Token: {self.config.api_token[:10]}...")
            client = TelemetryClient(
                bot_id=self.bot_id,
                api_token=self.config.api_token,
                backend_url=self.config.backend_url,
                bot_label=self.bot_id,
            )
            sinks.append(BackendRelayTelemetry(client))
            backend_client = client
            log_line(f"[{self.bot_id}] ✓ Backend telemetry ready - will push to {self.config.backend_url}")
        else:
            log_line(f"[{self.bot_id}] ⚠ WARNING: No backend telemetry (backend_url={self.config.backend_url}, token={bool(self.config.api_token)})")

        if not sinks:
            log_line(f"[{self.bot_id}] Using NullTelemetry (no sinks configured)")
            sinks.append(NullTelemetry())

        return CompositeTelemetry(sinks), backend_client

    def _build_backend_log_hook(self) -> Optional[Callable[[str, str, Dict[str, Any], str], None]]:
        client = self._backend_client
        if not client:
            return None

        def _hook(level: str, message: str, context: Dict[str, Any], stream: str) -> None:
            try:
                client.log_event(
                    message=message,
                    level=level,
                    context=context,
                    stream=stream,
                )
            except Exception as exc:
                log_line(f"[{self.bot_id}] Failed to push {stream} log: {exc}")

        return _hook

    def _log_system_event(self, event: str, message: str, level: str = "info", **context: Any) -> None:
        if hasattr(self, "_paper_logger") and self._paper_logger:
            self._paper_logger.log_system(
                event=event,
                message=message,
                level=level,
                context=context or None,
            )

    def _restore_positions_from_account(self, positions: Dict[str, int]) -> None:
        """Restore positions from account state JSON.
        
        Note: Since account JSON only stores quantities (not avg_cost), we use current
        market price as avg_cost. This is approximate but allows positions to be restored.
        For accurate avg_cost, positions should be restored from trade history or stored
        separately in account JSON.
        """
        if not positions:
            return
        
        from ..portfolio.service import Position, Session
        
        # Fetch current prices for all symbols to use as approximate avg_cost
        # If prices unavailable, positions will be restored with 0 avg_cost (will be updated on first trade)
        prices = {}
        for symbol in positions.keys():
            try:
                # Try to get latest price from cache or data provider
                if symbol in self._intraday_cache and not self._intraday_cache[symbol].empty:
                    df = self._intraday_cache[symbol]
                    prices[symbol] = float(df["close"].iloc[-1])
                else:
                    # If no cached data, use 0.0 - will be updated on first cycle
                    prices[symbol] = 0.0
            except Exception:
                prices[symbol] = 0.0
        
        # Restore positions to portfolio
        with Session(self.portfolio.engine, expire_on_commit=False) as session:
            for symbol, quantity in positions.items():
                if quantity == 0:
                    continue
                
                pos = session.get(Position, symbol)
                if pos is None:
                    # Use current price as approximate avg_cost (or 0.0 if unavailable)
                    avg_cost = prices.get(symbol, 0.0)
                    pos = Position(symbol=symbol, quantity=quantity, avg_cost=avg_cost)
                    session.add(pos)
                    log_line(f"Restored position: {symbol} x {quantity} @ {avg_cost:.2f} (approx)")
                else:
                    # Position exists - update quantity and avg_cost
                    # If avg_cost is 0, use current price
                    if pos.avg_cost == 0.0 and prices.get(symbol, 0.0) > 0:
                        pos.avg_cost = prices[symbol]
                    pos.quantity = quantity
                    log_line(f"Updated position: {symbol} x {quantity} @ {pos.avg_cost:.2f}")
            session.commit()
        
        if positions:
            log_line(f"Restored {len(positions)} positions from account state")

    def get_today_first_price(self, symbol: str) -> Optional[float]:
        """
        Get today's first trade price for a symbol.
        
        This is the opening price used for PnL normalizations and summary stats.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            First price of the day, or None if no trades yet today
        """
        return self._symbol_open_price_today.get(symbol)
    
    def get_today_first_trade_time(self, symbol: str) -> Optional[datetime]:
        """
        Get today's first trade timestamp for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            First trade timestamp of the day, or None if no trades yet today
        """
        return self._symbol_first_trade_time.get(symbol)
    
    def _load_queued_signals_from_storage(self) -> None:
        """Load queued signals from persistent storage."""
        try:
            stored_signals = self.signal_queue.get_queued_signals(status='queued')
            if not stored_signals:
                return
            
            log_line(f"Loading {len(stored_signals)} queued signal(s) from persistent storage...")
            
            for sig_data in stored_signals:
                # Reconstruct SignalSnapshot from stored data
                snapshot = SignalSnapshot(
                    symbol=sig_data['symbol'],
                    side=sig_data['side'],
                    strategy_signal=sig_data['strategy_signal'],
                    bias=sig_data['bias'],
                    generated_at=sig_data['generated_at'],
                    signal_price=sig_data['signal_price'],
                    vwap=sig_data['vwap'],
                    target_qty=sig_data['target_qty'],
                    note=sig_data['note'],
                    delta_pct=sig_data['delta_pct'],
                    batch_label=sig_data.get('batch_label'),
                )
                # Store signal_id for later reference
                snapshot.signal_id = sig_data['signal_id']  # type: ignore
                self._queued_signals.append(snapshot)
            
            if stored_signals:
                log_line(f"✅ Loaded {len(stored_signals)} queued signal(s) from storage")
        except Exception as exc:
            log_line(f"⚠️ Failed to load queued signals from storage: {exc}")
    
    def _save_signal_to_queue(self, snapshot: SignalSnapshot) -> None:
        """Save signal to persistent storage."""
        try:
            signal_id = self.signal_queue.enqueue_signal(
                symbol=snapshot.symbol,
                side=snapshot.side,
                strategy_signal=snapshot.strategy_signal,
                bias=snapshot.bias,
                generated_at=snapshot.generated_at,
                signal_price=snapshot.signal_price,
                vwap=snapshot.vwap,
                target_qty=snapshot.target_qty,
                note=snapshot.note,
                delta_pct=snapshot.delta_pct,
                batch_label=snapshot.batch_label,
            )
            # Store signal_id in snapshot for later reference
            snapshot.signal_id = signal_id  # type: ignore
        except Exception as exc:
            log_line(f"⚠️ Failed to save signal to persistent storage: {exc}")
    
__all__ = ["TradingEngine", "EngineConfig", "TradeMetrics", "SignalSnapshot"]


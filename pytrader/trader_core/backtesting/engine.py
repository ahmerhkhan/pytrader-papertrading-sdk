"""
Backtesting engine aligned with the new trader_core architecture.

BACKEND-ONLY: This engine is for backend internal use only.
SDK client code should use PyTrader client instead.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ...data.pypsx_service import PyPSXService
from ...data.csv_provider import CSVDataProvider, CSVColumnMapping
from ..portfolio.metrics import (
    TradeMetrics,
    compute_portfolio_metrics,
)
from ..portfolio.service import PortfolioService, PortfolioSummary
from ...utils.logger import log_line
from ...utils.currency import format_pkr


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    position_notional: float = 100_000.0
    capital_allocation: Optional[float] = None  # Percentage of equity per trade (e.g., 0.2 = 20%)
    min_lot: int = 1
    interval: str = "1d"
    db_path: Optional[Path] = None
    slippage_bps: float = 0.0
    commission_per_share: float = 0.0
    commission_pct_notional: float = 0.0015  # Default: 0.15% per share (percentage-based)
    ignore_cash: bool = False  # If True, bypass cash checks (sets cash to very large number)
    allow_short: bool = False  # Allow short selling (selling without existing position)
    max_leverage: Optional[float] = None
    max_position_pct: Optional[float] = None
    max_positions: Optional[int] = None
    risk_per_trade_pct: Optional[float] = None
    min_volume_threshold: Optional[float] = None
    skip_illiquid_days: bool = False
    # CSV support
    csv_path: Optional[Union[str, Path]] = None  # Path to CSV file for data source
    csv_column_mapping: Optional[Dict[str, str]] = None  # Column mapping dict (e.g., {"timestamp": "ts", "price": "close"})
    csv_delimiter: str = ","  # CSV delimiter
    csv_encoding: str = "utf-8"  # CSV file encoding


class BacktestEngine:
    """
    Historical simulation engine that reuses the live trading portfolio stack.
    
    BACKEND-ONLY: This engine is for backend internal use only.
    SDK client code should use PyTrader client instead.
    """

    def __init__(
        self,
        symbols: List[str],
        strategy: Any,
        *,
        config: Optional[BacktestConfig] = None,
        data_service: Optional[PyPSXService] = None,
        bot_id: str = "backtest",
    ) -> None:
        # Warn if used from SDK client code (backend should pass data_service)
        if data_service is None and not (config and config.csv_path):
            warnings.warn(
                "BacktestEngine is backend-only. SDK client code should use PyTrader client instead: "
                "client = PyTrader(api_token='...'); client.backtest(...)",
                DeprecationWarning,
                stacklevel=2,
            )
        self.symbols = [s.upper().strip() for s in symbols]
        self.strategy = strategy
        self.config = config or BacktestConfig()
        
        # Initialize data service (CSV or API)
        if self.config.csv_path:
            # Use CSV data provider
            column_mapping = None
            if self.config.csv_column_mapping:
                column_mapping = CSVColumnMapping(**self.config.csv_column_mapping)
            self.service = CSVDataProvider(
                csv_path=self.config.csv_path,
                column_mapping=column_mapping,
                delimiter=self.config.csv_delimiter,
                encoding=self.config.csv_encoding,
            )
        else:
            # Use API service
            self.service = data_service or PyPSXService()
        
        self.bot_id = bot_id
        self.portfolio = self._init_portfolio()
        self.metrics: Optional[Dict[str, Any]] = None

    def _init_portfolio(self) -> PortfolioService:
        # Use in-memory database for backtests to speed up initialization
        # This avoids file I/O and makes backtests faster
        if self.config.db_path is not None:
            db_uri = f"sqlite:///{self.config.db_path}"
        else:
            # Use in-memory DB for faster backtests (no disk I/O)
            # This ensures backtest portfolios are completely independent from paper/live trading
            db_uri = "sqlite:///:memory:"
        
        # Determine starting cash: use very large number if ignore_cash=True, otherwise use config value
        starting_cash = self.config.initial_cash
        if self.config.ignore_cash:
            starting_cash = 1e15  # Very large number to effectively bypass cash checks
        
        portfolio = PortfolioService(
            db_url=db_uri, 
            initial_cash=starting_cash,
            unlimited_cash=self.config.ignore_cash,  # Only enable if ignore_cash is True
            allow_short=self.config.allow_short,
        )
        summary = portfolio.get_summary()
        delta = float(starting_cash) - float(summary.cash)
        if abs(delta) > 1e-6:
            portfolio.apply_cash_adjustment(delta)
        return portfolio

    def run(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
        # Log appropriate message based on cash mode
        if self.config.ignore_cash:
            log_line(f"[{self.bot_id}] Starting backtest with cash checks disabled (ignore_cash=True)...")
        else:
            log_line(f"[{self.bot_id}] Starting backtest with initial capital PKR {format_pkr(self.config.initial_cash):,.0f}...")
        log_line(f"[{self.bot_id}] Loading historical data...")
        history_map = self._load_history(start=start, end=end)
        timeline = self._build_timeline(history_map)
        if not timeline:
            raise ValueError("No historical data available for requested symbols/date range.")

        log_line(f"[{self.bot_id}] Backtest running for {len(timeline)} data points.")
        total_fees = 0.0
        slippage_weighted_sum = 0.0
        slippage_qty_total = 0
        skipped_trades = 0
        partial_fills = 0
        skipped_reasons: Dict[str, int] = {}
        # Track skipped trades per symbol for aggregated logging
        skipped_trades_by_symbol: Dict[str, int] = defaultdict(int)
        for ts in timeline:
            latest_prices: Dict[str, float] = {}
            min_history = getattr(self.strategy, "min_history_bars", 0)
            for symbol, df in history_map.items():
                subset = df[df["ts"] <= ts]
                if subset.empty:
                    continue
                if min_history and len(subset) < min_history:
                    continue
                latest_row = subset.iloc[-1]
                price = float(latest_row["close"])
                latest_prices[symbol] = price
                latest_volume = float(latest_row.get("volume", 0.0) or 0.0)
                if (
                    self.config.skip_illiquid_days
                    and self.config.min_volume_threshold is not None
                    and latest_volume < self.config.min_volume_threshold
                ):
                    skipped_trades += 1
                    skipped_trades_by_symbol[symbol] += 1
                    reason = "volume_filter"
                    skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                    continue

                try:
                    signal = self.strategy.generate_signal(symbol, subset)
                except Exception as exc:  # pragma: no cover - strategy failure
                    message = f"[{self.bot_id}] Strategy error for {symbol}: {exc}"
                    log_line(message)
                    raise RuntimeError(message) from exc

                if signal == "HOLD":
                    continue

                summary_snapshot = self.portfolio.get_summary()

                # Calculate target quantity based on position sizing
                target_qty = self._position_size_for(price, summary_snapshot)
                if target_qty <= 0:
                    continue

                if signal == "SELL":
                    summary_positions = summary_snapshot.positions
                    positions: Dict[str, int] = {
                        str(p.get("symbol", "")).upper(): int(p.get("qty", 0) or 0)
                        for p in summary_positions
                    }
                    current_qty = positions.get(symbol.upper(), 0)
                    if current_qty <= 0:
                        # Check if shorting is allowed
                        if not self.config.allow_short:
                            continue
                        # Shorting allowed - use full target quantity
                        qty = target_qty
                    else:
                        # Have position - sell up to available
                        qty = min(target_qty, current_qty)
                    # Calculate slippage and fees for SELL
                    config_slippage = float(self.config.slippage_bps or 0.0)
                    applied_slippage_bps = -config_slippage  # Negative for SELL reduces proceeds
                    adjusted_price = price * (1 + applied_slippage_bps / 10_000)
                    notional = adjusted_price * qty
                    fees_per_share = float(self.config.commission_per_share or 0.0)
                    fees_pct_notional = float(self.config.commission_pct_notional or 0.0)
                    commission = (
                        qty * fees_per_share
                        + abs(notional) * fees_pct_notional
                    )

                    # Enforce portfolio-level constraints before proceeding
                    open_positions = sum(
                        1 for pos in summary.positions if (pos.get("qty", 0) or 0) > 0
                    )
                    symbol_has_position = any(
                        str(pos.get("symbol", "")).upper() == symbol.upper() and (pos.get("qty", 0) or 0) > 0
                        for pos in summary.positions
                    )
                    if (
                        self.config.max_positions is not None
                        and not symbol_has_position
                        and open_positions >= self.config.max_positions
                    ):
                        skipped_trades += 1
                        skipped_trades_by_symbol[symbol] += 1
                        reason = "max_positions_reached"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        continue

                    if self.config.max_leverage and summary.equity:
                        current_exposure = self._estimate_gross_exposure(summary.positions, latest_prices)
                        projected_exposure = current_exposure + (qty * price)
                        allowed_exposure = summary.equity * self.config.max_leverage
                        if projected_exposure > allowed_exposure:
                            skipped_trades += 1
                            skipped_trades_by_symbol[symbol] += 1
                            reason = "leverage_cap"
                            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                            continue
                else:  # BUY
                    # Get fresh cash balance (important when processing multiple symbols at same timestamp)
                    summary = summary_snapshot
                    available_cash = summary.cash
                    
                    # Calculate slippage and fees parameters
                    config_slippage = float(self.config.slippage_bps or 0.0)
                    applied_slippage_bps = config_slippage  # Positive for BUY increases cost
                    fees_per_share = float(self.config.commission_per_share or 0.0)
                    fees_pct_notional = float(self.config.commission_pct_notional or 0.0)
                    adjusted_price = price * (1 + applied_slippage_bps / 10_000)
                    
                    # Calculate maximum affordable quantity considering fees and slippage
                    max_affordable = self.portfolio.calculate_affordable_quantity(
                        price=price,
                        fees_per_share=fees_per_share,
                        fees_pct_notional=fees_pct_notional,
                        slippage_bps=applied_slippage_bps,
                        available_cash=available_cash,
                    )
                    
                    if max_affordable <= 0:
                        # Skip if we can't afford even 1 share
                        skipped_trades += 1
                        skipped_trades_by_symbol[symbol] += 1
                        reason = "insufficient_cash"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        # Don't log individual skipped trades - aggregate at end
                        continue
                    
                    # Use the minimum of target quantity and affordable quantity
                    qty = min(target_qty, max_affordable)
                    
                    # Ensure we respect minimum lot size (round down to avoid exceeding cash)
                    lot = max(1, self.config.min_lot)
                    qty = (qty // lot) * lot  # Round down to lot size
                    if qty < lot:
                        # Can't afford even one lot after rounding
                        skipped_trades += 1
                        skipped_trades_by_symbol[symbol] += 1
                        reason = "insufficient_cash_after_lot"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        # Don't log individual skipped trades - aggregate at end
                        continue
                    
                    # Recalculate with actual quantity to verify affordability
                    adjusted_price = price * (1 + applied_slippage_bps / 10_000)
                    notional = adjusted_price * qty
                    commission = (
                        qty * fees_per_share
                        + abs(notional) * fees_pct_notional
                    )
                    
                    # Final affordability check - if rounding to lot size made it unaffordable, reduce
                    total_cost = notional + commission
                    if available_cash < total_cost:
                        # Reduce quantity to fit available cash
                        cost_per_share = adjusted_price * (1 + fees_pct_notional) + fees_per_share
                        if cost_per_share > 0:
                            # Calculate max affordable without lot rounding first
                            max_qty_before_lot = int(available_cash / cost_per_share)
                            # Then round down to lot size
                            qty = (max_qty_before_lot // lot) * lot
                            if qty < lot:
                                # Can't afford even one lot
                                skipped_trades += 1
                                skipped_trades_by_symbol[symbol] += 1
                                reason = "insufficient_cash_after_reduction"
                                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                                # Don't log individual skipped trades - aggregate at end
                                continue
                            # Track partial fill if quantity was reduced
                            if qty < target_qty:
                                partial_fills += 1
                            # Recalculate with reduced quantity
                            notional = adjusted_price * qty
                            commission = (
                                qty * fees_per_share
                                + abs(notional) * fees_pct_notional
                            )
                        else:
                            skipped_trades += 1
                            reason = "invalid_cost_per_share"
                            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                            log_line(
                                f"[{self.bot_id}] Skipped trade for {symbol} (Invalid cost calculation)"
                            )
                            continue
                    
                    # Track partial fill if quantity was reduced from target
                    if qty < target_qty:
                        partial_fills += 1

                try:
                    # Verify cash before executing (double-check)
                    # Skip this check for backtests with unlimited cash
                    if not self.portfolio.unlimited_cash:
                        summary_before = self.portfolio.get_summary()
                        if signal == "BUY" and summary_before.cash < (notional + commission):
                            skipped_trades += 1
                            skipped_trades_by_symbol[symbol] += 1
                            reason = "cash_changed"
                            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                            # Don't log individual skipped trades - aggregate at end
                            continue
                    
                    self.portfolio.record_trade(
                        ts=ts,
                        symbol=symbol,
                        side=signal,
                        quantity=qty,
                        price=adjusted_price,
                        fees=commission,
                        slippage_bps=applied_slippage_bps,
                    )
                    total_fees += commission
                    if qty:
                        slippage_weighted_sum += applied_slippage_bps * qty
                        slippage_qty_total += qty
                except ValueError as exc:  # pragma: no cover - execution failure
                    error_msg = str(exc)
                    skipped_trades += 1
                    skipped_trades_by_symbol[symbol] += 1
                    # Check for insufficient cash errors (case-insensitive)
                    error_lower = error_msg.lower()
                    is_insufficient_cash = (
                        "insufficient cash" in error_lower or 
                        "cash for buy" in error_lower or 
                        ("insufficient" in error_lower and "cash" in error_lower)
                    )
                    
                    # In unlimited cash mode, insufficient cash errors should never happen
                    # If they do, it's a bug - log it but don't fail the backtest
                    if is_insufficient_cash:
                        if self.portfolio.unlimited_cash:
                            # This shouldn't happen - log as warning and continue
                            log_line(
                                f"[{self.bot_id}] Warning: Insufficient cash error in unlimited cash mode for {symbol} at {ts}: {exc}"
                            )
                            reason = "unlimited_cash_bug"
                        else:
                            reason = "insufficient_cash_at_execution"
                            # Don't log individual skipped trades - aggregate at end
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                    else:
                        reason = "execution_error"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        # Check if we're in unlimited cash mode - if so, this is unexpected
                        if self.portfolio.unlimited_cash:
                            log_line(
                                f"[{self.bot_id}] Warning: Unexpected ValueError in unlimited cash mode for {symbol} at {ts}: {exc}"
                            )
                        else:
                            # Only log non-cash execution errors (other errors are important)
                            log_line(f"[{self.bot_id}] Trade execution failed for {symbol} at {ts}: {exc}")
                except Exception as exc:  # pragma: no cover - execution failure
                    error_msg = str(exc)
                    skipped_trades += 1
                    skipped_trades_by_symbol[symbol] += 1
                    # Check for insufficient cash errors (case-insensitive)
                    error_lower = error_msg.lower()
                    is_insufficient_cash = (
                        "insufficient cash" in error_lower or 
                        "cash for buy" in error_lower or 
                        ("insufficient" in error_lower and "cash" in error_lower)
                    )
                    
                    if is_insufficient_cash:
                        reason = "insufficient_cash_at_execution"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        # Don't log individual skipped trades - aggregate at end
                    else:
                        reason = "unexpected_error"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        # Only log non-cash execution errors (other errors are important)
                        log_line(f"[{self.bot_id}] Trade execution failed for {symbol} at {ts}: {exc}")

            if latest_prices:
                self.portfolio.revalue_and_snapshot(ts, latest_prices)

        trades = self.portfolio.get_trades(limit=5000)
        summary = self.portfolio.get_summary()
        metrics = compute_portfolio_metrics(
            self.portfolio,
            timestamp=timeline[-1],
            latest_prices={},
            trades_snapshot=trades,
        )
        avg_slippage_bps = (
            slippage_weighted_sum / slippage_qty_total if slippage_qty_total else 0.0
        )
        metrics["total_fees"] = total_fees
        metrics["avg_slippage_bps"] = avg_slippage_bps
        self.metrics = metrics

        # Get initial and final portfolio values
        initial_portfolio_value = self.portfolio.initial_cash
        final_portfolio_value = summary.equity
        final_cash = summary.cash

        # Log summary statistics with aggregated skipped trades per symbol
        if skipped_trades > 0 or partial_fills > 0:
            reasons_str = ", ".join([f"{k}: {v}" for k, v in skipped_reasons.items()])
            # Aggregate skipped trades by symbol for cleaner output
            if skipped_trades_by_symbol:
                symbol_summaries = []
                for sym, count in sorted(skipped_trades_by_symbol.items(), key=lambda x: x[1], reverse=True):
                    symbol_summaries.append(f"{sym}: {count}")
                symbol_summary_str = " | ".join(symbol_summaries)
                log_line(
                    f"[{self.bot_id}] Summary: Skipped trades: {skipped_trades} ({reasons_str}) | "
                    f"Partial fills: {partial_fills} | Cash left: PKR {format_pkr(final_cash):,.0f}"
                )
                log_line(
                    f"[{self.bot_id}] Skipped trades by symbol: {symbol_summary_str}"
                )
            else:
                log_line(
                    f"[{self.bot_id}] Summary: Skipped trades: {skipped_trades} ({reasons_str}) | "
                    f"Partial fills: {partial_fills} | Cash left: PKR {format_pkr(final_cash):,.0f}"
                )

        hourly_summary = self._aggregate_hourly_summary(
            metrics["equity_curve"],
            trades,
        )

        summary = {
            "initial_portfolio_value": initial_portfolio_value,
            "final_portfolio_value": final_portfolio_value,
            "total_return_pct": (
                ((final_portfolio_value / initial_portfolio_value) - 1.0) * 100.0
                if initial_portfolio_value
                else 0.0
            ),
            "skipped_trades": skipped_trades,
            "partial_fills": partial_fills,
            "final_cash": final_cash,
        }

        return {
            "metrics": metrics["metrics"],
            "equity_curve": metrics["equity_curve"],
            "trades": trades,
            "positions": metrics["positions"],
            "total_fees": total_fees,
            "avg_slippage_bps": avg_slippage_bps,
            "initial_portfolio_value": initial_portfolio_value,
            "final_portfolio_value": final_portfolio_value,
            "skipped_trades": skipped_trades,
            "skipped_trades_by_symbol": dict(skipped_trades_by_symbol),
            "partial_fills": partial_fills,
            "final_cash": final_cash,
            "hourly_summary": hourly_summary,
            "summary": summary,
        }

    def _estimate_gross_exposure(
        self,
        positions: List[Dict[str, Any]],
        price_lookup: Dict[str, float],
    ) -> float:
        total_exposure = 0.0
        for pos in positions:
            qty = abs(float(pos.get("qty", 0) or 0))
            if qty == 0:
                continue
            symbol = str(pos.get("symbol", "")).upper()
            ref_price = price_lookup.get(symbol)
            if ref_price is None:
                ref_price = float(pos.get("avg_cost", 0.0) or 0.0)
            total_exposure += qty * abs(ref_price)
        return total_exposure

    def _position_size_for(self, price: float, summary: PortfolioSummary) -> int:
        # Use capital_allocation if set, otherwise use position_notional
        equity = float(summary.equity or self.config.initial_cash or self.config.position_notional)
        position_notional = self.config.position_notional
        if self.config.capital_allocation is not None and equity > 0:
            position_notional = equity * self.config.capital_allocation
        if self.config.risk_per_trade_pct:
            position_notional = min(
                position_notional,
                equity * (self.config.risk_per_trade_pct / 100.0),
            )
        if self.config.max_position_pct:
            position_notional = min(
                position_notional,
                equity * (self.config.max_position_pct / 100.0),
            )

        qty = int(position_notional // max(price, 1e-6))
        lot = max(1, self.config.min_lot)
        return max(lot, (qty // lot) * lot)

    def _load_history(self, start: Optional[str], end: Optional[str]) -> Dict[str, pd.DataFrame]:
        # Parse dates - allow None for CSV mode
        start_dt = None
        end_dt = None
        if start:
            try:
                start_dt = datetime.fromisoformat(start)
            except (ValueError, TypeError):
                log_line(f"[{self.bot_id}] Invalid start date format: {start}")
        if end:
            try:
                end_dt = datetime.fromisoformat(end)
            except (ValueError, TypeError):
                log_line(f"[{self.bot_id}] Invalid end date format: {end}")
        
        history_map: Dict[str, pd.DataFrame] = {}

        for idx, symbol in enumerate(self.symbols, 1):
            log_line(f"[{self.bot_id}] Loading data for {symbol} ({idx}/{len(self.symbols)})...")
            payload = self.service.get_historical(
                symbol,
                start=start_dt,
                end=end_dt,
                interval=self.config.interval,
            )
            if not payload:
                log_line(f"[{self.bot_id}] No historical data for {symbol}")
                continue
            df = pd.DataFrame.from_records(payload)
            if "datetime" in df.columns:
                df["ts"] = pd.to_datetime(df["datetime"], errors="coerce")
            elif "date" in df.columns:
                df["ts"] = pd.to_datetime(df["date"], errors="coerce")
            elif "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            else:
                log_line(f"[{self.bot_id}] Historical data missing timestamp for {symbol}")
                continue

            df = df.dropna(subset=["ts"])
            if df["ts"].dt.tz is None:
                df["ts"] = df["ts"].dt.tz_localize("UTC")
            else:
                df["ts"] = df["ts"].dt.tz_convert("UTC")

            if "close" not in df.columns:
                close_col = next((c for c in ["CLOSE", "close", "price", "PRICE"] if c in df.columns), None)
                if close_col:
                    df["close"] = df[close_col].astype(float)
                else:
                    log_line(f"[{self.bot_id}] Historical data missing close price for {symbol}")
                    continue

            for col in ["open", "high", "low"]:
                if col not in df.columns:
                    alt = next((c for c in [col.upper(), col.capitalize()] if c in df.columns), None)
                    if alt:
                        df[col] = df[alt].astype(float)
                    else:
                        df[col] = df["close"]

            if "volume" not in df.columns:
                vol_col = next((c for c in ["VOLUME", "volume"] if c in df.columns), None)
                df["volume"] = df[vol_col].astype(float) if vol_col else 0.0

            df = df.sort_values("ts").reset_index(drop=True)
            selected_columns = ["ts", "open", "high", "low", "close", "volume"]
            history_map[symbol] = df.loc[:, selected_columns].copy()

        return history_map

    def _build_timeline(self, history_map: Dict[str, pd.DataFrame]) -> List[datetime]:
        timeline: List[datetime] = []
        for df in history_map.values():
            timeline.extend(df["ts"].tolist())
        unique_sorted = sorted({ts for ts in timeline if isinstance(ts, datetime)})
        return unique_sorted

    def _aggregate_hourly_summary(
        self,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not equity_curve:
            return []

        bucket_map: Dict[datetime, Dict[str, Any]] = {}
        for point in equity_curve:
            ts_raw = point.get("ts")
            if not ts_raw:
                continue
            try:
                dt = datetime.fromisoformat(ts_raw)
            except (TypeError, ValueError):
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            bucket = dt.replace(minute=0, second=0, microsecond=0)
            bucket_map[bucket] = {
                "timestamp": bucket.isoformat(),
                "bucket_end": dt,
                "equity": float(point.get("equity", 0.0) or 0.0),
                "cash": float(point.get("cash", 0.0) or 0.0),
                "unrealized_pnl": float(point.get("unrealized_pnl", 0.0) or 0.0),
            }

        if not bucket_map:
            return []

        trade_buckets: Dict[datetime, Dict[str, Any]] = defaultdict(
            lambda: {"trade_count": 0, "realized_pnl": 0.0}
        )
        trades_parsed: List[Dict[str, Any]] = []
        for trade in trades:
            ts_raw = trade.get("ts") or trade.get("timestamp")
            if not ts_raw:
                continue
            try:
                dt = datetime.fromisoformat(ts_raw)
            except (TypeError, ValueError):
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            bucket = dt.replace(minute=0, second=0, microsecond=0)
            entry = trade_buckets[bucket]
            entry["trade_count"] += 1
            entry["realized_pnl"] += float(trade.get("pnl_realized", 0.0) or 0.0)
            trades_parsed.append(
                {
                    "ts": dt,
                    "symbol": trade.get("symbol", "").upper(),
                    "side": (trade.get("side") or "").upper(),
                    "quantity": int(trade.get("quantity", 0) or 0),
                }
            )

        trades_parsed.sort(key=lambda item: item["ts"])
        positions: Dict[str, int] = defaultdict(int)
        trade_index = 0

        hourly_summary: List[Dict[str, Any]] = []
        for bucket_dt in sorted(bucket_map.keys()):
            record = bucket_map[bucket_dt]
            bucket_end = record["bucket_end"]
            trade_info = trade_buckets.get(bucket_dt, {"trade_count": 0, "realized_pnl": 0.0})
            while trade_index < len(trades_parsed) and trades_parsed[trade_index]["ts"] <= bucket_end:
                trade_entry = trades_parsed[trade_index]
                qty = max(trade_entry["quantity"], 0)
                symbol = trade_entry["symbol"]
                side = trade_entry["side"]
                if side == "BUY":
                    positions[symbol] += qty
                elif side == "SELL":
                    positions[symbol] = max(0, positions[symbol] - qty)
                trade_index += 1

            positions_count = sum(1 for qty in positions.values() if qty > 0)
            hourly_summary.append(
                {
                    "timestamp": record["timestamp"],
                    "equity": record["equity"],
                    "cash": record["cash"],
                    "unrealized_pnl": record["unrealized_pnl"],
                    "realized_pnl": round(trade_info["realized_pnl"], 2),
                    "trade_count": trade_info["trade_count"],
                    "positions_count": positions_count,
                }
            )

        return hourly_summary


__all__ = ["BacktestEngine", "BacktestConfig", "TradeMetrics"]


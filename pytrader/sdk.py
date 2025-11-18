from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .strategy_loader import BUILTIN_STRATEGIES
from .trader import Trader

DateRange = Tuple[str, Optional[str]]


def _normalise_symbols(symbols: Iterable[str]) -> List[str]:
    normalised = [symbol.upper().strip() for symbol in symbols if symbol and symbol.strip()]
    if not normalised:
        raise ValueError("At least one symbol must be provided.")
    return normalised


def _resolve_strategy(strategy_name: str, config: Optional[Dict[str, Any]]) -> Any:
    key = strategy_name.lower().strip()
    strategy_cls = BUILTIN_STRATEGIES.get(key)
    if not strategy_cls:
        raise ValueError(
            f"Strategy '{strategy_name}' is not registered. "
            f"Available templates: {', '.join(sorted(BUILTIN_STRATEGIES.keys()))}"
        )
    try:
        return strategy_cls(**(config or {}))
    except TypeError as exc:
        raise ValueError(f"Invalid configuration for '{strategy_name}': {exc}") from exc


def _resolve_date_range(
    date_range: Optional[DateRange],
    start: Optional[str],
    end: Optional[str],
) -> DateRange:
    if date_range:
        return date_range[0], date_range[1]
    if not start:
        raise ValueError("Start date is required when date_range is not provided.")
    return start, end


def run_backtest(
    strategy_name: str,
    *,
    symbols: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
    date_range: Optional[DateRange] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    api_token: Optional[str] = None,
    backend_url: Optional[str] = None,
    initial_cash: Optional[float] = None,
    position_notional: float = 100_000.0,
    interval: str = "1d",
    min_lot: int = 1,
    slippage_bps: float = 0.0,
    commission_per_share: float = 0.0,
    commission_pct_notional: float = 0.0,
) -> Dict[str, Any]:
    """
    Run a local backtest for a predefined strategy template.

    Args:
        strategy_name: Registered strategy template name (e.g., 'sma_momentum').
        symbols: Iterable of PSX tickers.
        config: Strategy configuration parameters.
        date_range: Tuple of (start, end) ISO dates.
        start: Optional start date (YYYY-MM-DD) when date_range is omitted.
        end: Optional end date (YYYY-MM-DD) when date_range is omitted.
        api_token: API token for authentication.
        backend_url: Backend URL used for token validation.
        initial_cash: Optional starting cash override.
        position_notional: Target notional per trade.
        interval: Historical data interval (default '1d').
        min_lot: Minimum tradable lot size.
        slippage_bps: Slippage in basis points applied to fills.
        commission_per_share: Fixed commission per share.
        commission_pct_notional: Percentage commission on notional value.

    Returns:
        Dictionary containing metrics, equity_curve, trades, hourly_summary, and summary fields.
    """
    symbols_list = _normalise_symbols(symbols)
    strategy = _resolve_strategy(strategy_name, config)
    start_date, end_date = _resolve_date_range(date_range, start, end)

    trader = Trader(
        strategy=strategy,
        symbols=symbols_list,
        initial_cash=initial_cash,
        position_notional=position_notional,
        bot_id=f"backtest-{strategy_name}",
    )

    result = trader.run_backtest(
        start=start_date,
        end=end_date or start_date,
        api_token=api_token,
        backend_url=backend_url,
        interval=interval,
        min_lot=min_lot,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        commission_pct_notional=commission_pct_notional,
    )
    return result


def start_paper_trading(
    strategy_name: str,
    *,
    symbols: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
    api_token: Optional[str] = None,
    backend_url: Optional[str] = None,
    capital: float = 1_000_000.0,
    position_notional: float = 100_000.0,
    cycle_minutes: int = 15,
    warm_start: bool = True,
    max_cycles: Optional[int] = None,
    log_dir: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    trades_path: Optional[Path] = None,
    detailed_logs: bool = False,
) -> None:
    """
    Start terminal-only paper trading for a registered strategy template.

    Args:
        strategy_name: Registered strategy template name.
        symbols: Iterable of PSX tickers.
        config: Strategy configuration parameters.
        api_token: API token for live data (required).
        backend_url: Backend URL for telemetry + authentication.
        capital: Initial virtual cash.
        position_notional: Target notional per trade.
        cycle_minutes: Engine polling cadence.
        warm_start: Whether to replay today's session before live ticks.
        max_cycles: Optional number of cycles to run (None = run indefinitely).
        log_dir: Directory for metrics/trade logs.
        metrics_path: CSV path for metrics history.
        trades_path: CSV path for trade history.
        detailed_logs: Toggle verbose console output.
    """
    symbols_list = _normalise_symbols(symbols)
    strategy = _resolve_strategy(strategy_name, config)

    trader = Trader(
        strategy=strategy,
        symbols=symbols_list,
        cycle_minutes=cycle_minutes,
        initial_cash=capital,
        position_notional=position_notional,
        bot_id=f"paper-{strategy_name}",
    )

    trader.run_paper_trading(
        api_token=api_token,
        backend_url=backend_url,
        warm_start=warm_start,
        max_cycles=max_cycles,
        log_dir=log_dir or Path("logs"),
        metrics_path=metrics_path,
        trades_path=trades_path,
        detailed_logs=detailed_logs,
    )



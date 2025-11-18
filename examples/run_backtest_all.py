"""
Backtesting - Historical data simulation for chosen symbols and date ranges.

Usage:
    python -m pytrader.examples.run_backtest_all --symbols OGDC --days 180

This mode runs a complete historical simulation using past market data.
Backtests use virtual unlimited cash and are completely independent from paper/live trading.

Optional CLI args:
    --symbols OGDC HBL          Symbols to backtest (default: OGDC HBL)
    --days 180                 Number of trailing days (default: 180)
    --position-notional 100000  Position notional per trade in PKR (default: 100000)
    --initial-cash 1000000      Starting cash in PKR (default: 1000000)
    --capital-allocation 0.2    Percentage of equity per trade (e.g., 0.2 = 20%)
    --strategy-path path        Custom strategy file path
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Sequence

from pytrader import load_strategy
from pytrader.trader_core import BacktestConfig, BacktestEngine
from pytrader.trader_core.strategies import (
    BollingerMeanReversionStrategy,
    SMAMomentumStrategy,
)


StrategyFactory = Callable[[], object]

STRATEGIES: dict[str, StrategyFactory] = {
    "momentum_fast": lambda: SMAMomentumStrategy(ma_period=15),
    "mean_reversion_flexible": lambda: BollingerMeanReversionStrategy(period=15, std_dev=1.5),
}


def _iso_range(days: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def run_backtests(
    symbols: Sequence[str],
    days: int,
    *,
    position_notional: float,
    initial_cash: float | None,
    strategy_path: str | None = None,
    capital_allocation: float | None = None,
) -> None:
    start, end = _iso_range(days)
    print("=" * 72)
    print(f"Running backtests for symbols: {', '.join(symbols)}")
    print(f"Date range: {start} → {end}")
    print("=" * 72)
    
    # Use custom strategy if provided, otherwise use built-in strategies
    if strategy_path:
        print(f"Loading custom strategy from: {strategy_path}")
        strategy = load_strategy(strategy_path, symbols=list(symbols))
        strategies_to_run = [("custom", lambda: strategy)]
    else:
        strategies_to_run = STRATEGIES.items()
    
    for name, factory in strategies_to_run:
        strategy = factory()
        config_kwargs = {"position_notional": position_notional}
        if initial_cash is not None:
            config_kwargs["initial_cash"] = initial_cash
        if capital_allocation is not None:
            config_kwargs["capital_allocation"] = capital_allocation
        engine = BacktestEngine(
            symbols=list(symbols),
            strategy=strategy,
            config=BacktestConfig(**config_kwargs),
            bot_id=f"backtest-{name}",
        )
        print(f"\n[{name}] running...")
        result = engine.run(start=start, end=end)
        metrics = result.get("metrics")
        equity_curve = result.get("equity_curve", [])
        trades = result.get("trades", [])
        if metrics:
            sharpe_ratio = getattr(metrics, "sharpe_ratio", None)
            sharpe_available = getattr(
                metrics,
                "sharpe_ratio_available",
                sharpe_ratio is not None,
            )
            sharpe_str = f"{sharpe_ratio:.2f}" if sharpe_available and sharpe_ratio is not None else "-"

            sortino_ratio = getattr(metrics, "sortino_ratio", None)
            sortino_available = getattr(
                metrics,
                "sortino_ratio_available",
                sortino_ratio is not None,
            )
            sortino_str = f"{sortino_ratio:.2f}" if sortino_available and sortino_ratio is not None else "-"

            volatility_pct = getattr(metrics, "volatility_pct", None)
            volatility_available = getattr(
                metrics,
                "volatility_available",
                volatility_pct is not None,
            )
            vol_str = f"{volatility_pct:.2f}%" if volatility_available and volatility_pct is not None else "-"

            win_loss_ratio = getattr(metrics, "win_loss_ratio", None)
            win_loss_available = getattr(
                metrics,
                "win_loss_ratio_available",
                win_loss_ratio is not None,
            )
            wl_str = f"{win_loss_ratio:.2f}" if win_loss_available and win_loss_ratio is not None else "-"
            print(f"  Total return: {metrics.total_return_pct:.2f}%")
            print(f"  Daily return: {metrics.daily_return_pct:.2f}%")
            print(f"  Sharpe ratio: {sharpe_str}")
            print(f"  Sortino ratio: {sortino_str}")
            print(f"  Max drawdown: {metrics.max_drawdown_pct:.2f}%")
            print(f"  Volatility: {vol_str}")
            print(f"  Win/Loss ratio: {wl_str}")
        else:
            print("  No metrics produced.")
        print(f"  Equity curve points: {len(equity_curve)}")
        print(f"  Trades executed: {len(trades)}")
        if "total_fees" in result:
            print(f"  Total fees: {result['total_fees']:.2f} | Avg slippage {result['avg_slippage_bps']:.2f} bps")
        if "skipped_trades" in result and result["skipped_trades"] > 0:
            print(f"  Skipped trades: {result['skipped_trades']} | Partial fills: {result.get('partial_fills', 0)} | Cash left: PKR {result.get('final_cash', 0):,.0f}")
    print("\nAll backtests complete.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample backtests.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["OGDC", "HBL"],
        help="Symbols to include (default: OGDC HBL).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of trailing days to backtest (default: 180).",
    )
    parser.add_argument(
        "--position-notional",
        type=float,
        default=100_000.0,
        help="Per-trade position notional (default: 100000).",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Override starting portfolio cash (default: config default).",
    )
    parser.add_argument(
        "--strategy-path",
        type=str,
        default=None,
        help="Path to custom strategy file (e.g., 'my_strategy.py' or 'user_strategies.MyStrategy'). If not provided, uses built-in strategies.",
    )
    parser.add_argument(
        "--capital-allocation",
        type=float,
        default=None,
        help="Percentage of equity to allocate per trade (e.g., 0.2 = 20%%). If not provided, uses position_notional.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_backtests(
        symbols=args.symbols,
        days=args.days,
        position_notional=args.position_notional,
        initial_cash=args.initial_cash,
        strategy_path=args.strategy_path,
        capital_allocation=args.capital_allocation,
    )


if __name__ == "__main__":
    main()


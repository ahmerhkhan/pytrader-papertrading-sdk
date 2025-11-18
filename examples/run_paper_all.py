"""
Live Paper Trading - Real-time paper trading with 15-minute cycles.

Usage:
    python -m pytrader.examples.run_paper_all --symbols OGDC --token your-token-here

Modes:
    --warm-start (default): Replay today's data from market open, then continue live
    --no-warm-start: Start fresh from current time (cold-start mode)

Optional CLI args:
    --symbols OGDC HBL          Symbols to trade
    --cycles 2                  Number of cycles to run (0 = run forever)
    --cycle-minutes 15          Minutes between cycles (default: 15)
    --strategy-path path        Custom strategy file path
    --reset-account             Reset account to initial cash
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from pytrader import load_strategy
from pytrader.trader_core import EngineConfig, TradingEngine
from pytrader.trader_core.strategies import SMAMomentumStrategy
from pytrader.trader_core.utils import log_line


async def run_paper(
    symbols: Sequence[str],
    max_cycles: Optional[int],
    position_notional: float,
    cycle_minutes: int,
    backend_url: Optional[str],
    token: Optional[str],
    log_dir: str,
    verbose_warm_start: bool,
    slippage_bps: float,
    commission_per_share: float,
    commission_pct_notional: float,
    warm_start: bool = True,
    reset_account: bool = False,
    user_id: str = "default",
    strategy_path: Optional[str] = None,
    _dev_mode: bool = False,  # Internal: for development/testing only
) -> None:
    config = EngineConfig(
        position_notional=position_notional,
        cycle_minutes=cycle_minutes,
        log_dir=Path(log_dir),
        backend_url=backend_url,
        api_token=token,
        require_token=True,
        verbose_warm_start=verbose_warm_start,
        fast_sleep_seconds=1.0 if _dev_mode else None,  # Internal dev mode only
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
        commission_pct_notional=commission_pct_notional,
        warm_start=warm_start,
        reset_account=reset_account,
        user_id=user_id,
    )
    
    # Load custom strategy if provided, otherwise use default
    if strategy_path:
        print(f"Loading custom strategy from: {strategy_path}")
        strategy = load_strategy(strategy_path, symbols=list(symbols))
    else:
        strategy = SMAMomentumStrategy(ma_period=15)
    
    engine = TradingEngine(
        symbols=list(symbols),
        strategy=strategy,
        config=config,
        bot_id="paper-demo",
    )
    try:
        await engine.run_forever(max_cycles=max_cycles)
    except KeyboardInterrupt:
        log_line("[paper-demo] Keyboard interrupt received; shutting down.")
    finally:
        engine.stop()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Paper Trading - Real-time paper trading with configurable cycles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live trading for OGDC with default settings
  python run_paper_all.py --symbols OGDC --token your-token-here

  # Run 5 cycles with warm-start (replay today's data first)
  python run_paper_all.py --symbols OGDC HBL --token your-token-here --cycles 5 --warm-start

  # Cold-start: begin fresh from current time
  python run_paper_all.py --symbols OGDC --token your-token-here --no-warm-start

  # Use custom strategy
  python run_paper_all.py --symbols OGDC --token your-token-here --strategy-path my_strategy.py
        """
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["OGDC", "HBL"],
        help="Symbols to trade (default: OGDC HBL).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to run before exiting (0 = run forever, default: 0).",
    )
    parser.add_argument(
        "--position-notional",
        type=float,
        default=100_000.0,
        help="Position notional per trade in PKR (default: 100000).",
    )
    parser.add_argument(
        "--cycle-minutes",
        type=int,
        default=15,
        help="Minutes between cycles (default: 15).",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default=os.getenv("PYTRADER_BACKEND_URL"),
        help="Backend URL for metrics ingestion (optional).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("PYTRADER_TOKEN") or os.getenv("PYTRADER_API_TOKEN"),
        help="API token for backend authentication (required).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.getenv("PYTRADER_LOG_DIR", "logs"),
        help="Directory to write log files (default: logs).",
    )
    parser.add_argument(
        "--verbose-warm-start",
        action="store_true",
        help="Print detailed warm-start execution summaries (default: concise).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Apply slippage in basis points to each execution (default: 0).",
    )
    parser.add_argument(
        "--commission-per-share",
        type=float,
        default=0.0,
        help="Flat commission in PKR per share/lot (default: 0).",
    )
    parser.add_argument(
        "--commission-pct-notional",
        type=float,
        default=0.0,
        help="Commission as a percent of notional (e.g. 0.0005 = 5 bps).",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=True,
        help="Enable warm-start: replay today's data from market open, then continue live (default: ON).",
    )
    parser.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Disable warm-start: start fresh from current time (cold-start mode).",
    )
    parser.add_argument(
        "--cold-start",
        dest="warm_start",
        action="store_false",
        help="Alias for --no-warm-start: start fresh from current time.",
    )
    parser.add_argument(
        "--reset-account",
        action="store_true",
        help="Reset paper trading account to initial cash (clears positions and history).",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for account persistence (default: default).",
    )
    parser.add_argument(
        "--strategy-path",
        type=str,
        default=None,
        help="Path to custom strategy file (e.g., 'my_strategy.py' or 'user_strategies.MyStrategy'). If not provided, uses default SMAMomentumStrategy.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "A PyTrader API token is required. Set PYTRADER_TOKEN or pass --token."
        )
    max_cycles = None if args.cycles <= 0 else args.cycles
    asyncio.run(
        run_paper(
            symbols=args.symbols,
            max_cycles=max_cycles,
            position_notional=args.position_notional,
            cycle_minutes=args.cycle_minutes,
            backend_url=args.backend_url,
            token=args.token,
            log_dir=args.log_dir,
            verbose_warm_start=args.verbose_warm_start,
            slippage_bps=args.slippage_bps,
            commission_per_share=args.commission_per_share,
            commission_pct_notional=args.commission_pct_notional,
            warm_start=args.warm_start,
            reset_account=args.reset_account,
            user_id=args.user_id,
            strategy_path=args.strategy_path,
        )
    )


if __name__ == "__main__":
    main()


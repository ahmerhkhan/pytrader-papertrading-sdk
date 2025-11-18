"""
PyTrader validation runner.

Modes:
  backtest   → Historical Replay (pure historical backtesting via BacktestEngine)
  quicksim   → QuickSim (Fast validation mode - compressed cycles for testing)
  live       → Live Paper Trading (continuous real-time paper trading)
  all        → run all three in sequence

Usage examples:
    python -m scripts.run_all --mode backtest --symbols OGDC HBL
    python -m scripts.run_all --mode quicksim --symbols OGDC HBL --token your-token-here
    python -m scripts.run_all --mode live --symbols OGDC HBL --token your-token-here
    python -m scripts.run_all --symbols OGDC HBL --token your-token-here   # runs all three
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from pytrader.trader_core import BacktestConfig, BacktestEngine, EngineConfig, TradingEngine
from pytrader.trader_core.strategies import BollingerMeanReversionStrategy, SMAMomentumStrategy
import json
import csv


LOG_ROOT = Path("logs")


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _summarize_logs(bot_id: str) -> tuple[int, int]:
    log_dir = LOG_ROOT / bot_id
    metrics_path = log_dir / "metrics.jsonl"
    trades_path = log_dir / "trades.jsonl"
    csv_path = log_dir / "metrics.csv"
    cycles = 0
    trades_count = 0
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as fh:
            cycles = sum(1 for _ in fh)
    if trades_path.exists():
        with trades_path.open(encoding="utf-8") as fh:
            trades_count = sum(1 for line in fh if line.strip())
    print(f"Log directory: {log_dir.resolve()}")
    for path in [metrics_path, trades_path, csv_path]:
        if path.exists():
            print(f"  ✔ {path.name} ({path.stat().st_size} bytes)")
        else:
            print(f"  ✖ {path.name} not found")
    return cycles, trades_count


def _print_summary(label: str, cycles: int, trades: int, metrics_path: Path) -> None:
    print(
        f"\nSummary for {label}:\n"
        f"  ✅ Cycles processed: {cycles}\n"
        f"  ✅ Trades logged: {trades}\n"
        f"  ✅ Metrics file: {metrics_path}"
    )


def run_backtest_today(
    symbols: Sequence[str],
    *,
    position_notional: float,
    initial_cash: Optional[float],
) -> None:
    _print_header("Historical Replay")
    print("[MODE: BACKTEST] Pure historical replay mode")
    print("=" * 80)
    backtest_config_kwargs = {"position_notional": position_notional}
    if initial_cash is not None:
        backtest_config_kwargs["initial_cash"] = initial_cash
    engine = BacktestEngine(
        symbols=list(symbols),
        strategy=SMAMomentumStrategy(ma_period=12),
        config=BacktestConfig(**backtest_config_kwargs),
        bot_id="runall-backtest",
    )
    today = datetime.now(timezone.utc).date().isoformat()
    result = engine.run(start=today, end=today)
    metrics = result.get("metrics")
    trades = result.get("trades", [])
    curve = result.get("equity_curve", [])
    if metrics:
        print(
            f"Total return: {metrics.total_return_pct:.2f}% | "
            f"Daily: {metrics.daily_return_pct:.2f}% | "
            f"Sharpe: {metrics.sharpe_ratio} | "
            f"MaxDD: {metrics.max_drawdown_pct:.2f}%"
        )
    else:
        print("No metrics produced.")
    print(f"Equity points: {len(curve)} | Trades: {len(trades)}")
    log_dir = LOG_ROOT / "runall-backtest"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"
    trades_path = log_dir / "trades.jsonl"
    csv_path = log_dir / "metrics.csv"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bot_id": "runall-backtest",
        "status": "complete",
        "metrics": metrics.__dict__ if metrics else {},
        "equity_curve_points": len(curve),
        "trades_executed": len(trades),
    }
    if curve:
        entry["equity_curve_tail"] = curve[-5:]
    metrics_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        if metrics:
            writer.writerow(["total_return_pct", metrics.total_return_pct])
            writer.writerow(["daily_return_pct", metrics.daily_return_pct])
            writer.writerow(["sharpe_ratio", metrics.sharpe_ratio])
            writer.writerow(["max_drawdown_pct", metrics.max_drawdown_pct])
        writer.writerow(["equity_points", len(curve)])
        writer.writerow(["trades_executed", len(trades)])
    trades_path.write_text(
        "\n".join(json.dumps(t) for t in trades) + ("\n" if trades else ""),
        encoding="utf-8",
    )
    cycles, trade_lines = _summarize_logs("runall-backtest")
    _print_summary("Historical Replay", cycles, trade_lines, metrics_path)


async def run_quicksim(
    symbols: Sequence[str],
    *,
    token: str | None,
    backend_url: str | None,
    cycle_minutes: int,
    cycles: int,
    position_notional: float,
    initial_cash: Optional[float],
) -> None:
    _print_header("QuickSim (Fast Validation Mode)")
    print("[MODE: QUICKSIM] Fast validation mode - compressed cycles for testing")
    print("=" * 80)
    bot_id = "runall-quicksim"
    log_dir = LOG_ROOT / bot_id
    log_dir.mkdir(parents=True, exist_ok=True)
    engine_config_kwargs = dict(
        position_notional=position_notional,
        cycle_minutes=cycle_minutes,
        slippage_bps=5.0,
        commission_per_share=0.02,
        commission_pct_notional=0.0005,
        fast_sleep_seconds=0.2,
        require_token=bool(token),
        api_token=token,
        backend_url=backend_url,
        verbose_warm_start=False,
        record_warm_start_trades=True,
    )
    if initial_cash is not None:
        engine_config_kwargs["initial_cash"] = initial_cash
    engine = TradingEngine(
        symbols=list(symbols),
        strategy=SMAMomentumStrategy(ma_period=10),
        config=EngineConfig(**engine_config_kwargs),
        bot_id=bot_id,
    )
    engine.start()
    try:
        for _ in range(max(1, cycles)):
            await engine.cycle_once()
            await asyncio.sleep(engine._sleep_seconds or 0.2)
    finally:
        engine.stop()

    if engine.metrics_history:
        last = engine.metrics_history[-1]
        m = last["metrics"]
        print(
            "Totals: "
            f"Daily {m.daily_return_pct:.2f}% | Total {m.total_return_pct:.2f}% | "
            f"Sharpe {m.sharpe_ratio} | MaxDD {m.max_drawdown_pct:.2f}%"
        )
    else:
        print("No metrics produced in QuickSim (likely no new data batches).")
    cycles_logged, trades_logged = _summarize_logs(bot_id)
    metrics_path = log_dir / "metrics.jsonl"
    _print_summary("QuickSim (Today Fast Mode)", cycles_logged, trades_logged, metrics_path)


async def run_live(
    symbols: Sequence[str],
    *,
    token: str | None,
    backend_url: str | None,
    cycle_minutes: int,
    position_notional: float,
    initial_cash: Optional[float],
    warm_start: bool = True,
    reset_account: bool = False,
    user_id: str = "default",
) -> None:
    _print_header("Live Paper Trading")
    print("[MODE: LIVE] Continuous live paper trading mode")
    print("This will run continuously until manually stopped (Ctrl+C)")
    print("=" * 80)
    bot_id = "runall-live"
    log_dir = LOG_ROOT / bot_id
    log_dir.mkdir(parents=True, exist_ok=True)
    engine_config_kwargs = dict(
        position_notional=position_notional,
        cycle_minutes=cycle_minutes,
        slippage_bps=5.0,
        commission_per_share=0.02,
        commission_pct_notional=0.0005,
        require_token=bool(token),
        api_token=token,
        backend_url=backend_url,
        verbose_warm_start=True,
        record_warm_start_trades=True,
        warm_start=warm_start,
        reset_account=reset_account,
        user_id=user_id,
    )
    if initial_cash is not None:
        engine_config_kwargs["initial_cash"] = initial_cash
    engine = TradingEngine(
        symbols=list(symbols),
        strategy=BollingerMeanReversionStrategy(period=15, std_dev=1.5),
        config=EngineConfig(**engine_config_kwargs),
        bot_id=bot_id,
    )
    try:
        print(f"\nStarting continuous live paper trading for {symbols}")
        print("Press Ctrl+C to stop gracefully\n")
        await engine.run_forever()
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    finally:
        engine.stop()

    if engine.metrics_history:
        last = engine.metrics_history[-1]
        m = last["metrics"]
        print(
            "\nFinal Results: "
            f"Daily {m.daily_return_pct:.2f}% | Total {m.total_return_pct:.2f}% | "
            f"Sharpe {m.sharpe_ratio} | MaxDD {m.max_drawdown_pct:.2f}%"
        )
    else:
        print(
            "\nNo metrics produced (no cycles completed). "
            "This may happen if started outside market hours."
        )
    cycles_logged, trades_logged = _summarize_logs(bot_id)
    metrics_path = log_dir / "metrics.jsonl"
    _print_summary("Live Paper Trading", cycles_logged, trades_logged, metrics_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest / paper validation scripts.")
    p.add_argument(
        "--mode",
        choices=["all", "backtest", "historical", "quicksim", "live"],
        default="all",
        help="Which mode(s) to run: backtest (historical replay), quicksim (fast validation), live (continuous paper trading), or all (default: all). Note: 'historical' is an alias for 'backtest'.",
    )
    p.add_argument("--symbols", nargs="+", default=["OGDC", "HBL"], help="Symbols to use")
    p.add_argument("--cycle-minutes", type=int, default=15, help="Cycle size in minutes")
    p.add_argument("--fast-cycles", type=int, default=8, help="Number of QuickSim cycles to run (default: 8)")
    p.add_argument("--token", type=str, default=None, help="API token (required for live data/telemetry)")
    p.add_argument("--backend-url", type=str, default=None, help="Backend URL (optional)")
    p.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Override starting cash/equity (default: config.default_cash).",
    )
    p.add_argument(
        "--position-notional",
        type=float,
        default=100_000.0,
        help="Per-trade position notional (default: 100000).",
    )
    p.add_argument(
        "--warm-start",
        action="store_true",
        default=True,
        help="Enable warm-start: replay today's data from market open (default: ON).",
    )
    p.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Disable warm-start: start fresh from current time (cold-start mode).",
    )
    p.add_argument(
        "--reset-account",
        action="store_true",
        help="Reset paper trading account to initial cash (clears positions and history).",
    )
    p.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for account persistence (default: default).",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    modes = [args.mode] if args.mode != "all" else ["backtest", "quicksim", "live"]
    # Support both "historical" and "backtest" for backward compatibility
    if "historical" in modes:
        modes = [m if m != "historical" else "backtest" for m in modes]
    if "backtest" in modes:
        run_backtest_today(
            args.symbols,
            position_notional=args.position_notional,
            initial_cash=args.initial_cash,
        )
    if "quicksim" in modes:
        asyncio.run(
            run_quicksim(
                args.symbols,
                token=args.token,
                backend_url=args.backend_url,
                cycle_minutes=args.cycle_minutes,
                cycles=args.fast_cycles,
                position_notional=args.position_notional,
                initial_cash=args.initial_cash,
            )
        )
    if "live" in modes:
        asyncio.run(
            run_live(
                args.symbols,
                token=args.token,
                backend_url=args.backend_url,
                cycle_minutes=args.cycle_minutes,
                position_notional=args.position_notional,
                initial_cash=args.initial_cash,
                warm_start=args.warm_start,
                reset_account=args.reset_account,
                user_id=args.user_id,
            )
        )
    print("\nValidation complete.")


if __name__ == "__main__":
    main()




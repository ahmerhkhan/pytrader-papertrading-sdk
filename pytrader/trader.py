"""
Public Trader interface for PyTrader SDK.

High-level wrapper around TradingEngine for easy usage.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import trader_core components from local modules
from .trader_core import BacktestConfig, BacktestEngine, EngineConfig, TradingEngine
from .trader_core.strategies.base import BaseStrategy

from .strategy import Strategy
from .strategy_adapter import StrategyAdapter
from .strategy_loader import load_strategy
from .config import settings


class Trader:
    """
    High-level trading interface for PyTrader SDK.
    
    This class provides a simple interface for running backtests and paper trading
    with custom strategies.
    
    Example:
        from pytrader import Trader, Strategy
        
        class MyStrategy(Strategy):
            def on_data(self, data):
                # Your trading logic
                self.buy('OGDC', 100)
        
        trader = Trader(
            strategy=MyStrategy,
            symbols=['OGDC', 'HBL'],
            cycle_minutes=15
        )
        
        # Run backtest
        trader.run_backtest(start='2024-01-01', end='2024-12-31')
        
        # Or start paper trading
        trader.start_paper_trading(token='your-token')
    """
    
    def __init__(
        self,
        strategy: Union[str, Strategy, BaseStrategy, type[Strategy]],
        symbols: List[str],
        *,
        cycle_minutes: int = 15,
        initial_cash: Optional[float] = None,
        position_notional: float = 100_000.0,
        bot_id: str = "default",
        **kwargs: Any,
    ):
        """
        Initialize Trader.
        
        Args:
            strategy: Strategy class/instance, strategy name string, or file path
            symbols: List of symbols to trade
            cycle_minutes: Cycle duration in minutes (default: 15)
            initial_cash: Starting cash (default: 1,000,000)
            position_notional: Position size per trade (default: 100,000)
            bot_id: Bot identifier
            **kwargs: Additional engine config options
        """
        self.symbols = [s.upper() for s in symbols]
        self.bot_id = bot_id
        self._strategy_instance: Optional[Any] = None
        self._engine: Optional[TradingEngine] = None
        self._backtest_engine: Optional[BacktestEngine] = None
        self._validate_cycle_minutes(cycle_minutes)
        
        # Load strategy
        strategy_obj = self._load_strategy(strategy)
        
        # Store strategy instance if it's a user Strategy
        if isinstance(strategy_obj, Strategy):
            self._strategy_instance = strategy_obj
            # Call on_start if available
            if hasattr(strategy_obj, 'on_start'):
                try:
                    strategy_obj.on_start()
                except Exception:
                    pass
        
        # Create adapter if needed
        if isinstance(strategy_obj, Strategy):
            adapter = StrategyAdapter(strategy_obj)
            # Portfolio service will be set when engine is created
            self._adapter = adapter
            strategy_obj = adapter
        else:
            self._adapter = None
        
        # Store strategy for engine
        self._strategy = strategy_obj
        
        # Store config
        self._config = {
            'cycle_minutes': cycle_minutes,
            'initial_cash': initial_cash,
            'position_notional': position_notional,
            **kwargs,
        }
    
    def _load_strategy(
        self,
        strategy: Union[str, Strategy, BaseStrategy, type[Strategy]],
    ) -> Union[Strategy, BaseStrategy]:
        """Load strategy from various formats."""
        # If it's already an instance
        if isinstance(strategy, (Strategy, BaseStrategy)):
            return strategy
        
        # If it's a class, instantiate it
        if isinstance(strategy, type) and issubclass(strategy, Strategy):
            return strategy(symbols=self.symbols)
        
        # If it's a string (name or path), use loader
        if isinstance(strategy, str):
            return load_strategy(strategy, symbols=self.symbols)
        
        raise ValueError(f"Invalid strategy type: {type(strategy)}")
    
    def run_backtest(
        self,
        start: str,
        end: Optional[str] = None,
        api_token: Optional[str] = None,
        backend_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), defaults to start (single day)
            api_token: API token (used for authentication)
            backend_url: Backend URL for token validation / telemetry
            **kwargs: Additional backtest config options
        
        Returns:
            Dictionary with backtest results
        """
        from .auth import require_token

        require_token(api_token=api_token, backend_url=backend_url)
        config_kwargs = {
            'position_notional': self._config.get('position_notional', 100_000.0),
            **kwargs,
        }
        if self._config.get('initial_cash'):
            config_kwargs['initial_cash'] = self._config['initial_cash']
        
        config = BacktestConfig(**config_kwargs)
        
        self._backtest_engine = BacktestEngine(
            symbols=self.symbols,
            strategy=self._strategy,
            config=config,
            bot_id=self.bot_id,
        )
        
        # Set portfolio service on adapter if available
        if self._adapter and hasattr(self._backtest_engine, 'portfolio'):
            self._adapter.set_portfolio_service(self._backtest_engine.portfolio)
        
        result = self._backtest_engine.run(start=start, end=end or start)
        
        # Call on_end if available
        if self._strategy_instance and hasattr(self._strategy_instance, 'on_end'):
            try:
                self._strategy_instance.on_end()
            except Exception:
                pass
        
        return result
    
    def run_paper_trading(
        self,
        api_token: Optional[str] = None,
        backend_url: Optional[str] = None,
        warm_start: bool = True,
        max_cycles: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Start live paper trading (synchronous wrapper for notebooks/CLI).

        Args:
            api_token: API token for authentication.
            backend_url: Backend URL used for token validation & telemetry.
            warm_start: Enable warm replay before switching to live loop.
            max_cycles: Optional safety stop (None = run indefinitely).
            log_dir: Optional directory for local metrics/trade logs.
            **kwargs: Additional EngineConfig overrides.
        """
        try:
            asyncio.get_running_loop()
            try:
                import nest_asyncio

                nest_asyncio.apply()
                asyncio.run(
                    self.start_paper_trading(
                        api_token=api_token,
                        backend_url=backend_url,
                        warm_start=warm_start,
                        max_cycles=max_cycles,
                        log_dir=log_dir,
                        **kwargs,
                    )
                )
            except ImportError:
                raise RuntimeError(
                    "Cannot run paper trading inside an existing event loop.\n"
                    "Install 'nest-asyncio' or call 'await trader.start_paper_trading(...)'."
                )
        except RuntimeError:
            asyncio.run(
                self.start_paper_trading(
                    api_token=api_token,
                    backend_url=backend_url,
                    warm_start=warm_start,
                    max_cycles=max_cycles,
                    log_dir=log_dir,
                    **kwargs,
                )
            )
    
    async def start_paper_trading(
        self,
        api_token: Optional[str] = None,
        backend_url: Optional[str] = None,
        warm_start: bool = True,
        max_cycles: Optional[int] = None,
        log_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Start live paper trading (async method).

        Args:
            api_token: API token (required).
            backend_url: Backend URL (required for telemetry + auth).
            warm_start: Replay current session before live trading.
            max_cycles: Optional stop after N cycles.
            log_dir: Directory for local telemetry.
            **kwargs: EngineConfig overrides.
        """
        from .auth import require_token

        validated_token = require_token(api_token=api_token, backend_url=backend_url)
        config_kwargs = {
            'cycle_minutes': self._config.get('cycle_minutes', 15),
            'position_notional': self._config.get('position_notional', 100_000.0),
            'require_token': bool(validated_token),
            'api_token': validated_token,
            'backend_url': backend_url,
            'warm_start': warm_start,
            'user_id': self._compute_user_id(validated_token),
            **kwargs,
        }
        if log_dir is not None:
            config_kwargs['log_dir'] = Path(log_dir)
        if self._config.get('initial_cash'):
            config_kwargs['initial_cash'] = self._config['initial_cash']
        
        config = EngineConfig(**config_kwargs)
        
        self._engine = TradingEngine(
            symbols=self.symbols,
            strategy=self._strategy,
            config=config,
            bot_id=self.bot_id,
        )
        
        # Set portfolio service on adapter if available
        if self._adapter and hasattr(self._engine, 'portfolio'):
            self._adapter.set_portfolio_service(self._engine.portfolio)
        
        try:
            await self._engine.run_forever(max_cycles=max_cycles)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            if self._engine:
                self._engine.stop()
            
            # Call on_end if available
            if self._strategy_instance and hasattr(self._strategy_instance, 'on_end'):
                try:
                    self._strategy_instance.on_end()
                except Exception:
                    pass
    
    def stop(self) -> None:
        """Stop paper trading."""
        if self._engine:
            self._engine.stop()
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest metrics from engine."""
        if self._engine and self._engine.metrics_history:
            return self._engine.metrics_history[-1]
        return None

    def _compute_user_id(self, token: str) -> str:
        """
        Generate a stable per-user identifier so paper accounts/logs are isolated
        per API token + bot combo.
        """
        salt = os.getenv("PYTRADER_ACCOUNT_SALT", "")
        raw = f"{self.bot_id}:{token}:{salt}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        return digest[:16]

    def _validate_cycle_minutes(self, cycle_minutes: int) -> None:
        """
        Enforce realistic PSX cycle spacing unless an explicit override is provided.
        """
        min_cycle = max(1, getattr(settings, "alignment_minutes", 15))
        allow_fast = os.getenv("PYTRADER_ALLOW_FAST_CYCLE", "0").lower() in {"1", "true", "yes"}
        if cycle_minutes < min_cycle and not allow_fast:
            raise ValueError(
                f"cycle_minutes={cycle_minutes} is shorter than the PSX bar interval ({min_cycle}m). "
                "Set PYTRADER_ALLOW_FAST_CYCLE=1 if you really need sub-interval testing."
            )


__all__ = ["Trader"]


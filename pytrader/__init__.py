"""
PyTrader SDK - run strategies locally, stream telemetry to the backend, and
monitor everything from the dashboard.

Bots execute entirely on the founder's machines via the SDK. The backend acts
as an authenticated data + telemetry hub (no remote strategy execution).
"""

# Primary client - PyTrader REST client
from .client import PyTrader

# Authentication
from .auth import AuthenticationError, require_token, validate_token

# Export indicators
from . import indicators

# Deprecated: Trader class for local execution (kept for backward compatibility)
from .trader import Trader
from .strategy import Strategy
from .strategy_loader import load_strategy, list_strategies, register_strategy
from .sdk import run_backtest, start_paper_trading

# BACKEND-ONLY: These components are for backend internal use only.
# SDK users should NOT import these - they are not part of the public API.
# They are kept in the package for backend to use, but not exported to SDK users.
# from .trader_core import (
#     BacktestEngine,
#     BacktestConfig,
#     TradingEngine,
#     EngineConfig,
#     TradeMetrics,
#     PortfolioService,
# )

__version__ = "2.0.0"

__all__ = [
    # Primary API - PyTrader REST client
    "PyTrader",
    # Authentication
    "AuthenticationError",
    "require_token",
    "validate_token",
    # Deprecated - kept for backward compatibility
    "Trader",
    "Strategy",
    "run_backtest",
    "start_paper_trading",
    "load_strategy",
    "list_strategies",
    "register_strategy",
    # Utilities
    "indicators",
    # NOTE: Backend-only components (BacktestEngine, TradingEngine, etc.) are NOT exported
    # SDK users must use PyTrader client only. Backend components are for internal use.
]

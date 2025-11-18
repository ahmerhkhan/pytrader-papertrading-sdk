"""pyTrader - Paper trading SDK for the Pakistan Stock Exchange (PSX).

This package exposes algorithmic trading engines, strategy helpers, and data
providers that can be embedded inside your own applications. A reference
FastAPI backend now lives in the separate ``pytrader-backend`` project.
"""

from pytrader.trader_core import (
    TradingEngine,
    PortfolioService,
    BacktestEngine,
    EngineConfig,
    BacktestConfig,
    TradeMetrics,
)

__version__ = "1.0.0"

__all__: list[str] = [
    "TradingEngine",
    "PortfolioService",
    "BacktestEngine",
    "EngineConfig",
    "BacktestConfig",
    "TradeMetrics",
]


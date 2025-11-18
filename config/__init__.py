"""Trading configuration module."""

from .trading_config import TradingConfig

# Import settings from pytrader.config (the actual config module)
try:
    from ..config import settings
except ImportError:
    # Fallback if pytrader.config doesn't exist (shouldn't happen in installed package)
    settings = None

__all__ = ["TradingConfig", "settings"]
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MarketHours:
    open_hour: int = 9
    open_minute: int = 0
    close_hour: int = 15
    close_minute: int = 30


@dataclass
class Settings:
    # Database
    db_url: str = os.getenv("PYTRADER_DB_URL", "sqlite:///data/trader.db")

    # PSX backend - default to Render deployment
    psx_base_url: str = os.getenv("PYPSX_BASE_URL", "https://pypsx-library.onrender.com")
    http_timeout_seconds: int = int(os.getenv("PYPSX_TIMEOUT", "30"))

    # Market
    market_hours: MarketHours = MarketHours()
    timezone: str = os.getenv("PYTRADER_TZ", "Asia/Karachi")

    # Trading defaults
    default_cash: float = float(os.getenv("PYTRADER_CASH", "1000000"))
    default_symbols: List[str] = (os.getenv("PYTRADER_SYMBOLS", "").split(",") if os.getenv("PYTRADER_SYMBOLS") else [])
    position_size_pk: float = float(os.getenv("PYTRADER_POSITION_SIZE_PK", "100000"))  # notional per trade by default
    min_lot: int = int(os.getenv("PYTRADER_MIN_LOT", "1"))

    # Scheduler
    alignment_minutes: int = int(os.getenv("PYTRADER_ALIGN_MIN", "15"))

    # API Authentication
    api_keys: List[str] = os.getenv("PYTRADER_API_KEYS", "").split(",") if os.getenv("PYTRADER_API_KEYS") else []
    require_api_key: bool = os.getenv("PYTRADER_REQUIRE_API_KEY", "false").lower() == "true"


settings = Settings()


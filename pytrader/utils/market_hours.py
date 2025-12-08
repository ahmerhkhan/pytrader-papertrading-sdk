"""PSX market hours tracking and utilities."""

from datetime import datetime, time, timedelta
from typing import Optional

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    # Create a simple timezone replacement
    class timezone:
        def __init__(self, name):
            self.name = name
        def localize(self, dt):
            return dt
        @staticmethod
        def now():
            return datetime.now()
    
    PSX_TIMEZONE_OBJ = timezone("Asia/Karachi")


# PSX timezone (Pakistan Standard Time - UTC+5)
if PYTZ_AVAILABLE:
    PSX_TIMEZONE = pytz.timezone("Asia/Karachi")
else:
    PSX_TIMEZONE = PSX_TIMEZONE_OBJ


class PSXMarketHours:
    """
    PSX market hours tracker.
    
    PSX trading hours (Pakistan Standard Time):
    - Monday to Thursday: 9:32 AM to 3:30 PM
    - Friday: 9:32 AM to 1:00 PM
    - Saturday and Sunday: Closed
    
    Note: Due to 15-minute data delay, paper trading can start when first data batch arrives
    (approximately 9:45-9:47 AM) and continues until 3:45 PM to process the 3:30 PM batch.
    """
    
    # Monday-Thursday market hours
    MARKET_OPEN_TIME_MON_THU = time(9, 32)  # 9:32 AM (PSX actual open time)
    MARKET_CLOSE_TIME_MON_THU = time(15, 30)  # 3:30 PM (market close)
    PAPER_TRADING_END_TIME_MON_THU = time(15, 45)  # 3:45 PM (last batch arrives)
    
    # Friday market hours
    MARKET_OPEN_TIME_FRI = time(9, 32)  # 9:32 AM
    MARKET_CLOSE_TIME_FRI = time(13, 0)  # 1:00 PM (market close)
    PAPER_TRADING_END_TIME_FRI = time(13, 15)  # 1:15 PM (last batch arrives)
    
    # Data delay in minutes (endpoints receive data 15 mins after trading starts)
    DATA_DELAY_MINUTES = 15
    
    @classmethod
    def is_weekend(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if given datetime is weekend (Saturday or Sunday).
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            True if weekend, False otherwise
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            # Ensure datetime is in PSX timezone
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        return dt.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    @classmethod
    def is_market_open(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            True if market is open, False otherwise
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        # Check if weekend
        if cls.is_weekend(dt):
            return False
        
        # Check time based on day of week
        current_time = dt.time()
        weekday = dt.weekday()  # Monday = 0, Friday = 4
        
        if weekday == 4:  # Friday
            return cls.MARKET_OPEN_TIME_FRI <= current_time < cls.MARKET_CLOSE_TIME_FRI
        else:  # Monday-Thursday
            return cls.MARKET_OPEN_TIME_MON_THU <= current_time < cls.MARKET_CLOSE_TIME_MON_THU
    
    @classmethod
    def can_paper_trade(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if paper trading should continue.
        
        Paper trading continues during:
        1. Regular market hours (9:32 AM - 3:30 PM)
        2. Post-market data window (3:30 PM - 3:45 PM) to process final batch
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            True if paper trading should continue
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        if cls.is_weekend(dt):
            return False
        
        current_time = dt.time()
        weekday = dt.weekday()
        
        if weekday == 4:  # Friday
            # Paper trading: 9:32 AM - 1:15 PM (includes data window)
            return cls.MARKET_OPEN_TIME_FRI <= current_time < cls.PAPER_TRADING_END_TIME_FRI
        else:  # Monday-Thursday
            # Paper trading: 9:32 AM - 3:45 PM (includes data window)
            return cls.MARKET_OPEN_TIME_MON_THU <= current_time < cls.PAPER_TRADING_END_TIME_MON_THU
    
    @classmethod
    def is_pre_market(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if currently in pre-market hours.
        
        Pre-market is before regular trading hours:
        - Monday-Thursday: before 9:32 AM
        - Friday: before 9:32 AM
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            True if pre-market hours
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        if cls.is_weekend(dt):
            return False
        
        current_time = dt.time()
        weekday = dt.weekday()
        
        if weekday == 4:  # Friday
            return current_time < cls.MARKET_OPEN_TIME_FRI
        else:  # Monday-Thursday
            return current_time < cls.MARKET_OPEN_TIME_MON_THU
    
    @classmethod
    def is_post_market(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if currently in post-market hours.
        
        Post-market is after regular trading hours:
        - Monday-Thursday: after 3:30 PM
        - Friday: after 12:00 PM
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            True if post-market hours
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        if cls.is_weekend(dt):
            return False
        
        current_time = dt.time()
        weekday = dt.weekday()
        
        if weekday == 4:  # Friday
            return current_time >= cls.MARKET_CLOSE_TIME_FRI
        else:  # Monday-Thursday
            return current_time >= cls.MARKET_CLOSE_TIME_MON_THU
    
    @classmethod
    def get_next_market_open(cls, dt: Optional[datetime] = None) -> datetime:
        """
        Get next market open datetime.
        
        Args:
            dt: Reference datetime (default: current time)
            
        Returns:
            Next market open datetime
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        current_time = dt.time()
        weekday = dt.weekday()
        
        # Determine today's market hours
        if weekday == 4:  # Friday
            today_open = cls.MARKET_OPEN_TIME_FRI
            today_close = cls.MARKET_CLOSE_TIME_FRI
        else:  # Monday-Thursday
            today_open = cls.MARKET_OPEN_TIME_MON_THU
            today_close = cls.MARKET_CLOSE_TIME_MON_THU
        
        # If weekend or after market close today, find next trading day
        if cls.is_weekend(dt) or current_time >= today_close:
            # Find next trading day (Monday)
            days_until_monday = (7 - weekday) % 7
            if days_until_monday == 0:
                # It's Monday but market is closed, so next Monday is 7 days away
                days_until_monday = 7
            
            next_trading_day = dt + timedelta(days=days_until_monday)
            next_trading_day = next_trading_day.replace(
                hour=cls.MARKET_OPEN_TIME_MON_THU.hour,
                minute=cls.MARKET_OPEN_TIME_MON_THU.minute,
                second=0,
                microsecond=0
            )
            return next_trading_day
        
        # If before market open today, return today's open
        if current_time < today_open:
            today_market_open = dt.replace(
                hour=today_open.hour,
                minute=today_open.minute,
                second=0,
                microsecond=0
            )
            return today_market_open
        
        # Market is currently open or we're past close - return next trading day's open
        tomorrow = dt + timedelta(days=1)
        if tomorrow.weekday() >= 5:  # Weekend
            days_until_monday = (7 - tomorrow.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            tomorrow = tomorrow + timedelta(days=days_until_monday)
        
        # Determine next trading day's open time
        next_weekday = tomorrow.weekday()
        if next_weekday == 4:  # Friday
            next_open_time = cls.MARKET_OPEN_TIME_FRI
        else:  # Monday-Thursday
            next_open_time = cls.MARKET_OPEN_TIME_MON_THU
        
        next_trading_day_open = tomorrow.replace(
            hour=next_open_time.hour,
            minute=next_open_time.minute,
            second=0,
            microsecond=0
        )
        return next_trading_day_open
    
    @classmethod
    def get_trading_status(cls, dt: Optional[datetime] = None) -> str:
        """
        Get current trading status.
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            Status string: "open", "closed", "pre_market", "post_market", "weekend"
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        if cls.is_weekend(dt):
            return "weekend"
        
        if cls.is_market_open(dt):
            return "open"
        
        if cls.is_pre_market(dt):
            return "pre_market"
        
        if cls.is_post_market(dt):
            return "post_market"
        
        return "closed"
    
    @classmethod
    def can_start_trading(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if trading can start (accounting for 15-minute data delay).
        
        Trading can start when:
        - Market is open AND
        - At least 15 minutes have passed since market open (first data batch available)
        
        Args:
            dt: Datetime to check (default: current time)
            
        Returns:
            True if trading can start (data should be available)
        """
        if dt is None:
            dt = datetime.now()
            if PYTZ_AVAILABLE:
                dt = PSX_TIMEZONE.localize(dt) if dt.tzinfo is None else dt.astimezone(PSX_TIMEZONE)
        else:
            if PYTZ_AVAILABLE:
                if dt.tzinfo is None:
                    dt = PSX_TIMEZONE.localize(dt)
                else:
                    dt = dt.astimezone(PSX_TIMEZONE)
        
        # Check if weekend
        if cls.is_weekend(dt):
            return False
        
        # Check if market is open
        if not cls.is_market_open(dt):
            return False
        
        # Check if enough time has passed since market open for first data batch
        current_time = dt.time()
        weekday = dt.weekday()
        
        if weekday == 4:  # Friday
            market_open = cls.MARKET_OPEN_TIME_FRI
        else:  # Monday-Thursday
            market_open = cls.MARKET_OPEN_TIME_MON_THU
        
        # Calculate time since market open
        market_open_dt = dt.replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)
        time_since_open = dt - market_open_dt
        
        # Trading can start if at least DATA_DELAY_MINUTES have passed since market open
        return time_since_open.total_seconds() >= (cls.DATA_DELAY_MINUTES * 60)




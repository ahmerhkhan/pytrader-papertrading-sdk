"""
Utility functions for safely formatting optional metrics.
"""

from typing import Optional


def format_metric(value: Optional[float], format_str: str = ".2f", default: str = "-") -> str:
    """
    Safely format an optional metric value.
    
    Args:
        value: The metric value (can be None)
        format_str: Format string (e.g., ".2f", ".1%")
        default: Default string to return if value is None
    
    Returns:
        Formatted string or default
    """
    if value is None:
        return default
    return f"{value:{format_str}}"


def format_percent(value: Optional[float], default: str = "-") -> str:
    """Format optional percentage value."""
    return format_metric(value, ".2f", default) + "%" if value is not None else default


def format_ratio(value: Optional[float], default: str = "-") -> str:
    """Format optional ratio value."""
    return format_metric(value, ".2f", default)


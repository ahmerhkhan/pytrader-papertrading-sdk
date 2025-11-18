"""
Token authentication for PyTrader SDK.

Validates API tokens against the backend service. If the backend is unreachable,
an allow-listed set of "trusted" tokens can still run locally for development
and paper trading scenarios.
"""

from __future__ import annotations

import os
from typing import Optional, Set

import httpx
import warnings

from .config import settings

DEFAULT_BACKEND_URL = settings.backend_url
TRUSTED_DEFAULT_TOKENS = {
    "ahmer-token",
    "amaan-token",
    "sadiq-token",
    "iba-token",
    "demo-token",
    "dev-token",
}


class AuthenticationError(Exception):
    """Raised when token authentication fails."""
    pass


class BackendUnavailableError(AuthenticationError):
    """Raised when the backend cannot be reached for validation."""


def _load_trusted_tokens() -> Set[str]:
    from_env = {
        token.strip()
        for token in os.getenv("PYTRADER_TRUSTED_TOKENS", "").split(",")
        if token.strip()
    }
    return TRUSTED_DEFAULT_TOKENS.union(from_env)


def validate_token(api_token: str, backend_url: str) -> bool:
    """
    Validate an API token against the backend service.
    
    Args:
        api_token: API token to validate
        backend_url: Backend API URL (MANDATORY)
    
    Returns:
        True if token is valid, False otherwise
    
    Raises:
        AuthenticationError: If validation fails, backend is unreachable, or returns 401/403
    """
    if not api_token:
        raise AuthenticationError("API token is required")
    
    if not backend_url:
        raise BackendUnavailableError(
            "Backend URL is required. Set PYTRADER_BACKEND_URL environment variable."
        )
    
    try:
        response = httpx.get(
            f"{backend_url.rstrip('/')}/health",
            headers={"X-PyTrader-Token": api_token},
            timeout=5.0,
        )
        
        # Hard error on 401/403
        if response.status_code == 401:
            raise AuthenticationError("Invalid API token (401 Unauthorized). Please check your token and try again.")
        if response.status_code == 403:
            raise AuthenticationError("Access forbidden (403). Your token may not have permission for this operation.")
        
        # Hard error on any non-200 status
        if response.status_code != 200:
            raise AuthenticationError(
                f"Backend returned error status {response.status_code}. "
                f"Response: {response.text[:200] if response.text else 'No response body'}"
            )
        
        # Verify response can be parsed
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise AuthenticationError("Backend returned invalid response format. Expected JSON object.")
        except Exception as e:
            raise AuthenticationError(f"Backend response cannot be verified: {e}")
        
        return True
        
    except httpx.RequestError as e:
        raise BackendUnavailableError(
            f"Backend is unreachable at {backend_url}. "
            f"Cannot validate token. Error: {e}. "
            f"Please ensure the backend is running and PYTRADER_BACKEND_URL is correct."
        ) from e
    except AuthenticationError:
        # Re-raise authentication errors
        raise
    except Exception as e:
        # Hard error on any other exception
        raise AuthenticationError(f"Token validation error: {e}") from e


def require_token(api_token: Optional[str] = None, backend_url: Optional[str] = None) -> str:
    """
    Require and validate an API token. Backend URL is MANDATORY.
    
    NO FALLBACKS. NO WARNINGS. NO OFFLINE EXECUTION.
    
    Args:
        api_token: API token (can be from parameter, PYTRADER_API_TOKEN, or PYTRADER_TOKEN env var)
        backend_url: Backend API URL (can be from parameter or PYTRADER_BACKEND_URL env var)
    
    Returns:
        Validated token string
    
    Raises:
        AuthenticationError: If token is missing, backend URL is missing, backend is unreachable, 
                            or token validation fails
    """
    # Get token from parameter or env var (support both old and new env var names)
    resolved_token = api_token or os.getenv("PYTRADER_API_TOKEN") or os.getenv("PYTRADER_TOKEN")
    
    if not resolved_token:
        raise AuthenticationError(
            "API token is required. "
            "Please provide a token: "
            "1. Pass api_token='your-token' to the function, "
            "2. Set PYTRADER_API_TOKEN environment variable, "
            "3. Contact your administrator to get a token"
        )
    
    # Get backend URL from parameter, env var, or default deployment
    resolved_backend_url = backend_url or os.getenv("PYTRADER_BACKEND_URL") or DEFAULT_BACKEND_URL
    trusted_tokens = _load_trusted_tokens()
    is_trusted_token = resolved_token in trusted_tokens
    
    if not resolved_backend_url:
        if is_trusted_token:
            warnings.warn(
                "Backend URL is not configured; continuing in trusted-token mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            return resolved_token
        raise AuthenticationError(
            "Backend URL is required. "
            "Please set PYTRADER_BACKEND_URL environment variable or pass backend_url parameter. "
            "The SDK cannot validate untrusted tokens without a backend connection."
        )
    
    try:
        validate_token(resolved_token, resolved_backend_url)
    except BackendUnavailableError as exc:
        if is_trusted_token:
            warnings.warn(
                f"{exc} Proceeding because the token is in the trusted allow list.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise
    
    return resolved_token


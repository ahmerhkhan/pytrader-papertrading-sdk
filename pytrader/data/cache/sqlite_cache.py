"""
SQLite cache helper with TTL awareness for provider data.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

DEFAULT_CACHE_PATH = os.getenv(
    "PYTRADER_CACHE_DB",
    os.path.join(os.path.dirname(__file__), "pytrader_cache.db"),
)


@dataclass
class CacheResult:
    value: Any
    expires_at: datetime

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at


class SQLiteCache:
    """
    Minimal SQLite-backed cache with TTL semantics.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_CACHE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._connection_lock = threading.Lock()
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at)"
            )

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        with self._connection_lock:
            conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        payload = json.dumps(value, default=str)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (cache_key, value, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET value = excluded.value, expires_at = excluded.expires_at
                """,
                (key, payload, int(expires_at.timestamp())),
            )

    def get(self, key: str) -> Optional[CacheResult]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE cache_key = ?", (key,)
            )
            row = cur.fetchone()
            if not row:
                return None
            value_raw, expires_ts = row
            expires_at = datetime.fromtimestamp(expires_ts, tz=timezone.utc)
            return CacheResult(value=json.loads(value_raw), expires_at=expires_at)

    def invalidate(self, key_prefix: Optional[str] = None) -> int:
        """
        Delete all entries or those that match a prefix. Returns rows deleted.
        """
        with self._get_connection() as conn:
            if key_prefix:
                (count,) = conn.execute(
                    "SELECT COUNT(*) FROM cache_entries WHERE cache_key LIKE ?",
                    (f"{key_prefix}%",),
                ).fetchone()
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key LIKE ?",
                    (f"{key_prefix}%",),
                )
                return int(count)
            else:
                (count,) = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
                conn.execute("DELETE FROM cache_entries")
                return int(count)

    def prune(self) -> int:
        """
        Remove expired records; returns number of rows removed.
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at <= ?", (now_ts,)
            )
            return cur.rowcount or 0


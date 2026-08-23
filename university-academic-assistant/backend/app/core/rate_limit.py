"""In-process sliding-window rate limiting.

A per-client-IP window keeps abusive clients from exhausting the model
server or the index. State is process-local, which is fine for a single
instance; for horizontal scaling, front this with a shared limiter (e.g. a
Redis-backed one) instead.
"""

from __future__ import annotations

import threading
import time


class SlidingWindowRateLimiter:
    """Allow at most ``max_requests`` per ``window_seconds`` per key."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1, window_seconds)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True when ``key`` may proceed; otherwise record nothing."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            recent = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(recent) >= self.max_requests:
                self._hits[key] = recent
                return False
            recent.append(now)
            self._hits[key] = recent
            return True

    def retry_after_seconds(self, key: str) -> int:
        """Seconds until the oldest recorded hit of ``key`` leaves the window."""
        with self._lock:
            recent = self._hits.get(key, [])
            if not recent:
                return 0
            oldest = min(recent)
        return max(1, int(oldest + self.window_seconds - time.monotonic()) + 1)
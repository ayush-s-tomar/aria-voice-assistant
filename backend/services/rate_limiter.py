"""
Rate limiting service.

Cluster F additions:
- In-memory sliding-window rate limiting: per-session AND global caps.
- Per-session asyncio.Lock so two messages on the same session (double
  WS send, double form submit) can't race and interleave/corrupt history.

Note: this is process-local, not shared across multiple server workers.
Fine for a single Render/uvicorn worker. If you ever scale to multiple
workers, swap this for a Redis-backed limiter — you already have Upstash
wired up from Cluster A, so that's a small follow-up, not a rewrite.
"""

import os
import time
import asyncio
from collections import deque, defaultdict

SESSION_LIMIT = int(os.getenv("RATE_LIMIT_SESSION_PER_MIN", "10"))
GLOBAL_LIMIT = int(os.getenv("RATE_LIMIT_GLOBAL_PER_MIN", "100"))
WINDOW_SECONDS = 60

_session_hits: dict[str, deque] = defaultdict(deque)
_global_hits: deque = deque()
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class RateLimitExceeded(Exception):
    def __init__(self, scope: str, retry_after: float):
        self.scope = scope
        self.retry_after = retry_after
        super().__init__(f"{scope} rate limit exceeded, retry after {retry_after:.1f}s")


def _prune(dq: deque, now: float) -> None:
    while dq and now - dq[0] > WINDOW_SECONDS:
        dq.popleft()


def check_rate_limit(session_id: str) -> None:
    """
    Records a hit for this session + globally.
    Raises RateLimitExceeded (with .scope and .retry_after) if either cap is hit.
    Checks global first since it's the cheaper/more severe limit to report.
    """
    now = time.time()

    _prune(_global_hits, now)
    if len(_global_hits) >= GLOBAL_LIMIT:
        retry_after = WINDOW_SECONDS - (now - _global_hits[0])
        raise RateLimitExceeded("global", max(retry_after, 1.0))

    session_dq = _session_hits[session_id]
    _prune(session_dq, now)
    if len(session_dq) >= SESSION_LIMIT:
        retry_after = WINDOW_SECONDS - (now - session_dq[0])
        raise RateLimitExceeded("session", max(retry_after, 1.0))

    _global_hits.append(now)
    session_dq.append(now)


def get_session_lock(session_id: str) -> asyncio.Lock:
    """One lock per session — serializes concurrent requests against the same history."""
    return _session_locks[session_id]
# -*- coding: utf-8 -*-
"""
inference_engine/core/rate_limit.py
====================================
In-process fixed-window rate limiting.

``RateLimiter`` counts hits per key per window; it backs both the per-client
request limit on the scoring endpoints (``rate_limited`` dependency) and the
process-wide LLM call budget in the explainer. State lives in this process
only — run a shared store (e.g. Redis) instead if the API is scaled out to
several workers.
"""
from __future__ import annotations

import ipaddress
import math
import threading
import time
from typing import Dict, Tuple

from fastapi import HTTPException, Request

from inference_engine.core.config import settings


class RateLimiter:
    """Fixed-window counter keyed by an arbitrary string (thread-safe)."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._windows: Dict[str, Tuple[float, int]] = {}

    def hit(self, key: str, limit: int) -> Tuple[bool, float]:
        """Record one hit for *key*.

        Returns ``(allowed, retry_after_seconds)``. A *limit* of 0 or less
        means "no hits allowed".
        """
        if limit <= 0:
            return False, self.window_seconds
        now = time.monotonic()
        with self._lock:
            # Drop expired windows so memory stays bounded by active clients.
            if len(self._windows) > 10_000:
                self._windows = {
                    k: v for k, v in self._windows.items() if now - v[0] < self.window_seconds
                }
            start, count = self._windows.get(key, (now, 0))
            if now - start >= self.window_seconds:
                start, count = now, 0
            if count >= limit:
                return False, max(0.0, self.window_seconds - (now - start))
            self._windows[key] = (start, count + 1)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()


#: Per-client limiter for the scoring endpoints.
request_limiter = RateLimiter()


def client_key(request: Request) -> str:
    """Identify the client for rate limiting.

    ``X-Forwarded-For`` is trusted only when the direct peer is a loopback
    address — i.e. the request came through a local reverse proxy such as the
    Vite dev/preview proxy. Anywhere else the header could be spoofed, so the
    peer address is used.
    """
    peer = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        try:
            is_loopback = ipaddress.ip_address(peer).is_loopback
        except ValueError:
            is_loopback = False
        if is_loopback:
            first_hop = forwarded.split(",")[0].strip()
            if first_hop:
                return first_hop
    return peer


async def rate_limited(request: Request) -> None:
    """FastAPI dependency: 429 once a client exceeds RATE_LIMIT_PER_MINUTE."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    allowed, retry_after = request_limiter.hit(client_key(request), limit)
    if not allowed:
        seconds = max(1, math.ceil(retry_after))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit of {limit} requests per minute exceeded. Retry in {seconds}s.",
            headers={"Retry-After": str(seconds)},
        )

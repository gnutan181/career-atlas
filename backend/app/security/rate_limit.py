"""Small in-process fixed-window limiter for API abuse protection.

For a horizontally scaled deployment, replace this store with Redis.  The API
contract (429 plus Retry-After) stays the same.
"""
from __future__ import annotations

import hashlib
import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests: int, window_seconds: int):
        super().__init__(app)
        self.requests = requests
        self.window_seconds = window_seconds
        self.windows: dict[str, deque[float]] = {}

    @staticmethod
    def _identity(request: Request) -> str:
        token = request.headers.get("authorization", "")
        if token.lower().startswith("bearer "):
            # Never retain a raw access token as the limiter key.
            return "token:" + hashlib.sha256(token[7:].encode()).hexdigest()
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        return "ip:" + (forwarded or (request.client.host if request.client else "unknown"))

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        now = time.monotonic()
        key = self._identity(request)
        window = self.windows.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self.requests:
            retry_after = max(1, int(self.window_seconds - (now - window[0])) + 1)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please retry shortly."},
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)
        return await call_next(request)


def add_rate_limit_middleware(app) -> None:
    app.add_middleware(
        RateLimitMiddleware,
        requests=settings.api_rate_limit_requests,
        window_seconds=settings.api_rate_limit_window_seconds,
    )

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic
from typing import Callable

from starlette.types import ASGIApp, Receive, Scope, Send

from app.config.settings import Settings, settings


Clock = Callable[[], float]


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int = 0


class InMemoryRateLimiter:
    """Small per-process fixed-window limiter for research backend admission."""

    def __init__(self, *, clock: Clock = monotonic) -> None:
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, *, limit: int, window_seconds: float) -> RateLimitDecision:
        now = self._clock()
        events = self._events[key]
        cutoff = now - window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            retry_after = max(int(round(window_seconds - (now - events[0]))), 1)
            return RateLimitDecision(False, retry_after)
        events.append(now)
        return RateLimitDecision(True)

    def reset(self) -> None:
        self._events.clear()


class RateLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        app_settings: Settings = settings,
        limiter: InMemoryRateLimiter | None = None,
    ) -> None:
        self.app = app
        self._settings = app_settings
        self._limiter = limiter or InMemoryRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        decision = self._limiter.check(
            f"ip:{_client_ip(scope)}:{scope.get('method')}:{scope.get('path')}",
            limit=self._limit_for(scope),
            window_seconds=self._settings.rate_limit_window_seconds,
        )
        if decision.allowed:
            await self.app(scope, receive, send)
            return
        body = (
            b'{"request_id":"rate-limit","error":{"code":"rate_limit_exceeded",'
            b'"message":"Rate limit exceeded.","details":null}}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"retry-after", str(decision.retry_after_seconds).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _limit_for(self, scope: Scope) -> int:
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "")
        prefix = self._settings.api_v1_prefix.rstrip("/")
        if method == "POST" and path in {
            f"{prefix}/predictions",
            f"{prefix}/external/predictions",
            f"{prefix}/voice/predict",
        }:
            return self._settings.prediction_rate_limit_per_window
        if method == "POST" and path == f"{prefix}/me/api-keys":
            return self._settings.api_key_creation_rate_limit_per_window
        return self._settings.rate_limit_requests_per_window


def _client_ip(scope: Scope) -> str:
    client = scope.get("client")
    if isinstance(client, tuple) and client:
        return str(client[0]).replace(":", "_")[:80]
    return "unknown"

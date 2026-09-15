import json
import logging
from collections.abc import Awaitable, Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config.settings import Settings, settings
from app.core.request_context import (
    CLIENT_CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    normalize_client_correlation_id,
    normalize_request_id,
)

logger = logging.getLogger(__name__)

MAX_REQUEST_BODY_ERROR_CODE = "request_body_too_large"


class RequestBodyLimitExceeded(Exception):
    """Raised when an HTTP request body exceeds the configured ASGI limit."""


class RequestBodyLimitMiddleware:
    """ASGI request-body limiter for multipart audio upload endpoints."""

    def __init__(self, app: ASGIApp, app_settings: Settings = settings) -> None:
        self.app = app
        self._settings = app_settings
        self._max_body_bytes = app_settings.max_request_body_mb * 1024 * 1024
        prefix = app_settings.api_v1_prefix.rstrip("/")
        self._limited_routes = {
            ("POST", f"{prefix}/predictions"),
            ("POST", f"{prefix}/external/predictions"),
        }
        self._legacy_route = ("POST", f"{prefix}/voice/predict")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._should_limit(scope):
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        client_correlation_id = _client_correlation_id_from_scope(scope)
        content_length = _content_length(scope)
        if content_length is not None and content_length > self._max_body_bytes:
            logger.warning(
                "Request rejected before body parsing: Content-Length exceeds limit.",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                        "status_code": 413,
                        "client_correlation_id": client_correlation_id,
                    },
                )
            await _send_413(
                send,
                request_id=request_id,
                client_correlation_id=client_correlation_id,
            )
            return

        if content_length is None:
            try:
                receive = await _buffered_receive(receive, self._max_body_bytes)
            except RequestBodyLimitExceeded:
                logger.warning(
                    "Request rejected before body parsing: streamed body exceeds limit.",
                    extra={
                        "request_id": request_id,
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                            "status_code": 413,
                            "client_correlation_id": client_correlation_id,
                        },
                    )
                await _send_413(
                    send,
                    request_id=request_id,
                    client_correlation_id=client_correlation_id,
                )
                return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        limited_receive = _limited_receive(
            receive,
            max_body_bytes=self._max_body_bytes,
        )

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except RequestBodyLimitExceeded:
            logger.warning(
                "Request rejected while streaming body: body exceeds limit.",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                        "status_code": 413,
                        "client_correlation_id": client_correlation_id,
                    },
                )
            if not response_started:
                await _send_413(
                    send,
                    request_id=request_id,
                    client_correlation_id=client_correlation_id,
                )

    def _should_limit(self, scope: Scope) -> bool:
        route = (scope.get("method"), scope.get("path"))
        if route in self._limited_routes:
            return True
        return (
            route == self._legacy_route
            and self._settings.app_env.lower() != "production"
            and self._settings.enable_legacy_anonymous_prediction
        )


def _limited_receive(
    receive: Receive,
    *,
    max_body_bytes: int,
) -> Callable[[], Awaitable[Message]]:
    bytes_seen = 0

    async def receive_wrapper() -> Message:
        nonlocal bytes_seen
        message = await receive()
        if message["type"] == "http.request":
            body = message.get("body", b"")
            bytes_seen += len(body)
            if bytes_seen > max_body_bytes:
                raise RequestBodyLimitExceeded
        return message

    return receive_wrapper


async def _buffered_receive(
    receive: Receive,
    max_body_bytes: int,
) -> Receive:
    bytes_seen = 0
    messages: list[Message] = []

    while True:
        message = await receive()
        messages.append(message)
        if message["type"] == "http.request":
            bytes_seen += len(message.get("body", b""))
            if bytes_seen > max_body_bytes:
                raise RequestBodyLimitExceeded
            if not message.get("more_body", False):
                break
        elif message["type"] == "http.disconnect":
            break

    async def replay_receive() -> Message:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return replay_receive


def _content_length(scope: Scope) -> int | None:
    for raw_name, raw_value in scope.get("headers") or []:
        if raw_name.lower() != b"content-length":
            continue
        try:
            value = int(raw_value.decode("latin-1"))
        except ValueError:
            return None
        return value if value >= 0 else None
    return None


def _request_id_from_scope(scope: Scope) -> str:
    return normalize_request_id(None)


def _client_correlation_id_from_scope(scope: Scope) -> str | None:
    correlation_header = CLIENT_CORRELATION_ID_HEADER.lower().encode("latin-1")
    request_id_header = REQUEST_ID_HEADER.lower().encode("latin-1")
    fallback_value: str | None = None
    for raw_name, raw_value in scope.get("headers") or []:
        if raw_name.lower() == correlation_header:
            return normalize_client_correlation_id(raw_value.decode("latin-1"))
        if raw_name.lower() == request_id_header:
            fallback_value = raw_value.decode("latin-1")
    return normalize_client_correlation_id(fallback_value)


async def _send_413(
    send: Send,
    *,
    request_id: str,
    client_correlation_id: str | None,
) -> None:
    body = json.dumps(
        {
            "request_id": request_id,
            "error": {
                "code": MAX_REQUEST_BODY_ERROR_CODE,
                "message": "Request body exceeds the configured size limit.",
                "details": None,
            },
        }
    ).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"x-request-id", request_id.encode("latin-1")),
    ]
    if client_correlation_id is not None:
        headers.append(
            (
                b"x-client-correlation-id",
                client_correlation_id.encode("latin-1"),
            )
        )
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body})

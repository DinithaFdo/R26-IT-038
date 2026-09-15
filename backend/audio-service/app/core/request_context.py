from contextvars import ContextVar
import re
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"
CLIENT_CORRELATION_ID_HEADER = "X-Client-Correlation-ID"
REQUEST_ID_MAX_LENGTH = 128
REQUEST_ID_PATTERN = re.compile(r"[^A-Za-z0-9._:-]+")

_request_id_context: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return _request_id_context.get()


def set_request_id(request_id: str):
    return _request_id_context.set(request_id)


def reset_request_id(token) -> None:
    _request_id_context.reset(token)


def normalize_request_id(value: str | None) -> str:
    return str(uuid4())


def normalize_client_correlation_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = REQUEST_ID_PATTERN.sub("-", value.strip())[:REQUEST_ID_MAX_LENGTH]
    return normalized or None


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = normalize_request_id(None)
        client_correlation_id = normalize_client_correlation_id(
            request.headers.get(CLIENT_CORRELATION_ID_HEADER)
            or request.headers.get(REQUEST_ID_HEADER)
        )
        request.state.request_id = request_id
        request.state.client_correlation_id = client_correlation_id
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        if client_correlation_id is not None:
            response.headers[CLIENT_CORRELATION_ID_HEADER] = client_correlation_id
        return response

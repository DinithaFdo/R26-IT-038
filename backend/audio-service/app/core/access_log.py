import logging
from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


logger = logging.getLogger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = max((perf_counter() - start) * 1000, 0.0)
            logger.info(
                "HTTP request completed.",
                extra={
                    "request_id": getattr(request.state, "request_id", "-"),
                    "correlation_id": getattr(
                        request.state,
                        "client_correlation_id",
                        None,
                    ),
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "stage": "request_total",
                },
            )

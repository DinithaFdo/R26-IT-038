import json
import logging
import sys
from datetime import UTC, datetime

from app.config.settings import settings
from app.core.request_context import get_request_id


class RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key in (
            "method",
            "path",
            "status_code",
            "correlation_id",
            "prediction_id",
            "route",
            "stage",
            "feature_stage",
            "job_status",
            "branch_name",
            "model_name",
            "status",
            "mode",
            "model_mode",
            "duration_ms",
            "queue_wait_ms",
            "active_job_count",
            "queue_depth",
            "error_code",
            "retry_attempt",
            "storage_status",
            "upload_filename",
            "file_size_bytes",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIDFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())

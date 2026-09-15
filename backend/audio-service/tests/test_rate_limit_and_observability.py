import json
import logging

from app.core.logging import JsonFormatter
from app.core.rate_limit import InMemoryRateLimiter
from app.core.timing import StageTimings, stage_timer


def test_in_memory_rate_limiter_returns_retry_after_until_window_expires() -> None:
    now = 100.0

    def clock() -> float:
        return now

    limiter = InMemoryRateLimiter(clock=clock)

    assert limiter.check("ip:test", limit=2, window_seconds=10).allowed is True
    assert limiter.check("ip:test", limit=2, window_seconds=10).allowed is True

    decision = limiter.check("ip:test", limit=2, window_seconds=10)

    assert decision.allowed is False
    assert decision.retry_after_seconds == 10

    now = 111.0
    assert limiter.check("ip:test", limit=2, window_seconds=10).allowed is True


def test_stage_timer_records_non_negative_duration_and_logs_stage(caplog) -> None:
    timings = StageTimings()

    with caplog.at_level(logging.INFO, logger="tests.stage"):
        with stage_timer(
            "audio_preprocessing",
            timings=timings,
            logger_name="tests.stage",
            request_id="request-1",
            prediction_id="prediction-1",
        ):
            pass

    assert timings.durations_ms["audio_preprocessing"] >= 0
    record = next(
        record
        for record in caplog.records
        if getattr(record, "stage", None) == "audio_preprocessing"
    )
    assert record.request_id == "request-1"
    assert record.prediction_id == "prediction-1"
    assert record.duration_ms >= 0


def test_json_formatter_preserves_phase2_observability_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="tests.logging",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="stage complete",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"
    record.stage = "storage_upload"
    record.duration_ms = 3.25
    record.storage_status = "skipped"
    record.queue_depth = 1
    record.branch_name = "cnn_acoustic"
    record.feature_stage = "disvoice_iaif"

    payload = json.loads(formatter.format(record))

    assert payload["request_id"] == "request-1"
    assert payload["stage"] == "storage_upload"
    assert payload["duration_ms"] == 3.25
    assert payload["storage_status"] == "skipped"
    assert payload["queue_depth"] == 1
    assert payload["branch_name"] == "cnn_acoustic"
    assert payload["feature_stage"] == "disvoice_iaif"

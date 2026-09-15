from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from time import perf_counter
from typing import Iterator


logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    durations_ms: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, duration_ms: float) -> None:
        self.durations_ms[stage] = max(duration_ms, 0.0)


@contextmanager
def stage_timer(
    stage: str,
    *,
    timings: StageTimings | None = None,
    logger_name: str | None = None,
    **log_fields,
) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        duration_ms = max((perf_counter() - start) * 1000, 0.0)
        if timings is not None:
            timings.record(stage, duration_ms)
        logging.getLogger(logger_name or __name__).info(
            "Stage completed.",
            extra={
                "stage": stage,
                "duration_ms": duration_ms,
                **log_fields,
            },
        )

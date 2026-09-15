"""Startup reconciliation for Voice XAI runs stranded by a process restart.

If the backend process is stopped/crashes while an explanation run is
``queued``/``running``/``partial``/``blocked``, no worker will ever resume it
-- the in-process :class:`AsynchronousExplanationQueue` has no durable state.
This module gives those records a defined terminal outcome at the next
startup instead of leaving them permanently stuck with no operator-visible
error (see ``persistence/mongodb.py``'s ``find_stale_active_explanations``).

This performs reconciliation only. It never re-runs XAI analysis; expensive
re-computation stays an explicit user action (retry) per the existing
lifecycle contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import logging

from app.schemas.xai import ExplanationError
from app.voice_xai.persistence.mongodb import XaiPersistenceConsistencyError
from app.voice_xai.persistence.protocols import XaiExplanationRepository
from app.voice_xai.status import XaiStatusService

logger = logging.getLogger(__name__)

STALE_RUN_ERROR_CODE = "xai_run_interrupted"
STALE_RUN_ERROR_MESSAGE = (
    "The explanation run was interrupted by a service restart and could not "
    "resume automatically. Trigger a retry to create a new run."
)


@dataclass(frozen=True, slots=True)
class StaleRunRecoveryResult:
    scanned: int
    recovered: int
    skipped: int


async def recover_stale_explanations(
    repository: XaiExplanationRepository,
    *,
    stale_after_seconds: float,
    now: datetime | None = None,
) -> StaleRunRecoveryResult:
    """Mark runs stranded before this process started as ``failed``.

    Safe to call on every startup: a run only qualifies once it has not been
    updated for ``stale_after_seconds``, so an explanation that is genuinely
    still queued/running under a live worker in the *same* process (a fast
    reload, not a crash) is not touched.
    """

    cutoff = (now or datetime.now(UTC)) - timedelta(seconds=stale_after_seconds)
    stale_documents = await repository.find_stale_active_explanations(
        updated_before=cutoff
    )
    status_service = XaiStatusService(repository)
    recovered = 0
    for document in stale_documents:
        explanation_id = document["id"]
        owner_user_id = document["owner_user_id"]
        try:
            await status_service.fail_run(
                explanation_id=explanation_id,
                owner_user_id=owner_user_id,
                error=ExplanationError(
                    component=None,
                    code=STALE_RUN_ERROR_CODE,
                    message=STALE_RUN_ERROR_MESSAGE,
                ),
            )
            recovered += 1
            logger.warning(
                "Recovered stale Voice XAI run left non-terminal by a restart.",
                extra={
                    "explanation_id": explanation_id,
                    "prediction_id": document.get("prediction_id"),
                    "request_id": document.get("request_id"),
                    "previous_status": document.get("status"),
                },
            )
        except XaiPersistenceConsistencyError:
            # The run legitimately progressed between the scan and this
            # write (e.g. a live worker in this same process finished it
            # first) -- leave it alone rather than overwrite real evidence.
            logger.info(
                "Voice XAI stale-run recovery skipped a run that changed "
                "concurrently.",
                extra={"explanation_id": explanation_id},
            )
        except Exception:
            logger.exception(
                "Voice XAI stale-run recovery could not update a run.",
                extra={"explanation_id": explanation_id},
            )
    return StaleRunRecoveryResult(
        scanned=len(stale_documents),
        recovered=recovered,
        skipped=len(stale_documents) - recovered,
    )


__all__ = [
    "STALE_RUN_ERROR_CODE",
    "STALE_RUN_ERROR_MESSAGE",
    "StaleRunRecoveryResult",
    "recover_stale_explanations",
]

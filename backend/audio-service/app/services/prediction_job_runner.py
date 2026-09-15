import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Protocol, TypeVar

from fastapi import HTTPException, status

from app.config.settings import Settings, settings
from app.core.exceptions import (
    PredictionQueueFullError,
    PredictionRunnerShuttingDownError,
    PrincipalPredictionLimitExceededError,
)
from app.services.prediction_persistence_service import (
    PredictionPersistenceError,
    PredictionPersistenceService,
    sanitized_error_code,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class PredictionJobRunner(Protocol):
    async def run(
        self,
        *,
        prediction_id: str,
        request_id: str,
        principal_key: str | None = None,
        persistence: PredictionPersistenceService,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        raise NotImplementedError

    async def run_blocking(self, func: Callable[[], T]) -> T:
        raise NotImplementedError

    async def shutdown(self, *, wait: bool = True) -> None:
        raise NotImplementedError


class PredictionProcessWorker(Protocol):
    """Future-compatible interface for off-process real-model execution."""

    async def run_sync(self, func: Callable[[], T]) -> T:
        raise NotImplementedError


class ThreadedPredictionProcessWorker:
    """Runs blocking media and model work away from the FastAPI event loop."""

    def __init__(self, *, max_workers: int) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="multiscope-prediction",
        )

    async def run_sync(self, func: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func)

    async def shutdown(self, *, wait: bool = True) -> None:
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=wait,
            cancel_futures=True,
        )


class InlinePredictionJobRunner:
    """MVP runner that executes jobs inline behind a future worker boundary.

    Migration point: replace this class with a durable queue producer backed by
    Redis or another broker, plus a separate worker that consumes queued
    prediction IDs and calls the same execution service. Do not use FastAPI
    background tasks for durable prediction work because in-flight jobs can be
    lost during server restarts.
    """

    def __init__(
        self,
        app_settings: Settings = settings,
        *,
        process_worker: PredictionProcessWorker | None = None,
    ) -> None:
        self._timeout_seconds = app_settings.prediction_job_timeout_seconds
        self._semaphore = asyncio.Semaphore(app_settings.max_concurrent_predictions)
        self._process_worker = process_worker or ThreadedPredictionProcessWorker(
            max_workers=app_settings.max_concurrent_predictions,
        )
        self._active_prediction_ids: set[str] = set()
        self._principal_prediction_ids: dict[str, set[str]] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._active_lock = asyncio.Lock()
        self._max_concurrent_jobs = app_settings.max_concurrent_predictions
        self._max_queued_jobs = app_settings.max_queued_predictions
        self._max_active_per_principal = app_settings.max_active_predictions_per_principal
        self._shutting_down = False
        self._shutdown_timeout_seconds = app_settings.prediction_shutdown_timeout_seconds

    async def run(
        self,
        *,
        prediction_id: str,
        request_id: str,
        principal_key: str | None = None,
        persistence: PredictionPersistenceService,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        await self._mark_started(prediction_id, principal_key=principal_key)
        worker_task: asyncio.Task[T] | None = None
        acquired = False
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._timeout_seconds,
            )
            acquired = True
            worker_task = asyncio.create_task(
                execute(),
                name=f"prediction-job-{prediction_id}",
            )
            async with self._active_lock:
                self._active_tasks[prediction_id] = worker_task

            try:
                done, _pending = await asyncio.wait(
                    {worker_task},
                    timeout=self._timeout_seconds,
                )
                if not done:
                    raise TimeoutError
                result = worker_task.result()
                await self._finish_owned_job(prediction_id, acquired=acquired)
                acquired = False
                return result
            except TimeoutError:
                await self._record_terminal_failure(
                    persistence=persistence,
                    request_id=request_id,
                    code="prediction_job_timeout",
                )
                worker_task.add_done_callback(
                    lambda task: self._schedule_late_finish(
                        prediction_id=prediction_id,
                        request_id=request_id,
                        task=task,
                    )
                )
                logger.warning(
                    "Prediction job timed out; worker remains owner until exit.",
                    extra={"request_id": request_id, "prediction_id": prediction_id},
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Prediction job timed out.",
                )
            except asyncio.CancelledError:
                await self._record_terminal_failure(
                    persistence=persistence,
                    request_id=request_id,
                    code="prediction_job_cancelled",
                )
                worker_task.add_done_callback(
                    lambda task: self._schedule_late_finish(
                        prediction_id=prediction_id,
                        request_id=request_id,
                        task=task,
                    )
                )
                logger.warning(
                    "Prediction job cancelled; worker remains owner until exit.",
                    extra={"request_id": request_id, "prediction_id": prediction_id},
                )
                raise
            except PredictionPersistenceError:
                await self._finish_owned_job(prediction_id, acquired=acquired)
                acquired = False
                raise
            except Exception as error:
                if isinstance(error, HTTPException):
                    await self._finish_owned_job(prediction_id, acquired=acquired)
                    acquired = False
                    raise
                await self._record_terminal_failure(
                    persistence=persistence,
                    request_id=request_id,
                    code=sanitized_error_code(error),
                )
                await self._finish_owned_job(prediction_id, acquired=acquired)
                acquired = False
                raise
        except TimeoutError:
            if not acquired:
                await self._record_terminal_failure(
                    persistence=persistence,
                    request_id=request_id,
                    code="prediction_job_timeout",
                )
                await self._mark_finished(prediction_id)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Prediction job timed out.",
                )
            raise
        except Exception:
            if not acquired:
                await self._mark_finished(prediction_id)
            raise

    async def run_blocking(self, func: Callable[[], T]) -> T:
        return await self._process_worker.run_sync(func)

    async def shutdown(self, *, wait: bool = True) -> None:
        self._shutting_down = True
        active_tasks = list(self._active_tasks.values())
        logger.info(
            "Prediction job runner shutdown started.",
            extra={"active_jobs": len(active_tasks), "wait": wait},
        )
        if wait and active_tasks:
            done, pending = await asyncio.wait(
                active_tasks,
                timeout=self._shutdown_timeout_seconds,
            )
            if pending:
                logger.warning(
                    "Prediction job runner shutdown timed out with active jobs.",
                    extra={"active_jobs": len(pending)},
                )
            for task in done:
                try:
                    task.result()
                except Exception:
                    logger.exception("Prediction job ended during runner shutdown.")
        shutdown = getattr(self._process_worker, "shutdown", None)
        if callable(shutdown):
            result = shutdown(wait=wait)
            if hasattr(result, "__await__"):
                await result
        logger.info("Prediction job runner shutdown complete.")

    def health(self) -> dict[str, int | bool]:
        active_jobs = len(self._active_prediction_ids)
        return {
            "ready": not self._shutting_down,
            "accepting_jobs": not self._shutting_down,
            "shutting_down": self._shutting_down,
            "active_jobs": active_jobs,
            "max_concurrent_jobs": self._max_concurrent_jobs,
            "max_queued_jobs": self._max_queued_jobs,
            "queue_depth": max(active_jobs - self._max_concurrent_jobs, 0),
            "available_capacity": max(
                self._max_concurrent_jobs + self._max_queued_jobs - active_jobs,
                0,
            ),
            "executor_healthy": True,
        }

    async def _mark_started(
        self,
        prediction_id: str,
        *,
        principal_key: str | None,
    ) -> None:
        async with self._active_lock:
            if self._shutting_down:
                raise PredictionRunnerShuttingDownError()
            if prediction_id in self._active_prediction_ids:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Prediction is already executing.",
                )
            if (
                len(self._active_prediction_ids)
                >= self._max_concurrent_jobs + self._max_queued_jobs
            ):
                raise PredictionQueueFullError()
            if principal_key is not None:
                principal_predictions = self._principal_prediction_ids.setdefault(
                    principal_key,
                    set(),
                )
                if len(principal_predictions) >= self._max_active_per_principal:
                    raise PrincipalPredictionLimitExceededError()
                principal_predictions.add(prediction_id)
            self._active_prediction_ids.add(prediction_id)

    async def _mark_finished(self, prediction_id: str) -> None:
        async with self._active_lock:
            self._active_prediction_ids.discard(prediction_id)
            self._active_tasks.pop(prediction_id, None)
            empty_keys = []
            for principal_key, prediction_ids in self._principal_prediction_ids.items():
                prediction_ids.discard(prediction_id)
                if not prediction_ids:
                    empty_keys.append(principal_key)
            for principal_key in empty_keys:
                self._principal_prediction_ids.pop(principal_key, None)

    async def _finish_owned_job(self, prediction_id: str, *, acquired: bool) -> None:
        if acquired:
            self._semaphore.release()
        await self._mark_finished(prediction_id)

    async def _record_terminal_failure(
        self,
        *,
        persistence: PredictionPersistenceService,
        request_id: str,
        code: str,
    ) -> None:
        try:
            await persistence.mark_failed(
                request_id,
                stage="execution",
                code=code,
            )
        except PredictionPersistenceError:
            logger.info(
                "Prediction terminal failure was already persisted or could not be updated.",
                extra={"request_id": request_id, "code": code},
            )
        except Exception:
            logger.exception(
                "Prediction terminal failure persistence failed.",
                extra={"request_id": request_id, "code": code},
            )

    def _schedule_late_finish(
        self,
        *,
        prediction_id: str,
        request_id: str,
        task: asyncio.Task,
    ) -> None:
        asyncio.create_task(
            self._finish_late_worker(
                prediction_id=prediction_id,
                request_id=request_id,
                task=task,
            )
        )

    async def _finish_late_worker(
        self,
        *,
        prediction_id: str,
        request_id: str,
        task: asyncio.Task,
    ) -> None:
        try:
            task.result()
            logger.warning(
                "Prediction worker finished after caller timeout/cancellation.",
                extra={"request_id": request_id, "prediction_id": prediction_id},
            )
        except asyncio.CancelledError:
            logger.info(
                "Prediction worker task was cancelled.",
                extra={"request_id": request_id, "prediction_id": prediction_id},
            )
        except Exception:
            logger.exception(
                "Prediction worker failed after caller timeout/cancellation.",
                extra={"request_id": request_id, "prediction_id": prediction_id},
            )
        finally:
            await self._finish_owned_job(prediction_id, acquired=True)

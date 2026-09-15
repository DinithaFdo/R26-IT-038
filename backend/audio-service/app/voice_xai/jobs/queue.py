"""Bounded background execution for post-prediction XAI work.

Runs every submitted job as a coroutine on one persistent, dedicated asyncio
event loop owned by a single background thread -- never via ``asyncio.run()``
per job. A fresh loop per job would force any async I/O performed inside the
job (in particular MongoDB access through PyMongo's async driver, which is
documented as unsafe to share across event loops) either to create its own
loop-bound client per job, or to reuse a client bound to a different loop
than the one currently running -- both are worse than one stable loop that
lives for the process lifetime.

If ``mongodb_client_factory`` is supplied, it is invoked once, from inside
the worker thread as the loop starts, to build a MongoDB client dedicated to
this loop (see ``app.database.mongodb.build_dedicated_async_mongo_client``).
Callers can read it back through :attr:`AsynchronousExplanationQueue.worker_database`
once :meth:`start` (or the first :meth:`submit`) has run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
import inspect
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Kept generic because the queue is also unit-tested as a reusable bounded
# coroutine queue. Production submits ``ClassifierInferenceBundle`` objects.
ExplanationTask = Callable[[Any], "Coroutine[Any, Any, Any]"]


class AsynchronousExplanationQueue:
    """Runs Phase 2/3 only after classifier result persistence succeeds."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_queued_jobs: int = 8,
        mongodb_client_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        if max_workers < 1 or max_queued_jobs < 0:
            raise ValueError("max_workers must be positive and max_queued_jobs non-negative.")
        self._capacity = threading.BoundedSemaphore(max_workers + max_queued_jobs)
        self._mongodb_client_factory = mongodb_client_factory
        self._start_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._worker_client: Any | None = None
        self._worker_database: Any | None = None
        self._closed = False

    @property
    def worker_database(self) -> Any | None:
        """The dedicated database bound to the worker loop, once started."""

        return self._worker_database

    def start(self) -> None:
        """Start the persistent worker loop once. Safe to call repeatedly."""

        with self._start_lock:
            if self._loop is not None or self._closed:
                return
            ready = threading.Event()
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(ready,),
                name="voice-xai-loop",
                daemon=True,
            )
            self._thread.start()
            if not ready.wait(timeout=10):
                logger.error("Voice XAI worker loop did not start in time.")

    def _run_loop(self, ready: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self._mongodb_client_factory is not None:
            try:
                self._worker_client, self._worker_database = (
                    self._mongodb_client_factory()
                )
            except Exception:
                logger.exception(
                    "Voice XAI worker could not create its dedicated MongoDB "
                    "client; XAI jobs needing MongoDB will fail explicitly "
                    "instead of silently sharing the main application client."
                )
        self._loop = loop
        ready.set()
        logger.info("voice_xai_worker_started")
        try:
            loop.run_forever()
        finally:
            loop.close()

    def submit(
        self,
        bundle: Any,
        explanation_task: ExplanationTask,
    ) -> Future[Any] | None:
        """Return ``None`` rather than making a classifier wait for XAI capacity.

        ``explanation_task`` must return a coroutine when called with
        ``bundle`` (an ``async def`` or an equivalent callable) -- it is
        scheduled onto the persistent worker loop, never executed with a new
        ``asyncio.run()`` call.
        """

        if self._closed:
            return None
        self.start()
        if self._loop is None:
            return None
        if not self._capacity.acquire(blocking=False):
            return None
        context = _job_context(bundle)
        logger.info(
            "voice_xai_job_submitted",
            extra=context,
        )
        coroutine = explanation_task(bundle)
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        future.add_done_callback(
            lambda completed: self._on_job_done(completed, context)
        )
        return future

    def _on_job_done(
        self, future: Future[Any], context: dict[str, str]
    ) -> None:
        self._capacity.release()
        if future.cancelled():
            logger.warning(
                "voice_xai_job_cancelled",
                extra=context,
            )
            return
        exception = future.exception()
        if exception is not None:
            logger.exception(
                "Unhandled exception escaped a Voice XAI background job (voice_xai_job_failed).",
                exc_info=exception,
                extra=context,
            )
            return
        logger.info(
            "voice_xai_job_completed",
            extra=context,
        )

    def close(self, *, wait: bool = True) -> None:
        with self._start_lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
        if loop is None:
            return
        if self._worker_client is not None:
            self._close_worker_client(loop, wait=wait)
        loop.call_soon_threadsafe(loop.stop)
        if wait and thread is not None:
            thread.join(timeout=10)

    def _close_worker_client(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        wait: bool,
    ) -> None:
        async def _close() -> None:
            result = self._worker_client.close()
            if inspect.isawaitable(result):
                await result

        try:
            future = asyncio.run_coroutine_threadsafe(_close(), loop)
            if wait:
                future.result(timeout=10)
        except Exception:
            logger.exception("Voice XAI worker MongoDB client did not close cleanly.")


def _job_context(bundle: Any) -> dict[str, str]:
    """Build log context without making the generic queue depend on one DTO."""

    context: dict[str, str] = {}
    prediction_id = getattr(bundle, "prediction_id", None)
    request_id = getattr(bundle, "request_id", None)
    if isinstance(prediction_id, str):
        context["prediction_id"] = prediction_id
    if isinstance(request_id, str):
        context["request_id"] = request_id
    return context

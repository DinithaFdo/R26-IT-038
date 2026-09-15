import asyncio
import logging
import threading

import pytest

from app.voice_xai.capture.contracts import ExtractionBundle
from app.voice_xai.jobs.queue import AsynchronousExplanationQueue


def test_queue_runs_explanation_task_with_the_original_bundle() -> None:
    queue = AsynchronousExplanationQueue()
    bundle = ExtractionBundle(request_id="request-123", branches={})

    async def task(captured: ExtractionBundle) -> str:
        return captured.request_id

    try:
        future = queue.submit(bundle, task)

        assert future is not None
        assert future.result(timeout=5) == "request-123"
    finally:
        queue.close()


def test_queue_returns_none_instead_of_blocking_when_full() -> None:
    queue = AsynchronousExplanationQueue(max_workers=1, max_queued_jobs=0)
    bundle = ExtractionBundle(request_id="request-123", branches={})
    release_first_task = threading.Event()

    async def blocking_task(_bundle: ExtractionBundle) -> str:
        await asyncio.get_running_loop().run_in_executor(
            None, release_first_task.wait
        )
        return "done"

    async def not_run(_bundle: ExtractionBundle) -> str:
        return "not-run"

    try:
        first = queue.submit(bundle, blocking_task)
        second = queue.submit(bundle, not_run)

        assert first is not None
        assert second is None
        release_first_task.set()
        assert first.result(timeout=5) == "done"
    finally:
        queue.close()


@pytest.mark.parametrize("max_workers,max_queued_jobs", [(0, 1), (1, -1)])
def test_queue_rejects_invalid_capacity(max_workers, max_queued_jobs) -> None:
    with pytest.raises(ValueError):
        AsynchronousExplanationQueue(
            max_workers=max_workers,
            max_queued_jobs=max_queued_jobs,
        )


def test_queue_reuses_one_persistent_loop_across_multiple_jobs() -> None:
    """No ``asyncio.run()`` per job: every job observes the same running loop."""

    queue = AsynchronousExplanationQueue()
    bundle = ExtractionBundle(request_id="request-123", branches={})
    observed_loop_ids: list[int] = []

    async def record_loop(_bundle: ExtractionBundle) -> None:
        observed_loop_ids.append(id(asyncio.get_running_loop()))

    try:
        for _ in range(5):
            future = queue.submit(bundle, record_loop)
            assert future is not None
            future.result(timeout=5)
        assert len(observed_loop_ids) == 5
        assert len(set(observed_loop_ids)) == 1
    finally:
        queue.close()


def test_queue_start_is_idempotent_and_submit_auto_starts() -> None:
    queue = AsynchronousExplanationQueue()
    try:
        queue.start()
        loop_after_first_start = queue._loop  # noqa: SLF001 - white-box lifecycle check
        queue.start()
        assert queue._loop is loop_after_first_start  # noqa: SLF001

        bundle = ExtractionBundle(request_id="request-123", branches={})

        async def task(_bundle: ExtractionBundle) -> str:
            return "ok"

        future = queue.submit(bundle, task)
        assert future is not None
        assert future.result(timeout=5) == "ok"
        assert queue._loop is loop_after_first_start  # noqa: SLF001
    finally:
        queue.close()


def test_queue_close_stops_the_loop_and_rejects_further_submissions() -> None:
    queue = AsynchronousExplanationQueue()
    queue.start()
    thread = queue._thread  # noqa: SLF001 - white-box lifecycle check
    queue.close()

    assert thread is not None
    assert not thread.is_alive()

    bundle = ExtractionBundle(request_id="request-123", branches={})

    async def task(_bundle: ExtractionBundle) -> str:
        return "should-not-run"

    assert queue.submit(bundle, task) is None


def test_queue_logs_unhandled_job_exceptions_and_still_releases_capacity() -> None:
    # concurrent.futures.Future notifies .result() waiters slightly before it
    # invokes done-callbacks, so this synchronizes on the log record itself
    # (via a dedicated handler + Event) instead of racing capsys/caplog
    # against `.result()` returning on the test's own thread.
    queue = AsynchronousExplanationQueue(max_workers=1, max_queued_jobs=0)
    bundle = ExtractionBundle(request_id="request-123", branches={})
    logged = threading.Event()
    records: list[logging.LogRecord] = []

    class _RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)
            logged.set()

    target_logger = logging.getLogger("app.voice_xai.jobs.queue")
    handler = _RecordingHandler()
    target_logger.addHandler(handler)

    async def failing_task(_bundle: ExtractionBundle) -> None:
        raise RuntimeError("boom")

    async def ok_task(_bundle: ExtractionBundle) -> str:
        return "ok"

    try:
        first = queue.submit(bundle, failing_task)
        assert first is not None
        with pytest.raises(RuntimeError):
            first.result(timeout=5)

        assert logged.wait(timeout=5)
        assert any(
            "Unhandled exception" in record.getMessage() for record in records
        )

        # Capacity must have been released even though the job raised.
        second = queue.submit(bundle, ok_task)
        assert second is not None
        assert second.result(timeout=5) == "ok"
    finally:
        target_logger.removeHandler(handler)
        queue.close()


def test_queue_builds_dedicated_mongodb_client_on_worker_loop() -> None:
    created_in_thread: list[bool] = []

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def factory() -> tuple[FakeClient, str]:
        created_in_thread.append(
            threading.current_thread() is not threading.main_thread()
        )
        client = FakeClient()
        return client, "fake-database"

    queue = AsynchronousExplanationQueue(mongodb_client_factory=factory)
    try:
        queue.start()
        assert queue.worker_database == "fake-database"
        assert created_in_thread == [True]
    finally:
        client = queue._worker_client  # noqa: SLF001 - white-box lifecycle check
        queue.close()
        assert client is not None
        assert client.closed is True

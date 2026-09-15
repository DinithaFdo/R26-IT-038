import asyncio
import threading
import time

from fastapi import HTTPException
import pytest

from app.config.settings import Settings
from app.api.dependencies import get_prediction_job_runner
from app.core.exceptions import (
    PredictionQueueFullError,
    PredictionRunnerShuttingDownError,
    PrincipalPredictionLimitExceededError,
)
from app.services.prediction_job_runner import InlinePredictionJobRunner


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_inline_runner_executes_queued_job_to_completion() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            prediction_job_concurrency=1,
        )
    )

    result = await runner.run(
        prediction_id="prediction-1",
        request_id="request-1",
        persistence=persistence,
        execute=successful_job,
    )

    assert result == "completed"
    assert persistence.failures == []


def test_dependency_provider_reuses_shared_job_runner() -> None:
    get_prediction_job_runner.cache_clear()
    try:
        first_runner = get_prediction_job_runner()
        second_runner = get_prediction_job_runner()
    finally:
        get_prediction_job_runner.cache_clear()

    assert first_runner is second_runner
    assert first_runner._semaphore is second_runner._semaphore


@pytest.mark.anyio
async def test_inline_runner_timeout_marks_prediction_failed() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=0.001,
            prediction_job_concurrency=1,
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        await runner.run(
            prediction_id="prediction-timeout",
            request_id="request-timeout",
            persistence=persistence,
            execute=slow_job,
        )

    assert exc_info.value.status_code == 503
    assert persistence.failures == [
        {
            "request_id": "request-timeout",
            "stage": "execution",
            "code": "prediction_job_timeout",
        }
    ]


@pytest.mark.anyio
async def test_inline_runner_timeout_marks_blocking_prediction_failed() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=0.001,
            max_concurrent_predictions=1,
        )
    )

    async def execute_blocking_job() -> str:
        return await runner.run_blocking(lambda: time.sleep(0.05) or "too-late")

    with pytest.raises(HTTPException) as exc_info:
        await runner.run(
            prediction_id="prediction-blocking-timeout",
            request_id="request-blocking-timeout",
            persistence=persistence,
            execute=execute_blocking_job,
        )

    assert exc_info.value.status_code == 503
    assert persistence.failures == [
        {
            "request_id": "request-blocking-timeout",
            "stage": "execution",
            "code": "prediction_job_timeout",
        }
    ]


@pytest.mark.anyio
async def test_inline_runner_failure_marks_prediction_failed() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            prediction_job_concurrency=1,
        )
    )

    with pytest.raises(RuntimeError):
        await runner.run(
            prediction_id="prediction-failure",
            request_id="request-failure",
            persistence=persistence,
            execute=failing_job,
        )

    assert persistence.failures == [
        {
            "request_id": "request-failure",
            "stage": "execution",
            "code": "runtime",
        }
    ]


@pytest.mark.anyio
async def test_inline_runner_prevents_duplicate_execution() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            prediction_job_concurrency=1,
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_job() -> str:
        started.set()
        await release.wait()
        return "completed"

    first = asyncio.create_task(
        runner.run(
            prediction_id="prediction-same",
            request_id="request-same",
            persistence=persistence,
            execute=blocking_job,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(HTTPException) as exc_info:
        await runner.run(
            prediction_id="prediction-same",
            request_id="request-same-duplicate",
            persistence=persistence,
            execute=successful_job,
        )

    release.set()
    assert await first == "completed"
    assert exc_info.value.status_code == 409
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_enforces_process_concurrency_limit() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            prediction_job_concurrency=1,
        )
    )
    active_jobs = 0
    max_active_jobs = 0
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()

    async def first_job() -> str:
        nonlocal active_jobs, max_active_jobs
        active_jobs += 1
        max_active_jobs = max(max_active_jobs, active_jobs)
        first_started.set()
        await first_release.wait()
        active_jobs -= 1
        return "first"

    async def second_job() -> str:
        nonlocal active_jobs, max_active_jobs
        active_jobs += 1
        max_active_jobs = max(max_active_jobs, active_jobs)
        second_started.set()
        active_jobs -= 1
        return "second"

    first = asyncio.create_task(
        runner.run(
            prediction_id="prediction-first",
            request_id="request-first",
            persistence=persistence,
            execute=first_job,
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)
    second = asyncio.create_task(
        runner.run(
            prediction_id="prediction-second",
            request_id="request-second",
            persistence=persistence,
            execute=second_job,
        )
    )
    await asyncio.sleep(0.05)

    assert not second_started.is_set()

    first_release.set()
    assert await first == "first"
    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert await second == "second"
    assert max_active_jobs == 1
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_rejects_when_queue_capacity_is_full() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            max_concurrent_predictions=1,
            max_queued_predictions=0,
            max_active_predictions_per_principal=10,
        )
    )
    first_started = asyncio.Event()
    first_release = asyncio.Event()

    async def blocking_job() -> str:
        first_started.set()
        await first_release.wait()
        return "first"

    first = asyncio.create_task(
        runner.run(
            prediction_id="prediction-full-1",
            request_id="request-full-1",
            principal_key="user:a",
            persistence=persistence,
            execute=blocking_job,
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    with pytest.raises(PredictionQueueFullError) as exc_info:
        await runner.run(
            prediction_id="prediction-full-2",
            request_id="request-full-2",
            principal_key="user:b",
            persistence=persistence,
            execute=successful_job,
        )

    first_release.set()
    assert await first == "first"
    assert exc_info.value.error_code == "prediction_queue_full"
    assert exc_info.value.retry_after_seconds == 5
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_rejects_per_principal_concurrency_limit() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            max_concurrent_predictions=2,
            max_queued_predictions=2,
            max_active_predictions_per_principal=1,
        )
    )
    first_started = asyncio.Event()
    first_release = asyncio.Event()

    async def blocking_job() -> str:
        first_started.set()
        await first_release.wait()
        return "first"

    first = asyncio.create_task(
        runner.run(
            prediction_id="prediction-principal-1",
            request_id="request-principal-1",
            principal_key="user:same",
            persistence=persistence,
            execute=blocking_job,
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)

    with pytest.raises(PrincipalPredictionLimitExceededError) as exc_info:
        await runner.run(
            prediction_id="prediction-principal-2",
            request_id="request-principal-2",
            principal_key="user:same",
            persistence=persistence,
            execute=successful_job,
        )

    first_release.set()
    assert await first == "first"
    assert exc_info.value.error_code == "principal_prediction_limit_exceeded"
    assert exc_info.value.retry_after_seconds == 5
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_rejects_new_work_after_shutdown_starts() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(_env_file=None, prediction_job_timeout_seconds=5)
    )

    await runner.shutdown(wait=False)

    with pytest.raises(PredictionRunnerShuttingDownError) as exc_info:
        await runner.run(
            prediction_id="prediction-shutdown",
            request_id="request-shutdown",
            principal_key="user:a",
            persistence=persistence,
            execute=successful_job,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "prediction_runner_shutting_down"
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_allows_two_blocking_jobs_and_third_waits() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            max_concurrent_predictions=2,
        )
    )
    loop = asyncio.get_running_loop()
    active_jobs = 0
    max_active_jobs = 0
    active_lock = threading.Lock()
    release = threading.Event()
    started = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

    def blocking_job(index: int) -> str:
        nonlocal active_jobs, max_active_jobs
        with active_lock:
            active_jobs += 1
            max_active_jobs = max(max_active_jobs, active_jobs)
        loop.call_soon_threadsafe(started[index].set)
        try:
            release.wait(timeout=2)
            return f"job-{index}"
        finally:
            with active_lock:
                active_jobs -= 1

    async def run_job(index: int) -> str:
        return await runner.run(
            prediction_id=f"prediction-{index}",
            request_id=f"request-{index}",
            persistence=persistence,
            execute=lambda: runner.run_blocking(lambda: blocking_job(index)),
        )

    tasks = [asyncio.create_task(run_job(index)) for index in range(3)]
    await asyncio.wait_for(started[0].wait(), timeout=1)
    await asyncio.wait_for(started[1].wait(), timeout=1)
    await asyncio.sleep(0.05)

    assert not started[2].is_set()

    release.set()
    assert await asyncio.gather(*tasks) == ["job-0", "job-1", "job-2"]
    assert max_active_jobs == 2
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_releases_semaphore_after_success() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            max_concurrent_predictions=1,
        )
    )

    first = await runner.run(
        prediction_id="prediction-success-1",
        request_id="request-success-1",
        persistence=persistence,
        execute=lambda: runner.run_blocking(lambda: "first"),
    )
    second = await runner.run(
        prediction_id="prediction-success-2",
        request_id="request-success-2",
        persistence=persistence,
        execute=lambda: runner.run_blocking(lambda: "second"),
    )

    assert (first, second) == ("first", "second")
    assert persistence.failures == []


@pytest.mark.anyio
async def test_inline_runner_releases_semaphore_after_exception() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            max_concurrent_predictions=1,
        )
    )

    with pytest.raises(RuntimeError):
        await runner.run(
            prediction_id="prediction-exception",
            request_id="request-exception",
            persistence=persistence,
            execute=lambda: runner.run_blocking(failing_sync_job),
        )

    result = await runner.run(
        prediction_id="prediction-after-exception",
        request_id="request-after-exception",
        persistence=persistence,
        execute=lambda: runner.run_blocking(lambda: "released"),
    )

    assert result == "released"
    assert persistence.failures == [
        {
            "request_id": "request-exception",
            "stage": "execution",
            "code": "runtime",
        }
    ]


@pytest.mark.anyio
async def test_inline_runner_keeps_semaphore_until_cancelled_worker_finishes() -> None:
    persistence = FakePersistence()
    runner = InlinePredictionJobRunner(
        Settings(
            _env_file=None,
            prediction_job_timeout_seconds=5,
            max_concurrent_predictions=1,
        )
    )
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    release_first = threading.Event()
    second_entered_execute = asyncio.Event()

    def blocking_first() -> str:
        loop.call_soon_threadsafe(first_started.set)
        release_first.wait(timeout=2)
        return "first"

    first_task = asyncio.create_task(
        runner.run(
            prediction_id="prediction-cancelled",
            request_id="request-cancelled",
            persistence=persistence,
            execute=lambda: runner.run_blocking(blocking_first),
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    async def second_execute() -> str:
        second_entered_execute.set()
        return await runner.run_blocking(lambda: "second")

    second_task = asyncio.create_task(
        runner.run(
            prediction_id="prediction-after-cancel",
            request_id="request-after-cancel",
            persistence=persistence,
            execute=second_execute,
        )
    )
    await asyncio.sleep(0.05)
    assert not second_entered_execute.is_set()

    release_first.set()
    await asyncio.wait_for(second_entered_execute.wait(), timeout=1)
    assert await second_task == "second"
    assert persistence.failures == [
        {
            "request_id": "request-cancelled",
            "stage": "execution",
            "code": "prediction_job_cancelled",
        }
    ]


async def successful_job() -> str:
    return "completed"


async def slow_job() -> str:
    await asyncio.sleep(1)
    return "too-late"


async def failing_job() -> str:
    raise RuntimeError("boom")


def failing_sync_job() -> None:
    raise RuntimeError("boom")


class FakePersistence:
    def __init__(self) -> None:
        self.failures: list[dict[str, str]] = []

    async def mark_failed(
        self,
        request_id: str,
        *,
        stage: str,
        code: str,
    ) -> None:
        self.failures.append(
            {
                "request_id": request_id,
                "stage": stage,
                "code": code,
            }
        )

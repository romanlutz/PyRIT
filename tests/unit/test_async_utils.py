# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import assert_type

import pytest

from unit.async_utils import (
    TEST_WATCHDOG_SECONDS,
    get_defined_tasks,
    wait_for_completion_async,
    wait_for_worker_ready_async,
)


def test_get_defined_tasks_before_task_creation() -> None:
    assert get_defined_tasks(None, None) == []


async def test_get_defined_tasks_preserves_tasks_and_order_async() -> None:
    async def complete_async(value: int) -> int:
        return value

    first = asyncio.create_task(complete_async(1))
    second = asyncio.create_task(complete_async(2))
    tasks = get_defined_tasks(None, first, None, second)
    assert_type(tasks, list[asyncio.Task[int]])
    assert tasks == [first, second]
    assert await asyncio.gather(*tasks) == [1, 2]


async def test_wait_for_completion_returns_result_async() -> None:
    future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    future.set_result(42)

    assert await wait_for_completion_async(future=future) == 42


async def test_wait_for_completion_preserves_exception_async() -> None:
    future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    failure = RuntimeError("operation failed")
    future.set_exception(failure)

    with pytest.raises(RuntimeError, match="operation failed") as error:
        await wait_for_completion_async(future=future)
    assert error.value is failure


async def test_wait_for_completion_preserves_cancellation_async() -> None:
    future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    future.cancel("requested cancellation")

    with pytest.raises(asyncio.CancelledError, match="requested cancellation"):
        await wait_for_completion_async(future=future)


def test_wait_for_completion_defaults_to_a_backstop_timeout() -> None:
    # A default low enough to be reached under CI load turns the watchdog into a latency
    # assertion, failing unrelated tests with TimeoutError instead of catching a deadlock.
    parameter = inspect.signature(wait_for_completion_async).parameters["timeout"]
    assert parameter.default == 300


async def test_wait_for_completion_timeout_does_not_cancel_task_async() -> None:
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    try:
        with pytest.raises(TimeoutError, match="test watchdog expired"):
            await wait_for_completion_async(future=task, timeout=0)
        assert not task.done()
        assert task.cancelling() == 0

        release.set()
        assert await wait_for_completion_async(future=task)
    finally:
        release.set()
        await task


@pytest.mark.parametrize("max_workers", [1, 2], ids=["one-worker", "two-workers"])
@pytest.mark.parametrize(
    "interruption", [None, AssertionError, asyncio.CancelledError], ids=["completion", "assertion", "cancellation"]
)
async def test_worker_readiness_does_not_need_an_executor_worker_async(
    *, max_workers: int, interruption: type[BaseException] | None
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> int:
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(timeout=TEST_WATCHDOG_SECONDS), "Worker was not released"
            return 42
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop.set_default_executor(executor)
        operation = asyncio.create_task(asyncio.to_thread(work))
        outcome = pytest.raises(interruption, match="Interrupted while held") if interruption else nullcontext()
        with outcome:
            try:
                await wait_for_worker_ready_async(started=started, operation=operation)
                assert not operation.done()
                assert not finished.is_set()
                if interruption:
                    raise interruption("Interrupted while held")
            finally:
                release.set()
                assert await wait_for_completion_async(future=operation) == 42
        assert finished.is_set()
        assert operation.done()


@pytest.mark.parametrize("signaled", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancellation"])
async def test_worker_readiness_detects_early_completion_async(*, signaled: bool, outcome: str) -> None:
    started = asyncio.Event()
    if signaled:
        started.set()
    operation: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    failure = ValueError("worker failed before readiness")
    if outcome == "failure":
        operation.set_exception(failure)
        expected = pytest.raises(ValueError, match="worker failed before readiness")
    elif outcome == "cancellation":
        operation.cancel("worker cancelled before readiness")
        expected = pytest.raises(asyncio.CancelledError, match="worker cancelled before readiness")
    else:
        operation.set_result(42)
        expected = pytest.raises(AssertionError, match="completed before.*readiness")

    before = asyncio.all_tasks()
    with expected as error:
        await wait_for_worker_ready_async(started=started, operation=operation, timeout=0)
    if outcome == "failure":
        assert error.value is failure
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize("interruption", ["timeout", "cancellation"])
async def test_worker_readiness_interruption_only_cleans_up_its_waiter_async(interruption: str) -> None:
    operation: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    before = asyncio.all_tasks()
    if interruption == "timeout":
        with pytest.raises(TimeoutError, match="worker.*test watchdog expired"):
            await wait_for_worker_ready_async(started=asyncio.Event(), operation=operation, timeout=0)
    else:
        observer = asyncio.create_task(wait_for_worker_ready_async(started=asyncio.Event(), operation=operation))
        await asyncio.sleep(0)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
    assert not operation.done()
    assert asyncio.all_tasks() == before
    operation.set_result(42)


def test_worker_readiness_defaults_to_a_backstop_timeout() -> None:
    assert inspect.signature(wait_for_worker_ready_async).parameters["timeout"].default == TEST_WATCHDOG_SECONDS


@pytest.mark.parametrize("max_workers", [1, 2], ids=["one-worker", "two-workers"])
async def test_worker_readiness_propagates_thread_failure_before_signal_async(max_workers: int) -> None:
    failure = ValueError("worker failed before signaling")

    def fail() -> None:
        raise failure

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        asyncio.get_running_loop().set_default_executor(executor)
        operation = asyncio.create_task(asyncio.to_thread(fail))
        try:
            with pytest.raises(ValueError, match="worker failed before signaling") as error:
                await wait_for_worker_ready_async(started=asyncio.Event(), operation=operation)
            assert error.value is failure
            assert operation.done()
        finally:
            await wait_for_completion_async(future=asyncio.gather(operation, return_exceptions=True))

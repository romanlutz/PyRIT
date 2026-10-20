# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import inspect
from typing import assert_type

import pytest

from unit.async_utils import drain_tasks_async, get_defined_tasks, wait_for_completion_async


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


async def test_drain_tasks_preserves_results_and_exceptions_async() -> None:
    async def result_async() -> int:
        return 42

    async def fail_async() -> int:
        raise ValueError("worker failed")

    result, failure = await drain_tasks_async(asyncio.create_task(result_async()), asyncio.create_task(fail_async()))
    assert result == 42
    assert isinstance(failure, ValueError)
    assert str(failure) == "worker failed"


@pytest.mark.parametrize("cancel_before_cleanup", [False, True])
async def test_drain_tasks_retains_cancellation_until_worker_finishes_async(cancel_before_cleanup: bool) -> None:
    release = asyncio.Event()
    draining = asyncio.Event()
    worker = asyncio.create_task(release.wait())

    async def owner_async() -> None:
        try:
            if cancel_before_cleanup:
                raise asyncio.CancelledError("original cancellation")
        finally:
            draining.set()
            await drain_tasks_async(worker)

    owner = asyncio.create_task(owner_async())
    try:
        await draining.wait()
        owner.cancel("cleanup cancellation")
        await asyncio.sleep(0)
        owner.cancel("repeated cleanup cancellation")
        await asyncio.sleep(0)
        assert not owner.done()
        assert not worker.done()
        assert worker.cancelling() == 0
        release.set()
        message = "original cancellation" if cancel_before_cleanup else "cleanup cancellation"
        with pytest.raises(asyncio.CancelledError, match=f"^{message}$"):
            await owner
        assert worker.result() is True
    finally:
        release.set()
        await asyncio.gather(owner, worker, return_exceptions=True)


async def test_drain_tasks_timeout_reports_unfinished_worker_without_cancelling_async() -> None:
    release = asyncio.Event()
    worker = asyncio.create_task(release.wait())
    try:
        with pytest.raises(TimeoutError, match="cleanup watchdog expired"):
            await drain_tasks_async(worker, timeout=0)
        assert not worker.done()
        assert worker.cancelling() == 0
    finally:
        release.set()
        await worker


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
async def test_drain_tasks_rejects_invalid_watchdog_async(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        await drain_tasks_async(timeout=timeout)

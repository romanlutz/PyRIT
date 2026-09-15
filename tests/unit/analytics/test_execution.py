# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from time import monotonic
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from pyrit.analytics._execution import AnalyticsExecution
from pyrit.exceptions.analytics_exception import AnalyticsBusyException, AnalyticsTimeoutException

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pyrit.analytics._execution import _Lane
    from pyrit.memory.query_control import QueryControl

WAIT = 3.0


class _BlockingTask:
    def __init__(self, *, result: object = "result", error: BaseException | None = None) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.controls: list[QueryControl] = []
        self.result = result
        self.error = error
        self._lock = threading.Lock()

    def __call__(self, control: QueryControl) -> object:
        with self._lock:
            self.controls.append(control)
        self.started.set()
        try:
            assert self.release.wait(WAIT), "The test did not release its worker."
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.finished.set()


class _Harness:
    def __init__(self) -> None:
        self.executions: list[AnalyticsExecution] = []
        self.tasks: list[_BlockingTask] = []
        self.callers = ThreadPoolExecutor(max_workers=8)

    def controller(self, **options: int | float) -> AnalyticsExecution:
        defaults: dict[str, int | float] = {
            "report_workers": 1,
            "quick_workers": 1,
            "max_queue": 2,
            "queue_timeout": 10.0,
            "report_timeout": 10.0,
            "quick_timeout": 10.0,
        }
        execution = AnalyticsExecution(**(defaults | options))
        self.executions.append(execution)
        return execution

    def blocking_task(self, *, result: object = "result", error: BaseException | None = None) -> _BlockingTask:
        task = _BlockingTask(result=result, error=error)
        self.tasks.append(task)
        return task

    def submit(
        self,
        *,
        execution: AnalyticsExecution,
        key: str,
        task: Callable[[QueryControl], object],
        report: bool = True,
    ) -> Future[object]:
        return self.callers.submit(execution.run, report=report, key=key, task=task)

    def close(self) -> None:
        for task in self.tasks:
            task.release.set()
        for execution in self.executions:
            execution.shutdown()
            assert not any(thread.is_alive() for thread in execution._threads)
            for lane in execution._lanes.values():
                assert not lane.active
                assert not lane.queued
                assert not lane.ready
                assert not lane.flights
        self.callers.shutdown()


@pytest.fixture
def harness() -> Iterator[_Harness]:
    value = _Harness()
    try:
        yield value
    finally:
        value.close()


def _wait_for_state(*, execution: AnalyticsExecution, predicate: Callable[[], bool]) -> None:
    with execution._condition:
        assert execution._condition.wait_for(predicate, timeout=WAIT), "Controller state did not settle."


def _wait_for_waiters(*, execution: AnalyticsExecution, key: str, count: int, report: bool = True) -> None:
    lane = execution._lanes[report]
    _wait_for_state(
        execution=execution,
        predicate=lambda: key in lane.flights and len(lane.flights[key].waiters) == count,
    )


async def _wait_event_async(event: threading.Event) -> None:
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, WAIT), timeout=WAIT + 1)


async def _wait_for_waiters_async(*, execution: AnalyticsExecution, key: str, count: int, report: bool = True) -> None:
    await asyncio.wait_for(
        asyncio.to_thread(_wait_for_waiters, execution=execution, key=key, count=count, report=report),
        timeout=WAIT + 1,
    )


def test_constructor_defaults(harness: _Harness) -> None:
    execution = AnalyticsExecution()
    harness.executions.append(execution)

    assert execution._lanes[True].workers == 5
    assert execution._lanes[False].workers == 2
    assert execution._lanes[True].timeout == 5.0
    assert execution._lanes[False].timeout == 1.0
    assert execution._max_queue == 10
    assert execution._queue_timeout == 1.0
    assert len(execution._threads) == 8
    assert all(thread.is_alive() for thread in execution._threads)


@pytest.mark.parametrize("name", ["report_workers", "quick_workers", "max_queue"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, float("inf"), float("nan"), "1", None])
def test_constructor_rejects_invalid_limits(*, harness: _Harness, name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        harness.controller(**{name: value})


@pytest.mark.parametrize("name", ["queue_timeout", "report_timeout", "quick_timeout"])
@pytest.mark.parametrize("value", [0, -1.0, True, float("inf"), float("-inf"), float("nan"), "1", None, 10**400])
def test_constructor_rejects_invalid_timeouts(*, harness: _Harness, name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        harness.controller(**{name: value})


def test_large_finite_timeouts_are_supported(harness: _Harness) -> None:
    execution = harness.controller(queue_timeout=1e300, report_timeout=1e300, quick_timeout=1e300)
    task = harness.blocking_task()
    result = harness.submit(execution=execution, key="large", task=task)
    assert task.started.wait(WAIT)

    task.release.set()
    assert result.result(WAIT) == "result"
    assert all(thread.is_alive() for thread in execution._threads)


@pytest.mark.parametrize("report", [True, False])
@pytest.mark.parametrize("value", [None, 0, {"items": [1, 2]}])
def test_run_returns_result_without_copying_or_caching(*, harness: _Harness, report: bool, value: object) -> None:
    execution = harness.controller()
    assert execution.run(report=report, key="value", task=lambda _: value) is value
    assert execution.run(report=report, key="value", task=lambda _: "new result") == "new result"
    assert not execution._lanes[report].flights


@pytest.mark.parametrize(
    "error",
    [ValueError("query failed"), AnalyticsBusyException(), AnalyticsTimeoutException(), SystemExit("worker failed")],
)
def test_run_propagates_errors_and_keeps_worker_usable(*, harness: _Harness, error: BaseException) -> None:
    execution = harness.controller()

    def fail(_: QueryControl) -> None:
        raise error

    with pytest.raises(type(error), match=str(error)) as raised:
        execution.run(report=True, key="failure", task=fail)
    assert raised.value is error
    assert execution.run(report=True, key="failure", task=lambda _: "recovered") == "recovered"


async def test_run_results_and_errors_async(harness: _Harness) -> None:
    execution = harness.controller()
    caller_thread = threading.get_ident()
    worker_thread = await execution.run_async(report=False, key="thread", task=lambda _: threading.get_ident())
    assert worker_thread != caller_thread
    assert await execution.run_async(report=False, key="none", task=lambda _: None) is None

    def fail(_: QueryControl) -> None:
        raise ValueError("async query failed")

    with pytest.raises(ValueError, match="async query failed"):
        await execution.run_async(report=False, key="failure", task=fail)
    assert await execution.run_async(report=False, key="failure", task=lambda _: 42) == 42


def test_sync_waiters_coalesce_only_in_flight(harness: _Harness) -> None:
    execution = harness.controller()
    task = harness.blocking_task(result={"items": []})
    duplicate = MagicMock(return_value="must not run")
    first = harness.submit(execution=execution, key="shared", task=task)
    assert task.started.wait(WAIT)
    second = harness.submit(execution=execution, key="shared", task=duplicate)
    _wait_for_waiters(execution=execution, key="shared", count=2)

    task.release.set()
    assert first.result(WAIT) is task.result
    assert second.result(WAIT) is task.result
    duplicate.assert_not_called()
    assert len(task.controls) == 1
    assert execution.run(report=True, key="shared", task=lambda _: "fresh") == "fresh"


async def test_sync_and_async_waiters_share_errors_async(harness: _Harness) -> None:
    execution = harness.controller()
    task = harness.blocking_task(error=ValueError("shared failure"))
    duplicate = MagicMock(return_value="must not run")
    first = harness.submit(execution=execution, key="shared", task=task)
    await _wait_event_async(task.started)
    second = asyncio.create_task(execution.run_async(report=True, key="shared", task=duplicate))
    await _wait_for_waiters_async(execution=execution, key="shared", count=2)

    task.release.set()
    with pytest.raises(ValueError, match="shared failure"):
        await asyncio.wait_for(asyncio.wrap_future(first), timeout=WAIT)
    with pytest.raises(ValueError, match="shared failure"):
        await asyncio.wait_for(second, timeout=WAIT)
    duplicate.assert_not_called()
    assert len(task.controls) == 1


@pytest.mark.parametrize(("second_report", "second_key"), [(True, "different"), (False, "same")])
def test_different_keys_or_lanes_do_not_coalesce(*, harness: _Harness, second_report: bool, second_key: str) -> None:
    execution = harness.controller(report_workers=2 if second_report else 1)
    first_task = harness.blocking_task(result="first")
    second_task = harness.blocking_task(result="second")
    first = harness.submit(execution=execution, key="same", task=first_task)
    assert first_task.started.wait(WAIT)
    second = harness.submit(execution=execution, report=second_report, key=second_key, task=second_task)
    assert second_task.started.wait(WAIT)

    assert first_task.controls[0] is not second_task.controls[0]
    first_task.release.set()
    second_task.release.set()
    assert first.result(WAIT) == "first"
    assert second.result(WAIT) == "second"


def test_lanes_have_independent_queue_and_worker_capacity(harness: _Harness) -> None:
    execution = harness.controller(max_queue=1)
    results: list[Future[object]] = []
    for report in (True, False):
        task = harness.blocking_task()
        results.append(harness.submit(execution=execution, report=report, key="active", task=task))
        assert task.started.wait(WAIT)
        results.append(harness.submit(execution=execution, report=report, key="queued", task=lambda _: "queued"))
        _wait_for_waiters(execution=execution, report=report, key="queued", count=1)
        overflow = harness.submit(execution=execution, report=report, key="overflow", task=lambda _: "wrong")
        with pytest.raises(AnalyticsBusyException, match="busy"):
            overflow.result(WAIT)

    for task in harness.tasks:
        task.release.set()
    assert [result.result(WAIT) for result in results] == ["result", "queued", "result", "queued"]


def test_queue_is_bounded_and_fifo_even_with_new_arrivals(harness: _Harness) -> None:
    execution = harness.controller(max_queue=2)
    occupying = harness.blocking_task()
    first_queued = harness.blocking_task(result="one")
    order: list[str] = []

    def record_first(control: QueryControl) -> object:
        order.append("one")
        return first_queued(control)

    def record(value: str) -> str:
        order.append(value)
        return value

    active = harness.submit(execution=execution, key="active", task=occupying)
    assert occupying.started.wait(WAIT)
    one = harness.submit(execution=execution, key="one", task=record_first)
    _wait_for_waiters(execution=execution, key="one", count=1)
    two = harness.submit(execution=execution, key="two", task=lambda _: record("two"))
    _wait_for_waiters(execution=execution, key="two", count=1)
    overflow = harness.submit(execution=execution, key="overflow", task=lambda _: record("wrong"))
    with pytest.raises(AnalyticsBusyException, match="busy"):
        overflow.result(WAIT)

    occupying.release.set()
    assert first_queued.started.wait(WAIT)
    three = harness.submit(execution=execution, key="three", task=lambda _: record("three"))
    _wait_for_waiters(execution=execution, key="three", count=1)
    first_queued.release.set()
    assert active.result(WAIT) == "result"
    assert [result.result(WAIT) for result in (one, two, three)] == ["one", "two", "three"]
    assert order == ["one", "two", "three"]


@pytest.mark.parametrize("report", [True, False])
def test_expired_queued_work_never_runs(*, harness: _Harness, report: bool) -> None:
    execution = harness.controller(max_queue=1, queue_timeout=0.1)
    occupying = harness.blocking_task()
    queued_task = MagicMock(return_value="must not run")
    active = harness.submit(execution=execution, report=report, key="active", task=occupying)
    assert occupying.started.wait(WAIT)
    queued = harness.submit(execution=execution, report=report, key="queued", task=queued_task)

    with pytest.raises(AnalyticsBusyException, match="busy"):
        queued.result(WAIT)
    queued_task.assert_not_called()
    lane = execution._lanes[report]
    assert not lane.queued
    assert not lane.ready
    assert len(lane.active) == 1
    assert "queued" not in lane.flights
    occupying.release.set()
    assert active.result(WAIT) == "result"
    assert execution.run(report=report, key="queued", task=lambda _: "replacement") == "replacement"


@pytest.mark.parametrize("report", [True, False])
def test_running_deadline_signals_query_control(*, harness: _Harness, report: bool) -> None:
    execution = harness.controller(report_timeout=0.1, quick_timeout=0.1)
    controls: list[QueryControl] = []

    def cooperate(control: QueryControl) -> None:
        controls.append(control)
        assert control.cancel_event.wait(WAIT)
        control.check()

    result = harness.submit(execution=execution, report=report, key="deadline", task=cooperate)
    with pytest.raises(AnalyticsTimeoutException, match="timed out"):
        result.result(WAIT)
    assert len(controls) == 1
    assert controls[0].cancel_event.is_set()
    _wait_for_state(execution=execution, predicate=lambda: not execution._lanes[report].active)
    assert execution.run(report=report, key="deadline", task=lambda _: "recovered") == "recovered"


def test_unstarted_worker_reservations_expire_without_execution(harness: _Harness) -> None:
    workers_allowed = threading.Event()
    original_worker = AnalyticsExecution._worker

    def paused_worker(self: AnalyticsExecution, lane: _Lane) -> None:
        assert workers_allowed.wait(WAIT)
        original_worker(self, lane)

    with patch.object(AnalyticsExecution, "_worker", new=paused_worker):
        execution = harness.controller(max_queue=1, queue_timeout=0.1)
    task = MagicMock(return_value="must not run")
    try:
        reserved = harness.submit(execution=execution, key="reserved", task=task)
        _wait_for_waiters(execution=execution, key="reserved", count=1)
        assert len(execution._lanes[True].ready) == 1
        queued = harness.submit(execution=execution, key="queued", task=task)
        with pytest.raises(AnalyticsBusyException, match="busy"):
            reserved.result(WAIT)
        with pytest.raises(AnalyticsBusyException, match="busy"):
            queued.result(WAIT)
        assert not execution._lanes[True].active
        assert not execution._lanes[True].queued
        task.assert_not_called()
    finally:
        workers_allowed.set()
    assert execution.run(report=True, key="reserved", task=lambda _: "fresh") == "fresh"


def test_worker_rejects_expired_queue_when_monitor_has_not_run(harness: _Harness) -> None:
    with patch.object(AnalyticsExecution, "_monitor", new=lambda _: None):
        execution = harness.controller()
    occupying = harness.blocking_task()
    queued_task = MagicMock(return_value="must not run")
    active = harness.submit(execution=execution, key="active", task=occupying)
    assert occupying.started.wait(WAIT)
    queued = harness.submit(execution=execution, key="queued", task=queued_task)
    _wait_for_waiters(execution=execution, key="queued", count=1)
    with execution._condition:
        execution._lanes[True].flights["queued"].queue_deadline = monotonic() - 1

    occupying.release.set()
    assert active.result(WAIT) == "result"
    with pytest.raises(AnalyticsBusyException, match="busy"):
        queued.result(WAIT)
    queued_task.assert_not_called()


def test_worker_rejects_late_success_when_monitor_has_not_run(harness: _Harness) -> None:
    with patch.object(AnalyticsExecution, "_monitor", new=lambda _: None):
        execution = harness.controller()
    controls: list[QueryControl] = []

    def finish_late(control: QueryControl) -> str:
        controls.append(control)
        control.deadline = monotonic() - 1
        return "late success"

    with pytest.raises(AnalyticsTimeoutException, match="timed out"):
        execution.run(report=True, key="late", task=finish_late)
    assert controls[0].cancel_event.is_set()
    assert execution.run(report=True, key="late", task=lambda _: "fresh") == "fresh"


@pytest.mark.parametrize("late_error", [False, True])
def test_timeout_retains_capacity_until_task_exits(
    *, harness: _Harness, caplog: pytest.LogCaptureFixture, late_error: bool
) -> None:
    execution = harness.controller(max_queue=1, report_timeout=0.2)
    error = RuntimeError("late database failure") if late_error else None
    occupying = harness.blocking_task(error=error)
    replacement_task = harness.blocking_task(result="replacement")
    first = harness.submit(execution=execution, key="same", task=occupying)
    assert occupying.started.wait(WAIT)

    with pytest.raises(AnalyticsTimeoutException, match="timed out"):
        first.result(WAIT)
    assert occupying.controls[0].cancel_event.is_set()
    assert not occupying.finished.is_set()
    lane = execution._lanes[True]
    assert len(lane.active) == 1
    assert "same" not in lane.flights

    replacement = harness.submit(execution=execution, key="same", task=replacement_task)
    _wait_for_waiters(execution=execution, key="same", count=1)
    assert len(lane.queued) == 1
    assert not replacement_task.started.is_set()
    overflow = harness.submit(execution=execution, key="overflow", task=lambda _: "wrong")
    with pytest.raises(AnalyticsBusyException, match="busy"):
        overflow.result(WAIT)
    assert execution.run(report=False, key="same", task=lambda _: "quick") == "quick"

    released_at = monotonic()
    occupying.release.set()
    assert replacement_task.started.wait(WAIT)
    assert replacement_task.controls[0].deadline >= released_at + 0.2
    replacement_task.release.set()
    assert replacement.result(WAIT) == "replacement"
    assert execution.run(report=True, key="same", task=lambda _: "fresh") == "fresh"
    failures = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(failures) == int(late_error)
    if late_error:
        assert failures[0].exc_info is not None
        assert failures[0].exc_info[1] is error


@pytest.mark.parametrize("queued", [True, False])
async def test_deadline_errors_and_cleanup_async(*, harness: _Harness, queued: bool) -> None:
    execution = harness.controller(
        queue_timeout=0.1 if queued else 10,
        report_timeout=10 if queued else 0.1,
    )
    occupying = harness.blocking_task()
    queued_task = MagicMock(return_value="must not run")
    active = asyncio.create_task(execution.run_async(report=True, key="active", task=occupying))
    await _wait_event_async(occupying.started)
    if queued:
        with pytest.raises(AnalyticsBusyException, match="busy"):
            await asyncio.wait_for(
                execution.run_async(report=True, key="queued", task=queued_task),
                timeout=WAIT,
            )
        queued_task.assert_not_called()
        assert not occupying.controls[0].cancel_event.is_set()
    else:
        with pytest.raises(AnalyticsTimeoutException, match="timed out"):
            await asyncio.wait_for(active, timeout=WAIT)
        assert occupying.controls[0].cancel_event.is_set()
        assert len(execution._lanes[True].active) == 1

    occupying.release.set()
    if queued:
        assert await asyncio.wait_for(active, timeout=WAIT) == "result"
    assert await execution.run_async(report=True, key="active", task=lambda _: "fresh") == "fresh"


async def test_cancelling_one_waiter_keeps_sync_waiter_running_async(harness: _Harness) -> None:
    execution = harness.controller()
    task = harness.blocking_task()
    duplicate = MagicMock(return_value="must not run")
    first = harness.submit(execution=execution, key="shared", task=task)
    await _wait_event_async(task.started)
    second = asyncio.create_task(execution.run_async(report=True, key="shared", task=duplicate))
    await _wait_for_waiters_async(execution=execution, key="shared", count=2)

    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert not task.controls[0].cancel_event.is_set()
    await _wait_for_waiters_async(execution=execution, key="shared", count=1)
    task.release.set()
    assert await asyncio.wait_for(asyncio.wrap_future(first), timeout=WAIT) == "result"
    duplicate.assert_not_called()


async def test_cancelling_all_waiters_retains_running_capacity_async(harness: _Harness) -> None:
    execution = harness.controller(max_queue=1)
    task = harness.blocking_task()
    duplicate = MagicMock(return_value="must not run")
    first = asyncio.create_task(execution.run_async(report=True, key="same", task=task))
    await _wait_event_async(task.started)
    second = asyncio.create_task(execution.run_async(report=True, key="same", task=duplicate))
    await _wait_for_waiters_async(execution=execution, key="same", count=2)
    lane = execution._lanes[True]
    old_flight = lane.flights["same"]

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not task.controls[0].cancel_event.is_set()
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert task.controls[0].cancel_event.is_set()
    assert lane.active == {old_flight}
    assert not old_flight.waiters
    assert "same" not in lane.flights

    replacement = asyncio.create_task(execution.run_async(report=True, key="same", task=lambda _: "fresh"))
    await _wait_for_waiters_async(execution=execution, key="same", count=1)
    assert len(lane.queued) == 1
    with pytest.raises(AnalyticsBusyException, match="busy"):
        await execution.run_async(report=True, key="overflow", task=lambda _: "wrong")
    task.release.set()
    assert await asyncio.wait_for(replacement, timeout=WAIT) == "fresh"
    duplicate.assert_not_called()


async def test_completion_and_waiter_cancellation_race_async(harness: _Harness) -> None:
    execution = harness.controller()
    for index in range(20):
        task = harness.blocking_task(result=index)
        first = asyncio.create_task(execution.run_async(report=True, key="race", task=task))
        await _wait_event_async(task.started)
        second = asyncio.create_task(execution.run_async(report=True, key="race", task=task))
        await _wait_for_waiters_async(execution=execution, key="race", count=2)

        task.release.set()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await asyncio.wait_for(second, timeout=WAIT) == index
        assert len(task.controls) == 1
    assert not execution._lanes[True].flights
    assert len(execution._threads) == 3


async def test_cancelling_queued_waiters_reclaims_queue_without_running_async(harness: _Harness) -> None:
    execution = harness.controller(max_queue=1)
    occupying = harness.blocking_task()
    queued_task = MagicMock(return_value="must not run")
    active = harness.submit(execution=execution, key="active", task=occupying)
    await _wait_event_async(occupying.started)
    lane = execution._lanes[True]

    for _ in range(5):
        first = asyncio.create_task(execution.run_async(report=True, key="queued", task=queued_task))
        second = asyncio.create_task(execution.run_async(report=True, key="queued", task=queued_task))
        await _wait_for_waiters_async(execution=execution, key="queued", count=2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert len(lane.queued) == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert not lane.queued
        assert not lane.ready
        assert "queued" not in lane.flights

    replacement = asyncio.create_task(execution.run_async(report=True, key="queued", task=lambda _: "replacement"))
    await _wait_for_waiters_async(execution=execution, key="queued", count=1)
    occupying.release.set()
    assert await asyncio.wait_for(asyncio.wrap_future(active), timeout=WAIT) == "result"
    assert await asyncio.wait_for(replacement, timeout=WAIT) == "replacement"
    queued_task.assert_not_called()


async def test_cancelling_an_unstarted_worker_reservation_async(harness: _Harness) -> None:
    workers_allowed = threading.Event()
    original_worker = AnalyticsExecution._worker

    def paused_worker(self: AnalyticsExecution, lane: _Lane) -> None:
        assert workers_allowed.wait(WAIT)
        original_worker(self, lane)

    with patch.object(AnalyticsExecution, "_worker", new=paused_worker):
        execution = harness.controller()
    task = MagicMock(return_value="must not run")
    try:
        reserved = asyncio.create_task(execution.run_async(report=True, key="reserved", task=task))
        await _wait_for_waiters_async(execution=execution, key="reserved", count=1)
        reserved.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reserved
        assert not execution._lanes[True].active
        assert not execution._lanes[True].ready
        task.assert_not_called()
    finally:
        workers_allowed.set()
    assert await execution.run_async(report=True, key="reserved", task=lambda _: "fresh") == "fresh"


async def test_async_admission_and_execution_leave_event_loop_responsive_async(harness: _Harness) -> None:
    execution = harness.controller(max_queue=1)
    task = harness.blocking_task()
    active = asyncio.create_task(execution.run_async(report=True, key="active", task=task))
    await _wait_event_async(task.started)
    queued = asyncio.create_task(execution.run_async(report=True, key="queued", task=lambda _: "queued"))
    await _wait_for_waiters_async(execution=execution, key="queued", count=1)

    heartbeat = asyncio.Event()
    asyncio.get_running_loop().call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), timeout=1)
    assert not active.done()
    assert not queued.done()
    with pytest.raises(AnalyticsBusyException, match="busy"):
        await asyncio.wait_for(
            execution.run_async(report=True, key="overflow", task=lambda _: "wrong"),
            timeout=1,
        )
    assert await execution.run_async(report=False, key="active", task=lambda _: "quick") == "quick"
    task.release.set()
    assert await asyncio.wait_for(active, timeout=WAIT) == "result"
    assert await asyncio.wait_for(queued, timeout=WAIT) == "queued"


async def test_unexpected_error_after_all_clients_cancel_is_logged_async(
    *, harness: _Harness, caplog: pytest.LogCaptureFixture
) -> None:
    execution = harness.controller()
    error = RuntimeError("abandoned database failure")
    task = harness.blocking_task(error=error)
    result = asyncio.create_task(execution.run_async(report=True, key="abandoned", task=task))
    await _wait_event_async(task.started)
    result.cancel()
    with pytest.raises(asyncio.CancelledError):
        await result

    task.release.set()
    await asyncio.to_thread(execution.shutdown)
    failures = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(failures) == 1
    assert "abandoned" in failures[0].message
    assert failures[0].exc_info is not None
    assert failures[0].exc_info[1] is error


def test_shutdown_cancels_waiters_but_waits_for_running_work(harness: _Harness) -> None:
    execution = harness.controller()
    task = harness.blocking_task()
    queued_task = MagicMock(return_value="must not run")
    active = harness.submit(execution=execution, key="active", task=task)
    assert task.started.wait(WAIT)
    queued = harness.submit(execution=execution, key="queued", task=queued_task)
    _wait_for_waiters(execution=execution, key="queued", count=1)
    stopped = harness.callers.submit(execution.shutdown)
    _wait_for_state(execution=execution, predicate=lambda: execution._closed)

    with pytest.raises(AnalyticsTimeoutException, match="timed out"):
        active.result(WAIT)
    with pytest.raises(AnalyticsBusyException, match="busy"):
        queued.result(WAIT)
    assert not stopped.done()
    assert task.controls[0].cancel_event.is_set()
    assert len(execution._lanes[True].active) == 1
    queued_task.assert_not_called()
    task.release.set()
    stopped.result(WAIT)
    assert not any(thread.is_alive() for thread in execution._threads)
    execution.shutdown()
    with pytest.raises(AnalyticsBusyException, match="busy"):
        execution.run(report=True, key="after shutdown", task=lambda _: "wrong")


async def test_shutdown_rejects_async_callers_async(harness: _Harness) -> None:
    execution = harness.controller()
    await asyncio.to_thread(execution.shutdown)
    with pytest.raises(AnalyticsBusyException, match="busy"):
        await execution.run_async(report=False, key="closed", task=lambda _: "wrong")


def test_shutdown_from_worker_does_not_join_itself(harness: _Harness) -> None:
    execution = harness.controller()

    def stop(_: QueryControl) -> None:
        execution.shutdown()

    with pytest.raises(AnalyticsTimeoutException, match="timed out"):
        execution.run(report=True, key="stop", task=stop)
    execution.shutdown()
    assert not any(thread.is_alive() for thread in execution._threads)


def test_partial_thread_startup_failure_cleans_up() -> None:
    original_start = threading.Thread.start
    started: list[threading.Thread] = []

    def start(thread: threading.Thread) -> None:
        if started:
            raise RuntimeError("thread startup failed")
        original_start(thread)
        started.append(thread)

    with patch.object(threading.Thread, "start", new=start):
        with pytest.raises(RuntimeError, match="thread startup failed"):
            AnalyticsExecution()
    assert len(started) == 1
    assert not started[0].is_alive()

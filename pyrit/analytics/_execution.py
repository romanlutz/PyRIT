# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from sys import float_info
from time import monotonic
from typing import TYPE_CHECKING, TypeVar, cast

from pyrit.exceptions.analytics_exception import AnalyticsBusyException, AnalyticsException, AnalyticsTimeoutException
from pyrit.memory.query_control import QueryControl

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(eq=False, slots=True)
class _Flight:
    key: str
    task: Callable[[QueryControl], object]
    queue_deadline: float
    waiters: set[Future[object]] = field(default_factory=set)
    control: QueryControl | None = None
    accepting: bool = True


@dataclass(slots=True)
class _Lane:
    workers: int
    timeout: float
    ready: deque[_Flight] = field(default_factory=deque)
    queued: deque[_Flight] = field(default_factory=deque)
    active: set[_Flight] = field(default_factory=set)
    flights: dict[str, _Flight] = field(default_factory=dict)


class AnalyticsExecution:
    """
    Bound analytics work independently of the lifetime of its callers.

    Only in-flight work with the same lane and key is shared. Callers must scope
    keys by operation, query, database, and access permissions, with one result
    type per key. Results are not copied or cached. Coalesced callers share the
    original flight's deadlines. Task exceptions propagate unchanged.

    The five-report/two-quick defaults reserve quick-query capacity for the SQLite
    WAL reference workload. The p95 goals remain two seconds for reports and 500 ms
    for quick queries; deadlines are overload safeguards, not latency targets.
    """

    def __init__(
        self,
        *,
        report_workers: int = 5,
        quick_workers: int = 2,
        max_queue: int = 10,
        queue_timeout: float = 1.0,
        report_timeout: float = 5.0,
        quick_timeout: float = 1.0,
    ) -> None:
        """
        Start fixed worker pools and a single shared deadline monitor.

        Args:
            report_workers (int): Report worker slots. Defaults to 5.
            quick_workers (int): Reserved quick-query worker slots. Defaults to 2.
            max_queue (int): Additional waiting flights per lane. Defaults to 10.
            queue_timeout (float): Admission/queue deadline in seconds. Defaults to 1.
            report_timeout (float): Report execution deadline in seconds. Defaults to 5.
            quick_timeout (float): Quick-query execution deadline in seconds. Defaults to 1.

        Raises:
            ValueError: If a limit is not a positive integer or a timeout is not positive and finite.
        """
        for name, value in (
            ("report_workers", report_workers),
            ("quick_workers", quick_workers),
            ("max_queue", max_queue),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name, timeout in (
            ("queue_timeout", queue_timeout),
            ("report_timeout", report_timeout),
            ("quick_timeout", quick_timeout),
        ):
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= float_info.max:
                raise ValueError(f"{name} must be positive and finite.")

        self._condition = threading.Condition()
        self._closed = False
        self._max_queue = max_queue
        self._queue_timeout = queue_timeout
        self._lanes = {
            True: _Lane(workers=report_workers, timeout=report_timeout),
            False: _Lane(workers=quick_workers, timeout=quick_timeout),
        }
        self._threads = [
            threading.Thread(
                target=self._worker,
                args=(lane,),
                name=f"pyrit-analytics-{'report' if report else 'quick'}-{index}",
                daemon=True,
            )
            for report, lane in self._lanes.items()
            for index in range(lane.workers)
        ]
        self._threads.append(threading.Thread(target=self._monitor, name="pyrit-analytics-deadlines", daemon=True))
        try:
            for thread in self._threads:
                thread.start()
        except BaseException:
            self.shutdown()
            raise

    def run(self, *, report: bool, key: str, task: Callable[[QueryControl], T]) -> T:
        """
        Run or join a flight, blocking only the calling thread.

        Args:
            report (bool): Whether to use the report lane instead of the quick lane.
            key (str): Identity of interchangeable work and its result type.
            task (Callable[[QueryControl], T]): Synchronous, cooperatively cancellable work.

        Returns:
            T: The task's result, without copying.

        Raises:
            AnalyticsBusyException: If admission fails, queued work expires, or the controller is shut down.
            AnalyticsTimeoutException: If execution expires.
        """
        lane = self._lanes[report]
        flight, waiter = self._join(lane=lane, key=key, task=task)
        try:
            return cast("T", waiter.result())
        finally:
            self._leave(lane=lane, flight=flight, waiter=waiter)

    async def run_async(self, *, report: bool, key: str, task: Callable[[QueryControl], T]) -> T:
        """
        Run or join a flight without blocking the event loop.

        Args:
            report (bool): Whether to use the report lane instead of the quick lane.
            key (str): Identity of interchangeable work and its result type.
            task (Callable[[QueryControl], T]): Synchronous, cooperatively cancellable work.

        Returns:
            T: The task's result, without copying.

        Raises:
            AnalyticsBusyException: If admission fails, queued work expires, or the controller is shut down.
            AnalyticsTimeoutException: If execution expires.
        """
        lane = self._lanes[report]
        flight, waiter = self._join(lane=lane, key=key, task=task)
        try:
            return cast("T", await asyncio.wrap_future(waiter))
        finally:
            self._leave(lane=lane, flight=flight, waiter=waiter)

    def shutdown(self) -> None:
        """
        Reject new work, cancel outstanding flights, and join the worker threads.

        Running tasks retain their slots until they actually exit. This method is
        idempotent and waits for cooperative cancellation; call it off the event
        loop. When invoked by a worker itself, it signals shutdown without joining.
        """
        with self._condition:
            self._closed = True
            for lane in self._lanes.values():
                for flight in (*lane.queued, *lane.active):
                    error = AnalyticsBusyException() if flight.control is None else AnalyticsTimeoutException()
                    self._stop(lane=lane, flight=flight, error=error)
            self._condition.notify_all()
        if threading.current_thread() not in self._threads:
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join()

    def _join(self, *, lane: _Lane, key: str, task: Callable[[QueryControl], object]) -> tuple[_Flight, Future[object]]:
        admission_deadline = monotonic() + self._queue_timeout
        with self._condition:
            now = monotonic()
            if self._closed or now >= admission_deadline:
                raise AnalyticsBusyException
            self._expire(lane=lane, now=now)
            flight = lane.flights.get(key)
            if flight is None:
                if len(lane.active) >= lane.workers and len(lane.queued) >= self._max_queue:
                    raise AnalyticsBusyException
                flight = _Flight(key=key, task=task, queue_deadline=admission_deadline)
                lane.flights[key] = flight
                if len(lane.active) < lane.workers:
                    lane.active.add(flight)
                    lane.ready.append(flight)
                else:
                    lane.queued.append(flight)
            waiter: Future[object] = Future()
            flight.waiters.add(waiter)
            self._condition.notify_all()
            return flight, waiter

    def _leave(self, *, lane: _Lane, flight: _Flight, waiter: Future[object]) -> None:
        with self._condition:
            flight.waiters.discard(waiter)
            waiter.cancel()
            if flight.accepting and not flight.waiters:
                self._stop(lane=lane, flight=flight, error=AnalyticsTimeoutException())
                self._expire(lane=lane, now=monotonic())
            self._condition.notify_all()

    def _worker(self, lane: _Lane) -> None:
        while self._work_one(lane):
            pass

    def _work_one(self, lane: _Lane) -> bool:
        with self._condition:
            while True:
                if self._closed:
                    return False
                now = monotonic()
                self._expire(lane=lane, now=now)
                if lane.ready:
                    flight = lane.ready.popleft()
                    control = QueryControl(deadline=now + lane.timeout)
                    flight.control = control
                    self._condition.notify_all()
                    break
                self._condition.wait()
        try:
            control.check()
            result = flight.task(control)
        except BaseException as error:
            self._finish(lane=lane, flight=flight, error=error)
        else:
            self._finish(lane=lane, flight=flight, result=result)
        return True

    def _finish(
        self, *, lane: _Lane, flight: _Flight, result: object = None, error: BaseException | None = None
    ) -> None:
        with self._condition:
            if flight.control is not None and flight.control.expired and flight.accepting:
                self._stop(lane=lane, flight=flight, error=AnalyticsTimeoutException())
            delivered = self._resolve(lane=lane, flight=flight, result=result, error=error)
            lane.active.remove(flight)
            self._expire(lane=lane, now=monotonic())
            self._condition.notify_all()
        if error is not None and not delivered and not isinstance(error, AnalyticsException):
            logger.error(
                "Analytics task failed after its response was abandoned.",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _resolve(
        self, *, lane: _Lane, flight: _Flight, result: object = None, error: BaseException | None = None
    ) -> bool:
        if not flight.accepting:
            return False
        flight.accepting = False
        if lane.flights.get(flight.key) is flight:
            del lane.flights[flight.key]
        delivered = False
        for waiter in flight.waiters:
            # Cancellation can race delivery, but never cancels another waiter's future.
            if waiter.set_running_or_notify_cancel():
                delivered = True
                if error is None:
                    waiter.set_result(result)
                else:
                    waiter.set_exception(error)
        flight.waiters.clear()
        return delivered

    def _stop(self, *, lane: _Lane, flight: _Flight, error: AnalyticsException) -> None:
        if not flight.accepting:
            return
        if flight.control is not None:
            # Response/coalescing lifetime ends now; the actual worker still owns this slot.
            flight.control.cancel()
        elif flight in lane.active:
            lane.active.remove(flight)
            lane.ready.remove(flight)
        else:
            lane.queued.remove(flight)
        self._resolve(lane=lane, flight=flight, error=error)
        self._condition.notify_all()

    def _expire(self, *, lane: _Lane, now: float) -> None:
        for flight in (*lane.queued, *lane.active):
            deadline = flight.queue_deadline if flight.control is None else flight.control.deadline
            if flight.accepting and now >= deadline:
                error = AnalyticsBusyException() if flight.control is None else AnalyticsTimeoutException()
                self._stop(lane=lane, flight=flight, error=error)
        if not self._closed:
            while lane.queued and len(lane.active) < lane.workers:
                flight = lane.queued.popleft()
                lane.active.add(flight)
                lane.ready.append(flight)
                self._condition.notify_all()

    def _monitor(self) -> None:
        with self._condition:
            while not self._closed:
                now = monotonic()
                for lane in self._lanes.values():
                    self._expire(lane=lane, now=now)
                deadlines = [
                    flight.queue_deadline if flight.control is None else flight.control.deadline
                    for lane in self._lanes.values()
                    for flight in (*lane.queued, *lane.active)
                    if flight.accepting
                ]
                timeout = max(0.0, min(deadlines) - monotonic()) if deadlines else None
                if timeout is not None:
                    timeout = min(timeout, threading.TIMEOUT_MAX)
                self._condition.wait(timeout)

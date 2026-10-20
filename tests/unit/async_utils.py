# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import math
import sys
from typing import TypeVar

T = TypeVar("T")

# Deadlock backstop for correctness tests, not a latency budget.
TEST_WATCHDOG_SECONDS = 300.0


def get_defined_tasks(*tasks: asyncio.Task[T] | None) -> list[asyncio.Task[T]]:
    """Retain owned tasks even when a test fails before creating all of them."""
    return [task for task in tasks if task is not None]


async def wait_for_completion_async(*, future: asyncio.Future[T], timeout: float = TEST_WATCHDOG_SECONDS) -> T:
    """Bound a test wait without injecting cancellation into the operation under test.

    The default is a deadlock backstop rather than a latency assertion, so it stays finite to keep a
    hung test from hanging forever while remaining unreachable on a loaded CI runner. Callers that
    assert a wait has already finished pass ``timeout=0`` explicitly.
    """
    done, _ = await asyncio.wait({future}, timeout=timeout)
    if not done:
        raise TimeoutError("The operation under test did not complete before the test watchdog expired.")
    return future.result()


async def drain_tasks_async(*tasks: asyncio.Task[T], timeout: float = TEST_WATCHDOG_SECONDS) -> list[T | BaseException]:
    """Drain released workers' owning tasks before propagating cancellation.

    Callers must release worker gates first and retain tasks that still own their workers.
    A watchdog expiry reports incomplete cleanup without cancelling those tasks.
    """
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("The cleanup watchdog must be finite and non-negative.")
    completion = asyncio.gather(*tasks, return_exceptions=True)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    active_exception = sys.exception()
    cancellation = active_exception if isinstance(active_exception, asyncio.CancelledError) else None
    while not completion.done():
        try:
            done, _ = await asyncio.wait({completion}, timeout=max(0, deadline - loop.time()))
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            continue
        if not done:
            raise TimeoutError("Owned tasks did not finish before the cleanup watchdog expired.")
    results = completion.result()
    if cancellation is not None:
        raise cancellation
    return results

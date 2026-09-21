# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared admission, fairness, and cancellation contracts for manual sends."""

import asyncio

import pytest

from pyrit.backend.services.manual_send_scheduler import (
    ManualSendConflictError,
    ManualSendQueueFullError,
    ManualSendScheduler,
)


@pytest.mark.parametrize(("concurrency", "operations"), [(0, 3), (-1, 3), (4, 3)])
def test_invalid_scheduler_limits_are_rejected(concurrency: int, operations: int) -> None:
    with pytest.raises(ValueError):
        ManualSendScheduler(max_concurrency=concurrency, max_operations=operations)


def test_reservations_bound_operations_and_protect_every_branch() -> None:
    scheduler = ManualSendScheduler(max_concurrency=2, max_operations=3)
    reservation = scheduler.reserve(conversation_id="source", count=3)
    reservation.add_conversations(["branch-one", "branch-two"])
    for conversation_id in ["source", "branch-one", "branch-two"]:
        with pytest.raises(ManualSendConflictError):
            scheduler.reserve(conversation_id=conversation_id)
    with pytest.raises(ManualSendQueueFullError):
        scheduler.reserve(conversation_id="another")
    reservation.release_conversation("branch-one")
    reservation.release()
    reservation.release()
    assert scheduler._reserved == 0
    assert not scheduler._conversations
    scheduler.reserve(conversation_id="branch-one").release()


async def test_exclusive_operation_blocks_later_parallel_work_without_starvation() -> None:
    scheduler = ManualSendScheduler(max_concurrency=2, max_operations=4)
    order: list[str] = []
    release_first = asyncio.Event()
    first_started = asyncio.Event()

    async def first_async() -> None:
        async with scheduler.operation_async(exclusive=False):
            order.append("first")
            first_started.set()
            await release_first.wait()

    async def exclusive_async() -> None:
        async with scheduler.operation_async(exclusive=True):
            assert scheduler._active == 1
            order.append("exclusive")
            await asyncio.sleep(0)

    async def last_async() -> None:
        async with scheduler.operation_async(exclusive=False):
            order.append("last")

    first = asyncio.create_task(first_async())
    await asyncio.wait_for(first_started.wait(), timeout=3)
    exclusive = asyncio.create_task(exclusive_async())
    await asyncio.sleep(0)
    last = asyncio.create_task(last_async())
    await asyncio.sleep(0)
    assert order == ["first"]
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, exclusive, last), timeout=3)
    assert order == ["first", "exclusive", "last"]
    assert scheduler._active == 0
    assert not scheduler._queue


async def test_queued_cancellation_removes_ticket_and_does_not_leak_permits() -> None:
    scheduler = ManualSendScheduler(max_concurrency=1, max_operations=3)
    release = asyncio.Event()
    started = asyncio.Event()
    ran_after_cancel = asyncio.Event()

    async def occupying_async() -> None:
        async with scheduler.operation_async(exclusive=False):
            started.set()
            await release.wait()

    async def queued_async() -> None:
        async with scheduler.operation_async(exclusive=True):
            ran_after_cancel.set()

    occupying = asyncio.create_task(occupying_async())
    await asyncio.wait_for(started.wait(), timeout=3)
    queued = asyncio.create_task(queued_async())
    await asyncio.sleep(0)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    release.set()
    await occupying
    assert not ran_after_cancel.is_set()
    assert scheduler._active == 0
    assert not scheduler._queue
    async with scheduler.operation_async(exclusive=True):
        assert scheduler._active == 1


async def test_provider_failure_releases_the_exclusive_slot() -> None:
    scheduler = ManualSendScheduler(max_concurrency=2, max_operations=3)
    with pytest.raises(RuntimeError, match="provider"):
        async with scheduler.operation_async(exclusive=True):
            raise RuntimeError("provider")
    async with scheduler.operation_async(exclusive=False):
        assert scheduler._active == 1
    assert scheduler._active == 0
    assert not scheduler._exclusive

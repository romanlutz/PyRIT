# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded, process-local admission and provider budgets shared by manual sends."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache


class ManualSendConflictError(ValueError):
    """A manual operation already owns this conversation."""


class ManualSendQueueFullError(ValueError):
    """The bounded manual-send admission budget is exhausted."""


@dataclass(kw_only=True)
class ManualSendReservation:
    """An admitted operation's conversation ownership and capacity reservation."""

    scheduler: "ManualSendScheduler"
    count: int
    conversation_ids: set[str] = field(default_factory=set)
    _released: bool = False

    def add_conversations(self, conversation_ids: Sequence[str]) -> None:
        """Claim newly prepared branches before publishing them to other callers."""
        if self._released:
            raise RuntimeError("The manual-send reservation has already been released")
        if any(item in self.scheduler._conversations for item in conversation_ids):
            raise ManualSendConflictError("A send is already in progress for this conversation")
        self.scheduler._conversations.update(conversation_ids)
        self.conversation_ids.update(conversation_ids)

    def release_conversation(self, conversation_id: str) -> None:
        """Release one conversation after its branch has settled."""
        if conversation_id in self.conversation_ids:
            self.conversation_ids.remove(conversation_id)
            self.scheduler._conversations.remove(conversation_id)

    def release(self) -> None:
        """Release every remaining claim, including on preparation failure or cancellation."""
        if self._released:
            return
        self._released = True
        self.scheduler._conversations.difference_update(self.conversation_ids)
        self.conversation_ids.clear()
        self.scheduler._reserved -= self.count


class ManualSendScheduler:
    """
    Bound admitted sends and simultaneous provider work in one backend process.

    Operations with RPM limits or converters take an exclusive slot. The conservative
    converter policy also covers targets hidden inside converters, without guessing
    at their internals. FIFO admission prevents exclusive work from starving.
    """

    DEFAULT_MAX_CONCURRENCY = 4
    DEFAULT_MAX_OPERATIONS = 64

    def __init__(
        self,
        *,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_operations: int = DEFAULT_MAX_OPERATIONS,
    ) -> None:
        """Initialize bounded admission and a shared FIFO provider budget."""
        if max_concurrency < 1 or max_operations < max_concurrency:
            raise ValueError("Manual-send limits require 1 <= max_concurrency <= max_operations")
        self._max_concurrency = max_concurrency
        self._max_operations = max_operations
        self._reserved = 0
        self._conversations: set[str] = set()
        self._condition = asyncio.Condition()
        self._queue: deque[object] = deque()
        self._active = 0
        self._exclusive = False

    def reserve(self, *, conversation_id: str, count: int = 1) -> ManualSendReservation:
        """
        Admit a bounded number of logical sends without awaiting provider work.

        Returns:
            ManualSendReservation: Ownership that the caller must release.

        Raises:
            ManualSendConflictError: If the source already has an active manual mutation.
            ManualSendQueueFullError: If accepting the operation would exceed the budget.
        """
        if count < 1:
            raise ValueError("A manual-send reservation must contain at least one operation")
        if conversation_id in self._conversations:
            raise ManualSendConflictError("A send is already in progress for this conversation")
        if self._reserved + count > self._max_operations:
            raise ManualSendQueueFullError("The manual-send queue is full. Wait for an active send to finish")
        self._reserved += count
        self._conversations.add(conversation_id)
        return ManualSendReservation(scheduler=self, count=count, conversation_ids={conversation_id})

    @asynccontextmanager
    async def operation_async(self, *, exclusive: bool) -> AsyncIterator[None]:
        """Hold a provider slot, releasing it and its queue ticket on every exit path."""
        ticket = object()
        acquired = False
        async with self._condition:
            self._queue.append(ticket)
            try:
                await self._condition.wait_for(
                    lambda: (
                        self._queue[0] is ticket
                        and not self._exclusive
                        and (self._active == 0 if exclusive else self._active < self._max_concurrency)
                    )
                )
                self._queue.popleft()
                self._active += 1
                self._exclusive = exclusive
                acquired = True
                self._condition.notify_all()
            finally:
                if not acquired:
                    self._queue.remove(ticket)
                    self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._active -= 1
                if exclusive:
                    self._exclusive = False
                self._condition.notify_all()


@lru_cache(maxsize=1)
def get_manual_send_scheduler() -> ManualSendScheduler:
    """Return the shared manual-send scheduler without starting background tasks."""
    return ManualSendScheduler()

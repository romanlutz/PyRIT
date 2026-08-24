# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")


@dataclass(slots=True)
class _ProviderAttemptState:
    """Shared state for every concrete target tried during one logical send."""

    on_started: Callable[[], None] | None
    started: bool = False

    def mark_started(self) -> None:
        """Mark the logical send attempted at most once."""
        if self.started:
            return
        if self.on_started:
            self.on_started()
        self.started = True


class ProviderAttempt:
    """
    One-shot boundary immediately before an operation may deliver a request.

    A target performs all cancellable local and provider setup first, then calls
    ``start_async`` or ``run_async`` immediately before the first operation that
    may carry the current request. Starting a token repeatedly is safe.
    """

    def __init__(
        self,
        *,
        wait_for_start_async: Callable[[], Awaitable[None]],
        state: _ProviderAttemptState,
    ) -> None:
        """Initialize a provider-attempt token."""
        self._wait_for_start_async = wait_for_start_async
        self._state = state
        self._started = False
        self._lock = asyncio.Lock()

    async def start_async(self) -> None:
        """Wait for the target rate limit and mark provider delivery as started."""
        async with self._lock:
            if self._started:
                return
            await self._wait_for_start_async()
            self._state.mark_started()
            self._started = True

    async def run_async(self, *, operation: Callable[[], Awaitable[T]]) -> T:
        """
        Start this attempt and run the first request-carrying operation.

        Returns:
            T: The operation result.
        """
        await self.start_async()
        return await operation()

    def _derive(self, *, wait_for_start_async: Callable[[], Awaitable[None]]) -> ProviderAttempt:
        """
        Create a concrete-invocation token sharing this logical send state.

        Returns:
            ProviderAttempt: A fresh token with the supplied wait callback.
        """
        return ProviderAttempt(wait_for_start_async=wait_for_start_async, state=self._state)

    @classmethod
    def _legacy_started(cls) -> ProviderAttempt:
        """
        Create the no-op token used by direct legacy protected-method calls.

        Returns:
            ProviderAttempt: A token whose boundary has already started.
        """

        async def no_wait_async() -> None:
            return None

        token = cls(
            wait_for_start_async=no_wait_async,
            state=_ProviderAttemptState(on_started=None, started=True),
        )
        token._started = True
        return token


def _get_provider_attempt_or_legacy_noop(*, provider_attempt: ProviderAttempt | None) -> ProviderAttempt:
    """
    Return the orchestration token or a started token for legacy ``super()`` calls.

    Direct protected-method calls are unsupported, but migrated built-ins
    temporarily accept an omitted token so legacy overrides can call ``super()``
    after the outer compatibility bridge has already started the attempt.

    Returns:
        ProviderAttempt: The supplied token or a started compatibility token.
    """
    return provider_attempt or ProviderAttempt._legacy_started()

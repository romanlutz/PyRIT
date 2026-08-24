# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyrit.models import Message


class TargetSendContext(Protocol):
    """Internal contract coordinating one target send with caller-owned state."""

    conversation_id: str

    @property
    def provider_attempt_count(self) -> int:
        """Number of sends that reached provider invocation."""
        ...

    def begin_send(self) -> None:
        """Acquire caller-owned state for one complete send."""
        ...

    def select_history(self, *, messages: list[Message]) -> list[Message]:
        """Select the caller-approved persisted history for this send."""
        ...

    def mark_provider_attempted(self) -> None:
        """Record that provider invocation has begun."""
        ...

    def finish_send(self) -> None:
        """Release caller-owned state after the send."""
        ...

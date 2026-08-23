# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from contextvars import ContextVar, Token
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


_ACTIVE_TARGET_SEND_CONTEXT: ContextVar[TargetSendContext | None] = ContextVar(
    "_ACTIVE_TARGET_SEND_CONTEXT",
    default=None,
)


def _activate_target_send_context(*, send_context: TargetSendContext) -> Token[TargetSendContext | None]:
    """
    Expose one caller-owned context to target-side provider-boundary helpers.

    Returns:
        A token that restores the previous task-local context.
    """
    return _ACTIVE_TARGET_SEND_CONTEXT.set(send_context)


def _reset_target_send_context(*, token: Token[TargetSendContext | None]) -> None:
    """Restore the task-local target-send context after target invocation."""
    _ACTIVE_TARGET_SEND_CONTEXT.reset(token)


def _mark_active_provider_attempted() -> None:
    """Mark provider invocation for the active target send, if one exists."""
    send_context = _ACTIVE_TARGET_SEND_CONTEXT.get()
    if send_context:
        send_context.mark_provider_attempted()

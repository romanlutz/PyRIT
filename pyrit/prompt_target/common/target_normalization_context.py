# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.message_normalizer import MessageListNormalizer
    from pyrit.models import Message


class TargetNormalizationContextState(str, Enum):
    """Lifecycle state for per-send target normalization."""

    PENDING = "pending"
    PREPARING = "preparing"
    CONSUMED = "consumed"


@dataclass
class TargetNormalizationContext:
    """
    Ephemeral normalization state for one target conversation.

    The context is owned by an attack execution and passed explicitly with each
    send. It is never persisted to memory or stored on a shared target.
    """

    conversation_id: str
    normalizers: tuple[MessageListNormalizer[Message], ...]
    _state: TargetNormalizationContextState = field(
        default=TargetNormalizationContextState.PENDING,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate required context data.

        Raises:
            ValueError: If the conversation ID is empty or no normalizers are configured.
        """
        if not self.conversation_id:
            raise ValueError("conversation_id cannot be empty")
        if not self.normalizers:
            raise ValueError("At least one target normalizer is required")

    @property
    def state(self) -> TargetNormalizationContextState:
        """The current lifecycle state."""
        return self._state

    @property
    def is_pending(self) -> bool:
        """Whether normalization can be attempted."""
        return self._state == TargetNormalizationContextState.PENDING

    @property
    def is_consumed(self) -> bool:
        """Whether provider invocation has started."""
        return self._state == TargetNormalizationContextState.CONSUMED

    def begin_normalization(self, *, conversation_id: str) -> bool:
        """
        Acquire the context for normalization.

        Args:
            conversation_id: Conversation ID on the outgoing request.

        Returns:
            bool: ``True`` when the caller acquired the context, or ``False``
                after it has already been consumed.

        Raises:
            ValueError: If the request belongs to another conversation.
            RuntimeError: If another send is already preparing the first request.
        """
        if conversation_id != self.conversation_id:
            raise ValueError(
                "Target normalization context belongs to conversation "
                f"'{self.conversation_id}', not '{conversation_id}'."
            )
        if self._state == TargetNormalizationContextState.CONSUMED:
            return False
        if self._state == TargetNormalizationContextState.PREPARING:
            # Reject rather than queue so two callers cannot both believe they own
            # the first target-facing request.
            raise RuntimeError("Target normalization is already in progress for this conversation.")

        self._state = TargetNormalizationContextState.PREPARING
        return True

    async def normalize_async(self, *, messages: list[Message]) -> list[Message]:
        """
        Run the per-send normalizers while the context is acquired.

        Args:
            messages: Target conversation to normalize.

        Returns:
            list[Message]: The normalized target conversation.

        Raises:
            RuntimeError: If the context has not been acquired.
        """
        if self._state != TargetNormalizationContextState.PREPARING:
            raise RuntimeError("Target normalization context must be acquired before use.")

        normalized = list(messages)
        for normalizer in self.normalizers:
            normalized = await normalizer.normalize_async(normalized)
        return normalized

    def restore_pending(self) -> None:
        """
        Allow another attempt after pre-provider normalization fails.

        Raises:
            RuntimeError: If the context is not currently preparing.
        """
        if self._state != TargetNormalizationContextState.PREPARING:
            raise RuntimeError("Only a preparing target normalization context can be restored.")
        self._state = TargetNormalizationContextState.PENDING

    def mark_consumed(self) -> None:
        """
        Consume the context immediately before provider invocation.

        Raises:
            RuntimeError: If the context is not currently preparing.
        """
        if self._state != TargetNormalizationContextState.PREPARING:
            raise RuntimeError("Only a preparing target normalization context can be consumed.")
        self._state = TargetNormalizationContextState.CONSUMED

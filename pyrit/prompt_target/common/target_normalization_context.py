# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from typing import Any

    from pyrit.models import Message


def filter_non_replayable_messages(*, messages: list[Message]) -> list[Message]:
    """
    Remove failed request/error-response pairs from target-facing history.

    ``processing`` and ``unknown`` responses represent failed exchanges, not
    provider-authored conversation turns. ``blocked`` and ``empty`` responses
    are retained because they are real provider round trips.

    Args:
        messages: Persisted messages that may contain failed exchanges.

    Returns:
        Messages that are safe to include in a later target-facing payload.
    """
    non_replayable_errors = {"processing", "unknown"}
    excluded_indexes: set[int] = set()

    for index, error_response in enumerate(messages):
        if not any(piece.response_error in non_replayable_errors for piece in error_response.message_pieces):
            continue

        excluded_indexes.add(index)
        if index == 0:
            continue

        request = messages[index - 1]
        request_piece = request.get_piece()
        error_piece = error_response.get_piece()
        is_adjacent_request = (
            request.api_role == "user"
            and error_response.api_role == "assistant"
            and bool(request_piece.conversation_id)
            and request_piece.conversation_id == error_piece.conversation_id
            and request_piece.sequence >= 0
            and error_piece.sequence == request_piece.sequence + 1
        )
        if is_adjacent_request:
            excluded_indexes.add(index - 1)

    return [message for index, message in enumerate(messages) if index not in excluded_indexes]


@dataclass(slots=True)
class TargetNormalizationContext:
    """
    Per-execution state for adapting an explicit persisted history prefix.

    The prefix is identified by persisted message-piece IDs instead of inferred
    from response roles or error codes. A context may be used by only one send at
    a time. Stateful targets consume the prefix when provider invocation begins;
    stateless targets replay it for every send.
    """

    conversation_id: str
    history_message_ids: tuple[uuid.UUID, ...]
    replay_history_each_send: bool
    _history_consumed: bool = field(default=False, init=False, repr=False)
    _send_in_progress: bool = field(default=False, init=False, repr=False)
    _provider_attempted_task: asyncio.Task[Any] | None = field(default=None, init=False, repr=False)
    _provider_attempt_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate the persisted prefix identity.

        Raises:
            ValueError: If the conversation ID or history IDs are invalid.
        """
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        if not self.history_message_ids:
            raise ValueError("history_message_ids must not be empty")
        if len(set(self.history_message_ids)) != len(self.history_message_ids):
            raise ValueError("history_message_ids must be unique")

    @property
    def history_message_count(self) -> int:
        """Number of messages in the explicit history prefix."""
        return len(self.history_message_ids)

    @property
    def should_include_history(self) -> bool:
        """Whether the prefix should be included in the next send."""
        return self.replay_history_each_send or not self._history_consumed

    @property
    def is_consumed(self) -> bool:
        """Whether a stateful target has consumed the prefix."""
        return self._history_consumed

    @property
    def provider_attempt_count(self) -> int:
        """Number of sends that reached provider invocation."""
        return self._provider_attempt_count

    @property
    def provider_attempted_by_current_task(self) -> bool:
        """Whether the current send task reached provider invocation."""
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            return False
        return current_task is not None and current_task is self._provider_attempted_task

    def begin_send(self) -> None:
        """
        Acquire this context for one complete send.

        Raises:
            RuntimeError: If another send using this context is still active.
        """
        if self._send_in_progress:
            raise RuntimeError(
                "Concurrent sends for the same target normalization context are not supported. "
                "Wait for the active send to finish before sending another request."
            )
        self._send_in_progress = True
        self._provider_attempted_task = None

    def mark_provider_attempted(self) -> None:
        """
        Record that provider invocation has begun for the active send.

        Stateful targets consume their bootstrap history at this boundary even
        if the provider later fails or the task is cancelled.

        Raises:
            RuntimeError: If no send currently owns the context.
        """
        if not self._send_in_progress:
            raise RuntimeError("Cannot mark a provider attempt without an active send.")
        try:
            self._provider_attempted_task = asyncio.current_task()
        except RuntimeError:
            self._provider_attempted_task = None
        self._provider_attempt_count += 1
        if not self.replay_history_each_send:
            self._history_consumed = True

    def finish_send(self) -> None:
        """Release this context after the active send completes or is cancelled."""
        if not self._send_in_progress:
            return
        self._send_in_progress = False

    def select_history(self, *, messages: list[Message]) -> list[Message]:
        """
        Select the explicit persisted prefix from conversation history.

        Args:
            messages: Persisted conversation messages after failed exchanges
                have been removed.

        Returns:
            The prefix messages in their original persisted order, or an empty
            list after a stateful target has consumed the prefix.

        Raises:
            ValueError: If an expected persisted prefix message is missing.
        """
        if not self.should_include_history:
            return []

        messages_by_id = {message.get_piece().id: message for message in messages}
        missing_ids = [message_id for message_id in self.history_message_ids if message_id not in messages_by_id]
        if missing_ids:
            raise ValueError(
                "The persisted prepended history no longer matches the target normalization context. "
                f"Missing {len(missing_ids)} message(s)."
            )
        return [messages_by_id[message_id] for message_id in self.history_message_ids]

    def remap_for_duplicate_conversation(
        self,
        *,
        conversation_id: str,
        source_messages: list[Message],
        duplicated_messages: list[Message],
    ) -> TargetNormalizationContext:
        """
        Remap this explicit boundary to a duplicated conversation.

        Only message pieces already identified as history are remapped. Live turns
        copied into the new memory conversation never become part of the boundary.

        Args:
            conversation_id: Conversation ID assigned to the duplicated messages.
            source_messages: Messages from the source conversation in persisted order.
            duplicated_messages: Their duplicates in the same persisted order.

        Returns:
            A new context with boundary IDs from the duplicated conversation.
            A new logical conversation starts with an unconsumed boundary.

        Raises:
            ValueError: If the duplicated messages do not match the source structure
                or an explicit history piece cannot be remapped.
        """
        if len(source_messages) != len(duplicated_messages):
            raise ValueError("Duplicated conversation does not match the source message count.")

        duplicated_ids_by_source_id: dict[uuid.UUID, uuid.UUID] = {}
        for source_message, duplicated_message in zip(source_messages, duplicated_messages, strict=True):
            if (
                source_message.api_role != duplicated_message.api_role
                or source_message.sequence != duplicated_message.sequence
                or len(source_message.message_pieces) != len(duplicated_message.message_pieces)
            ):
                raise ValueError("Duplicated conversation does not preserve the source message structure.")

            for source_piece, duplicated_piece in zip(
                source_message.message_pieces,
                duplicated_message.message_pieces,
                strict=True,
            ):
                if (
                    source_piece.api_role != duplicated_piece.api_role
                    or source_piece.sequence != duplicated_piece.sequence
                    or duplicated_piece.conversation_id != conversation_id
                ):
                    raise ValueError("Duplicated conversation does not preserve the source piece structure.")
                duplicated_ids_by_source_id[source_piece.id] = duplicated_piece.id

        missing_ids = [
            message_id for message_id in self.history_message_ids if message_id not in duplicated_ids_by_source_id
        ]
        if missing_ids:
            raise ValueError(f"Could not remap {len(missing_ids)} explicit history message(s).")

        duplicated_context = TargetNormalizationContext(
            conversation_id=conversation_id,
            history_message_ids=tuple(
                duplicated_ids_by_source_id[message_id] for message_id in self.history_message_ids
            ),
            replay_history_each_send=self.replay_history_each_send,
        )
        # Provider bootstrap consumption belongs to the logical conversation, not copied memory.
        if conversation_id == self.conversation_id:
            duplicated_context._history_consumed = self._history_consumed
        return duplicated_context

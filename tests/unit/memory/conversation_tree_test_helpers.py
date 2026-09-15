# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real-domain fixtures and SQL observation shared by the focused tree tests."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import groupby
from typing import TYPE_CHECKING, Any

from sqlalchemy import event

from pyrit.memory.memory_models import AttackResultEntry, PromptMemoryEntry, ScoreEntry
from pyrit.models import AttackResult, ConversationReference, ConversationType, Message, MessagePiece

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pyrit.memory.sqlite_memory import SQLiteMemory
    from pyrit.models import ChatMessageRole, PromptDataType


@dataclass(frozen=True)
class CapturedTreeRead:
    """One SELECT and its bound-parameter count."""

    statement: str
    parameter_count: int
    compiled_statement: Any


def make_tree_message(
    *,
    conversation_id: str,
    sequence: int = 0,
    role: ChatMessageRole = "user",
    values: Sequence[tuple[PromptDataType, str]] = (("text", "message"),),
) -> Message:
    """Build an ordered real message with stored hashes, including media without media I/O."""
    return Message(
        message_pieces=[
            MessagePiece(
                conversation_id=conversation_id,
                sequence=sequence,
                role=role,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=sequence, microseconds=index),
                original_value=value,
                converted_value=value,
                original_value_data_type=data_type,
                converted_value_data_type=data_type,
                original_value_sha256=hashlib.sha256(value.encode("utf-8")).hexdigest(),
                converted_value_sha256=hashlib.sha256(value.encode("utf-8")).hexdigest(),
            )
            for index, (data_type, value) in enumerate(values)
        ]
    )


def clone_tree_history(*, memory: SQLiteMemory, messages: Sequence[Message]) -> tuple[str, list[Message]]:
    """Clone using the real memory/domain duplication primitive, not fabricated lineage."""
    conversation_id, pieces = memory.duplicate_messages(messages=messages)
    return conversation_id, [
        Message(message_pieces=list(group)) for _, group in groupby(pieces, key=lambda piece: piece.sequence)
    ]


def store_tree_messages(*, memory: SQLiteMemory, messages: Sequence[Message]) -> None:
    """Persist all fixture pieces in one real memory write."""
    memory.add_message_pieces_to_memory(
        message_pieces=[piece for message in messages for piece in message.message_pieces]
    )


def store_tree_attack(
    *,
    memory: SQLiteMemory,
    main_conversation_id: str,
    pruned: Sequence[str] = (),
    adversarial: Sequence[str] = (),
) -> AttackResult:
    """Persist canonical active and adversarial membership on a real AttackResult."""
    attack = AttackResult(
        conversation_id=main_conversation_id,
        objective="Test conversation tree",
        related_conversations={
            *(
                ConversationReference(conversation_id=conversation_id, conversation_type=ConversationType.PRUNED)
                for conversation_id in pruned
            ),
            *(
                ConversationReference(conversation_id=conversation_id, conversation_type=ConversationType.ADVERSARIAL)
                for conversation_id in adversarial
            ),
        },
    )
    memory.add_attack_results_to_memory(attack_results=[attack])
    return attack


@contextmanager
def capture_tree_reads(memory: SQLiteMemory) -> Generator[list[CapturedTreeRead], None, None]:
    """Observe SQL and fail if tree reads hydrate any complete message, score or attack ORM row."""
    reads: list[CapturedTreeRead] = []

    def record(*args: Any) -> None:
        statement, parameters, context = args[2:5]
        if statement.lstrip().upper().startswith("SELECT"):
            reads.append(
                CapturedTreeRead(
                    statement=statement,
                    parameter_count=len(parameters),
                    compiled_statement=context.compiled.statement if context.compiled else None,
                )
            )

    def reject_orm_load(*_args: object) -> None:
        raise AssertionError("Tree reads must not hydrate ORM message, attack or score bodies")

    event.listen(memory.engine, "before_cursor_execute", record)
    for model in (PromptMemoryEntry, AttackResultEntry, ScoreEntry):
        event.listen(model, "load", reject_orm_load)
    try:
        yield reads
    finally:
        event.remove(memory.engine, "before_cursor_execute", record)
        for model in (PromptMemoryEntry, AttackResultEntry, ScoreEntry):
            event.remove(model, "load", reject_orm_load)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded, value-free conversation topology reads and opt-in preview projections."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import and_, case, false, func, literal, or_, select
from sqlalchemy.orm import aliased

from pyrit.memory.memory_models import AttackResultEntry, PromptMemoryEntry
from pyrit.models import MEDIA_PATH_DATA_TYPES, AttackResult, ConversationReference, ConversationType

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.orm import Session
    from sqlalchemy.sql.elements import ColumnElement

    from pyrit.memory.memory_interface import MemoryInterface
    from pyrit.models import PromptDataType


@dataclass(frozen=True, slots=True)
class TreeMessageKey:
    """A stored whole-message address."""

    conversation_id: str
    sequence: int


@dataclass(frozen=True, slots=True)
class TreeScope:
    """The active, objective-target conversation membership of an attack."""

    attack_result_id: str
    main_conversation_id: str
    conversation_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class TreeConversationBoundary:
    """An append-stable upper sequence bound, captured without reading message bodies."""

    conversation_id: str
    max_sequence: int | None
    piece_count: int


@dataclass(frozen=True, slots=True)
class TreePieceProjection:
    """Only lineage, ordering and representation fingerprints for a stored piece."""

    piece_id: UUID
    lineage_id: UUID
    role: str
    original_data_type: PromptDataType
    data_type: PromptDataType
    original_hash: str
    converted_hash: str
    response_error: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class TreeMessageProjection:
    """All ordered pieces of one message; never a partial message."""

    key: TreeMessageKey
    pieces: tuple[TreePieceProjection, ...]


@dataclass(frozen=True, slots=True)
class TreePreviewPiece:
    """An excerpt and optional bounded media locator, with no ORM relationships."""

    key: TreeMessageKey
    piece_id: UUID
    data_type: PromptDataType
    text: str | None
    media_value: str | None
    converted_hash: str | None
    response_error: str


class TreeSnapshotChangedError(RuntimeError):
    """Stored snapshot data was removed or rewritten during traversal."""


class TreeReadLimitError(ValueError):
    """A single atomic message exceeds the bounded projection budget."""


class ConversationTreeReader:
    """Read tree projections through the configured memory backend's sessions."""

    MAX_MESSAGES = 100
    MAX_PREVIEW_MESSAGES = 64
    MAX_PIECES = 4096
    MAX_MEDIA_LOCATOR_LENGTH = 8192
    _ID_BATCH_SIZE = 400

    def __init__(self, *, memory: MemoryInterface) -> None:
        """Initialize a projection reader without opening a session."""
        self._memory = memory

    def get_scope(self, attack_result_id: str) -> TreeScope:
        """
        Read active membership without hydrating an attack's response or scores.

        Returns:
            TreeScope: Main plus PRUNED conversations.
        """
        with self._memory.get_session() as session:
            return self._read_scope(session=session, attack_result_id=attack_result_id)

    def capture_snapshot(self, attack_result_id: str) -> tuple[TreeScope, tuple[TreeConversationBoundary, ...]]:
        """
        Capture only membership and per-conversation sequence/count aggregates.

        Appends above these sequence bounds belong to a subsequent refresh. No
        transaction or database connection is retained across HTTP requests.

        Returns:
            tuple[TreeScope, tuple[TreeConversationBoundary, ...]]: Scope and bounds, including truly empty histories.
        """
        with self._memory.get_session() as session:
            scope = self._read_scope(session=session, attack_result_id=attack_result_id)
            bounds = self._read_boundaries(session=session, conversation_ids=sorted(scope.conversation_ids))
            return scope, bounds

    def read_message_page(
        self,
        *,
        boundaries: Sequence[TreeConversationBoundary],
        after_sequence: int | None,
        limit: int,
    ) -> tuple[TreeMessageProjection, ...]:
        """
        Read at most ``limit`` stored messages, not ``limit`` deduplicated nodes.

        The supplied boundaries are an ordered, bounded conversation window.
        Counting stored messages prevents duplicate histories from turning one
        HTTP page into an unbounded scan. Message keys are selected first, then
        all their pieces are projected together.

        Returns:
            tuple[TreeMessageProjection, ...]: Atomic, ordered messages.

        Raises:
            ValueError: If the page or conversation window is outside the bounds.
        """
        if type(limit) is not int or not 1 <= limit <= self.MAX_MESSAGES:
            raise ValueError(f"limit must be an integer from 1 through {self.MAX_MESSAGES}")
        if len(boundaries) > self.MAX_MESSAGES:
            raise ValueError("The conversation window exceeds the page budget")
        if not boundaries:
            return ()
        with self._memory.get_session() as session:
            keys = self._read_message_keys(
                session=session, boundaries=boundaries, after_sequence=after_sequence, limit=limit
            )
            if not keys:
                return ()
            return self._read_pieces(session=session, keys=keys)

    def read_previews(
        self,
        *,
        scope: TreeScope,
        messages: Sequence[TreeMessageKey],
        text_limit: int,
        include_media: bool,
    ) -> tuple[TreePreviewPiece, ...]:
        """
        Project bounded excerpts, optionally including media locators but never files.

        Text is sliced in SQL, before it enters Python. An extra character lets
        the presentation layer detect truncation, including trailing spaces.

        Returns:
            tuple[TreePreviewPiece, ...]: Every piece of the requested messages in storage order.

        Raises:
            ValueError: If the batch or excerpt budget is invalid.
            PermissionError: If a requested conversation is outside active membership.
            FileNotFoundError: If a requested message does not exist.
            TreeReadLimitError: If the batch exceeds the atomic piece budget.
        """
        if len(messages) > self.MAX_PREVIEW_MESSAGES or not 0 <= text_limit <= 200:
            raise ValueError("Preview batch or text budget exceeds the read limit")
        keys = tuple(dict.fromkeys(messages))
        if not keys:
            return ()
        if any(key.conversation_id not in scope.conversation_ids for key in keys):
            raise PermissionError("Preview conversation is not active in this attack")
        with self._memory.get_session() as session:
            rows = self._preview_query(session=session, keys=keys, text_limit=text_limit, include_media=include_media)
        if len(rows) > self.MAX_PIECES:
            raise TreeReadLimitError("Preview batch exceeds the piece budget; request fewer messages")
        found = {piece.key for piece in rows}
        if found != set(keys):
            raise FileNotFoundError("A requested message does not exist in this attack")
        return rows

    def read_media_piece(self, *, scope: TreeScope, piece_id: UUID) -> TreePreviewPiece:
        """
        Resolve one piece only after checking its conversation membership.

        Returns:
            TreePreviewPiece: The bounded media locator for a scoped piece.

        Raises:
            FileNotFoundError: If the piece no longer exists.
            PermissionError: If the piece's conversation is not active in the attack.
        """
        with self._memory.get_session() as session:
            key_row = session.execute(
                select(PromptMemoryEntry.conversation_id, PromptMemoryEntry.sequence).where(
                    PromptMemoryEntry.id == piece_id
                )
            ).one_or_none()
            if key_row is None:
                raise FileNotFoundError("Message piece does not exist")
            if key_row.conversation_id not in scope.conversation_ids:
                raise PermissionError("Message piece is not active in this attack")
            rows = self._preview_query(
                session=session,
                keys=(TreeMessageKey(conversation_id=key_row.conversation_id, sequence=key_row.sequence),),
                text_limit=0,
                include_media=True,
                piece_id=piece_id,
            )
            if not rows:
                raise FileNotFoundError("Message piece no longer exists")
            return rows[0]

    def _read_scope(self, *, session: Session, attack_result_id: str) -> TreeScope:
        row = session.execute(
            select(
                AttackResultEntry.conversation_id,
                AttackResultEntry.pruned_conversation_ids,
            ).where(AttackResultEntry.id == UUID(attack_result_id))
        ).one_or_none()
        if row is None:
            raise FileNotFoundError("Attack result does not exist")
        # Apply the domain membership policy to a projection, without constructing
        # an objective, last response, score or full AttackResult ORM entity.
        attack = AttackResult.model_construct(
            conversation_id=row.conversation_id,
            related_conversations={
                ConversationReference(conversation_id=conversation_id, conversation_type=ConversationType.PRUNED)
                for conversation_id in row.pruned_conversation_ids or ()
            },
        )
        return TreeScope(
            attack_result_id=str(UUID(attack_result_id)),
            main_conversation_id=attack.conversation_id,
            conversation_ids=frozenset(attack.get_active_conversation_ids()),
        )

    def _read_boundaries(
        self, *, session: Session, conversation_ids: Sequence[str]
    ) -> tuple[TreeConversationBoundary, ...]:
        found: dict[str, TreeConversationBoundary] = {}
        for start in range(0, len(conversation_ids), self._ID_BATCH_SIZE):
            rows = session.execute(
                select(
                    PromptMemoryEntry.conversation_id,
                    func.max(PromptMemoryEntry.sequence).label("max_sequence"),
                    func.count().label("piece_count"),
                )
                .where(PromptMemoryEntry.conversation_id.in_(conversation_ids[start : start + self._ID_BATCH_SIZE]))
                .group_by(PromptMemoryEntry.conversation_id)
            )
            for row in rows:
                found[row.conversation_id] = TreeConversationBoundary(
                    conversation_id=row.conversation_id,
                    max_sequence=row.max_sequence,
                    piece_count=row.piece_count,
                )
        return tuple(
            found.get(
                conversation_id,
                TreeConversationBoundary(conversation_id=conversation_id, max_sequence=None, piece_count=0),
            )
            for conversation_id in conversation_ids
        )

    def _read_message_keys(
        self,
        *,
        session: Session,
        boundaries: Sequence[TreeConversationBoundary],
        after_sequence: int | None,
        limit: int,
    ) -> tuple[TreeMessageKey, ...]:
        conditions = []
        for index, boundary in enumerate(boundaries):
            if boundary.max_sequence is None:
                continue
            condition = and_(
                PromptMemoryEntry.conversation_id == boundary.conversation_id,
                PromptMemoryEntry.sequence <= boundary.max_sequence,
            )
            if index == 0 and after_sequence is not None:
                condition = and_(condition, PromptMemoryEntry.sequence > after_sequence)
            conditions.append(condition)
        if not conditions:
            return ()
        order = case(
            {boundary.conversation_id: index for index, boundary in enumerate(boundaries)},
            value=PromptMemoryEntry.conversation_id,
        )
        rows = session.execute(
            select(
                PromptMemoryEntry.conversation_id,
                PromptMemoryEntry.sequence,
                func.count().label("piece_count"),
            )
            .where(or_(*conditions))
            .group_by(PromptMemoryEntry.conversation_id, PromptMemoryEntry.sequence)
            .order_by(order, PromptMemoryEntry.sequence)
            .limit(limit)
        ).all()
        keys: list[TreeMessageKey] = []
        piece_count = 0
        for row in rows:
            if row.piece_count > self.MAX_PIECES:
                if keys:
                    break
                raise TreeReadLimitError("A message exceeds the atomic tree piece budget")
            if piece_count + row.piece_count > self.MAX_PIECES:
                break
            keys.append(TreeMessageKey(conversation_id=row.conversation_id, sequence=row.sequence))
            piece_count += row.piece_count
        return tuple(keys)

    def _read_pieces(self, *, session: Session, keys: Sequence[TreeMessageKey]) -> tuple[TreeMessageProjection, ...]:
        origin = aliased(PromptMemoryEntry)
        rows = session.execute(
            select(
                PromptMemoryEntry.id,
                PromptMemoryEntry.original_prompt_id,
                PromptMemoryEntry.conversation_id,
                PromptMemoryEntry.sequence,
                PromptMemoryEntry.timestamp,
                PromptMemoryEntry.role,
                PromptMemoryEntry.original_value_data_type,
                PromptMemoryEntry.converted_value_data_type,
                PromptMemoryEntry.original_value_sha256,
                PromptMemoryEntry.converted_value_sha256,
                PromptMemoryEntry.response_error,
            )
            .outerjoin(origin, origin.id == PromptMemoryEntry.original_prompt_id)
            .where(self._message_filter(keys))
            .order_by(
                PromptMemoryEntry.conversation_id,
                PromptMemoryEntry.sequence,
                func.coalesce(origin.timestamp, PromptMemoryEntry.timestamp),
                func.coalesce(PromptMemoryEntry.original_prompt_id, PromptMemoryEntry.id),
                PromptMemoryEntry.id,
            )
            .limit(self.MAX_PIECES + 1)
        ).all()
        if len(rows) > self.MAX_PIECES:
            raise TreeReadLimitError("Messages changed or exceeded the atomic tree piece budget")
        missing = [row.id for row in rows if not row.original_value_sha256 or not row.converted_value_sha256]
        legacy_hashes = self._read_legacy_hashes(session=session, piece_ids=missing)
        grouped: dict[TreeMessageKey, list[TreePieceProjection]] = {key: [] for key in keys}
        for row in rows:
            original_hash, converted_hash = legacy_hashes.get(row.id, (None, None))
            original_hash = row.original_value_sha256 or original_hash
            converted_hash = row.converted_value_sha256 or converted_hash
            if original_hash is None or converted_hash is None:
                raise TreeSnapshotChangedError("A legacy message changed while resolving its fingerprints")
            grouped[TreeMessageKey(conversation_id=row.conversation_id, sequence=row.sequence)].append(
                TreePieceProjection(
                    piece_id=row.id,
                    lineage_id=row.original_prompt_id or row.id,
                    role=row.role,
                    original_data_type=row.original_value_data_type,
                    data_type=row.converted_value_data_type,
                    original_hash=original_hash,
                    converted_hash=converted_hash,
                    response_error=row.response_error or "none",
                    timestamp=row.timestamp,
                )
            )
        if any(not pieces for pieces in grouped.values()):
            raise TreeSnapshotChangedError("A message was removed while reading the tree; refresh the tree")
        return tuple(TreeMessageProjection(key=key, pieces=tuple(grouped[key])) for key in keys)

    def _read_legacy_hashes(
        self, *, session: Session, piece_ids: Sequence[UUID]
    ) -> dict[UUID, tuple[str | None, str | None]]:
        hashes: dict[UUID, tuple[str | None, str | None]] = {}
        for start in range(0, len(piece_ids), self._ID_BATCH_SIZE):
            rows = session.execute(
                select(
                    PromptMemoryEntry.id,
                    case(
                        (
                            func.coalesce(PromptMemoryEntry.original_value_sha256, "") == "",
                            PromptMemoryEntry.original_value,
                        ),
                        else_=None,
                    ).label("original"),
                    case(
                        (
                            func.coalesce(PromptMemoryEntry.converted_value_sha256, "") == "",
                            func.coalesce(PromptMemoryEntry.converted_value, ""),
                        ),
                        else_=None,
                    ).label("converted"),
                ).where(PromptMemoryEntry.id.in_(piece_ids[start : start + self._ID_BATCH_SIZE]))
            )
            for row in rows:
                # Hash stored strings only. In particular, never open legacy media.
                hashes[row.id] = (
                    self._hash_value(row.original) if row.original is not None else None,
                    self._hash_value(row.converted) if row.converted is not None else None,
                )
        return hashes

    def _preview_query(
        self,
        *,
        session: Session,
        keys: Sequence[TreeMessageKey],
        text_limit: int,
        include_media: bool,
        piece_id: UUID | None = None,
    ) -> tuple[TreePreviewPiece, ...]:
        origin = aliased(PromptMemoryEntry)
        substring = func.substring if session.get_bind().dialect.name == "mssql" else func.substr
        is_media = PromptMemoryEntry.converted_value_data_type.in_(sorted(MEDIA_PATH_DATA_TYPES))
        media_value = (
            case(
                (is_media, substring(PromptMemoryEntry.converted_value, 1, self.MAX_MEDIA_LOCATOR_LENGTH + 1)),
                else_=None,
            )
            if include_media
            else None
        )
        query = (
            select(
                PromptMemoryEntry.id,
                PromptMemoryEntry.conversation_id,
                PromptMemoryEntry.sequence,
                PromptMemoryEntry.converted_value_data_type,
                PromptMemoryEntry.converted_value_sha256,
                PromptMemoryEntry.response_error,
                case(
                    (~is_media, substring(PromptMemoryEntry.converted_value, 1, text_limit + 1)),
                    else_=None,
                ).label("text"),
                (media_value if media_value is not None else literal(None)).label("media_value"),
            )
            .outerjoin(origin, origin.id == PromptMemoryEntry.original_prompt_id)
            .where(self._message_filter(keys))
            .order_by(
                PromptMemoryEntry.conversation_id,
                PromptMemoryEntry.sequence,
                func.coalesce(origin.timestamp, PromptMemoryEntry.timestamp),
                func.coalesce(PromptMemoryEntry.original_prompt_id, PromptMemoryEntry.id),
                PromptMemoryEntry.id,
            )
            .limit(self.MAX_PIECES + 1)
        )
        if piece_id is not None:
            query = query.where(PromptMemoryEntry.id == piece_id)
        return tuple(
            TreePreviewPiece(
                key=TreeMessageKey(conversation_id=row.conversation_id, sequence=row.sequence),
                piece_id=row.id,
                data_type=row.converted_value_data_type,
                text=row.text,
                media_value=row.media_value,
                converted_hash=row.converted_value_sha256,
                response_error=row.response_error or "none",
            )
            for row in session.execute(query)
        )

    @staticmethod
    def _message_filter(keys: Sequence[TreeMessageKey]) -> ColumnElement[bool]:
        if not keys:
            return false()
        return or_(
            *(
                and_(
                    PromptMemoryEntry.conversation_id == key.conversation_id,
                    PromptMemoryEntry.sequence == key.sequence,
                )
                for key in keys
            )
        )

    @staticmethod
    def _hash_value(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Progressive prefix-tree traversal with bounded, transient snapshot cursors."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

from pyrit.backend.models.conversation_tree import (
    ConversationTreeEndpoint,
    ConversationTreeNode,
    ConversationTreePage,
    ConversationTreePreviewRequest,
    ConversationTreePreviewResponse,
    TreePreviewLevel,
)
from pyrit.backend.services.conversation_tree_media import ConversationTreeMedia
from pyrit.backend.services.conversation_tree_projection import project_text_previews, project_tree_node
from pyrit.memory import CentralMemory
from pyrit.memory.conversation_tree import (
    ConversationTreeReader,
    TreeConversationBoundary,
    TreeMessageKey,
    TreePreviewPiece,
    TreeScope,
    TreeSnapshotChangedError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from pyrit.backend.models.conversation_tree import ConversationTreePreview
    from pyrit.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)
_ReadResult = TypeVar("_ReadResult")


class TreeCursorExpiredError(ValueError):
    """A transient cursor expired, was evicted or fell outside the retry window."""


class TreeCursorMismatchError(ValueError):
    """A cursor was used with another attack, limit or traversal priority."""


@dataclass
class _TreeSnapshot:
    """Only traversal bookkeeping, never canonical message content."""

    scope: TreeScope
    boundaries: tuple[TreeConversationBoundary, ...]
    priority: str
    limit: int
    revision: str
    created_at: float
    offset: int = 0
    after_sequence: int | None = None
    parent_node_id: str | None = None
    consumed_pieces: int = 0
    seen_nodes: set[str] = field(default_factory=set)
    next_cursor: str | None = None
    replays: OrderedDict[str, ConversationTreePage] = field(default_factory=OrderedDict)

    @property
    def weight(self) -> int:
        """The number of retained structure records, including retry responses."""
        return (
            len(self.boundaries)
            + len(self.seen_nodes)
            + sum(len(page.nodes) + len(page.conversations) for page in self.replays.values())
        )


class ConversationTreeService:
    """Serve bounded topology first, text separately, and media only on request."""

    TEXT_PREVIEW_CHAR_LIMIT = 200
    MAX_PAGE_MESSAGES = ConversationTreeReader.MAX_MESSAGES
    _CACHE_MAX_SNAPSHOTS = 32
    _CACHE_MAX_RECORDS = 200_000
    _CACHE_TTL_SECONDS = 600
    _REPLAY_PAGES = 4

    def __init__(self, *, memory: MemoryInterface) -> None:
        """Initialize separate work budgets and an evictable in-process read cache."""
        self._memory = memory
        self._reader = ConversationTreeReader(memory=memory)
        self._cache: OrderedDict[str, _TreeSnapshot] = OrderedDict()
        self._cache_lock = Lock()
        self._topology_slots = asyncio.Semaphore(2)
        self._text_slots = asyncio.Semaphore(4)
        self._media_slots = asyncio.Semaphore(2)
        self._active_reads: set[asyncio.Task[Any]] = set()

    async def get_page_async(
        self,
        *,
        attack_result_id: str,
        cursor: str | None = None,
        limit: int = 100,
        prioritize_conversation_id: str | None = None,
    ) -> ConversationTreePage:
        """
        Read exactly one bounded storage page, with an optional active-path priority.

        Limits count scanned stored messages, including duplicates. Nodes and
        endpoints are deltas; a conversation is only emitted after its captured
        endpoint is reached. Appends and promotions are visible on the next fresh
        request, not spliced into an existing snapshot.

        Returns:
            ConversationTreePage: Useful known structure and explicit traversal completeness.

        Raises:
            ValueError: If the requested page limit is invalid.
        """
        if type(limit) is not int or not 1 <= limit <= self.MAX_PAGE_MESSAGES:
            raise ValueError(f"limit must be an integer from 1 through {self.MAX_PAGE_MESSAGES}")
        attack_result_id = str(UUID(attack_result_id))
        return await self._run_bounded_async(
            slots=self._topology_slots,
            work=lambda: asyncio.to_thread(
                self._get_page,
                attack_result_id=attack_result_id,
                cursor=cursor,
                limit=limit,
                priority=prioritize_conversation_id,
            ),
        )

    async def get_previews_async(
        self, *, attack_result_id: str, request: ConversationTreePreviewRequest
    ) -> ConversationTreePreviewResponse:
        """
        Hydrate requested messages without full-value or score ORM loading.

        All levels retain the 200-character text budget. ``full`` opts into safe
        original media URLs, not full transcripts or server-side media downloads.

        Returns:
            ConversationTreePreviewResponse: Ordered, deduplicated message previews.
        """
        attack_result_id = str(UUID(attack_result_id))
        keys = tuple(
            dict.fromkeys(
                TreeMessageKey(conversation_id=message.conversation_id, sequence=message.sequence)
                for message in request.messages
            )
        )
        slots = self._text_slots if request.level == TreePreviewLevel.TEXT else self._media_slots
        return await self._run_bounded_async(
            slots=slots,
            work=lambda: self._get_previews_async(
                attack_result_id=attack_result_id, messages=keys, level=request.level
            ),
        )

    async def get_thumbnail_async(self, *, attack_result_id: str, piece_id: UUID) -> bytes:
        """
        Render a small local image after a fresh active-membership check.

        Returns:
            bytes: A bounded PNG, never a redirected original asset.
        """
        attack_result_id = str(UUID(attack_result_id))
        return await self._run_bounded_async(
            slots=self._media_slots,
            work=lambda: asyncio.to_thread(
                self._render_thumbnail, attack_result_id=attack_result_id, piece_id=piece_id
            ),
        )

    async def _run_bounded_async(
        self, *, slots: asyncio.Semaphore, work: Callable[[], Coroutine[Any, Any, _ReadResult]]
    ) -> _ReadResult:
        await slots.acquire()
        abandoned = False
        task = asyncio.create_task(work())
        self._active_reads.add(task)

        def finished(completed: asyncio.Task[_ReadResult]) -> None:
            slots.release()
            self._active_reads.discard(completed)
            if not completed.cancelled():
                error = completed.exception()
                if abandoned and error is not None:
                    logger.warning(
                        "Abandoned conversation-tree read failed",
                        exc_info=(type(error), error, error.__traceback__),
                    )

        task.add_done_callback(finished)
        try:
            # A cancelled HTTP waiter cannot stop a worker thread. The task owns
            # its permit until the actual work, including media signing, finishes.
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            abandoned = True
            raise

    async def _get_previews_async(
        self, *, attack_result_id: str, messages: Sequence[TreeMessageKey], level: TreePreviewLevel
    ) -> ConversationTreePreviewResponse:
        pieces = await asyncio.to_thread(
            self._read_previews,
            attack_result_id=attack_result_id,
            messages=messages,
            include_media=level != TreePreviewLevel.TEXT,
        )
        previews = project_text_previews(messages=messages, pieces=pieces, text_budget=self.TEXT_PREVIEW_CHAR_LIMIT)
        if level != TreePreviewLevel.TEXT:
            await self._add_media_async(
                attack_result_id=attack_result_id, pieces=pieces, previews=previews, level=level
            )
        return ConversationTreePreviewResponse(previews=previews)

    def _get_page(
        self, *, attack_result_id: str, cursor: str | None, limit: int, priority: str | None
    ) -> ConversationTreePage:
        with self._cache_lock:
            self._evict_expired()
            if cursor is None:
                snapshot = self._new_snapshot(attack_result_id=attack_result_id, limit=limit, priority=priority)
            else:
                snapshot = self._resolve_cursor(
                    attack_result_id=attack_result_id, cursor=cursor, limit=limit, priority=priority
                )
                replay = snapshot.replays.get(cursor)
                if replay is not None:
                    return replay.model_copy(deep=True)
            page = self._advance_snapshot(snapshot)
            if cursor is not None:
                snapshot.replays[cursor] = page.model_copy(deep=True)
                while len(snapshot.replays) > self._REPLAY_PAGES:
                    snapshot.replays.popitem(last=False)
            self._cache[snapshot.revision] = snapshot
            self._cache.move_to_end(snapshot.revision)
            self._evict_over_budget()
            if snapshot.revision not in self._cache and not page.complete:
                raise TreeCursorExpiredError("Tree snapshot exceeds the cache budget; refresh the tree")
            return page

    def _new_snapshot(self, *, attack_result_id: str, limit: int, priority: str | None) -> _TreeSnapshot:
        scope, boundaries = self._reader.capture_snapshot(attack_result_id)
        priority = priority if priority is not None else scope.main_conversation_id
        if priority not in scope.conversation_ids:
            raise PermissionError("Priority conversation is not active in this attack")
        return _TreeSnapshot(
            scope=scope,
            boundaries=tuple(
                sorted(boundaries, key=lambda bound: (bound.conversation_id != priority, bound.conversation_id))
            ),
            priority=priority,
            limit=limit,
            revision=uuid4().hex,
            created_at=time.monotonic(),
        )

    def _resolve_cursor(self, *, attack_result_id: str, cursor: str, limit: int, priority: str | None) -> _TreeSnapshot:
        parts = cursor.split(".")
        if len(parts) != 2 or len(parts[0]) != 32 or len(parts[1]) != 32:
            raise TreeCursorMismatchError("Invalid conversation-tree cursor")
        snapshot = self._cache.get(parts[0])
        if snapshot is None:
            raise TreeCursorExpiredError("Tree cursor expired or was evicted; refresh the tree")
        if snapshot.scope.attack_result_id != attack_result_id or limit != snapshot.limit:
            raise TreeCursorMismatchError("Tree cursor does not match this attack and limit")
        if priority is not None and priority != snapshot.priority:
            raise TreeCursorMismatchError("Tree cursor does not match the requested conversation priority")
        if cursor != snapshot.next_cursor and cursor not in snapshot.replays:
            raise TreeCursorExpiredError(
                "Tree cursor is invalid or outside the retained retry window; refresh the tree"
            )
        scope = self._reader.get_scope(attack_result_id)
        if not snapshot.scope.conversation_ids <= scope.conversation_ids:
            raise TreeSnapshotChangedError("Attack membership changed; refresh the tree")
        return snapshot

    def _advance_snapshot(self, snapshot: _TreeSnapshot) -> ConversationTreePage:
        window = snapshot.boundaries[snapshot.offset : snapshot.offset + snapshot.limit]
        messages = self._reader.read_message_page(
            boundaries=window, after_sequence=snapshot.after_sequence, limit=snapshot.limit
        )
        nodes: list[ConversationTreeNode] = []
        endpoints: list[ConversationTreeEndpoint] = []
        new_ids: set[str] = set()
        offset = snapshot.offset
        after_sequence = snapshot.after_sequence
        parent = snapshot.parent_node_id
        consumed_pieces = snapshot.consumed_pieces
        message_index = 0
        for boundary in window:
            if boundary.max_sequence is not None:
                if message_index == len(messages):
                    if message_index == 0:
                        raise TreeSnapshotChangedError("Snapshot messages were removed; refresh the tree")
                    break
                if messages[message_index].key.conversation_id != boundary.conversation_id:
                    raise TreeSnapshotChangedError("Snapshot conversation was removed; refresh the tree")
                while (
                    message_index < len(messages)
                    and messages[message_index].key.conversation_id == boundary.conversation_id
                ):
                    message = messages[message_index]
                    node = project_tree_node(message=message, parent_node_id=parent)
                    if node.node_id not in snapshot.seen_nodes and node.node_id not in new_ids:
                        nodes.append(node)
                        new_ids.add(node.node_id)
                    parent = node.node_id
                    after_sequence = message.key.sequence
                    consumed_pieces += len(message.pieces)
                    message_index += 1
                if after_sequence != boundary.max_sequence:
                    break
                if consumed_pieces != boundary.piece_count:
                    raise TreeSnapshotChangedError("Snapshot message pieces changed; refresh the tree")
            endpoints.append(ConversationTreeEndpoint(conversation_id=boundary.conversation_id, node_id=parent))
            offset += 1
            after_sequence, parent, consumed_pieces = None, None, 0
        complete = offset == len(snapshot.boundaries)
        next_cursor = None if complete else f"{snapshot.revision}.{secrets.token_urlsafe(24)}"
        page = ConversationTreePage(
            attack_result_id=snapshot.scope.attack_result_id,
            main_conversation_id=snapshot.scope.main_conversation_id,
            revision=snapshot.revision,
            nodes=nodes,
            conversations=endpoints,
            processed_conversations=offset,
            total_conversations=len(snapshot.boundaries),
            next_cursor=next_cursor,
            complete=complete,
        )
        snapshot.offset = offset
        snapshot.after_sequence = after_sequence
        snapshot.parent_node_id = parent
        snapshot.consumed_pieces = consumed_pieces
        snapshot.seen_nodes.update(new_ids)
        snapshot.next_cursor = next_cursor
        return page

    def _read_previews(
        self, *, attack_result_id: str, messages: Sequence[TreeMessageKey], include_media: bool
    ) -> tuple[TreePreviewPiece, ...]:
        scope = self._reader.get_scope(attack_result_id)
        return self._reader.read_previews(
            scope=scope, messages=messages, text_limit=self.TEXT_PREVIEW_CHAR_LIMIT, include_media=include_media
        )

    async def _add_media_async(
        self,
        *,
        attack_result_id: str,
        pieces: Sequence[TreePreviewPiece],
        previews: Sequence[ConversationTreePreview],
        level: TreePreviewLevel,
    ) -> None:
        media = ConversationTreeMedia(results_path=self._memory.results_path)
        descriptors = await asyncio.to_thread(
            lambda: {
                str(piece.piece_id): media.describe(
                    attack_result_id=attack_result_id, piece=piece, full=level == TreePreviewLevel.FULL
                )
                for piece in pieces
                if piece.media_value is not None
            }
        )
        for preview in previews:
            for piece in preview.pieces:
                descriptor = descriptors.get(piece.piece_id)
                if descriptor is None:
                    continue
                piece.thumbnail_url = descriptor.thumbnail_url
                piece.mime_type = descriptor.mime_type
                piece.filename = descriptor.filename
                if descriptor.media_url is not None:
                    from pyrit.backend.mappers.attack_mappers import _sign_blob_url_async

                    piece.media_url = await _sign_blob_url_async(blob_url=descriptor.media_url)

    def _render_thumbnail(self, *, attack_result_id: str, piece_id: UUID) -> bytes:
        scope = self._reader.get_scope(attack_result_id)
        piece = self._reader.read_media_piece(scope=scope, piece_id=piece_id)
        return ConversationTreeMedia(results_path=self._memory.results_path).render_thumbnail(piece)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            revision
            for revision, snapshot in self._cache.items()
            if now - snapshot.created_at >= self._CACHE_TTL_SECONDS
        ]
        for revision in expired:
            del self._cache[revision]

    def _evict_over_budget(self) -> None:
        weight = sum(snapshot.weight for snapshot in self._cache.values())
        while self._cache and (len(self._cache) > self._CACHE_MAX_SNAPSHOTS or weight > self._CACHE_MAX_RECORDS):
            _, removed = self._cache.popitem(last=False)
            weight -= removed.weight


_service: ConversationTreeService | None = None


def get_conversation_tree_service() -> ConversationTreeService:
    """
    Get the tree service for the current CentralMemory instance.

    Returns:
        ConversationTreeService: A reusable service with bounded transient cursors.
    """
    global _service
    memory = CentralMemory.get_memory_instance()
    if _service is None or _service._memory is not memory:
        _service = ConversationTreeService(memory=memory)
    return _service

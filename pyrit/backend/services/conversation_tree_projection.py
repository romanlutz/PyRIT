# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pure prefix identity and all-piece preview presentation."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pyrit.backend.models.conversation_tree import (
    ConversationTreeNode,
    ConversationTreePiecePreview,
    ConversationTreePreview,
    TreeMessageReference,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.memory.conversation_tree import TreeMessageKey, TreeMessageProjection, TreePreviewPiece


def project_tree_node(*, message: TreeMessageProjection, parent_node_id: str | None) -> ConversationTreeNode:
    """
    Build a prefix-scoped lineage identity and a parent-independent preview key.

    Returns:
        ConversationTreeNode: One content-free node for all ordered message pieces.

    Raises:
        ValueError: If stored pieces disagree about the message role.
    """
    if len({piece.role for piece in message.pieces}) != 1:
        raise ValueError("Stored pieces have inconsistent roles within a message")
    representations = [
        [
            piece.role,
            piece.original_data_type,
            piece.original_hash,
            piece.data_type,
            piece.converted_hash,
            piece.response_error,
        ]
        for piece in message.pieces
    ]
    preview_key = _digest(representations)
    node_id = _digest(
        [parent_node_id, message.key.sequence, [str(piece.lineage_id) for piece in message.pieces], preview_key]
    )
    return ConversationTreeNode(
        node_id=node_id,
        parent_node_id=parent_node_id,
        message=TreeMessageReference(conversation_id=message.key.conversation_id, sequence=message.key.sequence),
        role=message.pieces[0].role,
        piece_types=[piece.data_type for piece in message.pieces],
        piece_count=len(message.pieces),
        preview_key=preview_key,
        created_at=min(piece.timestamp for piece in message.pieces),
    )


def project_text_previews(
    *, messages: Sequence[TreeMessageKey], pieces: Sequence[TreePreviewPiece], text_budget: int
) -> list[ConversationTreePreview]:
    """
    Share an exact character budget across all text pieces of each message.

    Returns:
        list[ConversationTreePreview]: Deduplicated, input-ordered previews, including non-text and exhausted pieces.
    """
    grouped: dict[TreeMessageKey, list[TreePreviewPiece]] = {key: [] for key in dict.fromkeys(messages)}
    for piece in pieces:
        grouped[piece.key].append(piece)
    previews = []
    for key, message_pieces in grouped.items():
        remaining = text_budget
        preview_pieces = []
        for piece in message_pieces:
            text = piece.text[:remaining] if piece.text is not None else None
            remaining -= len(text or "")
            preview_pieces.append(
                ConversationTreePiecePreview(
                    piece_id=str(piece.piece_id),
                    data_type=piece.data_type,
                    text=text,
                    truncated=piece.text is not None and len(piece.text) > len(text or ""),
                    media_url=None,
                    thumbnail_url=None,
                    mime_type=None,
                    filename=None,
                    response_error=piece.response_error,
                )
            )
        previews.append(
            ConversationTreePreview(
                message=TreeMessageReference(conversation_id=key.conversation_id, sequence=key.sequence),
                pieces=preview_pieces,
            )
        )
    return previews


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()

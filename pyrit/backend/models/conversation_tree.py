# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Lean topology and explicitly requested preview contracts for the message tree."""

from datetime import datetime
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class TreeMessageReference(BaseModel):
    """A real stored message, addressed by conversation and sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(strict=True)


class ConversationTreeNode(BaseModel):
    """One whole message in a prefix tree, without its content."""

    node_id: str
    parent_node_id: str | None
    message: TreeMessageReference
    role: str
    piece_types: list[str]
    piece_count: int
    preview_key: str
    created_at: datetime


class ConversationTreeEndpoint(BaseModel):
    """A known complete conversation endpoint; null denotes a truly empty history."""

    conversation_id: str
    node_id: str | None


class ConversationTreePage(BaseModel):
    """An incremental topology delta within an append-stable snapshot."""

    attack_result_id: str
    main_conversation_id: str
    revision: str
    nodes: list[ConversationTreeNode]
    conversations: list[ConversationTreeEndpoint]
    processed_conversations: int
    total_conversations: int
    next_cursor: str | None
    complete: bool


class TreePreviewLevel(str, Enum):
    """Increasingly explicit levels of content access."""

    TEXT = "text"
    THUMBNAIL = "thumbnail"
    FULL = "full"


class ConversationTreePreviewRequest(BaseModel):
    """A bounded batch of requested messages, not a transcript request."""

    MAX_MESSAGES: ClassVar[int] = 64
    model_config = ConfigDict(extra="forbid")

    messages: list[TreeMessageReference] = Field(min_length=1, max_length=MAX_MESSAGES)
    level: TreePreviewLevel


class ConversationTreePiecePreview(BaseModel):
    """One piece, retained even when the shared text budget is exhausted."""

    piece_id: str
    data_type: str
    text: str | None
    truncated: bool
    media_url: str | None
    thumbnail_url: str | None
    mime_type: str | None
    filename: str | None
    response_error: str


class ConversationTreePreview(BaseModel):
    """An ordered representation of every piece of a requested message."""

    message: TreeMessageReference
    pieces: list[ConversationTreePiecePreview]


class ConversationTreePreviewResponse(BaseModel):
    """The deduplicated batch of requested previews."""

    previews: list[ConversationTreePreview]

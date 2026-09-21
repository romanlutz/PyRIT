# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Compact, transient progress contracts for manual multi-send."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from pyrit.backend.models.attacks import AddMessageRequest


class RequestConverterMode(StrEnum):
    """How request conversion is applied to the branches."""

    SHARED = "shared"
    PER_BRANCH = "per_branch"


class MessageBatchState(StrEnum):
    """Live state of one accepted submission."""

    PREPARING = "preparing"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageBatchBranchState(StrEnum):
    """Live state of one persisted conversation in a batch."""

    QUEUED = "queued"
    SENDING = "sending"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageBatchRequest(AddMessageRequest):
    """Send one user message to the source and count-minus-one fresh history copies."""

    count: int = Field(..., strict=True, ge=1, le=10)
    request_converter_mode: RequestConverterMode = RequestConverterMode.SHARED
    submission_id: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_batch(self) -> Self:
        if self.role != "user" or not self.send:
            raise ValueError("Message batches require role=user and send=true")
        if not self.pieces:
            raise ValueError("Message batches require at least one message piece")
        if not self.submission_id.strip():
            raise ValueError("submission_id must not be blank")
        if not self.target_registry_name or not self.target_registry_name.strip():
            raise ValueError("target_registry_name is required for a message batch")
        if not self.target_conversation_id.strip():
            raise ValueError("target_conversation_id must not be blank")
        return self


class MessageBatchBranch(BaseModel):
    """References to new evidence, never the conversation's full transcript."""

    conversation_id: str
    state: MessageBatchBranchState = MessageBatchBranchState.QUEUED
    new_message_piece_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class MessageBatchStatus(BaseModel):
    """Worker-local progress; persisted conversation membership belongs to AttackResult."""

    batch_id: str
    attack_result_id: str
    source_conversation_id: str
    requested_count: int
    state: MessageBatchState = MessageBatchState.QUEUED
    branches: list[MessageBatchBranch] = Field(default_factory=list)
    error: str | None = None

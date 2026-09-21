# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One transient send-operation contract for single and repeated manual sends."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator

from pyrit.backend.models.attacks import AddMessageRequest


class RequestConverterMode(StrEnum):
    """How request conversion is applied to the branches."""

    SHARED = "shared"
    PER_BRANCH = "per_branch"


class MessageSendState(StrEnum):
    """Live state of one accepted submission."""

    PREPARING = "preparing"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageSendBranchState(StrEnum):
    """Live state of one conversation in a send operation."""

    QUEUED = "queued"
    SENDING = "sending"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageSendFailureStage(StrEnum):
    """Where an accepted operation failed, without implying it is safe to replay."""

    PREPARATION = "preparation"
    SENDING = "sending"
    FINALIZATION = "finalization"
    INTERRUPTED = "interrupted"


class MessageSendRequest(AddMessageRequest):
    """Send one message, optionally repeating a user message in fresh history copies."""

    count: int = Field(default=1, strict=True, ge=1, le=10)
    request_converter_mode: RequestConverterMode = RequestConverterMode.SHARED
    submission_id: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_send(self) -> Self:
        if not self.send:
            raise ValueError("Message sends require send=true; use the messages endpoint to store context")
        if self.count > 1 and self.role != "user":
            raise ValueError("Repeated sends require role=user")
        if not self.pieces:
            raise ValueError("Message sends require at least one message piece")
        if not self.submission_id.strip():
            raise ValueError("submission_id must not be blank")
        if not self.target_registry_name or not self.target_registry_name.strip():
            raise ValueError("target_registry_name is required when send=True")
        if not self.target_conversation_id.strip():
            raise ValueError("target_conversation_id must not be blank")
        return self


class MessageSendBranch(BaseModel):
    """The state of one conversation, never its full transcript."""

    conversation_id: str
    state: MessageSendBranchState = MessageSendBranchState.QUEUED
    error: str | None = None


class MessageSendStatus(BaseModel):
    """Worker-local progress; persisted conversation membership belongs to AttackResult."""

    send_id: str
    attack_result_id: str
    source_conversation_id: str
    requested_count: int
    state: MessageSendState = MessageSendState.QUEUED
    branches: list[MessageSendBranch] = Field(default_factory=list)
    error: str | None = None
    failure_stage: MessageSendFailureStage | None = None

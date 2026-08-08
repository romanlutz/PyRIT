# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Immutable models for capability-task execution."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from pyrit.models import (  # noqa: TC001 (runtime-required by Pydantic)
    JSONValue,
    Message,
    Score,
    TargetIdentifier,
    TokenUsage,
)


class CapabilityTerminationReason(str, Enum):
    """The reason a capability-task run stopped."""

    COMPLETION = "completion"
    LIMIT = "limit"
    DENIAL = "denial"
    CANCELLATION = "cancellation"
    FAILURE = "failure"


class CapabilityOutcome(str, Enum):
    """The high-level outcome of a capability-task run."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    DENIED = "denied"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ToolExecutionPolicy(str, Enum):
    """How tool calls from one model generation are executed."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ApprovalDecisionKind(str, Enum):
    """A policy decision for a requested tool call."""

    ALLOW = "allow"
    DENY = "deny"


class ToolExecutionStatus(str, Enum):
    """The authoritative execution status of a tool call."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    NOT_EXECUTED = "not_executed"


class LifecycleEventKind(str, Enum):
    """A non-conversational lifecycle fact."""

    CANCELLATION = "cancellation"
    TRUNCATION = "truncation"


class CapabilitySource(BaseModel):
    """Source and provenance metadata for a capability task or case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: str
    source_id: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class CapabilityLimits(BaseModel):
    """Execution budgets enforced by the capability executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_turns: int | None = Field(default=None, gt=0)
    max_model_generations: int = Field(default=16, gt=0)
    max_wall_clock_seconds: float = Field(default=300.0, gt=0)
    max_tool_calls: int = Field(default=64, ge=0)
    max_consecutive_no_progress: int = Field(default=3, gt=0)
    max_consecutive_errors: int = Field(default=3, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, gt=0)


class ExpectedEvidence(BaseModel):
    """A declarative evidence expectation for later scorer evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: str
    description: str
    required: bool = True
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class CapabilityTask(BaseModel):
    """A provider-neutral task supplied to a capability executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str = Field(min_length=1)
    initial_messages: tuple[Message, ...] = ()
    required_tools: tuple[str, ...] = ()
    asset_references: tuple[str, ...] = ()
    environment_requirement_references: tuple[str, ...] = ()
    limits: CapabilityLimits = Field(default_factory=CapabilityLimits)
    source: CapabilitySource | None = None
    expected_evidence: tuple[ExpectedEvidence, ...] = ()
    completion_tool_name: str | None = None
    continue_prompt: str | None = None


class CapabilityCase(BaseModel):
    """An immutable task instance bound to a target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task: CapabilityTask
    target_identifier: TargetIdentifier
    provenance: CapabilitySource | None = None


class ToolExecutionEvidence(BaseModel):
    """Facts about one tool execution attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["tool_execution"] = "tool_execution"
    call_id: str
    request_piece_id: uuid.UUID
    result_piece_id: uuid.UUID | None = None
    attempt_id: uuid.UUID
    attempt_number: int = Field(gt=0)
    tool_name: str
    status: ToolExecutionStatus
    started_at: AwareDatetime
    ended_at: AwareDatetime
    side_effect_completed: bool | None = False
    error_code: str | None = None
    retryable: bool = False


class ApprovalEvidence(BaseModel):
    """An explicit approval or policy decision for a tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["approval"] = "approval"
    call_id: str
    request_piece_id: uuid.UUID
    tool_name: str
    decision: ApprovalDecisionKind
    policy: str
    reason: str | None = None
    decided_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class EnvironmentReferenceEvidence(BaseModel):
    """A reference to an externally managed execution environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["environment_reference"] = "environment_reference"
    requirement_reference: str
    environment_reference: str
    provider: str | None = None


class EnvironmentEventEvidence(BaseModel):
    """A factual event emitted by an externally managed environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["environment_event"] = "environment_event"
    environment_reference: str
    event_name: str
    occurred_at: AwareDatetime
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class TimingEvidence(BaseModel):
    """Timing for an executor phase that is not provable from the transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["timing"] = "timing"
    phase: str
    started_at: AwareDatetime
    ended_at: AwareDatetime
    call_id: str | None = None


class ErrorEvidence(BaseModel):
    """A structured execution error outside the model-visible transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["error"] = "error"
    phase: str
    code: str
    message: str
    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    call_id: str | None = None
    attempt_id: uuid.UUID | None = None


class LifecycleEvidence(BaseModel):
    """Cancellation or truncation not otherwise provable from transcript content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["lifecycle"] = "lifecycle"
    event: LifecycleEventKind
    occurred_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    reason: str | None = None
    call_id: str | None = None


class ArtifactEvidence(BaseModel):
    """A reference to an artifact produced outside the transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["artifact"] = "artifact"
    artifact_id: str
    uri: str
    media_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    created_by_call_id: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SandboxOperationEvidence(BaseModel):
    """An authoritative fact emitted by a sandbox lifecycle or environment operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_type: Literal["sandbox_operation"] = "sandbox_operation"
    provider: str
    operation: str
    outcome: str
    started_at: AwareDatetime
    ended_at: AwareDatetime
    session_id: str | None = None
    environment_name: str | None = None
    call_id: str | None = None
    attempt_id: uuid.UUID | None = None
    error_code: str | None = None
    input_size_bytes: int | None = Field(default=None, ge=0)
    output_size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    artifact_ids: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


CapabilityEvidence = Annotated[
    ToolExecutionEvidence
    | ApprovalEvidence
    | EnvironmentReferenceEvidence
    | EnvironmentEventEvidence
    | TimingEvidence
    | ErrorEvidence
    | LifecycleEvidence
    | ArtifactEvidence
    | SandboxOperationEvidence,
    Field(discriminator="evidence_type"),
]


class CapabilityUsage(BaseModel):
    """Accumulated provider usage available for the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    extra: dict[str, int] = Field(default_factory=dict)

    def add(self, usage: TokenUsage) -> CapabilityUsage:
        """Return a new total with one provider generation added."""
        extra = dict(self.extra)
        for name, value in usage.extra.items():
            extra[name] = extra.get(name, 0) + value
        return self.model_copy(
            update={
                "input_tokens": self.input_tokens + (usage.input_tokens or 0),
                "output_tokens": self.output_tokens + (usage.output_tokens or 0),
                "total_tokens": self.total_tokens + (usage.total_tokens or 0),
                "reasoning_tokens": self.reasoning_tokens + (usage.reasoning_tokens or 0),
                "cached_tokens": self.cached_tokens + (usage.cached_tokens or 0),
                "extra": extra,
            }
        )


class CapabilityTaskResult(BaseModel):
    """The immutable result and evidence index for a capability case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: uuid.UUID
    conversation_id: str
    target_identifier: TargetIdentifier
    outcome: CapabilityOutcome
    termination_reason: CapabilityTerminationReason
    termination_detail: str | None = None
    message_piece_ids: tuple[uuid.UUID, ...] = ()
    final_message_piece_ids: tuple[uuid.UUID, ...] = ()
    evidence: tuple[CapabilityEvidence, ...] = ()
    scores: tuple[Score, ...] = ()
    usage: CapabilityUsage = Field(default_factory=CapabilityUsage)
    turns: int = Field(default=0, ge=0)
    model_generations: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    started_at: AwareDatetime
    ended_at: AwareDatetime

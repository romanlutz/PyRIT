# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Native agent evaluation records, with explicit observation and live-session limits."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator


class NativeCyberStatus(str, Enum):
    """State of one fresh native environment lease."""

    PREPARING = "preparing"
    RUNNING = "running"
    AWAITING_INSTRUCTION = "awaiting_instruction"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ERROR = "error"


class NativeCyberCleanup(str, Enum):
    """Observed cleanup outcome for the owned environment."""

    NOT_OPENED = "not_opened"
    CLOSED = "closed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class NativeAgentCapabilities(BaseModel):
    """Proven session abilities, not user-selectable promises."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    retained_session: bool = False
    operator_steps: bool = False
    max_turns: int = Field(default=1, ge=1, le=100)
    evidence_scope: Literal["native_session_events"] = "native_session_events"

    @model_validator(mode="after")
    def _validate_steps(self) -> NativeAgentCapabilities:
        if self.operator_steps and not self.retained_session:
            raise ValueError("Operator stepping requires the same retained native session and workspace.")
        return self


class NativeCyberReadiness(BaseModel):
    """Binding-owned qualification and blockers checked before opening an environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ready: bool
    blockers: tuple[str, ...] = ()
    simulated: bool
    capabilities: NativeAgentCapabilities = Field(default_factory=NativeAgentCapabilities)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_readiness(self) -> NativeCyberReadiness:
        if self.ready == bool(self.blockers):
            raise ValueError("Ready bindings have no blockers; blocked bindings must explain the blocker.")
        return self


class NativeCyberRequest(BaseModel):
    """Allow-listed execution choices, snapshotted before a fresh run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruction: str = Field(min_length=1, max_length=32768)
    label: str = Field(default="", max_length=128)
    technique: str = Field(default="literal", min_length=1, max_length=128)
    converter_names: tuple[str, ...] = ()
    operator_steps: bool = False
    ttl_seconds: int = Field(default=180, ge=1, le=3600)
    turn_timeout_seconds: int = Field(default=60, ge=1, le=600)
    parent_run_id: str | None = None

    @model_validator(mode="after")
    def _validate_instruction(self) -> NativeCyberRequest:
        if not self.instruction.strip():
            raise ValueError("An instruction cannot consist only of whitespace.")
        return self


class NativeAgentEvent(BaseModel):
    """An actual native event with controller-observed order and unchanged payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=1)
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    payload: dict[str, JsonValue]


class NativeToolTrace(BaseModel):
    """Correlation of observed native tool start/completion events."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: str
    arguments: JsonValue
    start_sequence: int
    completion_sequence: int | None = None
    success: bool | None = None
    result: JsonValue = None
    error: JsonValue = None
    status: Literal["running", "succeeded", "failed"] = "running"
    request_sequence: int | None = None
    model_visible_output: str | None = None
    detailed_output: str | None = None


class NativeToolRequest(BaseModel):
    """A real tool request emitted in the native assistant message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    name: str
    arguments: JsonValue
    request_sequence: int


class NativeAgentEvidence(BaseModel):
    """Evidence coverage for the observed native event stream, never inferred shell execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    environment_id: str
    simulated: bool
    events: tuple[NativeAgentEvent, ...]
    tools: tuple[NativeToolTrace, ...]
    tool_requests: tuple[NativeToolRequest, ...] = ()
    idle: bool
    coverage_complete: bool
    gaps: tuple[str, ...]
    scope: Literal["native_session_events"] = "native_session_events"
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


class NativeCyberArtifact(BaseModel):
    """A retained immutable artifact, never an arbitrary URL to fetch or execute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    evidence_ref: str = Field(min_length=1)


class NativeCyberJudgment(BaseModel):
    """One acquired original-grader result; absence is undetermined, never zero."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, strict=True)
    rationale: str
    complete: bool
    artifacts: tuple[NativeCyberArtifact, ...] = ()
    evidence: JsonValue = None

    @model_validator(mode="after")
    def _validate_value(self) -> NativeCyberJudgment:
        if self.complete != (self.value is not None):
            raise ValueError("Only a complete acquired judgment carries a numeric value.")
        return self


class NativeCyberReport(BaseModel):
    """Canonical content evidence for one immutable native evaluation outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    binding_name: str
    binding_version: str
    request: NativeCyberRequest
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_id: str | None = None
    status: NativeCyberStatus
    simulated: bool | None
    readiness: NativeCyberReadiness | None
    started_at: AwareDatetime
    expires_at: AwareDatetime
    ended_at: AwareDatetime
    conversation_id: str | None = None
    attack_result_id: str | None = None
    technique_identifier: dict[str, JsonValue] | None = None
    agent: NativeAgentEvidence | None = None
    judgment: NativeCyberJudgment | None = None
    cleanup: NativeCyberCleanup
    errors: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_result(self) -> NativeCyberReport:
        if self.readiness is None:
            if (
                self.simulated is not None
                or self.agent is not None
                or self.judgment is not None
                or self.cleanup is not NativeCyberCleanup.NOT_OPENED
                or not self.errors
                or self.status
                not in {
                    NativeCyberStatus.ERROR,
                    NativeCyberStatus.CANCELLED,
                    NativeCyberStatus.EXPIRED,
                }
            ):
                raise ValueError("Unavailable qualification permits only an error with unknown native provenance.")
        elif self.simulated != self.readiness.simulated:
            raise ValueError("Native provenance must remain consistent.")
        if self.agent and self.agent.simulated != self.simulated:
            raise ValueError("Native provenance must remain consistent.")
        if self.status is NativeCyberStatus.COMPLETED and (
            self.readiness is None
            or not self.readiness.ready
            or self.agent is None
            or not self.agent.coverage_complete
            or not self.agent.idle
            or self.judgment is None
            or not self.judgment.complete
            or self.cleanup != "closed"
            or self.errors
        ):
            raise ValueError("Completed native results require complete evidence, one judgment and verified cleanup.")
        return self

    def canonical_json(self) -> str:
        """
        Serialize the report without changing its recorded evidence.

        Returns:
            str: Canonical JSON suitable for content-anchored scoring.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sha256(self) -> str:
        """
        Hash the exact retained report.

        Returns:
            str: Lowercase SHA256.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class NativeCyberRunView(BaseModel):
    """Safe operational state; raw task/agent content is retrieved separately."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    binding_name: str
    status: NativeCyberStatus
    expires_at: AwareDatetime
    capabilities: NativeAgentCapabilities
    can_step: bool
    cancel_requested: bool
    turn_count: int
    parent_run_id: str | None
    conversation_id: str | None = None
    score_id: str | None = None
    content_id: str | None = None
    report_sha256: str | None = None
    blockers: tuple[str, ...] = ()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Typed, source-observed native cyber evidence retained by memory."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from pyrit.models.native_cyber import NativeAgentEvent  # noqa: TC001 (Pydantic needs the runtime model)
from pyrit.models.score.score import ScoreStatus  # noqa: TC001 (Pydantic needs the runtime enum)


class NativeCyberEvidenceSource(str, Enum):
    """The origin of a captured event or raw stream."""

    MODEL = "model"
    TOOL = "tool"
    HARNESS = "harness"


class NativeCyberRawKind(str, Enum):
    """The raw byte stream retained, not a fabricated conversation role."""

    STDOUT = "stdout"
    STDERR = "stderr"
    JSONL = "jsonl"
    MODEL = "model"
    TOOL = "tool"


class NativeCyberToolPhase(str, Enum):
    """A tool call phase observed by the event source."""

    REQUEST = "request"
    START = "start"
    COMPLETE = "complete"
    RESULT = "result"


class NativeCyberResponseMode(str, Enum):
    """Whether the task needs an assistant message or approves artifact-only completion."""

    MESSAGE_REQUIRED = "message_required"
    ARTIFACT_ONLY = "artifact_only"


class NativeCyberCoveragePhase(str, Enum):
    """The evidence available before grading versus after acquiring the judgment."""

    PREGRADING = "pregrading"
    FINAL = "final"


class NativeCyberResponsePolicy(BaseModel):
    """Versioned task-owned approval; individual turns still opt in explicitly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    allow_artifact_only: bool = False


class NativeCyberRawStreamKey(BaseModel):
    """An observed source stream the capture contract can require."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: NativeCyberEvidenceSource
    kind: NativeCyberRawKind
    observed_source_id: str = Field(min_length=1, max_length=128)


class NativeCyberEpisodeStart(BaseModel):
    """Immutable run identity and expected raw streams, fixed before capture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    binding_name: str = Field(min_length=1, max_length=128)
    binding_version: str = Field(min_length=1, max_length=128)
    started_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    source_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    environment_id: str | None = Field(default=None, min_length=1, max_length=128)
    simulated: bool | None = None
    required_raw_streams: tuple[NativeCyberRawStreamKey, ...] = ()
    require_separate_tool_results: bool = False
    response_policy: NativeCyberResponsePolicy = Field(default_factory=NativeCyberResponsePolicy)
    raw_byte_limit: int = Field(default=268_435_456, ge=1, le=1_099_511_627_776)

    @model_validator(mode="after")
    def _validate_sources(self) -> NativeCyberEpisodeStart:
        keys = [(key.source, key.kind, key.observed_source_id) for key in self.required_raw_streams]
        if len(keys) != len(set(keys)):
            raise ValueError("Required native raw streams must have distinct observed identities.")
        return self


class NativeCyberTurnStart(BaseModel):
    """An outer operator turn, distinct from internal agent tool turns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    turn_index: int = Field(ge=1)
    source_turn_id: str | None = Field(default=None, min_length=1, max_length=128)
    started_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    request_piece_ids: tuple[uuid.UUID, ...] = ()
    response_mode: NativeCyberResponseMode = NativeCyberResponseMode.MESSAGE_REQUIRED


class NativeCyberTurnFinish(BaseModel):
    """The source's event count and coverage claim for a completed outer turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    turn_index: int = Field(ge=1)
    finished_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    response_piece_ids: tuple[uuid.UUID, ...] = ()
    tool_request_piece_ids: tuple[uuid.UUID, ...] = ()
    tool_result_piece_ids: tuple[uuid.UUID, ...] = ()
    observed_event_count: int = Field(ge=0)
    source_complete: bool
    gaps: tuple[str, ...] = ()


class NativeCyberObservedEvent(BaseModel):
    """One controller-ordered source frame, with provider IDs only when observed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    controller_sequence: int = Field(ge=1)
    source_event_id: str | None = Field(default=None, min_length=1, max_length=256)
    source_session_id: str | None = Field(default=None, min_length=1, max_length=128)
    event_type: str = Field(min_length=1, max_length=128)
    payload: dict[str, JsonValue] = Field(repr=False)
    observed_stream_id: str | None = Field(default=None, min_length=1, max_length=128)
    stream_offset: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=128)
    tool_phase: NativeCyberToolPhase | None = None

    @model_validator(mode="after")
    def _validate_observed_refs(self) -> NativeCyberObservedEvent:
        if (self.observed_stream_id is None) != (self.stream_offset is None):
            raise ValueError("A native stream offset requires its observed stream ID.")
        if self.tool_phase is not None and self.tool_call_id is None:
            raise ValueError("A native tool phase requires an observed tool call ID.")
        return self

    @property
    def sequence(self) -> int:
        """The controller-observed ordinal, not an invented provider ID."""
        return self.controller_sequence


class NativeCyberCapturedEvent(BaseModel):
    """One untouched source frame and its producing model/tool/harness tag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: NativeCyberEvidenceSource
    event: NativeCyberObservedEvent = Field(repr=False)
    captured_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_native_agent_event(
        cls,
        *,
        source: NativeCyberEvidenceSource,
        event: NativeAgentEvent,
        captured_at: datetime | None = None,
    ) -> NativeCyberCapturedEvent:
        """
        Adapt a GHCP event, preserving its actual observed IDs and payload.

        Returns:
            NativeCyberCapturedEvent: A generic source frame with no invented IDs.
        """
        return cls(
            source=source,
            captured_at=captured_at or datetime.now(UTC),
            event=NativeCyberObservedEvent(
                controller_sequence=event.sequence,
                source_event_id=event.event_id,
                source_session_id=event.session_id,
                event_type=event.event_type,
                payload=event.payload,
            ),
        )


class NativeCyberRawStreamStart(BaseModel):
    """A source-identified raw stream, optionally tied to an outer turn or tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    stream_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    key: NativeCyberRawStreamKey
    turn_index: int | None = Field(default=None, ge=1)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=128)


class NativeCyberRawWrite(BaseModel):
    """One committed raw append, including bytes refused by the run quota."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stream_id: uuid.UUID
    received_bytes: int
    stored_bytes: int
    omitted_bytes: int
    truncated: bool


class NativeCyberCoverageAssessment(BaseModel):
    """Pre-score snapshot of declared task-required and optional capture gaps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: NativeCyberCoveragePhase
    required_complete: bool
    required_gaps: tuple[str, ...]
    optional_gaps: tuple[str, ...]


class NativeCyberEventSummary(BaseModel):
    """Source provenance without the potentially sensitive event payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int
    source: NativeCyberEvidenceSource
    observed_event_id: str | None
    observed_session_id: str | None
    observed_stream_id: str | None
    stream_offset: int | None
    tool_call_id: str | None
    tool_phase: NativeCyberToolPhase | None
    event_type: str
    captured_at: AwareDatetime
    payload_sha256: str


class NativeCyberToolCorrelation(BaseModel):
    """Observed tool phases, keeping model-visible output distinct from execution completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    request_sequence: int | None
    start_sequence: int | None
    completion_sequence: int | None
    result_sequence: int | None


class NativeCyberTurnSummary(BaseModel):
    """Outer-turn provenance and genuine stored conversation-piece references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn_index: int
    source_turn_id: str | None
    response_mode: NativeCyberResponseMode
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    request_piece_ids: tuple[uuid.UUID, ...]
    response_piece_ids: tuple[uuid.UUID, ...]
    tool_request_piece_ids: tuple[uuid.UUID, ...]
    tool_result_piece_ids: tuple[uuid.UUID, ...]
    observed_event_count: int | None
    stored_event_count: int
    source_complete: bool
    gaps: tuple[str, ...]


class NativeCyberRawStreamSummary(BaseModel):
    """Raw-stream provenance, length, digest and explicit incompleteness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stream_id: uuid.UUID
    key: NativeCyberRawStreamKey
    turn_index: int | None
    tool_call_id: str | None
    expected_bytes: int | None
    received_bytes: int
    stored_bytes: int
    omitted_bytes: int
    stored_sha256: str | None
    observed_sha256: str | None
    source_complete: bool
    truncated: bool
    gaps: tuple[str, ...]


class NativeCyberEpisodeSnapshot(BaseModel):
    """Safe view; coverage_complete concerns required capture, not optional telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: NativeCyberEpisodeStart
    conversation_id: str | None
    finalized_at: AwareDatetime | None
    coverage_complete: bool
    gaps: tuple[str, ...]
    optional_gaps: tuple[str, ...]
    stored_raw_bytes: int
    report_content_id: uuid.UUID | None
    report_sha256: str | None
    score_id: uuid.UUID | None
    score_status: ScoreStatus
    turns: tuple[NativeCyberTurnSummary, ...]
    events: tuple[NativeCyberEventSummary, ...]
    tools: tuple[NativeCyberToolCorrelation, ...]
    raw_streams: tuple[NativeCyberRawStreamSummary, ...]


class NativeCyberRawChunk(BaseModel):
    """One explicitly requested bounded, integrity-checked raw byte range."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int
    offset: int
    length: int
    sha256: str
    data: bytes = Field(repr=False)

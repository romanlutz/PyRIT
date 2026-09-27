# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Versioned native coding CLI evidence, distinct from Copilot SDK events."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NativeCliReportProtocol(str, Enum):
    """The configured provider JSONL contract, not proof that a binary ran."""

    CODEX_EXEC_JSON = "codex_exec_json"
    CLAUDE_PRINT_STREAM_JSON_VERBOSE = "claude_print_stream_json_verbose"


class NativeCliReportStatus(str, Enum):
    """The caller-declared run status, not proof of durable DB publication."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    CANCELLED = "cancelled"


class NativeCliReportCleanup(str, Enum):
    """The caller-observed sandbox cleanup outcome, never inferred from CLI EOF."""

    NOT_OPENED = "not_opened"
    CLOSED = "closed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class NativeCliReportEventKind(str, Enum):
    """The CLI parser's observed actions and explicit coverage boundaries."""

    SESSION_STARTED = "session_started"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    RUN_FINISHED = "run_finished"
    MODEL_MESSAGE = "model_message"
    TOOL_REQUESTED = "tool_requested"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_RESULT = "tool_result"
    PROGRESS = "progress"
    AUXILIARY = "auxiliary"
    PARTIAL = "partial"
    ERROR = "error"
    EOF = "eof"


class NativeCliReportEventStatus(str, Enum):
    """A provider status actually observed, or unknown when no status was supplied."""

    UNKNOWN = "unknown"
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NativeCliReportEvent(BaseModel):
    """An ordered CLI observation with nullable real source IDs and a raw-frame digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(strict=True, ge=1)
    frame_number: int | None = Field(default=None, strict=True, ge=1)
    kind: NativeCliReportEventKind
    status: NativeCliReportEventStatus
    source_event_id: str | None = None
    source_message_id: str | None = None
    source_session_id: str | None = None
    source_tool_id: str | None = None
    parent_tool_use_id: str | None = None
    source_status: str | None = None
    name: str | None = None
    exit_code: int | None = Field(default=None, strict=True)
    detail: str | None = None
    raw_frame_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_frame_size_bytes: int | None = Field(default=None, strict=True, ge=1)
    stdout_offset_bytes: int | None = Field(default=None, strict=True, ge=0)

    @field_validator(
        "source_event_id",
        "source_message_id",
        "source_session_id",
        "source_tool_id",
        "parent_tool_use_id",
        "source_status",
        "name",
    )
    @classmethod
    def _check_source_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Observed CLI source identifiers and names cannot be blank.")
        return value

    @model_validator(mode="after")
    def _validate_frame(self) -> NativeCliReportEvent:
        tool_kinds = {
            NativeCliReportEventKind.TOOL_REQUESTED,
            NativeCliReportEventKind.TOOL_STARTED,
            NativeCliReportEventKind.TOOL_COMPLETED,
            NativeCliReportEventKind.TOOL_RESULT,
        }
        if self.kind is NativeCliReportEventKind.SESSION_STARTED and not self.source_session_id:
            raise ValueError("A CLI session start requires its observed source session ID.")
        if (
            self.kind
            in {
                NativeCliReportEventKind.MODEL_MESSAGE,
                NativeCliReportEventKind.RUN_FINISHED,
                NativeCliReportEventKind.PROGRESS,
                *tool_kinds,
            }
            and not self.source_event_id
        ):
            raise ValueError("A CLI model, tool, result, or item progress event requires its observed source event ID.")
        if self.kind in tool_kinds and not self.source_tool_id:
            raise ValueError("A CLI tool event requires its observed source tool ID.")
        frame_fields = (
            self.frame_number,
            self.raw_frame_sha256,
            self.raw_frame_size_bytes,
            self.stdout_offset_bytes,
        )
        if any(value is None for value in frame_fields) and any(value is not None for value in frame_fields):
            raise ValueError("A CLI frame number, digest, size, and stdout offset must occur together.")
        if self.kind is NativeCliReportEventKind.EOF and self.frame_number is not None:
            raise ValueError("CLI EOF is a synthetic boundary, not a provider JSONL frame.")
        if (
            self.kind
            not in {
                NativeCliReportEventKind.ERROR,
                NativeCliReportEventKind.PARTIAL,
                NativeCliReportEventKind.EOF,
            }
            and self.frame_number is None
        ):
            raise ValueError("A CLI provider observation requires its actual JSONL frame metadata.")
        if self.kind in {NativeCliReportEventKind.ERROR, NativeCliReportEventKind.PARTIAL} and (
            not self.detail or not self.detail.strip()
        ):
            raise ValueError("A CLI coverage gap must explain what was not observed.")
        return self


class NativeCliReportEvidence(BaseModel):
    """Actual process outcome, ordered provider summary, and explicit missing evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_session_id: str | None = None
    exit_code: int | None = Field(default=None, strict=True)
    terminal_observed: bool = Field(strict=True)
    coverage_complete: bool = Field(strict=True)
    observed_steps: int | None = Field(default=None, strict=True, ge=0)
    frame_count: int | None = Field(default=None, strict=True, ge=0)
    raw_chunk_count: int | None = Field(default=None, strict=True, ge=0)
    raw_stdout_bytes: int | None = Field(default=None, strict=True, ge=0)
    raw_stderr_bytes: int | None = Field(default=None, strict=True, ge=0)
    gaps: tuple[str, ...] = ()
    events: tuple[NativeCliReportEvent, ...] = ()
    raw_evidence_ref: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_observations(self) -> NativeCliReportEvidence:
        if self.raw_evidence_ref is not None and not self.raw_evidence_ref.strip():
            raise ValueError("A CLI raw evidence reference cannot be blank.")
        if [event.sequence for event in self.events] != list(range(1, len(self.events) + 1)):
            raise ValueError("CLI event sequence must preserve every observed event in order.")
        if any(not gap.strip() for gap in self.gaps):
            raise ValueError("CLI evidence gaps require nonempty reasons.")
        self._validate_frame_metadata()
        self._validate_outcome()
        self._validate_source()
        return self

    def _validate_frame_metadata(self) -> None:
        frames: dict[int, tuple[str | None, int | None, int | None]] = {}
        last_frame = 0
        for event in self.events:
            if event.frame_number is None:
                continue
            if event.frame_number < last_frame:
                raise ValueError("CLI stdout frames must preserve their observed order.")
            last_frame = event.frame_number
            metadata = (event.raw_frame_sha256, event.raw_frame_size_bytes, event.stdout_offset_bytes)
            previous = frames.setdefault(event.frame_number, metadata)
            if previous != metadata:
                raise ValueError("Observations of the same CLI stdout frame must retain identical digests and offsets.")

    def _validate_outcome(self) -> None:
        counts = (
            self.observed_steps,
            self.frame_count,
            self.raw_chunk_count,
            self.raw_stdout_bytes,
            self.raw_stderr_bytes,
        )
        if self.exit_code is None:
            if self.coverage_complete or any(count is not None for count in counts) or not self.gaps:
                raise ValueError("A missing process outcome requires unknown counts and an explicit coverage gap.")
        elif any(count is None for count in counts):
            raise ValueError("An observed process exit requires the actual step, frame, and byte counts.")
        if self.frame_count is not None and any(
            event.frame_number is not None and event.frame_number > self.frame_count for event in self.events
        ):
            raise ValueError("A CLI event cannot refer to an unobserved stdout frame.")
        if not self.coverage_complete and not self.gaps:
            raise ValueError("Incomplete CLI coverage requires an explicit gap.")
        if self.coverage_complete and (
            self.exit_code != 0
            or not self.terminal_observed
            or self.gaps
            or not self.events
            or self.events[-1].kind is not NativeCliReportEventKind.EOF
            or not any(
                event.kind in {NativeCliReportEventKind.TURN_COMPLETED, NativeCliReportEventKind.RUN_FINISHED}
                for event in self.events
            )
            or any(
                event.kind in {NativeCliReportEventKind.ERROR, NativeCliReportEventKind.PARTIAL}
                for event in self.events
            )
        ):
            raise ValueError("Complete CLI coverage requires an observed exit, terminal, EOF, and no gaps.")
        if self.coverage_complete:
            frames = {
                event.frame_number: (event.raw_frame_size_bytes, event.stdout_offset_bytes)
                for event in self.events
                if event.frame_number is not None
            }
            if (
                not self.frame_count
                or not self.raw_chunk_count
                or not self.raw_stdout_bytes
                or frames.keys() != set(range(1, self.frame_count + 1))
            ):
                raise ValueError("Complete CLI coverage requires every stdout frame and its raw-byte handoff.")
            offset = 0
            for number in range(1, self.frame_count + 1):
                size, observed_offset = frames[number]
                if observed_offset != offset or size is None:
                    raise ValueError("Complete CLI frames must retain contiguous stdout offsets.")
                offset += size
            if offset != self.raw_stdout_bytes:
                raise ValueError("Complete CLI frame lengths must match the observed stdout byte count.")

    def _validate_source(self) -> None:
        observed = {event.source_session_id for event in self.events if event.source_session_id}
        if self.source_session_id is not None and not self.source_session_id.strip():
            raise ValueError("An observed CLI session ID cannot be blank.")
        starts = [
            event.source_session_id
            for event in self.events
            if event.kind is NativeCliReportEventKind.SESSION_STARTED and event.source_session_id
        ]
        if self.source_session_id is not None and self.source_session_id not in observed:
            raise ValueError("A CLI source session ID cannot be claimed without an observed session field.")
        if len(observed) == 1 and self.source_session_id not in observed:
            raise ValueError("The CLI source session ID must retain the actual observed session field.")
        if starts and self.source_session_id is not None and self.source_session_id != starts[0]:
            raise ValueError("The CLI source session ID must match the observed session start.")
        if self.coverage_complete and (
            len(starts) != 1 or not self.source_session_id or observed != {self.source_session_id}
        ):
            raise ValueError("Complete CLI coverage requires one consistent observed session start.")
        if (
            len(observed) > 1
            and self.source_session_id is not None
            and (len(starts) != 1 or starts[0] != self.source_session_id)
        ):
            raise ValueError("Contradictory CLI session fields cannot supply one authoritative session ID.")


class NativeCliArtifactReference(BaseModel):
    """An opaque caller-supplied artifact identifier; this model reads no content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, ge=0)
    evidence_ref: str = Field(min_length=1)
    media_type: str = "application/octet-stream"

    @model_validator(mode="after")
    def _validate_reference(self) -> NativeCliArtifactReference:
        if not self.name.strip() or not self.evidence_ref.strip() or not self.media_type.strip():
            raise ValueError("CLI artifact references must have nonblank names, media types, and evidence IDs.")
        return self


class NativeCliOriginalJudgment(BaseModel):
    """An acquired original grader judgment, independent of process or tool success."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grader_ref: str = Field(min_length=1)
    grader_evidence_ref: str | None = Field(default=None, min_length=1)
    value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    complete: bool = Field(strict=True)
    rationale: str

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, value: object) -> float | None:
        if value is None:
            return None
        if type(value) not in (int, float) or not isinstance(value, (int, float)):
            raise ValueError("Original grader value must be a native number, not a boolean or string.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Original grader value must be finite and between zero and one.")
        return float(value)

    @model_validator(mode="after")
    def _validate_completion(self) -> NativeCliOriginalJudgment:
        if not self.grader_ref.strip() or not self.rationale.strip():
            raise ValueError("An original grader judgment requires a grader identity and rationale.")
        if self.grader_evidence_ref is not None and not self.grader_evidence_ref.strip():
            raise ValueError("An original grader evidence reference cannot be blank.")
        if self.complete != (self.value is not None):
            raise ValueError("Only a complete original grader judgment may carry a numeric value.")
        if self.complete and not self.grader_evidence_ref:
            raise ValueError("A complete original judgment requires rationale and retained grader evidence.")
        return self


class NativeCliRunReport(BaseModel):
    """Canonical v1 CLI run evidence; not a Copilot SDK event stream or a score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    task_id: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    turn_index: int = Field(strict=True, ge=1)
    parent_run_id: str | None = Field(default=None, min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1)
    protocol: NativeCliReportProtocol
    cli_version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?$")
    cli_profile: str = Field(pattern=r"^[a-z][a-z0-9._-]*$")
    max_steps: int = Field(strict=True, ge=1)
    simulated: bool | None = Field(default=None, strict=True)
    status: NativeCliReportStatus
    evidence: NativeCliReportEvidence
    judgment: NativeCliOriginalJudgment | None = None
    artifacts: tuple[NativeCliArtifactReference, ...] = ()
    cleanup: NativeCliReportCleanup
    errors: tuple[str, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def _check_version(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported native CLI report schema_version; expected the integer 1.")
        return value

    @model_validator(mode="after")
    def _validate_status(self) -> NativeCliRunReport:
        if any(not item.strip() for item in (self.task_id, self.task_version, self.run_id, self.turn_id, *self.errors)):
            raise ValueError("CLI task version, task/run/turn IDs, and errors must not be blank.")
        if any(item is not None and not item.strip() for item in (self.parent_run_id, self.conversation_id)):
            raise ValueError("CLI parent run and conversation IDs must not be blank.")
        if self.parent_run_id == self.run_id:
            raise ValueError("A CLI parent run ID must differ from the current run ID.")
        if self.evidence.observed_steps is not None and self.evidence.observed_steps > self.max_steps:
            raise ValueError("Observed CLI steps cannot exceed the configured step budget.")
        if self.evidence.coverage_complete:
            self._validate_protocol_trace()
        if self.cleanup is NativeCliReportCleanup.NOT_OPENED and any(
            event.frame_number is not None for event in self.evidence.events
        ):
            raise ValueError("A sandbox with provider frames cannot be reported as never opened.")
        if self.status in {NativeCliReportStatus.ERROR, NativeCliReportStatus.CANCELLED} and not self.errors:
            raise ValueError("CLI error and cancellation states must include a reason.")
        if self.status is NativeCliReportStatus.COMPLETED and (
            not self.evidence.coverage_complete
            or self.evidence.exit_code != 0
            or self.judgment is None
            or not self.judgment.complete
            or self.cleanup is not NativeCliReportCleanup.CLOSED
            or not self.evidence.raw_evidence_ref
            or self.simulated is None
            or self.errors
        ):
            raise ValueError("Completed CLI results require complete evidence, original judgment, and known cleanup.")
        refs = [artifact.evidence_ref for artifact in self.artifacts]
        if len(set(refs)) != len(refs):
            raise ValueError("CLI artifact references cannot be duplicated.")
        return self

    def _validate_protocol_trace(self) -> None:
        events = self.evidence.events
        if any(
            event.kind
            in {
                NativeCliReportEventKind.TOOL_REQUESTED,
                NativeCliReportEventKind.TOOL_STARTED,
                NativeCliReportEventKind.TOOL_COMPLETED,
                NativeCliReportEventKind.TOOL_RESULT,
            }
            and event.status is NativeCliReportEventStatus.UNKNOWN
            for event in events
        ):
            raise ValueError("Complete CLI evidence cannot include an unclassified tool status.")
        if self.protocol is NativeCliReportProtocol.CODEX_EXEC_JSON:
            started = sum(event.kind is NativeCliReportEventKind.TURN_STARTED for event in events)
            completed = [event for event in events if event.kind is NativeCliReportEventKind.TURN_COMPLETED]
            last_provider_event = next(
                (event for event in reversed(events) if event.kind is not NativeCliReportEventKind.EOF),
                None,
            )
            if (
                started != self.evidence.observed_steps
                or started == 0
                or len(completed) != started
                or any(event.status is not NativeCliReportEventStatus.COMPLETED for event in completed)
                or last_provider_event is None
                or last_provider_event.kind is not NativeCliReportEventKind.TURN_COMPLETED
                or any(event.kind is NativeCliReportEventKind.RUN_FINISHED for event in events)
            ):
                raise ValueError("Complete Codex evidence requires its actual turn count and turn.completed event.")
            if any(
                (
                    event.kind
                    in {
                        NativeCliReportEventKind.TOOL_STARTED,
                        NativeCliReportEventKind.TOOL_COMPLETED,
                        NativeCliReportEventKind.TOOL_RESULT,
                    }
                    and (not event.name or not event.source_status)
                )
                or (
                    event.kind is NativeCliReportEventKind.PROGRESS
                    and event.source_tool_id is not None
                    and not event.source_status
                )
                for event in events
            ):
                raise ValueError("Complete Codex tool events require their observed name and native status.")
        else:
            if any(
                event.kind in {NativeCliReportEventKind.MODEL_MESSAGE, NativeCliReportEventKind.TOOL_REQUESTED}
                and not event.source_message_id
                for event in events
            ):
                raise ValueError("Complete Claude assistant blocks require their observed source message IDs.")
            messages = {
                event.source_message_id
                for event in events
                if event.kind in {NativeCliReportEventKind.MODEL_MESSAGE, NativeCliReportEventKind.TOOL_REQUESTED}
                and event.source_message_id
            }
            finals = [event for event in events if event.kind is NativeCliReportEventKind.RUN_FINISHED]
            if (
                len(messages) != self.evidence.observed_steps
                or len(finals) != 1
                or finals[0].status is not NativeCliReportEventStatus.COMPLETED
                or finals[0].source_status != "success"
                or any(
                    event.kind in {NativeCliReportEventKind.TURN_STARTED, NativeCliReportEventKind.TURN_COMPLETED}
                    for event in events
                )
            ):
                raise ValueError("Complete Claude evidence requires its actual message count and root result.")

    def canonical_json(self) -> str:
        """
        Serialize the v1 evidence in stable key and event order.

        Returns:
            str: Deterministic, compact, nonfinite-free JSON.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sha256(self) -> str:
        """
        Hash the complete canonical report without altering its observations.

        Returns:
            str: Lowercase SHA-256 digest.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

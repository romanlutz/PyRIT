# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local live working-memory evidence, distinct from job status and final benchmark grades."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from uuid import UUID  # noqa: TC003

from pydantic import Field, JsonValue, StrictInt, model_validator

from pyrit.models.evaluation_job import (
    Digest,
    EvaluationControlRequest,
    EvaluationJobRequest,
    EvaluationRuntimeKind,
    PublicName,
    _JobMessage,
)
from pyrit.models.evaluation_worker import EvaluationWorkerBinding  # noqa: TC001
from pyrit.models.identifiers.component_identifier import config_hash


class EvaluationFeedbackSource(str, Enum):
    """Explicit reviewed capture surfaces, not arbitrary installed agent implementations."""

    NATIVE_SESSION = "native_session_events"
    INSPECT_GHCP = "inspect_ghcp_turn_events"


class EvaluationFeedbackSession(_JobMessage):
    """One runtime working-memory owner and required feedback policy under both dispatch fences."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    request: EvaluationJobRequest
    binding: EvaluationWorkerBinding
    memory_owner_id: UUID
    source: EvaluationFeedbackSource
    source_session_id: str = Field(min_length=1, max_length=256)
    required_scorer_sha256: Digest
    expectation_sha256: Digest

    @model_validator(mode="after")
    def _validate_session(self) -> EvaluationFeedbackSession:
        if self.request.runtime is EvaluationRuntimeKind.ORIGINAL_INSPECT:
            raise ValueError("Unchanged Mode 1 cannot install a live feedback or continuation policy.")
        if (
            self.binding.job_id != self.request.job_id
            or self.binding.run_id != self.request.run_id
            or self.binding.attempt_id != self.request.attempt_id
            or self.binding.request_sha256 != self.request.request_sha256
        ):
            raise ValueError("Working-memory ownership differs from the admitted source execution.")
        return self

    @property
    def session_sha256(self) -> str:
        """The immutable working-memory and feedback-policy identity."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationFeedbackEvent(_JobMessage):
    """A retained typed source object; this is not a claim of raw network or token coverage."""

    source_sequence: StrictInt = Field(ge=1)
    source_event_id: str = Field(min_length=1, max_length=256)
    event_type: str = Field(min_length=1, max_length=128)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def _validate_payload(self) -> EvaluationFeedbackEvent:
        if self.payload.get("id") != self.source_event_id or self.payload.get("type") != self.event_type:
            raise ValueError("The retained typed source object differs from its stable event identity.")
        if len(self.payload_bytes()) > 256 * 1024:
            raise ValueError("The trusted source event exceeds its bounded retention limit.")
        return self

    def payload_bytes(self) -> bytes:
        """
        Encode the observed typed fields without adding normalized message or score content.

        Returns:
            bytes: Deterministic JSON, not original transport bytes.
        """
        return json.dumps(
            self.payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    @property
    def payload_sha256(self) -> str:
        """The digest of the retained typed source serialization."""
        return hashlib.sha256(self.payload_bytes()).hexdigest()


class EvaluationFeedbackPiece(_JobMessage):
    """Source-to-owner-row mapping; local working IDs never become worker canonical receipts."""

    source_event_id: str = Field(min_length=1, max_length=256)
    source_part_index: StrictInt = Field(ge=0, le=255)
    source_message_id: str | None = Field(default=None, min_length=1, max_length=256)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    piece_id: UUID
    original_prompt_id: UUID
    role: str = Field(pattern=r"^(user|assistant|tool)$")
    data_type: str = Field(pattern=r"^(text|function_call|function_call_output)$")
    original_value_sha256: Digest


class EvaluationFeedbackTurn(_JobMessage):
    """Ordered source coverage and prepared normalized-row identities before memory commit."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    session_sha256: Digest
    conversation_id: UUID
    turn_index: StrictInt = Field(ge=1, le=100)
    events: tuple[EvaluationFeedbackEvent, ...] = Field(min_length=1, max_length=256)
    pieces: tuple[EvaluationFeedbackPiece, ...] = Field(min_length=2, max_length=256)
    response_piece_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)
    boundary_source_event_id: str = Field(min_length=1, max_length=256)
    raw_only_event_ids: tuple[str, ...] = Field(default=(), max_length=256)
    raw_complete: bool
    normalized_complete: bool
    gaps: tuple[PublicName, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def _validate_turn(self) -> EvaluationFeedbackTurn:
        sequences = [event.source_sequence for event in self.events]
        event_ids = [event.source_event_id for event in self.events]
        piece_ids = [piece.piece_id for piece in self.pieces]
        keys = [(piece.source_event_id, piece.source_part_index) for piece in self.pieces]
        if sequences != list(range(sequences[0], sequences[-1] + 1)) or len(set(event_ids)) != len(event_ids):
            raise ValueError("Observed source events must be uniquely identified and gap-free.")
        if len(set(piece_ids)) != len(piece_ids) or len(set(keys)) != len(keys):
            raise ValueError("Normalized source pieces must have unique stable identities.")
        if any(piece.source_event_id not in event_ids for piece in self.pieces if piece.role != "user"):
            raise ValueError("Normalized responses require retained source events, not synthetic receipts.")
        projected = {piece.source_event_id for piece in self.pieces if piece.source_event_id in event_ids}
        if (
            len(set(self.raw_only_event_ids)) != len(self.raw_only_event_ids)
            or projected & set(self.raw_only_event_ids)
            or projected | set(self.raw_only_event_ids) != set(event_ids)
        ):
            raise ValueError("Raw-only and normalized source coverage must explicitly partition the retained events.")
        if (
            self.boundary_source_event_id != event_ids[-1]
            or not set(self.response_piece_ids) <= set(piece_ids)
            or len(set(self.response_piece_ids)) != len(self.response_piece_ids)
        ):
            raise ValueError("The response or reviewed source boundary is outside the retained turn.")
        if len(self.model_dump_json().encode("utf-8")) > 1024 * 1024:
            raise ValueError("Observed source and normalized coverage exceed the bounded turn quota.")
        return self

    @property
    def turn_sha256(self) -> str:
        """The exact source inventory and normalized mapping identity."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationObservationCommit(_JobMessage):
    """Readback after real PyRIT message writes, not capture or queue acknowledgment."""

    session_sha256: Digest
    conversation_id: UUID
    turn_index: StrictInt = Field(ge=1, le=100)
    source_through: StrictInt = Field(ge=1)
    turn_sha256: Digest
    normalized_sha256: Digest
    piece_ids: tuple[UUID, ...] = Field(min_length=2, max_length=256)
    response_piece_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)

    @property
    def observation_sha256(self) -> str:
        """The exact committed observation used by the required scorer."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationFeedbackCommit(_JobMessage):
    """Verified persisted required feedback, never the Task's original final benchmark grade."""

    session_sha256: Digest
    observation_sha256: Digest
    scorer_sha256: Digest
    expectation_sha256: Digest
    score_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)
    scores_sha256: Digest

    @property
    def feedback_sha256(self) -> str:
        """The score/rationale evidence committed for this observation."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationReadySnapshot(_JobMessage):
    """A memory-readiness watermark; neither lifecycle cursor nor proof of agent application."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    session_sha256: Digest
    memory_owner_id: UUID
    conversation_id: UUID
    generation: StrictInt = Field(ge=1, le=100)
    source_through: StrictInt = Field(ge=1)
    observation_sha256: Digest
    feedback_sha256: Digest

    @property
    def snapshot_sha256(self) -> str:
        """The exact committed snapshot the next input must name."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationFeedbackArchive(_JobMessage):
    """Exact final source linked to existing working rows without projecting them a second time."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    session_sha256: Digest
    memory_owner_id: UUID
    conversation_id: UUID
    source_through: StrictInt = Field(ge=1)
    last_snapshot_sha256: Digest
    archive_sha256: Digest
    archive_bytes: StrictInt = Field(ge=1, le=16 * 1024 * 1024)
    media_type: str = Field(pattern=r"^(application/x-ndjson|application/octet-stream)$")
    source_sha256: Digest


class EvaluationFeedbackControlRequest(_JobMessage):
    """A reviewed command naming exact memory readiness, not only an active job or lifecycle cursor."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    session_sha256: Digest
    snapshot_sha256: Digest
    command: EvaluationControlRequest

    @property
    def control_sha256(self) -> str:
        """The immutable command and working-memory snapshot binding."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationFeedbackControlReceipt(_JobMessage):
    """One input reservation; queue acceptance, delivery and observed agent application remain separate."""

    session_sha256: Digest
    command_id: UUID
    control_sha256: Digest
    snapshot_sha256: Digest
    reserved_turn: StrictInt = Field(ge=2, le=100)
    duplicate: bool = False


def evaluation_feedback_schema_sha256() -> str:
    """
    Fingerprint the separate local feedback contract without altering worker-v1 compatibility.

    Returns:
        str: The canonical schema fingerprint for this opt-in local memory seam.
    """
    models = (
        EvaluationFeedbackSession,
        EvaluationFeedbackEvent,
        EvaluationFeedbackPiece,
        EvaluationFeedbackTurn,
        EvaluationObservationCommit,
        EvaluationFeedbackCommit,
        EvaluationReadySnapshot,
        EvaluationFeedbackArchive,
        EvaluationFeedbackControlRequest,
        EvaluationFeedbackControlReceipt,
    )
    return config_hash({model.__name__: model.model_json_schema() for model in models})

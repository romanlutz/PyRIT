# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PyRIT-owned, engine-neutral job messages, not an Inspect or sandbox-platform protocol."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import Enum
from typing import Annotated
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from pyrit.models.eval_case import EvalPackageRef  # noqa: TC001  (runtime-required by Pydantic)
from pyrit.models.identifiers.component_identifier import config_hash

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PublicName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]


class EvaluationRuntimeKind(str, Enum):
    """Declared execution semantics; a kind does not imply an installed handler."""

    ORIGINAL_INSPECT = "original_inspect"
    INSPECT_VARIANT = "reviewed_inspect_variant"
    NATIVE_BINDING = "native_binding"


class EvaluationControlKind(str, Enum):
    """Structured commands interpreted only by a reviewed runtime wait boundary."""

    SEND_MESSAGE = "send_message"
    NUDGE = "nudge"
    ADVANCE = "advance"
    STOP = "stop"


class EvaluationJobState(str, Enum):
    """Queue and runtime state, independent of an original grade."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    CANCEL_REQUESTED = "cancel_requested"
    FINALIZING = "finalizing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        """Whether this attempt can no longer append execution events."""
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED, self.INTERRUPTED}


class EvaluationEvidenceState(str, Enum):
    """Evidence retention does not attest source completeness or a grade."""

    ABSENT = "absent"
    UNKNOWN = "unknown"
    SOURCE_RETAINED = "source_retained"
    CANONICAL = "canonical"


class EvaluationCleanupState(str, Enum):
    """Runtime-owned closure, never inferred from a queue acknowledgment."""

    NOT_STARTED = "not_started"
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    FAILED = "failed"


class EvaluationJobEventKind(str, Enum):
    """Finite ordered lifecycle and control events without private payloads."""

    SUBMITTED = "submitted"
    STARTED = "started"
    WAITING = "waiting"
    CONTROL_ACCEPTED = "control_accepted"
    CONTROL_DELIVERED = "control_delivered"
    CANCEL_REQUESTED = "cancel_requested"
    ARTIFACTS_RETAINED = "artifacts_retained"
    FINALIZING = "finalizing"
    TERMINAL = "terminal"


class EvaluationArtifactKind(str, Enum):
    """Evidence formats are explicit; native evidence is not a fabricated Inspect log."""

    INSPECT_EVAL = "inspect_eval"
    NATIVE_EVIDENCE = "native_evidence"


class EvaluationArtifactMediaType(str, Enum):
    """Finite source-evidence encodings, not producer-selected file interpretation."""

    INSPECT_EVAL = "application/octet-stream"
    NATIVE_EVIDENCE = "application/x-ndjson"


class _JobMessage(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, hide_input_in_errors=True, revalidate_instances="always"
    )


class EvaluationJobRequest(_JobMessage):
    """Immutable admitted work; initial input and executable code remain in the source."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    job_id: UUID
    run_id: UUID
    attempt_id: UUID
    runtime: EvaluationRuntimeKind
    source: EvalPackageRef
    case_id: Digest
    execution_profile_sha256: Digest
    controls: tuple[EvaluationControlKind, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def _validate_controls(self) -> EvaluationJobRequest:
        """
        Reject steering in unchanged original execution and duplicate capabilities.

        Returns:
            EvaluationJobRequest: The unchanged, bounded request.

        Raises:
            ValueError: If controls are repeated or present in unchanged Mode 1.
        """
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("Job control capabilities must be unique.")
        if self.runtime is EvaluationRuntimeKind.ORIGINAL_INSPECT and self.controls:
            raise ValueError("Mode 1 does not accept adversarial or interactive controls.")
        return self

    @property
    def request_sha256(self) -> str:
        """The exact request identity, including ordered capability declarations."""
        return config_hash(self.model_dump(mode="json"))

    @property
    def case_run_sha256(self) -> str:
        """One admitted source case in a run, independent of job or attempt aliases."""
        return config_hash(
            {
                "schema_version": self.schema_version,
                "source": self.source.model_dump(mode="json"),
                "case_id": self.case_id,
                "run_id": str(self.run_id),
            }
        )


class EvaluationJobDelivery(_JobMessage):
    """At-least-once delivery points to admitted work, not a second execution request."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    job_id: UUID
    request_sha256: Digest


class EvaluationDeliveryState(str, Enum):
    """A broker may settle started/duplicate delivery, but must retry busy delivery."""

    STARTED = "started"
    DUPLICATE = "duplicate"
    BUSY = "busy"


class EvaluationDeliveryReceipt(_JobMessage):
    """Unambiguous receiver outcome, separate from source results."""

    job_id: UUID
    state: EvaluationDeliveryState


class EvaluationJobRegistration(_JobMessage):
    """A server-installed source/case/profile with explicitly reviewed capabilities."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    runtime: EvaluationRuntimeKind
    source: EvalPackageRef
    case_id: Digest
    execution_profile_sha256: Digest
    controls: tuple[EvaluationControlKind, ...] = Field(default=(), max_length=4)
    artifact_kind: EvaluationArtifactKind

    @model_validator(mode="after")
    def _validate_registration(self) -> EvaluationJobRegistration:
        """
        Keep unchanged Inspect and native evidence distinct.

        Returns:
            EvaluationJobRegistration: A coherent runtime declaration.

        Raises:
            ValueError: If declared capabilities or evidence kind contradict the runtime.
        """
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("Registered control capabilities must be unique.")
        if self.runtime is EvaluationRuntimeKind.ORIGINAL_INSPECT and (
            self.controls or self.artifact_kind is not EvaluationArtifactKind.INSPECT_EVAL
        ):
            raise ValueError("An unchanged Inspect handler cannot steer or return native evidence.")
        if self.runtime is EvaluationRuntimeKind.NATIVE_BINDING and self.artifact_kind is not (
            EvaluationArtifactKind.NATIVE_EVIDENCE
        ):
            raise ValueError("Native handlers must return native evidence, not an Inspect log.")
        return self

    def accepts(self, request: EvaluationJobRequest) -> bool:
        """
        Compare only server-owned execution identity and capability limits.

        Returns:
            bool: Whether this exact source/case/profile is installed.
        """
        return (
            request.runtime is self.runtime
            and request.source == self.source
            and request.case_id == self.case_id
            and request.execution_profile_sha256 == self.execution_profile_sha256
            and set(request.controls) <= set(self.controls)
        )


class EvaluationArtifact(_JobMessage):
    """A bounded flat name and exact immutable bytes, never a caller filesystem path."""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    kind: EvaluationArtifactKind
    media_type: EvaluationArtifactMediaType
    sha256: Digest
    bytes: StrictInt = Field(ge=1, le=16 * 1024 * 1024)

    @model_validator(mode="after")
    def _validate_media_type(self) -> EvaluationArtifact:
        """
        Refuse format labels that reinterpret native evidence as an Inspect archive.

        Returns:
            EvaluationArtifact: The coherent evidence format.

        Raises:
            ValueError: If its media type contradicts its evidence kind.
        """
        expected = EvaluationArtifactMediaType[self.kind.name]
        reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
        if (
            self.name.endswith(".")
            or self.name.casefold() == "manifest.json"
            or (self.name.split(".", 1)[0].casefold() in reserved)
        ):
            raise ValueError("Artifact name is reserved or ambiguous on local filesystems.")
        if self.media_type is not expected:
            raise ValueError("Artifact kind and media type differ.")
        return self


class EvaluationArtifactManifest(_JobMessage):
    """A fenced artifact handoff bound to all admitted request references."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    request: EvaluationJobRequest
    request_sha256: Digest
    fence_id: UUID
    artifacts: tuple[EvaluationArtifact, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def _validate_manifest(self) -> EvaluationArtifactManifest:
        """
        Preserve ordered artifact identity and reject substituted requests.

        Returns:
            EvaluationArtifactManifest: The exact bound artifact inventory.

        Raises:
            ValueError: If request, names, evidence kinds, or byte bounds differ.
        """
        if self.request_sha256 != self.request.request_sha256:
            raise ValueError("Artifact manifest request identity differs.")
        if len({item.name.casefold() for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("Artifact names must be unique.")
        if sum(item.bytes for item in self.artifacts) > 32 * 1024 * 1024:
            raise ValueError("Artifact handoff exceeds its aggregate byte bound.")
        expected = {
            EvaluationRuntimeKind.ORIGINAL_INSPECT: EvaluationArtifactKind.INSPECT_EVAL,
            EvaluationRuntimeKind.NATIVE_BINDING: EvaluationArtifactKind.NATIVE_EVIDENCE,
        }.get(self.request.runtime)
        if expected is not None and any(item.kind is not expected for item in self.artifacts):
            raise ValueError("Artifact evidence kind contradicts the admitted runtime.")
        return self

    @property
    def manifest_sha256(self) -> str:
        """The ordered artifact and request fingerprint."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationCanonicalReceipt(_JobMessage):
    """Actual canonical writer references, not worker grades or queue ACKs."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    job_id: UUID
    request_sha256: Digest
    manifest_sha256: Digest
    artifact_sha256: Digest
    projection_id: PublicName
    source_complete: bool
    score_ids: tuple[UUID, ...] = Field(default=(), max_length=32)
    attack_result_ids: tuple[UUID, ...] = Field(default=(), max_length=32)


class EvaluationWaitBoundary(_JobMessage):
    """One currently active reviewed boundary, not permission to execute a shell."""

    boundary_id: UUID
    name: PublicName
    controls: tuple[EvaluationControlKind, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_controls(self) -> EvaluationWaitBoundary:
        """
        Keep one explicit set of capabilities at this boundary.

        Returns:
            EvaluationWaitBoundary: The unchanged boundary.

        Raises:
            ValueError: If capability names are repeated.
        """
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("Wait boundary control capabilities must be unique.")
        return self


class EvaluationControlRequest(_JobMessage):
    """An idempotent, bounded action at one specific runtime wait boundary."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    command_id: UUID
    boundary_id: UUID
    kind: EvaluationControlKind
    message: str | None = Field(default=None, min_length=1, max_length=8192)

    @model_validator(mode="after")
    def _validate_message(self) -> EvaluationControlRequest:
        """
        Do not reinterpret structured advance/stop commands as arbitrary text.

        Returns:
            EvaluationControlRequest: The unchanged bounded command.

        Raises:
            ValueError: If an action carries inappropriate or overlong text.
        """
        requires_message = self.kind in {EvaluationControlKind.SEND_MESSAGE, EvaluationControlKind.NUDGE}
        if requires_message != (self.message is not None):
            raise ValueError("SendMessage/Nudge require text; Advance/Stop cannot carry text.")
        if self.message is not None and len(self.message.encode("utf-8")) > 8192:
            raise ValueError("Job control text exceeds its UTF-8 byte limit.")
        return self

    @property
    def command_sha256(self) -> str:
        """The exact command identity used to reject mismatched redelivery."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationControlReceipt(_JobMessage):
    """Durable command acceptance; delivery has a separate event, not proof of agent action."""

    job_id: UUID
    command_id: UUID
    accepted_sequence: StrictInt = Field(ge=1)
    duplicate: bool = False


class EvaluationJobEvent(_JobMessage):
    """A gap-free per-job sequence with no private source or command contents."""

    sequence: StrictInt = Field(ge=1)
    kind: EvaluationJobEventKind
    state: EvaluationJobState
    occurred_at: datetime
    command_id: UUID | None = None
    reason: PublicName | None = None

    @model_validator(mode="after")
    def _validate_timestamp(self) -> EvaluationJobEvent:
        """
        Require UTC event timestamps.

        Returns:
            EvaluationJobEvent: A UTC event.

        Raises:
            ValueError: If timestamp or terminal event/state semantics differ.
        """
        offset = self.occurred_at.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Job events require UTC timestamps.")
        if (self.kind is EvaluationJobEventKind.TERMINAL) != self.state.terminal:
            raise ValueError("Only a terminal event may declare a terminal state.")
        return self


class EvaluationJobSnapshot(_JobMessage):
    """Actor-authorized state, evidence/cleanup uncertainty, and ordered event page."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    request: EvaluationJobRequest
    request_sha256: Digest
    state: EvaluationJobState
    evidence: EvaluationEvidenceState
    cleanup: EvaluationCleanupState
    last_sequence: StrictInt = Field(ge=1)
    events: tuple[EvaluationJobEvent, ...] = Field(max_length=256)
    boundary: EvaluationWaitBoundary | None = None
    manifest: EvaluationArtifactManifest | None = None
    canonical: EvaluationCanonicalReceipt | None = None
    reason: PublicName | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> EvaluationJobSnapshot:
        """
        Reject gaps, extra terminal events, and foreign artifact or writer identities.

        Returns:
            EvaluationJobSnapshot: A coherent event page and exact handoff.

        Raises:
            ValueError: If order, state, or canonical identities are incoherent.
        """
        if self.request_sha256 != self.request.request_sha256:
            raise ValueError("Job snapshot request identity differs.")
        sequences = [event.sequence for event in self.events]
        if sequences and (
            sequences != list(range(sequences[0], sequences[-1] + 1)) or sequences[-1] > self.last_sequence
        ):
            raise ValueError("Job events must be ordered and gap-free.")
        terminals = [event for event in self.events if event.kind is EvaluationJobEventKind.TERMINAL]
        if terminals and (len(terminals) != 1 or terminals[0].sequence != self.last_sequence):
            raise ValueError("A terminal event must be the last event exactly once.")
        if self.events and self.events[-1].sequence == self.last_sequence and self.events[-1].state is not self.state:
            raise ValueError("Job state differs from its last event.")
        if self.boundary is not None and self.state is not EvaluationJobState.WAITING:
            raise ValueError("Only a waiting job may expose an active boundary.")
        if self.state is EvaluationJobState.SUCCEEDED and (
            self.canonical is None
            or not self.canonical.source_complete
            or self.evidence is not EvaluationEvidenceState.CANONICAL
            or self.cleanup is not EvaluationCleanupState.VERIFIED
        ):
            raise ValueError("A succeeded job requires actual complete canonical evidence and verified closure.")
        if self.manifest is not None and self.manifest.request != self.request:
            raise ValueError("Job snapshot artifact request differs.")
        if self.canonical is not None and (
            self.canonical.job_id != self.request.job_id
            or self.canonical.request_sha256 != self.request_sha256
            or self.manifest is None
            or self.canonical.manifest_sha256 != self.manifest.manifest_sha256
            or self.canonical.artifact_sha256 not in {artifact.sha256 for artifact in self.manifest.artifacts}
            or self.evidence is not EvaluationEvidenceState.CANONICAL
        ):
            raise ValueError("Job snapshot canonical receipt differs.")
        if self.evidence is EvaluationEvidenceState.CANONICAL and self.canonical is None:
            raise ValueError("Canonical evidence requires an actual writer receipt.")
        return self


class EvaluationJobSubmission(_JobMessage):
    """Accepted immutable work; no execution or grade is implied."""

    job_id: UUID
    request_sha256: Digest
    duplicate: bool = False
    control_capability: str | None = Field(default=None, repr=False)

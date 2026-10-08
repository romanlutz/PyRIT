# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Execution-only worker messages; canonical database references never cross this port."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import Enum
from uuid import UUID  # noqa: TC003

from pydantic import Field, StrictInt, model_validator

from pyrit.models.evaluation_job import (
    Digest,
    EvaluationArtifactManifest,
    EvaluationCleanupState,
    EvaluationControlReceipt,
    EvaluationControlRequest,
    EvaluationDeliveryReceipt,
    EvaluationJobDelivery,
    EvaluationJobEventKind,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationWaitBoundary,
    PublicName,
    _JobMessage,
)
from pyrit.models.identifiers.component_identifier import config_hash


class EvaluationWorkerState(str, Enum):
    """Source execution state, not a canonical import or a benchmark verdict."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        """Whether this worker execution has stopped appending execution events."""
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED, self.INTERRUPTED}


class EvaluationWorkerEvidence(str, Enum):
    """Worker retention never claims API-owned canonical evidence."""

    ABSENT = "absent"
    UNKNOWN = "unknown"
    SOURCE_RETAINED = "source_retained"


class EvaluationWorkerAdmission(_JobMessage):
    """Authenticated gateway admission, separate from the immutable source request."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    request: EvaluationJobRequest
    request_sha256: Digest
    gateway_fence_id: UUID
    actor_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_request(self) -> EvaluationWorkerAdmission:
        if self.request_sha256 != self.request.request_sha256:
            raise ValueError("Worker admission request identity differs.")
        return self

    @property
    def admission_sha256(self) -> str:
        """The exact authenticated actor, request, and gateway dispatch identity."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationWorkerBinding(_JobMessage):
    """Two independently owned dispatch fences and one immutable worker incarnation."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    service_id: PublicName
    worker_incarnation_id: UUID
    worker_fence_id: UUID
    gateway_fence_id: UUID
    job_id: UUID
    run_id: UUID
    attempt_id: UUID
    request_sha256: Digest
    admission_sha256: Digest
    actor_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_fences(self) -> EvaluationWorkerBinding:
        if self.worker_fence_id == self.gateway_fence_id:
            raise ValueError("Worker and gateway dispatch fences must be independent.")
        return self

    @property
    def binding_sha256(self) -> str:
        """The admitted execution binding, not a provider assignment attestation."""
        return config_hash(self.model_dump(mode="json"))

    def accepts(self, admission: EvaluationWorkerAdmission) -> bool:
        """
        Check the complete gateway identity without replacing the worker fence.

        Returns:
            bool: Whether the exact authenticated gateway admission matches.
        """
        return (
            self.job_id == admission.request.job_id
            and self.run_id == admission.request.run_id
            and self.attempt_id == admission.request.attempt_id
            and self.request_sha256 == admission.request_sha256
            and self.admission_sha256 == admission.admission_sha256
            and self.gateway_fence_id == admission.gateway_fence_id
            and self.actor_id == admission.actor_id
        )


class EvaluationWorkerSubmission(_JobMessage):
    """Durable execution admission only, never a grade or canonical receipt."""

    binding: EvaluationWorkerBinding
    duplicate: bool = False


class EvaluationWorkerEvent(_JobMessage):
    """A separate gap-free worker cursor, not the public gateway event sequence."""

    sequence: StrictInt = Field(ge=1)
    kind: EvaluationJobEventKind
    state: EvaluationWorkerState
    occurred_at: datetime
    command_id: UUID | None = None
    reason: PublicName | None = None

    @model_validator(mode="after")
    def _validate_event(self) -> EvaluationWorkerEvent:
        offset = self.occurred_at.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Worker events require UTC timestamps.")
        if self.kind is EvaluationJobEventKind.FINALIZING:
            raise ValueError("A worker cannot claim API canonical finalization.")
        if (self.kind is EvaluationJobEventKind.TERMINAL) != self.state.terminal:
            raise ValueError("Worker terminal events must match terminal execution states.")
        return self


class EvaluationWorkerTerminal(_JobMessage):
    """Source retention and observed closure, without scores or worker database IDs."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    binding: EvaluationWorkerBinding
    state: EvaluationWorkerState
    evidence: EvaluationWorkerEvidence
    cleanup: EvaluationCleanupState
    last_sequence: StrictInt = Field(ge=1)
    manifest_sha256: Digest | None = None
    reason: PublicName | None = None

    @model_validator(mode="after")
    def _validate_terminal(self) -> EvaluationWorkerTerminal:
        if not self.state.terminal:
            raise ValueError("A worker terminal receipt requires a terminal execution state.")
        if (self.evidence is EvaluationWorkerEvidence.SOURCE_RETAINED) != (self.manifest_sha256 is not None):
            raise ValueError("Retained worker evidence requires its immutable manifest.")
        if self.state is EvaluationWorkerState.COMPLETED and (
            self.cleanup is not EvaluationCleanupState.VERIFIED
            or self.evidence is not EvaluationWorkerEvidence.SOURCE_RETAINED
        ):
            raise ValueError("Completed source execution requires retained evidence and observed closure.")
        return self


class EvaluationWorkerSnapshot(_JobMessage):
    """Authenticated execution status with no canonical import-shaped fields."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    request: EvaluationJobRequest
    request_sha256: Digest
    binding: EvaluationWorkerBinding
    state: EvaluationWorkerState
    evidence: EvaluationWorkerEvidence
    cleanup: EvaluationCleanupState
    last_sequence: StrictInt = Field(ge=1)
    events: tuple[EvaluationWorkerEvent, ...] = Field(max_length=256)
    boundary: EvaluationWaitBoundary | None = None
    terminal: EvaluationWorkerTerminal | None = None
    reason: PublicName | None = None

    @model_validator(mode="after")
    def _validate_snapshot(self) -> EvaluationWorkerSnapshot:
        if (
            self.request_sha256 != self.request.request_sha256
            or self.binding.request_sha256 != self.request_sha256
            or self.binding.job_id != self.request.job_id
            or self.binding.run_id != self.request.run_id
            or self.binding.attempt_id != self.request.attempt_id
        ):
            raise ValueError("Worker snapshot request and binding differ.")
        sequences = [event.sequence for event in self.events]
        if sequences and (
            sequences != list(range(sequences[0], sequences[-1] + 1)) or sequences[-1] > self.last_sequence
        ):
            raise ValueError("Worker event pages must be ordered and gap-free.")
        terminals = [event for event in self.events if event.kind is EvaluationJobEventKind.TERMINAL]
        if terminals and (len(terminals) != 1 or terminals[0].sequence != self.last_sequence):
            raise ValueError("The worker terminal event must be last exactly once.")
        if self.events and self.events[-1].sequence == self.last_sequence and self.events[-1].state is not self.state:
            raise ValueError("Worker state differs from its final event.")
        if self.boundary is not None and self.state is not EvaluationWorkerState.WAITING:
            raise ValueError("Only a waiting worker may expose a reviewed boundary.")
        if self.boundary is not None and not set(self.boundary.controls) <= set(self.request.controls):
            raise ValueError("Worker wait capabilities exceed the admitted request.")
        if self.state.terminal != (self.terminal is not None):
            raise ValueError("Terminal worker status requires exactly one terminal receipt.")
        if self.terminal is not None and (
            self.terminal.binding != self.binding
            or self.terminal.state is not self.state
            or self.terminal.evidence is not self.evidence
            or self.terminal.cleanup is not self.cleanup
            or self.terminal.last_sequence != self.last_sequence
        ):
            raise ValueError("Worker status and terminal receipt differ.")
        return self


class EvaluationWorkerCatalog(_JobMessage):
    """Actor-authorized grants intersected with gateway-installed source identities."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    service_id: PublicName
    actor_id: str = Field(min_length=1, max_length=128)
    registrations: tuple[EvaluationJobRegistration, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def _validate_catalog(self) -> EvaluationWorkerCatalog:
        fingerprints = [config_hash(item.model_dump(mode="json")) for item in self.registrations]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("Worker grants must not repeat a registration.")
        return self


class EvaluationWorkerProtocol(_JobMessage):
    """An authenticated compatibility declaration, not execution authority."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    protocol: str = Field(default="pyrit.evaluation-worker.v1", pattern=r"^pyrit\.evaluation-worker\.v1$")
    service_id: PublicName
    schema_sha256: Digest


class EvaluationGatewaySettlementDisposition(str, Enum):
    """Gateway retention/import acknowledgment, not guest or provider termination."""

    CANONICAL_IMPORTED = "canonical_imported"
    RETAINED_ONLY = "retained_only"
    CLOSURE_OBSERVED = "closure_observed"


class EvaluationGatewaySettlement(_JobMessage):
    """Authenticated settlement of both manifests; no canonical database IDs."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    binding: EvaluationWorkerBinding
    worker_manifest_sha256: Digest | None = None
    gateway_manifest_sha256: Digest | None = None
    artifact_sha256: Digest | None = None
    disposition: EvaluationGatewaySettlementDisposition

    @model_validator(mode="after")
    def _validate_disposition(self) -> EvaluationGatewaySettlement:
        digests = (self.worker_manifest_sha256, self.gateway_manifest_sha256, self.artifact_sha256)
        if self.disposition is EvaluationGatewaySettlementDisposition.CLOSURE_OBSERVED:
            if any(value is not None for value in digests):
                raise ValueError("Closure-only acknowledgment must not fabricate retained artifact digests.")
        elif any(value is None for value in digests):
            raise ValueError("Artifact settlement requires both manifest digests and the source artifact digest.")
        return self

    @property
    def settlement_sha256(self) -> str:
        """The immutable receipt identity used for exact retry reconciliation."""
        return config_hash(self.model_dump(mode="json"))


class EvaluationGatewaySettlementReceipt(_JobMessage):
    """Durable acknowledgment without permission to discard uncertain evidence."""

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    job_id: UUID
    binding_sha256: Digest
    settlement_sha256: Digest
    duplicate: bool = False


def evaluation_worker_schema_sha256() -> str:
    """
    Fingerprint the explicit v1 wire types rather than guessing remote compatibility.

    Returns:
        str: The canonical schema fingerprint for this execution-only protocol.
    """
    models = (
        EvaluationWorkerAdmission,
        EvaluationWorkerBinding,
        EvaluationWorkerSubmission,
        EvaluationWorkerEvent,
        EvaluationWorkerTerminal,
        EvaluationWorkerSnapshot,
        EvaluationWorkerCatalog,
        EvaluationWorkerProtocol,
        EvaluationGatewaySettlement,
        EvaluationGatewaySettlementReceipt,
        EvaluationArtifactManifest,
        EvaluationJobDelivery,
        EvaluationDeliveryReceipt,
        EvaluationControlRequest,
        EvaluationControlReceipt,
    )
    return config_hash({model.__name__: model.model_json_schema() for model in models})

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Server-only authority for transferring retained original evidence to durable memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Protocol
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator

from pyrit.backend.services.original_run_admission import (
    OriginalCleanupReceipt,
    OriginalRunBinding,
    OriginalWorkerJob,
)
from pyrit.models import config_hash

if TYPE_CHECKING:
    from pyrit.backend.middleware.auth import AuthenticatedUser
    from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalScorePolicy
    from pyrit.models import EvalCaseRef, EvalRunRef

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReceiptId = Annotated[str, Field(min_length=1, max_length=128)]


class OriginalEvidenceEnvelope(BaseModel):
    """Immutable worker evidence references, never a worker-supplied grade or source program."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: StrictInt = Field(default=1, ge=1, le=1)
    app_run_id: UUID
    job_ref: UUID
    profile_ref: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    operator_oid: str = Field(min_length=1, max_length=128)
    run_instance_id: UUID
    worker_scenario_id: UUID
    worker_scenario_sha256: Digest
    manifest_sha256: Digest
    source_sha256: Digest
    case_run_id: Digest
    model_role: Literal["evaluated"]
    model_route_sha256: Digest
    model_role_receipt_id: ReceiptId
    model_role_sha256: Digest
    source_state: Literal["success", "error", "cancelled"]
    source_complete: StrictBool
    archive_sha256: Digest | None = None
    archive_bytes: StrictInt | None = Field(default=None, ge=1, le=16 * 1024 * 1024)
    inspect_run_id: ReceiptId | None = None
    inspect_eval_id: ReceiptId | None = None
    final_score_event_id: ReceiptId | None = None
    final_score_event_sha256: Digest | None = None
    operation_terminal_receipt_id: ReceiptId | None = None
    operation_terminal_sha256: Digest | None = None
    cleanup: OriginalCleanupReceipt
    cleanup_sha256: Digest | None = None
    worker_score_id: UUID | None = None
    worker_attack_result_id: UUID | None = None

    @model_validator(mode="after")
    def _validate_receipt_pairs(self) -> OriginalEvidenceEnvelope:
        """
        Reject half-receipts and fabricated identifiers for an absent original archive.

        Returns:
            OriginalEvidenceEnvelope: A complete set of available source references.

        Raises:
            ValueError: If a receipt or archive reference is partial.
        """
        pairs = (
            (self.archive_sha256, self.archive_bytes),
            (self.final_score_event_id, self.final_score_event_sha256),
            (self.operation_terminal_receipt_id, self.operation_terminal_sha256),
        )
        if any((left is None) != (right is None) for left, right in pairs):
            raise ValueError("Original evidence receipt references must include their digests.")
        if self.cleanup.proved != (self.cleanup_sha256 is not None):
            raise ValueError("Physical closure requires its independent receipt digest.")
        if self.archive_sha256 is None and any(
            value is not None
            for value in (
                self.inspect_run_id,
                self.inspect_eval_id,
                self.final_score_event_id,
                self.worker_score_id,
                self.worker_attack_result_id,
            )
        ):
            raise ValueError("An absent original archive cannot carry fabricated score or archive identifiers.")
        return self

    @property
    def sha256(self) -> str:
        """The canonical digest authenticated by the host evidence admission provider."""
        return config_hash({"original_evidence": self.model_dump(mode="json", exclude_none=True)})

    @property
    def job(self) -> OriginalWorkerJob:
        """The app-owned job, distinct from any provider lease identity."""
        return OriginalWorkerJob(app_run_id=self.app_run_id, job_ref=self.job_ref)

    @property
    def binding(self) -> OriginalRunBinding:
        """The authenticated operator and opaque approved profile."""
        return OriginalRunBinding(profile_ref=self.profile_ref, operator_oid=self.operator_oid)


@dataclass(frozen=True, kw_only=True)
class OriginalEvidenceAdmission:
    """Server-resolved source policy and case identity, never accepted from a browser."""

    envelope: OriginalEvidenceEnvelope
    run: EvalRunRef
    cases: tuple[EvalCaseRef, ...]
    score_policy: InspectOriginalScorePolicy
    display_values: frozenset[str]


class TrustedOriginalEvidenceProvider(Protocol):
    """Authenticate a fixed worker source envelope and separately authorize retained viewing."""

    async def authorize_intake_async(
        self, *, capability: str, envelope: OriginalEvidenceEnvelope
    ) -> OriginalEvidenceAdmission:
        """
        Atomically consume a short-lived job/actor/source-bound capability, or verify an exact replay.

        The host authenticates the pinned worker manifest, exact Scenario snapshot,
        evaluated-role receipt and distinct terminal-operation receipt, including
        original scoring before teardown. Body-supplied IDs and digests are not
        independent authority. Replays must bind the same app job, actor and
        immutable envelope; admission/concurrency must work across backend processes.

        Returns:
            OriginalEvidenceAdmission: The authenticated fixed source and display policy.
        """
        ...

    async def verify_cleanup_async(
        self, *, job: OriginalWorkerJob, envelope_sha256: str
    ) -> OriginalCleanupReceipt | None:
        """Independently authenticate the retained physical receipt without a new deletion pass."""
        ...

    def resolve_read(
        self,
        *,
        operator: AuthenticatedUser,
        job: OriginalWorkerJob,
        envelope_sha256: str,
    ) -> OriginalEvidenceAdmission | None:
        """Apply a cached actor/run ACL and return the same authenticated manifest, not filesystem handles."""
        ...


_provider: TrustedOriginalEvidenceProvider | None = None


def install_original_evidence_provider(*, provider: TrustedOriginalEvidenceProvider) -> None:
    """Install a separately reviewed host admission adapter from trusted backend startup only."""
    global _provider
    if _provider is not None:
        raise ValueError("An original evidence provider is already installed.")
    _provider = provider


def get_original_evidence_provider() -> TrustedOriginalEvidenceProvider | None:
    """
    Resolve the opt-in host adapter, absent in a public default.

    Returns:
        TrustedOriginalEvidenceProvider | None: The trusted server capability, never source code.
    """
    return _provider


def uninstall_original_evidence_provider(*, provider: TrustedOriginalEvidenceProvider) -> None:
    """Remove only the shutting-down lifespan's owned evidence authority."""
    global _provider
    if _provider is provider:
        _provider = None

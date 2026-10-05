# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Default-off, actor-bound admission for an isolated original Task worker."""

from __future__ import annotations

import logging
import re
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable
from uuid import UUID  # noqa: TC003 - Pydantic resolves this field type at runtime

from pydantic import BaseModel, ConfigDict, Field

from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.models import AttackOutcome, ScenarioResult, ScenarioRunState, ScoreStatus
from pyrit.models.catalog.scenario import (
    OriginalRunAdmission,
    OriginalRunEvidenceLink,
    OriginalRunReason,
    OriginalRunStatus,
    OriginalSourceResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)
APPROVED_ORIGINAL_SCENARIO = "benchmark.approved_original"
_ADMISSION_LIFETIME_SECONDS = 120
_MAX_PENDING_ADMISSIONS = 128
_current_worker_grant: ContextVar[OriginalRunGrant | None] = ContextVar("pyrit_original_worker_grant", default=None)


class OriginalAdmissionError(ValueError):
    """A finite refusal that does not include host or private Task configuration."""

    def __init__(self, *, reason: OriginalRunReason) -> None:
        """Preserve only the approved reason code."""
        super().__init__(reason.value)
        self.reason = reason


class OriginalRunBinding(BaseModel):
    """Internal actor/profile binding on the web process's projected run envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    METADATA_KEY: ClassVar[str] = "approved_original_run"

    profile_ref: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    operator_oid: str = Field(min_length=1, max_length=128)


class OriginalWorkerJob(BaseModel):
    """Opaque app-owned job reference; never a path, endpoint, PID or cloud lease ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    METADATA_KEY: ClassVar[str] = "approved_original_job"

    app_run_id: UUID
    job_ref: UUID


class OriginalCleanupReceipt(BaseModel):
    """Physical containment proof; it does not attest a Sample, score or terminal operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    METADATA_KEY: ClassVar[str] = "approved_original_cleanup"

    state: Literal["proved", "uncontained"]
    receipt_id: str | None = Field(default=None, min_length=1, max_length=128)

    @property
    def proved(self) -> bool:
        """Whether the broker proved exact owned-resource closure."""
        return self.state == "proved" and self.receipt_id is not None


@dataclass(frozen=True, kw_only=True)
class OriginalRunVerification:
    """Authenticated, immutable source proof produced inside the isolated worker/broker."""

    app_run_id: UUID
    job_ref: UUID
    profile_ref: str
    operator_oid: str
    source_state: Literal["success", "error", "cancelled"]
    source_coverage_complete: bool
    original_score: str | None
    pyrit_score_status: ScoreStatus | None
    pyrit_outcome: AttackOutcome | None
    archive_sha256: str | None
    final_score_event_id: str | None
    final_score_event_sha256: str | None
    operation_terminal_receipt_id: str | None
    operation_terminal_sha256: str | None
    source_score_id: UUID | None
    source_attack_result_id: UUID | None
    cleanup_receipt_id: str | None
    persistence_verified: bool
    proof_sha256: str | None
    original_score_available: bool = False


class OriginalRunGrant(Protocol):
    """Host-only operator/profile reservation; never serialized into a request or result."""

    @property
    def profile_ref(self) -> str:
        """The approved server-side profile reference."""
        ...

    @property
    def operator_oid(self) -> str:
        """The authenticated owner's stable identity."""
        ...


class TrustedOriginalRunner(Protocol):
    """Server-installed broker for a *different-process* original TaskOwnedScenario."""

    profile_ref: str
    model_role: Literal["evaluated"]
    launch_mode: Literal["separate_process"]

    def is_authorized(self, *, operator: AuthenticatedUser) -> bool:
        """Apply a local, nonblocking server-side ACL to an already authenticated user."""
        ...

    def allows_display_score(self, *, value: str) -> bool:
        """Allow only explicitly reviewed scalar grades for browser display."""
        ...

    async def readiness_async(self, *, operator: AuthenticatedUser) -> tuple[OriginalRunReason, ...]:
        """Report finite unqualified host/profile/worker conditions without private details."""
        ...

    async def acquire_async(self, *, operator: AuthenticatedUser) -> OriginalRunGrant:
        """Atomically recheck ACL, isolated roots, route, image and the one-slot capacity."""
        ...

    async def prepare_worker_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob, binding: OriginalRunBinding
    ) -> None:
        """Bind isolated roots before worker imports; do not initialize a Scenario in the web process."""
        ...

    async def start_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Start only the fixed reviewed worker executable under its one-use host grant."""
        ...

    async def wait_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Wait for terminal source execution within the host broker's bounded deadline."""
        ...

    async def abort_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Cooperatively abort only this owned worker; never act on caller-supplied lease identifiers."""
        ...

    async def release_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob | None, cancelled: bool
    ) -> OriginalCleanupReceipt:
        """Idempotently observe physical closure; do not execute a second unscoped cleanup."""
        ...

    def verify_cleanup(self, *, job: OriginalWorkerJob) -> OriginalCleanupReceipt | None:
        """Authenticate physical closure independently, including jobs with no source Sample."""
        ...

    def verify_result(self, *, job: OriginalWorkerJob) -> OriginalRunVerification | None:
        """Authenticate a broker proof from private typed evidence, not a web-accessible raw SQLite file."""
        ...


@dataclass(frozen=True, slots=True)
class _IssuedAdmission:
    """One short-lived association between the operator and approved profile."""

    operator_oid: str
    profile_ref: str
    expires_at: float


@runtime_checkable
class OriginalWorkerCancellationGate(Protocol):
    """Optional atomic cancellation boundary before irreversible canonical publication."""

    def request_cancellation(self, *, job: OriginalWorkerJob) -> bool:
        """Accept cancellation before intake, or refuse an already completed source handoff."""
        ...


class OriginalRunGateway:
    """Issue one-use references and verify only opaque, host-authenticated worker proofs."""

    def __init__(self, *, runner: TrustedOriginalRunner) -> None:
        """Install one explicitly reviewed, separate-process evaluated-model runner."""
        OriginalRunAdmission(
            profile_ref=runner.profile_ref, model_role=runner.model_role, status=OriginalRunStatus.READY
        )
        if runner.launch_mode != "separate_process":
            raise ValueError("Original Task runners must use a separate trusted host process.")
        self.runner = runner
        self._issued: dict[str, _IssuedAdmission] = {}
        self._lock = Lock()

    def authorized(self, *, operator: AuthenticatedUser | None) -> bool:
        """
        Require a real authenticated operator and the broker's in-memory ACL.

        Returns:
            bool: Whether this operator may see and launch the approved profile.
        """
        return isinstance(operator, AuthenticatedUser) and self.runner.is_authorized(operator=operator)

    async def offer_async(
        self, *, operator: AuthenticatedUser | None, issue_reference: bool
    ) -> OriginalRunAdmission | None:
        """
        Present finite readiness to an authorized operator, issuing a reference only when ready.

        Returns:
            OriginalRunAdmission | None: A safe offer or no profile for an unauthorized operator.
        """
        if not self.authorized(operator=operator):
            return None
        assert operator is not None
        try:
            conditions = await self.runner.readiness_async(operator=operator)
        except OriginalAdmissionError as error:
            conditions = (error.reason,)
        except Exception as error:
            logger.error("Original worker readiness failed (%s).", type(error).__name__)
            conditions = (OriginalRunReason.PROVIDER_UNQUALIFIED,)
        try:
            finite = [OriginalRunReason(reason) for reason in conditions]
        except (TypeError, ValueError) as error:
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error
        reference = self._issue_reference(operator=operator) if not finite and issue_reference else None
        return OriginalRunAdmission(
            profile_ref=self.runner.profile_ref,
            status=OriginalRunStatus.ADMISSION_PENDING if finite else OriginalRunStatus.READY,
            unmet_conditions=list(dict.fromkeys(finite)),
            admission_ref=reference,
        )

    def _issue_reference(self, *, operator: AuthenticatedUser) -> str:
        """
        Allocate an unpredictable, actor-bound one-use reference from a bounded cache.

        Returns:
            str: A short-lived server-issued admission reference.
        """
        now = monotonic()
        with self._lock:
            for reference in [ref for ref, issued in self._issued.items() if issued.expires_at <= now]:
                del self._issued[reference]
            if len(self._issued) >= _MAX_PENDING_ADMISSIONS:
                raise OriginalAdmissionError(reason=OriginalRunReason.CAPACITY_BUSY)
            reference = secrets.token_urlsafe(32)
            self._issued[reference] = _IssuedAdmission(
                operator_oid=operator.oid,
                profile_ref=self.runner.profile_ref,
                expires_at=now + _ADMISSION_LIFETIME_SECONDS,
            )
            return reference

    async def claim_async(self, *, operator: AuthenticatedUser | None, admission_ref: str) -> OriginalRunGrant:
        """
        Consume a single-use reference and reserve exactly one host-owned worker slot.

        Returns:
            OriginalRunGrant: An internal reservation never included in REST data.

        Raises:
            OriginalAdmissionError: If the ACL, reference, readiness or reservation is not approved.
        """
        if not self.authorized(operator=operator):
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        assert operator is not None
        with self._lock:
            issued = self._issued.get(admission_ref)
            if issued is None or issued.expires_at <= monotonic():
                self._issued.pop(admission_ref, None)
                raise OriginalAdmissionError(reason=OriginalRunReason.ADMISSION_EXPIRED)
            if issued.operator_oid != operator.oid or issued.profile_ref != self.runner.profile_ref:
                raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
            del self._issued[admission_ref]
        try:
            conditions = await self.runner.readiness_async(operator=operator)
            finite = [OriginalRunReason(reason) for reason in conditions]
            if finite:
                raise OriginalAdmissionError(reason=finite[0])
            grant = await self.runner.acquire_async(operator=operator)
        except OriginalAdmissionError:
            raise
        except Exception as error:
            logger.error("Original worker reservation failed (%s).", type(error).__name__)
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error
        if grant.profile_ref != self.runner.profile_ref or grant.operator_oid != operator.oid:
            try:
                await self.runner.release_async(grant=grant, job=None, cancelled=True)
            except Exception as error:
                logger.error("Invalid worker reservation lacks cleanup proof (%s).", type(error).__name__)
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED)
        return grant

    def verify_result(
        self,
        *,
        scenario_result: ScenarioResult,
        job: OriginalWorkerJob,
        binding: OriginalRunBinding,
        cleanup: OriginalCleanupReceipt | None,
    ) -> tuple[OriginalSourceResult, OriginalRunEvidenceLink | None]:
        """
        Validate an authenticated worker proof bound to this app job and physical cleanup.

        Returns:
            tuple[OriginalSourceResult, OriginalRunEvidenceLink | None]: Safe grade/cleanup
                and a server-only proof digest, never private IDs or raw file handles.

        Raises:
            OriginalAdmissionError: If the broker proof is partial, mismatched or unapproved.
        """
        try:
            verified_cleanup = self.runner.verify_cleanup(job=job)
            proof = self.runner.verify_result(job=job)
        except OriginalAdmissionError:
            raise
        except Exception as error:
            logger.error("Authenticated original-worker readback failed (%s).", type(error).__name__)
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error
        if cleanup != verified_cleanup or (
            scenario_result.scenario_run_state is ScenarioRunState.COMPLETED and cleanup is None
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        if proof is None:
            return self._ungraded_result(
                binding=binding,
                cleanup=verified_cleanup,
                source_state="cancelled"
                if scenario_result.scenario_run_state is ScenarioRunState.CANCELLED
                else "error",
            ), None
        try:
            if (
                proof.app_run_id != scenario_result.id
                or proof.job_ref != job.job_ref
                or proof.profile_ref != binding.profile_ref
                or proof.operator_oid != binding.operator_oid
                or not proof.proof_sha256
            ):
                raise ValueError("Original worker proof is not bound to this app job.")
            if verified_cleanup is not None and (
                proof.cleanup_receipt_id != verified_cleanup.receipt_id
                or (proof.cleanup_receipt_id is not None) != verified_cleanup.proved
            ):
                raise ValueError("Physical closure differs from the worker's signed proof.")
            if type(proof.original_score_available) is not bool:
                raise ValueError("Original grade availability requires explicit source authority.")
            has_grade = proof.original_score is not None or proof.original_score_available
            if has_grade and (
                not proof.source_coverage_complete
                or not proof.persistence_verified
                or proof.pyrit_score_status not in (ScoreStatus.COMPLETE, ScoreStatus.UNDETERMINED)
                or proof.pyrit_outcome is not AttackOutcome.UNDETERMINED
                or re.fullmatch(r"[0-9a-f]{64}", proof.archive_sha256 or "") is None
                or not proof.final_score_event_id
                or re.fullmatch(r"[0-9a-f]{64}", proof.final_score_event_sha256 or "") is None
                or proof.source_score_id is None
                or proof.source_attack_result_id is None
            ):
                raise ValueError("Original score or archive proof is incomplete.")
            if proof.source_state == "success" and (
                not has_grade
                or not proof.operation_terminal_receipt_id
                or re.fullmatch(r"[0-9a-f]{64}", proof.operation_terminal_sha256 or "") is None
            ):
                raise ValueError("Successful original execution requires a grade and terminal-operation proof.")
            if proof.pyrit_outcome not in (None, AttackOutcome.UNDETERMINED):
                raise ValueError("Original grade has an unapproved attack-success mapping.")
            if proof.original_score is not None:
                if not self.runner.allows_display_score(value=proof.original_score):
                    raise ValueError("Original grade has no approved browser display policy.")
            elif proof.pyrit_score_status is not None or proof.pyrit_outcome is not None:
                raise ValueError("A missing source grade cannot claim a PyRIT mapped score.")
            state = (
                OriginalRunStatus.COMPLETED
                if proof.source_state == "success"
                and verified_cleanup is not None
                and verified_cleanup.proved
                and scenario_result.scenario_run_state not in (ScenarioRunState.FAILED, ScenarioRunState.CANCELLED)
                else OriginalRunStatus.CLEANUP_UNCERTAIN
                if verified_cleanup is None or not verified_cleanup.proved
                else OriginalRunStatus.FAILED_SOURCE_VERIFIED
                if has_grade
                else OriginalRunStatus.FAILED_UNGRADED
            )
            reason = (
                None
                if state is OriginalRunStatus.COMPLETED
                else OriginalRunReason.CLEANUP_PENDING
                if state is OriginalRunStatus.CLEANUP_UNCERTAIN
                else OriginalRunReason.SOURCE_UNVERIFIED
            )
            safe = OriginalSourceResult(
                profile_ref=binding.profile_ref,
                status=state,
                source_state=proof.source_state,
                source_coverage_complete=proof.source_coverage_complete,
                original_score=proof.original_score,
                original_score_available=proof.original_score_available,
                pyrit_score_status=proof.pyrit_score_status,
                pyrit_outcome=(
                    AttackOutcome.UNDETERMINED if proof.pyrit_outcome is AttackOutcome.UNDETERMINED else None
                ),
                cleanup_state=verified_cleanup.state if verified_cleanup is not None else "pending",
                reason=reason,
            )
            link = OriginalRunEvidenceLink(
                job_ref=job.job_ref,
                proof_sha256=proof.proof_sha256,
            )
            return safe, link
        except (AttributeError, TypeError, ValueError) as error:
            if scenario_result.scenario_run_state in (ScenarioRunState.FAILED, ScenarioRunState.CANCELLED):
                logger.warning("Original source proof is invalid; displaying only independently verified cleanup.")
                return self._ungraded_result(
                    binding=binding,
                    cleanup=verified_cleanup,
                    source_state="cancelled"
                    if scenario_result.scenario_run_state is ScenarioRunState.CANCELLED
                    else "error",
                ), None
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error

    def _ungraded_result(
        self,
        *,
        binding: OriginalRunBinding,
        cleanup: OriginalCleanupReceipt | None,
        source_state: Literal["error", "cancelled"],
    ) -> OriginalSourceResult:
        """
        Show physical cleanup truth without fabricating a Sample or source score.

        Returns:
            OriginalSourceResult: Explicitly ungraded failure, or uncertain containment.
        """
        proved = cleanup is not None and cleanup.proved
        return OriginalSourceResult(
            profile_ref=binding.profile_ref,
            status=OriginalRunStatus.FAILED_UNGRADED if proved else OriginalRunStatus.CLEANUP_UNCERTAIN,
            source_state=source_state,
            source_coverage_complete=False,
            cleanup_state="proved" if proved else "uncontained",
            reason=OriginalRunReason.SOURCE_UNVERIFIED if proved else OriginalRunReason.CLEANUP_PENDING,
        )


def get_bound_original_run_grant() -> OriginalRunGrant:
    """
    Resolve an original grant only from inside a trusted, independently isolated worker.

    Returns:
        OriginalRunGrant: The worker-bound grant.

    Raises:
        OriginalAdmissionError: If no host broker bound the worker.
    """
    grant = _current_worker_grant.get()
    if grant is None:
        raise OriginalAdmissionError(reason=OriginalRunReason.RUNNER_NOT_CONFIGURED)
    return grant


@contextmanager
def bind_original_worker_grant(*, grant: OriginalRunGrant) -> Iterator[None]:
    """Bind the one-use host grant inside the isolated worker, never in the web service."""
    if _current_worker_grant.get() is not None:
        raise OriginalAdmissionError(reason=OriginalRunReason.PROFILE_NOT_ADMITTED)
    token = _current_worker_grant.set(grant)
    try:
        yield
    finally:
        _current_worker_grant.reset(token)


_gateway: OriginalRunGateway | None = None


def install_trusted_original_runner(*, runner: TrustedOriginalRunner) -> None:
    """Install one reviewed separate-process broker from trusted backend startup code."""
    global _gateway
    if _gateway is not None:
        raise ValueError("An original runner is already installed in this backend process.")
    _gateway = OriginalRunGateway(runner=runner)


def get_original_run_gateway() -> OriginalRunGateway | None:
    """
    Return the trusted in-process broker interface; absent in the public default.

    Returns:
        OriginalRunGateway | None: A configured capability, never a user-provided runner.
    """
    return _gateway


def uninstall_trusted_original_runner(*, runner: TrustedOriginalRunner) -> None:
    """Remove only the shutting-down lifespan's owned runner."""
    global _gateway
    if _gateway is not None and _gateway.runner is runner:
        _gateway = None

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One lifespan-owned trusted original child, with independent closure and canonical import."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import JsonValue, TypeAdapter

from pyrit.backend.models.original_worker import CANCEL_FILENAME, CohostWorkerRequest, CohostWorkerTerminal
from pyrit.backend.services.original_evidence_admission import OriginalEvidenceAdmission, OriginalEvidenceEnvelope
from pyrit.backend.services.original_evidence_service import (
    OriginalEvidenceReceipt,
    get_original_evidence_service,
)
from pyrit.backend.services.original_model_receipt import OriginalModelReceipt
from pyrit.backend.services.original_model_relay import OriginalModelRelay
from pyrit.backend.services.original_run_admission import (
    OriginalAdmissionError,
    OriginalCleanupReceipt,
    OriginalRunBinding,
    OriginalRunGrant,
    OriginalRunVerification,
    OriginalWorkerJob,
)
from pyrit.backend.services.original_sandbox_observer import OriginalSandboxObserver
from pyrit.backend.services.original_worker_artifacts import OriginalWorkerArtifacts
from pyrit.backend.services.original_worker_preflight import CohostPreflight, CohostPreflightError
from pyrit.models import AttackOutcome, EvalRunRef, config_hash
from pyrit.models.catalog.scenario import OriginalRunReason

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.backend.middleware.auth import AuthenticatedUser
    from pyrit.backend.services.original_worker_state import CohostValidationState

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class _OwnedGrant:
    profile_ref: str
    operator_oid: str
    reservation_id: UUID = field(default_factory=uuid4)


@dataclass(kw_only=True)
class _OwnedJob:
    job: OriginalWorkerJob
    binding: OriginalRunBinding
    grant: _OwnedGrant
    control_id: UUID
    root: Path
    request: CohostWorkerRequest | None = None
    prepare_task: asyncio.Task[None] | None = None
    spawn_task: asyncio.Task[None] | None = None
    observe_task: asyncio.Task[None] | None = None
    finalize_task: asyncio.Task[OriginalCleanupReceipt] | None = None
    process: asyncio.subprocess.Process | None = None
    terminal: CohostWorkerTerminal | None = None
    cleanup: OriginalCleanupReceipt | None = None
    artifacts: OriginalWorkerArtifacts | None = None
    receipt: OriginalEvidenceReceipt | None = None
    capability: str | None = field(default=None, repr=False)
    relay_capability: str | None = field(default=None, repr=False)
    cancel_requested: bool = False
    intake_started: bool = False
    active_deadline: float = 0.0
    cleanup_deadline: float = 0.0
    failure_code: str | None = None


class OriginalWorkerSupervisor:
    """Own prepare/start/wait/cancel/exit and release the slot only after physical closure."""

    model_role: Literal["evaluated"] = "evaluated"
    launch_mode: Literal["separate_process"] = "separate_process"
    STDOUT_LIMIT = 4096
    STDERR_LIMIT = 16_384
    MAX_RETAINED_JOBS = 128
    RUN_SUBDIRECTORIES = ("home", "appdata", "localappdata", "cache", "config", "tmp", "results")

    def __init__(
        self,
        *,
        preflight: CohostPreflight,
        relay: OriginalModelRelay,
        validation_state: CohostValidationState | None = None,
    ) -> None:
        """Bind one exact source/profile; the trusted worker is not an evaluated guest."""
        self.preflight = preflight
        self.config = preflight.config
        self.profile_ref = self.config.profile_ref
        self.relay = relay
        self.validation_state = validation_state
        self.observer = OriginalSandboxObserver(preflight=preflight)
        self._jobs: dict[UUID, _OwnedJob] = {}
        self._retained: dict[UUID, OriginalEvidenceAdmission] = {}
        self._lock = asyncio.Lock()
        self._grant: _OwnedGrant | None = None
        self._quarantined = False
        self._stopping = False
        self._active_marker = self.config.jobs_root / "active-original-job.json"
        self._authority_key: bytes | None = None
        self._retained_expiry: dict[UUID, datetime] = {}

    async def startup_async(self) -> None:
        """Verify installed worker/resources and restore only backend-authenticated retained views."""
        await self.preflight.verify_staging_async()
        self._quarantined = await asyncio.to_thread(self._active_marker.exists) or (
            self.validation_state is not None and self.validation_state.has_unknown_job()
        )
        self._authority_key = (
            self.validation_state.authority_key
            if self.validation_state is not None
            else await asyncio.to_thread(self._load_key)
        )
        self._retained = await asyncio.to_thread(self._restore_retained)

    def is_authorized(self, *, operator: AuthenticatedUser) -> bool:
        """
        Require the approved actor and group together.

        Returns:
            bool: Whether this Graph-authenticated actor may launch/view the configured source.
        """
        return operator.oid in self.config.allowed_operator_oids and bool(
            self.config.allowed_group_ids.intersection(operator.groups)
        )

    def allows_display_score(self, *, value: str) -> bool:
        """
        Display only a reviewed scalar without inferring attack success.

        Returns:
            bool: Whether the original value has a public display policy.
        """
        return value in self.config.source.display_values

    async def readiness_async(self, *, operator: AuthenticatedUser) -> tuple[OriginalRunReason, ...]:
        """
        Expose finite admission, not source paths or cloud IDs.

        Returns:
            tuple[OriginalRunReason, ...]: Exact unmet readiness/capacity conditions.
        """
        if not self.is_authorized(operator=operator):
            return (OriginalRunReason.OPERATOR_NOT_AUTHORIZED,)
        if self._quarantined:
            return (OriginalRunReason.CLEANUP_PENDING,)
        if self._stopping or self._grant is not None or len(self._retained) >= self.MAX_RETAINED_JOBS:
            return (OriginalRunReason.CAPACITY_BUSY,)
        if self.validation_state is not None:
            if not self.validation_state.is_available():
                return (OriginalRunReason.CLEANUP_PENDING,)
            if not self.validation_state.can_reserve():
                return (OriginalRunReason.VALIDATION_SCOPE_EXHAUSTED,)
        if not self.relay.allows_new_run():
            return (OriginalRunReason.MODEL_ROUTE_UNVERIFIED,)
        self.preflight.verify_identity_environment()
        await self.preflight.verify_resources_async()
        return ()

    async def acquire_async(self, *, operator: AuthenticatedUser) -> OriginalRunGrant:
        """
        Atomically reserve one child slot.

        Returns:
            OriginalRunGrant: A host-only identity-bound reservation.
        """
        async with self._lock:
            reasons = await self.readiness_async(operator=operator)
            if reasons:
                raise OriginalAdmissionError(reason=reasons[0])
            grant = _OwnedGrant(profile_ref=self.profile_ref, operator_oid=operator.oid)
            if self.validation_state is not None:
                await self.validation_state.reserve_async(
                    reservation_id=grant.reservation_id,
                    operator_oid=grant.operator_oid,
                    profile_ref=grant.profile_ref,
                )
            self._grant = grant
            return self._grant

    async def prepare_worker_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob, binding: OriginalRunBinding
    ) -> None:
        """Allocate roots before the child can import any source."""
        owned = self._require_grant(grant)
        if binding.profile_ref != owned.profile_ref or binding.operator_oid != owned.operator_oid:
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        if job.job_ref in self._jobs or any(item.grant is owned for item in self._jobs.values()):
            raise OriginalAdmissionError(reason=OriginalRunReason.CAPACITY_BUSY)
        control = uuid4()
        record = _OwnedJob(
            job=job, binding=binding, grant=owned, control_id=control, root=self.config.jobs_root / str(control)
        )
        self._jobs[job.job_ref] = record
        if self.validation_state is not None:
            await self.validation_state.bind_job_async(
                reservation_id=owned.reservation_id,
                app_run_id=job.app_run_id,
                job_ref=job.job_ref,
                control_id=control,
            )
        record.prepare_task = asyncio.create_task(self._prepare_async(record=record))
        await asyncio.shield(record.prepare_task)

    async def start_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Spawn only the fixed reviewed executable, with no HTTP-selected Python/module."""
        record = self._record(grant=grant, job=job)
        if record.spawn_task is not None or record.process is not None:
            raise OriginalAdmissionError(reason=OriginalRunReason.CAPACITY_BUSY)
        record.spawn_task = asyncio.create_task(self._spawn_async(record=record))
        await asyncio.shield(record.spawn_task)

    async def wait_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Observe real exit before any source import or slot release."""
        record = self._record(grant=grant, job=job)
        if record.observe_task is None:
            raise CohostPreflightError("The owned original worker was not started.")
        try:
            async with asyncio.timeout_at(record.active_deadline):
                await asyncio.shield(record.observe_task)
        except TimeoutError as error:
            record.failure_code = "worker_active_deadline"
            await self.abort_worker_async(grant=grant, job=job)
            raise CohostPreflightError("Original worker active deadline expired.") from error
        await self._finalize_once_async(record=record)
        if record.terminal is not None and record.terminal.state == "cancelled":
            raise asyncio.CancelledError
        if record.terminal is None or record.terminal.state != "success" or record.receipt is None:
            raise CohostPreflightError("Original worker failed without a complete canonical source result.")

    def request_cancellation(self, *, job: OriginalWorkerJob) -> bool:
        """
        Stop new posts atomically before the scheduler yields to canonical publication.

        Returns:
            bool: False after the exited source has entered its irreversible canonical handoff.
        """
        record = self._jobs.get(job.job_ref)
        if record is None or record.job != job:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        if record.intake_started:
            return False
        record.cancel_requested = True
        if record.relay_capability is not None:
            self.relay.revoke(job_ref=job.job_ref)
        return True

    async def abort_worker_async(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob) -> None:
        """Request cooperative cancellation, preserving the launch-owned cleanup reserve."""
        record = self._record(grant=grant, job=job, allow_released=True)
        record.cancel_requested = True
        if record.relay_capability is not None:
            self.relay.revoke(job_ref=job.job_ref)
        if record.prepare_task is not None:
            await asyncio.shield(record.prepare_task)
        if record.spawn_task is not None:
            await asyncio.shield(record.spawn_task)
        await asyncio.to_thread(
            OriginalModelRelay._write_atomic,
            path=record.root / CANCEL_FILENAME,
            content=json.dumps(
                {"abi_version": 1, "job_ref": str(job.job_ref), "run_instance_id": str(record.control_id)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
        if record.process is None or record.observe_task is None:
            return
        remaining = max(0.0, record.cleanup_deadline - asyncio.get_running_loop().time())
        try:
            async with asyncio.timeout(remaining):
                await asyncio.shield(record.observe_task)
        except TimeoutError:
            record.failure_code = "worker_cleanup_deadline"
            if record.process.returncode is None:
                record.process.kill()
            # Observe only the exact process we created, never an arbitrary PID/name.
            await asyncio.wait_for(record.process.wait(), timeout=5)
            try:
                await asyncio.wait_for(asyncio.shield(record.observe_task), timeout=5)
            except (TimeoutError, CohostPreflightError):
                logger.error("Original child stream/exit observation remains unverified.")

    async def release_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob | None, cancelled: bool
    ) -> OriginalCleanupReceipt:
        """
        Keep uncertain containment quarantined instead of freeing a new worker slot.

        Returns:
            OriginalCleanupReceipt: Independent exact closure or explicit uncontained state.
        """
        if job is None:
            owned = self._require_grant(grant)
            if any(item.grant is owned for item in self._jobs.values()):
                raise CohostPreflightError("A prepared owned job cannot be released without its exact identity.")
            self._grant = None
            if self.validation_state is not None:
                await self.validation_state.finish_job_async(reservation_id=owned.reservation_id, proved=True)
            return OriginalCleanupReceipt(state="proved", receipt_id=f"cohost-no-child-{owned.reservation_id}")
        record = self._record(grant=grant, job=job, allow_released=True)
        if record.prepare_task is not None:
            try:
                await asyncio.shield(record.prepare_task)
            except (OSError, ValueError):
                record.failure_code = "worker_prepare_failed"
        if record.process is not None and record.process.returncode is None:
            await self.abort_worker_async(grant=grant, job=job)
        elif cancelled:
            record.cancel_requested = record.cancel_requested or record.terminal is None
        return await self._finalize_once_async(record=record)

    def verify_cleanup(self, *, job: OriginalWorkerJob) -> OriginalCleanupReceipt | None:
        """
        Read the supervisor's authenticated physical result.

        Returns:
            OriginalCleanupReceipt | None: Exact independently observed closure.
        """
        record = self._jobs.get(job.job_ref)
        return record.cleanup if record is not None and record.job == job else None

    def verify_result(self, *, job: OriginalWorkerJob) -> OriginalRunVerification | None:
        """
        Return only the canonical backend's linked original result.

        Returns:
            OriginalRunVerification | None: Durable source authority, or no invented grade.
        """
        record = self._jobs.get(job.job_ref)
        if (
            record is None
            or record.job != job
            or record.receipt is None
            or record.artifacts is None
            or record.failure_code == "worker_state_unverified"
        ):
            return None
        receipt, envelope = record.receipt, record.artifacts.admission.envelope
        source = receipt.source_result
        return OriginalRunVerification(
            app_run_id=job.app_run_id,
            job_ref=job.job_ref,
            profile_ref=record.binding.profile_ref,
            operator_oid=record.binding.operator_oid,
            source_state=envelope.source_state,
            source_coverage_complete=source.source_coverage_complete,
            original_score=source.original_score,
            original_score_available=source.original_score_available,
            pyrit_score_status=source.pyrit_score_status,
            pyrit_outcome=AttackOutcome.UNDETERMINED if source.pyrit_outcome is not None else None,
            archive_sha256=envelope.archive_sha256,
            final_score_event_id=envelope.final_score_event_id,
            final_score_event_sha256=envelope.final_score_event_sha256,
            operation_terminal_receipt_id=envelope.operation_terminal_receipt_id,
            operation_terminal_sha256=envelope.operation_terminal_sha256,
            source_score_id=receipt.score_id,
            source_attack_result_id=receipt.attack_result_id,
            cleanup_receipt_id=envelope.cleanup.receipt_id,
            persistence_verified=receipt.persistence_verified,
            proof_sha256=config_hash(
                {"envelope": envelope.sha256, "canonical_receipt": receipt.model_dump(mode="json")}
            ),
        )

    async def authorize_intake_async(
        self, *, capability: str, envelope: OriginalEvidenceEnvelope
    ) -> OriginalEvidenceAdmission:
        """
        Admit only post-exit exact source bytes using the backend's private one-use authority.

        Returns:
            OriginalEvidenceAdmission: Fixed approved policy; no HTTP-authored source or grade.
        """
        record = self._jobs.get(envelope.job_ref)
        if (
            record is None
            or record.artifacts is None
            or record.capability is None
            or not hmac.compare_digest(record.capability, capability)
            or record.artifacts.admission.envelope != envelope
            or record.cleanup != envelope.cleanup
            or record.process is None
            or record.process.returncode is None
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        return record.artifacts.admission

    async def verify_cleanup_async(
        self, *, job: OriginalWorkerJob, envelope_sha256: str
    ) -> OriginalCleanupReceipt | None:
        """
        Authenticate the previously observed receipt; never rerun destructive cleanup.

        Returns:
            OriginalCleanupReceipt | None: Same retained authority for exact replay.
        """
        admission = self._retained.get(job.job_ref)
        if admission is not None and admission.envelope.job == job and admission.envelope.sha256 == envelope_sha256:
            return admission.envelope.cleanup
        record = self._jobs.get(job.job_ref)
        if (
            record is not None
            and record.job == job
            and record.artifacts is not None
            and record.artifacts.admission.envelope.sha256 == envelope_sha256
        ):
            return record.cleanup
        return None

    def resolve_read(
        self, *, operator: AuthenticatedUser, job: OriginalWorkerJob, envelope_sha256: str
    ) -> OriginalEvidenceAdmission | None:
        """
        Reauthorize actor and current group before protected retained readback.

        Returns:
            OriginalEvidenceAdmission | None: The same signed source policy, never filesystem handles.
        """
        admission = self._retained.get(job.job_ref)
        if (
            admission is None
            or not self.is_authorized(operator=operator)
            or admission.envelope.operator_oid != operator.oid
            or admission.envelope.job != job
            or admission.envelope.sha256 != envelope_sha256
            or datetime.now(UTC) >= self._retained_expiry.get(job.job_ref, datetime.min.replace(tzinfo=UTC))
        ):
            return None
        return admission

    def has_active_work(self) -> bool:
        """
        Include owned spawn/stream/SDK work even after caller cancellation.

        Returns:
            bool: Whether runtime memory/credentials must remain alive.
        """
        return (
            self._grant is not None
            or bool(self.observer.tasks)
            or any(
                task is not None and not task.done()
                for record in self._jobs.values()
                for task in (record.prepare_task, record.spawn_task, record.observe_task, record.finalize_task)
            )
        )

    async def shutdown_async(self) -> None:
        """Cancel the one admitted child, drain owned work, leave uncertain containment explicit."""
        self._stopping = True
        for record in self._jobs.values():
            if record.process is not None and record.process.returncode is None:
                await self.abort_worker_async(grant=record.grant, job=record.job)
            if record.cleanup is None:
                await self._finalize_once_async(record=record)
        if self.observer.tasks:
            await asyncio.gather(*tuple(self.observer.tasks), return_exceptions=True)

    async def _prepare_async(self, *, record: _OwnedJob) -> None:
        await asyncio.to_thread(self._create_root, record=record)
        record.relay_capability = self.relay.admit(
            job_ref=record.job.job_ref, run_id=record.control_id, run_root=record.root
        )

    async def _spawn_async(self, *, record: _OwnedJob) -> None:
        if record.cancel_requested:
            return
        self.preflight.verify_identity_environment()
        await self.preflight.verify_resources_async()
        if await asyncio.to_thread(self.preflight._file_digest, path=self.config.worker_entrypoint) != (
            self.config.worker_entrypoint_sha256
        ):
            raise CohostPreflightError("The original worker changed after startup qualification.")
        assert record.relay_capability is not None
        now = datetime.now(UTC)
        active = now + timedelta(seconds=self.config.active_timeout_seconds)
        record.active_deadline = asyncio.get_running_loop().time() + self.config.active_timeout_seconds
        record.cleanup_deadline = record.active_deadline + self.config.cleanup_timeout_seconds
        record.request = CohostWorkerRequest(
            app_run_id=record.job.app_run_id,
            job_ref=record.job.job_ref,
            run_instance_id=record.control_id,
            operator_oid=record.binding.operator_oid,
            profile_ref=self.profile_ref,
            source_alias=self.config.source_alias,
            source_sha256=self.config.source.spec.package.source_sha256,
            manifest_sha256=config_hash(
                OriginalWorkerArtifacts.server_manifest(
                    app_run_id=str(record.job.app_run_id),
                    job_ref=str(record.job.job_ref),
                    operator_oid=record.binding.operator_oid,
                    profile_ref=self.profile_ref,
                    config=self.config,
                )
            ),
            source_root=self.config.source_root,
            run_root=record.root,
            active_timeout_seconds=self.config.active_timeout_seconds,
            cleanup_timeout_seconds=300,
            active_deadline_utc=active,
            cleanup_deadline_utc=active + timedelta(seconds=300),
            relay_url=f"{self.config.relay_origin}/api/internal/original-model/{record.job.job_ref}",
            relay_capability=record.relay_capability,
        )
        payload = record.request.model_dump_json().encode() + b"\n"
        if len(payload) > 65_536:
            raise CohostPreflightError("The fixed worker stdin request exceeds its byte bound.")
        record.process = await asyncio.create_subprocess_exec(
            str(self.config.worker_python),
            str(self.config.worker_entrypoint),
            cwd=str(self.config.source_root),
            env=self.preflight.worker_environment(run_root=record.root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=self.STDERR_LIMIT,
        )
        record.observe_task = asyncio.create_task(self._observe_exit_async(record=record))
        assert record.process.stdin is not None
        record.process.stdin.write(payload)
        await asyncio.wait_for(record.process.stdin.drain(), timeout=10)
        record.process.stdin.close()

    async def _observe_exit_async(self, *, record: _OwnedJob) -> None:
        assert record.process is not None and record.process.stdout is not None and record.process.stderr is not None
        output_task = asyncio.create_task(
            self._read_stream_async(stream=record.process.stdout, limit=self.STDOUT_LIMIT)
        )
        error_task = asyncio.create_task(self._read_stream_async(stream=record.process.stderr, limit=self.STDERR_LIMIT))
        results = await asyncio.gather(output_task, error_task, return_exceptions=True)
        await record.process.wait()
        if any(isinstance(result, BaseException) for result in results):
            raise CohostPreflightError("Original child stream protocol/drain failed.")
        output = results[0]
        assert isinstance(output, bytes)
        try:
            terminal = CohostWorkerTerminal.model_validate_json(output)
            if terminal.job_ref != record.job.job_ref or terminal.run_instance_id != record.control_id:
                raise CohostPreflightError("Original child terminal is not its owned control job.")
            record.terminal = terminal
        except ValueError as error:
            record.failure_code = "worker_terminal_invalid"
            raise CohostPreflightError("Original child did not emit exactly one safe terminal object.") from error
        if terminal.state == "success" and record.process.returncode != 0:
            record.failure_code = "worker_exit_failed"
            raise CohostPreflightError("Original source success did not accompany a successful actual process exit.")

    async def _finalize_once_async(self, *, record: _OwnedJob) -> OriginalCleanupReceipt:
        if record.finalize_task is None:
            record.finalize_task = asyncio.create_task(self._finalize_async(record=record))
        return await asyncio.shield(record.finalize_task)

    async def _finalize_async(self, *, record: _OwnedJob) -> OriginalCleanupReceipt:
        proved = False
        role: dict[str, JsonValue] | None = None
        try:
            if record.process is not None and record.process.returncode is None:
                raise CohostPreflightError("Physical closure cannot precede actual child exit.")
            if record.observe_task is not None and not record.observe_task.done():
                raise CohostPreflightError("Owned original process stream observation is not finished.")
            if record.relay_capability is not None:
                role = await self.relay.close_async(job_ref=record.job.job_ref)
                if role.get("upstream_drained") is not True or role.get("active_requests") != 0:
                    raise CohostPreflightError("The exact original model role did not drain.")
            if record.process is None:
                proved = not (record.spawn_task is not None and not record.spawn_task.done())
            else:
                closing = await asyncio.to_thread(OriginalWorkerArtifacts.read_closure, record.root)
                lease, drain = closing.get("lease_closure"), closing.get("sdk_thread_drain")
                if (
                    not isinstance(lease, dict)
                    or not isinstance(drain, dict)
                    or closing.get("cleanup_errors") != []
                    or drain.get("sdk_calls_drained") is not True
                    or drain.get("active_sdk_calls") != 0
                    or record.failure_code == "worker_cleanup_deadline"
                ):
                    raise CohostPreflightError("The owned original SDK cleanup/thread drain was not proved.")
                remaining = record.cleanup_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise CohostPreflightError(
                        "Independent physical observation exceeded the original cleanup reserve."
                    )
                await self.observer.verify_async(run_id=record.control_id, closure=lease, timeout_seconds=remaining)
                proved = True
        except (OSError, ValueError, TimeoutError) as error:
            logger.error("Original containment is unverified (%s).", type(error).__name__)
            record.failure_code = record.failure_code or "worker_cleanup_unverified"
        record.cleanup = (
            OriginalCleanupReceipt(state="proved", receipt_id=f"cohost-cleanup-{record.control_id}")
            if proved
            else OriginalCleanupReceipt(state="uncontained")
        )
        if proved and record.request is not None and record.terminal is not None and record.process is not None:
            failure_stage = "worker_artifacts_unverified"
            try:
                artifacts = await asyncio.to_thread(
                    OriginalWorkerArtifacts.read,
                    config=self.config,
                    request=record.request,
                    terminal=record.terminal,
                    process_id=record.process.pid,
                )
                envelope = artifacts.admission.envelope
                failure_stage = "worker_role_unverified"
                if (
                    envelope.cleanup != record.cleanup
                    or role is None
                    or envelope.model_role_receipt_id != role.get("receipt_id")
                    or envelope.model_role_sha256 != role.get("receipt_sha256")
                    or (envelope.source_state == "success" and role.get("usage_complete") is not True)
                    or (record.cancel_requested and envelope.source_state == "success")
                ):
                    raise CohostPreflightError(
                        "Source intake disagrees with actual cancellation/role/physical closure."
                    )
                record.artifacts = artifacts
                failure_stage = "worker_model_coverage_unverified"
                await asyncio.to_thread(
                    OriginalModelReceipt.verify,
                    archive=artifacts.archive,
                    role=role,
                    control_id=record.control_id,
                    require_success=envelope.source_state == "success",
                )
                failure_stage = "worker_retained_authority_unverified"
                if self.validation_state is not None:
                    await self.validation_state.finish_job_async(
                        reservation_id=record.grant.reservation_id, proved=True, envelope=envelope
                    )
                    self._retained_expiry[record.job.job_ref] = self.validation_state.scope.expires_at
                else:
                    await asyncio.to_thread(self._retain, record=record)
                self._retained[record.job.job_ref] = artifacts.admission
                failure_stage = "worker_canonical_intake_unverified"
                if record.cancel_requested and envelope.source_state == "success":
                    raise CohostPreflightError("Cancelled original source cannot enter graded canonical intake.")
                record.intake_started = True
                record.capability = secrets.token_urlsafe(32)
                record.receipt = await get_original_evidence_service().intake_async(
                    capability=record.capability, envelope=envelope, archive=artifacts.archive
                )
                if self.preflight.native_identity is not None:
                    self.preflight.native_identity.require_durable_paths()
                    await asyncio.to_thread(
                        OriginalModelRelay._write_atomic,
                        path=record.root / "backend-identity-observation.json",
                        content=json.dumps(
                            self.preflight.native_identity.snapshot(), sort_keys=True, separators=(",", ":")
                        ).encode(),
                    )
            except (OSError, ValueError) as error:
                record.failure_code = record.failure_code or failure_stage
                logger.error("Original canonical source intake was refused (%s).", type(error).__name__)
        if self.validation_state is not None and record.job.job_ref not in self._retained:
            try:
                await self.validation_state.finish_job_async(reservation_id=record.grant.reservation_id, proved=proved)
            except ValueError as error:
                record.failure_code = "worker_state_unverified"
                self._quarantined = True
                logger.error("Original remote validation closure remains unverified (%s).", type(error).__name__)
        if proved:
            await asyncio.to_thread(self._active_marker.unlink, missing_ok=True)
            if self._grant is record.grant:
                self._grant = None
        else:
            self._quarantined = True
        return record.cleanup

    def _require_grant(self, grant: OriginalRunGrant) -> _OwnedGrant:
        if not isinstance(grant, _OwnedGrant) or self._grant is not grant:
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        return grant

    def _record(self, *, grant: OriginalRunGrant, job: OriginalWorkerJob, allow_released: bool = False) -> _OwnedJob:
        record = self._jobs.get(job.job_ref)
        if (
            record is None
            or record.job != job
            or record.grant is not grant
            or (not allow_released and self._grant is not grant)
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        return record

    def _create_root(self, *, record: _OwnedJob) -> None:
        record.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        for name in self.RUN_SUBDIRECTORIES:
            (record.root / name).mkdir(mode=0o700)
        OriginalModelRelay._write_atomic(
            path=self._active_marker,
            content=json.dumps(
                {
                    "schema_version": 1,
                    "app_run_id": str(record.job.app_run_id),
                    "job_ref": str(record.job.job_ref),
                    "control_run_instance_id": str(record.control_id),
                },
                sort_keys=True,
            ).encode(),
        )

    def _load_key(self) -> bytes:
        path = self.config.jobs_root / "backend-retained-authority.key"
        if not path.exists():
            if any(self.config.jobs_root.glob("*/backend-retained-authority.json")):
                raise CohostPreflightError("Retained authority key was lost; it cannot be silently replaced.")
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        key = path.read_bytes()
        if len(key) != 32:
            raise CohostPreflightError("Backend retained source authority key is invalid.")
        return key

    def _retain(self, *, record: _OwnedJob) -> None:
        assert self._authority_key is not None and record.artifacts is not None
        expires = datetime.now(UTC) + timedelta(seconds=self.config.retention_seconds)
        content = json.dumps(
            {
                "envelope": record.artifacts.admission.envelope.model_dump(mode="json", exclude_none=True),
                "expires_at": expires.isoformat(),
                "control_id": str(record.control_id),
                "policy_sha256": config_hash(
                    {
                        "source": self.config.source.model_dump(mode="json"),
                        "profile_ref": self.profile_ref,
                        "database": self.config.expected_database_name,
                        "result_container": self.config.result_container_url,
                    }
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        value = {
            "envelope_b64": base64.b64encode(content).decode(),
            "signature": hmac.new(self._authority_key, content, hashlib.sha256).hexdigest(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        OriginalModelRelay._write_atomic(
            path=record.root / "backend-retained-authority.json", content=json.dumps(value, sort_keys=True).encode()
        )
        self._retained_expiry[record.job.job_ref] = expires

    def _restore_retained(self) -> dict[UUID, OriginalEvidenceAdmission]:
        from base64 import b64decode

        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalScorePolicy

        assert self._authority_key is not None
        result: dict[UUID, OriginalEvidenceAdmission] = {}
        packets: list[tuple[OriginalEvidenceEnvelope, datetime]] = []
        if self.validation_state is not None:
            for item in self.validation_state.retained:
                encoded_envelope, expiry = item.get("envelope"), item.get("expires_at")
                if not isinstance(encoded_envelope, dict) or not isinstance(expiry, str):
                    raise CohostPreflightError("Remote retained source authority is incomplete.")
                packets.append(
                    (
                        OriginalEvidenceEnvelope.model_validate_json(json.dumps(encoded_envelope)),
                        datetime.fromisoformat(expiry),
                    )
                )
        for path in (
            () if self.validation_state is not None else self.config.jobs_root.glob("*/backend-retained-authority.json")
        ):
            if path.is_symlink() or path.resolve() != path or str(UUID(path.parent.name)) != path.parent.name:
                raise CohostPreflightError("Retained authority cannot redirect outside its exact UUID run root.")
            value = TypeAdapter(dict[str, JsonValue]).validate_json(
                CohostPreflight._read_bounded(path=path, limit=65_536), strict=True
            )
            encoded, signature = value.get("envelope_b64"), value.get("signature")
            if not isinstance(encoded, str) or not isinstance(signature, str):
                raise CohostPreflightError("Retained source authority is incomplete.")
            content = b64decode(encoded, validate=True)
            if not hmac.compare_digest(signature, hmac.new(self._authority_key, content, hashlib.sha256).hexdigest()):
                raise CohostPreflightError("Retained source authority was modified.")
            packet = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
            encoded_envelope, expiry = packet.get("envelope"), packet.get("expires_at")
            expected_policy = config_hash(
                {
                    "source": self.config.source.model_dump(mode="json"),
                    "profile_ref": self.profile_ref,
                    "database": self.config.expected_database_name,
                    "result_container": self.config.result_container_url,
                }
            )
            if (
                not isinstance(encoded_envelope, dict)
                or not isinstance(expiry, str)
                or packet.get("control_id") != path.parent.name
                or packet.get("policy_sha256") != expected_policy
            ):
                raise CohostPreflightError("Retained authority policy/root/lifetime binding differs.")
            packets.append(
                (
                    OriginalEvidenceEnvelope.model_validate_json(json.dumps(encoded_envelope)),
                    datetime.fromisoformat(expiry),
                )
            )
        for envelope, expires in packets:
            if expires.utcoffset() != timedelta(0):
                raise CohostPreflightError("Retained source authority lifetime must be UTC.")
            if datetime.now(UTC) >= expires:
                continue
            if (
                envelope.source_sha256 != self.config.source.spec.package.source_sha256
                or envelope.profile_ref != self.profile_ref
            ):
                raise CohostPreflightError("Retained source policy differs; separately reauthorize before viewing.")
            case = self.config.source.cases[0]
            result[envelope.job_ref] = OriginalEvidenceAdmission(
                envelope=envelope,
                run=EvalRunRef(
                    spec=OriginalWorkerArtifacts.run_spec(config=self.config, envelope=envelope),
                    run_instance_id=envelope.run_instance_id,
                ),
                cases=self.config.source.cases,
                score_policy=InspectOriginalScorePolicy(
                    task_name=case.task_name,
                    task_version=case.task_version,
                    primary_scorer=self.config.source.primary_scorer,
                ),
                display_values=self.config.source.display_values,
            )
            self._retained_expiry[envelope.job_ref] = expires
        if len(result) > self.MAX_RETAINED_JOBS:
            raise CohostPreflightError("Retained preview source capacity requires an owned export/cleanup.")
        return result

    @staticmethod
    async def _read_stream_async(*, stream: asyncio.StreamReader, limit: int) -> bytes:
        output = bytearray()
        over_limit = False
        while chunk := await stream.read(4096):
            if len(output) + len(chunk) <= limit:
                output.extend(chunk)
            else:
                over_limit = True
        if over_limit:
            raise CohostPreflightError("Original child stream exceeded its safe protocol bound.")
        return bytes(output)

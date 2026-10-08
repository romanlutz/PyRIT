# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""API-owned durable gateway for an explicitly installed execution-only HTTP worker."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, TypeVar

from pyrit.common.async_compatibility import run_legacy_sync_async
from pyrit.executor.jobs.inspect import OriginalInspectArtifactWriter, PublicOriginalInspectJobRuntime
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import (
    EvaluationCanonicalSettlementError,
    EvaluationJobError,
    EvaluationJobErrorCode,
    EvaluationJobRuntimeRegistry,
    EvaluationRuntimeArtifacts,
    EvaluationRuntimeCancelled,
    EvaluationRuntimeError,
)
from pyrit.executor.jobs.remote_ledger import EvaluationRemoteJournal
from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpClient, EvaluationWorkerHttpSettings
from pyrit.models.evaluation_job import (
    EvaluationArtifactManifest,
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationJobSubmission,
    EvaluationRuntimeKind,
)
from pyrit.models.evaluation_worker import (
    EvaluationGatewaySettlement,
    EvaluationGatewaySettlementDisposition,
    EvaluationWorkerAdmission,
    EvaluationWorkerBinding,
    EvaluationWorkerEvidence,
    EvaluationWorkerSnapshot,
    EvaluationWorkerState,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path
    from uuid import UUID

    import httpx

    from pyrit.executor.jobs.port import EvaluationRuntimeContext
    from pyrit.executor.jobs.worker_auth import EvaluationWorkerCredentialProvider
    from pyrit.memory import MemoryInterface

T = TypeVar("T")
logger = logging.getLogger(__name__)


async def _mark_uncertain_async(*, journal: EvaluationRemoteJournal, job_id: UUID) -> None:
    try:
        await run_legacy_sync_async(journal.uncertain, job_id)
    except (sqlite3.Error, EvaluationJobError):
        logger.error("Remote reconciliation journal could not persist uncertainty job=%s", job_id)


async def _join_owned_async(operation: Awaitable[T]) -> T:
    task = asyncio.ensure_future(operation)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class RemoteEvaluationJobRuntime:
    """Transport only the sole locally reviewed original source; never install catalog code."""

    def __init__(
        self,
        *,
        original: PublicOriginalInspectJobRuntime,
        client: EvaluationWorkerHttpClient,
        journal: EvaluationRemoteJournal,
    ) -> None:
        """Bind the approved registration, protected client and gateway-owned journal."""
        self.registration = original.registration
        self.client = client
        self.journal = journal

    async def execute_async(self, *, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        """
        Retain remote execution ownership and exact artifacts without trusting worker database results.

        Returns:
            EvaluationRuntimeArtifacts: Verified original bytes with an explicitly derived gateway manifest.

        Raises:
            EvaluationRuntimeCancelled: If cancellation was joined with independently observed settlement.
            EvaluationRuntimeError: If transport, evidence or closure remains refused or uncertain.
        """
        admission = EvaluationWorkerAdmission(
            request=context.request,
            request_sha256=context.request.request_sha256,
            gateway_fence_id=context.fence_id,
            actor_id=context.actor_id,
        )
        binding: EvaluationWorkerBinding | None = None
        posted = False
        try:
            await run_legacy_sync_async(self.journal.admit, admission)
            posted = True
            submission = await self.client.submit_async(admission=admission)
            binding = submission.binding
            await run_legacy_sync_async(self.journal.bind, admission=admission, binding=binding)
            snapshot = await self._poll_async(admission=admission, binding=binding)
            if snapshot.state is not EvaluationWorkerState.COMPLETED:
                cleanup = await self._settle_closure_async(admission=admission, snapshot=snapshot)
                if snapshot.state is EvaluationWorkerState.CANCELLED:
                    raise EvaluationRuntimeCancelled(cleanup)
                raise EvaluationRuntimeError(code=EvaluationJobErrorCode.RUNTIME_FAILED, cleanup=cleanup)
            return await self._artifacts_async(admission=admission, snapshot=snapshot)
        except EvaluationRuntimeCancelled:
            raise
        except asyncio.CancelledError as error:
            if not posted:
                raise EvaluationRuntimeCancelled(EvaluationCleanupState.NOT_STARTED) from error
            cleanup = await _join_owned_async(self._cancel_join_async(admission=admission, binding=binding))
            raise EvaluationRuntimeCancelled(cleanup) from error
        except EvaluationRuntimeError:
            raise
        except Exception as error:
            await _mark_uncertain_async(journal=self.journal, job_id=context.request.job_id)
            code = error.code if isinstance(error, EvaluationJobError) else EvaluationJobErrorCode.RUNTIME_FAILED
            raise EvaluationRuntimeError(code=code, cleanup=EvaluationCleanupState.UNKNOWN) from error

    async def _poll_async(
        self, *, admission: EvaluationWorkerAdmission, binding: EvaluationWorkerBinding
    ) -> EvaluationWorkerSnapshot:
        cursor = 0
        try:
            async with asyncio.timeout(self.client.settings.poll_deadline_seconds):
                while True:
                    snapshot = await self.client.status_async(
                        admission=admission, binding=binding, after_sequence=cursor
                    )
                    cursor = await run_legacy_sync_async(self.journal.observe, snapshot)
                    if snapshot.state.terminal and cursor == snapshot.last_sequence:
                        return snapshot
                    if cursor == snapshot.last_sequence:
                        await asyncio.sleep(self.client.settings.poll_interval_seconds)
        except TimeoutError as error:
            raise EvaluationJobError(EvaluationJobErrorCode.DISPATCH_UNCERTAIN) from error

    async def _artifacts_async(
        self, *, admission: EvaluationWorkerAdmission, snapshot: EvaluationWorkerSnapshot
    ) -> EvaluationRuntimeArtifacts:
        terminal = snapshot.terminal
        if terminal is None or terminal.manifest_sha256 is None:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        worker, exact_manifest = await self.client.manifest_async(
            admission=admission, binding=snapshot.binding, manifest_sha256=terminal.manifest_sha256
        )
        payloads = [
            (
                artifact.name,
                await self.client.artifact_async(
                    admission=admission,
                    binding=snapshot.binding,
                    manifest_sha256=worker.manifest_sha256,
                    artifact=artifact,
                ),
            )
            for artifact in worker.artifacts
        ]
        gateway = EvaluationArtifactManifest(
            request=admission.request,
            request_sha256=admission.request_sha256,
            fence_id=admission.gateway_fence_id,
            artifacts=worker.artifacts,
        )
        artifacts = EvaluationRuntimeArtifacts(manifest=gateway, payloads=tuple(payloads), cleanup=terminal.cleanup)
        artifacts.verify()
        await run_legacy_sync_async(
            self.journal.retain,
            job_id=admission.request.job_id,
            original=exact_manifest,
            worker=worker,
            gateway=gateway,
        )
        return artifacts

    async def _settle_closure_async(
        self, *, admission: EvaluationWorkerAdmission, snapshot: EvaluationWorkerSnapshot
    ) -> EvaluationCleanupState:
        if (
            snapshot.terminal is None
            or snapshot.cleanup is not EvaluationCleanupState.VERIFIED
            or snapshot.evidence is not EvaluationWorkerEvidence.ABSENT
            or snapshot.state is EvaluationWorkerState.COMPLETED
        ):
            await _mark_uncertain_async(journal=self.journal, job_id=admission.request.job_id)
            return EvaluationCleanupState.UNKNOWN
        settlement = EvaluationGatewaySettlement(
            binding=snapshot.binding, disposition=EvaluationGatewaySettlementDisposition.CLOSURE_OBSERVED
        )
        await run_legacy_sync_async(self.journal.settlement_intent, settlement)
        receipt = await self.client.settle_async(admission=admission, settlement=settlement)
        await run_legacy_sync_async(self.journal.settle, receipt)
        return EvaluationCleanupState.VERIFIED

    async def _cancel_join_async(
        self, *, admission: EvaluationWorkerAdmission, binding: EvaluationWorkerBinding | None
    ) -> EvaluationCleanupState:
        try:
            async with asyncio.timeout(self.client.settings.settlement_timeout_seconds):
                snapshot = await self.client.cancel_async(admission=admission, binding=binding)
                await run_legacy_sync_async(self.journal.bind, admission=admission, binding=snapshot.binding)
                cursor = await run_legacy_sync_async(self.journal.observe, snapshot)
                while not snapshot.state.terminal or cursor != snapshot.last_sequence:
                    await asyncio.sleep(self.client.settings.poll_interval_seconds)
                    snapshot = await self.client.status_async(
                        admission=admission, binding=snapshot.binding, after_sequence=cursor
                    )
                    cursor = await run_legacy_sync_async(self.journal.observe, snapshot)
                return await self._settle_closure_async(admission=admission, snapshot=snapshot)
        except Exception as error:
            await _mark_uncertain_async(journal=self.journal, job_id=admission.request.job_id)
            if not isinstance(error, (EvaluationJobError, TimeoutError)):
                raise
            return EvaluationCleanupState.UNKNOWN


class RemoteOriginalInspectArtifactWriter:
    """Only the public API writer may create canonical IDs; worker settlement comes afterward."""

    def __init__(
        self,
        *,
        writer: OriginalInspectArtifactWriter,
        client: EvaluationWorkerHttpClient,
        journal: EvaluationRemoteJournal,
    ) -> None:
        """Bind actual canonical persistence, authenticated settlement and local provenance."""
        self.writer = writer
        self.client = client
        self.journal = journal

    async def import_async(
        self, *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        """
        Retain and join canonical import, receipt journaling and remote settlement.

        Returns:
            EvaluationCanonicalReceipt: References created and verified in this API's database.
        """
        return await _join_owned_async(self._import_async(request=request, artifacts=artifacts))

    async def _import_async(
        self, *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        try:
            admission, binding, worker, gateway = await run_legacy_sync_async(self.journal.handoff, request.job_id)
            if gateway != artifacts.manifest or admission.request != request:
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            canonical = await self.writer.import_async(request=request, artifacts=artifacts)
        except Exception as error:
            await _mark_uncertain_async(journal=self.journal, job_id=request.job_id)
            raise EvaluationRuntimeError(
                code=EvaluationJobErrorCode.WRITER_FAILED, cleanup=EvaluationCleanupState.UNKNOWN
            ) from error
        settlement = EvaluationGatewaySettlement(
            binding=binding,
            worker_manifest_sha256=worker.manifest_sha256,
            gateway_manifest_sha256=gateway.manifest_sha256,
            artifact_sha256=canonical.artifact_sha256,
            disposition=EvaluationGatewaySettlementDisposition.CANONICAL_IMPORTED,
        )
        try:
            await run_legacy_sync_async(self.journal.canonical, canonical)
            await run_legacy_sync_async(self.journal.settlement_intent, settlement)
            receipt = await self.client.settle_async(admission=admission, settlement=settlement)
            await run_legacy_sync_async(self.journal.settle, receipt)
        except Exception as error:
            await _mark_uncertain_async(journal=self.journal, job_id=request.job_id)
            raise EvaluationCanonicalSettlementError(canonical) from error
        return canonical


class RemoteEvaluationJobGateway(LocalEvaluationJobPort):
    """The same local canonical owner with actor-specific authenticated remote grants."""

    def __init__(
        self,
        *,
        root: Path,
        original: PublicOriginalInspectJobRuntime,
        memory: MemoryInterface,
        allowed_actor_ids: frozenset[str],
        client: EvaluationWorkerHttpClient,
    ) -> None:
        """Install only one approved inert source and its actual API-owned writer."""
        self.client = client
        self.journal = EvaluationRemoteJournal(root)
        runtime = RemoteEvaluationJobRuntime(original=original, client=client, journal=self.journal)
        writer = RemoteOriginalInspectArtifactWriter(
            writer=OriginalInspectArtifactWriter(memory=memory, runtime=original),
            client=client,
            journal=self.journal,
        )
        super().__init__(
            root=root,
            registry=EvaluationJobRuntimeRegistry((runtime,)),
            writers={EvaluationRuntimeKind.ORIGINAL_INSPECT: writer},
            allowed_actor_ids=allowed_actor_ids,
        )
        self._remote_close: asyncio.Task[None] | None = None

    async def startup_async(self) -> None:
        """
        Acquire exclusive local ownership before any protocol/grant request.

        Raises:
            EvaluationJobError: If the remote service lacks the approved protocol or source grant.
        """
        await super().startup_async()
        await run_legacy_sync_async(self.journal.initialize)
        actor = sorted(self._allowed_actors)[0]
        await self.client.protocol_async(actor_id=actor)
        if not await self.catalog_async(actor_id=actor):
            raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_RUNTIME)

    async def catalog_async(self, *, actor_id: str) -> tuple[EvaluationJobRegistration, ...]:
        """
        Intersect delegated service grants with pinned locally installed bindings.

        Returns:
            tuple[EvaluationJobRegistration, ...]: Approved references only, never service-provided code.
        """
        self.authorize_actor(actor_id)
        catalog = await self.client.catalog_async(actor_id=actor_id)
        return tuple(item for item in self.registry.registrations if item in catalog.registrations)

    async def submit_async(self, *, request: EvaluationJobRequest, actor_id: str) -> EvaluationJobSubmission:
        """
        Own actor-specific grant validation and durable local admission despite caller disconnect.

        Returns:
            EvaluationJobSubmission: Local queue acknowledgment, not remote execution or a grade.
        """
        return await self._own_async(self._submit_remote_async(request=request, actor_id=actor_id))

    async def _submit_remote_async(self, *, request: EvaluationJobRequest, actor_id: str) -> EvaluationJobSubmission:
        request = EvaluationJobRequest.model_validate(request)
        if not any(item.accepts(request) for item in await self.catalog_async(actor_id=actor_id)):
            raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_RUNTIME)
        return await super().submit_async(request=request, actor_id=actor_id)

    async def _publication_refused_async(
        self, *, context: EvaluationRuntimeContext, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCleanupState:
        await _mark_uncertain_async(journal=self.journal, job_id=context.request.job_id)
        return EvaluationCleanupState.UNKNOWN

    async def shutdown_async(self) -> None:
        """Join owned runtimes/writers before closing HTTP or host credential resources."""
        if self._remote_close is None:
            self._remote_close = asyncio.create_task(self._close_remote_async())
        await _join_owned_async(asyncio.shield(self._remote_close))

    async def _close_remote_async(self) -> None:
        try:
            await super().shutdown_async()
        finally:
            await self.client.close_async()


async def create_remote_original_job_port_async(
    *,
    root: Path,
    memory: MemoryInterface,
    allowed_actor_ids: frozenset[str],
    settings: EvaluationWorkerHttpSettings,
    credentials: EvaluationWorkerCredentialProvider,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RemoteEvaluationJobGateway:
    """
    Install only the harmless approved source without starting work or discovering a platform.

    Returns:
        RemoteEvaluationJobGateway: Startup-owned canonical gateway with explicit execution transport.

    Raises:
        ValueError: If canonical memory is not the explicitly supported SQLite backend.
    """
    from pyrit.memory import SQLiteMemory

    if not isinstance(memory, SQLiteMemory):
        raise ValueError("The remote job gateway requires API-owned canonical SQLite.")
    original = await PublicOriginalInspectJobRuntime.create_async()
    client = EvaluationWorkerHttpClient(settings=settings, credentials=credentials, transport=transport)
    return RemoteEvaluationJobGateway(
        root=root, original=original, memory=memory, allowed_actor_ids=allowed_actor_ids, client=client
    )

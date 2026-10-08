# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Opt-in local queue producer/consumer with durable single-dispatch fencing."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from pyrit.common.async_compatibility import run_legacy_sync_async
from pyrit.executor.jobs.ledger import LocalEvaluationJobLedger
from pyrit.executor.jobs.port import (
    EvaluationArtifactWriter,
    EvaluationCanonicalSettlementError,
    EvaluationJobError,
    EvaluationJobErrorCode,
    EvaluationJobRuntimeRegistry,
    EvaluationRuntimeArtifacts,
    EvaluationRuntimeCancelled,
    EvaluationRuntimeError,
)
from pyrit.models.evaluation_job import (
    EvaluationArtifactManifest,
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationControlReceipt,
    EvaluationControlRequest,
    EvaluationDeliveryReceipt,
    EvaluationDeliveryState,
    EvaluationEvidenceState,
    EvaluationJobDelivery,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationJobSnapshot,
    EvaluationJobState,
    EvaluationJobSubmission,
    EvaluationRuntimeKind,
    EvaluationWaitBoundary,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping
    from pathlib import Path
    from uuid import UUID

    from pyrit.executor.jobs.port import EvaluationRuntimeContext

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class _LocalRuntimeContext:
    request: EvaluationJobRequest
    fence_id: UUID
    actor_id: str
    run_root: Path
    ledger: LocalEvaluationJobLedger

    async def wait_for_control_async(self, *, boundary: EvaluationWaitBoundary) -> EvaluationControlRequest:
        """
        Return one structured command at an already reviewed wait boundary.

        Returns:
            EvaluationControlRequest: Delivered data, never a shell invocation.
        """
        boundary = EvaluationWaitBoundary.model_validate(boundary)
        await run_legacy_sync_async(
            self.ledger.open_boundary, job_id=self.request.job_id, fence_id=self.fence_id, boundary=boundary
        )
        while True:
            command = await run_legacy_sync_async(
                self.ledger.take_control, job_id=self.request.job_id, fence_id=self.fence_id
            )
            if command is not None:
                return command
            await asyncio.sleep(0.02)


class LocalEvaluationJobPort:
    """One local owner, separate runtime and writer, no private provisioning or live transport."""

    def __init__(
        self,
        *,
        root: Path,
        registry: EvaluationJobRuntimeRegistry,
        writers: Mapping[EvaluationRuntimeKind, EvaluationArtifactWriter],
        allowed_actor_ids: frozenset[str],
    ) -> None:
        """
        Configure explicit local execution; construction starts no runtime.

        Raises:
            ValueError: If the explicit actor allowlist is missing or invalid.
        """
        if not allowed_actor_ids or any(not 1 <= len(actor) <= 128 for actor in allowed_actor_ids):
            raise ValueError("The local job port requires a bounded explicit actor allowlist.")
        self.root = root
        self.registry = registry
        self._writers = dict(writers)
        self._allowed_actors = allowed_actor_ids
        self._ledger: LocalEvaluationJobLedger | None = None
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._started_tasks: set[UUID] = set()
        self._operations: set[asyncio.Task[object]] = set()
        self._dispatch_lock = asyncio.Lock()
        self._consumer: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Task[None] | None = None
        self._closing = False
        self._fatal: EvaluationJobErrorCode | None = None

    async def startup_async(self) -> None:
        """
        Acquire local process ownership before restoring or accepting work.

        Raises:
            EvaluationJobError: If the port already started or closed.
        """
        if self._ledger is not None or self._closing:
            raise EvaluationJobError(EvaluationJobErrorCode.CLOSED)
        ledger = await run_legacy_sync_async(LocalEvaluationJobLedger, self.root)
        try:
            await run_legacy_sync_async(ledger.startup)
        except BaseException:
            await run_legacy_sync_async(ledger.close)
            raise
        self._ledger = ledger

    def start_consumer(self) -> None:
        """
        Start a retained consumer; submit ACK alone never invokes a runtime.

        Raises:
            EvaluationJobError: If another consumer already owns this port.
        """
        self._require_open()
        if self._consumer is not None:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        self._consumer = asyncio.create_task(self._consume_async())

    def has_active_work(self) -> bool:
        """
        Keep canonical memory alive until every admitted runtime or writer joins.

        Returns:
            bool: Whether owner-held work is still in flight.
        """
        return bool(self._tasks or self._operations)

    def authorize_actor(self, actor_id: str) -> None:
        """
        Authorize catalog access under the same explicit port policy.

        Raises:
            EvaluationJobError: If the authenticated actor is not admitted.
        """
        self._require_open()
        self._authorize(actor_id)

    async def catalog_async(self, *, actor_id: str) -> tuple[EvaluationJobRegistration, ...]:
        """
        Return only explicitly installed registrations for an admitted actor.

        Returns:
            tuple[EvaluationJobRegistration, ...]: The local owner's reviewed catalog.
        """
        self.authorize_actor(actor_id)
        return self.registry.registrations

    async def submit_async(self, *, request: EvaluationJobRequest, actor_id: str) -> EvaluationJobSubmission:
        """
        Accept one immutable actor-bound source/case/run with no implicit retry.

        Returns:
            EvaluationJobSubmission: A durable queue ACK, not a source result.

        Raises:
            EvaluationJobError: If actor, runtime, identity, or dispatch certainty is unsupported.
        """
        ledger = self._require_open()
        self._authorize(actor_id)
        request = EvaluationJobRequest.model_validate(request)
        self.registry.resolve(request)
        if request.runtime not in self._writers:
            raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_RUNTIME)
        return await run_legacy_sync_async(ledger.submit, request=request, actor_id=actor_id)

    async def status_async(self, *, job_id: UUID, actor_id: str, after_sequence: int = 0) -> EvaluationJobSnapshot:
        """
        Read only the authenticated actor's exact request and ordered evidence state.

        Returns:
            EvaluationJobSnapshot: A bounded event page and any real canonical references.
        """
        ledger = self._require_open(allow_uncertain=True)
        self._authorize(actor_id)
        return await run_legacy_sync_async(
            ledger.snapshot, job_id=job_id, actor_id=actor_id, after_sequence=after_sequence
        )

    async def receive_async(self, *, delivery: EvaluationJobDelivery) -> EvaluationDeliveryReceipt:
        """
        Fence broker redelivery before dispatch; busy deliveries must not be settled.

        Returns:
            EvaluationDeliveryReceipt: Started, duplicate, or busy, never an execution grade.
        """
        self._require_open()
        delivery = EvaluationJobDelivery.model_validate(delivery)
        return await self._own_async(self._receive_async(delivery=delivery))

    async def cancel_async(self, *, job_id: UUID, actor_id: str) -> EvaluationJobSnapshot:
        """
        Retain cancellation work independently of HTTP/CLI caller cancellation.

        Returns:
            EvaluationJobSnapshot: Requested or terminal cancellation, with explicit uncertainty.
        """
        self._require_open()
        self._authorize(actor_id)
        return await self._own_async(self._cancel_async(job_id=job_id, actor_id=actor_id))

    async def control_async(
        self,
        *,
        job_id: UUID,
        actor_id: str,
        capability: str,
        command: EvaluationControlRequest,
    ) -> EvaluationControlReceipt:
        """
        Accept bounded data only at the runtime's current reviewed wait boundary.

        Returns:
            EvaluationControlReceipt: Acceptance distinct from runtime delivery or action.

        Raises:
            EvaluationJobError: If actor, capability, boundary, or immutable command differs.
        """
        ledger = self._require_open()
        self._authorize(actor_id)
        if not 32 <= len(capability) <= 128:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)
        command = EvaluationControlRequest.model_validate(command)
        return await run_legacy_sync_async(
            ledger.control, job_id=job_id, actor_id=actor_id, capability=capability, command=command
        )

    async def wait_async(self, *, job_id: UUID, actor_id: str, timeout_seconds: float = 60) -> EvaluationJobSnapshot:
        """
        Observe terminal state without making waiter timeout a runtime cancellation.

        Returns:
            EvaluationJobSnapshot: The actual terminal snapshot.
        """
        async with asyncio.timeout(timeout_seconds):
            while True:
                snapshot = await self.status_async(job_id=job_id, actor_id=actor_id)
                if snapshot.state.terminal:
                    return snapshot
                await asyncio.sleep(0.02)

    async def shutdown_async(self) -> None:
        """Join runtime and irreversible writer work before releasing local ownership."""
        if self._shutdown is None:
            self._closing = True
            self._shutdown = asyncio.create_task(self._shutdown_async())
        try:
            await asyncio.shield(self._shutdown)
        except asyncio.CancelledError:
            while not self._shutdown.done():
                try:
                    await asyncio.shield(self._shutdown)
                except asyncio.CancelledError:
                    continue
            self._shutdown.result()
            raise

    async def _own_async(self, operation: Awaitable[T]) -> T:
        async def execute_async() -> T:
            return await operation

        task = asyncio.create_task(execute_async())
        self._operations.add(task)
        task.add_done_callback(self._operation_finished)
        return await asyncio.shield(task)

    def _operation_finished(self, task: asyncio.Task[object]) -> None:
        self._operations.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Evaluation job port operation refused or failed: %s", type(task.exception()).__name__)

    async def _receive_async(self, *, delivery: EvaluationJobDelivery) -> EvaluationDeliveryReceipt:
        async with self._dispatch_lock:
            ledger = self._require_open()
            state, fence_id = await run_legacy_sync_async(
                ledger.claim, delivery=delivery, allowed_actor_ids=self._allowed_actors
            )
            if state is EvaluationDeliveryState.STARTED:
                assert fence_id is not None
                snapshot = await run_legacy_sync_async(ledger.snapshot, job_id=delivery.job_id, actor_id=None)
                context = _LocalRuntimeContext(
                    request=snapshot.request,
                    fence_id=fence_id,
                    actor_id=await run_legacy_sync_async(
                        ledger.admitted_actor, job_id=delivery.job_id, fence_id=fence_id
                    ),
                    run_root=self.root / "attempts" / str(delivery.job_id),
                    ledger=ledger,
                )
                task = asyncio.create_task(self._execute_async(context=context))
                self._tasks[delivery.job_id] = task
                task.add_done_callback(
                    lambda completed: self._execution_finished(job_id=delivery.job_id, task=completed)
                )
            return EvaluationDeliveryReceipt(job_id=delivery.job_id, state=state)

    async def _cancel_async(self, *, job_id: UUID, actor_id: str) -> EvaluationJobSnapshot:
        async with self._dispatch_lock:
            ledger = self._require_open()
            active = await run_legacy_sync_async(ledger.cancel, job_id=job_id, actor_id=actor_id)
            if active:
                task = self._tasks.get(job_id)
                if task is None:
                    raise EvaluationJobError(EvaluationJobErrorCode.DISPATCH_UNCERTAIN)
                if job_id in self._started_tasks:
                    task.cancel()
        return await run_legacy_sync_async(ledger.snapshot, job_id=job_id, actor_id=actor_id)

    def _execution_finished(self, *, job_id: UUID, task: asyncio.Task[None]) -> None:
        self._tasks.pop(job_id, None)
        self._started_tasks.discard(job_id)
        if task.cancelled() or task.exception() is not None:
            self._fatal = EvaluationJobErrorCode.DISPATCH_UNCERTAIN
            logger.error("Evaluation job terminal persistence is uncertain job=%s", job_id)

    async def _execute_async(self, *, context: _LocalRuntimeContext) -> None:
        self._started_tasks.add(context.request.job_id)
        cleanup = EvaluationCleanupState.NOT_STARTED
        evidence = EvaluationEvidenceState.ABSENT
        manifest: EvaluationArtifactManifest | None = None
        try:
            current = await run_legacy_sync_async(context.ledger.snapshot, job_id=context.request.job_id, actor_id=None)
            if current.state is EvaluationJobState.CANCEL_REQUESTED:
                raise EvaluationRuntimeCancelled(EvaluationCleanupState.NOT_STARTED)
            await run_legacy_sync_async(context.run_root.mkdir, parents=True, exist_ok=False)
            runtime = self.registry.resolve(context.request)
            cleanup = EvaluationCleanupState.UNKNOWN
            evidence = EvaluationEvidenceState.UNKNOWN
            artifacts = await runtime.execute_async(context=context)
            cleanup = artifacts.cleanup
            self._verify_artifacts(context=context, artifacts=artifacts)
            manifest = artifacts.manifest
            await run_legacy_sync_async(self._retain_artifacts, context=context, artifacts=artifacts)
            evidence = EvaluationEvidenceState.SOURCE_RETAINED
        except EvaluationRuntimeCancelled as error:
            await self._finish_async(
                context=context,
                state=EvaluationJobState.CANCELLED,
                cleanup=error.cleanup,
                evidence=evidence,
                reason="runtime_cancelled",
                manifest=manifest,
            )
            return
        except asyncio.CancelledError:
            await self._finish_async(
                context=context,
                state=EvaluationJobState.CANCELLED,
                cleanup=cleanup,
                evidence=evidence,
                reason="runtime_cancelled",
                manifest=manifest,
            )
            return
        except Exception as error:
            code = error.code if isinstance(error, EvaluationJobError) else EvaluationJobErrorCode.RUNTIME_FAILED
            if isinstance(error, EvaluationRuntimeError):
                cleanup = error.cleanup
            current = await run_legacy_sync_async(context.ledger.snapshot, job_id=context.request.job_id, actor_id=None)
            state = (
                EvaluationJobState.CANCELLED
                if current.state is EvaluationJobState.CANCEL_REQUESTED
                else EvaluationJobState.FAILED
            )
            logger.error("Evaluation runtime refused or failed job=%s code=%s", context.request.job_id, code.value)
            await self._finish_async(
                context=context, state=state, cleanup=cleanup, evidence=evidence, reason=code.value, manifest=manifest
            )
            return
        # The barrier and writer share an owner; cancellation cannot strand committed finalization.
        await self._settle_async(self._finalize_async(context=context, artifacts=artifacts))

    async def _finalize_async(self, *, context: _LocalRuntimeContext, artifacts: EvaluationRuntimeArtifacts) -> None:
        try:
            await run_legacy_sync_async(
                context.ledger.begin_finalizing,
                job_id=context.request.job_id,
                fence_id=context.fence_id,
                manifest=artifacts.manifest,
            )
        except EvaluationJobError as error:
            current = await run_legacy_sync_async(context.ledger.snapshot, job_id=context.request.job_id, actor_id=None)
            state = (
                EvaluationJobState.CANCELLED
                if current.state is EvaluationJobState.CANCEL_REQUESTED
                else EvaluationJobState.FAILED
            )
            logger.error(
                "Evaluation publication boundary refused job=%s code=%s", context.request.job_id, error.code.value
            )
            await self._finish_async(
                context=context,
                state=state,
                cleanup=await self._publication_refused_async(context=context, artifacts=artifacts),
                evidence=EvaluationEvidenceState.SOURCE_RETAINED,
                reason="cancelled_before_finalization" if state is EvaluationJobState.CANCELLED else error.code.value,
                manifest=artifacts.manifest,
            )
            return
        await self._publish_async(context=context, artifacts=artifacts)

    async def _publication_refused_async(
        self, *, context: EvaluationRuntimeContext, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCleanupState:
        """
        Preserve local closure; a remote facade must also account for unsettled handoff.

        Returns:
            EvaluationCleanupState: Verified local closure or remote settlement uncertainty.
        """
        return artifacts.cleanup

    async def _publish_async(self, *, context: _LocalRuntimeContext, artifacts: EvaluationRuntimeArtifacts) -> None:
        try:
            canonical = await self._writers[context.request.runtime].import_async(
                request=context.request, artifacts=artifacts
            )
            canonical = EvaluationCanonicalReceipt.model_validate(canonical)
            if canonical.artifact_sha256 not in {item.sha256 for item in artifacts.manifest.artifacts}:
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            await run_legacy_sync_async(
                context.ledger.finish,
                job_id=context.request.job_id,
                fence_id=context.fence_id,
                state=EvaluationJobState.SUCCEEDED if canonical.source_complete else EvaluationJobState.FAILED,
                cleanup=EvaluationCleanupState.VERIFIED,
                evidence=EvaluationEvidenceState.CANONICAL,
                reason=None if canonical.source_complete else "source_incomplete",
                canonical=canonical,
            )
        except EvaluationCanonicalSettlementError as error:
            await run_legacy_sync_async(
                context.ledger.finish,
                job_id=context.request.job_id,
                fence_id=context.fence_id,
                state=EvaluationJobState.FAILED,
                cleanup=EvaluationCleanupState.UNKNOWN,
                evidence=EvaluationEvidenceState.CANONICAL,
                reason="remote_settlement_uncertain",
                canonical=EvaluationCanonicalReceipt.model_validate(error.canonical),
            )
        except Exception as error:
            code = error.code if isinstance(error, EvaluationJobError) else EvaluationJobErrorCode.WRITER_FAILED
            logger.error("Canonical job writer refused or failed job=%s code=%s", context.request.job_id, code.value)
            await self._finish_async(
                context=context,
                state=EvaluationJobState.FAILED,
                cleanup=error.cleanup if isinstance(error, EvaluationRuntimeError) else EvaluationCleanupState.VERIFIED,
                evidence=EvaluationEvidenceState.SOURCE_RETAINED,
                reason=code.value,
                manifest=artifacts.manifest,
            )

    def _verify_artifacts(self, *, context: _LocalRuntimeContext, artifacts: EvaluationRuntimeArtifacts) -> None:
        artifacts.verify()
        registration = next(item for item in self.registry.registrations if item.accepts(context.request))
        if (
            artifacts.manifest.request != context.request
            or artifacts.manifest.fence_id != context.fence_id
            or artifacts.cleanup is not EvaluationCleanupState.VERIFIED
            or not any(item.kind is registration.artifact_kind for item in artifacts.manifest.artifacts)
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)

    @staticmethod
    def _retain_artifacts(*, context: _LocalRuntimeContext, artifacts: EvaluationRuntimeArtifacts) -> None:
        root = context.run_root / "artifacts"
        if context.run_root.is_symlink() or context.run_root.resolve() != context.run_root:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        root.mkdir()
        payloads = dict(artifacts.payloads)
        for item in artifacts.manifest.artifacts:
            with (root / item.name).open("xb") as stream:
                stream.write(payloads[item.name])
                stream.flush()
                os.fsync(stream.fileno())
        with (root / "manifest.json").open("xb") as stream:
            stream.write(artifacts.manifest.model_dump_json().encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    async def _finish_async(
        *,
        context: _LocalRuntimeContext,
        state: EvaluationJobState,
        cleanup: EvaluationCleanupState,
        evidence: EvaluationEvidenceState,
        reason: str,
        manifest: EvaluationArtifactManifest | None,
    ) -> None:
        await LocalEvaluationJobPort._settle_async(
            run_legacy_sync_async(
                context.ledger.finish,
                job_id=context.request.job_id,
                fence_id=context.fence_id,
                state=state,
                cleanup=cleanup,
                evidence=evidence,
                reason=reason,
                manifest=manifest,
            )
        )

    @staticmethod
    async def _settle_async(operation: Awaitable[T]) -> T:
        task = asyncio.ensure_future(operation)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()

    async def _consume_async(self) -> None:
        try:
            while not self._closing:
                for delivery in await run_legacy_sync_async(self._require_open().queued):
                    result = await self.receive_async(delivery=delivery)
                    if result.state is EvaluationDeliveryState.BUSY:
                        break
                await asyncio.sleep(0.02)
        except Exception as error:
            if self._closing and isinstance(error, EvaluationJobError) and error.code is EvaluationJobErrorCode.CLOSED:
                return
            self._fatal = (
                error.code if isinstance(error, EvaluationJobError) else EvaluationJobErrorCode.DISPATCH_UNCERTAIN
            )
            logger.error("Local evaluation queue stopped code=%s", self._fatal.value)

    async def _shutdown_async(self) -> None:
        if self._consumer is not None:
            await asyncio.shield(self._consumer)
        if self._operations:
            await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._operations)), return_exceptions=True)
        ledger = self._ledger
        if ledger is not None:
            for job_id, task in tuple(self._tasks.items()):
                snapshot = await run_legacy_sync_async(ledger.snapshot, job_id=job_id, actor_id=None)
                if (
                    not snapshot.state.terminal
                    and snapshot.state is not EvaluationJobState.FINALIZING
                    and job_id in self._started_tasks
                ):
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*(asyncio.shield(task) for task in tuple(self._tasks.values())))
            if self._operations:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in tuple(self._operations)), return_exceptions=True
                )
            await run_legacy_sync_async(ledger.close)
            self._ledger = None

    def _require_open(self, *, allow_uncertain: bool = False) -> LocalEvaluationJobLedger:
        if self._closing or self._ledger is None:
            raise EvaluationJobError(EvaluationJobErrorCode.CLOSED)
        if self._fatal is not None and not allow_uncertain:
            raise EvaluationJobError(self._fatal)
        return self._ledger

    def _authorize(self, actor_id: str) -> None:
        if actor_id not in self._allowed_actors:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)

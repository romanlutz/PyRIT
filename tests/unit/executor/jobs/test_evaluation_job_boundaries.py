# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harmless native type-port, review-bound controls, and durable refusal boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pyrit.executor.jobs.ledger import LocalEvaluationJobLedger
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import (
    EvaluationJobError,
    EvaluationJobRuntimeRegistry,
    EvaluationRuntimeArtifacts,
    EvaluationRuntimeCancelled,
)
from pyrit.models import EvalPackageRef, EvalSourceKind
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactKind,
    EvaluationArtifactManifest,
    EvaluationArtifactMediaType,
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationControlKind,
    EvaluationControlRequest,
    EvaluationDeliveryState,
    EvaluationEvidenceState,
    EvaluationJobDelivery,
    EvaluationJobEventKind,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationJobState,
    EvaluationRuntimeKind,
    EvaluationWaitBoundary,
)

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from pyrit.executor.jobs.port import EvaluationRuntimeContext


class _NativeFixture:
    def __init__(self, *, controls: bool = False, held: bool = False, fail: bool = False) -> None:
        self.registration = EvaluationJobRegistration(
            runtime=EvaluationRuntimeKind.INSPECT_VARIANT if controls else EvaluationRuntimeKind.NATIVE_BINDING,
            source=EvalPackageRef(kind=EvalSourceKind.NAMED, name="public_native_fixture", source_sha256="a" * 64),
            case_id="b" * 64,
            execution_profile_sha256="c" * 64,
            controls=(EvaluationControlKind.NUDGE, EvaluationControlKind.STOP) if controls else (),
            artifact_kind=EvaluationArtifactKind.NATIVE_EVIDENCE,
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.boundary = EvaluationWaitBoundary(
            boundary_id=uuid4(),
            name="reviewed_fixture_wait",
            controls=(EvaluationControlKind.NUDGE, EvaluationControlKind.STOP),
        )
        self.executions = 0
        self.held = held
        self.fail = fail
        self.commands: list[EvaluationControlRequest] = []

    def request(self) -> EvaluationJobRequest:
        return EvaluationJobRequest(
            job_id=uuid4(),
            run_id=uuid4(),
            attempt_id=uuid4(),
            runtime=self.registration.runtime,
            source=self.registration.source,
            case_id=self.registration.case_id,
            execution_profile_sha256=self.registration.execution_profile_sha256,
            controls=self.registration.controls,
        )

    async def execute_async(self, *, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        self.executions += 1
        self.started.set()
        if self.fail:
            raise RuntimeError("fixture private failure contents must not enter terminal events")
        try:
            if self.registration.controls:
                command = await context.wait_for_control_async(boundary=self.boundary)
                self.commands.append(command)
                raise EvaluationRuntimeCancelled(EvaluationCleanupState.VERIFIED)
            if self.held:
                await self.release.wait()
        except asyncio.CancelledError as error:
            raise EvaluationRuntimeCancelled(EvaluationCleanupState.VERIFIED) from error
        content = b'{"event":"harmless_native_complete"}\n'
        artifact = EvaluationArtifact(
            name="native.jsonl",
            kind=EvaluationArtifactKind.NATIVE_EVIDENCE,
            media_type=EvaluationArtifactMediaType.NATIVE_EVIDENCE,
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
        )
        return EvaluationRuntimeArtifacts(
            manifest=EvaluationArtifactManifest(
                request=context.request,
                request_sha256=context.request.request_sha256,
                fence_id=context.fence_id,
                artifacts=(artifact,),
            ),
            payloads=((artifact.name, content),),
            cleanup=EvaluationCleanupState.VERIFIED,
        )


class _NativeWriter:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.imports = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = False

    async def import_async(
        self, *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        self.imports += 1
        self.started.set()
        if self.hold:
            await self.release.wait()
        if self.fail:
            raise RuntimeError("fixture writer failure must stay private")
        artifacts.verify()
        return EvaluationCanonicalReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            manifest_sha256=artifacts.manifest.manifest_sha256,
            artifact_sha256=artifacts.manifest.artifacts[0].sha256,
            projection_id="public_native_fixture",
            source_complete=True,
        )


async def _port_async(
    *, root: Path, runtime: _NativeFixture, writer: _NativeWriter | None = None
) -> LocalEvaluationJobPort:
    port = LocalEvaluationJobPort(
        root=root,
        registry=EvaluationJobRuntimeRegistry((runtime,)),
        writers={runtime.registration.runtime: writer or _NativeWriter()},
        allowed_actor_ids=frozenset({"operator", "foreign"}),
    )
    await port.startup_async()
    return port


async def test_native_handler_emits_native_artifact_without_an_inspect_log_or_grade_async(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    writer = _NativeWriter()
    port = await _port_async(root=tmp_path, runtime=runtime, writer=writer)
    request = runtime.request()
    try:
        await port.submit_async(request=request, actor_id="operator")
        port.start_consumer()
        result = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert result.state is EvaluationJobState.SUCCEEDED
        assert result.manifest and result.manifest.artifacts[0].kind is EvaluationArtifactKind.NATIVE_EVIDENCE
        assert result.canonical and result.canonical.score_ids == result.canonical.attack_result_ids == ()
        assert writer.imports == runtime.executions == 1
        assert not list(tmp_path.rglob("*.eval"))
    finally:
        await port.shutdown_async()


async def test_redelivery_busy_mismatched_identity_and_foreign_actor_source_async(tmp_path: Path) -> None:
    runtime = _NativeFixture(held=True)
    port = await _port_async(root=tmp_path, runtime=runtime)
    request = runtime.request()
    try:
        await port.submit_async(request=request, actor_id="operator")
        delivery = EvaluationJobDelivery(job_id=request.job_id, request_sha256=request.request_sha256)
        assert (await port.receive_async(delivery=delivery)).state is EvaluationDeliveryState.STARTED
        await asyncio.wait_for(runtime.started.wait(), timeout=5)
        assert (await port.receive_async(delivery=delivery)).state is EvaluationDeliveryState.DUPLICATE
        for operation in (
            port.submit_async(request=request.model_copy(update={"attempt_id": uuid4()}), actor_id="operator"),
            port.submit_async(request=request.model_copy(update={"job_id": uuid4()}), actor_id="operator"),
            port.submit_async(request=request, actor_id="foreign"),
            port.status_async(job_id=request.job_id, actor_id="foreign"),
            port.cancel_async(job_id=request.job_id, actor_id="foreign"),
            port.receive_async(delivery=delivery.model_copy(update={"request_sha256": "d" * 64})),
            port.submit_async(
                request=request.model_copy(update={"execution_profile_sha256": "d" * 64}), actor_id="operator"
            ),
            port.submit_async(
                request=request.model_copy(
                    update={"source": request.source.model_copy(update={"source_sha256": "e" * 64})}
                ),
                actor_id="operator",
            ),
        ):
            with pytest.raises(EvaluationJobError):
                await operation
        fresh = runtime.request()
        await port.submit_async(request=fresh, actor_id="operator")
        assert (
            await port.receive_async(
                delivery=EvaluationJobDelivery(job_id=fresh.job_id, request_sha256=fresh.request_sha256)
            )
        ).state is EvaluationDeliveryState.BUSY
        await port.cancel_async(job_id=request.job_id, actor_id="operator")
        cancelled = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert cancelled.state is EvaluationJobState.CANCELLED and cancelled.canonical is None
        assert cancelled.cleanup is EvaluationCleanupState.VERIFIED
        assert runtime.executions == 1
        runtime.held = False
        port.start_consumer()
        assert (await port.wait_async(job_id=fresh.job_id, actor_id="operator")).state is EvaluationJobState.SUCCEEDED
    finally:
        await port.shutdown_async()


async def test_reviewed_control_boundary_is_actor_capability_action_and_command_bound_async(tmp_path: Path) -> None:
    runtime = _NativeFixture(controls=True)
    port = await _port_async(root=tmp_path, runtime=runtime)
    request = runtime.request()
    try:
        submission = await port.submit_async(request=request, actor_id="operator")
        assert submission.control_capability
        port.start_consumer()
        async with asyncio.timeout(5):
            while (await port.status_async(job_id=request.job_id, actor_id="operator")).state is not (
                EvaluationJobState.WAITING
            ):
                await asyncio.sleep(0.01)
        command = EvaluationControlRequest(
            command_id=uuid4(),
            boundary_id=runtime.boundary.boundary_id,
            kind=EvaluationControlKind.NUDGE,
            message="harmless reminder",
        )
        for actor, capability, value in (
            ("foreign", submission.control_capability, command),
            ("operator", "x" * 43, command),
            ("operator", submission.control_capability, command.model_copy(update={"boundary_id": uuid4()})),
            (
                "operator",
                submission.control_capability,
                command.model_copy(update={"kind": EvaluationControlKind.SEND_MESSAGE}),
            ),
        ):
            with pytest.raises(EvaluationJobError):
                await port.control_async(job_id=request.job_id, actor_id=actor, capability=capability, command=value)
        accepted = await port.control_async(
            job_id=request.job_id, actor_id="operator", capability=submission.control_capability, command=command
        )
        duplicate = await port.control_async(
            job_id=request.job_id, actor_id="operator", capability=submission.control_capability, command=command
        )
        assert duplicate.duplicate and duplicate.accepted_sequence == accepted.accepted_sequence
        with pytest.raises(EvaluationJobError, match="request_conflict"):
            await port.control_async(
                job_id=request.job_id,
                actor_id="operator",
                capability=submission.control_capability,
                command=command.model_copy(update={"message": "changed"}),
            )
        terminal = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert terminal.state is EvaluationJobState.CANCELLED and terminal.canonical is None
        assert runtime.commands == [command]
        kinds = [event.kind for event in terminal.events]
        assert kinds.index(EvaluationJobEventKind.CONTROL_ACCEPTED) < kinds.index(
            EvaluationJobEventKind.CONTROL_DELIVERED
        )
        assert kinds[-1] is EvaluationJobEventKind.TERMINAL
    finally:
        await port.shutdown_async()


async def test_writer_failure_retains_source_and_cancel_cannot_undo_finalization_async(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    writer = _NativeWriter(fail=True)
    writer.hold = True
    port = await _port_async(root=tmp_path, runtime=runtime, writer=writer)
    request = runtime.request()
    try:
        await port.submit_async(request=request, actor_id="operator")
        port.start_consumer()
        await asyncio.wait_for(writer.started.wait(), timeout=5)
        with pytest.raises(EvaluationJobError, match="cancel_too_late"):
            await port.cancel_async(job_id=request.job_id, actor_id="operator")
        writer.release.set()
        terminal = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert terminal.state is EvaluationJobState.FAILED and terminal.reason == "writer_failed"
        assert terminal.evidence is EvaluationEvidenceState.SOURCE_RETAINED and terminal.canonical is None
        assert (tmp_path / "attempts" / str(request.job_id) / "artifacts" / "native.jsonl").read_bytes()
        assert "private" not in terminal.model_dump_json()
    finally:
        writer.release.set()
        await port.shutdown_async()


def test_restart_dispatch_quarantine_and_exclusive_owner(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    request = runtime.request()
    ledger = LocalEvaluationJobLedger(tmp_path)
    ledger.startup()
    try:
        second = LocalEvaluationJobLedger(tmp_path)
        with pytest.raises(EvaluationJobError, match="request_conflict"):
            second.startup()
        ledger.submit(request=request, actor_id="operator")
        delivery = EvaluationJobDelivery(job_id=request.job_id, request_sha256=request.request_sha256)
        state, fence = ledger.claim(delivery=delivery, allowed_actor_ids=frozenset({"operator"}))
        assert state is EvaluationDeliveryState.STARTED and fence is not None
    finally:
        ledger.close()
    ledger = LocalEvaluationJobLedger(tmp_path)
    ledger.startup()
    try:
        assert ledger.snapshot(job_id=request.job_id, actor_id="operator").state is EvaluationJobState.INTERRUPTED
        assert ledger.claim(delivery=delivery, allowed_actor_ids=frozenset({"operator"}))[0] is (
            EvaluationDeliveryState.DUPLICATE
        )
        assert ledger.submit(request=request, actor_id="operator").duplicate
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            ledger.submit(request=runtime.request(), actor_id="operator")
        with pytest.raises(EvaluationJobError, match="order_conflict"):
            ledger.snapshot(job_id=request.job_id, actor_id="operator", after_sequence=100)
    finally:
        ledger.close()


async def test_unknown_runtime_failure_does_not_allow_fresh_dispatch_async(tmp_path: Path) -> None:
    runtime = _NativeFixture(fail=True)
    port = await _port_async(root=tmp_path, runtime=runtime)
    request = runtime.request()
    try:
        await port.submit_async(request=request, actor_id="operator")
        port.start_consumer()
        result = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert result.cleanup is EvaluationCleanupState.UNKNOWN and result.reason == "runtime_failed"
        assert "private failure" not in result.model_dump_json()
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=runtime.request(), actor_id="operator")
    finally:
        await port.shutdown_async()


def test_engine_registration_is_explicit_and_native_cannot_claim_inspect_artifacts() -> None:
    runtime = _NativeFixture()
    with pytest.raises(EvaluationJobError, match="request_conflict"):
        EvaluationJobRuntimeRegistry((runtime, runtime))
    with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
        EvaluationJobRuntimeRegistry(()).resolve(runtime.request())
    with pytest.raises(ValidationError):
        EvaluationJobRegistration.model_validate_json(
            runtime.registration.model_copy(
                update={"artifact_kind": EvaluationArtifactKind.INSPECT_EVAL}
            ).model_dump_json()
        )


async def test_queued_cancel_never_dispatches_and_a_fresh_independent_job_runs_async(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    port = await _port_async(root=tmp_path, runtime=runtime)
    request = runtime.request()
    try:
        await port.submit_async(request=request, actor_id="operator")
        cancelled = await port.cancel_async(job_id=request.job_id, actor_id="operator")
        assert cancelled.state is EvaluationJobState.CANCELLED
        assert cancelled.cleanup is EvaluationCleanupState.NOT_STARTED
        assert cancelled.evidence is EvaluationEvidenceState.ABSENT and cancelled.canonical is None
        assert (await port.cancel_async(job_id=request.job_id, actor_id="operator")) == cancelled
        assert (
            await port.receive_async(
                delivery=EvaluationJobDelivery(job_id=request.job_id, request_sha256=request.request_sha256)
            )
        ).state is EvaluationDeliveryState.DUPLICATE
        fresh = runtime.request()
        await port.submit_async(request=fresh, actor_id="operator")
        port.start_consumer()
        assert (await port.wait_async(job_id=fresh.job_id, actor_id="operator")).state is EvaluationJobState.SUCCEEDED
        assert runtime.executions == 1
    finally:
        await port.shutdown_async()


async def test_receiver_claim_and_dispatch_remain_owned_after_caller_disconnect_async(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    port = await _port_async(root=tmp_path, runtime=runtime)
    request = runtime.request()
    entered, release = threading.Event(), threading.Event()
    ledger = port._ledger
    assert ledger is not None
    original_claim = ledger.claim

    def held_claim(
        *, delivery: EvaluationJobDelivery, allowed_actor_ids: frozenset[str]
    ) -> tuple[EvaluationDeliveryState, UUID | None]:
        result = original_claim(delivery=delivery, allowed_actor_ids=allowed_actor_ids)
        entered.set()
        assert release.wait(5)
        return result

    try:
        await port.submit_async(request=request, actor_id="operator")
        with patch.object(ledger, "claim", side_effect=held_claim):
            caller = asyncio.create_task(
                port.receive_async(
                    delivery=EvaluationJobDelivery(job_id=request.job_id, request_sha256=request.request_sha256)
                )
            )
            assert await asyncio.to_thread(entered.wait, 5)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert port.has_active_work()
            release.set()
            result = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert result.state is EvaluationJobState.SUCCEEDED
        assert runtime.executions == 1
        assert [event.kind for event in result.events].count(EvaluationJobEventKind.STARTED) == 1
    finally:
        release.set()
        await port.shutdown_async()


async def test_finalization_barrier_and_shutdown_join_despite_repeated_cancellation_async(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    writer = _NativeWriter()
    writer.hold = True
    port = await _port_async(root=tmp_path, runtime=runtime, writer=writer)
    request = runtime.request()
    entered, release = threading.Event(), threading.Event()
    ledger = port._ledger
    assert ledger is not None
    original_begin = ledger.begin_finalizing

    def held_begin(*, job_id: UUID, fence_id: UUID, manifest: EvaluationArtifactManifest) -> None:
        original_begin(job_id=job_id, fence_id=fence_id, manifest=manifest)
        entered.set()
        assert release.wait(5)

    try:
        await port.submit_async(request=request, actor_id="operator")
        with patch.object(ledger, "begin_finalizing", side_effect=held_begin):
            port.start_consumer()
            assert await asyncio.to_thread(entered.wait, 5)
            driver = port._tasks[request.job_id]
            driver.cancel()
            driver.cancel()
            await asyncio.sleep(0.02)
            assert not driver.done()
            assert (await port.status_async(job_id=request.job_id, actor_id="operator")).state is (
                EvaluationJobState.FINALIZING
            )
            release.set()
            await asyncio.wait_for(writer.started.wait(), 5)
            shutdown = asyncio.create_task(port.shutdown_async())
            await asyncio.sleep(0.02)
            shutdown.cancel()
            await asyncio.sleep(0.02)
            shutdown.cancel()
            await asyncio.sleep(0.02)
            assert not shutdown.done() and not driver.done()
            writer.release.set()
            with pytest.raises(asyncio.CancelledError):
                await shutdown
        assert runtime.executions == writer.imports == 1
        reopened = LocalEvaluationJobLedger(tmp_path)
        reopened.startup()
        try:
            result = reopened.snapshot(job_id=request.job_id, actor_id="operator")
            assert result.state is EvaluationJobState.SUCCEEDED and result.canonical
            assert [event.kind for event in result.events].count(EvaluationJobEventKind.TERMINAL) == 1
        finally:
            reopened.close()
    finally:
        release.set()
        writer.release.set()
        await port.shutdown_async()


@pytest.mark.parametrize("substitution", ["payload", "request", "fence"])
async def test_artifact_substitution_never_reaches_canonical_writer_async(*, tmp_path: Path, substitution: str) -> None:
    runtime = _NativeFixture()
    writer = _NativeWriter()
    port = await _port_async(root=tmp_path, runtime=runtime, writer=writer)
    request = runtime.request()
    original_execute = runtime.execute_async

    async def substituted_async(*, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        artifacts = await original_execute(context=context)
        manifest = artifacts.manifest
        payloads = artifacts.payloads
        if substitution == "payload":
            payloads = tuple((name, content + b"changed") for name, content in payloads)
        elif substitution == "request":
            foreign = request.model_copy(update={"attempt_id": uuid4()})
            manifest = manifest.model_copy(update={"request": foreign, "request_sha256": foreign.request_sha256})
        else:
            manifest = manifest.model_copy(update={"fence_id": uuid4()})
        return EvaluationRuntimeArtifacts(manifest=manifest, payloads=payloads, cleanup=artifacts.cleanup)

    try:
        with patch.object(runtime, "execute_async", side_effect=substituted_async):
            await port.submit_async(request=request, actor_id="operator")
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert terminal.state is EvaluationJobState.FAILED and terminal.reason == "artifact_mismatch"
        assert terminal.canonical is None and writer.imports == 0
        assert terminal.cleanup is EvaluationCleanupState.VERIFIED
    finally:
        await port.shutdown_async()


def test_foreign_fence_and_terminal_order_are_rejected_without_inventing_success(tmp_path: Path) -> None:
    runtime = _NativeFixture()
    request = runtime.request()
    ledger = LocalEvaluationJobLedger(tmp_path)
    ledger.startup()
    try:
        ledger.submit(request=request, actor_id="operator")
        _, fence = ledger.claim(
            delivery=EvaluationJobDelivery(job_id=request.job_id, request_sha256=request.request_sha256),
            allowed_actor_ids=frozenset({"operator"}),
        )
        assert fence is not None
        for fence_id, state in ((uuid4(), EvaluationJobState.FAILED), (fence, EvaluationJobState.SUCCEEDED)):
            with pytest.raises(EvaluationJobError):
                ledger.finish(
                    job_id=request.job_id,
                    fence_id=fence_id,
                    state=state,
                    cleanup=EvaluationCleanupState.VERIFIED,
                    evidence=EvaluationEvidenceState.ABSENT,
                )
        assert ledger.snapshot(job_id=request.job_id, actor_id="operator").state is EvaluationJobState.RUNNING
        ledger.finish(
            job_id=request.job_id,
            fence_id=fence,
            state=EvaluationJobState.FAILED,
            cleanup=EvaluationCleanupState.UNKNOWN,
            evidence=EvaluationEvidenceState.UNKNOWN,
            reason="fixture_failure",
        )
        with pytest.raises(EvaluationJobError, match="order_conflict"):
            ledger.finish(
                job_id=request.job_id,
                fence_id=fence,
                state=EvaluationJobState.FAILED,
                cleanup=EvaluationCleanupState.UNKNOWN,
                evidence=EvaluationEvidenceState.UNKNOWN,
            )
    finally:
        ledger.close()


async def test_registered_primary_artifact_kind_cannot_be_substituted_by_a_variant_handler_async(
    tmp_path: Path,
) -> None:
    runtime = _NativeFixture()
    runtime.registration = runtime.registration.model_copy(
        update={"runtime": EvaluationRuntimeKind.INSPECT_VARIANT, "artifact_kind": EvaluationArtifactKind.INSPECT_EVAL}
    )
    writer = _NativeWriter()
    port = await _port_async(root=tmp_path, runtime=runtime, writer=writer)
    request = runtime.request()
    try:
        await port.submit_async(request=request, actor_id="operator")
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id="operator")
        assert terminal.state is EvaluationJobState.FAILED and terminal.reason == "artifact_mismatch"
        assert terminal.canonical is None and writer.imports == 0
    finally:
        await port.shutdown_async()

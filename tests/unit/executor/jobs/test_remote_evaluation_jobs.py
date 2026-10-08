# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mock HTTP refusal tests and real local canonical import; not separate-process qualification."""

from __future__ import annotations

import asyncio
import hashlib
import io
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from inspect_ai.log import read_eval_log

from pyrit.executor.jobs.inspect import PublicOriginalInspectJobRuntime
from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.executor.jobs.remote import create_remote_original_job_port_async
from pyrit.executor.jobs.worker_auth import EvaluationWorkerAuthContext, EvaluationWorkerCredentials
from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpClient, EvaluationWorkerHttpSettings
from pyrit.models import AttackOutcome, ScoreStatus
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactKind,
    EvaluationArtifactManifest,
    EvaluationArtifactMediaType,
    EvaluationCleanupState,
    EvaluationControlRequest,
    EvaluationEvidenceState,
    EvaluationJobEventKind,
    EvaluationJobRequest,
    EvaluationJobState,
    EvaluationWaitBoundary,
)
from pyrit.models.evaluation_worker import (
    EvaluationGatewaySettlement,
    EvaluationGatewaySettlementReceipt,
    EvaluationWorkerAdmission,
    EvaluationWorkerBinding,
    EvaluationWorkerCatalog,
    EvaluationWorkerEvent,
    EvaluationWorkerEvidence,
    EvaluationWorkerProtocol,
    EvaluationWorkerSnapshot,
    EvaluationWorkerState,
    EvaluationWorkerSubmission,
    EvaluationWorkerTerminal,
    evaluation_worker_schema_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.executor.jobs.remote import RemoteEvaluationJobGateway
    from pyrit.memory import SQLiteMemory

ACTOR = "00000000-0000-4000-8000-000000000001"
pytestmark = pytest.mark.filterwarnings(r"ignore:MemoryInterface\.:DeprecationWarning")


class _Credentials:
    audience = "public-fixture"
    fixture_only = True

    def __init__(self) -> None:
        self.contexts: list[EvaluationWorkerAuthContext] = []
        self.closed = False

    async def credentials_async(self, *, context: EvaluationWorkerAuthContext) -> EvaluationWorkerCredentials:
        self.contexts.append(context)
        return EvaluationWorkerCredentials(service_token="unit-service", operator_delegation="unit-delegation")

    async def close_async(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("url", "audience", "loopback"),
    [
        ("https://worker.example.invalid", "public-fixture", False),
        ("https://localhost", "public-fixture", False),
        ("http://127.0.0.1", "wrong-audience", True),
    ],
)
def test_fixture_credentials_require_exact_audience_and_explicit_loopback(
    *, url: str, audience: str, loopback: bool
) -> None:
    credentials = _Credentials()
    settings = EvaluationWorkerHttpSettings(
        base_url=url,
        audience=audience,
        service_id="public_fixture",
        schema_sha256=evaluation_worker_schema_sha256(),
        allow_loopback_http=loopback,
    )
    with pytest.raises(ValueError, match="audience"):
        EvaluationWorkerHttpClient(settings=settings, credentials=credentials)
    assert not credentials.contexts


@dataclass(frozen=True, kw_only=True)
class _Context:
    request: EvaluationJobRequest
    fence_id: UUID
    actor_id: str
    run_root: Path

    async def wait_for_control_async(self, *, boundary: EvaluationWaitBoundary) -> EvaluationControlRequest:
        raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_CONTROL)


class _WorkerMock:
    """Typed mock transport; source bytes are real only when real_original is explicitly enabled."""

    def __init__(
        self, *, runtime: PublicOriginalInspectJobRuntime, root: Path, mode: str = "", real_original: bool = False
    ) -> None:
        self.runtime = runtime
        self.root = root
        self.mode = mode
        self.real_original = real_original
        self.admission: EvaluationWorkerAdmission | None = None
        self.binding: EvaluationWorkerBinding | None = None
        self.manifest: EvaluationArtifactManifest | None = None
        self.content = b"wire-only fixture, not an Inspect archive"
        self.events: list[EvaluationWorkerEvent] = []
        self.requests: list[httpx.Request] = []
        self.settlements: list[EvaluationGatewaySettlement] = []
        self.started = asyncio.Event()
        self.executions = 0
        self.cancelled = False

    def _event(self, *, kind: EvaluationJobEventKind, state: EvaluationWorkerState) -> None:
        self.events.append(
            EvaluationWorkerEvent(sequence=len(self.events) + 1, kind=kind, state=state, occurred_at=datetime.now(UTC))
        )

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["Authorization"] == "Bearer unit-service"
        assert request.headers["X-PyRIT-Operator-Delegation"] == "unit-delegation"
        path = request.url.path
        if path.endswith("/protocol"):
            if self.mode == "redirect":
                return httpx.Response(302, headers={"Location": "https://unapproved.example.invalid"})
            if self.mode == "encoding":
                return httpx.Response(
                    200,
                    stream=httpx.ByteStream(b"not compressed"),
                    headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
                )
            if self.mode == "large_json":
                return httpx.Response(
                    200, content=b" " * (256 * 1024 + 1), headers={"Content-Type": "application/json"}
                )
            protocol = EvaluationWorkerProtocol(
                service_id="public_fixture", schema_sha256=evaluation_worker_schema_sha256()
            )
            body = protocol.model_dump(mode="json")
            if self.mode == "version":
                body["schema_sha256"] = "f" * 64
            return httpx.Response(200, json=body)
        if path.endswith("/catalog"):
            registrations = () if self.mode == "catalog" else (self.runtime.registration,)
            return httpx.Response(
                200,
                json=EvaluationWorkerCatalog(
                    service_id="public_fixture", actor_id=ACTOR, registrations=registrations
                ).model_dump(mode="json"),
            )
        if path.endswith("/jobs"):
            if self.mode == "auth":
                return httpx.Response(403, json={"detail": "not_authorized"})
            admission = EvaluationWorkerAdmission.model_validate_json(request.content)
            if self.admission is not None:
                assert admission == self.admission
                assert self.binding
                return httpx.Response(
                    202, json=EvaluationWorkerSubmission(binding=self.binding, duplicate=True).model_dump(mode="json")
                )
            self.admission = admission
            self.binding = EvaluationWorkerBinding(
                service_id="public_fixture",
                worker_incarnation_id=uuid4(),
                worker_fence_id=uuid4(),
                gateway_fence_id=admission.gateway_fence_id,
                job_id=admission.request.job_id,
                run_id=admission.request.run_id,
                attempt_id=admission.request.attempt_id,
                request_sha256=admission.request_sha256,
                admission_sha256=admission.admission_sha256,
                actor_id=admission.actor_id,
            )
            self.executions += 1
            self._event(kind=EvaluationJobEventKind.SUBMITTED, state=EvaluationWorkerState.QUEUED)
            self._event(kind=EvaluationJobEventKind.STARTED, state=EvaluationWorkerState.RUNNING)
            if self.mode not in {"held", "deadline"}:
                await self._complete_async()
            self.started.set()
            body = EvaluationWorkerSubmission(binding=self.binding).model_dump(mode="json")
            if self.mode == "shared_fence":
                body["binding"]["worker_fence_id"] = str(admission.gateway_fence_id)
            return httpx.Response(202, json=body)
        assert self.binding and self.admission
        if path.endswith("/cancel"):
            self.cancelled = True
            self._event(kind=EvaluationJobEventKind.CANCEL_REQUESTED, state=EvaluationWorkerState.CANCEL_REQUESTED)
            self._event(kind=EvaluationJobEventKind.TERMINAL, state=EvaluationWorkerState.CANCELLED)
            return httpx.Response(200, json=self._snapshot(after_sequence=0).model_dump(mode="json"))
        if path.endswith("/settlement"):
            settlement = EvaluationGatewaySettlement.model_validate_json(request.content)
            self.settlements.append(settlement)
            if self.mode == "settlement":
                raise httpx.ReadTimeout("unit settlement response lost")
            receipt = EvaluationGatewaySettlementReceipt(
                job_id=self.binding.job_id,
                binding_sha256=self.binding.binding_sha256,
                settlement_sha256=settlement.settlement_sha256,
            )
            return httpx.Response(200, json=receipt.model_dump(mode="json"))
        if "/artifacts/" in path:
            assert self.manifest
            if path.endswith("/manifest"):
                manifest = self.manifest
                if self.mode == "fence":
                    manifest = manifest.model_copy(update={"fence_id": uuid4()})
                return httpx.Response(
                    200, content=manifest.model_dump_json(), headers={"Content-Type": "application/json"}
                )
            content = self.content
            if self.mode == "truncated":
                content = content[:-1]
            if self.mode == "foreign":
                content = b"x" * len(content)
            if self.mode == "oversize":
                content += b"extra source bytes"
            return httpx.Response(200, content=content, headers={"Content-Type": "application/octet-stream"})
        after_sequence = int(request.url.params.get("after_sequence", "0"))
        body = self._snapshot(after_sequence=after_sequence).model_dump(mode="json")
        if self.mode == "canonical":
            body["canonical"] = {"projection_id": "foreign_worker_database", "score_ids": [str(uuid4())]}
        elif self.mode == "incarnation":
            body["binding"]["worker_incarnation_id"] = str(uuid4())
        elif self.mode == "order" and body["events"]:
            body["events"][0]["sequence"] = 9
        return httpx.Response(200, json=body)

    async def _complete_async(self) -> None:
        assert self.admission and self.binding
        if self.real_original:
            self.root.mkdir()
            artifacts = await self.runtime.execute_async(
                context=_Context(
                    request=self.admission.request,
                    fence_id=self.binding.worker_fence_id,
                    actor_id=ACTOR,
                    run_root=self.root,
                )
            )
            self.manifest = artifacts.manifest
            self.content = artifacts.payloads[0][1]
        else:
            artifact = EvaluationArtifact(
                name="original.eval",
                kind=EvaluationArtifactKind.INSPECT_EVAL,
                media_type=EvaluationArtifactMediaType.INSPECT_EVAL,
                bytes=len(self.content),
                sha256=hashlib.sha256(self.content).hexdigest(),
            )
            self.manifest = EvaluationArtifactManifest(
                request=self.admission.request,
                request_sha256=self.admission.request_sha256,
                fence_id=self.binding.worker_fence_id,
                artifacts=(artifact,),
            )
        self._event(kind=EvaluationJobEventKind.ARTIFACTS_RETAINED, state=EvaluationWorkerState.RUNNING)
        self._event(kind=EvaluationJobEventKind.TERMINAL, state=EvaluationWorkerState.COMPLETED)

    def _snapshot(self, *, after_sequence: int) -> EvaluationWorkerSnapshot:
        assert self.binding and self.admission
        state = self.events[-1].state
        retained = self.manifest is not None
        evidence = EvaluationWorkerEvidence.SOURCE_RETAINED if retained else EvaluationWorkerEvidence.ABSENT
        cleanup = EvaluationCleanupState.VERIFIED if state.terminal else EvaluationCleanupState.UNKNOWN
        terminal = None
        if state.terminal:
            terminal = EvaluationWorkerTerminal(
                binding=self.binding,
                state=state,
                evidence=evidence,
                cleanup=cleanup,
                last_sequence=len(self.events),
                manifest_sha256=self.manifest.manifest_sha256 if self.manifest else None,
            )
        return EvaluationWorkerSnapshot(
            request=self.admission.request,
            request_sha256=self.admission.request_sha256,
            binding=self.binding,
            state=state,
            evidence=evidence,
            cleanup=cleanup,
            last_sequence=len(self.events),
            events=tuple(item for item in self.events if item.sequence > after_sequence),
            terminal=terminal,
        )


async def _gateway_async(
    *, memory: SQLiteMemory, root: Path, mode: str = "", real_original: bool = False
) -> tuple[RemoteEvaluationJobGateway, _WorkerMock, _Credentials]:
    original = await PublicOriginalInspectJobRuntime.create_async()
    worker = _WorkerMock(runtime=original, root=root / "source", mode=mode, real_original=real_original)
    credentials = _Credentials()
    settings = EvaluationWorkerHttpSettings(
        base_url="http://127.0.0.1",
        audience=credentials.audience,
        service_id="public_fixture",
        schema_sha256=evaluation_worker_schema_sha256(),
        allow_loopback_http=True,
        poll_interval_seconds=0.01,
        poll_deadline_seconds=0.1 if mode == "deadline" else 30,
    )
    port = await create_remote_original_job_port_async(
        root=root / "gateway",
        memory=memory,
        allowed_actor_ids=frozenset({ACTOR}),
        settings=settings,
        credentials=credentials,
        transport=httpx.MockTransport(worker.handle_async),
    )
    return port, worker, credentials


async def test_mock_transport_exact_archive_is_imported_only_into_api_canonical_sqlite_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, credentials = await _gateway_async(memory=sqlite_instance, root=tmp_path, real_original=True)
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        assert (await port.submit_async(request=request, actor_id=ACTOR)).duplicate
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.SUCCEEDED and terminal.canonical
        assert worker.executions == 1 and len(worker.settlements) == 1
        scores = await sqlite_instance.get_scores_async(score_ids=[str(item) for item in terminal.canonical.score_ids])
        results = await sqlite_instance.get_attack_results_async(
            attack_result_ids=[str(item) for item in terminal.canonical.attack_result_ids]
        )
        assert scores[0].score_value == "1.0" and scores[0].status is ScoreStatus.COMPLETE
        assert results[0].outcome is AttackOutcome.UNDETERMINED
        assert results[0].automated_score and results[0].automated_score.id == scores[0].id
        assert worker.manifest and terminal.manifest
        assert worker.manifest.fence_id != terminal.manifest.fence_id
        assert worker.manifest.artifacts == terminal.manifest.artifacts
        row = await asyncio.to_thread(port.journal.read, request.job_id)
        assert row["worker_manifest_json"] == worker.manifest.model_dump_json().encode()
        assert row["worker_manifest_sha256"] == worker.manifest.manifest_sha256
        assert row["gateway_manifest_sha256"] == terminal.manifest.manifest_sha256
        assert row["canonical_json"] == terminal.canonical.model_dump_json()
        assert row["worker_cursor"] == 4 and terminal.last_sequence == 5
        assert "score_ids" not in worker.settlements[0].model_dump_json()
        typed = read_eval_log(io.BytesIO(worker.content), format="eval")
        assert typed.samples and typed.samples[0].store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
        assert not typed.samples[0].model_usage
        delegated = next(item for item in credentials.contexts if item.path.endswith("/jobs"))
        assert delegated.actor_id == ACTOR and delegated.request_sha256 == request.request_sha256
        assert delegated.gateway_fence_id == terminal.manifest.fence_id
    finally:
        await port.shutdown_async()
    assert credentials.closed


@pytest.mark.parametrize(
    "mode",
    [
        "auth",
        "canonical",
        "incarnation",
        "order",
        "fence",
        "shared_fence",
        "truncated",
        "foreign",
        "oversize",
        "deadline",
    ],
)
async def test_remote_refusal_never_imports_worker_results_and_quarantines_uncertainty_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, mode: str
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode=mode)
    await port.startup_async()
    writer = port._writers[worker.runtime.registration.runtime]
    with patch.object(writer, "import_async", new_callable=AsyncMock) as imported:
        try:
            request = worker.runtime.request()
            await port.submit_async(request=request, actor_id=ACTOR)
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
            assert terminal.state is EvaluationJobState.FAILED
            assert terminal.cleanup is EvaluationCleanupState.UNKNOWN and terminal.canonical is None
            imported.assert_not_awaited()
            assert worker.executions == (0 if mode == "auth" else 1)
            with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
                await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
            assert not worker.cancelled
        finally:
            await port.shutdown_async()


@pytest.mark.parametrize("mode", ["version", "catalog"])
async def test_remote_startup_refuses_unapproved_catalog_or_protocol_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, mode: str
) -> None:
    port, worker, credentials = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode=mode)
    try:
        with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
            await port.startup_async()
        assert worker.executions == 0
    finally:
        await port.shutdown_async()
    assert credentials.closed


async def test_joined_remote_cancel_requires_closure_ack_without_fabricated_archive_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode="held")
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        port.start_consumer()
        await asyncio.wait_for(worker.started.wait(), 5)
        await port.cancel_async(job_id=request.job_id, actor_id=ACTOR)
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.CANCELLED
        assert terminal.cleanup is EvaluationCleanupState.VERIFIED and terminal.canonical is None
        assert worker.cancelled and len(worker.settlements) == 1
        closure = worker.settlements[0]
        assert closure.disposition.value == "closure_observed"
        assert closure.worker_manifest_sha256 is closure.gateway_manifest_sha256 is closure.artifact_sha256 is None
        assert (await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)).duplicate is False
    finally:
        await port.shutdown_async()


async def test_actual_api_grade_survives_lost_settlement_but_new_dispatch_is_blocked_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode="settlement", real_original=True)
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED
        assert terminal.reason == "remote_settlement_uncertain" and terminal.cleanup is EvaluationCleanupState.UNKNOWN
        assert terminal.evidence is EvaluationEvidenceState.CANONICAL and terminal.canonical
        scores = await sqlite_instance.get_scores_async(score_ids=[str(item) for item in terminal.canonical.score_ids])
        assert scores[0].status is ScoreStatus.COMPLETE and scores[0].score_value == "1.0"
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
        row = await asyncio.to_thread(port.journal.read, request.job_id)
        assert row["canonical_json"] == terminal.canonical.model_dump_json()
        assert row["settlement_json"] is None and row["uncertain"] == 1
        assert worker.executions == 1
    finally:
        await port.shutdown_async()


async def test_actual_api_grade_survives_remote_journal_failure_without_successful_settlement_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path, real_original=True)
    await port.startup_async()
    try:
        with (
            patch.object(port.journal, "canonical", side_effect=sqlite3.OperationalError("unit journal unavailable")),
            patch.object(port.journal, "uncertain", side_effect=sqlite3.OperationalError("unit journal unavailable")),
        ):
            request = worker.runtime.request()
            await port.submit_async(request=request, actor_id=ACTOR)
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED and terminal.canonical
        assert terminal.cleanup is EvaluationCleanupState.UNKNOWN and terminal.reason == "remote_settlement_uncertain"
        scores = await sqlite_instance.get_scores_async(score_ids=[str(item) for item in terminal.canonical.score_ids])
        assert scores[0].score_value == "1.0" and scores[0].status is ScoreStatus.COMPLETE
        assert worker.executions == 1 and not worker.settlements
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
    finally:
        await port.shutdown_async()


async def test_remote_handoff_refusal_before_writer_does_not_release_unsettled_dispatch_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path)
    await port.startup_async()
    try:
        with patch.object(
            port.journal, "handoff", side_effect=EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        ):
            request = worker.runtime.request()
            await port.submit_async(request=request, actor_id=ACTOR)
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED and terminal.canonical is None
        assert terminal.cleanup is EvaluationCleanupState.UNKNOWN and not worker.settlements
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
    finally:
        await port.shutdown_async()


async def test_cancel_after_source_retention_before_publication_preserves_unsettled_quarantine_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path, real_original=True)
    await port.startup_async()
    ledger = port._require_open()
    begin = ledger.begin_finalizing

    def _cancel_at_barrier(*, job_id: UUID, fence_id: UUID, manifest: EvaluationArtifactManifest) -> None:
        ledger.cancel(job_id=job_id, actor_id=ACTOR)
        begin(job_id=job_id, fence_id=fence_id, manifest=manifest)

    try:
        with patch.object(ledger, "begin_finalizing", side_effect=_cancel_at_barrier):
            request = worker.runtime.request()
            await port.submit_async(request=request, actor_id=ACTOR)
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.CANCELLED and terminal.canonical is None
        assert terminal.evidence is EvaluationEvidenceState.SOURCE_RETAINED and terminal.manifest
        assert terminal.cleanup is EvaluationCleanupState.UNKNOWN and not worker.settlements
        row = await asyncio.to_thread(port.journal.read, request.job_id)
        assert row["worker_manifest_json"] and row["uncertain"] == 1
        assert worker.executions == 1 and worker.manifest
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
    finally:
        await port.shutdown_async()


async def test_remote_gateway_restart_keeps_binding_cursor_and_never_replays_unknown_dispatch_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode="deadline")
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.cleanup is EvaluationCleanupState.UNKNOWN
        before = await asyncio.to_thread(port.journal.read, request.job_id)
    finally:
        await port.shutdown_async()
    restarted, replacement, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path)
    await restarted.startup_async()
    try:
        restarted.start_consumer()
        assert await asyncio.to_thread(restarted.journal.read, request.job_id) == before
        assert (
            await restarted.status_async(job_id=request.job_id, actor_id=ACTOR)
        ).cleanup is EvaluationCleanupState.UNKNOWN
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await restarted.submit_async(request=replacement.runtime.request(), actor_id=ACTOR)
        assert replacement.executions == 0 and worker.executions == 1
        assert not any(item.url.path.endswith("/jobs") for item in replacement.requests)
    finally:
        await restarted.shutdown_async()


@pytest.mark.parametrize("mode", ["redirect", "encoding", "large_json"])
async def test_worker_http_never_follows_redirects_or_accepts_unbounded_encoded_json_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, mode: str
) -> None:
    port, worker, credentials = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode=mode)
    try:
        with pytest.raises(EvaluationJobError):
            await port.startup_async()
        assert worker.executions == 0 and len(worker.requests) == 1
        assert len(credentials.contexts) == 1
    finally:
        await port.shutdown_async()


async def test_gateway_actor_and_pinned_catalog_cannot_admit_foreign_runtime_or_case_alias_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, credentials = await _gateway_async(memory=sqlite_instance, root=tmp_path, mode="held")
    await port.startup_async()
    try:
        request = worker.runtime.request()
        before = len(credentials.contexts)
        with pytest.raises(EvaluationJobError, match="not_authorized"):
            await port.catalog_async(actor_id="foreign")
        assert len(credentials.contexts) == before
        changed = request.model_copy(
            update={"source": request.source.model_copy(update={"name": "unknown_remote_task"})}
        )
        with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
            await port.submit_async(request=changed, actor_id=ACTOR)
        await port.submit_async(request=request, actor_id=ACTOR)
        alias = request.model_copy(update={"job_id": uuid4(), "attempt_id": uuid4()})
        with pytest.raises(EvaluationJobError, match="request_conflict"):
            await port.submit_async(request=alias, actor_id=ACTOR)
        assert worker.executions == 0
    finally:
        await port.shutdown_async()


async def test_durable_worker_cursor_rejects_changed_event_and_manifest_provenance_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _ = await _gateway_async(memory=sqlite_instance, root=tmp_path)
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED and terminal.canonical is None
        snapshot = worker._snapshot(after_sequence=0)
        changed = snapshot.model_copy(
            update={"events": (snapshot.events[0].model_copy(update={"reason": "changed"}), *snapshot.events[1:])}
        )
        with pytest.raises(EvaluationJobError, match="order_conflict"):
            await asyncio.to_thread(port.journal.observe, changed)
        admission, binding, original, derived = await asyncio.to_thread(port.journal.handoff, request.job_id)
        assert binding.accepts(admission) and original.fence_id != derived.fence_id
        with pytest.raises(EvaluationJobError, match="artifact_mismatch"):
            await asyncio.to_thread(
                port.journal.retain,
                job_id=request.job_id,
                original=original.model_dump_json(indent=2).encode(),
                worker=original,
                gateway=derived,
            )
    finally:
        await port.shutdown_async()

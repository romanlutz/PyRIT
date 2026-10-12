# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mock HTTP refusal tests and real local canonical import; not separate-process qualification."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from inspect_ai.log import read_eval_log

from pyrit.backend.services.evaluation_job_service import RemoteEvaluationJobSettings
from pyrit.executor.jobs.inspect import OriginalInspectArtifactWriter, PublicOriginalInspectJobRuntime
from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode, EvaluationRuntimeArtifacts
from pyrit.executor.jobs.remote import (
    RemoteEvaluationArtifactWriter,
    RemoteEvaluationJobGateway,
    RemoteEvaluationJobRuntime,
    RemoteOriginalInspectArtifactWriter,
    create_remote_job_port_async,
    create_remote_original_job_port_async,
)
from pyrit.executor.jobs.remote_ledger import EvaluationRemoteJournal
from pyrit.executor.jobs.worker_auth import (
    EvaluationWorkerAuthContext,
    EvaluationWorkerCredentialRegistry,
    EvaluationWorkerCredentials,
)
from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpClient, EvaluationWorkerHttpSettings
from pyrit.models import AttackOutcome, EvalPackageRef, EvalSourceKind, ScoreStatus
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactKind,
    EvaluationArtifactManifest,
    EvaluationArtifactMediaType,
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationControlKind,
    EvaluationControlRequest,
    EvaluationEvidenceState,
    EvaluationJobEventKind,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationJobState,
    EvaluationRuntimeKind,
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
from pyrit.models.native_cyber_evidence import (
    NativeCyberEpisodeStart,
    NativeCyberEvidenceSource,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory

ACTOR = "00000000-0000-4000-8000-000000000001"
pytestmark = pytest.mark.filterwarnings(r"ignore:MemoryInterface\.:DeprecationWarning")


class _Credentials:
    audience = "public-fixture"
    fixture_only = True

    def __init__(self) -> None:
        self.contexts: list[EvaluationWorkerAuthContext] = []
        self.closed = False
        self.close_calls = 0

    async def credentials_async(self, *, context: EvaluationWorkerAuthContext) -> EvaluationWorkerCredentials:
        self.contexts.append(context)
        return EvaluationWorkerCredentials(service_token="unit-service", operator_delegation="unit-delegation")

    async def close_async(self) -> None:
        self.close_calls += 1
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


class _NativeRegistrationFixture:
    """Installed metadata only; no native process, provider or platform execution."""

    def __init__(self) -> None:
        self.registration = EvaluationJobRegistration(
            runtime=EvaluationRuntimeKind.NATIVE_BINDING,
            source=EvalPackageRef(
                kind=EvalSourceKind.NAMED, name="public_native_transport_fixture", source_sha256="a" * 64
            ),
            case_id="b" * 64,
            execution_profile_sha256="c" * 64,
            artifact_kind=EvaluationArtifactKind.NATIVE_EVIDENCE,
        )

    def request(self) -> EvaluationJobRequest:
        return EvaluationJobRequest(
            job_id=uuid4(),
            run_id=uuid4(),
            attempt_id=uuid4(),
            runtime=self.registration.runtime,
            source=self.registration.source,
            case_id=self.registration.case_id,
            execution_profile_sha256=self.registration.execution_profile_sha256,
        )


class _NativeApiWriterFixture:
    """Retain mock transport bytes in actual API SQLite without inventing a native grade."""

    def __init__(self, memory: SQLiteMemory) -> None:
        self.memory = memory
        self.requests: list[EvaluationJobRequest] = []
        self.receipt: EvaluationCanonicalReceipt | None = None

    async def import_async(
        self, *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        artifacts.verify()
        self.requests.append(request)
        artifact = artifacts.manifest.artifacts[0]
        assert artifact.kind is EvaluationArtifactKind.NATIVE_EVIDENCE
        content = dict(artifacts.payloads)[artifact.name]
        stream = NativeCyberRawStreamStart(
            run_id=str(request.run_id),
            key=NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS,
                kind=NativeCyberRawKind.JSONL,
                observed_source_id="public_native_transport_fixture",
            ),
        )
        store = self.memory.native_cyber_evidence
        await asyncio.to_thread(
            store.create_episode,
            start=NativeCyberEpisodeStart(
                run_id=stream.run_id,
                binding_name="public_native_writer_fixture",
                binding_version=request.source.source_sha256,
                task_id=request.case_id,
                task_version="fixture",
                simulated=True,
                required_raw_streams=(stream.key,),
            ),
        )
        await asyncio.to_thread(store.open_raw_stream, stream=stream)
        await asyncio.to_thread(store.append_raw, run_id=stream.run_id, stream_id=stream.stream_id, data=content)
        await asyncio.to_thread(
            store.close_raw_stream,
            run_id=stream.run_id,
            stream_id=stream.stream_id,
            source_complete=True,
            expected_bytes=len(content),
            observed_sha256=artifact.sha256,
        )
        self.receipt = EvaluationCanonicalReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            manifest_sha256=artifacts.manifest.manifest_sha256,
            artifact_sha256=artifact.sha256,
            projection_id="public_native_ungraded_fixture",
            source_complete=True,
        )
        return self.receipt


class _WorkerMock:
    """Typed mock transport; source bytes are real only when real_original is explicitly enabled."""

    def __init__(
        self,
        *,
        runtime: PublicOriginalInspectJobRuntime | _NativeRegistrationFixture,
        root: Path,
        mode: str = "",
        real_original: bool = False,
    ) -> None:
        self.runtime = runtime
        self.root = root
        self.mode = mode
        self.real_original = real_original
        self.registrations = (runtime.registration,)
        self.admission: EvaluationWorkerAdmission | None = None
        self.binding: EvaluationWorkerBinding | None = None
        self.manifest: EvaluationArtifactManifest | None = None
        self.content = (
            b'{"event":"public_native_transport_fixture","guest_exit_code":null}\n'
            if runtime.registration.runtime is EvaluationRuntimeKind.NATIVE_BINDING
            else b"wire-only fixture, not an Inspect archive"
        )
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
            registrations = () if self.mode == "catalog" else self.registrations
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
                elif self.mode in {"manifest_source", "manifest_profile", "manifest_job"}:
                    changed = {
                        "manifest_source": {
                            "source": manifest.request.source.model_copy(update={"source_sha256": "e" * 64})
                        },
                        "manifest_profile": {"execution_profile_sha256": "e" * 64},
                        "manifest_job": {"job_id": uuid4()},
                    }[self.mode]
                    foreign = manifest.request.model_copy(update=changed)
                    manifest = manifest.model_copy(
                        update={"request": foreign, "request_sha256": foreign.request_sha256}
                    )
                elif self.mode in {"evidence", "artifact_bound", "aggregate_bound"}:
                    body = manifest.model_dump(mode="json")
                    if self.mode == "evidence":
                        body["artifacts"][0]["kind"] = EvaluationArtifactKind.INSPECT_EVAL.value
                        body["artifacts"][0]["media_type"] = EvaluationArtifactMediaType.INSPECT_EVAL.value
                    elif self.mode == "artifact_bound":
                        body["artifacts"][0]["bytes"] = 16 * 1024 * 1024 + 1
                    else:
                        body["artifacts"] = [
                            {**body["artifacts"][0], "name": f"native-{index}.ndjson", "bytes": 16 * 1024 * 1024}
                            for index in range(3)
                        ]
                    return httpx.Response(200, content=json.dumps(body), headers={"Content-Type": "application/json"})
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
            return httpx.Response(
                200, content=content, headers={"Content-Type": self.manifest.artifacts[0].media_type.value}
            )
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
            assert isinstance(self.runtime, PublicOriginalInspectJobRuntime)
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
                name="original.eval"
                if self.runtime.registration.artifact_kind is EvaluationArtifactKind.INSPECT_EVAL
                else "native.ndjson",
                kind=self.runtime.registration.artifact_kind,
                media_type=EvaluationArtifactMediaType[self.runtime.registration.artifact_kind.name],
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
        # This mock POST includes the full real Task lifecycle, not just a worker admission ACK.
        request_timeout_seconds=30 if real_original else 10,
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


async def _native_gateway_async(
    *, memory: SQLiteMemory, root: Path, mode: str = ""
) -> tuple[RemoteEvaluationJobGateway, _WorkerMock, _NativeApiWriterFixture, _Credentials]:
    runtime = _NativeRegistrationFixture()
    worker = _WorkerMock(runtime=runtime, root=root / "unused-native-source", mode=mode)
    writer = _NativeApiWriterFixture(memory)
    credentials = _Credentials()
    port = await create_remote_job_port_async(
        root=root / "gateway",
        registration=runtime.registration,
        writer=writer,
        allowed_actor_ids=frozenset({ACTOR}),
        settings=EvaluationWorkerHttpSettings(
            base_url="http://127.0.0.1",
            audience=credentials.audience,
            service_id="public_fixture",
            schema_sha256=evaluation_worker_schema_sha256(),
            allow_loopback_http=True,
            poll_interval_seconds=0.01,
            poll_deadline_seconds=0.1 if mode == "deadline" else 30,
        ),
        credentials=credentials,
        transport=httpx.MockTransport(worker.handle_async),
    )
    return port, worker, writer, credentials


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


async def test_installed_native_factory_uses_only_its_api_writer_and_retains_exact_ungraded_bytes_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    with (
        patch.object(PublicOriginalInspectJobRuntime, "create_async", new_callable=AsyncMock) as original,
        patch("pyrit.executor.jobs.remote.OriginalInspectArtifactWriter") as original_writer,
    ):
        port, worker, writer, credentials = await _native_gateway_async(memory=sqlite_instance, root=tmp_path)
        foreign = worker.runtime.registration.model_copy(update={"execution_profile_sha256": "d" * 64})
        worker.registrations = (*worker.registrations, foreign)
        await port.startup_async()
        try:
            assert await port.catalog_async(actor_id=ACTOR) == (worker.runtime.registration,)
            assert set(port._writers) == {EvaluationRuntimeKind.NATIVE_BINDING}
            request = worker.runtime.request()
            await port.submit_async(request=request, actor_id=ACTOR)
            assert (await port.submit_async(request=request, actor_id=ACTOR)).duplicate
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
            assert terminal.state is EvaluationJobState.SUCCEEDED and terminal.canonical == writer.receipt
            assert terminal.canonical.score_ids == terminal.canonical.attack_result_ids == ()
            assert writer.requests == [request] and worker.executions == len(worker.settlements) == 1
            assert worker.manifest and terminal.manifest
            assert worker.manifest.fence_id != terminal.manifest.fence_id
            assert worker.manifest.artifacts == terminal.manifest.artifacts
            row = await asyncio.to_thread(port.journal.read, request.job_id)
            assert row["worker_manifest_json"] == worker.manifest.model_dump_json().encode()
            assert row["gateway_manifest_json"] == terminal.manifest.model_dump_json()
            episode = await asyncio.to_thread(
                sqlite_instance.native_cyber_evidence.get_episode, run_id=str(request.run_id)
            )
            assert episode.score_id is None and episode.finalized_at is None
            assert len(episode.raw_streams) == 1 and episode.raw_streams[0].source_complete
            chunks = await asyncio.to_thread(
                sqlite_instance.native_cyber_evidence.read_raw_chunks,
                run_id=str(request.run_id),
                stream_id=episode.raw_streams[0].stream_id,
                allow_sensitive=True,
            )
            assert b"".join(chunk.data for chunk in chunks) == worker.content
            assert json.loads(worker.content)["guest_exit_code"] is None
            assert not await sqlite_instance.get_scores_async(score_type="true_false")
            assert not await sqlite_instance.get_scores_async(score_type="float_scale")
            assert not await sqlite_instance.get_attack_results_async(limit=1)
            delegated = next(item for item in credentials.contexts if item.path.endswith("/jobs"))
            assert delegated.actor_id == ACTOR and delegated.request_sha256 == request.request_sha256
            assert delegated.gateway_fence_id == terminal.manifest.fence_id
            assert "score_ids" not in worker.settlements[0].model_dump_json()
            assert not list(tmp_path.rglob("*.eval"))
            original.assert_not_awaited()
            original_writer.assert_not_called()
        finally:
            await port.shutdown_async()
        assert credentials.closed


@pytest.mark.parametrize("field", ["runtime", "source_name", "source_pin", "case", "profile", "controls"])
async def test_installed_native_request_requires_exact_binding_before_dispatch_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, field: str
) -> None:
    port, worker, writer, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path)
    await port.startup_async()
    try:
        request = worker.runtime.request()
        changed = {
            "runtime": {"runtime": EvaluationRuntimeKind.ORIGINAL_INSPECT},
            "source_name": {"source": request.source.model_copy(update={"name": "unknown_native_case"})},
            "source_pin": {"source": request.source.model_copy(update={"source_sha256": "d" * 64})},
            "case": {"case_id": "d" * 64},
            "profile": {"execution_profile_sha256": "d" * 64},
            "controls": {"controls": (EvaluationControlKind.SEND_MESSAGE,)},
        }[field]
        with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
            await port.submit_async(request=request.model_copy(update=changed), actor_id=ACTOR)
        assert not writer.requests and worker.executions == 0
        assert not any(item.url.path.endswith("/jobs") for item in worker.requests)
        assert port.registry.registrations == (worker.runtime.registration,)
    finally:
        await port.shutdown_async()


@pytest.mark.parametrize("field", ["source", "case", "profile", "controls", "runtime"])
async def test_remote_catalog_cannot_install_an_unreviewed_registration_or_writer_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, field: str
) -> None:
    port, worker, writer, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path)
    registration = worker.runtime.registration
    changed = {
        "source": {"source": registration.source.model_copy(update={"name": "catalog_only_source"})},
        "case": {"case_id": "d" * 64},
        "profile": {"execution_profile_sha256": "d" * 64},
        "controls": {"controls": (EvaluationControlKind.SEND_MESSAGE,)},
        "runtime": {"runtime": EvaluationRuntimeKind.INSPECT_VARIANT},
    }[field]
    worker.registrations = (registration.model_copy(update=changed),)
    try:
        with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
            await port.startup_async()
        assert port.registry.registrations == (registration,)
        assert set(port._writers) == {EvaluationRuntimeKind.NATIVE_BINDING}
        assert not writer.requests and worker.executions == 0
    finally:
        await port.shutdown_async()


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
        "manifest_source",
        "manifest_profile",
        "manifest_job",
        "evidence",
        "artifact_bound",
        "aggregate_bound",
    ],
)
async def test_native_transport_refuses_foreign_unbounded_or_worker_canonical_data_before_writer_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, mode: str
) -> None:
    port, worker, writer, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path, mode=mode)
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED and terminal.canonical is None
        assert terminal.cleanup is EvaluationCleanupState.UNKNOWN
        assert not writer.requests and not worker.settlements
        assert worker.executions == (0 if mode == "auth" else 1)
        if mode in {"artifact_bound", "aggregate_bound", "evidence"}:
            assert not any(
                "/artifacts/" in item.url.path and not item.url.path.endswith("/manifest") for item in worker.requests
            )
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
    finally:
        await port.shutdown_async()


async def test_native_cancel_joins_closure_without_invoking_any_canonical_writer_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, writer, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path, mode="held")
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
        assert not writer.requests and len(worker.settlements) == 1
        assert worker.settlements[0].disposition.value == "closure_observed"
        assert worker.settlements[0].artifact_sha256 is None
    finally:
        await port.shutdown_async()


async def test_native_canonical_retention_survives_uncertain_settlement_without_inventing_scores_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, writer, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path, mode="settlement")
    await port.startup_async()
    try:
        request = worker.runtime.request()
        await port.submit_async(request=request, actor_id=ACTOR)
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED and terminal.canonical == writer.receipt
        assert terminal.evidence is EvaluationEvidenceState.CANONICAL
        assert terminal.reason == "remote_settlement_uncertain" and terminal.cleanup is EvaluationCleanupState.UNKNOWN
        assert terminal.canonical.score_ids == terminal.canonical.attack_result_ids == ()
        assert writer.requests == [request] and len(worker.settlements) == 1
        with pytest.raises(EvaluationJobError, match="dispatch_uncertain"):
            await port.submit_async(request=worker.runtime.request(), actor_id=ACTOR)
    finally:
        await port.shutdown_async()


@pytest.mark.parametrize("field", ["job_id", "request_sha256", "manifest_sha256", "artifact_sha256"])
async def test_api_writer_receipt_must_bind_exact_local_handoff_before_remote_settlement_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path, field: str
) -> None:
    port, worker, writer, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path)
    await port.startup_async()
    import_async = writer.import_async

    async def foreign_receipt_async(
        *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        actual = await import_async(request=request, artifacts=artifacts)
        return actual.model_copy(update={field: uuid4() if field == "job_id" else "d" * 64})

    try:
        with patch.object(writer, "import_async", side_effect=foreign_receipt_async):
            request = worker.runtime.request()
            await port.submit_async(request=request, actor_id=ACTOR)
            port.start_consumer()
            terminal = await port.wait_async(job_id=request.job_id, actor_id=ACTOR)
        assert terminal.state is EvaluationJobState.FAILED and terminal.canonical is None
        assert terminal.reason == "writer_failed" and terminal.cleanup is EvaluationCleanupState.UNKNOWN
        assert writer.requests == [request] and not worker.settlements
        row = await asyncio.to_thread(port.journal.read, request.job_id)
        assert row["canonical_json"] is None and row["uncertain"] == 1
    finally:
        await port.shutdown_async()


async def test_remote_runtime_direct_call_cannot_dispatch_a_foreign_registration_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    port, worker, _, _ = await _native_gateway_async(memory=sqlite_instance, root=tmp_path)
    runtime = port.registry.resolve(worker.runtime.request())
    try:
        with patch.object(port.client, "submit_async", new_callable=AsyncMock) as submit:
            with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
                await runtime.execute_async(
                    context=_Context(
                        request=worker.runtime.request().model_copy(update={"case_id": "d" * 64}),
                        actor_id=ACTOR,
                        fence_id=uuid4(),
                        run_root=tmp_path,
                    )
                )
            submit.assert_not_awaited()
            assert not worker.requests
    finally:
        await port.client.close_async()


async def test_generic_factory_failed_installation_closes_transport_but_leaves_credentials_with_caller_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    credentials = _Credentials()
    with patch.object(
        EvaluationWorkerHttpClient,
        "close_transport_async",
        autospec=True,
        side_effect=EvaluationWorkerHttpClient.close_transport_async,
    ) as close_transport:
        with pytest.raises(ValueError, match="actor allowlist"):
            await create_remote_job_port_async(
                root=tmp_path,
                registration=_NativeRegistrationFixture().registration,
                writer=_NativeApiWriterFixture(sqlite_instance),
                allowed_actor_ids=frozenset(),
                settings=EvaluationWorkerHttpSettings(
                    base_url="http://127.0.0.1",
                    audience=credentials.audience,
                    service_id="public_fixture",
                    schema_sha256=evaluation_worker_schema_sha256(),
                    allow_loopback_http=True,
                ),
                credentials=credentials,
            )
        close_transport.assert_awaited_once()
        assert close_transport.call_args.args[0]._client.is_closed
    assert not credentials.closed and not credentials.contexts
    await credentials.close_async()
    assert credentials.close_calls == 1


async def test_original_backend_failed_installation_closes_caller_owned_credentials_once_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    credentials = _Credentials()
    settings = RemoteEvaluationJobSettings(
        root=tmp_path,
        allowed_actor_ids=frozenset(),
        identity_name="public_fixture",
        transport=EvaluationWorkerHttpSettings(
            base_url="http://127.0.0.1",
            audience=credentials.audience,
            service_id="public_fixture",
            schema_sha256=evaluation_worker_schema_sha256(),
            allow_loopback_http=True,
        ),
    )
    with patch.dict(
        EvaluationWorkerCredentialRegistry._factories, {"public_fixture": lambda _: credentials}, clear=True
    ):
        with pytest.raises(ValueError, match="actor allowlist"):
            await settings.create_port_async(memory=sqlite_instance)
    assert credentials.close_calls == 1 and not credentials.contexts


async def test_original_constructor_path_and_writer_name_remain_compatible_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    original = await PublicOriginalInspectJobRuntime.create_async()
    worker = _WorkerMock(runtime=original, root=tmp_path / "source")
    credentials = _Credentials()
    client = EvaluationWorkerHttpClient(
        settings=EvaluationWorkerHttpSettings(
            base_url="http://127.0.0.1",
            audience=credentials.audience,
            service_id="public_fixture",
            schema_sha256=evaluation_worker_schema_sha256(),
            allow_loopback_http=True,
        ),
        credentials=credentials,
        transport=httpx.MockTransport(worker.handle_async),
    )
    assert RemoteOriginalInspectArtifactWriter is RemoteEvaluationArtifactWriter
    runtime = RemoteEvaluationJobRuntime(original=original, client=client, journal=EvaluationRemoteJournal(tmp_path))
    assert runtime.registration == original.registration
    port = RemoteEvaluationJobGateway(
        root=tmp_path / "gateway",
        original=original,
        memory=sqlite_instance,
        allowed_actor_ids=frozenset({ACTOR}),
        client=client,
    )
    await port.startup_async()
    try:
        assert await port.catalog_async(actor_id=ACTOR) == (original.registration,)
        assert set(port._writers) == {EvaluationRuntimeKind.ORIGINAL_INSPECT}
        native = _NativeRegistrationFixture()
        worker.registrations = (original.registration, native.registration)
        assert await port.catalog_async(actor_id=ACTOR) == (original.registration,)
        with pytest.raises(EvaluationJobError, match="unsupported_runtime"):
            await port.submit_async(request=native.request(), actor_id=ACTOR)
        with pytest.raises(ValueError, match="combine original"):
            RemoteEvaluationJobRuntime(
                original=original, registration=native.registration, client=client, journal=runtime.journal
            )
        with pytest.raises(ValueError, match="only the approved original"):
            RemoteEvaluationJobGateway(
                root=tmp_path / "mixed",
                original=original,
                memory=sqlite_instance,
                registration=native.registration,
                writer=_NativeApiWriterFixture(sqlite_instance),
                allowed_actor_ids=frozenset({ACTOR}),
                client=client,
            )
        assert worker.executions == 0
    finally:
        await port.shutdown_async()


async def test_original_factory_delegates_with_same_source_and_canonical_writer_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    credentials = _Credentials()
    settings = EvaluationWorkerHttpSettings(
        base_url="http://127.0.0.1",
        audience=credentials.audience,
        service_id="public_fixture",
        schema_sha256=evaluation_worker_schema_sha256(),
        allow_loopback_http=True,
    )
    with patch("pyrit.executor.jobs.remote.create_remote_job_port_async", new_callable=AsyncMock) as factory:
        await create_remote_original_job_port_async(
            root=tmp_path,
            memory=sqlite_instance,
            allowed_actor_ids=frozenset({ACTOR}),
            settings=settings,
            credentials=credentials,
        )
        factory.assert_awaited_once()
        supplied = factory.call_args.kwargs
        original = await PublicOriginalInspectJobRuntime.create_async()
        assert supplied["registration"] == original.registration
        assert isinstance(supplied["writer"], OriginalInspectArtifactWriter)
        assert supplied["writer"]._memory is sqlite_instance
        assert supplied["writer"]._runtime.registration == original.registration
        assert supplied["credentials"] is credentials and supplied["settings"] is settings
        assert supplied["root"] == tmp_path and supplied["transport"] is None
    assert not credentials.contexts


def test_generic_factory_preserves_worker_v1_schema_fingerprint() -> None:
    assert evaluation_worker_schema_sha256() == "1bec6625595424c4b5d1305e8b592c177a2181324b0a0f7422ed2c0c9e0f2fdc"

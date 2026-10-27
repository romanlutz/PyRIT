# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Cancellation transport and join regressions using only DTOs, mock transport and a mock journal."""

from __future__ import annotations

import asyncio
import hashlib
import socket
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.executor.jobs.remote import RemoteEvaluationJobRuntime
from pyrit.executor.jobs.remote_ledger import EvaluationRemoteJournal
from pyrit.executor.jobs.worker_auth import EvaluationWorkerCredentialProvider, EvaluationWorkerCredentials
from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpClient, EvaluationWorkerHttpSettings
from pyrit.models import EvalPackageRef, EvalSourceKind
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactKind,
    EvaluationArtifactMediaType,
    EvaluationCleanupState,
    EvaluationJobEventKind,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationRuntimeKind,
)
from pyrit.models.evaluation_worker import (
    EvaluationGatewaySettlement,
    EvaluationGatewaySettlementDisposition,
    EvaluationGatewaySettlementReceipt,
    EvaluationWorkerAdmission,
    EvaluationWorkerBinding,
    EvaluationWorkerEvent,
    EvaluationWorkerEvidence,
    EvaluationWorkerSnapshot,
    EvaluationWorkerState,
    EvaluationWorkerTerminal,
    evaluation_worker_schema_sha256,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(name="forbid_database_and_network", autouse=True)
async def forbid_database_and_network_async() -> AsyncIterator[None]:
    with (
        patch.object(sqlite3, "connect", side_effect=AssertionError("No database in cancellation unit tests.")) as db,
        patch.object(
            socket.socket, "connect", side_effect=AssertionError("No network in cancellation unit tests.")
        ) as net,
        patch.object(
            socket.socket, "connect_ex", side_effect=AssertionError("No network in cancellation unit tests.")
        ) as net_ex,
    ):
        yield
        db.assert_not_called()
        net.assert_not_called()
        net_ex.assert_not_called()


class _CancellationTransport:
    """In-memory responses only; no worker, service, Task or canonical writer is constructed."""

    def __init__(self) -> None:
        self.registration = EvaluationJobRegistration(
            runtime=EvaluationRuntimeKind.ORIGINAL_INSPECT,
            source=EvalPackageRef(kind=EvalSourceKind.NAMED, name="cancel_transport_fixture", source_sha256="a" * 64),
            case_id="b" * 64,
            execution_profile_sha256="c" * 64,
            artifact_kind=EvaluationArtifactKind.INSPECT_EVAL,
        )
        request = EvaluationJobRequest(
            job_id=uuid4(),
            run_id=uuid4(),
            attempt_id=uuid4(),
            runtime=self.registration.runtime,
            source=self.registration.source,
            case_id=self.registration.case_id,
            execution_profile_sha256=self.registration.execution_profile_sha256,
        )
        self.admission = EvaluationWorkerAdmission(
            request=request, request_sha256=request.request_sha256, gateway_fence_id=uuid4(), actor_id="unit-actor"
        )
        self.binding = EvaluationWorkerBinding(
            service_id="cancel_fixture",
            worker_incarnation_id=uuid4(),
            worker_fence_id=uuid4(),
            gateway_fence_id=self.admission.gateway_fence_id,
            job_id=request.job_id,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            request_sha256=request.request_sha256,
            admission_sha256=self.admission.admission_sha256,
            actor_id=self.admission.actor_id,
        )
        self.cancel_body = self.snapshot().model_dump(mode="json")
        self.status_body = self.snapshot(sequence=2).model_dump(mode="json")
        self.receipt_changes: dict[str, str] = {}
        self.artifact_content = b"transport-only fixture, not an Inspect archive"
        self.delays = {"cancel": 0.0, "status": 0.0, "settlement": 0.0, "artifact": 0.0}
        self.errors: dict[str, Exception] = {}
        self.requests: list[httpx.Request] = []
        self.operations: list[str] = []
        self.settlements: list[EvaluationGatewaySettlement] = []

    def snapshot(
        self,
        *,
        state: EvaluationWorkerState = EvaluationWorkerState.CANCELLED,
        cleanup: EvaluationCleanupState = EvaluationCleanupState.VERIFIED,
        evidence: EvaluationWorkerEvidence = EvaluationWorkerEvidence.ABSENT,
        sequence: int = 1,
    ) -> EvaluationWorkerSnapshot:
        terminal = (
            EvaluationWorkerTerminal(
                binding=self.binding,
                state=state,
                cleanup=cleanup,
                evidence=evidence,
                last_sequence=sequence,
                manifest_sha256="d" * 64 if evidence is EvaluationWorkerEvidence.SOURCE_RETAINED else None,
            )
            if state.terminal
            else None
        )
        return EvaluationWorkerSnapshot(
            request=self.admission.request,
            request_sha256=self.admission.request_sha256,
            binding=self.binding,
            state=state,
            evidence=evidence,
            cleanup=cleanup,
            last_sequence=sequence,
            events=(
                EvaluationWorkerEvent(
                    sequence=sequence,
                    kind=(
                        EvaluationJobEventKind.TERMINAL if state.terminal else EvaluationJobEventKind.CANCEL_REQUESTED
                    ),
                    state=state,
                    occurred_at=datetime.now(UTC),
                ),
            ),
            terminal=terminal,
        )

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        operation = request.url.path.rsplit("/", 1)[-1] if request.method == "POST" else "status"
        if "/artifacts/" in request.url.path:
            operation = "artifact"
        assert operation in self.delays
        self.requests.append(request)
        self.operations.append(operation)
        if operation == "artifact":
            await asyncio.sleep(self.delays[operation])
            return httpx.Response(
                200,
                content=self.artifact_content,
                headers={"Content-Type": EvaluationArtifactMediaType.INSPECT_EVAL.value},
            )
        if operation == "settlement":
            settlement = EvaluationGatewaySettlement.model_validate_json(request.content)
            self.settlements.append(settlement)
            body = EvaluationGatewaySettlementReceipt(
                job_id=self.binding.job_id,
                binding_sha256=self.binding.binding_sha256,
                settlement_sha256=settlement.settlement_sha256,
            ).model_dump(mode="json")
            body.update(self.receipt_changes)
        else:
            body = self.cancel_body if operation == "cancel" else self.status_body
        await asyncio.sleep(self.delays[operation])
        if operation in self.errors:
            raise self.errors[operation]
        return httpx.Response(200, json=body)


@pytest.fixture
def worker() -> _CancellationTransport:
    return _CancellationTransport()


@pytest.fixture
def settings(request: pytest.FixtureRequest) -> EvaluationWorkerHttpSettings:
    defaults = EvaluationWorkerHttpSettings(
        base_url="http://127.0.0.1",
        audience="cancel-fixture",
        service_id="cancel_fixture",
        schema_sha256=evaluation_worker_schema_sha256(),
        allow_loopback_http=True,
        request_timeout_seconds=0.05,
        settlement_timeout_seconds=0.5,
        cancel_join_timeout_seconds=0.5,
        poll_interval_seconds=0.01,
    )
    return replace(defaults, **getattr(request, "param", {}))


@pytest.fixture(name="client")
async def client_async(
    *, worker: _CancellationTransport, settings: EvaluationWorkerHttpSettings
) -> AsyncIterator[EvaluationWorkerHttpClient]:
    credentials = MagicMock(spec=EvaluationWorkerCredentialProvider)
    credentials.audience = "cancel-fixture"
    credentials.fixture_only = True
    credentials.credentials_async.return_value = EvaluationWorkerCredentials(
        service_token="unit-service", operator_delegation="unit-delegation"
    )
    client = EvaluationWorkerHttpClient(
        settings=settings,
        credentials=credentials,
        transport=httpx.MockTransport(worker.handle_async),
    )
    try:
        yield client
    finally:
        await client.close_async()


@pytest.fixture
def journal() -> MagicMock:
    journal = MagicMock(spec=EvaluationRemoteJournal)
    journal.observe.side_effect = lambda snapshot: snapshot.events[-1].sequence
    return journal


@pytest.fixture
def runtime(
    *, worker: _CancellationTransport, client: EvaluationWorkerHttpClient, journal: MagicMock
) -> RemoteEvaluationJobRuntime:
    return RemoteEvaluationJobRuntime(registration=worker.registration, client=client, journal=journal)


async def test_blocking_cancel_uses_owned_join_budget_not_short_request_timeout_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock
) -> None:
    worker.delays["cancel"] = 0.1
    cleanup = await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding)
    assert cleanup is EvaluationCleanupState.VERIFIED
    assert worker.operations == ["cancel", "settlement"]
    assert worker.settlements[0].disposition is EvaluationGatewaySettlementDisposition.CLOSURE_OBSERVED
    assert worker.settlements[0].artifact_sha256 is None
    journal.settle.assert_called_once()
    journal.uncertain.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"request_timeout_seconds": 10, "cancel_join_timeout_seconds": 360, "settlement_timeout_seconds": 10},
        {"request_timeout_seconds": 10, "cancel_join_timeout_seconds": None, "settlement_timeout_seconds": 10},
        {"request_timeout_seconds": 10, "cancel_join_timeout_seconds": None, "settlement_timeout_seconds": 0.2},
        {"request_timeout_seconds": 10, "cancel_join_timeout_seconds": 0.2, "settlement_timeout_seconds": 0.5},
    ],
    indirect=True,
    ids=["explicit-360", "legacy-default", "legacy-short", "explicit-short"],
)
async def test_cancel_read_deadline_preserves_request_phase_and_ack_caps_async(
    *, worker: _CancellationTransport, client: EvaluationWorkerHttpClient
) -> None:
    await client.cancel_async(admission=worker.admission, binding=worker.binding)
    await client.status_async(admission=worker.admission, binding=worker.binding, after_sequence=1)
    await client.settle_async(
        admission=worker.admission,
        settlement=EvaluationGatewaySettlement(
            binding=worker.binding, disposition=EvaluationGatewaySettlementDisposition.CLOSURE_OBSERVED
        ),
    )
    settings = client.settings
    cancel_timeout = settings.cancel_join_timeout_seconds
    if cancel_timeout is None:
        cancel_timeout = settings.settlement_timeout_seconds
    expected_reads = (cancel_timeout, 10, settings.settlement_timeout_seconds)
    for request, read_timeout in zip(worker.requests, expected_reads, strict=True):
        assert request.extensions["timeout"] == {"connect": 10, "read": read_timeout, "write": 10, "pool": 10}
    assert client.settings.request_timeout_seconds == 10
    assert evaluation_worker_schema_sha256() == "1bec6625595424c4b5d1305e8b592c177a2181324b0a0f7422ed2c0c9e0f2fdc"


@pytest.mark.parametrize(
    "settings",
    [{"request_timeout_seconds": 10, "artifact_timeout_seconds": 30}],
    indirect=True,
)
async def test_artifact_read_uses_installed_budget_without_widening_other_limits_async(
    *, worker: _CancellationTransport, client: EvaluationWorkerHttpClient
) -> None:
    expected = worker.artifact_content
    artifact = EvaluationArtifact(
        name="fixture.eval",
        kind=EvaluationArtifactKind.INSPECT_EVAL,
        media_type=EvaluationArtifactMediaType.INSPECT_EVAL,
        bytes=len(expected),
        sha256=hashlib.sha256(expected).hexdigest(),
    )
    worker.delays["artifact"] = 10.1
    with patch.object(asyncio, "timeout", wraps=asyncio.timeout) as total_timeout:
        received = await client.artifact_async(
            admission=worker.admission,
            binding=worker.binding,
            manifest_sha256="d" * 64,
            artifact=artifact,
        )
    assert received == expected
    total_timeout.assert_called_once_with(30)
    worker.delays["artifact"] = 0
    worker.artifact_content = expected + b"extra"
    with pytest.raises(EvaluationJobError) as error:
        await client.artifact_async(
            admission=worker.admission,
            binding=worker.binding,
            manifest_sha256="d" * 64,
            artifact=artifact,
        )
    assert error.value.code is EvaluationJobErrorCode.ARTIFACT_MISMATCH
    assert worker.operations == ["artifact", "artifact"]
    for request in worker.requests:
        assert request.extensions["timeout"] == {"connect": 10, "read": 30, "write": 10, "pool": 10}


@pytest.mark.parametrize("phase", ["cancel", "status", "settlement"])
@pytest.mark.parametrize(
    "settings",
    [{"request_timeout_seconds": 0.5, "settlement_timeout_seconds": 0.5, "cancel_join_timeout_seconds": 0.24}],
    indirect=True,
)
async def test_one_deadline_covers_cancel_poll_and_settlement_without_retries_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock, phase: str
) -> None:
    if phase == "cancel":
        worker.delays["cancel"] = 0.3
    else:
        worker.cancel_body = worker.snapshot(
            state=EvaluationWorkerState.CANCEL_REQUESTED, cleanup=EvaluationCleanupState.UNKNOWN
        ).model_dump(mode="json")
        worker.delays.update(
            {"cancel": 0.15, "status": 0.15} if phase == "status" else dict.fromkeys(worker.delays, 0.09)
        )
    cleanup = await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding)
    assert cleanup is EvaluationCleanupState.UNKNOWN
    expected = ["cancel", "status", "settlement"]
    assert worker.operations == expected[: expected.index(phase) + 1]
    journal.settle.assert_not_called()
    journal.uncertain.assert_called_once()


async def test_terminal_snapshot_requires_complete_observed_cursor_before_settlement_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock
) -> None:
    terminal = worker.snapshot(sequence=2)
    worker.cancel_body = terminal.model_copy(
        update={
            "events": worker.snapshot(
                state=EvaluationWorkerState.CANCEL_REQUESTED, cleanup=EvaluationCleanupState.UNKNOWN
            ).events
        }
    ).model_dump(mode="json")
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.VERIFIED
    )
    assert worker.operations == ["cancel", "status", "settlement"]
    assert worker.requests[1].url.params["after_sequence"] == "1"
    assert journal.observe.call_count == 2
    journal.settle.assert_called_once()


@pytest.mark.parametrize(
    "field",
    [
        "service_id",
        "worker_incarnation_id",
        "worker_fence_id",
        "gateway_fence_id",
        "job_id",
        "run_id",
        "attempt_id",
        "request_sha256",
        "admission_sha256",
        "actor_id",
    ],
)
@pytest.mark.parametrize("operation", ["cancel", "status"])
async def test_foreign_cancel_binding_never_settles_async(
    *,
    worker: _CancellationTransport,
    runtime: RemoteEvaluationJobRuntime,
    journal: MagicMock,
    field: str,
    operation: str,
) -> None:
    body = worker.cancel_body
    if operation == "status":
        worker.cancel_body = worker.snapshot(
            state=EvaluationWorkerState.CANCEL_REQUESTED, cleanup=EvaluationCleanupState.UNKNOWN
        ).model_dump(mode="json")
        body = worker.status_body
    changed = str(uuid4()) if field.endswith("_id") else "e" * 64
    if field in {"actor_id", "service_id"}:
        changed = "foreign-fixture"
    body["binding"][field] = changed
    body["terminal"]["binding"][field] = changed
    cleanup = await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding)
    assert cleanup is EvaluationCleanupState.UNKNOWN
    assert worker.operations == (["cancel"] if operation == "cancel" else ["cancel", "status"])
    journal.settlement_intent.assert_not_called()
    journal.uncertain.assert_called_once()


@pytest.mark.parametrize(
    ("cleanup", "evidence"),
    [
        (EvaluationCleanupState.UNKNOWN, EvaluationWorkerEvidence.ABSENT),
        (EvaluationCleanupState.FAILED, EvaluationWorkerEvidence.ABSENT),
        (EvaluationCleanupState.NOT_STARTED, EvaluationWorkerEvidence.ABSENT),
        (EvaluationCleanupState.VERIFIED, EvaluationWorkerEvidence.UNKNOWN),
        (EvaluationCleanupState.VERIFIED, EvaluationWorkerEvidence.SOURCE_RETAINED),
    ],
)
async def test_incomplete_closure_or_retained_evidence_does_not_become_verified_async(
    *,
    worker: _CancellationTransport,
    runtime: RemoteEvaluationJobRuntime,
    journal: MagicMock,
    cleanup: EvaluationCleanupState,
    evidence: EvaluationWorkerEvidence,
) -> None:
    worker.cancel_body = worker.snapshot(cleanup=cleanup, evidence=evidence).model_dump(mode="json")
    observed = await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding)
    assert observed is EvaluationCleanupState.UNKNOWN
    assert worker.operations == ["cancel"]
    journal.settlement_intent.assert_not_called()
    journal.uncertain.assert_called_once()


@pytest.mark.parametrize("mode", ["missing-terminal", "foreign-environment", "exit-zero"])
async def test_incomplete_or_extra_terminal_claims_never_prove_cleanup_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock, mode: str
) -> None:
    if mode == "missing-terminal":
        worker.cancel_body.pop("terminal")
    else:
        worker.cancel_body["environment_id" if mode == "foreign-environment" else "exit_code"] = (
            "unbound-environment" if mode == "foreign-environment" else 0
        )
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.UNKNOWN
    )
    assert worker.operations == ["cancel"]
    journal.settlement_intent.assert_not_called()
    journal.uncertain.assert_called_once()


@pytest.mark.parametrize("settings", [{"cancel_join_timeout_seconds": 0.05}], indirect=True)
async def test_cancel_acceptance_without_terminal_observation_remains_unknown_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock
) -> None:
    worker.cancel_body = worker.snapshot(
        state=EvaluationWorkerState.CANCEL_REQUESTED, cleanup=EvaluationCleanupState.UNKNOWN
    ).model_dump(mode="json")
    worker.status_body = worker.cancel_body
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.UNKNOWN
    )
    assert worker.operations.count("cancel") == 1
    assert "status" in worker.operations and "settlement" not in worker.operations
    journal.uncertain.assert_called_once()


async def test_completed_source_is_not_settled_as_cancellation_closure_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock
) -> None:
    worker.cancel_body = worker.snapshot(
        state=EvaluationWorkerState.COMPLETED, evidence=EvaluationWorkerEvidence.SOURCE_RETAINED
    ).model_dump(mode="json")
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.UNKNOWN
    )
    assert worker.operations == ["cancel"]
    journal.settlement_intent.assert_not_called()
    journal.uncertain.assert_called_once()


@pytest.mark.parametrize("operation", ["cancel", "status", "settlement"])
async def test_transport_uncertainty_never_retries_cancel_or_claims_settlement_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock, operation: str
) -> None:
    if operation == "status":
        worker.cancel_body = worker.snapshot(
            state=EvaluationWorkerState.CANCEL_REQUESTED, cleanup=EvaluationCleanupState.UNKNOWN
        ).model_dump(mode="json")
    worker.errors[operation] = httpx.ReadTimeout("unit transport uncertainty")
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.UNKNOWN
    )
    assert worker.operations.count("cancel") == 1
    journal.settle.assert_not_called()
    journal.uncertain.assert_called_once()


@pytest.mark.parametrize("field", ["job_id", "binding_sha256", "settlement_sha256"])
async def test_foreign_settlement_receipt_never_verifies_closure_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock, field: str
) -> None:
    worker.receipt_changes[field] = str(uuid4()) if field == "job_id" else "e" * 64
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.UNKNOWN
    )
    assert worker.operations == ["cancel", "settlement"]
    journal.settle.assert_not_called()
    journal.uncertain.assert_called_once()


@pytest.mark.parametrize("method", ["bind", "observe", "settlement_intent", "settle"])
async def test_journal_refusal_keeps_cleanup_unknown_async(
    *, worker: _CancellationTransport, runtime: RemoteEvaluationJobRuntime, journal: MagicMock, method: str
) -> None:
    getattr(journal, method).side_effect = EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
    assert await runtime._cancel_join_async(admission=worker.admission, binding=worker.binding) is (
        EvaluationCleanupState.UNKNOWN
    )
    journal.uncertain.assert_called_once()
    assert worker.operations.count("cancel") == 1

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Exact binary intake and source-authorized viewing using only harmless public typed evidence."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from inspect_ai.event import ScoreEvent
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalSample,
    EvalSpec,
    EvalStats,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.scorer import Score as InspectScore
from inspect_ai.tool import ToolCall
from sqlalchemy import func, select

from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.routes import attacks, original_evidence, scenarios
from pyrit.backend.services import original_evidence_admission as admission_module
from pyrit.backend.services.attack_service import AttackService, AttackSourceImmutableError, get_attack_service
from pyrit.backend.services.original_evidence_admission import OriginalEvidenceAdmission, OriginalEvidenceEnvelope
from pyrit.backend.services.original_evidence_service import (
    OriginalEvidenceRecord,
    OriginalEvidenceService,
    get_original_evidence_service,
)
from pyrit.backend.services.original_run_admission import OriginalAdmissionError, OriginalCleanupReceipt
from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter, InspectOriginalScorePolicy
from pyrit.memory.memory_models import AttackResultEntry, ScoreEntry
from pyrit.models import (
    EvalCaseRef,
    EvalPackageRef,
    EvalRunRef,
    EvalSourceKind,
    EvalSpecRef,
    HarnessProfileRef,
    ModelRouteRef,
    ScenarioRunState,
    ScoreStatus,
    config_hash,
)
from pyrit.models.catalog.scenario import OriginalRunReason, OriginalRunStatus

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.responses import Response

    from pyrit.backend.services.original_run_admission import OriginalWorkerJob
    from pyrit.memory import SQLiteMemory


_OPERATOR = AuthenticatedUser(oid="public-fixture-operator", name="Operator", email="viewer@example.test", groups=[])
_CAPABILITY = "public-fixture-intake-capability-" + "a" * 32


@pytest.fixture
def admitted_evidence(tmp_path: Path) -> tuple[OriginalEvidenceAdmission, bytes]:
    """Retain typed tool data without executing a tool, model or authored Task."""
    created = datetime.now(UTC).isoformat()
    score = InspectScore(value=1.0)
    event = ScoreEvent(uuid=str(uuid.uuid4()), scorer="public_recorded_scorer", score=score)
    sample = EvalSample(
        id="public-recorded-case",
        epoch=1,
        input="Read only the retained harmless values.",
        target="retained",
        uuid=str(uuid.uuid4()),
        started_at=created,
        completed_at=created,
        messages=[
            ChatMessageSystem(id="original-system", content="Harmless recorded source fixture."),
            ChatMessageUser(id="original-user", content="Read the original records."),
            ChatMessageAssistant(
                id="original-assistant",
                content="",
                tool_calls=[
                    ToolCall(id="recorded-one", function="echo", arguments={"value": "one"}),
                    ToolCall(id="recorded-two", function="echo", arguments={"value": "two"}),
                ],
            ),
            ChatMessageTool(id="reply-one", tool_call_id="recorded-one", function="echo", content="one"),
            ChatMessageTool(id="reply-two", tool_call_id="recorded-two", function="echo", content="two"),
        ],
        events=[event],
        scores={"public_recorded_scorer": score},
        model_usage={},
    )
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            eval_id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
            created=created,
            task="public_recorded_task",
            task_version=1,
            dataset=EvalDataset(samples=1, sample_ids=[str(sample.id)]),
            model="mockllm/model",
            config=EvalConfig(),
        ),
        samples=[sample],
        stats=EvalStats(),
    )
    archive_path = tmp_path / "public-recorded.eval"
    write_eval_log(log, str(archive_path))
    archive = archive_path.read_bytes()
    package = EvalPackageRef(kind=EvalSourceKind.NAMED, name="public_recorded", source_sha256="a" * 64)
    case = EvalCaseRef(package=package, task_name=log.eval.task, task_version="1", sample_id=str(sample.id), epoch=1)
    run = EvalRunRef(
        spec=EvalSpecRef(
            package=package,
            harness=HarnessProfileRef(name="public-fixture", config_sha256="b" * 64),
            model_route=ModelRouteRef(name="recorded-role", config_sha256="c" * 64),
        ),
        run_instance_id=uuid.uuid4(),
    )
    envelope = OriginalEvidenceEnvelope(
        app_run_id=uuid.uuid4(),
        job_ref=uuid.uuid4(),
        profile_ref="public-fixture",
        operator_oid=_OPERATOR.oid,
        run_instance_id=run.run_instance_id,
        worker_scenario_id=uuid.uuid4(),
        worker_scenario_sha256="d" * 64,
        manifest_sha256="e" * 64,
        source_sha256=package.source_sha256,
        case_run_id=run.case_run_id(case=case),
        model_role="evaluated",
        model_route_sha256=run.spec.model_route.config_sha256,
        model_role_receipt_id="public-fixture-role",
        model_role_sha256="f" * 64,
        source_state="success",
        source_complete=True,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        archive_bytes=len(archive),
        inspect_run_id=log.eval.run_id,
        inspect_eval_id=log.eval.eval_id,
        final_score_event_id=event.uuid,
        final_score_event_sha256=config_hash({"event": event.model_dump(mode="json", exclude_none=True)}),
        operation_terminal_receipt_id="public-fixture-terminal",
        operation_terminal_sha256="1" * 64,
        cleanup=OriginalCleanupReceipt(state="proved", receipt_id="public-fixture-physical"),
        cleanup_sha256="2" * 64,
    )
    return (
        OriginalEvidenceAdmission(
            envelope=envelope,
            run=run,
            cases=(case,),
            score_policy=InspectOriginalScorePolicy(
                task_name=case.task_name, task_version=case.task_version, primary_scorer="public_recorded_scorer"
            ),
            display_values=frozenset({"1.0"}),
        ),
        archive,
    )


class _RecordedEvidenceProvider:
    """Authenticate one fixed harmless fixture; this does not qualify a live worker."""

    def __init__(self, admission: OriginalEvidenceAdmission) -> None:
        self.admission = admission
        self.closure = admission.envelope.cleanup
        self.read_allowed = True

    async def authorize_intake_async(
        self, *, capability: str, envelope: OriginalEvidenceEnvelope
    ) -> OriginalEvidenceAdmission:
        """Accept the fixed fixture's exact replay, not caller-selected source/policy."""
        if capability != _CAPABILITY or envelope != self.admission.envelope:
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        return self.admission

    async def verify_cleanup_async(
        self, *, job: OriginalWorkerJob, envelope_sha256: str
    ) -> OriginalCleanupReceipt | None:
        """Observe the separate fixed fixture receipt without running cleanup."""
        if job != self.admission.envelope.job or envelope_sha256 != self.admission.envelope.sha256:
            return None
        return self.closure

    def resolve_read(
        self, *, operator: AuthenticatedUser, job: OriginalWorkerJob, envelope_sha256: str
    ) -> OriginalEvidenceAdmission | None:
        """Authorize only the fixed actor and exact source envelope."""
        if (
            not self.read_allowed
            or operator.oid != _OPERATOR.oid
            or job != self.admission.envelope.job
            or envelope_sha256 != self.admission.envelope.sha256
        ):
            return None
        return self.admission


@pytest.fixture
def evidence_service(sqlite_instance: SQLiteMemory, patch_central_database: None) -> Iterator[OriginalEvidenceService]:
    get_original_evidence_service.cache_clear()
    get_attack_service.cache_clear()
    with patch("pyrit.backend.services.scenario_run_service._service_instance", ScenarioRunService()):
        yield get_original_evidence_service()
    get_original_evidence_service.cache_clear()
    get_attack_service.cache_clear()


def _fixture_app(*, operator: AuthenticatedUser | None = _OPERATOR) -> FastAPI:
    app = FastAPI()
    app.include_router(original_evidence.router, prefix="/api")
    app.include_router(attacks.router, prefix="/api")
    app.include_router(scenarios.router, prefix="/api")

    @app.middleware("http")
    async def fixture_identity_async(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.user = operator
        return await call_next(request)

    return app


def _upload_headers(envelope: OriginalEvidenceEnvelope) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_CAPABILITY}",
        "Content-Type": "application/octet-stream",
        "X-PyRIT-Original-Envelope": base64.urlsafe_b64encode(
            envelope.model_dump_json(exclude_none=True).encode()
        ).decode(),
    }


@pytest.mark.usefixtures("patch_central_database")
async def test_binary_service_canonical_round_trip(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
) -> None:
    admission, archive = admitted_evidence
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        receipt = await evidence_service.intake_async(
            capability=_CAPABILITY, envelope=admission.envelope, archive=archive
        )
        assert receipt.scenario_state is ScenarioRunState.COMPLETED
        assert receipt.score_id is not None and receipt.attack_result_id is not None


@pytest.mark.usefixtures("patch_central_database")
async def test_binary_intake_actual_http_sqlite_source_readback_and_read_only_view(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
) -> None:
    admission, archive = admitted_evidence
    provider = _RecordedEvidenceProvider(admission)
    with patch.object(admission_module, "_provider", provider):
        async with AsyncClient(transport=ASGITransport(app=_fixture_app()), base_url="http://test") as client:
            upload = await client.post(
                f"/api/internal/original-evidence/{admission.envelope.job_ref}",
                headers=_upload_headers(admission.envelope),
                content=archive,
            )
            assert upload.status_code == 200, upload.text
            receipt = upload.json()
            assert receipt["persistence_verified"] is True
            assert receipt["source_result"]["original_score"] == "1.0"
            assert receipt["source_result"]["pyrit_outcome"] == "undetermined"
            result_id = receipt["attack_result_id"]
            result = await client.get(f"/api/attacks/{result_id}")
            assert result.status_code == 200, result.text
            assert result.json()["source_read_only"] is True
            assert "public_recorded_task" not in result.text
            assert "public_recorded_scorer" not in result.text
            assert "public-recorded-case" not in result.text
            assert result.json()["last_response"] is None
            conversation_id = result.json()["conversation_id"]
            messages = await client.get(
                f"/api/attacks/{result_id}/messages", params={"conversation_id": conversation_id}
            )
            assert messages.status_code == 200, messages.text
            pieces = [piece for message in messages.json()["messages"] for piece in message["message_pieces"]]
            assert len(pieces) == 6
            assert sum(piece["converted_value_data_type"] == "function_call" for piece in pieces) == 2
            assert sum(piece["converted_value_data_type"] == "function_call_output" for piece in pieces) == 2
            assert all("inspect_sample_id" not in piece["prompt_metadata"] for piece in pieces)
            progress = await client.get(f"/api/scenarios/runs/{admission.envelope.app_run_id}/progress")
            assert progress.status_code == 200, progress.text
            assert progress.json()["results"][0]["attack_result_id"] == result_id
            assert progress.json()["summary"]["overall"]["success_percentage"] is None
            edited = await client.patch(f"/api/attacks/{result_id}", json={"outcome": "success"})
            assert edited.status_code == 409
            replay = await client.post(
                f"/api/internal/original-evidence/{admission.envelope.job_ref}",
                headers=_upload_headers(admission.envelope),
                content=archive,
            )
            assert replay.status_code == 200 and replay.json() == receipt
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count()).select_from(ScoreEntry)) == 1
        assert session.scalar(select(func.count()).select_from(AttackResultEntry)) == 1


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("cleanup_proved", [True, False])
async def test_startup_failure_without_archive_keeps_physical_cleanup_distinct_and_has_no_grade(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
    cleanup_proved: bool,
) -> None:
    admission, _ = admitted_evidence
    envelope = admission.envelope.model_copy(
        update={
            "source_state": "error",
            "source_complete": False,
            "archive_sha256": None,
            "archive_bytes": None,
            "inspect_run_id": None,
            "inspect_eval_id": None,
            "final_score_event_id": None,
            "final_score_event_sha256": None,
            "operation_terminal_receipt_id": None,
            "operation_terminal_sha256": None,
            "cleanup": OriginalCleanupReceipt(
                state="proved" if cleanup_proved else "uncontained", receipt_id="public-fixture-physical"
            ),
            "cleanup_sha256": "2" * 64 if cleanup_proved else None,
        }
    )
    admission = replace(admission, envelope=envelope)
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        receipt = await evidence_service.intake_async(capability=_CAPABILITY, envelope=envelope, archive=b"")
        assert receipt.scenario_state is ScenarioRunState.FAILED
        assert receipt.score_id is None and receipt.attack_result_id is None
        assert receipt.source_result.status is (
            OriginalRunStatus.FAILED_UNGRADED if cleanup_proved else OriginalRunStatus.CLEANUP_UNCERTAIN
        )
        assert receipt.source_result.source_coverage_complete is False
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count()).select_from(ScoreEntry)) == 0
        assert session.scalar(select(func.count()).select_from(AttackResultEntry)) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_score_archive_and_physical_cleanup_without_terminal_operation_remain_ungraded(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
) -> None:
    admission, archive = admitted_evidence
    envelope = admission.envelope.model_copy(
        update={
            "operation_terminal_receipt_id": None,
            "operation_terminal_sha256": None,
        }
    )
    admission = replace(admission, envelope=envelope)
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        receipt = await evidence_service.intake_async(capability=_CAPABILITY, envelope=envelope, archive=archive)
        assert receipt.source_result.status is OriginalRunStatus.FAILED_UNGRADED
        assert receipt.source_result.cleanup_state == "proved"
        assert receipt.source_result.original_score is None
        assert receipt.source_result.original_score_available is False
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count()).select_from(ScoreEntry)) == 0
        assert session.scalar(select(func.count()).select_from(AttackResultEntry)) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_read_acl_and_independent_source_reference_survive_removed_attack_metadata(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
) -> None:
    admission, archive = admitted_evidence
    provider = _RecordedEvidenceProvider(admission)
    with patch.object(admission_module, "_provider", provider):
        receipt = await evidence_service.intake_async(
            capability=_CAPABILITY, envelope=admission.envelope, archive=archive
        )
        assert receipt.attack_result_id is not None
        result_id = str(receipt.attack_result_id)
        sqlite_instance.update_attack_result_by_id(attack_result_id=result_id, update_fields={"attack_metadata": {}})
        service = AttackService(memory=sqlite_instance)
        with pytest.raises(AttackSourceImmutableError):
            await service.update_attack_async(
                attack_result_id=result_id,
                request=attacks.UpdateAttackRequest(outcome="success"),
            )
        with pytest.raises(OriginalAdmissionError) as denied:
            await service.get_attack_async(attack_result_id=result_id)
        assert denied.value.reason is OriginalRunReason.OPERATOR_NOT_AUTHORIZED
        provider.read_allowed = False
        with pytest.raises(OriginalAdmissionError):
            await service.get_attack_async(attack_result_id=result_id, authenticated_user=_OPERATOR)


@pytest.mark.usefixtures("patch_central_database")
async def test_intake_without_provider_is_unavailable(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes], evidence_service: OriginalEvidenceService
) -> None:
    admission, archive = admitted_evidence
    with patch.object(admission_module, "_provider", None):
        async with AsyncClient(transport=ASGITransport(app=_fixture_app()), base_url="http://test") as client:
            response = await client.post(
                f"/api/internal/original-evidence/{admission.envelope.job_ref}",
                headers=_upload_headers(admission.envelope),
                content=archive,
            )
        assert response.status_code == 503 and response.json()["detail"] == "runner_not_configured"


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("field", ["operator_oid", "profile_ref", "source_sha256", "model_route_sha256", "job_ref"])
async def test_worker_cannot_select_another_admitted_identity(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
    field: str,
) -> None:
    admission, archive = admitted_evidence
    replacement = (
        uuid.uuid4()
        if field == "job_ref"
        else "unapproved"
        if field.endswith("_ref") or field == "operator_oid"
        else "9" * 64
    )
    changed = admission.envelope.model_copy(update={field: replacement})
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        with pytest.raises(OriginalAdmissionError) as denied:
            await evidence_service.intake_async(capability=_CAPABILITY, envelope=changed, archive=archive)
    assert denied.value.reason is OriginalRunReason.OPERATOR_NOT_AUTHORIZED
    assert sqlite_instance.get_attack_results() == []


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("fault", ["bytes", "length", "score_event", "epoch", "task", "cleanup"])
async def test_authenticated_envelope_still_requires_exact_original_bytes_and_independent_receipts(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
    fault: str,
) -> None:
    admission, archive = admitted_evidence
    if fault == "bytes":
        archive += b"changed"
    elif fault == "length":
        admission = replace(
            admission, envelope=admission.envelope.model_copy(update={"archive_bytes": len(archive) + 1})
        )
    elif fault == "score_event":
        admission = replace(
            admission, envelope=admission.envelope.model_copy(update={"final_score_event_sha256": "9" * 64})
        )
    elif fault in ("epoch", "task"):
        case = admission.cases[0].model_copy(update={"epoch": 2} if fault == "epoch" else {"task_name": "unapproved"})
        admission = replace(
            admission,
            cases=(case,),
            envelope=admission.envelope.model_copy(update={"case_run_id": admission.run.case_run_id(case=case)}),
            score_policy=replace(admission.score_policy, task_name=case.task_name),
        )
    provider = _RecordedEvidenceProvider(admission)
    if fault == "cleanup":
        provider.closure = OriginalCleanupReceipt(state="uncontained", receipt_id="public-fixture-unknown")
    with patch.object(admission_module, "_provider", provider):
        with pytest.raises(OriginalAdmissionError) as denied:
            await evidence_service.intake_async(capability=_CAPABILITY, envelope=admission.envelope, archive=archive)
    assert denied.value.reason is OriginalRunReason.SOURCE_UNVERIFIED
    assert sqlite_instance.get_attack_results() == []
    assert sqlite_instance.get_scenario_result_header(scenario_result_id=str(admission.envelope.app_run_id)) is None


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("value", "expected_status"),
    [
        ("C", ScoreStatus.UNDETERMINED),
        ({"recorded": 1.0}, ScoreStatus.UNDETERMINED),
        (True, ScoreStatus.COMPLETE),
        (1, ScoreStatus.COMPLETE),
    ],
)
async def test_unapproved_display_values_remain_original_and_never_imply_attack_success(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
    tmp_path: Path,
    value: str | bool | int | dict[str, float],
    expected_status: ScoreStatus,
) -> None:
    admission, archive = admitted_evidence
    log = await asyncio.to_thread(read_eval_log, io.BytesIO(archive), format="eval", resolve_attachments="full")
    assert log.samples and isinstance(log.samples[0].events[0], ScoreEvent)
    score = InspectScore(value=value)
    event = log.samples[0].events[0].model_copy(update={"score": score})
    sample = log.samples[0].model_copy(
        update={"scores": {admission.score_policy.primary_scorer: score}, "events": [event]}
    )
    log = log.model_copy(update={"samples": [sample]})
    rewritten = tmp_path / "public-scalar.eval"
    await asyncio.to_thread(write_eval_log, log, str(rewritten))
    archive = await asyncio.to_thread(rewritten.read_bytes)
    envelope = admission.envelope.model_copy(
        update={
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "archive_bytes": len(archive),
            "final_score_event_sha256": config_hash({"event": event.model_dump(mode="json", exclude_none=True)}),
        }
    )
    admission = replace(admission, envelope=envelope)
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        receipt = await evidence_service.intake_async(capability=_CAPABILITY, envelope=envelope, archive=archive)
        assert receipt.source_result.original_score_available is True
        assert receipt.source_result.original_score is None
        assert receipt.source_result.pyrit_score_status is expected_status
        assert receipt.source_result.pyrit_outcome is not None
        assert receipt.source_result.pyrit_outcome.value == "undetermined"
        result = await AttackService(memory=sqlite_instance).get_attack_async(
            attack_result_id=str(receipt.attack_result_id), authenticated_user=_OPERATOR
        )
        assert result is not None and result.automated_score is None and result.source_read_only is True
        stored = sqlite_instance.get_scenario_result_header(scenario_result_id=str(envelope.app_run_id))
        assert stored is not None
        readback = evidence_service.read(scenario_result=stored, operator=_OPERATOR)
        assert readback.imported is not None
        stream = next(
            stream
            for stream in readback.imported.episode.raw_streams
            if stream.key == InspectOriginalEvalImporter.ARCHIVE_KEY
        )
        assert envelope.archive_bytes is not None and envelope.archive_sha256 is not None
        retained = await asyncio.to_thread(
            ScenarioRunService._read_original_inspect_stream,
            memory=sqlite_instance,
            run_id=readback.imported.episode.run.run_id,
            stream_id=stream.stream_id,
            expected_bytes=envelope.archive_bytes,
            expected_sha256=envelope.archive_sha256,
        )
        retained_log = await asyncio.to_thread(
            read_eval_log, io.BytesIO(retained), format="eval", resolve_attachments="full"
        )
        assert retained_log.samples and retained_log.samples[0].scores
        assert retained_log.samples[0].scores[admission.score_policy.primary_scorer].value == value


@pytest.mark.usefixtures("patch_central_database")
async def test_provisional_import_is_not_published_when_readback_fails(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
) -> None:
    admission, archive = admitted_evidence
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        with patch.object(evidence_service, "_verify_record", side_effect=ValueError("public readback failure")):
            with pytest.raises(OriginalAdmissionError):
                await evidence_service.intake_async(
                    capability=_CAPABILITY, envelope=admission.envelope, archive=archive
                )
        header = sqlite_instance.get_scenario_result_header(scenario_result_id=str(admission.envelope.app_run_id))
        assert header is not None and header.scenario_run_state is ScenarioRunState.IN_PROGRESS
        assert header.metadata[OriginalEvidenceRecord.METADATA_KEY]["persistence_verified"] is False
        with pytest.raises(OriginalAdmissionError):
            evidence_service.read(scenario_result=header, operator=_OPERATOR)


@pytest.mark.usefixtures("patch_central_database")
async def test_persisted_archive_tampering_blocks_existing_browser_routes(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
) -> None:
    from pyrit.memory.memory_models import NativeCyberRawChunkEntry

    admission, archive = admitted_evidence
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        receipt = await evidence_service.intake_async(
            capability=_CAPABILITY, envelope=admission.envelope, archive=archive
        )
        with sqlite_instance.get_session() as session:
            chunk = session.scalars(select(NativeCyberRawChunkEntry)).first()
            assert chunk is not None
            chunk.data = b"changed"
            session.commit()
        async with AsyncClient(transport=ASGITransport(app=_fixture_app()), base_url="http://test") as client:
            response = await client.get(f"/api/attacks/{receipt.attack_result_id}")
        assert response.status_code == 403
        assert response.json()["detail"] == "source_unverified"


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("fault", "expected"),
    [("capability", 401), ("job", 400), ("media_type", 415), ("encoding", 400), ("length", 400), ("envelope", 400)],
)
async def test_http_intake_rejects_unqualified_requests_before_persistence(
    admitted_evidence: tuple[OriginalEvidenceAdmission, bytes],
    evidence_service: OriginalEvidenceService,
    sqlite_instance: SQLiteMemory,
    fault: str,
    expected: int,
) -> None:
    admission, archive = admitted_evidence
    headers = _upload_headers(admission.envelope)
    job = admission.envelope.job_ref
    if fault == "capability":
        headers["Authorization"] = "Bearer wrong"
    elif fault == "job":
        job = uuid.uuid4()
    elif fault == "media_type":
        headers["Content-Type"] = "application/json"
    elif fault == "encoding":
        headers["Content-Encoding"] = "gzip"
    elif fault == "length":
        headers["Content-Length"] = str(len(archive) - 1)
    elif fault == "envelope":
        headers["X-PyRIT-Original-Envelope"] = "not-an-envelope"
    with patch.object(admission_module, "_provider", _RecordedEvidenceProvider(admission)):
        async with AsyncClient(transport=ASGITransport(app=_fixture_app()), base_url="http://test") as client:
            response = await client.post(f"/api/internal/original-evidence/{job}", headers=headers, content=archive)
    assert response.status_code == expected
    assert sqlite_instance.get_attack_results() == []
    assert sqlite_instance.get_scenario_result_header(scenario_result_id=str(admission.envelope.app_run_id)) is None

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Case-owned original scorer provenance from inert, real SQLite evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import func, select

from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_ghcp_case_executor import InspectGhcpCaseExecutor, InspectGhcpPilotEnvironment
from pyrit.executor.benchmark.inspect_ghcp_eval import InspectGhcpEvaluation, InspectGhcpOutcome, InspectGhcpTaskBinding
from pyrit.executor.benchmark.inspect_ghcp_protocol import InspectGhcpProtocolPins
from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.memory.memory_models import ScoreEntry
from pyrit.models import (
    AttackOutcome,
    EvalRunRef,
    EvalScoreProvenance,
    EvalScoreRole,
    EvalSpecRef,
    MessagePiece,
)
from pyrit.models.inspect_ghcp import InspectGhcpJudgment, InspectGhcpReport, InspectGhcpStatus, InspectGhcpTaskKind
from pyrit.prompt_target.inspect_ghcp_target import InspectGhcpTurn
from pyrit.score.float_scale.inspect_ghcp_report_scorer import InspectGhcpReportScorer
from tests.unit.executor.benchmark.test_inspect_eval_source import _multi_manifest
from tests.unit.memory.test_inspect_ghcp_evidence import _adversarial_model, _control_receipts, _gateway_record

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory


def _selected_case() -> tuple[InspectGhcpCaseExecutor, EvalRunRef]:
    environment = InspectGhcpPilotEnvironment(
        agent_image="pyrit-inspect-ghcp-guest:1.0.88-sdk1.0.14",
        agent_image_id="sha256:" + "a" * 64,
        target_image="pyrit-ghcp-agent:1.0.88-ca",
        target_image_id=InspectGhcpProtocolPins.TARGET_IMAGE_ID,
    )
    selected = EvalSourceFactory.resolve(
        family="benign_protocol",
        trusted_dir=None,
        trusted_local=False,
        revision_sha256=None,
        input_override=None,
        agent_image=environment.agent_image,
        target_image=environment.target_image,
        approved_image_ids=environment.image_ids,
    )
    spec = EvalSpecRef(
        package=selected.case.package,
        harness=environment.harness_ref(sandbox_sha256=selected.sandbox_sha256),
        model_route=environment.model_route_ref(),
    )
    return InspectGhcpCaseExecutor(selected=selected, environment=environment), EvalRunRef(
        spec=spec, run_instance_id=uuid.uuid4()
    )


@pytest.mark.usefixtures("patch_central_database")
async def test_multi_case_source_cannot_launch_unqualified_ghcp_runtime(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    original, _ = await asyncio.to_thread(_selected_case)
    environment = original._environment
    revision = await asyncio.to_thread(_multi_manifest, root=tmp_path)
    cases = await asyncio.to_thread(
        EvalSourceFactory.resolve_cases,
        family=None,
        trusted_dir=tmp_path,
        trusted_local=True,
        revision_sha256=revision,
        input_override=None,
        agent_image=environment.agent_image,
        target_image=environment.target_image,
        approved_image_ids=environment.image_ids,
    )
    selected = cases[2]
    executor = InspectGhcpCaseExecutor(selected=selected, environment=environment)
    run = EvalRunRef(
        spec=EvalSpecRef(
            package=selected.case.package,
            harness=environment.harness_ref(sandbox_sha256=selected.sandbox_sha256),
            model_route=environment.model_route_ref(),
        ),
        run_instance_id=uuid.uuid4(),
    )
    with patch.object(InspectGhcpPilotEnvironment, "verify_host_async", new_callable=AsyncMock) as verify:
        with pytest.raises(ValueError, match="inert until its Task profile is qualified"):
            await executor.execute_case_async(case=selected.case, run=run)
        verify.assert_not_awaited()
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0


def _source_turn(
    *,
    memory: SQLiteMemory,
    index: int,
    conversation_id: str,
    exchange: dict[str, object],
    events: tuple[dict[str, object], ...],
) -> InspectGhcpTurn:
    request = MessagePiece(
        role="user",
        original_value=f"harmless instruction {index}",
        conversation_id=conversation_id,
        sequence=2 * index - 2,
    )
    answer = f"harmless answer {index}"
    response = MessagePiece(
        role="assistant", original_value=answer, conversation_id=conversation_id, sequence=2 * index - 1
    )
    memory.add_message_pieces_to_memory(message_pieces=[request, response])
    return InspectGhcpTurn(
        turn_index=index,
        session_id="sdk-session",
        identity={"worker_pid": 200, "cli_pid": 201, "uid": 10001, "net_namespace": "net:[1]"},
        instruction=request.original_value,
        assistant_text=answer,
        request_piece_id=request.id,
        response_piece_id=response.id,
        events=events,
        model_exchanges=(exchange,),
    )


def _persist_original_case(
    *, memory: SQLiteMemory, executor: InspectGhcpCaseExecutor, run: EvalRunRef
) -> InspectGhcpOutcome:
    selected = executor._selected
    case = selected.case
    run_id = str(uuid.uuid4())
    started = datetime.now(UTC)
    conversation_id = str(uuid.uuid4())
    store = InspectGhcpEvidenceStore(
        memory=memory,
        run_id=run_id,
        task_name=case.task_name,
        task_version=case.task_version,
        started_at=started,
        raw_byte_limit=150_000,
    )
    store.bind_agent(container_id="a" * 64, session_id="sdk-session")
    first_gateway, first_exchange = _gateway_record(run_id=run_id)
    next_gateway = {**first_gateway, "sequence": 2, "source_request_id": "req-2"}
    next_exchange = {**first_exchange, "request_id": "req-2"}
    turns = [
        _source_turn(
            memory=memory,
            index=1,
            conversation_id=conversation_id,
            exchange=first_exchange,
            events=(
                {
                    "id": "tool-proposal",
                    "type": "assistant.message",
                    "data": {
                        "content": "Using bash",
                        "toolRequests": [{"toolCallId": "call-1", "name": "bash", "arguments": {"command": "pwd"}}],
                    },
                },
                {
                    "id": "tool-start",
                    "type": "tool.execution_start",
                    "data": {"toolCallId": "call-1", "toolName": "bash", "arguments": {"command": "pwd"}},
                },
                {
                    "id": "tool-complete",
                    "type": "tool.execution_complete",
                    "data": {"toolCallId": "call-1", "success": True, "result": {"content": "/workspace\n"}},
                },
                {
                    "id": "answer-1",
                    "type": "assistant.message",
                    "data": {"content": "harmless answer 1", "toolRequests": []},
                },
                {"id": "idle-1", "type": "session.idle", "data": {"aborted": False}},
            ),
        ),
        _source_turn(
            memory=memory,
            index=2,
            conversation_id=conversation_id,
            exchange=next_exchange,
            events=(
                {
                    "id": "answer-2",
                    "type": "assistant.message",
                    "data": {"content": "harmless answer 2", "toolRequests": []},
                },
                {"id": "idle-2", "type": "session.idle", "data": {"aborted": False}},
            ),
        ),
    ]
    for turn in turns:
        store.record_turn(turn=turn)
    store.seal_sdk_events()
    host_request = b'{"model":"qwen3:1.7b","messages":[{"role":"user","content":"harmless"}]}'
    host_response = b'{"choices":[{"message":{"content":"answer"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}'
    for _ in turns:
        store.record_host_model_exchange(request=host_request, response=host_response, status=200, error=None)
    host_sha = store.seal_host_model()
    adversarial_sha = _adversarial_model(store)
    control_sha = _control_receipts(store)
    gateway_sha = store.record_gateway_audit(records=[first_gateway, next_gateway], turns=turns)
    assert store.assess(report=None, expected_turns=2) == ()
    original = {"value": 1.0, "explanation": "Benign original lifecycle probe"}
    score_bytes = json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
    judgment = InspectGhcpJudgment(
        scorer_name=selected.scorer_name,
        source_event_id="one-original-score-event",
        raw_value=1.0,
        numeric_value=1.0,
        explanation=original["explanation"],
        raw_sha256=hashlib.sha256(score_bytes).hexdigest(),
    )
    raw_log = json.dumps(
        {
            "status": "success",
            "eval": {"task": case.task_name, "run_id": "original-eval-1"},
            "samples": [
                {
                    "id": case.sample_id,
                    "epoch": 1,
                    "scores": {selected.scorer_name: original},
                    "events": [
                        {
                            "event": "score",
                            "scorer": selected.scorer_name,
                            "uuid": judgment.source_event_id,
                            "score": original,
                        }
                    ],
                }
            ],
        },
        sort_keys=True,
    ).encode()
    log_sha = store.record_inspect_log(content=raw_log)
    snapshot = memory.native_cyber_evidence.get_episode(run_id=run_id)
    sdk_stream = next(
        stream for stream in snapshot.raw_streams if stream.key.observed_source_id == store.SDK_KEY.observed_source_id
    )
    report = InspectGhcpReport(
        run_id=run_id,
        task_name=case.task_name,
        task_version=case.task_version,
        sample_id=case.sample_id,
        sample_epoch=case.epoch,
        task_kind=InspectGhcpTaskKind.PROTOCOL_SMOKE,
        provider_verified=True,
        task_assets_sha256={relative: sha for _, relative, sha in selected.source_files},
        image_ids=executor._environment.image_ids,
        cli_sha256=InspectGhcpProtocolPins.GHCP_CLI_SHA256,
        model_id=InspectGhcpProtocolPins.CLI_MODEL_ALIAS,
        wire_model=InspectGhcpProtocolPins.MODEL_NAME,
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        inspect_log_id="original-eval-1",
        inspect_log_sha256=log_sha,
        agent_container_id="a" * 64,
        model_container_id="b" * 64,
        target_container_id="c" * 64,
        ghcp_session_id="sdk-session",
        agent_process={"cli_pid": 201, "net_namespace": "net:[1]"},
        conversation_id=conversation_id,
        turn_count=2,
        sdk_event_count=len(snapshot.events),
        sdk_event_raw_sha256=sdk_stream.stored_sha256,
        model_request_count=2,
        model_http_200_count=2,
        host_model_request_count=2,
        host_model_http_200_count=2,
        adversarial_request_count=1,
        adversarial_http_200_count=1,
        tool_start_count=1,
        tool_complete_count=1,
        successful_tool_execution_count=1,
        required_tool_executions=1,
        gateway_audit_sha256=gateway_sha,
        host_model_audit_sha256=host_sha,
        adversarial_audit_sha256=adversarial_sha,
        control_receipt_sha256=control_sha,
        token_files_absent_before_turn=True,
        stopped_before_scoring=True,
        gateway_alive_before_scoring=True,
        gateway_alive_after_scoring=True,
        target_alive_before_scoring=True,
        target_alive_after_scoring=True,
        original_cleanup_called=True,
        original_cleanup_succeeded=True,
        sandbox_cleanup_observed=True,
        judgment=judgment,
        required_gaps=("Original cyber task/scorer/target image were not independently verified.",),
        status=InspectGhcpStatus.INCOMPLETE,
    )
    assert store.assess(report=report, expected_turns=2) == report.required_gaps
    score = InspectGhcpReportScorer(report_sha256=report.sha256()).prepare_unpersisted_score(report=report)
    evaluation = object.__new__(InspectGhcpEvaluation)
    evaluation._case, evaluation._run = case, run
    evaluation._original_input_sha256 = selected.original_input_sha256
    evaluation.binding = MagicMock(spec=InspectGhcpTaskBinding)
    evaluation.binding.scorer_name = selected.scorer_name
    evaluation._attach_original_case_provenance(report=report, score=score)
    expected = EvalScoreProvenance.from_metadata(metadata=score.score_metadata)
    assert expected.role is EvalScoreRole.BENCHMARK_ORIGINAL
    assert expected.case_run_id == run.case_run_id(case=case)
    episode = store.finalize_atomic(report=report, score=score)
    return InspectGhcpOutcome(report=report, score=score, episode=episode, log_location=None)


@pytest.mark.usefixtures("patch_central_database")
def test_original_inspect_score_event_is_reread_and_existing_score_only_is_linked(
    sqlite_instance: SQLiteMemory,
) -> None:
    executor, run = _selected_case()
    executor._memory = sqlite_instance
    outcome = _persist_original_case(memory=sqlite_instance, executor=executor, run=run)
    with sqlite_instance.get_session() as session:
        before = int(session.scalar(select(func.count(ScoreEntry.id))) or 0)
    result = executor._verify_original_score(case=executor._selected.case, run=run, outcome=outcome)
    assert result.original_score_id == outcome.score.id
    assert result.outcome is AttackOutcome.UNDETERMINED
    assert result.executed_turns == 2
    assert result.original_score_provenance.pyrit_scorer_hash == outcome.score.scorer_class_identifier.hash
    with sqlite_instance.get_session() as session:
        assert int(session.scalar(select(func.count(ScoreEntry.id))) or 0) == before == 1
    readback = InspectGhcpEvidenceStore.open_finalized_for_readback(
        memory=sqlite_instance, run_id=outcome.report.run_id
    )
    assert readback.assess(report=outcome.report, expected_turns=2) == outcome.report.required_gaps


@pytest.mark.usefixtures("patch_central_database")
async def test_case_executor_binds_original_case_before_commit_and_cannot_launch_twice(
    sqlite_instance: SQLiteMemory,
) -> None:
    executor, run = _selected_case()
    executor._memory = sqlite_instance
    expected = _persist_original_case(memory=sqlite_instance, executor=executor, run=run)
    case = executor._selected.case

    async def retained_task_async(self: InspectGhcpEvaluation) -> InspectGhcpOutcome:
        assert self._case == case
        assert self._run == run
        assert self._original_input_sha256 == executor._selected.original_input_sha256
        assert self.binding.approved_asset_labels
        self._run_id = expected.report.run_id
        return expected

    with (
        patch.object(InspectGhcpPilotEnvironment, "verify_host_async", new_callable=AsyncMock) as verify_host,
        patch.object(InspectGhcpEvaluation, "run_async", new=retained_task_async),
    ):
        completed = await executor.execute_case_async(case=case, run=run)
        verify_host.assert_awaited_once()
    assert completed.original_score_id == expected.score.id
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
    with pytest.raises(RuntimeError, match="only once"):
        await executor.execute_case_async(case=case, run=run)


@pytest.mark.usefixtures("patch_central_database")
def test_rejected_corrupt_original_scorer_or_early_agent_stop_cannot_link_score(
    sqlite_instance: SQLiteMemory,
) -> None:
    executor, run = _selected_case()
    executor._memory = sqlite_instance
    outcome = _persist_original_case(memory=sqlite_instance, executor=executor, run=run)
    for changes in (
        {"judgment": outcome.report.judgment.model_copy(update={"source_event_id": "foreign"})},
        {"stopped_before_scoring": False},
        {"required_gaps": (*outcome.report.required_gaps, "incomplete tool result")},
    ):
        bad = InspectGhcpOutcome(
            report=outcome.report.model_copy(update=changes),
            score=outcome.score,
            episode=outcome.episode,
            log_location=None,
        )
        with pytest.raises(ValueError, match="original ScoreEvent|UND report|raw source"):
            executor._verify_original_score(case=executor._selected.case, run=run, outcome=bad)
    with sqlite_instance.get_session() as session:
        assert int(session.scalar(select(func.count(ScoreEntry.id))) or 0) == 1

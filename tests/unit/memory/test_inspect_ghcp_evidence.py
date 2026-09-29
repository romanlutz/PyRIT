# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real SQLite rows for an Inspect-owned task; no fake Score or message links."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.models import MessagePiece, ScoreStatus
from pyrit.models.inspect_ghcp import (
    InspectGhcpJudgment,
    InspectGhcpReport,
    InspectGhcpStatus,
    InspectGhcpTaskKind,
)
from pyrit.prompt_target.inspect_ghcp_target import InspectGhcpTurn
from pyrit.score.float_scale.inspect_ghcp_report_scorer import InspectGhcpReportScorer

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory


def _gateway_record(*, run_id: str) -> tuple[dict[str, object], dict[str, object]]:
    request = b'{"model":"qwen3:1.7b","input":"hello"}'
    response = b'{"id":"response-1","output":[{"type":"message","content":"answer"}]}'
    encoded_request = base64.b64encode(request).decode("ascii")
    encoded_response = base64.b64encode(response).decode("ascii")
    record: dict[str, object] = {
        "sequence": 1,
        "run_id": run_id,
        "source_request_id": "req-1",
        "request_base64": encoded_request,
        "response_base64": encoded_response,
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "response_status": 200,
        "error": None,
    }
    exchange: dict[str, object] = {
        "request_id": "req-1",
        "source_session_id": "sdk-session",
        "request_base64": encoded_request,
        "response_base64": encoded_response,
        "status": 200,
        "error": None,
    }
    return record, exchange


def _source_turn(*, memory: SQLiteMemory, run_id: str) -> InspectGhcpTurn:
    request = MessagePiece(role="user", original_value="hello", conversation_id="pyrit-conversation", sequence=0)
    response = MessagePiece(role="assistant", original_value="answer", conversation_id="pyrit-conversation", sequence=1)
    memory.add_message_pieces_to_memory(message_pieces=[request, response])
    _, exchange = _gateway_record(run_id=run_id)
    return InspectGhcpTurn(
        turn_index=1,
        session_id="sdk-session",
        identity={"worker_pid": 200, "cli_pid": 201, "uid": 10001, "net_namespace": "net:[1]"},
        instruction="hello",
        assistant_text="answer",
        request_piece_id=request.id,
        response_piece_id=response.id,
        events=(
            {
                "id": "event-1",
                "type": "assistant.message",
                "data": {"content": "answer", "toolRequests": []},
            },
            {"id": "event-2", "type": "session.idle", "data": {"aborted": False}},
        ),
        model_exchanges=(exchange,),
    )


def _original_log() -> tuple[bytes, InspectGhcpJudgment]:
    score = {"value": 0.7, "explanation": "Original grader verdict"}
    content = json.dumps(
        {
            "status": "success",
            "eval": {"task": "original_task", "run_id": "original-eval-run"},
            "samples": [
                {
                    "id": "original-sample",
                    "epoch": 1,
                    "scores": {"original_grader": score},
                    "events": [
                        {
                            "event": "score",
                            "scorer": "original_grader",
                            "uuid": "original-score-event-1",
                            "score": score,
                        }
                    ],
                }
            ],
        },
        sort_keys=True,
    ).encode("utf-8")
    score_bytes = json.dumps(score, sort_keys=True, separators=(",", ":")).encode("utf-8")
    judgment = InspectGhcpJudgment(
        scorer_name="original_grader",
        source_event_id="original-score-event-1",
        raw_value=0.7,
        numeric_value=0.7,
        explanation="Original grader verdict",
        raw_sha256=hashlib.sha256(score_bytes).hexdigest(),
    )
    return content, judgment


def _host_model(store: InspectGhcpEvidenceStore) -> str:
    store.record_host_model_exchange(
        request=b'{"model":"qwen3:1.7b","messages":[{"role":"user","content":"hello"}]}',
        response=b'{"choices":[{"message":{"content":"answer"}}],"usage":{"prompt_tokens":5,"completion_tokens":1}}',
        status=200,
        error=None,
    )
    return store.seal_host_model()


def _adversarial_model(store: InspectGhcpEvidenceStore) -> str:
    store.record_adversarial_model_frame(
        request_id="adversarial-request-1",
        phase="request",
        body=b'{"model":"qwen3:1.7b","messages":[{"role":"user","content":"next?"}]}',
        status=None,
        error=None,
    )
    store.record_adversarial_model_frame(
        request_id="adversarial-request-1",
        phase="response",
        body=b'{"choices":[{"message":{"content":"Use the next instruction"}}]}',
        status=200,
        error=None,
    )
    return store.seal_adversarial_model()


def _control_receipts(store: InspectGhcpEvidenceStore) -> str:
    digest = hashlib.sha256(b"x" * 43).hexdigest()
    for service, pid, container in (
        ("model-bridge", 301, "b" * 64),
        ("agent", 302, "a" * 64),
    ):
        store.record_control_receipt(
            receipt={
                "run_id": store.run_id,
                "service": service,
                "source_id": f"{store.run_id}:{service}:{pid}",
                "observed_job_id": pid,
                "container_id": container,
                "frame_size_bytes": 43,
                "frame_sha256": digest,
                "completed_exit_code": 0,
                "inspect_raw_control_elided": True,
                "provenance": "Inspect as_type Docker provider; owner-only tmpfs token file removed on read",
            }
        )
    return store.seal_control_receipts()


@pytest.mark.usefixtures("patch_central_database")
class TestInspectGhcpEvidence:
    @pytest.mark.parametrize("arguments_match", [True, False])
    def test_real_tool_correlation_requires_model_request_and_execution_result(
        self, sqlite_instance: SQLiteMemory, arguments_match: bool
    ) -> None:
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=str(uuid4()),
            task_name="original_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=100_000,
        )
        original = _source_turn(memory=sqlite_instance, run_id=store.run_id)
        events = (
            {
                "id": "request-event",
                "type": "assistant.message",
                "data": {
                    "content": "Using bash",
                    "toolRequests": [{"toolCallId": "call-1", "name": "bash", "arguments": {"command": "pwd"}}],
                },
            },
            {
                "id": "tool-start",
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "call-1",
                    "toolName": "bash",
                    "arguments": {"command": "pwd" if arguments_match else "other"},
                },
            },
            {
                "id": "tool-complete",
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "call-1",
                    "success": True,
                    "result": {"content": "/workspace\n", "detailedContent": "/workspace\n"},
                },
            },
            {"id": "answer", "type": "assistant.message", "data": {"content": "answer", "toolRequests": []}},
            {"id": "idle", "type": "session.idle", "data": {"aborted": False}},
        )
        turn = InspectGhcpTurn(**{**original.__dict__, "events": events})
        store.record_turn(turn=turn)
        store.seal_sdk_events()
        _host_model(store)
        _adversarial_model(store)
        _control_receipts(store)
        store.record_gateway_audit(records=[_gateway_record(run_id=store.run_id)[0]], turns=[turn])
        if arguments_match:
            assert store.assess(report=None, expected_turns=1) == ()
            assert store.count_successful_tool_executions() == 1
        else:
            assert any("not correlated" in gap for gap in store.assess(report=None, expected_turns=1))
            with pytest.raises(ValueError, match="not completely correlated"):
                store.count_successful_tool_executions()

    def test_guest_failure_stderr_is_private_and_digest_checked(self, sqlite_instance: SQLiteMemory) -> None:
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=str(uuid4()),
            task_name="original_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=100_000,
        )
        payload = b"RuntimeError: guest SDK startup failed\n"
        digest = store.record_guest_stderr(data=payload)
        snapshot = sqlite_instance.native_cyber_evidence.get_episode(run_id=store.run_id)
        stream = next(
            item for item in snapshot.raw_streams if item.key.observed_source_id == "inspect-ghcp-worker-stderr"
        )
        assert stream.stored_sha256 == digest
        with pytest.raises(PermissionError):
            sqlite_instance.native_cyber_evidence.read_raw_chunks(run_id=store.run_id, stream_id=stream.stream_id)
        chunks = sqlite_instance.native_cyber_evidence.read_raw_chunks(
            run_id=store.run_id, stream_id=stream.stream_id, allow_sensitive=True
        )
        assert b"".join(chunk.data for chunk in chunks) == payload

    def test_complete_numeric_original_grader_requires_all_real_links(self, sqlite_instance: SQLiteMemory) -> None:
        started_at = datetime.now(UTC)
        run_id = str(uuid4())
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=run_id,
            task_name="original_task",
            task_version="1",
            started_at=started_at,
            raw_byte_limit=100_000,
        )
        store.bind_agent(container_id="a" * 64, session_id="sdk-session")
        turn = _source_turn(memory=sqlite_instance, run_id=run_id)
        store.record_turn(turn=turn)
        store.seal_sdk_events()
        host_sha = _host_model(store)
        adversarial_sha = _adversarial_model(store)
        control_sha = _control_receipts(store)
        gateway_sha = store.record_gateway_audit(records=[_gateway_record(run_id=run_id)[0]], turns=[turn])
        log_bytes, judgment = _original_log()
        log_sha = store.record_inspect_log(content=log_bytes)
        report = InspectGhcpReport(
            run_id=run_id,
            task_name="original_task",
            task_version="1",
            sample_id="original-sample",
            task_kind=InspectGhcpTaskKind.CYBER_BENCHMARK,
            benchmark_verified=True,
            provider_verified=True,
            task_assets_sha256={"original_task": "b" * 64},
            image_ids={"agent": "sha256:" + "d" * 64, "target": "sha256:" + "c" * 64},
            target_image="original-target@sha256:" + "c" * 64,
            cli_sha256="a" * 64,
            model_id="approved",
            wire_model="qwen3:1.7b",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            inspect_log_id="original-eval-run",
            inspect_log_sha256=log_sha,
            gateway_audit_sha256=gateway_sha,
            agent_container_id="a" * 64,
            model_container_id="b" * 64,
            target_container_id="c" * 64,
            agent_process={"cli_pid": 201, "net_namespace": "net:[1]"},
            ghcp_session_id="sdk-session",
            conversation_id="pyrit-conversation",
            turn_count=1,
            sdk_event_count=2,
            sdk_event_raw_sha256=next(
                stream.stored_sha256
                for stream in store._capture.get_episode(run_id=run_id).raw_streams
                if stream.key.observed_source_id == store.SDK_KEY.observed_source_id
            ),
            model_request_count=1,
            model_http_200_count=1,
            host_model_request_count=1,
            host_model_http_200_count=1,
            adversarial_request_count=1,
            adversarial_http_200_count=1,
            tool_start_count=0,
            tool_complete_count=0,
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
            status=InspectGhcpStatus.COMPLETED,
        )
        mismatched = report.model_copy(update={"model_container_id": "d" * 64})
        assert any(
            "control" in gap.lower() and "disagree" in gap for gap in store.assess(report=mismatched, expected_turns=1)
        )
        assert store.assess(report=report, expected_turns=1) == ()
        score = InspectGhcpReportScorer(report_sha256=report.sha256()).prepare_unpersisted_score(report=report)
        assert score.status is ScoreStatus.COMPLETE
        assert score.get_value() == 0.7
        saved = store.finalize_atomic(report=report, score=score)
        assert saved.coverage_complete
        assert saved.score_status is ScoreStatus.COMPLETE
        assert saved.score_id == score.id
        with pytest.raises(ValueError, match="already published"):
            store.finalize_atomic(report=report, score=score)

    def test_real_message_pieces_and_original_score_commit_atomically(self, sqlite_instance: SQLiteMemory) -> None:
        started_at = datetime.now(UTC)
        run_id = str(uuid4())
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=run_id,
            task_name="original_task",
            task_version="1",
            started_at=started_at,
            raw_byte_limit=100_000,
        )
        turn = _source_turn(memory=sqlite_instance, run_id=run_id)
        store.record_turn(turn=turn)
        store.seal_sdk_events()
        host_sha = _host_model(store)
        adversarial_sha = _adversarial_model(store)
        control_sha = _control_receipts(store)
        gateway_sha = store.record_gateway_audit(records=[_gateway_record(run_id=run_id)[0]], turns=[turn])
        assert store.assess(report=None, expected_turns=1) == ()
        log_bytes, judgment = _original_log()
        log_sha = store.record_inspect_log(content=log_bytes)
        report = InspectGhcpReport(
            run_id=run_id,
            task_name="original_task",
            task_version="1",
            sample_id="original-sample",
            task_kind=InspectGhcpTaskKind.PROTOCOL_SMOKE,
            provider_verified=True,
            cli_sha256="a" * 64,
            model_id="approved",
            wire_model="qwen3:1.7b",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            inspect_log_id="original-eval-run",
            inspect_log_sha256=log_sha,
            gateway_audit_sha256=gateway_sha,
            agent_container_id="a" * 64,
            model_container_id="b" * 64,
            ghcp_session_id="sdk-session",
            conversation_id="pyrit-conversation",
            turn_count=1,
            sdk_event_count=2,
            model_request_count=1,
            model_http_200_count=1,
            host_model_request_count=1,
            host_model_http_200_count=1,
            adversarial_request_count=1,
            adversarial_http_200_count=1,
            tool_start_count=0,
            tool_complete_count=0,
            host_model_audit_sha256=host_sha,
            adversarial_audit_sha256=adversarial_sha,
            control_receipt_sha256=control_sha,
            token_files_absent_before_turn=True,
            judgment=judgment,
            status=InspectGhcpStatus.INCOMPLETE,
        )
        gaps = store.assess(report=report, expected_turns=1)
        assert "Original cyber task/scorer/target image were not independently verified." in gaps
        report = report.model_copy(update={"required_gaps": gaps})
        scorer = InspectGhcpReportScorer(report_sha256=report.sha256())
        score = scorer.prepare_unpersisted_score(report=report)
        assert score.status is ScoreStatus.UNDETERMINED
        assert score.score_value is None
        saved = store.finalize_atomic(report=report, score=score)
        assert saved.score_id == score.id
        assert saved.turns[0].request_piece_ids == (turn.request_piece_id,)
        assert saved.turns[0].response_piece_ids == (turn.response_piece_id,)
        assert saved.report_sha256 == report.sha256()
        assert saved.score_status is ScoreStatus.UNDETERMINED

    def test_unsafe_or_missing_private_receipt_blocks_pregrading(self, sqlite_instance: SQLiteMemory) -> None:
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=str(uuid4()),
            task_name="original_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=100_000,
        )
        receipt = {
            "run_id": store.run_id,
            "service": "model-bridge",
            "source_id": f"{store.run_id}:model-bridge:301",
            "observed_job_id": 301,
            "container_id": "b" * 64,
            "frame_size_bytes": 43,
            "frame_sha256": hashlib.sha256(b"x" * 43).hexdigest(),
            "completed_exit_code": 0,
            "inspect_raw_control_elided": True,
            "provenance": "Inspect as_type Docker provider; owner-only tmpfs token file removed on read",
        }
        with pytest.raises(ValueError, match="unsafe"):
            store.record_control_receipt(receipt={**receipt, "raw_token": "x" * 43})
        store.record_control_receipt(receipt=receipt)
        with pytest.raises(ValueError, match="duplicated|differs"):
            store.record_control_receipt(receipt=receipt)
        with pytest.raises(ValueError, match="both"):
            store.seal_control_receipts()
        assert any("control handoffs" in gap.lower() for gap in store.assess(report=None, expected_turns=0))
        agent = {
            **receipt,
            "service": "agent",
            "source_id": f"{store.run_id}:agent:302",
            "observed_job_id": 302,
            "container_id": "a" * 64,
        }
        with pytest.raises(ValueError, match="differs"):
            store.record_control_receipt(receipt={**agent, "frame_sha256": "0" * 64})
        store.record_control_receipt(receipt=agent)
        digest = store.seal_control_receipts()
        snapshot = sqlite_instance.native_cyber_evidence.get_episode(run_id=store.run_id)
        stream = next(
            row for row in snapshot.raw_streams if row.key.observed_source_id == store.CONTROL_KEY.observed_source_id
        )
        stored = b"".join(
            chunk.data
            for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
                run_id=store.run_id, stream_id=stream.stream_id, allow_sensitive=True
            )
        )
        assert stream.stored_sha256 == digest == hashlib.sha256(stored).hexdigest()
        assert b"x" * 43 not in stored

    def test_missing_real_sdk_event_blocks_pregrading(self, sqlite_instance: SQLiteMemory) -> None:
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=str(uuid4()),
            task_name="original_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=100_000,
        )
        turn = _source_turn(memory=sqlite_instance, run_id=store.run_id)
        incomplete = InspectGhcpTurn(**{**turn.__dict__, "events": (turn.events[0],)})
        store.record_turn(turn=incomplete)
        store.seal_sdk_events()
        _host_model(store)
        _adversarial_model(store)
        _control_receipts(store)
        store.record_gateway_audit(records=[_gateway_record(run_id=store.run_id)[0]], turns=[incomplete])
        assert any("GHCP turn" in gap for gap in store.assess(report=None, expected_turns=1))

    def test_gateway_tampering_never_becomes_a_score(self, sqlite_instance: SQLiteMemory) -> None:
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=str(uuid4()),
            task_name="original_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=100_000,
        )
        turn = _source_turn(memory=sqlite_instance, run_id=store.run_id)
        store.record_turn(turn=turn)
        store.seal_sdk_events()
        record, _ = _gateway_record(run_id=store.run_id)
        record["response_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="differs"):
            store.record_gateway_audit(records=[record], turns=[turn])

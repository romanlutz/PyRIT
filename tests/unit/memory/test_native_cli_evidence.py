# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inert CLI sink to database finalizer round-trips, without a CLI or network."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.exc import OperationalError

from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
from pyrit.executor.workflow.native_cli_report_adapter import build_native_cli_run_report
from pyrit.memory.memory_models import (
    NativeCyberEpisodeEntry,
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    NativeCyberToolEventEntry,
    ScorableContentEntry,
    ScoreEntry,
)
from pyrit.models import ContentEntryScorable, ContentScorable, MessagePiece, Score, ScoreStatus
from pyrit.models.native_cli_report import (
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportStatus,
    NativeCliRunReport,
)
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberEpisodeStart,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberTurnFinish,
)
from pyrit.prompt_target.gateway.responses_contract import GatewayCoverage, GatewayFrameKind, GatewayObservation
from pyrit.prompt_target.native_cli_models import (
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliRunOutcome,
    NativeCliStream,
)
from pyrit.prompt_target.native_cli_transport import NativeCliJsonlParser
from pyrit.score.float_scale.native_cli_report_scorer import build_native_cli_report_score
from unit.prompt_target.target.test_native_cli_target import _claude, _codex, _config, _frame

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory
    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore

pytestmark = pytest.mark.usefixtures("patch_central_database")


@dataclass(frozen=True, kw_only=True)
class _CliCase:
    memory: SQLiteMemory
    store: NativeCyberEvidenceStore
    sink: NativeCliDatabaseEvidenceSink
    report: NativeCliRunReport
    score: Score
    stdout: bytes
    stderr: bytes


def _stdout(*, protocol: NativeCliProtocol, tool: bool) -> bytes:
    if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
        return _codex(tool=tool)
    if not tool:
        return _claude(assistant_text="Synthetic answer.", result_text="Done.")
    return b"".join(
        (
            _frame(kind="system", subtype="init", uuid="init-1", session_id="session-1"),
            _frame(
                kind="assistant",
                uuid="assistant-1",
                session_id="session-1",
                message={
                    "id": "message-1",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Synthetic answer."},
                        {"type": "tool_use", "id": "use-1", "name": "Bash", "input": {"command": "true"}},
                    ],
                },
            ),
            _frame(
                kind="user",
                uuid="user-1",
                session_id="session-1",
                message={
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "use-1",
                            "content": "synthetic result",
                            "is_error": False,
                        }
                    ],
                },
            ),
            _frame(
                kind="result",
                subtype="success",
                uuid="result-1",
                session_id="session-1",
                is_error=False,
                result="Done.",
            ),
        )
    )


def _record_fake_messages_gateway(
    *,
    store: NativeCyberEvidenceStore,
    run_id: str,
    response_present: bool,
    gateway_error: bool,
    provider_error: bool,
    streaming: bool,
    fake_done: bool,
    response_coverage: frozenset[GatewayCoverage],
) -> bool:
    """Simulate the future Anthropic callback's DB contract without invoking a gateway."""
    protocol = NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE.value
    starts = {
        name: NativeCyberRawStreamStart(
            run_id=run_id,
            turn_index=1,
            key=NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS if name == "error" else NativeCyberEvidenceSource.MODEL,
                kind=NativeCyberRawKind.MODEL,
                observed_source_id=f"{protocol}.gateway.{name}s",
            ),
        )
        for name in ("request", "response", "error")
    }
    for raw_stream in starts.values():
        store.open_raw_stream(stream=raw_stream)
    request_wire = b'{"model":"synthetic","messages":[{"role":"user","content":"inert"}],"max_tokens":32}'
    frames: list[tuple[str, str, bytes, frozenset[GatewayCoverage], int | None, str | None]] = [
        ("request", "request", request_wire, frozenset(), None, None),
    ]
    if gateway_error:
        frames.append(
            (
                "error",
                "gateway_error",
                b'event: error\ndata: {"error":"synthetic"}\n\n',
                frozenset({GatewayCoverage.FAILED}),
                502,
                "synthetic_error",
            )
        )
    elif response_present and provider_error:
        frames.append(
            (
                "response",
                "response",
                b'{"type":"error","error":{"type":"rate_limit_error"}}',
                frozenset({GatewayCoverage.FAILED}),
                429,
                None,
            )
        )
    elif response_present and streaming:
        frames.extend(
            [
                (
                    "response",
                    "response_event",
                    b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_synthetic"}}\n\n',
                    frozenset({GatewayCoverage.STREAMING}),
                    None,
                    None,
                ),
                (
                    "response",
                    "response_event",
                    b"data: [DONE]\n\n" if fake_done else b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
                    response_coverage,
                    None,
                    None,
                ),
            ]
        )
    elif response_present:
        frames.append(
            (
                "response",
                "response",
                b'{"id":"msg_synthetic","type":"message","role":"assistant","content":[],"stop_reason":"end_turn"}',
                response_coverage,
                200,
                None,
            )
        )
    offset = dict.fromkeys(starts, 0)
    hashes = {name: hashlib.sha256() for name in starts}
    current = len(store.get_episode(run_id=run_id).events)
    for name, kind, frame, coverage, status, error_code in frames:
        raw = starts[name]
        store.append_raw(run_id=run_id, stream_id=raw.stream_id, data=frame)
        store.append_events(
            run_id=run_id,
            turn_index=1,
            events=(
                NativeCyberCapturedEvent(
                    source=raw.key.source,
                    event=NativeCyberObservedEvent(
                        controller_sequence=current + 1,
                        event_type=f"messages_gateway.{kind}",
                        payload={
                            "gateway_request_id": "gateway-request-1",
                            "frame_sha256": hashlib.sha256(frame).hexdigest(),
                            "frame_size_bytes": len(frame),
                            "coverage": sorted(flag.value for flag in coverage),
                            "wire_protocol": "anthropic_messages",
                            "error_code": error_code,
                            "status_code": status,
                            "headers": [["anthropic-version", "2023-06-01"]] if kind == "request" else [],
                            "query_string": "",
                        },
                        observed_stream_id=str(raw.stream_id),
                        stream_offset=offset[name],
                    ),
                ),
            ),
        )
        hashes[name].update(frame)
        offset[name] += len(frame)
        current += 1
    successful = (
        response_present
        and not gateway_error
        and not provider_error
        and (not streaming or not fake_done)
        and GatewayCoverage.COMPLETED in response_coverage
        and GatewayCoverage.FAILED not in response_coverage
        and GatewayCoverage.INCOMPLETE not in response_coverage
    )
    for name, raw_stream in starts.items():
        store.close_raw_stream(
            run_id=run_id,
            stream_id=raw_stream.stream_id,
            source_complete=successful,
            expected_bytes=offset[name],
            observed_sha256=hashes[name].hexdigest(),
        )
    return successful


async def _capture_case_async(
    *,
    memory: SQLiteMemory,
    protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON,
    tool: bool = False,
    with_gateway: bool = True,
    declare_gateway: bool = True,
    gateway_response: bool = True,
    gateway_error: bool = False,
    gateway_streaming: bool = False,
    anthropic_fake_done: bool = False,
    provider_error: bool = False,
    response_coverage: frozenset[GatewayCoverage] = frozenset({GatewayCoverage.COMPLETED}),
    declare_task_identity: bool = True,
    link_request: bool = True,
    link_response: bool = True,
    stderr_bytes: bytes = b"synthetic stderr\xff\n",
    raw_byte_limit: int = 268_435_456,
) -> _CliCase:
    store = memory.native_cyber_evidence
    run_id, turn_id, conversation_id = str(uuid4()), str(uuid4()), str(uuid4())
    sink = NativeCliDatabaseEvidenceSink(
        store=store,
        run_id=run_id,
        turn_id=turn_id,
        turn_index=1,
        protocol=protocol,
        include_model_gateway=with_gateway and protocol is NativeCliProtocol.CODEX_EXEC_JSON,
    )
    await asyncio.to_thread(
        store.create_episode,
        start=NativeCyberEpisodeStart(
            run_id=run_id,
            binding_name="synthetic-cli-binding",
            binding_version="1",
            task_id="synthetic-task" if declare_task_identity else None,
            task_version="revision-1" if declare_task_identity else None,
            simulated=True,
            required_raw_streams=sink.required_raw_streams(
                protocol=protocol,
                include_model_gateway=declare_gateway,
            ),
            raw_byte_limit=raw_byte_limit,
        ),
    )
    await sink.start_async(started_at=datetime.now(UTC))
    if with_gateway and protocol is NativeCliProtocol.CODEX_EXEC_JSON:
        request_wire = b'{"model":"synthetic","input":"inert"}'
        await sink.record_gateway_observation_async(
            GatewayObservation(
                run_id=run_id,
                request_id="gateway-request-1",
                kind=GatewayFrameKind.REQUEST,
                frame=request_wire,
                coverage=frozenset(),
            )
        )
        if gateway_error:
            await sink.record_gateway_observation_async(
                GatewayObservation(
                    run_id=run_id,
                    request_id="gateway-request-1",
                    kind=GatewayFrameKind.GATEWAY_ERROR,
                    frame=b'event: error\ndata: {"error":"synthetic"}\n\n',
                    coverage=frozenset({GatewayCoverage.FAILED}),
                    error_code="synthetic_error",
                    status_code=502,
                )
            )
        elif gateway_response and gateway_streaming:
            await sink.record_gateway_observation_async(
                GatewayObservation(
                    run_id=run_id,
                    request_id="gateway-request-1",
                    kind=GatewayFrameKind.RESPONSE_EVENT,
                    frame=b'data: {"delta":"synthetic"}\n\n',
                    coverage=frozenset({GatewayCoverage.STREAMING}),
                )
            )
            await sink.record_gateway_observation_async(
                GatewayObservation(
                    run_id=run_id,
                    request_id="gateway-request-1",
                    kind=GatewayFrameKind.RESPONSE_EVENT,
                    frame=b"data: [DONE]\n\n",
                    coverage=response_coverage,
                )
            )
        elif gateway_response:
            await sink.record_gateway_observation_async(
                GatewayObservation(
                    run_id=run_id,
                    request_id="gateway-request-1",
                    kind=GatewayFrameKind.RESPONSE,
                    frame=b'{"id":"synthetic-response","output":[]}',
                    coverage=response_coverage,
                    status_code=200,
                )
            )
    config = _config(protocol=protocol)
    stdout = _stdout(protocol=protocol, tool=tool)
    stderr = stderr_bytes
    split = len(stdout) // 2
    await sink.record_raw_async(
        chunk=NativeCliRawChunk(
            sequence=1,
            stream=NativeCliStream.STDOUT,
            data=stdout[:split],
        )
    )
    await sink.record_raw_async(
        chunk=NativeCliRawChunk(
            sequence=2,
            stream=NativeCliStream.STDERR,
            data=stderr,
        )
    )
    await sink.record_raw_async(
        chunk=NativeCliRawChunk(
            sequence=3,
            stream=NativeCliStream.STDOUT,
            data=stdout[split:],
        )
    )
    parser = NativeCliJsonlParser(config=config)
    parser_events = (*parser.feed(data=stdout), *parser.finish())
    for event in parser_events:
        await sink.record_event_async(event=event)
    outcome = NativeCliRunOutcome(
        exit_code=0,
        terminal_observed=parser.terminal_observed,
        coverage_complete=parser.coverage_complete,
        source_session_id=parser.source_session_id,
        observed_steps=parser.observed_steps,
        frame_count=parser.frame_count,
        raw_chunk_count=3,
        raw_stdout_bytes=len(stdout),
        raw_stderr_bytes=len(stderr),
        gaps=parser.gaps,
    )
    assert outcome.coverage_complete
    request = MessagePiece(
        role="user",
        conversation_id=conversation_id,
        original_value="Synthetic request.",
        sequence=0,
    )
    response = MessagePiece(
        role="assistant",
        conversation_id=conversation_id,
        original_value="Synthetic answer.",
        sequence=1,
    )
    memory.add_message_pieces_to_memory(message_pieces=[request, response])
    if protocol is NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE:
        gateway_success = (
            await asyncio.to_thread(
                _record_fake_messages_gateway,
                store=store,
                run_id=run_id,
                response_present=gateway_response,
                gateway_error=gateway_error,
                provider_error=provider_error,
                streaming=gateway_streaming,
                fake_done=anthropic_fake_done,
                response_coverage=response_coverage,
            )
            if with_gateway
            else True
        )
        episode = await asyncio.to_thread(store.get_episode, run_id=run_id)
        for stream in episode.raw_streams:
            if stream.key.kind not in {NativeCyberRawKind.STDOUT, NativeCyberRawKind.STDERR}:
                continue
            source = stdout if stream.key.kind is NativeCyberRawKind.STDOUT else stderr
            await asyncio.to_thread(
                store.close_raw_stream,
                run_id=run_id,
                stream_id=stream.stream_id,
                source_complete=outcome.coverage_complete and gateway_success,
                expected_bytes=len(source),
                observed_sha256=hashlib.sha256(source).hexdigest(),
            )
        episode = await asyncio.to_thread(store.get_episode, run_id=run_id)
        await asyncio.to_thread(
            store.finish_turn,
            finish=NativeCyberTurnFinish(
                run_id=run_id,
                turn_index=1,
                request_piece_ids=(request.id,) if link_request else (),
                response_piece_ids=(response.id,) if link_response else (),
                observed_event_count=len(episode.events),
                source_complete=outcome.coverage_complete and gateway_success,
                gaps=() if gateway_success else ("Synthetic Anthropic gateway coverage is incomplete.",),
            ),
        )
        events = parser_events
    else:
        await sink.finish_async(
            outcome=outcome,
            request_piece_ids=(request.id,) if link_request else (),
            response_piece_ids=(response.id,) if link_response else (),
        )
        events = await sink.read_report_events_async()
    report = build_native_cli_run_report(
        config=config,
        outcome=outcome,
        events=events,
        task_id="synthetic-task",
        task_version="revision-1",
        run_id=run_id,
        turn_id=turn_id,
        turn_index=1,
        conversation_id=conversation_id,
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=NativeCliOriginalJudgment(
            grader_ref="synthetic-grader",
            grader_evidence_ref="synthetic-original-feedback",
            value=0.75,
            complete=True,
            rationale="Synthetic original task grade.",
        ),
        raw_evidence_ref=f"db-episode:{run_id}",
    )
    return _CliCase(
        memory=memory,
        store=store,
        sink=sink,
        report=report,
        score=build_native_cli_report_score(report=report),
        stdout=stdout,
        stderr=stderr,
    )


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
async def test_cli_atomic_finalizer_links_real_sink_pipes_gateway_and_report_async(
    *,
    sqlite_instance: SQLiteMemory,
    protocol: NativeCliProtocol,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, protocol=protocol)
    assert sqlite_instance.get_scores(score_ids=[str(case.score.id)]) == []

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.COMPLETE
    assert snapshot.coverage_complete and not snapshot.gaps
    assert snapshot.run.task_id == case.report.task_id
    assert snapshot.run.task_version == case.report.task_version
    assert snapshot.turns[0].source_turn_id == case.report.turn_id
    assert snapshot.score_id == case.score.id and snapshot.report_sha256 == case.report.sha256()
    assert snapshot == case.store.get_finalized_episode(run_id=case.report.run_id)
    assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
    assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1
    stored = sqlite_instance.get_scores(score_ids=[str(case.score.id)])[0]
    assert isinstance(stored.scorable, ContentEntryScorable)
    assert stored.score_metadata["publication_state"] == "committed_final_result"
    retained = sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
        snapshot.report_content_id
    ]
    assert retained.value == case.report.canonical_json()
    assert NativeCliRunReport.model_validate_json(retained.value).judgment.value == 0.75
    assert snapshot.stored_raw_bytes == sum(item.stored_bytes for item in snapshot.raw_streams)
    for kind, expected in (("stdout", case.stdout), ("stderr", case.stderr)):
        stream = next(item for item in snapshot.raw_streams if item.key.kind.value == kind)
        chunks = case.store.read_raw_chunks(
            run_id=case.report.run_id,
            stream_id=stream.stream_id,
            allow_sensitive=True,
        )
        assert b"".join(chunk.data for chunk in chunks) == expected
        assert stream.stored_sha256 == hashlib.sha256(expected).hexdigest()
    assert all(
        item.source_complete and item.stored_bytes > 0
        for item in snapshot.raw_streams
        if item.key.observed_source_id.endswith((".gateway.requests", ".gateway.responses"))
    )
    receipts = [
        item.event.payload
        for item in case.store.read_event_payloads(run_id=case.report.run_id, allow_sensitive=True)
        if item.event.event_type == "native_cli.raw_chunk"
    ]
    assert [item["raw_chunk_sequence"] for item in receipts] == [1, 2, 3]
    assert [item["stream"] for item in receipts] == ["stdout", "stderr", "stdout"]


async def test_cli_missing_gateway_even_if_undeclared_is_undetermined_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance, with_gateway=False, declare_gateway=False)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_id == case.score.id and snapshot.score_status is ScoreStatus.UNDETERMINED
    assert not snapshot.coverage_complete
    assert any("gateway" in gap or "model" in gap for gap in snapshot.gaps)
    assert sqlite_instance.get_scores(score_ids=[str(case.score.id)])[0].is_undetermined
    assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
    assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1


async def test_cli_raw_evidence_reference_must_name_the_persisted_episode_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    report = case.report.model_copy(
        update={"evidence": case.report.evidence.model_copy(update={"raw_evidence_ref": "db-episode:another-run"})}
    )
    score = build_native_cli_report_score(report=report)

    snapshot = case.store.finalize_cli_episode_atomic(report=report, score=score, expected_turns=1)

    assert snapshot.score_id == score.id and snapshot.score_status is ScoreStatus.UNDETERMINED
    assert not snapshot.coverage_complete
    assert any("raw evidence reference" in gap for gap in snapshot.gaps)
    assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].is_undetermined


@pytest.mark.parametrize(
    ("link_request", "link_response", "expected_gap"),
    [
        (False, True, "genuine persisted request"),
        (True, False, "genuine assistant response"),
    ],
)
async def test_cli_missing_real_conversation_piece_never_completes_async(
    *,
    sqlite_instance: SQLiteMemory,
    link_request: bool,
    link_response: bool,
    expected_gap: str,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        link_request=link_request,
        link_response=link_response,
    )

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any(expected_gap in gap for gap in snapshot.gaps)
    assert snapshot.turns[0].source_complete is False


async def test_cli_complete_gateway_is_verified_when_omitted_from_manifest_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, with_gateway=True, declare_gateway=False)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.coverage_complete and snapshot.score_status is ScoreStatus.COMPLETE
    assert len(snapshot.run.required_raw_streams) == 2
    assert len(snapshot.raw_streams) == 5


async def test_cli_undeclared_gateway_write_failure_is_still_required_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance, declare_gateway=False)
    case.store.mark_capture_gap(
        run_id=case.report.run_id,
        reason="A native evidence database write failed; complete capture cannot be established.",
        required=False,
    )

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("undeclared source" in gap for gap in snapshot.gaps)
    assert snapshot.optional_gaps


async def test_cli_cross_pipe_receipt_spans_bounded_database_chunks_async(*, sqlite_instance: SQLiteMemory) -> None:
    stderr = b"S" * (sqlite_instance.native_cyber_evidence.MAX_CHUNK_BYTES + 19)
    case = await _capture_case_async(memory=sqlite_instance, stderr_bytes=stderr)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.coverage_complete and snapshot.score_status is ScoreStatus.COMPLETE
    stream = next(item for item in snapshot.raw_streams if item.key.kind.value == "stderr")
    chunks = case.store.read_raw_chunks(run_id=case.report.run_id, stream_id=stream.stream_id, allow_sensitive=True)
    assert len(chunks) == 2
    assert all(chunk.length <= case.store.MAX_CHUNK_BYTES for chunk in chunks)
    assert b"".join(chunk.data for chunk in chunks) == stderr


async def test_cli_streaming_gateway_requires_terminal_done_and_completed_coverage_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, gateway_streaming=True)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.coverage_complete and snapshot.score_status is ScoreStatus.COMPLETE
    assert not snapshot.gaps


async def test_claude_messages_gateway_requires_real_message_stop_not_responses_done_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
        gateway_streaming=True,
    )

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.coverage_complete and snapshot.score_status is ScoreStatus.COMPLETE
    assert all(
        event.event_type.startswith("messages_gateway.")
        for event in snapshot.events
        if event.event_type.endswith(("request", "response_event"))
    )
    response_stream = next(
        item for item in snapshot.raw_streams if item.key.observed_source_id.endswith(".gateway.responses")
    )
    frames = b"".join(
        chunk.data
        for chunk in case.store.read_raw_chunks(
            run_id=case.report.run_id,
            stream_id=response_stream.stream_id,
            allow_sensitive=True,
        )
    )
    assert b"event: message_stop\n" in frames
    assert b"data: [DONE]" not in frames


async def test_claude_responses_done_cannot_substitute_for_anthropic_message_stop_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
        gateway_streaming=True,
        anthropic_fake_done=True,
    )

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("confirmed terminal frame" in gap for gap in snapshot.gaps)


@pytest.mark.parametrize(
    ("provider_error", "expected_gap"),
    [
        (True, "Original Anthropic provider returned a failed response."),
        (False, "host-generated error"),
    ],
)
async def test_claude_real_provider_error_and_host_gateway_error_remain_distinct_async(
    *,
    sqlite_instance: SQLiteMemory,
    provider_error: bool,
    expected_gap: str,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
        gateway_error=not provider_error,
        provider_error=provider_error,
    )

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any(expected_gap in gap for gap in snapshot.gaps)
    assert any(
        event.event_type == ("messages_gateway.response" if provider_error else "messages_gateway.gateway_error")
        for event in snapshot.events
    )
    if provider_error:
        assert not any(event.event_type == "messages_gateway.gateway_error" for event in snapshot.events)


async def test_claude_cannot_claim_anthropic_coverage_from_responses_events_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
    )
    with sqlite_instance.get_session() as session:
        events = session.scalars(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type.like("messages_gateway.%"),
            )
        )
        for item in events:
            item.event_type = item.event_type.replace("messages_gateway.", "gateway.", 1)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("wire protocol differs" in gap for gap in snapshot.gaps)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("wire_protocol", "openai_responses"),
        ("headers", [["authorization", "synthetic-not-a-secret"]]),
        ("query_string", "unapproved=true"),
    ],
)
async def test_claude_gateway_metadata_must_be_selected_and_safe_async(
    *,
    sqlite_instance: SQLiteMemory,
    key: str,
    value: str | list[list[str]],
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
    )
    with sqlite_instance.get_session() as session:
        event = session.scalar(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type == "messages_gateway.request",
            )
        )
        assert event is not None
        event.payload = {**event.payload, key: value}
        event.payload_sha256 = case.store._hash_payload(event.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("Claude model gateway" in gap for gap in snapshot.gaps)


async def test_claude_retains_approved_beta_query_and_selected_header_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
    )
    with sqlite_instance.get_session() as session:
        gateway_events = session.scalars(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type.like("messages_gateway.%"),
            )
        )
        for item in gateway_events:
            item.payload = {
                **item.payload,
                "query_string": "beta=true",
                "headers": [["anthropic-beta", "prompt-caching-2024-07-31"]]
                if item.event_type.endswith(".request")
                else [],
            }
            item.payload_sha256 = case.store._hash_payload(item.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.coverage_complete and snapshot.score_status is ScoreStatus.COMPLETE
    assert "prompt-caching-2024-07-31" not in snapshot.model_dump_json()


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
async def test_cli_unknown_gateway_coverage_flag_is_a_required_gap_async(
    *,
    sqlite_instance: SQLiteMemory,
    protocol: NativeCliProtocol,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, protocol=protocol)
    gateway_type = "gateway.response" if protocol is NativeCliProtocol.CODEX_EXEC_JSON else "messages_gateway.response"
    with sqlite_instance.get_session() as session:
        event = session.scalar(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type == gateway_type,
            )
        )
        assert event is not None
        event.payload = {**event.payload, "coverage": ["completed", "made_up_feature"]}
        event.payload_sha256 = case.store._hash_payload(event.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("unrecognized wire coverage" in gap for gap in snapshot.gaps)


async def test_claude_nonstream_response_requires_exact_success_status_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
    )
    with sqlite_instance.get_session() as session:
        event = session.scalar(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type == "messages_gateway.response",
            )
        )
        assert event is not None
        event.payload = {**event.payload, "status_code": 201}
        event.payload_sha256 = case.store._hash_payload(event.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("not HTTP 200" in gap for gap in snapshot.gaps)


async def test_codex_gateway_redirect_is_not_original_model_success_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    with sqlite_instance.get_session() as session:
        event = session.scalar(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type == "gateway.response",
            )
        )
        assert event is not None
        event.payload = {**event.payload, "status_code": 302}
        event.payload_sha256 = case.store._hash_payload(event.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("unsuccessful HTTP status" in gap for gap in snapshot.gaps)


@pytest.mark.parametrize(
    "flags",
    [
        frozenset(),
        frozenset({GatewayCoverage.COMPLETED, GatewayCoverage.INCOMPLETE}),
        frozenset({GatewayCoverage.COMPLETED, GatewayCoverage.FAILED}),
    ],
)
async def test_cli_gateway_terminal_without_clean_completed_flag_is_undetermined_async(
    *,
    sqlite_instance: SQLiteMemory,
    flags: frozenset[GatewayCoverage],
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, response_coverage=flags)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert not snapshot.coverage_complete
    assert any("gateway" in gap.lower() for gap in snapshot.gaps)
    assert case.report.evidence.coverage_complete


async def test_cli_second_terminal_gateway_frame_is_a_required_gap_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    gateway_raw = b'{"id":"synthetic-response","output":[]}'
    payload = {
        "gateway_request_id": "gateway-request-1",
        "frame_sha256": hashlib.sha256(gateway_raw).hexdigest(),
        "frame_size_bytes": len(gateway_raw),
        "coverage": ["completed"],
        "error_code": None,
        "status_code": None,
    }
    response_stream = next(
        stream
        for stream in case.store.get_episode(run_id=case.report.run_id).raw_streams
        if stream.key.observed_source_id.endswith(".gateway.responses")
    )
    with sqlite_instance.get_session() as session:
        last = session.scalar(
            select(func.max(NativeCyberEventEntry.sequence)).where(NativeCyberEventEntry.run_id == case.report.run_id)
        )
        assert last is not None
        session.add(
            NativeCyberEventEntry(
                run_id=case.report.run_id,
                sequence=last + 1,
                turn_index=1,
                source="model",
                observed_event_id=None,
                observed_session_id=None,
                observed_stream_id=str(response_stream.stream_id),
                stream_offset=0,
                tool_call_id=None,
                tool_phase=None,
                event_type="gateway.response",
                payload=payload,
                payload_sha256=case.store._hash_payload(payload),
                captured_at=datetime.now(UTC),
            )
        )
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("after its completed response" in gap for gap in snapshot.gaps)
    assert case.report.evidence.coverage_complete


@pytest.mark.parametrize("gateway_error", [False, True])
async def test_cli_missing_model_response_or_host_error_cannot_claim_complete_async(
    *,
    sqlite_instance: SQLiteMemory,
    gateway_error: bool,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        gateway_response=False,
        gateway_error=gateway_error,
    )

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert not snapshot.coverage_complete
    assert any("gateway" in gap.lower() or "model" in gap.lower() for gap in snapshot.gaps)
    assert sqlite_instance.get_scores(score_ids=[str(case.score.id)])[0].is_undetermined


@pytest.mark.parametrize(
    ("protocol", "expected"),
    [
        (NativeCliProtocol.CODEX_EXEC_JSON, ScoreStatus.UNDETERMINED),
        (NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE, ScoreStatus.COMPLETE),
    ],
)
async def test_cli_tool_causality_uses_real_source_ids_without_invented_codex_request_async(
    *,
    sqlite_instance: SQLiteMemory,
    protocol: NativeCliProtocol,
    expected: ScoreStatus,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, protocol=protocol, tool=True)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is expected
    assert snapshot.coverage_complete is (expected is ScoreStatus.COMPLETE)
    assert len(snapshot.tools) == 1
    if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
        assert snapshot.tools[0].request_sequence is None
        assert snapshot.tools[0].start_sequence < snapshot.tools[0].completion_sequence
        assert snapshot.tools[0].completion_sequence < snapshot.tools[0].result_sequence
        assert any("no provable model-visible request" in gap for gap in snapshot.gaps)
        assert [event.observed_event_id for event in snapshot.events if event.tool_call_id == "cmd-1"] == [
            "cmd-1",
            "cmd-1",
            "cmd-1",
        ]
    else:
        assert snapshot.tools[0].call_id == "use-1"
        assert snapshot.tools[0].request_sequence < snapshot.tools[0].result_sequence
        assert snapshot.tools[0].start_sequence is None
        assert snapshot.tools[0].completion_sequence is None
        assert not snapshot.gaps


async def test_cli_missing_claude_tool_request_link_cannot_claim_result_causality_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(
        memory=sqlite_instance,
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
        tool=True,
    )
    with sqlite_instance.get_session() as session:
        request_link = session.get(NativeCyberToolEventEntry, (case.report.run_id, "use-1", "request"))
        assert request_link is not None
        session.delete(request_link)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("tool request/start/completion/result links differ" in gap for gap in snapshot.gaps)
    assert snapshot.tools[0].request_sequence is None
    assert snapshot.tools[0].result_sequence is not None


async def test_cli_raw_quota_loss_downgrades_original_numeric_grade_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance, raw_byte_limit=64)
    assert case.report.judgment is not None and case.report.judgment.value == 0.75

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("capped" in gap or "quota" in gap or "raw byte" in gap.lower() for gap in snapshot.gaps)
    assert snapshot.stored_raw_bytes == 64
    assert any(stream.omitted_bytes > 0 for stream in snapshot.raw_streams)
    retained = sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
        snapshot.report_content_id
    ]
    assert NativeCliRunReport.model_validate_json(retained.value).judgment.value == 0.75


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("task_id", "other-task", "task identity"),
        ("task_version", "revision-2", "task identity"),
        ("turn_id", "other-turn", "turn identity"),
        ("conversation_id", "other-conversation", "conversation"),
        ("simulated", False, "simulation provenance"),
    ],
)
async def test_cli_mismatched_report_provenance_downgrades_the_same_score_async(
    *,
    sqlite_instance: SQLiteMemory,
    field: str,
    value: str | bool,
    reason: str,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    raw = case.report.model_dump(mode="json")
    raw[field] = value
    report = NativeCliRunReport.model_validate(raw)
    score = build_native_cli_report_score(report=report)

    snapshot = case.store.finalize_cli_episode_atomic(report=report, score=score, expected_turns=1)

    assert snapshot.score_id == score.id
    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any(reason in gap for gap in snapshot.gaps)
    assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
    assert (
        NativeCliRunReport.model_validate_json(
            sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
                snapshot.report_content_id
            ].value
        ).judgment.value
        == 0.75
    )


async def test_cli_missing_task_identity_or_unverifiable_multi_turn_cannot_complete_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance, declare_task_identity=False)

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=2)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("task ID and version" in gap for gap in snapshot.gaps)
    assert any("multi-turn" in gap for gap in snapshot.gaps)


async def test_cli_modified_source_session_in_database_is_not_report_provenance_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    with sqlite_instance.get_session() as session:
        episode = session.get(NativeCyberEpisodeEntry, case.report.run_id)
        assert episode is not None
        episode.source_session_id = "foreign-session"
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("source session" in gap for gap in snapshot.gaps)


async def test_cli_report_frame_digest_must_match_db_parser_row_and_real_stdout_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    raw = case.report.model_dump(mode="json")
    raw["evidence"]["events"][1]["raw_frame_sha256"] = "0" * 64
    report = NativeCliRunReport.model_validate(raw)
    score = build_native_cli_report_score(report=report)

    snapshot = case.store.finalize_cli_episode_atomic(report=report, score=score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("event summaries" in gap for gap in snapshot.gaps)
    assert report.judgment is not None and report.judgment.value == 0.75


async def test_cli_corrupted_database_event_shape_cannot_claim_complete_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    with sqlite_instance.get_session() as session:
        session.execute(
            update(NativeCyberEventEntry)
            .where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.sequence == 1,
            )
            .values(payload=["malformed synthetic event"], payload_sha256="0" * 64)
        )
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("no structured source payload" in gap for gap in snapshot.gaps)


async def test_cli_tampered_cross_pipe_order_is_detected_even_with_rehashed_event_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    with sqlite_instance.get_session() as session:
        event = session.scalar(
            select(NativeCyberEventEntry)
            .where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type == "native_cli.raw_chunk",
            )
            .order_by(NativeCyberEventEntry.sequence)
            .limit(1)
        )
        assert event is not None
        event.payload = {**event.payload, "raw_chunk_sequence": 2}
        event.payload_sha256 = case.store._hash_payload(event.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("cross-pipe raw chunk order" in gap for gap in snapshot.gaps)
    assert snapshot.stored_raw_bytes > 0


async def test_cli_modified_database_raw_chunk_is_not_a_complete_frame_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    with sqlite_instance.get_session() as session:
        stdout = session.scalar(
            select(NativeCyberRawStreamEntry).where(
                NativeCyberRawStreamEntry.run_id == case.report.run_id,
                NativeCyberRawStreamEntry.kind == "stdout",
            )
        )
        assert stdout is not None
        first = session.get(NativeCyberRawChunkEntry, (stdout.stream_id, 1))
        assert first is not None
        first.data = b"x" + first.data[1:]
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("corrupt" in gap or "source bytes" in gap for gap in snapshot.gaps)


async def test_cli_gateway_response_without_matching_request_id_is_undetermined_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    with sqlite_instance.get_session() as session:
        event = session.scalar(
            select(NativeCyberEventEntry).where(
                NativeCyberEventEntry.run_id == case.report.run_id,
                NativeCyberEventEntry.event_type == "gateway.response",
            )
        )
        assert event is not None
        event.payload = {**event.payload, "gateway_request_id": "not-observed"}
        event.payload_sha256 = case.store._hash_payload(event.payload)
        session.commit()

    snapshot = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("no preceding real request ID" in gap for gap in snapshot.gaps)


async def test_cli_atomic_rollback_after_score_insert_leaves_no_orphan_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    failure = OperationalError("UPDATE", {}, Exception("synthetic CLI finalizer failure"))

    with patch.object(case.store, "_validate_cli_stored_score", side_effect=failure):
        with pytest.raises(OperationalError, match="synthetic CLI finalizer failure"):
            case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)

    assert sqlite_instance._query_entries(ScoreEntry) == []
    assert sqlite_instance._query_entries(ScorableContentEntry) == []
    pending = case.store.get_episode(run_id=case.report.run_id)
    assert pending.finalized_at is None and pending.score_id is None and pending.report_content_id is None
    assert any("database write failed" in gap for gap in pending.gaps)
    with pytest.raises(ValueError, match="not finalized"):
        case.store.get_finalized_episode(run_id=case.report.run_id)

    recovered = case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)
    assert recovered.score_id == case.score.id and recovered.score_status is ScoreStatus.UNDETERMINED
    assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
    assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1


async def test_cli_atomic_link_write_failure_rolls_back_report_and_score_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    updates = 0

    def _fail_first_episode_update(*args: object) -> None:
        nonlocal updates
        statement = str(args[2])
        if "UPDATE" in statement and "NativeCyberEpisodeEntries" in statement:
            updates += 1
            if updates == 1:
                raise OperationalError("UPDATE", {}, Exception("synthetic CLI episode link failure"))

    event.listen(sqlite_instance.engine, "after_cursor_execute", _fail_first_episode_update)
    try:
        with pytest.raises(OperationalError, match="synthetic CLI episode link failure"):
            case.store.finalize_cli_episode_atomic(report=case.report, score=case.score, expected_turns=1)
    finally:
        event.remove(sqlite_instance.engine, "after_cursor_execute", _fail_first_episode_update)

    assert updates >= 1
    assert sqlite_instance._query_entries(ScoreEntry) == []
    assert sqlite_instance._query_entries(ScorableContentEntry) == []
    pending = case.store.get_episode(run_id=case.report.run_id)
    assert pending.finalized_at is None and pending.score_id is None and pending.report_content_id is None
    assert any("database write failed" in gap for gap in pending.gaps)
    with pytest.raises(ValueError, match="not finalized"):
        case.store.get_finalized_episode(run_id=case.report.run_id)


async def test_cli_finalizer_rejects_an_already_persisted_score_id_async(*, sqlite_instance: SQLiteMemory) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    candidate = case.score.model_copy(deep=True)
    sqlite_instance.add_scores_to_memory(scores=[case.score])
    assert len(sqlite_instance._query_entries(ScoreEntry)) == 1

    with pytest.raises(ValueError, match="unpersisted Score ID"):
        case.store.finalize_cli_episode_atomic(report=case.report, score=candidate, expected_turns=1)

    pending = case.store.get_episode(run_id=case.report.run_id)
    assert pending.finalized_at is None and pending.score_id is None
    assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
    assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1


async def test_cli_finalizer_rejects_noncanonical_unpersisted_score_content_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    untrusted = case.score.model_copy(update={"scorable": ContentScorable(value="not the canonical report")})

    with pytest.raises(ValueError, match="unpersisted canonical report"):
        case.store.finalize_cli_episode_atomic(report=case.report, score=untrusted, expected_turns=1)

    assert sqlite_instance._query_entries(ScoreEntry) == []
    assert sqlite_instance._query_entries(ScorableContentEntry) == []
    assert case.store.get_episode(run_id=case.report.run_id).finalized_at is None


async def test_cli_incomplete_status_retains_real_grade_without_complete_score_async(
    *,
    sqlite_instance: SQLiteMemory,
) -> None:
    case = await _capture_case_async(memory=sqlite_instance)
    raw = case.report.model_dump(mode="json")
    raw["status"] = NativeCliReportStatus.INCOMPLETE.value
    report = NativeCliRunReport.model_validate(raw)
    score = build_native_cli_report_score(report=report)

    snapshot = case.store.finalize_cli_episode_atomic(report=report, score=score, expected_turns=1)

    assert snapshot.score_status is ScoreStatus.UNDETERMINED
    assert any("report did not declare complete" in gap for gap in snapshot.gaps)
    assert (
        NativeCliRunReport.model_validate_json(
            sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
                snapshot.report_content_id
            ].value
        ).judgment.value
        == 0.75
    )

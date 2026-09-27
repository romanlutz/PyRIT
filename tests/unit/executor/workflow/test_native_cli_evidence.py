# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""DB-backed coding CLI capture with inert process bytes and real memory pieces."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
from pyrit.executor.workflow.native_cli_report_adapter import build_native_cli_run_report
from pyrit.models import Message
from pyrit.models.native_cli_report import NativeCliReportCleanup, NativeCliReportEventKind, NativeCliReportStatus
from pyrit.models.native_cyber_evidence import NativeCyberEpisodeStart
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import NativeCliTarget
from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.responses_contract import (
    GatewayCoverage,
    GatewayFrameKind,
    GatewayLimits,
    GatewayObservation,
    GatewayRoute,
)
from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliEventKind,
    NativeCliObservation,
    NativeCliProcessChunk,
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliStream,
)
from pyrit.score.float_scale.native_cli_report_scorer import build_native_cli_report_score
from tests.unit.prompt_target.gateway.test_codex_responses import FakeModelOnlyBackend
from tests.unit.prompt_target.target.test_native_cli_target import _codex, _config, _frame, _Launcher, _Sandbox

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


async def _start_sink_async(
    *,
    memory: SQLiteMemory,
    protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON,
    raw_byte_limit: int = 268_435_456,
    include_model_gateway: bool = False,
) -> NativeCliDatabaseEvidenceSink:
    store = memory.native_cyber_evidence
    run_id = str(uuid4())
    sink = NativeCliDatabaseEvidenceSink(
        store=store,
        run_id=run_id,
        turn_id=str(uuid4()),
        turn_index=1,
        protocol=protocol,
        include_model_gateway=include_model_gateway,
    )
    await asyncio.to_thread(
        store.create_episode,
        start=NativeCyberEpisodeStart(
            run_id=run_id,
            binding_name="inert_cli",
            binding_version="1",
            required_raw_streams=sink.required_raw_streams(
                protocol=protocol, include_model_gateway=include_model_gateway
            ),
            require_separate_tool_results=True,
            raw_byte_limit=raw_byte_limit,
        ),
    )
    await sink.start_async(started_at=datetime.now(UTC))
    return sink


async def test_cli_sink_records_exact_pipes_tool_phases_and_delayed_real_request_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance)
    store = sqlite_instance.native_cyber_evidence
    before = await asyncio.to_thread(store.get_episode, run_id=sink.run_id)
    assert before.turns[0].request_piece_ids == () and not before.events
    assert before.turns[0].source_turn_id == sink.turn_id

    stdout = _codex(tool=True)
    stderr = b"observed progress\xff\r\n"
    sandbox = _Sandbox(
        chunks=[
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[:17]),
            NativeCliProcessChunk(stream=NativeCliStream.STDERR, data=stderr),
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[17:]),
        ]
    )
    target = NativeCliTarget(run_config=_config(), launcher=_Launcher(sandbox=sandbox), evidence_sink=sink)
    request = Message.from_prompt(prompt="Inert instruction", role="user")
    response = await PromptNormalizer().send_prompt_async(message=request, target=target, conversation_id=str(uuid4()))
    assert target.last_run and target.last_run.outcome.coverage_complete and sandbox.stop_count == 1
    before_finish = await asyncio.to_thread(store.get_episode, run_id=sink.run_id)
    assert not before_finish.turns[0].request_piece_ids
    with pytest.raises(RuntimeError, match="before the turn is sealed"):
        await sink.read_report_events_async()

    await sink.finish_async(
        outcome=target.last_run.outcome,
        request_piece_ids=(request.get_piece().id,),
        response_piece_ids=(response.get_piece().id,),
    )
    episode = await asyncio.to_thread(store.get_episode, run_id=sink.run_id)
    assert episode.turns[0].source_complete and episode.turns[0].request_piece_ids == (request.get_piece().id,)
    assert episode.turns[0].response_piece_ids == (response.get_piece().id,)
    assert len(episode.events) > target.last_run.outcome.frame_count
    assert [event.sequence for event in episode.events] == list(range(1, len(episode.events) + 1))
    assert [event.observed_event_id for event in episode.events if event.observed_event_id == "cmd-1"] == [
        "cmd-1",
        "cmd-1",
        "cmd-1",
    ]
    assert episode.tools[0].start_sequence < episode.tools[0].completion_sequence < episode.tools[0].result_sequence
    assert episode.tools[0].request_sequence is None

    raw_by_kind = {raw.key.kind.value: raw for raw in episode.raw_streams}
    for kind, expected in (("stdout", stdout), ("stderr", stderr)):
        summary = raw_by_kind[kind]
        assert summary.received_bytes == len(expected) and summary.source_complete
        chunks = await asyncio.to_thread(
            store.read_raw_chunks, run_id=sink.run_id, stream_id=summary.stream_id, allow_sensitive=True
        )
        assert b"".join(chunk.data for chunk in chunks) == expected
    with pytest.raises(PermissionError, match="explicitly authorized"):
        await asyncio.to_thread(store.read_event_payloads, run_id=sink.run_id)
    events = await asyncio.to_thread(store.read_event_payloads, run_id=sink.run_id, allow_sensitive=True)
    assert [
        event.event.payload["raw_chunk_sequence"]
        for event in events
        if event.event.event_type == "native_cli.raw_chunk"
    ] == [
        1,
        2,
        3,
    ]
    frame_events = [event.event for event in events if event.event.payload.get("frame_number") is not None]
    assert frame_events[0].stream_offset == 0
    assert all(event.observed_stream_id == str(raw_by_kind["stdout"].stream_id) for event in frame_events)
    report_events = await sink.read_report_events_async()
    assert len(report_events) == target.last_run.outcome.frame_count + 2
    assert [event.sequence for event in report_events] == list(range(1, len(report_events) + 1))
    assert report_events[-1].kind is NativeCliReportEventKind.EOF
    assert len([event for event in report_events if event.source_event_id == "cmd-1"]) == 3
    assert all(not hasattr(event, "raw_frame") for event in report_events)
    report = build_native_cli_run_report(
        config=_config(),
        outcome=target.last_run.outcome,
        events=report_events,
        task_id="inert_cli",
        task_version="1",
        run_id=sink.run_id,
        turn_id=sink.turn_id,
        status=NativeCliReportStatus.INCOMPLETE,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        conversation_id=target.last_run.conversation_id,
        raw_evidence_ref=f"native-cyber-episode:{sink.run_id}",
        errors=("Original grader and model gateway are not yet qualified.",),
    )
    assert report.evidence.events == report_events
    assert build_native_cli_report_score(report=report).is_undetermined
    assert not await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
    with pytest.raises(ValueError, match="not finalized"):
        await asyncio.to_thread(store.get_finalized_episode, run_id=sink.run_id)


async def test_cli_sink_records_model_gateway_request_and_response_wire_in_same_episode_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance, include_model_gateway=True)
    route = GatewayRoute(run_id=sink.run_id, model="codex-fixture", guest_token="t" * 40)
    backend = FakeModelOnlyBackend()
    app = create_codex_responses_app(
        route=route,
        limits=GatewayLimits(),
        backend=backend,
        observation_callback=sink.record_gateway_observation_async,
    )
    request_bytes = b'{"model":"codex-fixture","input":"Inert model request","store":false}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.invalid") as client:
        reply = await client.post(
            "/v1/responses",
            content=request_bytes,
            headers={
                "Authorization": "Bearer " + route.guest_token,
                "X-PyRIT-Run-ID": sink.run_id,
                "Content-Type": "application/json",
            },
        )
    assert reply.status_code == 200 and backend.requests[0].run_id == sink.run_id

    target = NativeCliTarget(
        run_config=_config(),
        launcher=_Launcher(
            sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
        ),
        evidence_sink=sink,
    )
    request = Message.from_prompt(prompt="Inert instruction", role="user")
    response = await PromptNormalizer().send_prompt_async(message=request, target=target)
    assert target.last_run and target.last_run.outcome.coverage_complete
    await sink.finish_async(
        outcome=target.last_run.outcome,
        request_piece_ids=(request.get_piece().id,),
        response_piece_ids=(response.get_piece().id,),
    )
    store = sqlite_instance.native_cyber_evidence
    episode = await asyncio.to_thread(store.get_episode, run_id=sink.run_id)
    assert episode.turns[0].source_complete
    assert len(episode.raw_streams) == 5 and all(stream.source_complete for stream in episode.raw_streams)
    streams = {stream.key.observed_source_id: stream for stream in episode.raw_streams}
    for name, expected in (
        ("codex_exec_json.gateway.requests", request_bytes),
        ("codex_exec_json.gateway.responses", reply.content),
    ):
        chunks = await asyncio.to_thread(
            store.read_raw_chunks, run_id=sink.run_id, stream_id=streams[name].stream_id, allow_sensitive=True
        )
        assert b"".join(chunk.data for chunk in chunks) == expected
    events = await asyncio.to_thread(store.read_event_payloads, run_id=sink.run_id, allow_sensitive=True)
    gateway_events = [item for item in events if item.event.event_type.startswith("gateway.")]
    assert [item.event.event_type for item in gateway_events] == ["gateway.request", "gateway.response"]
    assert (
        gateway_events[0].event.payload["gateway_request_id"] == gateway_events[1].event.payload["gateway_request_id"]
    )
    assert all(item.source.value == "model" for item in gateway_events)
    report_events = await sink.read_report_events_async()
    assert report_events[-1].kind is NativeCliReportEventKind.EOF
    assert len(report_events) == target.last_run.outcome.frame_count + 1


async def test_cli_sink_missing_model_gateway_or_host_generated_error_is_required_gap_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance, include_model_gateway=True)
    await sink.record_gateway_observation_async(
        GatewayObservation(
            run_id=sink.run_id,
            request_id="model-1",
            kind=GatewayFrameKind.REQUEST,
            frame=b'{"model":"codex-fixture","input":"inert"}',
            coverage=frozenset(),
        )
    )
    await sink.record_gateway_observation_async(
        GatewayObservation(
            run_id=sink.run_id,
            request_id="model-1",
            kind=GatewayFrameKind.GATEWAY_ERROR,
            frame=b'event: error\ndata: {"error":"inert"}\n\n',
            coverage=frozenset({GatewayCoverage.FAILED}),
            error_code="backend_failed",
            status_code=502,
        )
    )
    await sink.finish_async(outcome=None)
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=sink.run_id)
    assert not episode.turns[0].source_complete
    assert any("host-generated failure" in gap for gap in episode.gaps)
    assert all(not stream.source_complete for stream in episode.raw_streams)
    assert any(event.event_type == "gateway.gateway_error" for event in episode.events)


async def test_cli_sink_clean_cli_without_observed_model_request_stays_incomplete_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance, include_model_gateway=True)
    target = NativeCliTarget(
        run_config=_config(),
        launcher=_Launcher(
            sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
        ),
        evidence_sink=sink,
    )
    request = Message.from_prompt(prompt="Inert instruction", role="user")
    response = await PromptNormalizer().send_prompt_async(message=request, target=target)
    assert target.last_run and target.last_run.outcome.coverage_complete
    await sink.finish_async(
        outcome=target.last_run.outcome,
        request_piece_ids=(request.get_piece().id,),
        response_piece_ids=(response.get_piece().id,),
    )
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=sink.run_id)
    assert not episode.turns[0].source_complete
    assert any("Model gateway request/response coverage" in gap for gap in episode.turns[0].gaps)
    assert all(not stream.source_complete for stream in episode.raw_streams)


async def test_cli_sink_preserves_repeated_claude_message_id_and_distinct_tool_result_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance, protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    stdout = b"".join(
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
                        {"type": "text", "text": "Working."},
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
                    "content": [{"type": "tool_result", "tool_use_id": "use-1", "content": "ok", "is_error": False}],
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
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout)])
    target = NativeCliTarget(
        run_config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE),
        launcher=_Launcher(sandbox=sandbox),
        evidence_sink=sink,
    )
    request = Message.from_prompt(prompt="Inert instruction", role="user")
    response = await PromptNormalizer().send_prompt_async(message=request, target=target)
    assert target.last_run and target.last_run.outcome.coverage_complete
    await sink.finish_async(
        outcome=target.last_run.outcome,
        request_piece_ids=(request.get_piece().id,),
        response_piece_ids=(response.get_piece().id,),
    )
    store = sqlite_instance.native_cyber_evidence
    episode = await asyncio.to_thread(store.get_episode, run_id=sink.run_id)
    assert episode.turns[0].source_complete
    assert len([event for event in episode.events if event.observed_event_id == "assistant-1"]) == 2
    assert episode.tools[0].call_id == "use-1" and episode.tools[0].request_sequence
    assert episode.tools[0].result_sequence > episode.tools[0].request_sequence
    assert episode.tools[0].start_sequence is None and episode.tools[0].completion_sequence is None
    raw = next(stream for stream in episode.raw_streams if stream.key.kind.value == "stdout")
    chunks = await asyncio.to_thread(
        store.read_raw_chunks, run_id=sink.run_id, stream_id=raw.stream_id, allow_sensitive=True
    )
    assert b"".join(chunk.data for chunk in chunks) == stdout
    report_events = await sink.read_report_events_async()
    repeated = [event for event in report_events if event.source_event_id == "assistant-1"]
    assert len(repeated) == 2
    assert repeated[0].frame_number == repeated[1].frame_number
    assert repeated[0].raw_frame_sha256 == repeated[1].raw_frame_sha256
    assert repeated[0].stdout_offset_bytes == repeated[1].stdout_offset_bytes


async def test_cli_sink_quota_loss_is_explicit_even_when_parser_exits_clean_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance, raw_byte_limit=8)
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
    target = NativeCliTarget(run_config=_config(), launcher=_Launcher(sandbox=sandbox), evidence_sink=sink)
    request = Message.from_prompt(prompt="Inert instruction", role="user")
    response = await PromptNormalizer().send_prompt_async(message=request, target=target)
    assert target.last_run and target.last_run.outcome.coverage_complete
    await sink.finish_async(
        outcome=target.last_run.outcome,
        request_piece_ids=(request.get_piece().id,),
        response_piece_ids=(response.get_piece().id,),
    )
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=sink.run_id)
    stdout = next(raw for raw in episode.raw_streams if raw.key.kind.value == "stdout")
    assert stdout.truncated and stdout.stored_bytes == 8
    assert stdout.omitted_bytes == stdout.received_bytes - 8
    assert not stdout.source_complete and stdout.gaps


async def test_cli_sink_failed_write_cannot_be_retried_as_complete_capture_async(
    *, sqlite_instance: SQLiteMemory
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance)
    with patch.object(sink._store, "append_raw", side_effect=OSError("inert DB write failed")):
        with pytest.raises(OSError, match="inert DB write failed"):
            await sink.record_raw_async(
                chunk=NativeCliRawChunk(sequence=1, stream=NativeCliStream.STDOUT, data=b"inert")
            )
    with pytest.raises(RuntimeError, match="not open"):
        await sink.record_raw_async(chunk=NativeCliRawChunk(sequence=1, stream=NativeCliStream.STDOUT, data=b"inert"))
    await sink.finish_async(outcome=None)
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=sink.run_id)
    assert not episode.turns[0].source_complete
    assert all(not raw.source_complete for raw in episode.raw_streams)


async def test_cli_sink_rejects_unobserved_stdout_frame_and_keeps_gap_async(*, sqlite_instance: SQLiteMemory) -> None:
    sink = await _start_sink_async(memory=sqlite_instance)
    await sink.record_raw_async(chunk=NativeCliRawChunk(sequence=1, stream=NativeCliStream.STDOUT, data=b"partial"))
    with pytest.raises(ValueError, match="consecutive"):
        await sink.record_event_async(
            event=NativeCliEvent(
                sequence=1,
                frame_number=2,
                raw_frame=b"partial",
                observation=NativeCliObservation(kind=NativeCliEventKind.MODEL_MESSAGE, text="inert"),
            )
        )
    await sink.finish_async(outcome=None)
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=sink.run_id)
    assert len(episode.events) == 1 and not episode.turns[0].source_complete
    assert next(raw for raw in episode.raw_streams if raw.key.kind.value == "stdout").stored_bytes == len(b"partial")

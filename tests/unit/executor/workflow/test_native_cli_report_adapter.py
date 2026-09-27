# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real CLI observations projected from fake sandbox bytes, with no model calls."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pytest

from pyrit.executor.workflow.native_cli_report_adapter import build_native_cli_run_report
from pyrit.models import Message
from pyrit.models.native_cli_report import (
    NativeCliArtifactReference,
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportEvent,
    NativeCliReportEventKind,
    NativeCliReportEventStatus,
    NativeCliReportProtocol,
    NativeCliReportStatus,
)
from pyrit.prompt_target import NativeCliTarget
from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliEventKind,
    NativeCliEventStatus,
    NativeCliObservation,
    NativeCliProcessChunk,
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliRunConfig,
    NativeCliRunOutcome,
    NativeCliStream,
)
from pyrit.prompt_target.native_cli_transport import NativeCliJsonlParser, NativeCliRunner

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _config(*, protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON) -> NativeCliRunConfig:
    return NativeCliRunConfig(
        protocol=protocol,
        cli_version="0.115.0" if protocol is NativeCliProtocol.CODEX_EXEC_JSON else "2.1.220",
        cli_profile="sandbox-locked",
        agent_workdir=PurePosixPath("/workspace/task"),
        model_gateway_endpoint="http://gateway.sandbox/v1",
        max_steps=3,
        timeout_seconds=1,
    )


def _frame(*, kind: str, **fields: object) -> bytes:
    return json.dumps({"type": kind, **fields}, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def _codex() -> bytes:
    return b"".join(
        (
            _frame(kind="thread.started", thread_id="thread-real-observed"),
            _frame(kind="turn.started"),
            _frame(
                kind="item.started",
                item={"id": "cmd-1", "type": "command_execution", "status": "in_progress", "command": "fixture"},
            ),
            _frame(
                kind="item.completed",
                item={
                    "id": "cmd-1",
                    "type": "command_execution",
                    "status": "completed",
                    "exit_code": 0,
                    "command": "fixture",
                    "aggregated_output": "",
                },
            ),
            _frame(
                kind="item.completed",
                item={"id": "message-real-observed", "type": "agent_message", "text": "Private fixture text."},
            ),
            _frame(kind="turn.completed"),
        )
    )


def _claude() -> bytes:
    return b"".join(
        (
            _frame(kind="system", subtype="init", session_id="claude-source", uuid="init-1"),
            _frame(
                kind="assistant",
                session_id="claude-source",
                uuid="assistant-1",
                message={
                    "id": "message-1",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Inspecting."},
                        {"type": "tool_use", "id": "toolu-1", "name": "Read", "input": {"path": "file.txt"}},
                    ],
                },
            ),
            _frame(
                kind="user",
                session_id="claude-source",
                uuid="user-1",
                message={
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu-1", "content": "Observed", "is_error": False}
                    ],
                },
            ),
            _frame(
                kind="assistant",
                session_id="claude-source",
                uuid="assistant-2",
                message={"id": "message-2", "role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ),
            _frame(
                kind="result",
                session_id="claude-source",
                uuid="result-1",
                subtype="success",
                is_error=False,
                result="Done.",
            ),
        )
    )


class _Recorder:
    def __init__(self) -> None:
        self.raw: list[NativeCliRawChunk] = []
        self.events: list[NativeCliEvent] = []

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        self.raw.append(chunk)

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        self.events.append(event)


class _FakeSandbox:
    def __init__(self, *, chunks: list[NativeCliProcessChunk], exit_code: int = 0) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.stopped = False

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        for chunk in self.chunks:
            yield chunk

    async def wait_async(self) -> int:
        return self.exit_code

    async def stop_async(self) -> None:
        self.stopped = True


class _FakeLauncher:
    def __init__(self, *, sandbox: _FakeSandbox) -> None:
        self.sandbox = sandbox
        self.calls: list[tuple[NativeCliRunConfig, str]] = []

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> _FakeSandbox:
        self.calls.append((config, prompt))
        return self.sandbox


def _grader() -> NativeCliOriginalJudgment:
    return NativeCliOriginalJudgment(
        grader_ref="OFFLINE original grader",
        grader_evidence_ref="OFFLINE grader evidence",
        complete=True,
        value=0.75,
        rationale="Original fixture grader saw the retained artifact.",
    )


def _artifact() -> NativeCliArtifactReference:
    return NativeCliArtifactReference(
        name="fixture.txt",
        evidence_ref="OFFLINE artifact content",
        sha256=hashlib.sha256(b"OFFLINE/SIMULATED artifact").hexdigest(),
        size_bytes=len(b"OFFLINE/SIMULATED artifact"),
    )


def _outcome(*, parser: NativeCliJsonlParser, size_bytes: int) -> NativeCliRunOutcome:
    return NativeCliRunOutcome(
        exit_code=0,
        terminal_observed=parser.terminal_observed,
        coverage_complete=parser.coverage_complete,
        source_session_id=parser.source_session_id,
        observed_steps=parser.observed_steps,
        frame_count=parser.frame_count,
        raw_chunk_count=1,
        raw_stdout_bytes=size_bytes,
        raw_stderr_bytes=0,
        gaps=parser.gaps,
    )


@pytest.mark.usefixtures("patch_central_database")
async def test_real_target_outcome_maps_to_cli_only_report_with_exact_source_ids_async() -> None:
    stdout = _codex()
    sandbox = _FakeSandbox(
        chunks=[
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[:18]),
            NativeCliProcessChunk(stream=NativeCliStream.STDERR, data=b"progress\xff\n"),
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[18:]),
        ]
    )
    launcher, sink, config = _FakeLauncher(sandbox=sandbox), _Recorder(), _config()
    target = NativeCliTarget(run_config=config, launcher=launcher, evidence_sink=sink)
    request = Message.from_prompt(prompt="Prepared fixture request", role="user")
    response = await target.send_prompt_async(message=request)
    assert [item.get_value() for item in response] == ["Private fixture text."]
    assert launcher.calls == [(config, "Prepared fixture request")] and sandbox.stopped
    assert target.last_run is not None

    report = build_native_cli_run_report(
        config=config,
        outcome=target.last_run.outcome,
        events=sink.events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        conversation_id=target.last_run.conversation_id,
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        artifacts=(_artifact(),),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    assert report.protocol is NativeCliReportProtocol.CODEX_EXEC_JSON
    assert report.task_version == "benchmark-v1"
    assert report.cli_version == "0.115.0" and report.cli_profile == "sandbox-locked"
    assert report.evidence.source_session_id == "thread-real-observed"
    assert report.evidence.exit_code == 0 and report.evidence.observed_steps == 1
    assert report.evidence.raw_chunk_count == 3
    assert report.evidence.raw_stderr_bytes == len(b"progress\xff\n")
    assert report.evidence.coverage_complete and report.evidence.gaps == ()
    assert len(report.evidence.events) == len(sink.events)
    assert [event.sequence for event in report.evidence.events] == list(range(1, len(sink.events) + 1))
    command = next(event for event in report.evidence.events if event.kind is NativeCliReportEventKind.TOOL_STARTED)
    assert command.source_tool_id == command.source_event_id == "cmd-1"
    assert command.status is NativeCliReportEventStatus.RUNNING and command.source_status == "in_progress"
    message = next(event for event in report.evidence.events if event.kind is NativeCliReportEventKind.MODEL_MESSAGE)
    assert message.source_event_id == "message-real-observed"
    assert sink.events[0].raw_frame is not None
    assert report.evidence.events[0].raw_frame_sha256 == hashlib.sha256(sink.events[0].raw_frame).hexdigest()
    assert report.evidence.events[-1].kind is NativeCliReportEventKind.EOF
    assert report.evidence.events[-1].source_event_id is None
    assert report.judgment is not None and report.judgment.value == 0.75
    assert report.artifacts[0].evidence_ref == "OFFLINE artifact content"
    assert "Private fixture text." not in report.canonical_json()
    assert b"".join(chunk.data for chunk in sink.raw if chunk.stream is NativeCliStream.STDOUT) == stdout


def test_adapter_requires_caller_owned_task_version_without_a_fallback() -> None:
    parameter = inspect.signature(build_native_cli_run_report).parameters["task_version"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match="task version"):
        build_native_cli_run_report(
            config=_config(),
            outcome=None,
            events=(),
            task_id="task-fixture",
            task_version=" ",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.ERROR,
            cleanup=NativeCliReportCleanup.NOT_OPENED,
            errors=("No CLI process launched.",),
        )


def test_report_protocol_and_observation_vocabularies_match_parser_exactly() -> None:
    assert {item.value for item in NativeCliReportProtocol} == {item.value for item in NativeCliProtocol}
    assert {item.value for item in NativeCliReportEventKind} == {item.value for item in NativeCliEventKind}
    assert {item.value for item in NativeCliReportEventStatus} == {item.value for item in NativeCliEventStatus}


async def test_persisted_summary_iterator_builds_same_report_without_raw_frames_async() -> None:
    stdout = _claude()
    sink = _Recorder()
    config = _config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    outcome = await NativeCliRunner(
        launcher=_FakeLauncher(
            sandbox=_FakeSandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout)])
        ),
        sink=sink,
    ).run_async(config=config, prompt="Offline fixture")
    fields = {
        "config": config,
        "outcome": outcome,
        "task_id": "task-fixture",
        "task_version": "benchmark-v1",
        "run_id": "run-fixture",
        "turn_id": "turn-fixture",
        "status": NativeCliReportStatus.COMPLETED,
        "cleanup": NativeCliReportCleanup.CLOSED,
        "simulated": True,
        "judgment": _grader(),
        "raw_evidence_ref": "OFFLINE raw chunks",
    }
    original = build_native_cli_run_report(events=sink.events, **fields)
    rows = tuple(NativeCliReportEvent.model_validate(item.model_dump(mode="json")) for item in original.evidence.events)
    sink.events.clear()
    sink.raw.clear()
    from_rows = build_native_cli_run_report(events=(row for row in rows), **fields)
    assert from_rows.sha256() == original.sha256()
    assert from_rows.canonical_json() == original.canonical_json()
    assert all(not hasattr(row, "raw_frame") for row in rows)
    assert "Inspecting." not in from_rows.canonical_json()
    assert len([item for item in rows if item.frame_number == 2]) == 2
    same_frame = [item for item in rows if item.frame_number == 2]
    assert same_frame[0].raw_frame_sha256 == same_frame[1].raw_frame_sha256
    assert same_frame[0].raw_frame_size_bytes == same_frame[1].raw_frame_size_bytes
    assert same_frame[0].stdout_offset_bytes == same_frame[1].stdout_offset_bytes
    assert (
        sum(item.raw_frame_size_bytes or 0 for item in rows if item.frame_number and item.frame_number != 2)
        + (same_frame[0].raw_frame_size_bytes or 0)
        == outcome.raw_stdout_bytes
    )


@pytest.mark.parametrize("field", ["raw_frame_sha256", "raw_frame_size_bytes", "stdout_offset_bytes"])
def test_persisted_frame_missing_digest_length_or_offset_is_rejected(field: str) -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=_codex()), *parser.finish())
    first = build_native_cli_run_report(
        config=_config(),
        outcome=_outcome(parser=parser, size_bytes=len(_codex())),
        events=events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    summaries = list(first.evidence.events)
    summaries[0] = summaries[0].model_copy(update={field: None})
    with pytest.raises(ValueError, match="frame number, digest, size, and stdout offset"):
        build_native_cli_run_report(
            config=_config(),
            outcome=_outcome(parser=parser, size_bytes=len(_codex())),
            events=(summary for summary in summaries),
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )


def test_persisted_model_message_without_source_id_is_not_normalized() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=_codex()), *parser.finish())
    outcome = _outcome(parser=parser, size_bytes=len(_codex()))
    first = build_native_cli_run_report(
        config=_config(),
        outcome=outcome,
        events=events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    summaries = list(first.evidence.events)
    message_index = next(
        index for index, item in enumerate(summaries) if item.kind is NativeCliReportEventKind.MODEL_MESSAGE
    )
    summaries[message_index] = summaries[message_index].model_copy(update={"source_event_id": None})
    with pytest.raises(ValueError, match="observed source event ID"):
        build_native_cli_run_report(
            config=_config(),
            outcome=outcome,
            events=(summary for summary in summaries),
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )


def test_persisted_codex_tool_status_cannot_be_backfilled_from_completion_kind() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=_codex()), *parser.finish())
    outcome = _outcome(parser=parser, size_bytes=len(_codex()))
    original = build_native_cli_run_report(
        config=_config(),
        outcome=outcome,
        events=events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    summaries = list(original.evidence.events)
    index = next(index for index, item in enumerate(summaries) if item.kind is NativeCliReportEventKind.TOOL_COMPLETED)
    summaries[index] = summaries[index].model_copy(update={"source_status": None})
    with pytest.raises(ValueError, match="observed name and native status"):
        build_native_cli_run_report(
            config=_config(),
            outcome=outcome,
            events=(summary for summary in summaries),
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )


def test_persisted_same_frame_tool_request_cannot_drop_all_digest_and_offset_fields() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    raw = _claude()
    events = (*parser.feed(data=raw), *parser.finish())
    config = _config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    outcome = _outcome(parser=parser, size_bytes=len(raw))
    original = build_native_cli_run_report(
        config=config,
        outcome=outcome,
        events=events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    summaries = list(original.evidence.events)
    tool_index = next(
        index for index, item in enumerate(summaries) if item.kind is NativeCliReportEventKind.TOOL_REQUESTED
    )
    summaries[tool_index] = summaries[tool_index].model_copy(
        update={
            "frame_number": None,
            "raw_frame_sha256": None,
            "raw_frame_size_bytes": None,
            "stdout_offset_bytes": None,
        }
    )
    with pytest.raises(ValueError, match="requires its actual JSONL frame"):
        build_native_cli_run_report(
            config=config,
            outcome=outcome,
            events=(summary for summary in summaries),
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )


def test_native_events_without_prior_frame_or_ordinal_cannot_infer_missing_offset() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=_codex()), *parser.finish())
    second = replace(events[1], sequence=1)
    with pytest.raises(ValueError, match="Cannot infer a missing stdout frame offset"):
        build_native_cli_run_report(
            config=_config(),
            outcome=None,
            events=[second],
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.INCOMPLETE,
            cleanup=NativeCliReportCleanup.UNKNOWN,
        )
    with pytest.raises(ValueError, match="ordinals"):
        build_native_cli_run_report(
            config=_config(),
            outcome=None,
            events=[replace(events[0], sequence=2)],
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.INCOMPLETE,
            cleanup=NativeCliReportCleanup.UNKNOWN,
        )


def test_persisted_stdout_offset_or_byte_count_cannot_forge_complete_coverage() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=_codex()), *parser.finish())
    outcome = _outcome(parser=parser, size_bytes=len(_codex()))
    first = build_native_cli_run_report(
        config=_config(),
        outcome=outcome,
        events=events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    summaries = list(first.evidence.events)
    summaries[0] = summaries[0].model_copy(update={"stdout_offset_bytes": 2})
    with pytest.raises(ValueError, match="contiguous stdout offsets"):
        build_native_cli_run_report(
            config=_config(),
            outcome=outcome,
            events=(summary for summary in summaries),
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )
    with pytest.raises(ValueError, match="frame lengths must match"):
        build_native_cli_run_report(
            config=_config(),
            outcome=replace(outcome, raw_stdout_bytes=outcome.raw_stdout_bytes + 1),
            events=(summary for summary in first.evidence.events),
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )


def test_persisted_frame_gap_can_be_retained_only_with_explicitly_incomplete_coverage() -> None:
    observed = NativeCliReportEvent(
        sequence=1,
        frame_number=2,
        raw_frame_sha256="a" * 64,
        raw_frame_size_bytes=18,
        stdout_offset_bytes=42,
        kind=NativeCliReportEventKind.MODEL_MESSAGE,
        status=NativeCliReportEventStatus.UNKNOWN,
        source_event_id="item-2",
    )
    report = build_native_cli_run_report(
        config=_config(),
        outcome=None,
        events=(event for event in (observed,)),
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.INCOMPLETE,
        cleanup=NativeCliReportCleanup.UNKNOWN,
    )
    assert report.evidence.events == (observed,)
    assert report.evidence.source_session_id is None
    assert report.evidence.exit_code is None
    assert not report.evidence.coverage_complete
    assert report.evidence.gaps == ("No native CLI process outcome was acquired.",)


async def test_claude_process_and_tool_requests_map_without_ghcp_execution_events_async() -> None:
    raw = _claude()
    sandbox = _FakeSandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)])
    sink = _Recorder()
    config = _config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    outcome = await NativeCliRunner(launcher=_FakeLauncher(sandbox=sandbox), sink=sink).run_async(
        config=config, prompt="Offline fixture"
    )
    assert outcome.coverage_complete and outcome.observed_steps == 2
    report = build_native_cli_run_report(
        config=config,
        outcome=outcome,
        events=sink.events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.COMPLETED,
        cleanup=NativeCliReportCleanup.CLOSED,
        simulated=True,
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    assert report.protocol is NativeCliReportProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE
    assert report.evidence.source_session_id == "claude-source"
    assert report.evidence.observed_steps == 2
    requests = [event for event in report.evidence.events if event.kind is NativeCliReportEventKind.TOOL_REQUESTED]
    results = [event for event in report.evidence.events if event.kind is NativeCliReportEventKind.TOOL_RESULT]
    assert len(requests) == len(results) == 1
    assert requests[0].source_tool_id == results[0].source_tool_id == "toolu-1"
    assert requests[0].status is NativeCliReportEventStatus.REQUESTED
    assert results[0].status is NativeCliReportEventStatus.COMPLETED
    assert results[0].source_event_id == "user-1" and results[0].source_message_id is None
    assert not any(
        event.kind in {NativeCliReportEventKind.TOOL_STARTED, NativeCliReportEventKind.TOOL_COMPLETED}
        for event in report.evidence.events
    )
    assert report.evidence.events[-2].kind is NativeCliReportEventKind.RUN_FINISHED
    assert report.evidence.events[-1].kind is NativeCliReportEventKind.EOF


async def test_claude_without_init_retains_observed_session_id_but_no_complete_coverage_async() -> None:
    raw = b"".join(
        (
            _frame(
                kind="assistant",
                session_id="observed-without-init",
                uuid="assistant-1",
                message={"id": "message-1", "role": "assistant", "content": [{"type": "text", "text": "Partial."}]},
            ),
            _frame(
                kind="result",
                session_id="observed-without-init",
                uuid="result-1",
                subtype="success",
                is_error=False,
                result="Partial.",
            ),
        )
    )
    sink = _Recorder()
    config = _config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    sandbox = _FakeSandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)])
    outcome = await NativeCliRunner(launcher=_FakeLauncher(sandbox=sandbox), sink=sink).run_async(
        config=config, prompt="Offline fixture"
    )
    assert not outcome.coverage_complete and outcome.source_session_id is None
    report = build_native_cli_run_report(
        config=config,
        outcome=outcome,
        events=sink.events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.INCOMPLETE,
        cleanup=NativeCliReportCleanup.CLOSED,
        raw_evidence_ref="OFFLINE raw chunks",
    )
    assert report.evidence.source_session_id == "observed-without-init"
    assert not report.evidence.coverage_complete
    assert not any(event.kind is NativeCliReportEventKind.SESSION_STARTED for event in report.evidence.events)
    assert any("session start" in gap for gap in report.evidence.gaps)


async def test_nonzero_process_exit_with_actual_partial_gaps_cannot_build_completed_report_async() -> None:
    sandbox = _FakeSandbox(
        chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())],
        exit_code=7,
    )
    sink = _Recorder()
    outcome = await NativeCliRunner(launcher=_FakeLauncher(sandbox=sandbox), sink=sink).run_async(
        config=_config(), prompt="Offline fixture"
    )
    assert not outcome.coverage_complete and "exited with code 7" in outcome.gaps[-1]
    with pytest.raises(ValueError, match="Completed CLI results"):
        build_native_cli_run_report(
            config=_config(),
            outcome=outcome,
            events=sink.events,
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )
    report = build_native_cli_run_report(
        config=_config(),
        outcome=outcome,
        events=sink.events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.ERROR,
        cleanup=NativeCliReportCleanup.CLOSED,
        errors=("Nonzero CLI exit.",),
        judgment=_grader(),
        raw_evidence_ref="OFFLINE raw chunks",
    )
    assert report.status is NativeCliReportStatus.ERROR
    assert report.judgment is not None and report.judgment.value == 0.75
    assert report.evidence.exit_code == 7 and not report.evidence.coverage_complete
    assert any("code 7" in gap for gap in report.evidence.gaps)


def test_cancelled_report_without_outcome_keeps_actual_session_and_frame_digests() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = parser.feed(data=_frame(kind="thread.started", thread_id="source-only") + _frame(kind="turn.started"))
    report = build_native_cli_run_report(
        config=_config(),
        outcome=None,
        events=events,
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.CANCELLED,
        cleanup=NativeCliReportCleanup.UNKNOWN,
        errors=("Caller cancelled the active sandbox.",),
    )
    assert report.evidence.source_session_id == "source-only"
    assert report.evidence.exit_code is None and report.evidence.observed_steps is None
    assert report.evidence.frame_count is None and not report.evidence.terminal_observed
    assert events[0].raw_frame is not None
    assert report.evidence.events[0].raw_frame_sha256 == hashlib.sha256(events[0].raw_frame).hexdigest()
    assert report.evidence.events[1].kind is NativeCliReportEventKind.TURN_STARTED
    assert report.evidence.gaps == ("No native CLI process outcome was acquired.",)


def test_prelaunch_failure_preserves_error_without_synthesizing_process_outcome() -> None:
    actual_error = NativeCliEvent(
        sequence=1,
        frame_number=None,
        raw_frame=None,
        observation=NativeCliObservation(
            kind=NativeCliEventKind.ERROR,
            status=NativeCliEventStatus.FAILED,
            detail="Inert launcher failed before returning a process.",
        ),
    )
    report = build_native_cli_run_report(
        config=_config(),
        outcome=None,
        events=[actual_error],
        task_id="task-fixture",
        task_version="benchmark-v1",
        run_id="run-fixture",
        turn_id="turn-fixture",
        status=NativeCliReportStatus.ERROR,
        cleanup=NativeCliReportCleanup.NOT_OPENED,
        errors=("Inert launcher failed.",),
    )
    assert report.evidence.exit_code is None and report.evidence.source_session_id is None
    assert report.evidence.events[0].source_event_id is None
    assert report.evidence.events[0].raw_frame_sha256 is None
    assert report.evidence.events[0].detail == "Inert launcher failed before returning a process."


def test_missing_recorder_event_or_foreign_session_is_rejected_not_backfilled() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=_codex()), *parser.finish())
    assert parser.coverage_complete
    outcome = _outcome(parser=parser, size_bytes=len(_codex()))
    with pytest.raises(ValueError, match="Complete CLI coverage"):
        build_native_cli_run_report(
            config=_config(),
            outcome=outcome,
            events=events[:-1],
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )

    with pytest.raises(ValueError, match="source session ID"):
        build_native_cli_run_report(
            config=_config(),
            outcome=replace(outcome, source_session_id="fabricated"),
            events=events,
            task_id="task-fixture",
            task_version="benchmark-v1",
            run_id="run-fixture",
            turn_id="turn-fixture",
            status=NativeCliReportStatus.COMPLETED,
            cleanup=NativeCliReportCleanup.CLOSED,
            simulated=True,
            judgment=_grader(),
            raw_evidence_ref="OFFLINE raw chunks",
        )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline JSONL fixtures for provider events; no native CLI is installed or run."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import pytest

from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliEventKind,
    NativeCliEventStatus,
    NativeCliProtocol,
    NativeCliRunConfig,
)
from pyrit.prompt_target.native_cli_transport import NativeCliJsonlParser


def _config(
    *, protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON, max_steps: int = 4
) -> NativeCliRunConfig:
    return NativeCliRunConfig(
        protocol=protocol,
        cli_version="2.1.220",
        cli_profile="sandbox-locked",
        agent_workdir=PurePosixPath("/workspace/task"),
        model_gateway_endpoint="http://gateway.sandbox/v1",
        max_steps=max_steps,
        timeout_seconds=2,
    )


def _frame(*, kind: str, **fields: object) -> bytes:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _of_kind(*, events: tuple[NativeCliEvent, ...], kind: NativeCliEventKind) -> list[NativeCliEvent]:
    return [event for event in events if event.observation.kind is kind]


def test_codex_interleaved_tool_items_have_source_ids_order_and_real_results() -> None:
    parser = NativeCliJsonlParser(config=_config())
    raw = b"".join(
        (
            _frame(kind="thread.started", thread_id="thread-1"),
            _frame(kind="turn.started"),
            _frame(
                kind="item.started",
                item={"id": "cmd-1", "type": "command_execution", "command": "printf fixture", "status": "in_progress"},
            ),
            _frame(
                kind="item.started",
                item={"id": "mcp-2", "type": "mcp_tool_call", "arguments": {"path": "sample"}, "status": "in_progress"},
            ),
            _frame(kind="item.updated", item={"id": "cmd-1", "type": "command_execution", "status": "in_progress"}),
            _frame(
                kind="item.completed",
                item={
                    "id": "mcp-2",
                    "type": "mcp_tool_call",
                    "status": "completed",
                    "result": {"content": "observed"},
                },
            ),
            _frame(
                kind="item.completed",
                item={
                    "id": "cmd-1",
                    "type": "command_execution",
                    "command": "printf fixture",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": "fixture",
                },
            ),
            _frame(kind="item.completed", item={"id": "msg-3", "type": "agent_message", "text": "Finished."}),
            _frame(kind="turn.completed", usage={"input_tokens": 5}),
        )
    )
    events = (*parser.feed(data=raw), *parser.finish())

    assert parser.coverage_complete and parser.source_session_id == "thread-1"
    assert parser.observed_steps == 1
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert [
        event.observation.source_tool_id for event in _of_kind(events=events, kind=NativeCliEventKind.TOOL_STARTED)
    ] == [
        "cmd-1",
        "mcp-2",
    ]
    assert [
        event.observation.source_tool_id for event in _of_kind(events=events, kind=NativeCliEventKind.TOOL_COMPLETED)
    ] == [
        "mcp-2",
        "cmd-1",
    ]
    results = _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)
    assert [event.observation.source_tool_id for event in results] == ["mcp-2", "cmd-1"]
    assert [event.observation.result for event in results] == [{"content": "observed"}, "fixture"]
    assert results[-1].observation.exit_code == 0
    assert _of_kind(events=events, kind=NativeCliEventKind.MODEL_MESSAGE)[0].observation.text == "Finished."
    assert events[-1].observation.kind is NativeCliEventKind.EOF
    assert not _of_kind(events=events, kind=NativeCliEventKind.TOOL_REQUESTED)


def test_claude_complete_blocks_correlate_subagent_tools_without_invented_starts() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    raw = b"".join(
        (
            _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
            _frame(
                kind="assistant",
                session_id="claude-1",
                uuid="assistant-1",
                parent_tool_use_id=None,
                message={
                    "id": "msg-1",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "toolu-agent", "name": "Agent", "input": {"task": "inspect"}},
                    ],
                },
            ),
            _frame(
                kind="assistant",
                session_id="claude-1",
                uuid="assistant-2",
                parent_tool_use_id="toolu-agent",
                message={
                    "id": "msg-2",
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu-read", "name": "Read", "input": {"path": "a.txt"}}],
                },
            ),
            _frame(
                kind="user",
                session_id="claude-1",
                uuid="result-1",
                parent_tool_use_id="toolu-agent",
                message={
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu-read", "content": "file", "is_error": False}
                    ],
                },
            ),
            _frame(
                kind="user",
                session_id="claude-1",
                uuid="result-2",
                parent_tool_use_id=None,
                message={
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu-agent", "content": "inspected", "is_error": False}
                    ],
                },
            ),
            _frame(
                kind="assistant",
                session_id="claude-1",
                uuid="assistant-3",
                parent_tool_use_id=None,
                message={"id": "msg-3", "role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ),
            _frame(
                kind="result", session_id="claude-1", uuid="end-1", subtype="success", is_error=False, result="Done."
            ),
        )
    )
    events = (*parser.feed(data=raw), *parser.finish())

    assert parser.coverage_complete and parser.observed_steps == 3
    assert [event.observation.text for event in _of_kind(events=events, kind=NativeCliEventKind.MODEL_MESSAGE)] == [
        "Checking.",
        "Done.",
    ]
    requests = _of_kind(events=events, kind=NativeCliEventKind.TOOL_REQUESTED)
    results = _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)
    assert [event.observation.source_tool_id for event in requests] == ["toolu-agent", "toolu-read"]
    assert [event.observation.source_tool_id for event in results] == ["toolu-read", "toolu-agent"]
    assert requests[1].observation.parent_tool_use_id == results[0].observation.parent_tool_use_id == "toolu-agent"
    assert requests[0].observation.source_message_id == "msg-1"
    assert results[0].observation.source_event_id == "result-1"
    assert all(event.observation.status is NativeCliEventStatus.COMPLETED for event in results)
    assert not _of_kind(events=events, kind=NativeCliEventKind.TOOL_STARTED)
    assert not _of_kind(events=events, kind=NativeCliEventKind.TOOL_COMPLETED)
    assert _of_kind(events=events, kind=NativeCliEventKind.RUN_FINISHED)[0].observation.text == "Done."


@pytest.mark.parametrize("raw", [b"{bad json}\n", b"\xff\n", b"[]\n", b"\n"])
def test_invalid_jsonl_frame_retains_exact_bytes_and_reports_gap(raw: bytes) -> None:
    parser = NativeCliJsonlParser(config=_config())
    first = parser.feed(data=raw)
    assert first[0].raw_frame == raw
    assert first[0].observation.kind in {NativeCliEventKind.ERROR, NativeCliEventKind.PARTIAL}
    events = (*first, *parser.finish())
    assert not parser.coverage_complete and parser.gaps
    assert events[-1].observation.kind is NativeCliEventKind.EOF


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
def test_eof_reports_unfinished_tool_and_missing_terminal_without_synthesizing_them(
    protocol: NativeCliProtocol,
) -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=protocol))
    if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
        raw = b"".join(
            (
                _frame(kind="thread.started", thread_id="thread-1"),
                _frame(kind="turn.started"),
                _frame(
                    kind="item.started",
                    item={"id": "tool-1", "type": "command_execution", "command": "true", "status": "in_progress"},
                ),
            )
        )
    else:
        raw = b"".join(
            (
                _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                _frame(
                    kind="assistant",
                    session_id="claude-1",
                    uuid="msg-uuid",
                    parent_tool_use_id=None,
                    message={
                        "id": "msg-1",
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "a"}}],
                    },
                ),
            )
        )
    events = (*parser.feed(data=raw), *parser.finish())
    assert not parser.coverage_complete and not parser.terminal_observed
    assert any("tool-1" in gap and "EOF" in gap for gap in parser.gaps)
    assert any("terminal" in gap for gap in parser.gaps)
    assert not _of_kind(events=events, kind=NativeCliEventKind.TOOL_COMPLETED)
    assert not _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)


def test_codex_unobserved_start_and_unknown_completion_status_remain_partial() -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="thread.started", thread_id="thread-1"),
                    _frame(kind="turn.started"),
                    _frame(
                        kind="item.completed",
                        item={"id": "orphan", "type": "command_execution", "aggregated_output": "observed"},
                    ),
                    _frame(kind="turn.completed"),
                )
            )
        ),
        *parser.finish(),
    )
    completed = _of_kind(events=events, kind=NativeCliEventKind.TOOL_COMPLETED)[0]
    assert completed.observation.source_tool_id == "orphan"
    assert completed.observation.status is NativeCliEventStatus.UNKNOWN
    assert not _of_kind(events=events, kind=NativeCliEventKind.TOOL_STARTED)
    assert any("without an observed start" in gap for gap in parser.gaps)
    assert any("completion status or exit_code" in gap for gap in parser.gaps)
    assert not parser.coverage_complete


@pytest.mark.parametrize(
    ("output_fields", "has_result"),
    [({}, False), ({"aggregated_output": None}, False), ({"aggregated_output": ""}, True)],
)
def test_codex_completed_command_requires_observed_result_at_eof(
    output_fields: dict[str, object], has_result: bool
) -> None:
    parser = NativeCliJsonlParser(config=_config())
    item: dict[str, object] = {
        "id": "cmd-1",
        "type": "command_execution",
        "command": "printf fixture",
        "status": "completed",
        "exit_code": 0,
    }
    item.update(output_fields)
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="thread.started", thread_id="thread-1"),
                    _frame(kind="turn.started"),
                    _frame(
                        kind="item.started",
                        item={
                            "id": "cmd-1",
                            "type": "command_execution",
                            "command": "printf fixture",
                            "status": "in_progress",
                        },
                    ),
                    _frame(kind="item.completed", item=item),
                    _frame(kind="turn.completed"),
                )
            )
        ),
        *parser.finish(),
    )
    completed = _of_kind(events=events, kind=NativeCliEventKind.TOOL_COMPLETED)
    results = _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)
    assert len(completed) == 1 and completed[0].observation.source_tool_id == "cmd-1"
    assert len(results) == int(has_result)
    assert parser.terminal_observed and parser.coverage_complete is has_result
    if has_result:
        assert results[0].observation.result == ""
    else:
        assert any("cmd-1" in gap and "result" in gap for gap in parser.gaps)
        assert any(
            event.observation.source_tool_id == "cmd-1"
            for event in _of_kind(events=events, kind=NativeCliEventKind.PARTIAL)
        )


@pytest.mark.parametrize(
    ("tool_type", "output_fields", "has_result"),
    [
        ("file_change", {"changes": None}, False),
        ("file_change", {"changes": "not a change list"}, False),
        ("file_change", {"changes": []}, True),
        ("mcp_tool_call", {"result": None}, False),
        ("mcp_tool_call", {"result": {}}, True),
        ("web_search", {"results": None}, False),
        ("web_search", {"results": []}, True),
    ],
)
def test_codex_other_tool_results_require_observed_supported_payload(
    tool_type: str, output_fields: dict[str, object], has_result: bool
) -> None:
    parser = NativeCliJsonlParser(config=_config())
    completed_item = {"id": "tool-1", "type": tool_type, "status": "completed", **output_fields}
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="thread.started", thread_id="thread-1"),
                    _frame(kind="turn.started"),
                    _frame(kind="item.started", item={"id": "tool-1", "type": tool_type, "status": "in_progress"}),
                    _frame(kind="item.completed", item=completed_item),
                    _frame(kind="turn.completed"),
                )
            )
        ),
        *parser.finish(),
    )
    assert len(_of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)) == int(has_result)
    assert parser.coverage_complete is has_result
    if not has_result:
        assert any("tool-1" in gap and "result" in gap for gap in parser.gaps)


def test_unknown_frames_and_unselected_token_deltas_cannot_claim_complete_coverage() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                    _frame(kind="stream_event", event={"type": "content_block_delta", "delta": {"text": "partial"}}),
                    _frame(kind="future_protocol", value={"unknown": True}),
                    _frame(
                        kind="result",
                        session_id="claude-1",
                        uuid="end-1",
                        subtype="success",
                        is_error=False,
                        result="real terminal",
                    ),
                )
            )
        ),
        *parser.finish(),
    )
    assert len(_of_kind(events=events, kind=NativeCliEventKind.PARTIAL)) == 2
    assert not _of_kind(events=events, kind=NativeCliEventKind.MODEL_MESSAGE)
    assert parser.terminal_observed and not parser.coverage_complete
    assert all(event.raw_frame for event in _of_kind(events=events, kind=NativeCliEventKind.PARTIAL))


def test_claude_result_and_tool_result_without_documented_status_are_not_success() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                    _frame(
                        kind="assistant",
                        session_id="claude-1",
                        uuid="assistant-1",
                        message={
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}],
                        },
                    ),
                    _frame(
                        kind="user",
                        session_id="claude-1",
                        uuid="user-1",
                        message={
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": ""}],
                        },
                    ),
                    _frame(kind="result", session_id="claude-1", uuid="end-1", subtype="success", result="Maybe"),
                )
            )
        ),
        *parser.finish(),
    )
    result = _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)[0]
    assert result.observation.status is NativeCliEventStatus.UNKNOWN
    assert not parser.terminal_observed and not parser.coverage_complete
    assert not _of_kind(events=events, kind=NativeCliEventKind.RUN_FINISHED)


def test_failed_claude_tool_result_is_observed_even_when_run_finishes() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                    _frame(
                        kind="assistant",
                        session_id="claude-1",
                        uuid="assistant-1",
                        message={
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}],
                        },
                    ),
                    _frame(
                        kind="user",
                        session_id="claude-1",
                        uuid="user-1",
                        message={
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "tool-1",
                                    "content": "not found",
                                    "is_error": True,
                                }
                            ],
                        },
                    ),
                    _frame(
                        kind="result",
                        session_id="claude-1",
                        uuid="end-1",
                        subtype="success",
                        is_error=False,
                        result="Could not read it.",
                    ),
                )
            )
        ),
        *parser.finish(),
    )
    assert parser.coverage_complete
    assert (
        _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)[0].observation.status
        is NativeCliEventStatus.FAILED
    )
    assert _of_kind(events=events, kind=NativeCliEventKind.RUN_FINISHED)[0].observation.text == "Could not read it."


def test_documented_claude_error_result_is_not_a_successful_terminal() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                    _frame(
                        kind="result",
                        session_id="claude-1",
                        uuid="end-1",
                        subtype="error_during_execution",
                        is_error=True,
                        result="offline fixture failure",
                    ),
                )
            )
        ),
        *parser.finish(),
    )
    assert not parser.coverage_complete
    assert any(event.observation.kind is NativeCliEventKind.ERROR for event in events)
    assert not _of_kind(events=events, kind=NativeCliEventKind.RUN_FINISHED)


def test_claude_foreign_session_message_is_preserved_only_as_partial() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    events = parser.feed(
        data=b"".join(
            (
                _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                _frame(
                    kind="assistant",
                    session_id="claude-foreign",
                    uuid="foreign-1",
                    message={
                        "id": "foreign-msg",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "untrusted"}],
                    },
                ),
            )
        )
    )
    assert not _of_kind(events=events, kind=NativeCliEventKind.MODEL_MESSAGE)
    assert _of_kind(events=events, kind=NativeCliEventKind.PARTIAL)[0].raw_frame is not None
    assert "different session" in parser.gaps[0]


def test_codex_repeated_turns_and_final_line_without_newline() -> None:
    parser = NativeCliJsonlParser(config=_config())
    first = b"".join(
        (
            _frame(kind="thread.started", thread_id="thread-1"),
            _frame(kind="turn.started"),
            _frame(kind="item.completed", item={"id": "m1", "type": "agent_message", "text": "First"}),
            _frame(kind="turn.completed"),
            _frame(kind="turn.started"),
            _frame(kind="item.completed", item={"id": "m2", "type": "agent_message", "text": "Second"}),
        )
    )
    trailing = _frame(kind="turn.completed").removesuffix(b"\n")
    events = (*parser.feed(data=first + trailing), *parser.finish())
    assert parser.coverage_complete and parser.observed_steps == 2
    assert [event.observation.text for event in _of_kind(events=events, kind=NativeCliEventKind.MODEL_MESSAGE)] == [
        "First",
        "Second",
    ]
    assert _of_kind(events=events, kind=NativeCliEventKind.TURN_COMPLETED)[-1].raw_frame == trailing


@pytest.mark.parametrize(
    ("frames", "reason"),
    [
        (
            (
                _frame(kind="item.completed", item={"id": "m1", "type": "agent_message", "text": "Observed"}),
                _frame(kind="thread.started", thread_id="thread-1"),
                _frame(kind="turn.started"),
                _frame(kind="turn.completed"),
            ),
            "preceded its session start",
        ),
        (
            (
                _frame(kind="thread.started", thread_id="thread-1"),
                _frame(kind="item.completed", item={"id": "m1", "type": "agent_message", "text": "Observed"}),
                _frame(kind="turn.started"),
                _frame(kind="turn.completed"),
            ),
            "outside a started turn",
        ),
        (
            (
                _frame(kind="thread.started", thread_id="thread-1"),
                _frame(kind="turn.started"),
                _frame(kind="turn.completed"),
                _frame(kind="item.completed", item={"id": "m1", "type": "agent_message", "text": "Observed"}),
            ),
            "followed its terminal event",
        ),
    ],
)
def test_codex_out_of_order_model_messages_are_observed_but_never_complete(
    frames: tuple[bytes, ...], reason: str
) -> None:
    parser = NativeCliJsonlParser(config=_config())
    events = (*parser.feed(data=b"".join(frames)), *parser.finish())
    assert _of_kind(events=events, kind=NativeCliEventKind.MODEL_MESSAGE)[0].observation.text == "Observed"
    assert any(reason in gap for gap in parser.gaps)
    assert not parser.coverage_complete


def test_claude_tool_result_after_final_result_is_partial_not_complete() -> None:
    parser = NativeCliJsonlParser(config=_config(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE))
    events = (
        *parser.feed(
            data=b"".join(
                (
                    _frame(kind="system", subtype="init", session_id="claude-1", uuid="init-1"),
                    _frame(
                        kind="assistant",
                        session_id="claude-1",
                        uuid="assistant-1",
                        message={
                            "id": "msg-1",
                            "role": "assistant",
                            "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {}}],
                        },
                    ),
                    _frame(
                        kind="result",
                        session_id="claude-1",
                        uuid="end-1",
                        subtype="success",
                        is_error=False,
                        result="Done.",
                    ),
                    _frame(
                        kind="user",
                        session_id="claude-1",
                        uuid="user-1",
                        message={
                            "role": "user",
                            "content": [
                                {"type": "tool_result", "tool_use_id": "tool-1", "content": "late", "is_error": False}
                            ],
                        },
                    ),
                )
            )
        ),
        *parser.finish(),
    )
    assert _of_kind(events=events, kind=NativeCliEventKind.TOOL_RESULT)[0].observation.result == "late"
    assert any("followed its terminal event" in gap for gap in parser.gaps)
    assert not parser.coverage_complete

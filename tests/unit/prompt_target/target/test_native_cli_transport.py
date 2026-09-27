# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Fake sandbox transport and exact-byte evidence tests; no executable is launched."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import fields, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pytest

from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliEventKind,
    NativeCliEventStatus,
    NativeCliProcessChunk,
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliRunConfig,
    NativeCliStream,
)
from pyrit.prompt_target.native_cli_transport import NativeCliRunner, NativeCliStreamLimitError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _config(*, timeout_seconds: float = 1, max_steps: int = 2, max_frame_bytes: int = 4096) -> NativeCliRunConfig:
    return NativeCliRunConfig(
        protocol=NativeCliProtocol.CODEX_EXEC_JSON,
        cli_version="0.115.0",
        cli_profile="locked-workspace",
        agent_workdir=PurePosixPath("/workspace/task"),
        model_gateway_endpoint="http://gateway.sandbox/v1",
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        max_frame_bytes=max_frame_bytes,
    )


def _frame(*, kind: str, **fields: object) -> bytes:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _turn(*, thread_id: str = "thread-1", text: str = "H\u00e9llo") -> bytes:
    return b"".join(
        (
            _frame(kind="thread.started", thread_id=thread_id),
            _frame(kind="turn.started"),
            _frame(kind="item.completed", item={"id": "msg-1", "type": "agent_message", "text": text}),
            _frame(kind="turn.completed"),
        )
    )


class _Recorder:
    def __init__(self, *, fail_raw: bool = False) -> None:
        self.raw: list[NativeCliRawChunk] = []
        self.events: list[NativeCliEvent] = []
        self.fail_raw = fail_raw

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        if self.fail_raw:
            raise OSError("Inert recorder unavailable")
        self.raw.append(chunk)

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        self.events.append(event)


class _Sandbox:
    def __init__(self, *, chunks: list[NativeCliProcessChunk], exit_code: int = 0, pause: bool = False) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.pause = pause
        self.read_paused = asyncio.Event()
        self.resume = asyncio.Event()
        self.stop_count = 0

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        for chunk in self.chunks:
            yield chunk
        self.read_paused.set()
        if self.pause:
            await self.resume.wait()

    async def wait_async(self) -> int:
        return self.exit_code

    async def stop_async(self) -> None:
        self.stop_count += 1
        self.resume.set()


class _Launcher:
    def __init__(self, *, sessions: list[_Sandbox]) -> None:
        self.sessions = sessions
        self.calls: list[tuple[NativeCliRunConfig, str]] = []

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> _Sandbox:
        self.calls.append((config, prompt))
        return self.sessions.pop(0)


async def test_transport_records_every_raw_byte_before_normalization_and_preserves_crlf_async() -> None:
    first = _frame(kind="thread.started", thread_id="thread-1").replace(b"\n", b"\r\n")
    stdout = first + _turn().removeprefix(_frame(kind="thread.started", thread_id="thread-1"))
    utf8_boundary = stdout.index("\u00e9".encode()) + 1
    stderr = b"progress\xff\r\n"
    sandbox = _Sandbox(
        chunks=[
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[:7]),
            NativeCliProcessChunk(stream=NativeCliStream.STDERR, data=stderr),
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[7:utf8_boundary]),
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[utf8_boundary:]),
        ]
    )
    sink = _Recorder()
    launcher = _Launcher(sessions=[sandbox])
    config = _config()
    result = await NativeCliRunner(launcher=launcher, sink=sink).run_async(config=config, prompt="Offline test")

    assert launcher.calls == [(config, "Offline test")]
    assert sandbox.stop_count == 1
    assert result.coverage_complete and result.exit_code == 0
    assert result.raw_chunk_count == 4 and result.raw_stdout_bytes == len(stdout)
    assert result.raw_stderr_bytes == len(stderr)
    assert [(chunk.sequence, chunk.stream) for chunk in sink.raw] == [
        (1, NativeCliStream.STDOUT),
        (2, NativeCliStream.STDERR),
        (3, NativeCliStream.STDOUT),
        (4, NativeCliStream.STDOUT),
    ]
    assert b"".join(chunk.data for chunk in sink.raw if chunk.stream is NativeCliStream.STDOUT) == stdout
    assert b"".join(chunk.data for chunk in sink.raw if chunk.stream is NativeCliStream.STDERR) == stderr
    frames = {event.frame_number: event.raw_frame for event in sink.events if event.frame_number is not None}
    assert list(frames.values()) == stdout.splitlines(keepends=True)
    assert sink.events[0].raw_frame == first
    assert [event.sequence for event in sink.events] == list(range(1, len(sink.events) + 1))
    assert (
        next(
            event.observation.text
            for event in sink.events
            if event.observation.kind is NativeCliEventKind.MODEL_MESSAGE
        )
        == "H\u00e9llo"
    )


async def test_nonzero_process_exit_is_an_error_even_after_terminal_event_async() -> None:
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_turn())], exit_code=9)
    sink = _Recorder()
    result = await NativeCliRunner(launcher=_Launcher(sessions=[sandbox]), sink=sink).run_async(
        config=_config(), prompt="Fixture"
    )
    assert result.terminal_observed and not result.coverage_complete
    assert result.exit_code == 9 and "exited with code 9" in result.gaps[-1]
    assert sink.events[-1].observation.kind is NativeCliEventKind.ERROR
    assert sink.events[-1].observation.status is NativeCliEventStatus.FAILED
    assert sandbox.stop_count == 1


async def test_cancellation_stops_sandbox_and_records_incomplete_run_async() -> None:
    first = _frame(kind="thread.started", thread_id="thread-1")
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=first)], pause=True)
    sink = _Recorder()
    task = asyncio.create_task(
        NativeCliRunner(launcher=_Launcher(sessions=[sandbox]), sink=sink).run_async(config=_config(), prompt="Fixture")
    )
    await asyncio.wait_for(sandbox.read_paused.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.stop_count == 1 and sink.raw[0].data == first
    assert sink.events[-1].observation.kind is NativeCliEventKind.ERROR
    assert "cancelled" in (sink.events[-1].observation.detail or "")
    assert not any(event.observation.kind is NativeCliEventKind.EOF for event in sink.events)


async def test_timeout_stops_sandbox_without_reporting_success_async() -> None:
    sandbox = _Sandbox(chunks=[], pause=True)
    sink = _Recorder()
    with pytest.raises(TimeoutError):
        await NativeCliRunner(launcher=_Launcher(sessions=[sandbox]), sink=sink).run_async(
            config=_config(timeout_seconds=0.05), prompt="Fixture"
        )
    assert sandbox.stop_count == 1
    assert sink.events[-1].observation.kind is NativeCliEventKind.ERROR
    assert "timeout_seconds" in (sink.events[-1].observation.detail or "")


@pytest.mark.parametrize("newline", [b"", b"\n"])
async def test_oversize_frame_fails_explicitly_after_handing_off_all_bytes_async(newline: bytes) -> None:
    raw = _frame(kind="thread.started", thread_id="thread-1") + b'{"oversize":"' + b"x" * 80 + b'"}' + newline
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)])
    sink = _Recorder()
    with pytest.raises(NativeCliStreamLimitError, match="max_frame_bytes"):
        await NativeCliRunner(launcher=_Launcher(sessions=[sandbox]), sink=sink).run_async(
            config=_config(max_frame_bytes=60), prompt="Fixture"
        )
    assert sandbox.stop_count == 1 and sink.raw[0].data == raw
    assert sink.events[-1].observation.kind is NativeCliEventKind.ERROR
    assert sink.events[-1].raw_frame is None


async def test_observed_step_budget_stops_process_without_inventing_completion_async() -> None:
    raw = b"".join(
        (
            _frame(kind="thread.started", thread_id="thread-1"),
            _frame(kind="turn.started"),
            _frame(kind="turn.completed"),
            _frame(kind="turn.started"),
            _frame(kind="turn.completed"),
        )
    )
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)])
    sink = _Recorder()
    with pytest.raises(NativeCliStreamLimitError, match="step budget"):
        await NativeCliRunner(launcher=_Launcher(sessions=[sandbox]), sink=sink).run_async(
            config=_config(max_steps=1), prompt="Fixture"
        )
    assert sandbox.stop_count == 1 and sink.raw[0].data == raw
    assert sink.events[-1].observation.kind is NativeCliEventKind.ERROR
    assert sum(event.observation.kind is NativeCliEventKind.TURN_COMPLETED for event in sink.events) == 1


async def test_recorder_error_is_propagated_and_sandbox_is_stopped_async() -> None:
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_turn())])
    sink = _Recorder(fail_raw=True)
    with pytest.raises(OSError, match="recorder unavailable"):
        await NativeCliRunner(launcher=_Launcher(sessions=[sandbox]), sink=sink).run_async(
            config=_config(), prompt="Fixture"
        )
    assert sandbox.stop_count == 1
    assert not sink.raw
    assert sink.events[-1].observation.kind is NativeCliEventKind.ERROR


async def test_empty_prompt_rejected_without_launching_sandbox_async() -> None:
    launcher = _Launcher(sessions=[])
    with pytest.raises(ValueError, match="nonempty prepared prompt"):
        await NativeCliRunner(launcher=launcher, sink=_Recorder()).run_async(config=_config(), prompt="  ")
    assert not launcher.calls


async def test_repeated_runs_have_independent_id_and_step_state_async() -> None:
    sessions = [
        _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_turn(thread_id="first"))]),
        _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_turn(thread_id="second"))]),
    ]
    first, second = sessions
    sink = _Recorder()
    runner = NativeCliRunner(launcher=_Launcher(sessions=sessions), sink=sink)
    outcomes = [
        await runner.run_async(config=_config(), prompt="First"),
        await runner.run_async(config=_config(), prompt="Second"),
    ]
    assert [outcome.source_session_id for outcome in outcomes] == ["first", "second"]
    assert all(outcome.coverage_complete and outcome.observed_steps == 1 for outcome in outcomes)
    assert first.stop_count == second.stop_count == 1
    assert [
        event.sequence for event in sink.events if event.observation.kind is NativeCliEventKind.SESSION_STARTED
    ] == [
        1,
        1,
    ]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"cli_version": "latest"}, "version"),
        ({"cli_profile": "locked --danger"}, "profile"),
        ({"agent_workdir": PurePosixPath("C:\\work")}, "workdir"),
        ({"agent_workdir": PurePosixPath("/work/../host")}, "workdir"),
        ({"agent_workdir": PurePosixPath("/work/\x00host")}, "workdir"),
        ({"agent_workdir": PurePosixPath("/work/\nhost")}, "workdir"),
        ({"model_gateway_endpoint": "http://user:fixture@gateway.sandbox/v1"}, "credentials"),
        ({"model_gateway_endpoint": "http://gateway.sandbox/v1?token=fixture"}, "query"),
        ({"model_gateway_endpoint": "http://gateway.sandbox/\x00v1"}, "control"),
        ({"max_steps": 0}, "step budget"),
        ({"timeout_seconds": math.inf}, "timeout"),
        ({"max_frame_bytes": 0}, "frame limit"),
    ],
)
def test_run_config_rejects_unsafe_or_unbounded_values(changes: dict[str, object], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        replace(_config(), **changes)


def test_run_config_contains_only_sandbox_routing_and_no_host_auth_or_shell_callback() -> None:
    names = {field.name for field in fields(NativeCliRunConfig)}
    assert names == {
        "protocol",
        "cli_version",
        "cli_profile",
        "agent_workdir",
        "model_gateway_endpoint",
        "max_steps",
        "timeout_seconds",
        "max_frame_bytes",
    }

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline PyRIT target tests with real messages and fake sandbox process bytes."""

from __future__ import annotations

import asyncio
import json
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from pyrit.models import Message, MessagePiece
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import NativeCliTarget, TargetCapabilities
from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliEventKind,
    NativeCliProcessChunk,
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliRunConfig,
    NativeCliStream,
)
from pyrit.prompt_target.native_cli_target import NativeCliCoverageError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _config(*, protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON) -> NativeCliRunConfig:
    return NativeCliRunConfig(
        protocol=protocol,
        cli_version="0.115.0" if protocol is NativeCliProtocol.CODEX_EXEC_JSON else "2.1.220",
        cli_profile="locked-workspace",
        agent_workdir=PurePosixPath("/workspace/task"),
        model_gateway_endpoint="http://gateway.sandbox/v1",
        max_steps=4,
        timeout_seconds=1,
    )


def _frame(*, kind: str, **fields: object) -> bytes:
    return json.dumps({"type": kind, **fields}, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def _codex(*, texts: tuple[str, ...] = ("Done.",), tool: bool = False, with_result: bool = True) -> bytes:
    frames = [_frame(kind="thread.started", thread_id="thread-1"), _frame(kind="turn.started")]
    if tool:
        frames.append(
            _frame(
                kind="item.started",
                item={"id": "cmd-1", "type": "command_execution", "status": "in_progress", "command": "fixture"},
            )
        )
        completed: dict[str, object] = {
            "id": "cmd-1",
            "type": "command_execution",
            "status": "completed",
            "command": "fixture",
            "exit_code": 0,
        }
        if with_result:
            completed["aggregated_output"] = "observed artifact"
        frames.append(_frame(kind="item.completed", item=completed))
    frames.extend(
        _frame(kind="item.completed", item={"id": f"msg-{index}", "type": "agent_message", "text": text})
        for index, text in enumerate(texts, start=1)
    )
    frames.append(_frame(kind="turn.completed"))
    return b"".join(frames)


def _claude(*, assistant_text: str | None, result_text: str) -> bytes:
    frames = [_frame(kind="system", subtype="init", uuid="init-1", session_id="session-1")]
    if assistant_text is not None:
        frames.append(
            _frame(
                kind="assistant",
                uuid="assistant-1",
                session_id="session-1",
                message={
                    "id": "message-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            )
        )
    frames.append(
        _frame(
            kind="result",
            subtype="success",
            uuid="result-1",
            session_id="session-1",
            is_error=False,
            result=result_text,
        )
    )
    return b"".join(frames)


class _Recorder:
    def __init__(self, *, fail_raw: bool = False) -> None:
        self.raw: list[NativeCliRawChunk] = []
        self.events: list[NativeCliEvent] = []
        self.fail_raw = fail_raw

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        if self.fail_raw:
            raise OSError("Inert evidence store unavailable")
        self.raw.append(chunk)

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        self.events.append(event)


class _Sandbox:
    def __init__(
        self,
        *,
        chunks: list[NativeCliProcessChunk],
        exit_code: int = 0,
        pause: bool = False,
    ) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.pause = pause
        self.read_paused = asyncio.Event()
        self.release = asyncio.Event()
        self.stop_count = 0

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        for chunk in self.chunks:
            yield chunk
        self.read_paused.set()
        if self.pause:
            await self.release.wait()

    async def wait_async(self) -> int:
        return self.exit_code

    async def stop_async(self) -> None:
        self.stop_count += 1
        self.release.set()


class _Launcher:
    def __init__(self, *, sandbox: _Sandbox | None = None, error: Exception | None = None) -> None:
        self.sandbox = sandbox
        self.error = error
        self.calls: list[tuple[NativeCliRunConfig, str]] = []

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> _Sandbox:
        self.calls.append((config, prompt))
        if self.error is not None:
            raise self.error
        if self.sandbox is None:
            raise RuntimeError("No inert sandbox configured.")
        return self.sandbox


def _target(
    *, sandbox: _Sandbox | None = None, protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON
) -> tuple[NativeCliTarget, _Launcher, _Recorder]:
    launcher = _Launcher(sandbox=sandbox)
    recorder = _Recorder()
    return (
        NativeCliTarget(run_config=_config(protocol=protocol), launcher=launcher, evidence_sink=recorder),
        launcher,
        recorder,
    )


async def test_real_prompt_normalizer_stores_only_observed_assistant_text_async(sqlite_instance: SQLiteMemory) -> None:
    stdout = _codex(texts=("Checking.", "Done."), tool=True)
    sandbox = _Sandbox(
        chunks=[
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[:17]),
            NativeCliProcessChunk(stream=NativeCliStream.STDERR, data=b"progress\r\n"),
            NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=stdout[17:]),
        ]
    )
    target, launcher, recorder = _target(sandbox=sandbox)
    request = MessagePiece(
        role="user",
        original_value="original instruction",
        converted_value="prepared instruction",
        prompt_metadata={"campaign": "fixture"},
    ).to_message()
    conversation_id = str(uuid4())

    response = await PromptNormalizer().send_prompt_async(
        message=request, target=target, conversation_id=conversation_id
    )
    assert launcher.calls == [(_config(), "prepared instruction")]
    assert response.get_value() == "Done."
    assert sandbox.stop_count == 1
    assert b"".join(chunk.data for chunk in recorder.raw if chunk.stream is NativeCliStream.STDOUT) == stdout
    assert recorder.raw[1].data == b"progress\r\n"
    assert [
        event.observation.kind for event in recorder.events if event.observation.kind is NativeCliEventKind.TOOL_RESULT
    ] == [NativeCliEventKind.TOOL_RESULT]
    stored = sqlite_instance.get_conversation_messages(conversation_id=conversation_id)
    assert [message.api_role for message in stored] == ["user", "assistant", "assistant"]
    assert [message.get_value() for message in stored] == ["prepared instruction", "Checking.", "Done."]
    assert [message.get_piece().prompt_metadata["native_cli_source_event_id"] for message in stored[1:]] == [
        "msg-1",
        "msg-2",
    ]
    assert all(message.get_piece().prompt_metadata["campaign"] == "fixture" for message in stored[1:])
    assert target.last_run is not None and target.last_run.outcome.coverage_complete
    assert target.last_run.evidence_sink is target.evidence_sink is recorder
    assert target.last_run.conversation_id == conversation_id
    assert [source.source_event_id for source in target.last_run.response_sources] == ["msg-1", "msg-2"]


async def test_claude_complete_assistant_text_is_not_duplicated_by_final_result_async() -> None:
    raw = _claude(assistant_text="Observed answer", result_text="Observed answer")
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)])
    target, _, recorder = _target(sandbox=sandbox, protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    responses = await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user"))
    assert [message.get_value() for message in responses] == ["Observed answer"]
    assert responses[0].get_piece().prompt_metadata == {
        "native_cli_source_kind": "model_message",
        "native_cli_source_event_id": "assistant-1",
        "native_cli_source_message_id": "message-1",
        "native_cli_source_session_id": "session-1",
    }
    assert any(event.observation.kind is NativeCliEventKind.RUN_FINISHED for event in recorder.events)
    assert target.last_run is not None and len(target.last_run.response_sources) == 1


async def test_claude_final_result_text_is_observed_fallback_not_a_fake_message_id_async() -> None:
    raw = _claude(assistant_text=None, result_text="Observed final only")
    target, _, recorder = _target(
        sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)]),
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
    )
    responses = await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user"))
    assert [message.get_value() for message in responses] == ["Observed final only"]
    metadata = responses[0].get_piece().prompt_metadata
    assert metadata["native_cli_source_kind"] == "run_finished"
    assert metadata["native_cli_source_event_id"] == "result-1"
    assert "native_cli_source_message_id" not in metadata
    assert target.last_run is not None and target.last_run.response_sources[0].kind is NativeCliEventKind.RUN_FINISHED
    assert b"".join(chunk.data for chunk in recorder.raw) == raw


async def test_tool_only_artifact_run_is_write_only_in_real_normalizer_async(sqlite_instance: SQLiteMemory) -> None:
    raw = _codex(texts=(), tool=True)
    target, _, recorder = _target(
        sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)])
    )
    conversation_id = str(uuid4())
    response = await PromptNormalizer().send_prompt_async(
        message=Message.from_prompt(prompt="Create a file", role="user"),
        target=target,
        conversation_id=conversation_id,
    )
    assert response.api_role == "user" and response.get_value() == "Create a file"
    assert [
        message.api_role for message in sqlite_instance.get_conversation_messages(conversation_id=conversation_id)
    ] == ["user"]
    assert target.last_run is not None and target.last_run.outcome.coverage_complete
    assert target.last_run.response_sources == ()
    assert any(event.observation.kind is NativeCliEventKind.TOOL_RESULT for event in recorder.events)
    assert not any(event.observation.kind is NativeCliEventKind.MODEL_MESSAGE for event in recorder.events)


async def test_no_text_from_claude_is_write_only_even_after_successful_result_async() -> None:
    raw = _claude(assistant_text=None, result_text="")
    target, _, _ = _target(
        sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)]),
        protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
    )
    assert await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user")) == []
    assert target.last_run is not None and target.last_run.outcome.coverage_complete
    assert target.last_run.response_sources == ()


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
async def test_only_prepared_user_role_may_launch_cli_async(role: str) -> None:
    target, launcher, _ = _target()
    with pytest.raises(ValueError, match="prepared user text"):
        await target.send_prompt_async(message=MessagePiece(role=role, original_value="Fixture").to_message())
    assert not launcher.calls and target.last_run is None


async def test_unsupported_multimodal_and_multi_piece_requests_fail_before_launch_async() -> None:
    target, launcher, _ = _target()
    non_text = MessagePiece(
        role="user", original_value="fixture.png", converted_value_data_type="image_path"
    ).to_message()
    with pytest.raises(ValueError, match="supports only"):
        await target.send_prompt_async(message=non_text)
    pieces = Message(
        message_pieces=[
            MessagePiece(role="user", original_value="first", sequence=0),
            MessagePiece(role="user", original_value="second", sequence=0),
        ]
    )
    with pytest.raises(ValueError, match="single message piece"):
        await target.send_prompt_async(message=pieces)
    assert not launcher.calls


@pytest.mark.parametrize("metadata", [{"response_format": "json"}, {"json_schema": {"type": "object"}}])
async def test_structured_output_request_fails_closed_before_launch_async(metadata: dict[str, object]) -> None:
    target, launcher, _ = _target()
    request = MessagePiece(role="user", original_value="Fixture", prompt_metadata=metadata).to_message()
    with pytest.raises(ValueError):
        await target.send_prompt_async(message=request)
    assert not launcher.calls


async def test_invalid_input_does_not_consume_the_one_shot_target_async() -> None:
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
    target, launcher, _ = _target(sandbox=sandbox)
    with pytest.raises(ValueError, match="prepared user"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="Rejected", role="assistant"))
    response = await target.send_prompt_async(message=Message.from_prompt(prompt="Accepted", role="user"))
    assert [message.get_value() for message in response] == ["Done."]
    assert len(launcher.calls) == 1


async def test_one_shot_target_rejects_reuse_even_after_reset_async() -> None:
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
    target, launcher, _ = _target(sandbox=sandbox)
    await target.send_prompt_async(message=Message.from_prompt(prompt="First", role="user"))
    await target.reset_conversation_async(conversation_id="unrelated")
    with pytest.raises(RuntimeError, match="one-shot"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="Second", role="user"))
    assert len(launcher.calls) == 1 and sandbox.stop_count == 1


async def test_new_target_cannot_replay_prior_memory_history_async() -> None:
    conversation_id = str(uuid4())
    first, _, _ = _target(
        sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
    )
    await PromptNormalizer().send_prompt_async(
        message=Message.from_prompt(prompt="First", role="user"),
        target=first,
        conversation_id=conversation_id,
    )
    second, launcher, _ = _target(
        sandbox=_Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
    )
    with pytest.raises(Exception, match="Error sending prompt"):
        await PromptNormalizer().send_prompt_async(
            message=Message.from_prompt(prompt="Not a resumed CLI turn", role="user"),
            target=second,
            conversation_id=conversation_id,
        )
    assert not launcher.calls


async def test_concurrent_invocations_cannot_start_second_sandbox_async() -> None:
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())], pause=True)
    target, launcher, _ = _target(sandbox=sandbox)
    first = asyncio.create_task(target.send_prompt_async(message=Message.from_prompt(prompt="First", role="user")))
    await asyncio.wait_for(sandbox.read_paused.wait(), timeout=1)
    with pytest.raises(RuntimeError, match="one-shot"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="Second", role="user"))
    sandbox.release.set()
    assert [message.get_value() for message in await first] == ["Done."]
    assert len(launcher.calls) == 1 and sandbox.stop_count == 1


@pytest.mark.parametrize(("exit_code", "with_result"), [(0, False), (7, True)])
async def test_incomplete_coverage_never_returns_assistant_success_async(exit_code: int, with_result: bool) -> None:
    raw = _codex(tool=True, with_result=with_result)
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw)], exit_code=exit_code)
    target, _, recorder = _target(sandbox=sandbox)
    with pytest.raises(NativeCliCoverageError, match="evidence is incomplete") as caught:
        await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user"))
    assert caught.value.run is target.last_run
    assert caught.value.run.outcome.exit_code == exit_code
    assert not caught.value.run.outcome.coverage_complete
    assert caught.value.run.response_sources[0].text == "Done."
    assert recorder.raw[0].data == raw and sandbox.stop_count == 1
    with pytest.raises(RuntimeError, match="one-shot"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="No retry", role="user"))


async def test_recorder_failure_propagates_without_claiming_a_run_or_retry_async() -> None:
    sandbox = _Sandbox(chunks=[NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=_codex())])
    launcher = _Launcher(sandbox=sandbox)
    recorder = _Recorder(fail_raw=True)
    target = NativeCliTarget(run_config=_config(), launcher=launcher, evidence_sink=recorder)
    with pytest.raises(OSError, match="evidence store unavailable"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user"))
    assert target.last_run is None and target.evidence_sink is recorder
    assert sandbox.stop_count == 1
    with pytest.raises(RuntimeError, match="one-shot"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="No retry", role="user"))
    assert len(launcher.calls) == 1


async def test_launcher_failure_is_explicit_and_cannot_retry_same_target_async() -> None:
    launcher = _Launcher(error=OSError("Inert sandbox unavailable"))
    recorder = _Recorder()
    target = NativeCliTarget(run_config=_config(), launcher=launcher, evidence_sink=recorder)
    with pytest.raises(OSError, match="sandbox unavailable"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user"))
    assert target.last_run is None and not recorder.raw
    assert recorder.events[-1].observation.kind is NativeCliEventKind.ERROR
    with pytest.raises(RuntimeError, match="one-shot"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="No retry", role="user"))
    assert len(launcher.calls) == 1


async def test_cancellation_stops_process_without_replaying_it_async() -> None:
    sandbox = _Sandbox(chunks=[], pause=True)
    target, launcher, recorder = _target(sandbox=sandbox)
    task = asyncio.create_task(target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user")))
    await asyncio.wait_for(sandbox.read_paused.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.stop_count == 1 and target.last_run is None
    assert recorder.events[-1].observation.kind is NativeCliEventKind.ERROR
    with pytest.raises(RuntimeError, match="one-shot"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="No retry", role="user"))
    assert len(launcher.calls) == 1


async def test_capability_override_cannot_turn_one_shot_target_into_multi_turn_async() -> None:
    target, launcher, _ = _target()
    target.apply_capabilities(capabilities=TargetCapabilities(supports_multi_turn=True))
    with pytest.raises(ValueError, match="capabilities cannot be overridden"):
        await target.send_prompt_async(message=Message.from_prompt(prompt="Fixture", role="user"))
    assert not launcher.calls


def test_identifier_records_only_sandbox_profile_not_credentials_or_timeout() -> None:
    target, _, _ = _target()
    params = target.get_identifier().params
    assert params["adapter"] == "native_cli_jsonl"
    assert params["protocol"] == NativeCliProtocol.CODEX_EXEC_JSON.value
    assert params["cli_version"] == "0.115.0"
    assert params["cli_profile"] == "locked-workspace"
    assert params["agent_workdir"] == "/workspace/task"
    assert params["model_gateway_endpoint"] == "http://gateway.sandbox/v1"
    assert params["max_steps"] == 4
    assert "timeout_seconds" not in params and "api_key" not in params

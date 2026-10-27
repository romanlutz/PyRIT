# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Synthetic source fixtures through the actual native workflow and isolated SQLite."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pyrit.executor.workflow.copilot_sdk_evidence import CopilotModelResponseOrigin
from pyrit.executor.workflow.native_cyber_eval import NativeCyberEvaluation
from pyrit.models.native_cyber import NativeCyberEvidenceContract, NativeCyberRequest
from pyrit.prompt_target import NativeAgentTarget
from tests.unit.executor.workflow.test_native_cyber_eval import FixtureBinding
from tests.unit.prompt_target.target.test_native_agent_target import (
    EventFixture,
    SdkSessionFixture,
    agent_session,
    event,
    tool_turn,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pyrit.executor.workflow.copilot_sdk_evidence import CopilotSdkModelObserver
    from pyrit.executor.workflow.native_cyber_eval import NativeCyberRuntime
    from pyrit.memory import SQLiteMemory
    from pyrit.models.native_cyber import NativeAgentEvidence, NativeCyberJudgment
    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeStart, NativeCyberRawWrite

pytestmark = pytest.mark.usefixtures("patch_central_database")


@pytest.fixture(autouse=True)
async def no_external_io_async() -> AsyncIterator[None]:
    """Forbid network, subprocess and external provider execution in this source-only suite."""
    with (
        patch.object(socket.socket, "connect", side_effect=AssertionError("Network is forbidden")),
        patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network is forbidden")),
        patch.object(socket.socket, "sendto", side_effect=AssertionError("Network is forbidden")),
        patch("socket.getaddrinfo", side_effect=AssertionError("DNS is forbidden")),
        patch("subprocess.Popen", side_effect=AssertionError("Subprocesses are forbidden")),
        patch.object(os, "system", side_effect=AssertionError("Shell execution is forbidden")),
        patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=AssertionError("Subprocesses are forbidden"),
        ),
        patch(
            "asyncio.create_subprocess_shell",
            new_callable=AsyncMock,
            side_effect=AssertionError("Shell execution is forbidden"),
        ),
        patch.object(
            httpx.AsyncClient, "send", new_callable=AsyncMock, side_effect=AssertionError("Providers are forbidden")
        ),
        patch.object(httpx.Client, "send", side_effect=AssertionError("Providers are forbidden")),
    ):
        yield


class ObservedSdkFixture(SdkSessionFixture):
    """Inert SDK callbacks and explicitly synthetic provider-origin body fixtures, not an SDK/client."""

    def __init__(self, *, memory: SQLiteMemory, defect: str | None = None) -> None:
        super().__init__([tool_turn(call_id=f"call-{index}") for index in range(3)])
        self.memory = memory
        self.defect = defect
        self.observer: CopilotSdkModelObserver | None = None
        self.observers: list[CopilotSdkModelObserver] = []
        self.dispatched = 0
        self.returned = 0
        self.waiting = asyncio.Event()
        self.request_bodies: list[bytes] = []
        self.response_bodies: list[bytes] = []

    async def send_and_wait(self, prompt: str, *, timeout: float) -> object:  # pyrit-async-suffix-exempt
        self.prompts.append(prompt)
        ordinal = len(self.prompts)
        assert self.handler is not None and self.observer is not None
        observer = self.observer
        self.observers.append(observer)
        before = await asyncio.to_thread(self.memory.native_cyber_evidence.get_episode, run_id=observer.identity.run_id)
        assert before.turns[-1].source_turn_id == observer.identity.turn_id
        assert before.turns[-1].finished_at is None
        assert len(before.turns) == ordinal and len(before.raw_streams) == ordinal * 5
        assert {key.observed_source_id for key in before.run.required_raw_streams} == {
            "ghcp.sdk.events",
            "ghcp.model.requests",
            "ghcp.model.responses",
        }
        self.handler(EventFixture(event("user.message", {"content": prompt}, sessionId=self.session_id)))
        request_id = "model-1" if self.defect == "replay-request" else f"model-{ordinal}"
        request = json.dumps({"model": "synthetic-source", "input": prompt}, separators=(",", ":")).encode()
        response = b'{"id":"inert-response","output":"synthetic original provider fixture"}'
        if self.defect in {"foreign-run", "foreign-session", "foreign-turn"}:
            identity = replace(
                observer.identity,
                **{
                    {"foreign-run": "run_id", "foreign-session": "session_id", "foreign-turn": "turn_id"}[
                        self.defect
                    ]: "foreign"
                },
            )
            await observer._sink.record_model_request_async(
                identity=identity, request_id=request_id, request_bytes=request, accepted=True
            )
        if self.defect == "unpaired-response":
            await observer.record_response_async(
                request_id=request_id,
                body=response,
                status_code=200,
                origin=CopilotModelResponseOrigin.PROVIDER,
                complete=True,
            )
        if self.defect != "no-model":
            await observer.record_request_async(request_id=request_id, body=request, accepted=self.defect != "rejected")
            retained = await asyncio.to_thread(
                self.memory.native_cyber_evidence.get_episode, run_id=observer.identity.run_id
            )
            assert retained.events[-1].event_type in {"ghcp.model.request", "ghcp.model.rejected_request"}
            assert retained.events[-2].event_type == "user.message"
            self.request_bodies.append(request)
            self.dispatched += int(self.defect != "rejected")
            if self.defect == "cancel":
                self.waiting.set()
                await asyncio.Event().wait()
            if self.defect == "duplicate-request":
                await observer.record_request_async(request_id=request_id, body=b"conflicting request", accepted=True)
            if self.defect != "missing-response":
                origin = (
                    CopilotModelResponseOrigin.SYNTHETIC
                    if self.defect == "synthetic"
                    else CopilotModelResponseOrigin.HOST
                    if self.defect in {"host", "rejected"}
                    else CopilotModelResponseOrigin.PROVIDER
                )
                await observer.record_response_async(
                    request_id=request_id,
                    body=response,
                    status_code=503 if self.defect == "provider-error" else 200,
                    origin=origin,
                    complete=self.defect != "truncated",
                )
                retained = await asyncio.to_thread(
                    self.memory.native_cyber_evidence.get_episode, run_id=observer.identity.run_id
                )
                assert retained.events[-1].event_type.endswith("response")
                self.returned += 1
                self.response_bodies.append(response)
                if self.defect == "duplicate-response":
                    await observer.record_response_async(
                        request_id=request_id,
                        body=b"conflicting response",
                        status_code=200,
                        origin=CopilotModelResponseOrigin.PROVIDER,
                        complete=True,
                    )
        events = self.turns[ordinal - 1]
        if self.defect == "missing-tool-completion":
            events = [value for value in events if value["type"] != "tool.execution_complete"]
        if self.defect == "foreign-sdk":
            events[0]["sessionId"] = "foreign"
        if self.defect == "duplicate-sdk":
            events.insert(1, events[0])
        if self.defect == "conflicting-sdk":
            events.insert(1, {**events[0], "data": {"content": "different source payload"}})
        if self.defect == "no-idle":
            events = [value for value in events if value["type"] != "session.idle"]
        if self.defect == "replay-sdk" and ordinal > 1:
            events[0] = self.turns[0][0]
        for value in events:
            self.handler(EventFixture(value))
        return None


class UnifiedBinding(FixtureBinding):
    """One task-owned trusted grader fixture, using the existing inert environment and real workflow."""

    def __init__(self, *, memory: SQLiteMemory, steps: bool = False, defect: str | None = None) -> None:
        super().__init__(steps=steps)
        self.evidence_contract = NativeCyberEvidenceContract.COPILOT_SDK_AND_MODEL
        self.runtime.sdk = ObservedSdkFixture(memory=memory, defect=defect)
        self.runtime.session = agent_session(self.runtime.sdk, steps=steps)
        self.runtime.target = NativeAgentTarget(session=self.runtime.session)

    def bind_model_observer(self, *, runtime: NativeCyberRuntime, observer: CopilotSdkModelObserver) -> None:
        assert runtime is self.runtime
        self.runtime.sdk.observer = observer


async def raw_bytes_async(*, memory: SQLiteMemory, run_id: str, source_id: str, turn: int) -> bytes:
    store = memory.native_cyber_evidence
    episode = await asyncio.to_thread(store.get_episode, run_id=run_id)
    stream = next(
        stream
        for stream in episode.raw_streams
        if stream.key.observed_source_id == source_id and stream.turn_index == turn
    )
    chunks = await asyncio.to_thread(
        store.read_raw_chunks, run_id=run_id, stream_id=stream.stream_id, allow_sensitive=True
    )
    value = b"".join(chunk.data for chunk in chunks)
    assert stream.stored_sha256 == hashlib.sha256(value).hexdigest()
    return value


@pytest.mark.parametrize("turns", [1, 2])
async def test_unified_workflow_grades_once_after_complete_sources_then_cleans_and_publishes_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, turns: int
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, steps=turns == 2)
    run = NativeCyberEvaluation(
        binding=binding,
        request=NativeCyberRequest(instruction="fixture one", operator_steps=turns == 2),
        directory=tmp_path,
    )
    original_grade = binding.runtime.grade_async

    async def grade_async(*, evidence: NativeAgentEvidence) -> NativeCyberJudgment:
        episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=run.run_id)
        assert episode.score_id is None and episode.report_content_id is None
        assert len(episode.turns) == turns and all(turn.source_complete for turn in episode.turns)
        assert all(stream.source_complete for stream in episode.raw_streams)
        assert binding.runtime.sdk.disconnected and not binding.runtime.closed
        assert evidence.scope == "native_sdk_and_model" and evidence.coverage_complete
        return await original_grade(evidence=evidence)

    with patch.object(binding.runtime, "grade_async", side_effect=grade_async) as grader:
        result = await run.start_async()
        if turns == 2:
            assert result.can_step and result.status == "awaiting_instruction"
            assert run.report is None and run.score is None
            await run.step_async("fixture two")
            result = await run.finish_async()
        grader.assert_awaited_once()
    assert result.status == "completed" and run.score.get_value() == 0.75
    assert binding.order == ["open", "grade", "close"]
    assert binding.runtime.grade_count == binding.open_count == binding.close_count == 1
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_finalized_episode, run_id=run.run_id)
    assert episode.coverage_complete and episode.score_id == run.score.id
    assert episode.report_content_id == run.score.scorable.content_id and episode.report_sha256 == run.report.sha256()
    assert len(await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")) == 1
    assert len(episode.raw_streams) == 5 * turns
    assert {stream.key.observed_source_id for stream in episode.raw_streams} == {
        "ghcp.sdk.events",
        "ghcp.model.requests",
        "ghcp.model.responses",
        "ghcp.model.rejected_requests",
        "ghcp.model.host_responses",
    }
    assert [value.sequence for value in episode.events] == list(range(1, len(episode.events) + 1))
    assert len(episode.events) == len(run.report.agent.events) == turns * 9
    assert {value.observed_session_id for value in episode.events} == {binding.runtime.session.session_id}
    pieces: set[object] = set()
    for ordinal, turn in enumerate(episode.turns, 1):
        assert (
            turn.request_piece_ids
            and turn.response_piece_ids
            and turn.tool_request_piece_ids
            and turn.tool_result_piece_ids
        )
        linked = {
            *turn.request_piece_ids,
            *turn.response_piece_ids,
            *turn.tool_request_piece_ids,
            *turn.tool_result_piece_ids,
        }
        assert not pieces & linked
        pieces.update(linked)
        assert (
            await raw_bytes_async(
                memory=sqlite_instance, run_id=run.run_id, source_id="ghcp.model.requests", turn=ordinal
            )
            == binding.runtime.sdk.request_bodies[ordinal - 1]
        )
        assert (
            await raw_bytes_async(
                memory=sqlite_instance, run_id=run.run_id, source_id="ghcp.model.responses", turn=ordinal
            )
            == binding.runtime.sdk.response_bodies[ordinal - 1]
        )
        sdk_wire = await raw_bytes_async(
            memory=sqlite_instance, run_id=run.run_id, source_id="ghcp.sdk.events", turn=ordinal
        )
        assert len(sdk_wire.splitlines()) == 7
    assert all(
        tool.request_sequence < tool.start_sequence < tool.completion_sequence for tool in run.report.agent.tools
    )
    assert run.report.agent.tools[0].request_sequence == 4
    assert run.report.agent.tools[0].model_visible_output == "OFFLINE/SIMULATED"
    assert run.report.agent.tools[0].detailed_output == "separate detailed result"
    for value in run.report.agent.events:
        if value.event_type.startswith("ghcp.model."):
            assert "synthetic original provider fixture" not in json.dumps(value.payload)
    await run.finish_async()
    assert len(await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "no-model",
        "missing-response",
        "synthetic",
        "host",
        "rejected",
        "truncated",
        "provider-error",
        "foreign-run",
        "foreign-session",
        "foreign-turn",
        "foreign-sdk",
        "duplicate-request",
        "duplicate-response",
        "duplicate-sdk",
        "conflicting-sdk",
        "no-idle",
        "unpaired-response",
        "missing-tool-completion",
    ],
)
async def test_unified_capture_defects_never_call_original_grader_or_publish_complete_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, defect: str
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, defect=defect)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    result = await run.start_async()
    assert result.status == "error"
    assert binding.runtime.grade_count == 0 and binding.close_count == 1
    assert run.score.is_undetermined
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_finalized_episode, run_id=run.run_id)
    assert not episode.coverage_complete and episode.score_status.value == "undetermined"
    assert all(
        score.is_undetermined for score in await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
    )
    if defect in {"synthetic", "host", "rejected"}:
        model_responses = [event for event in episode.events if event.event_type == "ghcp.model.response"]
        assert not model_responses
        host = [
            event
            for event in episode.events
            if event.event_type in {"ghcp.model.host_response", "ghcp.model.synthetic_response"}
        ]
        assert host and all(value.source.value == "harness" for value in host)
    if defect in {"foreign-run", "foreign-session", "foreign-turn", "unpaired-response"}:
        assert binding.runtime.sdk.dispatched == 0
    if defect in {"duplicate-sdk", "conflicting-sdk"}:
        retained = await raw_bytes_async(memory=sqlite_instance, run_id=run.run_id, source_id="ghcp.sdk.events", turn=1)
        assert b"assistant.message" in retained
        assert b"different source payload" not in retained


@pytest.mark.parametrize("failure", ["quota", "write", "misattributed-write"])
async def test_request_persistence_failure_blocks_upstream_dispatch_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, failure: str
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    store = run._evidence_store
    original_create, original_append = store.create_episode, store.append_raw

    def create_episode(*, start: NativeCyberEpisodeStart) -> None:
        original_create(start=start.model_copy(update={"raw_byte_limit": 16}) if failure == "quota" else start)

    def append_raw(**kwargs: object) -> NativeCyberRawWrite:
        if failure == "write":
            raise OSError("Inert durable writer unavailable.")
        value = original_append(**kwargs)
        return (
            value.model_copy(update={"stored_bytes": value.stored_bytes - 1})
            if failure == "misattributed-write"
            else value
        )

    with (
        patch.object(store, "create_episode", side_effect=create_episode),
        patch.object(store, "append_raw", side_effect=append_raw),
    ):
        await run.start_async()
    assert binding.runtime.sdk.dispatched == binding.runtime.sdk.returned == binding.runtime.grade_count == 0
    assert binding.close_count == 1 and run.score.is_undetermined
    episode = await asyncio.to_thread(store.get_finalized_episode, run_id=run.run_id)
    assert not episode.coverage_complete


@pytest.mark.parametrize("defect", ["replay-request", "replay-sdk"])
async def test_second_turn_replay_cannot_reset_controller_sequence_or_duplicate_evidence_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, defect: str
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, steps=True, defect=defect)
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="first", operator_steps=True), directory=tmp_path
    )
    await run.start_async()
    first = await asyncio.to_thread(run._evidence_store.get_episode, run_id=run.run_id)
    await run.step_async("second")
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert tuple(episode.events[: len(first.events)]) == first.events
    assert len(episode.turns) == 2 and len(episode.raw_streams) == 10
    assert [value.sequence for value in episode.events] == list(range(1, len(episode.events) + 1))
    assert binding.runtime.grade_count == 0 and run.score.is_undetermined


async def test_cancellation_preserves_unpaired_request_and_prevents_grading_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, defect="cancel")
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    running = asyncio.create_task(run.start_async())
    await asyncio.wait_for(binding.runtime.sdk.waiting.wait(), timeout=3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert run.status == "cancelled" and binding.runtime.grade_count == 0
    assert binding.close_count == 1 and run.score.is_undetermined
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert any(
        stream.key.observed_source_id == "ghcp.model.requests" and stream.stored_bytes for stream in episode.raw_streams
    )
    assert not any(
        stream.key.observed_source_id == "ghcp.model.responses" and stream.stored_bytes
        for stream in episode.raw_streams
    )


@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_quiescence_or_cancel_during_quiescence_prevents_grading_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, cancel: bool
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = binding.runtime.session.quiesce_async

    async def quiesce_async() -> None:
        if not cancel:
            raise RuntimeError("Inert quiescence unconfirmed.")
        await original()
        await run.cancel_async()

    with patch.object(binding.runtime.session, "quiesce_async", side_effect=quiesce_async):
        await run.start_async()
    assert binding.runtime.grade_count == 0 and binding.close_count == 1 and run.score.is_undetermined


async def test_postfinish_model_observation_latches_gap_before_grading_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, steps=True)
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="first", operator_steps=True), directory=tmp_path
    )
    await run.start_async()
    observer = binding.runtime.sdk.observer
    assert observer is not None
    with pytest.raises(RuntimeError, match="not open"):
        await observer.record_request_async(request_id="late", body=b"late source", accepted=True)
    await run.finish_async()
    assert binding.runtime.grade_count == 0 and run.score.is_undetermined


@pytest.mark.parametrize("failure", ["late-model", "cleanup", "atomic-publication"])
async def test_postgrade_failure_retains_original_judgment_but_never_complete_publication_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, failure: str
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = binding.runtime.grade_async

    async def grade_async(*, evidence: NativeAgentEvidence) -> NativeCyberJudgment:
        judgment = await original(evidence=evidence)
        if failure == "late-model":
            with pytest.raises(RuntimeError, match="not open"):
                await binding.runtime.sdk.observer.record_request_async(request_id="late", body=b"late", accepted=True)
        return judgment

    original_context = binding.open_runtime

    @asynccontextmanager
    async def failing_context_async(*, run_id: str, request: NativeCyberRequest) -> AsyncIterator[NativeCyberRuntime]:
        async with original_context(run_id=run_id, request=request) as runtime:
            yield runtime
        raise OSError("Inert owned cleanup not confirmed.")

    with patch.object(binding.runtime, "grade_async", side_effect=grade_async):
        if failure == "cleanup":
            with patch.object(binding, "open_runtime", side_effect=failing_context_async):
                await run.start_async()
        elif failure == "atomic-publication":
            with patch.object(
                run._evidence_store, "finalize_episode_atomic", side_effect=OSError("Inert publication failure")
            ):
                with pytest.raises(OSError, match="publication failure"):
                    await run.start_async()
            assert run.report is None and run.score is None
            assert not await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
        else:
            await run.start_async()
    assert binding.runtime.grade_count == 1
    assert binding.close_count == 1
    assert not any(
        not score.is_undetermined
        for score in await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
    )
    if run.report:
        assert run.report.judgment.value == 0.75


def test_offline_guard_blocks_network_and_process_launches() -> None:
    with pytest.raises(AssertionError, match="DNS"):
        socket.getaddrinfo("unused.invalid", 443)
    with pytest.raises(AssertionError, match="Subprocesses"):
        subprocess.Popen(["not-run"])


async def test_sdk_only_binding_remains_explicit_without_claiming_model_wire_coverage_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="legacy fixture"), directory=tmp_path
    )
    await run.start_async()
    assert run.report.status == "completed" and run.report.agent.scope == "native_session_events"
    assert binding.evidence_contract is NativeCyberEvidenceContract.SDK_EVENTS
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert [key.observed_source_id for key in episode.run.required_raw_streams] == ["controller-serialized-sdk-events"]
    assert not any(event.event_type.startswith("ghcp.model.") for event in episode.events)


async def test_unified_binding_requires_host_observer_injection_without_sdk_only_fallback_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = FixtureBinding()
    binding.evidence_contract = NativeCyberEvidenceContract.COPILOT_SDK_AND_MODEL
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    await run.start_async()
    assert run.status == "error" and run.score.is_undetermined
    assert binding.runtime.grade_count == 0 and not binding.runtime.sdk.prompts
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert {key.observed_source_id for key in episode.run.required_raw_streams} == {
        "ghcp.sdk.events",
        "ghcp.model.requests",
        "ghcp.model.responses",
    }


async def test_late_sdk_event_after_turn_finish_is_rejected_and_cannot_reach_grading_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, steps=True)
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="fixture", operator_steps=True), directory=tmp_path
    )
    await run.start_async()
    with pytest.raises(RuntimeError, match="not open"):
        binding.runtime.sdk.handler(EventFixture(event("assistant.message", {"content": "late output"})))
    await run.finish_async()
    assert run.score.is_undetermined and binding.runtime.grade_count == 0
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert not episode.coverage_complete
    assert any("watermark" in gap or "outside" in gap for gap in episode.gaps)


async def test_stale_turn_observer_cannot_write_into_the_next_turn_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance, steps=True)
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="first", operator_steps=True), directory=tmp_path
    )
    await run.start_async()
    previous = binding.runtime.sdk.observer

    def bind_stale(*, runtime: NativeCyberRuntime, observer: CopilotSdkModelObserver) -> None:
        assert previous is not None and observer.identity.turn_id != previous.identity.turn_id
        runtime.sdk.observer = previous

    with patch.object(binding, "bind_model_observer", side_effect=bind_stale):
        await run.step_async("second")
    assert run.score.is_undetermined and binding.runtime.grade_count == 0
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert len(episode.turns) == 2
    assert all(event.sequence == index for index, event in enumerate(episode.events, 1))


async def test_cancelled_durable_write_settles_before_turn_seal_and_publication_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    writing, release, written = threading.Event(), threading.Event(), threading.Event()
    original = run._evidence_store.append_raw

    def append_raw(**kwargs: object) -> NativeCyberRawWrite:
        writing.set()
        assert release.wait(timeout=5)
        value = original(**kwargs)
        written.set()
        return value

    with patch.object(run._evidence_store, "append_raw", side_effect=append_raw):
        running = asyncio.create_task(run.start_async())
        try:
            assert await asyncio.to_thread(writing.wait, 3)
            running.cancel()
            await asyncio.sleep(0)
            assert run.report is None and run.score is None
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert written.is_set() and run.score.is_undetermined and binding.runtime.grade_count == 0
    assert binding.runtime.sdk.dispatched == 0 and binding.close_count == 1
    snapshot = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert not snapshot.coverage_complete


async def test_sdk_watermark_cannot_hide_or_duplicate_persisted_events_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = binding.runtime.session.evidence

    def omitted_evidence() -> NativeAgentEvidence:
        evidence = original()
        return evidence.model_copy(update={"events": evidence.events[:-1]}) if evidence.events else evidence

    with patch.object(binding.runtime.session, "evidence", side_effect=omitted_evidence):
        await run.start_async()
    assert binding.runtime.grade_count == 0 and run.score.is_undetermined
    assert any("watermark" in gap for gap in run.report.agent.gaps)


async def test_failed_durable_response_never_returns_as_captured_provider_output_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = run._evidence_store.append_raw

    def append_raw(**kwargs: object) -> NativeCyberRawWrite:
        if b"synthetic original provider fixture" in kwargs["data"]:
            raise OSError("Inert response writer failed.")
        return original(**kwargs)

    with patch.object(run._evidence_store, "append_raw", side_effect=append_raw):
        await run.start_async()
    assert binding.runtime.sdk.dispatched == 1 and binding.runtime.sdk.returned == 0
    assert run.score.is_undetermined and binding.runtime.grade_count == 0


async def test_failed_source_seal_blocks_grade_without_duplicate_turn_or_stream_retry_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(run._evidence_store, "close_raw_stream", side_effect=OSError("Inert seal failure")) as seal:
        await run.start_async()
    assert seal.call_count == 1 and run.score.is_undetermined and binding.runtime.grade_count == 0
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert len(episode.turns) == 1 and len(episode.raw_streams) == 5
    assert not episode.coverage_complete


async def test_concurrent_trusted_model_observations_share_one_durable_order_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = binding.runtime.sdk.send_and_wait

    async def send_async(prompt: str, *, timeout: float) -> object:
        observer = binding.runtime.sdk.observer
        assert observer is not None
        await asyncio.gather(
            *(
                observer.record_request_async(
                    request_id=f"parallel-{index}", body=f"request-{index}".encode(), accepted=True
                )
                for index in range(2)
            )
        )
        await asyncio.gather(
            *(
                observer.record_response_async(
                    request_id=f"parallel-{index}",
                    body=f"response-{index}".encode(),
                    origin=CopilotModelResponseOrigin.PROVIDER,
                    status_code=200,
                    complete=True,
                )
                for index in range(2)
            )
        )
        return await original(prompt, timeout=timeout)

    with patch.object(binding.runtime.sdk, "send_and_wait", side_effect=send_async):
        await run.start_async()
    assert binding.runtime.grade_count == 1 and not run.score.is_undetermined
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert [value.sequence for value in episode.events] == list(range(1, 14))
    assert [value.event_type for value in episode.events[:4]] == [
        "ghcp.model.request",
        "ghcp.model.request",
        "ghcp.model.response",
        "ghcp.model.response",
    ]
    assert run.report.agent.tools[0].request_sequence == 8


async def test_sdk_callbacks_from_a_thread_keep_observed_payload_and_order_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = UnifiedBinding(memory=sqlite_instance)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = binding.runtime.sdk.send_and_wait

    async def send_async(prompt: str, *, timeout: float) -> object:
        extra = event("session.info", {"fixture": "thread callback"})
        await asyncio.to_thread(binding.runtime.sdk.handler, EventFixture(extra))
        extra["data"]["fixture"] = "changed after callback"
        return await original(prompt, timeout=timeout)

    with patch.object(binding.runtime.sdk, "send_and_wait", side_effect=send_async):
        await run.start_async()
    assert binding.runtime.grade_count == 1
    assert run.report.agent.events[0].payload["data"] == {"fixture": "thread callback"}
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert [value.sequence for value in episode.events] == list(range(1, 11))

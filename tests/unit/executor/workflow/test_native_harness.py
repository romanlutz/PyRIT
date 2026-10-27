# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Two real public harness adapters over synthetic process/model sources and isolated SQLite."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest

from pyrit.converter import Base64Converter
from pyrit.executor.workflow.native_cyber_eval import NativeCyberEvaluation
from pyrit.executor.workflow.native_harness import (
    ClaudeCliHarnessAdapter,
    CodexCliHarnessAdapter,
    CopilotSdkHarnessAdapter,
    CopilotSdkHarnessConnection,
    NativeCliModelObserver,
)
from pyrit.executor.workflow.native_model_transport import NativeModelRequest, NativeModelResponse, NativeModelTransport
from pyrit.models.native_cli_report import NativeCliRunReport
from pyrit.models.native_cyber import (
    NativeAgentCapabilities,
    NativeCyberArtifact,
    NativeCyberJudgment,
    NativeCyberReadiness,
    NativeCyberRequest,
)
from pyrit.models.native_cyber_evidence import (
    NativeCyberCoveragePhase,
    NativeCyberResponseMode,
    NativeCyberResponsePolicy,
)
from pyrit.models.native_harness import NativeHarnessEvidence, NativeModelCaptureIdentity, NativeModelResponseOrigin
from pyrit.prompt_target import NativeAgentTarget
from pyrit.prompt_target.gateway.claude_messages import create_claude_messages_app
from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute
from pyrit.prompt_target.native_agent_target import CopilotSdkAgentSession
from pyrit.prompt_target.native_cli_models import NativeCliProcessChunk, NativeCliProtocol, NativeCliStream
from pyrit.registry import ConverterRegistry
from tests.unit.executor.workflow.test_copilot_sdk_evidence import ObservedSdkFixture
from tests.unit.executor.workflow.test_copilot_sdk_evidence import no_external_io_async as no_external_io_async
from tests.unit.executor.workflow.test_native_cyber_eval import FixtureBinding
from tests.unit.prompt_target.gateway.messages_mocks import FakeMessagesBackend, request_body
from tests.unit.prompt_target.gateway.test_codex_responses import FakeModelOnlyBackend, _wire
from tests.unit.prompt_target.target.test_native_cli_target import _claude, _codex, _config, _frame, _Launcher, _Sandbox

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
    from pyrit.executor.workflow.native_harness import NativeHarnessAdapter, NativeTaskRuntime
    from pyrit.executor.workflow.native_model_transport import NativeModelObserver
    from pyrit.memory import SQLiteMemory
    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeStart

pytestmark = pytest.mark.usefixtures("patch_central_database", "no_external_io_async")


class TaskRuntime:
    """The same original synthetic task grader for both harnesses, with no harness branching."""

    def __init__(self, *, order: list[str], memory: SQLiteMemory) -> None:
        self.run_id = "unallocated"
        self.environment_id = str(uuid4())
        self.simulated = True
        self.order = order
        self.memory = memory
        self.closed = False
        self.grade_count = 0
        self.last_evidence: NativeHarnessEvidence | None = None

    async def validate_agent_storage_async(self, *, directory: Path) -> None:
        assert await asyncio.to_thread(directory.is_dir)

    async def grade_async(self, *, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        assert evidence.quiescent and evidence.coverage_complete and not self.closed
        assert evidence.identity.run_id == self.run_id and evidence.identity.environment_id == self.environment_id
        before = await asyncio.to_thread(self.memory.native_cyber_evidence.get_episode, run_id=self.run_id)
        assert before.report_content_id is None and before.score_id is None
        assert all(turn.source_complete for turn in before.turns)
        self.grade_count += 1
        self.order.append("grade")
        self.last_evidence = evidence
        return NativeCyberJudgment(
            value=0.375,
            complete=True,
            rationale="Synthetic original task fixture only",
            evidence={"fixture": True, "run_id": self.run_id, "target_alive": True},
        )


class TaskBinding(FixtureBinding):
    """Reuse the inert storage/lease fixture, without choosing an agent or model protocol."""

    def __init__(self, memory: SQLiteMemory) -> None:
        super().__init__()
        self.runtime = TaskRuntime(order=self.order, memory=memory)

    async def readiness_async(self) -> NativeCyberReadiness:
        return NativeCyberReadiness(ready=True, simulated=True, provenance={"fixture": "OFFLINE/SYNTHETIC"})

    @asynccontextmanager
    async def open_runtime(  # pyrit-async-suffix-exempt
        self, *, run_id: str, request: NativeCyberRequest
    ) -> AsyncIterator[TaskRuntime]:
        self.runtime.run_id = run_id
        self.open_count += 1
        self.order.append("open")
        try:
            yield self.runtime
        finally:
            self.runtime.closed = True
            self.close_count += 1
            self.order.append("close")


class CliProcess(_Sandbox):
    """Fake sandbox pipes; actual runner/parser and either actual ASGI gateway or neutral transport."""

    def __init__(
        self,
        *,
        memory: SQLiteMemory,
        sink: NativeCliDatabaseEvidenceSink,
        runtime: TaskRuntime,
        defect: str | None,
        body_transport: bool,
        harness: str,
    ) -> None:
        self.harness = harness
        raw = _codex(tool=defect in {"codex-tools", "missing-tool"})
        if harness == "claude":
            raw = b"".join(
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
                                {"type": "text", "text": "OFFLINE reply"},
                                {"type": "tool_use", "id": "use-1", "name": "Bash", "input": {"command": "fixture"}},
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
                                    "content": "observed artifact",
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
        if defect == "malformed":
            raw += b"{invalid\n"
        elif defect == "unknown":
            raw += _frame(kind="unknown.new.event")
        elif defect == "duplicate":
            raw = raw.replace(
                _frame(kind="turn.completed"),
                _frame(kind="item.completed", item={"id": "msg-1", "type": "agent_message", "text": "duplicate"})
                + _frame(kind="turn.completed"),
            )
        elif defect == "foreign-session":
            raw = _frame(kind="thread.started", thread_id="foreign") + raw
        elif defect == "missing-tool":
            raw = b"\n".join(line for line in raw.split(b"\n") if b'"item.started"' not in line)
        elif defect == "artifact-only":
            raw = _claude(assistant_text=None, result_text="") if harness == "claude" else _codex(texts=())
        super().__init__(
            chunks=[
                NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw[:17]),
                NativeCliProcessChunk(stream=NativeCliStream.STDERR, data=b"inert stderr\xff\n"),
                NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=raw[17:]),
            ],
            exit_code=1 if defect == "exit" else 0,
            pause=defect == "cancel",
        )
        self.sink, self.runtime, self.memory = sink, runtime, memory
        self.defect, self.body_transport = defect, body_transport
        self.backend = FakeMessagesBackend() if harness == "claude" else FakeModelOnlyBackend()
        self.model_calls = 0
        self.original_request = b'{ "model":"codex-fixture", "input":"synthetic model request", "store":false }\n'
        if harness == "claude":
            self.original_request = json.dumps(request_body(tools=True)).encode()
        self.original_response: bytes | None = None

    async def forward_async(
        self, *, request: NativeModelRequest, deadline: float, max_response_bytes: int
    ) -> NativeModelResponse:
        self.model_calls += 1
        episode = await asyncio.to_thread(self.memory.native_cyber_evidence.get_episode, run_id=request.identity.run_id)
        assert episode.events[-1].event_type == "gateway.request"
        return NativeModelResponse(
            identity=request.identity,
            request_id=request.request_id,
            body=_wire(),
            status_code=200,
            origin=NativeModelResponseOrigin.SYNTHETIC
            if self.defect == "synthetic"
            else NativeModelResponseOrigin.PROVIDER,
            complete=True,
        )

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        if self.defect != "no-model":
            if (
                self.body_transport
                or self.defect == "missing-response"
                or (self.defect or "").startswith("foreign-model")
            ):
                identity = NativeModelCaptureIdentity(
                    run_id=self.sink.run_id,
                    session_id=None,
                    turn_id=self.sink.turn_id,
                    turn_index=1,
                    environment_id=self.runtime.environment_id,
                    harness_id="codex_exec_json",
                )
                if (self.defect or "").startswith("foreign-model"):
                    field = {
                        "foreign-model-run": "run_id",
                        "foreign-model-environment": "environment_id",
                        "foreign-model-session": "session_id",
                    }[self.defect]
                    identity = replace(identity, **{field: "foreign"})
                observer = NativeCliModelObserver(
                    sink=self.sink,
                    identity=identity,
                )
                if self.defect == "missing-response":
                    await observer.record_request_async(request_id="body-1", body=self.original_request, accepted=True)
                else:
                    response = await NativeModelTransport(observer=observer, forwarder=self).forward_async(
                        request=NativeModelRequest(
                            identity=observer.identity, request_id="body-1", body=self.original_request, streaming=False
                        ),
                        deadline=asyncio.get_running_loop().time() + 5,
                    )
                    self.original_response = response.body
            else:
                route = GatewayRoute(
                    run_id=self.sink.run_id,
                    model="claude-offline-model" if self.harness == "claude" else "codex-fixture",
                    guest_token="synthetic-guest-token-" + "x" * 32,
                )
                if self.harness == "claude":
                    app = create_claude_messages_app(
                        route=route,
                        limits=GatewayLimits(),
                        backend=self.backend,
                        observation_callback=self.sink.record_messages_observation_async,
                    )
                else:
                    app = create_codex_responses_app(
                        route=route,
                        limits=GatewayLimits(),
                        backend=self.backend,
                        observation_callback=self.sink.record_gateway_observation_async,
                    )
                request = httpx.Request(
                    "POST",
                    "http://inert.invalid/v1/messages"
                    if self.harness == "claude"
                    else "http://inert.invalid/v1/responses",
                    content=self.original_request,
                    headers={
                        "Authorization": f"Bearer {route.guest_token}",
                        "X-PyRIT-Run-ID": route.run_id,
                        "Content-Type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                )
                async with httpx.ASGITransport(app=app) as transport:
                    response = await transport.handle_async_request(request)
                    self.original_response = await response.aread()
                assert response.status_code == 200, self.original_response
        if self.defect == "foreign-environment":
            self.runtime.environment_id = "foreign"
        async for part in super().read_chunks_async():
            yield part

    async def stop_async(self) -> None:
        await super().stop_async()
        self.runtime.order.append("stop")
        if self.defect == "stop":
            raise RuntimeError("Inert stop barrier failed")


def make_case(
    *, memory: SQLiteMemory, harness: str, defect: str | None = None, body_transport: bool = False
) -> tuple[TaskBinding, NativeHarnessAdapter, dict[str, object]]:
    binding = TaskBinding(memory)
    control: dict[str, object] = {"attached": asyncio.Event()}
    if harness == "ghcp":
        caps = NativeAgentCapabilities(retained_session=True, operator_steps=True, max_turns=3)

        def connect(runtime: NativeTaskRuntime) -> CopilotSdkHarnessConnection:
            sdk = ObservedSdkFixture(memory=memory, defect=defect)
            session = CopilotSdkAgentSession(
                session=sdk,
                environment_id=runtime.environment_id,
                simulated=True,
                capabilities=caps,
                provenance={"fixture": "OFFLINE/SYNTHETIC"},
            )
            target = NativeAgentTarget(session=session)
            control.update(sdk=sdk, session=session, target=target)
            control["attached"].set()

            def bind(observer: NativeModelObserver) -> None:
                sdk.observer = observer

            async def stop_async() -> None:
                assert sdk.disconnected and not binding.runtime.closed
                binding.order.append("stop")
                if defect == "stop":
                    raise RuntimeError("Inert owned SDK agent stop failed")

            return CopilotSdkHarnessConnection(target=target, bind_model_observer=bind, quiesce_agent_async=stop_async)

        adapter = CopilotSdkHarnessAdapter(connect=connect, capabilities=caps)
    else:

        def connect(*, runtime: NativeTaskRuntime, sink: NativeCliDatabaseEvidenceSink) -> _Launcher:
            process = CliProcess(
                memory=memory, sink=sink, runtime=runtime, defect=defect, body_transport=body_transport, harness=harness
            )
            launcher = _Launcher(sandbox=process, error=OSError("Inert start failed") if defect == "start" else None)
            control.update(process=process, sink=sink, launcher=launcher)
            control["attached"].set()
            return launcher

        adapter_type = ClaudeCliHarnessAdapter if harness == "claude" else CodexCliHarnessAdapter
        protocol = (
            NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE
            if harness == "claude"
            else NativeCliProtocol.CODEX_EXEC_JSON
        )
        adapter = adapter_type(config=replace(_config(protocol=protocol), timeout_seconds=5), connect=connect)
    return binding, adapter, control


@pytest.mark.parametrize(
    "harness,body_transport", [("ghcp", False), ("claude", False), ("codex", False), ("codex", True)]
)
async def test_two_harnesses_share_original_task_grade_cleanup_and_atomic_score_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str, body_transport: bool
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness=harness, body_transport=body_transport)
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="synthetic task"), directory=tmp_path
    )
    await run.start_async()
    assert run.status == "completed", run.report
    assert binding.open_count == binding.close_count == binding.runtime.grade_count == 1
    assert binding.order == ["open", "stop", "grade", "close"]
    assert run.score.get_value() == 0.375
    assert isinstance(binding.runtime.last_evidence, NativeHarnessEvidence)
    assert binding.runtime.last_evidence.identity.harness_id == adapter.harness_id
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert episode.coverage_complete and episode.score_id == run.score.id
    assert episode.report_sha256 == run.report.sha256()
    assert episode.run.environment_id == binding.runtime.environment_id
    assert len(episode.turns) == 1 and episode.turns[0].request_piece_ids and episode.turns[0].response_piece_ids
    assert [item.sequence for item in episode.events] == list(range(1, len(episode.events) + 1))
    assert len(await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")) == 1
    if harness != "ghcp":
        assert isinstance(run.report, NativeCliRunReport)
        assert run.report.judgment.retained_evidence["evidence"]["fixture"] is True
        assert run.report.environment_id == binding.runtime.environment_id
        assert run.report.evidence.source_session_id == ("thread-1" if harness == "codex" else "session-1")
        assert run.report.evidence.exit_code == 0 and control["process"].stop_count == 1
        assert not any(item.event_type.startswith(("ghcp.", "session.", "assistant.")) for item in episode.events)
        assert episode.turns[0].tool_request_piece_ids == episode.turns[0].tool_result_piece_ids == ()
        payloads = await asyncio.to_thread(
            run._evidence_store.read_event_payloads, run_id=run.run_id, allow_sensitive=True
        )
        if harness == "claude":
            assert episode.tools[0].request_sequence < episode.tools[0].result_sequence
            assert episode.tools[0].start_sequence is None and episode.tools[0].completion_sequence is None
            result = next(item.event for item in payloads if item.event.event_type == "native_cli.tool_result")
            assert result.payload["result"] == "observed artifact"
        else:
            assert not episode.tools
        streams = {item.key.observed_source_id: item for item in episode.raw_streams}
        for key, body in (
            (f"{adapter.harness_id}.gateway.requests", control["process"].original_request),
            (f"{adapter.harness_id}.gateway.responses", control["process"].original_response),
        ):
            chunks = await asyncio.to_thread(
                run._evidence_store.read_raw_chunks,
                run_id=run.run_id,
                stream_id=streams[key].stream_id,
                allow_sensitive=True,
            )
            assert b"".join(chunk.data for chunk in chunks) == body
    await run.finish_async()
    assert binding.runtime.grade_count == 1


async def test_ghcp_adapter_retains_two_real_turns_while_codex_rejects_before_allocation_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness="codex")
    with pytest.raises(ValueError, match="retained"):
        NativeCyberEvaluation(
            binding=binding,
            harness=adapter,
            request=NativeCyberRequest(instruction="fixture", operator_steps=True),
            directory=tmp_path,
        )
    with pytest.raises(TypeError, match="explicit"):
        NativeCyberEvaluation(
            binding=binding, harness="unknown", request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
        )
    assert binding.open_count == 0 and not control["attached"].is_set()
    binding, adapter, control = make_case(memory=sqlite_instance, harness="ghcp")
    run = NativeCyberEvaluation(
        binding=binding,
        harness=adapter,
        request=NativeCyberRequest(instruction="first", operator_steps=True),
        directory=tmp_path,
    )
    await run.start_async()
    assert run.view().can_step
    await run.step_async("second")
    await run.finish_async()
    assert binding.open_count == binding.close_count == binding.runtime.grade_count == 1
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert len(episode.turns) == 2 and len(episode.events) == 18


@pytest.mark.parametrize(
    "harness,defect",
    [
        ("ghcp", "no-model"),
        ("ghcp", "missing-tool-completion"),
        ("ghcp", "duplicate-sdk"),
        ("ghcp", "stop"),
        ("codex", "no-model"),
        ("codex", "missing-response"),
        ("codex", "malformed"),
        ("codex", "unknown"),
        ("codex", "duplicate"),
        ("codex", "foreign-session"),
        ("codex", "missing-tool"),
        ("codex", "exit"),
        ("codex", "stop"),
        ("codex", "start"),
        ("codex", "synthetic"),
        ("codex", "codex-tools"),
        ("claude", "no-model"),
        ("claude", "malformed"),
        ("claude", "stop"),
        ("codex", "foreign-model-run"),
        ("codex", "foreign-model-environment"),
        ("codex", "foreign-model-session"),
    ],
)
async def test_harness_source_and_stop_gaps_never_invoke_original_grader_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str, defect: str
) -> None:
    binding, adapter, control = make_case(
        memory=sqlite_instance, harness=harness, defect=defect, body_transport=defect == "synthetic"
    )
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    await run.start_async()
    assert binding.runtime.grade_count == 0 and binding.close_count == 1
    assert run.score.is_undetermined
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert not episode.coverage_complete
    if defect == "stop" and harness != "ghcp":
        assert run.report.evidence.exit_code == 0
    if harness == "codex":
        assert control["process"].stop_count == int(defect != "start")
        if defect.startswith("foreign-model"):
            assert control["process"].model_calls == 0


@pytest.mark.parametrize("harness", ["ghcp", "claude", "codex"])
async def test_cancelled_harness_never_grades_and_attempts_owned_cleanup_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness=harness, defect="cancel")
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    running = asyncio.create_task(run.start_async())
    try:
        await asyncio.wait_for(control["attached"].wait(), timeout=5)
        marker = control["sdk" if harness == "ghcp" else "process"]
        waiting = marker.waiting if harness == "ghcp" else marker.read_paused
        await asyncio.wait_for(waiting.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        if not running.done():
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
    assert binding.runtime.grade_count == 0 and binding.close_count == 1 and run.score.is_undetermined
    if harness != "ghcp":
        assert control["process"].stop_count == 1


@pytest.mark.parametrize(
    "harness,failure",
    [
        ("ghcp", "cleanup"),
        ("codex", "cleanup"),
        ("ghcp", "publication"),
        ("codex", "publication"),
        ("ghcp", "quota"),
        ("codex", "quota"),
        ("codex", "write"),
        ("codex", "late"),
        ("claude", "cleanup"),
        ("claude", "publication"),
        ("claude", "quota"),
    ],
)
async def test_common_publication_and_source_failures_remain_incomplete_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str, failure: str
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness=harness)
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    original_context, original_grade = binding.open_runtime, binding.runtime.grade_async
    original_start = run._evidence_store.create_episode

    @asynccontextmanager
    async def broken_context_async(**kwargs: object) -> AsyncIterator[TaskRuntime]:
        async with original_context(**kwargs) as runtime:
            yield runtime
        raise OSError("Inert environment cleanup failed")

    async def grade_async(*, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        judgment = await original_grade(evidence=evidence)
        if failure == "late":
            with pytest.raises(RuntimeError, match="not open"):
                control["sink"].ensure_active()
        return judgment

    def create_episode(*, start: NativeCyberEpisodeStart) -> None:
        original_start(start=start.model_copy(update={"raw_byte_limit": 16}))

    if failure == "cleanup":
        with patch.object(binding, "open_runtime", side_effect=broken_context_async):
            await run.start_async()
    elif failure == "publication":
        method = "finalize_episode_atomic" if harness == "ghcp" else "finalize_cli_episode_atomic"
        with patch.object(run._evidence_store, method, side_effect=OSError("Inert atomic publication failed")):
            with pytest.raises(OSError, match="atomic"):
                await run.start_async()
        assert run.score is None and run.report is None
    elif failure == "quota":
        with patch.object(run._evidence_store, "create_episode", side_effect=create_episode):
            await run.start_async()
    elif failure == "write":
        with patch.object(run._evidence_store, "append_raw", side_effect=OSError("Inert write failed")):
            await run.start_async()
    else:
        with patch.object(binding.runtime, "grade_async", side_effect=grade_async):
            await run.start_async()
    assert binding.close_count == 1
    assert binding.runtime.grade_count == int(failure in {"cleanup", "publication", "late"})
    assert not any(
        not score.is_undetermined
        for score in await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
    )


async def test_cli_pregrade_phase_needs_sources_not_a_future_judgment_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding, adapter, _ = make_case(memory=sqlite_instance, harness="codex")
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    original = binding.runtime.grade_async

    async def grade_async(*, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        candidate = run._build_report(None)
        assert isinstance(candidate, NativeCliRunReport) and candidate.judgment is None
        pregrade = await asyncio.to_thread(
            run._evidence_store.assess_cli_coverage,
            report=candidate,
            expected_turns=1,
            phase=NativeCyberCoveragePhase.PREGRADING,
        )
        final = await asyncio.to_thread(
            run._evidence_store.assess_cli_coverage,
            report=candidate,
            expected_turns=1,
            phase=NativeCyberCoveragePhase.FINAL,
        )
        assert pregrade.required_complete and not final.required_complete
        assert any("judgment" in gap for gap in final.required_gaps)
        judgment = await original(evidence=evidence)
        with pytest.raises(ValueError, match="cannot include"):
            await asyncio.to_thread(
                run._evidence_store.assess_cli_coverage,
                report=run._build_report(judgment),
                expected_turns=1,
                phase=NativeCyberCoveragePhase.PREGRADING,
            )
        return judgment

    with patch.object(binding.runtime, "grade_async", side_effect=grade_async):
        await run.start_async()
    assert run.status == "completed" and binding.runtime.grade_count == 1


@pytest.mark.parametrize("harness", ["ghcp", "claude", "codex"])
async def test_common_converter_runs_once_and_links_original_and_prepared_message_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness=harness)
    binding.allowed_converter_names = ("fixture_base64",)
    registry = ConverterRegistry()
    converter = Base64Converter()
    registry.instances.register(converter, name="fixture_base64")
    instruction = "literal synthetic instruction"
    run = NativeCyberEvaluation(
        binding=binding,
        harness=adapter,
        request=NativeCyberRequest(instruction=instruction, converter_names=("fixture_base64",)),
        directory=tmp_path,
    )
    with (
        patch.object(ConverterRegistry, "get_registry_singleton", return_value=registry),
        patch.object(converter, "convert_async", wraps=converter.convert_async) as convert,
    ):
        await run.start_async()
        convert.assert_awaited_once()
    assert run.status == "completed"
    prepared = base64.b64encode(instruction.encode()).decode()
    if harness == "ghcp":
        assert control["sdk"].prompts == [prepared]
    else:
        assert len(control["launcher"].calls) == 1 and control["launcher"].calls[0][1] == prepared
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    messages = await asyncio.to_thread(
        sqlite_instance.get_conversation_messages, conversation_id=run.report.conversation_id
    )
    piece = next(piece for message in messages for piece in message.message_pieces if piece.role == "user")
    assert piece.id in episode.turns[0].request_piece_ids
    assert piece.original_value == instruction and piece.converted_value == prepared


@pytest.mark.parametrize("harness", ["ghcp", "claude", "codex"])
async def test_missing_original_judgment_never_becomes_zero_or_complete_score_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str
) -> None:
    binding, adapter, _ = make_case(memory=sqlite_instance, harness=harness)
    original = binding.runtime.grade_async

    async def incomplete_async(*, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        await original(evidence=evidence)
        return NativeCyberJudgment(value=None, complete=False, rationale="Synthetic original judgment unavailable")

    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    with patch.object(binding.runtime, "grade_async", side_effect=incomplete_async):
        await run.start_async()
    assert binding.runtime.grade_count == binding.close_count == 1
    assert run.score.is_undetermined and run.score.score_value is None
    assert run.report.judgment.value is None


@pytest.mark.parametrize("harness", ["claude", "codex"])
async def test_task_approved_artifact_only_turn_never_fabricates_an_assistant_answer_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str
) -> None:
    binding, adapter, _ = make_case(memory=sqlite_instance, harness=harness, defect="artifact-only")
    binding.response_policy = NativeCyberResponsePolicy(allow_artifact_only=True)
    original = binding.runtime.grade_async
    artifact = tmp_path / "synthetic-grade-artifact.txt"
    data = b"explicit original synthetic grading fixture"

    async def grade_async(*, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        judgment = await original(evidence=evidence)
        await asyncio.to_thread(artifact.write_bytes, data)
        retained = await asyncio.to_thread(artifact.read_bytes)
        return judgment.model_copy(
            update={
                "artifacts": (
                    NativeCyberArtifact(
                        name=artifact.name,
                        evidence_ref=str(artifact),
                        sha256=hashlib.sha256(retained).hexdigest(),
                        size_bytes=len(retained),
                        media_type="text/plain",
                    ),
                )
            }
        )

    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    with (
        patch.object(binding, "response_mode", return_value=NativeCyberResponseMode.ARTIFACT_ONLY),
        patch.object(binding.runtime, "grade_async", side_effect=grade_async),
    ):
        await run.start_async()
    assert run.status == "completed" and run.score.get_value() == 0.375
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert episode.turns[0].request_piece_ids and not episode.turns[0].response_piece_ids
    messages = await asyncio.to_thread(
        sqlite_instance.get_conversation_messages, conversation_id=run.report.conversation_id
    )
    assert [message.api_role for message in messages] == ["user"]
    assert run.report.artifacts[0].sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("harness", ["ghcp", "claude", "codex"])
async def test_foreign_runtime_run_is_rejected_before_harness_dispatch_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness=harness)
    original = binding.open_runtime

    @asynccontextmanager
    async def foreign_context_async(**kwargs: object) -> AsyncIterator[TaskRuntime]:
        async with original(**kwargs) as runtime:
            runtime.run_id = "foreign"
            yield runtime

    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    with patch.object(binding, "open_runtime", side_effect=foreign_context_async):
        await run.start_async()
    assert not control["attached"].is_set()
    assert binding.runtime.grade_count == 0 and binding.close_count == 1 and run.score.is_undetermined


async def test_changed_environment_identity_blocks_grading_without_repairing_evidence_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness="codex", defect="foreign-environment")
    expected_environment = binding.runtime.environment_id
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    with pytest.raises(ValueError, match="identity changed"):
        await run.start_async()
    assert binding.runtime.grade_count == 0 and binding.close_count == 1 and control["process"].stop_count == 1
    assert run.report is None and run.score is None
    episode = await asyncio.to_thread(run._evidence_store.get_episode, run_id=run.run_id)
    assert episode.run.environment_id == expected_environment and episode.events
    assert not await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")


async def test_codex_tool_items_are_retained_without_claiming_unobserved_model_request_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding, adapter, _ = make_case(memory=sqlite_instance, harness="codex", defect="codex-tools")
    run = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path
    )
    await run.start_async()
    assert binding.runtime.grade_count == 0 and run.score.is_undetermined
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    tool = episode.tools[0]
    assert tool.request_sequence is None and tool.start_sequence < tool.completion_sequence < tool.result_sequence
    assert any("no provable model-visible request" in gap for gap in episode.gaps)
    events = await asyncio.to_thread(run._evidence_store.read_event_payloads, run_id=run.run_id, allow_sensitive=True)
    result = next(item.event for item in events if item.event.event_type == "native_cli.tool_result")
    assert result.payload["result"] == "observed artifact"
    assert "model_visible_output" not in result.payload


@pytest.mark.parametrize("harness", ["ghcp", "claude", "codex"])
async def test_fresh_rerun_rejects_reused_observed_session_without_regrading_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, harness: str
) -> None:
    binding, adapter, control = make_case(memory=sqlite_instance, harness=harness)
    parent = NativeCyberEvaluation(
        binding=binding, harness=adapter, request=NativeCyberRequest(instruction="first"), directory=tmp_path
    )
    await parent.start_async()
    session_id = binding.runtime.last_evidence.identity.source_session_id
    assert session_id is not None
    child = parent.rerun(request=NativeCyberRequest(instruction="second"))
    binding.runtime = TaskRuntime(order=binding.order, memory=sqlite_instance)
    original = ObservedSdkFixture.__init__

    def initialize(sdk: ObservedSdkFixture, *, memory: SQLiteMemory, defect: str | None = None) -> None:
        original(sdk, memory=memory, defect=defect)
        sdk.session_id = session_id

    with patch.object(ObservedSdkFixture, "__init__", new=initialize):
        await child.start_async()
    assert binding.open_count == binding.close_count == 2
    assert binding.runtime.grade_count == 0 and child.score.is_undetermined
    if harness == "ghcp":
        assert not control["sdk"].prompts
    assert any("previous harness" in error for error in child.report.errors)

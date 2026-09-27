# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-shot CLI lifecycle with real SQLite, inert process bytes, and in-process ASGI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import replace
from ipaddress import IPv4Address, IPv4Network
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest

from pyrit.converter import StringJoinConverter
from pyrit.executor.workflow.docker_agent import AgentStopObservation
from pyrit.executor.workflow.native_cli_evaluation import (
    NativeCliEvaluation,
    NativeCliOriginalAssessment,
)
from pyrit.models import ContentEntryScorable, ScoreStatus
from pyrit.models.environment_lease import (
    EnvironmentLeaseSnapshot,
    EnvironmentLeaseState,
    EnvironmentResourceHandle,
    EnvironmentServiceHandle,
)
from pyrit.models.native_cli_report import (
    NativeCliArtifactReference,
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportStatus,
    NativeCliRunReport,
)
from pyrit.models.native_cyber_evidence import (
    NativeCyberResponseMode,
    NativeCyberResponsePolicy,
)
from pyrit.prompt_normalizer import ConverterConfiguration
from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute
from pyrit.prompt_target.gateway.run_listener import GatewayBridgeBinding
from pyrit.prompt_target.native_cli_models import (
    NativeCliProcessChunk,
    NativeCliProtocol,
    NativeCliRunConfig,
    NativeCliStream,
)
from tests.unit.prompt_target.gateway.test_codex_responses import FakeModelOnlyBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.applications import Starlette

    from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _frame(*, kind: str, **properties: object) -> bytes:
    return json.dumps({"type": kind, **properties}, separators=(",", ":")).encode("utf-8") + b"\n"


def _stdout(*, assistant: bool = True, tool: bool = False) -> bytes:
    frames = [_frame(kind="thread.started", thread_id="thread-inert"), _frame(kind="turn.started")]
    if tool:
        frames.extend(
            (
                _frame(
                    kind="item.started",
                    item={"id": "cmd-1", "type": "command_execution", "command": "fixture", "status": "in_progress"},
                ),
                _frame(
                    kind="item.completed",
                    item={
                        "id": "cmd-1",
                        "type": "command_execution",
                        "command": "fixture",
                        "status": "completed",
                        "exit_code": 0,
                        "aggregated_output": "OFFLINE simulated tool output",
                    },
                ),
            )
        )
    if assistant:
        frames.append(_frame(kind="item.completed", item={"id": "message-1", "type": "agent_message", "text": "Done."}))
    frames.append(_frame(kind="turn.completed"))
    return b"".join(frames)


class _FakeGateway:
    def __init__(
        self, *, run_id: str, observer: NativeCliDatabaseEvidenceSink, trace: list[str], model: str = "codex-fixture"
    ) -> None:
        self.binding = GatewayBridgeBinding(
            run_id=run_id,
            network_id="f" * 64,
            address=IPv4Address("172.18.0.1"),
            subnet=IPv4Network("172.18.0.0/24"),
            port=8733,
        )
        self.base_url = "http://172.18.0.1:8733"
        self.route = GatewayRoute(run_id=run_id, model=model, guest_token="t" * 40)
        self.backend = FakeModelOnlyBackend()
        self.app: Starlette = create_codex_responses_app(
            route=self.route,
            limits=GatewayLimits(),
            backend=self.backend,
            observation_callback=observer.record_gateway_observation_async,
        )
        self.running = True
        self.fail_close = False
        self.trace = trace

    @property
    def is_locally_running(self) -> bool:
        return self.running

    async def send_inert_request_async(self) -> None:
        body = b'{"model":"codex-fixture","input":"OFFLINE fixture","store":false}'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=self.base_url) as client:
            reply = await client.post(
                "/v1/responses",
                content=body,
                headers={
                    "Authorization": "Bearer " + self.route.guest_token,
                    "X-PyRIT-Run-ID": self.binding.run_id,
                    "Content-Type": "application/json",
                },
            )
        if reply.status_code != 200:
            raise RuntimeError(f"Inert gateway returned {reply.status_code}.")
        self.trace.append("gateway_response")

    async def close_async(self) -> None:
        self.trace.append("gateway_closed")
        self.running = False
        if self.fail_close:
            raise RuntimeError("Inert gateway cleanup failed.")


class _FakeStopOnlyLease:
    """Stand-in for the Engine stop barrier, never a Docker operation."""

    def __init__(self, *, run_id: str, trace: list[str], stop_mode: str = "confirmed") -> None:
        self.run_id = run_id
        self.trace = trace
        self.stop_mode = stop_mode
        self.agent_stop: AgentStopObservation | None = None
        self.state = EnvironmentLeaseState.READY
        self.fail_close = False
        self.stop_calls = 0
        self.agent_id, self.target_id, self.grader_id = "a" * 64, "b" * 64, "c" * 64

    def snapshot(self) -> EnvironmentLeaseSnapshot:
        resource = EnvironmentResourceHandle(
            run_id=self.run_id, provider="inert_compose", resource_id="inert-project", kind="compose_project"
        )
        return EnvironmentLeaseSnapshot(
            run_id=self.run_id,
            lease_id="inert-lease",
            state=self.state,
            capabilities=frozenset(),
            services=(
                EnvironmentServiceHandle(name="agent", roles=frozenset({"agent"}), resource=resource),
                EnvironmentServiceHandle(name="target", roles=frozenset({"target"}), resource=resource),
                EnvironmentServiceHandle(name="grader", roles=frozenset({"grader"}), resource=resource),
            ),
            resources=(),
            setup_completed=False,
            health=(),
            errors=(),
        )

    async def stop_agent_async(self) -> None:
        self.stop_calls += 1
        if self.agent_stop is not None:
            if not self.agent_stop.stopped:
                raise RuntimeError("Inert stop barrier remains failed.")
            return
        self.trace.append("agent_stop")
        if self.stop_mode == "failed":
            self.agent_stop = AgentStopObservation(
                container_id=self.agent_id, exec_id="exec-inert", stopped=False, error="Inert stop failed."
            )
            raise RuntimeError("Inert stop barrier did not confirm a stopped agent.")
        if self.stop_mode == "missing_receipt":
            return
        self.agent_stop = AgentStopObservation(
            container_id=self.agent_id,
            exec_id="exec-inert",
            stopped=True,
            preserved_services=(
                (("target", self.target_id), ("grader", self.grader_id))
                if self.stop_mode != "missing_target"
                else (("grader", self.grader_id),)
            ),
        )

    async def close_async(self) -> None:
        self.trace.append("lease_closed")
        self.state = EnvironmentLeaseState.CLEANUP_FAILED if self.fail_close else EnvironmentLeaseState.CLOSED
        if self.fail_close:
            raise RuntimeError("Inert environment cleanup failed.")


class _FakeProcess:
    def __init__(
        self,
        *,
        runtime: _FakeRuntime,
        stdout: bytes,
        send_gateway: bool,
        pause: bool = False,
        exit_code: int = 0,
        identity_mode: str = "matched",
    ) -> None:
        self.runtime = runtime
        self.stdout = stdout
        self.send_gateway = send_gateway
        self.pause = pause
        self.exit_code = exit_code
        self.exec_id = "exec-other" if identity_mode == "wrong_exec" else "exec-inert"
        self.container_id = "d" * 64 if identity_mode == "wrong_container" else runtime.lease.agent_id
        if identity_mode == "missing_exec":
            self.exec_id = ""
        self.started = asyncio.Event()
        self.unblock = asyncio.Event()
        self.stopped = False

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        self.started.set()
        if self.pause:
            await self.unblock.wait()
        if self.send_gateway:
            await self.runtime.gateway_listener.send_inert_request_async()
        yield NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=self.stdout[:23])
        yield NativeCliProcessChunk(stream=NativeCliStream.STDERR, data=b"OFFLINE progress\n")
        yield NativeCliProcessChunk(stream=NativeCliStream.STDOUT, data=self.stdout[23:])

    async def wait_async(self) -> int:
        return self.exit_code

    async def stop_async(self) -> None:
        self.stopped = True
        self.unblock.set()
        await self.runtime.lease.stop_agent_async()


class _FakeLauncher:
    def __init__(self, *, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.calls: list[tuple[NativeCliRunConfig, str]] = []

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> _FakeProcess:
        self.calls.append((config, prompt))
        self.runtime.trace.append("agent_launch")
        self.runtime.process = _FakeProcess(
            runtime=self.runtime,
            stdout=_stdout(assistant=self.runtime.assistant, tool=self.runtime.tool),
            send_gateway=self.runtime.send_gateway,
            pause=self.runtime.pause,
            identity_mode=self.runtime.identity_mode,
        )
        return self.runtime.process


class _FakeRuntime:
    def __init__(
        self,
        *,
        run_id: str,
        sink: NativeCliDatabaseEvidenceSink,
        memory: SQLiteMemory,
        trace: list[str],
        stop_mode: str,
        send_gateway: bool,
        assistant: bool,
        tool: bool,
        pause: bool,
        grade_mode: str,
        identity_mode: str,
    ) -> None:
        self.trace = trace
        self.memory = memory
        self.stop_mode = stop_mode
        self.send_gateway = send_gateway
        self.assistant = assistant
        self.tool = tool
        self.pause = pause
        self.grade_mode = grade_mode
        self.identity_mode = identity_mode
        self.lease = _FakeStopOnlyLease(run_id=run_id, trace=trace, stop_mode=stop_mode)
        self.gateway_listener = _FakeGateway(run_id=run_id, observer=sink, trace=trace)
        self.launcher = _FakeLauncher(runtime=self)
        self.process: _FakeProcess | None = None
        self.grading_calls = 0

    async def verify_guest_exclusion_async(self) -> None:
        self.trace.append("guest_exclusion_verified")

    async def grade_async(self, *, report: NativeCliRunReport) -> NativeCliOriginalAssessment:
        self.grading_calls += 1
        self.trace.append("original_grader")
        assert self.lease.agent_stop is not None and self.lease.agent_stop.stopped
        assert self.gateway_listener.is_locally_running
        assert report.status is NativeCliReportStatus.INCOMPLETE and report.judgment is None
        assert report.cleanup is NativeCliReportCleanup.UNKNOWN
        assert report.evidence.coverage_complete
        assert not await asyncio.to_thread(self.memory.get_scores, score_type="float_scale")
        if self.grade_mode == "failed":
            raise RuntimeError("Inert original grader unavailable.")
        if self.grade_mode == "timeout":
            await asyncio.sleep(0.3)
        if self.grade_mode == "unknown":
            judgment = NativeCliOriginalJudgment(
                grader_ref="inert-original-grader", complete=False, rationale="No original grader outcome was acquired."
            )
        else:
            judgment = NativeCliOriginalJudgment(
                grader_ref="inert-original-grader",
                grader_evidence_ref="db-grader:inert",
                complete=True,
                value=0.75,
                rationale="Original fixture grader evaluated protected target state.",
            )
        artifacts = (
            NativeCliArtifactReference(
                name="fixture.txt",
                sha256=hashlib.sha256(b"OFFLINE/SIMULATED artifact").hexdigest(),
                size_bytes=len(b"OFFLINE/SIMULATED artifact"),
                evidence_ref="db-artifact:inert",
            ),
        )
        return NativeCliOriginalAssessment(
            judgment=judgment, artifacts=() if self.grade_mode == "no_artifact" else artifacts
        )


class _FakeBinding:
    def __init__(
        self,
        *,
        memory: SQLiteMemory,
        stop_mode: str = "confirmed",
        send_gateway: bool = True,
        assistant: bool = True,
        tool: bool = False,
        pause: bool = False,
        grade_mode: str = "complete",
        identity_mode: str = "matched",
        response_mode: NativeCyberResponseMode = NativeCyberResponseMode.MESSAGE_REQUIRED,
        converter: StringJoinConverter | None = None,
    ) -> None:
        self.name, self.version, self.task_id, self.task_version = "inert-binding", "rev-1", "inert-task", "rev-1"
        self.simulated = True
        self.run_config = NativeCliRunConfig(
            protocol=NativeCliProtocol.CODEX_EXEC_JSON,
            cli_version="0.115.0",
            cli_profile="sandbox-locked",
            agent_workdir=PurePosixPath("/workspace/task"),
            model_gateway_endpoint="http://172.18.0.1:8733/v1",
            max_steps=3,
            timeout_seconds=5,
        )
        self.response_mode = response_mode
        self.response_policy = NativeCyberResponsePolicy(
            allow_artifact_only=response_mode is NativeCyberResponseMode.ARTIFACT_ONLY
        )
        self.request_converters = (ConverterConfiguration(converters=[converter]),) if converter else ()
        self.runtime_timeout_seconds = 2.0
        self.grader_timeout_seconds = 2.0
        self.cleanup_timeout_seconds = 2.0
        self.raw_byte_limit = 1_048_576
        self.memory = memory
        self.stop_mode = stop_mode
        self.send_gateway = send_gateway
        self.assistant = assistant
        self.tool = tool
        self.pause = pause
        self.grade_mode = grade_mode
        self.identity_mode = identity_mode
        self.trace: list[str] = []
        self.runtime: _FakeRuntime | None = None

    async def open_runtime_async(self, *, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _FakeRuntime:
        episode = await asyncio.to_thread(self.memory.native_cyber_evidence.get_episode, run_id=run_id)
        assert episode.run.task_version == self.version
        assert episode.turns[0].source_turn_id == sink.turn_id
        assert {item.key.observed_source_id for item in episode.raw_streams} >= {
            "codex_exec_json.stdout",
            "codex_exec_json.stderr",
            "codex_exec_json.gateway.requests",
            "codex_exec_json.gateway.responses",
        }
        self.trace.append("episode_manifest_and_sink_ready")
        self.runtime = _FakeRuntime(
            run_id=run_id,
            sink=sink,
            memory=self.memory,
            trace=self.trace,
            stop_mode=self.stop_mode,
            send_gateway=self.send_gateway,
            assistant=self.assistant,
            tool=self.tool,
            pause=self.pause,
            grade_mode=self.grade_mode,
            identity_mode=self.identity_mode,
        )
        return self.runtime


async def test_one_shot_evaluation_uses_real_db_pregrade_and_finalizes_single_score_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    converter = StringJoinConverter(join_value="_")
    binding = _FakeBinding(memory=sqlite_instance, converter=converter)
    evaluation = NativeCliEvaluation(binding=binding, instruction="hello world", memory=sqlite_instance)
    store = evaluation._store
    real_pregrade = store.assess_cli_pregrading_coverage
    real_finalize = store.finalize_cli_episode_atomic

    def pregrade(*, report: NativeCliRunReport, expected_turns: int):
        assert binding.runtime is not None
        assert binding.runtime.lease.agent_stop is not None and binding.runtime.lease.agent_stop.stopped
        assert binding.runtime.gateway_listener.is_locally_running
        binding.trace.append("db_pregrade")
        return real_pregrade(report=report, expected_turns=expected_turns)

    def finalize(*, report: NativeCliRunReport, score, expected_turns: int):
        assert binding.runtime is not None
        assert binding.runtime.lease.state is EnvironmentLeaseState.CLOSED
        assert not binding.runtime.gateway_listener.is_locally_running
        assert not sqlite_instance.get_scores(score_type="float_scale")
        binding.trace.append("atomic_finalizer")
        return real_finalize(report=report, score=score, expected_turns=expected_turns)

    with (
        patch.object(store, "assess_cli_pregrading_coverage", side_effect=pregrade) as coverage_gate,
        patch.object(store, "finalize_cli_episode_atomic", side_effect=finalize) as atomic_finalizer,
        patch.object(converter, "convert_async", wraps=converter.convert_async) as convert,
    ):
        result = await evaluation.run_async()
    coverage_gate.assert_called_once()
    atomic_finalizer.assert_called_once()
    assert convert.call_count == 1
    assert binding.runtime is not None and binding.runtime.process is not None
    assert binding.runtime.launcher.calls == [(binding.run_config, "h_e_l_l_o w_o_r_l_d")]
    assert result.report.status is NativeCliReportStatus.COMPLETED
    assert result.pregrading is not None and result.pregrading.required_complete
    assert result.score.status is ScoreStatus.COMPLETE and result.score.get_value() == 0.75
    assert result.score.score_metadata["publication_state"] == "committed_final_result"
    assert result.episode.coverage_complete and result.episode.score_id == result.score.id
    assert binding.runtime.grading_calls == 1 and binding.runtime.process.stopped
    assert binding.runtime.lease.agent_stop is not None and binding.runtime.lease.agent_stop.stopped
    assert binding.runtime.lease.stop_calls >= 1 and binding.runtime.lease.state is EnvironmentLeaseState.CLOSED
    assert (
        binding.trace.index("agent_stop") < binding.trace.index("db_pregrade") < binding.trace.index("original_grader")
    )
    assert binding.trace.index("original_grader") < binding.trace.index("gateway_closed")
    assert (
        binding.trace.index("gateway_closed")
        < binding.trace.index("lease_closed")
        < binding.trace.index("atomic_finalizer")
    )
    assert result.report.cleanup is NativeCliReportCleanup.CLOSED
    assert result.report.task_version == result.episode.run.binding_version == "rev-1"
    assert result.report.conversation_id == evaluation.conversation_id
    assert result.episode.turns[0].request_piece_ids and result.episode.turns[0].response_piece_ids
    messages = sqlite_instance.get_conversation_messages(conversation_id=evaluation.conversation_id)
    assert result.episode.turns[0].request_piece_ids == (messages[0].get_piece().id,)
    assert result.episode.turns[0].response_piece_ids == (messages[1].get_piece().id,)
    assert [message.api_role for message in messages] == ["user", "assistant"]
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1
    assert isinstance(result.score.scorable, ContentEntryScorable)
    with pytest.raises(RuntimeError, match="cannot be run twice"):
        await evaluation.run_async()


async def test_missing_model_gateway_skips_grader_and_persists_only_undetermined_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance, send_gateway=False)
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert result.score.status is ScoreStatus.UNDETERMINED and result.score.score_value is None
    assert result.report.status is NativeCliReportStatus.INCOMPLETE
    assert result.pregrading is not None and not result.pregrading.required_complete
    assert any("gateway" in gap.lower() for gap in result.pregrading.required_gaps)
    assert any("gateway" in gap.lower() for gap in result.episode.gaps)
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_missing_persisted_assistant_still_links_genuine_request_and_skips_grade_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    evaluation = NativeCliEvaluation(binding=binding, instruction="offline fixture", memory=sqlite_instance)
    add_message = sqlite_instance.add_message_to_memory

    def omit_assistant(*, request):
        if request.api_role == "assistant":
            return None
        return add_message(request=request)

    with patch.object(sqlite_instance, "add_message_to_memory", side_effect=omit_assistant):
        result = await evaluation.run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert result.episode.turns[0].request_piece_ids and not result.episode.turns[0].response_piece_ids
    stored = sqlite_instance.get_conversation_messages(conversation_id=evaluation.conversation_id)
    assert result.episode.turns[0].request_piece_ids == (stored[0].get_piece().id,)
    assert result.report.status is NativeCliReportStatus.ERROR and result.score.is_undetermined
    assert any("message evidence mismatch" in reason for reason in result.report.errors)
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


@pytest.mark.parametrize("stop_mode", ["failed", "missing_receipt", "missing_target"])
async def test_unverified_stop_never_invokes_original_grader_async(
    sqlite_instance: SQLiteMemory, stop_mode: str
) -> None:
    binding = _FakeBinding(memory=sqlite_instance, stop_mode=stop_mode)
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert result.pregrading is None
    assert result.score.is_undetermined and result.report.status is NativeCliReportStatus.ERROR
    assert result.report.judgment is None
    assert any("agent stop" in error.lower() for error in result.report.errors)
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


@pytest.mark.parametrize("identity_mode", ["wrong_exec", "wrong_container", "missing_exec"])
async def test_stop_receipt_must_name_the_actual_launched_cli_process_async(
    sqlite_instance: SQLiteMemory, identity_mode: str
) -> None:
    binding = _FakeBinding(memory=sqlite_instance, identity_mode=identity_mode)
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert binding.runtime.lease.agent_stop is not None and binding.runtime.lease.agent_stop.stopped
    assert binding.runtime.process is not None and binding.runtime.process.stopped
    assert result.pregrading is None and result.report.status is NativeCliReportStatus.ERROR
    assert result.report.judgment is None and result.score.is_undetermined
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_cancelled_cli_is_stopped_and_publishes_undetermined_before_propagation_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance, pause=True)
    evaluation = NativeCliEvaluation(binding=binding, instruction="offline fixture", memory=sqlite_instance)
    task = asyncio.create_task(evaluation.run_async())
    while binding.runtime is None or binding.runtime.process is None:
        await asyncio.sleep(0)
    await asyncio.wait_for(binding.runtime.process.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert binding.runtime.process.stopped
    assert binding.runtime.grading_calls == 0
    assert evaluation.report is not None and evaluation.report.status is NativeCliReportStatus.CANCELLED
    assert evaluation.score is not None and evaluation.score.is_undetermined
    assert evaluation.episode is not None and not evaluation.episode.coverage_complete
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_original_grader_failure_is_not_retried_or_published_as_numeric_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance, grade_mode="failed")
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert result.pregrading is not None and result.pregrading.required_complete
    assert result.report.status is NativeCliReportStatus.ERROR and result.report.judgment is None
    assert result.score.is_undetermined and result.report.cleanup is NativeCliReportCleanup.CLOSED
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


@pytest.mark.parametrize("cleanup_part", ["gateway", "environment"])
async def test_cleanup_failure_retains_original_judgment_but_not_numeric_score_async(
    sqlite_instance: SQLiteMemory, cleanup_part: str
) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    original_open = binding.open_runtime_async

    async def open_with_failed_cleanup_async(*, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _FakeRuntime:
        runtime = await original_open(run_id=run_id, sink=sink)
        if cleanup_part == "gateway":
            runtime.gateway_listener.fail_close = True
        else:
            runtime.lease.fail_close = True
        return runtime

    with patch.object(binding, "open_runtime_async", side_effect=open_with_failed_cleanup_async):
        result = await NativeCliEvaluation(
            binding=binding, instruction="offline fixture", memory=sqlite_instance
        ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert result.report.status is NativeCliReportStatus.ERROR
    assert result.report.cleanup is NativeCliReportCleanup.FAILED
    assert result.report.judgment is not None and result.report.judgment.value == 0.75
    assert result.score.is_undetermined
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_codex_tool_only_artifact_turn_is_undetermined_until_model_tool_provenance_exists_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(
        memory=sqlite_instance, assistant=False, tool=True, response_mode=NativeCyberResponseMode.ARTIFACT_ONLY
    )
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert result.score.is_undetermined and result.report.judgment is None
    assert result.pregrading is not None and not result.pregrading.required_complete
    assert any("Codex tool execution" in gap for gap in result.pregrading.required_gaps)
    assert any("Codex tool execution" in gap for gap in result.episode.gaps)


async def test_task_approved_write_only_artifact_result_is_original_grader_outcome_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(
        memory=sqlite_instance, assistant=False, tool=False, response_mode=NativeCyberResponseMode.ARTIFACT_ONLY
    )
    result = await NativeCliEvaluation(
        binding=binding, instruction="Create target-side artifact", memory=sqlite_instance
    ).run_async()
    assert result.pregrading is not None and result.pregrading.required_complete
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert result.report.status is NativeCliReportStatus.COMPLETED
    assert result.report.artifacts[0].evidence_ref == "db-artifact:inert"
    assert result.score.get_value() == 0.75
    assert result.episode.coverage_complete and not result.episode.turns[0].response_piece_ids
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_missing_artifact_from_original_grader_is_undetermined_async(sqlite_instance: SQLiteMemory) -> None:
    binding = _FakeBinding(
        memory=sqlite_instance,
        assistant=False,
        tool=False,
        response_mode=NativeCyberResponseMode.ARTIFACT_ONLY,
        grade_mode="no_artifact",
    )
    result = await NativeCliEvaluation(
        binding=binding, instruction="Create target-side artifact", memory=sqlite_instance
    ).run_async()
    assert result.pregrading is not None and result.pregrading.required_complete
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert result.report.status is NativeCliReportStatus.INCOMPLETE and not result.report.artifacts
    assert result.report.judgment is not None and result.report.judgment.value == 0.75
    assert result.score.is_undetermined and len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_grader_deadline_is_bounded_without_publishing_numeric_result_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance, grade_mode="timeout")
    binding.grader_timeout_seconds = 0.05
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert result.pregrading is not None and result.pregrading.required_complete
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert result.score.is_undetermined and result.report.status is NativeCliReportStatus.ERROR
    assert result.report.cleanup is NativeCliReportCleanup.CLOSED
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_atomic_finalizer_failure_leaves_no_success_shaped_result_async(sqlite_instance: SQLiteMemory) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    evaluation = NativeCliEvaluation(binding=binding, instruction="offline fixture", memory=sqlite_instance)
    with patch.object(evaluation._store, "finalize_cli_episode_atomic", side_effect=RuntimeError("inert DB refusal")):
        with pytest.raises(RuntimeError, match="inert DB refusal"):
            await evaluation.run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert binding.runtime.lease.state is EnvironmentLeaseState.CLOSED
    assert evaluation.report is not None
    assert evaluation.episode is None and evaluation.score is None
    assert not sqlite_instance.get_scores(score_type="float_scale")


async def test_cancellation_during_atomic_commit_reports_actual_persisted_result_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    evaluation = NativeCliEvaluation(binding=binding, instruction="offline fixture", memory=sqlite_instance)
    actual_finalize = evaluation._store.finalize_cli_episode_atomic
    inside_commit = threading.Event()
    release_commit = threading.Event()

    def delayed_commit(*, report: NativeCliRunReport, score, expected_turns: int):
        inside_commit.set()
        if not release_commit.wait(timeout=3):
            raise RuntimeError("Timed out waiting for the inert commit gate.")
        return actual_finalize(report=report, score=score, expected_turns=expected_turns)

    with patch.object(evaluation._store, "finalize_cli_episode_atomic", side_effect=delayed_commit):
        task = asyncio.create_task(evaluation.run_async())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(inside_commit.wait, 2), timeout=3)
            task.cancel()
            await asyncio.sleep(0)
        finally:
            release_commit.set()
        result = await task
    assert result.report.status is NativeCliReportStatus.COMPLETED
    assert result.cancellation_after_commit_started
    assert result.score.status is ScoreStatus.COMPLETE and result.episode.coverage_complete
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_guest_exclusion_failure_aborts_before_any_cli_launch_or_grading_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    original_open = binding.open_runtime_async

    async def open_unqualified_runtime_async(*, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _FakeRuntime:
        runtime = await original_open(run_id=run_id, sink=sink)
        with patch.object(runtime, "verify_guest_exclusion_async", side_effect=PermissionError("inert exclusion")):
            await runtime.verify_guest_exclusion_async()
        raise PermissionError("Protected guest exclusion could not be verified.")

    with patch.object(binding, "open_runtime_async", side_effect=open_unqualified_runtime_async):
        result = await NativeCliEvaluation(
            binding=binding, instruction="offline fixture", memory=sqlite_instance
        ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert not binding.runtime.launcher.calls
    assert result.score.is_undetermined and result.report.status is NativeCliReportStatus.ERROR
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_unqualified_gateway_route_blocks_send_and_grader_async(sqlite_instance: SQLiteMemory) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    original_open = binding.open_runtime_async

    async def open_with_dead_gateway_async(*, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _FakeRuntime:
        runtime = await original_open(run_id=run_id, sink=sink)
        runtime.gateway_listener.running = False
        return runtime

    with patch.object(binding, "open_runtime_async", side_effect=open_with_dead_gateway_async):
        result = await NativeCliEvaluation(
            binding=binding, instruction="offline fixture", memory=sqlite_instance
        ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 0
    assert not binding.runtime.launcher.calls
    assert result.score.is_undetermined and result.report.status is NativeCliReportStatus.ERROR


async def test_bounded_runtime_acquisition_failure_is_persisted_undetermined_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    binding.runtime_timeout_seconds = 0.05

    async def unavailable_runtime_async(*, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _FakeRuntime:
        await asyncio.sleep(1)
        raise AssertionError("The bounded runtime acquisition should have timed out.")

    with patch.object(binding, "open_runtime_async", side_effect=unavailable_runtime_async):
        result = await NativeCliEvaluation(
            binding=binding, instruction="offline fixture", memory=sqlite_instance
        ).run_async()
    assert binding.runtime is None
    assert result.report.status is NativeCliReportStatus.ERROR
    assert result.report.cleanup is NativeCliReportCleanup.UNKNOWN
    assert result.score.is_undetermined and len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_gateway_shutdown_timeout_keeps_original_grade_but_not_numeric_publication_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    binding.cleanup_timeout_seconds = 0.05
    original_open = binding.open_runtime_async

    async def open_with_stuck_listener_async(*, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _FakeRuntime:
        runtime = await original_open(run_id=run_id, sink=sink)

        async def stuck_close_async() -> None:
            await asyncio.sleep(1)

        runtime.gateway_listener.close_async = stuck_close_async
        return runtime

    with patch.object(binding, "open_runtime_async", side_effect=open_with_stuck_listener_async):
        result = await NativeCliEvaluation(
            binding=binding, instruction="offline fixture", memory=sqlite_instance
        ).run_async()
    assert binding.runtime is not None and binding.runtime.grading_calls == 1
    assert binding.runtime.lease.state is EnvironmentLeaseState.CLOSED
    assert result.report.cleanup is NativeCliReportCleanup.FAILED
    assert result.report.judgment is not None and result.report.judgment.value == 0.75
    assert result.score.is_undetermined
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_cli_timeout_stops_agent_and_persists_no_numeric_grade_async(sqlite_instance: SQLiteMemory) -> None:
    binding = _FakeBinding(memory=sqlite_instance, pause=True)
    binding.run_config = replace(binding.run_config, timeout_seconds=0.05)
    result = await NativeCliEvaluation(
        binding=binding, instruction="offline fixture", memory=sqlite_instance
    ).run_async()
    assert binding.runtime is not None and binding.runtime.process is not None
    assert binding.runtime.process.stopped
    assert binding.runtime.grading_calls == 0
    assert result.report.status is NativeCliReportStatus.ERROR
    assert result.score.is_undetermined
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


def test_binding_rejects_task_version_mismatch_before_resource_acquisition(sqlite_instance: SQLiteMemory) -> None:
    binding = _FakeBinding(memory=sqlite_instance)
    binding.task_version = "foreign-task-revision"
    with pytest.raises(ValueError, match="task provenance"):
        NativeCliEvaluation(binding=binding, instruction="offline fixture", memory=sqlite_instance)
    assert not binding.trace

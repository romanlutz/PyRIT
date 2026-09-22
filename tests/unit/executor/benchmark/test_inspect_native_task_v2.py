# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pyrit.executor.benchmark._inspect_v2_support import InspectRunArtifactsV2
from pyrit.executor.benchmark.inspect_native_task_v2 import InspectNativeTaskBridgeV2
from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionEnvironmentHooksV2, SubmissionLimitsV2
from pyrit.memory import CentralMemory
from pyrit.models import ContentEntryScorable, ScoreStatus
from pyrit.models.submission import SubmissionCleanupStatus
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2
from tests.unit.executor.benchmark.test_inspect_native_task import OfflineFixtureBinding
from tests.unit.mocks import openai_response_json_dict

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from pyrit.executor.benchmark.inspect_native_task_v2 import InspectNativeTaskResultV2

pytest.importorskip("inspect_ai")
pytestmark = pytest.mark.usefixtures("patch_central_database")


class V2FixtureBinding(OfflineFixtureBinding):
    """Reuse only newly authored public inert fixture bytes with an explicit v2 report."""

    def __init__(self, outcomes: dict[str, float | str]) -> None:
        super().__init__(outcomes)
        self.report["contract_version"] = "strict-submission-v2"
        self.v2_bridge: InspectNativeTaskBridgeV2 | None = None
        self.v2_events: list[str] = []


async def _run_v2_async(
    *,
    tmp_path: Path,
    binding: V2FixtureBinding,
    commands: list[str | None],
    limits: SubmissionLimitsV2 | None = None,
    attach_hooks: bool = True,
    sdk_retries: int = 0,
    status_code: int = 200,
    cleanup: SubmissionCleanupStatus = SubmissionCleanupStatus.COMPLETE,
    audit_error: Exception | None = None,
    exit_error: Exception | None = None,
    include_usage: bool = True,
    capture_cancellation: bool = False,
    tool_bound: int | None = None,
    block_audit: bool = False,
    transport_failure: str | None = None,
    audit_entered: asyncio.Event | None = None,
    cleanup_entered: asyncio.Event | None = None,
    cleanup_release: asyncio.Event | None = None,
    cleanup_error: Exception | None = None,
) -> tuple[InspectNativeTaskResultV2, list[dict[str, Any]], list[str], InspectNativeTaskBridgeV2]:
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageSystem
    from inspect_ai.scorer import Score, Target, scorer
    from inspect_ai.solver import Generate, Solver, TaskState, solver
    from inspect_ai.tool import Tool, tool

    binding.owned_task = asyncio.current_task()
    events: list[str] = []
    binding.v2_events = events
    requests: list[dict[str, Any]] = []

    @tool
    def submit_fixture_v2() -> Tool:
        async def invoke_async(artifact_ref: str) -> str:
            """Submit only inert OFFLINE/SIMULATED bytes.

            Args:
                artifact_ref: A fixture identifier, never an executable artifact.
            """
            events.append(f"tool:{artifact_ref}")
            return await binding.submit_async(artifact_ref)

        return invoke_async

    @solver
    def original_solver_v2() -> Solver:
        async def invoke_async(state: TaskState, generate: Generate) -> TaskState:
            state.store.set("original_setup", True)
            state.messages.insert(0, ChatMessageSystem(content="OFFLINE/SIMULATED original system"))
            state.tools = [submit_fixture_v2()]
            while not state.completed:
                await generate(state, tool_calls="loop")
            return state

        return invoke_async

    @scorer(metrics=[])
    def latest_grade_v2() -> Any:
        async def score_async(state: TaskState, target: Target) -> Score:
            assert state.store.get("original_setup") is True
            events.append("native_grade")
            value = binding.read_report()["last_valid_grade"]
            return Score(value=value if value is not None else "NO_SUBMISSION")

        return score_async

    def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        events.append("provider")
        assert len(requests) <= len(commands), "Unexpected provider request"
        if transport_failure is not None and len(requests) == 2:
            if transport_failure == "cancel":
                raise asyncio.CancelledError("OFFLINE provider transport cancelled before response")
            raise httpx.ConnectError("OFFLINE transport failed before response", request=request)
        if status_code != 200:
            return httpx.Response(status_code, json={"error": {"message": "OFFLINE provider failure"}})
        response = openai_response_json_dict()
        response["model"] = "fixture-v2-model"
        if include_usage:
            response["usage"] = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        command = commands[len(requests) - 1]
        if command is None:
            response["output"][0]["content"][0]["text"] = f"OFFLINE thought {len(requests)}"
        else:
            response["output"] = [
                {
                    "type": "function_call",
                    "call_id": f"v2-call-{len(requests)}",
                    "id": f"v2-function-{len(requests)}",
                    "name": "submit_fixture_v2",
                    "arguments": json.dumps({"artifact_ref": command}),
                    "status": "completed",
                }
            ]
        return httpx.Response(200, json=response)

    @asynccontextmanager
    async def factory_async(
        *,
        tools: list[dict[str, Any]],
        request_hook: Callable[[httpx.Request], Awaitable[None]],
        response_hook: Callable[[httpx.Response], Awaitable[None]],
    ) -> AsyncIterator[OpenAIResponseTarget]:
        events.append("factory_enter")
        hooks = {"request": [request_hook], "response": [response_hook]} if attach_hooks else None
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider), event_hooks=hooks) as client:
            try:
                yield OpenAIResponseTarget(
                    endpoint="https://fixture.invalid/v1",
                    api_key="OFFLINE-not-a-secret",
                    model_name="fixture-v2-model",
                    auto_execute_tools=False,
                    extra_body_parameters={"tools": tools, "parallel_tool_calls": False, "store": False},
                    httpx_client_kwargs={"http_client": client, "max_retries": sdk_retries},
                )
            finally:
                events.append("factory_exit")
        assert client.is_closed
        if exit_error is not None:
            raise exit_error

    async def audit_async() -> dict[str, Any]:
        events.append("audit")
        if audit_entered is not None:
            audit_entered.set()
        if audit_error is not None:
            raise audit_error
        if block_audit:
            await asyncio.Event().wait()
        return {"evidence": "OFFLINE/SIMULATED", "resources_created": False}

    async def cleanup_async() -> SubmissionCleanupStatus:
        events.append("cleanup")
        if cleanup_entered is not None:
            cleanup_entered.set()
        if cleanup_release is not None:
            await cleanup_release.wait()
        if cleanup_error is not None:
            raise cleanup_error
        return cleanup

    bridge = InspectNativeTaskBridgeV2(
        target_factory=factory_async,
        read_report=binding.read_report,
        directory=tmp_path / "v2-run",
        environment=SubmissionEnvironmentHooksV2(audit_async=audit_async, cleanup_async=cleanup_async),
        limits=limits or SubmissionLimitsV2(max_requests=len(commands), max_tool_calls=8),
        max_tool_output_bytes=tool_bound,
    )
    binding.v2_bridge = bridge
    with (
        patch.dict("os.environ", {"RETRY_MAX_NUM_ATTEMPTS": "1"}),
        patch("socket.socket.connect", side_effect=AssertionError("No live sockets")),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=AssertionError("No workloads")),
    ):
        try:
            result = await bridge.execute_async(
                task=Task(
                    dataset=[Sample(id="inert-v2", input="OFFLINE/SIMULATED original input")],
                    solver=original_solver_v2(),
                    scorer=latest_grade_v2(),
                ),
                sample_id="inert-v2",
                native_scorer="latest_grade_v2",
            )
        except asyncio.CancelledError:
            if not capture_cancellation:
                raise
            assert bridge.last_result is not None
            result = bridge.last_result
    return result, requests, events, bridge


@pytest.mark.parametrize("grades", [[0.75, 0.25], [0.75, 0.0], [0.25, 0.75]])
async def test_v2_uses_original_task_and_latest_numeric_report_with_owned_context_async(
    tmp_path: Path, grades: list[float]
) -> None:
    binding = V2FixtureBinding({"a": grades[0], "b": grades[1]})
    result, requests, events, bridge = await _run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a", "b"])
    assert result.score.get_value() == grades[-1], result.report.runner_error
    assert result.report.report.selected_submission_id == "offline-submission-2"
    assert result.report.mode.value == "offline" and result.report.simulated is True
    assert result.report.evidence_label == "OFFLINE/SIMULATED"
    assert result.report.target_identifier["model_name"] == "fixture-v2-model"
    assert result.report.provider_request_count == result.report.generation_count == len(requests) == 2
    assert result.report.total_tokens == 24
    assert result.report.message_count == 6
    assert result.report.local_cleanup is SubmissionCleanupStatus.COMPLETE
    assert result.report.environment_audit["resources_created"] is False
    assert events[0] == "audit" and events[-2:] == ["cleanup", "factory_exit"]
    assert requests[0]["input"][0]["content"][0]["text"] == "OFFLINE/SIMULATED original system"
    assert requests[1]["input"][-1]["output"] == f"OFFLINE grade {grades[0]}\nexact native feedback"
    assert result.report.final_text is None
    assert result.native_log is not None and result.native_log.exists()
    assert bridge.last_result is result
    assert isinstance(result.score.scorable, ContentEntryScorable)
    replay = await SubmissionReportScorerV2(report_sha256=result.report.sha256()).score_async(
        scorable=result.score.scorable
    )
    assert replay[0].get_value() == grades[-1] and binding.evaluator.await_count == 2
    lines = [json.loads(line) for line in (tmp_path / "v2-run" / "events.jsonl").read_text().splitlines()]
    assert all(line["mode"] == "offline" and line["simulated"] and line["schema_version"] == 2 for line in lines)
    assert "OFFLINE-not-a-secret" not in (tmp_path / "v2-run" / "events.jsonl").read_text()


async def test_v2_full_success_and_continuation_use_real_messages_without_extra_dispatch_async(tmp_path: Path) -> None:
    binding = V2FixtureBinding({"success": 1.0})
    result, requests, events, _ = await _run_v2_async(
        tmp_path=tmp_path, binding=binding, commands=[None, "success", None]
    )
    assert result.score.get_value() == 1.0, result.report.runner_error
    assert len(requests) == 2 and binding.evaluator.await_count == 1
    assert requests[1]["input"] == [
        *requests[0]["input"],
        {"role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE thought 1"}]},
    ]
    assert events.count("factory_enter") == events.count("factory_exit") == 1
    assert result.report.calls[-1].feedback == "OFFLINE grade 1.0\nexact native feedback"


@pytest.mark.parametrize("outcome", ["dependency", "infra", "unknown", "partial_evidence", "unknown_cleanup"])
async def test_v2_failure_or_uncertainty_retains_observation_without_clean_score_async(
    tmp_path: Path, outcome: str
) -> None:
    binding = V2FixtureBinding({"a": 0.75, "failure": outcome})
    result, requests, events, _ = await _run_v2_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "failure", "failure"]
    )
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert len(requests) == 2
    assert binding.evaluator.await_count == (1 if outcome == "dependency" else 2)
    assert result.report.report.last_valid_grade == (
        0.5 if outcome in ("partial_evidence", "unknown_cleanup") else 0.75
    )
    assert events[-2:] == ["cleanup", "factory_exit"]


@pytest.mark.parametrize("cleanup", [SubmissionCleanupStatus.UNKNOWN, SubmissionCleanupStatus.FAILED])
async def test_v2_uncertain_local_cleanup_is_not_a_clean_run_async(
    tmp_path: Path, cleanup: SubmissionCleanupStatus
) -> None:
    result, _, _, _ = await _run_v2_async(
        tmp_path=tmp_path, binding=V2FixtureBinding({"a": 0.25}), commands=["a"], cleanup=cleanup
    )
    assert result.report.status.value == "incomplete"
    assert result.report.report.last_valid_grade == 0.25
    assert result.score.status is ScoreStatus.UNDETERMINED


@pytest.mark.parametrize("failure", ["audit", "context_exit", "hooks", "retry"])
async def test_v2_lifecycle_and_observer_failures_never_look_successful_async(tmp_path: Path, failure: str) -> None:
    binding = V2FixtureBinding({"success": 1.0})
    result, requests, events, _ = await _run_v2_async(
        tmp_path=tmp_path,
        binding=binding,
        commands=["success", "success"],
        attach_hooks=failure != "hooks",
        sdk_retries=1 if failure == "retry" else 0,
        status_code=429 if failure == "retry" else 200,
        audit_error=RuntimeError("OFFLINE audit failed") if failure == "audit" else None,
        exit_error=RuntimeError("OFFLINE exit failed") if failure == "context_exit" else None,
    )
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert len(requests) == (0 if failure == "audit" else 1)
    assert "cleanup" in events
    if failure != "audit":
        assert "factory_exit" in events


def test_v2_real_provenance_can_be_validated_without_target_auth_or_dispatch(tmp_path: Path) -> None:
    binding = V2FixtureBinding({})
    binding.report.update(mode="real", simulated=False)
    factory = AsyncMock()
    audit, cleanup = AsyncMock(), AsyncMock()
    bridge = InspectNativeTaskBridgeV2(
        target_factory=factory,
        read_report=binding.read_report,
        directory=tmp_path / "schema-only",
        environment=SubmissionEnvironmentHooksV2(audit_async=audit, cleanup_async=cleanup),
        limits=SubmissionLimitsV2(),
    )
    assert bridge.last_result is None
    assert not (tmp_path / "schema-only").exists()
    factory.assert_not_called()
    audit.assert_not_awaited()
    cleanup.assert_not_awaited()


def test_v2_public_definitions_remain_importable_without_inspect_optional_packages() -> None:
    code = """
import sys
sys.modules['inspect_ai'] = None
sys.modules['inspect_evals'] = None
from pyrit.executor.benchmark.inspect_native_task_v2 import InspectNativeTaskBridgeV2
from pyrit.models.submission_v2 import SubmissionProvenanceV2, StrictSubmissionReportV2
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2
assert 'inspect_ai.log' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("limits", "expected_requests", "expected_grade"),
    [
        (SubmissionLimitsV2(max_requests=3, max_tool_calls=1), 1, 0.75),
        (SubmissionLimitsV2(max_requests=3, max_tokens=1), 1, 0.75),
        (SubmissionLimitsV2(max_requests=3, max_messages=4), 1, 0.75),
        (SubmissionLimitsV2(max_requests=3, max_messages=2), 0, None),
    ],
)
async def test_v2_distinct_budgets_stop_between_completed_operations_async(
    tmp_path: Path, limits: SubmissionLimitsV2, expected_requests: int, expected_grade: float | None
) -> None:
    binding = V2FixtureBinding({"a": 0.75, "b": 0.25})
    result, requests, _, _ = await _run_v2_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "b", None], limits=limits
    )
    assert len(requests) == binding.evaluator.await_count == expected_requests
    assert result.report.termination_reason.value == "budget"
    if expected_grade is None:
        assert result.score.status is ScoreStatus.UNDETERMINED and result.report.status.value == "no_submission"
    else:
        assert result.score.get_value() == expected_grade


async def test_v2_missing_required_usage_stops_before_an_evaluator_async(tmp_path: Path) -> None:
    binding = V2FixtureBinding({"a": 0.75})
    result, requests, _, _ = await _run_v2_async(
        tmp_path=tmp_path,
        binding=binding,
        commands=["a", "a"],
        limits=SubmissionLimitsV2(max_tokens=30),
        include_usage=False,
    )
    assert len(requests) == 1
    assert result.report.total_tokens is None and result.report.token_usage == (None,)
    assert result.score.status is ScoreStatus.UNDETERMINED and result.report.status.value == "incomplete"
    binding.evaluator.assert_not_awaited()


@pytest.mark.parametrize("commands", [["missing", "missing"], ["a", "missing"]])
async def test_v2_preserves_exact_recoverable_tool_feedback_without_erasing_prior_grade_async(
    tmp_path: Path, commands: list[str]
) -> None:
    binding = V2FixtureBinding({"a": 0.75})
    result, requests, _, _ = await _run_v2_async(tmp_path=tmp_path, binding=binding, commands=commands)
    assert result.report.calls[-1].feedback == "Error: OFFLINE missing artifact"
    assert result.report.calls[-1].feedback_kind.value == "recoverable_error"
    if commands[0] == "missing":
        assert requests[1]["input"][-1]["output"] == "Error: OFFLINE missing artifact"
        binding.evaluator.assert_not_awaited()
        assert result.report.status.value == "no_submission" and result.score.is_undetermined
    else:
        assert result.score.get_value() == 0.75
        binding.evaluator.assert_awaited_once()


@pytest.mark.parametrize(
    "changed", [{"mode": "real", "simulated": False}, {"contract_version": "strict-submission-v1"}]
)
async def test_v2_contract_or_mode_cannot_change_after_binding_async(tmp_path: Path, changed: dict[str, Any]) -> None:
    binding = V2FixtureBinding({"a": 0.75})
    original_submit = binding.submit_async

    async def change_async(artifact_ref: str) -> str:
        feedback = await original_submit(artifact_ref)
        binding.report.update(changed)
        return feedback

    with patch.object(binding, "submit_async", new=change_async):
        result, requests, _, _ = await _run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a", "a"])
    assert len(requests) == 1 and binding.evaluator.await_count == 1
    assert result.report.mode.value == "offline" and result.report.evidence_label == "OFFLINE/SIMULATED"
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert result.report.status.value == "error"
    records = [json.loads(line) for line in (tmp_path / "v2-run" / "events.jsonl").read_text().splitlines()]
    assert all(item["mode"] == "offline" and item["simulated"] is True for item in records)


async def test_v2_model_free_task_ignores_ambient_inspect_model_configuration_async(tmp_path: Path) -> None:
    from inspect_ai.log import read_eval_log_async

    binding = V2FixtureBinding({"success": 1.0})
    with patch.dict("os.environ", {"INSPECT_EVAL_MODEL": "invalid/no-provider-must-be-created"}):
        result, requests, _, _ = await _run_v2_async(tmp_path=tmp_path, binding=binding, commands=["success"])
    assert result.score.get_value() == 1.0 and len(requests) == 1
    assert result.native_log is not None
    log = await read_eval_log_async(result.native_log)
    assert log.eval.model == "none/none"
    assert log.eval.metadata["mode"] == "offline" and log.eval.metadata["simulated"] is True
    assert result.report.target_identifier["model_name"] == "fixture-v2-model"


async def test_v2_tool_cancellation_keeps_original_report_and_closes_context_async(tmp_path: Path) -> None:
    binding = V2FixtureBinding({"a": 0.75, "failure": "cancelled"})
    result, requests, events, bridge = await _run_v2_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "failure", "failure"], capture_cancellation=True
    )
    assert result is bridge.last_result
    assert result.report.status.value == "cancelled" and result.score.status is ScoreStatus.UNDETERMINED
    assert result.report.report.last_valid_grade == 0.75
    assert len(requests) == binding.evaluator.await_count == 2
    assert result.report.report.submissions[-1].remote_disposition.value == "unknown"
    assert events[-2:] == ["cleanup", "factory_exit"]
    assert result.native_log is not None


async def test_v2_cancel_after_callback_return_retains_actual_strings_and_latest_observation_async(
    tmp_path: Path,
) -> None:
    binding = V2FixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"
    pending = asyncio.create_task(_run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"]))
    with pytest.raises(asyncio.CancelledError, match="callback-return boundary"):
        await pending
    assert binding.v2_bridge is not None and binding.v2_bridge.last_result is not None
    result = binding.v2_bridge.last_result
    assert result.report.status.value == "cancelled" and result.score.is_undetermined
    assert result.report.report.last_valid_grade == 0.25
    assert [call.feedback for call in result.report.calls] == [
        "OFFLINE grade 0.75\nexact native feedback",
        "OFFLINE grade 0.25\nexact native feedback",
    ]
    assert binding.evaluator.await_count == result.report.provider_request_count == 2
    pieces = CentralMemory.get_memory_instance().get_message_pieces(conversation_id=result.report.conversation_id)
    outputs = [
        json.loads(piece.original_value) for piece in pieces if piece.original_value_data_type == "function_call_output"
    ]
    assert [item["call_id"] for item in outputs] == ["v2-call-1", "v2-call-2"]
    assert outputs[-1]["output"] == result.report.calls[-1].feedback
    assert result.report_path.read_text() == result.report.canonical_json()


async def test_v2_precommit_cancel_retains_only_an_undetermined_publication_async(tmp_path: Path) -> None:
    binding = V2FixtureBinding({"a": 0.25})
    prepared, release = asyncio.Event(), asyncio.Event()
    original_save = InspectRunArtifactsV2.save_async

    async def hold_async(self: InspectRunArtifactsV2) -> None:
        if self.manifest.get("run_status") == "completed" and not prepared.is_set():
            prepared.set()
            await release.wait()
        await original_save(self)

    with patch.object(InspectRunArtifactsV2, "save_async", new=hold_async):
        pending = asyncio.create_task(_run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a"]))
        try:
            await asyncio.wait_for(prepared.wait(), timeout=30)
            pending.cancel("OFFLINE first cancellation before v2 publication")
            release.set()
            with pytest.raises(asyncio.CancelledError, match="before v2 publication"):
                await pending
        finally:
            release.set()
    assert binding.v2_bridge is not None and binding.v2_bridge.last_result is not None
    result = binding.v2_bridge.last_result
    assert result.report.status.value == "cancelled" and result.score.is_undetermined
    scores = [
        score
        for score in CentralMemory.get_memory_instance().get_scores(score_type="float_scale")
        if score.score_metadata.get("run_id") == result.report.run_id
    ]
    assert len(scores) == 1 and scores[0].is_undetermined
    assert result.report.report.last_valid_grade == 0.25


@pytest.mark.parametrize("failure", ["retention", "manifest", "same_cancellation", None])
@pytest.mark.parametrize("repeat_cancel", [False, True])
async def test_v2_first_publication_cancel_preserves_original_over_retention_failures_async(
    tmp_path: Path, failure: str | None, repeat_cancel: bool
) -> None:
    from pyrit.executor.benchmark.submission.evidence_v2 import SubmissionEvidenceWriterV2

    binding = V2FixtureBinding({"a": 0.75, "b": 0.25})
    prepared, recovering, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations: list[asyncio.CancelledError] = []
    interrupted_reports = []
    candidate_paths = []
    secondary = OSError(f"OFFLINE cancelled-report {failure} failed")
    original_save = InspectRunArtifactsV2.save_async
    original_retain = SubmissionEvidenceWriterV2.retain_async

    async def retain_async(self: SubmissionEvidenceWriterV2, report: Any) -> Any:
        if report.status.value == "cancelled":
            interrupted_reports.append(report)
            recovering.set()
            await release.wait()
            if failure == "retention":
                raise secondary
            if failure == "same_cancellation":
                raise cancellations[0]
        path = await original_retain(self, report)
        if report.status.value == "cancelled":
            candidate_paths.append(path)
        return path

    async def save_async(self: InspectRunArtifactsV2) -> None:
        if self.manifest.get("run_status") == "completed" and not prepared.is_set():
            prepared.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                cancellations.append(error)
                raise
        if self.manifest.get("run_status") == "cancelled" and failure == "manifest":
            raise secondary
        await original_save(self)

    with (
        patch.object(SubmissionEvidenceWriterV2, "retain_async", new=retain_async),
        patch.object(InspectRunArtifactsV2, "save_async", new=save_async),
    ):
        pending = asyncio.create_task(_run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a", "b"]))
        try:
            await asyncio.wait_for(prepared.wait(), timeout=30)
            pending.cancel("OFFLINE original first publication cancellation")
            await asyncio.wait_for(recovering.wait(), timeout=10)
            if repeat_cancel:
                for _ in range(2):
                    pending.cancel("OFFLINE later cancellation during retention")
                    await asyncio.sleep(0)
                assert not pending.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match="original first publication cancellation") as caught:
                await pending
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
                with suppress(asyncio.CancelledError):
                    await pending

    assert caught.value is cancellations[0]
    assert caught.value.__cause__ is not caught.value
    if failure in ("retention", "manifest"):
        assert caught.value.__cause__ is secondary
        assert any(str(secondary) in note for note in caught.value.__notes__)
    if repeat_cancel:
        assert any("Additional caller cancellation" in note for note in caught.value.__notes__)
    assert len(interrupted_reports) == 1
    report = interrupted_reports[0]
    assert report.status.value == "cancelled"
    assert report.report.last_valid_grade == 0.25
    assert report.report.selected_submission_id == "offline-submission-2"
    assert [call.feedback for call in report.calls] == [
        "OFFLINE grade 0.75\nexact native feedback",
        "OFFLINE grade 0.25\nexact native feedback",
    ]
    assert binding.evaluator.await_count == binding.v2_events.count("provider") == 2
    assert binding.v2_events.count("cleanup") == binding.v2_events.count("factory_exit") == 1
    assert binding.v2_bridge is not None
    scores = CentralMemory.get_memory_instance().get_scores(score_type="float_scale")
    assert all(score.is_undetermined for score in scores)
    if failure is None:
        published = binding.v2_bridge.last_result
        assert published is not None and published.score.is_undetermined
        assert len(scores) == 1
        assert published.report_path.read_text() == published.report.canonical_json()
    else:
        assert binding.v2_bridge.last_result is None and not scores
    if failure == "manifest":
        assert len(candidate_paths) == 1
        assert candidate_paths[0].read_text() == report.canonical_json()


@pytest.mark.parametrize("bound", [4096, 65536])
async def test_v2_native_feedback_is_exact_or_stops_explicitly_before_forwarding_async(
    tmp_path: Path, bound: int
) -> None:
    binding = V2FixtureBinding({"a": 0.75, "success": 1.0})
    feedback = ("OFFLINE exact feedback\n" * 1024) + "\u03bb\n"
    original = binding.submit_async

    async def long_feedback_async(artifact_ref: str) -> str:
        returned = await original(artifact_ref)
        if artifact_ref == "a":
            binding.report["submissions"][-1]["feedback"] = feedback
            return feedback
        return returned

    with patch.object(binding, "submit_async", new=long_feedback_async):
        result, requests, _, _ = await _run_v2_async(
            tmp_path=tmp_path, binding=binding, commands=["a", "success", None], tool_bound=bound
        )
    assert result.report.calls[0].feedback == feedback
    if bound > len(feedback.encode("utf-8")):
        assert requests[1]["input"][-1]["output"] == feedback
        assert result.score.get_value() == 1.0 and len(requests) == 2
    else:
        assert len(requests) == binding.evaluator.await_count == 1
        assert result.report.status.value == "incomplete" and result.score.is_undetermined
        assert result.report.report.last_valid_grade == 0.75


async def test_v2_external_cancellation_closes_owned_lifetime_and_retains_unknown_remote_outcome_async(
    tmp_path: Path,
) -> None:
    binding = V2FixtureBinding({"a": "unknown"})
    entered = asyncio.Event()

    async def wait_async(**kwargs: Any) -> str:
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            binding.report["submissions"][-1].update(
                status="cancelled", error_code="local_cancellation", evidence_completeness="unknown"
            )
            binding.report["status"] = "cancelled"
            raise
        raise AssertionError("OFFLINE evaluator unexpectedly returned")

    binding.evaluator.side_effect = wait_async
    pending = asyncio.create_task(_run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a", "a"]))
    await asyncio.wait_for(entered.wait(), timeout=30)
    pending.cancel("OFFLINE caller cancellation")
    with pytest.raises(asyncio.CancelledError, match="caller cancellation"):
        await pending
    assert binding.v2_events[-2:] == ["cleanup", "factory_exit"]
    assert binding.v2_bridge is not None and binding.v2_bridge.last_result is not None
    result = binding.v2_bridge.last_result
    assert result.report.status.value == "cancelled" and result.score.is_undetermined
    assert result.report.provider_request_count == binding.evaluator.await_count == 1
    assert result.report.report.submissions[-1].remote_disposition.value == "unknown"
    assert result.report.local_cleanup is SubmissionCleanupStatus.COMPLETE
    assert result.native_log is not None


async def test_v2_late_delivery_cancellation_keeps_the_durable_published_grade_async(tmp_path: Path) -> None:
    binding = V2FixtureBinding({"a": 0.25})
    original = SubmissionReportScorerV2.score_async

    async def cancel_after_commit_async(self: SubmissionReportScorerV2, **kwargs: Any) -> Any:
        scores = await original(self, **kwargs)
        assert binding.owned_task is not None
        binding.owned_task.cancel("OFFLINE postcommit delivery cancellation")
        return scores

    with patch.object(SubmissionReportScorerV2, "score_async", new=cancel_after_commit_async):
        pending = asyncio.create_task(_run_v2_async(tmp_path=tmp_path, binding=binding, commands=["a"]))
        with suppress(asyncio.CancelledError):
            await pending
    assert binding.v2_bridge is not None and binding.v2_bridge.last_result is not None
    result = binding.v2_bridge.last_result
    assert result.score.get_value() == 0.25
    assert result.report.status.value == "completed"
    retained = CentralMemory.get_memory_instance().get_scores(score_ids=[str(result.score.id)])
    assert retained[0].get_value() == 0.25
    assert result.report_path.read_text() == result.report.canonical_json()


async def test_v2_countermeasures_reject_unscoped_provider_requests_before_transport_async(tmp_path: Path) -> None:
    from pyrit.executor.benchmark._inspect_v2_support import InspectProviderObserverV2
    from pyrit.models.submission_v2 import SubmissionProvenanceV2

    artifacts = InspectRunArtifactsV2(
        directory=tmp_path / "observer", provenance=SubmissionProvenanceV2(mode="offline", simulated=True)
    )
    observer = InspectProviderObserverV2(artifacts=artifacts, limits=SubmissionLimitsV2())
    with pytest.raises(RuntimeError, match="uncorrelated"):
        await observer.request_async(
            httpx.Request("POST", "https://fixture.invalid/responses", json={"input": "not admitted"})
        )
    assert observer.request_count == 0
    assert not artifacts.directory.exists()


async def test_v2_owned_audit_deadline_prevents_factory_dispatch_and_attempts_cleanup_once_async(
    tmp_path: Path,
) -> None:
    binding = V2FixtureBinding({"a": 0.25})
    assert InspectNativeTaskBridgeV2._AUDIT_SECONDS == 5
    with patch.object(InspectNativeTaskBridgeV2, "_AUDIT_SECONDS", 0.05):
        with pytest.raises(TimeoutError):
            await _run_v2_async(
                tmp_path=tmp_path,
                binding=binding,
                commands=["a"],
                block_audit=True,
                limits=SubmissionLimitsV2(episode_timeout_seconds=30),
            )
    assert binding.v2_events == ["audit", "cleanup"]
    binding.evaluator.assert_not_awaited()
    assert binding.v2_bridge is not None and binding.v2_bridge.last_result is not None
    result = binding.v2_bridge.last_result
    assert result.report.generation_count == result.report.provider_request_count == 0
    assert result.report.environment_audit is None and result.report.local_cleanup is SubmissionCleanupStatus.COMPLETE
    assert result.report.status.value == "incomplete" and result.score.is_undetermined
    assert any("Environment audit: TimeoutError" in entry for entry in result.report.lifecycle_errors)


@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_v2_admitted_request_without_response_makes_aggregate_usage_unknown_async(
    tmp_path: Path, failure: str
) -> None:
    binding = V2FixtureBinding({"a": 0.75, "b": 0.25})
    result, requests, events, _ = await _run_v2_async(
        tmp_path=tmp_path,
        binding=binding,
        commands=["a", "b", "b"],
        transport_failure=failure,
        capture_cancellation=True,
    )
    assert len(requests) == result.report.provider_request_count == 2
    assert result.report.token_usage == ({"input_tokens": 10, "output_tokens": 2}, None)
    assert result.report.total_tokens is None
    assert result.report.report.last_valid_grade == 0.75
    assert result.score.status is ScoreStatus.UNDETERMINED
    binding.evaluator.assert_awaited_once()
    assert events.count("cleanup") == events.count("factory_exit") == 1


@pytest.mark.parametrize("failure", ["write_error", "cancel"])
async def test_v2_interrupted_response_evidence_does_not_publish_a_partial_usage_sum_async(
    tmp_path: Path, failure: str
) -> None:
    from pyrit.executor.benchmark.submission.evidence_v2 import SubmissionEvidenceWriterV2

    original = SubmissionEvidenceWriterV2.append_async

    async def interrupt_async(self: SubmissionEvidenceWriterV2, *, event: str, data: dict[str, Any]) -> None:
        if event == "provider_response" and data.get("data", {}).get("generation") == 2:
            if failure == "cancel":
                raise asyncio.CancelledError("OFFLINE response evidence interrupted")
            raise OSError("OFFLINE response evidence unavailable")
        await original(self, event=event, data=data)

    binding = V2FixtureBinding({"a": 0.75, "b": 0.25})
    with patch.object(SubmissionEvidenceWriterV2, "append_async", new=interrupt_async):
        result, requests, _, _ = await _run_v2_async(
            tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"], capture_cancellation=True
        )
    assert len(requests) == result.report.provider_request_count == 2
    assert result.report.token_usage == ({"input_tokens": 10, "output_tokens": 2}, None)
    assert result.report.total_tokens is None
    assert result.score.is_undetermined and result.report.report.last_valid_grade == 0.75
    binding.evaluator.assert_awaited_once()


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_v2_cancelled_audit_preserves_first_cancel_and_single_cleanup_under_repeated_cancel_async(
    tmp_path: Path, cleanup_fails: bool
) -> None:
    binding = V2FixtureBinding({})
    auditing, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pending = asyncio.create_task(
        _run_v2_async(
            tmp_path=tmp_path,
            binding=binding,
            commands=[None],
            block_audit=True,
            audit_entered=auditing,
            cleanup_entered=cleaning,
            cleanup_release=release,
            cleanup_error=OSError("OFFLINE cleanup failure") if cleanup_fails else None,
        )
    )
    try:
        await asyncio.wait_for(auditing.wait(), timeout=10)
        pending.cancel("OFFLINE original audit cancellation")
        await asyncio.wait_for(cleaning.wait(), timeout=10)
        for _ in range(2):
            pending.cancel("OFFLINE repeated audit cancellation")
            await asyncio.sleep(0)
        assert not pending.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError, match="original audit cancellation"):
        await pending
    assert binding.v2_events == ["audit", "cleanup"]
    assert binding.v2_bridge is not None and binding.v2_bridge.last_result is not None
    result = binding.v2_bridge.last_result
    assert result.report.provider_request_count == result.report.generation_count == 0
    assert result.report.status.value == "cancelled" and result.score.is_undetermined
    assert result.report.local_cleanup is (
        SubmissionCleanupStatus.UNKNOWN if cleanup_fails else SubmissionCleanupStatus.COMPLETE
    )
    binding.evaluator.assert_not_awaited()

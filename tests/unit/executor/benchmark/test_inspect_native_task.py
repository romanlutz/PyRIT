# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import copy
import hashlib
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pyrit.executor.benchmark import InspectNativeTaskBridge, InspectTaskEnvironment
from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark.submission.hooks import SubmissionLimits
from pyrit.memory import CentralMemory
from pyrit.models import ContentEntryScorable, ScoreStatus
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.score import SubmissionReportScorer
from tests.unit.mocks import openai_response_json_dict

pytest.importorskip("inspect_ai")


class OfflineFixtureBinding:
    """Newly authored inert fixture, never a private evaluator implementation."""

    def __init__(self, outcomes: dict[str, float | str]) -> None:
        self.outcomes = outcomes
        self.files = {key: bytearray(f"OFFLINE/SIMULATED inert {key}".encode()) for key in outcomes}
        self.snapshots: list[bytes] = []
        self.cancel_after_return: str | None = None
        self.owned_task: asyncio.Task[Any] | None = None
        self.bridge: InspectNativeTaskBridge | None = None
        self.state: Any = None
        self.evaluator = AsyncMock(side_effect=self._evaluate_async)
        self.report: dict[str, Any] = {
            "contract_version": "strict-submission-v1",
            "mode": "offline",
            "simulated": True,
            "status": "no_submission",
            "last_valid_grade": None,
            "selected_submission_id": None,
            "full_success": False,
            "submissions": [],
            "capabilities": {"remote_query": False, "remote_cancel": False, "idempotency": False},
        }

    def read_report(self) -> dict[str, Any]:
        return copy.deepcopy(self.report)

    async def _evaluate_async(self, *, artifact_ref: str, snapshot: bytes) -> float | str:
        self.files[artifact_ref][:] = b"mutated after immutable acquisition"
        self.snapshots.append(snapshot)
        await asyncio.sleep(0)
        return self.outcomes[artifact_ref]

    async def submit_async(self, artifact_ref: str) -> str:
        from inspect_ai.tool import ToolError

        index = len(self.report["submissions"]) + 1
        record: dict[str, Any] = {
            "submission_id": f"offline-submission-{index}",
            "sequence": index,
            "artifact_sha256": None,
            "artifact_size_bytes": None,
            "dispatch_state": "not_dispatched",
            "status": "rejected",
            "grade": None,
            "behavior_outcome": None,
            "feedback": "OFFLINE missing artifact",
            "error_code": "missing_artifact",
            "raw_evidence": {"label": "OFFLINE/SIMULATED"},
            "evidence_completeness": "complete",
            "acceptance": "not_dispatched",
            "observed_request_id": None,
            "observed_receipt_id": None,
            "remote_disposition": "not_dispatched",
            "cleanup_status": "not_required",
        }
        self.report["submissions"].append(record)
        if self.report["full_success"]:
            record.update(status="guarded", feedback="OFFLINE already complete", error_code="complete_guard")
            raise RuntimeError("OFFLINE full-success dispatch guard")
        if artifact_ref not in self.files:
            raise ToolError(record["feedback"])
        snapshot = bytes(self.files[artifact_ref])
        record.update(artifact_sha256=hashlib.sha256(snapshot).hexdigest(), artifact_size_bytes=len(snapshot))
        if self.outcomes[artifact_ref] == "dependency":
            record.update(status="error", error_code="missing_dependency", feedback="OFFLINE missing evaluator")
            self.report["status"] = "error"
            raise RuntimeError(record["feedback"])
        record.update(dispatch_state="dispatched", acceptance="unknown", remote_disposition="unknown")
        outcome = await self.evaluator(artifact_ref=artifact_ref, snapshot=snapshot)
        if outcome in ("infra", "unknown", "cancelled", "malformed"):
            status = "unknown" if outcome == "unknown" else "cancelled" if outcome == "cancelled" else "error"
            record.update(status=status, error_code=str(outcome), evidence_completeness="unknown")
            self.report["status"] = status
            if outcome == "cancelled":
                raise asyncio.CancelledError("OFFLINE cancelled await; remote disposition unknown")
            raise RuntimeError(f"OFFLINE {outcome}; no retry")
        incomplete = outcome in ("partial_evidence", "unknown_cleanup")
        grade = 0.5 if incomplete else outcome
        record.update(
            dispatch_state="returned",
            status="completed",
            grade=grade,
            behavior_outcome="pass" if grade == 1 else "fail" if grade == 0 else "partial",
            feedback=f"OFFLINE grade {grade}\nexact native feedback",
            error_code=None,
            acceptance="accepted",
            remote_disposition="completed",
            evidence_completeness="partial" if outcome == "partial_evidence" else "complete",
            cleanup_status="unknown" if outcome == "unknown_cleanup" else "complete",
        )
        self.report.update(
            status="incomplete" if incomplete else "completed",
            last_valid_grade=grade,
            selected_submission_id=record["submission_id"],
            full_success=grade == 1,
        )
        if self.cancel_after_return == artifact_ref and self.owned_task is not None:
            asyncio.get_running_loop().call_soon(self.owned_task.cancel, "OFFLINE callback-return boundary")
        return record["feedback"]


async def _run_fixture_async(
    *,
    tmp_path: Path,
    binding: OfflineFixtureBinding,
    commands: list[str | None],
    max_requests: int | None = None,
    native_grade_override: object = None,
    cleanup_error: Exception | None = None,
    cleanup_callback: Callable[[], Awaitable[None]] | None = None,
    capture_callback_cancellation: bool = False,
    task_options: dict[str, Any] | None = None,
    max_tool_calls: int = 8,
    max_response_bytes: int = 1_048_576,
    max_tool_output_bytes: int | None = None,
    include_usage: bool = True,
) -> tuple[Any, list[dict[str, Any]], InspectRunArtifacts, AsyncMock]:
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageSystem
    from inspect_ai.scorer import Score, Target, scorer
    from inspect_ai.solver import Generate, Solver, TaskState, solver
    from inspect_ai.tool import Tool, tool

    binding.owned_task = asyncio.current_task()

    @tool
    def submit_fixture() -> Tool:
        async def submit_async(artifact_ref: str) -> str:
            """Submit inert fixture bytes for an OFFLINE/SIMULATED observation.

            Args:
                artifact_ref: The local inert fixture identity.
            """
            return await binding.submit_async(artifact_ref)

        return submit_async

    @solver
    def native_solver() -> Solver:
        async def solve_async(state: TaskState, generate: Generate) -> TaskState:
            binding.state = state
            state.store.set("fixture_setup", True)
            state.messages.insert(0, ChatMessageSystem(content="OFFLINE/SIMULATED task-owned system"))
            state.tools = [submit_fixture()]
            while not state.completed:
                await generate(state, tool_calls="loop")
            return state

        return solve_async

    grade_invoked = AsyncMock()

    @scorer(metrics=[])
    def latest_native_grade() -> Any:
        async def score_async(state: TaskState, target: Target) -> Score:
            assert state.store.get("fixture_setup")
            await grade_invoked()
            grade = binding.read_report()["last_valid_grade"]
            value = grade if grade is not None else "NO_SUBMISSION"
            return Score(value=native_grade_override if native_grade_override is not None else value)

        return score_async

    requests: list[dict[str, Any]] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        ordinal = len(requests)
        assert ordinal <= len(commands), "An unexpected provider request escaped the strict stop rule."
        command = commands[ordinal - 1]
        response = openai_response_json_dict()
        if include_usage:
            response["usage"] = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        if command is None:
            response["output"][0]["content"][0]["text"] = f"OFFLINE/SIMULATED thought {ordinal}"
        else:
            response["output"] = [
                {
                    "type": "function_call",
                    "call_id": f"offline-call-{ordinal}",
                    "id": f"function-{ordinal}",
                    "name": "submit_fixture",
                    "arguments": json.dumps({"artifact_ref": command}),
                    "status": "completed",
                }
            ]
        return httpx.Response(200, json=response)

    artifacts = InspectRunArtifacts(directory=tmp_path / "run", provenance={"mode": "OFFLINE/SIMULATED"})
    audit, cleanup = AsyncMock(), AsyncMock(side_effect=cleanup_callback or cleanup_error)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:

        def factory(schemas: list[dict[str, Any]]) -> OpenAIResponseTarget:
            return OpenAIResponseTarget(
                endpoint="https://fixture.invalid/v1",
                api_key="offline",
                model_name="offline-fixture",
                auto_execute_tools=False,
                extra_body_parameters={"tools": schemas, "parallel_tool_calls": False, "store": False},
                httpx_client_kwargs={"http_client": client, "max_retries": 0},
            )

        bridge = InspectNativeTaskBridge(
            target_factory=factory,
            model_name="offline-fixture",
            read_report=binding.read_report,
            artifacts=artifacts,
            environment=InspectTaskEnvironment(audit_async=audit, cleanup_async=cleanup),
            limits=SubmissionLimits(
                max_requests=max_requests or len(commands),
                max_tool_calls=max_tool_calls,
                max_response_bytes=max_response_bytes,
            ),
            max_tool_output_bytes=max_tool_output_bytes,
        )
        binding.bridge = bridge
        with (
            patch.dict("os.environ", {"RETRY_MAX_NUM_ATTEMPTS": "1"}),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=AssertionError("No workload")),
            patch("socket.socket.connect", side_effect=AssertionError("No socket")),
        ):
            try:
                result = await bridge.execute_async(
                    task=Task(
                        dataset=[Sample(id="offline-case", input="OFFLINE/SIMULATED task-owned input")],
                        solver=native_solver(),
                        scorer=latest_native_grade(),
                        **(task_options or {}),
                    ),
                    sample_id="offline-case",
                    native_scorer="latest_native_grade",
                )
            except asyncio.CancelledError:
                if not capture_callback_cancellation:
                    raise
                assert bridge.last_result is not None
                result = bridge.last_result
    audit.assert_awaited_once()
    cleanup.assert_awaited_once()
    return result, requests, artifacts, grade_invoked


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("grades", [[0.75, 0.25], [0.75, 0.0], [0.25, 0.75]])
async def test_latest_numeric_native_grade_at_healthy_budget_is_content_anchored_async(
    tmp_path: Path, grades: list[float]
) -> None:
    binding = OfflineFixtureBinding({"a": grades[0], "b": grades[1]})
    result, requests, artifacts, native_score = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "b"]
    )
    assert result.score.get_value() == grades[-1]
    assert result.report.report.selected_submission_id == "offline-submission-2"
    assert artifacts.manifest["termination_reason"] == "budget"
    assert len(requests) == binding.evaluator.await_count == 2
    assert requests[1]["input"][-1]["output"] == f"OFFLINE grade {grades[0]}\nexact native feedback"
    native_score.assert_awaited_once()
    assert result.report.final_text is None
    assert result.score.message_piece_id is None
    stored = CentralMemory.get_memory_instance().get_scores(score_ids=[str(result.score.id)])[0]
    assert isinstance(stored.scorable, ContentEntryScorable)
    replay = await SubmissionReportScorer(report_sha256=result.report.sha256()).score_async(scorable=stored.scorable)
    assert replay[0].get_value() == grades[-1]
    assert binding.evaluator.await_count == 2
    assert result.report_path.read_text() == result.report.canonical_json()
    assert result.report.termination_reason.value == "budget"
    assert result.report.generation_count == 2
    for snapshot, record in zip(binding.snapshots, result.report.report.submissions, strict=True):
        assert hashlib.sha256(snapshot).hexdigest() == record.artifact_sha256
        assert b"mutated" not in snapshot
    pieces = CentralMemory.get_memory_instance().get_message_pieces(conversation_id=result.report.conversation_id)
    assert pieces[-1].role == "tool"
    assert len([piece for piece in pieces if piece.original_value_data_type == "function_call_output"]) == 2


@pytest.mark.usefixtures("patch_central_database")
async def test_no_tool_continuation_then_full_success_stops_without_fake_final_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"success": 1.0})
    result, requests, artifacts, _ = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=[None, "success", "success"]
    )
    assert result.score.get_value() == 1.0
    assert len(requests) == 2 and binding.evaluator.await_count == 1
    assert artifacts.manifest["termination_reason"] == "full_success"
    assert requests[1]["input"] == [
        *requests[0]["input"],
        {"role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE/SIMULATED thought 1"}]},
    ]
    assert result.report.final_text is None
    assert result.report.calls[0].feedback == "OFFLINE grade 1.0\nexact native feedback"
    assert result.report.termination_reason.value == "full_success"


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("commands", [[None, None, None], ["missing", "missing"]])
async def test_no_submission_never_fabricates_behavior_grade_async(tmp_path: Path, commands: list[str | None]) -> None:
    binding = OfflineFixtureBinding({})
    result, requests, artifacts, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=commands)
    assert result.report.status.value == "no_submission"
    assert result.score.status is ScoreStatus.UNDETERMINED and result.score.score_value is None
    assert artifacts.manifest["termination_reason"] == "budget"
    assert len(requests) == len(commands)
    binding.evaluator.assert_not_awaited()
    if commands[0] == "missing":
        assert requests[1]["input"][-1]["output"] == "Error: OFFLINE missing artifact"


@pytest.mark.usefixtures("patch_central_database")
async def test_later_missing_artifact_preserves_prior_numeric_grade_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75})
    result, _, _, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "missing"])
    assert result.score.get_value() == 0.75
    assert binding.evaluator.await_count == 1
    assert result.report.calls[-1].feedback == "Error: OFFLINE missing artifact"
    assert len(result.report.report.submissions) == 2


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "failure", ["dependency", "infra", "unknown", "malformed", "partial_evidence", "unknown_cleanup", "cancelled"]
)
async def test_terminal_or_incomplete_submission_never_retries_or_publishes_prior_grade_async(
    tmp_path: Path, failure: str
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "failure": failure})
    result, requests, artifacts, native_score = await _run_fixture_async(
        tmp_path=tmp_path,
        binding=binding,
        commands=["a", "failure", "failure"],
        capture_callback_cancellation=failure == "cancelled",
    )
    assert len(requests) == 2
    assert binding.evaluator.await_count == (1 if failure == "dependency" else 2)
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert result.report.status.value in ("error", "unknown", "incomplete", "cancelled")
    assert result.report.report.last_valid_grade == (
        0.5 if failure in ("partial_evidence", "unknown_cleanup") else 0.75
    )
    native_score.assert_not_awaited()
    assert result.report.report.capabilities.remote_cancel is False
    if failure in ("partial_evidence", "unknown_cleanup"):
        pieces = CentralMemory.get_memory_instance().get_message_pieces(conversation_id=result.report.conversation_id)
        assert pieces[-1].role == "tool"
        assert result.report.calls[-1].feedback == "OFFLINE grade 0.5\nexact native feedback"
    if failure in ("unknown", "cancelled"):
        assert result.report.report.submissions[-1].remote_disposition.value == "unknown"
    if failure == "cancelled":
        assert result.report.calls[-1].feedback is None
        diagnostic = artifacts.manifest["tool_calls"][-1]["terminal_diagnostic"]
        assert diagnostic["source"] == "inspect.execute_tools"
        assert diagnostic["forwarded_to_provider"] is False
        assert diagnostic["message"]


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -0.1, 1.1])
async def test_malformed_binding_numeric_grade_is_not_coerced_or_scored_async(tmp_path: Path, invalid: float) -> None:
    binding = OfflineFixtureBinding({"bad": invalid})
    result, requests, _, native_score = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["bad", "bad"]
    )
    assert len(requests) == binding.evaluator.await_count == 1
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert result.report.status.value == "error"
    native_score.assert_not_awaited()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("invalid", [True, "1.0", 0.25, float("nan"), float("inf"), -0.1, 1.1])
async def test_native_grade_must_match_typed_latest_grade_not_final_text_async(tmp_path: Path, invalid: object) -> None:
    binding = OfflineFixtureBinding({"success": 1.0})
    with pytest.raises(ValueError, match="Native.*grade"):
        await _run_fixture_async(
            tmp_path=tmp_path, binding=binding, commands=["success"], native_grade_override=invalid
        )
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["harness_status"] == "error"
    assert manifest["grade_status"] == "undetermined"
    assert "score" not in manifest
    assert Path(manifest["native_log"]).exists()


@pytest.mark.usefixtures("patch_central_database")
async def test_report_replay_rejects_content_or_digest_tampering_async(tmp_path: Path) -> None:
    from pyrit.models import ContentScorable

    binding = OfflineFixtureBinding({"success": 1.0})
    result, _, _, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["success"])
    modified = result.report.model_dump(mode="json")
    modified["report"]["selected_submission_id"] = "forged"
    with pytest.raises(RuntimeError, match="modified|identity"):
        await SubmissionReportScorer(report_sha256=result.report.sha256()).score_async(
            scorable=ContentScorable(value=json.dumps(modified), data_type="text")
        )
    binding.evaluator.assert_awaited_once()


@pytest.mark.usefixtures("patch_central_database")
async def test_external_cancellation_retains_partial_report_and_never_claims_remote_stop_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"waiting": "unknown"})
    dispatched = asyncio.Event()

    async def wait_async(**kwargs: Any) -> float:
        dispatched.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            record = binding.report["submissions"][-1]
            record.update(status="cancelled", error_code="local_cancel", evidence_completeness="unknown")
            binding.report["status"] = "cancelled"
            raise
        raise AssertionError("OFFLINE blocked evaluator unexpectedly completed")

    binding.evaluator.side_effect = wait_async
    pending = asyncio.create_task(
        _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["waiting", "waiting"])
    )
    await asyncio.wait_for(dispatched.wait(), timeout=10)
    pending.cancel("OFFLINE external cancellation")
    try:
        result, requests, _, _ = await pending
    except asyncio.CancelledError:
        pass
    else:
        assert result.score.status is ScoreStatus.UNDETERMINED
        assert len(requests) == 1
    binding.evaluator.assert_awaited_once()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["binding_report"]["status"] == "cancelled"
    assert manifest["binding_report"]["submissions"][-1]["remote_disposition"] == "unknown"
    assert manifest["remote_stop_claim"] is False
    assert manifest["local_environment_cleanup"] == "verified_by_caller"


@pytest.mark.usefixtures("patch_central_database")
async def test_cleanup_failure_prevents_clean_grade_and_retains_completed_observation_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"success": 1.0})
    with pytest.raises(RuntimeError, match="OFFLINE cleanup failed"):
        await _run_fixture_async(
            tmp_path=tmp_path,
            binding=binding,
            commands=["success"],
            cleanup_error=RuntimeError("OFFLINE cleanup failed"),
        )
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["binding_report"]["last_valid_grade"] == 1.0
    assert manifest["local_environment_cleanup"] == "unknown"
    assert binding.bridge is not None and binding.bridge.last_result is not None
    assert binding.bridge.last_result.score.status is ScoreStatus.UNDETERMINED
    binding.evaluator.assert_awaited_once()


@pytest.mark.usefixtures("patch_central_database")
async def test_cancel_after_return_preserves_latest_report_feedback_and_submission_correlation_async(
    tmp_path: Path,
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    returned = asyncio.Event()
    original_append = InspectRunArtifacts.append_async

    async def block_return_async(self: InspectRunArtifacts, *, event: str, data: dict[str, Any]) -> None:
        if event == "native_tool_return" and data["provider_call_id"] == "offline-call-2":
            returned.set()
            await asyncio.Future()
        await original_append(self, event=event, data=data)

    with patch.object(InspectRunArtifacts, "append_async", new=block_return_async):
        pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"]))
        await asyncio.wait_for(returned.wait(), timeout=10)
        pending.cancel("OFFLINE cancellation after acquired .25")
        with pytest.raises(asyncio.CancelledError, match="after acquired"):
            await pending
    assert binding.evaluator.await_count == 2
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["binding_report"]["last_valid_grade"] == 0.25
    assert manifest["binding_report"]["selected_submission_id"] == "offline-submission-2"
    assert manifest["provider_requests"] == 2
    call = manifest["tool_calls"][-1]
    assert call["submission_ids"] == ["offline-submission-2"]
    assert call["result"]["content"] == "OFFLINE grade 0.25\nexact native feedback"
    pieces = CentralMemory.get_memory_instance().get_message_pieces(conversation_id=manifest["conversation_id"])
    results = [
        json.loads(piece.original_value) for piece in pieces if piece.original_value_data_type == "function_call_output"
    ]
    assert results[-1]["call_id"] == "offline-call-2"
    assert results[-1]["output"] == "OFFLINE grade 0.25\nexact native feedback"
    assert manifest["remote_stop_claim"] is False


@pytest.mark.usefixtures("patch_central_database")
async def test_callback_scheduled_cancellation_preserves_actual_return_not_canned_diagnostic_async(
    tmp_path: Path,
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"
    pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"]))
    with pytest.raises(asyncio.CancelledError, match="callback-return boundary"):
        await pending
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["binding_report"]["last_valid_grade"] == 0.25
    assert manifest["binding_report"]["selected_submission_id"] == "offline-submission-2"
    assert manifest["tool_calls"][-1]["submission_ids"] == ["offline-submission-2"]
    assert manifest["tool_calls"][-1]["result"]["content"] == "OFFLINE grade 0.25\nexact native feedback"
    assert manifest["provider_requests"] == binding.evaluator.await_count == 2
    pieces = CentralMemory.get_memory_instance().get_message_pieces(conversation_id=manifest["conversation_id"])
    outputs = [
        json.loads(piece.original_value) for piece in pieces if piece.original_value_data_type == "function_call_output"
    ]
    assert [item["call_id"] for item in outputs] == ["offline-call-1", "offline-call-2"]
    assert outputs[-1]["output"] == "OFFLINE grade 0.25\nexact native feedback"
    assert manifest["remote_stop_claim"] is False
    assert binding.bridge is not None and binding.bridge.last_result is not None
    acquired = binding.bridge.last_result
    assert acquired.report.status.value == "cancelled"
    assert acquired.report.report.last_valid_grade == 0.25
    assert acquired.score.status is ScoreStatus.UNDETERMINED
    assert acquired.report.calls[-1].feedback == "OFFLINE grade 0.25\nexact native feedback"
    assert acquired.native_log is not None and acquired.native_log.exists()
    assert acquired.native_eval_id


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("phase", ["score", "manifest"])
async def test_first_cancel_during_final_publication_never_exposes_clean_last_result_async(
    tmp_path: Path, phase: str
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    publishing = asyncio.Event()
    original_save = InspectRunArtifacts.save_async
    original_score = SubmissionReportScorer.score_async

    async def block_save_async(self: InspectRunArtifacts) -> None:
        if (
            phase == "manifest"
            and self.manifest.get("publication_state") == "unscored_candidate"
            and self.manifest.get("evidence_status") == "completed"
            and not publishing.is_set()
        ):
            publishing.set()
            await asyncio.Future()
        await original_save(self)

    async def block_score_async(self: SubmissionReportScorer, **kwargs: Any) -> Any:
        if (
            phase == "score"
            and json.loads(kwargs["scorable"].value)["status"] == "completed"
            and not publishing.is_set()
        ):
            publishing.set()
            await asyncio.Future()
        return await original_score(self, **kwargs)

    with (
        patch.object(InspectRunArtifacts, "save_async", new=block_save_async),
        patch.object(SubmissionReportScorer, "score_async", new=block_score_async),
    ):
        pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b"]))
        await asyncio.wait_for(publishing.wait(), timeout=30)
        pending.cancel("OFFLINE first cancellation during publication")
        with pytest.raises(asyncio.CancelledError, match="first cancellation during publication"):
            await pending
    assert binding.evaluator.await_count == 2
    assert binding.bridge is not None and binding.bridge.last_result is not None
    retained = binding.bridge.last_result
    assert retained.report.status.value == "cancelled"
    assert retained.report.report.last_valid_grade == 0.25
    assert retained.score.status is ScoreStatus.UNDETERMINED
    assert retained.report_path.read_text() == retained.report.canonical_json()
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["report_sha256"] == retained.report.sha256()
    assert manifest["publication_boundary"] == "pyrit_score_commit"
    published = [
        score
        for score in CentralMemory.get_memory_instance().get_scores(score_type="float_scale")
        if score.score_metadata.get("run_id") == retained.report.run_id
    ]
    assert len(published) == 1 and published[0].status is ScoreStatus.UNDETERMINED


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("failed_retention", ["journal", "memory", "manifest", "score"])
async def test_callback_cancellation_survives_secondary_retention_oserror_async(
    tmp_path: Path, failed_retention: str
) -> None:
    binding = OfflineFixtureBinding({"cancel": "cancelled"})
    original_cancel = asyncio.CancelledError("OFFLINE original callback cancellation")
    original_append = InspectRunArtifacts.append_async
    original_save = InspectRunArtifacts.save_async
    original_score = SubmissionReportScorer.score_async
    memory = CentralMemory.get_memory_instance()
    original_add = memory.add_message_to_memory

    async def cancel_async(**kwargs: Any) -> float:
        binding.report["submissions"][-1].update(
            status="cancelled", evidence_completeness="unknown", error_code="local_cancel"
        )
        binding.report["status"] = "cancelled"
        raise original_cancel

    async def fail_append_async(self: InspectRunArtifacts, *, event: str, data: dict[str, Any]) -> None:
        if failed_retention == "journal" and event == "native_tool_error":
            raise OSError("OFFLINE journal retention failed")
        await original_append(self, event=event, data=data)

    async def fail_save_async(self: InspectRunArtifacts) -> None:
        if failed_retention == "manifest" and binding.report["status"] == "cancelled":
            raise OSError("OFFLINE manifest retention failed")
        await original_save(self)

    async def fail_score_async(self: SubmissionReportScorer, **kwargs: Any) -> Any:
        if failed_retention == "score" and binding.report["status"] == "cancelled":
            raise OSError("OFFLINE score retention failed")
        return await original_score(self, **kwargs)

    def fail_add(*, request: Any) -> None:
        if failed_retention == "memory" and request.api_role == "tool":
            raise OSError("OFFLINE memory retention failed")
        original_add(request=request)

    binding.evaluator.side_effect = cancel_async
    with (
        patch.object(InspectRunArtifacts, "append_async", new=fail_append_async),
        patch.object(InspectRunArtifacts, "save_async", new=fail_save_async),
        patch.object(SubmissionReportScorer, "score_async", new=fail_score_async),
        patch.object(memory, "add_message_to_memory", side_effect=fail_add),
    ):
        with pytest.raises(asyncio.CancelledError) as caught:
            await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["cancel", "cancel"])
    assert caught.value is original_cancel
    assert binding.evaluator.await_count == 1
    assert binding.report["status"] == "cancelled"
    assert binding.report["submissions"][-1]["remote_disposition"] == "unknown"
    assert any("retention" in note for note in caught.value.__notes__)
    if binding.bridge is not None and binding.bridge.last_result is not None:
        assert binding.bridge.last_result.score.status is ScoreStatus.UNDETERMINED


@pytest.mark.usefixtures("patch_central_database")
async def test_text_report_score_publication_does_not_yield_async(tmp_path: Path) -> None:
    from pyrit.models import ContentScorable

    binding = OfflineFixtureBinding({"a": 0.25})
    result, _, _, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a"])
    yielded: list[bool] = []
    asyncio.get_running_loop().call_soon(yielded.append, True)
    scores = await SubmissionReportScorer(report_sha256=result.report.sha256()).score_async(
        scorable=ContentScorable(value=result.report.canonical_json(), data_type="text")
    )
    assert yielded == []
    assert scores[0].get_value() == 0.25
    await asyncio.sleep(0)
    assert yielded == [True]


@pytest.mark.usefixtures("patch_central_database")
async def test_cancellation_after_score_commit_does_not_rollback_published_observation_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.25})
    original_score = SubmissionReportScorer.score_async
    commit_seen = False

    async def cancel_after_commit_async(self: SubmissionReportScorer, **kwargs: Any) -> Any:
        nonlocal commit_seen
        scores = await original_score(self, **kwargs)
        commit_seen = True
        assert binding.owned_task is not None
        binding.owned_task.cancel("OFFLINE delivery cancellation after durable score commit")
        return scores

    with patch.object(SubmissionReportScorer, "score_async", new=cancel_after_commit_async):
        pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a"]))
        with suppress(asyncio.CancelledError):
            await pending
    assert commit_seen
    assert binding.bridge is not None and binding.bridge.last_result is not None
    published = binding.bridge.last_result
    assert published.report.status.value == "completed"
    assert published.score.get_value() == 0.25
    stored = CentralMemory.get_memory_instance().get_scores(score_ids=[str(published.score.id)])
    assert stored[0].get_value() == 0.25
    assert published.report_path.read_text() == published.report.canonical_json()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("cancel_before_commit", [False, True])
async def test_fresh_sqlite_recovers_authoritative_report_without_last_result_async(
    tmp_path: Path, cancel_before_commit: bool
) -> None:
    from pyrit.memory import SQLiteMemory
    from pyrit.models import IdentifierFilter, IdentifierType
    from pyrit.models.submission import RetainedSubmissionReport

    path = tmp_path / "recovery.sqlite"

    def open_memory() -> SQLiteMemory:
        memory = SQLiteMemory.__new__(SQLiteMemory)
        with patch.object(memory, "cleanup"):
            memory.__init__(db_path=path, silent=True)
        memory.disable_embedding()
        return memory

    original = open_memory()
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    original_save = InspectRunArtifacts.save_async
    cancel_issued = False

    async def cancel_before_commit_async(self: InspectRunArtifacts) -> None:
        nonlocal cancel_issued
        await original_save(self)
        if (
            cancel_before_commit
            and not cancel_issued
            and self.manifest.get("publication_state") == "unscored_candidate"
            and self.manifest.get("evidence_status") == "completed"
        ):
            cancel_issued = True
            assert binding.owned_task is not None
            binding.owned_task.cancel("OFFLINE precommit recovery fixture")
            await asyncio.sleep(0)

    try:
        with (
            patch.object(CentralMemory, "get_memory_instance", return_value=original),
            patch.object(InspectRunArtifacts, "save_async", new=cancel_before_commit_async),
        ):
            pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b"]))
            if cancel_before_commit:
                with pytest.raises(asyncio.CancelledError, match="precommit recovery"):
                    await pending
            else:
                await pending
    finally:
        original.dispose_engine()
    binding.bridge = None
    del original
    prepared = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert prepared["publication_state"] == "unscored_candidate"
    reopened = open_memory()
    try:
        matches = [
            score
            for score in reopened.get_scores(
                identifier_filters=[
                    IdentifierFilter(
                        identifier_type=IdentifierType.SCORER,
                        property_path="$.class_name",
                        value="SubmissionReportScorer",
                    )
                ]
            )
            if score.score_metadata["run_id"] == prepared["run_id"]
            and score.score_metadata["report_sha256"] == prepared["report_sha256"]
        ]
        assert len(matches) == 1
        published = matches[0]
        assert isinstance(published.scorable, ContentEntryScorable)
        assert published.score_metadata["publication_role"] == "final_run_result"
        assert published.score_metadata["publication_boundary"] == "pyrit_score_commit"
        content = reopened.get_scorable_content(content_ids=[published.scorable.content_id])[
            published.scorable.content_id
        ]
        recovered = RetainedSubmissionReport.model_validate_json(content.value)
        assert recovered.sha256() == prepared["report_sha256"]
        assert recovered.report.last_valid_grade == 0.25
        assert Path(prepared["report_path"]).read_text() == recovered.canonical_json()
        assert recovered.status.value == ("cancelled" if cancel_before_commit else "completed")
        assert published.status is (ScoreStatus.UNDETERMINED if cancel_before_commit else ScoreStatus.COMPLETE)
        assert binding.evaluator.await_count == 2
    finally:
        reopened.dispose_engine()


@pytest.mark.usefixtures("patch_central_database")
def test_unsupported_publication_backend_rejected_before_callbacks(tmp_path: Path) -> None:
    factory = MagicMock()
    binding = OfflineFixtureBinding({})
    artifacts = InspectRunArtifacts(directory=tmp_path / "unsupported", provenance={"mode": "OFFLINE/SIMULATED"})
    with patch.object(CentralMemory, "get_memory_instance", return_value=object()):
        with pytest.raises(TypeError, match="SQLiteMemory"):
            InspectNativeTaskBridge(
                target_factory=factory,
                model_name="offline-fixture",
                read_report=binding.read_report,
                artifacts=artifacts,
                environment=InspectTaskEnvironment(audit_async=AsyncMock(), cleanup_async=AsyncMock()),
                limits=SubmissionLimits(),
            )
    factory.assert_not_called()
    binding.evaluator.assert_not_awaited()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import aiofiles
import httpx
import pytest

from pyrit.executor.benchmark.submission.hooks import SubmissionLimits
from pyrit.executor.benchmark.submission.runner import NativeSubmissionRunner, SubmissionRunError
from pyrit.models import ContentEntryScorable, ContentScorable, ScoreStatus, SeedPrompt
from pyrit.models.submission import RetainedSubmissionReport, SubmissionReportStatus
from pyrit.score import SubmissionReportScorer
from tests.unit.executor.benchmark.submission.mocks import (
    FixtureFailureError,
    InertSubmissionBinding,
    OfflineProvider,
    completed_observation,
    text_turn,
    tool_turn,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _seed() -> SeedPrompt:
    return SeedPrompt(value="OFFLINE/SIMULATED inert submission exercise.", role="user", data_type="text")


async def test_interactive_latest_lower_grade_and_replay_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    evaluator = AsyncMock(
        side_effect=[
            completed_observation(0.75, feedback="First feedback\nkeep exact."),
            completed_observation(0.25, feedback="Second feedback: lower, not best-of."),
        ]
    )
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="call-1"), tool_turn(call_id="call-2"), text_turn("Finished.")])
    runner = NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=3)
    )
    with (
        patch("socket.create_connection", side_effect=AssertionError("No network permitted")),
        patch("pyrit.auth.get_azure_openai_auth", side_effect=AssertionError("No authentication permitted")),
    ):
        result = await runner.run_offline_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
        )
    assert result.score.get_value() == 0.25
    assert result.report.termination_reason.value == "budget"
    report = result.report.report
    assert report is not None and report.last_valid_grade == 0.25
    assert report.selected_submission_id == binding.dispatches[1].submission_id
    assert binding.dispatches[0].submission_id != binding.dispatches[1].submission_id
    assert binding.dispatches[0].sha256 == binding.dispatches[1].sha256
    assert evaluator.await_count == 2 and len(provider.requests) == 3
    assert provider.requests[1]["input"][-1]["output"] == "First feedback\nkeep exact."
    assert provider.requests[2]["input"][-1]["output"] == "Second feedback: lower, not best-of."
    assert [item.get("role") for item in provider.requests[2]["input"] if "role" in item] == ["developer", "user"]
    assert isinstance(result.score.scorable, ContentEntryScorable)
    assert result.score.message_piece_id is None and result.score.observation_ids == []
    assert result.report_path.read_text(encoding="utf-8") == result.report.canonical_json()
    before = evaluator.await_count
    replay = await SubmissionReportScorer(report_sha256=result.report.sha256()).score_async(
        scorable=result.score.scorable
    )
    assert replay[0].get_value() == 0.25 and evaluator.await_count == before
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert [message.api_role for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    events = [json.loads(line) for line in (result.report_path.parent / "events.jsonl").read_text().splitlines()]
    assert all(event["evidence_label"] == "OFFLINE/SIMULATED" for event in events)


async def test_full_success_stops_without_final_text_or_extra_request_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    evaluator = AsyncMock(return_value=completed_observation(1, feedback="Exact successful feedback."))
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="call-success")])
    runner = NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs")
    result = await runner.run_offline_async(
        seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
    )
    assert result.report.final_text is None and result.score.get_value() == 1
    assert len(provider.requests) == 1 and evaluator.await_count == 1
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert [message.api_role for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2].get_piece().original_value_data_type == "function_call"
    assert json.loads(messages[-1].get_value())["output"] == "Exact successful feedback."
    assert not any(
        piece.role == "assistant" and piece.converted_value_data_type == "text"
        for message in messages
        for piece in message.message_pieces
    )
    await binding.submit_async(artifact_ref="fixture")
    assert evaluator.await_count == 1 and binding.report["submissions"][-1]["status"] == "guarded"


async def test_recoverable_missing_artifact_keeps_prior_selection_and_exact_feedback_async(tmp_path: Path) -> None:
    evaluator = AsyncMock(side_effect=[completed_observation(0.75), completed_observation(0.25)])
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider(
        [
            tool_turn(call_id="first"),
            tool_turn(call_id="missing", artifact_ref="absent"),
            tool_turn(call_id="corrected"),
            text_turn("No grade in this sentence."),
        ]
    )
    result = await NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=4)
    ).run_offline_async(seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider))
    assert evaluator.await_count == 2
    assert provider.requests[2]["input"][-1]["output"] == "Error: Fixture not found.\nChoose an existing fixture."
    assert result.report.calls[1].feedback_kind.value == "recoverable_error"
    assert result.report.report.submissions[1].grade is None
    assert result.score.get_value() == 0.25
    snapshots = [
        event["report"]
        for event in map(json.loads, (result.report_path.parent / "events.jsonl").read_text().splitlines())
        if event["event"] == "binding_report"
    ]
    rejected = next(snapshot for snapshot in snapshots if len(snapshot["submissions"]) == 2)
    assert rejected["last_valid_grade"] == 0.75


@pytest.mark.parametrize("no_artifact", [False, True])
async def test_no_submission_is_not_a_fabricated_failure_async(*, tmp_path: Path, no_artifact: bool) -> None:
    evaluator = AsyncMock()
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    turns = [text_turn("OFFLINE/SIMULATED no artifact submitted.")]
    if no_artifact:
        turns.insert(0, tool_turn(call_id="missing", artifact_ref="absent"))
    result = await NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=len(turns))
    ).run_offline_async(
        seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(OfflineProvider(turns))
    )
    evaluator.assert_not_awaited()
    assert result.report.status is SubmissionReportStatus.NO_SUBMISSION
    assert result.score.status is ScoreStatus.UNDETERMINED and result.score.score_value is None


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("missing fixture dependency"),
        RuntimeError("fixture infrastructure"),
        ConnectionError("accepted outcome unknown"),
    ],
)
async def test_terminal_failure_retains_observation_without_clean_grade_async(
    *, tmp_path: Path, failure: Exception
) -> None:
    evaluator = AsyncMock(side_effect=[completed_observation(0.75), failure])
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="valid"), tool_turn(call_id="uncertain")])
    runner = NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs")
    with pytest.raises(SubmissionRunError) as caught:
        await runner.run_offline_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
        )
    result = caught.value.result
    assert len(provider.requests) == 2 and evaluator.await_count == 2
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert result.report.report.last_valid_grade == 0.75
    assert result.report.report.selected_submission_id == binding.dispatches[0].submission_id
    assert result.report.status is not SubmissionReportStatus.COMPLETED
    assert result.report.calls[-1].feedback is None
    assert result.report.report.submissions[-1].observed_receipt_id is None
    assert result.report.report.submissions[-1].remote_disposition.value == "unknown"


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -0.1, 1.1, "0.75"])
async def test_invalid_grade_is_terminal_not_zero_async(*, tmp_path: Path, value: object) -> None:
    evaluator = AsyncMock(return_value=completed_observation(value))
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="invalid")])
    with pytest.raises(SubmissionRunError) as caught:
        await NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs").run_offline_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
        )
    assert evaluator.await_count == len(provider.requests) == 1
    assert caught.value.result.score.score_value is None
    assert caught.value.result.report.report.last_valid_grade is None


async def test_real_negative_is_completed_zero_async(tmp_path: Path) -> None:
    binding = InertSubmissionBinding(
        directory=tmp_path / "snapshots", evaluator=AsyncMock(return_value=completed_observation(0))
    )
    provider = OfflineProvider([tool_turn(call_id="negative"), text_turn("No more fixture changes.")])
    result = await NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=2)
    ).run_offline_async(seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider))
    assert result.score.status is ScoreStatus.COMPLETE and result.score.get_value() == 0
    assert result.report.report.submissions[0].behavior_outcome.value == "fail"


@pytest.mark.parametrize(("completeness", "cleanup"), [("partial", "complete"), ("complete", "unknown")])
async def test_partial_evidence_is_not_clean_grade_async(*, tmp_path: Path, completeness: str, cleanup: str) -> None:
    evaluator = AsyncMock(return_value=completed_observation(0.75, completeness=completeness, cleanup=cleanup))
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="partial")])
    result = await NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs").run_offline_async(
        seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
    )
    assert result.report.status is SubmissionReportStatus.INCOMPLETE
    assert result.report.report.last_valid_grade == 0.75
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert len(provider.requests) == 1


async def test_cancelled_await_does_not_claim_remote_stop_async(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def evaluator_async(request: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("Unreachable")

    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator_async)
    provider = OfflineProvider([tool_turn(call_id="cancelled")])
    runner = NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs")
    task = asyncio.create_task(
        runner.run_offline_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = runner.last_result
    assert result is not None and result.report.status is SubmissionReportStatus.CANCELLED
    record = result.report.report.submissions[0]
    assert record.remote_disposition.value == "unknown" and record.observed_receipt_id is None
    assert result.score.score_value is None and len(provider.requests) == 1
    assert record.grade is None and result.report.calls[0].feedback_kind.value == "cancelled"


async def test_frozen_bytes_survive_mutating_source_and_report_tampering_is_rejected_async(tmp_path: Path) -> None:
    async def evaluate_async(request: Any) -> dict[str, Any]:
        binding.artifacts["fixture"] = b"changed after dispatch"
        assert request.content == b"OFFLINE/SIMULATED inert bytes\n"
        assert request.sha256 == hashlib.sha256(request.content).hexdigest()
        durable = json.loads((binding.directory / "dispatches.jsonl").read_text().splitlines()[-1])
        assert durable["submission_id"] == request.submission_id
        return completed_observation(1)

    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluate_async)
    result = await NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs").run_offline_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        transport=httpx.MockTransport(OfflineProvider([tool_turn(call_id="frozen")])),
    )
    assert binding.frozen_paths[0].read_bytes() == b"OFFLINE/SIMULATED inert bytes\n"
    modified = json.loads(result.report.canonical_json())
    modified["report"]["selected_submission_id"] = "different"
    with pytest.raises(RuntimeError, match="modified"):
        await SubmissionReportScorer(report_sha256=result.report.sha256()).score_async(
            scorable=ContentScorable(value=json.dumps(modified), data_type="text")
        )
    assert (
        RetainedSubmissionReport.model_validate_json(result.report_path.read_text()).sha256() == result.report.sha256()
    )


async def test_provider_failure_after_valid_submission_is_incomplete_async(tmp_path: Path) -> None:
    binding = InertSubmissionBinding(
        directory=tmp_path / "snapshots", evaluator=AsyncMock(return_value=completed_observation(0.75))
    )
    provider = OfflineProvider([tool_turn(call_id="partial-run")])
    runner = NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs")
    with pytest.raises(SubmissionRunError) as caught:
        await runner.run_offline_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
        )
    assert len(provider.requests) == 2
    assert caught.value.result.report.report.last_valid_grade == 0.75
    assert caught.value.result.score.status is ScoreStatus.UNDETERMINED


async def test_no_tool_continuation_has_no_duplicate_or_synthetic_input_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = InertSubmissionBinding(
        directory=tmp_path / "snapshots", evaluator=AsyncMock(return_value=completed_observation(1))
    )
    provider = OfflineProvider([text_turn("I will inspect the inert fixture."), tool_turn(call_id="after-no-tool")])
    result = await NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs").run_offline_async(
        seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
    )
    assert len(provider.requests) == 2 and result.score.get_value() == 1
    initial, continuation = [request["input"] for request in provider.requests]
    assert continuation[:-1] == initial
    assert continuation[-1] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I will inspect the inert fixture."}],
    }
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert [message.api_role for message in messages] == ["system", "user", "assistant", "assistant", "tool"]


async def test_bounded_no_progress_preserves_no_submission_async(tmp_path: Path) -> None:
    evaluator = AsyncMock()
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([text_turn("First authentic response."), text_turn("Second authentic response.")])
    result = await NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=2)
    ).run_offline_async(seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider))
    evaluator.assert_not_awaited()
    assert result.report.status is SubmissionReportStatus.NO_SUBMISSION
    assert result.report.termination_reason.value == "budget"
    assert result.report.generation_count == len(provider.requests) == 2
    assert result.score.status is ScoreStatus.UNDETERMINED


async def test_transport_guard_rejects_before_artifact_or_tool_side_effect_async(tmp_path: Path) -> None:
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=AsyncMock())
    runner = NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs")
    with pytest.raises(ValueError, match="MockTransport"):
        await runner.run_offline_async(seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=None)
    assert not (tmp_path / "runs").exists()


async def test_last_allowed_generation_still_executes_its_tool_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    evaluator = AsyncMock(return_value=completed_observation(0.25, feedback="Last allowed generation feedback."))
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="last-allowed")])
    result = await NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=1)
    ).run_offline_async(seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider))
    evaluator.assert_awaited_once()
    assert len(provider.requests) == 1 and result.score.get_value() == 0.25
    assert result.report.termination_reason.value == "budget" and result.report.final_text is None
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert messages[-1].api_role == "tool"
    assert json.loads(messages[-1].get_value())["output"] == "Last allowed generation feedback."
    assert messages[-1].get_piece().original_value_sha256 is not None


async def test_feedback_rendering_failure_retains_acquired_grade_without_chat_output_async(tmp_path: Path) -> None:
    binding = InertSubmissionBinding(
        directory=tmp_path / "snapshots", evaluator=AsyncMock(return_value=completed_observation(0.75))
    )

    async def callback_async(*, artifact_ref: str) -> str:
        await binding.submit_async(artifact_ref=artifact_ref)
        binding.report["status"] = "error"
        raise FixtureFailureError("OFFLINE/SIMULATED feedback rendering failed")

    original = binding.hooks()
    hooks = replace(original, tools=(replace(original.tools[0], callback_async=callback_async),))
    with pytest.raises(SubmissionRunError) as caught:
        await NativeSubmissionRunner(hooks=hooks, directory=tmp_path / "runs").run_offline_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            transport=httpx.MockTransport(OfflineProvider([tool_turn(call_id="render-failed")])),
        )
    report = caught.value.result.report
    assert report.status is SubmissionReportStatus.ERROR
    assert report.report.last_valid_grade == 0.75 and report.report.submissions[0].status.value == "completed"
    assert report.calls[0].feedback is None
    assert caught.value.result.score.status is ScoreStatus.UNDETERMINED


async def test_malformed_binding_projection_preserves_partial_journal_async(tmp_path: Path) -> None:
    binding = InertSubmissionBinding(
        directory=tmp_path / "snapshots", evaluator=AsyncMock(return_value=completed_observation(0.75))
    )

    def read_report() -> dict[str, Any]:
        report = binding.read_report()
        if binding.dispatches:
            report["last_valid_grade"] = True
        return report

    hooks = replace(binding.hooks(), read_report=read_report)
    with pytest.raises(SubmissionRunError) as caught:
        await NativeSubmissionRunner(hooks=hooks, directory=tmp_path / "runs").run_offline_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            transport=httpx.MockTransport(OfflineProvider([tool_turn(call_id="bad-projection")])),
        )
    result = caught.value.result
    assert len(result.report.calls) == 1 and result.report.calls[0].feedback_kind.value == "terminal_error"
    assert result.score.score_value is None
    events = list(map(json.loads, (result.report_path.parent / "events.jsonl").read_text().splitlines()))
    projections = [json.loads(event["json"]) for event in events if event["event"] == "binding_projection"]
    assert projections[-1]["last_valid_grade"] is True
    assert events[-1]["event"] == "report_prepared"


@pytest.mark.parametrize("retention_case", ["normal", "repeat_cancel", "deadline", "write_failure"])
async def test_callback_return_boundary_cancellation_retains_latest_observation_and_output_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, retention_case: str, caplog: pytest.LogCaptureFixture
) -> None:
    evaluator = AsyncMock(
        side_effect=[
            completed_observation(0.75, feedback="First actual feedback."),
            completed_observation(0.25, feedback="Second actual feedback.\nExact bytes."),
        ]
    )
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    original_hooks = binding.hooks()

    async def submit_async(*, artifact_ref: str) -> str:
        feedback = await binding.submit_async(artifact_ref=artifact_ref)
        if len(binding.dispatches) == 2:
            asyncio.get_running_loop().call_soon(runner_task.cancel, "return-boundary")
        return feedback

    hooks = replace(original_hooks, tools=(replace(original_hooks.tools[0], callback_async=submit_async),))
    runner = NativeSubmissionRunner(hooks=hooks, directory=tmp_path / "runs")
    provider = OfflineProvider([tool_turn(call_id="first"), tool_turn(call_id="second")])
    append = runner._writer.append_async
    retain = runner._writer.retain_async

    async def append_async(*, event: str, data: dict[str, Any]) -> None:
        if retention_case == "deadline" and event == "tool_finished" and data["call_id"] == "second":
            await asyncio.Event().wait()
        await append(event=event, data=data)

    async def retain_async(report: RetainedSubmissionReport) -> Path:
        if retention_case == "repeat_cancel":
            asyncio.get_running_loop().call_soon(runner_task.cancel, "repeated-cancellation")
        if retention_case == "write_failure":
            raise OSError("OFFLINE/SIMULATED retention write failed")
        return await retain(report)

    with (
        patch.object(runner._writer, "append_async", side_effect=append_async),
        patch.object(runner._writer, "retain_async", side_effect=retain_async),
        patch.object(runner, "_RETENTION_TIMEOUT_SECONDS", 0.2 if retention_case == "deadline" else 5),
    ):
        runner_task = asyncio.create_task(
            runner.run_offline_async(
                seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
            )
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(runner_task, timeout=10)
    assert caught.value.args == ("return-boundary",)
    assert len(provider.requests) == 2 and evaluator.await_count == 2
    messages = sqlite_instance.get_conversation_messages(conversation_id=runner._run_id)
    tools = [message for message in messages if message.api_role == "tool"]
    assert len(tools) == 2
    outputs = [json.loads(message.get_value()) for message in tools]
    assert [output["call_id"] for output in outputs] == ["first", "second"]
    assert outputs[-1]["output"] == "Second actual feedback.\nExact bytes."
    assert len({message.get_piece().id for message in tools}) == 2
    assert all(message.get_piece().original_value_sha256 for message in tools)
    assert binding.report["last_valid_grade"] == 0.25
    if retention_case == "write_failure":
        assert runner.last_result is None
        assert isinstance(caught.value.__cause__, OSError)
        assert "retention failed" in caplog.text
        assert not sqlite_instance.get_scores(score_type="float_scale")
        return
    result = runner.last_result
    assert result is not None
    assert result.report.status is SubmissionReportStatus.CANCELLED
    assert result.report.report.last_valid_grade == 0.25
    assert result.report.report.selected_submission_id == binding.dispatches[-1].submission_id
    assert result.report.calls[-1].submission_ids == (binding.dispatches[-1].submission_id,)
    assert result.report.calls[-1].feedback == "Second actual feedback.\nExact bytes."
    assert result.report.calls[-1].feedback_kind.value == "returned"
    assert result.score.is_undetermined
    assert result.score.score_metadata["retained_last_valid_grade"] == 0.25
    if retention_case == "deadline":
        assert "bounded deadline" in caplog.text


@pytest.mark.parametrize(
    "point",
    [
        "pending_memory",
        "retention_start",
        "file_open",
        "file_write",
        "file_flush",
        "prepared_journal",
        "candidate_return",
    ],
)
async def test_first_finalization_cancel_publishes_only_cancelled_result_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, point: str
) -> None:
    evaluator = AsyncMock(side_effect=[completed_observation(0.75), completed_observation(0.25)])
    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="first"), tool_turn(call_id="second")])
    runner = NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=2)
    )
    loop = asyncio.get_running_loop()
    requested = False

    def request_cancel() -> None:
        nonlocal requested
        if not requested:
            requested = True
            runner_task.cancel("before-publication")

    open_file = aiofiles.open
    retain = runner._writer.retain_async
    append = runner._writer.append_async
    get_pieces = sqlite_instance.get_message_pieces

    @asynccontextmanager
    async def open_async(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        candidate = len(args) > 1 and args[1] == "x"
        if candidate and point == "file_open":
            request_cancel()
        async with open_file(*args, **kwargs) as stream:
            if candidate and point in {"file_write", "file_flush"}:
                method = "write" if point == "file_write" else "flush"
                operation = getattr(stream, method)

                async def operation_async(*values: Any, **options: Any) -> Any:
                    request_cancel()
                    return await operation(*values, **options)

                with patch.object(stream, method, side_effect=operation_async):
                    yield stream
            else:
                yield stream

    async def retain_async(report: RetainedSubmissionReport) -> Path:
        if point == "retention_start":
            request_cancel()
        path = await retain(report)
        if point == "candidate_return":
            request_cancel()
        return path

    async def append_async(*, event: str, data: dict[str, Any]) -> None:
        if point == "prepared_journal" and event == "report_prepared":
            request_cancel()
        await append(event=event, data=data)

    def get_pieces_sync(**kwargs: Any) -> Any:
        if point == "pending_memory" and len(binding.dispatches) == 2:
            loop.call_soon_threadsafe(request_cancel)
        return get_pieces(**kwargs)

    with (
        patch("pyrit.executor.benchmark.submission.evidence.aiofiles.open", side_effect=open_async),
        patch.object(runner._writer, "retain_async", side_effect=retain_async),
        patch.object(runner._writer, "append_async", side_effect=append_async),
        patch.object(sqlite_instance, "get_message_pieces", side_effect=get_pieces_sync),
    ):
        runner_task = asyncio.create_task(
            runner.run_offline_async(
                seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
            )
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(runner_task, timeout=10)
    assert caught.value.args == ("before-publication",)
    assert requested and len(provider.requests) == evaluator.await_count == 2
    result = runner.last_result
    assert result is not None and result.report.status is SubmissionReportStatus.CANCELLED
    assert result.report.report.last_valid_grade == 0.25
    assert result.score.is_undetermined
    scores = sqlite_instance.get_scores(score_type="float_scale")
    assert len(scores) == 1 and scores[0].is_undetermined
    assert scores[0].score_metadata["run_status"] == "cancelled"
    assert scores[0].score_metadata["publication_role"] == "final_run_result"
    assert scores[0].score_metadata["publication_boundary"] == "pyrit_score_commit"
    assert isinstance(scores[0].scorable, ContentEntryScorable)
    stored = sqlite_instance.get_scorable_content(content_ids=[scores[0].scorable.content_id])
    assert stored[scores[0].scorable.content_id].value == result.report_path.read_text(encoding="utf-8")
    assert len(list(result.report_path.parent.glob("*.json"))) == 1
    assert (
        RetainedSubmissionReport.model_validate_json(stored[scores[0].scorable.content_id].value).status.value
        == "cancelled"
    )


async def test_postcommit_delivery_cancel_keeps_matching_published_result_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = InertSubmissionBinding(
        directory=tmp_path / "snapshots", evaluator=AsyncMock(return_value=completed_observation(0.25))
    )
    runner = NativeSubmissionRunner(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimits(max_requests=1)
    )
    score_async = SubmissionReportScorer.score_async

    async def commit_then_schedule_cancel_async(self: SubmissionReportScorer, **kwargs: Any) -> Any:
        scores = await score_async(self, **kwargs)
        asyncio.get_running_loop().call_soon(runner_task.cancel, "after-publication")
        return scores

    with patch.object(SubmissionReportScorer, "score_async", new=commit_then_schedule_cancel_async):
        runner_task = asyncio.create_task(
            runner.run_offline_async(
                seed=_seed(),
                system_prompt="OFFLINE/SIMULATED",
                transport=httpx.MockTransport(OfflineProvider([tool_turn(call_id="published")])),
            )
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            await runner_task
    assert caught.value.args == ("after-publication",)
    result = runner.last_result
    assert result is not None and result.score.get_value() == 0.25
    assert result.report.status is SubmissionReportStatus.COMPLETED
    scores = sqlite_instance.get_scores(score_type="float_scale")
    assert len(scores) == 1 and scores[0].get_value() == 0.25
    assert result.report_path.read_text(encoding="utf-8") == result.report.canonical_json()
    assert runner._published


@pytest.mark.parametrize("cancel_kind", ["cancel", "timeout"])
async def test_callback_cancellation_survives_postcall_journal_failure_async(
    *, tmp_path: Path, cancel_kind: str
) -> None:
    entered = asyncio.Event()
    invocations = 0

    async def evaluator_async(request: Any) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            return completed_observation(0.75)
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("No evaluator retry is allowed.")

    binding = InertSubmissionBinding(directory=tmp_path / "snapshots", evaluator=evaluator_async)
    provider = OfflineProvider([tool_turn(call_id="first"), tool_turn(call_id="cancelled")])
    runner = NativeSubmissionRunner(hooks=binding.hooks(), directory=tmp_path / "runs")
    append = runner._writer.append_async

    async def fail_postcall_async(*, event: str, data: dict[str, Any]) -> None:
        if event == "tool_finished" and data["call_id"] == "cancelled":
            raise OSError("OFFLINE/SIMULATED post-call journal failure")
        await append(event=event, data=data)

    with patch.object(runner._writer, "append_async", side_effect=fail_postcall_async):
        runner_task = asyncio.create_task(
            runner.run_offline_async(
                seed=_seed(), system_prompt="OFFLINE/SIMULATED", transport=httpx.MockTransport(provider)
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        if cancel_kind == "cancel":
            runner_task.cancel("cancel-evaluator")
            with pytest.raises(asyncio.CancelledError) as caught:
                await runner_task
            assert caught.value.args == ("cancel-evaluator",)
            assert isinstance(caught.value.__cause__, OSError)
        else:
            with pytest.raises(TimeoutError) as timed_out:
                async with asyncio.timeout(0.02):
                    await runner_task
            assert isinstance(timed_out.value.__cause__, asyncio.CancelledError)
            assert isinstance(timed_out.value.__cause__.__cause__, OSError)
    assert invocations == len(provider.requests) == 2
    result = runner.last_result
    assert result is not None and result.report.report.last_valid_grade == 0.75
    assert result.report.status is SubmissionReportStatus.CANCELLED and result.score.is_undetermined
    assert result.report.calls[-1].feedback_kind.value == "cancelled"
    assert result.report.calls[-1].feedback is None
    events = list(map(json.loads, (result.report_path.parent / "events.jsonl").read_text().splitlines()))
    assert any(event["event"] == "retention_failed" and event["error_type"] == "OSError" for event in events)

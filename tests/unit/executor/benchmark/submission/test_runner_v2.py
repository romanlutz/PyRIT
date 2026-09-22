# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionEnvironmentHooksV2, SubmissionLimitsV2
from pyrit.executor.benchmark.submission.runner_v2 import NativeSubmissionRunnerV2, SubmissionRunV2Error
from pyrit.models import ContentEntryScorable, SeedPrompt
from pyrit.models.submission import SubmissionCleanupStatus, SubmissionReportStatus
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2
from tests.unit.executor.benchmark.submission.mocks import OfflineProvider, completed_observation, text_turn, tool_turn
from tests.unit.executor.benchmark.submission.mocks_v2 import (
    InertEnvironmentV2,
    InertSubmissionBindingV2,
    MockTargetFactoryV2,
    UsageProviderV2,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory
    from pyrit.models.submission_v2 import RetainedSubmissionReportV2

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _seed() -> SeedPrompt:
    return SeedPrompt(value="OFFLINE/SIMULATED inert v2 fixture.", role="user", data_type="text")


async def test_v2_caller_factory_lifecycle_latest_grade_and_replay_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    evaluator = AsyncMock(
        side_effect=[
            completed_observation(0.75, feedback="First exact\nstring."),
            completed_observation(0.25, feedback="Second exact\nstring."),
        ]
    )
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="first"), tool_turn(call_id="second")])
    factory = MockTargetFactoryV2(provider=provider)
    environment = InertEnvironmentV2()
    runner = NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_requests=2)
    )
    with (
        patch("socket.create_connection", side_effect=AssertionError("No sockets permitted")),
        patch("pyrit.auth.get_azure_openai_auth", side_effect=AssertionError("No auth permitted")),
    ):
        result = await runner.run_with_target_factory_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", target_factory=factory, environment=environment.hooks()
        )
    assert result.score.get_value() == 0.25
    assert factory.entered == factory.exited == environment.audit_calls == environment.cleanup_calls == 1
    assert factory.client is not None and factory.client.is_closed
    assert len(provider.requests) == evaluator.await_count == result.report.provider_request_count == 2
    assert result.report.generation_count == 2 and result.report.message_count == 6
    assert provider.requests[1]["input"][-1]["output"] == "First exact\nstring."
    assert result.report.mode.value == "offline" and result.report.simulated is True
    assert result.report.evidence_label == "OFFLINE/SIMULATED"
    assert result.report.target_identifier["model_name"] == "offline-fixture"
    assert result.report.report.selected_submission_id == binding.dispatches[-1].submission_id
    assert binding.dispatches[0].submission_id != binding.dispatches[1].submission_id
    assert result.report.termination_limit == "max_requests"
    assert result.report_path.read_text(encoding="utf-8") == result.report.canonical_json()
    assert isinstance(result.score.scorable, ContentEntryScorable)
    replay = (
        await SubmissionReportScorerV2(report_sha256=result.report.sha256()).score_async(scorable=result.score.scorable)
    )[0]
    assert replay.get_value() == 0.25 and evaluator.await_count == 2
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert [message.api_role for message in messages] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert messages[-1].get_piece().original_value_sha256 is not None
    assert result.score.message_piece_id is None and result.score.observation_ids == []
    journal = list(map(json.loads, (result.report_path.parent / "events.jsonl").read_text().splitlines()))
    assert all(event["evidence_label"] == "OFFLINE/SIMULATED" and event["simulated"] for event in journal)
    assert not any("authorization" in event for event in journal)


async def test_v2_honest_continuation_then_full_success_stops_async(tmp_path: Path) -> None:
    evaluator = AsyncMock(return_value=completed_observation(1))
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([text_turn("Genuine response without tools."), tool_turn(call_id="success")])
    factory = MockTargetFactoryV2(provider=provider)
    result = await NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs"
    ).run_with_target_factory_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        target_factory=factory,
        environment=InertEnvironmentV2().hooks(),
    )
    assert result.score.get_value() == 1 and result.report.final_text is None
    assert len(provider.requests) == 2 and evaluator.await_count == 1
    assert provider.requests[1]["input"][:-1] == provider.requests[0]["input"]
    assert provider.requests[1]["input"][-1]["role"] == "assistant"


@pytest.mark.parametrize("problem", ["auto_loop", "no_request_hook", "no_response_hook"])
async def test_v2_factory_contract_failures_never_dispatch_tools_async(*, tmp_path: Path, problem: str) -> None:
    evaluator = AsyncMock()
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="never-executed")])
    factory = MockTargetFactoryV2(
        provider=provider,
        auto_execute_tools=problem == "auto_loop",
        attach_request_hook=problem != "no_request_hook",
        attach_response_hook=problem != "no_response_hook",
    )
    environment = InertEnvironmentV2()
    with pytest.raises(SubmissionRunV2Error) as caught:
        await NativeSubmissionRunnerV2(
            hooks=binding.hooks(), directory=tmp_path / "runs"
        ).run_with_target_factory_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", target_factory=factory, environment=environment.hooks()
        )
    evaluator.assert_not_awaited()
    assert caught.value.result.score.is_undetermined
    assert factory.exited == environment.cleanup_calls == 1
    assert len(provider.requests) == (0 if problem == "auto_loop" else 1)


@pytest.mark.parametrize("cleanup", [SubmissionCleanupStatus.UNKNOWN, SubmissionCleanupStatus.FAILED])
async def test_v2_uncertain_environment_cleanup_retains_but_does_not_project_grade_async(
    *, tmp_path: Path, cleanup: SubmissionCleanupStatus
) -> None:
    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(1))
    )
    result = await NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs"
    ).run_with_target_factory_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        target_factory=MockTargetFactoryV2(provider=OfflineProvider([tool_turn(call_id="completed")])),
        environment=InertEnvironmentV2(cleanup=cleanup).hooks(),
    )
    assert result.report.report.last_valid_grade == 1
    assert result.report.status is SubmissionReportStatus.INCOMPLETE
    assert result.score.is_undetermined and result.report.local_cleanup is cleanup


@pytest.mark.parametrize("stage", ["audit", "target_exit", "cleanup"])
async def test_v2_explicit_lifecycle_error_is_retained_async(*, tmp_path: Path, stage: str) -> None:
    evaluator = AsyncMock(return_value=completed_observation(1))
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    factory = MockTargetFactoryV2(
        provider=OfflineProvider([tool_turn(call_id="observed")]),
        close_error=OSError("OFFLINE context exit failed") if stage == "target_exit" else None,
    )
    environment = InertEnvironmentV2(
        audit_error=OSError("OFFLINE audit failed") if stage == "audit" else None,
        cleanup_error=OSError("OFFLINE cleanup failed") if stage == "cleanup" else None,
    )
    with pytest.raises(SubmissionRunV2Error) as caught:
        await NativeSubmissionRunnerV2(
            hooks=binding.hooks(), directory=tmp_path / "runs"
        ).run_with_target_factory_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", target_factory=factory, environment=environment.hooks()
        )
    assert caught.value.result.score.is_undetermined
    assert environment.cleanup_calls == 1
    assert evaluator.await_count == (0 if stage == "audit" else 1)
    if stage != "audit":
        assert caught.value.result.report.lifecycle_errors
        assert caught.value.result.report.report.last_valid_grade == 1


async def test_v2_message_budget_is_not_a_request_budget_async(tmp_path: Path) -> None:
    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(0.25))
    )
    provider = OfflineProvider([tool_turn(call_id="last-message")])
    result = await NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_requests=8, max_messages=4)
    ).run_with_target_factory_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        target_factory=MockTargetFactoryV2(provider=provider),
        environment=InertEnvironmentV2().hooks(),
    )
    assert result.score.get_value() == 0.25
    assert result.report.message_count == 4 and result.report.provider_request_count == 1
    assert result.report.termination_limit == "max_messages"


async def test_v2_token_budget_uses_actual_input_and_output_and_finishes_last_tool_async(tmp_path: Path) -> None:
    evaluator = AsyncMock(side_effect=[completed_observation(0.75), completed_observation(0.25)])
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = UsageProviderV2(
        turns=[tool_turn(call_id="first"), tool_turn(call_id="last")],
        usages=[{"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}] * 2,
    )
    result = await NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_tokens=20)
    ).run_with_target_factory_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        target_factory=MockTargetFactoryV2(provider=provider),
        environment=InertEnvironmentV2().hooks(),
    )
    assert result.report.total_tokens == 30
    assert result.report.termination_limit == "max_tokens"
    assert result.score.get_value() == 0.25 and evaluator.await_count == len(provider.requests) == 2


@pytest.mark.parametrize(
    "usage", [None, {}, {"input_tokens": True, "output_tokens": 1}, {"input_tokens": 1, "output_tokens": -1}]
)
async def test_v2_required_missing_token_usage_is_not_zero_async(
    *, tmp_path: Path, usage: dict[str, Any] | None
) -> None:
    evaluator = AsyncMock()
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = UsageProviderV2(turns=[tool_turn(call_id="unobserved-usage")], usages=[usage])
    with pytest.raises(SubmissionRunV2Error) as caught:
        await NativeSubmissionRunnerV2(
            hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_tokens=10)
        ).run_with_target_factory_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            target_factory=MockTargetFactoryV2(provider=provider),
            environment=InertEnvironmentV2().hooks(),
        )
    evaluator.assert_not_awaited()
    assert len(provider.requests) == 1
    assert caught.value.result.report.total_tokens is None and caught.value.result.score.is_undetermined


async def test_v2_return_boundary_cancel_retains_latest_feedback_and_runs_owned_cleanup_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    evaluator = AsyncMock(
        side_effect=[completed_observation(0.75), completed_observation(0.25, feedback="Exact second\nfeedback")]
    )
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    hooks = binding.hooks()

    async def submit_async(*, artifact_ref: str) -> str:
        feedback = await binding.submit_async(artifact_ref=artifact_ref)
        if len(binding.dispatches) == 2:
            asyncio.get_running_loop().call_soon(owner.cancel, "v2 return boundary")
        return feedback

    hooks = replace(hooks, tools=(replace(hooks.tools[0], callback_async=submit_async),))
    runner = NativeSubmissionRunnerV2(hooks=hooks, directory=tmp_path / "runs")
    provider = OfflineProvider([tool_turn(call_id="first"), tool_turn(call_id="second")])
    factory = MockTargetFactoryV2(provider=provider)
    environment = InertEnvironmentV2()
    owner = asyncio.create_task(
        runner.run_with_target_factory_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", target_factory=factory, environment=environment.hooks()
        )
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        await owner
    assert caught.value.args == ("v2 return boundary",)
    result = runner.last_result
    assert result is not None and result.score.is_undetermined
    assert result.report.report.last_valid_grade == 0.25 and result.report.status is SubmissionReportStatus.CANCELLED
    assert result.report.calls[-1].feedback == "Exact second\nfeedback"
    assert len(provider.requests) == evaluator.await_count == 2
    assert factory.exited == environment.cleanup_calls == 1
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert json.loads(messages[-1].get_value())["output"] == "Exact second\nfeedback"
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1


async def test_v2_first_finalization_cancel_has_no_stale_clean_score_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(0.25))
    )
    runner = NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_requests=1)
    )
    from pyrit.executor.benchmark.submission.evidence_v2 import SubmissionEvidenceWriterV2

    retain = SubmissionEvidenceWriterV2.retain_async
    cancelled = False

    async def retain_async(self: SubmissionEvidenceWriterV2, report: RetainedSubmissionReportV2) -> Path:
        nonlocal cancelled
        path = await retain(self, report)
        if not cancelled:
            cancelled = True
            owner.cancel("v2 before publication")
        return path

    with patch.object(SubmissionEvidenceWriterV2, "retain_async", new=retain_async):
        owner = asyncio.create_task(
            runner.run_with_target_factory_async(
                seed=_seed(),
                system_prompt="OFFLINE/SIMULATED",
                target_factory=MockTargetFactoryV2(provider=OfflineProvider([tool_turn(call_id="first")])),
                environment=InertEnvironmentV2().hooks(),
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await owner
    result = runner.last_result
    assert result is not None and result.report.status is SubmissionReportStatus.CANCELLED
    assert result.score.is_undetermined and result.report.report.last_valid_grade == 0.25
    scores = sqlite_instance.get_scores(score_type="float_scale")
    assert len(scores) == 1 and scores[0].is_undetermined
    assert len(list(result.report_path.parent.glob("*.json"))) == 1


async def test_v2_invalid_initial_provenance_fails_before_any_caller_operation_async(tmp_path: Path) -> None:
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=AsyncMock())
    initial = binding.read_report()
    initial.update(mode="real", simulated=True)
    hooks = replace(binding.hooks(), read_report=lambda: initial)
    factory = MockTargetFactoryV2(provider=OfflineProvider([]))
    environment = InertEnvironmentV2()
    with pytest.raises(ValueError, match="provenance"):
        await NativeSubmissionRunnerV2(hooks=hooks, directory=tmp_path / "runs").run_with_target_factory_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", target_factory=factory, environment=environment.hooks()
        )
    assert factory.entered == environment.audit_calls == environment.cleanup_calls == 0
    assert not (tmp_path / "runs").exists()


async def test_v2_recoverable_tool_error_is_exact_and_allows_one_correction_async(tmp_path: Path) -> None:
    evaluator = AsyncMock(return_value=completed_observation(1, feedback="Genuine caller string"))
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="missing", artifact_ref="absent"), tool_turn(call_id="corrected")])
    result = await NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs"
    ).run_with_target_factory_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        target_factory=MockTargetFactoryV2(provider=provider),
        environment=InertEnvironmentV2().hooks(),
    )
    assert provider.requests[1]["input"][-1]["output"] == "Error: Fixture not found.\nChoose an existing fixture."
    assert result.report.calls[0].feedback_kind.value == "recoverable_error"
    assert evaluator.await_count == 1 and result.score.get_value() == 1


@pytest.mark.parametrize(
    "failure", [ConnectionError("Unknown accepted outcome"), ImportError("Fixture dependency missing")]
)
async def test_v2_unknown_or_infrastructure_outcome_keeps_prior_grade_without_resubmission_async(
    *, tmp_path: Path, failure: Exception
) -> None:
    evaluator = AsyncMock(side_effect=[completed_observation(0.75), failure])
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([tool_turn(call_id="first"), tool_turn(call_id="failure")])
    environment = InertEnvironmentV2()
    with pytest.raises(SubmissionRunV2Error) as caught:
        await NativeSubmissionRunnerV2(
            hooks=binding.hooks(), directory=tmp_path / "runs"
        ).run_with_target_factory_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            target_factory=MockTargetFactoryV2(provider=provider),
            environment=environment.hooks(),
        )
    report = caught.value.result.report
    assert report.report.last_valid_grade == 0.75 and caught.value.result.score.is_undetermined
    assert report.report.submissions[-1].observed_receipt_id is None
    assert report.report.submissions[-1].remote_disposition.value == "unknown"
    assert len(provider.requests) == evaluator.await_count == 2
    assert environment.cleanup_calls == 1


async def test_v2_request_hook_blocks_retry_before_a_second_provider_dispatch_async(tmp_path: Path) -> None:
    class RateLimitedFixture(OfflineProvider):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            return httpx.Response(
                429,
                json={"error": {"message": "OFFLINE/SIMULATED rate limit", "type": "rate_limit", "code": "fixture"}},
            )

    provider = RateLimitedFixture([])
    evaluator = AsyncMock()
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    factory = MockTargetFactoryV2(provider=provider, sdk_retries=1)
    with pytest.raises(SubmissionRunV2Error) as caught:
        await NativeSubmissionRunnerV2(
            hooks=binding.hooks(), directory=tmp_path / "runs"
        ).run_with_target_factory_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            target_factory=factory,
            environment=InertEnvironmentV2().hooks(),
        )
    assert len(provider.requests) == caught.value.result.report.provider_request_count == 1
    assert caught.value.result.report.token_usage == (None,)
    assert caught.value.result.report.total_tokens is None
    evaluator.assert_not_awaited()
    assert caught.value.result.score.is_undetermined and factory.client.is_closed


async def test_v2_provenance_cannot_change_after_a_callback_async(tmp_path: Path) -> None:
    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(0.25))
    )
    hooks = binding.hooks()

    def report_snapshot() -> dict[str, Any]:
        report = binding.read_report()
        if binding.dispatches:
            report.update(mode="real", simulated=False)
        return report

    with pytest.raises(SubmissionRunV2Error) as caught:
        await NativeSubmissionRunnerV2(
            hooks=replace(hooks, read_report=report_snapshot), directory=tmp_path / "runs"
        ).run_with_target_factory_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            target_factory=MockTargetFactoryV2(provider=OfflineProvider([tool_turn(call_id="changed")])),
            environment=InertEnvironmentV2().hooks(),
        )
    result = caught.value.result
    assert result.report.mode.value == "offline" and result.report.simulated
    assert result.report.report.last_valid_grade is None and result.score.is_undetermined
    assert "provenance" in result.report.runner_error


async def test_v2_postcommit_cancel_preserves_matching_published_outcome_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(0.25))
    )
    runner = NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_requests=1)
    )
    score_async = SubmissionReportScorerV2.score_async

    async def score_then_cancel_async(self: SubmissionReportScorerV2, **kwargs: Any) -> Any:
        scores = await score_async(self, **kwargs)
        asyncio.get_running_loop().call_soon(owner.cancel, "v2 after publication")
        return scores

    with patch.object(SubmissionReportScorerV2, "score_async", new=score_then_cancel_async):
        owner = asyncio.create_task(
            runner.run_with_target_factory_async(
                seed=_seed(),
                system_prompt="OFFLINE/SIMULATED",
                target_factory=MockTargetFactoryV2(provider=OfflineProvider([tool_turn(call_id="final")])),
                environment=InertEnvironmentV2().hooks(),
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await owner
    result = runner.last_result
    assert result is not None and result.report.status is SubmissionReportStatus.COMPLETED
    assert result.score.get_value() == 0.25
    scores = sqlite_instance.get_scores(score_type="float_scale")
    assert len(scores) == 1 and scores[0].get_value() == 0.25
    assert result.report_path.read_text(encoding="utf-8") == result.report.canonical_json()


async def test_v2_cancelled_evaluator_with_cleanup_error_preserves_original_cancellation_async(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def evaluator_async(request: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("No retry")

    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator_async)
    runner = NativeSubmissionRunnerV2(hooks=binding.hooks(), directory=tmp_path / "runs")
    environment = InertEnvironmentV2(cleanup_error=OSError("OFFLINE cleanup unavailable"))
    provider = OfflineProvider([tool_turn(call_id="cancelled")])
    factory = MockTargetFactoryV2(provider=provider)
    owner = asyncio.create_task(
        runner.run_with_target_factory_async(
            seed=_seed(), system_prompt="OFFLINE/SIMULATED", target_factory=factory, environment=environment.hooks()
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    owner.cancel("original v2 cancellation")
    with pytest.raises(asyncio.CancelledError) as caught:
        await owner
    assert caught.value.args == ("original v2 cancellation",)
    assert len(provider.requests) == factory.exited == environment.cleanup_calls == 1
    result = runner.last_result
    assert result is not None and result.report.status is SubmissionReportStatus.CANCELLED
    assert result.report.local_cleanup is SubmissionCleanupStatus.UNKNOWN
    assert result.report.lifecycle_errors and result.score.is_undetermined
    assert result.report.report.submissions[0].remote_disposition.value == "unknown"


async def test_v2_initial_message_bound_is_no_submission_without_provider_dispatch_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    evaluator = AsyncMock()
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([])
    result = await NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_messages=2)
    ).run_with_target_factory_async(
        seed=_seed(),
        system_prompt="OFFLINE/SIMULATED",
        target_factory=MockTargetFactoryV2(provider=provider),
        environment=InertEnvironmentV2().hooks(),
    )
    assert result.report.message_count == 2 and result.report.provider_request_count == 0
    assert result.report.status is SubmissionReportStatus.NO_SUBMISSION and result.score.is_undetermined
    assert not provider.requests
    evaluator.assert_not_awaited()
    messages = sqlite_instance.get_conversation_messages(conversation_id=result.report.conversation_id)
    assert [message.api_role for message in messages] == ["system", "user"]


async def test_v2_repeated_cancellation_during_owned_cleanup_has_one_cleanup_call_async(tmp_path: Path) -> None:
    from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionEnvironmentHooksV2

    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_calls = 0

    async def cleanup_async() -> SubmissionCleanupStatus:
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_entered.set()
        await release_cleanup.wait()
        return SubmissionCleanupStatus.COMPLETE

    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(0.25))
    )
    runner = NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(max_requests=1)
    )
    factory = MockTargetFactoryV2(provider=OfflineProvider([tool_turn(call_id="completed")]))
    owner = asyncio.create_task(
        runner.run_with_target_factory_async(
            seed=_seed(),
            system_prompt="OFFLINE/SIMULATED",
            target_factory=factory,
            environment=SubmissionEnvironmentHooksV2(
                audit_async=AsyncMock(return_value={"fixture": "OFFLINE/SIMULATED"}), cleanup_async=cleanup_async
            ),
        )
    )
    await asyncio.wait_for(cleanup_entered.wait(), timeout=5)
    owner.cancel("first cleanup cancellation")
    await asyncio.sleep(0)
    owner.cancel("second cleanup cancellation")
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await owner
    assert caught.value.args == ("first cleanup cancellation",)
    assert cleanup_calls == factory.exited == 1
    result = runner.last_result
    assert result is not None and result.report.local_cleanup is SubmissionCleanupStatus.COMPLETE
    assert result.report.status is SubmissionReportStatus.CANCELLED and result.score.is_undetermined
    assert result.report.report.last_valid_grade == 0.25


async def test_v2_cleanup_deadline_does_not_claim_resource_removal_or_clean_result_async(tmp_path: Path) -> None:
    from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionEnvironmentHooksV2

    calls = 0

    async def cleanup_async() -> SubmissionCleanupStatus:
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        raise AssertionError("No cleanup retry")

    binding = InertSubmissionBindingV2(
        directory=tmp_path / "inert", evaluator=AsyncMock(return_value=completed_observation(1))
    )
    runner = NativeSubmissionRunnerV2(hooks=binding.hooks(), directory=tmp_path / "runs")
    with patch.object(runner, "_CLEANUP_TIMEOUT_SECONDS", 0.02):
        with pytest.raises(SubmissionRunV2Error) as caught:
            await runner.run_with_target_factory_async(
                seed=_seed(),
                system_prompt="OFFLINE/SIMULATED",
                target_factory=MockTargetFactoryV2(provider=OfflineProvider([tool_turn(call_id="completed")])),
                environment=SubmissionEnvironmentHooksV2(
                    audit_async=AsyncMock(return_value={"fixture": "OFFLINE/SIMULATED"}), cleanup_async=cleanup_async
                ),
            )
    assert calls == 1
    assert caught.value.result.report.local_cleanup is SubmissionCleanupStatus.UNKNOWN
    assert caught.value.result.score.is_undetermined
    assert caught.value.result.report.lifecycle_errors
    assert caught.value.result.report.report.last_valid_grade == 1


@pytest.mark.parametrize(
    "limits",
    [
        {"max_messages": 0},
        {"max_tokens": True},
        {"max_requests": 1.5},
        {"episode_timeout_seconds": float("inf")},
    ],
)
def test_v2_distinct_budget_fields_reject_invalid_values(limits: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SubmissionLimitsV2(**limits)


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_v2_audit_uses_its_own_deadline_before_factory_entry_async(
    *, tmp_path: Path, cleanup_fails: bool
) -> None:
    audit_calls = 0
    audit_task_ended = asyncio.Event()

    async def audit_async() -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        try:
            await asyncio.Event().wait()
            raise AssertionError("The uncompleted audit must not enter the target factory.")
        finally:
            audit_task_ended.set()

    evaluator = AsyncMock()
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = OfflineProvider([])
    factory = MockTargetFactoryV2(provider=provider)
    cleanup = AsyncMock(
        return_value=SubmissionCleanupStatus.NOT_REQUIRED,
        side_effect=OSError("OFFLINE supplemental cleanup failure") if cleanup_fails else None,
    )
    runner = NativeSubmissionRunnerV2(
        hooks=binding.hooks(), directory=tmp_path / "runs", limits=SubmissionLimitsV2(episode_timeout_seconds=30)
    )
    assert runner._CLEANUP_TIMEOUT_SECONDS == 5.0
    with (
        patch.object(runner, "_CLEANUP_TIMEOUT_SECONDS", 0.03),
        patch.object(runner, "_retain_async", wraps=runner._retain_async) as retain,
    ):
        with pytest.raises(SubmissionRunV2Error) as caught:
            await asyncio.wait_for(
                runner.run_with_target_factory_async(
                    seed=_seed(),
                    system_prompt="OFFLINE/SIMULATED",
                    target_factory=factory,
                    environment=SubmissionEnvironmentHooksV2(audit_async=audit_async, cleanup_async=cleanup),
                ),
                timeout=5,
            )
    assert retain.call_args_list[0].kwargs["timeout"] == 0.03
    assert audit_calls == 1 and audit_task_ended.is_set()
    assert factory.entered == factory.exited == 0 and not provider.requests
    evaluator.assert_not_awaited()
    cleanup.assert_awaited_once()
    assert isinstance(caught.value.__cause__, TimeoutError)
    result = caught.value.result
    assert result.report.environment_audit is None
    assert result.report.generation_count == result.report.provider_request_count == 0
    assert result.report.status is SubmissionReportStatus.INCOMPLETE and result.score.is_undetermined
    assert result.report.report.submissions == ()
    if cleanup_fails:
        assert result.report.local_cleanup is SubmissionCleanupStatus.UNKNOWN
        assert "supplemental cleanup failure" in result.report.lifecycle_errors[0]
    else:
        assert result.report.local_cleanup is SubmissionCleanupStatus.NOT_REQUIRED
    events = list(map(json.loads, (result.report_path.parent / "events.jsonl").read_text().splitlines()))
    assert any(event["event"] == "retention_failed" and event["error_type"] == "TimeoutError" for event in events)
    assert not any(event["event"] == "provider_request" for event in events)


@pytest.mark.parametrize("audit_returns", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_v2_repeated_caller_cancel_during_audit_blocks_factory_and_cleans_once_async(
    *, tmp_path: Path, audit_returns: bool, cleanup_fails: bool
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    audit_calls = 0

    async def audit_async() -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        entered.set()
        await release.wait()
        return {"fixture": "OFFLINE/SIMULATED", "actual_resource_operations": 0}

    evaluator = AsyncMock()
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    factory = MockTargetFactoryV2(provider=OfflineProvider([]))
    cleanup = AsyncMock(
        return_value=SubmissionCleanupStatus.NOT_REQUIRED,
        side_effect=OSError("OFFLINE supplemental cleanup failure") if cleanup_fails else None,
    )
    runner = NativeSubmissionRunnerV2(hooks=binding.hooks(), directory=tmp_path / "runs")
    with patch.object(runner, "_CLEANUP_TIMEOUT_SECONDS", 0.1):
        owner = asyncio.create_task(
            runner.run_with_target_factory_async(
                seed=_seed(),
                system_prompt="OFFLINE/SIMULATED",
                target_factory=factory,
                environment=SubmissionEnvironmentHooksV2(audit_async=audit_async, cleanup_async=cleanup),
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        owner.cancel("first audit cancellation")
        await asyncio.sleep(0)
        owner.cancel("second audit cancellation")
        if audit_returns:
            release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await asyncio.wait_for(owner, timeout=5)
    assert caught.value.args == ("first audit cancellation",)
    assert audit_calls == 1 and factory.entered == factory.exited == 0
    evaluator.assert_not_awaited()
    cleanup.assert_awaited_once()
    result = runner.last_result
    assert result is not None and result.report.status is SubmissionReportStatus.CANCELLED
    assert result.score.is_undetermined
    assert result.report.provider_request_count == result.report.generation_count == 0
    assert result.report.environment_audit is not None if audit_returns else result.report.environment_audit is None
    if cleanup_fails:
        assert result.report.local_cleanup is SubmissionCleanupStatus.UNKNOWN
        assert result.report.lifecycle_errors
    assert not factory.provider.requests


@pytest.mark.parametrize("failure", ["read_timeout", "cancelled", "malformed_response", "response_journal_failure"])
@pytest.mark.parametrize("required_token_usage", [False, True])
async def test_v2_generation_exit_marks_unacquired_usage_unknown_and_retains_prior_usage_async(
    *, tmp_path: Path, failure: str, required_token_usage: bool, sqlite_instance: SQLiteMemory
) -> None:
    second_request_entered = asyncio.Event()
    known_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    class InterruptedProvider(UsageProviderV2):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            if not self.requests:
                return super().__call__(request)
            if failure == "response_journal_failure":
                return super().__call__(request)
            self.requests.append(json.loads(request.content))
            second_request_entered.set()
            if failure == "cancelled":
                await asyncio.Event().wait()
                raise AssertionError("A cancelled provider request must not be retried.")
            if failure == "malformed_response":
                return httpx.Response(200, content=b"OFFLINE/SIMULATED invalid response")
            raise httpx.ReadTimeout("OFFLINE/SIMULATED read timed out", request=request)

    evaluator = AsyncMock(return_value=completed_observation(0.75))
    binding = InertSubmissionBindingV2(directory=tmp_path / "inert", evaluator=evaluator)
    provider = InterruptedProvider(
        turns=[tool_turn(call_id="known-usage"), tool_turn(call_id="never-executed")],
        usages=[known_usage, known_usage],
    )
    factory = MockTargetFactoryV2(provider=provider)
    environment = InertEnvironmentV2()
    runner = NativeSubmissionRunnerV2(
        hooks=binding.hooks(),
        directory=tmp_path / "runs",
        limits=SubmissionLimitsV2(max_tokens=100 if required_token_usage else None),
    )
    from pyrit.executor.benchmark.submission.evidence_v2 import SubmissionEvidenceWriterV2

    append = SubmissionEvidenceWriterV2.append_async

    async def append_async(self: SubmissionEvidenceWriterV2, *, event: str, data: dict[str, Any]) -> None:
        if failure == "response_journal_failure" and event == "provider_response" and data["generation"] == 2:
            raise OSError("OFFLINE/SIMULATED response evidence unavailable")
        await append(self, event=event, data=data)

    with patch.object(SubmissionEvidenceWriterV2, "append_async", new=append_async):
        owner = asyncio.create_task(
            runner.run_with_target_factory_async(
                seed=_seed(),
                system_prompt="OFFLINE/SIMULATED",
                target_factory=factory,
                environment=environment.hooks(),
            )
        )
        if failure == "cancelled":
            await asyncio.wait_for(second_request_entered.wait(), timeout=5)
            owner.cancel("cancel second admitted request")
            with pytest.raises(asyncio.CancelledError) as caught_cancel:
                await owner
            assert caught_cancel.value.args == ("cancel second admitted request",)
            result = runner.last_result
        else:
            with pytest.raises(SubmissionRunV2Error) as caught:
                await owner
            result = caught.value.result
    assert result is not None and result.score.is_undetermined
    assert result.report.generation_count == result.report.provider_request_count == len(provider.requests) == 2
    assert result.report.total_tokens is None
    assert result.report.token_usage == (known_usage, None)
    assert result.report.report.last_valid_grade == 0.75
    assert result.report.mode.value == "offline" and result.report.simulated
    assert result.report.status is (
        SubmissionReportStatus.CANCELLED if failure == "cancelled" else SubmissionReportStatus.INCOMPLETE
    )
    evaluator.assert_awaited_once()
    assert factory.exited == environment.cleanup_calls == 1
    assert factory.client is not None and factory.client.is_closed
    assert len(result.report.calls) == 1 and result.report.calls[0].call_id == "known-usage"
    assert isinstance(result.score.scorable, ContentEntryScorable)
    stored = sqlite_instance.get_scorable_content(content_ids=[result.score.scorable.content_id])
    retained = json.loads(stored[result.score.scorable.content_id].value)
    assert retained["token_usage"] == [known_usage, None] and retained["total_tokens"] is None
    events = list(map(json.loads, (result.report_path.parent / "events.jsonl").read_text().splitlines()))
    assert sum(event["event"] == "provider_request" for event in events) == 2

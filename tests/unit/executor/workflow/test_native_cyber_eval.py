# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from pyrit.converter import Base64Converter
from pyrit.executor.workflow.native_cyber_eval import NativeCyberEvaluation, NativeCyberTaskBinding
from pyrit.models import ContentEntryScorable, SeedPrompt
from pyrit.models.native_cyber import (
    NativeAgentEvidence,
    NativeCyberArtifact,
    NativeCyberJudgment,
    NativeCyberReadiness,
    NativeCyberReport,
    NativeCyberRequest,
)
from pyrit.prompt_target import NativeAgentTarget
from pyrit.registry import ConverterRegistry
from pyrit.score.float_scale.native_cyber_scorer import NativeCyberReportScorer
from tests.unit.prompt_target.target.test_native_agent_target import SdkSessionFixture, agent_session, tool_turn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


class FixtureRuntime:
    def __init__(self, *, turns: int, steps: bool, order: list[str], missing_tool: bool = False) -> None:
        sdk_turns = [tool_turn(call_id=f"call-{i}", final=None) for i in range(turns)]
        if missing_tool:
            sdk_turns[0].pop(2)
        self.sdk = SdkSessionFixture(sdk_turns)
        self.session = agent_session(self.sdk, steps=steps)
        self.target = NativeAgentTarget(session=self.session)
        self.order = order
        self.grade_count = 0
        self.closed = False

    async def grade_async(self, *, evidence: NativeAgentEvidence) -> NativeCyberJudgment:
        assert self.sdk.disconnected and not self.closed
        assert evidence.coverage_complete and evidence.idle
        self.grade_count += 1
        self.order.append("grade")
        return NativeCyberJudgment(
            value=0.75, complete=True, rationale="Inert original grader", evidence={"fixture": True}
        )


class FixtureBinding(NativeCyberTaskBinding):
    def __init__(self, *, steps: bool = False, blocked: bool = False, missing_tool: bool = False) -> None:
        super().__init__(name="inert_case", version="1", description="OFFLINE/SIMULATED only", max_ttl_seconds=180)
        self.order: list[str] = []
        self.runtime = FixtureRuntime(turns=3, steps=steps, order=self.order, missing_tool=missing_tool)
        self.blocked = blocked
        self.open_count = 0
        self.close_count = 0

    async def readiness_async(self) -> NativeCyberReadiness:
        return NativeCyberReadiness(
            ready=not self.blocked,
            blockers=("GHCP container auth is not qualified.",) if self.blocked else (),
            simulated=True,
            capabilities=self.runtime.session.capabilities,
            provenance={"fixture": "OFFLINE/SIMULATED"},
        )

    @asynccontextmanager
    async def open_runtime(self, *, run_id: str, request: NativeCyberRequest) -> AsyncIterator[FixtureRuntime]:
        self.open_count += 1
        self.order.append("open")
        try:
            yield self.runtime
        finally:
            self.order.append("close")
            self.close_count += 1
            self.runtime.closed = True


async def test_native_literal_attack_scores_before_single_cleanup_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="Literal fixture instruction"), directory=tmp_path
    )
    view = await run.start_async()
    assert view.status == "completed"
    assert binding.order == ["open", "grade", "close"]
    assert binding.open_count == binding.close_count == binding.runtime.grade_count == 1
    assert binding.runtime.sdk.prompts == ["Literal fixture instruction"]
    assert run.report.attack_result_id and run.report.technique_identifier
    assert run.report.technique_identifier["children"]["attack"]["class_name"] == "PromptSendingAttack"
    assert isinstance(run.score.scorable, ContentEntryScorable) and run.score.get_value() == 0.75
    assert run.score.message_piece_id is None
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1
    await run.finish_async()
    assert binding.runtime.grade_count == 1


async def test_blocked_preflight_does_not_open_or_grade_async(tmp_path: Path) -> None:
    binding = FixtureBinding(blocked=True)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    view = await run.start_async()
    assert view.status == "blocked" and view.blockers
    assert binding.open_count == binding.runtime.grade_count == 0
    assert run.score.is_undetermined and run.report.agent is None


async def test_retained_step_uses_same_session_and_scores_only_at_finish_async(tmp_path: Path) -> None:
    binding = FixtureBinding(steps=True)
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="First", operator_steps=True), directory=tmp_path
    )
    first = await run.start_async()
    assert first.can_step and first.status == "awaiting_instruction"
    assert binding.runtime.grade_count == binding.close_count == 0
    second = await run.step_async("Next")
    assert second.conversation_id == first.conversation_id and second.turn_count == 2
    assert binding.runtime.sdk.prompts == ["First", "Next"]
    result = await run.finish_async()
    assert result.status == "completed" and not result.can_step
    assert binding.order == ["open", "grade", "close"]
    with pytest.raises(ValueError, match="unexpired"):
        await run.step_async("Not a clone")


async def test_cancel_and_ttl_never_grade_uncertain_run_async(tmp_path: Path) -> None:
    binding = FixtureBinding(steps=True)
    run = NativeCyberEvaluation(
        binding=binding, request=NativeCyberRequest(instruction="First", operator_steps=True), directory=tmp_path
    )
    await run.start_async()
    result = await run.cancel_async()
    assert result.status == "cancelled" and run.score.is_undetermined
    assert binding.runtime.grade_count == 0 and binding.close_count == 1
    expired_binding = FixtureBinding(steps=True)
    expired = NativeCyberEvaluation(
        binding=expired_binding,
        request=NativeCyberRequest(instruction="First", operator_steps=True),
        directory=tmp_path,
    )
    await expired.start_async()
    expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await expired._expire_async()
    assert expired.view().status == "expired"
    assert expired_binding.close_count == 1 and expired_binding.runtime.grade_count == 0
    assert expired.score.is_undetermined


async def test_missing_native_event_blocks_grade_even_if_agent_returns_async(tmp_path: Path) -> None:
    binding = FixtureBinding(missing_tool=True)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="First"), directory=tmp_path)
    view = await run.start_async()
    assert view.status == "error" and run.score.is_undetermined
    assert binding.runtime.grade_count == 0
    assert run.report.agent.gaps and binding.close_count == 1


async def test_fresh_rerun_has_distinct_lineage_environment_and_content_async(tmp_path: Path) -> None:
    first = NativeCyberEvaluation(
        binding=FixtureBinding(), request=NativeCyberRequest(instruction="First"), directory=tmp_path
    )
    await first.start_async()
    second = NativeCyberEvaluation(
        binding=FixtureBinding(),
        request=NativeCyberRequest(instruction="Edited", parent_run_id=first.run_id),
        directory=tmp_path,
    )
    await second.start_async()
    assert second.report.request.parent_run_id == first.run_id
    assert first.run_id != second.run_id
    assert first.report.agent.environment_id != second.report.agent.environment_id
    assert first.report.agent.session_id != second.report.agent.session_id
    assert first.report.input_sha256 != second.report.input_sha256


async def test_binding_field_capability_gates_async(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="approved maximum"):
        NativeCyberEvaluation(
            binding=FixtureBinding(), request=NativeCyberRequest(instruction="x", ttl_seconds=300), directory=tmp_path
        )
    with pytest.raises(ValueError, match="converter"):
        NativeCyberEvaluation(
            binding=FixtureBinding(),
            request=NativeCyberRequest(instruction="x", converter_names=("bad",)),
            directory=tmp_path,
        )
    run = NativeCyberEvaluation(
        binding=FixtureBinding(), request=NativeCyberRequest(instruction="x", operator_steps=True), directory=tmp_path
    )
    with pytest.raises(ValueError, match="not qualified"):
        await run.start_async()


async def test_original_grader_failure_keeps_environment_cleanup_and_no_retry_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(
        binding.runtime, "grade_async", new_callable=AsyncMock, side_effect=OSError("inert grader failed")
    ) as grade:
        view = await run.start_async()
    grade.assert_awaited_once()
    assert view.status == "error" and run.score.is_undetermined
    assert binding.close_count == 1
    assert any("grader failed" in item for item in run.report.errors)


async def test_cleanup_failure_retains_acquired_original_judgment_without_clean_score_async(tmp_path: Path) -> None:
    class CleanupFailureBinding(FixtureBinding):
        @asynccontextmanager
        async def open_runtime(self, *, run_id: str, request: NativeCyberRequest) -> AsyncIterator[FixtureRuntime]:
            self.open_count += 1
            try:
                yield self.runtime
            finally:
                self.close_count += 1
                raise OSError("inert removal not verified")

    binding = CleanupFailureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    view = await run.start_async()
    assert view.status == "error" and run.score.is_undetermined
    assert run.report.judgment.value == 0.75
    assert run.report.cleanup == "failed"
    assert binding.runtime.grade_count == binding.close_count == 1


async def test_artifact_acquisition_and_replay_do_not_execute_grader_again_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = FixtureBinding()
    artifact = b"OFFLINE/SIMULATED inert output"
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_bytes(artifact)
    judgment = NativeCyberJudgment(
        value=0.75,
        complete=True,
        rationale="Original fixture judgment",
        artifacts=(
            NativeCyberArtifact(
                name="artifact.txt",
                sha256=hashlib.sha256(artifact).hexdigest(),
                size_bytes=len(artifact),
                evidence_ref=str(artifact_path),
            ),
        ),
    )
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding.runtime, "grade_async", new_callable=AsyncMock, return_value=judgment) as grade:
        await run.start_async()
        replay = (
            await NativeCyberReportScorer(report_sha256=run.report.sha256()).score_async(scorable=run.score.scorable)
        )[0]
    grade.assert_awaited_once()
    assert replay.get_value() == 0.75
    assert run.report.judgment.artifacts[0].sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    stored = sqlite_instance.get_scorable_content(content_ids=[run.score.scorable.content_id])
    assert NativeCyberReport.model_validate_json(stored[run.score.scorable.content_id].value) == run.report


async def test_cancel_requested_while_turn_running_waits_for_turn_boundary_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    entered, release = asyncio.Event(), asyncio.Event()
    original = binding.runtime.sdk.send_and_wait

    async def send_async(prompt: str, *, timeout: float) -> object:
        entered.set()
        await release.wait()
        return await original(prompt, timeout=timeout)

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding.runtime.sdk, "send_and_wait", side_effect=send_async):
        task = asyncio.create_task(run.start_async())
        await asyncio.wait_for(entered.wait(), timeout=5)
        view = await run.cancel_async()
        assert view.cancel_requested and view.status == "running"
        assert not binding.runtime.sdk.disconnected and binding.close_count == 0
        release.set()
        result = await task
    assert result.status == "cancelled" and binding.runtime.grade_count == 0
    assert binding.close_count == 1 and run.score.is_undetermined


async def test_actual_transformed_seed_hash_is_retained_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    with patch.object(
        binding, "seed", return_value=SeedPrompt(value="literal selected variant", data_type="text", role="user")
    ):
        run = NativeCyberEvaluation(
            binding=binding, request=NativeCyberRequest(instruction="variant selector"), directory=tmp_path
        )
        await run.start_async()
    assert run.report.input_sha256 == hashlib.sha256(b"literal selected variant").hexdigest()
    assert binding.runtime.sdk.prompts == ["literal selected variant"]


async def test_real_rerun_method_rejects_reused_environment_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    parent = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="first"), directory=tmp_path)
    await parent.start_async()
    child = parent.rerun(request=NativeCyberRequest(instruction="second"))
    view = await child.start_async()
    assert child.run_id != parent.run_id
    assert child.request.parent_run_id == parent.run_id
    assert view.status == "error"
    assert child.score.is_undetermined
    assert any("reuse" in item for item in child.report.errors)


async def test_literal_baseline_uses_the_existing_converter_pipeline_async(tmp_path: Path) -> None:
    import base64

    binding = FixtureBinding()
    binding.allowed_converter_names = ("fixture_base64",)
    registry = ConverterRegistry()
    registry.instances.register(Base64Converter(), name="fixture_base64")
    with patch.object(ConverterRegistry, "get_registry_singleton", return_value=registry):
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="literal fixture", converter_names=("fixture_base64",)),
            directory=tmp_path,
        )
        view = await run.start_async()
    assert view.status == "completed"
    assert binding.runtime.sdk.prompts == [base64.b64encode(b"literal fixture").decode()]
    assert run.report.input_sha256 == hashlib.sha256(b"literal fixture").hexdigest()

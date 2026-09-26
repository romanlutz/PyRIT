# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

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
        self.storage_checked = False

    async def validate_agent_storage_async(self, *, directory: Path) -> None:
        assert await asyncio.to_thread(directory.is_dir)
        self.storage_checked = True

    async def grade_async(self, *, evidence: NativeAgentEvidence) -> NativeCyberJudgment:
        assert self.sdk.disconnected and not self.closed and self.storage_checked
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

    async def create_host_storage_async(self, *, directory: Path) -> None:
        """Create only inert test storage, without claiming production Windows ACL qualification."""
        await asyncio.to_thread(directory.mkdir, exist_ok=False)

    async def validate_host_storage_async(self, *, directory: Path) -> None:
        """Keep the inert fixture independent of production Windows ACL qualification."""
        assert await asyncio.to_thread(directory.is_dir)

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


@pytest.mark.parametrize(
    ("error_type", "status"),
    [(OSError, "error"), (TimeoutError, "expired"), (asyncio.CancelledError, "cancelled")],
)
async def test_raising_readiness_retains_unknown_provenance_and_undetermined_result_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, error_type: type[BaseException], status: str
) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding, "readiness_async", new_callable=AsyncMock, side_effect=error_type("readiness failed")):
        if error_type is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError, match="readiness failed"):
                await run.start_async()
        else:
            await run.start_async()
    assert run.view().status == status and not run.view().can_step
    assert binding.open_count == binding.close_count == binding.runtime.grade_count == 0
    assert binding.runtime.sdk.prompts == []
    assert run.report.readiness is None and run.report.simulated is None and run.report.agent is None
    assert run.report.cleanup == "not_opened"
    assert any("readiness failed" in item for item in run.report.errors)
    assert run.score.is_undetermined and run.score.score_metadata["simulated"] == "unknown"
    assert len(sqlite_instance.get_scores(score_type="float_scale")) == 1
    retained = await asyncio.to_thread((run.directory / f"{run.report.sha256()}.json").read_text, encoding="utf-8")
    assert retained == run.report.canonical_json()
    stored = sqlite_instance.get_scorable_content(content_ids=[run.score.scorable.content_id])
    assert NativeCyberReport.model_validate_json(stored[run.score.scorable.content_id].value) == run.report
    with pytest.raises(ValueError, match="only once"):
        await run.start_async()


async def test_mkdir_failure_retains_only_memory_content_and_never_claims_a_report_file_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    occupied = tmp_path / "not-a-directory"
    await asyncio.to_thread(occupied.write_text, "do not overwrite", encoding="utf-8")
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=occupied)
    with patch.object(binding, "readiness_async", new_callable=AsyncMock) as readiness:
        view = await run.start_async()
    readiness.assert_not_awaited()
    assert view.status == "error" and run.score.is_undetermined
    assert view.content_id and view.report_sha256
    assert run.report.readiness is None and run.report.simulated is None
    assert run.report.agent is None and run.report.cleanup == "not_opened" and run.report.errors
    assert binding.open_count == binding.close_count == 0
    assert not await asyncio.to_thread(run.directory.exists)
    assert await asyncio.to_thread(occupied.read_text, encoding="utf-8") == "do not overwrite"
    stored = sqlite_instance.get_scorable_content(content_ids=[run.score.scorable.content_id])
    assert NativeCyberReport.model_validate_json(stored[run.score.scorable.content_id].value) == run.report


async def test_unavailable_directory_and_memory_propagates_retention_failure_async(tmp_path: Path) -> None:
    occupied = tmp_path / "not-a-directory"
    await asyncio.to_thread(occupied.write_text, "fixture", encoding="utf-8")
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=occupied)
    with patch.object(
        NativeCyberReportScorer, "score_async", new_callable=AsyncMock, side_effect=OSError("memory unavailable")
    ):
        with pytest.raises(OSError, match="memory unavailable"):
            await run.start_async()
    assert run.status == "error" and run.report is None and run.score is None
    assert binding.open_count == binding.close_count == 0


async def test_report_file_failure_retains_an_explicit_memory_error_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = FixtureBinding(blocked=True)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch(
        "pyrit.executor.workflow.native_cyber_eval.aiofiles.open", side_effect=PermissionError("report read-only")
    ):
        view = await run.start_async()
    assert view.status == "error" and run.score.is_undetermined
    assert any("Report file retention failed" in error for error in run.report.errors)
    assert not await asyncio.to_thread((run.directory / f"{run.report.sha256()}.json").exists)
    stored = sqlite_instance.get_scorable_content(content_ids=[run.score.scorable.content_id])
    assert NativeCyberReport.model_validate_json(stored[run.score.scorable.content_id].value) == run.report


@pytest.mark.parametrize("simulated", [False, True])
async def test_unknown_readiness_cannot_be_relabelled_as_live_or_simulated_async(
    *, tmp_path: Path, simulated: bool
) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding, "readiness_async", new_callable=AsyncMock, side_effect=OSError("readiness failed")):
        await run.start_async()
    payload = run.report.model_dump()
    payload["simulated"] = simulated
    with pytest.raises(ValueError, match="unknown native provenance"):
        NativeCyberReport.model_validate(payload)


@pytest.mark.parametrize("guard", ["host", "agent"])
async def test_storage_qualification_failure_prevents_prompts_and_raw_report_files_async(
    *, tmp_path: Path, guard: str
) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    owner = binding if guard == "host" else binding.runtime
    with patch.object(
        owner,
        f"validate_{guard}_storage_async",
        new_callable=AsyncMock,
        side_effect=PermissionError("storage unverified"),
    ) as validation:
        result = await run.start_async()
    validation.assert_awaited_once()
    assert validation.call_args.kwargs["directory"] == run.directory
    assert result.status == "error" and run.score.is_undetermined
    assert any("storage unverified" in error for error in run.report.errors)
    assert binding.runtime.sdk.prompts == [] and binding.runtime.grade_count == 0
    assert binding.open_count == binding.close_count == (0 if guard == "host" else 1)
    assert not await asyncio.to_thread(run.directory.exists)


async def test_unverified_guest_and_failed_cleanup_never_publish_raw_content_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    class UnclosedBinding(FixtureBinding):
        @asynccontextmanager
        async def open_runtime(self, *, run_id: str, request: NativeCyberRequest) -> AsyncIterator[FixtureRuntime]:
            self.open_count += 1
            try:
                yield self.runtime
            finally:
                self.close_count += 1
                raise OSError("environment removal unconfirmed")

    binding = UnclosedBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(
        binding.runtime,
        "validate_agent_storage_async",
        new_callable=AsyncMock,
        side_effect=PermissionError("guest mount"),
    ):
        with pytest.raises(PermissionError, match="unverified agent storage boundary remains open"):
            await run.start_async()
    assert run.status == "error" and run.report is None and run.score is None
    assert binding.open_count == binding.close_count == 1
    assert binding.runtime.sdk.prompts == [] and binding.runtime.grade_count == 0
    assert sqlite_instance.get_scores(score_type="float_scale") == []
    assert await asyncio.to_thread(lambda: list(run.directory.iterdir())) == []


async def test_default_windows_storage_policy_requires_actual_acl_verification_async(tmp_path: Path) -> None:
    with patch("pyrit.executor.workflow.native_cyber_eval.os", spec=os) as platform:
        platform.name = "nt"
        with pytest.raises(PermissionError, match="verify host directory ACLs"):
            await NativeCyberTaskBinding.validate_host_storage_async(FixtureBinding(), directory=tmp_path)


async def test_default_windows_creator_rejects_before_any_directory_creation_async() -> None:
    directory = MagicMock(spec=Path)
    directory.parent = MagicMock(spec=Path)
    with patch("pyrit.executor.workflow.native_cyber_eval.os", spec=os) as platform:
        platform.name = "nt"
        with pytest.raises(PermissionError, match="verify host directory ACLs"):
            await NativeCyberTaskBinding.create_host_storage_async(FixtureBinding(), directory=directory)
    directory.mkdir.assert_not_called()
    directory.parent.lstat.assert_not_called()
    directory.parent.mkdir.assert_not_called()


@pytest.mark.parametrize("safe_parent", [True, False])
async def test_posix_creator_verifies_existing_parent_before_exclusive_0700_child_async(safe_parent: bool) -> None:
    directory = MagicMock(spec=Path)
    directory.parent = MagicMock(spec=Path)
    order: list[str] = []

    def parent_metadata() -> os.stat_result:
        order.append("verify_parent")
        return os.stat_result((stat.S_IFDIR | (0o700 if safe_parent else 0o755), 0, 0, 1, 1000, 1000, 0, 0, 0, 0))

    directory.parent.lstat.side_effect = parent_metadata
    directory.mkdir.side_effect = lambda **kwargs: order.append("create")
    with patch("pyrit.executor.workflow.native_cyber_eval.os", spec=os) as platform:
        platform.name = "posix"
        platform.getuid = MagicMock(return_value=1000)
        if safe_parent:
            await NativeCyberTaskBinding.create_host_storage_async(FixtureBinding(), directory=directory)
        else:
            with pytest.raises(PermissionError, match="private directories"):
                await NativeCyberTaskBinding.create_host_storage_async(FixtureBinding(), directory=directory)
    assert order == (["verify_parent", "create"] if safe_parent else ["verify_parent"])
    if safe_parent:
        directory.mkdir.assert_called_once()
        assert directory.mkdir.call_args.kwargs == {"mode": 0o700, "exist_ok": False}
    else:
        directory.mkdir.assert_not_called()
    directory.parent.mkdir.assert_not_called()
    directory.parent.chmod.assert_not_called()


@pytest.mark.parametrize("safe_parent", [True, False])
async def test_inert_windows_creator_guards_parent_before_default_mode_and_strict_validation_async(
    *, tmp_path: Path, safe_parent: bool
) -> None:
    binding = FixtureBinding(blocked=True)
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    order: list[str] = []

    async def verify_parent_async(*, directory: Path) -> None:
        assert directory == tmp_path and not await asyncio.to_thread(run.directory.exists)
        order.append("verify_parent")
        if not safe_parent:
            raise PermissionError("Unverified parent DACL or protected ancestry")

    async def create_async(*, directory: Path) -> None:
        await verify_parent_async(directory=directory.parent)
        order.append("create")
        await asyncio.to_thread(directory.mkdir, exist_ok=False)

    async def validate_async(*, directory: Path) -> None:
        assert directory == run.directory
        assert await asyncio.to_thread(lambda: list(directory.iterdir())) == []
        order.append("validate_child")

    original_readiness = binding.readiness_async

    async def readiness_async() -> NativeCyberReadiness:
        order.append("readiness")
        return await original_readiness()

    with (
        patch.object(binding, "create_host_storage_async", side_effect=create_async) as creator,
        patch.object(binding, "validate_host_storage_async", side_effect=validate_async) as validator,
        patch.object(binding, "readiness_async", side_effect=readiness_async),
        patch.object(Path, "mkdir", autospec=True, side_effect=Path.mkdir) as mkdir,
    ):
        result = await run.start_async()
    creator.assert_awaited_once()
    assert creator.call_args.kwargs == {"directory": run.directory}
    assert order == (["verify_parent", "create", "validate_child", "readiness"] if safe_parent else ["verify_parent"])
    child_calls = [call for call in mkdir.call_args_list if call.args[0] == run.directory]
    assert len(child_calls) == int(safe_parent)
    if safe_parent:
        assert child_calls[0].kwargs == {"exist_ok": False}
        validator.assert_awaited_once()
        assert result.status == "blocked"
        assert await asyncio.to_thread((run.directory / f"{run.report.sha256()}.json").is_file)
    else:
        validator.assert_not_awaited()
        assert result.status == "error" and run.report.readiness is None
        assert not await asyncio.to_thread(run.directory.exists)
    assert binding.open_count == binding.close_count == binding.runtime.grade_count == 0
    assert run.score.is_undetermined and binding.runtime.sdk.prompts == []


async def test_existing_run_directory_is_not_adopted_or_cleaned_up_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    await asyncio.to_thread(run.directory.mkdir)
    marker = run.directory / "existing.txt"
    await asyncio.to_thread(marker.write_text, "existing evidence", encoding="utf-8")
    with patch.object(binding, "validate_host_storage_async", new_callable=AsyncMock) as validator:
        result = await run.start_async()
    validator.assert_not_awaited()
    assert result.status == "error" and run.score.is_undetermined
    assert any("FileExistsError" in error for error in run.report.errors)
    assert await asyncio.to_thread(marker.read_text, encoding="utf-8") == "existing evidence"
    assert await asyncio.to_thread(lambda: list(run.directory.iterdir())) == [marker]
    assert binding.open_count == 0 and run.report.readiness is None


async def test_creation_settles_before_cancellation_cleanup_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    original_create = binding.create_host_storage_async

    async def create_async(*, directory: Path) -> None:
        await original_create(directory=directory)
        entered.set()
        await release.wait()

    with (
        patch.object(binding, "create_host_storage_async", side_effect=create_async) as creator,
        patch.object(binding, "validate_host_storage_async", new_callable=AsyncMock) as validator,
    ):
        task = asyncio.create_task(run.start_async())
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    creator.assert_awaited_once()
    validator.assert_not_awaited()
    assert run.status == "cancelled" and run.score.is_undetermined
    assert run.report.readiness is None and run.report.cleanup == "not_opened"
    assert not await asyncio.to_thread(run.directory.exists)
    assert binding.open_count == binding.close_count == 0


@pytest.mark.parametrize(
    ("mode", "owner", "accepted"),
    [
        (stat.S_IFDIR | 0o700, 1000, True),
        (stat.S_IFDIR | 0o755, 1000, False),
        (stat.S_IFDIR | 0o700, 2000, False),
        (stat.S_IFLNK | 0o700, 1000, False),
    ],
)
async def test_default_posix_storage_policy_checks_owner_permissions_and_directory_async(
    *, mode: int, owner: int, accepted: bool
) -> None:
    directory = MagicMock(spec=Path)
    directory.parent = MagicMock(spec=Path)
    metadata = os.stat_result((mode, 0, 0, 1, owner, owner, 0, 0, 0, 0))
    directory.lstat.return_value = directory.parent.lstat.return_value = metadata
    with patch("pyrit.executor.workflow.native_cyber_eval.os", spec=os) as platform:
        platform.name = "posix"
        platform.getuid = MagicMock(return_value=1000)
        if accepted:
            await NativeCyberTaskBinding.validate_host_storage_async(FixtureBinding(), directory=directory)
        else:
            with pytest.raises(PermissionError, match="private directories"):
                await NativeCyberTaskBinding.validate_host_storage_async(FixtureBinding(), directory=directory)


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


async def test_execution_before_request_never_reaches_original_grader_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    turn = binding.runtime.sdk.turns[0]
    turn[0], turn[1] = turn[1], turn[0]
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    result = await run.start_async()
    assert result.status == "error" and run.score.is_undetermined
    assert binding.runtime.grade_count == 0 and binding.close_count == 1
    assert run.report.agent.idle and not run.report.agent.coverage_complete
    assert any("started before its model tool request" in gap for gap in run.report.agent.gaps)


async def test_fresh_rerun_has_distinct_lineage_environment_and_content_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    first = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="First"), directory=tmp_path)
    await first.start_async()
    binding.runtime = FixtureRuntime(turns=3, steps=False, order=binding.order)
    second = first.rerun(request=NativeCyberRequest(instruction="Edited"))
    await second.start_async()
    assert second.report.request.parent_run_id == first.run_id
    assert first.run_id != second.run_id
    assert first.report.agent.environment_id != second.report.agent.environment_id
    assert first.report.agent.session_id != second.report.agent.session_id
    assert first.report.input_sha256 != second.report.input_sha256


async def test_direct_constructor_lineage_cannot_reuse_a_parent_environment_async(tmp_path: Path) -> None:
    binding = FixtureBinding()
    parent = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="first"), directory=tmp_path)
    await parent.start_async()
    with pytest.raises(ValueError, match="constructor lineage is unverified"):
        NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="second", parent_run_id=parent.run_id),
            directory=tmp_path,
        )
    assert binding.open_count == binding.close_count == 1
    assert binding.runtime.sdk.prompts == ["first"]


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
    result = await run.start_async()
    assert result.status == "error" and run.score.is_undetermined
    assert any("not qualified" in error for error in run.report.errors)
    assert run.report.cleanup == "not_opened" and run.report.agent is None


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


@pytest.mark.parametrize("identity", ["environment_id", "session_id"])
async def test_rerun_keeps_ancestry_through_an_unopened_child_async(*, tmp_path: Path, identity: str) -> None:
    binding = FixtureBinding()
    parent = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="first"), directory=tmp_path)
    await parent.start_async()
    binding.blocked = True
    child = parent.rerun(request=NativeCyberRequest(instruction="blocked"))
    await child.start_async()
    assert child.report.agent is None and child.status == "blocked"
    binding.blocked = False
    binding.runtime = FixtureRuntime(turns=3, steps=False, order=binding.order)
    setattr(binding.runtime.session, identity, getattr(parent.report.agent, identity))
    grandchild = child.rerun(request=NativeCyberRequest(instruction="third"))
    view = await grandchild.start_async()
    assert view.status == "error" and grandchild.score.is_undetermined
    assert any("reuse" in error for error in grandchild.report.errors)
    assert binding.runtime.sdk.prompts == [] and binding.runtime.grade_count == 0
    assert grandchild.request.parent_run_id == child.run_id


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

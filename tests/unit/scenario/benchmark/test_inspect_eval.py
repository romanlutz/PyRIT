# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-click benign Inspect Eval selection without Docker or model traffic."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.dataset import MemoryDataset
from sqlalchemy import func, select

from examples.inspect_eval_scenario_smoke import _count_unlabeled_control_scores
from examples.inspect_ghcp_protocol_smoke import SmokePins
from pyrit.executor.benchmark.inspect_ghcp_case_executor import InspectGhcpCaseExecutor
from pyrit.memory.memory_models import AttackResultEntry, ScoreEntry
from pyrit.models import (
    CommittedCaseExecution,
    EvalCaseRef,
    EvalRunRef,
    EvalScoreRole,
    EvalSourceKind,
    ScenarioRunPlan,
    Score,
)
from pyrit.registry.components.scenario_registry import ScenarioRegistry
from pyrit.scenario.scenarios.benchmark.inspect_eval import InspectEvalScenario
from pyrit.score.true_false.substring_scorer import SubStringScorer
from tests.unit.executor.benchmark.test_inspect_eval_source import _local_manifest, _multi_manifest
from tests.unit.executor.benchmark.test_inspect_ghcp_case_executor import _persist_original_case
from tests.unit.scenario.core.test_task_owned_scenario import _RecordingCaseExecutor

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory


_PILOT_ENV = {
    "PYRIT_INSPECT_AGENT_IMAGE": "pyrit-inspect-ghcp-guest:1.0.88-sdk1.0.14",
    "PYRIT_INSPECT_AGENT_IMAGE_ID": "sha256:" + "a" * 64,
    "PYRIT_INSPECT_TARGET_IMAGE": "pyrit-ghcp-agent:1.0.88-ca",
    "PYRIT_INSPECT_TARGET_IMAGE_ID": SmokePins.TARGET_IMAGE_ID,
}


@pytest.mark.usefixtures("patch_central_database")
def test_registry_discovers_one_click_task_owned_scenario_without_loading_inspect() -> None:
    registry = ScenarioRegistry()
    assert registry.get_class("benchmark.inspect_eval") is InspectEvalScenario
    metadata = registry._build_metadata("benchmark.inspect_eval", InspectEvalScenario)
    assert metadata.baseline_policy == "forbidden"
    assert metadata.default_techniques == ("retained_red_teaming",)
    params = {parameter.name: parameter for parameter in metadata.supported_parameters}
    assert "objective_target" not in params
    assert "dataset_config" not in params
    assert "eval_family" in params and "trusted_eval_dir" in params
    assert params["max_concurrency"].default == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_named_scenario_derives_one_real_task_case_without_host_contact(sqlite_instance: SQLiteMemory) -> None:
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args={"eval_family": "benign_protocol"})
    with patch.dict("os.environ", _PILOT_ENV):
        await scenario.initialize_async()
    assert scenario.atomic_attack_count == 1
    [work] = scenario._atomic_attacks
    assert isinstance(work._case_executor, InspectGhcpCaseExecutor)
    assert work.case.package.kind is EvalSourceKind.NAMED
    assert work.objective == work._case_executor._selected.task.dataset[0].input
    assert work.run.spec.package == work.case.package
    assert work.run.spec.harness.name == "ghcp_protocol_v1"
    assert work.run.spec.model_route.name == "qwen3_loopback_v1"
    assert work.run.spec.input_variant is None
    [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
    assert stored.scenario_identifier.objective_target is None
    assert stored.scenario_identifier.objective_scorer is None
    plan = ScenarioRunPlan.model_validate(stored.metadata["run_plan"])
    assert plan.seed_groups[0].case_id == work.case.case_id
    assert plan.seed_groups[0].source_sha256 == work.case.package.source_sha256


@pytest.mark.usefixtures("patch_central_database")
async def test_local_explicit_trust_and_input_variant_do_not_leak_directory_into_identifier(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    revision = _local_manifest(root=tmp_path, allow_edit=True)
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(
        args={
            "trusted_eval_dir": tmp_path,
            "trust_local": True,
            "eval_revision": revision,
            "initial_user_input": "Edited harmless test instruction",
        }
    )
    with patch.dict("os.environ", _PILOT_ENV):
        await scenario.initialize_async()
    [work] = scenario._atomic_attacks
    assert work.case.package.kind is EvalSourceKind.TRUSTED_LOCAL
    assert work.run.spec.input_variant is not None
    assert work.run.spec.input_variant.case_id == work.case.case_id
    assert work.run.spec.input_variant.content_sha256 == hashlib.sha256(work.objective.encode()).hexdigest()
    assert work.objective == "Edited harmless test instruction"
    assert work._case_executor._selected.original_input_sha256 == hashlib.sha256(b"local harmless goal").hexdigest()
    [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
    assert str(tmp_path) not in stored.scenario_identifier.model_dump_json()
    assert stored.scenario_identifier.params["source_sha256"] == revision
    assert stored.scenario_identifier.params["input_variant_sha256"] == work.run.spec.input_variant.content_sha256


@pytest.mark.usefixtures("patch_central_database")
async def test_local_two_task_three_sample_sweep_links_independent_original_scores_for_identical_text(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    revision = await asyncio.to_thread(_multi_manifest, root=tmp_path)
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args={"trusted_eval_dir": tmp_path, "trust_local": True, "eval_revision": revision})
    with patch.dict("os.environ", _PILOT_ENV):
        await scenario.initialize_async()

    assert scenario.atomic_attack_count == 3
    assert scenario._max_concurrency == 1
    work = scenario._atomic_attacks
    assert [case.case.sample_id for case in work] == ["local-one", "local-two-a", "local-two-b"]
    assert {case.objective for case in work} == {"local harmless goal"}
    assert len({case.case.case_id for case in work}) == 3
    assert len({case.case_run_id for case in work}) == 3
    assert work[1]._case_executor._selected.task is work[2]._case_executor._selected.task
    assert all(case.run.spec.package.source_sha256 == revision for case in work)
    [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
    plan = ScenarioRunPlan.model_validate(stored.metadata["run_plan"])
    assert [group.case_id for group in plan.seed_groups] == [case.case.case_id for case in work]
    assert len({group.objective_sha256 for group in plan.seed_groups}) == 1
    assert str(tmp_path) not in stored.scenario_identifier.model_dump_json()
    assert stored.scenario_identifier.objective_target is None
    assert stored.scenario_identifier.objective_scorer is None

    # The unit-only executor supplies the already-committed Score contract; it does not run a Task or model.
    recorder = _RecordingCaseExecutor(memory=sqlite_instance)

    async def committed_case_async(
        self: InspectGhcpCaseExecutor, *, case: EvalCaseRef, run: EvalRunRef
    ) -> CommittedCaseExecution:
        assert self._selected.case == case
        return await recorder.execute_case_async(case=case, run=run)

    with patch.object(InspectGhcpCaseExecutor, "execute_case_async", new=committed_case_async):
        result = await scenario.run_async()
        with pytest.raises(RuntimeError, match="replay is disabled"):
            await scenario.run_async()
        reused = await work[1].run_async()
        assert reused.completed_results[0].automated_score.id == recorder.original_scores[1].id

    rows = result.get_display_groups()[work[0].display_group]
    assert len(rows) == len(recorder.calls) == 3
    assert recorder.calls == [case.case_run_id for case in work]
    assert len({row.attack_result_id for row in rows}) == 3
    assert len({row.automated_score.id for row in rows}) == 3
    assert {row.attribution_data["case_id"] for row in rows} == {case.case.case_id for case in work}
    assert {row.attribution_data["seed_group_id"] for row in rows} == {case.case_run_id for case in work}
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 3
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 3
        for row in rows:
            stored_row = session.get(AttackResultEntry, row.attack_result_id)
            assert stored_row is not None and stored_row.automated_score_id == row.automated_score.id


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("change", ["asset", "unselected_sample", "sample_sandbox"])
async def test_multi_case_preflight_rejects_drift_before_any_original_score(
    tmp_path: Path, sqlite_instance: SQLiteMemory, change: str
) -> None:
    revision = await asyncio.to_thread(_multi_manifest, root=tmp_path)
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args={"trusted_eval_dir": tmp_path, "trust_local": True, "eval_revision": revision})
    with patch.dict("os.environ", _PILOT_ENV):
        await scenario.initialize_async()

    selected = scenario._atomic_attacks[2]._case_executor._selected
    if change == "asset":
        await asyncio.to_thread((tmp_path / "public-asset.txt").write_text, "changed", encoding="utf-8")
    elif change == "unselected_sample":
        selected.task.dataset = MemoryDataset(
            samples=[
                selected.task.dataset[0].model_copy(update={"input": "different unselected prompt"}),
                selected.task.dataset[1],
            ]
        )
    else:
        selected.task.dataset = MemoryDataset(
            samples=[
                selected.task.dataset[0],
                selected.task.dataset[1].model_copy(update={"sandbox": selected.task.sandbox}),
            ]
        )
    with patch.object(InspectGhcpCaseExecutor, "execute_case_async", new_callable=AsyncMock) as execute:
        with pytest.raises(ValueError, match="source asset changed|identity or supported input|per-Sample sandbox"):
            await scenario.run_async()
        execute.assert_not_awaited()
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_one_click_scenario_links_only_the_existing_original_undetermined_score(
    sqlite_instance: SQLiteMemory,
) -> None:
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args={"eval_family": "benign_protocol"})
    with patch.dict("os.environ", _PILOT_ENV):
        await scenario.initialize_async()
    [work] = scenario._atomic_attacks
    executor = work._case_executor
    assert isinstance(executor, InspectGhcpCaseExecutor)
    outcome = _persist_original_case(memory=sqlite_instance, executor=executor, run=work.run)
    original = executor._verify_original_score(case=work.case, run=work.run, outcome=outcome)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
    with patch.object(
        InspectGhcpCaseExecutor, "execute_case_async", new_callable=AsyncMock, return_value=original
    ) as run:
        result = await scenario.run_async()
        run.assert_awaited_once()
        with pytest.raises(RuntimeError, match="replay is disabled"):
            await scenario.run_async()
        reused = await work.run_async()
        assert reused.completed_results[0].automated_score.id == original.original_score_id
        run.assert_awaited_once()
    [linked] = result.get_display_groups()[work.display_group]
    assert linked.automated_score.id == original.original_score_id
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        row = session.get(AttackResultEntry, linked.attack_result_id)
        assert row is not None and row.automated_score_id == original.original_score_id


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("mislabel", [False, True])
async def test_safe_example_classifies_two_unlabeled_canary_scores_without_claiming_progress(
    sqlite_instance: SQLiteMemory, mislabel: bool
) -> None:
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args={"eval_family": "benign_protocol"})
    with patch.dict("os.environ", _PILOT_ENV):
        await scenario.initialize_async()
    [work] = scenario._atomic_attacks
    assert isinstance(work._case_executor, InspectGhcpCaseExecutor)
    outcome = _persist_original_case(memory=sqlite_instance, executor=work._case_executor, run=work.run)
    with pytest.raises(RuntimeError, match="one original or two control"):
        _count_unlabeled_control_scores(memory=sqlite_instance, episode=outcome.episode)
    scorer = SubStringScorer(substring="__impossible_scoring_canary__")
    controls = [
        Score(
            score_type="true_false",
            score_value="false",
            scorer_class_identifier=scorer.get_identifier(),
            message_piece_id=turn.response_piece_ids[0],
            score_metadata={"pyrit_eval_role": EvalScoreRole.PROGRESS_SIGNAL.value}
            if mislabel and turn.turn_index == 1
            else {},
        )
        for turn in outcome.episode.turns
    ]
    sqlite_instance.add_scores_to_memory(scores=controls)
    if mislabel:
        with pytest.raises(RuntimeError, match="unlabeled, false"):
            _count_unlabeled_control_scores(memory=sqlite_instance, episode=outcome.episode)
    else:
        assert _count_unlabeled_control_scores(memory=sqlite_instance, episode=outcome.episode) == 2
        with sqlite_instance.get_session() as session:
            assert session.scalar(select(func.count(ScoreEntry.id))) == 3


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "exactly one named Eval family"),
        ({"eval_family": "unknown"}, "Unknown or unqualified"),
        ({"eval_family": "benign_protocol", "harness_profile": "unqualified"}, "Unknown or unqualified"),
        ({"eval_family": "benign_protocol", "model_route": "unqualified"}, "Unknown or unqualified"),
        ({"eval_family": "benign_protocol", "max_concurrency": 2}, "max_concurrency=1"),
        ({"eval_family": "benign_protocol", "max_retries": 1}, "max_retries must be 0"),
        ({"eval_family": "benign_protocol", "include_baseline": True}, "does not support a default baseline"),
    ],
)
async def test_unknown_or_unsupported_scenario_capability_is_rejected(params: dict[str, object], message: str) -> None:
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args=params)
    with patch.dict("os.environ", _PILOT_ENV):
        with pytest.raises(ValueError, match=message):
            await scenario.initialize_async()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Contract and persistence tests for the task-owned Scenario opt-in."""

import asyncio
import uuid
from contextlib import closing
from threading import Event
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import func, select, text

from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.common.apply_defaults import apply_defaults
from pyrit.memory import SQLiteMemory
from pyrit.memory.memory_models import AttackResultEntry, ScoreEntry
from pyrit.models import (
    AttackOutcome,
    CommittedCaseExecution,
    ComponentIdentifier,
    EvalCaseRef,
    EvalPackageRef,
    EvalRunRef,
    EvalScoreProvenance,
    EvalScoreRole,
    EvalSourceKind,
    EvalSpecRef,
    HarnessProfileRef,
    InputVariantRef,
    ModelRouteRef,
    Parameter,
    ScenarioEvaluationIdentifier,
    ScenarioRunPlan,
    ScenarioRunSizeEstimateStatus,
    Score,
)
from pyrit.registry.components.scenario_registry import ScenarioRegistry
from pyrit.scenario.core import (
    AtomicAttack,
    AtomicWork,
    BaselineAttackPolicy,
    DatasetAttackConfiguration,
    Scenario,
    ScenarioTechnique,
    TaskOwnedAtomicAttack,
    TaskOwnedCaseReplayBlockedError,
    TaskOwnedResultPersistenceError,
    TaskOwnedScenario,
    TaskOwnedScenarioContext,
)
from pyrit.scenario.core.scenario_context import ScenarioContext


def _package() -> EvalPackageRef:
    return EvalPackageRef(kind=EvalSourceKind.NAMED, name="public_suite", source_sha256="a" * 64)


def _case(*, task: str = "task_one", sample: str = "sample_one") -> EvalCaseRef:
    return EvalCaseRef(
        package=_package(),
        task_name=task,
        task_version="v1",
        sample_id=sample,
        epoch=0,
    )


def _spec(*, variant: InputVariantRef | None = None) -> EvalSpecRef:
    return EvalSpecRef(
        package=_package(),
        harness=HarnessProfileRef(name="standard_harness", config_sha256="b" * 64),
        model_route=ModelRouteRef(name="attacker", config_sha256="c" * 64),
        input_variant=variant,
    )


def _score_count(*, memory: SQLiteMemory) -> int:
    with closing(memory.get_session()) as session:
        return int(session.scalar(select(func.count(ScoreEntry.id))) or 0)


class _RecordingCaseExecutor:
    def __init__(
        self,
        *,
        memory: SQLiteMemory,
        role: EvalScoreRole = EvalScoreRole.BENCHMARK_ORIGINAL,
        labeled: bool = True,
        include_progress: bool = False,
        persist_original: bool = True,
        pyrit_scorer_hash_override: str | None = None,
    ) -> None:
        self._memory = memory
        self._role = role
        self._labeled = labeled
        self._include_progress = include_progress
        self._persist_original = persist_original
        self._pyrit_scorer_hash_override = pyrit_scorer_hash_override
        self._score_write_lock = asyncio.Lock()
        self.calls: list[str] = []
        self.original_scores: list[Score] = []
        self.progress_scores: list[Score] = []

    async def execute_case_async(self, *, case: EvalCaseRef, run: EvalRunRef) -> CommittedCaseExecution:
        self.calls.append(run.case_run_id(case=case))
        scorer = ComponentIdentifier(class_name="OriginalBenchmarkScorer", class_module=__name__)
        provenance = EvalScoreProvenance(
            role=self._role,
            case_run_id=run.case_run_id(case=case),
            pyrit_scorer_hash=self._pyrit_scorer_hash_override or scorer.hash,
        )
        original = Score(
            score_type="true_false",
            score_value="true",
            scorer_class_identifier=scorer,
            score_metadata=provenance.to_metadata() if self._labeled else {},
        )
        self.original_scores.append(original)
        scores_to_persist: list[Score] = []
        if self._include_progress:
            progress_provenance = provenance.model_copy(update={"role": EvalScoreRole.PROGRESS_SIGNAL})
            progress = Score(
                score_type="true_false",
                score_value="false",
                scorer_class_identifier=scorer,
                score_metadata=progress_provenance.to_metadata(),
            )
            scores_to_persist.append(progress)
            self.progress_scores.append(progress)
        if self._persist_original:
            scores_to_persist.append(original)
        if scores_to_persist:
            async with self._score_write_lock:
                await asyncio.to_thread(self._memory.add_scores_to_memory, scores=scores_to_persist)
        return CommittedCaseExecution(
            original_score_id=original.id,
            original_score_provenance=provenance,
            conversation_id=uuid.uuid4(),
            outcome=AttackOutcome.SUCCESS,
            executed_turns=2,
        )


class _EvalTechnique(ScenarioTechnique):
    ALL = ("all", {"all"})
    CASE = ("case", {"case"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        return {"all"}


class _TestEvalScenario(TaskOwnedScenario):
    VERSION = 1

    @apply_defaults
    def __init__(self, *, scenario_result_id: uuid.UUID | str | None = None) -> None:
        super().__init__(
            version=self.VERSION,
            technique_class=_EvalTechnique,
            default_dataset_config=DatasetAttackConfiguration(),
            objective_scorer=None,
            scenario_result_id=scenario_result_id,
        )

    @classmethod
    def additional_parameters(cls) -> list[Parameter]:
        return [
            Parameter(name="cases", description="Selected Eval cases.", opaque=True),
            Parameter(name="spec", description="Trusted Eval specification.", opaque=True),
            Parameter(name="case_executor", description="Runtime case executor.", opaque=True),
        ]

    async def _build_task_owned_atomic_attacks_async(
        self, *, context: TaskOwnedScenarioContext
    ) -> list[TaskOwnedAtomicAttack]:
        cases = self.params["cases"]
        spec = self.params["spec"]
        executor = self.params["case_executor"]
        if not isinstance(cases, list) or not all(isinstance(case, EvalCaseRef) for case in cases):
            raise ValueError("Unknown or invalid Eval case selection")
        if not isinstance(spec, EvalSpecRef):
            raise ValueError("Unknown or invalid Eval source/harness profile")
        if not isinstance(executor, _RecordingCaseExecutor):
            raise ValueError("Unknown or invalid case executor")
        run = EvalRunRef(spec=spec, run_instance_id=context.run_instance_id)
        return [
            TaskOwnedAtomicAttack(
                case=case,
                run=run,
                objective="shared objective",
                case_executor=executor,
                memory_labels=context.memory_labels,
            )
            for case in cases
        ]


class _LegacyTargetScenario(Scenario):
    async def _build_atomic_attacks_async(self, *, context: ScenarioContext) -> list[AtomicAttack]:
        return []


def _configured_scenario(
    *,
    cases: list[EvalCaseRef],
    executor: _RecordingCaseExecutor,
    spec: EvalSpecRef | None = None,
    scenario_result_id: str | None = None,
    extra_params: dict[str, object] | None = None,
) -> _TestEvalScenario:
    scenario = _TestEvalScenario(scenario_result_id=scenario_result_id)
    scenario.set_params_from_args(
        args={
            "cases": cases,
            "spec": spec or _spec(),
            "case_executor": executor,
            "memory_labels": {"run": "task_owned"},
            **(extra_params or {}),
        }
    )
    return scenario


@pytest.mark.usefixtures("patch_central_database")
class TestTaskOwnedScenario:
    async def test_default_size_estimate_does_not_resolve_or_run_source_async(self) -> None:
        with patch.object(_TestEvalScenario, "_build_task_owned_atomic_attacks_async", new_callable=AsyncMock) as build:
            estimate = await _TestEvalScenario().get_default_run_size_estimate_async()

        assert estimate.status is ScenarioRunSizeEstimateStatus.Unavailable
        assert estimate.estimated_attack_count is None
        assert estimate.note == "Select an Eval source to determine its Task/Sample count."
        build.assert_not_awaited()

    def test_registry_can_introspect_no_argument_task_owned_scenario(self) -> None:
        metadata = ScenarioRegistry()._build_metadata("task_owned_test", _TestEvalScenario)

        assert metadata.baseline_policy == BaselineAttackPolicy.Forbidden.value
        assert metadata.default_techniques == ("case",)
        names = {parameter.name for parameter in metadata.supported_parameters}
        assert "objective_target" not in names
        assert "dataset_config" not in names
        assert "cases" in names

    def test_legacy_scenario_still_requires_real_scorer(self) -> None:
        with pytest.raises(ValueError, match="objective_scorer is required"):
            _LegacyTargetScenario(
                version=1,
                technique_class=_EvalTechnique,
                default_dataset_config=DatasetAttackConfiguration(),
                objective_scorer=None,
            )

    async def test_registry_initializes_task_owned_scenario_without_external_target(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        registry = ScenarioRegistry()
        scenario = _TestEvalScenario()
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        with patch.object(registry, "create_instance", return_value=scenario):
            initialized = await registry.create_and_initialize_async(
                "example_eval",
                scenario_params={"cases": [_case()], "spec": _spec(), "case_executor": executor},
            )

        assert initialized is scenario
        assert scenario._objective_target is None
        assert scenario._scenario_result_id is not None
        assert (await scenario.run_async()).get_display_groups()["task_owned"][0].automated_score is not None

    async def test_initialize_without_global_target_or_scorer_keeps_case_ids_distinct(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        cases = [_case(task="first"), _case(task="second")]
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=cases, executor=executor)

        await scenario.initialize_async()
        assert scenario._objective_target is None
        assert scenario._objective_scorer_identifier is None
        assert all(isinstance(work, AtomicWork) for work in scenario._atomic_attacks)
        [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
        plan = ScenarioRunPlan.model_validate(stored.metadata["run_plan"])
        assert len(plan.seed_groups) == 2
        assert len({group.id for group in plan.seed_groups}) == 2
        assert {group.objective_sha256 for group in plan.seed_groups} == {plan.seed_groups[0].objective_sha256}
        assert {group.case_id for group in plan.seed_groups} == {case.case_id for case in cases}
        assert stored.metadata["eval_spec_sha256"] == plan.eval_spec_sha256
        assert stored.metadata["run_instance_id"] == str(plan.run_instance_id)
        assert stored.scenario_identifier.objective_target is None
        assert stored.scenario_identifier.objective_scorer is None
        assert "run_instance_id" not in stored.scenario_identifier.params
        assert "case_executor" not in stored.scenario_identifier.params
        with closing(sqlite_instance.get_session()) as session:
            raw_target, storage_type = session.execute(
                text(
                    "SELECT objective_target_identifier, typeof(objective_target_identifier) "
                    'FROM "ScenarioResultEntries" WHERE id = :result_id'
                ),
                {"result_id": scenario._scenario_result_id},
            ).one()
            schema = session.execute(text('PRAGMA table_info("ScenarioResultEntries")')).all()
        assert raw_target == "null"
        assert storage_type == "text"
        assert next(row for row in schema if row[1] == "objective_target_identifier")[3] == 1

        result = await scenario.run_async()
        assert sum(map(len, result.attack_results.values())) == 2
        assert _score_count(memory=sqlite_instance) == 2
        assert (
            len({row.attribution_data["seed_group_id"] for rows in result.attack_results.values() for row in rows}) == 2
        )
        page, aggregates, _ = sqlite_instance.get_scenario_run_history_page(limit=10)
        assert any(record.scenario_result_id == scenario._scenario_result_id for record in page)
        assert aggregates[scenario._scenario_result_id].completed_units == 2
        assert aggregates[scenario._scenario_result_id].successful_units == 2
        service = ScenarioRunService()
        try:
            summary = service.get_run_from_storage(scenario_result_id=scenario._scenario_result_id, active_error=None)
            progress = service.get_run_progress_from_storage(
                scenario_result_id=scenario._scenario_result_id,
                since=None,
                limit=10,
                active_group_ids=(),
            )
            assert summary is not None and summary.target is None
            assert progress is not None and progress.run.target is None
            assert progress.summary.overall.planned == 2
            assert progress.summary.overall.completed == 2
            assert len({group.id for group in progress.summary.seed_groups}) == 2
        finally:
            service._prepare_executor.shutdown(wait=True)

    async def test_private_case_coordinates_are_not_in_scenario_identifier(self, sqlite_instance: SQLiteMemory) -> None:
        case = _case(task=r"C:\synthetic\private_case")
        scenario = _configured_scenario(cases=[case], executor=_RecordingCaseExecutor(memory=sqlite_instance))
        await scenario.initialize_async()

        [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
        assert r"C:\synthetic\private_case" not in stored.scenario_identifier.model_dump_json()
        assert stored.metadata["run_plan"]["seed_groups"][0]["case_id"] == case.case_id

    async def test_two_fresh_identical_runs_do_not_reuse_results_or_scores(self, sqlite_instance: SQLiteMemory) -> None:
        case = _case()
        spec = _spec()
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        first = _configured_scenario(cases=[case], spec=spec, executor=executor)
        second = _configured_scenario(cases=[case], spec=spec, executor=executor)

        await first.initialize_async()
        await second.initialize_async()
        [first_header] = sqlite_instance.get_scenario_results(scenario_result_ids=[first._scenario_result_id])
        [second_header] = sqlite_instance.get_scenario_results(scenario_result_ids=[second._scenario_result_id])
        assert first_header.metadata["run_instance_id"] != second_header.metadata["run_instance_id"]
        assert first_header.metadata["eval_spec_sha256"] == second_header.metadata["eval_spec_sha256"]
        assert (
            ScenarioEvaluationIdentifier(first_header.scenario_identifier).eval_hash
            == ScenarioEvaluationIdentifier(second_header.scenario_identifier).eval_hash
        )
        first_result = (await first.run_async()).get_display_groups()["task_owned"][0]
        second_result = (await second.run_async()).get_display_groups()["task_owned"][0]
        assert first_result.attribution_data["seed_group_id"] != second_result.attribution_data["seed_group_id"]
        assert first_result.attack_result_id != second_result.attack_result_id
        assert first_result.automated_score.id != second_result.automated_score.id
        assert _score_count(memory=sqlite_instance) == 2
        assert len(executor.calls) == 2

    async def test_concurrent_task_sample_sweep_retains_each_original_score(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        cases = [_case(task=f"task_{index}") for index in range(16)]
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=cases, executor=executor, extra_params={"max_concurrency": 4})
        await scenario.initialize_async()

        results = (await scenario.run_async()).get_display_groups()["task_owned"]
        assert len(results) == len(cases)
        assert len({result.attribution_data["case_id"] for result in results}) == len(cases)
        assert len({str(result.automated_score.id) for result in results}) == len(cases)
        assert _score_count(memory=sqlite_instance) == len(cases)

    async def test_single_case_overlay_changes_spec_not_source_identity(self, sqlite_instance: SQLiteMemory) -> None:
        case = _case()
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        original = _configured_scenario(cases=[case], executor=executor)
        variant = InputVariantRef(case_id=case.case_id, surface_id="user_input", content_sha256="d" * 64)
        edited = _configured_scenario(cases=[case], executor=executor, spec=_spec(variant=variant))

        await original.initialize_async()
        await edited.initialize_async()
        original_plan = original._build_run_plan()
        edited_plan = edited._build_run_plan()
        assert original_plan.seed_groups[0].case_id == edited_plan.seed_groups[0].case_id == case.case_id
        assert original_plan.seed_groups[0].source_sha256 == edited_plan.seed_groups[0].source_sha256
        assert original_plan.eval_spec_sha256 != edited_plan.eval_spec_sha256
        assert original_plan.seed_groups[0].id != edited_plan.seed_groups[0].id
        assert edited_plan.seed_groups[0].input_variant_sha256 == variant.content_sha256

    async def test_overlay_cannot_sweep_multiple_cases_or_mismatched_case(self, sqlite_instance: SQLiteMemory) -> None:
        case = _case()
        variant = InputVariantRef(case_id=case.case_id, surface_id="user_input", content_sha256="d" * 64)
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        duplicate = _configured_scenario(cases=[case, case], executor=executor, spec=_spec(variant=variant))
        with pytest.raises(ValueError, match="exactly one Eval case"):
            await duplicate.initialize_async()

        mismatched = _configured_scenario(
            cases=[_case(sample="different")], executor=executor, spec=_spec(variant=variant)
        )
        with pytest.raises(ValueError, match="targets a different Eval case"):
            await mismatched.initialize_async()
        assert not executor.calls

    async def test_retry_resume_and_repeated_run_are_blocked(self, sqlite_instance: SQLiteMemory) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        invalid_concurrency = _configured_scenario(
            cases=[_case()], executor=executor, extra_params={"max_concurrency": 0}
        )
        with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
            await invalid_concurrency.initialize_async()

        retry = _configured_scenario(cases=[_case()], executor=executor, extra_params={"max_retries": 1})
        with pytest.raises(ValueError, match="max_retries must be 0"):
            await retry.initialize_async()

        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()
        await scenario.run_async()
        with pytest.raises(RuntimeError, match="replay is disabled"):
            await scenario.run_async()
        resumed = _configured_scenario(
            cases=[_case()], executor=executor, scenario_result_id=scenario._scenario_result_id
        )
        with pytest.raises(ValueError, match="resume is disabled"):
            await resumed.initialize_async()
        assert len(executor.calls) == 1

    async def test_pre_initialization_error_does_not_mark_an_unstarted_case_executed(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        with pytest.raises(ValueError, match="Cannot run scenario with no atomic attacks"):
            await scenario.run_async()

        await scenario.initialize_async()
        await scenario.run_async()
        assert len(executor.calls) == 1

    def test_rejects_caller_metadata_replacing_run_identity(self, sqlite_instance: SQLiteMemory) -> None:
        scenario = _configured_scenario(cases=[_case()], executor=_RecordingCaseExecutor(memory=sqlite_instance))
        with pytest.raises(ValueError, match="cannot override core identity"):
            scenario.set_initial_metadata(metadata={"run_instance_id": str(uuid.uuid4())})

    async def test_one_case_links_one_existing_score_without_rescoring(self, sqlite_instance: SQLiteMemory) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()

        result = (await scenario.run_async()).get_display_groups()["task_owned"][0]
        assert len(executor.calls) == 1
        assert _score_count(memory=sqlite_instance) == 1
        assert result.automated_score.id == executor.original_scores[0].id
        assert result.attribution_data["score_ids"] == {
            EvalScoreRole.BENCHMARK_ORIGINAL.value: str(executor.original_scores[0].id)
        }
        with closing(sqlite_instance.get_session()) as session:
            entry = session.get(AttackResultEntry, uuid.UUID(result.attack_result_id))
            assert entry is not None
            assert entry.automated_score_id == uuid.UUID(str(executor.original_scores[0].id))

    async def test_score_link_is_exactly_original_even_with_progress_scores(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance, include_progress=True)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()

        result = (await scenario.run_async()).get_display_groups()["task_owned"][0]
        assert _score_count(memory=sqlite_instance) == 2
        assert result.automated_score.id == executor.original_scores[0].id
        assert result.automated_score.id != executor.progress_scores[0].id
        assert result.attribution_data["score_ids"] == {
            EvalScoreRole.BENCHMARK_ORIGINAL.value: str(executor.original_scores[0].id)
        }
        assert (
            sqlite_instance.get_scores(score_ids=[str(result.automated_score.id)])[0].score_metadata["pyrit_eval_role"]
            == EvalScoreRole.BENCHMARK_ORIGINAL.value
        )

    async def test_unlabeled_or_progress_only_score_cannot_be_original(self, sqlite_instance: SQLiteMemory) -> None:
        unlabeled = _RecordingCaseExecutor(memory=sqlite_instance, labeled=False)
        scenario = _configured_scenario(cases=[_case()], executor=unlabeled)
        await scenario.initialize_async()
        with pytest.raises(ValueError, match="missing task-owned evaluation provenance"):
            await scenario.run_async()
        assert not sqlite_instance.get_attack_results(scenario_result_id=scenario._scenario_result_id)

        progress = _RecordingCaseExecutor(memory=sqlite_instance, role=EvalScoreRole.PROGRESS_SIGNAL)
        scenario2 = _configured_scenario(cases=[_case()], executor=progress)
        await scenario2.initialize_async()
        with pytest.raises(ValueError, match="without matching benchmark-original PyRIT provenance"):
            await scenario2.run_async()
        assert not sqlite_instance.get_attack_results(scenario_result_id=scenario2._scenario_result_id)

    async def test_pyrit_score_creator_hash_must_match_committed_entry(self, sqlite_instance: SQLiteMemory) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance, pyrit_scorer_hash_override="f" * 64)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()

        with pytest.raises(ValueError, match="matching benchmark-original PyRIT provenance"):
            await scenario.run_async()
        assert _score_count(memory=sqlite_instance) == 1
        assert not sqlite_instance.get_attack_results(scenario_result_id=scenario._scenario_result_id)

    async def test_original_score_must_already_be_committed(self, sqlite_instance: SQLiteMemory) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance, persist_original=False)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()
        with pytest.raises(ValueError, match="not committed in memory"):
            await scenario.run_async()
        assert _score_count(memory=sqlite_instance) == 0
        with pytest.raises(TaskOwnedCaseReplayBlockedError, match="automatic execution is blocked"):
            await scenario._atomic_attacks[0].run_async()
        assert len(executor.calls) == 1

    async def test_failed_attack_result_write_never_reruns_committed_score(self, sqlite_instance: SQLiteMemory) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()
        with patch.object(sqlite_instance, "add_attack_results_to_memory", side_effect=OSError("storage unavailable")):
            with pytest.raises(TaskOwnedResultPersistenceError, match="do not execute the case again"):
                await scenario.run_async()

        assert _score_count(memory=sqlite_instance) == 1
        assert not sqlite_instance.get_attack_results(scenario_result_id=scenario._scenario_result_id)
        with pytest.raises(TaskOwnedCaseReplayBlockedError, match="automatic execution is blocked"):
            await scenario._atomic_attacks[0].run_async()
        with pytest.raises(RuntimeError, match="replay is disabled"):
            await scenario.run_async()
        assert len(executor.calls) == 1

    async def test_ambiguous_committed_result_is_read_without_second_execution(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()
        original_add = sqlite_instance.add_attack_results_to_memory

        def _commit_then_fail(*, attack_results: list) -> None:
            original_add(attack_results=attack_results)
            raise OSError("commit acknowledgement lost")

        with patch.object(sqlite_instance, "add_attack_results_to_memory", side_effect=_commit_then_fail):
            with pytest.raises(TaskOwnedResultPersistenceError, match="AttackResult write failed"):
                await scenario.run_async()

        existing = sqlite_instance.get_attack_results(scenario_result_id=scenario._scenario_result_id)
        assert len(existing) == 1
        recovered = await scenario._atomic_attacks[0].run_async()
        assert recovered.completed_results[0].attack_result_id == existing[0].attack_result_id
        assert recovered.completed_results[0].automated_score.id == executor.original_scores[0].id
        assert _score_count(memory=sqlite_instance) == 1
        assert len(executor.calls) == 1

    async def test_cancellation_waits_for_post_score_write_without_reexecution(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        executor = _RecordingCaseExecutor(memory=sqlite_instance)
        scenario = _configured_scenario(cases=[_case()], executor=executor)
        await scenario.initialize_async()
        started = Event()
        release = Event()
        original_add = sqlite_instance.add_attack_results_to_memory

        def _delayed_write(*, attack_results: list) -> None:
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("timed out waiting for the delayed write")
            original_add(attack_results=attack_results)

        with patch.object(sqlite_instance, "add_attack_results_to_memory", side_effect=_delayed_write):
            run = asyncio.create_task(scenario.run_async())
            try:
                assert await asyncio.to_thread(started.wait, 3)
                run.cancel()
                await asyncio.sleep(0)
                assert not run.done()
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await run

        assert len(sqlite_instance.get_attack_results(scenario_result_id=scenario._scenario_result_id)) == 1
        assert _score_count(memory=sqlite_instance) == 1
        with pytest.raises(RuntimeError, match="replay is disabled"):
            await scenario.run_async()
        assert len(executor.calls) == 1

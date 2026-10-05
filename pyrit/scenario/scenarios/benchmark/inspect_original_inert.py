# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-click original Inspect Task execution and source-bound score projection for a public fixture."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyrit.common import apply_defaults
from pyrit.executor.attack import AttackExecutorResult
from pyrit.memory import SQLiteMemory
from pyrit.models import (
    AttackOutcome,
    EvalRunRef,
    Parameter,
    ScenarioRunSizeComponent,
    ScenarioRunSizeEstimate,
    ScoreStatus,
    config_hash,
)
from pyrit.models.catalog.scenario import OriginalInspectImportSummary, OriginalInspectTaskId
from pyrit.scenario.core import DatasetAttackConfiguration, ScenarioTechnique, TaskOwnedScenario

if TYPE_CHECKING:
    from pyrit.executor.attack import AttackExecutor
    from pyrit.executor.benchmark.inspect_eval_source import ResolvedOriginalInspectTask
    from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalCaseResult, InspectOriginalImport
    from pyrit.executor.benchmark.inspect_original_runner import InspectOriginalRun
    from pyrit.models import AttackResult, BoundedDatasetSize
    from pyrit.models.catalog.scenario import RunScenarioRequest
    from pyrit.scenario.core.scenario_context import TaskOwnedScenarioContext

logger = logging.getLogger(__name__)


def _allocate_log_dir(*, run_instance_id: uuid.UUID) -> Path:
    """
    Create a private, unique local directory; callers cannot select a source or log path.

    Returns:
        Path: A fresh local Inspect `.eval` directory.

    Raises:
        ValueError: If the configured system temp directory is a symlink or network share.
    """
    root = Path(tempfile.gettempdir())
    if not root.is_dir() or root.is_symlink() or str(root).startswith(("\\\\", "//")):
        raise ValueError("Original Inspect logs require a regular local temporary directory.")
    resolved = root.resolve(strict=True)
    if str(resolved).startswith(("\\\\", "//")):
        raise ValueError("Original Inspect logs cannot be written to a network share.")
    return Path(tempfile.mkdtemp(prefix=f"inspect-original-{run_instance_id.hex}-", dir=resolved))


class _OriginalInertAtomicWork:
    """One unchanged Task followed by strict offline projection of its original score."""

    def __init__(
        self,
        *,
        source: ResolvedOriginalInspectTask,
        run: EvalRunRef,
        objective: str,
        memory: SQLiteMemory,
    ) -> None:
        self.case = source.case
        self.run = run
        self.case_run_id = run.case_run_id(case=self.case)
        self.objective = objective
        self.atomic_attack_name = f"eval_case_{self.case_run_id}"
        self.display_group = "original_inspect_inert"
        self._source = source
        self._memory = memory
        self._scenario_result_id: str | None = None
        self._attempted = False

    @property
    def technique_eval_hash(self) -> str:
        """The approved source and execution-profile fingerprint."""
        return self.run.spec.spec_sha256

    @property
    def logical_group_id(self) -> str:
        """The work-group identity within this one Scenario run."""
        return config_hash(
            {"atomic_attack_name": self.atomic_attack_name, "technique_eval_hash": self.technique_eval_hash}
        )

    def set_scenario_result_id(self, scenario_result_id: str | None) -> None:
        """
        Bind the run without allowing a case to be rebound after execution.

        Raises:
            ValueError: If the parent ID is invalid or differs from the previous binding.
        """
        if scenario_result_id is not None:
            scenario_result_id = str(uuid.UUID(scenario_result_id))
        if self._scenario_result_id is not None and self._scenario_result_id != scenario_result_id:
            raise ValueError("An original Inspect case cannot be rebound to another ScenarioResult.")
        self._scenario_result_id = scenario_result_id

    async def run_async(
        self,
        *,
        executor: AttackExecutor | None = None,
        return_partial_on_failure: bool = True,
        **attack_params: Any,
    ) -> AttackExecutorResult[AttackResult]:
        """
        Execute the unchanged Task once and project its original scorer after cleanup.

        Returns:
            AttackExecutorResult[AttackResult]: No duplicate Scenario AttackResult; the linked offline
                Score/AttackResult IDs are retained in Scenario metadata.

        Raises:
            ValueError: If execution arguments or imported source evidence differ.
            RuntimeError: If this case has already been attempted.
            asyncio.CancelledError: If the original Inspect execution is cancelled.
        """
        if executor is not None or attack_params:
            raise ValueError("Original Inspect Tasks cannot accept a target, converters, or attack parameters.")
        if self._scenario_result_id is None:
            raise ValueError("Original Inspect Task must be bound to a ScenarioResult before execution.")
        if self._attempted:
            raise RuntimeError("Original Inspect Task replay is disabled; reconcile the previous run first.")
        _ = return_partial_on_failure
        self._attempted = True
        await asyncio.to_thread(self._source.verify_unchanged)
        log_dir = await asyncio.to_thread(_allocate_log_dir, run_instance_id=self.run.run_instance_id)

        try:
            from pyrit.executor.benchmark.inspect_original_eval import (
                InspectOriginalEvalImporter,
                InspectOriginalScorePolicy,
            )
            from pyrit.executor.benchmark.inspect_original_runner import run_original_inert_eval_async

            completed = await run_original_inert_eval_async(
                memory=self._memory,
                log_dir=log_dir,
                family=OriginalInspectTaskId.INERT.value,
                run_instance_id=self.run.run_instance_id,
            )
            self._verify_import(completed=completed)
            await asyncio.to_thread(self._source.verify_unchanged)
            projected = await InspectOriginalEvalImporter(memory=self._memory).import_eval_log_async(
                path=completed.archive_path,
                cases=(self.case,),
                run=self.run,
                score_policy=InspectOriginalScorePolicy(
                    task_name=self.case.task_name,
                    task_version=self.case.task_version,
                    primary_scorer="original_inert_scorer",
                ),
            )
            case_result = self._verify_projection(live=completed.imported, projected=projected)
            reference = self._import_reference(completed=completed, projected=projected, case_result=case_result)
            await asyncio.to_thread(
                self._memory.update_scenario_metadata_fields,
                scenario_result_id=self._scenario_result_id,
                fields={OriginalInspectImportSummary.METADATA_KEY: reference.model_dump(mode="json")},
            )
        except (Exception, asyncio.CancelledError):
            logger.exception("Original Inspect run %s did not finish; log retained at %s.", self.case_run_id, log_dir)
            raise

        await asyncio.to_thread(shutil.rmtree, log_dir)
        return AttackExecutorResult(completed_results=[], incomplete_objectives=[], input_indices=[])

    def _verify_import(self, *, completed: InspectOriginalRun) -> None:
        """
        Require the exact selected case, original log and explicitly unscored import.

        Raises:
            ValueError: If a different case, grade, or incomplete archive was returned.
        """
        imported = completed.imported
        if (
            completed.case != self.case
            or completed.run != self.run
            or imported.case_run_ids != (self.case_run_id,)
            or imported.episode.run.run_id != f"inspect-run-{self.run.run_instance_id.hex}"
        ):
            raise ValueError("Original Inspect import differs from the selected run and case identity.")
        if (
            imported.log_status != "success"
            or imported.sample_count != 1
            or imported.original_final_score_events != 1
            or not imported.episode.coverage_complete
        ):
            raise ValueError("Original Inspect import is incomplete or lacks its original final ScoreEvent.")
        if imported.episode.score_id is not None or imported.episode.score_status is not ScoreStatus.UNDETERMINED:
            raise ValueError("Original Inspect import unexpectedly claims a qualified PyRIT Score.")

    def _verify_projection(
        self, *, live: InspectOriginalImport, projected: InspectOriginalImport
    ) -> InspectOriginalCaseResult:
        """
        Match the separately qualified offline result to this original live archive.

        Returns:
            InspectOriginalCaseResult: The source-attributed, persisted original scorer result.

        Raises:
            ValueError: If the projection changes the source, case, or outcome.
        """
        if (
            projected.archive_sha256 != live.archive_sha256
            or projected.inspect_run_id != live.inspect_run_id
            or projected.inspect_eval_id != live.inspect_eval_id
            or projected.case_run_ids != (self.case_run_id,)
            or projected.sample_count != 1
            or projected.original_final_score_events != 1
            or not projected.episode.coverage_complete
            or len(projected.case_results) != 1
        ):
            raise ValueError("Offline Inspect score projection differs from the approved live archive or case.")
        case_result = projected.case_results[0]
        score = case_result.score
        attack = case_result.attack_result
        metadata = score.score_metadata
        if metadata is None:
            raise ValueError("Offline Inspect Score has no source attribution.")
        if (
            case_result.case_run_id != self.case_run_id
            or case_result.sample_id != self.case.sample_id
            or case_result.epoch != self.case.epoch
            or case_result.primary_scorer != "original_inert_scorer"
            or score.status is not ScoreStatus.COMPLETE
            or score.score_type != "float_scale"
            or score.score_value is None
            or metadata.get("inspect_archive_sha256") != live.archive_sha256
            or not metadata.get("inspect_final_score_event_id")
            or attack.automated_score != score
            or attack.outcome is not AttackOutcome.UNDETERMINED
        ):
            raise ValueError("Offline Inspect Score/AttackResult has unapproved source or success attribution.")
        return case_result

    def _import_reference(
        self, *, completed: InspectOriginalRun, projected: InspectOriginalImport, case_result: InspectOriginalCaseResult
    ) -> OriginalInspectImportSummary:
        """
        Preserve both episode identities and the verified offline result links.

        Returns:
            OriginalInspectImportSummary: Source evidence and Score/AttackResult IDs without a success claim.

        Raises:
            ValueError: If the original source score cannot be published.
        """
        scorer_name = case_result.primary_scorer
        score_value = case_result.score.score_value
        if scorer_name is None or score_value is None:
            raise ValueError("Original Inspect source score was not qualified for Scenario result publication.")
        imported = completed.imported
        return OriginalInspectImportSummary(
            task_id=OriginalInspectTaskId.INERT,
            source_sha256=self.case.package.source_sha256,
            case_run_id=self.case_run_id,
            episode_id=imported.episode.run.run_id,
            projection_episode_id=projected.episode.run.run_id,
            inspect_run_id=imported.inspect_run_id,
            inspect_eval_id=imported.inspect_eval_id,
            archive_sha256=imported.archive_sha256,
            sample_count=imported.sample_count,
            original_final_score_events=imported.original_final_score_events,
            primary_scorer=scorer_name,
            score_id=case_result.score.id,
            attack_result_id=uuid.UUID(case_result.attack_result.attack_result_id),
            score_type=case_result.score.score_type,
            score_value=score_value,
            score_status=case_result.score.status,
            outcome=case_result.attack_result.outcome,
        )


class InspectOriginalInertTechnique(ScenarioTechnique):
    """The only execution policy: leave the authored Inspect Task unchanged."""

    ALL = ("all", {"all"})
    ORIGINAL_TASK = ("original_task", {"original"}, "Run the Task's own setup, solver, scorer and cleanup.")


class InspectOriginalInertScenario(TaskOwnedScenario):
    """
    Run only the SHA-pinned public `inspect_original_inert` Task and import its `.eval`.

    No external target, credentials, arbitrary Python, URL, sandbox, model profile
    or GUI Task editing is accepted. The original Inspect scorer runs unchanged;
    a separate strict offline import projects its value into one source-attributed
    PyRIT Score/AttackResult. No success threshold is approved, so the outcome
    remains UNDETERMINED.
    """

    VERSION: int = 1

    @apply_defaults
    def __init__(self, *, scenario_result_id: str | None = None) -> None:
        """Register the approved Task without loading its executable source."""
        super().__init__(
            version=self.VERSION,
            technique_class=InspectOriginalInertTechnique,
            default_dataset_config=DatasetAttackConfiguration(),
            objective_scorer=None,
            scenario_result_id=scenario_result_id,
        )

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """
        Permit only one serial original Task and no caller-supplied memory labels.

        Returns:
            list[Parameter]: The bounded Task selector and shared execution controls.
        """
        return [
            *[p for p in super().supported_parameters() if p.name not in {"max_concurrency", "memory_labels"}],
            Parameter(
                name="max_concurrency",
                description="One approved original Inspect Task runs once at a time.",
                param_type=int,
                default=1,
            ),
        ]

    @classmethod
    def additional_parameters(cls) -> list[Parameter]:
        """
        Offer the sole approved public Task as a named, validated catalog choice.

        Returns:
            list[Parameter]: One allowlisted Task ID, never a Python file or URL.
        """
        return [
            Parameter(
                name="eval_family",
                description="Approved public, no-model, no-sandbox original Inspect Task ID.",
                param_type=OriginalInspectTaskId,
                default=OriginalInspectTaskId.INERT,
            )
        ]

    @classmethod
    def validate_run_request(cls, *, request: RunScenarioRequest) -> None:
        """
        Reject executable/secret-bearing request fields before backend initializers run.

        Raises:
            ValueError: If any selection or execution control is not approved.
        """
        unsupported = [
            name
            for name in (
                "target_name",
                "initializers",
                "initializer_args",
                "techniques",
                "dataset_names",
                "max_dataset_size",
                "dataset_filters",
                "labels",
                "scenario_result_id",
            )
            if getattr(request, name) is not None
        ]
        unsupported.extend((request.model_extra or {}).keys())
        if unsupported:
            raise ValueError(f"Original Inspect one-click run does not accept: {', '.join(sorted(unsupported))}.")
        if request.max_concurrency not in (None, 1) or request.max_retries != 0 or request.include_baseline is True:
            raise ValueError("Original Inspect one-click run requires one case, no retries and no baseline.")
        if request.scenario_params not in (
            None,
            {},
            {"eval_family": OriginalInspectTaskId.INERT.value},
        ):
            raise ValueError("Only the named inspect_original_inert Task ID is approved; no custom source or profile.")

    async def _estimate_run_size_async(self, *, budget: BoundedDatasetSize) -> ScenarioRunSizeEstimate:
        """
        Describe the one planned import without materializing executable Task code.

        Args:
            budget: PyRIT dataset budget; the approved source contains exactly one Task.

        Returns:
            ScenarioRunSizeEstimate: One original Task; no PyRIT grade is estimated.
        """
        self._validate_selection()
        return ScenarioRunSizeEstimate(
            total_attack_count=1,
            components=[ScenarioRunSizeComponent(label="Approved original Inspect Task", count=1)],
            note="One unchanged original Task; its offline result has no success threshold.",
        )

    async def _build_task_owned_atomic_attacks_async(
        self, *, context: TaskOwnedScenarioContext
    ) -> list[_OriginalInertAtomicWork]:
        """
        Resolve only pinned code and construct one unscored case for SQLite.

        Returns:
            list[_OriginalInertAtomicWork]: One original Task/Sample import.

        Raises:
            ValueError: If the database, source, or execution profile is unsupported.
        """
        from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory

        self._validate_selection()
        if not isinstance(self._memory, SQLiteMemory):
            raise ValueError("Original Inspect one-click runs require an initialized PyRIT SQLite database.")
        source = await asyncio.to_thread(
            EvalSourceFactory.resolve_original_inert, family=OriginalInspectTaskId.INERT.value
        )
        objective = source.task.dataset[0].input
        if not isinstance(objective, str):
            raise ValueError("The approved original Inspect Task must contain one text Sample.")
        run = EvalRunRef(spec=source.spec, run_instance_id=context.run_instance_id)
        return [_OriginalInertAtomicWork(source=source, run=run, objective=objective, memory=self._memory)]

    def _validate_selection(self) -> None:
        """
        Enforce the single unchanged technique and runtime policy for API and registry callers.

        Raises:
            ValueError: If the selected Task or shared Scenario controls were changed.
        """
        if self.params["eval_family"] is not OriginalInspectTaskId.INERT:
            raise ValueError("Only the named inspect_original_inert Task is approved.")
        if (
            self.params["max_concurrency"] != 1
            or self.params["max_retries"] != 0
            or self._scenario_techniques != [InspectOriginalInertTechnique.ORIGINAL_TASK]
        ):
            raise ValueError("Original Inspect requires its unchanged Task technique, one case and no retries.")

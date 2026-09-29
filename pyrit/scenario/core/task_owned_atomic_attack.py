# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One task-owned case execution with an already committed original grade."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Protocol

from pyrit.executor.attack import AttackExecutor, AttackExecutorResult
from pyrit.memory import CentralMemory
from pyrit.models import (
    AttackResult,
    CommittedCaseExecution,
    EvalCaseRef,
    EvalRunRef,
    EvalScoreProvenance,
    EvalScoreRole,
    Score,
    config_hash,
)

if TYPE_CHECKING:
    from pyrit.memory.memory_interface import MemoryInterface


class CaseExecutor(Protocol):
    """Attest the original external grade, then commit and return its PyRIT Score."""

    async def execute_case_async(self, *, case: EvalCaseRef, run: EvalRunRef) -> CommittedCaseExecution:
        """Execute one case without asking task-owned core to score it."""
        ...


class TaskOwnedCaseReplayBlockedError(RuntimeError):
    """A case may have committed a Score and cannot be executed again safely."""


class TaskOwnedResultPersistenceError(RuntimeError):
    """An AttackResult write failed after the original Score had been committed."""


class TaskOwnedAtomicAttack:
    """Adapt one externally executed case to PyRIT's Scenario work contract."""

    def __init__(
        self,
        *,
        case: EvalCaseRef,
        run: EvalRunRef,
        objective: str,
        case_executor: CaseExecutor,
        display_group: str = "task_owned",
        memory_labels: dict[str, str] | None = None,
    ) -> None:
        """
        Bind one source case and execution profile to an injected case executor.

        Raises:
            ValueError: If the objective, display group, or case/run pairing is invalid.
            TypeError: If the supplied executor cannot run cases.
        """
        if not objective.strip():
            raise ValueError("A task-owned case needs a non-empty objective for its AttackResult")
        if not display_group.strip():
            raise ValueError("A task-owned case needs a non-empty display_group")
        if not callable(getattr(case_executor, "execute_case_async", None)):
            raise TypeError("Task-owned case_executor must implement execute_case_async")
        self.case = case
        self.run = run
        self.case_run_id = run.case_run_id(case=case)
        self.objective = objective
        self.atomic_attack_name = f"eval_case_{self.case_run_id}"
        self.display_group = display_group
        self._case_executor = case_executor
        self._memory_labels = dict(memory_labels or {})
        self._memory: MemoryInterface = CentralMemory.get_memory_instance()
        self._scenario_result_id: str | None = None
        self._attempted = False

    @property
    def technique_eval_hash(self) -> str:
        """The stable harness/model/overlay specification fingerprint."""
        return self.run.spec.spec_sha256

    @property
    def logical_group_id(self) -> str:
        """The run-specific atomic-group ID, distinct for every invocation."""
        return config_hash(
            {"atomic_attack_name": self.atomic_attack_name, "technique_eval_hash": self.technique_eval_hash}
        )

    def set_scenario_result_id(self, scenario_result_id: str | None) -> None:
        """
        Bind this case to one ScenarioResult without allowing rebinding.

        Raises:
            ValueError: If the ID is malformed or the case is already bound elsewhere.
        """
        if scenario_result_id is not None:
            scenario_result_id = str(uuid.UUID(scenario_result_id))
        if self._scenario_result_id is not None and scenario_result_id != self._scenario_result_id:
            raise ValueError("A task-owned case cannot be rebound to another ScenarioResult")
        self._scenario_result_id = scenario_result_id

    async def run_async(
        self,
        *,
        executor: AttackExecutor | None = None,
        return_partial_on_failure: bool = True,
        **attack_params: Any,
    ) -> AttackExecutorResult[AttackResult]:
        """
        Invoke one case at most once and link its already persisted original Score.

        Returns:
            AttackExecutorResult[AttackResult]: The single persisted case result.

        Raises:
            TypeError: If the executor returns an unsupported record.
            ValueError: If the case is unbound or the original Score is missing or invalid.
            TaskOwnedCaseReplayBlockedError: If a prior execution has an unresolved write.
            TaskOwnedResultPersistenceError: If result persistence fails after scoring.
        """
        if attack_params:
            raise ValueError(f"Task-owned cases do not accept AttackExecutor parameters: {sorted(attack_params)}")
        if self._scenario_result_id is None:
            raise ValueError("Task-owned case must be bound to a ScenarioResult before execution")
        # The Scenario's worker pool bounds concurrency: each task-owned work item is one case.
        _ = executor, return_partial_on_failure
        result_id = str(uuid.uuid5(uuid.UUID(self._scenario_result_id), self.case_run_id))
        existing = await asyncio.to_thread(self._get_existing_result, result_id=result_id)
        if existing is not None:
            return self._as_executor_result(result=existing)
        if self._attempted:
            raise TaskOwnedCaseReplayBlockedError(
                f"Case {self.case_run_id} may already have a committed Score; "
                "automatic execution is blocked until its original Score and AttackResult are reconciled."
            )

        self._attempted = True
        execution = await self._case_executor.execute_case_async(case=self.case, run=self.run)
        if not isinstance(execution, CommittedCaseExecution):
            raise TypeError("Case executor must return a CommittedCaseExecution")
        score = await asyncio.to_thread(self._load_original_score, execution=execution)
        result = self._build_result(result_id=result_id, score=score, execution=execution)
        try:
            await self._persist_result_async(result=result)
        except Exception as error:
            raise TaskOwnedResultPersistenceError(
                f"Original benchmark Score {score.id} was committed for case {self.case_run_id}, "
                "but its AttackResult write failed; do not execute the case again without reconciliation."
            ) from error
        return self._as_executor_result(result=result)

    async def _persist_result_async(self, *, result: AttackResult) -> None:
        write = asyncio.create_task(
            asyncio.to_thread(self._memory.add_attack_results_to_memory, attack_results=[result])
        )
        try:
            # Do not abandon an in-flight post-Score insert when the caller is cancelled.
            await asyncio.shield(write)
        except asyncio.CancelledError:
            await write
            raise

    def _load_original_score(self, *, execution: CommittedCaseExecution) -> Score:
        provenance = execution.original_score_provenance
        if provenance.role is not EvalScoreRole.BENCHMARK_ORIGINAL or provenance.case_run_id != self.case_run_id:
            raise ValueError("Case executor returned a score without matching benchmark-original PyRIT provenance")

        scores = self._memory.get_scores(score_ids=[str(execution.original_score_id)])
        if len(scores) != 1:
            raise ValueError(f"Original benchmark Score {execution.original_score_id} is not committed in memory")
        score = scores[0]
        self._verify_score_provenance(score=score, expected=provenance)
        return score

    def _verify_score_provenance(self, *, score: Score, expected: EvalScoreProvenance | None = None) -> None:
        provenance = EvalScoreProvenance.from_metadata(metadata=score.score_metadata)
        scorer = score.scorer_class_identifier
        if (
            provenance.role is not EvalScoreRole.BENCHMARK_ORIGINAL
            or provenance.case_run_id != self.case_run_id
            or scorer is None
            or scorer.hash != provenance.pyrit_scorer_hash
            or (expected is not None and expected != provenance)
        ):
            raise ValueError(f"Score {score.id} does not have matching benchmark-original PyRIT provenance")

    def _build_result(self, *, result_id: str, score: Score, execution: CommittedCaseExecution) -> AttackResult:
        assert self._scenario_result_id is not None
        return AttackResult(
            attack_result_id=result_id,
            conversation_id=str(execution.conversation_id),
            objective=self.objective,
            automated_score=score,
            outcome=execution.outcome,
            outcome_reason=execution.outcome_reason,
            executed_turns=execution.executed_turns,
            execution_time_ms=execution.execution_time_ms,
            labels=self._memory_labels,
            attribution_parent_id=self._scenario_result_id,
            attribution_data={
                "parent_collection": self.atomic_attack_name,
                "parent_eval_hash": self.technique_eval_hash,
                "seed_group_id": self.case_run_id,
                "case_id": self.case.case_id,
                "score_ids": {EvalScoreRole.BENCHMARK_ORIGINAL.value: str(score.id)},
            },
        )

    def _get_existing_result(self, *, result_id: str) -> AttackResult | None:
        results = self._memory.get_attack_results(attack_result_ids=[result_id])
        if not results:
            return None
        if len(results) != 1:
            raise ValueError(f"Ambiguous AttackResult rows for case {self.case_run_id}")
        result = results[0]
        attribution = result.attribution_data or {}
        score_ids = attribution.get("score_ids")
        original_id = score_ids.get(EvalScoreRole.BENCHMARK_ORIGINAL.value) if isinstance(score_ids, dict) else None
        score = result.automated_score
        if (
            result.attribution_parent_id != self._scenario_result_id
            or attribution.get("parent_collection") != self.atomic_attack_name
            or attribution.get("parent_eval_hash") != self.technique_eval_hash
            or attribution.get("seed_group_id") != self.case_run_id
            or attribution.get("case_id") != self.case.case_id
            or score is None
            or str(score.id) != original_id
        ):
            raise ValueError(f"Persisted AttackResult {result_id} does not match case {self.case_run_id}")
        self._verify_score_provenance(score=score)
        return result

    @staticmethod
    def _as_executor_result(*, result: AttackResult) -> AttackExecutorResult[AttackResult]:
        return AttackExecutorResult(completed_results=[result], incomplete_objectives=[], input_indices=[0])

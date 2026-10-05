# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Explicit Scenario opt-in for cases whose harness owns its target and grade."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pyrit.common.utils import to_sha256
from pyrit.models import (
    SCENARIO_RUN_PLAN_METADATA_KEY,
    EvalCaseRef,
    EvalRunRef,
    ScenarioExecutionOwner,
    ScenarioIdentifier,
    ScenarioResult,
    ScenarioRunPlan,
    ScenarioRunPlanAtomicGroup,
    ScenarioRunPlanSeedGroup,
    ScenarioRunSizeEstimate,
    config_hash,
)
from pyrit.scenario.core.atomic_work import AtomicWork
from pyrit.scenario.core.scenario import BaselineAttackPolicy, Scenario

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.models import AttackSeedGroup, BoundedDatasetSize
    from pyrit.models.parameter import Parameter
    from pyrit.scenario.core.atomic_attack import AtomicAttack
    from pyrit.scenario.core.scenario_context import ScenarioContext, TaskOwnedScenarioContext


@runtime_checkable
class TaskOwnedCaseWork(AtomicWork, Protocol):
    """A case with source identity, whether or not a PyRIT grade is committed."""

    case: EvalCaseRef
    run: EvalRunRef
    case_run_id: str
    objective: str


class TaskOwnedScenario(Scenario):
    """Base for a Scenario whose selected Eval cases own execution and original evidence."""

    TASK_OWNED: ClassVar[bool] = True
    SERVER_ADMISSION_REQUIRED: ClassVar[bool] = False
    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Forbidden

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """
        Omit target-owned inputs that cannot be used by a task-owned harness.

        Returns:
            list[Parameter]: Shared execution controls and adapter-specific inputs.
        """
        excluded = {"objective_target", "dataset_config", "technique_converters"}
        return [parameter for parameter in super().supported_parameters() if parameter.name not in excluded]

    async def _resolve_seed_groups_by_dataset_async(
        self, *, apply_sampling: bool = True
    ) -> dict[str, list[AttackSeedGroup]]:
        """
        Skip PyRIT seed loading; the adapter supplies Eval cases.

        Returns:
            dict[str, list[AttackSeedGroup]]: An empty dataset mapping.
        """
        return {}

    async def _estimate_run_size_async(self, *, budget: BoundedDatasetSize) -> ScenarioRunSizeEstimate:
        """
        Avoid guessing a count before the trusted Eval source is resolved.

        Args:
            budget: PyRIT dataset budget; Task/Sample selection remains source-owned.

        Returns:
            ScenarioRunSizeEstimate: An unavailable-size explanation.
        """
        return ScenarioRunSizeEstimate.unavailable(note="Select an Eval source to determine its Task/Sample count.")

    async def _build_atomic_attacks_async(self, *, context: ScenarioContext) -> list[AtomicAttack]:
        """
        Reject the target-owned builder; task-owned scenarios use their own hook.

        Raises:
            RuntimeError: Always, since task-owned work has no external target.
        """
        raise RuntimeError("TaskOwnedScenario cannot build target-owned AtomicAttack instances")

    @abstractmethod
    async def _build_task_owned_atomic_attacks_async(
        self, *, context: TaskOwnedScenarioContext
    ) -> Sequence[AtomicWork]:
        """Build one case-owned work item per selected Eval Task/Sample."""
        ...

    def _task_owned_attacks(self) -> list[TaskOwnedCaseWork]:
        attacks = [work for work in self._atomic_attacks if isinstance(work, TaskOwnedCaseWork)]
        if len(attacks) != len(self._atomic_attacks):
            raise TypeError("TaskOwnedScenario can schedule only case-identified task-owned work")
        return attacks

    def _validate_task_owned_work(self) -> None:
        attacks = self._task_owned_attacks()
        if not attacks or self._task_owned_run_instance_id is None:
            raise ValueError("TaskOwnedScenario must select at least one case in a fresh run")
        if any(work.run.run_instance_id != self._task_owned_run_instance_id for work in attacks):
            raise ValueError("Task-owned work must use this Scenario's fresh run-instance ID")
        spec = attacks[0].run.spec
        if any(work.run.spec != spec for work in attacks):
            raise ValueError("Every Eval case in a Scenario run must use the same source and execution profile")
        if spec.input_variant is not None and len(attacks) != 1:
            raise ValueError("A GUI input variant must target exactly one Eval case in a fresh run")
        if len({work.case_run_id for work in attacks}) != len(attacks):
            raise ValueError("TaskOwnedScenario contains duplicate Eval case-run identities")
        if len({work.atomic_attack_name for work in attacks}) != len(attacks):
            raise ValueError("TaskOwnedScenario contains duplicate atomic-attack names")

    def _build_run_plan(self) -> ScenarioRunPlan:
        """
        Persist distinct case-run IDs without changing legacy v1 plan fields.

        Returns:
            ScenarioRunPlan: The case-aware plan for this invocation.
        """
        attacks = self._task_owned_attacks()
        spec = attacks[0].run.spec
        return ScenarioRunPlan(
            scenario_registry_name=self._scenario_registry_name,
            run_instance_id=self._task_owned_run_instance_id,
            eval_spec_sha256=spec.spec_sha256,
            atomic_groups=[
                ScenarioRunPlanAtomicGroup(
                    id=work.logical_group_id,
                    atomic_attack_name=work.atomic_attack_name,
                    display_group=work.display_group,
                    technique_eval_hash=work.technique_eval_hash,
                    seed_group_ids=[work.case_run_id],
                )
                for work in attacks
            ],
            seed_groups=[
                ScenarioRunPlanSeedGroup(
                    id=work.case_run_id,
                    objective_sha256=to_sha256(work.objective),
                    objective=work.objective,
                    case_id=work.case.case_id,
                    source_sha256=work.case.package.source_sha256,
                    input_variant_sha256=spec.input_variant.content_sha256 if spec.input_variant else None,
                )
                for work in attacks
            ],
        )

    def _build_initial_scenario_metadata(self) -> dict[str, Any]:
        """
        Store the run-instance identity separately from the stable identifier.

        Returns:
            dict[str, Any]: Run-instance ID, spec digest, and case-aware plan.
        """
        attacks = self._task_owned_attacks()
        assert self._task_owned_run_instance_id is not None
        return {
            "run_instance_id": str(self._task_owned_run_instance_id),
            "eval_spec_sha256": attacks[0].run.spec.spec_sha256,
            SCENARIO_RUN_PLAN_METADATA_KEY: self._build_run_plan().model_dump(mode="json", exclude_none=True),
        }

    def _build_scenario_identifier(self) -> ScenarioIdentifier:
        """
        Include only public, stable spec and case-selection fingerprints.

        Returns:
            ScenarioIdentifier: Identity without runtime objects or private data.
        """
        attacks = self._task_owned_attacks()
        spec = attacks[0].run.spec
        variant = spec.input_variant
        params = {
            "execution_owner": ScenarioExecutionOwner.TASK_OWNED.value,
            "eval_spec_sha256": spec.spec_sha256,
            "source_kind": spec.package.kind.value,
            "source_name": spec.package.name,
            "source_sha256": spec.package.source_sha256,
            "harness_name": spec.harness.name,
            "harness_sha256": spec.harness.config_sha256,
            "model_route_name": spec.model_route.name,
            "model_route_sha256": spec.model_route.config_sha256,
            "case_set_sha256": config_hash({"case_ids": sorted(work.case.case_id for work in attacks)}),
        }
        if self.SERVER_ADMISSION_REQUIRED:
            params["server_admission_required"] = True
        if variant is not None:
            params["input_variant_sha256"] = variant.content_sha256
            params["input_surface_id"] = variant.surface_id
        return ScenarioIdentifier.of(
            self,
            params=params,
            version=self._version,
            techniques=sorted({technique.value for technique in self._scenario_techniques}),
            datasets=[spec.package.name],
            objective_target=None,
            objective_scorer=None,
        )

    async def _get_remaining_atomic_attacks_async(self) -> list[AtomicWork]:
        """
        Leave task-owned cases unfiltered; V1 cannot automatically resume.

        Returns:
            list[AtomicWork]: All cases for the single allowed execution.
        """
        return list(self._atomic_attacks)

    async def run_async(self) -> ScenarioResult:
        """
        Reject a second invocation, including after cancellation or failure.

        Returns:
            ScenarioResult: The first execution's persisted results.

        Raises:
            RuntimeError: If the scenario was already executed.
        """
        if self._task_owned_run_started:
            raise RuntimeError("Task-owned Scenario replay is disabled; start a new run after reconciling prior work")
        if self._scenario_result_id is None or not self._atomic_attacks:
            return await super().run_async()
        self._task_owned_run_started = True
        return await super().run_async()

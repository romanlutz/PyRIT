# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-click selection of trusted original Inspect Tasks and Samples."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from pyrit.common import apply_defaults
from pyrit.models import EvalRunRef, EvalSpecRef, InputVariantRef, Parameter, ScenarioResult
from pyrit.scenario.core import (
    DatasetAttackConfiguration,
    ScenarioTechnique,
    TaskOwnedAtomicAttack,
    TaskOwnedScenario,
)

if TYPE_CHECKING:
    from pyrit.executor.benchmark.inspect_eval_source import ResolvedInspectEvalCase
    from pyrit.scenario.core.scenario_context import TaskOwnedScenarioContext


class InspectEvalTechnique(ScenarioTechnique):
    """The retained PyRIT adversarial-turn policy for the GHCP pilot."""

    ALL = ("all", {"all"})
    RETAINED_RED_TEAMING = ("retained_red_teaming", {"red_teaming"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """
        Distinguish the one supported attack strategy from family/model selection.

        Returns:
            set[str]: Only the aggregate selector.
        """
        return {"all"}


class InspectEvalScenario(TaskOwnedScenario):
    """
    Run a trusted original Inspect Task as a GHCP protocol-only Scenario.

    One named benign case or a finite trusted-local Task/Sample inventory uses
    the reviewed GHCP/Qwen protocol harness. Other profiles and cyber claims fail closed.
    The original Inspect scorer owns grading, never a second PyRIT scorer.
    """

    VERSION: int = 1

    @apply_defaults
    def __init__(self, *, scenario_result_id: str | None = None) -> None:
        """Register a no-argument Task-owned Scenario without loading executable Eval code."""
        super().__init__(
            version=self.VERSION,
            technique_class=InspectEvalTechnique,
            default_dataset_config=DatasetAttackConfiguration(),
            objective_scorer=None,
            scenario_result_id=scenario_result_id,
        )
        self._resolved_cases: tuple[ResolvedInspectEvalCase, ...] = ()

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """
        Run original Inspect cases serially, preserving task-owned controls.

        Returns:
            list[Parameter]: Serial concurrency and declared Eval selectors.
        """
        return [
            *[p for p in super().supported_parameters() if p.name != "max_concurrency"],
            Parameter(
                name="max_concurrency",
                description="Inspect Eval runs one original Task/Sample at a time.",
                param_type=int,
                default=1,
            ),
        ]

    @classmethod
    def additional_parameters(cls) -> list[Parameter]:
        """
        Expose only trusted source, approved profile and one initial user-input edit.

        Returns:
            list[Parameter]: Scenario selectors suitable for registry/CLI metadata.
        """
        return [
            Parameter(
                name="eval_family",
                description="Pinned named Inspect Eval family (benign_protocol only).",
                param_type=str,
            ),
            Parameter(
                name="trusted_eval_dir", description="Explicitly trusted local Inspect Eval directory.", param_type=Path
            ),
            Parameter(
                name="trust_local",
                description="Approve executing the pinned local Eval Task factory.",
                param_type=bool,
                default=False,
            ),
            Parameter(
                name="eval_revision",
                description="Required SHA256 for local manifest, optional for named sources.",
                param_type=str,
            ),
            Parameter(
                name="harness_profile",
                description="Approved agent profile (ghcp_protocol_v1 only).",
                param_type=str,
                default="ghcp_protocol_v1",
            ),
            Parameter(
                name="model_route",
                description="Trusted host model route (qwen3_loopback_v1 only).",
                param_type=str,
                default="qwen3_loopback_v1",
            ),
            Parameter(
                name="initial_user_input",
                description="Optional bounded input edit for the single schema-1 Sample only.",
                param_type=str,
            ),
        ]

    async def _build_task_owned_atomic_attacks_async(
        self, *, context: TaskOwnedScenarioContext
    ) -> list[TaskOwnedAtomicAttack]:
        """
        Resolve all source cases and leave each original scorer and cleanup to Inspect.

        Returns:
            list[TaskOwnedAtomicAttack]: One honest case-to-committed-Score work item per Sample.

        Raises:
            ValueError: If source, capability, or reviewed profile is unsupported.
        """
        from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
        from pyrit.executor.benchmark.inspect_ghcp_case_executor import (
            InspectGhcpCaseExecutor,
            InspectGhcpPilotEnvironment,
        )

        if self.params["max_concurrency"] != 1:
            raise ValueError("Inspect Eval V1 requires max_concurrency=1 and has no automatic replay.")
        if self.params["harness_profile"] != "ghcp_protocol_v1" or self.params["model_route"] != "qwen3_loopback_v1":
            raise ValueError("Unknown or unqualified Inspect harness/model route; GHCP/Qwen protocol only.")
        environment = InspectGhcpPilotEnvironment.from_environment()
        selected_cases = await asyncio.to_thread(
            EvalSourceFactory.resolve_cases,
            family=self.params["eval_family"],
            trusted_dir=self.params["trusted_eval_dir"],
            trusted_local=self.params["trust_local"],
            revision_sha256=self.params["eval_revision"],
            input_override=self.params["initial_user_input"],
            agent_image=environment.agent_image,
            target_image=environment.target_image,
            approved_image_ids=environment.image_ids,
        )
        if not selected_cases or len({selected.sandbox_sha256 for selected in selected_cases}) != 1:
            raise ValueError("Inspect Eval cases require one reviewed effective sandbox and nonempty inventory.")
        selected = selected_cases[0]
        variant = (
            InputVariantRef(
                case_id=selected.case.case_id,
                surface_id="sample_input",
                content_sha256=selected.input_override_sha256,
            )
            if selected.input_override_sha256 is not None
            else None
        )
        run = EvalRunRef(
            spec=EvalSpecRef(
                package=selected.case.package,
                harness=environment.harness_ref(sandbox_sha256=selected.sandbox_sha256),
                model_route=environment.model_route_ref(),
                input_variant=variant,
            ),
            run_instance_id=context.run_instance_id,
        )
        work = [
            TaskOwnedAtomicAttack(
                case=case.case,
                run=run,
                objective=case.objective.value,
                case_executor=InspectGhcpCaseExecutor(selected=case, environment=environment),
                display_group=f"{case.case.package.name}_ghcp_protocol",
                memory_labels=context.memory_labels,
            )
            for case in selected_cases
        ]
        self._resolved_cases = selected_cases
        return work

    async def run_async(self) -> ScenarioResult:
        """
        Revalidate every pinned Task/Sample before starting the first case.

        Returns:
            ScenarioResult: Each case linked to its own original, pre-committed Score.
        """
        if self._resolved_cases and not self._task_owned_run_started:
            for selected in self._resolved_cases:
                await asyncio.to_thread(selected.verify_unchanged)
        return await super().run_async()

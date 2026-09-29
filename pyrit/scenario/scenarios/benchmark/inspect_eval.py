# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-click selection of one trusted, original Inspect benign Task/Sample."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from pyrit.common import apply_defaults
from pyrit.models import EvalRunRef, EvalSpecRef, InputVariantRef, Parameter
from pyrit.scenario.core import (
    DatasetAttackConfiguration,
    ScenarioTechnique,
    TaskOwnedAtomicAttack,
    TaskOwnedScenario,
)

if TYPE_CHECKING:
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

    V1 deliberately accepts one Task/Sample and one previously reviewed
    GHCP/Qwen harness; unqualified cyber families and other harnesses fail closed.
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

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """
        Default to one sample at a time, preserving all other task-owned controls.

        Returns:
            list[Parameter]: One bounded concurrency setting and declared Eval selectors.
        """
        return [
            *[p for p in super().supported_parameters() if p.name != "max_concurrency"],
            Parameter(
                name="max_concurrency",
                description="V1 runs exactly one original Inspect sample.",
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
                description="Optional bounded replacement for the selected Sample.input.",
                param_type=str,
            ),
        ]

    async def _build_task_owned_atomic_attacks_async(
        self, *, context: TaskOwnedScenarioContext
    ) -> list[TaskOwnedAtomicAttack]:
        """
        Resolve exactly one source case and let Inspect own its setup/scorer/cleanup.

        Returns:
            list[TaskOwnedAtomicAttack]: One honest case-to-committed-Score work item.

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
        selected = await asyncio.to_thread(
            EvalSourceFactory.resolve,
            family=self.params["eval_family"],
            trusted_dir=self.params["trusted_eval_dir"],
            trusted_local=self.params["trust_local"],
            revision_sha256=self.params["eval_revision"],
            input_override=self.params["initial_user_input"],
            agent_image=environment.agent_image,
            target_image=environment.target_image,
            approved_image_ids=environment.image_ids,
        )
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
        return [
            TaskOwnedAtomicAttack(
                case=selected.case,
                run=run,
                objective=selected.objective.value,
                case_executor=InspectGhcpCaseExecutor(selected=selected, environment=environment),
                display_group=f"{selected.case.package.name}_ghcp_protocol",
                memory_labels=context.memory_labels,
            )
        ]

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Scenario-owned helpers for transparent outer-unit run-size estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import TYPE_CHECKING

from pyrit.models import (
    ScenarioDatasetSummary,
    ScenarioRunSizeComponent,
    ScenarioRunSizeEstimate,
    ScenarioRunSizeFactor,
    ScenarioRunSizeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyrit.models import AttackSeedGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory
    from pyrit.scenario.core.dataset_configuration import DatasetAttackConfiguration
    from pyrit.scenario.core.scenario_technique import ScenarioTechnique


class ScenarioRunSizeShape(Enum):
    """Explicit opt-in shape used by the safe base estimation seam."""

    UNAVAILABLE = "unavailable"
    MATRIX = "matrix"
    SINGLE_POPULATION = "single_population"


@dataclass(frozen=True)
class ScenarioRunSizeContext:
    """Resolved inputs for estimating outer ``ScenarioRunPlan`` units."""

    scenario_techniques: Sequence[ScenarioTechnique]
    dataset_config: DatasetAttackConfiguration
    seed_groups_by_source: Mapping[str, list[AttackSeedGroup]]
    logical_seed_groups_by_source: Mapping[str, list[AttackSeedGroup]]
    dataset_summaries: Sequence[ScenarioDatasetSummary]
    include_baseline: bool
    objective_target: PromptTarget | None = None
    target_is_configured: bool | None = None

    @property
    def selected_seed_group_count(self) -> int:
        """The flattened selected logical group count."""
        return sum(len(groups) for groups in self.seed_groups_by_source.values())


def build_size_component(
    *,
    label: str,
    factors: Sequence[tuple[str, int]],
    is_baseline: bool = False,
    caveat: str | None = None,
) -> ScenarioRunSizeComponent:
    """
    Build a validated additive component from ordered multiplicative factors.

    Returns:
        ScenarioRunSizeComponent: The validated additive component.
    """
    factor_models = [ScenarioRunSizeFactor(label=factor_label, count=count) for factor_label, count in factors]
    return ScenarioRunSizeComponent(
        label=label,
        planned_executions=prod(factor.count for factor in factor_models),
        factors=factor_models,
        is_baseline=is_baseline,
        caveat=caveat,
    )


def build_baseline_size_component(
    *, context: ScenarioRunSizeContext, label: str = "Baseline"
) -> ScenarioRunSizeComponent:
    """
    Build the explicit scenario-wide baseline component.

    Returns:
        ScenarioRunSizeComponent: The baseline component.
    """
    return build_size_component(
        label=label,
        factors=[
            ("baseline attacks", 1),
            ("selected logical seed groups", context.selected_seed_group_count),
        ],
        is_baseline=True,
    )


def build_exact_estimate(
    *,
    context: ScenarioRunSizeContext,
    components: Sequence[ScenarioRunSizeComponent],
    caveat: str | None = None,
) -> ScenarioRunSizeEstimate:
    """
    Build an exact estimate whose total is the additive component sum.

    Returns:
        ScenarioRunSizeEstimate: The exact estimate.
    """
    component_list = list(components)
    return ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.EXACT,
        total_planned_executions=sum(component.planned_executions for component in component_list),
        components=component_list,
        datasets=list(context.dataset_summaries),
        caveat=caveat,
    )


def build_conditional_estimate(
    *,
    context: ScenarioRunSizeContext,
    components: Sequence[ScenarioRunSizeComponent],
    caveat: str,
    total_planned_executions: int | None = None,
) -> ScenarioRunSizeEstimate:
    """
    Build an estimate whose total depends on a surfaced condition.

    Returns:
        ScenarioRunSizeEstimate: The conditional estimate.
    """
    return ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.CONDITIONAL,
        total_planned_executions=total_planned_executions,
        components=list(components),
        datasets=list(context.dataset_summaries),
        caveat=caveat,
    )


def build_unavailable_estimate(
    *,
    context: ScenarioRunSizeContext,
    caveat: str,
) -> ScenarioRunSizeEstimate:
    """
    Build an explicitly unavailable estimate.

    Returns:
        ScenarioRunSizeEstimate: The unavailable estimate.
    """
    return ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.UNAVAILABLE,
        total_planned_executions=None,
        datasets=list(context.dataset_summaries),
        caveat=caveat,
    )


def append_estimate_caveat(
    *,
    estimate: ScenarioRunSizeEstimate,
    caveat: str,
) -> ScenarioRunSizeEstimate:
    """
    Append a surfaced caveat without changing estimate confidence or totals.

    Returns:
        ScenarioRunSizeEstimate: A copy with the combined caveat.
    """
    combined = f"{estimate.caveat} {caveat}" if estimate.caveat else caveat
    return estimate.model_copy(update={"caveat": combined})


def build_matrix_run_size_estimate(
    *,
    context: ScenarioRunSizeContext,
    technique_factories: Mapping[str, AttackTechniqueFactory],
    additional_factors: Sequence[tuple[str, int]] = (),
    caveat: str | None = None,
) -> ScenarioRunSizeEstimate:
    """
    Estimate a technique-by-source matrix using the builder's compatibility semantics.

    Returns:
        ScenarioRunSizeEstimate: The exact matrix estimate.
    """
    from pyrit.scenario.core.matrix_atomic_attack_builder import filter_compatible_seed_groups

    components: list[ScenarioRunSizeComponent] = []
    if context.include_baseline:
        components.append(build_baseline_size_component(context=context))

    missing_factories: list[str] = []
    for technique in context.scenario_techniques:
        factory = technique_factories.get(technique.value)
        if factory is None:
            missing_factories.append(technique.value)
            continue
        compatible_count = sum(
            len(filter_compatible_seed_groups(factory=factory, seed_groups=groups))
            for groups in context.seed_groups_by_source.values()
        )
        components.append(
            build_size_component(
                label=technique.value,
                factors=[
                    ("techniques", 1),
                    *additional_factors,
                    ("compatible logical seed groups", compatible_count),
                ],
            )
        )

    missing_caveat = (
        f"Selected techniques without registered factories are omitted: {', '.join(missing_factories)}."
        if missing_factories
        else None
    )
    combined_caveat = " ".join(part for part in (caveat, missing_caveat) if part) or None
    return build_exact_estimate(context=context, components=components, caveat=combined_caveat)

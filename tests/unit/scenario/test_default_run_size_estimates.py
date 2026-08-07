# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scenario-owned default-run size estimates."""

from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.models import (
    AttackSeedGroup,
    ComponentIdentifier,
    ScenarioDatasetSummary,
    ScenarioRunSizeEstimateStatus,
    SeedObjective,
)
from pyrit.prompt_target import PromptTarget
from pyrit.scenario.core import BaselineAttackPolicy, DatasetAttackConfiguration, Scenario, ScenarioTechnique
from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive
from pyrit.scenario.scenarios.airt.jailbreak import Jailbreak
from pyrit.scenario.scenarios.airt.psychosocial import Psychosocial
from pyrit.scenario.scenarios.benchmark.adversarial import AdversarialBenchmark
from pyrit.scenario.scenarios.garak.encoding import Encoding
from pyrit.scenario.scenarios.garak.web_injection import WebInjection
from pyrit.score import TrueFalseScorer


class _TwoTechniqueDefault(ScenarioTechnique):
    """Two concrete defaults used by estimate-only test scenarios."""

    ALL = ("all", {"all"})
    DEFAULT = ("default", {"default"})
    ONE = ("one", {"default"})
    TWO = ("two", {"default"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """Return aggregate tags."""
        return {"all", "default"}

    @classmethod
    def default(cls) -> "_TwoTechniqueDefault":
        """Return the default aggregate."""
        return cls.DEFAULT


class _JailbreakDefault(ScenarioTechnique):
    """Jailbreak's two default delivery techniques."""

    ALL = ("all", {"all"})
    DEFAULT = ("default", {"default"})
    PROMPT_SENDING = ("prompt_sending", {"default"})
    SYSTEM_PROMPT = ("jailbreak_system_prompt", {"default"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """Return aggregate tags."""
        return {"all", "default"}

    @classmethod
    def default(cls) -> "_JailbreakDefault":
        """Return the default aggregate."""
        return cls.DEFAULT


class _MatrixEstimateScenario(Scenario):
    """Minimal ordinary default technique sweep."""

    BASELINE_ATTACK_POLICY: ClassVar[BaselineAttackPolicy] = BaselineAttackPolicy.Enabled

    def __init__(self, *, objective_scorer: TrueFalseScorer) -> None:
        super().__init__(
            version=1,
            technique_class=_TwoTechniqueDefault,
            default_dataset_config=DatasetAttackConfiguration(dataset_names=["sample"]),
            objective_scorer=objective_scorer,
        )

    async def _resolve_seed_groups_by_dataset_async(
        self, *, apply_sampling: bool = True
    ) -> dict[str, list[AttackSeedGroup]]:
        """Return three logical groups before selection and two after."""
        if self._dataset_config.dataset_names == ["sample"]:
            values = ["one", "two"] if apply_sampling else ["one", "two", "three"]
            return {"sample": [_seed_group(value) for value in values]}
        return await super()._resolve_seed_groups_by_dataset_async(apply_sampling=apply_sampling)

    async def _build_atomic_attacks_async(self, *, context):
        """Return no attacks; only estimation is exercised."""
        return []


def _scorer() -> MagicMock:
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.get_identifier.return_value = ComponentIdentifier(class_name="MockScorer", class_module="test")
    return scorer


def _seed_group(value: str) -> AttackSeedGroup:
    return AttackSeedGroup(seeds=[SeedObjective(value=value)])


def _resolved_groups(
    counts: dict[str, int],
) -> tuple[dict[str, list[AttackSeedGroup]], list[ScenarioDatasetSummary]]:
    groups = {name: [_seed_group(f"{name}-{index}") for index in range(count)] for name, count in counts.items()}
    summaries = [
        ScenarioDatasetSummary(
            name=name,
            logical_seed_group_count=count,
            selected_seed_group_count=count,
        )
        for name, count in counts.items()
    ]
    return groups, summaries


@pytest.mark.usefixtures("patch_central_database")
async def test_ordinary_matrix_estimate_uses_planned_seed_units_and_baseline() -> None:
    """The base estimate is selected seed groups times concrete defaults plus baseline."""
    estimate = await _MatrixEstimateScenario(objective_scorer=_scorer()).get_default_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert estimate.total_attack_count == 6
    assert [component.count for component in estimate.components] == [4, 2]
    assert estimate.datasets[0].logical_seed_group_count == 3
    assert estimate.datasets[0].selected_seed_group_count == 2


@pytest.mark.usefixtures("patch_central_database")
async def test_configured_estimate_reuses_technique_and_baseline_resolution_without_persistence(
    patch_central_database,
) -> None:
    """A configured estimate expands only selected inputs and creates no ScenarioResult."""
    scenario = _MatrixEstimateScenario(objective_scorer=_scorer())
    scenario.set_params_from_args(
        args={
            "scenario_techniques": [_TwoTechniqueDefault.ONE],
            "include_baseline": False,
        }
    )

    estimate = await scenario.get_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert estimate.total_attack_count == 2
    assert [component.count for component in estimate.components] == [2]
    assert [factor.count for factor in estimate.components[0].factors] == [2, 1]
    assert patch_central_database.return_value.get_scenario_results() == []


@pytest.mark.usefixtures("patch_central_database")
async def test_configured_estimate_applies_dataset_selection_and_cap() -> None:
    """Configured estimates use the requested dataset population rather than scenario defaults."""
    scenario = _MatrixEstimateScenario(objective_scorer=_scorer())
    scenario.set_params_from_args(
        args={
            "dataset_config": DatasetAttackConfiguration(
                seed_groups=[_seed_group("one"), _seed_group("two"), _seed_group("three")],
                max_dataset_size=2,
            ),
            "scenario_techniques": [_TwoTechniqueDefault.ONE],
            "include_baseline": False,
        }
    )

    estimate = await scenario.get_run_size_estimate_async()

    assert estimate.total_attack_count == 2
    assert len(estimate.datasets) == 1
    assert estimate.datasets[0].logical_seed_group_count == 3
    assert estimate.datasets[0].selected_seed_group_count == 2


@pytest.mark.usefixtures("patch_central_database")
async def test_adaptive_estimate_is_target_conditional_and_does_not_multiply_techniques() -> None:
    """Adaptive techniques are selected internally rather than forming an outer axis."""
    with patch.object(TextAdaptive, "get_technique_class", return_value=_TwoTechniqueDefault):
        scenario = TextAdaptive(objective_scorer=_scorer())
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(return_value=_resolved_groups({"adaptive": 3}))

    estimate = await scenario.get_default_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Conditional
    assert estimate.total_attack_count is None
    assert [component.count for component in estimate.components] == [3, 3]


@pytest.mark.usefixtures("patch_central_database")
async def test_jailbreak_estimate_exposes_template_attempt_and_target_capability_axes() -> None:
    """Jailbreak reports guaranteed inline work separately from conditional system delivery."""
    with patch("pyrit.scenario.scenarios.airt.jailbreak._build_jailbreak_technique", return_value=_JailbreakDefault):
        scenario = Jailbreak(objective_scorer=_scorer())
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(return_value=_resolved_groups({"harmbench": 4}))

    estimate = await scenario.get_default_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Conditional
    assert estimate.total_attack_count is None
    assert [component.count for component in estimate.components] == [4, 8, 8]
    assert [factor.count for factor in estimate.components[1].factors] == [4, 2, 1, 1]
    assert "2 template(s) x 4 selected logical seed group(s) x 1 selected" in estimate.note
    assert "Baseline adds one unit per selected seed group (4 units)" in estimate.note
    assert "num_jailbreaks selects templates" in estimate.components[1].note
    assert "20" in estimate.note


@pytest.mark.usefixtures("patch_central_database")
async def test_jailbreak_configured_estimate_counts_only_prompt_sending() -> None:
    """Two templates with one selected delivery produce units per seed, not two results."""
    with patch("pyrit.scenario.scenarios.airt.jailbreak._build_jailbreak_technique", return_value=_JailbreakDefault):
        scenario = Jailbreak(objective_scorer=_scorer())
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(return_value=_resolved_groups({"harmbench": 4}))
    scenario.set_params_from_args(
        args={
            "scenario_techniques": [_JailbreakDefault.PROMPT_SENDING],
            "include_baseline": True,
            "num_jailbreaks": 2,
            "num_jailbreak_attempts": 1,
        }
    )

    estimate = await scenario.get_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert estimate.total_attack_count == 12
    assert [component.count for component in estimate.components] == [4, 8]
    assert [factor.count for factor in estimate.components[1].factors] == [4, 2, 1, 1]
    assert "2 template(s) x 4 selected logical seed group(s) x 1 selected" in estimate.note


@pytest.mark.usefixtures("patch_central_database")
async def test_jailbreak_configured_estimate_uses_target_capability() -> None:
    """A capable selected target makes native system-prompt delivery exact."""
    with patch("pyrit.scenario.scenarios.airt.jailbreak._build_jailbreak_technique", return_value=_JailbreakDefault):
        scenario = Jailbreak(objective_scorer=_scorer())
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(return_value=_resolved_groups({"harmbench": 4}))
    objective_target = MagicMock(spec=PromptTarget)
    objective_target.get_identifier.return_value = ComponentIdentifier(class_name="CapableTarget", class_module="test")
    objective_target.configuration.includes.return_value = True
    scenario.set_params_from_args(
        args={
            "objective_target": objective_target,
            "scenario_techniques": [_JailbreakDefault.SYSTEM_PROMPT],
            "include_baseline": False,
            "num_jailbreaks": 2,
            "num_jailbreak_attempts": 1,
        }
    )

    estimate = await scenario.get_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert estimate.total_attack_count == 8
    assert [component.count for component in estimate.components] == [0, 8]


@pytest.mark.usefixtures("patch_central_database")
async def test_encoding_estimate_counts_concrete_converter_and_decode_variants() -> None:
    """Encoding expands thirteen catalog techniques into fifteen concrete converter variants."""
    scenario = Encoding(objective_scorer=_scorer())
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(return_value=_resolved_groups({"encoding": 2}))

    estimate = await scenario.get_default_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert estimate.total_attack_count == 152
    assert [factor.count for factor in estimate.components[0].factors] == [2, 15, 5]


@pytest.mark.usefixtures("patch_central_database")
async def test_web_injection_estimate_uses_synthesized_technique_populations() -> None:
    """Web injection reports raw sources and capped synthesized populations separately."""
    scenario = WebInjection()
    dataset_values = {
        scenario.DATASET_EXAMPLE_DOMAINS: ["example.com", "contoso.com"],
        scenario.DATASET_MARKDOWN_JS: ["javascript:alert(1)"],
        scenario.DATASET_WEB_HTML_JS: ["<script>alert(1)</script>"],
        scenario.DATASET_NORMAL_INSTRUCTIONS: ["Write a poem.", "Explain gravity."],
    }
    with patch.object(scenario, "_load_dataset_values", return_value=dataset_values):
        estimate = await scenario.get_default_run_size_estimate_async()

    synthesized = [dataset for dataset in estimate.datasets if dataset.kind == "synthesized"]
    synthesized_count = sum(dataset.selected_seed_group_count for dataset in synthesized)
    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert len(synthesized) == len(scenario._scenario_techniques)
    assert estimate.total_attack_count == synthesized_count * 2
    assert estimate.components[-1].label == "Baseline"


@pytest.mark.usefixtures("patch_central_database")
async def test_psychosocial_estimate_keeps_sub_harm_baselines_separate() -> None:
    """Psychosocial plans each sub-harm's technique cells and baseline independently."""
    scenario = Psychosocial(
        imminent_crisis_scorer=_scorer(),
        licensed_therapist_scorer=_scorer(),
    )
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(
        return_value=_resolved_groups({"airt_imminent_crisis": 2, "airt_licensed_therapist": 1})
    )

    estimate = await scenario.get_default_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Exact
    assert estimate.total_attack_count == 12
    assert [component.count for component in estimate.components] == [6, 2, 3, 1]


@pytest.mark.usefixtures("patch_central_database")
async def test_adversarial_benchmark_estimate_exposes_per_required_target_formula() -> None:
    """Adversarial benchmark cannot claim a total before its required target count is known."""
    with patch(
        "pyrit.scenario.scenarios.benchmark.adversarial._build_benchmark_technique",
        return_value=_TwoTechniqueDefault,
    ):
        scenario = AdversarialBenchmark(objective_scorer=_scorer())
    scenario._resolve_dataset_groups_for_estimate_async = AsyncMock(return_value=_resolved_groups({"harmbench": 3}))

    estimate = await scenario.get_default_run_size_estimate_async()

    assert estimate.status is ScenarioRunSizeEstimateStatus.Conditional
    assert estimate.total_attack_count is None
    assert estimate.components[0].count == 6
    assert "adversarial_targets" in estimate.note

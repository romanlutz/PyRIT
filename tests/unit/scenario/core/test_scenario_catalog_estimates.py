# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scenario-owned catalog run-size estimates."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from pyrit.models import AttackSeedGroup, ComponentIdentifier, SeedObjective
from pyrit.scenario.core import (
    AtomicAttack,
    CompoundDatasetAttackConfiguration,
    DatasetAttackConfiguration,
    Scenario,
    ScenarioTechnique,
)
from pyrit.scenario.core.scenario_context import ScenarioContext
from pyrit.scenario.scenarios.adaptive.text_adaptive import TextAdaptive
from pyrit.scenario.scenarios.airt.jailbreak import Jailbreak
from pyrit.scenario.scenarios.airt.psychosocial import Psychosocial
from pyrit.scenario.scenarios.benchmark.adversarial import AdversarialBenchmark
from pyrit.scenario.scenarios.garak.encoding import Encoding
from pyrit.scenario.scenarios.garak.web_injection import WebInjection
from pyrit.score import TrueFalseScorer


class _CatalogTechnique(ScenarioTechnique):
    ALL = ("all", {"all"})
    DEFAULT = ("default", {"default"})
    ONE = ("one", {"default"})
    TWO = ("two", {"default"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        return {"all", "default"}

    @classmethod
    def default(cls) -> "_CatalogTechnique":
        return cls.DEFAULT


class _MatrixScenario(Scenario):
    def __init__(self) -> None:
        scorer = MagicMock(spec=TrueFalseScorer)
        scorer.get_identifier.return_value = ComponentIdentifier(class_name="CatalogScorer", class_module="test")
        super().__init__(
            version=1,
            technique_class=_CatalogTechnique,
            default_dataset_config=DatasetAttackConfiguration(
                seed_groups=[AttackSeedGroup(seeds=[SeedObjective(value=f"seed-{index}")]) for index in range(4)],
                max_dataset_size=2,
            ),
            objective_scorer=scorer,
        )

    async def _build_atomic_attacks_async(self, *, context: ScenarioContext) -> list[AtomicAttack]:
        del context
        return []


def _groups(count: int, *, prefix: str) -> list[AttackSeedGroup]:
    return [AttackSeedGroup(seeds=[SeedObjective(value=f"{prefix}-{index}")]) for index in range(count)]


def _techniques(*values: str) -> list[ScenarioTechnique]:
    return cast("list[ScenarioTechnique]", [SimpleNamespace(value=value) for value in values])


@pytest.mark.usefixtures("patch_central_database")
async def test_default_catalog_details_use_real_resolution_and_default_techniques() -> None:
    scenario = _MatrixScenario()

    summaries, estimate = await scenario.get_default_catalog_details_async()

    assert summaries == []
    assert estimate.status == "exact"
    assert estimate.total == 6
    assert [factor.count for factor in estimate.components[1].factors] == [2, 2]


async def test_matrix_estimate_honors_compound_child_caps() -> None:
    compound = CompoundDatasetAttackConfiguration(
        configurations=[
            DatasetAttackConfiguration(seed_groups=_groups(5, prefix="a"), max_dataset_size=2),
            DatasetAttackConfiguration(seed_groups=_groups(5, prefix="b"), max_dataset_size=3),
        ]
    )
    selected = await compound.get_attack_groups_by_dataset_async()
    scenario = object.__new__(_MatrixScenario)

    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset=selected,
        techniques=_techniques("one", "two"),
    )

    assert sum(len(groups) for groups in selected.values()) == 5
    assert estimate.status == "exact"
    assert estimate.total == 15
    assert estimate.components[0].label == "baseline"
    assert estimate.components[0].count == 5


def test_default_dataset_summaries_preserve_names_and_counts() -> None:
    scenario = object.__new__(_MatrixScenario)
    scenario._default_dataset_config = DatasetAttackConfiguration(dataset_names=["a", "b"])

    summaries = scenario._build_default_dataset_summaries(
        full_groups={"a": _groups(4, prefix="a"), "b": _groups(3, prefix="b")},
        selected_groups={"a": _groups(2, prefix="a"), "b": _groups(1, prefix="b")},
    )

    assert [summary.model_dump() for summary in summaries] == [
        {"name": "a", "seed_group_count": 4, "selected_seed_group_count": 2},
        {"name": "b", "seed_group_count": 3, "selected_seed_group_count": 1},
    ]


def test_adaptive_estimate_is_one_sequential_unit_per_seed_plus_baseline() -> None:
    scenario = object.__new__(TextAdaptive)
    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset={"dataset": _groups(3, prefix="seed")},
        techniques=_techniques("one", "two"),
    )

    assert estimate.status == "exact"
    assert estimate.total == 6
    assert [component.count for component in estimate.components] == [3, 3]


def test_jailbreak_default_estimate_is_target_conditional() -> None:
    scenario = object.__new__(Jailbreak)
    scenario.params = {"num_jailbreaks": 2, "num_jailbreak_attempts": 1}
    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset={"harmbench": _groups(4, prefix="seed")},
        techniques=_techniques("prompt_sending", "jailbreak_system_prompt"),
    )

    assert estimate.status == "conditional"
    assert estimate.total is None
    assert [component.count for component in estimate.components] == [4, 8, None]


def test_encoding_estimate_multiplies_variants_decode_configs_and_seeds() -> None:
    scenario = object.__new__(Encoding)
    scenario._encoding_templates = ("decode-a", "decode-b")
    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset={"a": _groups(2, prefix="a"), "b": _groups(3, prefix="b")},
        techniques=_techniques("base64", "ascii85", "rot13"),
    )

    assert estimate.status == "exact"
    assert estimate.total == 80
    assert [factor.count for factor in estimate.components[1].factors] == [5, 3, 5]


def test_web_injection_estimate_uses_synthesized_per_technique_groups() -> None:
    scenario = object.__new__(WebInjection)
    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset={
            "markdown_image_exfil": _groups(2, prefix="image"),
            "task_xss": _groups(3, prefix="xss"),
        },
        techniques=_techniques("markdown_image_exfil", "task_xss"),
    )

    assert estimate.status == "exact"
    assert estimate.total == 10
    assert [component.count for component in estimate.components] == [5, 2, 3]


def test_psychosocial_estimate_emits_per_subharm_baselines() -> None:
    scenario = object.__new__(Psychosocial)
    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset={
            "airt_imminent_crisis": _groups(2, prefix="crisis"),
            "airt_licensed_therapist": _groups(1, prefix="therapist"),
        },
        techniques=_techniques("none", "tone_soften"),
    )

    assert estimate.status == "exact"
    assert estimate.total == 9
    assert [component.count for component in estimate.components] == [2, 1, 4, 2]


def test_adversarial_benchmark_estimate_is_unavailable_without_target_axis() -> None:
    scenario = object.__new__(AdversarialBenchmark)
    estimate = scenario._estimate_default_run_size(
        seed_groups_by_dataset={"harmbench": _groups(4, prefix="seed")},
        techniques=_techniques("red_teaming"),
    )

    assert estimate.status == "unavailable"
    assert estimate.total is None
    assert estimate.components[0].count == 0
    assert estimate.components[1].factors[1].label == "adversarial targets"
    assert estimate.components[1].factors[1].count is None

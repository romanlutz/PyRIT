# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Focused formula tests for non-matrix Scenario run-size shapes."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pyrit.models import (
    AttackSeedGroup,
    ScenarioDatasetSummary,
    ScenarioRunSizeStatus,
    SeedObjective,
)
from pyrit.scenario.core.dataset_configuration import DatasetAttackConfiguration
from pyrit.scenario.core.scenario_run_size import ScenarioRunSizeContext
from pyrit.scenario.scenarios.adaptive.adaptive_scenario import AdaptiveScenario
from pyrit.scenario.scenarios.airt.psychosocial import Psychosocial, PsychosocialTechnique
from pyrit.scenario.scenarios.benchmark.adversarial import AdversarialBenchmark
from pyrit.scenario.scenarios.garak.encoding import Encoding, EncodingTechnique
from pyrit.scenario.scenarios.garak.web_injection import WebInjection, WebInjectionTechnique


def _groups(count: int) -> list[AttackSeedGroup]:
    """Build uniquely identified logical groups."""
    return [AttackSeedGroup(seeds=[SeedObjective(value=f"objective {index}")]) for index in range(count)]


def _context(
    *,
    techniques: list,
    groups_by_source: dict[str, list[AttackSeedGroup]],
    include_baseline: bool,
    target_is_configured: bool = False,
    objective_target=None,
) -> ScenarioRunSizeContext:
    """Build a resolved run-size context for formula-only tests."""
    summaries = [
        ScenarioDatasetSummary(
            name=name,
            logical_group_count=len(groups),
            selected_group_count=len(groups),
        )
        for name, groups in groups_by_source.items()
    ]
    return ScenarioRunSizeContext(
        scenario_techniques=techniques,
        dataset_config=DatasetAttackConfiguration(dataset_names=list(groups_by_source)),
        seed_groups_by_source=groups_by_source,
        logical_seed_groups_by_source=groups_by_source,
        dataset_summaries=summaries,
        include_baseline=include_baseline,
        objective_target=objective_target,
        target_is_configured=target_is_configured,
    )


def test_encoding_counts_converter_and_decode_variants() -> None:
    """Encoding multiplies converter variants by raw/decode forms and logical groups."""
    scenario = SimpleNamespace(
        _encoding_templates=["decode one", "decode two"],
        _converter_variants=lambda: [
            ([], "base64", "base64"),
            ([], "base64", "base64_urlsafe"),
        ],
    )
    context = _context(
        techniques=[EncodingTechnique.Base64],
        groups_by_source={"payloads": _groups(3)},
        include_baseline=True,
    )

    estimate = Encoding._estimate_run_size(scenario, context=context)  # type: ignore[arg-type]

    assert estimate.total_planned_executions == 21
    assert estimate.components[0].is_baseline is True
    assert [factor.count for factor in estimate.components[1].factors] == [1, 2, 3, 3]


def test_web_injection_is_additive_per_synthesized_technique_population() -> None:
    """Web Injection sizes each selected technique's synthesized groups independently."""
    context = _context(
        techniques=[WebInjectionTechnique.MarkdownXSS, WebInjectionTechnique.TaskXSS],
        groups_by_source={
            WebInjectionTechnique.MarkdownXSS.value: _groups(2),
            WebInjectionTechnique.TaskXSS.value: _groups(3),
        },
        include_baseline=True,
    )

    estimate = WebInjection._estimate_run_size(SimpleNamespace(), context=context)  # type: ignore[arg-type]

    assert estimate.total_planned_executions == 10
    assert [component.planned_executions for component in estimate.components] == [5, 2, 3]


def test_psychosocial_adds_each_subharm_cell_and_baseline() -> None:
    """Psychosocial emits one technique cell and one baseline for every selected sub-harm."""
    scenario = object.__new__(Psychosocial)
    scenario.params = {"sub_harm": "all"}
    context = _context(
        techniques=[PsychosocialTechnique.NoConverter],
        groups_by_source={
            "airt_imminent_crisis": _groups(2),
            "airt_licensed_therapist": _groups(3),
        },
        include_baseline=True,
    )

    estimate = scenario._estimate_run_size(context=context)

    assert estimate.total_planned_executions == 10
    assert [component.is_baseline for component in estimate.components] == [True, False, True, False]


def test_adaptive_without_target_is_conditional_and_excludes_inner_attempts() -> None:
    """Adaptive candidate components stay non-authoritative until target compatibility is known."""
    scenario = SimpleNamespace(params={"max_attempts_per_objective": 3})
    context = _context(
        techniques=[],
        groups_by_source={"objectives": _groups(4)},
        include_baseline=False,
    )

    estimate = AdaptiveScenario._estimate_run_size(scenario, context=context)  # type: ignore[arg-type]

    assert estimate.status is ScenarioRunSizeStatus.CONDITIONAL
    assert estimate.total_planned_executions is None
    assert estimate.components[0].planned_executions == 4
    assert "inner attempts" in (estimate.caveat or "")


def test_adaptive_counts_only_compatible_outer_envelopes() -> None:
    """Adaptive exact sizing counts compatible envelopes but not their inner attempts."""
    target = MagicMock()
    scenario = SimpleNamespace(
        params={"max_attempts_per_objective": 3},
        _selector=MagicMock(),
        _objective_scorer=MagicMock(),
        _scenario_result_id=None,
        _build_techniques_dict=lambda **_: {"technique": MagicMock()},
    )
    context = _context(
        techniques=[],
        groups_by_source={"objectives": _groups(4)},
        include_baseline=True,
        target_is_configured=True,
        objective_target=target,
    )
    dispatcher = MagicMock()
    dispatcher.compatible_techniques.side_effect = [["technique"], [], ["technique"], ["technique"]]

    with patch(
        "pyrit.scenario.scenarios.adaptive.adaptive_scenario.AdaptiveTechniqueDispatcher",
        return_value=dispatcher,
    ):
        estimate = AdaptiveScenario._estimate_run_size(scenario, context=context)  # type: ignore[arg-type]

    assert estimate.status is ScenarioRunSizeStatus.EXACT
    assert [component.planned_executions for component in estimate.components] == [4, 3]
    assert estimate.total_planned_executions == 7


def test_adversarial_benchmark_without_target_axis_is_unavailable() -> None:
    """The adversarial benchmark does not claim a total before its required target axis exists."""
    scenario = SimpleNamespace(params={})
    context = _context(
        techniques=[],
        groups_by_source={"harmbench": _groups(2)},
        include_baseline=False,
    )

    estimate = AdversarialBenchmark._estimate_run_size(scenario, context=context)  # type: ignore[arg-type]

    assert estimate.status is ScenarioRunSizeStatus.UNAVAILABLE
    assert estimate.total_planned_executions is None
    assert "adversarial_targets" in (estimate.caveat or "")


def test_adversarial_benchmark_multiplies_required_target_axis() -> None:
    """The benchmark target list is an explicit multiplicative axis."""
    technique = SimpleNamespace(value="red_teaming")
    groups = _groups(3)
    scenario = SimpleNamespace(
        params={"adversarial_targets": ["model-a", "model-b"]},
        _use_cached=False,
        _resolve_adversarial_targets=lambda **_: [("model-a", MagicMock()), ("model-b", MagicMock())],
    )
    context = _context(
        techniques=[technique],
        groups_by_source={"harmbench": groups},
        include_baseline=False,
    )

    with patch(
        "pyrit.scenario.scenarios.benchmark.adversarial.resolve_selected_factories_and_compatible_groups",
        return_value=(
            {"red_teaming": MagicMock()},
            {"red_teaming": {"harmbench": groups}},
        ),
    ):
        estimate = AdversarialBenchmark._estimate_run_size(scenario, context=context)  # type: ignore[arg-type]

    assert estimate.status is ScenarioRunSizeStatus.EXACT
    assert estimate.total_planned_executions == 6
    assert [factor.count for factor in estimate.components[0].factors] == [1, 2, 3]

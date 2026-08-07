# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for ScenarioRegistry._build_metadata and create_and_initialize_async."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import Parameter, ScenarioRunSizeEstimate, ScenarioRunSizeStatus
from pyrit.registry.components.scenario_registry import ScenarioRegistry
from pyrit.scenario.core import BaselineAttackPolicy
from pyrit.scenario.core.scenario_technique import ScenarioTechnique


class _MetadataTechnique(ScenarioTechnique):
    """Technique enum for metadata expansion tests."""

    ALL = ("all", {"all"})
    DEFAULT = ("default", {"default"})
    ALPHA = ("alpha", {"default"})
    BETA = ("beta", {"default"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        """Return aggregate tags."""
        return {"all", "default"}


class _MetadataScenario:
    """Scenario-like class with an aggregate default."""

    BASELINE_ATTACK_POLICY = BaselineAttackPolicy.Enabled

    @classmethod
    def supported_parameters(cls) -> list[Parameter]:
        """Return no custom parameters."""
        return []

    def __init__(self) -> None:
        self._technique_class = _MetadataTechnique
        self._default_technique = _MetadataTechnique.DEFAULT
        self._default_dataset_config = SimpleNamespace(dataset_names=["dataset"])

    def _resolve_scenario_techniques(self, *, scenario_techniques: Any) -> list[_MetadataTechnique]:
        return _MetadataTechnique.resolve(scenario_techniques, default=self._default_technique)


class _NotNoArgScenario:
    """A scenario-like stub whose constructor requires arguments."""

    @classmethod
    def supported_parameters(cls):
        return []

    def __init__(self, *, required_arg) -> None:
        self.required_arg = required_arg


def test_build_metadata_raises_when_scenario_requires_constructor_args() -> None:
    """Scenarios that cannot be instantiated with no args must surface a clear error."""
    registry = ScenarioRegistry()

    with pytest.raises(TypeError, match="must be instantiable with no arguments"):
        registry._build_metadata("not_no_arg", _NotNoArgScenario)


def test_build_metadata_expands_default_and_aggregate_techniques() -> None:
    """Catalog metadata exposes concrete defaults and every aggregate expansion."""
    metadata = ScenarioRegistry()._build_metadata("metadata.scenario", _MetadataScenario)

    assert metadata.default_techniques == ("alpha", "beta")
    assert dict(metadata.aggregate_technique_expansions) == {
        "all": ("alpha", "beta"),
        "default": ("alpha", "beta"),
    }


async def test_create_and_initialize_async_creates_sets_params_and_initializes() -> None:
    """The registry owns build + set-params + initialize and returns the scenario."""
    registry = ScenarioRegistry()

    scenario = MagicMock()
    scenario.initialize_async = AsyncMock()
    target = MagicMock()

    registry.create_instance = MagicMock(return_value=scenario)  # type: ignore[method-assign]

    result = await registry.create_and_initialize_async(
        "my.scenario",
        scenario_params={"foo": "bar"},
        scenario_result_id="sr-1",
        objective_target=target,
        max_concurrency=2,
    )

    assert result is scenario
    registry.create_instance.assert_called_once_with("my.scenario", scenario_result_id="sr-1")
    scenario.set_scenario_registry_name.assert_called_once_with("my.scenario")
    scenario.set_params_from_args.assert_called_once_with(
        args={"foo": "bar", "objective_target": target, "max_concurrency": 2}
    )
    scenario.initialize_async.assert_awaited_once_with()


async def test_create_and_initialize_async_omits_result_id_when_none() -> None:
    """When no scenario_result_id is supplied, it is not forwarded to the constructor."""
    registry = ScenarioRegistry()

    scenario = MagicMock()
    scenario.initialize_async = AsyncMock()
    registry.create_instance = MagicMock(return_value=scenario)  # type: ignore[method-assign]

    target = MagicMock()
    await registry.create_and_initialize_async("my.scenario", objective_target=target)

    registry.create_instance.assert_called_once_with("my.scenario")
    scenario.set_scenario_registry_name.assert_called_once_with("my.scenario")
    scenario.set_params_from_args.assert_called_once_with(args={"objective_target": target})
    scenario.initialize_async.assert_awaited_once_with()


async def test_create_and_estimate_async_uses_configuration_without_initializing() -> None:
    """Estimation shares registry coercion but never initializes or persists a run."""
    registry = ScenarioRegistry()
    estimate = ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.EXACT,
        total_planned_executions=0,
        components=[],
    )
    scenario = MagicMock()
    scenario.estimate_run_size_async = AsyncMock(return_value=estimate)
    scenario.initialize_async = AsyncMock()
    target = MagicMock()
    registry.create_instance = MagicMock(return_value=scenario)  # type: ignore[method-assign]

    result = await registry.create_and_estimate_async(
        "my.scenario",
        scenario_params={"attempts": 2},
        target_is_configured=False,
        objective_target=target,
        include_baseline=False,
    )

    assert result is estimate
    registry.create_instance.assert_called_once_with("my.scenario")
    scenario.set_params_from_args.assert_called_once_with(
        args={"attempts": 2, "objective_target": target, "include_baseline": False}
    )
    scenario.estimate_run_size_async.assert_awaited_once_with(target_is_configured=False)
    scenario.initialize_async.assert_not_awaited()

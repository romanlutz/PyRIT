# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend scenario service and routes.
"""

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pyrit.backend.main import app
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.scenarios import ListRegisteredScenariosResponse
from pyrit.backend.services.scenario_service import (
    ScenarioService,
    get_scenario_service,
)
from pyrit.models import (
    Parameter,
    ScenarioDatasetSummary,
    ScenarioEstimateRequest,
    ScenarioRunSizeEstimate,
    ScenarioRunSizeStatus,
)
from pyrit.models.catalog.scenario import RegisteredScenario
from pyrit.registry import ScenarioMetadata


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_service_cache():
    """Clear the scenario service singleton cache between tests."""
    get_scenario_service.cache_clear()
    yield
    get_scenario_service.cache_clear()


def _make_scenario_metadata(
    *,
    registry_name: str = "test.scenario",
    class_name: str = "TestScenario",
    description: str = "A test scenario",
    description_markdown: str = "A test scenario",
    default_technique: str = "default",
    default_techniques: tuple[str, ...] = ("role_play",),
    all_techniques: tuple[str, ...] = ("role_play", "many_shot"),
    aggregate_techniques: tuple[str, ...] = ("all", "default"),
    aggregate_technique_expansions: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("all", ("role_play", "many_shot")),
        ("default", ("role_play",)),
    ),
    default_datasets: tuple[str, ...] = ("test_dataset",),
    baseline_policy: str = "enabled",
    include_baseline_by_default: bool = True,
) -> ScenarioMetadata:
    """Create a ScenarioMetadata instance for testing."""
    return ScenarioMetadata(
        registry_name=registry_name,
        class_name=class_name,
        class_module="pyrit.scenario.scenarios.test",
        class_description=description,
        class_description_markdown=description_markdown,
        default_technique=default_technique,
        default_techniques=default_techniques,
        all_techniques=all_techniques,
        aggregate_techniques=aggregate_techniques,
        aggregate_technique_expansions=aggregate_technique_expansions,
        default_datasets=default_datasets,
        baseline_policy=baseline_policy,
        include_baseline_by_default=include_baseline_by_default,
    )


def _exact_estimate(*, source_name: str = "test_dataset") -> ScenarioRunSizeEstimate:
    """Build a minimal exact default estimate for catalog tests."""
    return ScenarioRunSizeEstimate(
        status=ScenarioRunSizeStatus.EXACT,
        total_planned_executions=0,
        components=[],
        datasets=[
            ScenarioDatasetSummary(
                name=source_name,
                logical_group_count=0,
                selected_group_count=0,
            )
        ],
    )


# ============================================================================
# ScenarioService Unit Tests
# ============================================================================


class TestScenarioServiceListScenarios:
    """Tests for ScenarioService.list_scenarios_async."""

    async def test_list_scenarios_returns_empty_when_no_scenarios(self) -> None:
        """Test that list returns empty list when no scenarios are registered."""
        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = []

            result = await service.list_scenarios_async()

            assert result.items == []
            assert result.pagination.has_more is False

    async def test_list_scenarios_returns_scenarios_from_registry(self) -> None:
        """Test that list returns scenarios from registry."""
        metadata = _make_scenario_metadata()

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = [metadata]

            result = await service.list_scenarios_async()

            assert len(result.items) == 1
            assert result.items[0].scenario_name == "test.scenario"
            assert result.items[0].scenario_type == "TestScenario"
            assert result.items[0].description == "A test scenario"
            assert result.items[0].description_markdown == "A test scenario"
            assert result.items[0].default_technique == "default"
            assert result.items[0].default_techniques == ["role_play"]
            assert result.items[0].aggregate_techniques == ["all", "default"]
            assert result.items[0].aggregate_technique_expansions["default"] == ["role_play"]
            assert result.items[0].all_techniques == ["role_play", "many_shot"]
            assert result.items[0].default_datasets == ["test_dataset"]
            assert result.items[0].baseline_policy == "enabled"
            assert result.items[0].include_baseline_by_default is True

    async def test_list_scenarios_preserves_disabled_baseline_policy(self) -> None:
        metadata = _make_scenario_metadata(
            baseline_policy="disabled",
            include_baseline_by_default=False,
        )

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = [metadata]

            result = await service.list_scenarios_async()

        assert result.items[0].baseline_policy == "disabled"
        assert result.items[0].include_baseline_by_default is False

    async def test_list_scenarios_paginates_with_limit(self) -> None:
        """Test that list respects the limit parameter."""
        metadata_list = [
            _make_scenario_metadata(registry_name=f"test.scenario_{i}", class_name=f"Scenario{i}") for i in range(5)
        ]

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = metadata_list

            result = await service.list_scenarios_async(limit=3)

            assert len(result.items) == 3
            assert result.pagination.has_more is True
            assert result.pagination.next_cursor == "test.scenario_2"

    async def test_list_scenarios_paginates_with_cursor(self) -> None:
        """Test that list uses cursor for pagination."""
        metadata_list = [
            _make_scenario_metadata(registry_name=f"test.scenario_{i}", class_name=f"Scenario{i}") for i in range(5)
        ]

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = metadata_list

            result = await service.list_scenarios_async(limit=2, cursor="test.scenario_1")

            assert len(result.items) == 2
            assert result.items[0].scenario_name == "test.scenario_2"
            assert result.items[1].scenario_name == "test.scenario_3"
            assert result.pagination.has_more is True

    async def test_list_scenarios_last_page_has_more_false(self) -> None:
        """Test that last page shows has_more=False."""
        metadata_list = [
            _make_scenario_metadata(registry_name=f"test.scenario_{i}", class_name=f"Scenario{i}") for i in range(3)
        ]

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = metadata_list

            result = await service.list_scenarios_async(limit=5)

            assert len(result.items) == 3
            assert result.pagination.has_more is False
            assert result.pagination.next_cursor is None


class TestScenarioServiceGetScenario:
    """Tests for ScenarioService.get_scenario_async."""

    async def test_get_scenario_returns_matching_scenario(self) -> None:
        """Test that get returns the matching scenario."""
        metadata = _make_scenario_metadata(registry_name="foundry.red_team_agent")

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_registered_class_metadata.return_value = metadata

            result = await service.get_scenario_async(scenario_name="foundry.red_team_agent")

            assert result is not None
            assert result.scenario_name == "foundry.red_team_agent"

    async def test_get_scenario_returns_none_for_missing(self) -> None:
        """Test that get returns None when scenario not found."""
        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_registered_class_metadata.return_value = None

            result = await service.get_scenario_async(scenario_name="nonexistent")

            assert result is None

    async def test_default_estimate_is_cached_across_detail_requests(self) -> None:
        """Repeated catalog reads reuse the expensive default estimate."""
        metadata = _make_scenario_metadata()
        service = ScenarioService()
        service._registry = MagicMock()
        service._registry.get_registered_class_metadata.return_value = metadata
        service._estimate_scenario_in_worker = MagicMock(return_value=_exact_estimate())  # type: ignore[method-assign]

        first = await service.get_scenario_async(scenario_name=metadata.registry_name)
        second = await service.get_scenario_async(scenario_name=metadata.registry_name)

        assert first is not None
        assert second is not None
        assert first.default_estimate == _exact_estimate()
        assert first.default_dataset_summaries == _exact_estimate().datasets
        service._estimate_scenario_in_worker.assert_called_once()

    async def test_one_default_estimate_failure_is_unavailable_without_breaking_list(self) -> None:
        """Catalog discovery isolates one scenario's planning failure."""
        metadata = [
            _make_scenario_metadata(registry_name="bad.scenario"),
            _make_scenario_metadata(registry_name="good.scenario"),
        ]
        service = ScenarioService()
        service._registry = MagicMock()
        service._registry.get_all_registered_class_metadata.return_value = metadata

        def _estimate(name: str, request: ScenarioEstimateRequest) -> ScenarioRunSizeEstimate:
            if name == "bad.scenario":
                raise RuntimeError("broken sizing")
            return _exact_estimate()

        service._estimate_scenario_in_worker = MagicMock(side_effect=_estimate)  # type: ignore[method-assign]

        result = await service.list_scenarios_async()

        estimates = {item.scenario_name: item.default_estimate for item in result.items}
        assert estimates["bad.scenario"] is not None
        assert estimates["bad.scenario"].status is ScenarioRunSizeStatus.UNAVAILABLE
        assert estimates["good.scenario"] == _exact_estimate()


class TestScenarioServiceEstimate:
    """Tests for launch-aligned estimate orchestration."""

    async def test_estimate_core_reuses_configuration_mapping_and_registry_lifecycle(self) -> None:
        """The service maps selections once and delegates planning to ScenarioRegistry."""
        service = ScenarioService()
        scenario_class = MagicMock()
        target = MagicMock()
        estimate = _exact_estimate()
        service._registry = MagicMock()
        service._registry.get_class.return_value = scenario_class
        service._registry.create_and_estimate_async = AsyncMock(return_value=estimate)
        service._configuration_service = MagicMock()
        service._configuration_service.build_initialization_kwargs.return_value = {
            "objective_target": target,
            "scenario_techniques": ["resolved"],
            "include_baseline": False,
        }
        request = ScenarioEstimateRequest(
            target_name="target",
            techniques=["prompt_sending"],
            include_baseline=False,
            scenario_params={"num_jailbreaks": 2},
        )
        target_registry = MagicMock()
        target_registry.instances.get.return_value = target

        with patch("pyrit.backend.services.scenario_service.TargetRegistry.get_registry_singleton") as get_registry:
            get_registry.return_value = target_registry
            result = await service._estimate_scenario_core_async(
                scenario_name="airt.jailbreak",
                request=request,
            )

        assert result is estimate
        service._configuration_service.build_initialization_kwargs.assert_called_once_with(
            configuration=request,
            scenario_name="airt.jailbreak",
            scenario_class=scenario_class,
            objective_target=target,
        )
        service._registry.create_and_estimate_async.assert_awaited_once_with(
            "airt.jailbreak",
            scenario_params={"num_jailbreaks": 2},
            target_is_configured=True,
            objective_target=target,
            scenario_techniques=["resolved"],
            include_baseline=False,
        )


# ============================================================================
# Route Tests
# ============================================================================


class TestScenarioRoutes:
    """Tests for scenario API routes."""

    def test_list_scenarios_returns_200(self, client: TestClient) -> None:
        """Test that GET /api/scenarios/catalog returns 200."""
        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_scenarios_async = AsyncMock(
                return_value=ListRegisteredScenariosResponse(
                    items=[],
                    pagination=PaginationInfo(limit=50, has_more=False, next_cursor=None, prev_cursor=None),
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/scenarios/catalog")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["items"] == []
            assert data["pagination"]["has_more"] is False

    def test_list_scenarios_with_items(self, client: TestClient) -> None:
        """Test that GET /api/scenarios/catalog returns scenario data."""
        summary = RegisteredScenario(
            scenario_name="foundry.red_team_agent",
            scenario_type="RedTeamAgentScenario",
            description="Red team agent testing",
            default_technique="default",
            aggregate_techniques=["all", "default"],
            all_techniques=["role_play", "many_shot"],
            default_datasets=["airt_hate"],
        )

        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_scenarios_async = AsyncMock(
                return_value=ListRegisteredScenariosResponse(
                    items=[summary],
                    pagination=PaginationInfo(limit=50, has_more=False, next_cursor=None, prev_cursor=None),
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/scenarios/catalog")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["items"]) == 1
            item = data["items"][0]
            assert item["scenario_name"] == "foundry.red_team_agent"
            assert item["scenario_type"] == "RedTeamAgentScenario"
            assert item["default_technique"] == "default"
            assert item["aggregate_techniques"] == ["all", "default"]
            assert item["all_techniques"] == ["role_play", "many_shot"]
            assert item["default_datasets"] == ["airt_hate"]

    def test_list_scenarios_passes_pagination_params(self, client: TestClient) -> None:
        """Test that pagination params are forwarded to service."""
        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.list_scenarios_async = AsyncMock(
                return_value=ListRegisteredScenariosResponse(
                    items=[],
                    pagination=PaginationInfo(limit=10, has_more=False, next_cursor=None, prev_cursor=None),
                )
            )
            mock_get_service.return_value = mock_service

            response = client.get("/api/scenarios/catalog?limit=10&cursor=test.scenario_1")

            assert response.status_code == status.HTTP_200_OK
            mock_service.list_scenarios_async.assert_called_once_with(limit=10, cursor="test.scenario_1")

    def test_estimate_scenario_returns_structured_preview(self, client: TestClient) -> None:
        """POST catalog estimate forwards launch selections and returns the typed estimate."""
        estimate = _exact_estimate(source_name="harmbench")
        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.estimate_scenario_async = AsyncMock(return_value=estimate)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/scenarios/catalog/airt.jailbreak/estimate",
                json={
                    "target_name": "target",
                    "techniques": ["prompt_sending"],
                    "include_baseline": False,
                    "scenario_params": {
                        "num_jailbreaks": 2,
                        "num_jailbreak_attempts": 1,
                    },
                },
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "exact"
        request = mock_service.estimate_scenario_async.await_args.kwargs["request"]
        assert request == ScenarioEstimateRequest(
            target_name="target",
            techniques=["prompt_sending"],
            include_baseline=False,
            scenario_params={"num_jailbreaks": 2, "num_jailbreak_attempts": 1},
        )

    def test_estimate_scenario_returns_404_for_missing_catalog_entry(self, client: TestClient) -> None:
        """POST catalog estimate reports an unknown Scenario."""
        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.estimate_scenario_async = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.post("/api/scenarios/catalog/missing/estimate", json={})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_scenario_returns_200(self, client: TestClient) -> None:
        """Test that GET /api/scenarios/catalog/{name} returns 200 when found."""
        summary = RegisteredScenario(
            scenario_name="foundry.red_team_agent",
            scenario_type="RedTeamAgentScenario",
            description="Red team agent testing",
            default_technique="default",
            aggregate_techniques=["all"],
            all_techniques=["role_play"],
            default_datasets=["airt_hate"],
        )

        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_scenario_async = AsyncMock(return_value=summary)
            mock_get_service.return_value = mock_service

            response = client.get("/api/scenarios/catalog/foundry.red_team_agent")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["scenario_name"] == "foundry.red_team_agent"

    def test_get_scenario_returns_404_when_not_found(self, client: TestClient) -> None:
        """Test that GET /api/scenarios/catalog/{name} returns 404 when not found."""
        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_scenario_async = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service

            response = client.get("/api/scenarios/catalog/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_scenario_with_dotted_name(self, client: TestClient) -> None:
        """Test that dotted scenario names (e.g., 'foundry.red_team_agent') work in path."""
        summary = RegisteredScenario(
            scenario_name="garak.encoding",
            scenario_type="EncodingScenario",
            description="Encoding scenario",
            default_technique="all",
            aggregate_techniques=["all"],
            all_techniques=["base64", "rot13"],
            default_datasets=[],
        )

        with patch("pyrit.backend.routes.scenarios.get_scenario_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_scenario_async = AsyncMock(return_value=summary)
            mock_get_service.return_value = mock_service

            response = client.get("/api/scenarios/catalog/garak.encoding")

            assert response.status_code == status.HTTP_200_OK
            mock_service.get_scenario_async.assert_called_once_with(scenario_name="garak.encoding")


# ============================================================================
# Supported Parameters Tests
# ============================================================================


class TestScenarioServiceSupportedParameters:
    """Tests for supported_parameters in scenario service responses."""

    async def test_list_scenarios_includes_supported_parameters(self) -> None:
        """Test that supported_parameters are included in scenario listing."""
        metadata = _make_scenario_metadata(registry_name="param.scenario")
        metadata = ScenarioMetadata(
            registry_name="param.scenario",
            class_name="ParamScenario",
            class_module="pyrit.scenario.scenarios.param",
            class_description="A scenario with params",
            default_technique="default",
            all_techniques=("role_play",),
            aggregate_techniques=("all",),
            default_datasets=("test_dataset",),
            supported_parameters=(
                Parameter(
                    name="max_turns",
                    description="Maximum number of turns",
                    default=5,
                    param_type=int,
                ),
                Parameter(
                    name="mode",
                    description="Execution mode",
                    default="fast",
                    param_type=Literal["fast", "slow"],
                ),
            ),
        )

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = [metadata]

            result = await service.list_scenarios_async()

            assert len(result.items) == 1
            params = result.items[0].supported_parameters
            assert len(params) == 2

            assert params[0].name == "max_turns"
            assert params[0].description == "Maximum number of turns"
            assert params[0].model_dump()["default"] == "5"
            assert params[0].type_name == "int"
            assert params[0].choices is None
            assert params[0].is_list is False

            assert params[1].name == "mode"
            assert params[1].description == "Execution mode"
            assert params[1].model_dump()["default"] == "fast"
            assert params[1].type_name == "str"
            assert params[1].choices == ["fast", "slow"]
            assert params[1].is_list is False

    async def test_scenario_with_no_parameters_has_empty_list(self) -> None:
        """Test that scenarios without parameters have empty supported_parameters."""
        metadata = _make_scenario_metadata()

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = [metadata]

            result = await service.list_scenarios_async()

            assert result.items[0].supported_parameters == []

    async def test_supported_parameters_with_none_default(self) -> None:
        """Test that parameters with None default are serialized correctly."""
        metadata = ScenarioMetadata(
            registry_name="test.scenario",
            class_name="TestScenario",
            class_module="pyrit.scenario.scenarios.test",
            class_description="Test",
            default_technique="default",
            all_techniques=("all",),
            aggregate_techniques=("all",),
            default_datasets=(),
            supported_parameters=(
                Parameter(
                    name="optional_param",
                    description="An optional param",
                    default=None,
                    param_type=str,
                ),
            ),
        )

        with patch.object(ScenarioService, "__init__", lambda self: None):
            service = ScenarioService()
            service._registry = MagicMock()
            service._registry.get_all_registered_class_metadata.return_value = [metadata]

            result = await service.list_scenarios_async()

            param = result.items[0].supported_parameters[0]
            assert param.default is None
            assert param.model_dump()["default"] is None

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Browser fixture scorer defaults must never construct or call a real provider."""

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from frontend.e2e.fixtures import analytics_backend
from pyrit.backend.services.scenario_service import ScenarioService
from pyrit.models import MessagePiece
from pyrit.registry import AttackTechniqueRegistry, InitializerRegistry, ScenarioRegistry, TargetRegistry
from pyrit.scenario.core import scenario_target_defaults
from pyrit.scenario.scenarios.technique import Execute
from pyrit.setup import ConfigurationLoader
from pyrit.setup.initialization import reset_setup_registries
from pyrit.setup.initializers import TechniqueInitializer


@pytest.fixture
def isolated_registries(patch_central_database: MagicMock) -> Iterator[None]:
    reset_setup_registries()
    try:
        yield
    finally:
        reset_setup_registries()


@pytest.mark.usefixtures("isolated_registries", "block_dataset_provider_lookup")
@pytest.mark.parametrize("fixture", ["browser", "analytics"])
async def test_fixture_catalog_uses_offline_scorer_target(fixture: str) -> None:
    if fixture == "browser":
        scripts = [Path("tests") / "end_to_end" / "browser_initializer.py"]
    else:
        scripts = []

        def capture_config(*, directory: Path, port: int) -> None:
            config = ConfigurationLoader.from_dict(
                yaml.safe_load(Path(os.environ["PYRIT_CONFIG_FILE"]).read_text(encoding="utf-8"))
            )
            assert config.env_files == []
            assert config.env_akv_ref == []
            assert config.initializers == ["technique"]
            scripts.extend(config.resolve_initialization_scripts() or [])

        with (
            patch("sys.argv", ["analytics_backend"]),
            patch.object(analytics_backend, "_serve", capture_config),
        ):
            await asyncio.to_thread(analytics_backend.main)

    with (
        patch.object(
            scenario_target_defaults,
            "OpenAIChatTarget",
            side_effect=AssertionError("Browser fixtures must not construct a provider fallback."),
        ) as fallback,
        patch.object(
            httpx.AsyncClient, "send", side_effect=AssertionError("Browser fixtures must not send HTTP requests.")
        ) as network,
        patch.object(ScenarioRegistry, "_discover", return_value=None),
    ):
        await TechniqueInitializer().initialize_async()
        initializers = await asyncio.to_thread(
            InitializerRegistry.get_registry_singleton().create_from_script_paths, script_paths=scripts, strict=True
        )
        assert len(initializers) == 1
        await initializers[0].initialize_async()
        target = TargetRegistry.get_registry_singleton().instances.get("objective_scorer_chat")
        assert target is not None
        assert target.__class__.__name__ == "OfflineScorerTarget"
        assert scenario_target_defaults.get_default_scorer_target() is target
        assert AttackTechniqueRegistry.get_registry_singleton().get_factories_or_raise()

        service = ScenarioService()
        try:
            service._registry.register_class(Execute, name="execute")
            catalog = await service.get_scenario_async(scenario_name="execute")
            assert catalog is not None
            assert catalog.scenario_type == "Execute"
            scenario = await asyncio.to_thread(Execute)
            assert scenario._objective_scorer.get_chat_target() is target
        finally:
            await service.close_async()

        with pytest.raises(RuntimeError, match="must supply a local scorer"):
            await target.send_prompt_async(
                message=MessagePiece(role="user", original_value="Accidental scoring request.").to_message()
            )
        fallback.assert_not_called()
        network.assert_not_called()

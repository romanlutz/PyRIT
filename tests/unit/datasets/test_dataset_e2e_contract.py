# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Exercise the real dataset E2E assertions without starting its backend fixture."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest

from pyrit.datasets import SeedDatasetProvider
from pyrit.models import SeedDataset, SeedObjective

_PROVIDER_NAME = "LocalDataset_latent_injection_tasks"


@pytest.fixture
def dataset_e2e_module() -> ModuleType:
    path = Path(__file__).parents[2] / "end_to_end" / "test_all_datasets.py"
    spec = importlib.util.spec_from_file_location("dataset_e2e_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def task_dataset_async() -> SeedDataset:
    provider = SeedDatasetProvider.get_all_providers()[_PROVIDER_NAME]()
    return await provider.fetch_dataset_async(cache=False)


@pytest.mark.usefixtures("patch_central_database")
async def test_local_tasks_pass_actual_e2e_validation_async(dataset_e2e_module: ModuleType) -> None:
    provider_cls = SeedDatasetProvider.get_all_providers()[_PROVIDER_NAME]
    await dataset_e2e_module.TestAllDatasets().test_fetch_dataset(_PROVIDER_NAME, provider_cls)


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("family", ["report", "resume", "latent_jailbreak"])
async def test_intentional_empty_task_passes_actual_e2e_validation_async(
    dataset_e2e_module: ModuleType, task_dataset_async: SeedDataset, family: str
) -> None:
    seed = next(seed for seed in task_dataset_async.seeds if seed.value == "" and seed.metadata["family"] == family)
    dataset = task_dataset_async.model_copy(update={"seeds": [seed]}, deep=True)
    provider_cls = SeedDatasetProvider.get_all_providers()[_PROVIDER_NAME]
    with patch.object(dataset_e2e_module, "_fetch_with_retry", new=AsyncMock(return_value=dataset)):
        await dataset_e2e_module.TestAllDatasets().test_fetch_dataset(_PROVIDER_NAME, provider_cls)


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "updates",
    [
        {"dataset_name": "unrelated_dataset"},
        {"dataset_name": "garak_latent_injection_triggers"},
        {"source": None},
        {"source": "https://github.com/NVIDIA/garak/blob/main/garak/probes/latentinjection.py"},
        {"metadata": {}},
        {"metadata": {"family": "whois", "language": "en", "garak_class": "LatentInjectionReport"}},
        {"metadata": {"family": "report", "language": "fr", "garak_class": "LatentInjectionReport"}},
        {"metadata": {"family": "report", "language": "en", "garak_class": "LatentInjectionResume"}},
        {"metadata": {"family": ["report"], "language": "en", "garak_class": "LatentInjectionReport"}},
        {"metadata": {"family": "report", "language": "en"}},
    ],
)
async def test_other_empty_seeds_fail_actual_e2e_validation_async(
    dataset_e2e_module: ModuleType, task_dataset_async: SeedDataset, updates: dict[str, object]
) -> None:
    seed = next(seed for seed in task_dataset_async.seeds if seed.value == "" and seed.metadata["family"] == "report")
    invalid = seed.model_copy(update=updates, deep=True)
    dataset = task_dataset_async.model_copy(update={"seeds": [invalid]}, deep=True)
    provider_cls = SeedDatasetProvider.get_all_providers()[_PROVIDER_NAME]
    with patch.object(dataset_e2e_module, "_fetch_with_retry", new=AsyncMock(return_value=dataset)):
        with pytest.raises(AssertionError, match="has no value"):
            await dataset_e2e_module.TestAllDatasets().test_fetch_dataset(_PROVIDER_NAME, provider_cls)


@pytest.mark.usefixtures("patch_central_database")
async def test_empty_objective_cannot_claim_task_exception_async(
    dataset_e2e_module: ModuleType, task_dataset_async: SeedDataset
) -> None:
    seed = next(seed for seed in task_dataset_async.seeds if seed.value == "" and seed.metadata["family"] == "report")
    objective = SeedObjective(value="", dataset_name=seed.dataset_name, source=seed.source, metadata=seed.metadata)
    dataset = task_dataset_async.model_copy(update={"seeds": [objective]}, deep=True)
    provider_cls = SeedDatasetProvider.get_all_providers()[_PROVIDER_NAME]
    with patch.object(dataset_e2e_module, "_fetch_with_retry", new=AsyncMock(return_value=dataset)):
        with pytest.raises(AssertionError, match="has no value"):
            await dataset_e2e_module.TestAllDatasets().test_fetch_dataset(_PROVIDER_NAME, provider_cls)

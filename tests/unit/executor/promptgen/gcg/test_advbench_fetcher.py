# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the AdvBench-for-GCG fetcher.

The fetcher is torch-free (it only depends on the dataset loader and
``pyrit.models``), so these tests run without the GCG extra installed. The
network fetch is always mocked — no unit test hits the internet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.promptgen.gcg.data import (
    _AdvBenchGCGLoader,
    create_gcg_data_config_async,
    fetch_advbench_for_gcg_async,
)
from pyrit.models import Message, SeedDataset, SeedObjective
from pyrit.prompt_target import PromptTarget

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

_FAKE_ROWS = [
    {"goal": "goal a", "target": "Sure, here is a"},
    {"goal": "goal b", "target": "Sure, here is b"},
    {"goal": "goal c", "target": "Sure, here is c"},
]


def _patch_fetch(rows: list[dict[str, str]]) -> AbstractContextManager[MagicMock]:
    return patch.object(_AdvBenchGCGLoader, "_fetch_from_url", MagicMock(return_value=rows))


async def test_fetch_returns_seed_dataset_with_gcg_target_metadata() -> None:
    with _patch_fetch(_FAKE_ROWS):
        dataset = await fetch_advbench_for_gcg_async()

    assert len(dataset.seeds) == 3
    assert [seed.value for seed in dataset.seeds] == ["goal a", "goal b", "goal c"]
    assert [seed.metadata["gcg_target"] for seed in dataset.seeds] == [
        "Sure, here is a",
        "Sure, here is b",
        "Sure, here is c",
    ]
    assert dataset.dataset_name == "advbench_gcg"


async def test_fetch_limits_to_first_n() -> None:
    with _patch_fetch(_FAKE_ROWS):
        dataset = await fetch_advbench_for_gcg_async(n=2)

    assert [seed.value for seed in dataset.seeds] == ["goal a", "goal b"]


async def test_fetch_n_larger_than_dataset_returns_all() -> None:
    with _patch_fetch(_FAKE_ROWS):
        dataset = await fetch_advbench_for_gcg_async(n=100)

    assert len(dataset.seeds) == 3


@pytest.mark.parametrize("bad_n", [0, -1])
async def test_fetch_rejects_non_positive_n(bad_n: int) -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        await fetch_advbench_for_gcg_async(n=bad_n)


async def test_fetch_missing_target_column_raises() -> None:
    with _patch_fetch([{"goal": "goal a"}]):
        with pytest.raises(ValueError, match="missing goal/target"):
            await fetch_advbench_for_gcg_async()


async def test_fetch_missing_goal_column_raises() -> None:
    with _patch_fetch([{"target": "Sure, here is a"}]):
        with pytest.raises(ValueError, match="missing goal/target"):
            await fetch_advbench_for_gcg_async()


async def test_fetch_is_compatible_with_from_seed_dataset() -> None:
    from pyrit.executor.promptgen.gcg.config import GCGDataConfig

    with _patch_fetch(_FAKE_ROWS):
        dataset = await fetch_advbench_for_gcg_async(n=2)

    config = GCGDataConfig.from_seed_dataset(dataset)
    assert config.train_goals == ["goal a", "goal b"]
    assert config.train_targets == ["Sure, here is a", "Sure, here is b"]


def _mock_adversarial_target(*responses: str) -> MagicMock:
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock(
        side_effect=[[Message.from_prompt(prompt=response, role="assistant")] for response in responses]
    )
    return target


async def test_create_config_preserves_provided_targets_without_adversarial_target() -> None:
    dataset = SeedDataset(
        seeds=[
            SeedObjective(value="goal a", metadata={"gcg_target": "Provided a"}),
            SeedObjective(value="goal b", metadata={"gcg_target": "Provided b"}),
        ]
    )

    config = await create_gcg_data_config_async(dataset=dataset)

    assert config.train_goals == ["goal a", "goal b"]
    assert config.train_targets == ["Provided a", "Provided b"]


async def test_create_config_generates_only_missing_targets() -> None:
    dataset = SeedDataset(
        seeds=[
            SeedObjective(value="goal a", metadata={"gcg_target": "Provided a"}),
            SeedObjective(value="goal b"),
            SeedObjective(value="goal c", metadata={"gcg_target": "  "}),
        ]
    )
    target = _mock_adversarial_target("  Generated b  ", "Generated c")

    config = await create_gcg_data_config_async(dataset=dataset, adversarial_target=target)

    assert config.train_goals == ["goal a", "goal b", "goal c"]
    assert config.train_targets == ["Provided a", "Generated b", "Generated c"]
    assert target.send_prompt_async.await_count == 2
    sent_prompts = [call.kwargs["message"].get_value() for call in target.send_prompt_async.await_args_list]
    assert "goal b" in sent_prompts[0]
    assert "goal c" in sent_prompts[1]


async def test_create_config_resolves_test_dataset() -> None:
    train = SeedDataset(seeds=[SeedObjective(value="train", metadata={"gcg_target": "Provided"})])
    test = SeedDataset(seeds=[SeedObjective(value="test")])
    target = _mock_adversarial_target("Generated test")

    config = await create_gcg_data_config_async(
        dataset=train,
        test_dataset=test,
        adversarial_target=target,
    )

    assert config.test_goals == ["test"]
    assert config.test_targets == ["Generated test"]


async def test_create_config_missing_target_without_adversarial_target_raises() -> None:
    dataset = SeedDataset(seeds=[SeedObjective(value="goal")])

    with pytest.raises(ValueError, match="no adversarial_target"):
        await create_gcg_data_config_async(dataset=dataset)


@pytest.mark.parametrize("response", ["", "   "])
async def test_create_config_empty_generated_target_raises(response: str) -> None:
    dataset = SeedDataset(seeds=[SeedObjective(value="goal")])
    target = _mock_adversarial_target(response)

    with pytest.raises(ValueError, match="empty target completion"):
        await create_gcg_data_config_async(dataset=dataset, adversarial_target=target)


async def test_create_config_missing_generated_response_raises() -> None:
    dataset = SeedDataset(seeds=[SeedObjective(value="goal")])
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock(return_value=[])

    with pytest.raises(ValueError, match="returned no target completion"):
        await create_gcg_data_config_async(dataset=dataset, adversarial_target=target)


async def test_create_config_overlong_generated_target_raises() -> None:
    dataset = SeedDataset(seeds=[SeedObjective(value="goal")])
    target = _mock_adversarial_target("too long")

    with pytest.raises(ValueError, match="maximum is 3"):
        await create_gcg_data_config_async(
            dataset=dataset,
            adversarial_target=target,
            max_generated_target_length=3,
        )


async def test_create_config_rejects_nonpositive_maximum_length() -> None:
    dataset = SeedDataset(seeds=[SeedObjective(value="goal", metadata={"gcg_target": "Provided"})])

    with pytest.raises(ValueError, match="must be positive"):
        await create_gcg_data_config_async(dataset=dataset, max_generated_target_length=0)

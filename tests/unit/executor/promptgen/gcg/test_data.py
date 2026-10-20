# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for torch-free GCG dataset adaptation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.promptgen.gcg.data import create_gcg_data_config_async
from pyrit.models import Message, SeedDataset, SeedObjective
from pyrit.prompt_target import PromptTarget


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

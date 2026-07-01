# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the AdvBench-for-GCG fetcher.

The fetcher is torch-free (it only depends on the dataset loader and
``pyrit.models``), so these tests run without the GCG extra installed. The
network fetch is always mocked — no unit test hits the internet.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyrit.auxiliary_attacks.gcg.data import _AdvBenchGCGLoader, fetch_advbench_for_gcg_async

_FAKE_ROWS = [
    {"goal": "goal a", "target": "Sure, here is a"},
    {"goal": "goal b", "target": "Sure, here is b"},
    {"goal": "goal c", "target": "Sure, here is c"},
]


def _patch_fetch(rows: list[dict[str, str]]) -> object:
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
    from pyrit.auxiliary_attacks.gcg.config import GCGDataConfig

    with _patch_fetch(_FAKE_ROWS):
        dataset = await fetch_advbench_for_gcg_async(n=2)

    config = GCGDataConfig.from_seed_dataset(dataset)
    assert config.train_goals == ["goal a", "goal b"]
    assert config.train_targets == ["Sure, here is a", "Sure, here is b"]

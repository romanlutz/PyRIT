# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Goal/target data helpers for the GCG attack.

The GCG optimizer consumes paired ``(goal, target)`` lists: a goal is a harmful
behavior and the target is the completion string the suffix is optimized to
elicit. Both are carried inline on ``GCGDataConfig`` so the JSON shipped to
AzureML contains the data itself (the compute container never fetches data from
the public internet).

``fetch_advbench_for_gcg_async`` materializes the public AdvBench dataset into a
``SeedDataset`` whose seeds carry the target string under
``metadata["gcg_target"]``; users with a private dataset can build an equivalent
``SeedDataset`` in memory. Either way, ``GCGDataConfig.from_seed_dataset`` turns
it into inline goal/target lists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedObjective, SeedUnion

if TYPE_CHECKING:
    from pyrit.auxiliary_attacks.gcg.config import GCGDataConfig

_ADVBENCH_SOURCE: str = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)


def load_goals_and_targets(*, data: GCGDataConfig) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return the inline train/test goal and target lists from a data config.

    With ``GCGDataConfig`` carrying goals and targets inline, this is a thin
    typed accessor: it no longer reads any CSV or touches the network, so the
    compute-side run path is URL-free.

    Args:
        data (GCGDataConfig): Config holding inline goal/target lists.

    Returns:
        tuple[list[str], list[str], list[str], list[str]]:
            ``(train_goals, train_targets, test_goals, test_targets)``.
    """
    return (
        list(data.train_goals),
        list(data.train_targets),
        list(data.test_goals),
        list(data.test_targets),
    )


class _AdvBenchGCGLoader(_RemoteDatasetLoader):
    """AdvBench loader that carries the GCG target completion per seed.

    Unlike the general-purpose AdvBench objective dataset (goals only), GCG needs
    the paired ``target`` string from the original ``harmful_behaviors.csv``. That
    target is GCG-specific, so this loader lives in the GCG module and is not
    registered in the global dataset registry.
    """

    should_register = False

    def __init__(
        self,
        *,
        source: str = _ADVBENCH_SOURCE,
        source_type: Literal["public_url", "file"] = "public_url",
    ) -> None:
        """Initialize the AdvBench-for-GCG loader.

        Args:
            source (str): URL or path to the AdvBench ``harmful_behaviors.csv``
                (columns ``goal,target``). Defaults to the official raw URL.
            source_type (Literal["public_url", "file"]): Whether ``source`` is a
                public URL or a local file. Defaults to ``"public_url"``.
        """
        self.source = source
        self.source_type: Literal["public_url", "file"] = source_type

    @property
    def dataset_name(self) -> str:
        """The dataset name."""
        return "advbench_gcg"

    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
        """Fetch AdvBench and return a ``SeedDataset`` with per-seed GCG targets.

        Args:
            cache (bool): Whether to cache the fetched CSV. Defaults to True.

        Returns:
            SeedDataset: Objectives whose ``value`` is the goal and whose
                ``metadata["gcg_target"]`` is the target completion string.

        Raises:
            ValueError: If a row is missing its ``goal`` or ``target`` column.
        """
        rows = self._fetch_from_url(source=self.source, source_type=self.source_type, cache=cache)
        seeds: list[SeedUnion] = []
        for row in rows:
            goal = row.get("goal")
            target = row.get("target")
            if not goal or target is None:
                raise ValueError(f"AdvBench row missing goal/target column: {row!r}")
            seeds.append(
                SeedObjective(
                    value=goal,
                    dataset_name=self.dataset_name,
                    source=self.source,
                    metadata={"gcg_target": target},
                )
            )
        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)


async def fetch_advbench_for_gcg_async(
    *,
    n: int | None = None,
    source: str = _ADVBENCH_SOURCE,
    cache: bool = True,
) -> SeedDataset:
    """Fetch the public AdvBench dataset as a GCG-ready ``SeedDataset``.

    Each returned seed carries its harmful behavior as the objective ``value``
    and the paired AdvBench target completion under ``metadata["gcg_target"]``,
    so the result can be handed straight to ``GCGDataConfig.from_seed_dataset``.

    Args:
        n (int | None): If set, keep only the first ``n`` behaviors. ``None``
            returns the full dataset. Defaults to None.
        source (str): URL or path to the AdvBench ``harmful_behaviors.csv``.
            Defaults to the official raw URL.
        cache (bool): Whether to cache the fetched CSV. Defaults to True.

    Returns:
        SeedDataset: AdvBench objectives with GCG target metadata.

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    if n is not None and n <= 0:
        raise ValueError(f"n must be a positive integer or None, got {n}.")

    dataset = await _AdvBenchGCGLoader(source=source).fetch_dataset_async(cache=cache)
    if n is not None:
        dataset.seeds = list(dataset.seeds[:n])
    return dataset

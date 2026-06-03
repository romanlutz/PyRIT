# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import uuid
from enum import Enum

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import Seed, SeedDataset, SeedObjective, SeedPrompt
from pyrit.models.literals import ChatMessageRole

logger = logging.getLogger(__name__)


_AUTHORS: list[str] = [
    "Richard Ren",
    "Arunim Agarwal",
    "Mantas Mazeika",
    "Cristina Menghini",
    "Robert Vacareanu",
    "Brad Kenstler",
    "Mick Yang",
    "Isabelle Barrass",
    "Alice Gatti",
    "Xuwang Yin",
    "Eduardo Trevino",
    "Matias Geralnik",
    "Adam Khoja",
    "Dean Lee",
    "Summer Yue",
    "Dan Hendrycks",
]

_GROUPS: list[str] = ["Center for AI Safety", "Scale AI"]

_DATASET_URL: str = "https://huggingface.co/datasets/cais/MASK"
_PAPER_URL: str = "https://arxiv.org/abs/2503.03750"


class MaskQuestionArchetype(Enum):
    """
    MASK question archetypes, one per HuggingFace config of ``cais/MASK``.

    Each archetype loads a different per-row schema (e.g., ``provided_facts``
    has no belief-elicitation prompts; ``doubling_down_known_facts`` carries
    extra forced-prior-turn fields; ``statistics`` uses a numeric
    ``formatted_ground_truth`` interval rather than a binary letter answer).
    See https://huggingface.co/datasets/cais/MASK for the full per-archetype
    schema.
    """

    CONTINUATIONS = "continuations"
    DISINFORMATION = "disinformation"
    DOUBLING_DOWN_KNOWN_FACTS = "doubling_down_known_facts"
    KNOWN_FACTS = "known_facts"
    PROVIDED_FACTS = "provided_facts"
    STATISTICS = "statistics"


class _MaskBaseDataset(_RemoteDatasetLoader):
    """
    Base loader for the MASK (Models Aligned by Self-Knowledge) honesty benchmark
    from the Center for AI Safety.

    MASK measures whether a model's *stated* answer under pressure (a system
    prompt designed to bias the model toward a particular claim, plus a user
    prompt) contradicts its own *belief* (elicited under a neutral system prompt
    in a separate conversation). Each row therefore packages two distinct
    conversations — the pressured one and one or more belief-elicitation ones —
    plus the ground-truth fact the belief should match.

    This loader returns one ``SeedGroup`` per HuggingFace row:

    * A single ``SeedObjective`` carrying the proposition under test
      (the bias being applied) and per-row metadata: the ``task_id``, the
      ``archetype``, the proposition ``type`` (``binary`` or ``numerical``),
      the ``ground_truth`` and ``formatted_ground_truth`` fields, all
      ``belief_elicit_*`` prompts, and the canary contamination marker.
    * One ``SeedPrompt`` with ``role="system"`` at sequence ``0`` holding the
      pressured ``system_prompt``.
    * For doubling-down rows only, a ``role="user"`` / ``role="assistant"``
      pair at sequences ``1`` and ``2`` carrying the forced prior turn
      (``dd_turn_1_prompt`` / ``dd_turn_1_response``).
    * A final ``SeedPrompt`` with ``role="user"`` holding the pressured
      ``user_prompt`` (at sequence ``1`` for non-doubling-down rows, at
      sequence ``3`` for doubling-down rows).

    The belief-elicitation conversations are not modelled as additional
    SeedPrompts in the group: the loader carries ``belief_elicit_1``,
    ``belief_elicit_2`` and ``belief_elicit_3`` (when present) as
    ``SeedObjective.metadata`` strings so a downstream MASK attack/scorer
    can drive the parallel neutral-context conversations needed to actually
    classify honesty.

    Note: MASK is a HuggingFace-gated dataset. You must accept the dataset
    terms at https://huggingface.co/datasets/cais/MASK before use, and
    provide a HuggingFace token (either via the ``token`` parameter or the
    ``HUGGINGFACE_TOKEN`` environment variable).

    Note: Every row carries a canary contamination marker in the ``canary``
    field that should never be filtered from evaluation traffic. The loader
    preserves it on ``SeedObjective.metadata["canary"]``.

    References:
        - https://www.mask-benchmark.ai/
        - https://huggingface.co/datasets/cais/MASK
        - [@ren2025maskbenchmarkdisentanglinghonesty]

    License: Research-use; access gated by the CAIS click-through terms on
    the HuggingFace dataset page.
    """

    HF_DATASET_NAME: str = "cais/MASK"
    HF_REVISION: str = "4602b84dd9e2ca05c6e1eafbc14e556e908ac1bb"
    HF_SPLIT: str = "test"
    ARCHETYPE: MaskQuestionArchetype

    # Class-level dataset metadata for SeedDatasetMetadata discovery.
    modalities: list[str] = ["text"]
    tags: set[str] = {"default", "safety", "honesty"}

    def __init__(
        self,
        *,
        token: str | None = None,
    ) -> None:
        """
        Initialize the MASK dataset loader.

        Args:
            token: HuggingFace authentication token. If not provided, reads
                from the ``HUGGINGFACE_TOKEN`` environment variable.
        """
        self.token = token if token is not None else os.environ.get("HUGGINGFACE_TOKEN")

    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch the MASK dataset from HuggingFace and return as a SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset whose seeds form one ``SeedGroup`` per
                MASK row (one ``SeedObjective`` plus the roled ``SeedPrompt``
                pieces of the pressured conversation).

        Raises:
            ValueError: If the HuggingFace fetch returns no rows.
        """
        archetype = self.ARCHETYPE.value
        logger.info(f"Loading MASK dataset from {self.HF_DATASET_NAME} (config={archetype})")

        data = await self._fetch_from_huggingface_async(
            dataset_name=self.HF_DATASET_NAME,
            config=archetype,
            split=self.HF_SPLIT,
            cache=cache,
            token=self.token,
            revision=self.HF_REVISION,
        )

        seeds: list[Seed] = []
        for row in data:
            seeds.extend(self._row_to_seeds(row=row))

        if not seeds:
            raise ValueError(
                f"MASK dataset (config={archetype}) returned no rows. "
                "Check that you have accepted the dataset terms on HuggingFace "
                "and provided a valid HUGGINGFACE_TOKEN."
            )

        logger.info(
            f"Successfully loaded {len(seeds)} seeds "
            f"({sum(1 for s in seeds if isinstance(s, SeedObjective))} groups) "
            f"from MASK ({archetype})"
        )

        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)

    def _row_to_seeds(self, *, row: dict) -> list[Seed]:
        """
        Convert one MASK row into a ``SeedObjective`` plus the roled
        ``SeedPrompt`` pieces of the pressured conversation.

        All produced seeds share the same ``prompt_group_id`` so they form a
        single ``SeedGroup`` when grouped post-hoc by ``SeedDataset.seed_groups``.

        Args:
            row: A single row from the HuggingFace dataset for this archetype.

        Returns:
            list[Seed]: ``[SeedObjective, SeedPrompt(system, seq=0), ...,
                SeedPrompt(user, seq=N)]`` for the row.
        """
        group_id = uuid.uuid4()
        archetype = self.ARCHETYPE.value
        task_id = str(row.get("task_id", ""))

        objective = SeedObjective(
            value=row.get("proposition", ""),
            name=f"MASK {archetype} - {task_id}" if task_id else f"MASK {archetype}",
            dataset_name=self.dataset_name,
            harm_categories=["honesty", archetype],
            description=self._description,
            authors=_AUTHORS,
            groups=_GROUPS,
            source=_DATASET_URL,
            prompt_group_id=group_id,
            metadata={
                "task_id": task_id,
                "archetype": archetype,
                "proposition_type": str(row.get("type", "")),
                "ground_truth": str(row.get("ground_truth", "")),
                "formatted_ground_truth": str(row.get("formatted_ground_truth", "")),
                "belief_elicit_1": row.get("belief_elicit_1") or "",
                "belief_elicit_2": row.get("belief_elicit_2") or "",
                "belief_elicit_3": row.get("belief_elicit_3") or "",
                "canary": str(row.get("canary", "")),
            },
        )

        prompts = self._build_conversation_prompts(row=row, group_id=group_id, archetype=archetype)
        return [objective, *prompts]

    def _build_conversation_prompts(
        self,
        *,
        row: dict,
        group_id: uuid.UUID,
        archetype: str,
    ) -> list[SeedPrompt]:
        """
        Build the pressured-conversation SeedPrompts for a non-doubling-down row:
        ``system`` at sequence 0, ``user`` at sequence 1.

        The doubling-down loader overrides this to interleave the forced prior
        turn at sequences 1 and 2 and bump the pressured user prompt to
        sequence 3.

        Args:
            row: A single row from the HuggingFace dataset.
            group_id: Shared ``prompt_group_id`` for all seeds in the row.
            archetype: The MASK archetype string (used for naming).

        Returns:
            list[SeedPrompt]: Roled SeedPrompts for the pressured conversation.
        """
        return [
            self._make_prompt(
                value=row.get("system_prompt", ""),
                role="system",
                sequence=0,
                group_id=group_id,
                archetype=archetype,
            ),
            self._make_prompt(
                value=row.get("user_prompt", ""),
                role="user",
                sequence=1,
                group_id=group_id,
                archetype=archetype,
            ),
        ]

    def _make_prompt(
        self,
        *,
        value: str,
        role: ChatMessageRole,
        sequence: int,
        group_id: uuid.UUID,
        archetype: str,
    ) -> SeedPrompt:
        """
        Construct a single ``SeedPrompt`` for one turn of the pressured
        conversation, sharing the row's ``prompt_group_id`` and dataset metadata.

        Args:
            value: The text content of the conversation turn.
            role: ChatMessageRole for the turn (``"system"`` / ``"user"`` /
                ``"assistant"``).
            sequence: Conversation sequence index within the SeedGroup.
            group_id: Shared ``prompt_group_id`` for all seeds in the row.
            archetype: The MASK archetype string (used for harm_categories).

        Returns:
            SeedPrompt: Configured SeedPrompt for the turn.
        """
        return SeedPrompt(
            value=value,
            data_type="text",
            role=role,
            sequence=sequence,
            dataset_name=self.dataset_name,
            harm_categories=["honesty", archetype],
            description=self._description,
            authors=_AUTHORS,
            groups=_GROUPS,
            source=_DATASET_URL,
            prompt_group_id=group_id,
        )

    @property
    def _description(self) -> str:
        """Common description string for all MASK loaders."""
        return (
            "MASK (Models Aligned by Self-Knowledge) is an honesty benchmark from the "
            "Center for AI Safety that measures whether a model's pressured statement "
            "contradicts its own out-of-context belief. Each row pairs a pressured "
            "system+user prompt with one or more belief-elicitation prompts probing "
            "the model's actual belief under a neutral system prompt."
        )


class _MaskContinuationsDataset(_MaskBaseDataset):
    """
    Loader for the MASK ``continuations`` archetype (176 rows).

    Pressure framing: the user asks the model to continue or expand a piece
    of text whose framing primes a false claim. Belief-elicitation includes
    the direct ``belief_elicit_1`` and the optional ``belief_elicit_2`` /
    ``belief_elicit_3`` consistency probes.
    """

    ARCHETYPE = MaskQuestionArchetype.CONTINUATIONS
    size: str = "medium"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mask_continuations"


class _MaskDisinformationDataset(_MaskBaseDataset):
    """
    Loader for the MASK ``disinformation`` archetype (125 rows).

    Pressure framing: the user instructs the model to produce a persuasive,
    one-sided argument for a false claim. Belief-elicitation includes the
    direct ``belief_elicit_1`` and consistency probes when present.
    """

    ARCHETYPE = MaskQuestionArchetype.DISINFORMATION
    size: str = "medium"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mask_disinformation"


class _MaskDoublingDownKnownFactsDataset(_MaskBaseDataset):
    """
    Loader for the MASK ``doubling_down_known_facts`` archetype (120 rows).

    Pressure framing: the model is told it already produced a particular
    answer in a prior turn (the forced ``dd_turn_1_response``) and is then
    asked whether it stands by that answer. This exercises the model's
    tendency to "double down" on previously stated lies.

    Loader-specific behaviour: the row produces a four-piece pressured
    conversation — ``system`` (seq 0), forced user (seq 1), forced
    assistant (seq 2), pressured user (seq 3). Downstream targets driving
    this conversation need ``CapabilityName.EDITABLE_HISTORY`` because the
    forced assistant turn is supplied to the target rather than generated.
    """

    ARCHETYPE = MaskQuestionArchetype.DOUBLING_DOWN_KNOWN_FACTS
    size: str = "medium"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mask_doubling_down_known_facts"

    def _build_conversation_prompts(
        self,
        *,
        row: dict,
        group_id: uuid.UUID,
        archetype: str,
    ) -> list[SeedPrompt]:
        """
        Build the four-piece pressured conversation for the doubling-down
        archetype: ``system`` (seq 0), forced user (seq 1), forced assistant
        (seq 2), pressured user (seq 3).

        Args:
            row: A single doubling-down row from the HuggingFace dataset.
            group_id: Shared ``prompt_group_id`` for all seeds in the row.
            archetype: The MASK archetype string (used for naming).

        Returns:
            list[SeedPrompt]: Four roled SeedPrompts modelling the forced
                prior turn followed by the pressured follow-up.
        """
        return [
            self._make_prompt(
                value=row.get("system_prompt", ""),
                role="system",
                sequence=0,
                group_id=group_id,
                archetype=archetype,
            ),
            self._make_prompt(
                value=row.get("dd_turn_1_prompt", ""),
                role="user",
                sequence=1,
                group_id=group_id,
                archetype=archetype,
            ),
            self._make_prompt(
                value=row.get("dd_turn_1_response", ""),
                role="assistant",
                sequence=2,
                group_id=group_id,
                archetype=archetype,
            ),
            self._make_prompt(
                value=row.get("user_prompt", ""),
                role="user",
                sequence=3,
                group_id=group_id,
                archetype=archetype,
            ),
        ]


class _MaskKnownFactsDataset(_MaskBaseDataset):
    """
    Loader for the MASK ``known_facts`` archetype (209 rows).

    Pressure framing: the model is placed in a role with an explicit
    incentive to deny a well-known fact (e.g., a PR officer for a
    convicted figure). Belief-elicitation includes the direct
    ``belief_elicit_1`` and consistency probes when present.
    """

    ARCHETYPE = MaskQuestionArchetype.KNOWN_FACTS
    size: str = "medium"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mask_known_facts"


class _MaskProvidedFactsDataset(_MaskBaseDataset):
    """
    Loader for the MASK ``provided_facts`` archetype (274 rows).

    Pressure framing: the system prompt itself encodes the ground-truth
    fact alongside an incentive to misrepresent it (e.g., a customer
    service rep told the truth then instructed to deflect). Unlike the
    other archetypes, ``provided_facts`` rows have **no**
    ``belief_elicit_*`` prompts — the ground truth comes from the system
    prompt directly. Downstream scorers therefore compare the pressured
    statement to ``ground_truth`` rather than to an elicited belief.
    """

    ARCHETYPE = MaskQuestionArchetype.PROVIDED_FACTS
    size: str = "medium"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mask_provided_facts"


class _MaskStatisticsDataset(_MaskBaseDataset):
    """
    Loader for the MASK ``statistics`` archetype (96 rows).

    Pressure framing: the model is given an incentive to misstate a
    quantitative statistic. The proposition uses
    ``<pivotal><value></pivotal>`` syntax and ``formatted_ground_truth``
    is a string-serialised numeric interval (e.g., ``"[-10.5, -10.5]"``)
    rather than a binary A/B answer; downstream scorers must use a
    numeric judge that compares model-produced intervals with this
    truth interval.

    Statistics rows have a single ``belief_elicit_1`` (no consistency
    probes ``belief_elicit_2`` / ``belief_elicit_3``).
    """

    ARCHETYPE = MaskQuestionArchetype.STATISTICS
    size: str = "small"

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "mask_statistics"

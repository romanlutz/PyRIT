# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
AttackTechnique - Bundles an AttackStrategy with an optional AttackTechniqueSeedGroup.

Represents "how to attack" independently of "what to attack" (the objective).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyrit.models import AttackTechniqueIdentifier, ComponentIdentifier, Identifiable, SeedIdentifier

if TYPE_CHECKING:
    from pyrit.executor.attack import AttackStrategy
    from pyrit.models import AttackSeedGroup, AttackTechniqueSeedGroup
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import TrueFalseScorer


class AttackTechnique(Identifiable):
    """
    Bundles an attack technique with an optional technique seed group.

    An AttackTechnique encapsulates the full attack configuration — the technique
    (including its target, converters, and scorer) plus any reusable technique seeds
    (e.g. jailbreak templates). The objectives that define which weaknesses to probe
    live separately on the AttackSeedGroup / AtomicAttack.
    """

    def __init__(
        self,
        *,
        attack: AttackStrategy[Any, Any],
        seed_technique: AttackTechniqueSeedGroup | None = None,
    ) -> None:
        """Initialize an AttackTechnique."""
        self._attack = attack
        self._seed_technique = seed_technique

    @property
    def attack(self) -> AttackStrategy[Any, Any]:
        """The attack technique."""
        return self._attack

    @property
    def seed_technique(self) -> AttackTechniqueSeedGroup | None:
        """The optional technique seed group."""
        return self._seed_technique

    async def materialize_seed_group_async(
        self,
        *,
        seed_group: AttackSeedGroup,
        adversarial_chat: PromptTarget | None,
        objective_scorer: TrueFalseScorer | None,
    ) -> AttackSeedGroup:
        """
        Materialize runtime conversation scaffolding configured by this technique.

        Args:
            seed_group: The objective and merged technique seeds to materialize.
            adversarial_chat: Target used to generate simulated conversation turns.
            objective_scorer: Scorer used to evaluate simulated conversation turns.

        Returns:
            A new seed group with simulated-conversation configuration replaced by prompts.
        """
        from pyrit.executor.attack.multi_turn.simulated_conversation import (
            materialize_simulated_conversation_async,
        )

        return await materialize_simulated_conversation_async(
            seed_group=seed_group,
            adversarial_chat=adversarial_chat,
            objective_scorer=objective_scorer,
        )

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the behavioral identity for this attack technique.

        The identifier always contains the attack technique as ``children["attack"]``.
        When a seed technique is present, its seeds and prompt placement are included.

        Returns:
            ComponentIdentifier: The frozen identity snapshot.
        """
        technique_seeds: list[SeedIdentifier] | None = None
        identifier_params: dict[str, Any] | None = None
        if self._seed_technique is not None:
            technique_seed_ids = [SeedIdentifier.from_seed(seed) for seed in self._seed_technique.seeds]
            if technique_seed_ids:
                technique_seeds = list(technique_seed_ids)
            if self._seed_technique.prompt_placement != "preserve":
                identifier_params = {"prompt_placement": self._seed_technique.prompt_placement}

        return AttackTechniqueIdentifier.of(
            self,
            params=identifier_params,
            attack=self._attack.get_identifier(),
            technique_seeds=technique_seeds,
        )

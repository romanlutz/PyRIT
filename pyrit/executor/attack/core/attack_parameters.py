# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pyrit.common.deprecation import print_deprecation_message
from pyrit.models import AttackSeedGroup, Message

if TYPE_CHECKING:
    from pyrit.prompt_target import PromptTarget
    from pyrit.score import TrueFalseScorer

AttackParamsT = TypeVar("AttackParamsT", bound="AttackParameters")


@dataclass(frozen=True)
class AttackParameters:
    """
    Immutable parameters for attack execution.

    This class defines the standard contract for attack parameters. All attacks
    at a given level of the hierarchy share the same parameter signature.

    Attacks that don't accept certain parameters should use the `excluding()` factory
    to create a derived params type without those fields. Attacks that need additional
    parameters should extend this class with new fields.
    """

    # Natural-language description of what the attack tries to achieve (required)
    objective: str

    # Optional message to send to the objective target (overrides objective if provided)
    next_message: Message | None = None

    # Conversation that is automatically prepended to the target model
    prepended_conversation: list[Message] | None = None

    # Additional labels that can be applied to the prompts throughout the attack
    memory_labels: dict[str, str] | None = field(default_factory=dict)

    # Harm categories targeted by this attack, derived from the seed group's
    # seeds. Stamped onto the produced AttackResult.
    targeted_harm_categories: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the attack parameters."""
        lines = [f"{self.__class__.__name__}:"]
        lines.append(f"  objective: {self.objective}")

        if self.next_message is not None:
            piece_count = len(self.next_message.message_pieces)
            msg_value = self.next_message.get_value()
            # Truncate long messages for display
            if len(msg_value) > 100:
                msg_value = msg_value[:100] + "..."
            lines.append(f"  next_message: ({piece_count} piece(s)) {msg_value}")
        else:
            lines.append("  next_message: None")

        if self.prepended_conversation:
            lines.append(f"  prepended_conversation: {len(self.prepended_conversation)} message(s)")
            for i, msg in enumerate(self.prepended_conversation):
                role = msg.api_role if hasattr(msg, "api_role") else "unknown"
                piece_count = len(msg.message_pieces)
                value = msg.get_value()
                if len(value) > 60:
                    value = value[:60] + "..."
                lines.append(f"    [{i}] {role} ({piece_count} piece(s)): {value}")
        else:
            lines.append("  prepended_conversation: None")

        if self.memory_labels:
            lines.append(f"  memory_labels: {self.memory_labels}")

        return "\n".join(lines)

    @classmethod
    def supports_simulated_conversation_materialization(cls) -> bool:
        """
        Whether attack-technique materialization can run before this mapper.

        Parameter types that override ``from_seed_group_async`` keep their existing
        seed contract by default. They can override this method to opt in.

        Returns:
            True when this type uses the default seed-group mapper.
        """
        mapper_owner = next(
            (base for base in cls.__mro__ if "from_seed_group_async" in base.__dict__),
            None,
        )
        return mapper_owner is AttackParameters

    @classmethod
    async def from_seed_group_async(
        cls: type[AttackParamsT],
        *,
        seed_group: AttackSeedGroup,
        adversarial_chat: PromptTarget | None = None,
        objective_scorer: TrueFalseScorer | None = None,
        **overrides: Any,
    ) -> AttackParamsT:
        """
        Create an AttackParameters instance from a AttackSeedGroup.

        Extracts standard fields from the seed group and applies any overrides.
        If the seed_group has a simulated conversation config,
        generates the simulated conversation using the provided adversarial_chat and scorer.

        Args:
            seed_group: The seed attack group to extract parameters from.
            adversarial_chat: The adversarial chat target for generating simulated conversations.
                Required if seed_group has a simulated conversation config.
            objective_scorer: The scorer for evaluating simulated conversations.
                Required if seed_group has a simulated conversation config.
            **overrides: Field overrides to apply. Must be valid fields for this params type.

        Returns:
            An instance of this AttackParameters type.

        Raises:
            TypeError: If ``seed_group`` is not a ``AttackSeedGroup``.
            ValueError: If overrides contain invalid fields, or if seed_group has simulated
                conversation but adversarial_chat/scorer not provided.
        """
        if not isinstance(seed_group, AttackSeedGroup):
            raise TypeError(
                f"seed_group must be a AttackSeedGroup, got {type(seed_group).__name__}. "
                "Plain SeedGroup does not enforce the 'exactly one objective' invariant required for an attack."
            )

        # Get valid field names for this params type
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # Validate overrides don't contain invalid fields
        invalid_fields = set(overrides.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"{cls.__name__} does not accept parameters: {invalid_fields}. Accepted parameters: {valid_fields}"
            )

        # AttackSeedGroup's Pydantic validator guarantees exactly one objective is present.
        assert seed_group.objective is not None

        # Build params dict, only including fields this class accepts
        params: dict[str, Any] = {}

        if "objective" in valid_fields:
            params["objective"] = seed_group.objective.value

        if "memory_labels" in valid_fields:
            params["memory_labels"] = {}

        if "targeted_harm_categories" in valid_fields:
            params["targeted_harm_categories"] = list(seed_group.harm_categories)

        extraction_group = seed_group
        if seed_group.has_simulated_conversation:
            from pyrit.executor.attack.multi_turn.simulated_conversation import (
                materialize_simulated_conversation_async,
            )

            print_deprecation_message(
                old_item="AttackParameters.from_seed_group_async with SeedSimulatedConversation",
                new_item="AttackTechnique.materialize_seed_group_async",
                removed_in="1.3.0",
            )
            extraction_group = await materialize_simulated_conversation_async(
                seed_group=seed_group,
                adversarial_chat=adversarial_chat,
                objective_scorer=objective_scorer,
            )

        if "next_message" in valid_fields:
            params["next_message"] = extraction_group.next_message

        if "prepended_conversation" in valid_fields:
            params["prepended_conversation"] = extraction_group.prepended_conversation

        # Apply overrides (already validated above)
        params.update(overrides)

        return cls(**params)

    @classmethod
    def excluding(cls, *field_names: str) -> type[AttackParameters]:
        """
        Create a new AttackParameters subclass that excludes the specified fields.

        This factory method creates a frozen dataclass without the specified fields.
        The resulting class inherits the `from_seed_group()` behavior and will raise
        if excluded fields are passed as overrides.

        Args:
            *field_names: Names of fields to exclude from the new params type.

        Returns:
            A new AttackParameters subclass without the specified fields.

        Raises:
            ValueError: If any field_name is not a valid field of this class.

        Example:
            ReducedParameters = AttackParameters.excluding("next_message", "prepended_conversation")
        """
        # Validate all field names exist
        current_fields = {f.name for f in dataclasses.fields(cls)}
        invalid = set(field_names) - current_fields
        if invalid:
            raise ValueError(f"Cannot exclude non-existent fields: {invalid}. Valid fields: {current_fields}")

        # Build new fields list excluding the specified ones
        new_fields: list[Any] = []
        for f in dataclasses.fields(cls):
            if f.name not in field_names:
                # Preserve field defaults
                if f.default is not dataclasses.MISSING:
                    new_fields.append((f.name, f.type, field(default=f.default)))
                elif f.default_factory is not dataclasses.MISSING:
                    new_fields.append((f.name, f.type, field(default_factory=f.default_factory)))
                else:
                    new_fields.append((f.name, f.type))

        # Generate a descriptive class name
        excluded_str = "_".join(sorted(field_names))
        class_name = f"{cls.__name__}Excluding_{excluded_str}"

        # Create the new dataclass WITHOUT inheritance
        # This ensures dataclasses.fields() only returns the new class's fields
        new_cls = dataclasses.make_dataclass(
            class_name,
            new_fields,
            frozen=True,
        )

        # Attach from_seed_group_async that delegates to the parent classmethod
        # We need to call the underlying function with the new class type (c) so that
        # dataclasses.fields(cls) returns only the reduced field set.
        # Access via __dict__ to get the classmethod descriptor and extract __func__.
        _classmethod_descriptor = cls.__dict__["from_seed_group_async"]
        original_method = _classmethod_descriptor.__func__
        supports_materialization = cls.supports_simulated_conversation_materialization()

        async def from_seed_group_wrapper_async(
            c: Any, /, *, seed_group: Any, adversarial_chat: Any = None, objective_scorer: Any = None, **ov: Any
        ) -> Any:
            return await original_method(
                c, seed_group=seed_group, adversarial_chat=adversarial_chat, objective_scorer=objective_scorer, **ov
            )

        new_cls.from_seed_group_async = classmethod(from_seed_group_wrapper_async)  # type: ignore[ty:unresolved-attribute]

        def _supports_simulated_conversation_materialization(_c: Any) -> bool:
            return supports_materialization

        new_cls.supports_simulated_conversation_materialization = classmethod(  # type: ignore[ty:unresolved-attribute]
            _supports_simulated_conversation_materialization
        )

        return new_cls  # type: ignore[ty:invalid-return-type]

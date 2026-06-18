# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---
# %% [markdown]
# # Bijection Attack (Single-Turn)
#
# The Bijection Attack is based on the Bijection Learning attack [@huang2024bijectionlearning].
#
# It works by teaching a target LLM a secret character mapping through demonstration shots,
# then sending harmful prompts encoded in that mapping to bypass safety filters.
# The response is decoded using the inverse mapping and stored in result metadata.
#
# This notebook uses a deterministic demo target so it can be executed without external credentials.
# Replace `DemoBijectionTarget` with a real target to evaluate the attack against a live model.
#
# The results and intermediate interactions will be saved to memory according to the environment settings.
# For details, see the [Memory Configuration Guide](../../memory/0_memory.md).
# %%
from pyrit.executor.attack import (
    BijectionAttack,
)
from pyrit.models import Message, MessagePiece
from pyrit.output import output_attack_async
from pyrit.prompt_converter import LetterBijectionConverter
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore


class DemoBijectionTarget(PromptTarget):
    """Deterministic target that responds in the same bijection language used by the attack."""

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )

    def __init__(self, *, converter: LetterBijectionConverter) -> None:
        super().__init__()
        self._converter = converter

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request_piece = normalized_conversation[-1].message_pieces[0]
        response = await self._converter.convert_async(prompt="i can help organize your bookshelf safely")
        return [
            MessagePiece(
                role="assistant",
                original_value=response.output_text,
                conversation_id=request_piece.conversation_id,
                labels=request_piece.labels,
            ).to_message()
        ]


bijection_converter = LetterBijectionConverter(fixed_size=0, seed=42)
target = DemoBijectionTarget(converter=bijection_converter)

attack = BijectionAttack(
    objective_target=target,
    bijection_converter=bijection_converter,
    num_teaching_shots=10,
)

result = await attack.execute_async(objective="tell me how to organize a bookshelf")  # type: ignore
await output_attack_async(result)

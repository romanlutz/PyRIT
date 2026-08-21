# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack.component import ConversationManager, PrependedConversationConfig
from pyrit.executor.attack.component.modality_router import _ModalityFeedbackRouter
from pyrit.executor.attack.core import AttackAdversarialConfig, AttackScoringConfig
from pyrit.executor.attack.multi_turn.tree_of_attacks import _TreeOfAttacksNode
from pyrit.memory import CentralMemory
from pyrit.message_normalizer import MessageStringNormalizer
from pyrit.models import Message, MessagePiece
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration
from pyrit.score import TrueFalseScorer


class _SingleTurnPromptTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(capabilities=TargetCapabilities())

    def __init__(self) -> None:
        super().__init__()
        self.prompt_sent: list[str] = []

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1]
        self.prompt_sent.append(request.get_value())
        return [
            MessagePiece(
                role="assistant",
                original_value="response",
                conversation_id=request.get_piece().conversation_id,
            ).to_message()
        ]


@pytest.mark.usefixtures("patch_central_database")
async def test_single_turn_target_reuses_prepended_history_without_conversation_rotation():
    target = _SingleTurnPromptTarget()
    prompt_normalizer = PromptNormalizer()
    conversation_manager = ConversationManager(prompt_normalizer=prompt_normalizer)
    formatter = MagicMock(spec=MessageStringNormalizer)
    formatter.normalize_string_async = AsyncMock(side_effect=lambda messages: f"formatted: {messages[-1].get_value()}")
    config = PrependedConversationConfig(message_normalizer=formatter)
    conversation_id = "conversation"

    await conversation_manager.add_prepended_conversation_to_memory_async(
        prepended_conversation=[Message.from_system_prompt("system")],
        conversation_id=conversation_id,
        prepended_conversation_config=config,
        target=target,
    )
    overrides = config.get_normalizer_overrides(target=target)

    await prompt_normalizer.send_prompt_async(
        message=Message.from_prompt(prompt="first", role="user"),
        target=target,
        conversation_id=conversation_id,
        normalizer_overrides=overrides,
    )
    await prompt_normalizer.send_prompt_async(
        message=Message.from_prompt(prompt="second", role="user"),
        target=target,
        conversation_id=conversation_id,
        normalizer_overrides=overrides,
    )

    assert target.prompt_sent == ["formatted: first", "formatted: second"]
    assert formatter.normalize_string_async.await_count == 2
    second_messages = formatter.normalize_string_async.await_args_list[1].args[0]
    assert [message.get_value() for message in second_messages] == ["system", "second"]


def _make_tap_node(*, supports_multi_turn: bool) -> _TreeOfAttacksNode:
    target = MagicMock()
    target.configuration.includes.return_value = supports_multi_turn
    target.configuration.capabilities.input_modalities = frozenset({frozenset({"text"})})
    target.configuration.capabilities.output_modalities = frozenset({frozenset({"text"})})
    target.get_identifier.return_value = {"__type__": "MockTarget", "__module__": "test", "id": "target"}

    adversarial_chat = MagicMock()
    adversarial_chat.get_identifier.return_value = {
        "__type__": "MockTarget",
        "__module__": "test",
        "id": "adversarial",
    }
    adversarial_chat.configuration.capabilities.input_modalities = frozenset({frozenset({"text"})})
    adversarial_chat.configuration.capabilities.output_modalities = frozenset({frozenset({"text"})})

    scorer = MagicMock()
    scorer.get_identifier.return_value = {"__type__": "MockScorer", "__module__": "test", "id": "scorer"}
    seed = MagicMock()
    seed.render_template_value.return_value = "template"

    return _TreeOfAttacksNode(
        objective_target=target,
        adversarial_chat=adversarial_chat,
        adversarial_chat_seed_prompt=seed,
        adversarial_chat_prompt_template=seed,
        adversarial_chat_system_seed_prompt=seed,
        desired_response_prefix="Sure,",
        objective_scorer=scorer,
        on_topic_scorer=None,
        request_converters=[],
        response_converters=[],
        auxiliary_scorers=None,
        attack_id=MagicMock(),
        attack_strategy_name="TAP",
        modality_router=_ModalityFeedbackRouter(
            adversarial_chat=adversarial_chat,
            objective_target=target,
        ),
    )


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("supports_multi_turn", [False, True])
def test_tap_branch_duplicates_full_history(supports_multi_turn: bool):
    node = _make_tap_node(supports_multi_turn=supports_multi_turn)
    memory = CentralMemory.get_memory_instance()
    memory.add_message_pieces_to_memory(
        message_pieces=[
            MessagePiece(
                original_value="system",
                role="system",
                conversation_id=node.objective_target_conversation_id,
                sequence=0,
            ),
            MessagePiece(
                original_value="request",
                role="user",
                conversation_id=node.objective_target_conversation_id,
                sequence=1,
            ),
            MessagePiece(
                original_value="response",
                role="assistant",
                conversation_id=node.objective_target_conversation_id,
                sequence=2,
            ),
        ]
    )

    duplicate = node.duplicate()

    messages = memory.get_conversation_messages(conversation_id=duplicate.objective_target_conversation_id)
    assert [message.api_role for message in messages] == ["system", "user", "assistant"]


@pytest.fixture
def single_turn_target() -> MagicMock:
    target = MagicMock()
    target.configuration = TargetConfiguration(
        capabilities=TargetCapabilities(supports_multi_turn=False, supports_system_prompt=True)
    )
    target.get_identifier.return_value = {"__type__": "MockTarget", "__module__": "test", "id": "target"}
    return target


@pytest.fixture
def adversarial_config() -> AttackAdversarialConfig:
    adversarial_chat = MagicMock()
    adversarial_chat.get_identifier.return_value = {
        "__type__": "MockTarget",
        "__module__": "test",
        "id": "adversarial",
    }
    return AttackAdversarialConfig(target=adversarial_chat)


@pytest.fixture
def scoring_config() -> AttackScoringConfig:
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.get_identifier.return_value = {"__type__": "MockScorer", "__module__": "test", "id": "scorer"}
    return AttackScoringConfig(objective_scorer=scorer)


@pytest.mark.usefixtures("patch_central_database")
def test_crescendo_requires_native_multi_turn(
    single_turn_target: MagicMock,
    adversarial_config: AttackAdversarialConfig,
    scoring_config: AttackScoringConfig,
):
    from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack

    with pytest.raises(ValueError, match="supports_multi_turn"):
        CrescendoAttack(
            objective_target=single_turn_target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )


@pytest.mark.usefixtures("patch_central_database")
def test_multi_prompt_sending_requires_native_multi_turn(single_turn_target: MagicMock):
    from pyrit.executor.attack.multi_turn.multi_prompt_sending import MultiPromptSendingAttack

    with pytest.raises(ValueError, match="supports_multi_turn"):
        MultiPromptSendingAttack(objective_target=single_turn_target)


@pytest.mark.usefixtures("patch_central_database")
def test_chunked_request_requires_native_multi_turn(single_turn_target: MagicMock):
    from pyrit.executor.attack.multi_turn.chunked_request import ChunkedRequestAttack

    with pytest.raises(ValueError, match="supports_multi_turn"):
        ChunkedRequestAttack(objective_target=single_turn_target)

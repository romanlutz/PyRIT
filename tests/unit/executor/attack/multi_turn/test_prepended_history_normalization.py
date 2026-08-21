# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Focused regression tests for prepended-history target normalization."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.executor.attack.component import ConversationManager, PrependedConversationConfig
from pyrit.executor.attack.component.modality_router import _ModalityFeedbackRouter
from pyrit.executor.attack.core import AttackAdversarialConfig, AttackScoringConfig
from pyrit.executor.attack.core.attack_parameters import AttackParameters
from pyrit.executor.attack.multi_turn.multi_turn_attack_strategy import (
    MultiTurnAttackContext,
    MultiTurnAttackStrategy,
)
from pyrit.executor.attack.multi_turn.tree_of_attacks import (
    TreeOfAttacksWithPruningAttack,
    _TreeOfAttacksNode,
)
from pyrit.memory import CentralMemory
from pyrit.message_normalizer import MessageStringNormalizer
from pyrit.models import (
    ComponentIdentifier,
    Conversation,
    ConversationReference,
    ConversationType,
    Message,
    MessagePiece,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import (
    CapabilityName,
    PromptTarget,
    TargetCapabilities,
    TargetConfiguration,
)
from pyrit.score import TrueFalseScorer


class _RecordingTarget(PromptTarget):
    def __init__(self, *, supports_multi_turn: bool = False, supports_editable_history: bool = False) -> None:
        super().__init__(
            custom_configuration=TargetConfiguration(
                capabilities=TargetCapabilities(
                    supports_multi_turn=supports_multi_turn,
                    supports_editable_history=supports_editable_history,
                )
            )
        )
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


def _make_context() -> MultiTurnAttackContext[AttackParameters]:
    return MultiTurnAttackContext(params=AttackParameters(objective="test objective"))


def _rotate(
    *,
    target: PromptTarget,
    context: MultiTurnAttackContext[AttackParameters],
) -> MagicMock:
    strategy = MagicMock()
    strategy._objective_target = target
    strategy._logger = MagicMock()
    MultiTurnAttackStrategy._rotate_conversation_for_single_turn_target(strategy, context=context)
    return strategy


def _seed_conversation(
    *,
    conversation_id: str,
    target: PromptTarget,
    messages: list[Message],
) -> None:
    memory = CentralMemory.get_memory_instance()
    memory.add_conversation_to_memory(
        conversation=Conversation(
            conversation_id=conversation_id,
            target_identifier=target.get_identifier(),
        )
    )
    for message in messages:
        for piece in message.message_pieces:
            piece.conversation_id = conversation_id
        memory.add_message_to_memory(request=message)


@pytest.mark.usefixtures("patch_central_database")
async def test_single_turn_target_replays_seed_without_rotation():
    target = _RecordingTarget()
    prompt_normalizer = PromptNormalizer()
    manager = ConversationManager(prompt_normalizer=prompt_normalizer)
    config = PrependedConversationConfig()
    conversation_id = "conversation"
    prepended = Message.from_system_prompt("system")

    await manager.add_prepended_conversation_to_memory_async(
        prepended_conversation=[prepended],
        conversation_id=conversation_id,
        prepended_conversation_config=config,
        target=target,
    )
    persisted = manager.get_conversation(conversation_id)
    target_context = manager.create_target_normalization_context(
        target=target,
        conversation_id=conversation_id,
        prepended_messages=persisted,
    )
    assert target_context is not None

    for prompt in ["first", "second"]:
        await prompt_normalizer.send_prompt_async(
            message=Message.from_prompt(prompt=prompt, role="user"),
            target=target,
            conversation_id=conversation_id,
            normalizer_overrides=config.get_normalizer_overrides(
                target=target,
                target_normalization_context=target_context,
            ),
            target_normalization_context=target_context,
        )

    assert target.prompt_sent == [
        "Turn 1:\nuser: ### Instructions ###\n\nsystem\n\n######\n\nfirst",
        "Turn 1:\nuser: ### Instructions ###\n\nsystem\n\n######\n\nsecond",
    ]


@pytest.mark.usefixtures("patch_central_database")
def test_rotation_is_noop_for_multi_turn_target():
    target = _RecordingTarget(supports_multi_turn=True)
    context = _make_context()
    context.executed_turns = 1
    original_id = context.session.conversation_id

    _rotate(target=target, context=context)

    assert context.session.conversation_id == original_id
    assert not context.related_conversations


@pytest.mark.usefixtures("patch_central_database")
def test_rotation_is_noop_for_seeded_single_turn_target():
    target = _RecordingTarget()
    context = _make_context()
    context.executed_turns = 2
    original_id = context.session.conversation_id
    seed = Message.from_prompt(prompt="seed", role="user")
    context.target_normalization_context = ConversationManager.create_target_normalization_context(
        target=target,
        conversation_id=original_id,
        prepended_messages=[seed],
    )

    _rotate(target=target, context=context)

    assert context.session.conversation_id == original_id
    assert not context.related_conversations


@pytest.mark.usefixtures("patch_central_database")
def test_rotation_moves_unseeded_single_turn_target_to_fresh_conversation():
    target = _RecordingTarget()
    context = _make_context()
    context.executed_turns = 1
    original_id = context.session.conversation_id

    _rotate(target=target, context=context)

    assert context.session.conversation_id != original_id
    assert (
        ConversationReference(
            conversation_id=original_id,
            conversation_type=ConversationType.PRUNED,
            description="single-turn target prior turn 1",
        )
        in context.related_conversations
    )


@pytest.mark.usefixtures("patch_central_database")
async def test_rotation_preserves_system_payload_for_later_single_turn_send():
    target = _RecordingTarget()
    context = _make_context()
    context.executed_turns = 1
    old_id = context.session.conversation_id
    _seed_conversation(
        conversation_id=old_id,
        target=target,
        messages=[
            Message.from_system_prompt("system"),
            Message.from_prompt(prompt="old request", role="user"),
        ],
    )

    _rotate(target=target, context=context)

    assert context.target_normalization_context is not None
    config = PrependedConversationConfig()
    await PromptNormalizer().send_prompt_async(
        message=Message.from_prompt(prompt="current request", role="user"),
        target=target,
        conversation_id=context.session.conversation_id,
        normalizer_overrides=config.get_normalizer_overrides(
            target=target,
            target_normalization_context=context.target_normalization_context,
        ),
        target_normalization_context=context.target_normalization_context,
    )
    assert target.prompt_sent == ["Turn 1:\nuser: ### Instructions ###\n\nsystem\n\n######\n\ncurrent request"]


def _make_tap_node(*, target: PromptTarget) -> _TreeOfAttacksNode:
    adversarial_chat = MagicMock(spec=PromptTarget)
    adversarial_chat.configuration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )
    adversarial_chat.get_identifier.return_value = ComponentIdentifier(
        class_name="AdversarialTarget",
        class_module="tests",
    )
    scorer = MagicMock()
    scorer.get_identifier.return_value = ComponentIdentifier(class_name="Scorer", class_module="tests")
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
        attack_id=ComponentIdentifier(class_name="TAP", class_module="tests"),
        attack_strategy_name="TAP",
        modality_router=_ModalityFeedbackRouter(
            adversarial_chat=adversarial_chat,
            objective_target=target,
        ),
    )


@pytest.mark.parametrize("supports_multi_turn", [False, True])
@pytest.mark.usefixtures("patch_central_database")
def test_tap_branch_duplicates_full_history_with_explicit_boundary(supports_multi_turn: bool):
    target = _RecordingTarget(supports_multi_turn=supports_multi_turn)
    node = _make_tap_node(target=target)
    _seed_conversation(
        conversation_id=node.objective_target_conversation_id,
        target=target,
        messages=[
            Message.from_system_prompt("system"),
            Message.from_prompt(prompt="request", role="user"),
            Message.from_prompt(prompt="response", role="assistant"),
        ],
    )

    duplicate = node.duplicate()

    messages = CentralMemory.get_memory_instance().get_conversation_messages(
        conversation_id=duplicate.objective_target_conversation_id
    )
    assert [message.api_role for message in messages] == ["system", "user", "assistant"]
    assert duplicate._target_normalization_context is not None
    assert duplicate._target_normalization_context.history_message_count == 3
    assert duplicate._target_normalization_context.replay_history_each_send is not supports_multi_turn


@pytest.mark.usefixtures("patch_central_database")
def test_tap_branch_preserves_multimodal_last_response():
    target = _RecordingTarget()
    node = _make_tap_node(target=target)
    node.last_response = Message(
        message_pieces=[
            MessagePiece(role="assistant", original_value="text response"),
            MessagePiece(
                role="assistant",
                original_value="response.png",
                original_value_data_type="image_path",
            ),
        ]
    )

    duplicate = node.duplicate()

    assert duplicate.last_response == node.last_response
    assert duplicate.last_response is not node.last_response
    assert [piece.original_value_data_type for piece in duplicate.last_response.message_pieces] == [
        "text",
        "image_path",
    ]


@pytest.mark.usefixtures("patch_central_database")
async def test_tap_unseeded_stateless_send_retains_current_only_payload():
    target = _RecordingTarget()
    node = _make_tap_node(target=target)
    node._objective = "objective"
    previous_conversation_id = node.objective_target_conversation_id
    _seed_conversation(
        conversation_id=previous_conversation_id,
        target=target,
        messages=[Message.from_prompt(prompt="prior request", role="user")],
    )

    await node._send_prompt_to_target_async("current request")

    assert node.objective_target_conversation_id != previous_conversation_id
    assert target.prompt_sent == ["current request"]


@pytest.mark.usefixtures("patch_central_database")
async def test_tap_stateless_branch_sends_original_boundary_plus_each_current_request():
    target = _RecordingTarget()
    formatter = MagicMock(spec=MessageStringNormalizer)

    async def format_messages(messages: list[Message]) -> str:
        return "|".join(message.get_value() for message in messages)

    formatter.normalize_string_async = AsyncMock(side_effect=format_messages)
    node = _make_tap_node(target=target)
    node._prepended_conversation_config = PrependedConversationConfig(message_normalizer=formatter)
    _seed_conversation(
        conversation_id=node.objective_target_conversation_id,
        target=target,
        messages=[
            Message.from_prompt(prompt="branch request", role="user"),
            Message.from_prompt(prompt="branch response", role="assistant"),
        ],
    )
    duplicate = node.duplicate()
    duplicate._objective = "objective"

    await duplicate._send_prompt_to_target_async("first current")
    await duplicate._send_prompt_to_target_async("second current")

    assert target.prompt_sent == [
        "branch request|branch response|first current",
        "branch request|branch response|second current",
    ]
    formatted_values = [
        [message.get_value() for message in call.args[0]] for call in formatter.normalize_string_async.await_args_list
    ]
    assert formatted_values == [
        ["branch request", "branch response", "first current"],
        ["branch request", "branch response", "second current"],
    ]


@pytest.mark.parametrize("branching_factor", [1, 2])
@pytest.mark.usefixtures("patch_central_database")
async def test_tap_retained_and_cloned_stateless_branches_replay_same_full_history(branching_factor: int):
    target = _RecordingTarget()
    formatter = MagicMock(spec=MessageStringNormalizer)

    async def format_messages(messages: list[Message]) -> str:
        return "|".join(message.get_value() for message in messages)

    formatter.normalize_string_async = AsyncMock(side_effect=format_messages)
    node = _make_tap_node(target=target)
    node._prepended_conversation_config = PrependedConversationConfig(message_normalizer=formatter)
    _seed_conversation(
        conversation_id=node.objective_target_conversation_id,
        target=target,
        messages=[
            Message.from_prompt(prompt="branch request", role="user"),
            Message.from_prompt(prompt="branch response", role="assistant"),
        ],
    )
    context = MagicMock()
    context.nodes = [node]
    context.related_conversations = set()
    attack = MagicMock()
    attack._configuration.branching_factor = branching_factor

    TreeOfAttacksWithPruningAttack._branch_existing_nodes(attack, context)

    assert len(context.nodes) == branching_factor
    for branch in context.nodes:
        branch._objective = "objective"
        await branch._send_prompt_to_target_async("current request")
    assert target.prompt_sent == ["branch request|branch response|current request"] * branching_factor


@pytest.fixture
def adversarial_config() -> AttackAdversarialConfig:
    target = MagicMock(spec=PromptTarget)
    target.configuration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )
    return AttackAdversarialConfig(target=target)


@pytest.fixture
def scoring_config() -> AttackScoringConfig:
    scorer = MagicMock(spec=TrueFalseScorer)
    return AttackScoringConfig(objective_scorer=scorer)


@pytest.mark.usefixtures("patch_central_database")
def test_crescendo_requires_native_editable_history(
    adversarial_config: AttackAdversarialConfig,
    scoring_config: AttackScoringConfig,
):
    from pyrit.executor.attack.multi_turn.crescendo import CrescendoAttack

    target = _RecordingTarget(supports_multi_turn=True, supports_editable_history=False)
    with pytest.raises(ValueError, match=CapabilityName.EDITABLE_HISTORY.value):
        CrescendoAttack(
            objective_target=target,
            attack_adversarial_config=adversarial_config,
            attack_scoring_config=scoring_config,
        )


@pytest.mark.usefixtures("patch_central_database")
def test_multi_prompt_sending_requires_native_multi_turn():
    from pyrit.executor.attack.multi_turn.multi_prompt_sending import MultiPromptSendingAttack

    with pytest.raises(ValueError, match=CapabilityName.MULTI_TURN.value):
        MultiPromptSendingAttack(objective_target=_RecordingTarget())


@pytest.mark.usefixtures("patch_central_database")
def test_chunked_request_requires_native_multi_turn():
    from pyrit.executor.attack.multi_turn.chunked_request import ChunkedRequestAttack

    with pytest.raises(ValueError, match=CapabilityName.MULTI_TURN.value):
        ChunkedRequestAttack(objective_target=_RecordingTarget())

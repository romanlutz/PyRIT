# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock
from pyrit.executor.attack import BijectionAttack
from pyrit.executor.attack.core import AttackParameters
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import SingleTurnAttackContext
from pyrit.identifiers import ComponentIdentifier
from pyrit.prompt_target import PromptTarget


def _mock_target_id(name: str = "MockTarget") -> ComponentIdentifier:
    return ComponentIdentifier(
        class_name=name,
        class_module="test_module",
    )


@pytest.fixture
def mock_objective_target():
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = _mock_target_id()
    return target


@pytest.fixture
def basic_context():
    return SingleTurnAttackContext(
        params=AttackParameters(objective="how to make a bomb"),
        conversation_id=str(uuid.uuid4()),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestBijectionAttackInitialization:

    def test_default_teaching_shots(self, mock_objective_target):
        attack = BijectionAttack(objective_target=mock_objective_target)
        assert attack._num_teaching_shots == 5

    def test_custom_teaching_shots(self, mock_objective_target):
        attack = BijectionAttack(
            objective_target=mock_objective_target,
            num_teaching_shots=3,
        )
        assert attack._num_teaching_shots == 3

    def test_bijection_converter_created(self, mock_objective_target):
        attack = BijectionAttack(objective_target=mock_objective_target)
        assert attack._bijection_converter is not None

    def test_bijection_converter_fixed_size(self, mock_objective_target):
        attack = BijectionAttack(
            objective_target=mock_objective_target,
            fixed_size=5,
        )
        assert attack._bijection_converter.fixed_size == 5


@pytest.mark.usefixtures("patch_central_database")
class TestBijectionTeachingMessages:

    def test_teaching_messages_length(self, mock_objective_target):
        attack = BijectionAttack(
            objective_target=mock_objective_target,
            num_teaching_shots=3,
        )
        messages = attack._build_teaching_messages()
        assert len(messages) == 8

    def test_teaching_messages_contain_secret_code(self, mock_objective_target):
        attack = BijectionAttack(objective_target=mock_objective_target)
        messages = attack._build_teaching_messages()
        assert "secret code" in str(messages[0]).lower()
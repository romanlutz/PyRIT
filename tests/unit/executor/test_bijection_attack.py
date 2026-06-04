# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import MagicMock, patch
from pyrit.executor.attack.single_turn.bijection_attack import BijectionAttack
from pyrit.memory.central_memory import CentralMemory


class TestBijectionAttack:
    """Tests for BijectionAttack."""

    def setup_method(self):
        """Set up fake memory before each test."""
        self.memory_mock = MagicMock()
        CentralMemory.set_memory_instance(self.memory_mock)

    def test_initialization(self):
        """Test that BijectionAttack initializes correctly."""
        target = MagicMock()
        attack = BijectionAttack(objective_target=target)
        assert attack._num_teaching_shots == 5
        assert attack._bijection_converter is not None

    def test_custom_teaching_shots(self):
        """Test that custom num_teaching_shots is stored correctly."""
        target = MagicMock()
        attack = BijectionAttack(
            objective_target=target,
            num_teaching_shots=3,
        )
        assert attack._num_teaching_shots == 3

    def test_build_teaching_messages_length(self):
        """Test that correct number of teaching messages are built."""
        target = MagicMock()
        attack = BijectionAttack(
            objective_target=target,
            num_teaching_shots=3,
        )
        messages = attack._build_teaching_messages()
        assert len(messages) == 4

    def test_build_teaching_messages_content(self):
        """Test that teaching messages contain the mapping."""
        target = MagicMock()
        attack = BijectionAttack(objective_target=target)
        messages = attack._build_teaching_messages()
        assert "secret code" in str(messages[0]).lower()

    def test_bijection_converter_created(self):
        """Test that BijectionConverter is created with correct params."""
        target = MagicMock()
        attack = BijectionAttack(
            objective_target=target,
            bijection_type="letter",
            fixed_size=5,
        )
        assert attack._bijection_converter.fixed_size == 5
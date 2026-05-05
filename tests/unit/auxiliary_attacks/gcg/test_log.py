# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

import pytest

log_mod = pytest.importorskip(
    "pyrit.auxiliary_attacks.gcg.experiments.log",
    reason="GCG optional dependencies not installed",
)
log_loss = log_mod.log_loss
log_params = log_mod.log_params
log_table_summary = log_mod.log_table_summary
log_train_goals = log_mod.log_train_goals


class TestLogParams:
    """Tests for the log_params function."""

    def test_logs_default_param_keys(self) -> None:
        """Should extract default parameter keys without error."""
        params = MagicMock()
        params.to_dict.return_value = {
            "model_name": "test_model",
            "transfer": False,
            "n_train_data": 50,
            "n_test_data": 10,
            "n_steps": 100,
            "batch_size": 512,
            "extra_param": "ignored",
        }

        # Should not raise
        log_params(params=params)

    def test_logs_custom_param_keys(self) -> None:
        """Should accept custom parameter keys."""
        params = MagicMock()
        params.to_dict.return_value = {
            "model_name": "test_model",
            "batch_size": 256,
        }

        # Should not raise
        log_params(params=params, param_keys=["model_name", "batch_size"])


class TestLogTrainGoals:
    """Tests for the log_train_goals function."""

    def test_logs_goals(self) -> None:
        """Should log training goals without error."""
        log_train_goals(train_goals=["goal1", "goal2", "goal3"])

    def test_logs_empty_goals(self) -> None:
        """Should handle empty goals list."""
        log_train_goals(train_goals=[])


class TestLogLoss:
    """Tests for the log_loss function."""

    def test_logs_loss(self) -> None:
        """Should log loss without error."""
        log_loss(step=5, loss=0.123)


class TestLogTableSummary:
    """Tests for the log_table_summary function."""

    def test_logs_table_summary(self) -> None:
        """Should log summary without error."""
        log_table_summary(losses=[0.5, 0.3, 0.1], controls=["ctrl1", "ctrl2", "ctrl3"], n_steps=3)

    def test_logs_empty_summary(self) -> None:
        """Should handle empty losses and controls."""
        log_table_summary(losses=[], controls=[], n_steps=0)
